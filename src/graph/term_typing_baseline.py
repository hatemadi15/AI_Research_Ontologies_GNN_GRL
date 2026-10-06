"""
term_typing_baseline.py - Leakage-free, cross-validated term-typing baselines

Task (LLMs4OL Task B style): given a gold term span from the MaterioMiner
annotations, predict its Materials Mechanics Ontology (MMO) class. Gold term
spans are given, so this is *term typing*, not NER; there is no published
term-typing result for MaterioMiner -> MMO, these numbers are baselines.

Universe: the graph terms (entity texts in >= 2 sentences, 315 terms), or all
1,055 annotated surfaces with --min-df 1. Every candidate class of the
ontology (428) can be predicted.

Folds: StratifiedGroupKFold over terms (5 folds by default, repeated over
--seeds). Surface variants ('crack' / 'cracks') share a group and never
straddle folds. Supervised methods only see the gold types of training-fold
terms; zero-shot methods use no gold types at all.

Methods
  majority     most frequent training class                  [supervised-CV]
  exact        exact label / alt-label / local-name match     [zero-shot]
  label        SBERT cosine to class labels (max over labels) [zero-shot]
  definition   SBERT cosine to "label: skos:definition"       [zero-shot]
  label+def    mean of label and definition scores            [zero-shot]
  knn          k nearest training terms vote for their types  [supervised-CV]
  hybrid       LAMBDA * knn + (1 - LAMBDA) * label+def        [supervised-CV]
  hybrid+llm   LLM picks among the hybrid top-5, given class definitions, a
               context sentence and training-fold examples    [supervised-CV + LLM]

K and LAMBDA are fixed in advance (not tuned on test folds).

Usage:
  python term_typing_baseline.py                 # all non-LLM methods
  python term_typing_baseline.py --llm           # + hybrid+llm (first seed)
  python term_typing_baseline.py --min-df 1      # all annotated surfaces
"""

import argparse
import json
import os
import re
import statistics
import time

import numpy as np

import conll
import metrics
import splits
from config import DEFAULT_EMBEDDER_MODEL, PROCESSED_DIR, RAW_DIR, SEED
from ontology_utils import load_ontology
from text_match import TermMatcher

KNN_K = 5
LAMBDA = 0.5
LLM_TOPK = 5
LLM_N_EXAMPLES = 5

METHOD_SETTINGS = {
    'majority': 'supervised-CV',
    'exact': 'zero-shot',
    'label': 'zero-shot',
    'definition': 'zero-shot',
    'label+def': 'zero-shot',
    'knn': 'supervised-CV',
    'hybrid': 'supervised-CV',
    'hybrid+llm': 'supervised-CV + LLM',
}


def _normalize_rows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def load_data(min_df):
    """Terms of the universe with their gold class sets and majority class."""
    onto = load_ontology()
    _, doc_entities, _ = conll.load_corpus(
        os.path.join(RAW_DIR, 'fine_grained_ner'))
    terms = conll.valid_terms(doc_entities, min_df=min_df)
    counts = conll.term_type_counts(doc_entities)
    majority = conll.majority_types(doc_entities, terms)
    gold = {t: {onto.resolve(ty) for ty in counts[t]} - {None} for t in terms}
    gold_counts = {t: {onto.resolve(ty): c for ty, c in counts[t].items()
                       if onto.resolve(ty)} for t in terms}
    majority = {t: onto.resolve(majority[t]) for t in terms}
    return onto, terms, gold, gold_counts, majority


def build_contexts(terms):
    """First annotated (CoNLL) sentence mentioning each term, for the LLM."""
    sentences, _, _ = conll.load_corpus(os.path.join(RAW_DIR, 'fine_grained_ner'))
    matcher = TermMatcher(terms)
    context = {}
    for sent in sentences:
        for term in matcher.terms_in(sent):
            context.setdefault(term, sent)
    return context


def zero_shot_scores(onto, terms, embedder):
    """Label, definition and exact-match score matrices (terms x classes)."""
    uris = onto.uris
    term_emb = _normalize_rows(embedder.encode(terms, show_progress_bar=False))

    # Label score: max cosine over all labels of the class
    label_texts, owners = [], []
    for j, uri in enumerate(uris):
        for label in onto.labels(uri):
            label_texts.append(label)
            owners.append(j)
    label_emb = _normalize_rows(embedder.encode(label_texts, show_progress_bar=False))
    sims = term_emb @ label_emb.T
    label_scores = np.full((len(terms), len(uris)), -1.0)
    for col, j in enumerate(owners):
        label_scores[:, j] = np.maximum(label_scores[:, j], sims[:, col])

    def_texts = [f"{onto.primary_label(u)}: {onto.definition(u)}"
                 if onto.definition(u) else onto.primary_label(u) for u in uris]
    def_emb = _normalize_rows(embedder.encode(def_texts, show_progress_bar=False))
    def_scores = term_emb @ def_emb.T

    exact = {}
    for i, t in enumerate(terms):
        uri = onto.resolve(t)
        if uri is None:
            # also try a simple singular form ('cracks' -> 'crack')
            uri = onto.resolve(re.sub(r's$', '', t)) if t.endswith('s') else None
        if uri is not None:
            exact[t] = uri
    return term_emb, label_scores, def_scores, exact


def knn_scores(term_emb, train_idx, test_idx, terms, gold_counts, uri_index, k):
    """kNN class scores for test terms from training-fold terms only."""
    scores = np.zeros((len(test_idx), len(uri_index)))
    sims = term_emb[test_idx] @ term_emb[train_idx].T
    for row, i in enumerate(test_idx):
        order = np.argsort(-sims[row])[:k]
        total = 0.0
        for o in order:
            s = max(float(sims[row, o]), 0.0)
            counts = gold_counts[terms[train_idx[o]]]
            n = sum(counts.values())
            for uri, c in counts.items():
                scores[row, uri_index[uri]] += s * c / n
            total += s
        if total > 0:
            scores[row] /= total
    return scores


def ranked(row_scores, uris, k=5):
    order = np.argsort(-row_scores, kind='stable')[:k]
    return [uris[j] for j in order]


def llm_rerank(onto, term, context, candidates, examples):
    """Ask the LLM to choose among candidate classes; None if no valid answer."""
    from llm_validator import _cache_key, _get_client, _load_cache, _save_cache
    from config import LLM_MODEL

    lines = []
    for n, uri in enumerate(candidates, 1):
        definition = onto.definition(uri) or 'no definition'
        lines.append(f"{n}. {onto.primary_label(uri)} - {definition[:220]}")
    demo = '\n'.join(f"- '{t}' -> {onto.primary_label(u)}" for t, u in examples)
    prompt = (
        "You assign terms from materials-mechanics papers to classes of the "
        "Materials Mechanics Ontology.\n"
        f"Annotated examples:\n{demo}\n\n"
        f"Term: '{term}'\n"
        f"Context: '{context[:300]}'\n\n"
        f"Candidate classes:\n" + '\n'.join(lines) + "\n\n"
        "Answer with the number of the best candidate only."
    )
    cache = _load_cache()
    key = _cache_key('term_typing_rerank', term, candidates, examples, context)
    if key in cache:
        answer = cache[key]
    else:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5, temperature=0.0,
        )
        answer = (response.choices[0].message.content or '').strip()
        cache[key] = answer
        _save_cache(cache)
    m = re.search(r'\d+', answer)
    if m and 1 <= int(m.group(0)) <= len(candidates):
        return candidates[int(m.group(0)) - 1]
    return None


def evaluate_predictions(onto, pred, cand, gold, majority, seen_classes):
    ancestors = lambda c: onto.ancestors(c, include_self=True)  # noqa: E731
    res = metrics.typing_metrics(pred, gold, majority)
    res.update(metrics.topk_accuracy(cand, gold, ks=(1, 3, 5)))
    res.update(metrics.hierarchical_prf(pred, gold, majority, ancestors))
    seen = [t for t in gold if majority[t] in seen_classes]
    unseen = [t for t in gold if majority[t] not in seen_classes]
    res['acc_seen_classes'] = round(metrics.safe_div(
        sum(1 for t in seen if pred.get(t) in gold[t]), len(seen)), 4)
    res['acc_unseen_classes'] = round(metrics.safe_div(
        sum(1 for t in unseen if pred.get(t) in gold[t]), len(unseen)), 4)
    res['n_unseen'] = len(unseen)
    return res


def run(min_df=2, n_splits=5, seeds=(0, 1, 2), use_llm=False, embedder_model=None):
    from sentence_transformers import SentenceTransformer

    start = time.time()
    onto, terms, gold, gold_counts, majority = load_data(min_df)
    uris = onto.uris
    uri_index = {u: j for j, u in enumerate(uris)}
    embedder_model = embedder_model or os.environ.get(
        'EMBEDDER_MODEL', DEFAULT_EMBEDDER_MODEL)
    print(f"Term typing: {len(terms)} terms (min_df={min_df}), "
          f"{len(uris)} candidate classes, embedder={embedder_model}")
    embedder = SentenceTransformer(embedder_model)
    term_emb, label_scores, def_scores, exact = zero_shot_scores(onto, terms, embedder)
    labeldef_scores = (label_scores + def_scores) / 2
    contexts = build_contexts(terms) if use_llm else {}
    term_idx = {t: i for i, t in enumerate(terms)}

    methods = [m for m in METHOD_SETTINGS if use_llm or m != 'hybrid+llm']
    per_fold = {m: [] for m in methods}
    pooled = {m: ({}, {}) for m in methods}  # first seed: pred, candidates
    llm_calls = 0

    for seed in seeds:
        folds = splits.group_kfold_splits(
            terms, [majority[t] for t in terms], n_splits=n_splits, seed=seed)
        for fold_no, (train_terms, test_terms) in enumerate(folds):
            train_idx = [term_idx[t] for t in train_terms]
            test_idx = [term_idx[t] for t in test_terms]
            fold_gold = {t: gold[t] for t in test_terms}
            fold_major = {t: majority[t] for t in test_terms}
            seen_classes = set().union(*(gold[t] for t in train_terms))
            train_major = max(sorted(seen_classes),
                              key=lambda c: sum(1 for t in train_terms if majority[t] == c))

            knn = knn_scores(term_emb, train_idx, test_idx, terms, gold_counts,
                             uri_index, KNN_K)
            hybrid = LAMBDA * knn + (1 - LAMBDA) * labeldef_scores[test_idx]
            matrices = {
                'label': label_scores[test_idx],
                'definition': def_scores[test_idx],
                'label+def': labeldef_scores[test_idx],
                'knn': knn,
                'hybrid': hybrid,
            }

            fold_preds = {}
            for m in methods:
                pred, cand = {}, {}
                if m == 'majority':
                    for t in test_terms:
                        pred[t] = train_major
                        cand[t] = [train_major]
                elif m == 'exact':
                    for t in test_terms:
                        if t in exact:
                            pred[t] = exact[t]
                            cand[t] = [exact[t]]
                elif m == 'hybrid+llm':
                    if seed != seeds[0]:
                        continue  # LLM only on the first seed (cost control)
                    base_pred, base_cand = fold_preds['hybrid']
                    train_emb = term_emb[train_idx]
                    for row, t in enumerate(test_terms):
                        candidates = base_cand[t][:LLM_TOPK]
                        sims = train_emb @ term_emb[term_idx[t]]
                        demo_idx = np.argsort(-sims)[:LLM_N_EXAMPLES]
                        examples = [(train_terms[d], majority[train_terms[d]])
                                    for d in demo_idx]
                        try:
                            choice = llm_rerank(onto, t, contexts.get(t, ''),
                                                candidates, examples)
                            llm_calls += 1
                        except Exception as e:  # keep the base prediction
                            print(f"  LLM error for '{t}': {e}")
                            choice = None
                        pred[t] = choice or base_pred[t]
                        cand[t] = [pred[t]] + [c for c in candidates if c != pred[t]]
                else:
                    mat = matrices[m]
                    for row, t in enumerate(test_terms):
                        cand[t] = ranked(mat[row], uris, k=5)
                        pred[t] = cand[t][0]
                fold_preds[m] = (pred, cand)
                per_fold[m].append(evaluate_predictions(
                    onto, pred, cand, fold_gold, fold_major, seen_classes))
                if seed == seeds[0]:
                    pooled[m][0].update(pred)
                    pooled[m][1].update(cand)
            print(f"  seed {seed} fold {fold_no}: hybrid acc="
                  f"{per_fold['hybrid'][-1]['accuracy']:.3f}")

    keys = ['accuracy', 'set_f1', 'macro_f1', 'acc@3', 'acc@5', 'h_f1',
            'coverage', 'acc_seen_classes', 'acc_unseen_classes']
    results = {
        'task': 'term typing (gold term spans given) on MaterioMiner -> MMO',
        'config': {'min_df': min_df, 'n_terms': len(terms), 'n_classes': len(uris),
                   'n_splits': n_splits, 'seeds': list(seeds),
                   'embedder': embedder_model, 'knn_k': KNN_K, 'lambda': LAMBDA,
                   'llm': use_llm, 'llm_calls': llm_calls},
        'methods': {},
    }
    if use_llm:
        from config import LLM_MODEL
        results['config']['llm_model'] = LLM_MODEL
    for m in methods:
        runs = per_fold[m]
        mean = {k: round(statistics.mean(r[k] for r in runs), 4) for k in keys}
        std = {k: round(statistics.pstdev(r[k] for r in runs), 4) for k in keys}
        pred, cand = pooled[m]
        # Pooled over the first seed's folds: every term predicted exactly once
        pooled_metrics = evaluate_predictions(
            onto, pred, cand, gold, majority,
            set().union(*gold.values()))
        results['methods'][m] = {
            'setting': METHOD_SETTINGS[m],
            'mean_over_folds': mean,
            'std_over_folds': std,
            'n_fold_runs': len(runs),
            'pooled_first_seed': {k: pooled_metrics[k] for k in
                                  ['accuracy', 'set_precision', 'set_recall',
                                   'set_f1', 'macro_f1', 'h_f1', 'coverage']},
            'accuracy_ci95_first_seed': metrics.accuracy_ci(pred, gold, seed=SEED),
        }

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, f'term_typing_results_mindf{min_df}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Per-term predictions of the first seed (for error analysis)
    pred_path = os.path.join(PROCESSED_DIR, f'term_typing_predictions_mindf{min_df}.json')
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump({t: {'gold': sorted(onto.primary_label(u) for u in gold[t]),
                       **{m: (onto.primary_label(pooled[m][0][t])
                              if pooled[m][0].get(t) else None)
                          for m in methods}}
                   for t in terms}, f, indent=2, ensure_ascii=False)

    print(f"\n{'Method':<12} {'Setting':<20} {'Acc':>7} {'SetF1':>7} {'Macro':>7} "
          f"{'Acc@5':>7} {'hF1':>7} {'Unseen':>7}")
    for m in methods:
        r = results['methods'][m]['mean_over_folds']
        s = results['methods'][m]['std_over_folds']
        print(f"{m:<12} {METHOD_SETTINGS[m]:<20} {r['accuracy']:>7.3f} "
              f"{r['set_f1']:>7.3f} {r['macro_f1']:>7.3f} {r['acc@5']:>7.3f} "
              f"{r['h_f1']:>7.3f} {r['acc_unseen_classes']:>7.3f}  (acc sd {s['accuracy']:.3f})")
    print(f"\nSaved {out_path} ({time.time() - start:.0f}s)")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--min-df', type=int, default=conll.MIN_TERM_DF)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--llm', action='store_true',
                        help='also run hybrid+llm (needs an LLM API key)')
    parser.add_argument('--embedder', default=None)
    args = parser.parse_args()
    run(min_df=args.min_df, n_splits=args.folds, seeds=tuple(args.seeds),
        use_llm=args.llm, embedder_model=args.embedder)


if __name__ == '__main__':
    main()
