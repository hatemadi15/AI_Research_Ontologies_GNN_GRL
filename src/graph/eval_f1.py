"""
eval_f1.py - Leakage-free Multi-level Evaluation

What is evaluated
  The pipeline starts from the gold entity spans of MaterioMiner (term
  extraction is not evaluated here). The evaluation universe is the set of
  graph terms (entity texts in >= 2 annotated sentences, 315 terms). Each term
  has one or more gold classes: its annotated fine-grained NER types, which
  are MMO class names (all 177 resolve to a class of ontology.ttl).

Levels
  1. Typing (primary): the pipeline's decision per term
     (predicted_types.json) vs. the term's gold classes. Reports accuracy
     (= P = R = F1 at full coverage) with a bootstrap 95% CI, LLMs4OL Task-B
     set precision/recall/F1, macro-F1 over majority classes, acc@1/3/5,
     hierarchical P/R/F1 (ancestors without the ontology root), the area
     under the risk-coverage curve, and a score-threshold sweep.
  2. Alignment rows (ontology_alignment.csv): pairwise P/R/F1 per
     similarity threshold; a row counts only if its class is a gold class of
     the term. This replaces the old "Term F1", whose precision only checked
     that the target was *some* ontology label.
  3. Clustering (concept level): ARI / V-measure / NMI / B-cubed of the term
     clusters vs. majority classes, with all-singletons and one-cluster
     baselines. (The old concept level derived each cluster's class from the
     gold types.)
  4. Taxonomy (class level): term is-a edges lifted to class edges through
     the predicted types and compared with the ontology's subClassOf.
  5. Dataset checks (not model metrics): e.g. how many NER labels resolve to
     ontology classes - the old "Type F1" measured only this.

The results record whether ORACLE_TYPES / NER_TYPE_CLUSTERING were on; such
runs are upper bounds, not model results.

Usage:
  python eval_f1.py
  python eval_f1.py --legacy-alignment-csv path/to/ontology_alignment.csv
      (score the top-1 row per term of an old alignment file, e.g. v5)
"""

import os
import json
import argparse

import pandas as pd

import conll
import metrics
from config import (NER_TYPE_CLUSTERING, ORACLE_TYPES, PREDICTED_TYPES_BASE_FILE,
                    PREDICTED_TYPES_FILE, PROCESSED_DIR, RAW_DIR)
from ontology_utils import load_ontology

ALIGNMENT_PATH = os.path.join(PROCESSED_DIR, 'ontology_alignment.csv')
CLUSTERS_PATH = os.path.join(PROCESSED_DIR, 'fine_auto_clusters.json')
TAXONOMY_EDGES_PATH = os.path.join(PROCESSED_DIR, 'taxonomy_edges.csv')

# Typing-score thresholds for the abstention sweep. FIXED_THRESHOLD is chosen
# in advance; report it (or the full-coverage accuracy), not the best value.
TYPING_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
FIXED_THRESHOLD = 0.5
ALIGNMENT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def load_gold(onto):
    """Universe terms with their gold class sets and majority class."""
    _, doc_entities, _ = conll.load_corpus(os.path.join(RAW_DIR, 'fine_grained_ner'))
    terms = conll.valid_terms(doc_entities)
    counts = conll.term_type_counts(doc_entities)
    majority_types = conll.majority_types(doc_entities, terms)

    def resolve(ner_type):
        return onto.resolve(ner_type) or f'UNRESOLVED:{ner_type}'

    gold = {t: {resolve(ty) for ty in counts[t]} for t in terms}
    majority = {t: resolve(majority_types[t]) for t in terms}
    return terms, gold, majority, doc_entities


def load_predictions(path):
    """Predictions, scores and ranked candidates from predicted_types*.json."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    pred = {t: p.get('class_uri') for t, p in data.items()}
    scores = {t: float(p.get('score', 0.0)) for t, p in data.items()}
    cand = {}
    for t, p in data.items():
        ranked = [c[0] for c in p.get('top5', [])]
        if pred[t] in ranked:
            ranked.remove(pred[t])
        cand[t] = [pred[t]] + ranked
    return pred, scores, cand


def _row_class_uri(row, onto):
    if 'class_uri' in row and isinstance(row['class_uri'], str) and row['class_uri']:
        return row['class_uri']
    return onto.resolve(str(row['reference']))


def alignment_rows(csv_path, onto):
    """(term, class_uri, similarity) rows of an alignment CSV."""
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        uri = _row_class_uri(row, onto)
        rows.append((str(row['discovered']), uri, float(row['similarity'])))
    return rows


def predictions_from_alignment_csv(csv_path, onto):
    """Legacy mode: top-1 row per term (by similarity) as the typing decision."""
    best = {}
    ranked = {}
    for term, uri, sim in alignment_rows(csv_path, onto):
        ranked.setdefault(term, []).append((sim, uri))
        if term not in best or sim > best[term][0]:
            best[term] = (sim, uri)
    pred = {t: u for t, (s, u) in best.items()}
    scores = {t: s for t, (s, u) in best.items()}
    cand = {}
    for t, items in ranked.items():
        seen = []
        for _, uri in sorted(items, key=lambda x: -x[0]):
            if uri not in seen:
                seen.append(uri)
        cand[t] = seen
    return pred, scores, cand


def evaluate_typing(pred, scores, cand, gold, majority, onto):
    def ancestors(c):
        if c in onto.classes:
            return onto.ancestors(c, include_self=True)
        return {c}

    res = metrics.typing_metrics(pred, gold, majority)
    res.update(metrics.topk_accuracy(cand, gold, ks=(1, 3, 5)))
    res.update(metrics.hierarchical_prf(pred, gold, majority, ancestors))
    res.update(metrics.risk_coverage(pred, scores, gold))
    res['accuracy_ci95'] = metrics.accuracy_ci(pred, gold)
    res['fixed_threshold'] = FIXED_THRESHOLD
    res['thresholds'] = metrics.threshold_sweep(pred, scores, gold, TYPING_THRESHOLDS)
    return res


def evaluate_clustering(majority):
    if not os.path.exists(CLUSTERS_PATH):
        return {'note': 'no clusters file'}
    with open(CLUSTERS_PATH) as f:
        clusters = json.load(f)
    cluster_of = {t: cid for cid, terms in clusters.items() for t in terms
                  if t in majority}
    return metrics.clustering_metrics(cluster_of, majority)


def evaluate_taxonomy(pred, gold, majority, onto):
    if not os.path.exists(TAXONOMY_EDGES_PATH):
        return {'note': 'no taxonomy edges file'}
    edges = pd.read_csv(TAXONOMY_EDGES_PATH)
    active = set().union(*gold.values()) & set(onto.classes)

    def lift(type_of):
        lifted = set()
        for child, parent in zip(edges['child'], edges['parent']):
            c, p = type_of.get(child), type_of.get(parent)
            if c in onto.classes and p in onto.classes and c != p:
                lifted.add((c, p))
        return lifted

    res = metrics.taxonomy_edge_metrics(lift(pred), onto, active)
    res['n_term_edges'] = int(len(edges))
    # Diagnostic: lifting through the gold majority types isolates the quality
    # of the term-level edges from typing errors (gold used only to evaluate).
    res['diagnostic_gold_lifted'] = metrics.taxonomy_edge_metrics(
        lift(majority), onto, active)
    return res


def dataset_checks(doc_entities, onto):
    types = sorted({ty for sent in doc_entities for _, ty in sent})
    by_local = sum(1 for ty in types if ty in onto.local_to_uri)
    return {
        'note': 'Properties of the dataset, not model metrics. The old '
                '"Type F1" (~0.997) measured only this label/ontology overlap.',
        'n_fine_ner_types': len(types),
        'ner_types_resolvable_by_local_name': by_local,
        'n_ontology_classes': len(onto.classes),
        'n_restriction_triples': len(onto.restriction_triples()),
    }


def majority_class_reference(gold, majority):
    """Accuracy of always predicting the most frequent majority class."""
    top = max(sorted(set(majority.values())),
              key=lambda c: sum(1 for t in majority if majority[t] == c))
    return {'class': top,
            'accuracy': round(metrics.safe_div(
                sum(1 for t in gold if top in gold[t]), len(gold)), 4)}


def run_evaluation(legacy_alignment_csv=None, out_path=None):
    onto = load_ontology()
    terms, gold, majority, doc_entities = load_gold(onto)
    n_ambiguous = sum(1 for t in terms if len(gold[t]) > 1)
    print(f"Universe: {len(terms)} terms ({n_ambiguous} with >1 gold class), "
          f"{len(set().union(*gold.values()))} gold classes")

    results = {
        'oracle': bool(ORACLE_TYPES or NER_TYPE_CLUSTERING),
        'oracle_flags': {'ORACLE_TYPES': ORACLE_TYPES,
                         'NER_TYPE_CLUSTERING': NER_TYPE_CLUSTERING},
        'universe': {'n_terms': len(terms), 'n_ambiguous_terms': n_ambiguous,
                     'n_gold_classes': len(set().union(*gold.values())),
                     'n_majority_classes': len(set(majority.values())),
                     'gold_term_spans_given': True},
        'reference_majority_class': majority_class_reference(gold, majority),
    }

    if legacy_alignment_csv:
        print(f"Legacy mode: top-1 row per term of {legacy_alignment_csv}")
        pred, scores, cand = predictions_from_alignment_csv(legacy_alignment_csv, onto)
        results['prediction_source'] = f'legacy alignment CSV: {legacy_alignment_csv}'
        results['warning'] = ('Predictions come from an external alignment file; the '
                              'evaluator cannot tell whether they used gold types '
                              '(v5 files did, via type_match rows).')
        align_csv = legacy_alignment_csv
    else:
        pred_path = os.path.join(PROCESSED_DIR, PREDICTED_TYPES_FILE)
        if not os.path.exists(pred_path):
            pred_path = os.path.join(PROCESSED_DIR, PREDICTED_TYPES_BASE_FILE)
        pred, scores, cand = load_predictions(pred_path)
        results['prediction_source'] = os.path.basename(pred_path)
        align_csv = ALIGNMENT_PATH
        base_path = os.path.join(PROCESSED_DIR, PREDICTED_TYPES_BASE_FILE)
        if os.path.exists(base_path) and pred_path != base_path:
            bp, bs, bc = load_predictions(base_path)
            results['typing_before_rag'] = evaluate_typing(bp, bs, bc, gold, majority, onto)

    results['typing'] = evaluate_typing(pred, scores, cand, gold, majority, onto)

    if os.path.exists(align_csv):
        rows = alignment_rows(align_csv, onto)
        results['alignment_rows'] = {
            'n_rows': len(rows),
            'thresholds': metrics.pairwise_alignment_metrics(
                rows, gold, ALIGNMENT_THRESHOLDS),
        }
    results['clustering'] = evaluate_clustering(majority)
    results['taxonomy'] = evaluate_taxonomy(pred, gold, majority, onto)
    results['dataset_checks'] = dataset_checks(doc_entities, onto)

    out_path = out_path or os.path.join(PROCESSED_DIR, 'eval_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_summary(results)
    write_summary_csv(results, os.path.join(os.path.dirname(out_path), 'eval_summary.csv'))
    print(f"\nSaved detailed results to {out_path}")
    return results


def summary_rows(results):
    """Flat (metric, value) pairs of the headline numbers."""
    ty = results['typing']
    rows = [
        ('typing_accuracy', ty['accuracy']),
        ('typing_set_f1', ty['set_f1']),
        ('typing_macro_f1', ty['macro_f1']),
        ('typing_acc@5', ty['acc@5']),
        ('typing_h_f1', ty['h_f1']),
        ('typing_aurc', ty['aurc']),
        (f"typing_f1@{FIXED_THRESHOLD}", ty['thresholds'][str(FIXED_THRESHOLD)]['f1']),
    ]
    al = results.get('alignment_rows', {}).get('thresholds', {})
    if al:
        rows.append(('alignment_pairwise_f1@0.5', al['0.5']['f1']))
    cl = results.get('clustering', {})
    if 'ari' in cl:
        rows += [('clustering_ari', cl['ari']), ('clustering_v_measure', cl['v_measure']),
                 ('clustering_bcubed_f1', cl['bcubed_f1'])]
    tx = results.get('taxonomy', {})
    if 'f1' in tx:
        rows += [('taxonomy_edge_f1', tx['f1']),
                 ('taxonomy_precision_closure', tx['precision_closure'])]
    return rows


def write_summary_csv(results, path):
    pd.DataFrame(summary_rows(results), columns=['metric', 'value']).assign(
        oracle=results['oracle']).to_csv(path, index=False)


def print_summary(results):
    print("\n" + "=" * 70)
    title = "EVALUATION SUMMARY (leakage-free)"
    if results['oracle']:
        title += "  [ORACLE RUN: upper bound, gold types used]"
    print(title)
    print("=" * 70)
    if 'warning' in results:
        print(f"WARNING: {results['warning']}")
    ty = results['typing']
    ref = results['reference_majority_class']
    print(f"Typing ({ty['n_terms']} terms, coverage {ty['coverage']:.2f}):")
    print(f"  accuracy {ty['accuracy']:.4f}  95% CI {ty['accuracy_ci95']}  "
          f"(majority-class reference {ref['accuracy']:.4f})")
    print(f"  set P/R/F1 {ty['set_precision']:.4f}/{ty['set_recall']:.4f}/"
          f"{ty['set_f1']:.4f}   macro-F1 {ty['macro_f1']:.4f}")
    print(f"  acc@1/3/5 {ty['acc@1']:.4f}/{ty['acc@3']:.4f}/{ty['acc@5']:.4f}   "
          f"hierarchical F1 {ty['h_f1']:.4f}   AURC {ty['aurc']:.4f}")
    print(f"  {'thresh':>7} {'P':>7} {'R':>7} {'F1':>7} {'n':>5}")
    for th, m in ty['thresholds'].items():
        print(f"  {float(th):>7.2f} {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['f1']:>7.4f} {m['n_predicted']:>5}")
    if 'typing_before_rag' in results:
        b = results['typing_before_rag']
        print(f"  (before LLM re-ranking: accuracy {b['accuracy']:.4f}, "
              f"set F1 {b['set_f1']:.4f})")
    al = results.get('alignment_rows')
    if al:
        print(f"\nAlignment rows ({al['n_rows']}), pairwise:")
        print(f"  {'thresh':>7} {'P':>7} {'R':>7} {'F1':>7} {'rows':>6}")
        for th, m in al['thresholds'].items():
            print(f"  {float(th):>7.2f} {m['precision']:>7.4f} {m['recall']:>7.4f} "
                  f"{m['f1']:>7.4f} {m['n_rows']:>6}")
    cl = results.get('clustering', {})
    if 'ari' in cl:
        print(f"\nClustering ({cl['n_clusters']} clusters): ARI {cl['ari']:.4f}  "
              f"V {cl['v_measure']:.4f}  B-cubed F1 {cl['bcubed_f1']:.4f}  "
              f"(singletons B3-F1 {cl['baseline_singletons']['bcubed_f1']:.4f}, "
              f"one-cluster {cl['baseline_one_cluster']['bcubed_f1']:.4f})")
    tx = results.get('taxonomy', {})
    if 'f1' in tx:
        print(f"\nTaxonomy (class level, {tx['n_edges']} lifted edges): "
              f"P {tx['precision']:.4f} R {tx['recall']:.4f} F1 {tx['f1']:.4f}  "
              f"P_closure {tx['precision_closure']:.4f} "
              f"(chance {tx['random_precision_closure']:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Leakage-free evaluation')
    parser.add_argument('--legacy-alignment-csv', default=None,
                        help='score the top-1 row per term of an alignment CSV')
    parser.add_argument('--out', default=None, help='output JSON path')
    args = parser.parse_args()
    run_evaluation(args.legacy_alignment_csv, args.out)


if __name__ == "__main__":
    main()
