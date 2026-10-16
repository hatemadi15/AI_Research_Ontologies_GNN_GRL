"""
ner_consistency.py - document-level label consistency for saved NER test predictions

A post-processing rule for the test predictions of a results file (ner_baseline.py
--save-predictions, or a vote of ner_ensemble.py), fixed before it is applied to any real
predictions (ROADMAP §4). The new predictions are scored with ner_baseline.score and written
in the results-file schema, so that ner_seedmeans.py can pair them with their input run by
run. No training.

Rule. A document is a paper of MaterioMiner or an article of SOFC-Exp, and only the run's
test sentences of it are used. Each predicted entity gives a surface form, its exact words.
An occurrence of a form is a place in the document where its words appear: tagged if a
predicted entity has exactly that span, untagged if every one of its words is predicted O,
and otherwise left alone. A form is propagated if it has more tagged than untagged
occurrences and one type is the most frequent among its tagged ones: each untagged
occurrence becomes an entity of that type. Longer forms go first, and an occurrence becomes
an entity only while all its words are still O, so no prediction is removed or changed.
With --retype, the tagged occurrences of a propagated form also take its type.

The input must be valid IOB2 (a CRF head with IOB2 constraints, or a vote of such models),
so that rewriting a sentence's tags changes nothing but the rule's entities; the words come
from the corpus (ner_baseline.load_split_corpus) through each run's test_index.

Output: ner_results_<granularity>[_<protocol>]_<variant>.json next to the input (--out-dir
overrides), where the variant is the input's with _dc (with --retype, _dcr) before its seed
tag: crf_m300 gives crf_dc_m300, crf_ens3 gives crf_ens3_dc. Each run has its run_id, the
seqeval metrics, the number of entities added (n_added) and retyped (n_retyped), and the
input's f1 and f1_strict as input_f1 and input_f1_strict; with --save-predictions also
test_index, test_gold and the new test_pred (the rule regenerates them from the input).

Usage:
  python ner_consistency.py FILE [FILE ...] [--retype] [--out-dir DIR] [--save-predictions]
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter

from ner_baseline import load_split_corpus, score, summarize
from ner_ensemble import MEMBER_TAG, results_name, spans_to_tags, tags_to_spans

RULE = 'document_majority_propagation'
# top-level keys of the input that are copied to the output
COPIED_KEYS = ('granularity', 'protocol', 'model', 'labels', 'label_tie', 'paper_test_f1')


class ConsistencyError(ValueError):
    """The results file cannot be post-processed (reported without a traceback)."""


# ---------------------------------------------------------------------------
# The rule (pure Python)
# ---------------------------------------------------------------------------
def occurrences(form, words):
    """Start positions of the word sequence `form` in `words`."""
    n = len(form)
    return [i for i in range(len(words) - n + 1) if tuple(words[i:i + n]) == form]


def document_rule(sentences, preds, retype=False):
    """(tag lists, n_added, n_retyped): the rule on the test sentences of one document.

    sentences: their word lists; preds: their predicted IOB2 tag lists, which are not
    changed. A sentence the rule leaves alone keeps its tag list.
    """
    spans = [tags_to_spans(tags)[0] for tags in preds]
    types = {}
    for words, sentence_spans in zip(sentences, spans, strict=True):
        for start, end, kind in sentence_spans:
            types.setdefault(tuple(words[start:end]), Counter())[kind] += 1
    out = [set(s) for s in spans]
    added = retyped = 0
    for form in sorted(types, key=lambda f: (-len(f), f)):
        ranked = types[form].most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            continue                                # no single most frequent type
        kind = ranked[0][0]
        tagged, untagged = [], []
        for s, words in enumerate(sentences):
            for start in occurrences(form, words):
                end = start + len(form)
                exact = [span for span in spans[s] if span[:2] == (start, end)]
                if exact:
                    tagged.append((s, exact[0]))
                elif all(tag == 'O' for tag in preds[s][start:end]):
                    untagged.append((s, start))
        if len(tagged) <= len(untagged):
            continue
        for s, start in untagged:
            end = start + len(form)
            if all(e <= start or b >= end for b, e, _ in out[s]):
                out[s].add((start, end, kind))
                added += 1
        if retype:
            for s, span in tagged:
                if span[2] != kind:
                    out[s].discard(span)
                    out[s].add((span[0], span[1], kind))
                    retyped += 1
    tags = [spans_to_tags(sorted(new), len(words)) if new != set(old) else list(pred)
            for new, old, words, pred in zip(out, spans, sentences, preds)]
    return tags, added, retyped


# ---------------------------------------------------------------------------
# Results files
# ---------------------------------------------------------------------------
def dc_variant(variant, retype=False):
    """The output variant: _dc (or _dcr) before the input's seed tag m<N>, else at the end."""
    suffix = '_dcr' if retype else '_dc'
    tag = MEMBER_TAG.search(variant)
    if tag is None:
        return variant + suffix
    if tag.start() == 0:                            # a seed tag alone (m100)
        return suffix[1:] + '_' + variant
    return variant[:tag.start()] + suffix + variant[tag.start():]


def apply(results, corpus, retype=False, save_predictions=False):
    """The rule applied to every run of a parsed results file, scored (ner_baseline schema);
    save_predictions keeps the test sentences and the new predictions in each run."""
    out = {k: results[k] for k in COPIED_KEYS if k in results}
    out.update({'variant': dc_variant(results['variant'], retype),
                'config': {'input': results['variant'], 'rule': RULE, 'retype': retype}})
    runs = []
    for run in results['runs']:
        where = f"run {run.get('run_id')}"
        missing = [k for k in ('test_index', 'test_gold', 'test_pred') if k not in run]
        if missing:
            raise ConsistencyError(f"{where}: no {', '.join(missing)} (run with "
                                   f"--save-predictions)")
        gold = [g.split() for g in run['test_gold']]
        pred = [p.split() for p in run['test_pred']]
        if not len(run['test_index']) == len(gold) == len(pred):
            raise ConsistencyError(f'{where}: test_index, test_gold and test_pred differ in '
                                   f'length')
        words, docs = [], []
        for k, i in enumerate(run['test_index']):
            if not 0 <= i < len(corpus):
                raise ConsistencyError(f'{where}: test sentence {i} is not in the corpus')
            tokens, tags, doc = corpus[i]
            if not len(tokens) == len(gold[k]) == len(pred[k]):
                raise ConsistencyError(f'{where}: test sentence {i} does not match the corpus '
                                       f'({len(tokens)} words, {len(gold[k])} gold tags)')
            words.append(tokens)
            docs.append(doc)
        if any(tags_to_spans(p)[1] for p in pred):
            raise ConsistencyError(f'{where}: the predictions are not valid IOB2 (an I- tag '
                                   f'continues nothing); use a CRF head or a vote')
        new, added, retyped = list(pred), 0, 0
        for doc in dict.fromkeys(docs):
            positions = [k for k, d in enumerate(docs) if d == doc]
            tags, a, r = document_rule([words[k] for k in positions],
                                       [pred[k] for k in positions], retype)
            for k, t in zip(positions, tags):
                new[k] = t
            added, retyped = added + a, retyped + r
        res = score(gold, new)
        res.update({'run_id': run['run_id'], 'n_added': added, 'n_retyped': retyped,
                    'input_f1': run['f1'], 'input_f1_strict': run['f1_strict']})
        if save_predictions:
            res.update({'test_index': run['test_index'], 'test_gold': run['test_gold'],
                        'test_pred': [' '.join(t) for t in new]})
        runs.append(res)
    out['runs'] = runs
    summarize(out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Document-level label consistency for saved NER test predictions')
    parser.add_argument('files', nargs='+', metavar='FILE',
                        help='results files with test predictions (valid IOB2)')
    parser.add_argument('--retype', action='store_true',
                        help='also give the tagged occurrences of a propagated form its type '
                             '(variant _dcr)')
    parser.add_argument('--out-dir', default=None, metavar='DIR',
                        help='where to write (default: next to each input)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='keep test_index, test_gold and the new test_pred in each run')
    args = parser.parse_args(argv)
    corpora = {}
    try:
        for path in args.files:
            with open(path) as f:
                results = json.load(f)
            g = results['granularity']
            if g not in corpora:
                corpora[g] = load_split_corpus(g)
            out = apply(results, corpora[g], args.retype, args.save_predictions)
            name = os.path.join(args.out_dir or os.path.dirname(path) or '.',
                                results_name(g, out['protocol'], out['variant']))
            if os.path.realpath(name) == os.path.realpath(path):
                raise ConsistencyError(f'{name} is the input; choose another --out-dir')
            with open(name, 'w') as f:
                json.dump(out, f, indent=2)
            before = statistics.mean(r['input_f1'] for r in out['runs'])
            print(f"{os.path.basename(name)}: micro F1 {out['summary']['micro']['mean']:.2f} "
                  f"(input {before:.2f}) over {len(out['runs'])} runs; "
                  f"{sum(r['n_added'] for r in out['runs'])} entities added, "
                  f"{sum(r['n_retyped'] for r in out['runs'])} retyped")
    except (ConsistencyError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
