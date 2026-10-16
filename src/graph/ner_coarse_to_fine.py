"""
ner_coarse_to_fine.py - coarse-to-fine completion of fine-grained test predictions

A rule for the saved test predictions of both MaterioMiner granularities, fixed before it is
applied to any real predictions (ROADMAP §4, week 7, part 2c). No training.

Rule. For a leave-one-paper-out fold (the run_id is the held-out paper), a mapping comes from
the gold annotations of the other papers, all their sentences, both granularities: a
coarse-grained type C maps to the fine-grained type F if C has at least MIN_SPANS spans there
and at least SHARE of them are fine-grained entities of type F with the same span. In each
test sentence, a coarse-grained predicted entity of a mapped type whose words are all
predicted O at the fine-grained level becomes a fine-grained entity of type F. No
fine-grained prediction is removed or changed. A sentence that the two corpora tokenise
differently (15 in MaterioMiner) cannot be aligned word by word: it counts for neither the
mapping nor the completion, and its fine-grained predictions stay as they are.

Inputs: pairs of results files (--fine, --coarse, paired by position) of the same seed and
kind of system, with saved predictions (ner_baseline.py --save-predictions, or a vote of
ner_ensemble.py --save-predictions), valid IOB2, on the same test sentences per run.

Output: ner_results_fine_leave_one_paper_out_<variant>.json next to the fine-grained input
(--out-dir overrides), the variant being the fine-grained input's with _cf before its seed tag
(crf_m300 gives crf_cf_m300, crf_mt_ens3 gives crf_mt_ens3_cf). Each run has the seqeval
metrics, the number of entities added (n_added), the input's f1 and f1_strict (input_f1,
input_f1_strict) and the fold's mapping; with --save-predictions also test_index, test_gold
and the new test_pred.

Usage:
  python ner_coarse_to_fine.py --fine FILE [FILE ...] --coarse FILE [FILE ...] [--out-dir DIR]
      [--save-predictions]
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter

from ner_baseline import granularity_data, parse_args, score, summarize
from ner_ensemble import MEMBER_TAG, results_name, spans_to_tags, tags_to_spans

RULE = 'coarse_to_fine_completion'
MIN_SPANS = 10
SHARE = 0.9
COPIED_KEYS = ('granularity', 'protocol', 'model', 'labels', 'label_tie', 'paper_test_f1')


class CompletionError(ValueError):
    """The results files cannot be used (reported without a traceback)."""


# ---------------------------------------------------------------------------
# The rule (pure Python)
# ---------------------------------------------------------------------------
def type_mapping(fine, coarse, min_spans=MIN_SPANS, share=SHARE):
    """{coarse type: fine type} from gold sentences: fine and coarse are lists of (words,
    tags, ...) for the same corpus positions; a pair tokenised differently is left out. A
    coarse type maps if it has at least min_spans spans and at least `share` of them are fine
    entities of one type, same span."""
    seen, same = Counter(), {}
    for f, c in zip(fine, coarse, strict=True):
        if list(f[0]) != list(c[0]):
            continue                                # not aligned word by word
        fine_spans = {(s, e): k for s, e, k in tags_to_spans(f[1])[0]}
        for s, e, k in tags_to_spans(c[1])[0]:
            seen[k] += 1
            if (s, e) in fine_spans:
                same.setdefault(k, Counter())[fine_spans[(s, e)]] += 1
    mapping = {}
    for kind, n in seen.items():
        if n >= min_spans and kind in same:
            target, count = same[kind].most_common(1)[0]
            if count >= share * n:
                mapping[kind] = target
    return dict(sorted(mapping.items()))


def complete(fine_tags, coarse_tags, mapping):
    """(new fine tags, n_added) of one sentence: each coarse predicted entity of a mapped
    type whose words are all O at the fine level becomes a fine entity of the mapped type."""
    if len(fine_tags) != len(coarse_tags):
        raise CompletionError('fine-grained and coarse-grained tags of different lengths')
    spans = set(tags_to_spans(fine_tags)[0])
    added = 0
    for s, e, kind in tags_to_spans(coarse_tags)[0]:
        if kind in mapping and all(t == 'O' for t in fine_tags[s:e]):
            spans.add((s, e, mapping[kind]))
            added += 1
    if not added:
        return list(fine_tags), 0
    return spans_to_tags(sorted(spans), len(fine_tags)), added


# ---------------------------------------------------------------------------
# Results files
# ---------------------------------------------------------------------------
def cf_variant(variant):
    """The output variant: _cf before the input's seed tag m<N>, else at the end."""
    tag = MEMBER_TAG.search(variant)
    if tag is None:
        return variant + '_cf'
    if tag.start() == 0:                            # a seed tag alone (m100)
        return 'cf_' + variant
    return variant[:tag.start()] + '_cf' + variant[tag.start():]


def gold_corpora():
    """The gold sentences of both granularities as the runs saw them (the fine-grained
    labels restricted to the default top-k classes), with the fine-grained labels."""
    args = parse_args(['--crf'])
    fine, labels, _ = granularity_data('fine', args)
    coarse, _, _ = granularity_data('coarse', args)
    if len(fine) != len(coarse):
        raise CompletionError('the fine-grained and coarse-grained corpora differ in length')
    return fine, coarse, labels


def apply(fine_results, coarse_results, corpora, save_predictions=False):
    """The rule on every run of a pair of parsed results files (ner_baseline schema)."""
    fine, coarse, labels = corpora
    for name, res, granularity in (('fine', fine_results, 'fine'),
                                   ('coarse', coarse_results, 'coarse')):
        if res.get('granularity') != granularity:
            raise CompletionError(f'--{name}: a {res.get("granularity")} file')
        if res.get('protocol') != 'leave_one_paper_out':
            raise CompletionError(f'--{name}: not a leave-one-paper-out file')
    if fine_results['labels'] != labels:
        raise CompletionError('the fine-grained labels differ from the corpus top-k classes')
    coarse_runs = {r['run_id']: r for r in coarse_results['runs']}
    out = {k: fine_results[k] for k in COPIED_KEYS if k in fine_results}
    out.update({'variant': cf_variant(fine_results['variant']),
                'config': {'input': fine_results['variant'],
                           'coarse_input': coarse_results['variant'], 'rule': RULE,
                           'min_spans': MIN_SPANS, 'share': SHARE}})
    runs = []
    for run in fine_results['runs']:
        paper = run['run_id']
        other = coarse_runs.get(paper)
        if other is None:
            raise CompletionError(f'run {paper}: not in the coarse-grained file')
        for r in (run, other):
            missing = [k for k in ('test_index', 'test_gold', 'test_pred') if k not in r]
            if missing:
                raise CompletionError(f"run {paper}: no {', '.join(missing)} (run with "
                                      f"--save-predictions)")
        if run['test_index'] != other['test_index']:
            raise CompletionError(f'run {paper}: the two files test different sentences')
        if any(fine[i][2] != paper for i in run['test_index']):
            raise CompletionError(f'run {paper}: test sentences from another paper')
        train = [i for i in range(len(fine)) if fine[i][2] != paper]
        mapping = type_mapping([fine[i] for i in train], [coarse[i] for i in train])
        gold = [g.split() for g in run['test_gold']]
        pred = [p.split() for p in run['test_pred']]
        coarse_pred = [p.split() for p in other['test_pred']]
        if any(tags_to_spans(p)[1] for p in pred + coarse_pred):
            raise CompletionError(f'run {paper}: the predictions are not valid IOB2')
        new, added, unaligned = [], 0, 0
        for k, i in enumerate(run['test_index']):
            if len(fine[i][0]) != len(gold[k]) or len(coarse[i][0]) != len(coarse_pred[k]):
                raise CompletionError(f'run {paper}: test sentence {i} does not match the corpus')
            if list(fine[i][0]) != list(coarse[i][0]):
                new.append(pred[k])                     # not aligned word by word: unchanged
                unaligned += 1
                continue
            tags, n = complete(pred[k], coarse_pred[k], mapping)
            new.append(tags)
            added += n
        res = score(gold, new)
        res.update({'run_id': paper, 'n_added': added, 'n_unaligned': unaligned,
                    'input_f1': run['f1'], 'input_f1_strict': run['f1_strict'],
                    'mapping': mapping})
        if save_predictions:
            res.update({'test_index': run['test_index'], 'test_gold': run['test_gold'],
                        'test_pred': [' '.join(t) for t in new]})
        runs.append(res)
    out['runs'] = runs
    summarize(out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Coarse-to-fine completion of fine-grained test predictions')
    parser.add_argument('--fine', nargs='+', required=True, metavar='FILE',
                        help='fine-grained results files with test predictions')
    parser.add_argument('--coarse', nargs='+', required=True, metavar='FILE',
                        help='the coarse-grained files of the same seeds, in the same order')
    parser.add_argument('--out-dir', default=None, metavar='DIR',
                        help='where to write (default: next to each fine-grained input)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='keep test_index, test_gold and the new test_pred in each run')
    args = parser.parse_args(argv)
    if len(args.fine) != len(args.coarse):
        parser.error('--fine and --coarse need as many files each')
    try:
        corpora = gold_corpora()
        for fine_path, coarse_path in zip(args.fine, args.coarse):
            with open(fine_path) as f:
                fine_results = json.load(f)
            with open(coarse_path) as f:
                coarse_results = json.load(f)
            out = apply(fine_results, coarse_results, corpora, args.save_predictions)
            name = os.path.join(args.out_dir or os.path.dirname(fine_path) or '.',
                                results_name('fine', out['protocol'], out['variant']))
            if os.path.realpath(name) == os.path.realpath(fine_path):
                raise CompletionError(f'{name} is the input; choose another --out-dir')
            with open(name, 'w') as f:
                json.dump(out, f, indent=2)
            before = statistics.mean(r['input_f1'] for r in out['runs'])
            print(f"{os.path.basename(name)}: micro F1 {out['summary']['micro']['mean']:.2f} "
                  f"(input {before:.2f}) over {len(out['runs'])} runs; "
                  f"{sum(r['n_added'] for r in out['runs'])} entities added")
    except (CompletionError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
