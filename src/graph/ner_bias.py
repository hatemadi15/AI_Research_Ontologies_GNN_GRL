"""
ner_bias.py - the entity bias of each test set, chosen without its labels, and the runs at it

Runs of ner_baseline.py --entity-bias store, per bias D added to the emission of every tag
but O before the Viterbi decode, the entity counts [correct, predicted, gold] of each test
paper (bias_test) and of the validation sentences (bias_val), and with --save-predictions
the test predictions (test_pred_bias). This script chooses a D under a rule fixed before
any run (ROADMAP §4, week 7) and writes the runs at it in the results-file schema, so that
ner_ensemble.py votes over them and ner_seedmeans.py pairs them with the D = 0 runs (their
input, the CRF head). No training.

Choice (--choose):
  inner  for leave-one-paper-out runs (MaterioMiner). For the held-out paper P, the inner
         folds of --leave-two-papers-out (--inner FILE ...) that hold out P and another paper
         Q: each trains on the two papers left, i.e. on two of P's three training papers, and
         is scored on Q, never on P. D_P has the highest mean F1 over the three Q and the
         inner files (their model seeds); the held-out paper's own labels are never read.
  val    for any protocol, each run on its own: the D with the highest F1 on the run's
         validation sentences (SOFC-Exp's official split, whose validation articles are
         unseen articles; on MaterioMiner's papers the validation sentences come from the
         training papers).
  F1 is micro F1 from the counts, 200 * correct / (predicted + gold). Ties go to the D
  nearest 0, then to the smaller. Every file must hold the same biases.

Output: per input file the variant with _bias replaced by _eb (inner) or _ebv (val), e.g.
crf_bias_m300 gives crf_eb_m300, next to the input (--out-dir overrides). Each run is
rescored (ner_baseline.score) from its test predictions at its D and stores bias (D),
input_f1 and input_f1_strict (its D = 0 scores) and entities ({input, output, gold}: the
predicted entities at 0 and at D, and the gold ones); with --save-predictions also
test_index, test_gold and the test_pred at D. And ner_entity_bias_<granularity>_<protocol>
_<choice>.json: per test set the curve the choice used (mean F1 per D), the chosen D, and
the inputs' mean test F1 per D (reported, never used to choose).

Usage (the files first: --inner takes every file after it):
  python ner_bias.py FILE [FILE ...] --choose inner --inner INNER_FILE [...] [--out-dir DIR]
  python ner_bias.py FILE [FILE ...] --choose val [--out-dir DIR] [--save-predictions]
"""

import argparse
import json
import os
import statistics
import sys

from ner_baseline import score, summarize
from ner_ensemble import results_name

COPIED_KEYS = ('granularity', 'protocol', 'model', 'labels', 'label_tie', 'paper_test_f1')
SUFFIX = {'inner': '_eb', 'val': '_ebv'}


class BiasError(ValueError):
    """The results files cannot be used (reported without a traceback)."""


def f1(counts):
    """Micro F1 from [correct, predicted, gold] entity counts."""
    correct, n_pred, n_gold = counts
    return 200 * correct / (n_pred + n_gold) if n_pred + n_gold else 0.0


def best_bias(curve):
    """The bias (a key of `curve`, {bias key: mean F1}) with the highest F1; ties go to the
    bias nearest 0, then to the smaller."""
    return max(curve, key=lambda k: (round(curve[k], 9), -abs(float(k)), -float(k)))


def bias_keys(results, where):
    """The biases of every run of a parsed results file (all runs must hold the same)."""
    keys = None
    for run in results['runs']:
        if 'bias_test' not in run:
            raise BiasError(f"{where}: run {run.get('run_id')} has no bias_test (run with "
                            f"--entity-bias)")
        if keys is None:
            keys = list(run['bias_test'])
        elif list(run['bias_test']) != keys or list(run['bias_val']) != keys:
            raise BiasError(f'{where}: runs with different biases')
    if not keys or '0' not in keys:
        raise BiasError(f'{where}: no runs, or no bias 0')
    return keys


def inner_curves(inner, papers, keys):
    """{P: {bias: mean F1}} for each held-out paper P of `papers` from the parsed inner files
    (folds '<A>+<B>' of --leave-two-papers-out): the folds holding out P and a Q, scored on
    Q. The Q are every other paper of the inner folds."""
    curves = {}
    for paper in papers:
        scores = {k: [] for k in keys}
        for where, results in inner:
            folds = {tuple(r['run_id'].split('+')): r for r in results['runs']}
            others = sorted({p for fold in folds for p in fold} - {paper})
            if paper not in {p for fold in folds for p in fold} or len(others) < 2:
                raise BiasError(f'{where}: no inner folds for {paper}')
            for other in others:
                run = folds.get(tuple(sorted((paper, other))))
                if run is None:
                    raise BiasError(f'{where}: no fold holds out {paper} and {other}')
                for k in keys:
                    scores[k].append(f1(run['bias_test'][k][other]))
        curves[paper] = {k: statistics.mean(v) for k, v in scores.items()}
    return curves


def biased_run(run, bias, save_predictions):
    """A run's record at a bias: rescored from its test predictions there."""
    where = f"run {run.get('run_id')}"
    missing = [k for k in ('test_index', 'test_gold', 'test_pred_bias') if k not in run]
    if missing:
        raise BiasError(f"{where}: no {', '.join(missing)} (run with --save-predictions)")
    gold = [g.split() for g in run['test_gold']]
    pred = [p.split() for p in run['test_pred_bias'][bias]]
    if len(gold) != len(pred) or any(len(g) != len(p) for g, p in zip(gold, pred)):
        raise BiasError(f'{where}: the predictions at bias {bias} do not match test_gold')
    totals = {k: [sum(c[i] for c in run['bias_test'][k].values()) for i in range(3)]
              for k in ('0', bias)}
    res = score(gold, pred)
    res.update({'run_id': run['run_id'], 'bias': float(bias), 'input_f1': run['f1'],
                'input_f1_strict': run['f1_strict'],
                'entities': {'input': totals['0'][1], 'output': totals[bias][1],
                             'gold': totals['0'][2]}})
    if save_predictions:
        res.update({'test_index': run['test_index'], 'test_gold': run['test_gold'],
                    'test_pred': run['test_pred_bias'][bias]})
    return res


def eb_variant(variant, choice):
    """The output variant: the input's _bias replaced by _eb (inner) or _ebv (val)."""
    if variant.count('_bias') != 1:
        raise BiasError(f'variant {variant}: not an --entity-bias variant (one _bias)')
    return variant.replace('_bias', SUFFIX[choice])


def apply(results, choices, choice, save_predictions=False):
    """The parsed results file at the chosen biases ({run_id: bias key})."""
    out = {k: results[k] for k in COPIED_KEYS if k in results}
    out.update({'variant': eb_variant(results['variant'], choice),
                'config': {'input': results['variant'], 'choice': choice}})
    out['runs'] = [biased_run(run, choices[run['run_id']], save_predictions)
                   for run in results['runs']]
    summarize(out)
    return out


def test_curves(files, keys):
    """{run_id: {bias: mean test F1 over the files}}: the inputs' own curves (reported)."""
    curves = {}
    for _, results in files:
        for run in results['runs']:
            curve = curves.setdefault(run['run_id'], {k: [] for k in keys})
            for k in keys:
                totals = [sum(c[i] for c in run['bias_test'][k].values()) for i in range(3)]
                curve[k].append(f1(totals))
    return {r: {k: round(statistics.mean(v), 4) for k, v in c.items()}
            for r, c in curves.items()}


def load(paths):
    out = []
    for path in paths:
        with open(path) as f:
            out.append((path, json.load(f)))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='The entity bias of each test set, chosen without its labels')
    parser.add_argument('files', nargs='+', metavar='FILE',
                        help='results files of --entity-bias runs with --save-predictions, '
                             'one granularity and protocol')
    parser.add_argument('--choose', choices=('inner', 'val'), required=True)
    parser.add_argument('--inner', nargs='+', default=[], metavar='FILE',
                        help='with --choose inner: the --leave-two-papers-out --entity-bias '
                             'files of the same granularity')
    parser.add_argument('--out-dir', default=None, metavar='DIR',
                        help='where to write (default: next to each input)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='keep test_index, test_gold and the test_pred at the chosen bias')
    args = parser.parse_args(argv)
    try:
        files = load(args.files)
        kinds = {(r['granularity'], r['protocol']) for _, r in files}
        if len(kinds) != 1:
            raise BiasError(f'the files mix granularities or protocols: {sorted(kinds)}')
        ((granularity, protocol),) = kinds
        keys = bias_keys(files[0][1], files[0][0])
        for path, results in files:
            if bias_keys(results, path) != keys:
                raise BiasError(f'{path}: other biases than {files[0][0]}')
        summary = {'granularity': granularity, 'protocol': protocol, 'choice': args.choose,
                   'biases': [float(k) for k in keys], 'inputs': [r['variant'] for _, r in files]}
        if args.choose == 'inner':
            if protocol != 'leave_one_paper_out':
                raise BiasError('--choose inner is for leave-one-paper-out files')
            if not args.inner:
                raise BiasError('--choose inner needs --inner files')
            inner = load(args.inner)
            for path, results in inner:
                if (results['granularity'], results['protocol']) != (granularity,
                                                                     'leave_two_papers_out'):
                    raise BiasError(f'{path}: not a {granularity} --leave-two-papers-out file')
                if bias_keys(results, path) != keys or results['labels'] != files[0][1]['labels']:
                    raise BiasError(f'{path}: other biases or labels than the inputs')
            papers = sorted({r['run_id'] for _, res in files for r in res['runs']})
            curves = inner_curves(inner, papers, keys)
            chosen = {p: best_bias(c) for p, c in curves.items()}
            summary.update({'inner': [r['variant'] for _, r in inner],
                            'curves': {p: {k: round(v, 4) for k, v in c.items()}
                                       for p, c in curves.items()},
                            'chosen': {p: float(b) for p, b in chosen.items()}})
            choices = [{p: chosen[p] for p in papers} for _ in files]
        else:
            choices = [{run['run_id']: best_bias({k: f1(run['bias_val'][k]) for k in keys})
                        for run in results['runs']} for _, results in files]
            summary['chosen'] = {res['variant']: {r: float(b) for r, b in c.items()}
                                 for (_, res), c in zip(files, choices)}
        summary['test_curves'] = test_curves(files, keys)
        for (path, results), choice in zip(files, choices):
            out = apply(results, choice, args.choose, args.save_predictions)
            name = os.path.join(args.out_dir or os.path.dirname(path) or '.',
                                results_name(granularity, protocol, out['variant']))
            with open(name, 'w') as f:
                json.dump(out, f, indent=2)
            before = statistics.mean(r['input_f1'] for r in out['runs'])
            print(f"{os.path.basename(name)}: micro F1 {out['summary']['micro']['mean']:.2f} "
                  f"(at 0: {before:.2f}) over {len(out['runs'])} runs")
        name = os.path.join(args.out_dir or os.path.dirname(args.files[0]) or '.',
                            f'ner_entity_bias_{granularity}_{protocol}_{args.choose}.json')
        with open(name, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"{os.path.basename(name)}: chosen {summary['chosen']}")
    except (BiasError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
