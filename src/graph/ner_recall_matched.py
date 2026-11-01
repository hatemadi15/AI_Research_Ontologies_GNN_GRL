"""The names against a recall-matched control (week 18; ROADMAP §4, the week-18 protocol).

MR6 on MuLMS's 7 held-out subfields at the decision seeds 300 to 900, the stored crf_mr6 jobs run
again with --entity-bias over a grid G (files crf_mr6_bias_m<N>: the test predictions per bias,
test_pred_bias), against the stored names runs (crf_mrm_topic6_m<N>). Per subfield u, D*_u is the
bias at which the control's recall over all test entities, pooled over the seeds, reaches the
names' pooled recall: the first crossing going from D = 0 toward it along G, by linear
interpolation between the two grid values that bracket it (weight w). A subfield whose target lies
outside the control's range along that path is matched at the end of the path and flagged. Every
quantity of the control at D*_u is the same interpolation of its grid values: counts pooled over
the seeds for pooled rates, each run's micro F1 for the seed mean.

Decisions (paired over the 7 subfields, the 95% Student-t interval above 0):
  18.1  recall on the unseen test names the lexicon holds (week 15's split, ner_lexicon_recall),
        names minus the control at D*_u;
  18.2  micro F1 (the mean over the seeds), names minus the control at D*_u.
Pre-specified, no decision: precision on all test entities; recall, precision and F1 on the unseen
test names; recall on the unseen names the lexicon does not hold and in - out; the bias's own F1
change (the control at D*_u minus at 0); D*_u, w and the flags; the control's curves.

Post hoc (added after the result, 1 November): the protocol's "nearer end of G" for a flagged
subfield assumed the control's recall to rise with the bias; in the three flagged subfields it
peaks at D = 2.5 or 3 and falls after, so the end of G (5) puts the control 2.4 to 2.9 points of
recall below the names. post_hoc_nearest matches a flagged subfield instead at the grid value
whose recall comes nearest the names' (match_nearest); the other subfields are unchanged.

Checks before any number is used: every control run at D = 0 equals its stored crf_mr6 run (F1,
precision, recall, best epoch, training history, test predictions, and test_pred_bias at 0); the
rebuilt lexicons pass week 15's checks against the stored names runs. The control is read from
the new runs at every D either way.

Inputs: results/gpu (the three families) and results/gpu_names (the rebuilt lexicons, not
committed). Output: results/gpu/ner_recall_matched.json, aggregate counts and scores only.
"""
import argparse
import json
import os
from collections import Counter

import metrics
import ner_lexicon_recall as lr
import ner_mentions as nm
from ner_baseline import entity_counts
from ner_seedmeans import DECISION_OFFSETS

OUTPUT = nm.RESULTS_DIR / 'ner_recall_matched.json'
GRANULARITY, PROTOCOL = 'mulms', 'leave_one_subfield_out'
CONTROL, NAMES, STORED = 'crf_mr6_bias', 'crf_mrm_topic6', 'crf_mr6'
NAMES_PATTERN = lr.SETTINGS['mulms'][5]
D0_FIELDS = ('f1', 'precision', 'recall', 'best_epoch', 'history', 'test_pred')


def run_counts(words_of, test, gold_tags, pred_tags, forms, lexicon):
    """The Counter of one run's test predictions: ('all', 'correct'|'pred'|'gold') over all test
    entities (seqeval's counting, entity_counts) and week 15's unseen split by lexicon
    membership (lr.sentence_counts: ('gold'|'correct'|'pred'|'pred_correct', 'in'|'out'))."""
    golds = [g.split() for g in gold_tags]
    preds = [p.split() for p in pred_tags]
    correct, pred, gold = entity_counts(golds, preds)
    counts = Counter({('all', 'correct'): correct, ('all', 'pred'): pred, ('all', 'gold'): gold})
    for i, g, p in zip(test, golds, preds, strict=True):
        counts.update(lr.sentence_counts(words_of(i), g, p, forms, lexicon))
    return counts


def f1_of(counts):
    """Micro F1 (in %) of a Counter's all-entity counts, as seqeval computes it."""
    den = counts['all', 'pred'] + counts['all', 'gold']
    return 200 * counts['all', 'correct'] / den if den else 0.0


def rates(counts):
    """Pooled rates (in %) of a Counter: recall and precision over all test entities, recall,
    precision and F1 on the unseen names, recall on the names in and out of the lexicon."""
    def rate(num, den):
        return 100 * num / den if den else None
    unseen = {k: counts[k, 'in'] + counts[k, 'out'] for k in ('gold', 'correct', 'pred',
                                                              'pred_correct')}
    r_u, p_u = rate(unseen['correct'], unseen['gold']), rate(unseen['pred_correct'], unseen['pred'])
    return {'recall': rate(counts['all', 'correct'], counts['all', 'gold']),
            'precision': rate(counts['all', 'correct'], counts['all', 'pred']),
            'unseen_recall': r_u, 'unseen_precision': p_u,
            'unseen_f1': 2 * r_u * p_u / (r_u + p_u) if r_u and p_u else 0.0,
            'recall_in': rate(counts['correct', 'in'], counts['gold', 'in']),
            'recall_out': rate(counts['correct', 'out'], counts['gold', 'out'])}


def match(curve, target):
    """(D_a, D_b, w, flagged): the first crossing of `target` by the recall curve {D: recall},
    going from D = 0 toward it along the grid, w the weight of D_b in the linear interpolation
    (target = (1 - w) curve[D_a] + w curve[D_b]); without a crossing, the end of the path,
    flagged."""
    if curve[0.0] == target:
        return 0.0, 0.0, 0.0, False
    grid = sorted(curve)
    up = target > curve[0.0]
    path = [d for d in grid if d >= 0] if up else [d for d in reversed(grid) if d <= 0]
    for prev, d in zip(path, path[1:]):
        if (curve[d] >= target) if up else (curve[d] <= target):
            ra, rb = curve[prev], curve[d]
            return prev, d, ((target - ra) / (rb - ra) if rb != ra else 0.0), False
    return path[-1], path[-1], 0.0, True


def match_nearest(curve, target):
    """Post hoc (module docstring): as match, but a flagged subfield is matched at the grid value
    whose recall comes nearest the target (the smaller |D| on a tie), still flagged."""
    d_a, d_b, w, flagged = match(curve, target)
    if not flagged:
        return d_a, d_b, w, flagged
    d = min(curve, key=lambda x: (abs(curve[x] - target), abs(x)))
    return d, d, 0.0, True


def blend(a, b, w):
    """(1 - w) a + w b of two Counters (or numbers)."""
    if isinstance(a, Counter):
        return Counter({k: (1 - w) * a[k] + w * b[k] for k in set(a) | set(b)})
    return (1 - w) * a + w * b


def paired(values):
    """The mean paired difference of per-unit values rounded to 2 decimals, its 95% CI, the
    units above 0 and the verdict of the decision rule."""
    vals = [round(v, 2) for v in values.values()]
    ci = metrics.mean_ci(vals)
    lo, hi = (round(x, 2) for x in ci['ci'])
    return {'per_unit': {u: round(v, 2) for u, v in values.items()},
            'mean_diff': round(ci['mean'], 2), 'ci95': [lo, hi], 'n_units': len(vals),
            'n_positive': sum(v > 0 for v in vals), 'above_0': lo > 0}


def d0_check(control_run, stored_run):
    """The fields of D0_FIELDS (and test_pred_bias at 0) that differ between a control run and
    its stored crf_mr6 run; [] when identical."""
    diff = [k for k in D0_FIELDS if control_run.get(k) != stored_run.get(k)]
    if control_run.get('test_pred_bias', {}).get('0') != control_run.get('test_pred'):
        diff.append('test_pred_bias[0]')
    return diff


def unit_analysis(names_runs, control_runs, matcher=match):
    """Per subfield: the matching (by `matcher`) and every compared quantity. names_runs: [Counter]
    per seed; control_runs: [{D: Counter}] per seed, the same seeds in the same order."""
    pooled_names = sum(names_runs, Counter())
    grid = sorted(control_runs[0])
    pooled = {d: sum((r[d] for r in control_runs), Counter()) for d in grid}
    curve = {d: rates(pooled[d])['recall'] for d in grid}
    target = rates(pooled_names)['recall']
    d_a, d_b, w, flagged = matcher(curve, target)
    at = blend(pooled[d_a], pooled[d_b], w)
    f1_control = [blend(f1_of(r[d_a]), f1_of(r[d_b]), w) for r in control_runs]
    f1_names = [f1_of(r) for r in names_runs]
    names, control = rates(pooled_names), rates(at)
    mean = sum(f1_control) / len(f1_control)
    return {'target_recall': round(target, 2), 'control_recall_at_0': round(curve[0.0], 2),
            'd_star': round(blend(d_a, d_b, w), 3), 'between': [d_a, d_b],
            'w': round(w, 4), 'flagged': flagged,
            'names': {k: round(v, 2) for k, v in names.items() if v is not None}
            | {'f1': round(sum(f1_names) / len(f1_names), 2)},
            'control_at_d_star': {k: round(v, 2) for k, v in control.items() if v is not None}
            | {'f1': round(mean, 2)},
            'control_f1_at_0': round(sum(f1_of(r[0.0]) for r in control_runs)
                                     / len(control_runs), 2),
            'curve': {f'{d:g}': {'recall': round(curve[d], 2),
                                 'precision': round(rates(pooled[d])['precision'], 2),
                                 'f1_pooled': round(f1_of(pooled[d]), 2),
                                 'f1_seed_mean': round(sum(f1_of(r[d]) for r in control_runs)
                                                       / len(control_runs), 2)}
                      for d in grid},
            '_diffs': {'recall_in': names['recall_in'] - control['recall_in'],
                       'f1': sum(f1_names) / len(f1_names) - mean,
                       'precision': names['precision'] - control['precision'],
                       'unseen_recall': names['unseen_recall'] - control['unseen_recall'],
                       'unseen_precision': names['unseen_precision'] - control['unseen_precision'],
                       'unseen_f1': names['unseen_f1'] - control['unseen_f1'],
                       'recall_out': names['recall_out'] - control['recall_out'],
                       'in_minus_out': (names['recall_in'] - control['recall_in'])
                       - (names['recall_out'] - control['recall_out']),
                       'bias_f1_change': mean - sum(f1_of(r[0.0]) for r in control_runs)
                       / len(control_runs)}}


def comparisons(units, keep=None):
    """The decisions and the pre-specified comparisons over the units (all, or those in keep)."""
    keys = [u for u in units if keep is None or u in keep]
    out = {'18.1': paired({u: units[u]['_diffs']['recall_in'] for u in keys}),
           '18.2': paired({u: units[u]['_diffs']['f1'] for u in keys})}
    for d in ('18.1', '18.2'):
        out[d]['verdict'] = 'passes' if out[d]['above_0'] else 'no'
    pre = {k: paired({u: units[u]['_diffs'][k] for u in keys})
           for k in ('precision', 'unseen_recall', 'unseen_precision', 'unseen_f1', 'recall_out',
                     'in_minus_out', 'bias_f1_change')}
    return {'decisions': out, 'prespecified': pre}


def analyse(directory=nm.RESULTS_DIR, names_dir=lr.NAMES_DIR, offsets=DECISION_OFFSETS):
    """The week-18 analysis (module docstring); None without the control's files."""
    control_files = nm.family_files(directory, GRANULARITY, PROTOCOL, CONTROL, offsets)
    names_files = nm.family_files(directory, GRANULARITY, PROTOCOL, NAMES, offsets)
    stored_files = nm.family_files(directory, GRANULARITY, PROTOCOL, STORED, offsets)
    if not (control_files and names_files and stored_files):
        return None
    stores = lr.load_stores(names_dir, NAMES_PATTERN)
    lexicon_checks = lr.checks(stores, names_files)
    lexicons = {u: lr.lexicon_of(stores[u]) for u, v in lexicon_checks.items()
                if v['teachers'] and v['coverage']}
    data = nm.corpus(GRANULARITY)
    d0 = {}
    per_unit_names, per_unit_control = {}, {}
    for off, control, names, stored in zip(offsets, control_files, names_files, stored_files,
                                           strict=True):
        stored_runs = {r['run_id']: r for r in stored['runs']}
        names_runs = {r['run_id']: r for r in names['runs']}
        for run in control['runs']:
            u = run['run_id']
            d0[f'{u}_m{off}'] = d0_check(run, stored_runs.get(u, {}))
            if u not in lexicons or u not in names_runs:
                continue
            run_data, train, test = nm.run_split(control, run, data, GRANULARITY)
            forms = nm.entity_forms(run_data, train)
            def words_of(i, run_data=run_data):
                return run_data[i][0]
            by_bias = {float(key): run_counts(words_of, test, run['test_gold'], preds, forms,
                                              lexicons[u])
                       for key, preds in run['test_pred_bias'].items()}
            per_unit_control.setdefault(u, []).append(by_bias)
            n = names_runs[u]
            n_data, n_train, n_test = nm.run_split(names, n, data, GRANULARITY)
            if n_test != test or n['test_gold'] != run['test_gold']:
                raise ValueError(f'{u} m{off}: the names run and the control differ in test set')
            def names_words(i, n_data=n_data):
                return n_data[i][0]
            per_unit_names.setdefault(u, []).append(
                run_counts(names_words, n_test, n['test_gold'], n['test_pred'],
                           nm.entity_forms(n_data, n_train), lexicons[u]))
    units = {u: unit_analysis(per_unit_names[u], per_unit_control[u])
             for u in sorted(per_unit_control) if len(per_unit_control[u]) == len(offsets)}
    result = {'protocol': 'ROADMAP §4, week 18', 'seeds': list(offsets),
              'grid': sorted({float(k) for f in control_files for r in f['runs']
                              for k in r['test_pred_bias']}),
              'units': sorted(units),
              'checks': {'d0_identical': all(not v for v in d0.values()),
                         'd0_differences': {k: v for k, v in d0.items() if v},
                         'n_runs_checked': len(d0), 'lexicons': lexicon_checks},
              **comparisons(units)}
    flagged = [u for u in units if units[u]['flagged']]
    if flagged:
        result['without_flagged'] = {'flagged': flagged,
                                     **comparisons(units, keep=set(units) - set(flagged))}
        nearest = {u: unit_analysis(per_unit_names[u], per_unit_control[u], match_nearest)
                   for u in units}
        result['post_hoc_nearest'] = {
            'note': 'post hoc: each flagged subfield matched at the grid value whose pooled '
                    "recall comes nearest the names' (module docstring)",
            'matched': {u: {'d': nearest[u]['d_star'],
                            'control_recall': nearest[u]['control_at_d_star']['recall'],
                            'target_recall': nearest[u]['target_recall']} for u in flagged},
            **comparisons(nearest)}
    result['per_unit'] = {u: {k: v for k, v in units[u].items() if k != '_diffs'}
                          for u in units}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--results', default=str(nm.RESULTS_DIR))
    parser.add_argument('--names', default=str(lr.NAMES_DIR))
    parser.add_argument('--out', default=str(OUTPUT))
    args = parser.parse_args(argv)
    out = analyse(args.results, args.names)
    if out is None:
        raise SystemExit(f'no {CONTROL} files in {args.results} (kaggle_jobs.py collect after w18a)')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    c = out['checks']
    print(f"checks: D = 0 identical in {c['n_runs_checked'] - len(c['d0_differences'])} of "
          f"{c['n_runs_checked']} runs; units {out['units']}")
    for key, d in out['decisions'].items():
        print(f"  {key}: {d['mean_diff']:+.2f} {d['ci95']} ({d['n_positive']}/{d['n_units']} "
              f"above 0): {d['verdict']}")
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
