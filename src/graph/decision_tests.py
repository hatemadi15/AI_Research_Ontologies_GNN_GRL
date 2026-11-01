"""
decision_tests.py - the t statistic and P value of each decision of paper A's names route

No training. npj Computational Materials asks for the name, n, sidedness and actual P value
of every statistical test. Paper A's twelve decisions (and D5', the post hoc reading of D5 on
MuLMS's decision seeds) were taken by the registered rule: the 95% Student-t interval of the
mean paired difference lies above 0 (for the pooled decisions, with neither corpus's interval
below 0). That rule is a one-sided t test at 2.5%. From the per-unit differences each decision
used, this module computes the t statistic with n - 1 degrees of freedom and the one- and
two-sided P values, and the exact two-sided sign-flip P value as well. It also recomputes the
interval and checks it against the stored one. It reports and changes no decision; the paper
labels the P values as computed for the journal's reporting rules. `post_hoc_stratified` (round
14 of the paper's internal review, post hoc) weights the two corpora of each pooled decision
(D5, D5', D7) equally: the mean of the two corpus means with a Welch interval.

Reads results/gpu/ner_week5_seed_analysis.json, ner_week12_comparisons.json,
ner_recall_matched.json and ner_robustness.json; writes results/gpu/ner_decision_tests.json.

Usage:
  python src/graph/decision_tests.py [--results results/gpu] [--out results/gpu/ner_decision_tests.json]
"""

import argparse
import json
import math
import os
import statistics

from ner_robustness import sign_flip_p

ANALYSIS = 'ner_week5_seed_analysis.json'
WEEK12 = 'ner_week12_comparisons.json'
WEEK18 = 'ner_recall_matched.json'
ROBUST = 'ner_robustness.json'
# the recomputed interval may differ from the stored one by the per-unit rounding (two decimals)
TOLERANCE = 0.02


def t_test(values):
    """n, mean, sd, t, df, one-sided (mean > 0) and two-sided P of a one-sample t test of the
    mean against 0, and the 95% Student-t interval."""
    from scipy import stats
    n = len(values)
    mean, sd = statistics.fmean(values), statistics.stdev(values)
    se = sd / math.sqrt(n)
    t = mean / se
    half = float(stats.t.ppf(0.975, n - 1)) * se
    return {'n_units': n, 'mean_diff': round(mean, 2), 'sd': round(sd, 2), 't': round(t, 2),
            'df': n - 1, 'p_one_sided': round(float(stats.t.sf(t, n - 1)), 4),
            'p_two_sided': round(float(2 * stats.t.sf(abs(t), n - 1)), 4),
            'ci95': [round(mean - half, 2), round(mean + half, 2)]}


def entries(analysis, week12, matched, robust):
    """[(decision, label, stored entry with per-unit values, stored verdict)] in Table 2's order."""
    winner = analysis['week13']['week12_winner']
    mulms = analysis['mulms']['week12'][winner]['decision']
    analogue = {e['granularity']: e for e in week12.values()
                if isinstance(e, dict) and e.get('role') == 'decision'
                and e.get('protocol') == 'leave_one_paper_out'}
    pooled = analysis['week13']['pooled'][winner]
    copies = analysis['week14']['copies']['pooled']
    topic = analysis['week14']['topic']['decision']
    week17 = analysis['week17']
    d18 = matched['decisions']
    unit = lambda e: dict(e, per_run=list(e['per_unit'].values()))  # noqa: E731
    return [
        ('D1', 'MuLMS: MR6 + names vs CRF head', mulms['minus_crf']['micro'], mulms['decision_1']),
        ('D2', 'MuLMS: MR6 + names vs MR3', mulms['minus_crf_mr']['micro'], mulms['decision_2']),
        ('D3', 'MaterioMiner fine: MR6 + names vs CRF head', analogue['fine']['micro'],
         analogue['fine']['micro']['ci95'][0] > 0),
        ('D4', 'MaterioMiner coarse: MR6 + names vs CRF head', analogue['coarse']['micro'],
         analogue['coarse']['micro']['ci95'][0] > 0),
        ('D5', 'pooled 11 units (MuLMS seeds 0-900): MR6 + names vs CRF head',
         pooled['minus_crf'], pooled['passes'] and pooled['no_corpus_loses']),
        ("D5'", 'pooled 11 units (MuLMS seeds 300-900, post hoc reading): MR6 + names vs CRF head',
         robust['pooled']['decision']['names_vs_crf'],
         robust['pooled']['decision']['names_vs_crf']['ci95'][0] > 0),
        ('D6', 'SOFC-Exp folds: MR6 + names vs CRF head',
         analysis['week13']['folds']['crf_mrm6']['minus_crf']['micro'],
         analysis['week13']['folds']['crf_mrm6']['passes']),
        ('D7', 'pooled 11 units: MR6 + names vs MR6', copies['minus_control'],
         copies['passes'] and copies['no_corpus_loses']),
        ('D8', 'MuLMS: own-topic vs mixed corpus', topic['minus_crf_mrm_mixed6']['micro'],
         topic['passes']),
        ('D9', 'MaterioMiner fine: multi-task + MR6 + names vs multi-task + MR6',
         week17['fine']['minus_crf_mt_mr6']['micro'], week17['fine']['passes']),
        ('D10', 'MaterioMiner coarse: multi-task + MR6 + names vs multi-task + MR6',
         week17['coarse']['minus_crf_mt_mr6']['micro'], week17['coarse']['passes']),
        ('D11', 'MuLMS: MR6 + names vs recall-matched MR6, recall on the unseen names the lexicon holds',
         unit(d18['18.1']), d18['18.1']['verdict'] == 'passes'),
        ('D12', 'MuLMS: MR6 + names vs recall-matched MR6, micro F1', unit(d18['18.2']),
         d18['18.2']['verdict'] == 'passes'),
    ]


def decision_tests(analysis, week12, matched, robust):
    """Per decision: the t test, the sign-flip P, the stored interval and verdict, and whether
    the recomputed interval matches the stored one and the one-sided P agrees with the verdict
    (P < 0.025 exactly when the interval clears 0; the pooled decisions add a corpus condition)."""
    rows = []
    for key, label, stored, passes in entries(analysis, week12, matched, robust):
        values = [float(v) for v in stored['per_run']]
        test = t_test(values)
        matches = all(abs(a - b) <= TOLERANCE for a, b in zip(test['ci95'], stored['ci95']))
        clears = test['p_one_sided'] < 0.025
        rows.append(dict(test, decision=key, comparison=label, per_unit=values,
                         stored_ci95=list(stored['ci95']), stored_mean_diff=stored['mean_diff'],
                         sign_flip_p_two_sided=sign_flip_p(values), passes=bool(passes),
                         interval_matches=matches,
                         p_agrees=clears == (stored['ci95'][0] > 0)))
    return rows


def welch_average(a, b):
    """The mean of two groups' means with its 95% Welch interval (Welch-Satterthwaite degrees
    of freedom), each group weighted equally whatever its size."""
    from scipy import stats
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    mean = (statistics.fmean(a) + statistics.fmean(b)) / 2
    se = 0.5 * math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = float(stats.t.ppf(0.975, df)) * se
    return {'mean': round(mean, 2), 'ci95': [round(mean - half, 2), round(mean + half, 2)],
            'df': round(df, 1)}


def stratified(rows, first=7):
    """Per pooled decision (11 units: MuLMS's 7 subfields first, then MaterioMiner's 4 papers),
    the corpus-weighted mean of its per-unit differences with the Welch interval."""
    return {r['decision']: welch_average(r['per_unit'][:first], r['per_unit'][first:])
            for r in rows if r['n_units'] == 11}


def load(path):
    with open(path) as f:
        return json.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(description="t statistics and P values of paper A's decisions")
    root = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'gpu')
    parser.add_argument('--results', default=root)
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    rows = decision_tests(*(load(os.path.join(args.results, name))
                            for name in (ANALYSIS, WEEK12, WEEK18, ROBUST)))
    out = {'note': ('Reporting only, computed on 1 November 2026 for the journal: the decisions '
                    'were taken by the registered interval rule, which is a one-sided t test at '
                    '2.5%; the P values are those of the same t tests on the same per-unit '
                    'differences, with the exact two-sided sign-flip P beside them.'),
           'decisions': rows,
           'post_hoc_stratified': {'note': ('post hoc: each pooled decision with its two corpora '
                                            'weighted equally, the mean of the corpus means and '
                                            'its 95% Welch interval'),
                                   'decisions': stratified(rows)}}
    path = args.out or os.path.join(args.results, 'ner_decision_tests.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    for r in rows:
        flags = '' if r['interval_matches'] and r['p_agrees'] else '  CHECK'
        print(f"{r['decision']:4s} n={r['n_units']:2d} {r['mean_diff']:+6.2f} [{r['ci95'][0]:+.2f}, "
              f"{r['ci95'][1]:+.2f}] (stored [{r['stored_ci95'][0]:+.2f}, {r['stored_ci95'][1]:+.2f}]) "
              f"t({r['df']}) = {r['t']:.2f}, P one-sided {r['p_one_sided']:.4f}, two-sided "
              f"{r['p_two_sided']:.4f}, sign-flip {r['sign_flip_p_two_sided']:.4f}, "
              f"{'passes' if r['passes'] else 'no'}{flags}")
    for key, e in out['post_hoc_stratified']['decisions'].items():
        print(f"{key:4s} corpora weighted equally: {e['mean']:+.2f} [{e['ci95'][0]:+.2f}, "
              f"{e['ci95'][1]:+.2f}] (Welch, df {e['df']})")


if __name__ == '__main__':
    main()
