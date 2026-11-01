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

`post_hoc_dependence` (round 15, post hoc) drops the t test's assumption that the units'
differences are independent where their training sets share documents: MaterioMiner's papers
(leave one paper out, so any two training sets share two of three papers), SOFC-Exp's folds
(three of four folds) and the MuLMS subfields whose 400-sentence training pools share an
article (MULMS_POOLS). Such units get the correlation that the corrected resampled t test
(Nadeau and Bengio 2003) assumes for k-fold designs, 1/K (1/4 and 1/5), and the MuLMS pairs that
share an article MaterioMiner's 1/4; other pairs, and pairs across corpora, 0. Under that
model the variance of the mean of n units is S^2 / (1 - rho_bar) * (n + sum of rho_ij) / n^2,
which for K units all at 1/K is the corrected test's S^2 (1/K + 1/(K - 1)); the t statistic
keeps n - 1 degrees of freedom. `post_hoc_reported` gives the t test of mention replacement's
pooled gain over the CRF head (MR3, 11 units, both seed bases), which the paper quotes.

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
LEXICON = 'ner_lexicon_recall.json'
MENTIONS = 'ner_mentions.json'
# the recomputed interval may differ from the stored one by the per-unit rounding (two decimals)
TOLERANCE = 0.02
# the verdict printed for an interval above 0 where "passes" would mislead: D5' is a post hoc
# reading, not a decision, and D11 passes as registered only with three flagged subfields
VERDICTS = {"D5'": 'above 0 (post hoc)', 'D11': 'passes (as registered)'}
# Each MuLMS held-out subfield's training pool: the articles of ner_baseline.subfield_folds
# (split seed 0, --train-pool 400, MuLMS at revision bc1c97d) in pool order, the last one cut.
# tests/test_decision_tests.py recomputes them when data/processed/mulms is present.
MULMS_POOLS = {'electrolysis': ('graphene_02', 'poly_11', 'sofc_01'),
               'graphene': ('steel_06', 'pemfc_02', 'steel_07'),
               'pemfc': ('oastm_02', 'poly_02'),
               'poly': ('graphene_02', 'oastm_06', 'pemfc_13'),
               'semi': ('poly_11', 'poly_09', 'poly_04'),
               'sofc': ('semi_05', 'poly_11'),
               'steel': ('sofc_04', 'pemfc_10', 'semi_05')}
MULMS_UNITS = tuple(sorted(MULMS_POOLS))
# the leave-one-out designs and their number of units K: the correlation 1/K between any two
LEAVE_ONE_OUT = {'materiominer': 4, 'sofc_exp': 5}
SHARED_POOL_RHO = 0.25      # two MuLMS subfields whose pools share an article: MaterioMiner's 1/4


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
        ('D5', 'pooled 11 units (MuLMS seeds 0–900): MR6 + names vs CRF head',
         pooled['minus_crf'], pooled['passes'] and pooled['no_corpus_loses']),
        ("D5'", 'pooled 11 units (MuLMS seeds 300–900, post hoc reading): MR6 + names vs CRF head',
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
                         verdict=VERDICTS.get(key, 'passes') if passes else 'no',
                         interval_matches=matches,
                         p_agrees=clears == (stored['ci95'][0] > 0)))
    return rows


def layout(n):
    """[(corpus, unit)] of a decision's n per-unit values, in the stored order: MuLMS's 7
    subfields (sorted), MaterioMiner's 4 papers, SOFC-Exp's 5 folds, or the 11 pooled units
    (MuLMS's first, then MaterioMiner's)."""
    mulms = [('mulms', u) for u in MULMS_UNITS]
    papers = [('materiominer', i) for i in range(LEAVE_ONE_OUT['materiominer'])]
    folds = [('sofc_exp', i) for i in range(LEAVE_ONE_OUT['sofc_exp'])]
    layouts = {len(mulms): mulms, len(papers): papers, len(folds): folds,
               len(mulms) + len(papers): mulms + papers}
    if n not in layouts:
        raise ValueError(f'no unit layout for {n} units')
    return layouts[n]


def pair_rho(a, b, rho=None):
    """The assumed correlation of two units' differences: 1/K within a leave-one-out design,
    SHARED_POOL_RHO for two MuLMS subfields whose pools share an article, else 0. rho:
    {'materiominer', 'sofc_exp', 'mulms': correlation} overriding those values."""
    rho = rho or {}
    if a[0] != b[0]:
        return 0.0
    if a[0] in LEAVE_ONE_OUT:
        return rho.get(a[0], 1 / LEAVE_ONE_OUT[a[0]])
    shared = set(MULMS_POOLS[a[1]]) & set(MULMS_POOLS[b[1]])
    return rho.get('mulms', SHARED_POOL_RHO) if shared else 0.0


def dependent_t(values, units, rho=None):
    """The t test of the mean of `values` when unit pairs are correlated (pair_rho): the
    variance of the mean S^2 / (1 - rho_bar) * (n + sum_ij rho_ij) / n^2 (sum over ordered pairs
    i != j, rho_bar its mean), n - 1 degrees of freedom; with the ratio of its standard error to
    the independent one and the number of correlated pairs."""
    from scipy import stats
    n = len(values)
    mean, s2 = statistics.fmean(values), statistics.variance(values)
    rho = [pair_rho(units[i], units[j], rho) for i in range(n) for j in range(n) if i != j]
    off = sum(rho)
    se = math.sqrt(s2 / (1 - off / (n * (n - 1))) * (n + off) / n ** 2)
    t = mean / se
    half = float(stats.t.ppf(0.975, n - 1)) * se
    return {'mean_diff': round(mean, 2), 'ci95': [round(mean - half, 2), round(mean + half, 2)],
            'p_two_sided': round(float(2 * stats.t.sf(abs(t), n - 1)), 4),
            'se_ratio': round(se / math.sqrt(s2 / n), 3),
            'correlated_pairs': sum(1 for r in rho if r > 0) // 2}


def dependence(rows):
    """Per decision, dependent_t of its per-unit differences (post hoc), and whether the
    interval still lies above 0."""
    out = {}
    for r in rows:
        e = dependent_t(r['per_unit'], layout(r['n_units']))
        out[r['decision']] = dict(e, above_0=e['ci95'][0] > 0)
    return out


# the correlation each passing decision's dependence check turns on: the papers' for those
# with MaterioMiner units (the MuLMS pairs held at SHARED_POOL_RHO), the shared pools' for D11
TIPPING = {'D3': 'materiominer', 'D4': 'materiominer', 'D5': 'materiominer',
           "D5'": 'materiominer', 'D11': 'mulms'}


def tipping(rows, top=0.95, steps=60):
    """Post hoc: per passing decision, the correlation (of TIPPING's group; the others as in
    pair_rho) at which its dependence-adjusted interval reaches 0, by bisection on [0, top]; 0
    if it covers 0 without correlation, None if it stays above 0 up to `top`."""
    out = {}
    for r in rows:
        group = TIPPING.get(r['decision'])
        if not group or not r['passes']:
            continue
        units = layout(r['n_units'])

        def lower(x):
            return dependent_t(r['per_unit'], units, {group: x})['ci95'][0]
        if lower(0.0) <= 0:
            out[r['decision']] = {'group': group, 'rho': 0.0}
            continue
        if lower(top) > 0:
            out[r['decision']] = {'group': group, 'rho': None}
            continue
        lo, hi = 0.0, top
        for _ in range(steps):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if lower(mid) > 0 else (lo, mid)
        out[r['decision']] = {'group': group, 'rho': round(hi, 2)}
    return out


def shared_pool_pairs():
    """The pairs of MuLMS subfields whose training pools share an article, with the articles."""
    return [[a, b, sorted(set(MULMS_POOLS[a]) & set(MULMS_POOLS[b]))]
            for i, a in enumerate(MULMS_UNITS) for b in MULMS_UNITS[i + 1:]
            if set(MULMS_POOLS[a]) & set(MULMS_POOLS[b])]


def reported(robust):
    """The t test of MR3 minus the CRF head over the 11 pooled units on both MuLMS seed bases
    (ner_robustness.json), which the paper quotes with its P value."""
    return {basis: t_test([float(v) for v in robust['pooled'][basis]['mr3_vs_crf']['per_run']])
            for basis in ('decision', 'ten')}


def p_of(entry, n=None):
    """The two-sided P value of the one-sample t test behind a stored paired difference: from
    its per-unit values ('per_run', or 'per_unit' as a list or a {unit: value} map) if it has
    them, else from its mean and 95% Student-t interval over n units, which determine it."""
    from scipy import stats
    values = entry.get('per_run', entry.get('per_unit'))
    if isinstance(values, dict):
        values = list(values.values())
    if values:
        return t_test([float(v) for v in values])['p_two_sided']
    mean, (lo, hi) = entry.get('mean_diff', entry.get('mean')), entry['ci95']
    se = (hi - lo) / 2 / float(stats.t.ppf(0.975, n - 1))
    return round(float(2 * stats.t.sf(abs(mean / se), n - 1)), 4)


def inferences(analysis, robust, matched, lexicon, mentions):
    """Post hoc (round 16): the two-sided t test P values of the reported, non-decision
    comparisons from which the paper draws an inference, each with its stored interval; the
    unseen-name recall and precision on MuLMS (ner_mentions.json `chain`) store no per-unit
    values, so theirs come from the interval and n = 7."""
    chain = mentions['chain']['mulms_leave_one_subfield_out']['crf_mrm_topic6']['decision']['unseen']
    split = lexicon['mulms']['comparisons']['names_vs_mr6']

    def part(key):
        return {'mean_diff': split[key]['mean_diff'], 'ci95': split[key]['ci95'],
                'per_unit': {u: v[key] for u, v in split['per_unit'].items()}}
    entries = {
        'mr3_vs_crf_11_decision': robust['pooled']['decision']['mr3_vs_crf'],
        'mr6_vs_mr3_11_decision': robust['pooled']['decision']['mr6_vs_mr3'],
        'mr3_vs_crf_16': analysis['week13']['pooled']['crf_mr']['minus_crf'],
        'mr6_vs_crf_mulms_decision': analysis['week14']['copies']['mulms']['decision']['control']['minus_crf']['micro'],
        'sofc_names_vs_mr3': analysis['week13']['folds']['crf_mrm6']['minus_crf_mr']['micro'],
        'unseen_recall_names_vs_mr6': chain['recall'],
        'unseen_precision_names_vs_mr6': chain['precision'],
        'held_recall_names_vs_mr6': part('gain_in'),
        'held_in_minus_out': part('in_minus_out'),
        'held_names_vs_best_bias': matched['post_hoc_curves']['names_minus_best']['recall_in'],
        'other_unseen_at_matched_bias': matched['prespecified']['recall_out'],
    }
    out = {}
    for key, e in entries.items():
        n = len(e.get('per_run') or e.get('per_unit') or []) or 7
        out[key] = {'mean_diff': e.get('mean_diff', e.get('mean')), 'ci95': list(e['ci95']),
                    'n_units': n, 'p_two_sided': p_of(e, n)}
    return out


def mr3_materiominer(results_dir, offsets=range(300, 1000, 100)):
    """Post hoc (round 16): the companion study's MR3 decisions on MaterioMiner's held-out
    papers (crf_mr against crf, leave one paper out) on the seeds the stage-1 screen did not
    use, 300 to 900, and on all ten as decided: per granularity the t test over the papers,
    each paper's value the difference of the two systems' seed means."""
    out = {}
    for g in ('fine', 'coarse'):
        means = {}
        for system in ('crf', 'crf_mr'):
            runs = {}
            for off in offsets:
                path = os.path.join(results_dir,
                                    f'ner_results_{g}_leave_one_paper_out_{system}_m{off}.json')
                for r in load(path)['runs']:
                    runs.setdefault(r['run_id'], []).append(r['f1'])
            means[system] = {k: statistics.fmean(v) for k, v in runs.items()}
        papers = sorted(means['crf'])
        out[g] = t_test([means['crf_mr'][p] - means['crf'][p] for p in papers])
    return out


def d5_coarse(rows):
    """Post hoc: D5's pooled comparison with MaterioMiner's coarse-grained papers (D4's
    per-unit values) in place of the fine-grained ones, MuLMS on all ten seeds as in D5."""
    by = {r['decision']: r for r in rows}
    return t_test(by['D5']['per_unit'][:len(MULMS_UNITS)] + by['D4']['per_unit'])


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
    inputs = [load(os.path.join(args.results, name)) for name in (ANALYSIS, WEEK12, WEEK18, ROBUST)]
    lexicon, mentions = (load(os.path.join(args.results, name)) for name in (LEXICON, MENTIONS))
    rows = decision_tests(*inputs)
    out = {'note': ('Reporting only, computed on 1 November 2026 for the journal: the decisions '
                    'were taken by the registered interval rule, which is a one-sided t test at '
                    '2.5%; the P values are those of the same t tests on the same per-unit '
                    'differences, with the exact two-sided sign-flip P beside them.'),
           'decisions': rows,
           'post_hoc_stratified': {'note': ('post hoc: each pooled decision with its two corpora '
                                            'weighted equally, the mean of the corpus means and '
                                            'its 95% Welch interval'),
                                   'decisions': stratified(rows)},
           'post_hoc_dependence': {'note': ('post hoc (round 15): each decision with its units '
                                            'correlated where their training sets share documents, '
                                            'at the corrected resampled t test\'s 1/K (Nadeau and '
                                            'Bengio 2003): 1/4 between MaterioMiner\'s papers, 1/5 '
                                            'between SOFC-Exp\'s folds, 1/4 between MuLMS subfields '
                                            'whose pools share an article, 0 otherwise'),
                                   'rho': {'materiominer': 1 / LEAVE_ONE_OUT['materiominer'],
                                           'sofc_exp': 1 / LEAVE_ONE_OUT['sofc_exp'],
                                           'mulms_shared_pool': SHARED_POOL_RHO},
                                   'mulms_pools': {k: list(v) for k, v in MULMS_POOLS.items()},
                                   'shared_pool_pairs': shared_pool_pairs(),
                                   'decisions': dependence(rows),
                                   'tipping': tipping(rows)},
           'post_hoc_reported': {'note': ('t tests of comparisons the paper quotes with a P value: '
                                          'MR3 minus the CRF head over the 11 pooled units, on '
                                          "MuLMS's decision seeds (decision) and all ten (ten)"),
                                 'mr3_vs_crf': reported(inputs[3]),
                                 'inferences': inferences(inputs[0], inputs[3], inputs[2],
                                                          lexicon, mentions),
                                 'mr3_materiominer_decision_seeds': mr3_materiominer(args.results),
                                 'mr3_materiominer_all_seeds': mr3_materiominer(
                                     args.results, range(0, 1000, 100)),
                                 'd5_coarse': d5_coarse(rows)}}
    path = args.out or os.path.join(args.results, 'ner_decision_tests.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    for r in rows:
        flags = '' if r['interval_matches'] and r['p_agrees'] else '  CHECK'
        print(f"{r['decision']:4s} n={r['n_units']:2d} {r['mean_diff']:+6.2f} [{r['ci95'][0]:+.2f}, "
              f"{r['ci95'][1]:+.2f}] (stored [{r['stored_ci95'][0]:+.2f}, {r['stored_ci95'][1]:+.2f}]) "
              f"t({r['df']}) = {r['t']:.2f}, P one-sided {r['p_one_sided']:.4f}, two-sided "
              f"{r['p_two_sided']:.4f}, sign-flip {r['sign_flip_p_two_sided']:.4f}, "
              f"{r['verdict']}{flags}")
    for key, e in out['post_hoc_stratified']['decisions'].items():
        print(f"{key:4s} corpora weighted equally: {e['mean']:+.2f} [{e['ci95'][0]:+.2f}, "
              f"{e['ci95'][1]:+.2f}] (Welch, df {e['df']})")
    for key, e in out['post_hoc_dependence']['decisions'].items():
        print(f"{key:4s} units correlated: {e['mean_diff']:+.2f} [{e['ci95'][0]:+.2f}, "
              f"{e['ci95'][1]:+.2f}], P two-sided {e['p_two_sided']:.4f}, SE x{e['se_ratio']:.3f}, "
              f"{e['correlated_pairs']} correlated pairs{'' if e['above_0'] else ', covers 0'}")
    for key, e in out['post_hoc_dependence']['tipping'].items():
        print(f"{key:4s} interval reaches 0 at a {e['group']} correlation of {e['rho']}")
    rep = out['post_hoc_reported']
    for key, e in rep['inferences'].items():
        print(f"{key}: {e['mean_diff']:+.2f} {e['ci95']}, n {e['n_units']}, P {e['p_two_sided']:.4f}")
    for basis in ('mr3_materiominer_decision_seeds', 'mr3_materiominer_all_seeds'):
        for g, e in rep[basis].items():
            print(f"{basis} {g}: {e['mean_diff']:+.2f} {e['ci95']}, P {e['p_two_sided']:.4f}")
    e = rep['d5_coarse']
    print(f"D5 with coarse papers: {e['mean_diff']:+.2f} {e['ci95']}, P {e['p_two_sided']:.4f}")
    for basis, e in out['post_hoc_reported']['mr3_vs_crf'].items():
        print(f"MR3 - CRF head, 11 units, {basis}: {e['mean_diff']:+.2f} [{e['ci95'][0]:+.2f}, "
              f"{e['ci95'][1]:+.2f}], P two-sided {e['p_two_sided']:.4f}")


if __name__ == '__main__':
    main()
