"""
ner_robustness.py - robustness checks of the names method's pooled comparisons (paper A)

No training: reads results/gpu/ner_week5_seed_analysis.json and the week-7 and week-12
comparisons files and writes results/gpu/ner_robustness.json:
  pooled: over the 11 held-out units of decision 13.2 (MuLMS's 7 subfields and MaterioMiner's
      4 fine-grained papers), every pairwise comparison among the CRF head, mention replacement
      with 3 copies (MR3), with 6 copies (MR6) and the names method (MR6 + harvested names),
      on two seed bases for the MuLMS units: `ten` (seed offsets 0-900, as decision 13.2 was
      executed) and `decision` (offsets 300-900, the seeds the screen did not see; MaterioMiner
      is always 10 seeds). Per comparison the per-unit differences, the mean with its 95%
      Student-t interval, the number of positive units, an exact sign-flip permutation p-value
      (two-sided; all 2^11 sign patterns), and, for the names method against the CRF head,
      the leave-one-unit-out intervals.
  grids: the same comparisons on MaterioMiner's 4 held-out papers per granularity.
  coverage_gain: per MuLMS subfield, the lexicon's share of the unseen test names, the names'
      gain over MR6 and over the CRF head (decision seeds) and the CRF head's F1 (10 seeds),
      with Pearson and Spearman correlations.
  compute: per system on MuLMS's decision seeds, runs, mean and maximum runtime, mean best
      epoch and the number of augmented sentences.
Everything here is post hoc and labelled so in the paper; no decision rests on it.

Usage:
  python src/graph/ner_robustness.py [--results results/gpu] [--out results/gpu/ner_robustness.json]
"""

import argparse
import itertools
import json
import os
import statistics

import metrics
from ner_seed_analysis import (COPIES_CONTROL, WEEK14_ANALOGUE, WEEK14_WINNER, WEEK17_CONTROL,
                               WEEK17_NAMES)

ANALYSIS = 'ner_week5_seed_analysis.json'
WEEK7 = 'ner_week7_comparisons.json'
WEEK12 = 'ner_week12_comparisons.json'
MULMS = 'mulms_leave_one_subfield_out'
DECISION = tuple(range(300, 1000, 100))
SYSTEMS = {'crf': 'CRF head', 'crf_mr': 'MR3', COPIES_CONTROL: 'MR6', WEEK14_WINNER: 'MR6 + names'}
COMPARISONS = (('names_vs_crf', 'MR6 + names', 'CRF head'), ('names_vs_mr3', 'MR6 + names', 'MR3'),
               ('names_vs_mr6', 'MR6 + names', 'MR6'), ('mr6_vs_crf', 'MR6', 'CRF head'),
               ('mr6_vs_mr3', 'MR6', 'MR3'), ('mr3_vs_crf', 'MR3', 'CRF head'))


def interval(values):
    """{'mean_diff', 'ci95', 'per_run', 'n_positive'} with the 95% Student-t interval."""
    ci = metrics.mean_ci(values)
    return {'mean_diff': round(statistics.mean(values), 2),
            'ci95': [round(ci['ci'][0], 2), round(ci['ci'][1], 2)],
            'per_run': [round(v, 2) for v in values], 'n_positive': sum(v > 0 for v in values)}


def sign_flip_p(values):
    """Exact two-sided sign-flip permutation p-value of the mean over all 2^n sign patterns."""
    n = len(values)
    observed = abs(sum(values))
    hits = sum(abs(sum(s * v for s, v in zip(signs, values))) >= observed - 1e-9
               for signs in itertools.product((1, -1), repeat=n))
    return round(hits / 2 ** n, 4)


def leave_one_out(units, values):
    """[(unit left out, mean, lower, upper)] over the units, and whether every lower bound
    is above 0."""
    rows = []
    for k, unit in enumerate(units):
        rest = values[:k] + values[k + 1:]
        e = interval(rest)
        rows.append({'without': unit, 'mean_diff': e['mean_diff'], 'ci95': e['ci95']})
    return {'rows': rows, 'all_above_zero': all(r['ci95'][0] > 0 for r in rows),
            'min_lower': min(r['ci95'][0] for r in rows)}


def _micro(entry):
    return entry['micro']['per_run'] if 'micro' in entry else entry['per_run']


def unit_differences(analysis, comparisons):
    """{basis: {comparison: (units, per-unit differences)}} over the 11 pooled units, from the
    analysis file's week13.pooled, week14.copies and mulms.week12 sections and the week-12
    comparisons file (the fine-grained analogue against MR3); None where a part is missing."""
    pooled = analysis.get('week13', {}).get('pooled', {})
    copies = analysis.get('week14', {}).get('copies', {})
    winner = analysis.get('week13', {}).get('week12_winner', WEEK14_WINNER)
    w12 = analysis.get('mulms', {}).get('week12', {}).get(winner, {})
    names = pooled.get(winner)
    if not names or not copies.get('mulms') or not copies.get('fine'):
        return {}
    m_units = list(names['corpora']['mulms']['units'])
    f_units = list(names['corpora']['fine']['units'])
    fine_v4_mr3 = comparisons.get(f'ner_results_fine_leave_one_paper_out_{WEEK14_ANALOGUE}_vs_crf_mr',
                                  {}).get('micro', {}).get('per_run')
    fine = {'names_vs_crf': names['corpora']['fine']['minus_crf']['per_run'],
            'names_vs_mr6': _micro(copies['fine']['minus_crf_mr6']),
            'mr6_vs_crf': _micro(copies['fine']['control']['minus_crf']),
            'mr6_vs_mr3': _micro(copies['fine']['control']['minus_crf_mr']),
            'mr3_vs_crf': pooled['crf_mr']['corpora']['fine']['minus_crf']['per_run'],
            'names_vs_mr3': fine_v4_mr3}
    out = {}
    for basis, part in (('ten', 'ten_seeds'), ('decision', 'decision')):
        c = copies['mulms'].get(part)
        w = w12.get(part)
        if not c or not w:
            continue
        mr6_crf = _micro(c['control']['minus_crf'])
        mr6_mr3 = _micro(c['control']['minus_crf_mr'])
        mulms = {'names_vs_crf': _micro(w['minus_crf']), 'names_vs_mr3': _micro(w['minus_crf_mr']),
                 'names_vs_mr6': _micro(c['minus_crf_mr6']), 'mr6_vs_crf': mr6_crf,
                 'mr6_vs_mr3': mr6_mr3, 'mr3_vs_crf': [a - b for a, b in zip(mr6_crf, mr6_mr3)]}
        out[basis] = {key: (m_units + f_units, list(mulms[key]) + list(fine[key]))
                      for key, *_ in COMPARISONS if fine.get(key) is not None}
    return out


def pooled(analysis, comparisons):
    """The pooled section: per basis and comparison, interval, n_positive, sign_flip_p, and
    for names_vs_crf the leave-one-out table."""
    out = {}
    for basis, entries in unit_differences(analysis, comparisons).items():
        out[basis] = {}
        for key, (units, values) in entries.items():
            e = interval(values)
            e['sign_flip_p'] = sign_flip_p(values)
            e['n_units'] = len(values)
            if key == 'names_vs_crf':
                e['leave_one_out'] = leave_one_out(units, values)
            out[basis][key] = e
        out[basis]['units'] = units
        out[basis]['stratified_names_vs_crf'] = stratified(units, entries['names_vs_crf'][1])
    return out


def stratified(units, values):
    """The mean of the per-corpus means of names_vs_crf: MuLMS's subfields and MaterioMiner's papers
    weighted equally as corpora rather than as units."""
    groups = {}
    for unit, value in zip(units, values):
        groups.setdefault('mulms' if not unit.endswith('.conll') else 'materiominer', []).append(value)
    means = [statistics.mean(v) for v in groups.values()]
    return {'corpus_means': {k: round(statistics.mean(v), 2) for k, v in groups.items()},
            'mean': round(statistics.mean(means), 2)} if len(means) == 2 else {}


def mde(half_width, df, power=0.80):
    """The smallest true mean difference a paired comparison with this CI half-width and these
    degrees of freedom detects with the given power at the two-sided 5% level: the standard error
    implied by the half-width times the sum of the 97.5% and the power quantiles of Student's t."""
    from scipy import stats
    t975 = stats.t.ppf(0.975, df)
    return round(half_width / t975 * (t975 + stats.t.ppf(power, df)), 2)


def detectable(analysis, robust_pooled, robust_grids):
    """Per comparison the CI half-width, degrees of freedom and MDE at 80% power, from the stored
    intervals: the pooled comparisons of both bases (10 degrees of freedom), the MaterioMiner
    grids (3), the MuLMS decisions D1, D2 and D8 and the names against MR6 there (6), the
    3-seed ablations against the method (6), and the week-17 decisions on the strongest base
    with the names' increment over the week-9 system there (3)."""
    out = {}

    def add(name, entry, df):
        m = entry.get('micro', entry) if isinstance(entry, dict) else None
        if m and 'ci95' in m:
            half = (m['ci95'][1] - m['ci95'][0]) / 2
            out[name] = {'half_width': round(half, 2), 'df': df, 'mde_80': mde(half, df)}

    for basis, entries in robust_pooled.items():
        for key, e in entries.items():
            if isinstance(e, dict) and 'ci95' in e:
                add(f'pooled {basis}: {key}', e, e['n_units'] - 1)
    for g, cells in robust_grids.items():
        for key, e in cells.items():
            add(f'MaterioMiner {g}: {key}', e, 3)
    winner = analysis.get('week13', {}).get('week12_winner', WEEK14_WINNER)
    dec = analysis.get('mulms', {}).get('week12', {}).get(winner, {}).get('decision', {})
    cop = analysis.get('week14', {}).get('copies', {}).get('mulms', {}).get('decision', {})
    top = analysis.get('week14', {}).get('topic', {}).get('decision', {})
    add('D1 MuLMS: names_vs_crf', dec.get('minus_crf'), 6)
    add('D2 MuLMS: names_vs_mr3', dec.get('minus_crf_mr'), 6)
    add('D8 MuLMS: own_vs_mixed', top.get('minus_crf_mrm_mixed6'), 6)
    folds = analysis.get('week13', {}).get('folds', {}).get('crf_mrm6', {})
    add('D6 SOFC-Exp: names_vs_crf', folds.get('minus_crf'), 4)      # 5 article folds
    add('MuLMS: names_vs_mr6', cop.get('minus_crf_mr6'), 6)
    for name, entry in (analysis.get('week14', {}).get('ablations', {}) or {}).items():
        add(f'ablation {name}: vs_method (3 seeds)', entry.get(f'minus_{winner}') if isinstance(entry, dict) else None, 6)
    for g, entry in (analysis.get('week17') or {}).items():
        # week 17 (post review): the names on the multi-task model with 6 copies against the
        # matched control (decisions 17.1 and 17.2) and against multi-task + MR3 (reported)
        add(f"{entry.get('decision', '17')} MaterioMiner {g}: {WEEK17_NAMES}_vs_{WEEK17_CONTROL}",
            entry.get(f'minus_{WEEK17_CONTROL}'), 3)
        add(f'MaterioMiner {g}: {WEEK17_NAMES}_vs_crf_mt_mr',
            (entry.get('names') or {}).get('minus_crf_mt_mr'), 3)
    return out


def grids(analysis, week7, week12):
    """{granularity: {comparison: interval}} on MaterioMiner's 4 held-out papers."""
    copies = analysis.get('week14', {}).get('copies', {})
    out = {}
    for g in ('fine', 'coarse'):
        c = copies.get(g)
        if not c:
            continue
        prefix = f'ner_results_{g}_leave_one_paper_out_'
        cells = {'names_vs_mr6': c['minus_crf_mr6'],
                 'mr6_vs_crf': c['control']['minus_crf'],
                 'mr6_vs_mr3': c['control']['minus_crf_mr'],
                 'names_vs_crf': week12.get(f'{prefix}{WEEK14_ANALOGUE}'),
                 'names_vs_mr3': week12.get(f'{prefix}{WEEK14_ANALOGUE}_vs_crf_mr'),
                 'mr3_vs_crf': week7.get(f'{prefix}crf_mr')}
        out[g] = {key: stored_interval(v) for key, v in cells.items() if v and v.get('micro')}
    return out


def stored_interval(entry):
    """interval()'s format from an analysis entry's stored mean and CI, which were computed from
    the unrounded per-run values; the per-run values are stored rounded, so recomputing from them
    can move a bound by 0.01."""
    m = entry['micro'] if isinstance(entry, dict) else None
    if not m or 'ci95' not in m:
        return interval(list(entry))
    per = list(m.get('per_run', []))
    return {'mean_diff': m.get('mean_diff', m.get('mean')), 'ci95': list(m['ci95']),
            'per_run': per, 'n_positive': sum(v > 0 for v in per)}


def partial(r_xy, r_xz, r_yz):
    """The partial correlation of x and y given z from the three pairwise correlations."""
    return (r_xy - r_xz * r_yz) / ((1 - r_xz ** 2) * (1 - r_yz ** 2)) ** 0.5


def coverage_gain(analysis):
    """Per MuLMS subfield the lexicon's coverage of the unseen test names, the names' gain over
    MR6 and over the CRF head on the decision seeds and the CRF head's 10-seed F1, with the
    correlations; {} without the week-14 decision part."""
    winner = analysis.get('week13', {}).get('week12_winner', WEEK14_WINNER)
    dec = analysis.get('mulms', {}).get('week12', {}).get(winner, {}).get('decision')
    cop = analysis.get('week14', {}).get('copies', {}).get('mulms', {}).get('decision')
    held = analysis.get('mulms', {}).get('leave_one_subfield_out', {}).get('subfields', {})
    if not dec or not cop or not held:
        return {}
    subs = list(dec['names'])
    cov = [dec['names'][s]['n_test_unseen_in_silver'] / dec['names'][s]['n_test_unseen'] for s in subs]
    names_gain = _micro(cop['minus_crf_mr6'])
    method_gain = _micro(dec['minus_crf'])
    crf = [held[s]['crf'] for s in subs]
    from scipy import stats
    r_cov_f1 = stats.pearsonr(cov, crf)[0]
    r_cov_gain = stats.pearsonr(cov, names_gain)[0]
    r_f1_gain = stats.pearsonr(crf, names_gain)[0]
    keep = [i for i, s in enumerate(subs) if s not in ('semi', 'sofc')]
    sub = lambda xs: [xs[i] for i in keep]  # noqa: E731
    return {'subfields': subs, 'coverage': [round(x, 3) for x in cov],
            'names_minus_mr6': names_gain, 'names_minus_crf': method_gain, 'crf_f1': crf,
            'pearson': {'coverage_vs_names_gain': round(r_cov_gain, 2),
                        'crf_f1_vs_names_gain': round(r_f1_gain, 2),
                        'crf_f1_vs_method_gain': round(stats.pearsonr(crf, method_gain)[0], 2),
                        'coverage_vs_crf_f1': round(r_cov_f1, 2)},
            'partial': {'coverage_vs_names_gain_given_crf_f1': round(partial(r_cov_gain, r_cov_f1, r_f1_gain), 2),
                        'crf_f1_vs_names_gain_given_coverage': round(partial(r_f1_gain, r_cov_f1, r_cov_gain), 2)},
            'without_semi_sofc': {'coverage_vs_names_gain': round(stats.pearsonr(sub(cov), sub(names_gain))[0], 2),
                                  'crf_f1_vs_names_gain': round(stats.pearsonr(sub(crf), sub(names_gain))[0], 2)},
            'spearman': {'coverage_vs_names_gain': round(stats.spearmanr(cov, names_gain)[0], 2),
                         'crf_f1_vs_names_gain': round(stats.spearmanr(crf, names_gain)[0], 2)}}


def compute(directory):
    """Per system on MuLMS's decision seeds: runs, mean and max runtime_sec, the mean best
    epoch (best_epoch is a 0-based index in the run files; epochs_to_best_mean counts the
    epochs trained up to and including it), the mean count of augmented sentences, and
    against the CRF head the ratio of sentences per epoch and of updates up to the best
    epoch; systems whose files are missing are left out."""
    out = {}
    for system in SYSTEMS:
        runs = []
        for off in DECISION:
            path = os.path.join(directory, f'ner_results_{MULMS}_{system}_m{off}.json')
            if not os.path.exists(path):
                runs = None
                break
            with open(path) as f:
                runs += json.load(f)['runs']
        if runs:
            augmented = [r.get('n_augmented') for r in runs]
            out[system] = {'label': SYSTEMS[system], 'n_runs': len(runs),
                           'runtime_sec_mean': round(statistics.mean(r['runtime_sec'] for r in runs)),
                           'runtime_sec_max': round(max(r['runtime_sec'] for r in runs)),
                           'best_epoch_mean': round(statistics.mean(r['best_epoch'] for r in runs), 1),
                           'epochs_to_best_mean': round(statistics.mean(r['best_epoch'] + 1 for r in runs), 1),
                           'n_augmented_mean': (round(statistics.mean(augmented))
                                                if all(a is not None for a in augmented) else None)}
            if runs[0].get('self_training'):
                out[system].update(job_costs(runs, out.get('crf_mr6', {}).get('runtime_sec_mean')))
            if out[system]['n_augmented_mean'] is not None and 'crf' in out:
                n_train = statistics.mean(r['n_train'] for r in runs)
                sentences = (n_train + statistics.mean(augmented)) / n_train
                epochs = statistics.mean(r['best_epoch'] + 1 for r in runs)
                out[system]['sentences_per_epoch_ratio'] = round(sentences, 2)
                out[system]['updates_to_best_ratio'] = round(
                    sentences * epochs / out['crf']['epochs_to_best_mean_exact'], 2)
            if system == 'crf':
                out[system]['epochs_to_best_mean_exact'] = statistics.mean(r['best_epoch'] + 1 for r in runs)
    if 'crf' in out:
        del out['crf']['epochs_to_best_mean_exact']
    return out


def job_costs(runs, student_sec=None):
    """A teacher system's runs store their job's time (teachers, tagging and the job's students
    together), one value shared by the students of a job: the jobs (distinct run_id and
    runtime), the students per job, the job mean, the mean per student (job / students) and,
    given a plain student's time (MR6's run mean), the harvest alone (job - students * that)."""
    jobs = {}
    for r in runs:
        jobs.setdefault((r['run_id'], r['runtime_sec']), 0)
        jobs[(r['run_id'], r['runtime_sec'])] += 1
    times = [(t, n) for (_, t), n in jobs.items()]
    out = {'n_jobs': len(times), 'students_per_job': sorted({n for _, n in times}),
           'job_sec_mean': round(statistics.mean(t for t, _ in times)),
           'per_student_sec_mean': round(statistics.mean(t / n for t, n in times))}
    if student_sec:
        out['harvest_sec_mean'] = round(statistics.mean(t - n * student_sec for t, n in times))
        out['harvest_student_sec'] = student_sec
    return out


def load(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Robustness checks of the names method's pooled comparisons")
    root = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'gpu')
    parser.add_argument('--results', default=root)
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    analysis = load(os.path.join(args.results, ANALYSIS))
    week7, week12 = load(os.path.join(args.results, WEEK7)), load(os.path.join(args.results, WEEK12))
    robust, grid = pooled(analysis, week12), grids(analysis, week7, week12)
    out = {'pooled': robust, 'grids': grid, 'coverage_gain': coverage_gain(analysis),
           'detectable': detectable(analysis, robust, grid), 'compute': compute(args.results)}
    path = args.out or os.path.join(args.results, 'ner_robustness.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    for basis, entries in out['pooled'].items():
        for key, label, ref in COMPARISONS:
            e = entries.get(key)
            if e:
                print(f"  {basis:8s} {label} minus {ref}: {e['mean_diff']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}], "
                      f"{e['n_positive']}/{e['n_units']} positive, sign-flip p = {e['sign_flip_p']}")
    cg = out['coverage_gain']
    if cg:
        print(f"  coverage vs names gain r = {cg['pearson']['coverage_vs_names_gain']}, "
              f"CRF F1 vs names gain r = {cg['pearson']['crf_f1_vs_names_gain']}")
    print(f'Saved {path}')
    return out


if __name__ == '__main__':
    main()
