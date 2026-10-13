"""
ner_seedmeans.py - week-3 NER comparisons of 3-seed means (ROADMAP §4, week-3 protocol)

Week 3 scores a model trained with three seeds on a split by the mean of its three runs
(model seeds K, K + 100 and K + 200 on random split K; 0, 100 and 200 on a held-out
paper), an ensemble or a one-seed run by its one run, and compares these scores split
by split. No training: the script reads the results files of ner_baseline.py and
ner_ensemble.py.

  - Families of three seeds (the variants of their results files): crf = crf_m0,
    crf_m100, crf_m200 (the reference: the week-2 ensemble members); crf_mt = crf_mt,
    crf_mt_m100, crf_mt_m200; crf_ls, crf_st and crf_dapt2 = <family>_m0, _m100, _m200.
  - Single systems, scored once per split: crf_ens3, crf_mix3, crf_scibert and
    crf_pubmedbert.
  - A family's score for a run_id is the mean over its members' runs with that run_id,
    for micro, strict micro, macro and weighted F1. A run_id counts only if all three
    members have it.
  - Comparisons, in the protocol's reporting order: crf_mix3 vs crf_ens3 (the primary
    one of variant 1); crf_mix3, crf_ens3, crf_scibert and crf_pubmedbert vs crf; crf_ls,
    crf_st, crf_mt and crf_dapt2 vs crf. Each is made per granularity over the 5 random
    splits and the 4 held-out papers and paired by run_id: the per-run differences,
    their mean and the 95% Student-t CI, rounded as in ner_comparisons.json.
  - role: 'decision' on the protocol where the week-3 rule decides (crf_mix3 vs
    crf_ens3, crf_mt and crf_dapt2 over the random splits; crf_ls and crf_st over the
    held-out papers), 'reported' otherwise. complete: 5 pairs over the random splits,
    4 over the held-out papers.

A comparison is skipped while one of its results files is missing. The output is one
JSON object keyed like ner_comparisons.json:

    ner_results_<granularity>_<protocol>_<variant>[_vs_<reference>, unless it is crf]

An entry holds variant, reference, granularity, protocol, role, the files of both sides,
n_pairs, run_ids, complete, the per-split scores of both sides (variant_scores and
reference_scores: mean, 95% CI and per_run for every F1 average, in run_ids order), and
micro, micro_strict, macro and weighted, each with mean_diff, ci95 and per_run.

Usage:
  python src/graph/ner_seedmeans.py [--results results] [--out results/ner_week3_comparisons.json]
"""

import argparse
import json
import os
import statistics
from pathlib import Path

import metrics
from ner_baseline import F1_AVERAGES, result_suffix

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results'
OUT_NAME = 'ner_week3_comparisons.json'

GRANULARITIES = ('fine', 'coarse')
# pairs of a complete comparison: the 5 random splits and the 4 held-out papers
PROTOCOLS = {'random_split_vary': 5, 'leave_one_paper_out': 4}

# three model seeds per split: the variants of their results files
FAMILIES = {
    'crf': ('crf_m0', 'crf_m100', 'crf_m200'),
    'crf_mt': ('crf_mt', 'crf_mt_m100', 'crf_mt_m200'),
    'crf_ls': ('crf_ls_m0', 'crf_ls_m100', 'crf_ls_m200'),
    'crf_st': ('crf_st_m0', 'crf_st_m100', 'crf_st_m200'),
    'crf_dapt2': ('crf_dapt2_m0', 'crf_dapt2_m100', 'crf_dapt2_m200'),
}
# one run per split: the ensembles and the backbones of the mixed one
SINGLE = ('crf_ens3', 'crf_mix3', 'crf_scibert', 'crf_pubmedbert')
SYSTEMS = {**FAMILIES, **{name: (name,) for name in SINGLE}}

# (variant, reference, protocol on which the week-3 rule decides; None: reported only),
# in the protocol's reporting order
COMPARISONS = (
    ('crf_mix3', 'crf_ens3', 'random_split_vary'),
    ('crf_mix3', 'crf', None),
    ('crf_ens3', 'crf', None),
    ('crf_scibert', 'crf', None),
    ('crf_pubmedbert', 'crf', None),
    ('crf_ls', 'crf', 'leave_one_paper_out'),
    ('crf_st', 'crf', 'leave_one_paper_out'),
    ('crf_mt', 'crf', 'random_split_vary'),
    ('crf_dapt2', 'crf', 'random_split_vary'),
)


def file_names(granularity, protocol, system):
    """Results files of a system (a family's three seeds, or one file) for one protocol."""
    return [f'ner_results_{granularity}{result_suffix(protocol, v)}.json' for v in SYSTEMS[system]]


def comparison_key(granularity, protocol, variant, reference):
    """Key of a comparison: as in ner_comparisons.json, plus _vs_<reference> unless crf."""
    key = f'ner_results_{granularity}{result_suffix(protocol, variant)}'
    return key if reference == 'crf' else f'{key}_vs_{reference}'


def system_scores(files):
    """{run_id: {F1 average: score}} of a system from its parsed results files.

    One file gives the scores of its runs; a family's files give the mean over its
    members, for the run_ids that every member has (in the order of the first file).
    """
    by_id = [{r.get('run_id'): r for r in f['runs']} for f in files]
    ids = [i for i in by_id[0] if all(i in b for b in by_id[1:])]
    return {i: {name: statistics.fmean(get(b[i]) for b in by_id)
                for name, get in F1_AVERAGES.items()} for i in ids}


def spread(values, mean_key='mean'):
    """Mean, 95% Student-t CI and per-run values, rounded as in ner_comparisons.json."""
    return {mean_key: round(statistics.mean(values), 2),
            'ci95': [round(x, 2) for x in metrics.mean_ci(values)['ci']],
            'per_run': [round(v, 2) for v in values]}


def compare(variant, reference):
    """Scores of both sides and their differences (variant - reference), paired by run_id.

    variant, reference: {run_id: {F1 average: score}} as from system_scores, with at
    least one run_id in common. The pairs follow the order of the variant's run_ids.
    """
    ids = [i for i in variant if i in reference]
    out = {'run_ids': ids}
    for side, scores in (('variant_scores', variant), ('reference_scores', reference)):
        out[side] = {name: spread([scores[i][name] for i in ids]) for name in F1_AVERAGES}
    for name in F1_AVERAGES:
        out[name] = spread([variant[i][name] - reference[i][name] for i in ids], 'mean_diff')
    return out


def load_scores(directory, names):
    """system_scores of the results files `names` in `directory`."""
    files = []
    for name in names:
        with open(os.path.join(directory, name)) as f:
            files.append(json.load(f))
    return system_scores(files)


def week3_comparisons(directory):
    """Every week-3 comparison that the results files in `directory` allow, keyed as above.

    Prints one line per comparison: its micro-F1 difference, or why it was skipped (a
    missing results file, or no run_id with scores on both sides).
    """
    comparisons = {}
    for variant, reference, decides in COMPARISONS:
        for protocol, need in PROTOCOLS.items():
            for granularity in GRANULARITIES:
                key = comparison_key(granularity, protocol, variant, reference)
                files = {side: file_names(granularity, protocol, side)
                         for side in (variant, reference)}
                missing = [fn for names in files.values() for fn in names
                           if not os.path.exists(os.path.join(directory, fn))]
                if missing:
                    print(f"  {key}: no {', '.join(missing)}, skipped")
                    continue
                scores = {side: load_scores(directory, names) for side, names in files.items()}
                if not any(i in scores[reference] for i in scores[variant]):
                    print(f"  {key}: no run_id with scores on both sides, skipped")
                    continue
                comp = compare(scores[variant], scores[reference])
                entry = {'variant': variant, 'reference': reference, 'granularity': granularity,
                         'protocol': protocol,
                         'role': 'decision' if protocol == decides else 'reported',
                         'variant_files': files[variant], 'reference_files': files[reference],
                         'n_pairs': len(comp['run_ids']), 'run_ids': comp['run_ids'],
                         'complete': len(comp['run_ids']) >= need, **comp}
                comparisons[key] = entry
                d = entry['micro']
                print(f"  {key}: micro F1 difference {d['mean_diff']:+.2f} vs {reference}, "
                      f"95% CI {d['ci95']} over {entry['n_pairs']} paired runs ({entry['role']}"
                      f"{'' if entry['complete'] else ', incomplete'})")
    return comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Week-3 NER comparisons of 3-seed means (no training): reads the results '
                    'files and writes ner_week3_comparisons.json')
    parser.add_argument('--results', default=str(RESULTS_DIR), metavar='DIR',
                        help='directory with the NER results files (default: results/ of the repo)')
    parser.add_argument('--out', default=None, metavar='PATH',
                        help=f'output file (default: DIR/{OUT_NAME})')
    args = parser.parse_args(argv)
    if not os.path.isdir(args.results):
        parser.error(f'results directory not found: {args.results}')
    comparisons = week3_comparisons(args.results)
    out = args.out or os.path.join(args.results, OUT_NAME)
    with open(out, 'w') as f:
        json.dump(comparisons, f, indent=2)
    complete = sum(c['complete'] for c in comparisons.values())
    print(f"Saved {out}: {len(comparisons)} comparisons, {complete} complete")
    return comparisons


if __name__ == '__main__':
    main()
