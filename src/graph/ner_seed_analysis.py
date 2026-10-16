"""
ner_seed_analysis.py - exploratory analyses of the week-5 10-seed runs (ROADMAP §4, week-5
protocol, "Reported alongside")

No training and no decision: reads the GPU results files of the 10-seed families and their
votes (ner_seedmeans.WEEK5_SYSTEMS) and writes ner_week5_seed_analysis.json next to them.

  - spread: for each family, protocol and granularity, the population SD of micro F1 over
    the 10 seeds on each split, and its mean over the splits.
  - triples: for each family against the CRF head, the comparison of 3-seed means that weeks
    3 and 4 made, repeated for each of the 120 triples of the 10 seed offsets (the same triple
    on both sides): the share of triples whose 95% CI lies above 0 (the change would pass the
    rule), below 0, and whose mean difference has the opposite sign to the 10-seed one. How
    far a 3-seed design can mislead.
  - types: for each vote family (crf_ens3, crf_mt_ens3, crf_ls_ens3, crf_st3_ens3) and entity
    type, the F1 over the family's 3 votes and the splits whose test set or predictions have
    the type, its mean test support per split, and the macro F1 without the type: the types
    that pull macro F1 down, for whichever vote is the reported system.
  - published_split (week-5 amendment): per granularity, plain MatSciBERT and the CRF head on
    the published split, 10 model seeds each: the mean and 95% CI over the seeds of every F1
    average and where the published number lies; the CRF head minus plain MatSciBERT, paired
    by seed; and the CRF votes over the seed triples 0-200, 300-500 and 600-800. Week 6 adds
    SOFC-Exp's official split (granularity sofc, against the published MatSciBERT-CRF
    scores) and, where their files exist, the week-6 methods on a published split
    (SPLIT_METHODS): their scores and their difference from the CRF head, paired by seed
    (SOFC-Exp's S1 and S2 of the week-6 protocol). Week 7 adds the consistency rule on the
    CRF head (crf_dc, and crf_dcr with retyping) and recall calibration with the bias chosen
    on the validation articles (crf_ebv, part 2a) in the same way; part 2d adds SOFC-Exp's
    slot level (sofc_slot, against the published MatSciBERT-CRF scores of SOFC-Slot) and
    multi-task training on SOFC-Exp's two levels (crf_mt), whose votes over the seed triples
    are also set against the CRF votes, paired by triple (VOTED_METHODS); part 2b adds
    mention replacement (crf_mr, S2).
  - errors (exploratory; cited by the week-7 protocol, part 1): per protocol and
    granularity, the kinds of error of the CRF head over its 10 seeds: the share of gold
    entities that are correct, missed (no overlapping prediction), of the right span and
    the wrong type, or boundary errors (of the right type, or of another type), and the
    share of predictions that overlap no gold entity (spurious).
  - week6_types (week 6, reported alongside its decisions): per granularity, each week-6
    method with all 10 seeds on the 4 held-out papers against the CRF head, per entity type:
    the F1 of both, paired by seed offset and paper over the pairs whose test set has the
    type, and their mean difference.

A family or vote family is used only if all its files exist. Usage:
  python src/graph/ner_seed_analysis.py [--results results/gpu] [--out PATH]
"""

import argparse
import itertools
import json
import os
import statistics
from pathlib import Path

import metrics
import ner_ensemble
from ner_baseline import F1_AVERAGES
from ner_seedmeans import (GRANULARITIES, LOPO, PROTOCOLS, WEEK5_OFFSETS, WEEK5_SYSTEMS,
                           WEEK5_TRIPLES, WEEK6_METHODS, file_names, seed_family, spread,
                           system_scores)

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'gpu'
OUT_NAME = 'ner_week5_seed_analysis.json'
FAMILIES = ('crf', 'crf_mt', 'crf_ls', 'crf_st3')
VOTES = tuple(f'{f}_ens3' for f in FAMILIES)
AVERAGES = ('micro avg', 'macro avg', 'weighted avg')
# micro F1: Durmaz et al. 2024 (MaterioMiner); SOFC-Exp and its slots (SOFC-Slot):
# MatSciBERT-CRF (Gupta et al. 2022, Supplementary Table 3 and Table 1)
PUBLISHED_F1 = {'fine': 69.92, 'coarse': 72.32, 'sofc': 84.77, 'sofc_slot': 72.85}
PUBLISHED_MACRO = {'sofc': 82.39, 'sofc_slot': 65.35}
PUBLISHED_SPLITS = ('fine', 'coarse', 'sofc', 'sofc_slot')
# methods on a published split, compared with the CRF head by seed: the week-6 methods,
# the week-7 consistency rule on the CRF head's predictions (ner_consistency.py) and
# recall calibration (ner_bias.py --choose val); multi-task training on SOFC-Exp's levels
SPLIT_METHODS = ('crf_deberta', 'crf_ctx', 'crf_dc', 'crf_dcr', 'crf_ebv', 'crf_mt', 'crf_mr')
VOTED_METHODS = ('crf_mt',)


def load_files(directory, names):
    """The parsed results files, or None if one is missing."""
    paths = [os.path.join(directory, n) for n in names]
    if not all(os.path.exists(p) for p in paths):
        return None
    out = []
    for p in paths:
        with open(p) as f:
            out.append(json.load(f))
    return out


def seed_spread(files):
    """Per run_id, the population SD of micro F1 over the members; and its mean."""
    by_id = [{r['run_id']: r['f1'] for r in f['runs']} for f in files]
    ids = [i for i in by_id[0] if all(i in b for b in by_id)]
    per_split = {i: round(statistics.pstdev(b[i] for b in by_id), 2) for i in ids}
    return {'per_split': per_split,
            'mean_sd': round(statistics.mean(per_split.values()), 2) if ids else None}


def triple_comparisons(variant_files, reference_files):
    """The 3-seed paired comparisons of every triple of the 10 seeds against the 10-seed one.

    The files are those of two 10-seed families, member k of both with seed offset 100 k. Each
    triple scores the mean micro F1 of its members per run_id on both sides; the triple passes
    if the 95% Student-t CI of the mean paired difference lies above 0.
    """
    def diffs(members):
        v = system_scores([variant_files[k] for k in members])
        r = system_scores([reference_files[k] for k in members])
        return [v[i]['micro'] - r[i]['micro'] for i in v if i in r]

    full = diffs(range(len(WEEK5_OFFSETS)))
    above = below = flipped = 0
    triples = list(itertools.combinations(range(len(WEEK5_OFFSETS)), 3))
    for members in triples:
        d = diffs(members)
        lo, hi = metrics.mean_ci(d)['ci']
        above += lo > 0
        below += hi < 0
        flipped += statistics.mean(d) * statistics.mean(full) < 0    # strictly opposite
    n = len(triples)
    lo, hi = metrics.mean_ci(full)['ci']
    return {'n_triples': n, 'ten_seed': {'mean_diff': round(statistics.mean(full), 2),
                                         'ci95': [round(lo, 2), round(hi, 2)],
                                         'passes': lo > 0},
            'share_above_0': round(above / n, 3), 'share_below_0': round(below / n, 3),
            'share_opposite_sign': round(flipped / n, 3)}


def type_scores(vote_files):
    """Per entity type: mean F1 and support over (vote, split) where the type occurs, and the
    mean macro F1 of those (vote, split) pairs without the type."""
    rows = {}
    for f in vote_files:
        for run in f['runs']:
            classes = {t: s for t, s in run['per_class'].items() if t not in AVERAGES}
            macro = statistics.mean(s['f1-score'] for s in classes.values())
            for t, s in classes.items():
                others = [o['f1-score'] for u, o in classes.items() if u != t]
                row = rows.setdefault(t, {'f1': [], 'support': [], 'macro': [], 'without': []})
                row['f1'].append(100 * s['f1-score'])
                row['support'].append(s['support'])
                row['macro'].append(100 * macro)
                row['without'].append(100 * statistics.mean(others) if others else None)
    out = {}
    for t, row in sorted(rows.items(), key=lambda kv: statistics.mean(kv[1]['f1'])):
        without = [w for w in row['without'] if w is not None]
        out[t] = {'f1': round(statistics.mean(row['f1']), 2),
                  'support_per_split': round(statistics.mean(row['support']), 1),
                  'n': len(row['f1']),
                  'macro_without': round(statistics.mean(without), 2) if without else None,
                  'macro_with': round(statistics.mean(row['macro']), 2)}
    return out


def type_differences(method_files, crf_files):
    """Per entity type: the F1 of a method and of the CRF head, paired by member (seed offset)
    and run_id (paper) over the pairs whose test set has the type; their means, the mean
    difference, the number of papers and the mean test support. Most frequent type first."""
    rows = {}
    for method, crf in zip(method_files, crf_files, strict=True):
        ours = {r['run_id']: r for r in method['runs']}
        for ref in crf['runs']:
            run = ours.get(ref['run_id'])
            if run is None:
                continue
            for t, s in ref['per_class'].items():
                if t in AVERAGES or not s['support']:
                    continue
                f1 = run['per_class'].get(t, {'f1-score': 0.0})['f1-score']
                row = rows.setdefault(t, {'crf': [], 'method': [], 'papers': set(),
                                          'support': []})
                row['crf'].append(100 * s['f1-score'])
                row['method'].append(100 * f1)
                row['papers'].add(ref['run_id'])
                row['support'].append(s['support'])
    out = {}
    for t, row in sorted(rows.items(), key=lambda kv: -statistics.mean(kv[1]['support'])):
        out[t] = {'crf': round(statistics.mean(row['crf']), 2),
                  'method': round(statistics.mean(row['method']), 2),
                  'diff': round(statistics.mean(m - c for m, c in zip(row['method'],
                                                                         row['crf'])), 2),
                  'n_splits': len(row['papers']),
                  'support_per_split': round(statistics.mean(row['support']), 1)}
    return out


def error_kinds(files):
    """The kinds of error over every run of the parsed results files (their test_gold and
    test_pred): shares of the gold entities and, for 'spurious', of the predictions."""
    counts = dict.fromkeys(('correct', 'missed', 'wrong_type', 'boundary',
                            'boundary_and_type', 'spurious'), 0)
    n_gold = n_pred = 0
    for f in files:
        for run in f['runs']:
            for gold_tags, pred_tags in zip(run['test_gold'], run['test_pred'], strict=True):
                gold = set(ner_ensemble.tags_to_spans(gold_tags.split())[0])
                pred = set(ner_ensemble.tags_to_spans(pred_tags.split())[0])
                n_gold, n_pred = n_gold + len(gold), n_pred + len(pred)
                for g in gold:
                    over = [p for p in pred if p[0] < g[1] and g[0] < p[1]]
                    kind = ('correct' if g in pred else 'missed' if not over
                            else 'wrong_type' if any(p[:2] == g[:2] for p in over)
                            else 'boundary' if any(p[2] == g[2] for p in over)
                            else 'boundary_and_type')
                    counts[kind] += 1
                counts['spurious'] += sum(not any(g[0] < p[1] and p[0] < g[1] for g in gold)
                                          for p in pred)
    return {'n_gold': n_gold, 'n_pred': n_pred,
            **{k: round(v / (n_pred if k == 'spurious' else n_gold), 3) if v else 0.0
               for k, v in counts.items()}}


def week6_type_differences(directory):
    """{granularity: {method: type_differences}} on the 4 held-out papers, for the week-6
    methods whose 10 seeds and the CRF head's are all there."""
    out = {}
    for g in GRANULARITIES:
        crf = load_files(directory, file_names(g, LOPO, 'crf', WEEK5_SYSTEMS))
        for method in WEEK6_METHODS:
            files = load_files(directory, file_names(g, LOPO, method,
                                                     {method: seed_family(method)}))
            if crf and files:
                out.setdefault(g, {})[method] = type_differences(files, crf)
    return out


def triple_votes(by_id):
    """The votes of a system's runs {run_id: run} over the seed triples 0-200, 300-500 and
    600-800 of a published split (run ids seed<K>); None unless every member is there."""
    triples = [[f'seed{first + 100 * j}' for j in range(3)] for first in WEEK5_TRIPLES]
    if not all(i in by_id for t in triples for i in t):
        return None
    return [ner_ensemble.ensemble_run(f'seed{t[0][4:]}_triple', [by_id[i] for i in t])
            for t in triples]


def published_split(directory):
    """The published-split analysis per granularity (see the module docstring); granularities
    without both files are left out."""
    out = {}
    for g in PUBLISHED_SPLITS:
        files = load_files(directory, [f'ner_results_{g}_published_split.json',
                                       f'ner_results_{g}_published_split_crf.json'])
        if not files:
            continue
        runs = {name: {r['run_id']: r for r in f['runs']}
                for name, f in zip(('baseline', 'crf'), files)}
        paper = PUBLISHED_F1[g]
        entry = {'published': paper}
        if g in PUBLISHED_MACRO:
            entry['published_macro'] = PUBLISHED_MACRO[g]
        for name, by_id in runs.items():
            entry[name] = {avg: spread([get(r) for r in by_id.values()])
                           for avg, get in F1_AVERAGES.items()}
            lo, hi = entry[name]['micro']['ci95']
            entry[name]['published_lies'] = ('within' if lo <= paper <= hi else
                                             'below' if paper < lo else 'above')
        ids = [i for i in runs['crf'] if i in runs['baseline']]
        entry['crf_minus_baseline'] = {
            avg: spread([get(runs['crf'][i]) - get(runs['baseline'][i]) for i in ids],
                        'mean_diff') for avg, get in F1_AVERAGES.items()}
        crf_votes = triple_votes(runs['crf'])
        if crf_votes:
            entry['crf_votes'] = {avg: spread([get(v) for v in crf_votes])
                                  for avg, get in F1_AVERAGES.items()}
        for method in SPLIT_METHODS:
            found = load_files(directory, [f'ner_results_{g}_published_split_{method}.json'])
            if not found:
                continue
            by_id = {r['run_id']: r for r in found[0]['runs']}
            ids = [i for i in runs['crf'] if i in by_id]
            found = entry.setdefault('methods', {})[method] = {
                **{avg: spread([get(r) for r in by_id.values()])
                   for avg, get in F1_AVERAGES.items()},
                'minus_crf': {avg: spread([get(by_id[i]) - get(runs['crf'][i]) for i in ids],
                                          'mean_diff') for avg, get in F1_AVERAGES.items()},
                'n_pairs': len(ids)}
            votes = triple_votes(by_id) if method in VOTED_METHODS else None
            if votes and crf_votes:
                found['votes'] = {avg: spread([get(v) for v in votes])
                                  for avg, get in F1_AVERAGES.items()}
                found['votes_minus_crf_votes'] = {
                    avg: spread([get(v) - get(c) for v, c in zip(votes, crf_votes)],
                                'mean_diff') for avg, get in F1_AVERAGES.items()}
        out[g] = entry
    return out


def analyse(directory):
    """The analyses for every protocol and granularity that has the files."""
    out = {'spread': {}, 'triples': {}, 'types': {}}
    for protocol in PROTOCOLS:
        for g in GRANULARITIES:
            key = f'{g}_{protocol}'
            files = {fam: load_files(directory, file_names(g, protocol, fam, WEEK5_SYSTEMS))
                     for fam in FAMILIES}
            for fam, fs in files.items():
                if fs:
                    out['spread'].setdefault(key, {})[fam] = seed_spread(fs)
                if fs and fam != 'crf' and files['crf']:
                    out['triples'].setdefault(key, {})[fam] = triple_comparisons(
                        fs, files['crf'])
            for system in VOTES:
                votes = load_files(directory, file_names(g, protocol, system, WEEK5_SYSTEMS))
                if votes:
                    out['types'].setdefault(key, {})[system] = type_scores(votes)
    out['published_split'] = published_split(directory)
    out['week6_types'] = week6_type_differences(directory)
    out['errors'] = {}
    for protocol in PROTOCOLS:
        for g in GRANULARITIES:
            crf = load_files(directory, file_names(g, protocol, 'crf', WEEK5_SYSTEMS))
            if crf and all('test_pred' in r for f in crf for r in f['runs']):
                out['errors'][f'{g}_{protocol}'] = error_kinds(crf)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='Exploratory analyses of the 10-seed runs '
                                                 '(no training, no decision)')
    parser.add_argument('--results', default=str(RESULTS_DIR), metavar='DIR')
    parser.add_argument('--out', default=None, metavar='PATH')
    args = parser.parse_args(argv)
    result = analyse(args.results)
    for key, fams in result['triples'].items():
        for fam, t in fams.items():
            ten = t['ten_seed']
            print(f"  {key} {fam} vs crf: 10 seeds {ten['mean_diff']:+.2f} {ten['ci95']}; of "
                  f"{t['n_triples']} 3-seed triples {100 * t['share_above_0']:.0f}% pass, "
                  f"{100 * t['share_below_0']:.0f}% below 0, "
                  f"{100 * t['share_opposite_sign']:.0f}% opposite sign")
    for g, entry in result['published_split'].items():
        base, crf = entry['baseline']['micro'], entry['crf']['micro']
        print(f"  published split, {g}: plain MatSciBERT {base['mean']:.2f} {base['ci95']} "
              f"(published {entry['published']:.2f}, {entry['baseline']['published_lies']}), "
              f"CRF head {crf['mean']:.2f} {crf['ci95']}")
        for method, m in entry.get('methods', {}).items():
            d = m['minus_crf']['micro']
            print(f"    {method}: {m['micro']['mean']:.2f} {m['micro']['ci95']}, minus the CRF "
                  f"head {d['mean_diff']:+.2f} {d['ci95']} over {m['n_pairs']} seeds")
    for g, methods in result['week6_types'].items():
        for method, types in methods.items():
            print(f"  {method}, {g}-grained, per type (minus the CRF head): "
                  + ', '.join(f"{t} {row['diff']:+.2f}" for t, row in types.items()))
    out = args.out or os.path.join(args.results, OUT_NAME)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out}")
    return result


if __name__ == '__main__':
    main()
