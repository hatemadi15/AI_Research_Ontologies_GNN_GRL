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
    mention replacement (crf_mr, S2); part 2e the CRF head and plain MatSciBERT scored on
    SOFC-Exp's test articles cut into words as its training articles are (crf_cdetest
    against the CRF head; cdetest against plain MatSciBERT, baseline_retokenized_test);
    week 9 mention replacement on the multi-task model on SOFC-Exp's two levels (crf_mt_mr,
    S2, voted as crf_mt).
    Per entity type of the test set (types), its support and each system's mean F1 over
    its seeds (part 2d reports them on the slot level; not the retokenized runs).
  - errors (exploratory; cited by the week-7 protocol, part 1): per protocol and
    granularity, the kinds of error of the CRF head over its 10 seeds: the share of gold
    entities that are correct, missed (no overlapping prediction), of the right span and
    the wrong type, or boundary errors (of the right type, or of another type), and the
    share of predictions that overlap no gold entity (spurious).
  - week6_types (week 6, reported alongside its decisions): per granularity, each week-6
    method with all 10 seeds on the 4 held-out papers against the CRF head, per entity type:
    the F1 of both, paired by seed offset and paper over the pairs whose test set has the
    type, and their mean difference. Later weeks add multi-task training (crf_mt), mention
    replacement (crf_mr, week 7) and the two together (crf_mt_mr, week 9): TYPE_METHODS.
  - paper_count (exploratory): what a third training paper is worth on an unseen paper. Per
    granularity and held-out paper P, the CRF head's F1 on P trained on the 3 other papers
    (crf_m0 to _m200, leave-one-paper-out) and on 2 of them (the inner folds of week 7, part
    2a, --leave-two-papers-out with seeds 0 to 200: each fold that tests P trains on the 2
    papers other than P and its partner; the counts at D = 0), each the mean over its runs,
    and their difference with its mean and 95% CI over the 4 papers. Three papers also hold
    more sentences than two, so this is the value of a third paper with its sentences.
  - paper_diversity (week 8, part 2; its decision): more papers or more sentences. Per
    granularity and held-out paper P, the CRF head's F1 on P (seeds 0 to 200) trained on two
    of the other papers (the inner folds, as in paper_count), on all three at the size of
    the pair (crf_p3n2, --two-paper-size, folds P|Q+R), and on all three (crf); each the mean
    over its pairs and seeds. Then over the 4 papers, mean and 95% CI: diversity, three
    papers at the size of two minus two papers (the decision: does a third paper help at
    the same number of training sentences?), and size, three papers minus three papers at
    the size of two.
  - sofc_draws (week 9, part 2; its decision): SOFC-Exp's entity level trained on 10 of its
    26 training articles (--train-articles 10, 5 draws, the official validation and test
    articles). Per draw, the 10-seed means of the CRF head and of mention replacement under
    every F1 average, and over the 5 draws their means and mention replacement minus the
    CRF head with its 95% CI (passes: the CI above 0); reported, the CRF head on 10
    articles minus on all 26 (the official split's run of the same model seed; the mean
    over the draws), paired by seed.
  - mulms (week 10; its decisions): MuLMS, a third corpus, its outermost entities.
    leave_one_subfield_out: per held-out subfield (7), trained on 400 sentences of the other
    subfields' documents (--train-pool 400), the 10-seed means of the CRF head and of
    mention replacement under every F1 average, and over the 7 subfields their means and
    mention replacement minus the CRF head with its 95% CI (decision 1, passes: the CI
    above 0); then the votes over the seed triples 0-200, 300-500 and 600-800, per subfield
    the mean of its 3 votes, and mention replacement's votes minus the CRF votes (decision
    2). random_split_pool (descriptive): the CRF head on 5 random splits of a 400-sentence
    pool (the next 20% of the sentences testing), its 10-seed mean per split, and its votes
    over the same seed triples (the mean of the 3 votes per split); and the gap, random
    splits minus held-out subfields (the CRF head), per F1 average. silver_names (week 11;
    its MuLMS decisions): mention replacement with the teachers' silver names in its pool
    (crf_mrs) on the held-out subfields, its 10-seed means per subfield, minus the CRF head
    (decision 1) and minus mention replacement (decision 2), each with its 95% CI over the
    subfields; its votes against the CRF votes and mention replacement's (reported); and per
    subfield the silver names' counts (name_coverage of ner_baseline.py), with the share of
    the unseen test entities whose name is a silver name, pooled over the subfields. week12
    (week 12; its screen and MuLMS decisions): every other mention replacement variant with
    files on the held-out subfields (crf_mr<...>: names from a topic corpus, crf_mrs_topic
    and crf_mrm_topic, or more copies, crf_mr6), each analysed as crf_mrs is against the
    CRF head and mention replacement at the same seed offsets, in three parts, each present
    when the variant has every file of its offsets: screen, the offsets 0 to 200 (the means
    and CIs, no decision; the screen picks the variant by its mean minus mention
    replacement); decision, the offsets 300 to 900, seeds the screen did not see (decision_1
    and decision_2); ten_seeds, all 10 offsets (reported).
  - sofc_splits (week 8, part 1; descriptive): SOFC-Exp beyond its official split. Per
    protocol, its 5 random sentence splits and its 5 article folds (--document-folds 5), the
    CRF head's 10-seed mean per split under every F1 average, with the mean and 95% CI over
    the splits; the CRF votes over the seed triples 0-200, 300-500 and 600-800 (the mean of
    the 3 votes per split); and the gap, random splits minus article folds, per F1 average.

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
from ner_seedmeans import (DECISION_OFFSETS, GRANULARITIES, LOPO, PROTOCOLS, SCREEN_OFFSETS,
                           WEEK5_OFFSETS, WEEK5_SYSTEMS, WEEK5_TRIPLES, WEEK6_METHODS,
                           file_names, seed_family, spread, system_scores)

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
SPLIT_METHODS = ('crf_deberta', 'crf_ctx', 'crf_dc', 'crf_dcr', 'crf_ebv', 'crf_mt', 'crf_mr',
                 'crf_cdetest', 'crf_mt_mr')
# week 7, part 2e: plain MatSciBERT scored on SOFC-Exp's retokenized test articles
RETOKENIZED_BASELINE = 'cdetest'
SOFC_PROTOCOLS = ('random_split_vary', 'document_folds')     # week 8, part 1
ARTICLE_DRAWS = 'article_draws'                               # week 9, part 2
MULMS_PROTOCOLS = ('leave_one_subfield_out', 'random_split_pool')   # week 10
VOTED_METHODS = ('crf_mt', 'crf_mt_mr')
TYPE_METHODS = WEEK6_METHODS + ('crf_mt', 'crf_mr', 'crf_mt_mr')   # week6_types


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
    """{granularity: {method: type_differences}} on the 4 held-out papers, for the methods of
    TYPE_METHODS whose 10 seeds and the CRF head's are all there."""
    out = {}
    for g in GRANULARITIES:
        crf = load_files(directory, file_names(g, LOPO, 'crf', WEEK5_SYSTEMS))
        for method in TYPE_METHODS:
            family = WEEK5_SYSTEMS.get(method) or seed_family(method)
            files = load_files(directory, file_names(g, LOPO, method, {method: family}))
            if crf and files:
                out.setdefault(g, {})[method] = type_differences(files, crf)
    return out


def split_types(systems):
    """Per entity type of a published split's test set, the most frequent first: its support
    and each system's mean F1 over its seeds ({system: {run_id: run}}). Types that only a
    prediction has are left out."""
    rows = {}
    for name, by_id in systems.items():
        for run in by_id.values():
            for t, s in run['per_class'].items():
                if t not in AVERAGES and s['support']:
                    row = rows.setdefault(t, {'support': int(s['support'])})
                    row.setdefault(name, []).append(100 * s['f1-score'])
    return {t: {k: v if k == 'support' else round(statistics.mean(v), 2)
                for k, v in row.items()}
            for t, row in sorted(rows.items(), key=lambda kv: (-kv[1]['support'], kv[0]))}


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
        typed = dict(runs)
        for method in SPLIT_METHODS:
            found = load_files(directory, [f'ner_results_{g}_published_split_{method}.json'])
            if not found:
                continue
            by_id = {r['run_id']: r for r in found[0]['runs']}
            if method != f'crf_{RETOKENIZED_BASELINE}':      # another test set's words
                typed[method] = by_id
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
        found = load_files(directory,
                           [f'ner_results_{g}_published_split_{RETOKENIZED_BASELINE}.json'])
        if found:
            by_id = {r['run_id']: r for r in found[0]['runs']}
            ids = [i for i in runs['baseline'] if i in by_id]
            entry['baseline_retokenized_test'] = {
                **{avg: spread([get(r) for r in by_id.values()])
                   for avg, get in F1_AVERAGES.items()},
                'minus_baseline': {avg: spread([get(by_id[i]) - get(runs['baseline'][i])
                                                for i in ids], 'mean_diff')
                                   for avg, get in F1_AVERAGES.items()},
                'n_pairs': len(ids)}
        entry['types'] = split_types(typed)
        out[g] = entry
    return out


def paper_count(directory):
    """The paper_count section (see the module docstring); granularities without all the files
    are left out."""
    out = {}
    for g in GRANULARITIES:
        two = load_files(directory, [f'ner_results_{g}_leave_two_papers_out_crf_bias_m{off}.json'
                                     for off in SCREEN_OFFSETS])
        three = load_files(directory, [f'ner_results_{g}_{LOPO}_crf_m{off}.json'
                                       for off in SCREEN_OFFSETS])
        if not two or not three:
            continue
        f1 = {'two': {}, 'three': {}}
        for f in two:
            for run in f['runs']:
                for paper, (correct, pred, gold) in run['bias_test']['0'].items():
                    f1['two'].setdefault(paper, []).append(
                        200 * correct / (pred + gold) if pred + gold else 0.0)
        for f in three:
            for run in f['runs']:
                f1['three'].setdefault(run['run_id'], []).append(run['f1'])
        papers = sorted(set(f1['two']) & set(f1['three']))
        rows = {p: {k: round(statistics.mean(f1[k][p]), 2) for k in ('two', 'three')}
                for p in papers}
        out[g] = {'papers': {p: {**r, 'n_two': len(f1['two'][p]),
                                 'n_three': len(f1['three'][p])} for p, r in rows.items()},
                  'two': spread([r['two'] for r in rows.values()]),
                  'three': spread([r['three'] for r in rows.values()]),
                  'three_minus_two': spread([r['three'] - r['two'] for r in rows.values()],
                                            'mean_diff')}
    return out


def paper_diversity(directory):
    """The paper_diversity section (see the module docstring); a granularity is left out
    unless every held-out paper has every pair, with every seed, on every side."""
    out = {}
    for g in GRANULARITIES:
        names = {kind: [f'ner_results_{g}_{protocol}_{variant}_m{off}.json'
                        for off in SCREEN_OFFSETS]
                 for kind, protocol, variant in (('two', 'leave_two_papers_out', 'crf_bias'),
                                                 ('sized', LOPO, 'crf_p3n2'),
                                                 ('full', LOPO, 'crf'))}
        files = {kind: load_files(directory, n) for kind, n in names.items()}
        if not all(files.values()):
            continue
        f1 = {kind: {} for kind in files}
        for f in files['two']:                    # held out: P and S; trained on the pair
            for run in f['runs']:
                held = run['run_id'].split('+')
                for paper, (correct, pred, gold) in run['bias_test']['0'].items():
                    third = next(h for h in held if h != paper)
                    f1['two'].setdefault((paper, third), []).append(
                        200 * correct / (pred + gold) if pred + gold else 0.0)
        for f in files['sized']:
            for run in f['runs']:
                paper, pair = run['run_id'].split('|')
                f1['sized'].setdefault((paper, pair), []).append(run['f1'])
        for f in files['full']:
            for run in f['runs']:
                f1['full'].setdefault(run['run_id'], []).append(run['f1'])
        papers = sorted(f1['full'])
        rows, complete = {}, True
        for paper in papers:
            others = [p for p in papers if p != paper]
            pairs = [(pair, next(p for p in others if p not in pair))
                     for pair in itertools.combinations(others, 2)]
            keys = [((paper, third), (paper, '+'.join(pair))) for pair, third in pairs]
            n = len(SCREEN_OFFSETS)
            if not all(len(f1['two'].get(k2, ())) == n and len(f1['sized'].get(k3, ())) == n
                       for k2, k3 in keys) or len(f1['full'][paper]) != n:
                complete = False
                break
            rows[paper] = {'two': statistics.mean(statistics.mean(f1['two'][k2])
                                                  for k2, _ in keys),
                           'three_at_two': statistics.mean(statistics.mean(f1['sized'][k3])
                                                           for _, k3 in keys),
                           'three': statistics.mean(f1['full'][paper])}
        if not complete:
            continue
        out[g] = {'papers': {p: {k: round(v, 2) for k, v in r.items()}
                             for p, r in rows.items()},
                  **{k: spread([r[k] for r in rows.values()])
                     for k in ('two', 'three_at_two', 'three')},
                  'diversity': spread([r['three_at_two'] - r['two'] for r in rows.values()],
                                      'mean_diff'),
                  'size': spread([r['three'] - r['three_at_two'] for r in rows.values()],
                                 'mean_diff')}
        out[g]['diversity']['passes'] = out[g]['diversity']['ci95'][0] > 0
    return out


def sofc_splits(directory):
    """The sofc_splits section (see the module docstring); protocols without all 10 seed files
    of the CRF head are left out, the votes too without their predictions."""
    out = {}
    for protocol in SOFC_PROTOCOLS:
        files = load_files(directory, [f'ner_results_sofc_{protocol}_crf_m{off}.json'
                                       for off in WEEK5_OFFSETS])
        if not files:
            continue
        per_split = system_scores(files)
        entry = {avg: spread([v[avg] for v in per_split.values()]) for avg in F1_AVERAGES}
        entry['n_splits'] = len(per_split)
        if all('test_pred' in r for f in files for r in f['runs']):
            by_id = [{r['run_id']: r for r in f['runs']} for f in files]
            votes = {i: [ner_ensemble.ensemble_run(i, [by_id[first // 100 + j][i]
                                                       for j in range(3)])
                         for first in WEEK5_TRIPLES] for i in per_split}
            entry['crf_votes'] = {avg: spread([statistics.mean(get(v) for v in votes[i])
                                               for i in per_split])
                                  for avg, get in F1_AVERAGES.items()}
        out[protocol] = entry
    if len(out) == len(SOFC_PROTOCOLS):
        out['gap'] = {avg: round(out[SOFC_PROTOCOLS[0]][avg]['mean']
                                 - out[SOFC_PROTOCOLS[1]][avg]['mean'], 2)
                      for avg in F1_AVERAGES}
    return out


def sofc_draws(directory):
    """The sofc_draws section (see the module docstring); {} without all 10 seed files of
    both systems."""
    files = {s: load_files(directory, [f'ner_results_sofc_{ARTICLE_DRAWS}_{s}_m{off}.json'
                                       for off in WEEK5_OFFSETS]) for s in ('crf', 'crf_mr')}
    if not all(files.values()):
        return {}
    scores = {s: system_scores(f) for s, f in files.items()}
    ids = [i for i in scores['crf'] if i in scores['crf_mr']]
    out = {s: {avg: spread([scores[s][i][avg] for i in ids]) for avg in F1_AVERAGES}
           for s in scores}
    out['crf_mr_minus_crf'] = {
        avg: spread([scores['crf_mr'][i][avg] - scores['crf'][i][avg] for i in ids],
                    'mean_diff') for avg in F1_AVERAGES}
    out['n_draws'] = len(ids)
    out['passes'] = out['crf_mr_minus_crf']['micro']['ci95'][0] > 0
    official = load_files(directory, ['ner_results_sofc_published_split_crf.json'])
    if official:
        by_seed = {r['run_id']: r['f1'] for r in official[0]['runs']}
        pairs = [(statistics.mean(r['f1'] for r in f['runs']), by_seed[f'seed{off}'])
                 for off, f in zip(WEEK5_OFFSETS, files['crf']) if f'seed{off}' in by_seed]
        if pairs:
            out['ten_minus_all_articles'] = spread([a - b for a, b in pairs], 'mean_diff')
    return out


def fold_votes(files):
    """{run_id: {F1 average: the mean over the seed triples 0-200, 300-500 and 600-800 of
    their votes}} of a 10-seed family's files (ordered by seed offset), from the stored
    predictions; of the triples the files cover (the 3-seed screen: the first one)."""
    by_id = [{r['run_id']: r for r in f['runs']} for f in files]
    triples = [first for first in WEEK5_TRIPLES if first // 100 + 3 <= len(by_id)]
    votes = {i: [ner_ensemble.ensemble_run(i, [by_id[first // 100 + j][i] for j in range(3)])
                 for first in triples] for i in by_id[0]}
    return {i: {avg: statistics.mean(get(v) for v in vs) for avg, get in F1_AVERAGES.items()}
            for i, vs in votes.items()}


def mulms(directory):
    """The mulms section (see the module docstring); a part without all 10 seed files of its
    systems is left out, the votes too without their predictions."""
    out = {}
    files = {s: load_files(directory, [f'ner_results_mulms_{MULMS_PROTOCOLS[0]}_{s}_m{off}.json'
                                       for off in WEEK5_OFFSETS]) for s in ('crf', 'crf_mr')}
    if all(files.values()):
        scores = {s: system_scores(f) for s, f in files.items()}
        ids = [i for i in scores['crf'] if i in scores['crf_mr']]
        entry = {s: {avg: spread([scores[s][i][avg] for i in ids]) for avg in F1_AVERAGES}
                 for s in scores}
        entry['subfields'] = {i: {s: round(scores[s][i]['micro'], 2) for s in scores}
                              for i in ids}
        entry['crf_mr_minus_crf'] = {
            avg: spread([scores['crf_mr'][i][avg] - scores['crf'][i][avg] for i in ids],
                        'mean_diff') for avg in F1_AVERAGES}
        entry['n_subfields'] = len(ids)
        entry['passes'] = entry['crf_mr_minus_crf']['micro']['ci95'][0] > 0
        if all('test_pred' in r for fs in files.values() for f in fs for r in f['runs']):
            votes = {s: fold_votes(f) for s, f in files.items()}
            entry['votes'] = {s: {avg: spread([votes[s][i][avg] for i in ids])
                                  for avg in F1_AVERAGES} for s in votes}
            entry['votes']['crf_mr_minus_crf'] = {
                avg: spread([votes['crf_mr'][i][avg] - votes['crf'][i][avg] for i in ids],
                            'mean_diff') for avg in F1_AVERAGES}
            entry['votes']['passes'] = (
                entry['votes']['crf_mr_minus_crf']['micro']['ci95'][0] > 0)
        out[MULMS_PROTOCOLS[0]] = entry
    pool = load_files(directory, [f'ner_results_mulms_{MULMS_PROTOCOLS[1]}_crf_m{off}.json'
                                  for off in WEEK5_OFFSETS])
    if pool:
        per_split = system_scores(pool)
        entry = {avg: spread([v[avg] for v in per_split.values()]) for avg in F1_AVERAGES}
        entry['n_splits'] = len(per_split)
        if all('test_pred' in r for f in pool for r in f['runs']):
            votes = fold_votes(pool)
            entry['crf_votes'] = {avg: spread([votes[i][avg] for i in per_split])
                                  for avg in F1_AVERAGES}
        out[MULMS_PROTOCOLS[1]] = entry
    if all(p in out for p in MULMS_PROTOCOLS):
        out['gap'] = {avg: round(out[MULMS_PROTOCOLS[1]][avg]['mean']
                                 - out[MULMS_PROTOCOLS[0]]['crf'][avg]['mean'], 2)
                      for avg in F1_AVERAGES}
    mrs = load_files(directory, [f'ner_results_mulms_{MULMS_PROTOCOLS[0]}_crf_mrs_m{off}.json'
                                 for off in WEEK5_OFFSETS])
    if mrs and all(files.values()):
        out['silver_names'] = silver_names({**files, 'crf_mrs': mrs})
    week12 = week12_variants(directory)
    if week12:
        out['week12'] = week12
    return out


def silver_names(files):
    """The silver_names section of mulms (week 11) from the 10 seed files of crf, crf_mr and
    crf_mrs on the held-out subfields."""
    return names_variant(files, 'crf_mrs')


def names_variant(files, system, decide=True):
    """A mention replacement variant (`system`) against the CRF head and mention replacement
    on the held-out subfields, from files {'crf', 'crf_mr', system} of the same seed offsets:
    its means per subfield and over them, minus each reference with the 95% CI over the
    subfields (and, with `decide`, decision_1 and decision_2: the CI above 0), the votes when
    the predictions are saved, and the silver names' counts when the runs have them."""
    scores = {s: system_scores(f) for s, f in files.items()}
    ids = [i for i in scores[system] if all(i in v for v in scores.values())]

    def minus(get, reference):
        return {avg: spread([get(system, i)[avg] - get(reference, i)[avg] for i in ids],
                            'mean_diff') for avg in F1_AVERAGES}
    entry = {system: {avg: spread([scores[system][i][avg] for i in ids])
                      for avg in F1_AVERAGES},
             'subfields': {i: round(scores[system][i]['micro'], 2) for i in ids},
             'n_subfields': len(ids)}
    for reference, decision in (('crf', 'decision_1'), ('crf_mr', 'decision_2')):
        entry[f'minus_{reference}'] = minus(lambda s, i: scores[s][i], reference)
        if decide:
            entry[decision] = entry[f'minus_{reference}']['micro']['ci95'][0] > 0
    if all('test_pred' in r for fs in files.values() for f in fs for r in f['runs']):
        votes = {s: fold_votes(f) for s, f in files.items()}
        entry['votes'] = {system: {avg: spread([votes[system][i][avg] for i in ids])
                                   for avg in F1_AVERAGES},
                          **{f'minus_{r}': minus(lambda s, i: votes[s][i], r)
                             for r in ('crf', 'crf_mr')}}
    runs = files[system][0]['runs']
    if all('names' in r.get('self_training', {}) for r in runs):
        names = {r['run_id']: r['self_training']['names'] for r in runs}
        entry['names'] = {i: names[i] for i in ids}
        unseen = sum(names[i]['n_test_unseen'] for i in ids)
        entry['unseen_in_silver'] = (round(sum(names[i]['n_test_unseen_in_silver']
                                               for i in ids) / unseen, 3) if unseen else None)
        filters = {r['self_training'].get('silver_filter') for r in runs}
        if len(filters) == 1:
            entry['silver_filter'] = filters.pop()
    return entry


def week12_variants(directory):
    """The week12 section of mulms: {variant: {part: names_variant(...)}} for every mention
    replacement variant other than crf_mr and crf_mrs with files on the held-out subfields;
    the parts screen (SCREEN_OFFSETS, no decision), decision (DECISION_OFFSETS) and ten_seeds
    (WEEK5_OFFSETS, no decision), each present when the variant and both references have
    every file of its offsets."""
    prefix = f'ner_results_mulms_{MULMS_PROTOCOLS[0]}_'
    variants = set()
    for path in sorted(Path(directory).glob(f'{prefix}crf_mr*_m*.json')):
        variant, _, offset = path.name[len(prefix):-len('.json')].rpartition('_m')
        if variant not in ('crf_mr', 'crf_mrs') and offset.isdigit():
            variants.add(variant)
    out = {}
    for variant in sorted(variants):
        entry = {}
        for part, offsets, decide in (('screen', SCREEN_OFFSETS, False),
                                      ('decision', DECISION_OFFSETS, True),
                                      ('ten_seeds', WEEK5_OFFSETS, False)):
            files = {s: load_files(directory, [f'{prefix}{s}_m{off}.json' for off in offsets])
                     for s in ('crf', 'crf_mr', variant)}
            if all(files.values()):
                entry[part] = {'seeds': list(offsets), **names_variant(files, variant, decide)}
        if entry:
            out[variant] = entry
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
    out['sofc_splits'] = sofc_splits(directory)
    out['sofc_draws'] = sofc_draws(directory)
    out['mulms'] = mulms(directory)
    out['paper_count'] = paper_count(directory)
    out['paper_diversity'] = paper_diversity(directory)
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
        if 'baseline_retokenized_test' in entry:
            m = entry['baseline_retokenized_test']
            d = m['minus_baseline']['micro']
            print(f"    plain MatSciBERT on the retokenized test articles: {m['micro']['mean']:.2f} "
                  f"{m['micro']['ci95']}, minus plain MatSciBERT {d['mean_diff']:+.2f} "
                  f"{d['ci95']} over {m['n_pairs']} seeds")
    for g, entry in result['paper_count'].items():
        d = entry['three_minus_two']
        print(f"  {g}-grained, an unseen paper: trained on 2 papers {entry['two']['mean']:.2f}, "
              f"on 3 {entry['three']['mean']:.2f}; the third paper {d['mean_diff']:+.2f} "
              f"{d['ci95']}")
    for g, entry in result['paper_diversity'].items():
        d, z = entry['diversity'], entry['size']
        print(f"  {g}-grained, three papers at the size of two minus two papers: "
              f"{d['mean_diff']:+.2f} {d['ci95']} ({'passes' if d['passes'] else 'no'}); "
              f"three papers minus three at the size of two: {z['mean_diff']:+.2f} {z['ci95']}")
    for protocol, entry in result['sofc_splits'].items():
        if protocol != 'gap':
            print(f"  SOFC-Exp, {protocol}: the CRF head {entry['micro']['mean']:.2f} "
                  f"{entry['micro']['ci95']} over {entry['n_splits']} splits")
    if result['sofc_draws']:
        e = result['sofc_draws']
        d = e['crf_mr_minus_crf']['micro']
        print(f"  SOFC-Exp on 10 training articles ({e['n_draws']} draws): the CRF head "
              f"{e['crf']['micro']['mean']:.2f}, mention replacement "
              f"{e['crf_mr']['micro']['mean']:.2f}, minus the CRF head {d['mean_diff']:+.2f} "
              f"{d['ci95']} ({'passes' if e['passes'] else 'no'})")
    held = result['mulms'].get(MULMS_PROTOCOLS[0])
    if held:
        d = held['crf_mr_minus_crf']['micro']
        print(f"  MuLMS, held-out subfields ({held['n_subfields']}): the CRF head "
              f"{held['crf']['micro']['mean']:.2f}, mention replacement "
              f"{held['crf_mr']['micro']['mean']:.2f}, minus the CRF head {d['mean_diff']:+.2f} "
              f"{d['ci95']} ({'passes' if held['passes'] else 'no'})")
        if 'votes' in held:
            v = held['votes']['crf_mr_minus_crf']['micro']
            print(f"    votes: the CRF votes {held['votes']['crf']['micro']['mean']:.2f}, "
                  f"mention replacement's {held['votes']['crf_mr']['micro']['mean']:.2f}, "
                  f"minus {v['mean_diff']:+.2f} {v['ci95']} "
                  f"({'passes' if held['votes']['passes'] else 'no'})")
    if 'gap' in result['mulms']:
        print(f"  MuLMS, random pool splits minus held-out subfields: "
              f"{result['mulms']['gap']['micro']:+.2f} micro F1")
    names = result['mulms'].get('silver_names')
    if names:
        for reference, decision in (('crf', 'decision_1'), ('crf_mr', 'decision_2')):
            d = names[f'minus_{reference}']['micro']
            print(f"  MuLMS, silver names ({names['n_subfields']} subfields): crf_mrs "
                  f"{names['crf_mrs']['micro']['mean']:.2f}, minus {reference} "
                  f"{d['mean_diff']:+.2f} {d['ci95']} ({'passes' if names[decision] else 'no'})")
        print(f"    unseen test names among the silver names: {names['unseen_in_silver']}")
    for variant, parts in result['mulms'].get('week12', {}).items():
        for part, entry in parts.items():
            for reference, decision in (('crf', 'decision_1'), ('crf_mr', 'decision_2')):
                d = entry[f'minus_{reference}']['micro']
                verdict = ('passes' if entry[decision] else 'no') if decision in entry else part
                print(f"  MuLMS, week 12 {variant}, {part} ({len(entry['seeds'])} seeds, "
                      f"{entry['n_subfields']} subfields): {entry[variant]['micro']['mean']:.2f}"
                      f", minus {reference} {d['mean_diff']:+.2f} {d['ci95']} ({verdict})")
            if 'unseen_in_silver' in entry:
                print(f"    unseen test names among the silver names: "
                      f"{entry['unseen_in_silver']} ({entry.get('silver_filter')} filter)")
    if 'gap' in result['sofc_splits']:
        print(f"  SOFC-Exp, random splits minus article folds: "
              f"{result['sofc_splits']['gap']['micro']:+.2f} micro F1")
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
