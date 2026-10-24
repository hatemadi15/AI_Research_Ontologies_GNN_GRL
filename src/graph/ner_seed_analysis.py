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
    and crf_mrm_topic, or more copies, crf_mr6; not the consistency rule's outputs _dc and
    _dcr, which belong to week13), each analysed as crf_mrs is against the
    CRF head and mention replacement at the same seed offsets, in three parts, each present
    when the variant has every file of its offsets: screen, the offsets 0 to 200 (the means
    and CIs, no decision; the screen picks the variant by its mean minus mention
    replacement); decision, the offsets 300 to 900, seeds the screen did not see (decision_1
    and decision_2); ten_seeds, all 10 offsets (reported).
  - week13 (week 13, the methods push; its decisions 13.1a, 13.1b and 13.1d, and the pooled
    estimates of 13.2 and 13.3): consistency, the document consistency rule of week 7
    (ner_consistency.py; _dc, and _dcr with retyping) applied to the saved test predictions
    of held-out documents, per corpus (MuLMS's held-out subfields, MaterioMiner's held-out
    papers at both granularities, SOFC-Exp's article folds) and per input system (the CRF
    head, mention replacement, its multi-task combination) with all 10 seed files of the
    input and the output: per unit (a subfield, a paper, a fold) the 10-seed means of both,
    the output minus the input with its mean and 95% CI over the units, the entities the
    rule adds and retypes per seed, and, where the comparison is a pre-registered decision
    (CONSISTENCY_DECISIONS: week 7 on the CRF head on MaterioMiner, 13.1a and 13.1b on
    MuLMS, 13.1d on mention replacement on MaterioMiner), its verdict (passes: the CI above
    0); the rest is reported. pooled: a method against the CRF head over the held-out units
    of every corpus that has all 10 seed files of both (POOLED_METHODS names the method's
    variant on each corpus: a MuLMS topic variant and its MaterioMiner analogue, mention
    replacement and its multi-task combination on SOFC-Exp's folds as well), per corpus the
    units' differences with their mean and 95% CI and whether the corpus loses (its CI below
    0), and over all units the pooled mean and 95% CI (passes: above 0); the pooled
    comparison is the decision 13.2 for the week-12 screen's winner only (week12_winner:
    the screen variant with the largest 3-seed mean gain over mention replacement, named
    once every variant has its screen, if that gain is above 0), reported for every other
    method (plain mention replacement's pooled estimate is post hoc, ROADMAP §4, week 13).
    folds: 13.3, mention replacement (crf_mr, decision 13.3a) and its multi-task combination
    (crf_mt_mr, decision 13.3b) on SOFC-Exp's 5 article folds (batches w13a and w13c) against week 8's
    CRF head on the folds, each with all 10 seed files of both: per fold the 10-seed means,
    the method minus the CRF head with its mean and 95% CI over the folds (passes: above 0),
    and, for the multi-task combination, its difference against mention replacement and its
    slot level (sofc_slot_document_folds; reported); 13.4 (conditional on week 12's decision
    1): the week-12 winner's analogue crf_mrm6 on the folds (batch w13b) against the CRF
    head, decision 13.4, and against mention replacement (reported); the pooled entry of
    the winner then carries with_sofc_folds, the three-corpus estimate (POOLED_EXTENSIONS;
    reported, 13.2 stays the two-corpus decision).
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
# week 13: the consistency rule's inputs per corpus (the results-file prefix and the systems
# it was applied to), the comparisons that are pre-registered decisions (the others are
# reported), the corpora of the pooled estimates and each method's variant on them
CONSISTENCY_CORPORA = {
    'mulms': ('mulms_leave_one_subfield_out', ('crf', 'crf_mr')),
    'fine': ('fine_leave_one_paper_out', ('crf', 'crf_mr', 'crf_mt_mr')),
    'coarse': ('coarse_leave_one_paper_out', ('crf', 'crf_mr', 'crf_mt_mr')),
    'sofc_folds': ('sofc_document_folds', ('crf',)),
}
CONSISTENCY_DECISIONS = {('mulms', 'crf'): '13.1a', ('mulms', 'crf_mr'): '13.1b',
                         ('fine', 'crf'): 'week 7', ('coarse', 'crf'): 'week 7',
                         ('fine', 'crf_mr'): '13.1d', ('coarse', 'crf_mr'): '13.1d'}
POOLED_CORPORA = {'mulms': 'mulms_leave_one_subfield_out', 'fine': 'fine_leave_one_paper_out',
                  'sofc_folds': 'sofc_document_folds'}
POOLED_METHODS = {
    'crf_mr': {'mulms': 'crf_mr', 'fine': 'crf_mr', 'sofc_folds': 'crf_mr'},
    'crf_mt_mr': {'fine': 'crf_mt_mr', 'sofc_folds': 'crf_mt_mr'},
    'crf_mrs': {'mulms': 'crf_mrs', 'fine': 'crf_mrs'},
    'crf_mrs_topic': {'mulms': 'crf_mrs_topic', 'fine': 'crf_mrs'},     # V1's analogue: w11b
    'crf_mrm_topic': {'mulms': 'crf_mrm_topic', 'fine': 'crf_mrm'},
    'crf_mr6': {'mulms': 'crf_mr6', 'fine': 'crf_mr6'},
    'crf_mrm_topic6': {'mulms': 'crf_mrm_topic6', 'fine': 'crf_mrm6'},
    'crf_st3_topic': {'mulms': 'crf_st3_topic', 'fine': 'crf_st3'},     # V6's analogue: week 4
    'crf_gzm_topic': {'mulms': 'crf_gzm_topic', 'fine': 'crf_gzm'},
    'crf_gzs_topic': {'mulms': 'crf_gzs_topic', 'fine': 'crf_gzs'},
}
SCREEN_VARIANTS = ('crf_mrs_topic', 'crf_mrm_topic', 'crf_mr6', 'crf_mrm_topic6',
                   'crf_st3_topic', 'crf_gzm_topic', 'crf_gzs_topic')     # the week-12 screen


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
    """The week12 section of mulms: {variant: {part: names_variant(...)}} for every variant
    other than crf, crf_mr and crf_mrs with files on the held-out subfields (crf_mrs_topic,
    crf_mrm_topic, crf_mrm_topic6, crf_st3_topic, crf_mr6); the consistency rule's outputs
    (*_dc and *_dcr, no training; the week13 section) are not variants;
    the parts screen (SCREEN_OFFSETS, no decision), decision (DECISION_OFFSETS) and ten_seeds
    (WEEK5_OFFSETS, no decision), each present when the variant and both references have
    every file of its offsets."""
    prefix = f'ner_results_mulms_{MULMS_PROTOCOLS[0]}_'
    variants = set()
    for path in sorted(Path(directory).glob(f'{prefix}crf_*_m*.json')):
        variant, _, offset = path.name[len(prefix):-len('.json')].rpartition('_m')
        if (variant not in ('crf', 'crf_mr', 'crf_mrs') and offset.isdigit()
                and variant.rpartition('_')[2] not in ('dc', 'dcr')):
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


FOLDS_PREFIX = 'sofc_document_folds'                   # week 8's article folds, fold0 to fold4
FOLDS_DECISIONS = {'crf_mr': '13.3a', 'crf_mt_mr': '13.3b',     # week 13, 13.3: against crf
                   'crf_mrm6': '13.4'}                          # 13.4: the winner's analogue
# the three-corpus estimate of a pooled method (reported): its variant on SOFC-Exp's folds
POOLED_EXTENSIONS = {'crf_mrm_topic6': {'sofc_folds': 'crf_mrm6'}}


def folds_entry(method_files, crf_files, decision):
    """A method against the CRF head on the article folds (10 seed files each, paired by
    fold): both means, the difference with its 95% CI over the folds, and the verdict of
    `decision`."""
    scores = {'system': system_scores(method_files), 'crf': system_scores(crf_files)}
    ids = [i for i in scores['system'] if i in scores['crf']]
    entry = {side: {avg: spread([scores[side][i][avg] for i in ids]) for avg in F1_AVERAGES}
             for side in scores}
    entry['minus_crf'] = {
        avg: spread([scores['system'][i][avg] - scores['crf'][i][avg] for i in ids],
                    'mean_diff') for avg in F1_AVERAGES}
    entry['units'] = {i: round(scores['system'][i]['micro'] - scores['crf'][i]['micro'], 2)
                      for i in ids}
    entry['n_units'] = len(ids)
    entry['decision'] = decision
    entry['passes'] = entry['minus_crf']['micro']['ci95'][0] > 0
    return entry


def folds(directory):
    """The folds part of week13 (see the module docstring): {method: entry} for every
    method of FOLDS_DECISIONS with all 10 seed files on SOFC-Exp's article folds, against
    the CRF head's; the multi-task combination's entry adds its difference against mention
    replacement (minus_crf_mr) and its slot level (slot), reported, when their files
    exist."""
    crf = load_files(directory, [f'ner_results_{FOLDS_PREFIX}_crf_m{off}.json'
                                 for off in WEEK5_OFFSETS])
    out, files = {}, {}
    if not crf:
        return out
    for method, decision in FOLDS_DECISIONS.items():
        files[method] = load_files(directory, [f'ner_results_{FOLDS_PREFIX}_{method}_m{off}.json'
                                               for off in WEEK5_OFFSETS])
        if files[method]:
            out[method] = folds_entry(files[method], crf, decision)
    if files.get('crf_mr'):
        mr = system_scores(files['crf_mr'])
        for method, f in files.items():
            if method == 'crf_mr' or not f:
                continue
            scores = system_scores(f)
            ids = [i for i in scores if i in mr]
            out[method]['minus_crf_mr'] = {
                avg: spread([scores[i][avg] - mr[i][avg] for i in ids], 'mean_diff')
                for avg in F1_AVERAGES}
    slot = load_files(directory, [f'ner_results_sofc_slot_document_folds_crf_mt_mr_m{off}.json'
                                  for off in WEEK5_OFFSETS])
    if slot and 'crf_mt_mr' in out:
        per_fold = system_scores(slot)
        out['crf_mt_mr']['slot'] = {avg: spread([v[avg] for v in per_fold.values()])
                                    for avg in F1_AVERAGES}
        out['crf_mt_mr']['slot']['n_units'] = len(per_fold)
    return out


def consistency_entry(inputs, outputs, decision=None):
    """The rule's output family against its input family (10 seed files each, paired by
    run_id): both families' means, the difference with its 95% CI over the units, the
    entities added and retyped per seed, and the verdict when the comparison is a
    decision (`decision` names it)."""
    scores = {'input': system_scores(inputs), 'output': system_scores(outputs)}
    ids = [i for i in scores['input'] if i in scores['output']]
    entry = {side: {avg: spread([scores[side][i][avg] for i in ids]) for avg in F1_AVERAGES}
             for side in scores}
    entry['minus_input'] = {
        avg: spread([scores['output'][i][avg] - scores['input'][i][avg] for i in ids],
                    'mean_diff') for avg in F1_AVERAGES}
    entry['units'] = {i: round(scores['output'][i]['micro'] - scores['input'][i]['micro'], 2)
                      for i in ids}
    entry['n_units'] = len(ids)
    for key in ('n_added', 'n_retyped'):
        entry[f'{key}_per_seed'] = round(sum(r.get(key, 0) for f in outputs for r in f['runs']
                                             if r['run_id'] in ids) / len(outputs), 1)
    if decision:
        entry['decision'] = decision
        entry['passes'] = entry['minus_input']['micro']['ci95'][0] > 0
    return entry


def consistency(directory):
    """The consistency part of week13 (see the module docstring): {corpus: {output variant:
    consistency_entry}} for every input system whose input and output families have all
    their files; a corpus without any is left out."""
    out = {}
    for corpus, (prefix, systems) in CONSISTENCY_CORPORA.items():
        for system in systems:
            inputs = load_files(directory, [f'ner_results_{prefix}_{system}_m{off}.json'
                                            for off in WEEK5_OFFSETS])
            if not inputs:
                continue
            for rule in ('dc', 'dcr'):
                outputs = load_files(directory, [f'ner_results_{prefix}_{system}_{rule}_m{off}.json'
                                                 for off in WEEK5_OFFSETS])
                if outputs:
                    decision = CONSISTENCY_DECISIONS.get((corpus, system)) if rule == 'dc' else None
                    out.setdefault(corpus, {})[f'{system}_{rule}'] = consistency_entry(
                        inputs, outputs, decision)
    return out


def week12_winner(week12):
    """The week-12 screen's winner (ROADMAP §4, week 12): the screen variant with the largest
    3-seed mean micro-F1 gain over mention replacement, if that gain is above 0; None until
    every variant of SCREEN_VARIANTS has its screen in the week12 section."""
    gains = {v: week12[v]['screen']['minus_crf_mr']['micro']['mean_diff']
             for v in SCREEN_VARIANTS if 'screen' in (week12 or {}).get(v, {})}
    if len(gains) < len(SCREEN_VARIANTS):
        return None
    best = max(gains, key=gains.get)
    return best if gains[best] > 0 else None


def pooled(directory, winner=None):
    """The pooled part of week13 (see the module docstring): {method: entry} for every method
    of POOLED_METHODS with all 10 seed files of its variant and of the CRF head on two or
    more of POOLED_CORPORA: per corpus its units' differences against the CRF head, and the
    pooled difference over every unit; the entry is the decision 13.2 for `winner` (the
    week-12 screen's winner), reported otherwise."""
    out = {}
    for method, variants in POOLED_METHODS.items():
        corpora, diffs = {}, []
        for corpus, variant in variants.items():
            entry = pooled_corpus(directory, corpus, variant)
            if entry is None:
                continue
            corpora[corpus] = entry
            diffs.extend(entry['units'].values())
        if len(corpora) < 2:
            continue
        entry = {'corpora': corpora, 'n_units': len(diffs),
                 'minus_crf': spread(diffs, 'mean_diff'),
                 'status': 'decision' if method == winner else 'reported'}
        entry['passes'] = entry['minus_crf']['ci95'][0] > 0
        entry['no_corpus_loses'] = not any(c['loses'] for c in corpora.values())
        for corpus, variant in POOLED_EXTENSIONS.get(method, {}).items():
            extra = pooled_corpus(directory, corpus, variant)
            if extra is None:
                continue
            more = {**corpora, corpus: extra}
            all_diffs = diffs + list(extra['units'].values())
            entry[f'with_{corpus}'] = {
                'corpora': {c: {'variant': v['variant'], 'n_units': v['n_units']} for c, v in more.items()},
                corpus: extra, 'n_units': len(all_diffs),
                'minus_crf': spread(all_diffs, 'mean_diff'), 'status': 'reported'}
            entry[f'with_{corpus}']['passes'] = entry[f'with_{corpus}']['minus_crf']['ci95'][0] > 0
            entry[f'with_{corpus}']['no_corpus_loses'] = not any(c['loses'] for c in more.values())
        out[method] = entry
    return out


def pooled_corpus(directory, corpus, variant):
    """A method's variant against the CRF head on one corpus of POOLED_CORPORA (the entry of
    pooled's corpora), or None without all 10 seed files of both."""
    prefix = POOLED_CORPORA[corpus]
    files = {s: load_files(directory, [f'ner_results_{prefix}_{s}_m{off}.json'
                                       for off in WEEK5_OFFSETS]) for s in ('crf', variant)}
    if not all(files.values()):
        return None
    scores = {s: system_scores(f) for s, f in files.items()}
    ids = [i for i in scores[variant] if i in scores['crf']]
    units = [scores[variant][i]['micro'] - scores['crf'][i]['micro'] for i in ids]
    entry = {'variant': variant, 'n_units': len(ids), 'minus_crf': spread(units, 'mean_diff'),
             'units': {i: round(d, 2) for i, d in zip(ids, units)}}
    entry['loses'] = entry['minus_crf']['ci95'][1] < 0
    return entry


def week13(directory, week12=None):
    """The week13 section (see the module docstring); week12 is the mulms section's week12
    part, which names the screen's winner."""
    winner = week12_winner(week12)
    return {'consistency': consistency(directory), 'week12_winner': winner,
            'pooled': pooled(directory, winner), 'folds': folds(directory)}


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
    out['week13'] = week13(directory, out['mulms'].get('week12'))
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
    for corpus, entries in result['week13']['consistency'].items():
        for variant, e in entries.items():
            d = e['minus_input']['micro']
            verdict = (f"{e['decision']}: {'passes' if e['passes'] else 'no'}"
                       if 'decision' in e else 'reported')
            print(f"  week 13, the consistency rule, {corpus}, {variant} ({e['n_units']} units): "
                  f"{e['output']['micro']['mean']:.2f} against {e['input']['micro']['mean']:.2f}, "
                  f"{d['mean_diff']:+.2f} {d['ci95']} ({verdict}); {e['n_added_per_seed']:.0f} "
                  f"entities added per seed")
    for method, e in result['week13'].get('folds', {}).items():
        d = e['minus_crf']['micro']
        print(f"  week 13, SOFC-Exp's article folds, {method} ({e['n_units']} units): "
              f"{e['system']['micro']['mean']:.2f} against the CRF head "
              f"{e['crf']['micro']['mean']:.2f}, {d['mean_diff']:+.2f} {d['ci95']} "
              f"({e['decision']}: {'passes' if e['passes'] else 'no'})")
        if 'minus_crf_mr' in e:
            d = e['minus_crf_mr']['micro']
            print(f"    against crf_mr: {d['mean_diff']:+.2f} {d['ci95']} (reported)")
        if 'slot' in e:
            print(f"    slot level: {e['slot']['micro']['mean']:.2f} {e['slot']['micro']['ci95']} "
                  f"(reported)")
    if result['week13']['week12_winner']:
        print(f"  week 13, the week-12 screen's winner: {result['week13']['week12_winner']}")
    for method, e in result['week13']['pooled'].items():
        d = e['minus_crf']
        parts = ', '.join(f"{c} {v['n_units']} units {v['minus_crf']['mean_diff']:+.2f}"
                          f"{' (loses)' if v['loses'] else ''}" for c, v in e['corpora'].items())
        print(f"  week 13, pooled over {e['n_units']} held-out units, {method} minus the CRF "
              f"head: {d['mean_diff']:+.2f} {d['ci95']} ({e['status']}; "
              f"{'passes' if e['passes'] else 'no'}; {parts})")
        for key, w in e.items():
            if key.startswith('with_'):
                d = w['minus_crf']
                print(f"    with {key[5:]} ({w[key[5:]]['variant']}, {w[key[5:]]['n_units']} "
                      f"units {w[key[5:]]['minus_crf']['mean_diff']:+.2f}): {w['n_units']} units, "
                      f"{d['mean_diff']:+.2f} {d['ci95']} (reported; "
                      f"{'passes' if w['passes'] else 'no'})")
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
