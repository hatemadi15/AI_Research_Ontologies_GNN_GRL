"""
paper_b_checks.py - post hoc checks of paper B's 111 pre-registered decisions

No training, and nothing here decides anything. Paper B (docs/PAPER_B.md, Section 7.7) counts
111 pre-registered decisions, 24 passing, each taken by its registered rule: the 95% Student-t
interval of the mean paired micro-F1 difference above 0, which is a one-sided t test at 2.5%
(the pooled decisions also need neither corpus's interval below 0). Its first referee asked for
post hoc checks of them. Every item here is post hoc, computed for round 1 of paper B's internal
review (3 November 2026), and labelled so; the decisions stand as taken.

`decisions` lists the 111 with the per-unit (or per-seed) differences each used: Table 2's 70
(paper_tables.COMPARISONS, role 'decision'); Table 6's 20 rows (paper_tables.held_out_entries)
less the 4 that repeat Table 2 rows (week 7's crf_dc and week 12's crf_mrm6, both levels),
which are checked against them; week 2's 6, one run per split (results/ner_comparisons.json);
week 8 part 2's 2 (`paper_diversity`); SOFC-Exp's 12 on its official split, paired by seed
(`published_split`); the training-size decision (`sofc_draws`); and paper A's D9 to D12
(`week17`, ner_recall_matched.json). Each carries its week, corpus, comparison, design, the
family of Section 7.7's by-addition table, the stored mean, interval and verdict. The interval
is recomputed from the per-unit values (decision_tests.t_test) and compared with the stored one
within 0.02; the totals, by design and by family, are checked against Section 7.7 (`SECTION_7_7`),
and a mismatch stops the run with each discrepancy named.

Per decision: the one- and two-sided P of the t test; the difference it detects with 80% power
(ner_robustness.mde); and the 95% interval with units that share training documents correlated
(paper A's model, decision_tests.dependent_t: MaterioMiner's 4 papers at 1/4, MuLMS's 7
subfields at 1/4 where their pools share an article, SOFC-Exp's 5 folds at 1/5, the 11 pooled
units in two blocks uncorrelated across corpora) or, on random sentence splits, the corrected
resampled t test (Nadeau and Bengio 2003: the variance of the mean (1/J + n_test/n_train) S^2,
J = 5, n - 1 degrees of freedom, the sizes from the reference runs). Decisions paired by seed on
SOFC-Exp's one official split, the draws of training articles scored on that split, and week
2's single runs get no correction, and say so. Over the decisions: Benjamini-Hochberg q-values of
the 111 one-sided P values (unrounded), Holm within each family, one-sided at a family-wise 2.5%
(a family is a row of Section 7.7's by-addition table for the 81 decisions with 7 or 10 seeds,
and for the 30 others an addition at its design, as the paragraph after the table groups them),
and the passes expected by chance. Then weeks 3 and 4's four 3-seed passes recomputed from each
machine's runs at the same seeds, and every CPU run against its GPU repeat; unseen- against
seen-name recall gains where ner_mentions.json `methods` stores per-unit values (otherwise it
says so and computes nothing); and per corpus the CRF head's gap between random splits and
held-out documents with a Welch interval. Means recomputed from run files are exact from the
stored two-decimal scores and rounded half up, so that a tie does not round by float error.

Reads the comparisons files of paper_tables.COMPARISONS, results/ner_comparisons.json,
results/gpu/ner_week5_seed_analysis.json, ner_week12_comparisons.json, ner_recall_matched.json,
ner_decision_tests.json and ner_mentions.json, and the run files under results/ and results/gpu/
(scores, seeds and split sizes only); writes results/gpu/paper_b_checks.json.

Usage:
  python src/graph/paper_b_checks.py [--results results] [--out results/gpu/paper_b_checks.json]
"""

import argparse
import json
import math
import os
import statistics
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import decision_tests as dt
import paper_tables as pt
from ner_robustness import mde

WEEK2 = 'ner_comparisons.json'
ANALYSIS = os.path.join('gpu', dt.ANALYSIS)
WEEK12 = os.path.join('gpu', dt.WEEK12)
WEEK18 = os.path.join('gpu', dt.WEEK18)
TESTS = os.path.join('gpu', 'ner_decision_tests.json')
MENTIONS = os.path.join('gpu', dt.MENTIONS)
OUT = os.path.join('gpu', 'paper_b_checks.json')
TOLERANCE = dt.TOLERANCE          # a recomputed bound may differ by the per-unit rounding
ALPHA = 0.025                     # the registered rule is a one-sided test at 2.5%
FDR = (0.05, 0.10)
SPLITS = 5                        # J of the corrected resampled t test: the 5 random splits
OFFSETS = tuple(range(0, 1000, 100))
NOTE = ("Post hoc, for round 1 of paper B's internal review (3 November 2026), at its first "
        "referee's request. Nothing here decides anything: each decision stands as taken by its "
        "registered rule, the 95% Student-t interval of the mean paired difference above 0.")

# the designs, and the seed bases of the 7- or 10-seed decisions
ONE_RUN, THREE, SEVEN_OR_TEN = 'one run per split', '3 seeds', '7 or 10 seeds'
DESIGNS = (ONE_RUN, THREE, SEVEN_OR_TEN)
TEN, SEVEN, BOTH = '10', '7 (300-900)', 'MuLMS 7 (300-900), MaterioMiner 10'
PROTOCOLS = {'random_split_vary': 'random splits', 'leave_one_paper_out': 'unseen papers',
             'leave_one_subfield_out': 'held-out subfields', 'document_folds': 'article folds',
             'pooled': 'MuLMS subfields + MaterioMiner papers',
             'published_split': 'official split, paired by seed',
             'article_draws': '10-article draws, official split'}

# the rows of Section 7.7's by-addition table, which counts the decisions with 7 or 10 seeds
CRF = 'CRF head over plain MatSciBERT'
VOTE = 'seed vote over the single model'
MT = 'multi-task training (two type levels)'
MR = 'mention replacement'
MTMR = 'mention replacement on the multi-task model'
NAMES = "harvested names by mention replacement (week 11; paper A's D1 to D12)"
DC = 'document-level label propagation'
OTHER = ('label semantics, self-training, document context, multi-task self-training, '
         'recall calibration, coarse-to-fine completion')
TOKENS = "SOFC-Exp's test articles cut as its training articles are (a measurement)"
OTHERS = ('label semantics', 'self-training', 'document context', 'multi-task self-training',
          'recall calibration', 'coarse-to-fine completion')
FAMILY = {'CRF head': CRF, 'seed vote': VOTE, 'multi-task training': MT, 'mention replacement': MR,
          'mention replacement on the multi-task model': MTMR, 'harvested names': NAMES,
          'document-level label propagation': DC, 'test tokens': TOKENS,
          **{a: OTHER for a in OTHERS}}
# the addition a decision's variant tests
ADDITION = {
    'crf': 'CRF head', 'crf_ens3': 'seed vote', 'crf_mix3': 'the mixed-backbone vote',
    'crf_dapt': 'domain-adaptive pre-training', 'crf_dapt2': 'in-domain pre-training',
    'crf_mt': 'multi-task training', 'crf_mt_ens3': 'multi-task training',
    'crf_ls': 'label semantics', 'crf_ls_ens3': 'label semantics', 'crf_mt_ls': 'label semantics',
    'crf_st': 'self-training', 'crf_st_ens3': 'self-training', 'crf_st3': 'self-training',
    'crf_st3_ens3': 'self-training',
    'crf_ctx': 'document context', 'crf_ctx_ens3': 'document context',
    'crf_mt_st3': 'multi-task self-training', 'crf_mt_st3_ens3': 'multi-task self-training',
    'crf_eb': 'recall calibration', 'crf_eb_ens3': 'recall calibration',
    'crf_ebv': 'recall calibration',
    'crf_cf': 'coarse-to-fine completion', 'crf_mt_ens3_cf': 'coarse-to-fine completion',
    'crf_dc': DC, 'crf_dcr': DC, 'crf_ens3_dc': DC, 'crf_mt_ens3_dc': DC, 'crf_mr_dc': DC,
    'crf_mr': 'mention replacement', 'crf_mr_ens3': 'mention replacement',
    'crf_mt_mr': 'mention replacement on the multi-task model',
    'crf_mt_mr_ens3': 'mention replacement on the multi-task model',
    'crf_mrs': 'harvested names', 'crf_mrm6': 'harvested names',
    'crf_mrm_topic6': 'harvested names',
    'crf_mt_mrm6': 'harvested names',
    'crf_cdetest': 'test tokens', 'cdetest': 'test tokens',
    'crf_p3n2': 'a third paper at equal size',
}
# paper A's labels of the Table 6 rows (by their decision label) and of Table 2's week-12 rows
TABLE6_LABELS = {'week 12, decision 1': 'D1', 'week 12, decision 2': 'D2', 'week 13, 13.2': 'D5',
                 'week 13, 13.4': 'D6', 'week 14, 14.1': 'D7', 'week 14, 14.2': 'D8'}
TABLE6_SEEDS = {'D1': SEVEN, 'D2': SEVEN, 'D7': BOTH, 'D8': SEVEN}
TABLE2_LABELS = {(12, 'fine', 'crf_mrm6'): 'D3', (12, 'coarse', 'crf_mrm6'): 'D4'}
CORPORA = {'mulms': ('MuLMS', None, 'leave_one_subfield_out'),
           'fine': ('MaterioMiner', 'fine', 'leave_one_paper_out'),
           'coarse': ('MaterioMiner', 'coarse', 'leave_one_paper_out'),
           'sofc_folds': ('SOFC-Exp', None, 'document_folds')}
WEEK2_VARIANTS = ('crf_mt', 'crf_dapt', 'crf_ens3')
# SOFC-Exp's 12 decisions on its official split, paired by seed (ROADMAP weeks 6, 7 and 9):
# (week, the protocol's label, level, path in published_split.<level>, variant, reference)
SOFC_OFFICIAL = (
    (6, 'S1', 'sofc', ('crf_minus_baseline',), 'crf', 'baseline'),
    (6, 'S2, document context', 'sofc', ('methods', 'crf_ctx', 'minus_crf'), 'crf_ctx', 'crf'),
    (7, 'part 1, the rule', 'sofc', ('methods', 'crf_dc', 'minus_crf'), 'crf_dc', 'crf'),
    (7, 'part 1 (amendment), the rule with retyping', 'sofc', ('methods', 'crf_dcr', 'minus_crf'),
     'crf_dcr', 'crf'),
    (7, 'part 2a, decision 3', 'sofc', ('methods', 'crf_ebv', 'minus_crf'), 'crf_ebv', 'crf'),
    (7, 'part 2b, S2', 'sofc', ('methods', 'crf_mr', 'minus_crf'), 'crf_mr', 'crf'),
    (7, 'part 2d, decision 1', 'sofc_slot', ('methods', 'crf_mt', 'minus_crf'), 'crf_mt', 'crf'),
    (7, 'part 2d, decision 2', 'sofc', ('methods', 'crf_mt', 'minus_crf'), 'crf_mt', 'crf'),
    (7, 'part 2e, decision 1', 'sofc', ('methods', 'crf_cdetest', 'minus_crf'), 'crf_cdetest',
     'crf'),
    (7, 'part 2e, decision 2', 'sofc', ('baseline_retokenized_test', 'minus_baseline'), 'cdetest',
     'baseline'),
    (9, 'part 1, S2, slot level', 'sofc_slot', ('methods', 'crf_mt_mr', 'minus_crf'), 'crf_mt_mr',
     'crf'),
    (9, 'part 1, S2, entity level', 'sofc', ('methods', 'crf_mt_mr', 'minus_crf'), 'crf_mt_mr',
     'crf'),
)
SOFC_LEVELS = {'sofc': 'entity', 'sofc_slot': 'slot'}

# What Section 7.7 states (with the week-2 protocol's two levels per variant): the totals, by
# design, the seed bases, the by-addition table (decisions and passes per corpus, with the
# breakdowns it gives) and the 3-seed and one-run families of the paragraph below it.
SECTION_7_7 = {
    'decisions': 111, 'passes': 24,
    'weeks': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18],
    'by_design': {ONE_RUN: [6, 0], THREE: [24, 6], SEVEN_OR_TEN: [81, 18]},
    'three_seeds': {'weeks 3 and 4': [22, 4], 'week 8': [2, 2]},
    'seed_bases': {TEN: 75, SEVEN: 5, BOTH: 1},
    'random_splits_7_or_10': 8,
    'by_addition_rows': [81, 18],
    'by_addition': {
        CRF: {'decisions': {'SOFC-Exp': 1}, 'passes': {'SOFC-Exp': 1},
              'sofc_protocols': {'published_split': 1}},
        VOTE: {'decisions': {}, 'passes': {}},
        MT: {'decisions': {'MaterioMiner': 6, 'SOFC-Exp': 2}, 'passes': {'MaterioMiner': 3},
             'random_splits': 4},
        MR: {'decisions': {'MaterioMiner': 4, 'MuLMS': 2, 'SOFC-Exp': 3},
             'passes': {'MaterioMiner': 2, 'MuLMS': 1},
             'sofc_protocols': {'article_draws': 1, 'document_folds': 1, 'published_split': 1}},
        MTMR: {'decisions': {'MaterioMiner': 4, 'SOFC-Exp': 3}, 'passes': {'MaterioMiner': 3},
               'sofc_protocols': {'document_folds': 1, 'published_split': 2}},
        NAMES: {'decisions': {'MaterioMiner': 6, 'MuLMS': 7, 'SOFC-Exp': 1, 'pooled': 2},
                'passing': ['D11', 'D3', 'D4', 'D5']},
        DC: {'decisions': {'MaterioMiner': 6, 'MuLMS': 2, 'SOFC-Exp': 2}, 'n_passes': 2},
        OTHER: {'n_decisions': 28, 'n_passes': 0, 'sofc_exp': 2,
                'additions': {'label semantics': 6, 'self-training': 6, 'document context': 5,
                              'multi-task self-training': 4, 'recall calibration': 5,
                              'coarse-to-fine completion': 2}},
        TOKENS: {'n_decisions': 2, 'n_passes': 2},
    },
    'three_seed_additions': {'label semantics': [8, 2], 'self-training': [6, 1],
                             'multi-task training': [4, 1], 'the mixed-backbone vote': [2, 0],
                             'in-domain pre-training': [2, 0],
                             'a third paper at equal size': [2, 2]},
    'three_seed_label_semantics_on_the_multi_task_model': 4,
    'one_run_additions': {'multi-task training': [2, 0], 'domain-adaptive pre-training': [2, 0],
                          'seed vote': [2, 0]},
}


class CountMismatch(AssertionError):
    """The enumeration does not give Section 7.7's counts; the message names each discrepancy."""


def load(path):
    with open(path) as f:
        return json.load(f)


def plain(comparison):
    """A table's comparison without its backticks."""
    return comparison.replace('`', '')


def family(addition, design):
    """The family of a decision: its row of Section 7.7's by-addition table for 7 or 10 seeds,
    else the addition with the design, as the paragraph after the table groups them."""
    return FAMILY[addition] if design == SEVEN_OR_TEN else f'{addition} ({design})'


def record(week, corpus, level, protocol, comparison, addition, design, seeds, entry, passes,
           source, label=None, machine='GPU', per_unit=None, verdict=None, measure='micro F1',
           **extra):
    """One decision: where it was taken, what it compares, its design and family, its per-unit
    differences (entry['per_run'] unless given) and the stored mean, interval and verdict."""
    values = entry['per_run'] if per_unit is None else per_unit
    where = f"{corpus}{' ' + level if level else ''}, {PROTOCOLS[protocol]}"
    what = comparison if measure == 'micro F1' else f'{comparison}, {measure}'
    return {'id': f"w{week}{' CPU' if machine == 'CPU' else ''} {where}: {what}",
            'label': label, 'week': week, 'machine': machine, 'corpus': corpus, 'level': level,
            'protocol': protocol, 'comparison': comparison, 'measure': measure,
            'addition': addition, 'family': family(addition, design),
            'in_by_addition_table': design == SEVEN_OR_TEN, 'design': design, 'seeds': seeds,
            'source': source, 'n_units': len(values), 'per_unit': [float(v) for v in values],
            'stored': {'mean_diff': entry['mean_diff'], 'ci95': list(entry['ci95'])},
            'passes': bool(passes), 'verdict': verdict or ('passes' if passes else 'no'), **extra}


def table2_decisions(results_dir):
    """Table 2's decisions: paper_tables.comparison_rows of each comparisons file, role
    'decision', with the reference's results files (for the split sizes)."""
    out = []
    for week, machine, seeds, name in pt.COMPARISONS:
        data = pt.load(os.path.join(results_dir, name))
        if data is None:
            continue
        for _, _, _, e in pt.comparison_rows([(week, machine, seeds, data)], ('decision',)):
            design = THREE if seeds == '3' else SEVEN_OR_TEN
            label = TABLE2_LABELS.get((int(week), e['granularity'], e['variant']))
            out.append(record(
                int(week), 'MaterioMiner', e['granularity'], e['protocol'],
                f"{e['variant']} vs {e['reference']}", ADDITION[e['variant']], design,
                TEN if design == SEVEN_OR_TEN else seeds, e['micro'], e['micro']['ci95'][0] > 0,
                f'Table 2, results/{name}', label=label, machine=machine,
                results_dir=os.path.dirname(name), variant_files=list(e['variant_files']),
                reference_files=list(e['reference_files'])))
    return out


def table6_entries(analysis, week12):
    """Table 6's 20 rows as records (paper_tables.held_out_entries)."""
    out = []
    for week, corpus, comparison, decision, _, diff, passes in pt.held_out_entries(analysis,
                                                                                    week12):
        key = next((k for k, v in pt.HELD_OUT_CORPORA.items() if v == corpus), None)
        name, level, protocol = CORPORA[key] if key else ('pooled', None, 'pooled')
        label = next((v for k, v in TABLE6_LABELS.items() if decision.startswith(k)), decision)
        variant = comparison.split('`')[1]
        out.append(record(week, name, level, protocol, plain(comparison), ADDITION[variant],
                          SEVEN_OR_TEN, TABLE6_SEEDS.get(label, TEN), diff, passes,
                          'Table 6, results/gpu/ner_week5_seed_analysis.json'
                          + (', ner_week12_comparisons.json' if 'analogue' in decision else ''),
                          label=label, decision=decision))
    return out


def same_decision(a, b):
    """Whether a Table 6 record repeats a Table 2 record: the same week, corpus, level,
    protocol and comparison."""
    return all(a[k] == b[k] for k in ('week', 'corpus', 'level', 'protocol', 'comparison'))


def week2_decisions(week2):
    """Week 2's 6 decisions: each variant against the CRF head over the 5 random splits, one
    run per split (ROADMAP's week-2 protocol; the held-out papers were reported)."""
    out = []
    for variant in WEEK2_VARIANTS:
        for g in ('fine', 'coarse'):
            e = week2[f'ner_results_{g}_random_split_vary_{variant}']
            assert e['reference'] == 'crf' and e['n_pairs'] == 5, e
            out.append(record(2, 'MaterioMiner', g, 'random_split_vary', f'{variant} vs crf',
                              ADDITION[variant], ONE_RUN, '1', e['micro'],
                              e['micro']['ci95'][0] > 0, f'results/{WEEK2}', machine='CPU'))
    return out


def week8_decisions(analysis):
    """Week 8 part 2's 2 decisions: diversity, three papers at the size of two minus the two
    papers alone, over the 4 held-out papers, 3 seeds (`paper_diversity`)."""
    return [record(8, 'MaterioMiner', g, 'leave_one_paper_out',
                   'crf_p3n2 (three papers at the size of two) vs two papers', 'a third paper at '
                   'equal size', THREE, '3', e['diversity'], e['diversity']['passes'],
                   'results/gpu/ner_week5_seed_analysis.json, paper_diversity',
                   label=f'week 8, part 2, decision {i}')
            for i, (g, e) in enumerate(analysis['paper_diversity'].items(), 1)]


def sofc_official_decisions(analysis):
    """SOFC-Exp's 12 decisions on its official split, paired by the 10 seeds (SOFC_OFFICIAL)."""
    out = []
    for week, label, level, path, variant, reference in SOFC_OFFICIAL:
        e = analysis['published_split'][level]
        for key in path:
            e = e[key]
        out.append(record(week, 'SOFC-Exp', SOFC_LEVELS[level], 'published_split',
                          f'{variant} vs {reference}', ADDITION[variant], SEVEN_OR_TEN, TEN,
                          e['micro'], e['micro']['ci95'][0] > 0,
                          'results/gpu/ner_week5_seed_analysis.json, published_split.'
                          + '.'.join((level,) + path), label=f'week {week}, {label}'))
    return out


def sofc_draws_decision(analysis):
    """Week 9 part 2: mention replacement against the CRF head on 10 of the 26 official
    training articles, paired over the 5 draws (`sofc_draws`)."""
    d = analysis['sofc_draws']
    return [record(9, 'SOFC-Exp', 'entity', 'article_draws', 'crf_mr vs crf', 'mention replacement',
                   SEVEN_OR_TEN, TEN, d['crf_mr_minus_crf']['micro'], d['passes'],
                   'results/gpu/ner_week5_seed_analysis.json, sofc_draws',
                   label='week 9, part 2')]


def names_decisions(analysis, matched):
    """Paper A's D9 and D10 (week 17) and D11 and D12 (week 18, the subfields in MULMS_UNITS
    order)."""
    out = []
    for label, g in (('D9', 'fine'), ('D10', 'coarse')):
        e = analysis['week17'][g]
        out.append(record(17, 'MaterioMiner', g, 'leave_one_paper_out',
                          'crf_mt_mrm6 vs crf_mt_mr6', 'harvested names', SEVEN_OR_TEN, TEN,
                          e['minus_crf_mt_mr6']['micro'], e['passes'],
                          'results/gpu/ner_week5_seed_analysis.json, week17', label=label,
                          decision=e['decision']))
    for label, key, measure in (
            ('D11', '18.1', 'recall on the unseen test names the lexicon holds'),
            ('D12', '18.2', 'micro F1')):
        e = matched['decisions'][key]
        passes = e['verdict'] == 'passes'
        out.append(record(18, 'MuLMS', None, 'leave_one_subfield_out',
                          'crf_mrm_topic6 vs crf_mr6_bias at the recall-matched bias',
                          'harvested names', SEVEN_OR_TEN, SEVEN, e, passes,
                          'results/gpu/ner_recall_matched.json, decisions', label=label,
                          per_unit=[e['per_unit'][u] for u in dt.MULMS_UNITS],
                          verdict=dt.VERDICTS.get(label) if passes else None, measure=measure,
                          decision=key))
    return out


def enumerate_decisions(results_dir):
    """(the 111 decisions, Table 6's 4 rows that repeat Table 2, with their agreement), in the
    order: Table 2, Table 6's others, week 2, week 8, SOFC-Exp's official split, its draws, D9
    to D12."""
    analysis = load(os.path.join(results_dir, ANALYSIS))
    table2 = table2_decisions(results_dir)
    table6 = table6_entries(analysis, load(os.path.join(results_dir, WEEK12)))
    if len(table2) != 70 or len(table6) != 20:
        raise CountMismatch(f'Table 2 has {len(table2)} decisions (70 expected), Table 6 '
                            f'{len(table6)} rows (20 expected)')
    others, repeats = [], []
    for r in table6:
        twin = next((d for d in table2 if same_decision(r, d)), None)
        if twin is None:
            others.append(r)
            continue
        repeats.append({'table6': r['id'], 'table2': twin['id'], 'label': twin['label'],
                        'same_mean': abs(r['stored']['mean_diff'] - twin['stored']['mean_diff'])
                        <= TOLERANCE,
                        'same_interval': all(abs(a - b) <= TOLERANCE for a, b in
                                             zip(r['stored']['ci95'], twin['stored']['ci95'])),
                        'same_per_unit': sorted(r['per_unit']) == sorted(twin['per_unit']),
                        'same_verdict': r['passes'] == twin['passes']})
    decisions = (table2 + others + week2_decisions(load(os.path.join(results_dir, WEEK2)))
                 + week8_decisions(analysis) + sofc_official_decisions(analysis)
                 + sofc_draws_decision(analysis)
                 + names_decisions(analysis, load(os.path.join(results_dir, WEEK18))))
    ids = [d['id'] for d in decisions]
    if len(set(ids)) != len(ids):
        raise CountMismatch(f'duplicate ids: {sorted({i for i in ids if ids.count(i) > 1})}')
    return decisions, repeats


def counts(decisions):
    """The counts of SECTION_7_7, from an enumeration."""
    def tally(rows, key):
        out = {}
        for d in rows:
            out[d[key]] = out.get(d[key], 0) + 1
        return dict(sorted(out.items()))

    def both(rows):
        return [len(rows), sum(d['passes'] for d in rows)]
    tables = [d for d in decisions if d['design'] == SEVEN_OR_TEN]
    three = [d for d in decisions if d['design'] == THREE]
    one = [d for d in decisions if d['design'] == ONE_RUN]
    by_addition = {}
    for row in SECTION_7_7['by_addition']:
        rows = [d for d in tables if d['family'] == row]
        passing = [d for d in rows if d['passes']]
        sofc = [d for d in rows if d['corpus'] == 'SOFC-Exp']
        by_addition[row] = {'decisions': tally(rows, 'corpus'), 'passes': tally(passing, 'corpus'),
                            'n_decisions': len(rows), 'n_passes': len(passing),
                            'sofc_exp': len(sofc), 'sofc_protocols': tally(sofc, 'protocol'),
                            'random_splits': sum(d['protocol'] == 'random_split_vary'
                                                 for d in rows),
                            'additions': tally(rows, 'addition'),
                            'passing': sorted(d['label'] or d['id'] for d in passing)}
    return {'decisions': len(decisions), 'passes': sum(d['passes'] for d in decisions),
            'weeks': sorted({d['week'] for d in decisions}),
            'by_design': {k: both([d for d in decisions if d['design'] == k]) for k in DESIGNS},
            'three_seeds': {'weeks 3 and 4': both([d for d in three if d['week'] in (3, 4)]),
                            'week 8': both([d for d in three if d['week'] == 8])},
            'seed_bases': tally(tables, 'seeds'),
            'random_splits_7_or_10': sum(d['protocol'] == 'random_split_vary' for d in tables),
            'by_addition_rows': both([d for d in tables
                                      if d['family'] in SECTION_7_7['by_addition']]),
            'by_addition': by_addition,
            'three_seed_additions': {a: both([d for d in three if d['addition'] == a])
                                     for a in sorted({d['addition'] for d in three})},
            'three_seed_label_semantics_on_the_multi_task_model': sum(
                d['comparison'].startswith('crf_mt_ls ') for d in three),
            'one_run_additions': {a: both([d for d in one if d['addition'] == a])
                                  for a in sorted({d['addition'] for d in one})}}


# the counts compared as wholes (a tally must hold exactly the corpora, protocols, additions or
# seed bases stated); the other dicts are compared key by key, on the keys Section 7.7 states
WHOLE = {'decisions', 'passes', 'passing', 'additions', 'sofc_protocols', 'seed_bases',
         'three_seed_additions', 'one_run_additions'}


def discrepancies(found, expected=None, path=''):
    """Each place where `found` (counts) differs from what Section 7.7 states."""
    expected = SECTION_7_7 if expected is None else expected
    if isinstance(expected, dict) and path.rpartition('/')[2] not in WHOLE:
        found = found if isinstance(found, dict) else {}
        out = []
        for key, want in expected.items():
            out += discrepancies(found.get(key), want, f'{path}/{key}' if path else key)
        return out
    if found == expected:
        return []
    return [f'{path}: enumerated {found}, Section 7.7 states {expected}']


def check_counts(decisions):
    """counts(decisions), or CountMismatch naming every count that differs from Section 7.7."""
    found = counts(decisions)
    wrong = discrepancies(found)
    if wrong:
        raise CountMismatch('the enumeration does not match Section 7.7:\n  ' + '\n  '.join(wrong))
    return found


def t_raw(values):
    """The unrounded one-sample t test of `values` against 0: mean, standard error, t, df,
    the 95% half-width and the one- and two-sided P."""
    from scipy import stats
    n = len(values)
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    t = mean / se
    return {'mean': mean, 'se': se, 't': t, 'df': n - 1,
            'half': float(stats.t.ppf(0.975, n - 1)) * se,
            'p_one_sided': float(stats.t.sf(t, n - 1)),
            'p_two_sided': float(2 * stats.t.sf(abs(t), n - 1))}


def corrected_resampled_t(values, ratio, splits=SPLITS):
    """The corrected resampled t test (Nadeau and Bengio 2003) of the mean of `values` over
    `splits` random splits: the variance of the mean (1/J + n_test/n_train) S^2, n - 1 degrees of
    freedom; with the ratio of its standard error to the plain one."""
    from scipy import stats
    n = len(values)
    mean, s2 = statistics.fmean(values), statistics.variance(values)
    se = math.sqrt((1 / splits + ratio) * s2)
    t = mean / se
    half = float(stats.t.ppf(0.975, n - 1)) * se
    return {'mean_diff': round(mean, 2), 'ci95': [round(mean - half, 2), round(mean + half, 2)],
            'p_one_sided': round(float(stats.t.sf(t, n - 1)), 4),
            'p_two_sided': round(float(2 * stats.t.sf(abs(t), n - 1)), 4),
            'se_ratio': round(se / math.sqrt(s2 / n), 3), 'df': n - 1}


def split_sizes(directory, names):
    """[(n_train, n_test)] of every run of the results files `names` in `directory`; a vote's
    runs carry no sizes, so its members' runs (config.members), on the same splits, are read."""
    sizes = []
    for name in names:
        data = load(os.path.join(directory, name))
        members = (data.get('config') or {}).get('members')
        if members and not all('n_train' in r for r in data['runs']):
            sizes += split_sizes(directory, members)
        else:
            sizes += [(r['n_train'], r['n_test']) for r in data['runs']]
    return sizes


# the dependence model of each design ('random_split_vary' only with seeds; week 2 has none)
LEAVE_OUT_UNITS = {'leave_one_paper_out': 4, 'leave_one_subfield_out': 7, 'document_folds': 5,
                   'pooled': 11}
MODELS = {
    'leave_one_paper_out': "MaterioMiner's 4 held-out papers, every pair correlated at 1/4 "
                           '(decision_tests.dependent_t)',
    'leave_one_subfield_out': "MuLMS's 7 held-out subfields, the pairs whose training pools "
                              'share an article (MULMS_POOLS) at 1/4 (decision_tests.dependent_t)',
    'document_folds': "SOFC-Exp's 5 article folds, every pair at 1/5 (decision_tests.dependent_t)",
    'pooled': "the 11 pooled units in two blocks uncorrelated across corpora: MuLMS's 7 "
              'subfields (pools sharing an article at 1/4) and MaterioMiner\'s 4 papers (at 1/4) '
              '(decision_tests.dependent_t)',
    'random_split_vary': 'the corrected resampled t test (Nadeau and Bengio 2003): the variance '
                         'of the mean (1/J + n_test/n_train) S^2, J = 5, n - 1 degrees of freedom',
}
UNCORRECTED = {
    'published_split': "none: paired by seed on SOFC-Exp's one official split",
    'article_draws': ('none: the 5 draws of 10 of the 26 official training articles are scored '
                      'on the one official test split and paired by draw, as seeds on one split '
                      'are; their training articles overlap, which no model here describes'),
    ONE_RUN: "none: week 2's single runs are left uncorrected",
}


def dependence(d, root):
    """The dependence-adjusted 95% interval of decision `d` (post hoc; MODELS), or why it has
    none (UNCORRECTED); `root` is the results directory, for the split sizes."""
    protocol, values = d['protocol'], d['per_unit']
    if d['design'] == ONE_RUN or protocol in UNCORRECTED:
        return {'model': UNCORRECTED[ONE_RUN if d['design'] == ONE_RUN else protocol],
                'corrected': False}
    if protocol in LEAVE_OUT_UNITS:
        if len(values) != LEAVE_OUT_UNITS[protocol]:
            raise ValueError(f"{d['id']}: {len(values)} units for {protocol}")
        e = dt.dependent_t(values, dt.layout(len(values)))
    else:
        directory = os.path.join(root, d['results_dir'])
        sizes = split_sizes(directory, d['reference_files'])
        ratio = statistics.fmean(n_test / n_train for n_train, n_test in sizes)
        e = dict(corrected_resampled_t(values, ratio),
                 n_train=sorted({s[0] for s in sizes}), n_test=sorted({s[1] for s in sizes}),
                 ratio=round(ratio, 4), J=SPLITS)
    return dict(e, model=MODELS[protocol], corrected=True, above_0=e['ci95'][0] > 0)


def dependence_summary(decisions):
    """Post hoc: per dependence model, its decisions and passes, and which passes' adjusted
    intervals stay above 0 (or are left uncorrected); the n_test/n_train ratios used on the
    random splits."""
    models, tally = {}, {'above_0': 0, 'covers_0': 0, 'uncorrected': 0}
    for d in decisions:
        dep = d['dependence']
        e = models.setdefault(dep['model'], {'decisions': 0, 'passes': 0})
        e['decisions'] += 1
        if d['passes']:
            e['passes'] += 1
            kind = ('uncorrected' if not dep['corrected'] else
                    'above_0' if dep['above_0'] else 'covers_0')
            tally[kind] += 1
            e.setdefault(f'passes_{kind}', []).append(name(d))
    return {'note': ('post hoc: per dependence model, the passes whose 95% interval stays above 0 '
                     'with the units correlated, those whose interval then covers 0, and those '
                     'left uncorrected'),
            'passes': tally, 'models': models,
            'random_split_ratios': sorted({d['dependence']['ratio'] for d in decisions
                                           if 'ratio' in d['dependence']})}


def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg adjusted q-values (step-up over all m), in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    q, running = [0.0] * m, 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * m / rank)
        q[i] = running
    return q


def holm(pvalues):
    """Holm's step-down adjusted P values, in the input order: H_i is rejected at family-wise
    level a exactly when its adjusted P is at most a."""
    k = len(pvalues)
    order = sorted(range(k), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * k, 0.0
    for j, i in enumerate(order):
        running = max(running, min(1.0, (k - j) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def per_decision(decisions, root):
    """Per decision (post hoc), added to its record: the t test of its per-unit values
    (decision_tests.t_test), whether that interval reproduces the stored one and its one-sided
    P agrees with the verdict, the unrounded one-sided P (4 significant digits), the 80%-power
    detectable difference and the dependence-adjusted interval. Returns the unrounded one-sided
    P values, for multiplicity."""
    raw_p = []
    for d in decisions:
        test, raw = dt.t_test(d['per_unit']), t_raw(d['per_unit'])
        raw_p.append(raw['p_one_sided'])
        d['t_test'] = test
        d['interval_matches'] = all(abs(a - b) <= TOLERANCE
                                    for a, b in zip(test['ci95'], d['stored']['ci95']))
        d['p_agrees'] = (test['p_one_sided'] < ALPHA) == (d['stored']['ci95'][0] > 0)
        d['p_one_sided_unrounded'] = float(f"{raw['p_one_sided']:.4g}")
        d['detectable_80'] = {'half_width': round(raw['half'], 2), 'df': raw['df'],
                              'difference': mde(raw['half'], raw['df'])}
        d['dependence'] = dependence(d, root)
    return raw_p


def name(d):
    """A decision's label and id for the lists of the multiplicity section."""
    return f"{d['label']}: {d['id']}" if d['label'] else d['id']


def multiplicity(decisions, raw_p):
    """Post hoc: Benjamini-Hochberg over the decisions (one-sided P, `raw_p` unrounded), Holm
    within each family and the passes expected by chance; each record gets its q-value
    (`bh_q`), its Holm-adjusted P (`holm_p`) and whether it survives Holm (`holm_survives`)."""
    q = benjamini_hochberg(raw_p)
    families = {}
    for i, d in enumerate(decisions):
        d['bh_q'] = round(q[i], 4)
        families.setdefault(d['family'], []).append(i)
    for members in families.values():
        for i, adjusted in zip(members, holm([raw_p[i] for i in members])):
            decisions[i]['holm_p'] = round(adjusted, 4)
            decisions[i]['holm_survives'] = adjusted <= ALPHA
    holm_out = {f: {'decisions': len(rows), 'passes': sum(decisions[i]['passes'] for i in rows),
                    'survivors': [name(decisions[i]) for i in rows
                                  if decisions[i]['holm_survives']]}
                for f, rows in families.items()}
    passes = sum(d['passes'] for d in decisions)
    return {
        'note': ('post hoc: Benjamini-Hochberg q-values of the one-sided t-test P values (mean > '
                 '0, unrounded) over all the decisions; Holm within each family (the rows of '
                 "Section 7.7's by-addition table for 7 or 10 seeds, the 3-seed and one-run "
                 'groups of the paragraph after it), one-sided, family-wise 2.5%'),
        'benjamini_hochberg': {
            'm': len(decisions),
            **{f'fdr_{level:.2f}': {'n': sum(x <= level for x in q),
                                    'decisions': [name(d) for d, x in zip(decisions, q)
                                                  if x <= level]}
               for level in FDR}},
        'holm': {'alpha': ALPHA, 'families': holm_out,
                 'survivors': [name(d) for d in decisions if d['holm_survives']]},
        'chance': {'alpha': ALPHA, 'decisions': len(decisions),
                   'expected_passes': round(len(decisions) * ALPHA, 3), 'passes': passes,
                   'share_of_passes': round(len(decisions) * ALPHA / passes, 3)}}


def exact(x):
    """A stored score (two decimals) as an exact fraction."""
    return Fraction(Decimal(repr(x)))


def half_up(q, places=2):
    """An exact number rounded to `places` decimals, a tie away from 0."""
    d = Decimal(q.numerator) / Decimal(q.denominator)
    return float(d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def family_means(directory, names):
    """({run_id: the exact mean F1 of the files' runs with that run_id}, {run_id: seeds}) of
    the results files `names` in `directory`, run_ids in all of them; None if one is missing."""
    paths = [os.path.join(directory, n) for n in names]
    if not all(os.path.exists(p) for p in paths):
        return None
    runs = [{r['run_id']: r for r in load(p)['runs']} for p in paths]
    ids = [i for i in runs[0] if all(i in r for r in runs[1:])]
    return ({i: sum((exact(r[i]['f1']) for r in runs), Fraction(0)) / len(runs) for i in ids},
            {i: [r[i].get('seed') for r in runs] for i in ids})


def seed_paired(root, machine, variant_files, reference_files):
    """A comparison of seed means recomputed from one machine's run files (MACHINES), paired
    by run_id: per unit the variant's mean minus the reference's, the mean (exact, rounded half
    up) with its 95% Student-t interval and one-sided P; None if a file is missing there."""
    from scipy import stats
    directory = os.path.join(root, MACHINES[machine])
    v, r = family_means(directory, variant_files), family_means(directory, reference_files)
    if v is None or r is None:
        return None
    ids = [i for i in v[0] if i in r[0]]
    diffs = [v[0][i] - r[0][i] for i in ids]
    mean = sum(diffs, Fraction(0)) / len(diffs)
    values = [float(x) for x in diffs]
    se = statistics.stdev(values) / math.sqrt(len(values))
    half = float(stats.t.ppf(0.975, len(values) - 1)) * se
    lo, hi = round(float(mean) - half, 2), round(float(mean) + half, 2)
    return {'directory': os.path.join('results', MACHINES[machine]).rstrip(os.sep),
            'variant_files': list(variant_files), 'reference_files': list(reference_files),
            'units': [pt.short_unit(i) for i in ids], 'seeds': {
                'variant': sorted({s for i in ids for s in v[1][i] if s is not None}),
                'reference': sorted({s for i in ids for s in r[1][i] if s is not None})},
            'per_unit': [half_up(x) for x in diffs], 'mean_diff': half_up(mean), 'ci95': [lo, hi],
            'p_one_sided': round(float(stats.t.sf(float(mean) / se, len(values) - 1)), 4),
            'above_0': lo > 0}


# the results directory of each machine under results/ (the same file names on both)
MACHINES = {'CPU': '', 'GPU': 'gpu'}
# the GPU pass whose design never ran on the CPU, and the nearest design that did
NEAREST = {'crf_st3': ('crf_st', "week 3's self-training (one teacher, 2,000 corpus sentences), "
                                 'the only self-training run on the CPU')}


def hardware(decisions, root):
    """Post hoc: weeks 3 and 4's 3-seed passes recomputed from each machine's runs at the same
    seeds 0, 100 and 200 (the same file names under results/ and results/gpu/). Where the
    passing design never ran on the other machine, that side is the nearest design that did
    (NEAREST), which `nearest` also gives on both machines."""
    out = []
    for d in decisions:
        if d['design'] != THREE or d['week'] not in (3, 4) or not d['passes']:
            continue
        variant = d['comparison'].split(' vs ')[0]
        sides = {m: seed_paired(root, m, d['variant_files'], d['reference_files'])
                 for m in MACHINES}
        item = {'decision': d['id'], 'decided_on': d['machine'], 'stored': d['stored'],
                'cpu': sides['CPU'], 'gpu': sides['GPU']}
        if variant in NEAREST and None in sides.values():
            near, why = NEAREST[variant]
            files = [f.replace(f'_{variant}_', f'_{near}_') for f in d['variant_files']]
            item['nearest'] = {'variant': near, 'why': why,
                               **{m.lower(): seed_paired(root, m, files, d['reference_files'])
                                  for m in MACHINES}}
            missing = 'cpu' if sides['CPU'] is None else 'gpu'
            item[missing] = dict(item['nearest'][missing], variant=near, why=why)
        out.append(item)
    return out


def is_vote(data):
    """Whether a results file holds votes: ensembles of other runs (config.members)."""
    variant = data.get('variant') or ''
    return bool((data.get('config') or {}).get('members')) or 'ens3' in variant or 'mix3' in variant


def matched_runs(root):
    """Post hoc: every run present on both machines, matched by (file, run_id, seed) over the
    results files of the same name under results/ and results/gpu/, votes and files whose runs
    lack a seed left out; the count and the median, mean and maximum of |F1_CPU - F1_GPU|."""
    gpu = os.path.join(root, 'gpu')
    names = sorted(set(os.listdir(root)) & set(os.listdir(gpu)))
    diffs, files, skipped = [], [], {'votes': 0, 'no_seed': 0, 'no_runs': 0}
    for n in names:
        if not n.endswith('.json'):
            continue
        cpu_data, gpu_data = load(os.path.join(root, n)), load(os.path.join(gpu, n))
        if not all(isinstance(x, dict) and isinstance(x.get('runs'), list)
                   for x in (cpu_data, gpu_data)):
            skipped['no_runs'] += 1
            continue
        if is_vote(cpu_data) or is_vote(gpu_data):
            skipped['votes'] += 1
            continue
        if not all(r.get('seed') is not None for x in (cpu_data, gpu_data) for r in x['runs']):
            skipped['no_seed'] += 1
            continue
        on_gpu = {(r['run_id'], r['seed']): r['f1'] for r in gpu_data['runs']}
        matched = [(abs(exact(r['f1']) - exact(on_gpu[(r['run_id'], r['seed'])])), n,
                    r['run_id'], r['seed'])
                   for r in cpu_data['runs'] if (r['run_id'], r['seed']) in on_gpu]
        if matched:
            files.append(n)
            diffs += matched
    if not diffs:
        return {'files': 0, 'runs': 0, 'skipped_files': skipped}
    values = sorted(x[0] for x in diffs)
    largest = max(diffs, key=lambda x: x[0])
    return {'files': len(files), 'runs': len(values), 'skipped_files': skipped,
            'identical': sum(x == 0 for x in values),
            'abs_diff_f1': {'median': half_up(statistics.median(values)),
                            'mean': half_up(sum(values, Fraction(0)) / len(values)),
                            'max': half_up(values[-1])},
            'max_at': {'file': largest[1], 'run_id': pt.short_unit(largest[2]),
                       'seed': largest[3]}}


def per_unit_values(entry):
    """An entry's per-unit values as {unit: value} (a list keyed by position), or None."""
    values = (entry or {}).get('per_run', (entry or {}).get('per_unit'))
    if isinstance(values, dict):
        return {str(k): float(v) for k, v in values.items()}
    if isinstance(values, list) and values:
        return {str(i): float(v) for i, v in enumerate(values)}
    return None


def unseen_minus_seen(mentions):
    """Post hoc: per method of ner_mentions.json `methods`, the paired difference over units
    of its recall gain on unseen test names minus its recall gain on seen ones (mean, 95%
    interval, two-sided P), where both gains store per-unit values; otherwise only that the
    entries hold summaries, and nothing is computed."""
    methods = mentions.get('methods') or {}
    computed, summaries, fields = {}, 0, set()
    for setting, systems in methods.items():
        for system, e in systems.items():
            fields |= {k for part in ('unseen', 'seen') for k in ((e.get(part) or {}).get('recall')
                                                                   or {})}
            unseen = per_unit_values((e.get('unseen') or {}).get('recall'))
            seen = per_unit_values((e.get('seen') or {}).get('recall'))
            units = [u for u in (unseen or {}) if u in (seen or {})]
            if len(units) < 2:
                summaries += 1
                continue
            test = dt.t_test([unseen[u] - seen[u] for u in units])
            computed[f'{setting}/{system}'] = {
                'reference': e.get('reference'), 'units': units, 'mean_diff': test['mean_diff'],
                'ci95': test['ci95'], 'p_two_sided': test['p_two_sided'],
                'unseen_gain': (e['unseen'].get('recall') or {}).get('mean_diff'),
                'seen_gain': (e['seen'].get('recall') or {}).get('mean_diff')}
    out = {'entries': sum(len(s) for s in methods.values()), 'with_per_unit_values': len(computed),
           'summaries_only': summaries, 'recall_fields': sorted(fields)}
    if not computed:
        out['note'] = ('post hoc, not computed: every entry of ner_mentions.json `methods` stores '
                       'only summaries of its recall gains (the two systems\' pooled recall, the '
                       'mean paired gain and its 95% interval, the number of pairs), no per-unit '
                       'values, so the paired difference of the unseen and seen gains over the '
                       'units cannot be formed from it')
    else:
        out['note'] = ('post hoc: per method, the unseen-name recall gain minus the seen-name '
                       'gain, paired over the units, with the 95% Student-t interval')
        out['methods'] = computed
    return out


def welch(a, b):
    """mean(a) - mean(b) with its 95% Welch interval (Welch-Satterthwaite degrees of freedom),
    the t statistic and the two-sided P; given exact values (Fractions) the difference is exact
    and rounded half up, the interval is computed in floating point either way."""
    from scipy import stats
    fa, fb = [float(x) for x in a], [float(x) for x in b]
    exact_gap = all(isinstance(x, Fraction) for x in list(a) + list(b))
    gap = (sum(a) / len(a) - sum(b) / len(b)) if exact_gap else (statistics.fmean(fa)
                                                                  - statistics.fmean(fb))
    va, vb = statistics.variance(fa) / len(fa), statistics.variance(fb) / len(fb)
    se, g = math.sqrt(va + vb), float(gap)
    df = (va + vb) ** 2 / (va ** 2 / (len(fa) - 1) + vb ** 2 / (len(fb) - 1))
    half = float(stats.t.ppf(0.975, df)) * se
    return {'gap': half_up(gap) if exact_gap else round(gap, 2),
            'ci95': [round(g - half, 2), round(g + half, 2)], 'df': round(df, 1),
            't': round(g / se, 2), 'p_two_sided': float(f'{2 * stats.t.sf(abs(g / se), df):.4g}')}


def crf_seed_means(directory, stem, offsets=OFFSETS):
    """{run_id: the CRF head's exact mean micro F1 over the seed offsets} of
    ner_results_<stem>_crf_m<N>.json."""
    names = [f'ner_results_{stem}_crf_m{off}.json' for off in offsets]
    found = family_means(directory, names)
    if found is None:
        raise FileNotFoundError(f'{directory}: not every ner_results_{stem}_crf_m<N>.json')
    return found[0]


def gaps(root, analysis):
    """Post hoc: per corpus, the CRF head's micro F1 on random splits minus on held-out
    documents, each unit the mean of its 10 seeds, with the Welch 95% interval and two-sided P:
    MaterioMiner's 5 random splits against its 4 papers per level (from the run files),
    MuLMS's pool splits against its 7 subfields and SOFC-Exp's random splits against its 5
    folds (the analysis file, whose stored point gaps they reproduce)."""
    gpu = os.path.join(root, 'gpu')
    parts = {}
    for g in ('fine', 'coarse'):
        parts[f'MaterioMiner {g}'] = (crf_seed_means(gpu, f'{g}_random_split_vary'),
                                      crf_seed_means(gpu, f'{g}_leave_one_paper_out'), None,
                                      f'results/gpu/ner_results_{g}_{{random_split_vary,'
                                      'leave_one_paper_out}_crf_m0..m900.json')
    mulms, sofc = analysis['mulms'], analysis['sofc_splits']
    parts['MuLMS'] = (mulms['random_split_pool']['micro']['per_run'],
                      mulms['leave_one_subfield_out']['crf']['micro']['per_run'],
                      mulms['gap']['micro'],
                      'ner_week5_seed_analysis.json, mulms.random_split_pool and '
                      'mulms.leave_one_subfield_out.crf')
    parts['SOFC-Exp'] = (sofc['random_split_vary']['micro']['per_run'],
                         sofc['document_folds']['micro']['per_run'], sofc['gap']['micro'],
                         'ner_week5_seed_analysis.json, sofc_splits')
    out = {'note': ('post hoc: the CRF head\'s micro F1 on random splits minus on held-out '
                    'documents, each unit the mean of its 10 seeds, a difference of two '
                    'independent means with its 95% Welch interval and two-sided P (the means '
                    'exact from the two-decimal scores, rounded half up)')}

    def side(values):
        values = [v if isinstance(v, Fraction) else exact(v) for v in values]
        return values, {'n': len(values), 'mean': half_up(sum(values) / len(values)),
                        'per_unit': [half_up(v) for v in values]}
    for key, (random, held, stored, source) in parts.items():
        a, random_part = side(random.values() if isinstance(random, dict) else random)
        b, held_part = side(held.values() if isinstance(held, dict) else held)
        w = welch(a, b)
        out[key] = dict(w, random_splits=random_part, held_out=held_part, source=source)
        if stored is not None:
            out[key]['stored_gap'] = stored
            out[key]['reproduces'] = w['gap'] == stored
    return out


def dependence_matches(decisions, stored_tests):
    """Whether this module's dependence-adjusted values of D5 and D11 equal those of
    results/gpu/ner_decision_tests.json (post_hoc_dependence)."""
    stored = stored_tests['post_hoc_dependence']['decisions']
    out = {}
    for d in decisions:
        if d['label'] in ('D5', 'D11'):
            mine = {k: d['dependence'][k] for k in stored[d['label']]}
            out[d['label']] = {'equal': mine == stored[d['label']], 'here': mine,
                               'ner_decision_tests': stored[d['label']]}
    return out


def checks(root):
    """Everything this module writes, from the results directory `root`."""
    decisions, repeats = enumerate_decisions(root)
    found = check_counts(decisions)
    raw_p = per_decision(decisions, root)
    adjusted = multiplicity(decisions, raw_p)        # adds bh_q, holm_p, holm_survives
    mismatched = [{'id': d['id'], 'label': d['label'], 'stored': d['stored']['ci95'],
                   'recomputed': d['t_test']['ci95']}
                  for d in decisions if not d['interval_matches']]
    return {
        'note': NOTE,
        'decisions_note': ("post hoc: the 111 pre-registered decisions of paper B's Section 7.7, "
                           'each with its per-unit differences, its stored mean, interval and '
                           'verdict, the t test of its per-unit values (decision_tests.t_test), '
                           'the difference it detects with 80% power at the two-sided 5% level '
                           '(ner_robustness.mde), its dependence-adjusted 95% interval, its '
                           'Benjamini-Hochberg q-value over all 111 and its Holm-adjusted P '
                           'within its family'),
        'decisions': decisions,
        'counts': found,
        'table6_repeats': repeats,
        'interval_check': {'tolerance': TOLERANCE, 'matched': len(decisions) - len(mismatched),
                           'mismatched': mismatched},
        'dependence_summary': dependence_summary(decisions),
        'dependence_check': dependence_matches(decisions, load(os.path.join(root, TESTS))),
        'multiplicity': adjusted,
        'hardware': {'note': ("post hoc: weeks 3 and 4's four 3-seed passes recomputed from the "
                              "other machine's runs at the same seeds 0, 100 and 200 (each unit "
                              'the difference of the two seed means, the mean exact from the '
                              'stored two-decimal F1 values, rounded half up), and every run '
                              'present on both machines matched by file, run_id and seed'),
                     'passes': hardware(decisions, root), 'runs': matched_runs(root)},
        'unseen_minus_seen': unseen_minus_seen(load(os.path.join(root, MENTIONS))),
        'gaps': gaps(root, load(os.path.join(root, ANALYSIS))),
    }


def signed(x):
    return f'{x:+.2f}'


def ci(pair):
    return f'[{pair[0]:+.2f}, {pair[1]:+.2f}]'


def main(argv=None):
    parser = argparse.ArgumentParser(description="Post hoc checks of paper B's 111 decisions")
    parser.add_argument('--results', default=str(pt.ROOT / 'results'))
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    out = checks(args.results)
    path = args.out or os.path.join(args.results, OUT)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    c = out['counts']
    print(f"{c['decisions']} decisions, {c['passes']} passing; by design "
          + ', '.join(f'{k} {n}/{p}' for k, (n, p) in c['by_design'].items())
          + ' (decisions/passes; Section 7.7 matches)')
    for m in out['interval_check']['mismatched']:
        print(f"interval not reproduced: {m['id']}: stored {m['stored']}, recomputed "
              f"{m['recomputed']}")
    print('passes: one-sided P, BH q, detectable difference, dependence-adjusted 95% interval')
    for d in out['decisions']:
        if d['passes']:
            dep = d['dependence']
            adj = (f"{ci(dep['ci95'])}{'' if dep['above_0'] else ' covers 0'}" if dep['corrected']
                   else 'none')
            print(f"  {name(d)}: {signed(d['stored']['mean_diff'])} "
                  f"{ci(d['stored']['ci95'])}, P {d['t_test']['p_one_sided']:.4f}, "
                  f"q {d['bh_q']:.4f}, "
                  f"MDE {d['detectable_80']['difference']:.2f}, adjusted {adj}")
    mult = out['multiplicity']
    for level in FDR:
        e = mult['benjamini_hochberg'][f'fdr_{level:.2f}']
        print(f"BH at FDR {level:.0%}: {e['n']}: " + '; '.join(e['decisions']))
    print(f"Holm within families (2.5%): {len(mult['holm']['survivors'])}: "
          + '; '.join(mult['holm']['survivors']))
    ch = mult['chance']
    print(f"expected by chance: {ch['expected_passes']} of {ch['passes']} passes "
          f"({ch['share_of_passes']:.1%})")
    for item in out['hardware']['passes']:
        sides = ', '.join(' '.join(filter(None, (m.upper(), item[m].get('variant'),
                                                 signed(item[m]['mean_diff']),
                                                 ci(item[m]['ci95']))))
                          for m in ('cpu', 'gpu') if item[m])
        print(f"hardware: {item['decision']}: {sides}")
    runs = out['hardware']['runs']
    print(f"matched CPU-GPU runs: {runs['runs']} in {runs['files']} files, |F1 difference| median "
          f"{runs['abs_diff_f1']['median']}, mean {runs['abs_diff_f1']['mean']}, max "
          f"{runs['abs_diff_f1']['max']}")
    print(f"unseen minus seen: {out['unseen_minus_seen']['note']}")
    for key, g in out['gaps'].items():
        if key != 'note':
            print(f"gap {key}: {g['gap']:+.2f} {ci(g['ci95'])} (Welch, df {g['df']}), "
                  f"P {g['p_two_sided']}")
    print(f'wrote {path}')
    return out


if __name__ == '__main__':
    main()
