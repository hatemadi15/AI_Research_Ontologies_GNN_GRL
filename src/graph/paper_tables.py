"""
paper_tables.py - the tables of the paper plan (docs/PAPER.md) from the results files

No training: reads the committed comparisons and analyses and writes Markdown tables to
docs/tables/:
  table1_published_split.md (C1, C3, C13): per published split (MaterioMiner's fine and
      coarse levels, SOFC-Exp's official split), plain MatSciBERT and the CRF head over their
      10 model seeds under every F1 average, mean [95% CI], next to the published number; the
      CRF head minus plain MatSciBERT, paired by seed; and on SOFC-Exp each method minus the
      CRF head (ner_week5_seed_analysis.json, published_split).
  table2_decisions.md (C5-C8): every pre-registered decision and screen on MaterioMiner,
      weeks 3 to 7 (the comparisons files): its design, the mean paired difference of micro
      F1 with its 95% CI, and whether the CI lies above 0.
  table3_strict.md (C4): default against strict IOB2 scoring on the published splits, for
      plain MatSciBERT, the CRF head and their difference.
  table4_unseen.md (C17, C18): per setting, the share of the gold test entities whose names
      are unseen in the training sentences, the CRF head's recall on seen and unseen names
      (ner_mentions.json), and its F1 under the four matching criteria (ner_matching.json).
  table5_probe.md (C24): per MuLMS subfield, what the topic corpus holds of the unseen test
      names (mulms_coverage.json) and what the two silver filters admit at 0.9, the held-out
      CRF head's precision and recall, the dictionary rule's F1 change with each 0.9 lexicon
      and with the gold unseen names (silver_lexicons/<subfield>.json).
  table6_held_out_decisions.md (C22-C25): every pre-registered decision on held-out
      documents of the three corpora (ner_week5_seed_analysis.json): MuLMS's held-out
      subfields (weeks 10 to 12), the consistency rule of weeks 7 and 13, SOFC-Exp's article
      folds (13.3) and the pooled decision 13.2; its units, the mean paired difference of
      micro F1 with its 95% CI, and whether the CI lies above 0.
  table7_ablations.md (C26, paper A's Section 4.3): the ablations of the names method
      (week 14, 14.3; reported): on MuLMS's held-out subfields at the screen's seeds, the
      method with one setting changed (the corpus's first 10,000 or 20,000 sentences, the
      marginal threshold 0.7, 2 of 3 teachers, the harvested names in half of the draws),
      each against the method, mention replacement and the CRF head, with the distinct
      names the teachers admit and the share of the unseen test names among them.
  table8_types.md (paper A): per entity type on held-out documents, the CRF head, mention
      replacement with 6 copies (MR6) and MR6 with harvested names: F1 per type averaged over
      the (seed, unit) pairs whose test set has the type, and the differences; MuLMS's 13
      types over the decision's seeds, MaterioMiner's coarse-grained types over 10 seeds
      (week14.types of the seed analysis); tableA2_types_fine.md the fine-grained classes.
  table10_lexicon.md (paper A): recall on the unseen test names in and out of the rebuilt
      lexicon per system, and the paired gains per part (week 15, ner_lexicon_recall.json)
  table9_robustness.md (paper A): the pooled comparisons over the 11 held-out units of
      decision 13.2 among the CRF head, MR3, MR6 and MR6 + names, on both MuLMS seed bases
      (ner_robustness.json): mean difference with its 95% CI, positive units, the exact
      sign-flip permutation p-value, and the leave-one-unit-out minimum for the names
      against the CRF head; then the same comparisons on MaterioMiner's papers per
      granularity.
  table16_decision_tests.md (paper A): per decision of the names route, the t statistic with
      its degrees of freedom and the one- and two-sided P values from the per-unit differences,
      with the exact sign-flip P (ner_decision_tests.json, decision_tests.py; reporting only).
  tableA1_all_comparisons.md: every comparison of the comparisons files, the reported and
      descriptive ones too.
A table whose results are missing is skipped (and named on stdout).

Usage:
  python src/graph/paper_tables.py [--results results] [--out docs/tables]
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# (week, machine, seeds per split, comparisons file under the results directory)
COMPARISONS = (('3', 'CPU', '3', 'ner_week3_comparisons.json'),
               ('4', 'CPU', '3', 'ner_week4_comparisons.json'),
               ('4', 'GPU', '3', 'gpu/ner_week4_comparisons.json'),
               ('5', 'GPU', '10', 'gpu/ner_week5_comparisons.json'),
               ('6', 'GPU', '3', 'gpu/ner_week6_screen.json'),
               ('6', 'GPU', '10', 'gpu/ner_week6_comparisons.json'),
               ('7', 'GPU', '3', 'gpu/ner_week7_screen.json'),
               ('7', 'GPU', '10', 'gpu/ner_week7_comparisons.json'),
               ('9', 'GPU', '10', 'gpu/ner_week9_comparisons.json'),
               ('11', 'GPU', '10', 'gpu/ner_week11_comparisons.json'),
               ('12', 'GPU', '10', 'gpu/ner_week12_comparisons.json'))
DECIDED = ('decision', 'screen')
PROTOCOL_NAMES = {'random_split_vary': 'random splits', 'leave_one_paper_out': 'unseen papers'}
AVERAGES = (('micro', 'micro'), ('macro', 'macro'), ('weighted', 'weighted'),
            ('micro_strict', 'strict micro'))
SPLIT_NAMES = {'fine': 'MaterioMiner, fine-grained', 'coarse': 'MaterioMiner, coarse-grained',
               'sofc': 'SOFC-Exp', 'sofc_slot': 'SOFC-Exp, slot types'}
# the system whose published score a split's entry holds: plain MatSciBERT on MaterioMiner
# (Durmaz et al. 2024), MatSciBERT-CRF on SOFC-Exp (Gupta et al. 2022)
PUBLISHED_SYSTEM = {'fine': 'baseline', 'coarse': 'baseline', 'sofc': 'crf', 'sofc_slot': 'crf'}
UNSEEN_SETTINGS = (('fine_random_split_vary', 'fine, random splits'),
                   ('fine_leave_one_paper_out', 'fine, unseen papers'),
                   ('coarse_random_split_vary', 'coarse, random splits'),
                   ('coarse_leave_one_paper_out', 'coarse, unseen papers'),
                   ('sofc_random_split_vary', 'SOFC-Exp, random splits'),
                   ('sofc_document_folds', 'SOFC-Exp, article folds'),
                   ('sofc_published_split', 'SOFC-Exp, official split'),
                   ('sofc_slot_published_split', 'SOFC-Exp slot types, official split'),
                   ('mulms_random_split_pool', 'MuLMS, random splits'),
                   ('mulms_leave_one_subfield_out', 'MuLMS, held-out subfields'))
CRITERIA = ('strict', 'exact', 'type', 'overlap')


def number(x, signed=False):
    """x with 2 decimals and a typographic minus; with signed, a + on positive numbers."""
    return (f'{x:+.2f}' if signed else f'{x:.2f}').replace('-', '\u2212')


def interval(entry, key='mean', signed=False):
    """'mean [lo, hi]' of a spread entry ({key, 'ci95'}), signed for differences."""
    lo, hi = entry['ci95']
    return f"{number(entry[key], signed)} [{number(lo, signed)}, {number(hi, signed)}]"


def markdown(header, rows):
    """A Markdown table."""
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines) + '\n'


def table1(published):
    """Table 1 from the published_split analysis {granularity: entry}."""
    rows = []
    for g, entry in published.items():
        name = SPLIT_NAMES.get(g, g)
        reference = number(entry['published'])
        if 'published_macro' in entry:
            reference += f" (macro {number(entry['published_macro'])})"
        for system, label in (('baseline', 'plain MatSciBERT'), ('crf', 'CRF head')):
            rows.append([name, label, *(interval(entry[system][a]) for a, _ in AVERAGES),
                         reference if system == PUBLISHED_SYSTEM.get(g, 'baseline') else ''])
        rows.append([name, 'CRF head − plain', *(interval(entry['crf_minus_baseline'][a],
                                                          'mean_diff', signed=True)
                                                 for a, _ in AVERAGES), ''])
        for method, m in entry.get('methods', {}).items():
            rows.append([name, f'`{method}` − CRF head',
                         *(interval(m['minus_crf'][a], 'mean_diff', signed=True)
                           for a, _ in AVERAGES), ''])
    header = ['split', 'system', *(label for _, label in AVERAGES), 'published']
    return ('Table 1. Published splits, 10 model seeds: mean F1 [95% CI]; differences '
            'paired by seed. The published score is plain MatSciBERT\'s on MaterioMiner '
            '(Durmaz et al. 2024) and MatSciBERT-CRF\'s on SOFC-Exp (Gupta et al. 2022).\n\n'
            + markdown(header, rows))


def comparison_rows(files, roles=None):
    """[(week, machine, seeds, entry)] of the comparisons files, in their order, only the
    entries whose role is in `roles` (all if None)."""
    rows = []
    for week, machine, seeds, data in files:
        for entry in data.values():
            if roles is None or entry.get('role') in roles:
                rows.append((week, machine, seeds, entry))
    return rows


def table2(rows, title='Table 2. Pre-registered decisions and screens on MaterioMiner'):
    """Table 2 from comparison_rows: one row per comparison."""
    out = []
    for week, machine, seeds, e in rows:
        decided = e.get('role') in DECIDED
        lo = e['micro']['ci95'][0]
        out.append([week, machine, seeds, e['granularity'],
                    PROTOCOL_NAMES.get(e['protocol'], e['protocol']),
                    f"`{e['variant']}` vs `{e['reference']}`", e.get('role', ''),
                    e['n_pairs'], interval(e['micro'], 'mean_diff', signed=True),
                    ('passes' if lo > 0 else 'no') if decided else ''])
    header = ['week', 'machine', 'seeds', 'granularity', 'protocol', 'comparison', 'role',
              'pairs', 'Δ micro F1 [95% CI]', 'CI above 0']
    return (f'{title}: mean paired difference of micro F1 over splits or held-out papers, '
            f'each side the mean over its seeds.\n\n' + markdown(header, out))


HELD_OUT_CORPORA = {'mulms': 'MuLMS, held-out subfields',
                    'fine': 'MaterioMiner fine-grained, unseen papers',
                    'coarse': 'MaterioMiner coarse-grained, unseen papers',
                    'sofc_folds': 'SOFC-Exp, article folds'}


def held_out_rows(analysis, week12=None):
    """The rows of Table 6 from the week-5 analyses file, in week order: every pre-registered
    decision on held-out documents that the seed analysis holds (MuLMS's decisions of weeks
    10 and 11, the week-12 winner's, the consistency rule's of weeks 7 and 13, 13.3's on
    SOFC-Exp's folds, and the pooled decision 13.2), and from the week-12 comparisons
    (`week12`, ner_week12_comparisons.json) the winner's MaterioMiner analogue against the CRF
    head on the held-out papers, which 13.2 pools; week 14's decisions 14.1 (the names
    against matched copies, pooled) and 14.2 (the topic's own corpus against the mixed one);
    with its units, difference and verdict; reported comparisons are left out
    (MaterioMiner's other decisions are in Table 2)."""
    rows = []

    def row(week, corpus, comparison, decision, units, diff, passes):
        rows.append([week, corpus, comparison, decision, units,
                     interval(diff, 'mean_diff', signed=True), 'passes' if passes else 'no'])

    mulms = analysis.get('mulms') or {}
    held = mulms.get('leave_one_subfield_out')
    if held:
        row(10, HELD_OUT_CORPORA['mulms'], '`crf_mr` vs `crf`', 'week 10, decision 1',
            held['n_subfields'], held['crf_mr_minus_crf']['micro'], held['passes'])
        votes = held.get('votes')
        if votes:
            row(10, HELD_OUT_CORPORA['mulms'], '`crf_mr` votes vs `crf` votes',
                'week 10, decision 2', held['n_subfields'], votes['crf_mr_minus_crf']['micro'],
                votes['passes'])
    names = mulms.get('silver_names')
    if names:
        for reference, key, label in (('crf', 'decision_1', 'week 11, decision 1'),
                                      ('crf_mr', 'decision_2', 'week 11, decision 2')):
            row(11, HELD_OUT_CORPORA['mulms'], f'`crf_mrs` vs `{reference}`', label,
                names['n_subfields'], names[f'minus_{reference}']['micro'], names[key])
    week13 = analysis.get('week13') or {}
    winner = week13.get('week12_winner')
    part = ((mulms.get('week12') or {}).get(winner) or {}).get('decision') if winner else None
    if part:
        for reference, key, label in (('crf', 'decision_1', 'week 12, decision 1'),
                                      ('crf_mr', 'decision_2', 'week 12, decision 2')):
            row(12, HELD_OUT_CORPORA['mulms'], f'`{winner}` vs `{reference}`', label,
                part['n_subfields'], part[f'minus_{reference}']['micro'], part[key])
    for e in (week12 or {}).values():
        if e.get('role') == 'decision' and e.get('protocol') == 'leave_one_paper_out':
            row(12, HELD_OUT_CORPORA.get(e['granularity'], e['granularity']),
                f"`{e['variant']}` vs `{e['reference']}`", "week 12, the winner's analogue",
                e['n_pairs'], e['micro'], e['micro']['ci95'][0] > 0)
    for corpus, entries in (week13.get('consistency') or {}).items():
        for variant, e in entries.items():
            if 'decision' not in e:
                continue
            system = variant.rpartition('_')[0]
            recomputed = e['decision'] == 'week 7'
            row(7 if recomputed else 13, HELD_OUT_CORPORA.get(corpus, corpus),
                f'`{variant}` vs `{system}`',
                'week 7 (recomputed)' if recomputed else f"week 13, {e['decision']}",
                e['n_units'], e['minus_input']['micro'], e['passes'])
    for method, e in (week13.get('folds') or {}).items():
        row(13, HELD_OUT_CORPORA['sofc_folds'], f'`{method}` vs `crf`',
            f"week 13, {e['decision']}", e['n_units'], e['minus_crf']['micro'], e['passes'])
    for method, e in (week13.get('pooled') or {}).items():
        if e.get('status') != 'decision':
            continue
        corpora = ' + '.join(f"{HELD_OUT_CORPORA.get(c, c)} ({v['n_units']})"
                             for c, v in e['corpora'].items())
        row(13, f'pooled: {corpora}', f'`{method}` vs `crf`',
            'week 13, 13.2' + ('' if e['no_corpus_loses'] else ', a corpus loses'),
            e['n_units'], e['minus_crf'], e['passes'] and e['no_corpus_loses'])
    week14 = analysis.get('week14') or {}
    pooled = (week14.get('copies') or {}).get('pooled')
    if pooled:
        corpora = ' + '.join(f"{HELD_OUT_CORPORA.get(c, c)} ({v['n_units']})"
                             for c, v in pooled['corpora'].items())
        row(14, f'pooled: {corpora}', '`crf_mrm_topic6` / `crf_mrm6` vs `crf_mr6`',
            'week 14, 14.1' + ('' if pooled['no_corpus_loses'] else ', a corpus loses'),
            pooled['n_units'], pooled['minus_control'],
            pooled['passes'] and pooled['no_corpus_loses'])
    mixed = (week14.get('topic') or {}).get('decision')
    if mixed:
        row(14, HELD_OUT_CORPORA['mulms'], '`crf_mrm_topic6` vs `crf_mrm_mixed6`', 'week 14, 14.2',
            mixed['n_units'], mixed['minus_crf_mrm_mixed6']['micro'], mixed['passes'])
    rows.sort(key=lambda r: r[0])
    return rows


def table6(rows):
    """Table 6 from held_out_rows."""
    header = ['week', 'corpus, protocol', 'comparison', 'decision', 'units',
              'Δ micro F1 [95% CI]', 'CI above 0']
    return ('Table 6. Pre-registered decisions on held-out documents of three corpora from '
            'weeks 7 to 13: MuLMS\'s held-out subfields, the consistency rule, the week-12 '
            "winner's MaterioMiner analogue, SOFC-Exp's article folds and the pooled decision "
            '(GPU runs, 10 model seeds; a unit is a held-out subfield, paper or article fold, '
            'each side the mean over its seeds; MaterioMiner\'s other unseen-paper decisions '
            'are in Table 2): mean paired difference of micro F1 over the units with its 95% '
            'CI, and whether the CI lies above 0.\n\n' + markdown(header, rows))


# Table 7: the ablations of the names method (week 14, 14.3), in this order
ABLATION_LABELS = {'crf_mrm_topic10k6': "the corpus's first 10,000 sentences (40,000)",
                   'crf_mrm_topic20k6': "the corpus's first 20,000 sentences (40,000)",
                   'crf_mrm_topiclow6': 'marginal threshold 0.7 (0.9)',
                   'crf_mrm_topictwo6': '2 of 3 teachers agree (3 of 3)',
                   'crf_mrm_topiccap6': 'harvested names in half of the draws (a uniform pool)'}
METHOD_LABEL = 'the method: 40,000 sentences, threshold 0.9, 3 of 3 teachers, a uniform pool'


def ablation_rows(analysis):
    """The rows of Table 7 from the week-5 analyses file: the names method on the screen's
    seeds first (the week-12 screen entry of week13.week12_winner), then each ablation of
    week14.ablations in ABLATION_LABELS order: label, held-out units, distinct silver names
    (summed over the units), the share of the unseen test names among the silver names,
    micro F1, and the paired difference (micro, with its CI) against the method, mention
    replacement and the CRF head, None where a reference is missing."""
    rows = []
    winner = analysis.get('week13', {}).get('week12_winner', 'crf_mrm_topic6')
    screen = analysis.get('mulms', {}).get('week12', {}).get(winner, {}).get('screen')
    if screen and winner in screen:
        names = screen.get('names', {})
        rows.append([METHOD_LABEL, screen.get('n_subfields', len(names)),
                     sum(n['n_distinct'] for n in names.values()), screen.get('unseen_in_silver'),
                     screen[winner]['micro']['mean'], None,
                     screen.get('minus_crf_mr', {}).get('micro'),
                     screen.get('minus_crf', {}).get('micro')])
    for variant, label in ABLATION_LABELS.items():
        entry = analysis.get('week14', {}).get('ablations', {}).get(variant)
        if not entry or variant not in entry:
            continue
        names = entry.get('names', {})
        rows.append([label, entry['n_units'], sum(n['n_distinct'] for n in names.values()),
                     entry.get('unseen_in_silver'), entry[variant]['micro']['mean'],
                     entry.get(f'minus_{winner}', {}).get('micro'),
                     entry.get('minus_crf_mr', {}).get('micro'),
                     entry.get('minus_crf', {}).get('micro')])
    return rows


def table7(rows):
    """Table 7 from ablation_rows."""
    header = ["setting (the method's value)", 'units', 'distinct silver names',
              'unseen test names among them (%)', 'micro F1', 'Δ vs the method [95% CI]',
              'Δ vs mention replacement [95% CI]', 'Δ vs CRF head [95% CI]']
    cells = []
    for label, units, distinct, share, micro, vs_method, vs_mr, vs_crf in rows:
        cells.append([label, units, f'{distinct:,}', '' if share is None else f'{100 * share:.1f}',
                      number(micro), *(interval(d, 'mean_diff', signed=True) if d else ''
                                       for d in (vs_method, vs_mr, vs_crf))])
    return ("Table 7. The ablations of the names method (week 14, 14.3; reported, no decision): "
            "on MuLMS's 7 held-out subfields at the screen's seed offsets 0 to 200, the method "
            'with one setting changed, each against the method, mention replacement with 3 '
            'copies and the CRF head on the same seeds (mean paired micro-F1 difference over '
            'the subfields with its 95% CI); the distinct names the three teachers admit, '
            "summed over the subfields, and the share of the held-out subfields' unseen test "
            'names among them.\n\n' + markdown(header, cells))


# Table 8: per entity type (week14.types); Table 9: the pooled comparisons (ner_robustness.json)
TYPE_PARTS = {'mulms': "MuLMS, 7 held-out subfields (seeds 300–900)",
              'coarse': "MaterioMiner coarse-grained, 4 held-out papers (10 seeds)",
              'fine': "MaterioMiner fine-grained, 4 held-out papers (10 seeds)"}
ROBUST_LABELS = {'names_vs_crf': 'MR6 + names − CRF head', 'names_vs_mr3': 'MR6 + names − MR3',
                 'names_vs_mr6': 'MR6 + names − MR6', 'mr6_vs_crf': 'MR6 − CRF head',
                 'mr6_vs_mr3': 'MR6 − MR3', 'mr3_vs_crf': 'MR3 − CRF head'}
BASES = {'ten': 'MuLMS seeds 0–900 (as decided)', 'decision': 'MuLMS seeds 300–900 (not seen by the screen)'}


def type_rows(analysis, part):
    """Rows of Table 8 for one part of week14.types: per type, most frequent first: type,
    mean test support per unit, units with the type, the CRF head's F1, MR6's, the names
    method's, the names method minus the CRF head and minus MR6 (None without MR6)."""
    types = analysis.get('week14', {}).get('types', {}).get(part, {})
    method = types.get('crf_mrm_topic6') or types.get('crf_mrm6')
    control = types.get('crf_mr6', {})
    rows = []
    for name, e in (method or {}).items():
        c = control.get(name)
        rows.append([name, e['support_per_split'], e['n_splits'], e['crf'],
                     None if c is None else c['method'], e['method'], e['diff'],
                     None if c is None else round(e['method'] - c['method'], 2)])
    return rows


def table8(parts, title='Table 8', note=''):
    """Table 8 from {part: type_rows}; one block per part."""
    header = ['type', 'test support per unit', 'units', 'CRF head', 'MR6', 'MR6 + names',
              'names − CRF head', 'names − MR6']
    blocks = []
    for part, rows in parts.items():
        cells = [[name, support, units, number(crf), '' if mr6 is None else number(mr6),
                  number(method), number(d_crf, signed=True),
                  '' if d_mr6 is None else number(d_mr6, signed=True)]
                 for name, support, units, crf, mr6, method, d_crf, d_mr6 in rows]
        blocks.append(f'**{TYPE_PARTS.get(part, part)}**\n\n' + markdown(header, cells))
    return (f'{title}. Per entity type on held-out documents: micro F1 of the CRF head, mention '
            'replacement with 6 copies (MR6) and MR6 with harvested names, averaged over the '
            '(seed, held-out unit) pairs whose test set has the type, with the names method\'s '
            'differences; types ordered by test support. Exploratory: per-type differences are '
            f'not decided on.{note}\n\n' + '\n\n'.join(blocks))


def robustness_rows(robust):
    """Rows of Table 9's first block from ner_robustness.json's pooled section: per basis
    and comparison in ROBUST_LABELS order: basis label, comparison label, interval entry,
    positives of units, sign-flip p, and for the names against the CRF head the
    leave-one-out minimum lower bound."""
    rows = []
    for basis, label in BASES.items():
        entries = robust.get('pooled', {}).get(basis)
        if not entries:
            continue
        for key, name in ROBUST_LABELS.items():
            e = entries.get(key)
            if e:
                loo = e.get('leave_one_out', {})
                rows.append([label, name, e, f"{e['n_positive']}/{e['n_units']}", e['sign_flip_p'],
                             loo.get('min_lower')])
    return rows


def table9(robust):
    """Table 9 from ner_robustness.json: the pooled comparisons on both bases, then the
    MaterioMiner grids."""
    rows = robustness_rows(robust)
    cells = [[basis, name, interval(e, 'mean_diff', signed=True), pos, f'{p_value:.4f}',
              '' if loo is None else number(loo, signed=True)]
             for basis, name, e, pos, p_value, loo in rows]
    header = ['MuLMS seed basis', 'comparison', 'Δ micro F1 [95% CI], 11 units', 'units above 0',
              'sign-flip p', 'leave-one-unit-out: lowest lower bound']
    blocks = [markdown(header, cells)]
    grids = robust.get('grids', {})
    if grids:
        gh = ['comparison', *(f'{g}-grained, 4 papers [95% CI]' for g in grids)]
        gcells = [[name, *(interval(grids[g][key], 'mean_diff', signed=True) if key in grids[g] else ''
                           for g in grids)] for key, name in ROBUST_LABELS.items()]
        blocks.append('**MaterioMiner\'s held-out papers (10 seeds)**\n\n' + markdown(gh, gcells))
    return ('Table 9. Post hoc robustness of the pooled comparisons (ner_robustness.py): over the '
            '11 held-out units of decision 13.2 (MuLMS\'s 7 subfields and MaterioMiner\'s 4 '
            'fine-grained papers), every comparison among the CRF head, mention replacement with '
            '3 copies (MR3), with 6 copies (MR6) and MR6 with harvested names, with the MuLMS '
            'units as 10-seed means (as decided) and as means over the 7 seeds the screen did '
            'not see; the mean paired difference with its 95% Student-t CI, the number of units '
            'above 0, the exact two-sided sign-flip permutation p-value of the mean, and for the '
            'names against the CRF head the lowest lower bound over the 11 leave-one-unit-out '
            'intervals. Only decision 13.2 (first row of the first basis) was pre-registered.'
            '\n\n' + '\n\n'.join(blocks))


LEXICON_SETTINGS = {'mulms': "MuLMS, 7 held-out subfields (seeds 300–900)",
                    'fine': "MaterioMiner fine-grained, 4 held-out papers (10 seeds)",
                    'coarse': "MaterioMiner coarse-grained, 4 held-out papers (10 seeds)"}
LEXICON_COMPARISONS = {'names_vs_mr6': 'MR6 + names − MR6', 'names_vs_crf': 'MR6 + names − CRF head',
                       'mr6_vs_crf': 'MR6 − CRF head', 'mr3_vs_crf': 'MR3 − CRF head',
                       'mixed_vs_mr6': 'MR6 + names (mixed corpus) − MR6'}


CHAIN_SETTINGS = {'mulms_leave_one_subfield_out': "MuLMS, 7 subfields",
                  'fine_leave_one_paper_out': 'MaterioMiner fine, 4 papers',
                  'coarse_leave_one_paper_out': 'MaterioMiner coarse, 4 papers',
                  'sofc_document_folds': "SOFC-Exp, 5 folds"}
CHAIN_SYSTEMS = {'crf': 'CRF head', 'crf_mr': 'MR3', 'crf_mr6': 'MR6', 'crf_mrm_topic6': 'MR6 + names',
                 'crf_mrm6': 'MR6 + names', 'crf_mrm_mixed6': 'MR6 + names (mixed corpus)',
                 'crf_mt': 'multi-task', 'crf_mt_mr': 'multi-task + MR3',
                 'crf_mt_mr6': 'multi-task + MR6', 'crf_mt_mrm6': 'multi-task + MR6 + names'}


def table14(mentions):
    """Table 14 from ner_mentions.json's `chain` block: per setting, each system against the one it
    builds on, the paired differences in F1, recall and precision on the unseen and on the seen
    test names and over all entities, on all ten seeds and on MuLMS's decision seeds."""
    header = ['setting', 'comparison', 'seeds', 'part', 'Δ F1 [95% CI]', 'Δ recall [95% CI]',
              'Δ precision [95% CI]']
    cells = []
    for key, label in CHAIN_SETTINGS.items():
        for system, e in (mentions.get('chain', {}).get(key) or {}).items():
            name = f"{CHAIN_SYSTEMS.get(system, system)} − {CHAIN_SYSTEMS.get(e['reference'], e['reference'])}"
            top = 'all ten' if tuple(e.get('seeds') or range(0, 1000, 100)) == tuple(range(0, 1000, 100)) \
                else 'decision seeds (300–900)'
            for basis, block in ((top, e), ('decision seeds (300–900)', e.get('decision'))):
                if not block or (block is not e and not key.startswith('mulms')):
                    continue
                for part in ('unseen', 'seen', 'all'):
                    q = block.get(part)
                    if not q:
                        continue
                    cells.append([label, name, basis, {'all': 'all entities'}.get(part, part + ' names'),
                                  interval(q, 'mean_diff', signed=True),
                                  interval(q['recall'], 'mean_diff', signed=True) if 'recall' in q else '',
                                  interval(q['precision'], 'mean_diff', signed=True) if 'precision' in q else ''])
    return ('Table 14. The unseen-name trade in full (ner_mentions.json, `chain`): per setting, each '
            'system against the one it builds on, the mean paired difference over the units in F1, '
            'recall and precision on the unseen and the seen test names and over all entities, with '
            "95% CIs, on all ten seeds and, on MuLMS, on the decision seeds (300–900).\n\n"
            + markdown(header, cells))


def short_unit(unit):
    """A MaterioMiner paper by the short name the paper uses (s11340 for
    10.1007_s11340-021-00758-x.conll); a MuLMS subfield or SOFC-Exp fold as stored."""
    return unit[:-len('.conll')].partition('_')[2].split('-')[0] if unit.endswith('.conll') else unit


def table15(data):
    """Table 15 from ner_lexicon_recall.json: per setting and comparison, each unit's recall gain on
    the unseen test names the lexicon holds and on the others, and their difference."""
    header = ['setting', 'comparison', 'unit', 'gain, names in the lexicon', 'gain, names not in it',
              'in − out']
    cells = []
    for setting, label in LEXICON_SETTINGS.items():
        for key, name in LEXICON_COMPARISONS.items():
            c = ((data.get(setting) or {}).get('comparisons') or {}).get(key)
            if not c:
                continue
            for unit, v in c['per_unit'].items():
                cells.append([label, name, short_unit(unit), number(v['gain_in'], True),
                              number(v['gain_out'], True), number(v['in_minus_out'], True)])
    return ('Table 15. The lexicon split per unit (ner_lexicon_recall.json, post hoc): per setting '
            "and comparison, each held-out unit's recall gain on the unseen test names its lexicon "
            'holds and on the others, and their difference (the paired means are Table 10).\n\n'
            + markdown(header, cells))


def table16(tests):
    """Table 16 from ner_decision_tests.json (decision_tests.py): per decision of paper A's
    names route, the units, the stored difference with its 95% CI, the standard deviation of
    the per-unit differences, the t statistic with its degrees of freedom, the one- and
    two-sided P values and the exact two-sided sign-flip P, the verdict, and (post hoc) the
    interval and P with the units correlated where their training sets share documents;
    reporting only, the decisions rest on the intervals."""
    dependence = (tests.get('post_hoc_dependence') or {}).get('decisions', {})
    header = ['decision or reading', 'comparison', 'units', 'Δ [95% CI]', 'SD', 't (df)', 'P one-sided',
              'P two-sided', 'sign-flip P', 'verdict', 'units correlated: 95% CI, P (post hoc)']

    def correlated(key):
        e = dependence.get(key)
        if not e:
            return ''
        return (interval({'mean_diff': e['mean_diff'], 'ci95': e['ci95']}, 'mean_diff',
                         signed=True).split(' ', 1)[1] + f", {e['p_two_sided']:.3f}")
    rows = [[r['decision'], r['comparison'], r['n_units'],
             interval({'mean_diff': r['stored_mean_diff'], 'ci95': r['stored_ci95']},
                      'mean_diff', signed=True), f"{r['sd']:.2f}",
             f"{r['t']:.2f} ({r['df']})".replace('-', '−'), f"{r['p_one_sided']:.4f}",
             f"{r['p_two_sided']:.4f}", f"{r['sign_flip_p_two_sided']:.4f}",
             r.get('verdict') or ('passes' if r['passes'] else 'no'), correlated(r['decision'])]
            for r in tests['decisions']]
    return ("Table 16. The t statistic and P values of paper A's decisions (D5' the post hoc "
            'reading of D5 on MuLMS\'s decision seeds): a one-sample Student-t test of the mean '
            'paired difference against 0 over the units, n - 1 degrees of freedom, with the '
            'standard deviation (SD) of the per-unit differences, from the '
            'per-unit differences the decision used (rounded to two decimals, so t and P can '
            'differ from the unrounded ones in the last digit), the exact two-sided sign-flip '
            'P over all 2ⁿ sign patterns beside it. Reporting only: each decision was taken by '
            'the registered rule, the 95% interval above 0 (for the pooled ones, with neither '
            "corpus's interval below 0), which is the one-sided test at 2.5%. The last column "
            '(post hoc, round 15) gives the interval and two-sided P with the units correlated '
            'where their training sets share documents, at the corrected resampled t test\'s '
            "1/K: 1/4 between MaterioMiner's papers, 1/5 between SOFC-Exp's folds, 1/4 between "
            'the 5 of 21 pairs of MuLMS subfields whose training pools share an article, 0 '
            'otherwise and across corpora (decision_tests.py, post_hoc_dependence).\n\n'
            + markdown(header, rows))


def lexicon_rows(data):
    """[(setting, system label, pooled scores)] and [(setting, comparison label, comparison)]
    from ner_lexicon_recall.json, settings in LEXICON_SETTINGS order."""
    scores, comparisons = [], []
    for key, label in LEXICON_SETTINGS.items():
        part = data.get(key)
        if not part:
            continue
        for system, name in part['systems'].items():
            scores.append((label, name, part['pooled'][system]))
        for ckey, cname in LEXICON_COMPARISONS.items():
            if ckey in part['comparisons']:
                comparisons.append((label, cname, part['comparisons'][ckey]))
    return scores, comparisons


WEEK17_NAMES, WEEK17_CONTROL = 'crf_mt_mrm6', 'crf_mt_mr6'
WEEK17_LABELS = {WEEK17_NAMES: 'multi-task + MR6 + names', WEEK17_CONTROL: 'multi-task + MR6 (the control)',
                 'crf_mt_mr': 'multi-task + MR3 (week 9)', 'crf_mt': 'multi-task model', 'crf': 'CRF head'}


def week17_rows(analysis):
    """[(granularity, decision, names entry)] from the week17 section of the seed analysis
    (ner_seed_analysis.week17), in granularity order; empty without it."""
    w = analysis.get('week17') or {}
    return [(g, e['decision'], e) for g, e in w.items()]


def table11(rows):
    """Table 11 from week17_rows: per granularity the names' and the control's micro F1 over the
    4 held-out papers (10 seeds), the names against the control (the decision), and both against
    multi-task + MR3, the multi-task model and the CRF head (reported)."""
    header = ['granularity', 'system', 'micro F1 [95% CI], 4 papers', 'Δ vs the control [95% CI]',
              'Δ vs multi-task + MR3', 'Δ vs the multi-task model', 'Δ vs the CRF head']
    cells = []
    for g, decision, e in rows:
        names, control = e['names'], e['control']
        cells.append([f'{g}-grained', WEEK17_LABELS[WEEK17_NAMES], interval(e[WEEK17_NAMES]['micro']),
                      interval(e[f'minus_{WEEK17_CONTROL}']['micro'], 'mean_diff', signed=True)
                      + f" (decision {decision}, {'passes' if e['passes'] else 'no'})",
                      interval(names['minus_crf_mt_mr']['micro'], 'mean_diff', signed=True),
                      interval(names['minus_crf_mt']['micro'], 'mean_diff', signed=True),
                      interval(names['minus_crf']['micro'], 'mean_diff', signed=True)])
        cells.append([f'{g}-grained', WEEK17_LABELS[WEEK17_CONTROL],
                      interval(control[WEEK17_CONTROL]['micro']), '',
                      interval(control['minus_crf_mt_mr']['micro'], 'mean_diff', signed=True),
                      interval(control['minus_crf_mt']['micro'], 'mean_diff', signed=True),
                      interval(control['minus_crf']['micro'], 'mean_diff', signed=True)])
    return ('Table 11. The harvested names on the strongest MaterioMiner base (week 17, post review; '
            'the multi-task model with 6-copy mention replacement, with and without the fatigue '
            "corpus's names; 4 held-out papers, 10 seeds, both granularities from one model): micro "
            'F1 with its 95% CI over the papers, the names against the matched control (decisions '
            '17.1 and 17.2, the week-6 rule), and both against multi-task + MR3, the multi-task model '
            'and the CRF head (reported).\n\n' + markdown(header, cells))


def table12(matched):
    """Table 12 from ner_recall_matched.json (week 18): per MuLMS subfield the names' pooled recall,
    the control's at bias 0, the matched bias D* (flagged outside the grid) and the names minus the
    control at D* in recall on the unseen names the lexicon holds, micro F1 and precision, a
    flagged subfield's post hoc nearest-recall values in parentheses; then the two decisions and
    the pre-specified comparisons over the subfields, the same without the flagged subfields
    (pre-specified) and with them matched at the nearest recall (post hoc)."""
    header = ['subfield', "names' recall", "control's recall at 0", 'D*',
              'Δ recall, unseen names in the lexicon', 'Δ micro F1', 'Δ precision']
    d, pre = matched['decisions'], matched['prespecified']
    nearest = matched.get('post_hoc_nearest') or {}
    near_d, near_pre = nearest.get('decisions') or {}, nearest.get('prespecified') or {}

    def cell(block, u, alt=None):
        out = number(block['per_unit'][u], True)
        return out + (f" ({number(alt['per_unit'][u], True)})" if alt else '')
    cells = []
    for u, v in matched['per_unit'].items():
        near = v['flagged'] and u in (nearest.get('matched') or {})
        star = number(v['d_star'])
        if v['flagged']:
            star += f" (flagged; {nearest['matched'][u]['d']:g} post hoc)" if near else ' (flagged)'
        cells.append([u, number(v['target_recall']), number(v['control_recall_at_0']), star,
                      cell(d['18.1'], u, near and near_d['18.1']),
                      cell(d['18.2'], u, near and near_d['18.2']),
                      cell(pre['precision'], u, near and near_pre['precision'])])
    rows = [(f"mean [95% CI], {d['18.1']['n_units']} subfields", d, pre, True)]
    without = matched.get('without_flagged')
    if without:
        rows.append((f"without the flagged subfields, {without['decisions']['18.1']['n_units']} "
                     '(pre-specified)', without['decisions'], without['prespecified'], False))
    if nearest:
        rows.append(('the flagged subfields at the nearest recall, '
                     f"{near_d['18.1']['n_units']} (post hoc)", near_d, near_pre, False))
    for label, dec, pr, verdict in rows:
        cells.append([label, '', '', ''] + [
            interval(dec[k], 'mean_diff', signed=True)
            + (f" ({k}, {dec[k]['verdict']})" if verdict else
               f", {dec[k]['n_positive']}/{dec[k]['n_units']} above 0")
            for k in ('18.1', '18.2')] + [interval(pr['precision'], 'mean_diff', signed=True)])
    extra = '; '.join(f"{name} {interval(pre[key], 'mean_diff', signed=True)}"
                      for key, name in (('unseen_recall', 'recall on the unseen names'),
                                        ('unseen_precision', 'precision on them'),
                                        ('unseen_f1', 'F1 on them'),
                                        ('recall_out', 'recall on the unseen names not in the lexicon'),
                                        ('in_minus_out', 'in minus out'),
                                        ('bias_f1_change', "the bias's own F1 change")))
    return ('Table 12. The names against a recall-matched control (week 18, post review): MR6 on '
            "MuLMS's 7 held-out subfields at seeds 300 to 900 with a bias D on the entity "
            "emissions, matched per subfield to the names' recall over all test entities pooled "
            'over the seeds; the names minus the control at D*, paired over the subfields '
            '(decisions 18.1 and 18.2, the week-6 rule; the rest pre-specified). A flagged '
            "subfield's recall lies outside the control's range over the grid and is matched at "
            'its end as the protocol fixed; in parentheses, matched post hoc at the grid value of '
            "the control's recall nearest the names'. Also, paired: "
            f'{extra}.\n\n' + markdown(header, cells) + '\n'
            + table12_curves(matched))


def table12_curves(matched):
    """Table 12, second to fourth parts: per subfield the control's pooled recall, precision and
    F1 over all test entities at every bias of the grid, beside the names' (pre-specified); the
    per-subfield bracket and the pre-specified comparisons at D* (a flagged subfield's post hoc
    nearest-recall value in parentheses); and, post hoc, the control's recall on the unseen test
    names in and out of the lexicon at every bias, beside the names'. '' without the curves."""
    units = matched['per_unit']
    if not all('curve' in v for v in units.values()):
        return ''
    grid = list(next(iter(units.values()))['curve'])
    dcols = [f'D = {g}'.replace('-', '\u2212') for g in grid]

    def f1(r, p):
        return 2 * r * p / (r + p) if r + p else 0.0
    rows = []
    for u, v in units.items():
        names = {'recall': v['names']['recall'], 'precision': v['names']['precision'],
                 'F1': f1(v['names']['recall'], v['names']['precision'])}
        for label, key in (('recall', 'recall'), ('precision', 'precision'), ('F1', 'f1_pooled')):
            rows.append([u, label, number(names[label])]
                        + [number(v['curve'][g][key]) for g in grid])
    out = ("Table 12, second part (pre-specified). The control's recall, precision and F1 over "
           'all test entities, pooled over the seeds, at every bias of the grid, beside the '
           "names' (pooled the same way; the matching's input is the recall rows).\n\n"
           + markdown(['subfield', 'rate', 'names'] + dcols, rows))
    pre = matched['prespecified']
    near = (matched.get('post_hoc_nearest') or {}).get('prespecified') or {}
    keys = (('unseen_recall', 'Δ recall, unseen names'), ('unseen_precision', 'Δ precision, unseen names'),
            ('unseen_f1', 'Δ F1, unseen names'), ('recall_out', 'Δ recall, unseen names not in the lexicon'),
            ('in_minus_out', 'in − out'), ('bias_f1_change', "the bias's own F1 change"))
    rows = []
    for u, v in units.items():
        a, b = v['between']
        cells = [u, f"{number(a)} to {number(b)}, w = {v['w']:.2f}" if a != b else number(a)]
        for key, _ in keys:
            cell = number(pre[key]['per_unit'][u], True)
            if v['flagged'] and near.get(key):
                cell += f" ({number(near[key]['per_unit'][u], True)})"
            cells.append(cell)
        rows.append(cells)
    rows.append(['mean [95% CI]', ''] + [interval(pre[k], 'mean_diff', signed=True) for k, _ in keys])
    if near and all(k in near for k, _ in keys):
        rows.append(['mean, the flagged at the nearest recall (post hoc)', '']
                    + [interval(near[k], 'mean_diff', signed=True) for k, _ in keys])
    out += ('\nTable 12, third part (pre-specified). Per subfield the grid values that bracket D* '
            'with the weight w of the upper one, and the names minus the control at D* on the '
            "unseen test names; a flagged subfield's post hoc nearest-recall value in "
            'parentheses.\n\n' + markdown(['subfield', 'D* between'] + [n for _, n in keys], rows))
    curves = (matched.get('post_hoc_curves') or {}).get('units')
    if curves:
        rows = []
        for u, v in curves.items():
            for label, key in (('held', 'recall_in'), ('not held', 'recall_out')):
                values = [v['control'][g][key] for g in grid]
                rows.append([u, label, number(v['names'][key])] + [number(x) for x in values]
                            + [number(v['names'][key] - max(values), True)])
        best = (matched.get('post_hoc_curves') or {}).get('names_minus_best') or {}
        for label, key in (('held', 'recall_in'), ('not held', 'recall_out')):
            if key in best:
                rows.append(['mean [95% CI]', label, ''] + [''] * len(grid)
                            + [interval(best[key], 'mean_diff', signed=True)
                               + f", {best[key]['n_positive']}/{best[key]['n_units']} above 0"])
        out += ("\nTable 12, fourth part (post hoc). The control's recall on the unseen test names "
                'the lexicon holds (held) and on the others (not held), pooled over the seeds, at '
                "every bias, beside the names'; the last column is the names' recall minus the "
                "control's highest over the grid.\n\n"
                + markdown(['subfield', 'unseen names', 'names'] + dcols
                           + ['names − the control at its best bias'], rows))
    return out + table12_post_hoc(matched)


def extra_names(types, key):
    """The names MR6 + names recalls beyond MR6 at D = 0 in a type cell of post_hoc_types, per
    seed summed over the subfields (each subfield's recall gain times its names in the cell),
    with the cell's share of its part's total where no cell of the part loses names ('' without
    the gains)."""
    gains, gold = types.get('minus_mr6_at_0') or {}, types.get('gold_per_run') or {}

    def extra(k):
        e = gains.get(k)
        if not e or 'per_unit' not in e:
            return None
        return sum(e['per_unit'][u] / 100 * gold[u][k] for u in e['per_unit'] if u in gold)
    mine = extra(key)
    if mine is None:
        return ''
    part = key.rsplit('_', 1)[1]
    parts = [x for x in (extra(k) for k in gains if k.rsplit('_', 1)[1] == part) if x is not None]
    count = f'{mine:+.0f}'.replace('-', '\u2212')
    # a share only where every cell of the part adds names, so that the shares are parts of a whole
    if parts and min(parts) >= 0 and sum(parts) > 0:
        return f'{count} ({100 * mine / sum(parts):.0f}%)'
    return count


def table12_post_hoc(matched):
    """Table 12's post hoc parts added after paper A's round-11 review: the decomposition of 18.1,
    the held / not-held split within type groups, the sign-flip p-values and the rebuilt
    lexicons' composition by type ('' without them)."""
    out = ''
    dec = matched.get('post_hoc_decomposition')
    if dec:
        rows = [['all 7 subfields', interval(dec['names_minus_mr6_at_0'], 'mean_diff', signed=True),
                 interval(dec['mr6_at_0_minus_control_at_d_star'], 'mean_diff', signed=True)]]
        m = dec.get('matched_only')
        if m:
            rows.append([f"the {m['names_minus_mr6_at_0']['n_units']} matched subfields",
                         interval(m['names_minus_mr6_at_0'], 'mean_diff', signed=True),
                         interval(m['mr6_at_0_minus_control_at_d_star'], 'mean_diff', signed=True)])
        out += ("\nTable 12, fifth part (post hoc). 18.1 per subfield is the sum of the names' gain "
                "over MR6 at D = 0 on the unseen names the lexicon holds and MR6's own loss on them "
                'from D = 0 to D*.\n\n'
                + markdown(['subfields', 'names − MR6 at D = 0', 'MR6 at D = 0 − the control at D*'],
                           rows))
    types = matched.get('post_hoc_types')
    if types:
        rows = []
        for key in types['minus_control_at_d_star']:
            group, part = key.rsplit('_', 1)
            cells = [group, 'held' if part == 'in' else 'not held']
            for block in ('minus_control_at_d_star', 'minus_control_nearest', 'minus_mr6_at_0'):
                e = types[block].get(key)
                cells.append('' if e is None else interval(e, 'mean_diff', signed=True)
                             + f", {e['n_positive']}/{e['n_units']}")
            gold = [v[key] for v in types['gold_per_run'].values()]
            cells.append(f'{min(gold):.0f}–{max(gold):.0f}')
            cells.append(extra_names(types, key))
            rows.append(cells)
        out += ('\nTable 12, sixth part (post hoc). Recall on the unseen test names by type group and '
                'lexicon membership: the names minus the control at D*, at the nearest grid value for '
                'the flagged subfields, and minus MR6 at D = 0, paired over the subfields with the '
                'subfields above 0; the unseen test names per subfield in each cell; and the names '
                'MR6 + names recalls beyond MR6 at D = 0 in each cell, per seed summed over the '
                "subfields (each subfield's gain times its names), with the cell's share of its "
                "part's total where every cell of the part adds names.\n\n"
                + markdown(['type', 'unseen names', 'names − control at D*',
                            'the flagged at the nearest recall', 'names − MR6 at D = 0',
                            'test names per subfield', 'names recalled beyond MR6 at D = 0, per seed'],
                           rows))
    flips = matched.get('post_hoc_sign_flip')
    if flips:
        out += ('\nExact two-sided sign-flip p over the 7 subfields (post hoc): 18.1 '
                f"{flips['18.1']}, 18.2 {flips['18.2']}.\n")
    lex = matched.get('post_hoc_lexicon_types')
    if lex:
        kinds = sorted({k for v in lex.values() for k in v},
                       key=lambda k: -sum(v.get(k, {}).get('distinct', 0) for v in lex.values()))
        rows = [[u] + [f"{v[k]['distinct']:,} ({v[k]['entries']:,})" if k in v else ''
                       for k in kinds] for u, v in lex.items()]
        out += ('\nTable 12, seventh part. The rebuilt lexicons by type: distinct names and, in '
                'parentheses, their entries (occurrences in the corpus, which set how often a name is '
                'drawn).\n\n' + markdown(['subfield'] + kinds, rows))
    return out


# the systems of each comparison of ner_lexicon_recall.py ('names': the setting's names system)
LEXICON_PAIRS = {'names_vs_mr6': ('names', 'crf_mr6'), 'names_vs_crf': ('names', 'crf'),
                 'mr6_vs_crf': ('crf_mr6', 'crf'), 'mr3_vs_crf': ('crf_mr', 'crf'),
                 'mixed_vs_mr6': ('crf_mrm_mixed6', 'crf_mr6')}
LEXICON_NAMES = {'mulms': 'crf_mrm_topic6', 'fine': 'crf_mrm6', 'coarse': 'crf_mrm6'}


def f1_of(recall, precision):
    """F1 from a recall and a precision in percent (None if either is missing or both are 0)."""
    if recall is None or precision is None or recall + precision == 0:
        return None
    return 2 * recall * precision / (recall + precision)


def part_f1(e, part):
    """F1 in percent on one lexicon part ('in' or 'out') of a ner_lexicon_recall.json score
    entry: from its counts, 2 correct / (gold + predicted), where it has them (exact), else
    from its rounded recall and precision."""
    counts = [(e.get(k) or {}).get(part) for k in ('correct', 'gold', 'pred')]
    if all(c is not None for c in counts):
        correct, gold, pred = counts
        return 200 * correct / (gold + pred) if gold + pred else None
    return f1_of(e.get(f'recall_{part}'), e.get(f'precision_{part}'))


def held_f1_gain(part, setting, key):
    """Post hoc (round 15): the paired gain in F1 on the unseen test names the lexicon holds,
    each unit's F1 on them pooled over the seeds (part_f1, from the counts), the method
    minus its reference, with the 95% Student-t interval over the units; None without the
    per-unit scores."""
    import statistics
    from scipy import stats
    method, reference = LEXICON_PAIRS[key]
    method = LEXICON_NAMES.get(setting, method) if method == 'names' else method
    per_unit = part.get('per_unit') or {}
    if method not in per_unit or reference not in per_unit:
        return None
    diffs = []
    for unit in part['units']:
        a, b = per_unit[method].get(unit), per_unit[reference].get(unit)
        fa = a and part_f1(a, 'in')
        fb = b and part_f1(b, 'in')
        if fa is None or fb is None:
            return None
        diffs.append(fa - fb)
    n = len(diffs)
    if n < 2:
        return None
    mean = statistics.fmean(diffs)
    half = float(stats.t.ppf(0.975, n - 1)) * statistics.stdev(diffs) / n ** 0.5
    return {'mean_diff': round(mean, 2), 'ci95': [round(mean - half, 2), round(mean + half, 2)],
            'n_units': n}


def table10(data):
    """Table 10 from ner_lexicon_recall.json: per setting and system, recall on the unseen test
    names in and out of the rebuilt lexicon, the unseen predictions' lexicon share, precision
    and F1 on each part, pooled over seeds and units; then the paired gains per part, their
    difference and the paired F1 gain on the held names (post hoc)."""
    scores, _ = lexicon_rows(data)

    def pct(x):
        return '' if x is None else f'{x:.1f}'
    header = ['setting', 'system', 'unseen test names in the lexicon (%)', 'recall, in',
              'recall, out', 'unseen predictions in the lexicon (%)', 'precision, in',
              'precision, out', 'F1, in', 'F1, out']
    cells = [[setting, system, pct(e['gold_share_in']), pct(e['recall_in']), pct(e['recall_out']),
              pct(e['pred_share_in']), pct(e['precision_in']), pct(e['precision_out']),
              pct(part_f1(e, 'in')), pct(part_f1(e, 'out'))]
             for setting, system, e in scores]
    cheader = ['setting', 'comparison', 'recall gain on the names in the lexicon [95% CI]',
               'recall gain on the names not in it [95% CI]', 'in − out [95% CI]',
               'units with in > out', 'F1 gain on the names in the lexicon [95% CI]']
    ccells = []
    for key, label in LEXICON_SETTINGS.items():
        part = data.get(key)
        if not part:
            continue
        for ckey, name in LEXICON_COMPARISONS.items():
            c = part['comparisons'].get(ckey)
            if not c:
                continue
            gain = held_f1_gain(part, key, ckey)
            ccells.append([label, name, interval(c['gain_in'], 'mean_diff', signed=True),
                           interval(c['gain_out'], 'mean_diff', signed=True),
                           interval(c['in_minus_out'], 'mean_diff', signed=True),
                           f"{c['in_minus_out']['n_positive']}/{c['in_minus_out']['n_units']}",
                           interval(gain, 'mean_diff', signed=True) if gain else ''])
    units = {k: ', '.join(v['units']) for k, v in data.items() if v}
    excluded = {k: v['excluded'] for k, v in data.items() if v and v['excluded']}
    return ('Table 10. Recall on the unseen test names by lexicon membership (week 15, post hoc; '
            'ner_lexicon_recall.py from the saved predictions and the lexicons rebuilt by the '
            "names method's own teachers, batch w15a): per system, pooled over seeds and held-out "
            'units, recall on the unseen test names that the unit\'s lexicon holds (in) and on '
            'those it does not (out), and among the predicted entities whose name is unseen the '
            'share that are lexicon names and the precision of each part, with the F1 of each '
            'part from its pooled counts (round 15); then, paired over the '
            "units, a method's recall gain over its reference on each part and the difference of "
            'the two gains with its 95% Student-t CI, and the F1 gain on the held names, each '
            "unit's F1 from its own pooled counts on them (round 15). Units used: "
            + '; '.join(f'{LEXICON_SETTINGS[k]}: {u}' for k, u in units.items())
            + (f". Excluded (rebuilt teachers or coverage differ): {excluded}" if excluded else '')
            + '.\n\n' + markdown(header, cells) + '\n\n**Paired gains**\n\n' + markdown(cheader, ccells))


def table3(published):
    """Table 3: default against strict IOB2 scoring on the published splits."""
    rows = []
    for g, entry in published.items():
        for system, label in (('baseline', 'plain MatSciBERT'), ('crf', 'CRF head')):
            rows.append([SPLIT_NAMES.get(g, g), label, interval(entry[system]['micro']),
                         interval(entry[system]['micro_strict'])])
        d = entry['crf_minus_baseline']
        rows.append([SPLIT_NAMES.get(g, g), 'CRF head − plain',
                     interval(d['micro'], 'mean_diff', signed=True),
                     interval(d['micro_strict'], 'mean_diff', signed=True)])
    return ('Table 3. Default (conlleval) against strict IOB2 scoring, published splits, 10 '
            'model seeds: micro F1 [95% CI].\n\n'
            + markdown(['split', 'system', 'default', 'strict IOB2'], rows))


def table4(mentions, matching):
    """Table 4: unseen names and the matching criteria, CRF head, per setting."""
    rows = []
    for key, label in UNSEEN_SETTINGS:
        names = mentions.get('systems', {}).get(key, {}).get('crf')
        scores = matching.get('systems', {}).get(key, {}).get('crf')
        if not names or not scores:
            continue
        rows.append([label, f"{100 * names['unseen']['share']:.1f}",
                     f"{names['seen']['recall']:.1f}", f"{names['unseen']['recall']:.1f}",
                     *(number(scores[c]['mean']) for c in CRITERIA)])
    header = ['setting', 'unseen names (%)', 'recall, seen', 'recall, unseen',
              *(f'F1, {c}' for c in CRITERIA)]
    return ('Table 4. The CRF head (10 model seeds): the share of gold test entities whose '
            'words are no entity of their type in the training sentences, recall on seen and '
            'unseen names (pooled), and F1 under the matching criteria of SemEval-2013 Task '
            '9.1 (mean over splits, papers, subfields or seeds).\n\n'
            + markdown(header, rows))


def probe_row(report, coverage=None):
    """One row of table 5 from a probe report (silver_lexicon.py) and, if given, the
    subfield's entry of mulms_coverage.json: the numbers as floats, the subfield first."""
    sub = report['subfield']
    held = None
    if coverage:
        own = coverage.get('corpora', {}).get(f'topic_{sub}', {})
        held = own.get('exact')
    before = [v['before'] for v in report['probe']['sentences_0.9']['fill']['per_seed'].values()]
    precision = sum(b['precision'] for b in before) / len(before)
    recall = sum(b['recall'] for b in before) / len(before)
    return [sub, held, report['lexicons']['sentences_0.9']['unseen_share_in_lexicon'],
            report['lexicons']['mentions_0.9']['unseen_share_in_lexicon'], precision, recall,
            report['probe']['sentences_0.9']['fill']['delta_f1'],
            report['probe']['mentions_0.9']['fill']['delta_f1'],
            report['probe']['ceiling_unseen_gold']['fill']['delta_f1'],
            report['probe']['ceiling_unseen_gold']['override']['delta_f1']]


def table5(reports, coverage=None):
    """Table 5: the CPU probe per held-out subfield, with a mean row."""
    rows = [probe_row(r, (coverage or {}).get('subfields', {}).get(r['subfield']))
            for r in sorted(reports, key=lambda r: r['subfield'])]

    def cells(row):
        sub, held, adm_s, adm_m, precision, recall, fill_s, fill_m, gold_fill, gold_over = row
        return [sub, '' if held is None else f'{100 * held:.1f}', f'{100 * adm_s:.1f}',
                f'{100 * adm_m:.1f}', f'{precision:.1f}', f'{recall:.1f}',
                number(fill_s, signed=True), number(fill_m, signed=True),
                number(gold_fill, signed=True), number(gold_over, signed=True)]
    out = [cells(r) for r in rows]
    if len(rows) > 1:
        means = ['mean']
        for k in range(1, len(rows[0])):
            values = [r[k] for r in rows if r[k] is not None]
            means.append(sum(values) / len(values) if values else None)
        out.append(cells(means))
    header = ['subfield', 'corpus holds (%)', 'sentence filter admits (%)',
              'mention filter admits (%)', 'CRF precision', 'CRF recall',
              'Δ F1, sentence lexicon', 'Δ F1, mention lexicon', 'Δ F1, gold names (fill)',
              'Δ F1, gold names (override)']
    return ('Table 5. The CPU probe (no decision): per held-out MuLMS subfield, the share of '
            'the unseen test names that the topic corpus holds and that the two silver '
            'filters admit at threshold 0.9; the held-out CRF head\'s precision and recall '
            '(10 model seeds); and the change of its micro F1 when every verbatim match of a '
            'lexicon name is tagged where it predicted O (fill), for each 0.9 lexicon and for '
            'the gold unseen names themselves, which also override its tags in the last '
            'column.\n\n' + markdown(header, out))


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(description='The tables of the paper plan')
    parser.add_argument('--results', default=str(ROOT / 'results'))
    parser.add_argument('--out', default=str(ROOT / 'docs' / 'tables'))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    written = []

    def write(name, text):
        with open(os.path.join(args.out, name), 'w') as f:
            f.write(text)
        written.append(name)

    analysis = load(os.path.join(args.results, 'gpu', 'ner_week5_seed_analysis.json')) or {}
    published = analysis.get('published_split')
    if published:
        write('table1_published_split.md', table1(published))
        write('table3_strict.md', table3(published))
    else:
        print('tables 1 and 3: no published-split analysis')
    files = [(w, m, s, load(os.path.join(args.results, name)))
             for w, m, s, name in COMPARISONS]
    files = [f for f in files if f[3]]
    if files:
        write('table2_decisions.md', table2(comparison_rows(files, DECIDED)))
        write('tableA1_all_comparisons.md',
              table2(comparison_rows(files), 'Table A1. Every comparison on MaterioMiner'))
    else:
        print('table 2: no comparisons files')
    mentions = load(os.path.join(args.results, 'gpu', 'ner_mentions.json'))
    matching = load(os.path.join(args.results, 'gpu', 'ner_matching.json'))
    if mentions and matching:
        write('table4_unseen.md', table4(mentions, matching))
    else:
        print('table 4: no mention or matching analysis')
    probe_dir = Path(args.results) / 'silver_lexicons'
    reports = [load(str(path)) for path in sorted(probe_dir.glob('*.json'))]
    if reports:
        write('table5_probe.md', table5(reports, load(os.path.join(args.results,
                                                                    'mulms_coverage.json'))))
    else:
        print('table 5: no probe reports')
    rows = held_out_rows(analysis, load(os.path.join(args.results, 'gpu',
                                                     'ner_week12_comparisons.json')))
    if rows:
        write('table6_held_out_decisions.md', table6(rows))
    else:
        print('table 6: no held-out decisions')
    rows = ablation_rows(analysis)
    if rows:
        write('table7_ablations.md', table7(rows))
    else:
        print('table 7: no ablations')
    parts = {part: type_rows(analysis, part) for part in ('mulms', 'coarse')}
    parts = {k: v for k, v in parts.items() if v}
    if parts:
        write('table8_types.md', table8(parts))
        fine = type_rows(analysis, 'fine')
        if fine:
            write('tableA2_types_fine.md', table8({'fine': fine}, 'Table A2'))
    else:
        print('table 8: no per-type analysis of the names method')
    robust = load(os.path.join(args.results, 'gpu', 'ner_robustness.json'))
    if robust and robust.get('pooled'):
        write('table9_robustness.md', table9(robust))
    else:
        print('table 9: no robustness file')
    lexicon = load(os.path.join(args.results, 'gpu', 'ner_lexicon_recall.json'))
    if lexicon:
        write('table10_lexicon.md', table10(lexicon))
        write('table15_lexicon_units.md', table15(lexicon))
    else:
        print('table 10: no lexicon-membership file')
    if mentions and mentions.get('chain'):
        write('table14_chain.md', table14(mentions))
    tests = load(os.path.join(args.results, 'gpu', 'ner_decision_tests.json'))
    if tests:
        write('table16_decision_tests.md', table16(tests))
    else:
        print('table 16: no decision tests file')
    rows = week17_rows(analysis)
    if rows:
        write('table11_week17.md', table11(rows))
    else:
        print('table 11: no week-17 runs')
    matched = load(os.path.join(args.results, 'gpu', 'ner_recall_matched.json'))
    if matched and matched.get('decisions'):
        write('table12_week18.md', table12(matched))
    else:
        print('table 12: no recall-matched control')
    print(f"wrote {', '.join(written) or 'nothing'} to {args.out}")
    return written


if __name__ == '__main__':
    main()
