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
    control at D* in recall on the unseen names the lexicon holds, micro F1 and precision; then
    the two decisions and the pre-specified comparisons over the subfields."""
    header = ['subfield', "names' recall", "control's recall at 0", 'D*',
              'Δ recall, unseen names in the lexicon', 'Δ micro F1', 'Δ precision']
    d, pre = matched['decisions'], matched['prespecified']
    cells = []
    for u, v in matched['per_unit'].items():
        cells.append([u, number(v['target_recall']), number(v['control_recall_at_0']),
                      f"{v['d_star']:.2f}{' (flagged)' if v['flagged'] else ''}",
                      number(d['18.1']['per_unit'][u], True), number(d['18.2']['per_unit'][u], True),
                      number(pre['precision']['per_unit'][u], True)])
    cells.append([f"mean [95% CI], {d['18.1']['n_units']} subfields", '', '', '',
                  interval(d['18.1'], 'mean_diff', signed=True) + f" (18.1, {d['18.1']['verdict']})",
                  interval(d['18.2'], 'mean_diff', signed=True) + f" (18.2, {d['18.2']['verdict']})",
                  interval(pre['precision'], 'mean_diff', signed=True)])
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
            '(decisions 18.1 and 18.2, the week-6 rule; the rest pre-specified). Also, paired: '
            f'{extra}.{readings(matched)}\n\n' + markdown(header, cells))


def readings(matched):
    """The week-18 decisions without the flagged subfields (pre-specified) and with them matched
    at the grid value of the nearest recall (post hoc), as a sentence ('' without flags)."""
    out = ''
    without, nearest = matched.get('without_flagged'), matched.get('post_hoc_nearest')
    if without:
        d = without['decisions']
        out += (f" Without the flagged subfields ({', '.join(without['flagged'])}; pre-specified): "
                f"18.1 {interval(d['18.1'], 'mean_diff', signed=True)}, "
                f"{d['18.1']['n_positive']}/{d['18.1']['n_units']} above 0; "
                f"18.2 {interval(d['18.2'], 'mean_diff', signed=True)}.")
    if nearest:
        d = nearest['decisions']
        at = ', '.join(f"{u} at {v['d']:g} ({number(v['control_recall'])} against "
                       f"{number(v['target_recall'])})" for u, v in nearest['matched'].items())
        out += (' Post hoc, each flagged subfield matched instead at the grid value of the '
                f"control's recall nearest the names' ({at}): "
                f"18.1 {interval(d['18.1'], 'mean_diff', signed=True)}, "
                f"{d['18.1']['n_positive']}/{d['18.1']['n_units']} above 0; "
                f"18.2 {interval(d['18.2'], 'mean_diff', signed=True)}.")
    return out


def table10(data):
    """Table 10 from ner_lexicon_recall.json: per setting and system, recall on the unseen test
    names in and out of the rebuilt lexicon and the unseen predictions' lexicon share and
    precision, pooled over seeds and units; then the paired gains per part and their
    difference."""
    scores, comparisons = lexicon_rows(data)

    def pct(x):
        return '' if x is None else f'{x:.1f}'
    header = ['setting', 'system', 'unseen test names in the lexicon (%)', 'recall, in',
              'recall, out', 'unseen predictions in the lexicon (%)', 'precision, in',
              'precision, out']
    cells = [[setting, system, pct(e['gold_share_in']), pct(e['recall_in']), pct(e['recall_out']),
              pct(e['pred_share_in']), pct(e['precision_in']), pct(e['precision_out'])]
             for setting, system, e in scores]
    cheader = ['setting', 'comparison', 'recall gain on the names in the lexicon [95% CI]',
               'recall gain on the names not in it [95% CI]', 'in − out [95% CI]', 'units with in > out']
    ccells = [[setting, name, interval(c['gain_in'], 'mean_diff', signed=True),
               interval(c['gain_out'], 'mean_diff', signed=True),
               interval(c['in_minus_out'], 'mean_diff', signed=True),
               f"{c['in_minus_out']['n_positive']}/{c['in_minus_out']['n_units']}"]
              for setting, name, c in comparisons]
    units = {k: ', '.join(v['units']) for k, v in data.items() if v}
    excluded = {k: v['excluded'] for k, v in data.items() if v and v['excluded']}
    return ('Table 10. Recall on the unseen test names by lexicon membership (week 15, post hoc; '
            'ner_lexicon_recall.py from the saved predictions and the lexicons rebuilt by the '
            "names method's own teachers, batch w15a): per system, pooled over seeds and held-out "
            'units, recall on the unseen test names that the unit\'s lexicon holds (in) and on '
            'those it does not (out), and among the predicted entities whose name is unseen the '
            'share that are lexicon names and the precision of each part; then, paired over the '
            "units, a method's recall gain over its reference on each part and the difference of "
            'the two gains with its 95% Student-t CI. Units used: '
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
    else:
        print('table 10: no lexicon-membership file')
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
