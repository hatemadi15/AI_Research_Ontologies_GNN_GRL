"""
names_route_record.py - every comparison of the names route in one table (paper A's SI)

No training: reads the stored results in results/gpu and writes
docs/tables/table13_names_route.md, one row per comparison of the protocols of 17 October
20:21 to 1 November (weeks 10 to 14, 17 and 18; ROADMAP §4), in week order: its week and the
commit of the protocol or amendment that named it, the corpus and units, the comparison in
paper A's system names, its status (a decision, by its ID in paper A's Table 2, D1 to D12, or
by week and number; a screen; pre-specified; post hoc), and the mean paired difference of
micro, macro, weighted and strict IOB2 micro F1 with its 95% CI (micro with the units above
0), and of the votes. A second part gives each row's per-unit values and its votes' macro and
weighted F1. Sources, all under results/gpu:
  - ner_week5_seed_analysis.json: mulms (week 10: leave_one_subfield_out, random_split_pool
    and gap; week 11: silver_names; week 12: the screen of every variant of week12 and the
    winner's decision and ten_seeds parts), week13 (consistency, folds, pooled), week14
    (copies, topic, ablations) and week17;
  - ner_week11_comparisons.json and ner_week12_comparisons.json: MaterioMiner's held-out
    papers; their random-split rows (week 5's self-training) are left out, and MR6's rows
    come from week14.copies, which holds the same numbers and MR6's votes;
  - ner_mentions.json (methods): F1 on the seen and on the unseen test names in weeks 10 and
    11, whose protocols name them;
  - ner_robustness.json (pooled): the 11 units' comparisons on both MuLMS seed bases, D5' and
    the post hoc ones;
  - ner_recall_matched.json: week 18's decisions and pre-specified comparisons, the readings
    without the flagged subfields, and the post hoc ones.
A comparison that two sources hold (the same corpus and comparison) is printed once, from the
first; the two must agree. A part whose results are missing is left out; without any row
nothing is written.

Usage:
  python src/graph/names_route_record.py [--results results] [--out docs/tables]
"""

import argparse
import os
from pathlib import Path

from paper_tables import interval, load, markdown, number

ROOT = Path(__file__).resolve().parents[2]
OUT_NAME = 'table13_names_route.md'
LOPO = 'leave_one_paper_out'
GRANULARITIES = ('fine', 'coarse')
TEN = tuple(range(0, 1000, 100))          # the 10 seed offsets
DECISION_SEEDS = TEN[3:]                  # 300 to 900, the seeds a screen did not see
SCREEN_SEEDS = TEN[:3]                    # 0 to 200

# a row's week: the protocol or amendment that named its comparison, with its commit (UTC)
W10 = ('10', '17 October 20:21')
W11 = ('11', '18 October 01:13')
W12 = ('12', '18 October 11:00')
W12_A1 = ('12', 'amendment 1, 19 October 17:15')
W12_A2 = ('12', 'amendment 2, 19 October 21:03')
W12_A3 = ('12', 'amendment 3, 20 October 01:50')
W12_ANALOGUE = ('12', 'the analogue batches, 20 October 00:51')
W13 = ('13', '20 October 09:14')
W13_AMENDED = ('13', 'amendment, 20 October 09:25')
W13_4 = ('13', 'amendment 13.4, 20 October 13:53')
W14 = ('14', '29 October 12:07')
W17 = ('17', '31 October 19:15')
W18 = ('18', '1 November 10:51')

# paper A's system names
SYSTEMS = {'crf': 'CRF head', 'crf_mr': 'MR3', 'crf_mr6': 'MR6', 'crf_mrm_topic6': 'MR6 + names',
           'crf_mrm6': 'MR6 + names', 'crf_mt': 'multi-task', 'crf_mt_mr': 'multi-task + MR3',
           'crf_mt_mr6': 'multi-task + MR6', 'crf_mt_mrm6': 'multi-task + MR6 + names',
           'crf_mrm_mixed6': 'the mixed-corpus variant'}
OTHER_SUBFIELDS = "names from the other subfields' sentences"    # week 11, MuLMS (crf_mrs)
FATIGUE_SENTENCES = "names from the fatigue corpus's sentences"  # week 11, MaterioMiner
V6_ANALOGUE = "self-training on the fatigue corpus's silver sentences (V6's analogue)"
# the week-12 screen in the order its variants joined it (Supplementary Table S5's labels)
SCREEN = (('crf_mrs_topic', 'MR3 + names, sentence filter (V1)', W12),
          ('crf_mrm_topic', 'MR3 + names, mention filter (V2)', W12),
          ('crf_mr6', 'MR6 (V3)', W12),
          ('crf_mrm_topic6', 'MR6 + names (V4)', W12_A1),
          ('crf_st3_topic', "self-training on the corpus's silver sentences (V6)", W12_A1),
          ('crf_gzm_topic', 'lexicon as a gazetteer feature, mention filter (V7)', W12_A2),
          ('crf_gzs_topic', 'lexicon as a gazetteer feature, sentence filter (V8)', W12_A3))
WINNER = 'crf_mrm_topic6'
# week 14, 14.3: MR6 + names with one setting changed (Supplementary Table S2's labels)
ABLATIONS = {'crf_mrm_topic10k6': 'MR6 + names, corpus: its first 10,000 sentences',
             'crf_mrm_topic20k6': 'MR6 + names, corpus: its first 20,000 sentences',
             'crf_mrm_topiclow6': 'MR6 + names, filter: marginal threshold 0.7',
             'crf_mrm_topictwo6': 'MR6 + names, filter: 2 of 3 teachers agree',
             'crf_mrm_topiccap6': 'MR6 + names, pool: harvested names in half of the draws'}
# week 13: the label-propagation rule's inputs; the pooled estimates' corpora
PROPAGATED = {'crf': 'CRF head', 'crf_mr': 'MR3', 'crf_mt_mr': 'multi-task + MR3'}
CORPORA = {'mulms': 'MuLMS', 'fine': 'MaterioMiner fine', 'coarse': 'MaterioMiner coarse',
           'sofc_folds': 'SOFC-Exp'}
# ner_robustness.json's pooled comparisons: their sides and weeks; D5 and D7 come from the
# seed analysis first and are printed from there
ROBUST = {'names_vs_crf': ('MR6 + names − CRF head', W13),
          'names_vs_mr3': ('MR6 + names − MR3', W13),
          'mr3_vs_crf': ('MR3 − CRF head', W13),
          'names_vs_mr6': ('MR6 + names − MR6', W14),
          'mr6_vs_crf': ('MR6 − CRF head', W14),
          'mr6_vs_mr3': ('MR6 − MR3', W14)}
# week 18 (ner_recall_matched.json): the pre-specified comparisons at the matched bias
MATCHED = 'MR6 + names − MR6 at matched recall'
HELD = 'the unseen test names the lexicon holds'
WEEK18 = (('18.1', 'D11', f'{MATCHED}: recall on {HELD}'),
          ('18.2', 'D12', f'{MATCHED}: micro F1'))
WEEK18_PRESPECIFIED = (
    ('precision', f'{MATCHED}: precision on all test entities'),
    ('unseen_recall', f'{MATCHED}: recall on the unseen test names'),
    ('unseen_precision', f'{MATCHED}: precision on the unseen test names'),
    ('unseen_f1', f'{MATCHED}: F1 on the unseen test names'),
    ('recall_out', f"{MATCHED}: recall on the unseen test names the lexicon does not hold"),
    ('in_minus_out', f'{MATCHED}: the recall difference on the names the lexicon holds minus '
                     'that on the others'),
    ('bias_f1_change', 'MR6 at matched recall − MR6 at bias 0: micro F1'))
WEEK18_DECOMPOSITION = (
    ('names_minus_mr6_at_0', f'MR6 + names − MR6 at bias 0: recall on {HELD}'),
    ('mr6_at_0_minus_control_at_d_star', f'MR6 at bias 0 − MR6 at matched recall: recall on '
                                         f'{HELD}'))

HEADER = ['#', 'week (protocol, UTC)', 'corpus, units (MuLMS seeds)',
          'comparison (system − reference)', 'status', 'Δ micro F1 [95% CI], units above 0',
          'Δ macro F1 [95% CI]', 'Δ weighted F1 [95% CI]', 'Δ strict IOB2 micro F1 [95% CI]',
          'votes: Δ micro F1 [95% CI], units above 0']
DETAIL_HEADER = ['#', 'corpus, units (MuLMS seeds)', 'comparison', 'per unit', 'votes per unit',
                 'votes: Δ macro F1 [95% CI]', 'votes: Δ weighted F1 [95% CI]']


# --------------------------------------------------------------------------------
# Rows and cells
# --------------------------------------------------------------------------------
def row(week, corpus, comparison, status, diff, votes=None, units=None, vote_units=None):
    """One row: `diff` the comparison under the F1 averages ({'micro': entry, 'macro': ...,
    'weighted': ..., 'micro_strict': ...}) or a single entry, for the micro column alone;
    `votes` the votes' comparison in the same form; `units` the units in the order of the
    entries' per_run (`vote_units` the votes' where it differs)."""
    units = list(units or ())
    return {'week': week, 'corpus': corpus, 'comparison': comparison, 'status': status,
            'diff': averages(diff), 'votes': averages(votes), 'units': units,
            'vote_units': units if vote_units is None else list(vote_units)}


def averages(entry):
    """{average: entry} of a comparison under the F1 averages, or of a single entry (a
    difference, a gap or a level) as its micro column; {} without one."""
    if not entry:
        return {}
    if 'ci95' in entry or 'gap' in entry or 'level' in entry:
        return {'micro': entry}
    return entry


def verdict(passes):
    return 'passes' if passes else 'no'


def system_name(variant):
    """Paper A's name of a system, else its screen label, else the variant itself."""
    return SYSTEMS.get(variant) or {v: label for v, label, _ in SCREEN}.get(variant, variant)


def above_0(entry):
    """True if the 95% CI of a difference lies above 0."""
    return entry['ci95'][0] > 0


def unit_values(entry, units=()):
    """[(unit, value)] of a difference's per-unit values (its per_run with `units`, or its
    per_unit), [] without them or when the units do not match them."""
    if isinstance(entry.get('per_unit'), dict):
        return list(entry['per_unit'].items())
    values = entry.get('per_run')
    if values is None or len(values) != len(units):
        return []
    return list(zip(units, values))


def positives(entry, units=()):
    """'k/n', the units whose difference lies above 0; '' without per-unit values or counts."""
    values = [v for _, v in unit_values(entry, units)] or entry.get('per_run')
    if values:
        return f'{sum(v > 0 for v in values)}/{len(values)}'
    if 'n_positive' in entry and 'n_units' in entry:
        return f"{entry['n_positive']}/{entry['n_units']}"
    return ''


def cell(entry, units=None):
    """A cell: the mean difference with its 95% CI (and, given `units`, the units above 0), a
    difference without interval (a gap, not paired) or a level; '' without an entry."""
    if not entry:
        return ''
    if 'gap' in entry:
        return number(entry['gap'], signed=True)
    if 'level' in entry:
        return f"{interval(entry['level'])} (its level)"
    text = interval(entry, 'mean_diff', signed=True)
    count = positives(entry, units) if units is not None else ''
    return f'{text}, {count}' if count else text


def unit_name(unit):
    """A unit's short name: a MaterioMiner paper's identifier (s11340 for
    10.1007_s11340-021-00758-x.conll), a MuLMS subfield or SOFC-Exp fold as stored."""
    if unit.endswith('.conll'):
        return unit[:-len('.conll')].partition('_')[2].split('-')[0]
    return unit


def unit_order(pair):
    """Sort key of a (unit, value): MuLMS's subfields, then MaterioMiner's papers, then
    SOFC-Exp's folds, each in the order of their stored names."""
    unit = pair[0]
    return (unit.endswith('.conll') + 2 * unit.startswith('fold'), unit)


def per_unit(entry, units=()):
    """'unit +x, ...' of a difference's per-unit values, in unit_order; '' without them."""
    return ', '.join(f'{unit_name(u)} {number(v, signed=True)}'
                     for u, v in sorted(unit_values(entry or {}, units), key=unit_order))


def same(a, b):
    """True if two cells' entries hold the same mean (or gap) and CI."""
    def numbers(e):
        e = (e or {}).get('level') or e or {}
        return e.get('mean_diff', e.get('mean', e.get('gap'))), tuple(e.get('ci95') or ())
    return numbers(a) == numbers(b)


def dedupe(rows):
    """The rows without the repeats of a (corpus, comparison) already printed; a repeat must
    hold the first one's micro numbers (ValueError otherwise: two comparisons share a label)."""
    first, out = {}, []
    for r in rows:
        key = (r['corpus'], r['comparison'])
        if key in first:
            if not same(first[key]['diff'].get('micro'), r['diff'].get('micro')):
                raise ValueError(f'two different comparisons are labelled {key}')
            continue
        first[key] = r
        out.append(r)
    return out


# --------------------------------------------------------------------------------
# Corpus labels
# --------------------------------------------------------------------------------
def span(seeds):
    return f'{seeds[0]}–{seeds[-1]}'


def mulms(n, seeds=TEN):
    return f'MuLMS, {n} subfields (seeds {span(seeds)})'


def papers(granularity, n=4):
    return f'MaterioMiner {granularity}, {n} papers'


def folds(n=5):
    return f'SOFC-Exp, {n} folds'


def pooled_label(counts, mulms_seeds=TEN):
    """'pooled, n units: MuLMS 7 (0–900) + MaterioMiner fine 4 + ...' from [(corpus, units)]."""
    parts = [f'{CORPORA.get(c, c)} {n}' + (f' ({span(mulms_seeds)})' if c == 'mulms' else '')
             for c, n in counts]
    return f"pooled, {sum(n for _, n in counts)} units: {' + '.join(parts)}"


# --------------------------------------------------------------------------------
# The weeks
# --------------------------------------------------------------------------------
def kinds_rows(week, mentions, setting, system, corpus, name):
    """F1 on the unseen and on the seen test names (ner_mentions.json, methods): `system`
    against the CRF head over the same 10 seeds, its votes against the CRF votes;
    pre-specified (weeks 10 and 11)."""
    methods = ((mentions or {}).get('methods') or {}).get(setting) or {}
    e, votes = methods.get(system), methods.get(f'{system}_ens3')
    if not e or e.get('reference') != 'crf' or tuple(e.get('seeds') or ()) != TEN:
        return []
    if votes and votes.get('reference') != 'crf_ens3':
        votes = None
    return [row(week, corpus, f'{name} − CRF head: F1 on the {kind} test names', 'pre-specified',
                e[kind], (votes or {}).get(kind)) for kind in ('unseen', 'seen') if kind in e]


def week10(analysis, mentions):
    """Week 10: MR3 against the CRF head on MuLMS's held-out subfields (decision 10.1, its
    votes decision 10.2), the CRF head's gap between random pool splits and held-out
    subfields, and F1 on the unseen and seen test names (pre-specified)."""
    m = analysis.get('mulms') or {}
    held = m.get('leave_one_subfield_out')
    if not held:
        return []
    votes = held.get('votes') or {}
    status = f"decision 10.1: {verdict(held['passes'])}"
    if votes:
        status += f"; votes: decision 10.2: {verdict(votes['passes'])}"
    units = list(held['subfields'])
    rows = [row(W10, mulms(held['n_subfields']), 'MR3 − CRF head', status,
                held['crf_mr_minus_crf'], votes.get('crf_mr_minus_crf'), units)]
    pool, gap = m.get('random_split_pool'), m.get('gap')
    if pool and gap:
        rows.append(row(W10, f"MuLMS, {pool['n_splits']} random pool splits against the "
                             f"{held['n_subfields']} subfields (seeds 0–900; not paired)",
                        'CRF head on the random pool splits − CRF head on the held-out subfields',
                        'pre-specified', {avg: {'gap': v} for avg, v in gap.items()}))
    return rows + kinds_rows(W10, mentions, 'mulms_leave_one_subfield_out', 'crf_mr',
                             mulms(held['n_subfields']), 'MR3')


def find(comparisons, granularity, variant, reference):
    """The comparison of `variant` with `reference` on the held-out papers at `granularity`
    in a comparisons file, or None."""
    for e in (comparisons or {}).values():
        if (e.get('variant'), e.get('reference'), e.get('granularity'),
                e.get('protocol')) == (variant, reference, granularity, LOPO):
            return e
    return None


def paper_row(week, granularity, e, comparison, status, votes=None):
    """A row from a comparisons file's entry (and its votes' entry), paired by paper."""
    return row(week, papers(granularity, e.get('n_pairs', len(e['run_ids']))), comparison, status,
               e, votes, e['run_ids'], (votes or {}).get('run_ids'))


def names_part_rows(week, part, name, comparisons):
    """Rows of a names variant's entry of the seed analysis (mulms.silver_names, a part of
    mulms.week12: its subfields, minus_<reference>, votes and seeds, 0 to 900 if not given):
    `name` against each reference of `comparisons`, [(reference, status)]."""
    votes = part.get('votes') or {}
    corpus = mulms(part['n_subfields'], part.get('seeds', TEN))
    return [row(week, corpus, f'{name} − {SYSTEMS[reference]}', status,
                part[f'minus_{reference}'], votes.get(f'minus_{reference}'), part['subfields'])
            for reference, status in comparisons if f'minus_{reference}' in part]


def week11(analysis, comparisons, mentions):
    """Week 11: the names of unlabeled sentences in MR3's pool. MuLMS (the other subfields'
    sentences): decisions 11.1 and 11.2 against the CRF head and MR3; MaterioMiner (the
    fatigue corpus): decisions 11.3 and 11.4 against MR3; reported, against the CRF head,
    the multi-task votes, and F1 on the unseen and seen test names."""
    rows = []
    names = (analysis.get('mulms') or {}).get('silver_names')
    if names:
        rows += names_part_rows(W11, names, OTHER_SUBFIELDS,
                                (('crf', f"decision 11.1: {verdict(names['decision_1'])}"),
                                 ('crf_mr', f"decision 11.2: {verdict(names['decision_2'])}")))
    for g, k in zip(GRANULARITIES, (3, 4)):
        e = find(comparisons, g, 'crf_mrs', 'crf_mr')
        if e:
            rows.append(paper_row(W11, g, e, f'{FATIGUE_SENTENCES} − MR3',
                                  f"decision 11.{k}: {verdict(above_0(e['micro']))}",
                                  find(comparisons, g, 'crf_mrs_ens3', 'crf_mr_ens3')))
    for g in GRANULARITIES:
        e = find(comparisons, g, 'crf_mrs', 'crf')
        if e:
            rows.append(paper_row(W11, g, e, f'{FATIGUE_SENTENCES} − CRF head', 'pre-specified',
                                  find(comparisons, g, 'crf_mrs_ens3', 'crf_ens3')))
    for g in GRANULARITIES:
        e = find(comparisons, g, 'crf_mrs_ens3', 'crf_mt_ens3')
        if e:
            rows.append(paper_row(W11, g, e, f'the votes: {FATIGUE_SENTENCES} − multi-task',
                                  'pre-specified'))
    if names:
        rows += kinds_rows(W11, mentions, 'mulms_leave_one_subfield_out', 'crf_mrs',
                           mulms(names['n_subfields']), OTHER_SUBFIELDS)
    for g in GRANULARITIES:
        rows += kinds_rows(W11, mentions, f'{g}_{LOPO}', 'crf_mrs', papers(g), FATIGUE_SENTENCES)
    return rows


def week12(analysis, comparisons):
    """Week 12: names from a topic corpus. The screen of seven variants on MuLMS at seeds 0
    to 200 (no decision; against MR3 its selection statistic); the winner, MR6 + names, on
    the seeds 300 to 900 (D1, D2) and on all 10 (pre-specified); its MaterioMiner analogue
    against the CRF head (D3, D4) and MR3 (pre-specified); V6's analogue, the companion
    study's self-training (week 5's decision against the CRF head; against MR3 post hoc)."""
    rows = []
    week12 = (analysis.get('mulms') or {}).get('week12') or {}
    winner = (analysis.get('week13') or {}).get('week12_winner') or WINNER
    d5 = ((analysis.get('week13') or {}).get('pooled') or {}).get(winner) or {}
    d5 = d5.get('corpora', {}) if d5.get('status') == 'decision' else {}

    def d5_part(corpus, entry, name):
        """'; D5's <name> part' if the pooled decision's corpus holds this difference."""
        found = corpus in d5 and same(d5[corpus].get('minus_crf'), entry)
        return f"; D5's {name} part" if found else ''
    for variant, label, week in SCREEN:
        screen = (week12.get(variant) or {}).get('screen')
        if not screen:
            continue
        chosen = '; selected' if variant == winner else ''
        rows += names_part_rows(week, screen, label,
                                (('crf_mr', f'screen, its statistic (no decision){chosen}'),
                                 ('crf', 'screen (no decision)')))
    parts = week12.get(winner) or {}
    decision = parts.get('decision')
    if decision and 'decision_1' in decision:
        rows += names_part_rows(W12, decision, system_name(winner),
                                (('crf', f"D1: {verdict(decision['decision_1'])}"),
                                 ('crf_mr', f"D2: {verdict(decision['decision_2'])}")))
    ten = parts.get('ten_seeds')
    if ten and 'minus_crf' in ten:
        part = d5_part('mulms', ten['minus_crf']['micro'], 'MuLMS')
        rows += names_part_rows(W12, ten, system_name(winner),
                                (('crf', f'pre-specified{part}'), ('crf_mr', 'pre-specified')))
    for g, d in zip(GRANULARITIES, ('D3', 'D4')):
        e = find(comparisons, g, 'crf_mrm6', 'crf')
        if e:
            rows.append(paper_row(W12, g, e, 'MR6 + names − CRF head',
                                  f"{d}: {verdict(above_0(e['micro']))}"
                                  + d5_part(g, e['micro'], 'MaterioMiner'),
                                  find(comparisons, g, 'crf_mrm6_ens3', 'crf_ens3')))
    for g in GRANULARITIES:
        e = find(comparisons, g, 'crf_mrm6', 'crf_mr')
        if e:
            rows.append(paper_row(W12_ANALOGUE, g, e, 'MR6 + names − MR3', 'pre-specified',
                                  find(comparisons, g, 'crf_mrm6_ens3', 'crf_mr_ens3')))
    for g in GRANULARITIES:
        e = find(comparisons, g, 'crf_st3', 'crf')
        if e:
            votes = find(comparisons, g, 'crf_st3_ens3', 'crf_ens3')
            status = ('reported again; the companion study\'s week-5 decision (15 October): '
                      + verdict(above_0(e['micro'])))
            if votes:
                status += f"; votes: {verdict(above_0(votes['micro']))}"
            rows.append(paper_row(W12_ANALOGUE, g, e, f'{V6_ANALOGUE} − CRF head', status, votes))
    for g in GRANULARITIES:
        e = find(comparisons, g, 'crf_st3', 'crf_mr')
        if e:
            rows.append(paper_row(W12_ANALOGUE, g, e, f'{V6_ANALOGUE} − MR3', 'post hoc',
                                  find(comparisons, g, 'crf_st3_ens3', 'crf_mr_ens3')))
    return rows


def consistency_status(corpus, system, e):
    """The status of a label-propagation comparison of week13.consistency: on the CRF head on
    MaterioMiner's papers the companion study's week-7 comparisons (the rule a decision,
    retyping reported), recomputed; the rest week 13's (13.1c on SOFC-Exp's folds)."""
    if e.get('decision') == 'week 7':
        return ("the companion study's week-7 decision (16 October), recomputed: "
                + verdict(e['passes']))
    if 'decision' in e:
        return f"decision {e['decision']}: {verdict(e['passes'])}"
    if system == 'crf' and corpus in GRANULARITIES:
        return "the companion study's week-7 comparison (16 October; reported), recomputed"
    return 'pre-specified (13.1c)' if corpus == 'sofc_folds' else 'pre-specified'


def corpus_label(corpus, n):
    return {'mulms': lambda: mulms(n), 'sofc_folds': lambda: folds(n)}.get(
        corpus, lambda: papers(corpus, n))()


def pooled_rows(entry, label, status, week):
    """The rows of a week13.pooled entry: the pooled comparison over its corpora's units
    (10-seed means), and its with_<corpus> extensions (SOFC-Exp's folds: 13.4's three-corpus
    estimate, pre-specified; any other post hoc)."""
    counts = [(c, v['n_units']) for c, v in entry['corpora'].items()]
    units = [u for v in entry['corpora'].values() for u in v.get('units', {})]
    rows = [row(week, pooled_label(counts), label, status, entry['minus_crf'], units=units)]
    for key, ext in entry.items():
        if key.startswith('with_') and isinstance(ext, dict) and 'minus_crf' in ext:
            corpus = key[len('with_'):]
            more = [(c, v['n_units']) for c, v in ext['corpora'].items()]
            extra = list((ext.get(corpus) or {}).get('units', {}))
            folds_13_4 = corpus == 'sofc_folds'
            rows.append(row(W13_4 if folds_13_4 else week, pooled_label(more), label,
                            'pre-specified (13.4)' if folds_13_4 else 'post hoc',
                            ext['minus_crf'], units=units + extra))
    return rows


def robust_rows(robust, keys):
    """ner_robustness.json's pooled comparisons of `keys` over the 11 units of D5 on both MuLMS
    seed bases: post hoc but for D5 (the 10-seed basis), D5' (D5 on the decision seeds) and
    D7 (MR6 + names − MR6 on the decision seeds)."""
    rows = []
    for basis, seeds in (('ten', TEN), ('decision', DECISION_SEEDS)):
        entries = (robust or {}).get('pooled', {}).get(basis) or {}
        units = entries.get('units') or []
        n_mulms = sum(not u.endswith('.conll') for u in units)
        label = pooled_label([('mulms', n_mulms), ('fine', len(units) - n_mulms)], seeds)
        for key in keys:
            e = entries.get(key)
            if not e:
                continue
            name, week = ROBUST[key]
            status = 'post hoc'
            if key == 'names_vs_crf':
                status = ('D5' if basis == 'ten' else
                          f"D5′, D5 on the decision seeds (post hoc reading): "
                          f"{'above 0' if above_0(e) else 'not above 0'}")
            elif key == 'names_vs_mr6' and basis == 'decision':
                status = f'D7: {verdict(above_0(e))}'
            rows.append(row(week, label, name, status, e, units=units))
    return rows


def with_measures(rows, robust):
    """The rows, the 11-unit pooled comparisons given the macro and weighted F1 of
    ner_robustness.json's pooled_measures (post hoc, recomputed from the run files) where a row
    has none; `rows` are changed in place and returned."""
    measures = (robust or {}).get('pooled_measures') or {}
    keys = {name: key for key, (name, _) in ROBUST.items()}
    for basis, seeds in (('ten', TEN), ('decision', DECISION_SEEDS)):
        entries = measures.get(basis) or {}
        units = entries.get('units') or []
        n_mulms = sum(not u.endswith('.conll') for u in units)
        corpus = pooled_label([('mulms', n_mulms), ('fine', len(units) - n_mulms)], seeds)
        for r in rows:
            e = entries.get(keys.get(r['comparison'])) if r['corpus'] == corpus else None
            for avg in ('macro', 'weighted'):
                if e and avg in e and avg not in r['diff']:
                    r['diff'][avg] = e[avg]
    return rows


def week13(analysis, robust):
    """Week 13: the label-propagation rule (13.1a to 13.1d and the reported applications),
    SOFC-Exp's article folds (13.3a, 13.3b, D6 and their reported comparisons), and the pooled
    comparisons (D5, the pre-specified three-corpus estimates, the post hoc ones)."""
    w = analysis.get('week13') or {}
    rows = []
    for corpus, entries in (w.get('consistency') or {}).items():
        for variant, e in entries.items():
            system, _, rule = variant.rpartition('_')
            name = PROPAGATED.get(system, system)
            retyping = ' with retyping' if rule == 'dcr' else ''
            rows.append(row(W13 if corpus == 'mulms' else W13_AMENDED,
                            corpus_label(corpus, e['n_units']),
                            f'{name} + label propagation{retyping} − {name}',
                            consistency_status(corpus, system, e), e['minus_input'],
                            units=e['units']))
    for method, e in (w.get('folds') or {}).items():
        units, d6 = list(e['units']), e.get('decision') == '13.4'
        name = SYSTEMS.get(method, method)
        status = 'D6' if d6 else f"decision {e['decision']}"
        week = W13_4 if d6 else W13
        rows.append(row(week, folds(e['n_units']), f'{name} − CRF head',
                        f"{status}: {verdict(e['passes'])}", e['minus_crf'], units=units))
        if 'minus_crf_mr' in e:
            rows.append(row(week, folds(e['n_units']), f'{name} − MR3', 'pre-specified',
                            e['minus_crf_mr'], units=units))
        if 'slot' in e:
            slot = e['slot']
            rows.append(row(week, f"{folds(slot.get('n_units', e['n_units']))}, slot level",
                            f'{name}, its slot level (no reference at that level)',
                            'pre-specified', {avg: {'level': slot[avg]} for avg in
                                              ('micro', 'macro', 'weighted', 'micro_strict')
                                              if avg in slot}))
    # the pooled estimates: D5 for the screen's winner; the methods' pooled estimates over
    # three corpora that 13.3 names; the week-11 method's, post hoc (MR6's is week 14's)
    pooled = w.get('pooled') or {}
    winner = w.get('week12_winner') or WINNER
    for method, label, status in (
            (winner, f'{system_name(winner)} − CRF head', 'post hoc'),
            ('crf_mr', 'MR3 − CRF head', 'pre-specified (13.3)'),
            ('crf_mt_mr', 'multi-task + MR3 − CRF head', 'pre-specified (13.3)'),
            ('crf_mrs', f"{OTHER_SUBFIELDS} (MuLMS) and the fatigue corpus's (MaterioMiner) "
                        '− CRF head', 'post hoc')):
        e = pooled.get(method)
        if not e:
            continue
        if e.get('status') == 'decision':
            status = f"D5: {pooled_verdict(e)}"
        rows += pooled_rows(e, label, status, W13)
    return rows + robust_rows(robust, ('names_vs_crf', 'names_vs_mr3', 'mr3_vs_crf'))


def pooled_verdict(e):
    """The verdict of a pooled decision: its interval above 0 with neither corpus's below 0."""
    return (verdict(e['passes'] and e['no_corpus_loses'])
            + (' (neither corpus below 0)' if e['no_corpus_loses'] else ' (a corpus below 0)'))


def against_rows(week, corpus, entry, system, references, statuses):
    """Rows of an `against` entry of the seed analysis: `system` against each reference, with
    the votes and the entry's units."""
    votes = entry.get('votes') or {}
    return [row(week, corpus, f'{system} − {SYSTEMS.get(r, r)}', status, entry[f'minus_{r}'],
                votes.get(f'minus_{r}'), entry['units'])
            for r, status in zip(references, statuses) if f'minus_{r}' in entry]


def week14(analysis, robust):
    """Week 14: D7 (MR6 + names against MR6, pooled) with its parts and MR6's own comparisons;
    D8 (the own-topic corpus against the mixed one) and the mixed-corpus variant; the
    ablations of 14.3 (pre-specified); and the post hoc pooled comparisons of MR6."""
    w = analysis.get('week14') or {}
    copies = w.get('copies') or {}
    rows = []
    pooled = copies.get('pooled')
    m = copies.get('mulms') or {}
    if pooled:
        parts = [m.get('decision') or {}, copies.get('fine') or {}]
        units = [u for p in parts for u in p.get('units', {})]
        rows.append(row(W14, pooled_label([(c, v['n_units']) for c, v in pooled['corpora'].items()],
                                          DECISION_SEEDS),
                        'MR6 + names − MR6', f'D7: {pooled_verdict(pooled)}',
                        pooled['minus_control'], units=units))
    loses = {c: v.get('loses') for c, v in ((pooled or {}).get('corpora') or {}).items()}

    def part_status(corpus):
        if corpus not in loses:
            return 'pre-specified'
        return ("D7's per-corpus condition: its interval "
                + ('below 0' if loses[corpus] else 'not below 0'))
    parts = (('mulms', m.get('decision'), lambda n: mulms(n, DECISION_SEEDS)),
             ('mulms ten', m.get('ten_seeds'), mulms),
             ('fine', copies.get('fine'), lambda n: papers('fine', n)),
             ('coarse', copies.get('coarse'), lambda n: papers('coarse', n)))
    for corpus, e, label in parts:
        if e:
            rows += against_rows(W14, label(e['n_units']), e, 'MR6 + names', ('crf_mr6',),
                                 (part_status(corpus),))
    for corpus, e, label in parts:
        if e and e.get('control'):
            c = e['control']
            rows += against_rows(W14, label(c['n_units']), c, 'MR6', ('crf', 'crf_mr'),
                                 ('pre-specified', 'pre-specified'))
    mr6 = (analysis.get('week13') or {}).get('pooled', {}).get('crf_mr6')
    if mr6:
        rows += pooled_rows(mr6, 'MR6 − CRF head', 'post hoc', W14)
    rows += robust_rows(robust, ('names_vs_mr6', 'mr6_vs_crf', 'mr6_vs_mr3'))
    topic = (w.get('topic') or {}).get('decision')
    if topic:
        votes = topic.get('votes') or {}
        rows.append(row(W14, mulms(topic['n_units'], topic['seeds']),
                        'MR6 + names, own-topic corpus − mixed corpus',
                        f"D8: {verdict(topic['passes'])}", topic['minus_crf_mrm_mixed6'],
                        votes.get('minus_crf_mrm_mixed6'), topic['units']))
        mixed = topic.get('mixed')
        if mixed:
            rows += against_rows(W14, mulms(mixed['n_units'], topic['seeds']), mixed,
                                 SYSTEMS['crf_mrm_mixed6'], ('crf', 'crf_mr', 'crf_mr6'),
                                 ('pre-specified',) * 3)
    for variant, label in ABLATIONS.items():
        e = (w.get('ablations') or {}).get(variant)
        if e:
            rows += against_rows(W14, mulms(e['n_units'], e.get('seeds', SCREEN_SEEDS)), e, label,
                                 ('crf_mrm_topic6', 'crf_mr', 'crf'), ('pre-specified',) * 3)
    return rows


def week17(analysis):
    """Week 17 (follow-up): per granularity multi-task + MR6 + names against multi-task + MR6
    (D9, D10), and both against multi-task + MR3, multi-task and the CRF head
    (pre-specified)."""
    rows = []
    references = ('crf_mt_mr', 'crf_mt', 'crf')
    for g, d in zip(GRANULARITIES, ('D9', 'D10')):
        e = (analysis.get('week17') or {}).get(g)
        if not e:
            continue
        corpus = papers(g, e['n_units'])
        rows += against_rows(W17, corpus, e, SYSTEMS['crf_mt_mrm6'], ('crf_mt_mr6',),
                             (f"{d}: {verdict(e['passes'])}",))
        for side, system in (('names', 'crf_mt_mrm6'), ('control', 'crf_mt_mr6')):
            if e.get(side):
                rows += against_rows(W17, corpus, e[side], SYSTEMS[system], references,
                                     ('pre-specified',) * 3)
    return rows


def week18(matched):
    """Week 18 (follow-up): MR6 + names against MR6 at matched recall on MuLMS's 7 subfields,
    seeds 300 to 900: D11, D12 and the pre-specified comparisons; the same without the flagged
    subfields (pre-specified) and with them matched at the nearest recall of the grid (post
    hoc); the post hoc comparisons at the control's best bias and the decomposition of D11."""
    d = (matched or {}).get('decisions')
    if not d:
        return []
    seeds = tuple(matched.get('seeds') or DECISION_SEEDS)
    flagged = (matched.get('without_flagged') or {}).get('flagged') or []

    def corpus(n, reading=''):
        return f'MuLMS, {n} subfields (seeds {span(seeds)}{reading})'
    without = f"; without the flagged {', '.join(flagged)}" if flagged else ''
    readings = ((matched, '', None, 'pre-specified'),
                (matched.get('without_flagged') or {}, without,
                 "pre-specified ({}'s reading without the flagged subfields)", 'pre-specified'),
                (matched.get('post_hoc_nearest') or {}, '; the flagged at the grid value of the '
                 'nearest recall', "post hoc ({}'s reading with the flagged at the nearest "
                 'recall)', 'post hoc'))
    rows = []
    for part, reading, alias, status in readings:
        decisions, pre = part.get('decisions') or {}, part.get('prespecified') or {}
        for key, paper_id, label in WEEK18:
            e = decisions.get(key)
            if e:
                text = (alias.format(paper_id) if alias else
                        f"{paper_id}: {e.get('verdict', verdict(above_0(e)))}")
                rows.append(row(W18, corpus(e['n_units'], reading), label, text, e))
        for key, label in WEEK18_PRESPECIFIED:
            e = pre.get(key)
            if e:
                rows.append(row(W18, corpus(e['n_units'], reading), label, status, e))
    best = (matched.get('post_hoc_curves') or {}).get('names_minus_best') or {}
    for key, what in (('recall_in', HELD), ('recall_out', 'the unseen test names the lexicon '
                                                          'does not hold')):
        e = best.get(key)
        if e:
            rows.append(row(W18, corpus(e['n_units']), f'MR6 + names − MR6 at its best bias of '
                                                       f'the grid: recall on {what}', 'post hoc', e))
    decomposition = matched.get('post_hoc_decomposition') or {}
    for part, reading in ((decomposition, ''), (decomposition.get('matched_only') or {}, without)):
        for key, label in WEEK18_DECOMPOSITION:
            e = part.get(key)
            if e:
                rows.append(row(W18, corpus(e['n_units'], reading), label, 'post hoc', e))
    return rows


# --------------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------------
def build(results):
    """The rows of the table from the results directory (its gpu/ files), in week order,
    each comparison once."""
    def gpu(name):
        return load(os.path.join(results, 'gpu', name))
    analysis = gpu('ner_week5_seed_analysis.json') or {}
    mentions = gpu('ner_mentions.json')
    robust = gpu('ner_robustness.json')
    rows = (week10(analysis, mentions)
            + week11(analysis, gpu('ner_week11_comparisons.json'), mentions)
            + week12(analysis, gpu('ner_week12_comparisons.json'))
            + week13(analysis, robust) + week14(analysis, robust) + week17(analysis)
            + week18(gpu('ner_recall_matched.json')))
    return with_measures(dedupe(rows), robust)


def strict_equals_default(rows):
    """True if every row that stores strict IOB2 F1 has the default scoring's numbers."""
    pairs = [(r['diff']['micro'], r['diff']['micro_strict']) for r in rows
             if r['diff'].get('micro_strict') and r['diff'].get('micro')]
    return bool(pairs) and all(same(a, b) for a, b in pairs)


def table13(rows):
    """The Markdown of the table from build's rows: the caption, the first part (one row per
    comparison) and the second (per-unit values and the votes' macro and weighted F1)."""
    cells, details = [], []
    for i, r in enumerate(rows, 1):
        diff, votes = r['diff'], r['votes']
        cells.append([i, f"{r['week'][0]} ({r['week'][1]})", r['corpus'], r['comparison'],
                      r['status'], cell(diff.get('micro'), r['units']), cell(diff.get('macro')),
                      cell(diff.get('weighted')), cell(diff.get('micro_strict')),
                      cell(votes.get('micro'), r['vote_units'])])
        detail = [per_unit(diff.get('micro'), r['units']),
                  per_unit(votes.get('micro'), r['vote_units']),
                  cell(votes.get('macro')), cell(votes.get('weighted'))]
        if any(detail):
            details.append([i, r['corpus'], r['comparison'], *detail])
    strict = (' Strict IOB2 scoring gives the default scoring\'s numbers in every row that '
              'stores both.' if strict_equals_default(rows) else '')
    return (
        "Table 13. The names route in full: every comparison of the protocols of 17 October "
        "20:21 to 1 November 2026 (weeks 10 to 14, 17 and 18) that the results files hold, one "
        "row each, in week order (ner_week5_seed_analysis.json, ner_week11_comparisons.json, "
        "ner_week12_comparisons.json, ner_mentions.json, ner_robustness.json and "
        "ner_recall_matched.json in results/gpu; names_route_record.py). A row's week is the "
        "protocol whose comparison it reports or re-reads, with the commit time (UTC) of the "
        "protocol or amendment that named the comparison (for a post hoc row, of the protocol "
        "it re-reads). A unit is a held-out MuLMS subfield, MaterioMiner "
        "paper or SOFC-Exp fold of articles, each side the mean over its seeds (MuLMS's in "
        "parentheses; MaterioMiner's and SOFC-Exp's 0–900). Status: a decision by its ID in "
        "Table 2 (D1 to D12) or, outside it, by week and number (10.1 is week 10's decision 1; "
        "13.1a as its protocol names it), with its verdict; a screen, which decides nothing; "
        "pre-specified, named in a protocol without a verdict rule; or post hoc, specified "
        "after the results it analyses were known. A cell is the mean paired difference over "
        "the units with its 95% Student-t CI, in the micro and vote columns with the units "
        "above 0; a row that names another measure (recall, precision, F1 on the seen or unseen "
        "test names, a level) gives it in the micro column, and a gap between unpaired settings "
        "has no interval. The votes are the entity-level majority of the three models of each "
        "seed triple (0–200, 300–500 and 600–800; on MuLMS's seeds 300–900 the last two, on "
        "its seeds 0–200 the first), each unit the mean over its triples. The status is the "
        "micro (or named) comparison's; the other columns are reported beside it and decide "
        "nothing, except the votes of decision 10.2 and, in the rows of V6's analogue, those "
        "of the companion study's week-5 decision; the macro and weighted cells of the pooled "
        "11-unit rows are post hoc, recomputed from the run files (ner_robustness.py, "
        "pooled_measures, whose micro values equal the rows' to within 0.01)." + strict
        + " An empty cell: the results do not hold that quantity, or, for the macro and "
        "weighted cells of the other pooled rows, it was not computed. Left out: the per-type analyses (Tables 8 and A2), the "
        "lexicon split of week 15 (Table 10), F1 on the seen and unseen test names after week 11 "
        "(ner_mentions.json; exploratory), and the companion study's comparisons, among them "
        "MR3 and multi-task + MR3 against the CRF head on MaterioMiner's papers, whose units "
        "some pooled rows include (Table 3, Supplementary Table S1); of those, only the ones the "
        "names route reports again are rows: the label-propagation rule's on MaterioMiner's "
        "papers (week 7, recomputed in week 13) and self-training's on them (week 5, V6's "
        "analogue in week 12), not its random splits.\n\n"
        + markdown(HEADER, cells)
        + "\nTable 13, second part. Per row of the first part (#): the per-unit differences of "
        "its micro (or named) comparison and of its votes' micro F1 (MuLMS's subfields, "
        "MaterioMiner's papers s11340, ncomms7164, s41467 and met10111458, SOFC-Exp's folds), "
        "and its votes' macro and weighted F1 differences with their 95% CIs.\n\n"
        + markdown(DETAIL_HEADER, details))


def main(argv=None):
    parser = argparse.ArgumentParser(description="The names route's comparisons in one table")
    parser.add_argument('--results', default=str(ROOT / 'results'))
    parser.add_argument('--out', default=str(ROOT / 'docs' / 'tables'))
    args = parser.parse_args(argv)
    rows = build(args.results)
    if not rows:
        print('table 13: no results of the names route')
        return None
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, OUT_NAME)
    with open(path, 'w') as f:
        f.write(table13(rows))
    print(f'wrote {OUT_NAME} ({len(rows)} rows) to {args.out}')
    return path


if __name__ == '__main__':
    main()
