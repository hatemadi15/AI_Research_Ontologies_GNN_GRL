"""The names route's comparisons in one table (names_route_record.py) from toy results."""
import json
import os

import pytest

import names_route_record as nr

SUBFIELDS = ('electrolysis', 'graphene', 'pemfc', 'poly', 'semi', 'sofc', 'steel')
PAPERS = ('10.1007_s11340-021-00758-x.conll', '10.1038_ncomms7164.conll',
          '10.1038_s41467-020-19071-7.conll', '10.3390_met10111458.conll')
FOLDS = tuple(f'fold{i}' for i in range(5))
REAL_RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
M = '−'


def _d(mean, lo, hi, per_run=None):
    """A paired difference, with its per-unit values if given."""
    out = {'mean_diff': mean, 'ci95': [lo, hi]}
    if per_run is not None:
        out['per_run'] = list(per_run)
    return out


def _avgs(micro, per_run=None, macro=(0.5, -0.5, 1.5), weighted=(0.25, -0.75, 1.25)):
    """A difference under every F1 average; strict IOB2 equal to micro."""
    return {'micro': _d(*micro, per_run), 'micro_strict': _d(*micro, per_run),
            'macro': _d(*macro), 'weighted': _d(*weighted)}


def _units(names, values):
    return dict(zip(names, values))


def _line(table, number):
    """The cells of row `number` of a rendered table's first part."""
    for line in table.split('\n\n')[1].splitlines():
        cells = [c.strip() for c in line.strip('|').split('|')]
        if cells[0] == str(number):
            return cells
    raise AssertionError(f'no row {number}')


# --------------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------------
def test_cells_print_signed_differences_with_typographic_minus_and_units_above_0():
    per_run = (1.42, 1.66, 2.28, -0.73, 2.52, 7.39, 0.15)
    assert nr.cell(_d(2.1, -0.31, 4.51, per_run), list(SUBFIELDS)) == \
        f'+2.10 [{M}0.31, +4.51], 6/7'
    assert nr.cell(_d(2.1, -0.31, 4.51, per_run)) == f'+2.10 [{M}0.31, +4.51]'
    assert nr.cell(_d(0.0, -1, 1, (0.0, 0.5, -0.5)), ['a', 'b', 'c']) == \
        f'+0.00 [{M}1.00, +1.00], 1/3'                       # a unit at 0 is not above it
    # counts stored with the entry (no per-unit values), a gap, a level, nothing
    assert nr.cell({'mean_diff': 5.92, 'ci95': [-1.1, 12.94], 'n_units': 4, 'n_positive': 4},
                   []) == f'+5.92 [{M}1.10, +12.94], 4/4'
    assert nr.cell({'gap': 17.06}) == '+17.06' and nr.cell({'gap': -0.5}) == f'{M}0.50'
    assert nr.cell({'level': {'mean': 74.01, 'ci95': [67.89, 80.13]}}) == \
        '74.01 [67.89, 80.13] (its level)'
    assert nr.cell(None) == '' and nr.cell({}) == ''


def test_units_are_named_short_and_listed_subfields_papers_folds():
    assert [nr.unit_name(u) for u in PAPERS] == ['s11340', 'ncomms7164', 's41467', 'met10111458']
    assert nr.unit_name('steel') == 'steel' and nr.unit_name('fold3') == 'fold3'
    entry = {'per_unit': {'fold1': 1.0, PAPERS[1]: 0.5, 'semi': -2.0}}
    assert nr.per_unit(entry) == f'semi {M}2.00, ncomms7164 +0.50, fold1 +1.00'
    assert nr.per_unit(_d(1, 0, 2, (0.1, -0.2)), ['b', 'a']) == f'a {M}0.20, b +0.10'
    assert nr.per_unit(_d(1, 0, 2, (0.1, -0.2)), ['a']) == ''    # units that do not match


def test_dedupe_keeps_the_first_of_a_repeat_and_rejects_two_comparisons_under_one_label():
    first = nr.row(nr.W13, 'pooled, 11 units', 'MR6 + names − CRF head', 'D5: passes',
                   _d(3.58, 0.95, 6.21))
    again = nr.row(nr.W13, 'pooled, 11 units', 'MR6 + names − CRF head', 'D5', _d(3.58, 0.95, 6.21))
    other = nr.row(nr.W13, 'pooled, 16 units', 'MR6 + names − CRF head', 'x', _d(2.45, 0.5, 4.4))
    assert nr.dedupe([first, again, other]) == [first, other]
    wrong = nr.row(nr.W13, 'pooled, 11 units', 'MR6 + names − CRF head', 'D5', _d(3.0, 0.9, 6.2))
    with pytest.raises(ValueError):
        nr.dedupe([first, wrong])


# --------------------------------------------------------------------------------
# the weeks
# --------------------------------------------------------------------------------
def _mentions():
    def method(reference, seeds, unseen, seen):
        return {'reference': reference, 'seeds': list(seeds), 'unseen': _d(*unseen),
                'seen': _d(*seen), 'all': _d(0, -1, 1)}
    return {'methods': {
        'mulms_leave_one_subfield_out': {
            'crf_mr': method('crf', nr.TEN, (2.19, -0.25, 4.63), (1.3, -1.21, 3.81)),
            'crf_mr_ens3': method('crf_ens3', (0, 300, 600), (2.92, 0.41, 5.44), (1.17, -1.69, 4.04)),
            'crf_mrs': method('crf', nr.TEN, (2.13, -0.84, 5.09), (1.21, -0.97, 3.39)),
            'crf_mrm_topic6': method('crf', nr.SCREEN_SEEDS, (3.4, -1.7, 8.6), (1, 0, 2))},
        'fine_leave_one_paper_out': {
            'crf_mrs': method('crf', nr.TEN, (3.19, 2.03, 4.35), (2.66, 0.03, 5.29))}}}


def _mulms():
    held = {'n_subfields': 7, 'passes': False, 'subfields': _units(SUBFIELDS, [{}] * 7),
            'crf_mr_minus_crf': _avgs((2.1, -0.31, 4.51), (1.42, 1.66, 2.28, -0.73, 2.52, 7.39, 0.15),
                                      macro=(2.12, 1.14, 3.09), weighted=(2.68, 0.32, 5.04)),
            'votes': {'passes': True, 'crf_mr_minus_crf': _avgs(
                (2.71, 0.22, 5.2), (2.12, 2.12, 3.18, -0.65, 3.38, 7.96, 0.88))}}
    names = {'n_subfields': 7, 'subfields': _units(SUBFIELDS, range(7)),
             'decision_1': False, 'decision_2': False,
             'minus_crf': _avgs((2.04, -0.85, 4.92), (1, 1, 1, -1, 1, 8, 0)),
             'minus_crf_mr': _avgs((-0.06, -1.09, 0.96), (1, -1, -1, 1, -1, 1, 0)),
             'votes': {'minus_crf': _avgs((2.48, -0.44, 5.4)), 'minus_crf_mr': _avgs((-0.23, -1.15, 0.68))}}

    def part(seeds, crf, mr, **flags):
        return {'seeds': list(seeds), 'n_subfields': 7, 'subfields': _units(SUBFIELDS, range(7)),
                'minus_crf': _avgs(crf, (1,) * 7), 'minus_crf_mr': _avgs(mr, (-1,) * 7),
                'votes': {'minus_crf': _avgs((3.23, -1.15, 7.61)),
                          'minus_crf_mr': _avgs((0.57, -2.22, 3.36))}, **flags}
    week12 = {'crf_mrm_topic6': {'screen': part(nr.SCREEN_SEEDS, (3.16, -1.96, 8.28), (1.45, -1.35, 4.25)),
                                 'decision': part(nr.DECISION_SEEDS, (3.19, -1.13, 7.51),
                                                  (0.92, -1.63, 3.48), decision_1=False,
                                                  decision_2=False),
                                 'ten_seeds': part(nr.TEN, (3.18, -1.3, 7.66), (1.08, -1.12, 3.28))},
              'crf_mrs_topic': {'screen': part(nr.SCREEN_SEEDS, (2.5, -2.09, 7.09), (0.8, -0.91, 2.51))},
              'crf_mr6': {'screen': part(nr.SCREEN_SEEDS, (1.5, -1.65, 4.66), (-0.2, -3.2, 2.8)),
                          'decision': part(nr.DECISION_SEEDS, (2.68, 0.13, 5.23), (0.41, -0.35, 1.17))}}
    return {'leave_one_subfield_out': held, 'random_split_pool': {'n_splits': 5},
            'gap': {'micro': 17.06, 'micro_strict': 17.06, 'macro': 15.06, 'weighted': 18.74},
            'silver_names': names, 'week12': week12}


def test_week10_decision_votes_gap_and_unseen_names():
    rows = nr.week10({'mulms': _mulms()}, _mentions())
    assert [r['comparison'] for r in rows] == [
        'MR3 − CRF head', 'CRF head on the random pool splits − CRF head on the held-out subfields',
        'MR3 − CRF head: F1 on the unseen test names', 'MR3 − CRF head: F1 on the seen test names']
    table = nr.table13(rows)
    assert _line(table, 1) == [
        '1', '10 (17 October 20:21)', 'MuLMS, 7 subfields (seeds 0–900)', 'MR3 − CRF head',
        'decision 10.1: no; votes: decision 10.2: passes', f'+2.10 [{M}0.31, +4.51], 6/7',
        '+2.12 [+1.14, +3.09]', '+2.68 [+0.32, +5.04]', f'+2.10 [{M}0.31, +4.51]',
        '+2.71 [+0.22, +5.20], 6/7']
    assert _line(table, 2)[5:] == ['+17.06', '+15.06', '+18.74', '+17.06', '']
    assert 'not paired' in _line(table, 2)[2]
    assert _line(table, 3)[4:] == ['pre-specified', f'+2.19 [{M}0.25, +4.63]', '', '', '',
                                   '+2.92 [+0.41, +5.44]']
    # the second part: per subfield, the votes' per subfield and their macro and weighted F1
    assert (f'| 1 | MuLMS, 7 subfields (seeds 0–900) | MR3 − CRF head | electrolysis +1.42, '
            f'graphene +1.66, pemfc +2.28, poly {M}0.73, semi +2.52, sofc +7.39, steel +0.15 | '
            f'electrolysis +2.12, graphene +2.12, pemfc +3.18, poly {M}0.65, semi +3.38, '
            f'sofc +7.96, steel +0.88 | +0.50 [{M}0.50, +1.50] | +0.25 [{M}0.75, +1.25] |') in table
    assert nr.week10({}, None) == []


def test_unseen_name_rows_need_the_protocols_reference_and_ten_seeds():
    mentions = _mentions()
    assert nr.kinds_rows(nr.W12, mentions, 'mulms_leave_one_subfield_out', 'crf_mrm_topic6',
                         'MuLMS', 'MR6 + names') == []          # the screen's 3 seeds
    methods = mentions['methods']['mulms_leave_one_subfield_out']
    methods['crf_mr_ens3']['reference'] = 'crf_mr_ens3'          # votes against another vote
    rows = nr.kinds_rows(nr.W10, mentions, 'mulms_leave_one_subfield_out', 'crf_mr', 'MuLMS', 'MR3')
    assert len(rows) == 2 and all(r['votes'] == {} for r in rows)
    methods['crf_mr']['reference'] = 'crf_mr'
    assert nr.kinds_rows(nr.W10, mentions, 'mulms_leave_one_subfield_out', 'crf_mr', 'MuLMS',
                         'MR3') == []


def _entry(variant, reference, granularity, micro, per_run=(1, 1, 1, 1), role='reported',
           protocol=nr.LOPO, run_ids=PAPERS):
    return {'variant': variant, 'reference': reference, 'granularity': granularity,
            'protocol': protocol, 'role': role, 'n_pairs': len(run_ids), 'run_ids': list(run_ids),
            **_avgs(micro, per_run)}


def test_week11_its_four_decisions_the_reported_comparisons_and_the_vote_only_rows():
    comparisons = {str(i): e for i, e in enumerate((
        _entry('crf_mrs', 'crf_mr', 'fine', (0.23, -0.69, 1.14), (-0.5, 0.12, 0.42, 0.87), 'decision'),
        _entry('crf_mrs', 'crf_mr', 'coarse', (0.61, -0.96, 2.18), role='decision'),
        _entry('crf_mrs', 'crf', 'fine', (3.0, 1.98, 4.02)),
        _entry('crf_mrs_ens3', 'crf_mr_ens3', 'fine', (0.44, -1.85, 2.73), (-1, 1, -1, 1),
               run_ids=PAPERS[::-1]),
        _entry('crf_mrs_ens3', 'crf_ens3', 'fine', (4.05, 3.03, 5.07)),
        _entry('crf_mrs_ens3', 'crf_mt_ens3', 'coarse', (2.81, 0.95, 4.67)),
        _entry('crf_mrs', 'crf', 'fine', (9.0, 8.0, 10.0), protocol='random_split_vary')))}
    rows = nr.week11({'mulms': _mulms()}, comparisons, _mentions())
    labels = [(r['corpus'], r['comparison'], r['status']) for r in rows]
    assert labels[:5] == [
        ('MuLMS, 7 subfields (seeds 0–900)', "names from the other subfields' sentences − CRF head",
         'decision 11.1: no'),
        ('MuLMS, 7 subfields (seeds 0–900)', "names from the other subfields' sentences − MR3",
         'decision 11.2: no'),
        ('MaterioMiner fine, 4 papers', "names from the fatigue corpus's sentences − MR3",
         'decision 11.3: no'),
        ('MaterioMiner coarse, 4 papers', "names from the fatigue corpus's sentences − MR3",
         'decision 11.4: no'),
        ('MaterioMiner fine, 4 papers', "names from the fatigue corpus's sentences − CRF head",
         'pre-specified')]
    assert labels[5][1:] == ("the votes: names from the fatigue corpus's sentences − multi-task",
                             'pre-specified')
    assert [r['comparison'].rpartition(': ')[2] for r in rows[6:]] == [
        'F1 on the unseen test names', 'F1 on the seen test names'] * 2
    assert [r['corpus'] for r in rows[6::2]] == ['MuLMS, 7 subfields (seeds 0–900)',
                                                 'MaterioMiner fine, 4 papers']
    assert not any('9.00' in nr.cell(r['diff']['micro']) for r in rows)    # random splits left out
    table = nr.table13(rows)
    assert _line(table, 3)[5] == f'+0.23 [{M}0.69, +1.14], 3/4'
    assert _line(table, 3)[9] == f'+0.44 [{M}1.85, +2.73], 2/4'
    assert _line(table, 5)[9] == '+4.05 [+3.03, +5.07], 4/4'
    assert _line(table, 6)[5] == '+2.81 [+0.95, +4.67], 4/4' and _line(table, 6)[9] == ''
    # the votes' per-unit values follow their own entry's papers
    assert (f's11340 +1.00, ncomms7164 {M}1.00, s41467 +1.00, met10111458 {M}1.00') in table


def test_week12_the_screen_the_decisions_and_the_analogues():
    analysis = {'mulms': _mulms(), 'week13': {
        'week12_winner': 'crf_mrm_topic6',
        'pooled': {'crf_mrm_topic6': {'status': 'decision', 'corpora': {
            'mulms': {'minus_crf': _d(3.18, -1.3, 7.66)}, 'fine': {'minus_crf': _d(4.27, 1.39, 7.15)}}}}}}
    comparisons = {str(i): e for i, e in enumerate((
        _entry('crf_mrm6', 'crf', 'fine', (4.27, 1.39, 7.15), role='decision'),
        _entry('crf_mrm6', 'crf', 'coarse', (1.77, -0.67, 2.88), role='decision'),
        _entry('crf_mrm6_ens3', 'crf_ens3', 'fine', (4.99, 3.27, 6.72)),
        _entry('crf_mrm6', 'crf_mr', 'fine', (1.5, -1.76, 4.75)),
        _entry('crf_st3', 'crf', 'fine', (1.74, -1.16, 4.64)),
        _entry('crf_st3_ens3', 'crf_ens3', 'fine', (1.12, -2.71, 4.96)),
        _entry('crf_st3', 'crf_mr', 'coarse', (0.28, -1.46, 2.01)),
        _entry('crf_mr6', 'crf', 'fine', (2.98, 0.09, 5.86)),
        _entry('crf_st3', 'crf', 'fine', (1.05, 0.1, 2.0), protocol='random_split_vary')))}
    rows = nr.week12(analysis, comparisons)
    got = [(r['week'], r['corpus'], r['comparison'], r['status']) for r in rows]
    assert got[:4] == [
        (nr.W12, 'MuLMS, 7 subfields (seeds 0–200)', 'MR3 + names, sentence filter (V1) − MR3',
         'screen, its statistic (no decision)'),
        (nr.W12, 'MuLMS, 7 subfields (seeds 0–200)', 'MR3 + names, sentence filter (V1) − CRF head',
         'screen (no decision)'),
        (nr.W12, 'MuLMS, 7 subfields (seeds 0–200)', 'MR6 (V3) − MR3',
         'screen, its statistic (no decision)'),
        (nr.W12, 'MuLMS, 7 subfields (seeds 0–200)', 'MR6 (V3) − CRF head', 'screen (no decision)')]
    assert got[4][0] == nr.W12_A1 and got[4][3].endswith('; selected')
    assert [g[1:] for g in got[6:12]] == [
        ('MuLMS, 7 subfields (seeds 300–900)', 'MR6 + names − CRF head', 'D1: no'),
        ('MuLMS, 7 subfields (seeds 300–900)', 'MR6 + names − MR3', 'D2: no'),
        ('MuLMS, 7 subfields (seeds 0–900)', 'MR6 + names − CRF head',
         "pre-specified; D5's MuLMS part"),
        ('MuLMS, 7 subfields (seeds 0–900)', 'MR6 + names − MR3', 'pre-specified'),
        ('MaterioMiner fine, 4 papers', 'MR6 + names − CRF head',
         "D3: passes; D5's MaterioMiner part"),
        ('MaterioMiner coarse, 4 papers', 'MR6 + names − CRF head', 'D4: no')]
    assert got[12][::3] == (nr.W12_ANALOGUE, 'pre-specified')
    v6 = [g for g in got if "V6's analogue" in g[2]]
    assert [g[3] for g in v6] == [
        "reported again; the companion study's week-5 decision (15 October): no; votes: no",
        'post hoc']
    # MR6's rows of the week-12 file are week 14's; random splits and MR6 at seeds 300-900 out
    assert not any(g[2].startswith('MR6 − ') or g[2].startswith('MR6 (V3)') and '300' in g[1]
                   for g in got)
    assert len(rows) == 15
    assert rows[10]['votes']['micro']['mean_diff'] == 4.99


def _consistency(decision=None, passes=False, n=4, units=PAPERS, mean=0.2):
    e = {'n_units': n, 'units': _units(units, (0.1,) * n),
         'minus_input': _avgs((mean, mean - 0.5, mean + 0.5), (0.1,) * n)}
    if decision:
        e.update({'decision': decision, 'passes': passes})
    return e


def _pooled(status, corpora, mean, lo, hi, passes=True, loses=False, **extra):
    units = [u for _, names in corpora for u in names]
    return {'status': status, 'passes': passes, 'no_corpus_loses': not loses,
            'n_units': len(units), 'minus_crf': _d(mean, lo, hi, (1,) * len(units)),
            'corpora': {c: {'n_units': len(names), 'units': _units(names, (1,) * len(names)),
                            'minus_crf': _d(mean, lo, hi), 'loses': False}
                        for c, names in corpora}, **extra}


def test_week13_label_propagation_folds_and_the_pooled_estimates():
    analysis = {'week13': {
        'week12_winner': 'crf_mrm_topic6',
        'consistency': {
            'mulms': {'crf_dc': _consistency('13.1a', True, 7, SUBFIELDS),
                      'crf_dcr': _consistency(n=7, units=SUBFIELDS)},
            'fine': {'crf_dc': _consistency('week 7'), 'crf_dcr': _consistency(),
                     'crf_mr_dc': _consistency('13.1d'), 'crf_mt_mr_dcr': _consistency()},
            'sofc_folds': {'crf_dc': _consistency(n=5, units=FOLDS, mean=-0.44)}},
        'folds': {
            'crf_mr': {'decision': '13.3a', 'passes': False, 'n_units': 5,
                       'units': _units(FOLDS, (-0.39, -0.42, -0.81, -1.36, 0.56)),
                       'minus_crf': _avgs((-0.48, -1.36, 0.39), (-0.39, -0.42, -0.81, -1.36, 0.56))},
            'crf_mt_mr': {'decision': '13.3b', 'passes': False, 'n_units': 5,
                          'units': _units(FOLDS, (0,) * 5), 'minus_crf': _avgs((-0.11, -0.69, 0.46)),
                          'minus_crf_mr': _avgs((0.37, -0.11, 0.84)),
                          'slot': {'n_units': 5, 'micro': {'mean': 74.01, 'ci95': [67.89, 80.13]},
                                   'macro': {'mean': 70.71, 'ci95': [64.62, 76.8]}}},
            'crf_mrm6': {'decision': '13.4', 'passes': False, 'n_units': 5,
                         'units': _units(FOLDS, (0,) * 5), 'minus_crf': _avgs((-0.04, -1.21, 1.14)),
                         'minus_crf_mr': _avgs((0.45, 0.04, 0.86))}},
        'pooled': {
            'crf_mrm_topic6': _pooled(
                'decision', (('mulms', SUBFIELDS), ('fine', PAPERS)), 3.58, 0.95, 6.21,
                with_sofc_folds={'corpora': {'mulms': {'n_units': 7}, 'fine': {'n_units': 4},
                                             'sofc_folds': {'n_units': 5}},
                                 'sofc_folds': {'units': _units(FOLDS, (0,) * 5)},
                                 'minus_crf': _d(2.45, 0.5, 4.4, (1,) * 16)},
                with_coarse={'corpora': {'mulms': {'n_units': 7}, 'coarse': {'n_units': 4}},
                             'coarse': {'units': {}}, 'minus_crf': _d(2.67, 0.09, 5.25)}),
            'crf_mr': _pooled('reported', (('mulms', SUBFIELDS), ('fine', PAPERS),
                                           ('sofc_folds', FOLDS)), 1.46, 0.28, 2.64),
            'crf_mrs': _pooled('reported', (('mulms', SUBFIELDS), ('fine', PAPERS)), 2.39, 0.71, 4.06),
            'crf_mr6': _pooled('reported', (('mulms', SUBFIELDS), ('fine', PAPERS)), 2.56, 0.94, 4.18)}}}
    rows = nr.week13(analysis, None)
    status = {(r['corpus'], r['comparison']): (r['week'], r['status']) for r in rows}
    assert status[('MuLMS, 7 subfields (seeds 0–900)', 'CRF head + label propagation − CRF head')] \
        == (nr.W13, 'decision 13.1a: passes')
    assert status[('MuLMS, 7 subfields (seeds 0–900)',
                   'CRF head + label propagation with retyping − CRF head')][1] == 'pre-specified'
    fine = 'MaterioMiner fine, 4 papers'
    assert status[(fine, 'CRF head + label propagation − CRF head')] == (
        nr.W13_AMENDED, "the companion study's week-7 decision (16 October), recomputed: no")
    assert status[(fine, 'CRF head + label propagation with retyping − CRF head')][1] == \
        "the companion study's week-7 comparison (16 October; reported), recomputed"
    assert status[(fine, 'MR3 + label propagation − MR3')][1] == 'decision 13.1d: no'
    assert status[(fine, 'multi-task + MR3 + label propagation with retyping − multi-task + MR3')][1] \
        == 'pre-specified'
    assert status[('SOFC-Exp, 5 folds', 'CRF head + label propagation − CRF head')][1] == \
        'pre-specified (13.1c: reported, not a decision)'
    assert status[('SOFC-Exp, 5 folds', 'MR3 − CRF head')] == (nr.W13, 'decision 13.3a: no')
    assert status[('SOFC-Exp, 5 folds', 'multi-task + MR3 − MR3')][1] == 'pre-specified'
    assert status[('SOFC-Exp, 5 folds', 'MR6 + names − CRF head')] == (nr.W13_4, 'D6: no')
    assert status[('SOFC-Exp, 5 folds', 'MR6 + names − MR3')] == (nr.W13_4, 'pre-specified')
    eleven = 'pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4'
    assert status[(eleven, 'MR6 + names − CRF head')] == (
        nr.W13, 'D5: passes (neither corpus below 0)')
    assert status[('pooled, 16 units: MuLMS 7 (0–900) + MaterioMiner fine 4 + SOFC-Exp 5',
                   'MR6 + names − CRF head')] == (nr.W13_4, 'pre-specified (13.4)')
    assert status[('pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner coarse 4',
                   'MR6 + names − CRF head')][1] == 'post hoc'      # an extension 13.4 never named
    assert status[('pooled, 16 units: MuLMS 7 (0–900) + MaterioMiner fine 4 + SOFC-Exp 5',
                   'MR3 − CRF head')][1] == 'pre-specified (13.3)'
    assert status[(eleven, "names from the other subfields' sentences (MuLMS) and the fatigue "
                           "corpus's (MaterioMiner) − CRF head")][1] == 'post hoc'
    assert not any(r['comparison'] == 'MR6 − CRF head' for r in rows)    # MR6's pooled: week 14
    table = nr.table13(rows)
    slot = next(line for line in table.splitlines() if 'slot level' in line and '74.01' in line)
    assert '74.01 [67.89, 80.13] (its level) | 70.71 [64.62, 76.80] (its level) |  |  |  |' in slot
    assert (f'fold0 {M}0.39, fold1 {M}0.42, fold2 {M}0.81, fold3 {M}1.36, fold4 +0.56') in table
    d5 = next(r for r in rows if r['status'].startswith('D5'))
    assert len(d5['units']) == 11 and nr.cell(d5['diff']['micro'], d5['units']).endswith(', 11/11')
    # a pooled decision with a corpus below 0 does not pass
    analysis['week13']['pooled']['crf_mrm_topic6']['no_corpus_loses'] = False
    assert any(r['status'] == 'D5: no (a corpus below 0)' for r in nr.week13(analysis, None))


def _robust():
    units = list(SUBFIELDS) + list(PAPERS)

    def e(mean, lo, hi):
        return {**_d(mean, lo, hi, (1,) * 10 + (-1,)), 'n_positive': 10, 'n_units': 11}
    return {'pooled': {
        'ten': {'units': units, 'names_vs_crf': e(3.58, 0.95, 6.21), 'mr3_vs_crf': e(2.35, 0.94, 3.75),
                'mr6_vs_crf': e(2.56, 0.94, 4.18), 'names_vs_mr6': e(1.02, -0.56, 2.59)},
        'decision': {'units': units, 'names_vs_crf': e(3.58, 1.04, 6.13),
                     'names_vs_mr6': e(0.8, -0.66, 2.26), 'mr6_vs_mr3': e(0.33, -0.25, 0.92)}}}


def test_robustness_rows_d5_prime_and_the_post_hoc_ones_each_printed_once():
    rows = nr.robust_rows(_robust(), ('names_vs_crf', 'mr3_vs_crf'))
    assert [(r['corpus'], r['comparison'], r['status']) for r in rows] == [
        ('pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4', 'MR6 + names − CRF head', 'D5'),
        ('pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4', 'MR3 − CRF head', 'post hoc'),
        ('pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4', 'MR6 + names − CRF head',
         'D5′, D5 on the decision seeds (post hoc reading): above 0')]
    assert nr.cell(rows[1]['diff']['micro'], rows[1]['units']) == '+2.35 [+0.94, +3.75], 10/11'
    assert nr.robust_rows(None, ('names_vs_crf',)) == []
    # with the seed analysis, D5 and D7 come from it once and MR6's 10-seed pooled estimate too
    analysis = {'week13': {'week12_winner': 'crf_mrm_topic6', 'pooled': {
        'crf_mrm_topic6': _pooled('decision', (('mulms', SUBFIELDS), ('fine', PAPERS)), 3.58, 0.95, 6.21),
        'crf_mr6': _pooled('reported', (('mulms', SUBFIELDS), ('fine', PAPERS)), 2.56, 0.94, 4.18)}},
        'week14': {'copies': {'pooled': {
            'decision': '14.1', 'n_units': 11, 'passes': False, 'no_corpus_loses': True,
            'minus_control': _d(0.8, -0.66, 2.26, (1,) * 11),
            'corpora': {'mulms': {'n_units': 7, 'loses': False},
                        'fine': {'n_units': 4, 'loses': False}}}}}}
    rows = nr.dedupe(nr.week13(analysis, _robust()) + nr.week14(analysis, _robust()))
    statuses = [(r['comparison'], r['status']) for r in rows]
    assert statuses.count(('MR6 + names − CRF head', 'D5: passes (neither corpus below 0)')) == 1
    assert ('MR6 + names − CRF head', 'D5') not in statuses
    assert ('MR6 + names − MR6', 'D7: no (neither corpus below 0)') in statuses
    assert ('MR6 + names − MR6', 'D7: no') not in statuses
    assert statuses.count(('MR6 − CRF head', 'post hoc')) == 1
    assert ('MR6 + names − MR6', 'post hoc') in statuses and ('MR6 − MR3', 'post hoc') in statuses


def _against(system, references, n, units, seeds=None, votes=True):
    e = {'n_units': n, 'units': _units(units, (50,) * n),
         **{f'minus_{r}': _avgs((1.0 + i, 0.5 + i, 1.5 + i), (1,) * n) for i, r in enumerate(references)}}
    if votes:
        e['votes'] = {f'minus_{r}': _avgs((0.35, -1.67, 2.36), (1,) * n) for r in references}
    if seeds:
        e['seeds'] = list(seeds)
    return e


def test_week14_d7_its_parts_mr6_d8_the_mixed_corpus_and_the_ablations():
    decision = _against('crf_mrm_topic6', ('crf_mr6',), 7, SUBFIELDS, nr.DECISION_SEEDS)
    decision['control'] = _against('crf_mr6', ('crf', 'crf_mr'), 7, SUBFIELDS)
    fine = _against('crf_mrm6', ('crf_mr6',), 4, PAPERS, nr.TEN)
    fine['control'] = _against('crf_mr6', ('crf', 'crf_mr'), 4, PAPERS)
    topic = _against('crf_mrm_topic6', ('crf_mrm_mixed6',), 7, SUBFIELDS, nr.DECISION_SEEDS)
    topic.update({'decision': '14.2', 'passes': False,
                  'mixed': _against('crf_mrm_mixed6', ('crf', 'crf_mr', 'crf_mr6'), 7, SUBFIELDS)})
    analysis = {'week14': {
        'copies': {'mulms': {'decision': decision}, 'fine': fine,
                   'pooled': {'decision': '14.1', 'n_units': 11, 'passes': False,
                              'no_corpus_loses': True, 'minus_control': _d(0.8, -0.66, 2.26, (1,) * 11),
                              'corpora': {'mulms': {'n_units': 7, 'loses': False},
                                          'fine': {'n_units': 4, 'loses': True}}}},
        'topic': {'decision': topic},
        'ablations': {'crf_mrm_topiclow6': _against(
            'crf_mrm_topiclow6', ('crf_mrm_topic6', 'crf_mr', 'crf'), 7, SUBFIELDS, nr.SCREEN_SEEDS)}}}
    rows = nr.week14(analysis, None)
    got = [(r['corpus'], r['comparison'], r['status']) for r in rows]
    assert got == [
        ('pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4', 'MR6 + names − MR6',
         'D7: no (neither corpus below 0)'),
        ('MuLMS, 7 subfields (seeds 300–900)', 'MR6 + names − MR6',
         "D7's per-corpus condition: its interval not below 0"),
        ('MaterioMiner fine, 4 papers', 'MR6 + names − MR6',
         "D7's per-corpus condition: its interval below 0"),
        ('MuLMS, 7 subfields (seeds 300–900)', 'MR6 − CRF head', 'pre-specified'),
        ('MuLMS, 7 subfields (seeds 300–900)', 'MR6 − MR3', 'pre-specified'),
        ('MaterioMiner fine, 4 papers', 'MR6 − CRF head', 'pre-specified'),
        ('MaterioMiner fine, 4 papers', 'MR6 − MR3', 'pre-specified'),
        ('MuLMS, 7 subfields (seeds 300–900)', 'MR6 + names, own-topic corpus − mixed corpus',
         'D8: no'),
        ('MuLMS, 7 subfields (seeds 300–900)', 'the mixed-corpus variant − CRF head', 'pre-specified'),
        ('MuLMS, 7 subfields (seeds 300–900)', 'the mixed-corpus variant − MR3', 'pre-specified'),
        ('MuLMS, 7 subfields (seeds 300–900)', 'the mixed-corpus variant − MR6', 'pre-specified'),
        ('MuLMS, 7 subfields (seeds 0–200)',
         'MR6 + names, filter: marginal threshold 0.7 − MR6 + names', 'pre-specified'),
        ('MuLMS, 7 subfields (seeds 0–200)', 'MR6 + names, filter: marginal threshold 0.7 − MR3',
         'pre-specified'),
        ('MuLMS, 7 subfields (seeds 0–200)', 'MR6 + names, filter: marginal threshold 0.7 − CRF head',
         'pre-specified')]
    assert len(rows[0]['units']) == 11 and rows[0]['votes'] == {}
    assert nr.cell(rows[1]['votes']['micro'], rows[1]['units']) == f'+0.35 [{M}1.67, +2.36], 7/7'
    assert nr.week14({}, None) == []


def test_week17_d9_d10_and_the_reported_comparisons():
    references = ('crf_mt_mr', 'crf_mt', 'crf')
    week17 = {}
    for g, decision in (('fine', '17.1'), ('coarse', '17.2')):
        e = _against('crf_mt_mrm6', ('crf_mt_mr6',), 4, PAPERS, nr.TEN)
        e.update({'decision': decision, 'passes': False,
                  'names': _against('crf_mt_mrm6', references, 4, PAPERS),
                  'control': _against('crf_mt_mr6', references, 4, PAPERS)})
        week17[g] = e
    rows = nr.week17({'week17': week17})
    assert len(rows) == 14 and rows[0]['week'] == nr.W17
    assert [(r['comparison'], r['status']) for r in rows[:2]] == [
        ('multi-task + MR6 + names − multi-task + MR6', 'D9: no'),
        ('multi-task + MR6 + names − multi-task + MR3', 'pre-specified')]
    assert rows[6]['comparison'] == 'multi-task + MR6 − CRF head'
    assert (rows[7]['corpus'], rows[7]['status']) == ('MaterioMiner coarse, 4 papers', 'D10: no')


def test_week18_decisions_prespecified_readings_and_post_hoc():
    def e(mean, lo, hi, units=SUBFIELDS, verdict=None):
        out = {'per_unit': _units(units, (1.0,) * len(units)), 'mean_diff': mean, 'ci95': [lo, hi],
               'n_units': len(units), 'n_positive': len(units)}
        if verdict:
            out['verdict'] = verdict
        return out
    kept = ('graphene', 'poly', 'semi', 'steel')
    matched = {
        'seeds': list(nr.DECISION_SEEDS),
        'decisions': {'18.1': e(11.24, 3.31, 19.18, verdict='passes'),
                      '18.2': e(3.26, -0.79, 7.31, verdict='no')},
        'prespecified': {'precision': e(5.78, -1.22, 12.79), 'bias_f1_change': e(-2.75, -5.77, 0.27)},
        'without_flagged': {'flagged': ['electrolysis', 'pemfc', 'sofc'],
                            'decisions': {'18.1': e(6.05, -3.33, 15.44, kept)},
                            'prespecified': {'precision': e(0.56, -6.74, 7.85, kept)}},
        'post_hoc_nearest': {'decisions': {'18.2': e(1.37, -0.66, 3.41)}},
        'post_hoc_curves': {'names_minus_best': {'recall_out': e(-4.16, -5.82, -2.5)}},
        'post_hoc_decomposition': {'names_minus_mr6_at_0': e(7.29, 3.27, 11.3),
                                   'matched_only': {'mr6_at_0_minus_control_at_d_star': {
                                       'mean_diff': 0.14, 'ci95': [-4.89, 5.16], 'n_units': 4,
                                       'n_positive': 1}}}}
    rows = nr.week18(matched)
    got = [(r['corpus'], r['comparison'], r['status']) for r in rows]
    seven = 'MuLMS, 7 subfields (seeds 300–900)'
    four = 'MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc)'
    assert got == [
        (seven, f'{nr.MATCHED}: recall on {nr.HELD}',
         "D11: passes as registered, with the 3 flagged subfields matched at the grid's end"),
        (seven, f'{nr.MATCHED}: micro F1', 'D12: no'),
        (seven, f'{nr.MATCHED}: precision on all test entities', 'pre-specified'),
        (seven, 'MR6 at matched recall − MR6 at bias 0: micro F1', 'pre-specified'),
        (four, f'{nr.MATCHED}: recall on {nr.HELD}',
         "pre-specified (D11's reading without the flagged subfields)"),
        (four, f'{nr.MATCHED}: precision on all test entities', 'pre-specified'),
        ('MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall)',
         f'{nr.MATCHED}: micro F1', "post hoc (D12's reading with the flagged at the nearest recall)"),
        (seven, 'MR6 + names − MR6 at its best bias of the grid: recall on the unseen test names '
                'the lexicon does not hold', 'post hoc'),
        (seven, f'MR6 + names − MR6 at bias 0: recall on {nr.HELD}', 'post hoc'),
        (four, f'MR6 at bias 0 − MR6 at matched recall: recall on {nr.HELD}', 'post hoc')]
    table = nr.table13(rows)
    assert _line(table, 1)[5:] == ['+11.24 [+3.31, +19.18], 7/7', '', '', '', '']
    assert _line(table, 10)[5] == f'+0.14 [{M}4.89, +5.16], 1/4'
    assert nr.week18({}) == [] and nr.week18(None) == []


# --------------------------------------------------------------------------------
# the table and main
# --------------------------------------------------------------------------------
def test_the_table_numbers_its_rows_and_says_when_strict_scoring_equals_the_default():
    rows = nr.week10({'mulms': _mulms()}, _mentions())
    table = nr.table13(rows)
    caption, first, second = table.split('\n\n')[:3]
    assert caption.startswith('Table 13. The names route in full')
    assert 'Strict IOB2 scoring gives the default scoring\'s numbers' in caption
    assert first.splitlines()[0] == '| ' + ' | '.join(nr.HEADER) + ' |'
    assert [line.split(' | ')[0] for line in first.splitlines()[2:]] == ['| 1', '| 2', '| 3', '| 4']
    assert second.startswith('Table 13, second part.')
    # only the rows with per-unit values or vote averages are in the second part
    assert [line.split(' | ')[0] for line in table.split('\n\n')[3].splitlines()[2:]] == ['| 1']
    rows[0]['diff']['micro_strict'] = _d(1.9, -0.5, 4.3)
    assert 'Strict IOB2' not in nr.table13(rows).split('\n\n')[0]


def test_main_writes_the_table_from_what_it_has(tmp_path, capsys):
    results = tmp_path / 'results'
    (results / 'gpu').mkdir(parents=True)
    assert nr.main(['--results', str(results), '--out', str(tmp_path / 'tables')]) is None
    assert 'table 13: no results of the names route' in capsys.readouterr().out
    (results / 'gpu' / 'ner_week5_seed_analysis.json').write_text(json.dumps({'mulms': _mulms()}))
    (results / 'gpu' / 'ner_mentions.json').write_text(json.dumps(_mentions()))
    path = nr.main(['--results', str(results), '--out', str(tmp_path / 'tables')])
    assert path == str(tmp_path / 'tables' / nr.OUT_NAME)
    text = open(path).read()
    assert text.count('| 10 (17 October 20:21) |') == 4
    # weeks 11 and 12 from the seed analysis alone (no comparisons files): MuLMS's two
    # decisions and the unseen and seen names on MuLMS and fine-grained MaterioMiner
    assert text.count('| 11 (18 October 01:13) |') == 6 and '| D1: no |' in text
    assert 'wrote table13_names_route.md' in capsys.readouterr().out
    # D5 in both the seed analysis and ner_robustness.json: printed once, from the first
    analysis = {'mulms': _mulms(), 'week13': {'week12_winner': 'crf_mrm_topic6', 'pooled': {
        'crf_mrm_topic6': _pooled('decision', (('mulms', SUBFIELDS), ('fine', PAPERS)),
                                  3.58, 0.95, 6.21)}}}
    (results / 'gpu' / 'ner_week5_seed_analysis.json').write_text(json.dumps(analysis))
    (results / 'gpu' / 'ner_robustness.json').write_text(json.dumps(_robust()))
    rows = nr.build(str(results))
    d5 = [r['status'] for r in rows if r['comparison'] == 'MR6 + names − CRF head'
          and r['corpus'] == 'pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4']
    assert d5 == ['D5: passes (neither corpus below 0)']
    assert len({(r['corpus'], r['comparison']) for r in rows}) == len(rows)


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL_RESULTS, 'gpu',
                                                    'ner_week5_seed_analysis.json')),
                    reason='the stored results are not in this checkout')
def test_smoke_the_stored_results():
    rows = nr.build(REAL_RESULTS)
    table = nr.table13(rows)
    statuses = [r['status'] for r in rows]
    for d in (f'D{i}' for i in range(1, 13)):
        assert any(s.startswith(f'{d}:') for s in statuses), d
    assert all(r['status'] and r['diff'].get('micro') for r in rows)
    assert len({(r['corpus'], r['comparison']) for r in rows}) == len(rows)
    for numbers in ('+2.12 [+1.14, +3.09]', '+2.71 [+0.22, +5.20]', '+3.00 [+1.98, +4.02]',
                    '+3.78 [+2.47, +5.10]', f'+0.19 [{M}2.28, +2.66]', '74.01 [67.89, 80.13]'):
        assert numbers in table, numbers


def test_pooled_rows_get_the_post_hoc_macro_and_weighted_f1():
    """The 11-unit pooled rows take ner_robustness.json's pooled_measures by basis and
    comparison; a row's own macro cell is kept, and other corpora's rows are left alone."""
    units = list(SUBFIELDS) + list(PAPERS)

    def e(mean):
        return {'mean_diff': mean, 'ci95': [mean - 1, mean + 1], 'per_run': [mean] * 11, 'n_positive': 11}
    robust = {'pooled_measures': {'decision': {'units': units, 'names_vs_mr6': {
        'micro': e(0.8), 'macro': e(0.83), 'weighted': e(0.89)}}}}
    decision = 'pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4'
    rows = [nr.row(nr.W14, decision, 'MR6 + names − MR6', 'D7: no', e(0.8)),
            nr.row(nr.W14, 'pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4',
                   'MR6 + names − MR6', 'post hoc', e(1.02)),
            nr.row(nr.W14, decision, 'MR6 − MR3', 'post hoc', {'micro': e(0.33), 'macro': e(9.9)})]
    out = nr.with_measures(rows, robust)
    assert nr.cell(out[0]['diff']['macro']) == '+0.83 [−0.17, +1.83]'
    assert out[0]['diff']['weighted']['mean_diff'] == 0.89
    assert 'macro' not in out[1]['diff']                      # the 10-seed basis has no entry here
    assert out[2]['diff']['macro']['mean_diff'] == 9.9        # a row's own value is kept
    assert nr.with_measures(rows, None) == rows
