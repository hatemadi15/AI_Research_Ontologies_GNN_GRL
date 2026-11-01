"""The recall-matched control (ner_recall_matched.py, week 18): the matching of a target recall
on a bias curve, the interpolation, the pooled rates, the paired comparisons and the D = 0
check."""
from collections import Counter

import pytest

import ner_recall_matched as rm


def test_match_finds_the_first_crossing_from_zero():
    curve = {-1.0: 40.0, -0.5: 42.0, 0.0: 44.0, 0.5: 45.0, 1.0: 47.0, 2.0: 50.0}
    assert rm.match(curve, 44.0) == (0.0, 0.0, 0.0, False)          # no bias needed
    d_a, d_b, w, flagged = rm.match(curve, 46.0)                    # between 0.5 and 1
    assert (d_a, d_b, flagged) == (0.5, 1.0, False) and w == pytest.approx(0.5)
    d_a, d_b, w, flagged = rm.match(curve, 41.0)                    # downward, between -0.5 and -1
    assert (d_a, d_b, flagged) == (-0.5, -1.0, False) and w == pytest.approx(0.5)
    assert rm.match(curve, 55.0) == (2.0, 2.0, 0.0, True)           # beyond the grid: flagged
    assert rm.match(curve, 30.0) == (-1.0, -1.0, 0.0, True)
    # a curve that dips first: the first crossing along the path from 0 is taken
    wavy = {0.0: 44.0, 0.5: 43.0, 1.0: 46.0, 2.0: 48.0}
    d_a, d_b, w, _ = rm.match(wavy, 45.0)
    assert (d_a, d_b) == (0.5, 1.0) and w == pytest.approx(2 / 3)


def test_blend_rates_and_f1():
    a = Counter({('all', 'correct'): 40, ('all', 'pred'): 80, ('all', 'gold'): 100})
    b = Counter({('all', 'correct'): 50, ('all', 'pred'): 120, ('all', 'gold'): 100})
    mid = rm.blend(a, b, 0.5)
    assert mid['all', 'correct'] == 45 and mid['all', 'pred'] == 100
    assert rm.f1_of(a) == pytest.approx(200 * 40 / 180) and rm.blend(1.0, 3.0, 0.25) == 1.5
    c = Counter({('all', 'correct'): 5, ('all', 'pred'): 10, ('all', 'gold'): 20,
                 ('gold', 'in'): 4, ('correct', 'in'): 3, ('gold', 'out'): 6,
                 ('correct', 'out'): 1, ('pred', 'in'): 4, ('pred_correct', 'in'): 3,
                 ('pred', 'out'): 4, ('pred_correct', 'out'): 1})
    r = rm.rates(c)
    assert (r['recall'], r['precision']) == (25.0, 50.0)
    assert (r['recall_in'], r['recall_out']) == (75.0, pytest.approx(100 / 6))
    assert r['unseen_recall'] == 40.0 and r['unseen_precision'] == 50.0
    assert r['unseen_f1'] == pytest.approx(2 * 40 * 50 / 90)


def _run(correct, pred, gold, cin, gin, cout, gout, pin=None, pcin=None):
    return Counter({('all', 'correct'): correct, ('all', 'pred'): pred, ('all', 'gold'): gold,
                    ('correct', 'in'): cin, ('gold', 'in'): gin, ('correct', 'out'): cout,
                    ('gold', 'out'): gout, ('pred', 'in'): pin or cin,
                    ('pred_correct', 'in'): pcin or cin, ('pred', 'out'): cout + 5,
                    ('pred_correct', 'out'): cout})


def test_unit_analysis_matches_the_pooled_recall():
    # the names: recall 50 over all entities, 8 of 10 held names found
    names = [_run(50, 90, 100, 8, 10, 30, 70), _run(50, 90, 100, 8, 10, 30, 70)]
    # the control: recall 40 at 0, 60 at 1 (so D* = 0.5); held names 5 of 10 at 0, 7 at 1
    control = [{0.0: _run(40, 70, 100, 5, 10, 25, 70), 1.0: _run(60, 130, 100, 7, 10, 40, 70)}
               for _ in range(2)]
    u = rm.unit_analysis(names, control)
    assert u['target_recall'] == 50.0 and u['d_star'] == 0.5 and not u['flagged']
    assert u['control_at_d_star']['recall'] == 50.0 and u['control_at_d_star']['recall_in'] == 60.0
    assert u['_diffs']['recall_in'] == pytest.approx(20.0)      # 80 against 60
    # F1: the names 200*50/190; the control halfway between 200*40/170 and 200*60/230
    f1_c = (200 * 40 / 170 + 200 * 60 / 230) / 2
    assert u['_diffs']['f1'] == pytest.approx(200 * 50 / 190 - f1_c)
    assert u['_diffs']['bias_f1_change'] == pytest.approx(f1_c - 200 * 40 / 170)
    assert u['curve']['1'] == {'recall': 60.0, 'precision': round(6000 / 130, 2),
                               'f1_pooled': round(200 * 120 / 460, 2),
                               'f1_seed_mean': round(200 * 60 / 230, 2)}


def test_comparisons_and_d0_check():
    pytest.importorskip('scipy')
    units = {u: {'_diffs': {k: v for k in ('recall_in', 'f1', 'precision', 'unseen_recall',
                                           'unseen_precision', 'unseen_f1', 'recall_out',
                                           'in_minus_out', 'bias_f1_change')}}
             for u, v in (('a', 2.0), ('b', 3.0), ('c', 4.0))}
    out = rm.comparisons(units)
    assert out['decisions']['18.1']['mean_diff'] == 3.0 and out['decisions']['18.1']['verdict'] == 'passes'
    assert out['decisions']['18.2']['n_positive'] == 3
    assert rm.comparisons(units, keep={'a', 'b'})['decisions']['18.2']['n_units'] == 2
    stored = {'f1': 50.0, 'precision': 49.0, 'recall': 51.0, 'best_epoch': 4,
              'history': [{'epoch': 1}], 'test_pred': ['O B-MAT']}
    same = dict(stored, test_pred_bias={'0': ['O B-MAT'], '1': ['B-MAT B-MAT']})
    assert rm.d0_check(same, stored) == []
    other = dict(same, f1=50.1, test_pred_bias={'0': ['O O']})
    assert rm.d0_check(other, stored) == ['f1', 'test_pred_bias[0]']
