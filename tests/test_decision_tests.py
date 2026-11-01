"""The t statistics and P values of paper A's decisions (decision_tests.py): the test on a
small case, and on the committed results every recomputed interval is the stored one and every
one-sided P agrees with its decision's verdict."""
import os

import pytest

import decision_tests as dt

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results', 'gpu')


def test_t_test_small_case():
    pytest.importorskip('scipy')
    r = dt.t_test([1.0, 2.0, 3.0])
    assert r['n_units'] == 3 and r['df'] == 2 and r['mean_diff'] == 2.0 and r['sd'] == 1.0
    assert r['t'] == 3.46                                 # 2 / (1 / sqrt 3)
    assert r['p_one_sided'] == 0.0371 and r['p_two_sided'] == 0.0742
    assert r['ci95'] == [-0.48, 4.48]                     # 2 +- 4.303 / sqrt 3


def test_committed_decisions():
    pytest.importorskip('scipy')
    names = (dt.ANALYSIS, dt.WEEK12, dt.WEEK18, dt.ROBUST)
    if not all(os.path.exists(os.path.join(RESULTS, n)) for n in names):
        pytest.skip('results files not present')
    rows = dt.decision_tests(*(dt.load(os.path.join(RESULTS, n)) for n in names))
    assert [r['decision'] for r in rows] == ['D1', 'D2', 'D3', 'D4', 'D5', "D5'", 'D6', 'D7',
                                             'D8', 'D9', 'D10', 'D11', 'D12']
    assert all(r['interval_matches'] and r['p_agrees'] for r in rows)
    passing = {r['decision'] for r in rows if r['passes']}
    assert passing == {'D3', 'D4', 'D5', "D5'", 'D11'}
    assert all(r['p_one_sided'] < 0.025 for r in rows if r['passes'])
