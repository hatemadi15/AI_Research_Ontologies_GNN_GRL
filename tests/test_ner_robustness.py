"""Robustness checks of the pooled comparisons (ner_robustness.py): the sign-flip test, the
leave-one-out table, the per-unit differences from a synthetic analysis file."""
import json

import pytest

import ner_robustness as nr


def test_sign_flip_and_leave_one_out():
    pytest.importorskip('scipy')
    eleven = [1.0] * 11
    assert nr.sign_flip_p(eleven) == round(2 / 2 ** 11, 4)              # all signs flipped, or none
    assert nr.sign_flip_p([1.0, -1.0, 2.0, -2.0]) == 1.0                # the mean is 0
    loo = nr.leave_one_out(['a', 'b', 'c', 'd', 'e'], [2.0, 2.5, 3.0, 3.5, 4.0])
    assert [r['without'] for r in loo['rows']] == ['a', 'b', 'c', 'd', 'e']
    assert loo['rows'][4]['mean_diff'] == 2.75                         # without the largest
    assert loo['all_above_zero'] is True and loo['min_lower'] > 0
    wide = nr.leave_one_out(['a', 'b', 'c', 'd'], [2.0, 2.5, 3.0, 20.0])
    assert wide['all_above_zero'] is False                             # three left, one far off
    e = nr.interval([1.0, 2.0, 3.0, -1.0])
    assert e['mean_diff'] == 1.25 and e['n_positive'] == 3 and e['ci95'][0] < 1.25 < e['ci95'][1]


def _spread(values):
    return {'micro': {'mean_diff': sum(values) / len(values), 'ci95': [0, 0], 'per_run': values}}


def test_unit_differences_and_pooled(tmp_path):
    """The 11 units are MuLMS's subfields and the fine-grained papers; MR3 minus the CRF head on
    MuLMS is derived from the control's two comparisons; the decision basis needs the decision
    parts; the fine-grained names-vs-MR3 cell comes from the week-12 comparisons file."""
    pytest.importorskip('scipy')
    subs = ['s1', 's2', 's3']
    papers = ['p1', 'p2']
    pooled = {'crf_mrm_topic6': {'corpora': {
                  'mulms': {'units': dict.fromkeys(subs, 50), 'minus_crf': {'per_run': [3, 4, 5]}},
                  'fine': {'units': dict.fromkeys(papers, 50), 'minus_crf': {'per_run': [4, 6]}}}},
              'crf_mr': {'corpora': {'mulms': {'minus_crf': {'per_run': [2, 2, 2]}},
                                     'fine': {'minus_crf': {'per_run': [3, 3]}}}}}
    control = lambda crf, mr: {'minus_crf': _spread(crf), 'minus_crf_mr': _spread(mr)}  # noqa: E731
    copies = {'mulms': {'ten_seeds': {'minus_crf_mr6': _spread([1, 1, 1]), 'control': control([2, 3, 4], [0, 1, 2])},
                        'decision': {'minus_crf_mr6': _spread([0.5, 0.5, 0.5]),
                                     'control': control([2.5, 3.5, 4.5], [0.5, 1.5, 2.5])}},
              'fine': {'minus_crf_mr6': _spread([1, 2]), 'control': control([3, 4], [0, 1])}}
    w12 = {'crf_mrm_topic6': {'ten_seeds': {'minus_crf': _spread([3, 4, 5]), 'minus_crf_mr': _spread([1, 2, 3])},
                              'decision': {'minus_crf': _spread([3, 4, 5]), 'minus_crf_mr': _spread([1, 2, 3])}}}
    analysis = {'week13': {'pooled': pooled, 'week12_winner': 'crf_mrm_topic6'},
                'week14': {'copies': copies}, 'mulms': {'week12': w12}}
    comparisons = {'ner_results_fine_leave_one_paper_out_crf_mrm6_vs_crf_mr': {'micro': {'per_run': [1, 3]}}}
    diffs = nr.unit_differences(analysis, comparisons)
    assert set(diffs) == {'ten', 'decision'}
    units, values = diffs['ten']['names_vs_crf']
    assert units == subs + papers and values == [3, 4, 5, 4, 6]
    assert diffs['ten']['mr3_vs_crf'][1] == [2, 2, 2, 3, 3]         # control minus (control minus MR3)
    assert diffs['decision']['mr6_vs_crf'][1] == [2.5, 3.5, 4.5, 3, 4]
    assert diffs['ten']['names_vs_mr3'][1] == [1, 2, 3, 1, 3]
    out = nr.pooled(analysis, comparisons)
    assert out['ten']['names_vs_crf']['n_positive'] == 5
    assert out['ten']['names_vs_crf']['leave_one_out']['all_above_zero'] is True
    assert out['ten']['units'] == subs + papers
    assert nr.unit_differences({}, {}) == {} and nr.pooled({}, {}) == {}
    assert nr.coverage_gain({}) == {} and nr.compute(str(tmp_path)) == {}
    (tmp_path / 'x.json').write_text(json.dumps({}))
