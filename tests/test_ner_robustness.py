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


def test_mde_stratified_partial_and_detectable():
    """The detectable difference from a stored interval, the stratified mean over the two corpora,
    the partial correlation, and the per-comparison table from the pooled, grid and analysis parts."""
    pytest.importorskip('scipy')
    assert nr.mde(1.46, 10) == 2.04                                   # D7's half-width: 2.0 points
    assert nr.mde(4.32, 6) > nr.mde(1.83, 6)                          # a wider interval detects less
    st = nr.stratified(['s1', 's2', 'p1.conll', 'p2.conll'], [2.0, 4.0, 5.0, 7.0])
    assert st == {'corpus_means': {'mulms': 3.0, 'materiominer': 6.0}, 'mean': 4.5}
    assert nr.stratified(['s1', 's2'], [1.0, 2.0]) == {}              # one corpus: no stratified mean
    assert round(nr.partial(0.6, 0.5, 0.5), 4) == 0.4667
    robust = {'decision': {'names_vs_mr6': {'ci95': [-0.66, 2.26], 'n_units': 11}, 'units': []}}
    grids = {'fine': {'names_vs_crf': {'ci95': [1.39, 7.15]}}}
    analysis = {'week13': {'week12_winner': 'w'},
                'mulms': {'week12': {'w': {'decision': {'minus_crf': {'micro': {'ci95': [-1.13, 7.51]}}}}}},
                'week14': {'copies': {'mulms': {'decision': {}}}, 'topic': {'decision': {}},
                           'ablations': {'x': {'minus_w': {'micro': {'ci95': [-1.0, 1.0]}}}}},
                'week17': {'coarse': {'decision': '17.2',
                                      'minus_crf_mt_mr6': {'micro': {'ci95': [-1.0, 2.0]}},
                                      'names': {'minus_crf_mt_mr': {'micro': {'ci95': [0.0, 4.0]}}}}}}
    d = nr.detectable(analysis, robust, grids)
    assert d['pooled decision: names_vs_mr6'] == {'half_width': 1.46, 'df': 10, 'mde_80': 2.04}
    assert d['MaterioMiner fine: names_vs_crf']['df'] == 3
    assert d['D1 MuLMS: names_vs_crf']['half_width'] == 4.32
    assert 'ablation x: vs_method (3 seeds)' in d and 'D2 MuLMS: names_vs_mr3' not in d
    # week 17: the decision against the matched control and the names against multi-task + MR3,
    # 3 degrees of freedom (4 papers); absent without the section
    assert d['17.2 MaterioMiner coarse: crf_mt_mrm6_vs_crf_mt_mr6'] == {'half_width': 1.5, 'df': 3,
                                                                        'mde_80': nr.mde(1.5, 3)}
    assert d['MaterioMiner coarse: crf_mt_mrm6_vs_crf_mt_mr']['half_width'] == 2.0
    assert not [k for k in nr.detectable({}, {}, {}) if '17' in k]
    # week 18: the two decisions against the recall-matched control, also without the flagged
    # subfields (3 degrees of freedom here) and in the post hoc reading; absent without the file
    one = {'ci95': [3.31, 19.18], 'n_units': 7}
    matched = {'decisions': {'18.1': one, '18.2': {'ci95': [-0.79, 7.31], 'n_units': 7}},
               'without_flagged': {'decisions': {'18.1': {'ci95': [-3.33, 15.44], 'n_units': 4}}},
               'post_hoc_nearest': {'decisions': {'18.2': {'ci95': [-0.66, 3.41], 'n_units': 7}}}}
    d18 = nr.detectable({}, {}, {}, matched)
    assert d18['D11 MuLMS: recall_in_vs_recall_matched'] == {'half_width': 7.93, 'df': 6,
                                                            'mde_80': nr.mde(7.935, 6)}
    assert d18['D11 MuLMS: recall_in_vs_recall_matched without the flagged subfields']['df'] == 3
    assert 'D12 MuLMS: f1_vs_recall_matched (post hoc, nearest recall)' in d18
    assert not [k for k in nr.detectable({}, {}, {}) if 'D11' in k or 'D12' in k]
    # D6, SOFC-Exp's 5 article folds (4 degrees of freedom), from week13.folds
    d6 = nr.detectable({'week13': {'folds': {'crf_mrm6': {'minus_crf': {'micro': {'ci95': [-1.21, 1.14]}}}}}}, {}, {})
    assert d6['D6 SOFC-Exp: names_vs_crf'] == {'half_width': 1.17, 'df': 4, 'mde_80': 1.57}   # (1.14 + 1.21) / 2 rounds to 1.17


def test_job_costs_share_a_job_over_its_students():
    """Three students of one job share its runtime; the per-student cost divides it, and the
    harvest is the job less the students' own training."""
    runs = [{'run_id': 'a', 'runtime_sec': 900.0}] * 3 + [{'run_id': 'b', 'runtime_sec': 1200.0}] * 4
    out = nr.job_costs(runs, student_sec=100)
    assert out == {'n_jobs': 2, 'students_per_job': [3, 4], 'job_sec_mean': 1050,
                   'per_student_sec_mean': 300, 'harvest_sec_mean': 700, 'harvest_student_sec': 100}


def test_compute_means_and_ratios(tmp_path):
    def write(system, best, augmented):
        for off in nr.DECISION:
            run = {'runtime_sec': 100.0, 'best_epoch': best, 'n_train': 100}
            if augmented:
                run['n_augmented'] = augmented
            path = tmp_path / f'ner_results_{nr.MULMS}_{system}_m{off}.json'
            path.write_text(json.dumps({'runs': [run]}))
    write('crf', 7, None)
    write('crf_mr', 3, 200)
    out = nr.compute(str(tmp_path))
    assert out['crf']['epochs_to_best_mean'] == 8 and out['crf']['n_augmented_mean'] is None
    assert 'epochs_to_best_mean_exact' not in out['crf']
    assert out['crf_mr']['n_augmented_mean'] == 200
    assert out['crf_mr']['sentences_per_epoch_ratio'] == 3.0      # (100 + 200) / 100
    assert out['crf_mr']['updates_to_best_ratio'] == 1.5           # 3 x 4 epochs / 8
