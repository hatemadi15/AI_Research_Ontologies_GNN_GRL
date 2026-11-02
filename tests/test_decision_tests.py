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
    strat = dt.stratified(rows)
    assert strat['D5'] == {'mean': 3.73, 'ci95': [1.39, 6.07], 'df': 8.3}
    assert strat["D5'"]['ci95'] == [1.46, 6.0] and strat['D7']['ci95'] == [-0.88, 2.69]


def test_verdict_labels():
    """D5' (a post hoc reading) and D11 (a pass only with the flagged subfields) carry their
    own verdict words; the other passes read "passes", the rest "no"."""
    pytest.importorskip('scipy')
    names = (dt.ANALYSIS, dt.WEEK12, dt.WEEK18, dt.ROBUST)
    if not all(os.path.exists(os.path.join(RESULTS, n)) for n in names):
        pytest.skip('results files not present')
    rows = {r['decision']: r for r in dt.decision_tests(*(dt.load(os.path.join(RESULTS, n))
                                                           for n in names))}
    assert rows["D5'"]['verdict'] == 'above 0 (post hoc)'
    assert rows['D11']['verdict'] == 'passes (as registered)'
    assert {k for k, r in rows.items() if r['verdict'] == 'passes'} == {'D3', 'D4', 'D5'}
    assert all(r['verdict'] == 'no' for r in rows.values() if not r['passes'])


def test_dependent_t_is_the_corrected_resampled_test():
    """With K units all correlated at 1/K, the variance of the mean is the corrected
    resampled t test's S^2 (1/K + 1/(K - 1)); with none correlated, the plain one."""
    pytest.importorskip('scipy')
    import math
    import statistics
    values = [1.99, 5.67, 3.64, 5.78]
    papers = dt.layout(4)
    e = dt.dependent_t(values, papers)
    s2 = statistics.variance(values)
    assert e['se_ratio'] == round(math.sqrt(s2 * (1 / 4 + 1 / 3)) / math.sqrt(s2 / 4), 3)
    assert e['ci95'] == [-0.13, 8.67] and e['correlated_pairs'] == 6
    alone = [('mulms', u) for u in ('graphene', 'pemfc', 'steel')]   # pools share no article
    assert dt.dependent_t([1.0, 2.0, 3.0], alone)['se_ratio'] == 1.0
    assert dt.pair_rho(('mulms', 'semi'), ('mulms', 'sofc')) == dt.SHARED_POOL_RHO
    assert dt.pair_rho(('mulms', 'semi'), ('materiominer', 0)) == 0.0
    assert dt.pair_rho(('sofc_exp', 0), ('sofc_exp', 1)) == 1 / 5
    assert len(dt.shared_pool_pairs()) == 5


def test_committed_dependence_and_reported():
    """On the committed results, with the units correlated D3's interval covers 0 and D4's,
    D5's and D11's stay above 0, and no failing decision clears 0; MR3's pooled P values."""
    pytest.importorskip('scipy')
    names = (dt.ANALYSIS, dt.WEEK12, dt.WEEK18, dt.ROBUST)
    if not all(os.path.exists(os.path.join(RESULTS, n)) for n in names):
        pytest.skip('results files not present')
    inputs = [dt.load(os.path.join(RESULTS, n)) for n in names]
    rows = dt.decision_tests(*inputs)
    dep = dt.dependence(rows)
    assert {k for k, e in dep.items() if e['above_0']} == {'D4', 'D5', "D5'", 'D11'}
    assert dep['D3']['ci95'] == [-0.13, 8.67] and dep['D5']['ci95'] == [0.27, 6.88]
    rep = dt.reported(inputs[3])
    assert rep['decision']['p_two_sided'] == 0.0038 and rep['ten']['p_two_sided'] == 0.004


def test_mulms_pools_match_the_folds():
    """MULMS_POOLS are the articles of ner_baseline.subfield_folds (split seed 0, pool 400),
    recomputed from data/processed/mulms when it is present."""
    import ner_baseline as nb
    if not os.path.isfile(os.path.join(nb.MULMS_DIR, 'subfields.json')):
        pytest.skip('MuLMS not converted here (python src/graph/mulms.py)')
    data = nb.load_split_corpus('mulms')
    for name, train, val, test in nb.subfield_folds(data, nb.mulms_subfields(), 0, 400):
        docs = []
        for _, _, doc in train + val:
            if doc[:-len('.conll')] not in docs:
                docs.append(doc[:-len('.conll')])
        assert set(docs) == set(dt.MULMS_POOLS[name]), name


def test_p_of_from_values_and_from_the_interval():
    """p_of gives the t test's two-sided P from per-unit values when an entry has them, and
    the same P from the mean and 95% interval when it has not."""
    pytest.importorskip('scipy')
    values = [1.0, 2.0, 3.0, 4.0]
    full = dt.t_test(values)
    assert dt.p_of({'per_run': values}) == full['p_two_sided']
    assert dt.p_of({'per_unit': {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0}}) == full['p_two_sided']
    from_ci = dt.p_of({'mean_diff': full['mean_diff'], 'ci95': full['ci95']}, n=4)
    assert abs(from_ci - full['p_two_sided']) < 0.002


def test_committed_tipping_inferences_and_readings():
    """On the committed results: the correlation at which each pass's adjusted interval
    reaches 0, the P values of the reported inferences, MR3's MaterioMiner decisions on the
    decision seeds (its stored all-seed decisions reproduced) and D5 with coarse papers."""
    pytest.importorskip('scipy')
    names = (dt.ANALYSIS, dt.WEEK12, dt.WEEK18, dt.ROBUST, dt.LEXICON, dt.MENTIONS)
    if not all(os.path.exists(os.path.join(RESULTS, n)) for n in names):
        pytest.skip('results files not present')
    analysis, week12, matched, robust, lexicon, mentions = (dt.load(os.path.join(RESULTS, n))
                                                            for n in names)
    rows = dt.decision_tests(analysis, week12, matched, robust)
    tip = dt.tipping(rows)
    assert {k: v['rho'] for k, v in tip.items()} == {'D3': 0.23, 'D4': 0.28, 'D5': 0.45,
                                                     "D5'": 0.54, 'D11': 0.53}
    # round 17: the bisection runs on the unrounded bound, which changes sign within 0.005 of
    # the reported correlation (D5's bound is +0.011 at 0.44, the rounded one's stopping point)
    by = {r['decision']: r for r in rows}
    for decision, e in tip.items():
        r, group = by[decision], dt.TIPPING[decision]
        units = dt.layout(r['n_units'])
        assert dt.dependent_lower(r['per_unit'], units, {group: e['rho'] - 0.005}) > 0
        assert dt.dependent_lower(r['per_unit'], units, {group: e['rho'] + 0.005}) < 0
    assert dt.dependent_lower(by['D5']['per_unit'], dt.layout(11), {'materiominer': 0.44}) > 0.005
    week5 = (dt.load(os.path.join(RESULTS, dt.WEEK5))
             if os.path.exists(os.path.join(RESULTS, dt.WEEK5)) else None)
    inf = dt.inferences(analysis, robust, matched, lexicon, mentions, week12, week5)
    assert inf['held_recall_names_vs_mr6']['p_two_sided'] == 0.0044
    assert inf['st3_votes_vs_mr3_votes_coarse']['p_two_sided'] == 0.004
    assert inf['mt_mr3_vs_crf_materiominer_fine']['p_two_sided'] == 0.0001
    if week5:
        assert inf['mt_vs_crf_materiominer_fine']['p_two_sided'] == 0.0331
    assert inf['mr3_vs_crf_16']['n_units'] == 16 and inf['mr3_vs_crf_16']['p_two_sided'] == 0.0184
    assert {k: inf[k]['p_two_sided'] for k in (
        'mr6_vs_crf_mulms_decision', 'd11_flagged_at_nearest_recall',
        'other_unseen_vs_their_best_bias', 'held_materials_at_0', 'held_other_types_at_0')} == {
        'mr6_vs_crf_mulms_decision': 0.0424, 'd11_flagged_at_nearest_recall': 0.0071,
        'other_unseen_vs_their_best_bias': 0.0009, 'held_materials_at_0': 0.0155,
        'held_other_types_at_0': 0.0099}
    if not os.path.exists(os.path.join(RESULTS, 'ner_results_fine_leave_one_paper_out_crf_mr_m300.json')):
        pytest.skip('run files not present')
    every = dt.mr3_materiominer(RESULTS, range(0, 1000, 100))
    assert every['fine']['ci95'] == [1.51, 4.03] and every['coarse']['ci95'] == [0.06, 2.51]
    seven = dt.mr3_materiominer(RESULTS)
    assert seven['fine']['ci95'] == [0.1, 5.03] and seven['coarse']['ci95'] == [-0.63, 4.08]
    assert dt.d5_coarse(rows)['ci95'] == [0.09, 5.25]
