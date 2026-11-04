"""The post hoc checks of paper B's decisions (paper_b_checks.py): Benjamini-Hochberg, Holm,
the corrected resampled t test, the Welch gap and the exact rounding on small cases; the
unseen-minus-seen reading with and without per-unit values; and on the committed results the
enumeration of Section 7.8's 118 decisions with its counts, every stored interval reproduced,
the dependence entries of D5, D11 and weeks 20 and 21, the hardware re-decisions, the gaps
and the written file."""
import json
import os
import statistics
from fractions import Fraction

import pytest

import paper_b_checks as pb

ROOT = os.path.join(os.path.dirname(__file__), '..', 'results')
NEEDED = (pb.WEEK2, pb.ANALYSIS, pb.WEEK12, pb.WEEK18, pb.TESTS, pb.MENTIONS)


def committed():
    if not all(os.path.exists(os.path.join(ROOT, n)) for n in NEEDED):
        pytest.skip('results files not present')
    return pb.enumerate_decisions(ROOT)


@pytest.fixture(scope='module')
def out():
    pytest.importorskip('scipy')
    committed()
    return pb.checks(ROOT)


def by_id(decisions):
    return {d['id']: d for d in decisions}


def test_benjamini_hochberg_and_holm_small_case():
    p = [0.01, 0.04, 0.03, 0.005]
    # BH: sorted 0.005, 0.01, 0.03, 0.04 -> 4p/rank 0.02, 0.02, 0.04, 0.04, monotone from the top
    assert [round(q, 6) for q in pb.benjamini_hochberg(p)] == [0.02, 0.04, 0.04, 0.02]
    # Holm: (k - j) p in order 0.02, 0.03, 0.06, 0.04, cumulative maximum
    assert [round(x, 6) for x in pb.holm(p)] == [0.03, 0.06, 0.06, 0.02]
    assert [x <= pb.ALPHA for x in pb.holm(p)] == [False, False, False, True]
    assert pb.holm([0.5]) == [0.5] and pb.benjamini_hochberg([0.5]) == [0.5]


def test_corrected_resampled_t_small_case():
    pytest.importorskip('scipy')
    import decision_tests as dt
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    e = pb.corrected_resampled_t(values, 0.25)
    # variance of the mean (1/5 + 1/4) * 2.5 = 1.125; the plain one 2.5 / 5 = 0.5
    assert e['se_ratio'] == 1.5 and e['df'] == 4 and e['mean_diff'] == 3.0
    assert e['ci95'] == [0.06, 5.94]                      # 3 +- 2.776 * sqrt(1.125)
    # without the training-test overlap term it is the plain t test over the 5 splits
    plain = pb.corrected_resampled_t(values, 0.0)
    assert plain['ci95'] == dt.t_test(values)['ci95'] and plain['se_ratio'] == 1.0


def test_welch_matches_scipy():
    pytest.importorskip('scipy')
    from scipy import stats
    a, b = [86.65, 84.81, 86.7, 85.41, 87.76], [82.85, 88.76, 82.28, 79.77, 80.27]
    w = pb.welch(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert w['gap'] == 3.48 and w['df'] == round(ref.df, 1)
    assert w['p_two_sided'] == float(f'{ref.pvalue:.4g}') and w['ci95'] == [-0.9, 7.86]


def test_exact_means_break_ties_upwards():
    """A mean of two-decimal scores can be an exact tie at the third decimal, which a float sum
    rounds either way: week 3's self-training on the CPU, 2.915 exactly, reads 2.91 by float."""
    assert pb.exact(77.95) == Fraction(7795, 100)
    diffs = [Fraction(909, 300), Fraction(1831, 300), Fraction(299, 300), Fraction(459, 300)]
    assert sum(diffs) / 4 == Fraction(583, 200)
    # the same per-paper differences of the seed means in floating point, as the runs give them
    floats = [3.029999999999994, 6.103333333333332, 0.9966666666666697, 1.5300000000000011]
    assert all(abs(f - float(x)) < 1e-12 for f, x in zip(floats, diffs))
    assert round(statistics.fmean(floats), 2) == 2.91
    assert pb.half_up(sum(diffs) / 4) == 2.92 and pb.half_up(Fraction(-17, 200)) == -0.09


def test_unseen_minus_seen_needs_per_unit_values():
    summary = {'methods': {'fine_leave_one_paper_out': {'crf_x': {
        'reference': 'crf', 'unseen': {'recall': {'mean_diff': 2.0, 'ci95': [0.0, 4.0]}},
        'seen': {'recall': {'mean_diff': 1.0, 'ci95': [-1.0, 3.0]}}}}}}
    e = pb.unseen_minus_seen(summary)
    assert e['with_per_unit_values'] == 0 and e['summaries_only'] == 1 and 'methods' not in e
    assert e['note'].startswith('post hoc, not computed')
    pytest.importorskip('scipy')
    units = {'unseen': {'recall': {'per_unit': {'a': 3.0, 'b': 5.0, 'c': 4.0}}},
             'seen': {'recall': {'per_unit': {'a': 1.0, 'b': 2.0, 'c': 3.0}}}}
    e = pb.unseen_minus_seen({'methods': {'s': {'m': {'reference': 'crf', **units}}}})
    pair = e['methods']['s/m']
    assert pair['mean_diff'] == 2.0 and pair['units'] == ['a', 'b', 'c']    # 2, 3, 1


def test_committed_enumeration_matches_section_7_8():
    """Section 7.8's count from the stored decisions: 118, 27 passing, by design and by family;
    Table 6's 4 repeats of Table 2 rows agree with them; no scipy needed."""
    decisions, repeats = committed()
    found = pb.check_counts(decisions)
    assert found['decisions'] == 118 and found['passes'] == 27
    assert found['by_design'] == {pb.ONE_RUN: [6, 0], pb.THREE: [24, 6], pb.SEVEN_OR_TEN: [88, 21]}
    sources = [d['source'].split(',')[0] for d in decisions]
    assert sources.count('Table 2') == 70 and sources.count('Table 6') == 16
    assert len(repeats) == 4 and {r['label'] for r in repeats} == {None, 'D3', 'D4'}
    assert all(r['same_mean'] and r['same_interval'] and r['same_per_unit'] and r['same_verdict']
               for r in repeats)
    sizes = {'random_split_vary': 5, 'leave_one_paper_out': 4, 'leave_one_subfield_out': 7,
             'document_folds': 5, 'pooled': 11, 'published_split': 10, 'article_draws': 5}
    assert all(d['n_units'] == sizes[d['protocol']] for d in decisions)
    official = [d for d in decisions if d['protocol'] == 'published_split']
    assert len(official) == 12 and {d['label'] for d in official if d['passes']} == {
        'week 6, S1', 'week 7, part 2e, decision 1', 'week 7, part 2e, decision 2'}
    labels = {d['label'] for d in decisions if d['label'] and d['label'].startswith('D')}
    assert labels == {f'D{i}' for i in range(1, 13)}
    passing = {d['label'] for d in decisions if d['passes'] and d['label']}
    assert {'D3', 'D4', 'D5', 'D11', 'week 10, decision 2', 'week 13, 13.1a'} <= passing
    # weeks 20 and 21: their own rows of the by-addition table, 7 or 10 seeds
    late = {d['label']: d for d in decisions if d['week'] in (20, 21)}
    assert sorted(late) == ['week 20, 20.1', 'week 20, 20.2', 'week 20, 20.3', 'week 21, 21.1',
                            'week 21, 21.2', 'week 21, 21.3', 'week 21, 21.4']
    assert {k for k, d in late.items() if d['passes']} == {'week 20, 20.1', 'week 20, 20.2',
                                                          'week 20, 20.3'}
    assert {d['family'] for k, d in late.items() if k.startswith('week 20')} == {pb.W20}
    assert {d['family'] for k, d in late.items() if k.startswith('week 21')} == {pb.W21}
    assert [late[f'week 20, 20.{i}']['seeds'] for i in (1, 2, 3)] == [pb.TEN, pb.SEVEN, pb.SEVEN]
    assert late['week 21, 21.2']['reading'] == 'below the CRF head'
    assert late['week 21, 21.4']['protocol'] == 'document_folds'
    # week 8's 3-seed decisions on a third paper keep their own group
    assert {d['family'] for d in decisions if d['week'] == 8} == {
        'a third paper at equal size (3 seeds)'}


def test_counts_name_each_discrepancy():
    decisions, _ = committed()
    flipped = [dict(d) for d in decisions]
    flipped[0]['passes'] = True                       # week 3's mixed-backbone vote, fine
    with pytest.raises(pb.CountMismatch) as err:
        pb.check_counts(flipped)
    text = str(err.value)
    assert 'passes: enumerated 28, Section 7.8 states 27' in text
    assert "by_design/3 seeds: enumerated [24, 7], Section 7.8 states [24, 6]" in text
    assert "three_seed_additions: enumerated" in text
    with pytest.raises(pb.CountMismatch, match='decisions: enumerated 117'):
        pb.check_counts(decisions[1:])


def test_committed_intervals_dependence_and_multiplicity(out):
    decisions = out['decisions']
    assert out['interval_check']['mismatched'] == [] and out['interval_check']['matched'] == 118
    assert all(d['p_agrees'] for d in decisions)
    assert all(e['equal'] for e in out['dependence_check'].values())
    assert set(out['dependence_check']) == {'D5', 'D11', 'week 20, 20.1', 'week 20, 20.2',
                                            'week 20, 20.3', 'week 21, 21.1', 'week 21, 21.2',
                                            'week 21, 21.3', 'week 21, 21.4'}
    summary = out['dependence_summary']
    assert summary['passes'] == {'above_0': 8, 'covers_0': 16, 'uncorrected': 3}
    assert summary['random_split_ratios'] == [0.3107]       # 96 / 309 on every random split
    above = {d['label'] or d['id'] for d in decisions
             if d['passes'] and d['dependence'].get('above_0')}
    assert above == {'w3 CPU MaterioMiner coarse, unseen papers: crf_ls vs crf',
                     'w7 MaterioMiner fine, unseen papers: crf_mr vs crf', 'D4', 'D5', 'D11',
                     'week 13, 13.1a', 'week 8, part 2, decision 2', 'week 20, 20.3'}
    ids = by_id(decisions)
    d3 = ids['w12 MaterioMiner fine, unseen papers: crf_mrm6 vs crf']
    assert d3['dependence']['ci95'] == [-0.13, 8.67] and d3['detectable_80']['difference'] == 3.77
    mt = ids['w5 MaterioMiner fine, random splits: crf_mt vs crf']
    assert mt['dependence']['se_ratio'] == 1.598 and mt['dependence']['ci95'] == [-0.59, 4.24]
    assert not any(d['dependence']['corrected'] for d in decisions
                   if d['protocol'] in ('published_split', 'article_draws') or d['week'] == 2)
    mult = out['multiplicity']
    assert mult['benjamini_hochberg']['m'] == 118
    assert mult['benjamini_hochberg']['fdr_0.05']['n'] == 3
    # every pass at 10%: the 27th smallest P, 0.0223, is under its step-up threshold 27/118 x 10%
    assert mult['benjamini_hochberg']['fdr_0.10']['n'] == 27
    survivors = {d['label'] or d['id'] for d in decisions if d['holm_survives']}
    assert survivors == {'w3 CPU MaterioMiner coarse, unseen papers: crf_ls vs crf',
                         'week 8, part 2, decision 1', 'week 8, part 2, decision 2', 'week 6, S1',
                         'week 7, part 2e, decision 1', 'week 7, part 2e, decision 2',
                         'week 20, 20.1', 'week 20, 20.2', 'week 20, 20.3'}
    assert all(d['passes'] for d in decisions if d['bh_q'] <= 0.10 or d['holm_survives'])
    assert mult['holm']['families'][pb.W21] == {'decisions': 4, 'passes': 0, 'survivors': []}
    assert mult['chance'] == {'alpha': 0.025, 'decisions': 118, 'expected_passes': 2.95,
                              'passes': 27, 'share_of_passes': 0.109}


def test_committed_hardware_and_gaps(out):
    """The four 3-seed passes of weeks 3 and 4 on both machines, seeds 0-200 (the GPU pass's CPU
    side is week 3's self-training, the only one run on the CPU); every CPU run against its GPU
    repeat; and the gaps, whose MuLMS and SOFC-Exp points reproduce the analysis file's."""
    got = {(item['decision'], m): (item[m]['mean_diff'], item[m]['ci95'])
           for item in out['hardware']['passes'] for m in ('cpu', 'gpu')}
    ls, mt, vote, st = ('w3 CPU MaterioMiner coarse, unseen papers: crf_ls vs crf',
                        'w3 CPU MaterioMiner coarse, random splits: crf_mt vs crf',
                        'w4 CPU MaterioMiner coarse, unseen papers: crf_ls_ens3 vs crf_ens3',
                        'w4 MaterioMiner fine, unseen papers: crf_st3 vs crf')
    assert got == {(ls, 'cpu'): (1.17, [0.94, 1.41]), (ls, 'gpu'): (0.07, [-2.47, 2.61]),
                   (mt, 'cpu'): (0.75, [0.1, 1.39]), (mt, 'gpu'): (-0.08, [-1.71, 1.55]),
                   (vote, 'cpu'): (1.63, [0.11, 3.16]), (vote, 'gpu'): (0.4, [-1.36, 2.16]),
                   (st, 'cpu'): (2.92, [-0.73, 6.56]), (st, 'gpu'): (1.07, [0.33, 1.8])}
    st_item = next(i for i in out['hardware']['passes'] if i['decision'] == st)
    assert st_item['cpu']['variant'] == 'crf_st' and st_item['nearest']['gpu']['mean_diff'] == 2.14
    # the same seeds on both machines and both sides: 0, 100 and 200 on a paper, K, K + 100 and
    # K + 200 on random split K; the votes' runs record none
    for item in out['hardware']['passes']:
        assert item['cpu']['seeds'] == item['gpu']['seeds']
        assert item['cpu']['seeds']['variant'] == item['cpu']['seeds']['reference']
    assert st_item['gpu']['seeds']['variant'] == [0, 100, 200]
    assert next(i for i in out['hardware']['passes'] if i['decision'] == vote)['cpu']['seeds'] == {
        'variant': [], 'reference': []}
    runs = out['hardware']['runs']
    assert (runs['files'], runs['runs'], runs['identical']) == (42, 186, 0)
    assert runs['abs_diff_f1'] == {'median': 1.43, 'mean': 2.0, 'max': 10.08}
    assert runs['skipped_files'] == {'votes': 10, 'no_seed': 0, 'no_runs': 1}
    gaps = out['gaps']
    assert gaps['MuLMS']['reproduces'] and gaps['SOFC-Exp']['reproduces']
    assert (gaps['MuLMS']['gap'], gaps['MuLMS']['ci95']) == (17.06, [9.31, 24.81])
    assert (gaps['SOFC-Exp']['gap'], gaps['SOFC-Exp']['ci95']) == (3.48, [-0.9, 7.86])
    assert (gaps['MaterioMiner fine']['gap'], gaps['MaterioMiner coarse']['gap']) == (31.03, 18.36)
    # the seed means behind MaterioMiner's gaps are the CRF head's stored 10-seed means
    week5 = pb.load(os.path.join(ROOT, 'gpu', 'ner_week5_comparisons.json'))
    for g in ('fine', 'coarse'):
        for side, key in (('random_splits', f'ner_results_{g}_random_split_vary_crf_mt'),
                          ('held_out', f'ner_results_{g}_leave_one_paper_out_crf_ls')):
            stored = week5[key]['reference_scores']['micro']['per_run']
            assert gaps[f'MaterioMiner {g}'][side]['per_unit'] == stored
    seen = out['unseen_minus_seen']
    assert seen['entries'] == 82 and seen['with_per_unit_values'] == 0 and 'methods' not in seen


def test_written_file_is_current(out):
    """results/gpu/paper_b_checks.json is what the module computes from the committed results."""
    path = os.path.join(ROOT, pb.OUT)
    if not os.path.exists(path):
        pytest.skip('paper_b_checks.json not written')
    assert pb.load(path) == json.loads(json.dumps(out))
