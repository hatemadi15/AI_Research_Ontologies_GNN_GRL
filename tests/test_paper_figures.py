"""The data behind the paper's figures (paper_figures.py); the drawing needs matplotlib, the
CIs scipy (skipped without)."""
import json

import pytest

import paper_figures as pf


def _write(directory, granularity, protocol, variant, scores):
    runs = [{'run_id': f'split{k}', 'f1': s, 'f1_strict': s,
             'per_class': {'macro avg': {'f1-score': s / 100},
                           'weighted avg': {'f1-score': s / 100}}}
            for k, s in enumerate(scores)]
    (directory / f'ner_results_{granularity}_{protocol}_{variant}.json').write_text(
        json.dumps({'runs': runs}))


def _families(tmp_path, gain=lambda off: 1.0):
    """crf and crf_mt (untagged at offset 0) on both protocols; crf_mt = crf + gain(offset)."""
    for g in ('fine', 'coarse'):
        for protocol, base in (('random_split_vary', [78, 76, 80, 77, 79]),
                               ('leave_one_paper_out', [50, 45, 48, 44])):
            for off in range(0, 1000, 100):
                scores = [b + (off % 300) / 100 for b in base]
                _write(tmp_path, g, protocol, f'crf_m{off}', scores)
                _write(tmp_path, g, protocol, 'crf_mt' if off == 0 else f'crf_mt_m{off}',
                       [s + gain(off) + 0.1 * k for k, s in enumerate(scores)])


def test_protocol_points_need_both_protocols(tmp_path):
    pytest.importorskip('scipy')
    _families(tmp_path)
    points = pf.protocol_points(tmp_path)
    assert [label for label, *_ in points['fine']] == ['CRF head', 'multi-task']
    (_, crf_random, crf_unseen), (_, mt_random, _) = points['fine']
    # 10 seeds of offsets 0, 1, 2, 0, 1, 2, ... (cycling by 300) added to each split
    assert crf_random[0] == pytest.approx(78 + 0.9)
    assert crf_unseen[0] == pytest.approx(46.75 + 0.9)
    assert crf_random[1] < crf_random[0] < crf_random[2]
    assert mt_random[0] == pytest.approx(78 + 0.9 + 1.0 + 0.2)


def test_sofc_points_need_both_protocols():
    """Figure 1's SOFC-Exp rows: the CRF head and its votes on random splits and article
    folds, from the analysis's sofc_splits; none unless both protocols are there."""
    def entry(m, v):
        return {'micro': {'mean': m, 'ci95': [m - 1, m + 1]},
                'crf_votes': {'micro': {'mean': v, 'ci95': [v - 1, v + 1]}}}
    analysis = {'sofc_splits': {'random_split_vary': entry(86.0, 87.0),
                                'document_folds': entry(83.0, 84.0), 'gap': {'micro': 3.0}}}
    assert pf.sofc_points(analysis) == [('CRF head', (86.0, 85.0, 87.0), (83.0, 82.0, 84.0)),
                                        ('CRF vote', (87.0, 86.0, 88.0), (84.0, 83.0, 85.0))]
    del analysis['sofc_splits']['document_folds']
    assert pf.sofc_points(analysis) == [] and pf.sofc_points({}) == []


def _mulms(pool, held, votes=None):
    """The analysis's mulms section: the CRF head's micro F1 on the pool splits and held-out
    subfields, and with votes (pool, held) its votes there."""
    def micro(m):
        return {'micro': {'mean': m, 'ci95': [m - 1, m + 1]}}
    out = {'random_split_pool': micro(pool),
           'leave_one_subfield_out': {'crf': micro(held), 'crf_mr': micro(held + 1)}}
    if votes:
        out['random_split_pool']['crf_votes'] = micro(votes[0])
        out['leave_one_subfield_out']['votes'] = {'crf': micro(votes[1])}
    return out


def test_mulms_points_need_both_protocols():
    """Figure 1's MuLMS rows: the CRF head on the random pool splits and the held-out
    subfields, and its votes if both protocols have them."""
    analysis = {'mulms': _mulms(62.0, 51.0, votes=(63.0, 53.0))}
    assert pf.mulms_points(analysis) == [('CRF head', (62.0, 61.0, 63.0), (51.0, 50.0, 52.0)),
                                         ('CRF vote', (63.0, 62.0, 64.0), (53.0, 52.0, 54.0))]
    del analysis['mulms']['random_split_pool']['crf_votes']
    assert [r[0] for r in pf.mulms_points(analysis)] == ['CRF head']
    del analysis['mulms']['random_split_pool']
    assert pf.mulms_points(analysis) == [] and pf.mulms_points({}) == []


def test_triple_differences(tmp_path):
    pytest.importorskip('scipy')
    _families(tmp_path, gain=lambda off: off / 900)       # seeds with larger offsets gain more
    v, r = (pf.family_files(tmp_path, 'fine', 'leave_one_paper_out', s)
            for s in ('crf_mt', 'crf'))
    means, passes, ten = pf.triple_differences(v, r)
    assert len(means) == len(passes) == 120
    assert ten == pytest.approx(0.5 + 0.15)               # mean gain 0.5, mean of 0.1 k 0.15
    # the screen's triple (offsets 0, 100, 200) has the smallest gain: 0.1 + 0.15
    assert means[0] == pytest.approx(min(means)) == pytest.approx(100 / 900 + 0.15)
    assert all(passes)                                    # every split gains on every triple


def test_figures_are_written(tmp_path):
    pytest.importorskip('matplotlib')
    pytest.importorskip('scipy')
    results = tmp_path / 'gpu'
    results.mkdir()
    _families(results)
    for g in ('fine', 'coarse'):
        for protocol in ('random_split_vary', 'leave_one_paper_out'):
            for off in range(0, 1000, 100):
                src = results / f'ner_results_{g}_{protocol}_crf_m{off}.json'
                (results / f'ner_results_{g}_{protocol}_crf_ctx_m{off}.json').write_text(
                    src.read_text())
    folds = {'micro': {'mean': 83.0, 'ci95': [82.0, 84.0]},
             'crf_votes': {'micro': {'mean': 84.0, 'ci95': [83.0, 85.0]}}}
    (results / pf.ANALYSIS).write_text(json.dumps({
        'sofc_splits': {'random_split_vary': folds, 'document_folds': folds},
        'mulms': _mulms(62.0, 25.0, votes=(63.0, 27.0)),     # an interval below 30 too
        'spread': {'fine_random_split_vary': {'crf': {'per_split': {'a': 1.5, 'b': 2.0}}}},
        'types': {'coarse_random_split_vary': {'crf_mt_ens3': {
            'SDE': {'f1': 17.7, 'support_per_split': 3.0},
            'MEC': {'f1': 80.0, 'support_per_split': 53.0}}}}}))
    out = tmp_path / 'figures'
    written = pf.main(['--results', str(results), '--out', str(out)])
    assert written == ['fig1_random_vs_unseen.png', 'fig2_seeds.png', 'fig3_types.png']
    assert all((out / name).stat().st_size > 10000 for name in written)


def test_threshold_data_and_figure4(tmp_path):
    """Figure 4 (C15): per held-out paper the mean F1 over seeds at each bias, and, pooled, the
    share of the entities added between two grid points that are right, against F1/2 at 0."""
    def run(run_id, counts):            # counts: {D: [correct, predicted, gold]} of one paper
        return {'run_id': run_id, 'bias_test': {d: {run_id: c} for d, c in counts.items()}}
    for off in range(0, 1000, 100):
        for g in ('fine', 'coarse'):
            runs = [run('p1.conll', {'0': [4, 10, 10], '1': [5, 14, 10], '-1': [3, 6, 10]}),
                    run('p2.conll', {'0': [6, 10, 10], '1': [6, 12, 10], '-1': [5, 8, 10]})]
            (tmp_path / f'ner_results_{g}_leave_one_paper_out_crf_bias_m{off}.json').write_text(
                json.dumps({'runs': runs}))
    sofc = {'runs': [{'run_id': f'seed{k}', 'bias_test': {
        '0': {'a1': [8, 10, 10], 'a2': [2, 2, 4]}, '1': {'a1': [8, 12, 10], 'a2': [2, 2, 4]}}}
        for k in range(3)]}
    (tmp_path / pf.SOFC_BIAS).write_text(json.dumps(sofc))
    curves, steps = pf.threshold_data(tmp_path)
    assert curves['fine']['p1'] == [(-1.0, 37.5), (0.0, 40.0), (1.0, pytest.approx(41.67,
                                                                                   abs=0.01))]
    points, breakeven = steps['fine-grained']
    # pooled over 2 papers and 10 seeds: from -1 to 0, 60 more predicted and 20 more right
    assert points == [(-0.5, pytest.approx(1 / 3)), (0.5, pytest.approx(1 / 6))]
    assert breakeven == 25.0
    points, breakeven = steps['SOFC-Exp']
    assert points == [(0.5, 0.0)] and breakeven == pytest.approx(100 * 30 / 78)
    pytest.importorskip('matplotlib')
    pf.figure4(curves, steps, tmp_path / 'fig4.png')
    assert (tmp_path / 'fig4.png').stat().st_size > 10000


def test_mention_data_and_figure5(tmp_path):
    """Figure 5 (C17): per setting the CRF head's shares of seen and unseen test entities and
    its recall on each; on held-out papers (fine-grained) each method's paired differences,
    the screen's marked, the settings and methods without results left out."""
    def kinds(seen, unseen, recall_seen, recall_unseen):
        return {'seen': {'share': seen, 'recall': recall_seen},
                'seen_other': {'share': round(1 - seen - unseen, 3), 'recall': 0.0},
                'unseen': {'share': unseen, 'recall': recall_unseen},
                'all': {'recall': seen * recall_seen + unseen * recall_unseen}}

    def diff(mean):
        return {'mean_diff': mean, 'ci95': [mean - 1, mean + 1]}
    mentions = {'systems': {'fine_random_split_vary': {'crf': kinds(0.65, 0.345, 88, 59)},
                            'fine_leave_one_paper_out': {'crf': kinds(0.17, 0.81, 71, 35)},
                            'sofc_published_split': {'crf': kinds(0.51, 0.49, 89, 82)}},
                'methods': {'fine_leave_one_paper_out': {
                    'crf_mt': {'seeds': list(range(0, 1000, 100)), 'seen': diff(3),
                               'unseen': diff(4)},
                    'crf_mt_ens3': {'seeds': [0, 300, 600], 'seen': diff(3), 'unseen': diff(4)},
                    'crf_mt_st3': {'seeds': [0, 100, 200], 'seen': diff(5), 'unseen': diff(6)},
                    'crf_unknown': {'seeds': [0], 'seen': diff(0), 'unseen': diff(0)}}}}
    settings, methods = pf.mention_data(mentions)
    assert [s[0] for s in settings] == ['fine, random splits', 'fine, unseen papers',
                                        'SOFC-Exp, official split']
    assert settings[1][1] == {'seen': 0.17, 'seen_other': 0.02, 'unseen': 0.81}
    assert settings[1][2] == (71, 35)
    assert methods == [('multi-task', (3, 2, 4), (4, 3, 5)),
                       ('multi-task vote', (3, 2, 4), (4, 3, 5)),
                       ('multi-task self-training (screen, 3 seeds)', (5, 4, 6), (6, 5, 7))]
    assert pf.mention_data({}) == ([], [])
    pytest.importorskip('matplotlib')
    pf.figure5(settings, methods, tmp_path / 'fig5.png')
    assert (tmp_path / 'fig5.png').stat().st_size > 10000


def test_gain_data_and_figure6(tmp_path):
    """Figure 6: a method's 10-seed comparison on the held-out papers, else its screen, in
    FIG6_METHODS order, and the third paper last; comparisons against another reference or
    on the random splits are left out."""
    def entry(g, variant, mean, reference='crf', protocol='leave_one_paper_out'):
        return {'granularity': g, 'protocol': protocol, 'variant': variant,
                'reference': reference, 'micro': {'mean_diff': mean,
                                                  'ci95': [mean - 1, mean + 1]}}
    (tmp_path / 'ner_week5_comparisons.json').write_text(json.dumps({
        'a': entry('fine', 'crf_mt', 3.7), 'b': entry('fine', 'crf_ens3', 2.4),
        'c': entry('fine', 'crf_mt_ens3', 3.6, reference='crf_ens3'),
        'd': entry('fine', 'crf_ls', 9.9, protocol='random_split_vary')}))
    (tmp_path / 'ner_week7_screen.json').write_text(json.dumps({
        'a': entry('fine', 'crf_mr', 3.25), 'b': entry('coarse', 'crf_mr', 0.25)}))
    (tmp_path / 'ner_week7_comparisons.json').write_text(json.dumps({
        'a': entry('coarse', 'crf_mr', 0.5)}))
    analysis = {'paper_count': {'fine': {'three_minus_two': {'mean_diff': 7.36,
                                                             'ci95': [1.98, 12.74]}}}}
    gains = pf.gain_data(tmp_path, analysis)
    assert gains['fine'] == [('seed vote', 2.4, 1.4, 3.4, 'ten'),
                             ('multi-task', 3.7, 2.7, 4.7, 'ten'),
                             ('mention replacement', 3.25, 2.25, 4.25, 'screen'),
                             ('a third training paper', 7.36, 1.98, 12.74, 'papers')]
    assert gains['coarse'] == [('mention replacement', 0.5, -0.5, 1.5, 'ten')]
    assert pf.gain_data(tmp_path / 'missing', {}) == {}
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    pf.figure6(gains, tmp_path / 'fig6.png')
    assert (tmp_path / 'fig6.png').stat().st_size > 0
