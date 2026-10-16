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
    (results / pf.ANALYSIS).write_text(json.dumps({
        'spread': {'fine_random_split_vary': {'crf': {'per_split': {'a': 1.5, 'b': 2.0}}}},
        'types': {'coarse_random_split_vary': {'crf_mt_ens3': {
            'SDE': {'f1': 17.7, 'support_per_split': 3.0},
            'MEC': {'f1': 80.0, 'support_per_split': 53.0}}}}}))
    out = tmp_path / 'figures'
    written = pf.main(['--results', str(results), '--out', str(out)])
    assert written == ['fig1_random_vs_unseen.png', 'fig2_seeds.png', 'fig3_types.png']
    assert all((out / name).stat().st_size > 10000 for name in written)
