"""NER baseline protocol helpers (no torch or model download needed)."""
from types import SimpleNamespace

import pytest

import ner_baseline as nb


def _run(run_id, micro, strict, macro, weighted):
    return {'run_id': run_id, 'f1': micro, 'f1_strict': strict,
            'per_class': {'macro avg': {'f1-score': macro / 100},
                          'weighted avg': {'f1-score': weighted / 100}}}


def _corpus():
    """Toy corpus: one-token sentences from four 'papers' of different sizes."""
    data = []
    for doc, n in (('a.conll', 10), ('b.conll', 6), ('c.conll', 8), ('d.conll', 4)):
        data += [([f'{doc}-{i}'], ['O'], doc) for i in range(n)]
    return data


def test_summarize_reports_every_average_with_ci():
    res = {'runs': [_run('seed0', 70.0, 72.0, 60.0, 69.0),
                    _run('seed1', 74.0, 76.0, 64.0, 73.0)]}
    nb.summarize(res)
    # the old top-level keys keep their meaning (micro F1, population SD)
    assert (res['test_f1_mean'], res['test_f1_std'], res['test_f1_strict_mean']) == (72.0, 2.0, 74.0)
    s = res['summary']
    assert s['n_runs'] == 2 and s['run_ids'] == ['seed0', 'seed1']
    assert s['macro']['mean'] == 62.0 and s['weighted']['mean'] == 71.0
    assert s['micro']['per_run'] == [70.0, 74.0] and s['micro']['pop_sd'] == 2.0
    # t(0.975, df=1) = 12.71: two runs give a very wide interval
    lo, hi = s['micro']['ci95']
    assert lo == pytest.approx(72.0 - 12.706 * 2.8284 / 1.4142, abs=0.05)
    assert hi == pytest.approx(72.0 + 12.706 * 2.8284 / 1.4142, abs=0.05)


def test_summarize_ignores_empty_results():
    res = {'runs': []}
    assert 'summary' not in nb.summarize(res)


def test_leave_one_paper_out_tests_each_paper_once():
    data = _corpus()
    folds = nb.make_folds(data, SimpleNamespace(leave_one_paper_out=True, split_seed=0))
    assert [f[0] for f in folds] == ['a.conll', 'b.conll', 'c.conll', 'd.conll']
    for doc, train, val, test in folds:
        assert {x[2] for x in test} == {doc}
        assert len(test) == sum(1 for x in data if x[2] == doc)
        assert doc not in {x[2] for x in train + val}
        ids = [x[0][0] for x in train + val + test]
        assert len(ids) == len(set(ids)) == len(data)
        assert len(val) == round(0.15 * (len(data) - len(test)))


def test_variant_names_and_resume_isolation(tmp_path):
    assert nb.variant_name(SimpleNamespace(class_weights=False)) == 'baseline'
    assert nb.variant_name(SimpleNamespace(class_weights=True)) == 'cw'
    assert nb.variant_name(SimpleNamespace(class_weights=False, crf=True)) == 'crf'
    # a file written before variants existed holds baseline runs
    path = tmp_path / 'ner_results_fine_random_split_vary.json'
    path.write_text('{"labels": ["A"], "model": "m", "runs": [{"run_id": "seed0"}]}')
    assert nb.resumable_runs(str(path), ['A'], 'm', 'baseline') == [{'run_id': 'seed0'}]
    assert nb.resumable_runs(str(path), ['A'], 'm', 'cw') == []
    assert nb.resumable_runs(str(path), ['B'], 'm', 'baseline') == []
    assert nb.resumable_runs(str(tmp_path / 'missing.json'), ['A'], 'm', 'baseline') == []
    assert nb.VARIANT_FILE.match('ner_results_fine_random_split_vary_cw.json')
    assert nb.VARIANT_FILE.match('ner_results_coarse_leave_one_paper_out_crf.json')
    assert not nb.VARIANT_FILE.match('ner_results_fine_random_split_vary.json')
    assert not nb.VARIANT_FILE.match('ner_results_fine_random_split_vary_cw_smoke16.json')


def test_compare_to_baseline_pairs_runs_by_id():
    base = {'runs': [_run('seed0', 70.0, 72.0, 60.0, 69.0),
                     _run('seed1', 74.0, 76.0, 64.0, 73.0),
                     _run('seed2', 72.0, 74.0, 62.0, 71.0)]}
    variant = {'runs': [_run('seed1', 75.0, 77.0, 66.0, 74.0),
                        _run('seed0', 71.0, 73.0, 61.0, 70.0),
                        _run('seed2', 74.0, 75.0, 61.0, 72.0),
                        _run('seed9', 99.0, 99.0, 99.0, 99.0)]}   # no baseline partner
    comp = nb.compare_to_baseline(base, variant)
    assert comp['n_pairs'] == 3 and comp['run_ids'] == ['seed1', 'seed0', 'seed2']
    assert comp['micro']['per_run'] == [1.0, 1.0, 2.0]
    assert comp['micro']['mean_diff'] == pytest.approx(1.33, abs=0.01)
    assert comp['macro']['per_run'] == [2.0, 1.0, -1.0]
    lo, hi = comp['micro']['ci95']
    assert lo < 1.33 < hi


def test_compare_all_writes_comparisons(tmp_path):
    import json
    base = {'protocol': 'random_split_vary', 'runs': [_run('seed0', 70.0, 72.0, 60.0, 69.0),
                                                      _run('seed1', 74.0, 76.0, 64.0, 73.0)]}
    var = {'protocol': 'random_split_vary', 'runs': [_run('seed0', 72.0, 73.0, 61.0, 70.0),
                                                     _run('seed1', 75.0, 77.0, 66.0, 74.0)]}
    (tmp_path / 'ner_results_fine_random_split_vary.json').write_text(json.dumps(base))
    (tmp_path / 'ner_results_fine_random_split_vary_cw.json').write_text(json.dumps(var))
    (tmp_path / 'ner_results_coarse_leave_one_paper_out_crf.json').write_text(json.dumps(var))
    comps = nb.compare_all(str(tmp_path))
    assert list(comps) == ['ner_results_fine_random_split_vary_cw']   # no coarse baseline
    c = comps['ner_results_fine_random_split_vary_cw']
    assert (c['variant'], c['granularity'], c['n_pairs']) == ('cw', 'fine', 2)
    assert c['micro']['per_run'] == [2.0, 1.0]
    assert json.loads((tmp_path / 'ner_comparisons.json').read_text()) == comps


def test_random_split_protocol():
    assert nb.make_folds(_corpus(), SimpleNamespace(leave_one_paper_out=False)) is None
    tr, va, te = nb.random_split(476, 0)
    assert (len(tr), len(va), len(te)) == (309, 71, 96)
    assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)
    assert nb.random_split(476, 0) == (tr, va, te)
    assert nb.random_split(476, 1) != (tr, va, te)
