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


def test_random_split_protocol():
    assert nb.make_folds(_corpus(), SimpleNamespace(leave_one_paper_out=False)) is None
    tr, va, te = nb.random_split(476, 0)
    assert (len(tr), len(va), len(te)) == (309, 71, 96)
    assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)
    assert nb.random_split(476, 0) == (tr, va, te)
    assert nb.random_split(476, 1) != (tr, va, te)
