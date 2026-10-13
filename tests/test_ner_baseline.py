"""NER baseline protocol helpers (no torch or model download needed).

The tiny-model run tests at the end need torch, transformers, seqeval and scipy
and are skipped without them.
"""
import json
import os
import sys
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


def test_leave_one_paper_out_folds_as_corpus_positions():
    data = _corpus()
    args = SimpleNamespace(leave_one_paper_out=True, split_seed=0)
    folds, positions = nb.make_folds(data, args), nb.make_folds(data, args, indices=True)
    for (doc, train, val, test), (doc_i, train_i, val_i, test_i) in zip(folds, positions):
        assert doc_i == doc
        assert [data[i] for i in train_i] == train and [data[i] for i in val_i] == val
        assert [data[i] for i in test_i] == test
        assert test_i == [i for i, x in enumerate(data) if x[2] == doc]    # corpus order
        assert sorted(train_i + val_i + test_i) == list(range(len(data)))
    assert nb.make_folds(data, SimpleNamespace(leave_one_paper_out=False), indices=True) is None


def test_prediction_fields_join_the_tags_of_each_sentence():
    test = [(['a', 'b'], ['B-X', 'O'], 'p.conll'), (['c'], ['O'], 'p.conll')]
    got = nb.prediction_fields((7, 3), test, [['B-X', 'I-X'], ['O']])
    assert got == {'test_index': [7, 3], 'test_gold': ['B-X O', 'O'],
                   'test_pred': ['B-X I-X', 'O']}
    assert all(type(i) is int for i in got['test_index'])


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


def test_variant_names_with_tags():
    def name(**kw):
        return nb.variant_name(SimpleNamespace(
            **{'class_weights': False, 'crf': False, 'tag': None, **kw}))
    assert name(tag='') == name() == 'baseline'
    assert name(tag='m100') == 'm100'                    # the baseline is named by the tag alone
    assert name(class_weights=True, tag='m100') == 'cw_m100'
    assert name(crf=True, tag='m100') == 'crf_m100'
    assert nb.variant_name(SimpleNamespace(class_weights=False, tag='x1')) == 'x1'   # no crf field
    # the file suffix follows the variant name; smoke tests append to it
    assert nb.result_suffix('random_split_fixed', 'baseline') == ''
    assert nb.result_suffix('random_split_fixed', 'crf_m100') == '_crf_m100'
    assert nb.result_suffix('random_split_vary', 'baseline') == '_random_split_vary'
    assert nb.result_suffix('leave_one_paper_out', 'm100') == '_leave_one_paper_out_m100'


def test_reference_variant():
    assert nb.reference_variant('crf_m100') == 'crf'
    assert nb.reference_variant('cw_m100') == 'cw'
    assert [nb.reference_variant(v) for v in ('cw', 'crf', 'm100', 'baseline')] == ['baseline'] * 4
    # only the underscore after crf / cw makes a tagged variant, not a shared prefix
    assert nb.reference_variant('crf2') == nb.reference_variant('cwx') == 'baseline'
    # the seeds of the week-3 families
    for variant in ('crf_dapt2_m100', 'crf_ls_m0', 'crf_st_m200', 'crf_mt_m100', 'crf_scibert'):
        assert nb.reference_variant(variant) == 'crf'


def test_result_file_names_parse_to_protocol_and_variant():
    parse = nb.parse_result_name
    # the baseline file of a protocol is no variant named like the protocol
    assert parse('ner_results_fine_random_split_vary.json') == (
        'fine', 'random_split_vary', 'baseline')
    assert parse('ner_results_coarse_leave_one_paper_out.json') == (
        'coarse', 'leave_one_paper_out', 'baseline')
    assert parse('ner_results_fine.json') == ('fine', 'random_split_fixed', 'baseline')
    assert parse('ner_results_fine_random_split_vary_cw.json') == (
        'fine', 'random_split_vary', 'cw')
    assert parse('ner_results_coarse_leave_one_paper_out_crf_m100.json') == (
        'coarse', 'leave_one_paper_out', 'crf_m100')
    # without a protocol part the file belongs to the fixed split
    assert parse('ner_results_fine_crf.json') == ('fine', 'random_split_fixed', 'crf')
    assert parse('ner_results_fine_m100.json') == ('fine', 'random_split_fixed', 'm100')
    # smoke-test files keep their marker in the variant; other files are no results
    assert parse('ner_results_fine_random_split_vary_crf_m100_smoke16.json') == (
        'fine', 'random_split_vary', 'crf_m100_smoke16')
    assert parse('ner_results_fine_random_split_vary_smoke16.json')[2] == 'smoke16'
    for fn in ('ner_comparisons.json', 'ner_results_fine.json.bak', 'ner_results_medium.json',
               'ner_results_fine_CRF.json', 'ner_results_fine_.json', 'ner_results_fine_a-b.json'):
        assert parse(fn) is None
    # every name a run can write reads back
    for protocol in ('random_split_fixed', 'random_split_vary', 'leave_one_paper_out'):
        for variant in ('baseline', 'cw', 'crf', 'm100', 'cw_m100', 'crf_m100', 'crf_dapt2_m100',
                        'crf_mt_m200', 'crf_ls_m0', 'crf_mix3', 'crf_pubmedbert'):
            fn = f'ner_results_coarse{nb.result_suffix(protocol, variant)}.json'
            assert parse(fn) == ('coarse', protocol, variant)


def test_every_results_file_parses_to_the_variant_it_holds():
    """The names of the files in results/ give the granularity, protocol and variant they store."""
    results = os.path.join(os.path.dirname(__file__), '..', 'results')
    names = sorted(fn for fn in os.listdir(results) if fn.startswith('ner_results_'))
    assert {'ner_results_fine.json', 'ner_results_coarse_leave_one_paper_out.json',
            'ner_results_fine_random_split_vary_crf.json',
            'ner_results_coarse_random_split_vary_crf_m100.json',
            'ner_results_fine_leave_one_paper_out_crf_ens3.json'} <= set(names)
    for fn in names:
        with open(os.path.join(results, fn)) as f:
            stored = json.load(f)
        # files written before variants existed hold baseline runs
        assert nb.parse_result_name(fn) == (stored['granularity'], stored['protocol'],
                                            stored.get('variant', 'baseline')), fn


def test_resume_isolation_for_seed_offset_and_predictions(tmp_path):
    path = tmp_path / 'ner_results_fine_random_split_vary_crf_m100.json'
    runs = [{'run_id': 'seed0', 'test_pred': ['O']}, {'run_id': 'seed1'}]   # seed1: no predictions

    def write(config):
        path.write_text(json.dumps({'labels': ['A'], 'model': 'm', 'variant': 'crf_m100',
                                    'config': config, 'runs': runs}))

    def resumable(*args, **kwargs):
        return nb.resumable_runs(str(path), ['A'], 'm', 'crf_m100', *args, **kwargs)

    write({'seed_offset': 100})
    assert resumable(100) == runs
    assert resumable(200) == [] and resumable(0) == [] and resumable() == []
    assert resumable(100, save_predictions=True) == runs[:1]
    assert resumable(200, save_predictions=True) == []
    write({'seed_offset': 100, 'save_predictions': False})
    assert resumable(100, save_predictions=True) == runs[:1]      # the stored flag is no matter
    # files written before offsets existed have none stored: offset 0
    write({})
    assert resumable() == runs and resumable(100) == []
    path.write_text('{"labels": ["A"], "model": "m", "variant": "crf_m100", "runs": []}')
    assert resumable() == [] and resumable(100) == []


def _stored(path, **fields):
    """A results file of two crf_m100 runs, seed offset 100; only seed0 stored predictions."""
    results = {'labels': ['A', 'B'], 'model': 'm', 'variant': 'crf_m100',
               'config': {'seed_offset': 100},
               'runs': [{'run_id': 'seed0', 'test_pred': ['O']}, {'run_id': 'seed1'}]}
    path.write_text(json.dumps({**results, **fields}))
    return str(path)


def test_check_overwrite_names_the_file_the_reason_and_the_fix(tmp_path):
    path = _stored(tmp_path / 'ner_results_fine_crf_m100.json')
    same = {'labels': ['A', 'B'], 'model': 'm', 'variant': 'crf_m100', 'seed_offset': 100}
    nb.check_overwrite(path, **same)                       # a matching file resumes as before
    assert nb.resumable_runs(path, **same) == json.loads(
        (tmp_path / 'ner_results_fine_crf_m100.json').read_text())['runs']
    for change, reason in (({'model': 'dapt'}, "model: 'm' in the file, 'dapt' now"),
                           ({'seed_offset': 200}, 'seed_offset: 100 in the file, 200 now'),
                           ({'variant': 'crf_x'}, "variant: 'crf_m100' in the file, 'crf_x' now"),
                           ({'labels': ['A', 'C']}, "labels: ['B'] only in the file, ['C'] only now"),
                           ({'labels': ['B', 'A']}, 'labels: the same classes in another order')):
        with pytest.raises(nb.OverwriteError) as err:
            nb.check_overwrite(path, **{**same, **change})
        message = str(err.value)
        assert message.startswith(f'{path}: ') and reason in message
        assert '(2 of 2 stored)' in message and '--tag NAME' in message and '--overwrite' in message
    with pytest.raises(nb.OverwriteError, match="'x' now; seed_offset: 100 in the file, 0 now"):
        nb.check_overwrite(path, **{**same, 'model': 'x', 'seed_offset': 0})   # every reason
    # --save-predictions: the runs that stored none would be replaced
    with pytest.raises(nb.OverwriteError,
                       match=r"runs \['seed1'\] have no stored predictions\) .* \(1 of 2 stored\)"):
        nb.check_overwrite(path, **same, save_predictions=True)
    # nothing to lose: no file, no runs, every run with predictions, a file from before variants
    nb.check_overwrite(str(tmp_path / 'missing.json'), **same)
    nb.check_overwrite(_stored(tmp_path / 'empty.json', runs=[]), **{**same, 'model': 'x'})
    nb.check_overwrite(_stored(tmp_path / 'pred.json', runs=[{'run_id': 's', 'test_pred': []}]),
                       **same, save_predictions=True)
    (tmp_path / 'old.json').write_text('{"labels": ["A"], "model": "m", "runs": [{"run_id": "s"}]}')
    nb.check_overwrite(str(tmp_path / 'old.json'), ['A'], 'm', 'baseline')


def test_results_files_may_be_replaced_with_overwrite_or_in_smoke_tests(tmp_path):
    files = [(_stored(tmp_path / 'f.json', model=nb.DEFAULT_MODEL), ['A', 'B'])]
    args = ['--crf', '--tag', 'm100', '--seed-offset']
    nb.check_results_files(nb.parse_args(args + ['100']), files)          # same offset: resumed
    with pytest.raises(nb.OverwriteError, match='seed_offset: 100 in the file, 200 now'):
        nb.check_results_files(nb.parse_args(args + ['200']), files)
    nb.check_results_files(nb.parse_args(args + ['200', '--overwrite']), files)
    nb.check_results_files(nb.parse_args(args + ['200', '--max-train', '16']), files)
    assert nb.parse_args([]).overwrite is False
    # how a run is invoked, not a setting of its results: not stored, so the config is as before
    assert 'overwrite' not in nb.stored_config(nb.parse_args(['--crf', '--overwrite']))


def test_run_stops_before_any_training(tmp_path, monkeypatch):
    """A coarse file that cannot be resumed stops a run of both granularities before fine trains."""
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: _toy_ner_corpus())
    monkeypatch.setattr(nb, 'train_one', lambda *args, **kwargs: pytest.fail('trained'))
    (tmp_path / 'ner').mkdir()
    coarse = tmp_path / 'ner' / 'ner_results_coarse_random_split_vary_crf.json'
    coarse.write_text(json.dumps({'labels': ['COND', 'MAT', 'PROP'], 'model': 'other',
                                  'variant': 'crf', 'runs': [{'run_id': 'seed0'}]}))
    before = coarse.read_text()
    argv = ['--crf', '--vary-split', '--seeds', '0']
    with pytest.raises(nb.OverwriteError, match=f"{coarse.name}: .*model: 'other' in the file, "
                                                f"'{nb.DEFAULT_MODEL}' now"):
        nb.run(nb.parse_args(argv))
    assert coarse.read_text() == before and os.listdir(tmp_path / 'ner') == [coarse.name]
    monkeypatch.setattr(sys, 'argv', ['ner_baseline.py', *argv])
    with pytest.raises(SystemExit) as exit_info:        # the command line reports it as an error
        nb.main()
    assert str(exit_info.value.code).startswith(f'error: {coarse}: ')


def test_seed_offset_needs_a_tag_and_tags_are_checked(capsys):
    args = nb.parse_args(['--crf', '--tag', 'm100', '--seed-offset', '100', '--save-predictions'])
    assert (args.tag, args.seed_offset, args.save_predictions) == ('m100', 100, True)
    args = nb.parse_args([])
    assert (args.tag, args.seed_offset, args.save_predictions) == (None, 0, False)
    assert nb.parse_args(['--tag', 'rerun']).seed_offset == 0      # a tag alone is fine
    assert nb.parse_args(['--seed-offset', '0']).tag is None
    for argv in (['--seed-offset', '100'], ['--crf', '--seed-offset', '-1']):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert '--seed-offset needs --tag' in capsys.readouterr().err
    for tag in ('M100', 'm_100', 'm-100', 'm 100', ''):
        with pytest.raises(SystemExit):
            nb.parse_args(['--tag', tag])
        assert 'invalid tag' in capsys.readouterr().err
    # a bare tag must not take the name of a plain variant: that file holds other results
    for tag in ('baseline', 'cw', 'crf'):
        with pytest.raises(SystemExit):
            nb.parse_args(['--tag', tag])
        assert 'would overwrite' in capsys.readouterr().err
    assert nb.parse_args(['--crf', '--tag', 'cw']).tag == 'cw'      # crf_cw is a variant of its own
    # model seeds are seed + offset and set_seed rejects negative ones
    with pytest.raises(SystemExit):
        nb.parse_args(['--tag', 'x', '--seed-offset', '-1'])
    assert 'must not be negative' in capsys.readouterr().err
    args = nb.parse_args(['--tag', 'x', '--seeds', '5', '6', '--seed-offset', '-5'])
    assert args.seed_offset == -5


def test_seed_family_tags(capsys):
    """<name>_m<N> tags name one seed of a week-3 family, e.g. in-domain pre-training."""
    args = nb.parse_args(['--crf', '--model', 'ckpt', '--tag', 'dapt2_m100', '--seed-offset', '100',
                          '--vary-split'])
    assert (nb.variant_name(args), args.seed_offset, args.model) == ('crf_dapt2_m100', 100, 'ckpt')
    assert nb.results_path('out', 'fine', args) == os.path.join(
        'out', 'ner_results_fine_random_split_vary_crf_dapt2_m100.json')
    assert nb.parse_args(['--crf', '--tag', 'xyz_m0']).tag == 'xyz_m0'
    assert nb.variant_name(nb.parse_args(['--tag', 'x2_m5'])) == 'x2_m5'        # baseline: the tag
    assert nb.variant_name(nb.parse_args(['--class-weights', '--tag', 'x2_m5'])) == 'cw_x2_m5'
    for tag in ('dapt2_m', 'dapt2_100', 'dapt2_n100', 'dapt2__m100', '_m100', 'dapt2_m100_m200',
                'dapt_2_m100', 'Dapt2_m100', 'dapt2-m100'):
        with pytest.raises(SystemExit):
            nb.parse_args(['--crf', '--tag', tag])
        assert 'invalid tag' in capsys.readouterr().err
    # the guards stay: a tag, or the name in a seed tag, must not give a plain variant
    for argv in (['--crf', '--tag', 'mt'], ['--crf', '--tag', 'mt_m100'], ['--tag', 'crf_m100'],
                 ['--tag', 'cw_m0'], ['--tag', 'baseline_m100'], ['--tag', 'crf'],
                 ['--crf', '--tag', 'ls_m0']):    # crf_ls_m0: a label-semantics seed
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert 'would overwrite' in capsys.readouterr().err
    # ... and the multi-task family keeps its own names
    args = nb.parse_args(['--crf', '--multi-task', '--tag', 'm100', '--seed-offset', '100'])
    assert nb.variant_name(args) == 'crf_mt_m100'
    assert nb.variant_name(nb.parse_args(['--crf', '--multi-task', '--tag', 'x_m100'])) == \
        'crf_mt_x_m100'


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
    assert c['reference'] == 'baseline' and 'baseline_file' not in c
    assert c['reference_file'] == 'ner_results_fine_random_split_vary.json'
    assert c['micro']['per_run'] == [2.0, 1.0]
    assert json.loads((tmp_path / 'ner_comparisons.json').read_text()) == comps


def test_compare_all_pairs_tagged_variants_with_their_reference(tmp_path, capsys):
    vary, lopo = 'ner_results_fine_random_split_vary', 'ner_results_fine_leave_one_paper_out'

    def write(name, run_id, micro, protocol, **extra):
        run = dict(_run(run_id, micro, micro + 2, micro - 10, micro - 1), **extra)
        (tmp_path / f'{name}.json').write_text(json.dumps({'protocol': protocol, 'runs': [run]}))

    write(vary, 'seed0', 70.0, 'random_split_vary')
    write(vary + '_crf', 'seed0', 72.0, 'random_split_vary')
    write(vary + '_crf_m100', 'seed0', 75.0, 'random_split_vary', seed=100)  # same split, new seed
    write(vary + '_x2', 'seed0', 73.0, 'random_split_vary')                  # bare tag
    write(vary + '_crf_m100_smoke16', 'seed0', 99.0, 'random_split_vary')
    write(vary + '_smoke16', 'seed0', 99.0, 'random_split_vary')
    write(lopo, 'a.conll', 60.0, 'leave_one_paper_out')
    write(lopo + '_cw', 'a.conll', 61.0, 'leave_one_paper_out')
    write(lopo + '_cw_m100', 'a.conll', 64.0, 'leave_one_paper_out')
    write('ner_results_coarse_crf_m100', 'seed0', 80.0, 'random_split_fixed')   # no coarse crf file
    (tmp_path / 'ner_comparisons.json').write_text('{"stale": {}}')
    comps = nb.compare_all(str(tmp_path))
    # the baselines, the smoke files and the old comparisons are not compared
    assert list(comps) == [lopo + '_cw', lopo + '_cw_m100', vary + '_crf', vary + '_crf_m100',
                           vary + '_x2']
    ref = {k: (c['variant'], c['reference'], c['reference_file']) for k, c in comps.items()}
    assert ref == {lopo + '_cw': ('cw', 'baseline', lopo + '.json'),
                   lopo + '_cw_m100': ('cw_m100', 'cw', lopo + '_cw.json'),
                   vary + '_crf': ('crf', 'baseline', vary + '.json'),
                   vary + '_crf_m100': ('crf_m100', 'crf', vary + '_crf.json'),
                   vary + '_x2': ('x2', 'baseline', vary + '.json')}
    diffs = {k: c['micro']['per_run'] for k, c in comps.items()}
    assert diffs == {lopo + '_cw': [1.0], lopo + '_cw_m100': [3.0], vary + '_crf': [2.0],
                     vary + '_crf_m100': [3.0], vary + '_x2': [3.0]}
    assert comps[vary + '_crf_m100']['run_ids'] == ['seed0']
    assert all('baseline_file' not in c and c['granularity'] == 'fine' for c in comps.values())
    assert comps[lopo + '_cw']['protocol'] == 'leave_one_paper_out'
    assert 'ner_results_coarse_crf_m100.json: no reference file ner_results_coarse_crf.json' \
        in capsys.readouterr().out
    assert json.loads((tmp_path / 'ner_comparisons.json').read_text()) == comps


def test_bio_constraints():
    labels = ['O', 'B-A', 'I-A', 'B-B', 'I-B']
    allowed, start = nb.bio_allowed(labels)
    ok = {(p, c) for i, p in enumerate(labels) for j, c in enumerate(labels) if allowed[i][j]}
    assert ('B-A', 'I-A') in ok and ('I-A', 'I-A') in ok and ('O', 'B-B') in ok
    assert ('O', 'I-A') not in ok and ('B-B', 'I-A') not in ok and ('I-B', 'I-A') not in ok
    assert start == [True, True, False, True, False]


def test_random_split_protocol():
    assert nb.make_folds(_corpus(), SimpleNamespace(leave_one_paper_out=False)) is None
    tr, va, te = nb.random_split(476, 0)
    assert (len(tr), len(va), len(te)) == (309, 71, 96)
    assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)
    assert nb.random_split(476, 0) == (tr, va, te)
    assert nb.random_split(476, 1) != (tr, va, te)


# --- runs with a tiny random model (torch, transformers, seqeval and scipy; skipped without)
TOY_WORDS = ['steel', 'alloy', 'tensile', 'strength', 'yield', 'room', 'temperature', 'high']
LEGACY_RUN_KEYS = {'f1', 'precision', 'recall', 'f1_strict', 'per_class', 'seed', 'best_epoch',
                   'val_f1', 'n_train', 'n_val', 'n_test', 'history', 'run_id', 'runtime_sec'}
PREDICTION_KEYS = {'test_index', 'test_gold', 'test_pred'}


def _toy_ner_corpus():
    """24 sentences of four 'papers'; the extra words make every length different.

    Each sentence has two MAT, two PROP and one COND entity, so `--top-k 2` drops COND.
    """
    data = []
    for p, doc in enumerate(('a.conll', 'b.conll', 'c.conll', 'd.conll')):
        for k in range(6):
            extra = 6 * p + k
            data.append((['steel', 'alloy', 'tensile', 'strengths', 'yield', 'room',
                          'temperature'] + ['high'] * extra,
                         ['B-MAT', 'B-MAT', 'B-PROP', 'I-PROP', 'B-PROP', 'B-COND',
                          'I-COND'] + ['O'] * extra, doc))
    return data


@pytest.fixture(scope='module')
def tiny_model(tmp_path_factory):
    """A tiny random BERT with a word-piece vocabulary ('strengths' is two subwords)."""
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    path = tmp_path_factory.mktemp('tiny_bert')
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', *TOY_WORDS, '##s']
    (path / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    transformers.BertTokenizerFast(str(path / 'vocab.txt')).save_pretrained(path)
    torch.manual_seed(0)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64)).save_pretrained(path)
    return str(path)


@pytest.fixture
def tiny_run(tiny_model, tmp_path, monkeypatch):
    """run(*flags): ner_baseline.run on the toy corpus (two classes kept) in tmp_path/ner."""
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: _toy_ner_corpus())

    def run(*flags):
        nb.run(nb.parse_args(['--granularity', 'fine', '--top-k', '2', '--model', tiny_model,
                              '--epochs', '1', '--threads', '1', '--device', 'cpu', *flags]))
    return run


def _load(tmp_path, name):
    return json.loads((tmp_path / 'ner' / name).read_text())


def _toy_data():
    """The toy corpus as `run` sees it: tags outside the kept classes mapped to O."""
    corpus = _toy_ner_corpus()
    labels, _ = nb.select_labels(corpus, 2)
    return labels, [(t, nb.restrict_tags(g, set(labels)), d) for t, g, d in corpus]


def test_run_stores_predictions_and_offsets_only_the_model_seed(tiny_run, tmp_path):
    tiny_run('--vary-split', '--seeds', '0', '1', '--crf', '--tag', 't1', '--seed-offset', '100',
             '--save-predictions')
    res = _load(tmp_path, 'ner_results_fine_random_split_vary_crf_t1.json')
    labels, data = _toy_data()
    assert labels == ['MAT', 'PROP']
    assert (res['variant'], res['protocol']) == ('crf_t1', 'random_split_vary')
    assert (res['config']['tag'], res['config']['seed_offset']) == ('t1', 100)
    assert [r['run_id'] for r in res['runs']] == ['seed0', 'seed1']      # run ids ignore the offset
    assert [r['seed'] for r in res['runs']] == [100, 101]                # the model seed
    for run, split_seed in zip(res['runs'], (0, 1)):                     # ... and so does the split
        index = run['test_index']
        assert set(run) == LEGACY_RUN_KEYS | PREDICTION_KEYS
        assert index == nb.random_split(len(data), split_seed)[2]
        assert run['n_test'] == len(index) == len(run['test_gold']) == len(run['test_pred'])
        # gold tags are those of the corpus at test_index, after restrict_tags (no COND left)
        assert run['test_gold'] == [' '.join(data[i][1]) for i in index]
        assert not any('COND' in gold for gold in run['test_gold'])
        # one predicted tag per word, from the kept classes (also for 'strengths', 2 subwords)
        assert all(len(pred.split()) == len(data[i][0]) for pred, i in zip(run['test_pred'], index))
        assert set(' '.join(run['test_pred']).split()) <= {
            'O', 'B-MAT', 'I-MAT', 'B-PROP', 'I-PROP'}


def test_leave_one_paper_out_run_stores_the_held_out_paper(tiny_run, tmp_path):
    tiny_run('--leave-one-paper-out', '--seeds', '3', '--tag', 't1', '--seed-offset', '10',
             '--save-predictions')
    res = _load(tmp_path, 'ner_results_fine_leave_one_paper_out_t1.json')
    _, data = _toy_data()
    assert res['variant'] == 't1'                                        # a bare tag, no cw / crf
    assert [r['run_id'] for r in res['runs']] == ['a.conll', 'b.conll', 'c.conll', 'd.conll']
    assert {r['seed'] for r in res['runs']} == {13}                      # seeds[0] + offset
    for run in res['runs']:
        index = run['test_index']
        assert index == [i for i, x in enumerate(data) if x[2] == run['run_id']]
        assert run['test_gold'] == [' '.join(data[i][1]) for i in index]
        assert [len(p.split()) for p in run['test_pred']] == [len(data[i][0]) for i in index]


def test_fixed_split_run_keeps_the_split_seed_under_an_offset(tiny_run, tmp_path):
    tiny_run('--split-seed', '3', '--seeds', '0', '1', '--tag', 't1', '--seed-offset', '7',
             '--save-predictions')
    res = _load(tmp_path, 'ner_results_fine_t1.json')
    _, data = _toy_data()
    assert [(r['run_id'], r['seed']) for r in res['runs']] == [('seed0', 7), ('seed1', 8)]
    assert all(r['test_index'] == nb.random_split(len(data), 3)[2] for r in res['runs'])


def test_run_resumes_only_matching_offsets_and_predictions(tiny_run, tmp_path, monkeypatch):
    calls = []
    train_one = nb.train_one

    def counting(seed, *args, **kwargs):
        calls.append(seed)
        return train_one(seed, *args, **kwargs)
    monkeypatch.setattr(nb, 'train_one', counting)
    baseline = tmp_path / 'ner' / 'ner_results_fine.json'

    tiny_run('--seeds', '0', '1')                  # the default path: fixed split, no extra fields
    res = _load(tmp_path, baseline.name)
    assert (res['variant'], res['protocol']) == ('baseline', 'random_split_fixed')
    assert [(r['run_id'], r['seed']) for r in res['runs']] == [('seed0', 0), ('seed1', 1)]
    assert all(set(r) == LEGACY_RUN_KEYS for r in res['runs'])
    assert res['config']['seed_offset'] == 0 and calls == [0, 1]
    tiny_run('--seeds', '0', '1')                  # finished runs are skipped ...
    assert calls == [0, 1]
    before = baseline.read_text()                  # ... runs without predictions are not reused
    with pytest.raises(nb.OverwriteError, match='have no stored predictions'):
        tiny_run('--seeds', '0', '1', '--save-predictions')   # and not replaced by accident
    assert calls == [0, 1] and baseline.read_text() == before
    tiny_run('--seeds', '0', '1', '--save-predictions', '--overwrite')
    assert calls == [0, 1, 0, 1]
    assert all(PREDICTION_KEYS <= set(r) for r in _load(tmp_path, baseline.name)['runs'])
    tiny_run('--seeds', '0', '1', '--save-predictions')
    assert calls == [0, 1, 0, 1]

    before = baseline.read_text()
    tiny_run('--seeds', '0', '1', '--tag', 't2', '--seed-offset', '5')   # its own file
    assert calls[4:] == [5, 6] and baseline.read_text() == before
    tagged = _load(tmp_path, 'ner_results_fine_t2.json')
    assert [(r['run_id'], r['seed']) for r in tagged['runs']] == [('seed0', 5), ('seed1', 6)]
    tiny_run('--seeds', '0', '1', '--tag', 't2', '--seed-offset', '5')   # same offset: reused
    assert calls[4:] == [5, 6]
    with pytest.raises(nb.OverwriteError, match='seed_offset: 5 in the file, 7 now'):
        tiny_run('--seeds', '0', '1', '--tag', 't2', '--seed-offset', '7')   # other offset: not mixed
    assert calls[4:] == [5, 6]
    tiny_run('--seeds', '0', '1', '--tag', 't2', '--seed-offset', '7', '--overwrite')
    assert calls[4:] == [5, 6, 7, 8]
    runs = _load(tmp_path, 'ner_results_fine_t2.json')['runs']
    assert [(r['run_id'], r['seed']) for r in runs] == [('seed0', 7), ('seed1', 8)]
    assert 'overwrite' not in _load(tmp_path, 'ner_results_fine_t2.json')['config']
