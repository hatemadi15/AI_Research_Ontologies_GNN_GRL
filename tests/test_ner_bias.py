"""Choosing the entity bias of each test set (ner_bias.py) from --entity-bias results files.
The choice is pure Python; writing the runs rescores them with seqeval (skipped without)."""
import itertools
import json

import pytest

import ner_bias as eb

PAPERS = ['a.conll', 'b.conll', 'c.conll', 'd.conll']
KEYS = ['-1', '0', '1', '2']
# inner F1 per scored paper and bias (counts [c, 10, 10] give F1 10 * c): paper a peaks at 2,
# the others at 1, so a paper's choice depends on whether paper a is among its inner papers
INNER_C = {True: {'-1': 1, '0': 2, '1': 3, '2': 9}, False: {'-1': 1, '0': 2, '1': 5, '2': 4}}


def _inner(variant='crf_bias_m0', drop=None):
    runs = []
    for pair in itertools.combinations(PAPERS, 2):
        if pair == drop:
            continue
        runs.append({'run_id': '+'.join(pair),
                     'bias_test': {k: {p: [INNER_C[p == 'a.conll'][k], 10, 10] for p in pair}
                                   for k in KEYS},
                     'bias_val': {k: [1, 1, 1] for k in KEYS}})
    return {'granularity': 'fine', 'protocol': 'leave_two_papers_out', 'variant': variant,
            'labels': ['MAT'], 'runs': runs}


# the test predictions of every outer run (two sentences, gold 'B-MAT O' twice), per bias
PRED = {'-1': ['O O', 'O O'], '0': ['O O', 'B-MAT O'], '1': ['B-MAT O', 'B-MAT O'],
        '2': ['B-MAT B-MAT', 'B-MAT O']}
COUNTS = {'-1': [0, 0, 2], '0': [1, 1, 2], '1': [2, 2, 2], '2': [2, 3, 2]}


def _outer(variant='crf_bias_m0', protocol='leave_one_paper_out', val=None):
    runs = []
    for paper in PAPERS:
        runs.append({'run_id': paper, 'f1': 66.67, 'f1_strict': 66.67, 'test_index': [0, 1],
                     'test_gold': ['B-MAT O', 'B-MAT O'], 'test_pred': PRED['0'],
                     'test_pred_bias': PRED, 'bias_test': {k: {paper: COUNTS[k]} for k in KEYS},
                     'bias_val': val or {k: [1, 1, 1] for k in KEYS}})
    return {'granularity': 'fine', 'protocol': protocol, 'variant': variant, 'model': 'm',
            'labels': ['MAT'], 'label_tie': None, 'paper_test_f1': 69.92, 'runs': runs}


def test_best_bias_breaks_ties_towards_zero():
    assert eb.best_bias({'-1': 5.0, '0': 5.0, '1': 5.0}) == '0'
    assert eb.best_bias({'-1': 5.0, '1': 5.0, '2': 3.0}) == '-1'
    assert eb.best_bias({'-1': 5.0, '0': 4.0, '1': 5.0 + 1e-12, '2': 5.0}) == '-1'   # float noise
    assert eb.best_bias({'0': 1.0, '0.5': 2.0}) == '0.5'
    assert eb.f1([2, 3, 2]) == 80.0 and eb.f1([0, 0, 0]) == 0.0


def test_a_held_out_paper_never_chooses_with_its_own_labels():
    curves = eb.inner_curves([('inner', _inner())], PAPERS, KEYS)
    # paper a: its inner papers are b, c and d, all peaking at 1
    assert curves['a.conll'] == {'-1': 10.0, '0': 20.0, '1': 50.0, '2': 40.0}
    assert eb.best_bias(curves['a.conll']) == '1'
    # papers b, c, d: one inner paper is a, whose scores pull the mean to 2
    for p in PAPERS[1:]:
        assert eb.best_bias(curves[p]) == '2'
        assert curves[p]['2'] == pytest.approx((90 + 40 + 40) / 3)
    # two inner files (seeds) are averaged
    two = eb.inner_curves([('m0', _inner()), ('m100', _inner('crf_bias_m100'))], PAPERS, KEYS)
    assert two == curves
    with pytest.raises(eb.BiasError, match='no fold holds out'):
        eb.inner_curves([('m0', _inner(drop=('a.conll', 'b.conll')))], PAPERS, KEYS)


def test_variant_names():
    assert eb.eb_variant('crf_bias_m300', 'inner') == 'crf_eb_m300'
    assert eb.eb_variant('crf_ctx_bias', 'val') == 'crf_ctx_ebv'
    with pytest.raises(eb.BiasError):
        eb.eb_variant('crf_m300', 'inner')


def test_inner_choice_writes_the_runs_at_their_bias(tmp_path, capsys):
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    inner = tmp_path / 'ner_results_fine_leave_two_papers_out_crf_bias_m0.json'
    inner.write_text(json.dumps(_inner()))
    outer = tmp_path / 'ner_results_fine_leave_one_paper_out_crf_bias_m300.json'
    outer.write_text(json.dumps(_outer('crf_bias_m300')))
    assert eb.main([str(outer), '--choose', 'inner', '--inner', str(inner),
                    '--save-predictions']) == 0
    out = json.loads((tmp_path / 'ner_results_fine_leave_one_paper_out_crf_eb_m300.json')
                     .read_text())
    assert out['variant'] == 'crf_eb_m300' and out['protocol'] == 'leave_one_paper_out'
    assert out['config'] == {'input': 'crf_bias_m300', 'choice': 'inner'}
    runs = {r['run_id']: r for r in out['runs']}
    a, b = runs['a.conll'], runs['b.conll']
    assert (a['bias'], a['f1'], a['input_f1']) == (1.0, 100.0, 66.67)
    assert a['entities'] == {'input': 1, 'output': 2, 'gold': 2} and a['test_pred'] == PRED['1']
    assert (b['bias'], b['f1']) == (2.0, 80.0) and b['test_pred'] == PRED['2']
    assert out['summary']['n_runs'] == 4
    summary = json.loads((tmp_path / 'ner_entity_bias_fine_leave_one_paper_out_inner.json')
                         .read_text())
    assert summary['chosen'] == {'a.conll': 1.0, 'b.conll': 2.0, 'c.conll': 2.0, 'd.conll': 2.0}
    assert summary['biases'] == [-1.0, 0.0, 1.0, 2.0] and summary['inner'] == ['crf_bias_m0']
    assert summary['test_curves']['a.conll'] == {'-1': 0.0, '0': 66.6667, '1': 100.0, '2': 80.0}
    assert 'crf_eb_m300' in capsys.readouterr().out
    # without --save-predictions the runs keep only their scores
    lean = tmp_path / 'lean'
    lean.mkdir()
    assert eb.main([str(outer), '--choose', 'inner', '--inner', str(inner), '--out-dir',
                    str(lean)]) == 0
    run = json.loads((lean / outer.name.replace('bias', 'eb')).read_text())['runs'][0]
    assert not {'test_index', 'test_gold', 'test_pred'} & set(run)


def test_validation_choice(tmp_path):
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    val = {'-1': [0, 0, 4], '0': [2, 2, 4], '1': [3, 4, 4], '2': [3, 6, 4]}   # 1 is best
    path = tmp_path / 'ner_results_fine_published_split_crf_bias.json'
    path.write_text(json.dumps(_outer('crf_bias', 'published_split', val)))
    assert eb.main([str(path), '--choose', 'val']) == 0
    out = json.loads((tmp_path / 'ner_results_fine_published_split_crf_ebv.json').read_text())
    assert out['variant'] == 'crf_ebv' and {r['bias'] for r in out['runs']} == {1.0}
    summary = json.loads((tmp_path / 'ner_entity_bias_fine_published_split_val.json').read_text())
    assert summary['chosen'] == {'crf_bias': {p: 1.0 for p in PAPERS}}


def test_unusable_inputs_are_reported(tmp_path, capsys):
    inner = tmp_path / 'inner.json'
    inner.write_text(json.dumps(_inner()))
    outer = tmp_path / 'outer.json'
    outer.write_text(json.dumps(_outer()))
    assert eb.main([str(outer), '--choose', 'inner']) == 2
    assert 'needs --inner' in capsys.readouterr().err
    other = _outer()
    for run in other['runs']:
        run['bias_test'] = {k: v for k, v in run['bias_test'].items() if k != '2'}
    (tmp_path / 'other.json').write_text(json.dumps(other))
    assert eb.main([str(outer), str(tmp_path / 'other.json'), '--choose', 'inner',
                    '--inner', str(inner)]) == 2
    assert 'biases' in capsys.readouterr().err
    lopo_as_inner = tmp_path / 'lopo.json'
    lopo_as_inner.write_text(json.dumps(_outer()))
    assert eb.main([str(outer), '--choose', 'inner', '--inner', str(lopo_as_inner)]) == 2
    assert 'leave-two-papers-out' in capsys.readouterr().err
    bare = _outer()
    for run in bare['runs']:
        del run['test_pred_bias']
    (tmp_path / 'bare.json').write_text(json.dumps(bare))
    assert eb.main([str(tmp_path / 'bare.json'), '--choose', 'inner', '--inner', str(inner)]) == 2
    assert 'save-predictions' in capsys.readouterr().err
    no_bias = _outer()
    for run in no_bias['runs']:
        del run['bias_test']
    (tmp_path / 'plain.json').write_text(json.dumps(no_bias))
    assert eb.main([str(tmp_path / 'plain.json'), '--choose', 'val']) == 2
    assert 'entity-bias' in capsys.readouterr().err
