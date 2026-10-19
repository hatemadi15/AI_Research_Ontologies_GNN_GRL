"""Document-level label consistency for saved test predictions (ner_consistency.py). The rule
is pure Python; the tests that score a results file need seqeval and scipy (skipped
without them)."""
import json

import pytest

import ner_consistency as nc


def _doc(*sentences):
    """(word lists, tag lists) from 'word/TAG word/TAG ...' strings (no /TAG: O)."""
    words, tags = [], []
    for s in sentences:
        pairs = [t.split('/') if '/' in t else (t, 'O') for t in s.split()]
        words.append([w for w, _ in pairs])
        tags.append([t for _, t in pairs])
    return words, tags


def _rule(*sentences, retype=False):
    words, tags = _doc(*sentences)
    out, added, retyped = nc.document_rule(words, tags, retype)
    assert tags == _doc(*sentences)[1]                        # the input is not changed
    return [' '.join(t) for t in out], added, retyped


def test_a_form_tagged_more_often_than_not_is_propagated():
    out, added, retyped = _rule('Ti64/B-MAT was tested', 'Ti64/B-MAT fails', 'the Ti64 sample')
    assert out == ['B-MAT O O', 'B-MAT O', 'O B-MAT O'] and (added, retyped) == (1, 0)
    # multi-word forms keep their B-/I- tags
    out, added, _ = _rule('Inconel/B-MAT 718/I-MAT and', 'Inconel/B-MAT 718/I-MAT', 'Inconel 718')
    assert out == ['B-MAT I-MAT O', 'B-MAT I-MAT', 'B-MAT I-MAT'] and added == 1


def test_a_form_needs_more_tagged_than_untagged_occurrences():
    out, added, _ = _rule('Ti64/B-MAT was tested', 'the Ti64 sample')
    assert out == ['B-MAT O O', 'O O O'] and added == 0
    # occurrences in the same sentence count too
    out, added, _ = _rule('Ti64/B-MAT and Ti64/B-MAT and Ti64')
    assert out == ['B-MAT O B-MAT O B-MAT'] and added == 1


def test_a_form_without_one_most_frequent_type_is_left_alone():
    out, added, _ = _rule('Ti64/B-MAT a', 'Ti64/B-PRO b', 'Ti64 c')
    assert out == ['B-MAT O', 'B-PRO O', 'O O'] and added == 0
    out, added, _ = _rule('Ti64/B-MAT a', 'Ti64/B-MAT b', 'Ti64/B-PRO c', 'Ti64 d')
    assert out[3] == 'B-MAT O' and added == 1


def test_longer_forms_go_first_and_nothing_is_overwritten():
    # "Ti64 alloy" and "Ti64" are both propagated; the longer form takes the untagged
    # "Ti64 alloy", so "Ti64" cannot tag its first word again
    out, added, _ = _rule('Ti64/B-MAT alloy/I-MAT x', 'Ti64/B-MAT alloy/I-MAT y',
                          'Ti64/B-MAT z', 'Ti64/B-MAT w', 'Ti64 alloy v')
    assert out[4] == 'B-MAT I-MAT O' and added == 1
    # an occurrence that overlaps a predicted entity of another span is neither tagged nor
    # untagged: "Ti64" inside "Ti64 alloy", and nothing changes there
    out, added, _ = _rule('Ti64/B-MAT x', 'Ti64/B-MAT y', 'Ti64/B-MAT alloy/I-MAT z')
    assert out[2] == 'B-MAT I-MAT O' and added == 0
    # nor is one with a predicted word inside it
    out, added, _ = _rule('Ti64/B-MAT alloy/I-MAT x', 'Ti64/B-MAT alloy/I-MAT y',
                          'Ti64/B-MAT alloy z')
    assert out[2] == 'B-MAT O O' and added == 0


def test_retyping_is_optional():
    sentences = ('Ti64/B-MAT a', 'Ti64/B-MAT b', 'Ti64/B-PRO c', 'Ti64 d')
    out, added, retyped = _rule(*sentences)
    assert out[2] == 'B-PRO O' and (added, retyped) == (1, 0)
    out, added, retyped = _rule(*sentences, retype=True)
    assert out[2] == 'B-MAT O' and out[3] == 'B-MAT O' and (added, retyped) == (1, 1)
    # a form that is not propagated is not retyped either
    out, _, retyped = _rule('Ti64/B-MAT a', 'Ti64/B-PRO b', 'Ti64/B-MAT c', 'Ti64 d', 'Ti64 e',
                            'Ti64 f', retype=True)
    assert out[1] == 'B-PRO O' and retyped == 0


def test_variant_names():
    assert nc.dc_variant('crf_m300') == 'crf_dc_m300'
    assert nc.dc_variant('crf_mt') == 'crf_mt_dc'
    assert nc.dc_variant('crf_mt_m100') == 'crf_mt_dc_m100'
    assert nc.dc_variant('crf_ens3') == 'crf_ens3_dc'
    assert nc.dc_variant('crf_ens3_m600', retype=True) == 'crf_ens3_dcr_m600'
    assert nc.dc_variant('crf') == 'crf_dc'


CORPUS = [(['Ti64', 'was', 'tested'], ['B-MAT', 'O', 'O'], 'a.conll'),
          (['the', 'Ti64', 'sample'], ['O', 'B-MAT', 'O'], 'a.conll'),
          (['Ti64', 'fails'], ['B-MAT', 'O'], 'a.conll'),
          (['Ti64', 'here'], ['B-MAT', 'O'], 'b.conll'),
          (['no', 'Ti64'], ['O', 'O'], 'b.conll')]


def _results(pred):
    """A results file of two runs over the corpus above; pred[k]: the tags of sentence k."""
    def run(run_id, idx):
        return {'run_id': run_id, 'f1': 50.0, 'f1_strict': 50.0, 'test_index': idx,
                'test_gold': [' '.join(CORPUS[i][1]) for i in idx],
                'test_pred': [pred[i] for i in idx]}
    return {'granularity': 'fine', 'protocol': 'leave_one_paper_out', 'variant': 'crf_m100',
            'model': 'm', 'labels': ['MAT'], 'label_tie': 'x', 'paper_test_f1': None,
            'runs': [run('a', [0, 1, 2]), run('b', [3, 4])]}


def test_results_are_rescored_per_document(tmp_path, monkeypatch, capsys):
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    pred = ['B-MAT O O', 'O O O', 'B-MAT O', 'B-MAT O', 'O O']
    path = tmp_path / 'ner_results_fine_leave_one_paper_out_crf_m100.json'
    path.write_text(json.dumps(_results(pred)))
    monkeypatch.setattr(nc, 'load_split_corpus', lambda g: CORPUS)
    assert nc.main([str(path), '--save-predictions']) == 0
    out = json.loads((tmp_path / 'ner_results_fine_leave_one_paper_out_crf_dc_m100.json')
                     .read_text())
    assert out['variant'] == 'crf_dc_m100' and out['protocol'] == 'leave_one_paper_out'
    assert out['config'] == {'input': 'crf_m100', 'rule': nc.RULE, 'retype': False}
    a, b = out['runs']
    # paper a: Ti64 tagged twice, untagged once, so it is added (gold has it): F1 100
    assert a['test_pred'] == ['B-MAT O O', 'O B-MAT O', 'B-MAT O'] and a['n_added'] == 1
    assert (a['f1'], a['input_f1'], a['n_retyped']) == (100.0, 50.0, 0)
    # paper b counts only its own sentences: one tagged, one untagged, nothing added
    assert b['test_pred'] == ['B-MAT O', 'O O'] and b['n_added'] == 0
    assert out['summary']['n_runs'] == 2 and out['summary']['run_ids'] == ['a', 'b']
    assert 'crf_dc_m100' in capsys.readouterr().out
    # without --save-predictions the runs keep only their scores and counts
    assert nc.main([str(path), '--out-dir', str(tmp_path / 'lean')]) == 2   # no such dir
    (tmp_path / 'lean').mkdir()
    assert nc.main([str(path), '--out-dir', str(tmp_path / 'lean')]) == 0
    lean = json.loads((tmp_path / 'lean' / 'ner_results_fine_leave_one_paper_out_crf_dc_m100.json')
                      .read_text())
    assert not {'test_index', 'test_gold', 'test_pred'} & set(lean['runs'][0])
    assert [r['f1'] for r in lean['runs']] == [r['f1'] for r in out['runs']]


def test_unusable_inputs_are_reported(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(nc, 'load_split_corpus', lambda g: CORPUS)
    path = tmp_path / 'in.json'
    results = _results(['B-MAT O O', 'O O O', 'B-MAT O', 'B-MAT O', 'O O'])
    del results['runs'][1]['test_pred']
    path.write_text(json.dumps(results))
    assert nc.main([str(path)]) == 2 and 'save-predictions' in capsys.readouterr().err
    # an I- tag that continues nothing: only valid IOB2 predictions (a CRF head, a vote)
    path.write_text(json.dumps(_results(['I-MAT O O', 'O O O', 'B-MAT O', 'B-MAT O', 'O O'])))
    assert nc.main([str(path)]) == 2 and 'IOB2' in capsys.readouterr().err
    # test sentences that do not match the corpus
    results = _results(['B-MAT O O', 'O O O', 'B-MAT O', 'B-MAT O', 'O O'])
    results['runs'][0]['test_index'] = [3, 1, 2]
    path.write_text(json.dumps(results))
    assert nc.main([str(path)]) == 2 and 'corpus' in capsys.readouterr().err
