"""Coarse-to-fine completion of fine-grained test predictions (ner_coarse_to_fine.py). The
rule is pure Python; scoring a results file needs seqeval and scipy (skipped without)."""
import json

import pytest

import ner_coarse_to_fine as cf


def _sentence(spec, doc='b.conll'):
    """(words, fine tags, coarse tags, doc) from 'word/FINE/COARSE ...' (missing: O)."""
    words, fine, coarse = [], [], []
    for token in spec.split():
        parts = token.split('/') + ['O', 'O']
        words.append(parts[0])
        fine.append(parts[1])
        coarse.append(parts[2])
    return (words, fine, doc), (words, coarse, doc)


def _gold(specs, doc='b.conll'):
    pairs = [_sentence(s, doc) for s in specs]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def test_a_coarse_type_maps_when_most_of_its_spans_are_one_fine_type():
    specs = (['5/B-Value/B-VAL MPa/I-Value/I-VAL'] * 10              # VAL: 10 of 10 Value
             + ['Ti64/B-Alloy/B-MAT'] * 8 + ['steel/O/B-MAT'] * 2      # MAT: 8 of 10 Alloy
             + ['crack/B-Crack/B-DAM'] * 9)                            # DAM: 9 spans only
    fine, coarse = _gold(specs)
    assert cf.type_mapping(fine, coarse) == {'VAL': 'Value'}
    assert cf.type_mapping(fine, coarse, share=0.8) == {'MAT': 'Alloy', 'VAL': 'Value'}
    assert cf.type_mapping(fine, coarse, min_spans=9) == {'DAM': 'Crack', 'VAL': 'Value'}
    # a fine entity on another span does not count for the coarse span
    fine, coarse = _gold(['5/B-Value/B-VAL MPa/O/I-VAL'] * 10)
    assert cf.type_mapping(fine, coarse) == {}
    # a sentence tokenised differently at the two levels counts for nothing
    fine, coarse = _gold(['5/B-Value/B-VAL'] * 10)
    other, _ = _gold(['5x/B-Value/B-VAL'] * 10)
    assert cf.type_mapping(fine + other[:1], coarse + coarse[:1]) == {'VAL': 'Value'}
    assert cf.type_mapping(other, coarse) == {}


def test_completion_adds_only_where_the_fine_level_is_all_o():
    mapping = {'VAL': 'Value', 'MAT': 'Alloy'}
    tags, added = cf.complete(['O', 'O', 'O', 'B-Mechanism'],
                              ['B-VAL', 'I-VAL', 'O', 'B-MEC'], mapping)
    assert tags == ['B-Value', 'I-Value', 'O', 'B-Mechanism'] and added == 1
    # a fine entity inside the coarse span: nothing changes (not even a partial overlap)
    tags, added = cf.complete(['O', 'B-Alloy', 'O'], ['B-MAT', 'I-MAT', 'O'], mapping)
    assert tags == ['O', 'B-Alloy', 'O'] and added == 0
    # unmapped coarse types and coarse O leave the fine tags alone
    tags, added = cf.complete(['O', 'O'], ['B-MEC', 'O'], mapping)
    assert tags == ['O', 'O'] and added == 0
    with pytest.raises(cf.CompletionError):
        cf.complete(['O'], ['O', 'O'], mapping)


def test_variant_names():
    assert cf.cf_variant('crf_m300') == 'crf_cf_m300'
    assert cf.cf_variant('crf_mt') == 'crf_mt_cf'
    assert cf.cf_variant('crf_mt_m100') == 'crf_mt_cf_m100'
    assert cf.cf_variant('crf_mt_ens3') == 'crf_mt_ens3_cf'
    assert cf.cf_variant('crf_mt_ens3_m600') == 'crf_mt_ens3_cf_m600'


# the corpus: paper a is held out (3 test sentences), b and c give the mapping
TRAIN = ['5/B-Value/B-VAL'] * 6 + ['steel/O/B-MAT'] * 6
TEST = ['7/O/B-VAL MPa/O/I-VAL x', 'Ti64/B-Alloy/B-MAT y', 'z/O/O 9/B-Value/B-VAL']


def _corpora():
    fine, coarse = [], []
    for doc, specs in (('a.conll', TEST), ('b.conll', TRAIN), ('c.conll', TRAIN)):
        f, c = _gold(specs, doc)
        fine += f
        coarse += c
    return fine, coarse, ['Alloy', 'Value']


def _files(tmp_path, fine_pred, coarse_pred, coarse_index=(0, 1, 2)):
    fine, coarse, labels = _corpora()

    def results(granularity, gold, pred, index):
        return {'granularity': granularity, 'protocol': 'leave_one_paper_out',
                'variant': 'crf_m100', 'model': 'm',
                'labels': labels if granularity == 'fine' else ['MAT', 'VAL'],
                'runs': [{'run_id': 'a.conll', 'f1': 50.0, 'f1_strict': 50.0,
                          'test_index': list(index),
                          'test_gold': [' '.join(gold[i][1]) for i in index],
                          'test_pred': pred}]}
    paths = []
    for g, gold, pred, index in (('fine', fine, fine_pred, (0, 1, 2)),
                                 ('coarse', coarse, coarse_pred, coarse_index)):
        path = tmp_path / f'ner_results_{g}_leave_one_paper_out_crf_m100.json'
        path.write_text(json.dumps(results(g, gold, pred, index)))
        paths.append(str(path))
    return paths


def test_results_are_completed_and_rescored(tmp_path, monkeypatch, capsys):
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    monkeypatch.setattr(cf, 'gold_corpora', _corpora)
    fine_path, coarse_path = _files(tmp_path, ['O O O', 'B-Alloy O', 'O O'],
                                    ['B-VAL I-VAL O', 'B-MAT O', 'O B-VAL'])
    assert cf.main(['--fine', fine_path, '--coarse', coarse_path, '--save-predictions']) == 0
    out = json.loads((tmp_path / 'ner_results_fine_leave_one_paper_out_crf_cf_m100.json')
                     .read_text())
    (run,) = out['runs']
    # b and c: VAL spans are all Value (12), MAT spans never fine (0 of 12): only VAL maps
    assert run['mapping'] == {'VAL': 'Value'} and run['n_added'] == 2
    assert run['test_pred'] == ['B-Value I-Value O', 'B-Alloy O', 'O B-Value']
    # '7 MPa' is fine-grained O in the gold tags: 2 of the 3 predictions are right
    assert run['f1'] == 80.0 and run['input_f1'] == 50.0 and run['n_unaligned'] == 0
    assert out['config'] == {'input': 'crf_m100', 'coarse_input': 'crf_m100',
                             'rule': cf.RULE, 'min_spans': 10, 'share': 0.9}
    assert 'crf_cf_m100' in capsys.readouterr().out


def test_a_sentence_tokenised_differently_is_left_alone(tmp_path, monkeypatch):
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')

    def corpora():
        fine, coarse, labels = _corpora()
        words, tags, doc = coarse[0]
        coarse[0] = (['7MPa', 'x', 'y'], tags, doc)        # same length, other words
        return fine, coarse, labels
    monkeypatch.setattr(cf, 'gold_corpora', corpora)
    fine_path, coarse_path = _files(tmp_path, ['O O O', 'B-Alloy O', 'O O'],
                                    ['B-VAL I-VAL O', 'B-MAT O', 'O B-VAL'])
    assert cf.main(['--fine', fine_path, '--coarse', coarse_path, '--save-predictions']) == 0
    (run,) = json.loads((tmp_path / 'ner_results_fine_leave_one_paper_out_crf_cf_m100.json')
                        .read_text())['runs']
    assert run['test_pred'] == ['O O O', 'B-Alloy O', 'O B-Value']
    assert (run['n_added'], run['n_unaligned'], run['f1']) == (1, 1, 100.0)


def test_unusable_inputs_are_reported(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cf, 'gold_corpora', _corpora)
    paths = _files(tmp_path, ['O O O', 'B-Alloy O', 'O O'], ['O O O', 'O O', 'O O'],
                   coarse_index=(0, 2, 1))
    assert cf.main(['--fine', paths[0], '--coarse', paths[1]]) == 2
    assert 'different sentences' in capsys.readouterr().err
    assert cf.main(['--fine', paths[1], '--coarse', paths[0]]) == 2      # swapped
    assert 'coarse file' in capsys.readouterr().err
    paths = _files(tmp_path, ['I-Value O O', 'B-Alloy O', 'O O'], ['O O O', 'O O', 'O O'])
    assert cf.main(['--fine', paths[0], '--coarse', paths[1]]) == 2
    assert 'IOB2' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cf.main(['--fine', paths[0], paths[0], '--coarse', paths[1]])
