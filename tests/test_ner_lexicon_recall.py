"""The recall split by lexicon membership (ner_lexicon_recall.py): the per-sentence counts, the
unit scores, the paired comparison and the checks against the stored names runs."""
from collections import Counter

import pytest

import ner_lexicon_recall as lr


def test_sentence_counts_and_unit_scores():
    words = ['Pt', 'on', 'YSZ', 'at', '800', 'C']
    gold = ['B-MAT', 'O', 'B-MAT', 'O', 'B-VALUE', 'I-VALUE']
    pred = ['B-MAT', 'O', 'O', 'O', 'B-VALUE', 'I-VALUE']
    forms = {('800', 'C'): {'VALUE'}}                      # the value is a training entity
    lexicon = {('MAT', ('YSZ',))}                          # YSZ is in the lexicon, Pt is not
    c = lr.sentence_counts(words, gold, pred, forms, lexicon)
    assert c['gold', 'in'] == 1 and c['correct', 'in'] == 0      # YSZ missed
    assert c['gold', 'out'] == 1 and c['correct', 'out'] == 1    # Pt found
    assert c['pred', 'out'] == 1 and c['pred_correct', 'out'] == 1 and c['pred', 'in'] == 0
    s = lr.unit_scores(c)
    assert s['recall_in'] == 0.0 and s['recall_out'] == 100.0 and s['precision_in'] is None
    assert s['gold_share_in'] == 50.0 and s['pred_share_in'] == 0.0
    assert lr.lexicon_of({'names': [['MAT', 'Y S Z', 2]]}) == {('MAT', ('Y', 'S', 'Z'))}


def _unit(gold_in, correct_in, gold_out, correct_out):
    return Counter({('gold', 'in'): gold_in, ('correct', 'in'): correct_in,
                    ('gold', 'out'): gold_out, ('correct', 'out'): correct_out})


def test_compare_pairs_the_units():
    pytest.importorskip('scipy')
    method = {'a': _unit(10, 8, 10, 5), 'b': _unit(10, 6, 10, 5), 'c': _unit(10, 7, 10, 6)}
    reference = {'a': _unit(10, 4, 10, 4), 'b': _unit(10, 4, 10, 4), 'c': _unit(10, 4, 10, 5)}
    out = lr.compare(method, reference, ['a', 'b', 'c'])
    assert out['per_unit']['a'] == {'gain_in': 40.0, 'gain_out': 10.0, 'in_minus_out': 30.0}
    assert out['gain_in']['mean_diff'] == 30.0 and out['gain_in']['n_positive'] == 3
    assert out['in_minus_out']['mean_diff'] == 20.0 and out['in_minus_out']['n_units'] == 3
    assert out['in_minus_out']['ci95'][0] < 20.0 < out['in_minus_out']['ci95'][1]


def test_checks_against_the_stored_runs():
    stores = {'semi': {'teachers': [{'test_f1': 30.0}, {'test_f1': 31.0}],
                       'coverage': {'n_mentions': 5}, 'names': [['MAT', 'x', 1]]}}
    good = {'runs': [{'run_id': 'semi', 'self_training': {
        'teachers': [{'test_f1': 30.0}, {'test_f1': 31.0}], 'names': {'n_mentions': 5}}}]}
    bad = {'runs': [{'run_id': 'semi', 'self_training': {
        'teachers': [{'test_f1': 30.0}, {'test_f1': 31.5}], 'names': {'n_mentions': 5}}}]}
    assert lr.checks(stores, [good])['semi'] == {
        'teachers': True, 'coverage': True, 'runs_checked': 1,
        'rebuilt_teachers_f1': [30.0, 31.0], 'n_distinct': 1}
    assert lr.checks(stores, [good, bad])['semi']['teachers'] is False
    assert lr.checks(stores, None)['semi']['teachers'] is False      # nothing to check against


def test_week19_split_by_each_arms_lexicon(tmp_path, monkeypatch):
    """--week19: the arms against the 6-copy control on the unseen test names in and out of
    each arm's own lexicon, the unsplit rates paired over the units, and the expected share
    of the replacement draws that are names; {} until the names files and arm files exist."""
    pytest.importorskip('scipy')
    import json
    import ner_mentions as nm
    data = [(['steel', 'is', 'hard'], ['B-MAT', 'O', 'O'], 'd1'),
            (['the', 'alloy', 'AA2024'], ['O', 'O', 'B-MAT'], 'd1'),
            (['Ti6Al4V', 'fails'], ['B-MAT', 'O'], 'd2'),
            (['copper', 'bends'], ['B-MAT', 'O'], 'd2')]
    monkeypatch.setattr(nm, 'corpus', lambda granularity: data)
    monkeypatch.setattr(nm, 'run_split', lambda results, run, d, g: (data, [0, 1], [2, 3]))
    monkeypatch.setattr(lr, 'WEEK19_SETTINGS', {'mulms': ('mulms', 'leave_one_subfield_out',
                                                          (300,))})
    week19 = tmp_path / 'week19'
    week19.mkdir()
    preds = {'crf_mr6': ['O O', 'B-MAT O'], 'crf_mrg6': ['B-MAT O', 'B-MAT O'],
             'crf_mrv6': ['O O', 'O B-MAT']}
    for system, pred in preds.items():
        runs = [{'run_id': u, 'test_index': [2, 3], 'test_gold': ['B-MAT O', 'B-MAT O'],
                 'test_pred': pred} for u in ('u1', 'u2')]
        (tmp_path / f'ner_results_mulms_leave_one_subfield_out_{system}_m300.json').write_text(
            json.dumps({'protocol': 'leave_one_subfield_out', 'runs': runs}))
    assert lr.analyse_week19(tmp_path, week19) == {}
    names = [['MAT', 'Ti6Al4V'], ['MAT', 'Ti6Al4V'], ['MAT', 'Inconel 718']]
    for arm in ('g', 'v'):
        (week19 / f'names_mulms_{arm}.json').write_text(json.dumps(
            {'arm': arm, 'runs': {u: {'names': names} for u in ('u1', 'u2')}}))
    out = lr.analyse_week19(tmp_path, week19)['mulms']
    assert out['units'] == ['u1', 'u2'] and out['seeds'] == [300]
    g = out['arms']['crf_mrg6']
    assert g['lexicon']['u1'] == {'n_distinct': 2, 'n_entries': 3}
    assert g['names_share'] == {'u1': 60.0, 'u2': 60.0}          # 2 * 3 / (2 + 3) / 2
    assert g['pooled']['crf_mrg6']['recall_in'] == 100.0 and g['pooled']['crf_mr6']['recall_in'] == 0
    assert g['minus_control']['gain_in']['mean_diff'] == 100.0
    assert g['minus_control']['gain_out']['mean_diff'] == 0.0
    assert out['unseen']['crf_mr6']['u1'] == {'recall': 50.0, 'precision': 100.0}
    assert out['unseen']['crf_mrv6']['u1'] == {'recall': 0.0, 'precision': 0.0}
    assert out['graph_minus_vote']['recall']['mean_diff'] == 100.0
    assert out['graph_minus_vote']['precision']['mean_diff'] == 100.0
    assert out['vote_minus_control']['recall']['mean_diff'] == -50.0
