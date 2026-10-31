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
