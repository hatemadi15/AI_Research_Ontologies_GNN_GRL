"""The saved NER predictions under looser matching criteria (ner_matching.py): one-to-one pairs,
the four criteria, the near misses, and the strict criterion against the stored micro F1."""
import json
import os
import shutil

import pytest

import ner_matching as nm

GPU = os.path.join(os.path.dirname(__file__), '..', 'results', 'gpu')


def test_the_four_criteria_on_a_sentence():
    gold = 'B-A I-A O B-B O O'.split()
    pred = 'B-A O O B-C O B-A'.split()
    # (0,1,A) inside the gold (0,2,A); (3,4,C) the span of (3,4,B); (5,6,A) spurious
    assert nm.sentence_counts(gold, pred) == {'gold': 2, 'pred': 3, 'strict': 0, 'exact': 1,
                                              'type': 1, 'overlap': 2}
    assert nm.sentence_counts(gold, gold) == {'gold': 2, 'pred': 2, 'strict': 2, 'exact': 2,
                                              'type': 2, 'overlap': 2}


def test_pairs_are_one_to_one_and_prefer_the_same_span():
    gold = {(0, 3, 'A')}
    pred = {(0, 1, 'A'), (2, 3, 'A')}
    assert nm.paired(gold, pred, same_type=True) == 1           # one gold entity, one pair
    gold = {(0, 2, 'A'), (2, 4, 'B')}
    pred = {(0, 2, 'B'), (1, 3, 'B')}
    # the same span pairs first: (0,2,A)-(0,2,B); then (2,4,B)-(1,3,B) by overlap
    assert nm.paired(gold, pred, same_type=False) == 2
    assert nm.paired(gold, pred, same_type=True) == 1           # only (1,3,B)-(2,4,B)


def test_near_misses():
    run = {'test_gold': ['B-A I-A O B-B O O O', 'B-C O'],
           'test_pred': ['B-A O O B-C O B-A I-A', 'B-C I-C']}
    out = nm.near_misses([{'runs': [run]}], related=lambda a, b: {a, b} == {'B', 'C'})
    assert out == {'n_pred': 4, 'right': 0.0, 'wrong_type': 0.25, 'wrong_type_related': 0.25,
                   'inside_longer_gold': 0.25, 'around_shorter_gold': 0.25, 'crossing': 0.0,
                   'spurious': 0.25, 'wrong_type_pairs': [['B', 'C', 1.0]]}
    # the pairs per run, the most frequent first, at most `top`
    other = {'test_gold': ['B-B O B-D', 'B-B'], 'test_pred': ['B-C O B-E', 'B-C']}
    pairs = nm.near_misses([{'runs': [run, other]}], top=2)['wrong_type_pairs']
    assert pairs == [['B', 'C', 1.5], ['D', 'E', 0.5]]
    assert 'wrong_type_related' not in nm.near_misses([{'runs': [run]}])


def test_mmo_related():
    pytest.importorskip('rdflib')
    assert nm.mmo_related('FatigueTestSpecimen', 'TestSpecimen')
    assert nm.mmo_related('TestSpecimen', 'FatigueTestSpecimen')
    assert not nm.mmo_related('Crack', 'Value')


@pytest.mark.parametrize('name', ['ner_results_fine_leave_one_paper_out_crf_m0.json',
                                  'ner_results_coarse_published_split.json',
                                  'ner_results_sofc_published_split_crf.json'])
def test_strict_is_the_stored_micro_f1(name):
    """Plain MatSciBERT's predictions need not be valid IOB2; seqeval's default mode and
    tags_to_spans read them alike."""
    with open(os.path.join(GPU, name)) as f:
        results = json.load(f)
    for run in results['runs']:
        scores = nm.run_f1(run)
        assert scores['strict'] == pytest.approx(run['f1'], abs=0.01)
        assert scores['strict'] <= scores['exact'] <= scores['overlap']
        assert scores['strict'] <= scores['type'] <= scores['overlap']


def test_analyse_the_crf_head_on_both_protocols(tmp_path):
    pytest.importorskip('scipy')
    pytest.importorskip('rdflib')
    for protocol in ('random_split_vary', 'leave_one_paper_out'):
        for k in range(0, 1000, 100):
            name = f'ner_results_coarse_{protocol}_crf_m{k}.json'
            shutil.copy(os.path.join(GPU, name), tmp_path / name)
    out = nm.analyse(str(tmp_path))
    assert set(out['systems']) == {'coarse_random_split_vary', 'coarse_leave_one_paper_out'}
    held_out = out['systems']['coarse_leave_one_paper_out']
    assert set(held_out) == {'crf', 'crf_ens3'} and held_out['crf']['n_pairs'] == 4
    assert held_out['crf']['strict']['mean'] == 58.6           # the committed 10-seed mean
    assert held_out['crf_ens3']['strict']['mean'] == 61.1      # the seed vote's
    gap = out['gap']['coarse']['crf']
    assert gap['strict'] > gap['exact'] > gap['overlap'] > 0
    assert set(out['near_misses']) == {'coarse_random_split_vary', 'coarse_leave_one_paper_out'}
    assert 'wrong_type_related' not in out['near_misses']['coarse_leave_one_paper_out']


def test_the_mulms_gap_is_its_pool_splits_minus_its_held_out_subfields(tmp_path):
    """MuLMS's two protocols are its random pool splits and held-out subfields (the coarse
    CRF files stand in for its files)."""
    pytest.importorskip('scipy')
    pytest.importorskip('rdflib')
    for source, protocol in (('random_split_vary', 'random_split_pool'),
                             ('leave_one_paper_out', 'leave_one_subfield_out')):
        for k in range(0, 1000, 100):
            shutil.copy(os.path.join(GPU, f'ner_results_coarse_{source}_crf_m{k}.json'),
                        tmp_path / f'ner_results_mulms_{protocol}_crf_m{k}.json')
    out = nm.analyse(str(tmp_path))
    assert set(out['systems']) == {'mulms_random_split_pool', 'mulms_leave_one_subfield_out'}
    pool, held = (out['systems'][f'mulms_{p}']['crf']
                  for p in ('random_split_pool', 'leave_one_subfield_out'))
    assert out['gap']['mulms']['crf'] == {
        c: round(pool[c]['mean'] - held[c]['mean'], 2) for c in nm.CRITERIA}
    assert set(out['near_misses']) == set(out['systems'])


def test_conventions_of_the_gold_annotations():
    data = [(['the', 'samples'], ['O', 'B-TestSpecimen'], 'p1'),
            (['the', 'samples'], ['O', 'B-FatigueTestSpecimen'], 'p2'),
            (['fatigue'], ['B-Mechanism'], 'p1'),
            (['fatigue'], ['B-FatigueTest'], 'p2'),
            (['EBSD', 'data'], ['B-DataSet', 'I-DataSet'], 'p1'),
            (['data'], ['B-DataSet'], 'p2'),
            (['data'], ['B-DataSet'], 'p3'),
            (['crack'], ['B-Crack'], 'p1')]
    related = {frozenset({'TestSpecimen', 'FatigueTestSpecimen'})}
    out = nm.conventions(data, related=lambda a, b: frozenset({a, b}) in related,
                         min_entities=1, min_papers=2)
    assert (out['n_shared'], out['n_retyped'], out['n_retyped_related']) == (3, 2, 1)
    assert out['retyped'] == {'fatigue': {'p1': ['Mechanism'], 'p2': ['FatigueTest']},
                              'samples': {'p1': ['TestSpecimen'],
                                          'p2': ['FatigueTestSpecimen']}}
    assert out['span_length']['DataSet'] == {'p1': 2.0, 'p2': 1.0, 'p3': 1.0}
    assert 'Crack' not in out['span_length']                 # one paper only
    assert 'n_retyped_related' not in nm.conventions(data)
