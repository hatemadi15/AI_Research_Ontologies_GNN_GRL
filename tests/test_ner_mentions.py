"""Seen and unseen test mentions of the saved NER runs (ner_mentions.py): the kinds, the counts
and scores per kind, the rebuilt splits of the committed runs, the votes, the paired
comparisons and the recall gap. The tests that compute a confidence interval need scipy."""
import json
import os
import shutil
from collections import Counter

import pytest

import ner_mentions as nm
from ner_baseline import paper_fold

GPU = os.path.join(os.path.dirname(__file__), '..', 'results', 'gpu')

# a training paper A, each entity of it in two sentences (so that one of them may be a
# validation sentence), and a held-out paper B
TOY = [(['The', 'Ti-6Al-4V', 'alloy'], ['O', 'B-Alloy', 'O'], 'A.conll'),
       (['Ti-6Al-4V', 'cracks'], ['B-Alloy', 'O'], 'A.conll'),
       (['at', '200', 'MPa'], ['O', 'B-Value', 'I-Value'], 'A.conll'),
       (['200', 'MPa', 'here'], ['B-Value', 'I-Value', 'O'], 'A.conll'),
       (['a', 'crack', 'grew'], ['O', 'B-Crack', 'O'], 'A.conll'),
       (['the', 'crack', 'stopped'], ['O', 'B-Crack', 'O'], 'A.conll'),
       (['Ti-6Al-4V', 'and', '316L'], ['B-Alloy', 'O', 'B-Alloy'], 'B.conll'),
       (['crack', 'at', '300', 'MPa'], ['B-Value', 'O', 'B-Value', 'I-Value'], 'B.conll')]


def _toy_results(preds, test_index=(6, 7)):
    return {'protocol': 'leave_one_paper_out', 'config': {'split_seed': 0},
            'runs': [{'run_id': 'B.conll', 'test_index': list(test_index),
                      'test_gold': [' '.join(TOY[i][1]) for i in test_index],
                      'test_pred': [' '.join(p) for p in preds]}]}


# Ti-6Al-4V found (seen), 316L missed (unseen), 'and' a spurious Alloy (unseen); crack as a
# Value missed (seen as a Crack only), 300 MPa found (unseen)
PREDS = [['B-Alloy', 'B-Alloy', 'O'], ['O', 'O', 'B-Value', 'I-Value']]


def test_kinds_of_entities():
    train, _, test = paper_fold(TOY, ('B.conll',), 0, indices=True)
    assert test == [6, 7] and len(train) == 5
    forms = nm.entity_forms(TOY, range(6))
    assert forms == {('Ti-6Al-4V',): {'Alloy'}, ('200', 'MPa'): {'Value'},
                     ('crack',): {'Crack'}}
    assert nm.kind_of(['Ti-6Al-4V'], 'Alloy', forms) == 'seen'
    assert nm.kind_of(['crack'], 'Value', forms) == 'seen_other'
    assert nm.kind_of(['316L'], 'Alloy', forms) == 'unseen'
    assert nm.kind_of(['200'], 'Value', forms) == 'unseen'       # part of a form is not it


def test_counts_and_scores_per_kind():
    counts = nm.run_counts(_toy_results(PREDS), TOY, 'fine')['B.conll']
    assert {k: v for k, v in counts.items() if len(k) == 2} == {
        ('gold', 'seen'): 1, ('correct', 'seen'): 1, ('pred', 'seen'): 1,
        ('gold', 'seen_other'): 1, ('correct', 'seen_other'): 0,
        ('gold', 'unseen'): 2, ('correct', 'unseen'): 1, ('pred', 'unseen'): 2}
    s = nm.scores(counts)
    assert (s['seen']['recall'], s['seen']['precision'], s['seen']['share']) == (100, 100, 0.25)
    assert (s['seen_other']['recall'], s['seen_other']['f1']) == (0, 0)
    assert (s['unseen']['recall'], s['unseen']['precision'], s['unseen']['f1'],
            s['unseen']['share']) == (50, 50, 50, 0.5)
    assert (s['all']['n_gold'], s['all']['n_pred'], s['all']['n_correct'],
            s['all']['f1']) == (4, 3, 2, 57.14)
    types = nm.type_kinds(counts, n_runs=1)
    assert list(types) == ['Alloy', 'Value']
    assert types['Alloy'] == {'n_gold_per_run': 2.0, 'unseen_share': 0.5, 'recall_seen': 100.0,
                              'recall_unseen': 0.0}
    assert types['Value'] == {'n_gold_per_run': 2.0, 'unseen_share': 0.5, 'recall_seen': None,
                              'recall_unseen': 100.0}


def test_a_run_must_match_its_rebuilt_split():
    with pytest.raises(nm.MentionError, match='test sentences'):
        nm.run_counts(_toy_results(PREDS[:1], test_index=(6,)), TOY, 'fine')
    bad = _toy_results(PREDS)
    bad['runs'][0]['test_gold'][1] = 'O O B-Value I-Value'
    with pytest.raises(nm.MentionError, match='gold tags'):
        nm.run_counts(bad, TOY, 'fine')
    del bad['runs'][0]['test_pred']
    with pytest.raises(nm.MentionError, match='no test_pred'):
        nm.run_counts(bad, TOY, 'fine')


def test_mulms_splits_are_rebuilt(monkeypatch):
    """Week 10: a held-out subfield's run (its fold rebuilt from subfields.json and the training
    pool) and a random pool split's run."""
    subfields = {'A.conll': 'alpha', 'B.conll': 'beta'}
    monkeypatch.setattr(nm, 'mulms_subfields', lambda: subfields)
    from ner_baseline import pool_split, subfield_folds
    name, train, _, test = subfield_folds(TOY, subfields, 0, 2, indices=True)[1]
    run = {'run_id': name, 'test_index': test}
    results = {'protocol': 'leave_one_subfield_out',
               'config': {'split_seed': 0, 'train_pool': 2}}
    assert nm.run_split(results, run, TOY, 'mulms') == (TOY, train, test)
    tr, _, te = pool_split(len(TOY), 1, 5)
    results = {'protocol': 'random_split_pool', 'config': {'train_pool': 5}}
    assert nm.run_split(results, {'run_id': 'seed1', 'test_index': te}, TOY, 'mulms') == (
        TOY, tr, te)
    with pytest.raises(nm.MentionError, match='test sentences'):
        nm.run_split(results, {'run_id': 'seed2', 'test_index': te}, TOY, 'mulms')


@pytest.mark.parametrize('name, granularity', [
    ('ner_results_fine_leave_one_paper_out_crf_m0.json', 'fine'),
    ('ner_results_coarse_random_split_vary_crf_m300.json', 'coarse'),
    ('ner_results_fine_published_split_crf.json', 'fine'),
    ('ner_results_sofc_published_split_crf.json', 'sofc')])
def test_the_committed_runs_match_their_rebuilt_splits(name, granularity):
    """Every run's test sentences and gold tags are those of its rebuilt split, and its F1
    over all kinds is the stored micro F1 (seqeval's default mode)."""
    with open(os.path.join(GPU, name)) as f:
        results = json.load(f)
    counts = nm.run_counts(results, nm.corpus(granularity), granularity)
    for run in results['runs']:
        assert nm.scores(counts[run['run_id']])['all']['f1'] == pytest.approx(run['f1'],
                                                                              abs=0.01)
    if 'leave_one_paper_out' in name:
        paper = nm.scores(counts['10.1007_s11340-021-00758-x.conll'])
        assert (paper['all']['n_gold'], paper['unseen']['n_gold']) == (60, 43)


@pytest.mark.filterwarnings('ignore:Precision is ill-defined')      # a vote of all O
def test_votes_take_their_members_by_triple_and_run_id():
    def member(j):
        tag = 'B-Alloy' if j in (0, 1, 3) else 'O'      # the first triple's majority tags it
        runs = [{'run_id': i, 'test_index': [0], 'test_gold': ['B-Alloy'], 'test_pred': [tag],
                 'f1': 0.0, 'f1_strict': 0.0} for i in ('p2', 'p1')]
        return {'protocol': 'leave_one_paper_out', 'config': {'split_seed': 0},
                'runs': runs[::-1] if j == 1 else runs}
    out = nm.votes([member(j) for j in range(10)])
    assert len(out) == 3
    assert [r['run_id'] for r in out[0]['runs']] == ['p2', 'p1']
    assert [r['test_pred'] for r in out[0]['runs']] == [['B-Alloy'], ['B-Alloy']]
    assert [r['test_pred'] for r in out[1]['runs']] == [['O'], ['O']]


def test_paired_comparison_per_kind():
    pytest.importorskip('scipy')

    def counts(found):
        return Counter({('gold', 'unseen'): 4, ('pred', 'unseen'): 4,
                        ('correct', 'unseen'): found, ('gold', 'seen'): 2,
                        ('pred', 'seen'): 2, ('correct', 'seen'): 2})
    method = [{'p1': counts(3), 'p2': counts(4)}, {'p1': counts(3), 'p2': counts(2)}]
    reference = [{'p1': counts(2), 'p2': counts(2)}, {'p1': counts(2), 'p2': counts(2)}]
    out = nm.paired(method, reference)
    assert out['seen'] == {'method': 100.0, 'reference': 100.0, 'mean_diff': 0.0,
                           'ci95': [0.0, 0.0], 'n_pairs': 2}
    # unseen F1: p1 75 and 75 against 50; p2 100 and 50 against 50 -> +25 per paper
    assert (out['unseen']['method'], out['unseen']['reference'],
            out['unseen']['mean_diff'], out['unseen']['ci95']) == (75.0, 50.0, 25.0,
                                                                   [25.0, 25.0])


def test_recall_gap_with_the_other_protocols_mix():
    random = {'seen': {'share': 0.6, 'recall': 90.0}, 'seen_other': {'share': 0.0, 'recall': 0.0},
              'unseen': {'share': 0.4, 'recall': 60.0}, 'all': {'recall': 78.0}}
    held_out = {'seen': {'share': 0.2, 'recall': 70.0},
                'seen_other': {'share': 0.0, 'recall': 0.0},
                'unseen': {'share': 0.8, 'recall': 35.0}, 'all': {'recall': 42.0}}
    out = nm.gap(random, held_out)
    assert out['held_out_recall_with_random_mix'] == 56.0       # 0.6 * 70 + 0.4 * 35
    assert out['random_recall_with_held_out_mix'] == 66.0       # 0.2 * 90 + 0.8 * 60
    assert out['unseen_share'] == {'random_splits': 0.4, 'held_out': 0.8}


def test_analyse_the_crf_head_on_both_protocols(tmp_path):
    pytest.importorskip('scipy')
    for protocol in ('random_split_vary', 'leave_one_paper_out'):
        for k in range(0, 1000, 100):
            name = f'ner_results_fine_{protocol}_crf_m{k}.json'
            shutil.copy(os.path.join(GPU, name), tmp_path / name)
    out = nm.analyse(str(tmp_path), corpus_path=tmp_path / 'no_corpus.txt')
    assert out['corpus_coverage'] == {}
    assert set(out['systems']) == {'fine_random_split_vary', 'fine_leave_one_paper_out'}
    assert set(out['systems']['fine_leave_one_paper_out']) == {'crf', 'crf_ens3'}
    held_out = out['systems']['fine_leave_one_paper_out']['crf']
    assert held_out['n_runs'] == 40 and held_out['all']['n_gold'] == 13280
    assert out['papers']['fine']['10.1007_s11340-021-00758-x.conll'] == {
        'n_gold': 60, 'seen': 0.283, 'seen_other': 0.0, 'unseen': 0.717}
    assert out['gap']['fine']['unseen_share'] == {'random_splits': 0.345, 'held_out': 0.812}
    assert out['types']['fine_leave_one_paper_out']['Value']['n_gold_per_run'] == 41.8
    assert out['methods']['fine_leave_one_paper_out'] == {}         # no method, no vote pair


def test_corpus_coverage_of_the_unseen_names(tmp_path, monkeypatch):
    """The unseen gold names of a held-out paper (316L, 300 MPa), the share of them that the
    self-training corpus holds (316L), cut into words as self-training cuts it."""
    monkeypatch.setattr(nm, 'corpus', lambda granularity, retokenized_test=False: TOY)
    results = _toy_results(PREDS)
    (tmp_path / 'ner_results_fine_leave_one_paper_out_crf_m0.json').write_text(
        json.dumps(results))
    path = tmp_path / 'corpus.txt'
    path.write_text('The 316L steel was tested.\nNo name here.\n')
    out = nm.corpus_coverage(str(tmp_path), path)
    assert out['fine'] == {'B.conll': {'n_unseen': 2, 'in_corpus': 0.5,
                                       'distinct_in_corpus': 0.5}}
    assert out['corpus']['sentences'] == 2 and len(out['corpus']['sha256']) == 64
    assert 'coarse' not in out
    assert nm.corpus_coverage(str(tmp_path), tmp_path / 'missing.txt') == {}


def test_document_folds_are_rebuilt():
    """A run of --document-folds (week 8): its fold's test documents and training sentences."""
    from types import SimpleNamespace

    from ner_baseline import make_folds
    data = [(['the', name], ['O', 'B-Alloy'], f'{doc}.conll')
            for doc, names in (('A', ['Ti64', 'Ti64', 'IN718']), ('B', ['Ti64', '316L']),
                               ('C', ['IN718', 'IN718']), ('D', ['316L', 'Ti64']))
            for name in names]
    folds = make_folds(data, SimpleNamespace(leave_one_paper_out=False, document_folds=2,
                                             split_seed=0), indices=True)
    results = {'protocol': 'document_folds', 'config': {'split_seed': 0, 'document_folds': 2},
               'runs': [{'run_id': name, 'test_index': test,
                         'test_gold': [' '.join(data[i][1]) for i in test],
                         'test_pred': [' '.join(data[i][1]) for i in test]}
                        for name, _, _, test in folds]}
    counts = nm.run_counts(results, data, 'fine')
    for name, train, _, test in folds:
        forms = nm.entity_forms(data, train)
        unseen = sum(nm.kind_of(data[i][0][1:], 'Alloy', forms) == 'unseen' for i in test)
        assert counts[name]['gold', 'unseen'] == unseen
        assert counts[name]['correct', 'seen'] + counts[name]['correct', 'unseen'] == len(test)
    results['runs'][0]['test_index'] = results['runs'][1]['test_index']
    with pytest.raises(nm.MentionError, match='test sentences'):
        nm.run_counts(results, data, 'fine')
