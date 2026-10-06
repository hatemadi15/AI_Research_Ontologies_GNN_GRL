"""Tests for term matching (text_match.py) and leakage-safe folds (splits.py)."""

import random

import splits
from text_match import TermMatcher, lemma_key


def test_word_boundaries_and_longest_match():
    m = TermMatcher(['al', 'fatigue', 'fatigue crack', 'crack'])
    assert m.terms_in('The material failed') == set()          # no 'al' in 'material'
    assert m.find('fatigue crack growth') == [(0, 13, 'fatigue crack')]
    assert m.terms_in('Al alloys and a crack') == {'al', 'crack'}


def test_match_phrase_is_deterministic():
    terms = ['crack', 'fatigue crack', 'growth rate', 'crack growth rate']
    phrase = 'the fatigue crack growth rate'
    results = set()
    for seed in range(5):
        shuffled = terms[:]
        random.Random(seed).shuffle(shuffled)
        results.add(TermMatcher(shuffled).match_phrase(phrase))
    assert len(results) == 1
    assert TermMatcher(terms).match_phrase('crack') == 'crack'
    assert TermMatcher(terms).match_phrase('nothing here') is None


def test_lemma_key_groups_surface_variants():
    assert lemma_key('cracks') == lemma_key('crack')
    assert lemma_key('the carbon steels') == 'carbon steel'
    assert lemma_key('boundaries') == 'boundary'
    assert lemma_key('stress') == 'stress'


def test_group_kfold_is_leakage_safe():
    terms = ['crack', 'cracks', 'alloy', 'alloys', 'grain', 'grains', 'stress',
             'strain', 'fatigue', 'model', 'models', 'data']
    labels = ['Crack', 'Crack', 'Alloy', 'Alloy', 'Grain', 'Grain', 'Stress',
              'Strain', 'Mechanism', 'Model', 'Model', 'DataSet']
    folds = splits.group_kfold_splits(terms, labels, n_splits=3, seed=0)
    assert sorted(t for _, test in folds for t in test) == sorted(terms)
    group = dict(zip(terms, splits.term_groups(terms)))
    for train, test in folds:
        assert not set(train) & set(test)
        assert not {group[t] for t in train} & {group[t] for t in test}
