"""Tests for the CoNLL BIO readers (conll.py)."""

import os
from collections import Counter

import conll
from config import RAW_DIR

FINE_DIR = os.path.join(RAW_DIR, 'fine_grained_ner')


def test_bio_spans_basic_rules():
    tokens = ['Fatigue', 'crack', 'growth', 'in', 'Al', 'alloys', 'x']
    tags = ['B-Crack', 'I-Crack', 'I-Mechanism', 'O', 'B-Alloy', 'I-Alloy', 'I-Alloy']
    # I- of another type closes the entity and is dropped; I- without B is dropped
    assert conll.bio_spans(tokens, tags) == [('fatigue crack', 'Crack'),
                                             ('al alloys x', 'Alloy')]


def test_bio_spans_skips_short_and_dangling():
    tokens = ['a', 'b', 'grain']
    tags = ['B-Value', 'O', 'I-Grain']  # 1-char entity, dangling I-
    assert conll.bio_spans(tokens, tags) == []


def test_hash_token_is_not_a_comment(tmp_path):
    path = tmp_path / 'x.conll'
    path.write_text('where O\n# B-StrainAmplitude\na I-StrainAmplitude\n\n'
                    'next O\n', encoding='utf-8')
    sentences = conll.read_conll_sentences(str(path))
    assert sentences[0] == (['where', '#', 'a'], ['O', 'B-StrainAmplitude',
                                                  'I-StrainAmplitude'])
    _, ents = conll.parse_conll_bio(str(path))
    assert ents[0] == [('# a', 'StrainAmplitude')]


def test_dataset_statistics():
    sentences, doc_entities, doc_ids = conll.load_corpus(FINE_DIR)
    mentions = [m for ents in doc_entities for m in ents]
    assert len(sentences) == 476
    assert len(mentions) == 2168
    assert len(set(doc_ids)) == 4
    counts = conll.term_type_counts(doc_entities)
    assert len(counts) == 1055
    valid = conll.valid_terms(doc_entities)
    assert len(valid) == 315
    assert sum(1 for t in valid if len(counts[t]) > 1) == 47


def test_majority_type_tie_break_uses_global_frequency():
    counts = Counter({'Mechanism': 9, 'FatigueTest': 9, 'LoadingType': 1})
    freq = {'Mechanism': 143, 'FatigueTest': 33}
    assert conll.majority_type(counts, freq) == 'Mechanism'
    assert conll.majority_type(Counter({'B': 1, 'A': 1})) == 'A'
