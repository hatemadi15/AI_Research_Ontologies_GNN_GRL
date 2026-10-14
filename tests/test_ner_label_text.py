"""Label descriptions of --label-semantics (ner_label_text.py; no torch needed).

The real descriptions come from the MMO file and the MaterioMiner corpora, as in
test_ontology_utils.py.
"""
import hashlib
import random
from collections import Counter

import pytest

import ner_baseline as nb
import ner_ensemble as ne
import ner_label_text as lt


def _sent(tokens, tags, doc='p.conll'):
    return tokens.split(), tags.split(), doc


def test_class_names_split_into_words_with_their_spelling():
    assert lt.split_name('CrackInitation') == 'Crack Initation'        # the class's own spelling
    assert lt.split_name('MechanicalTestDataSet') == 'Mechanical Test Data Set'
    assert lt.split_name('SNCurve') == 'SN Curve'
    assert lt.split_name('FacecenteredCubic') == 'Facecentered Cubic'
    assert [lt.split_name(n) for n in ('H', 'Al', 'Value', 'MAT')] == ['H', 'Al', 'Value', 'MAT']


def test_fine_descriptions_name_and_definition_or_the_name_alone():
    definitions = {'CrackInitation': ' Stage after\n the  nucleation. ', 'Value': ''}
    assert lt.fine_descriptions(['Value', 'CrackInitation', 'GrainSize'], definitions) == [
        'Value', 'Crack Initation: Stage after the nucleation.', 'Grain Size']
    assert lt.fine_descriptions([], definitions) == []


def test_fine_descriptions_of_the_27_classes_use_their_mmo_definitions():
    labels, _ = nb.select_labels(nb.load_split_corpus('fine'), 27)
    definitions = lt.mmo_definitions(labels)
    assert all(definitions.values())                    # every top-27 class has a definition
    texts = dict(zip(labels, lt.fine_descriptions(labels)))
    assert texts['Value'] == ('Value: A Value is composed of a numerical value and optionally '
                              'a physical unit.')
    assert texts['StrainRate'] == 'Strain Rate: The time derivate of the strain'
    assert texts['CrackInitation'].startswith('Crack Initation: The crack initiation stage is ')
    for label, text in texts.items():
        assert text == f"{lt.split_name(label)}: {' '.join(definitions[label].split())}"
    # a name outside the MMO keeps its name alone
    assert lt.mmo_definitions(['NoSuchClass']) == {'NoSuchClass': ''}
    assert lt.fine_descriptions(['NoSuchClass']) == ['No Such Class']


def test_entity_spans_read_tags_as_the_ensemble_does():
    assert lt.entity_spans('B-A I-A O B-B B-B I-B'.split()) == [(0, 2, 'A'), (3, 4, 'B'),
                                                                 (4, 6, 'B')]
    # an I- tag that continues nothing starts an entity (seqeval's default mode)
    assert lt.entity_spans('I-A I-B O I-B'.split()) == [(0, 1, 'A'), (1, 2, 'B'), (3, 4, 'B')]
    assert lt.entity_spans([]) == [] and lt.entity_spans(['O']) == []
    rng = random.Random(0)
    for _ in range(500):
        tags = [rng.choice(['O', 'B-A', 'I-A', 'B-B', 'I-B']) for _ in range(rng.randint(0, 8))]
        assert lt.entity_spans(tags) == ne.tags_to_spans(tags)[0]


def test_coarse_types_are_described_by_the_fine_classes_of_the_same_spans():
    fine = [_sent('steel and alloy crack', 'B-Steel O B-Alloy B-Crack'),
            _sent('high cycle fatigue', 'B-HighcycleFatigue I-HighcycleFatigue I-HighcycleFatigue'),
            _sent('the steel', 'O B-Steel'),               # tokenised differently: skipped
            _sent('steel bar', 'B-Steel I-Steel'),          # no coarse entity of the same span
            _sent('alloy', 'B-Alloy')]
    coarse = [_sent('steel and alloy crack', 'B-MAT O B-MAT B-DAM'),
              _sent('high cycle fatigue', 'B-MEC I-MEC I-MEC'),
              _sent('the steel .', 'O B-MAT O'),
              _sent('steel bar', 'B-MAT B-OBJ'),
              _sent('alloy', 'B-MAT')]
    assert lt.coarse_classes(fine, coarse) == {
        'MAT': Counter({'Alloy': 2, 'Steel': 1}), 'DAM': Counter({'Crack': 1}),
        'MEC': Counter({'HighcycleFatigue': 1})}
    # by decreasing count, names in words; a type without a paired class: its code alone
    assert lt.coarse_descriptions(['OBJ', 'MAT', 'MEC', 'DAM'], fine, coarse) == [
        'OBJ', 'MAT: Alloy, Steel', 'MEC: Highcycle Fatigue', 'DAM: Crack']
    with pytest.raises(ValueError):                    # the corpora hold the same sentences
        lt.coarse_classes(fine, coarse[:4])


def test_coarse_descriptions_break_ties_by_name_and_keep_ten_names():
    counts = {'Zeta': 3, 'GammaRay': 2, 'Beta': 2, **{f'Class{c}': 1 for c in 'HGFEDCBA'}}
    pairs = [(_sent('x', f'B-{name}'), _sent('x', 'B-T')) for name, n in counts.items()
             for _ in range(n)]
    expected = ['T: Zeta, Beta, Gamma Ray, Class A, Class B, Class C, Class D, Class E, '
                'Class F, Class G']
    for seed in range(5):                              # the corpus order does not matter
        random.Random(seed).shuffle(pairs)
        fine, coarse = [f for f, _ in pairs], [c for _, c in pairs]
        assert lt.coarse_descriptions(['T'], fine, coarse) == expected
    assert lt.coarse_descriptions(['T'], fine, coarse, max_names=2) == ['T: Zeta, Beta']


def test_coarse_descriptions_of_materiominer():
    fine, coarse = nb.load_split_corpus('fine'), nb.load_split_corpus('coarse')
    types, _ = nb.select_labels(coarse)
    texts = dict(zip(types, lt.coarse_descriptions(types, fine, coarse)))
    assert len(texts) == 27
    assert texts['VAL'] == 'VAL: Value' and texts['MAT'] == 'MAT: Alloy, Material'
    assert texts['DAT'] == 'DAT: Data Set, Mechanical Test Data Set, SN Curve'
    # 16 classes for MEC: the 10th place is a tie at 3 entities, broken by name
    assert texts['MEC'] == ('MEC: Mechanism, Crack Initation, Highcycle Fatigue, Crack '
                            'Propagation, Failure, Fracture, Precipitation Hardening, Very '
                            'Highcycle Fatigue, Lowcycle Fatigue, Crack Growth Behaviour')
    assert all(text.startswith(f'{code}: ') and len(text.split(', ')) <= lt.MAX_NAMES
               for code, text in texts.items())
    # 2,088 fine-grained entities in the 461 sentences tokenised alike, 2,084 of them paired
    assert sum(fine[i][0] == coarse[i][0] for i in range(len(fine))) == 461
    classes = lt.coarse_classes(fine, coarse)
    assert sum(sum(c.values()) for c in classes.values()) == 2084
    assert lt.coarse_descriptions(types, fine, coarse) == list(texts.values())


def test_descriptions_hash_covers_labels_texts_and_order():
    sha = lt.descriptions_sha256(['A', 'B'], ['A: x', 'B'])
    assert sha == hashlib.sha256(b'[["A", "A: x"], ["B", "B"]]').hexdigest()
    assert sha != lt.descriptions_sha256(['B', 'A'], ['B', 'A: x'])   # the head's order
    assert sha != lt.descriptions_sha256(['A', 'B'], ['A: y', 'B'])
    with pytest.raises(ValueError):
        lt.descriptions_sha256(['A'], ['A', 'B'])
