"""Tests for the shared ontology loader (ontology_utils.py) on the real TTL."""

import os

import conll
from config import RAW_DIR
from ontology_utils import MMO_NS, PROV_NS, load_ontology, local_name


def test_classes_labels_definitions():
    onto = load_ontology()
    assert len(onto.classes) == 428
    assert sum(1 for u in onto.uris if onto.definition(u)) == 420
    uri = onto.resolve('FatigueTest')
    assert onto.primary_label(uri) == 'Fatigue test'
    assert onto.resolve('fatigue test') == uri
    # alternative labels resolve too
    assert onto.resolve('Endurance limit') == onto.resolve('FatigueLimit')


def test_all_fine_ner_types_resolve():
    onto = load_ontology()
    _, doc_entities, _ = conll.load_corpus(os.path.join(RAW_DIR, 'fine_grained_ner'))
    types = {t for ents in doc_entities for _, t in ents}
    assert len(types) == 177
    assert all(onto.resolve(t) for t in types)


def test_ancestors_exclude_root_by_default():
    onto = load_ontology()
    uri = onto.resolve('FatigueTest')
    names = {local_name(a) for a in onto.ancestors(uri)}
    assert {'MechanicalTest', 'CharacterizationProcess', 'Process'} <= names
    assert onto.root not in onto.ancestors(uri)
    assert onto.root in onto.ancestors(uri, include_root=True)
    assert uri in onto.ancestors(uri, include_self=True)
    assert onto.is_subclass(uri, onto.resolve('MechanicalTest'))
    assert not onto.is_subclass(onto.resolve('MechanicalTest'), uri)


def test_reduced_edges_skip_intermediate_classes():
    onto = load_ontology()
    test, mech, proc = (onto.resolve(n) for n in ('FatigueTest', 'MechanicalTest', 'Process'))
    assert onto.reduced_edges({test, proc}) == {(test, proc)}
    assert onto.reduced_edges({test, mech, proc}) >= {(test, mech)}
    assert (test, proc) not in onto.reduced_edges({test, mech, proc})


def test_restrictions_and_property_hierarchy():
    onto = load_ontology()
    triples = onto.restriction_triples()
    assert len(triples) == 70
    assert (onto.resolve('FatigueTest'), 'https://w3id.org/pmd/co/input',
            onto.resolve('FatigueTestSpecimen')) in triples
    cause_ancestors = onto.property_ancestors(MMO_NS + 'causeOf')
    assert PROV_NS + 'influenced' in cause_ancestors
    assert MMO_NS + 'associatedWith' in cause_ancestors
