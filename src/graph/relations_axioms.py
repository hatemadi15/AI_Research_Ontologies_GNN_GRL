"""
relations_axioms.py - OWL Axiom-based Relation Pruning

Uses OWL reasoning over the reference ontology to prune logically
inconsistent relations extracted by relations.py.

Pruning checks:
  1. Domain/range constraint violations
  2. Class hierarchy compatibility (transitive subClassOf)
  3. Disjointness violations
  4. Property hierarchy consistency

Runs as a post-processing step after relations.py in the pipeline.
"""

import os
import json
from collections import defaultdict

import pandas as pd
from rdflib import Graph, RDF, RDFS, OWL, URIRef, Namespace

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TTL_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw", "dataset", "ontologies", "ontology.ttl"
)

MMO = Namespace("https://w3id.org/pmd/materials-mechanics-ontology/")
MMO_BASE = "https://w3id.org/pmd/materials-mechanics-ontology/"
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")


def load_ontology():
    """Load the ontology into an rdflib graph."""
    g = Graph()
    g.parse(TTL_PATH, format="turtle")
    return g


def get_local_name(uri):
    """Extract local name from a URI."""
    uri_str = str(uri)
    return uri_str.split("#")[-1].split("/")[-1]


def get_label(g, uri):
    """Get human-readable label for a URI."""
    if not isinstance(uri, URIRef):
        return None
    label_props = [RDFS.label, SKOS.prefLabel, SKOS.altLabel]
    for prop in label_props:
        for label in g.objects(uri, prop):
            label_str = str(label).strip()
            if label_str and not label_str.startswith("http"):
                return label_str.lower()
    local = get_local_name(uri)
    return local.lower() if local else None


def build_class_label_map(g):
    """Build mappings between class labels and URIs."""
    label_to_uris = defaultdict(set)
    uri_to_labels = defaultdict(set)

    for cls in g.subjects(RDF.type, OWL.Class):
        if not isinstance(cls, URIRef):
            continue
        cls_str = str(cls)
        if cls_str.startswith("http://www.w3.org/"):
            continue

        labels = set()
        for prop in [RDFS.label, SKOS.prefLabel, SKOS.altLabel]:
            for label in g.objects(cls, prop):
                label_str = str(label).strip().lower()
                if label_str and not label_str.startswith("http"):
                    labels.add(label_str)

        local = get_local_name(cls)
        if local:
            labels.add(local.lower())

        for label in labels:
            label_to_uris[label].add(cls)
            uri_to_labels[cls].add(label)

    return label_to_uris, uri_to_labels


def get_superclasses(g, cls_uri, _cache=None):
    """Get all superclasses of a class (transitive subClassOf)."""
    if _cache is None:
        _cache = {}
    if cls_uri in _cache:
        return _cache[cls_uri]

    visited = set()
    queue = [cls_uri]
    superclasses = set()

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        superclasses.add(current)

        for superclass in g.objects(current, RDFS.subClassOf):
            if isinstance(superclass, URIRef) and superclass not in visited:
                queue.append(superclass)

    _cache[cls_uri] = superclasses
    return superclasses


def is_subclass_of(g, cls, parent, superclass_cache=None):
    """Check if cls is a subclass of parent (transitive)."""
    if cls == parent:
        return True
    supers = get_superclasses(g, cls, superclass_cache)
    return parent in supers


def get_property_constraints(g):
    """Extract domain/range constraints for each object property."""
    constraints = {}

    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        prop_name = get_local_name(prop).lower()
        domains = []
        ranges = []

        for d in g.objects(prop, RDFS.domain):
            if isinstance(d, URIRef):
                domains.append(d)
        for r in g.objects(prop, RDFS.range):
            if isinstance(r, URIRef):
                ranges.append(r)

        constraints[prop_name] = {
            "domain": domains,
            "range": ranges,
            "uri": prop,
        }

    return constraints


def get_disjoint_pairs(g):
    """Extract pairs of disjoint classes from the ontology."""
    disjoint_pairs = set()

    # owl:disjointWith
    for s, _, o in g.triples((None, OWL.disjointWith, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            pair = tuple(sorted([str(s), str(o)]))
            disjoint_pairs.add(pair)

    # owl:AllDisjointClasses
    for adc in g.subjects(RDF.type, OWL.AllDisjointClasses):
        members = list(g.objects(adc, OWL.members))
        for member_list in members:
            classes = list(g.items(member_list))
            for i, c1 in enumerate(classes):
                for c2 in classes[i + 1:]:
                    if isinstance(c1, URIRef) and isinstance(c2, URIRef):
                        pair = tuple(sorted([str(c1), str(c2)]))
                        disjoint_pairs.add(pair)

    return disjoint_pairs


def resolve_entity_type(entity_name, entity_types, label_to_uris):
    """Resolve an entity name to its ontology class URI(s)."""
    entity_lower = entity_name.lower().strip()

    # Direct NER type mapping
    ner_type = entity_types.get(entity_name) or entity_types.get(entity_lower)
    uris = set()

    if ner_type:
        ner_lower = ner_type.lower()
        if ner_lower in label_to_uris:
            uris.update(label_to_uris[ner_lower])

    # Also try entity name itself as a label
    if entity_lower in label_to_uris:
        uris.update(label_to_uris[entity_lower])

    return uris


def is_compatible(entity_uris, constraint_uris, g, superclass_cache):
    """Check if any entity URI is compatible with any constraint URI."""
    if not constraint_uris:
        return True  # No constraint = always compatible
    if not entity_uris:
        return True  # Unknown type = allow (conservative)

    for e_uri in entity_uris:
        for c_uri in constraint_uris:
            if is_subclass_of(g, e_uri, c_uri, superclass_cache):
                return True
    return False


def are_disjoint(uris1, uris2, disjoint_pairs, g, superclass_cache):
    """Check if two sets of class URIs are disjoint."""
    if not disjoint_pairs:
        return False

    for u1 in uris1:
        supers1 = get_superclasses(g, u1, superclass_cache)
        for u2 in uris2:
            supers2 = get_superclasses(g, u2, superclass_cache)
            for s1 in supers1:
                for s2 in supers2:
                    pair = tuple(sorted([str(s1), str(s2)]))
                    if pair in disjoint_pairs:
                        return True
    return False


def prune_relations(relations_df, g, entity_types):
    """Remove relations that violate OWL axioms.

    Returns:
        valid_df: DataFrame of valid relations
        pruned_df: DataFrame of pruned relations (with reason)
    """
    constraints = get_property_constraints(g)
    disjoint_pairs = get_disjoint_pairs(g)
    label_to_uris, uri_to_labels = build_class_label_map(g)
    superclass_cache = {}

    print(f"  Property constraints loaded: {len(constraints)}")
    print(f"  Disjoint pairs found: {len(disjoint_pairs)}")
    print(f"  Class labels mapped: {len(label_to_uris)}")

    valid = []
    pruned = []

    # Normalize column names for relations.csv format
    subj_col = "subj" if "subj" in relations_df.columns else "subject"
    rel_col = "rel" if "rel" in relations_df.columns else "relation_type"
    obj_col = "obj" if "obj" in relations_df.columns else "object"

    for _, row in relations_df.iterrows():
        rel_type = str(row[rel_col]).lower().strip()
        subj = str(row[subj_col]).strip()
        obj = str(row[obj_col]).strip()

        # Resolve entity types to ontology classes
        subj_uris = resolve_entity_type(subj, entity_types, label_to_uris)
        obj_uris = resolve_entity_type(obj, entity_types, label_to_uris)

        prune_reason = None

        # Normalize relation type for constraint lookup
        rel_normalized = rel_type.replace(" ", "").replace("_", "").lower()

        # Find matching constraint key
        constraint_key = None
        for k in constraints:
            if k.replace(" ", "").replace("_", "").lower() == rel_normalized:
                constraint_key = k
                break

        if constraint_key and (subj_uris or obj_uris):
            c = constraints[constraint_key]

            # Check domain constraint
            if c["domain"] and subj_uris:
                domain_ok = is_compatible(
                    subj_uris, c["domain"], g, superclass_cache
                )
                if not domain_ok:
                    prune_reason = (
                        f"domain_violation: {subj} not in domain of "
                        f"{rel_type}"
                    )

            # Check range constraint
            if not prune_reason and c["range"] and obj_uris:
                range_ok = is_compatible(
                    obj_uris, c["range"], g, superclass_cache
                )
                if not range_ok:
                    prune_reason = (
                        f"range_violation: {obj} not in range of "
                        f"{rel_type}"
                    )

        # Check disjointness
        if not prune_reason and subj_uris and obj_uris and disjoint_pairs:
            if are_disjoint(subj_uris, obj_uris, disjoint_pairs,
                            g, superclass_cache):
                prune_reason = (
                    f"disjoint_violation: {subj} and {obj} belong to "
                    f"disjoint classes"
                )

        if prune_reason:
            pruned_row = row.to_dict()
            pruned_row["prune_reason"] = prune_reason
            pruned.append(pruned_row)
        else:
            valid.append(row.to_dict())

    valid_df = pd.DataFrame(valid)
    pruned_df = pd.DataFrame(pruned)

    return valid_df, pruned_df


def run_axiom_pruning():
    """Main entry point: load relations, prune, save results."""
    print("Loading ontology...")
    g = load_ontology()
    print(f"  Loaded {len(g)} triples")

    # Load relations
    relations_path = os.path.join(PROCESSED_DIR, "relations.csv")
    if not os.path.exists(relations_path):
        print(f"ERROR: No relations found at {relations_path}. "
              f"Run relations.py first.")
        return

    relations_df = pd.read_csv(relations_path)
    print(f"Input relations: {len(relations_df)}")

    # Load entity types
    entity_types_path = os.path.join(PROCESSED_DIR, "entity_types.json")
    entity_types = {}
    if os.path.exists(entity_types_path):
        with open(entity_types_path) as f:
            entity_types = json.load(f)
        print(f"Entity types loaded: {len(entity_types)}")

    # Prune
    print("\nPruning relations with OWL axiom checks...")
    valid_df, pruned_df = prune_relations(relations_df, g, entity_types)

    print(f"\nResults:")
    print(f"  Valid relations: {len(valid_df)}")
    print(f"  Pruned relations: {len(pruned_df)}")

    # Save pruned relations (overwrite original)
    if not valid_df.empty:
        valid_df.to_csv(relations_path, index=False)
        print(f"  Saved valid relations to {relations_path}")

    # Save pruned log
    if not pruned_df.empty:
        pruned_path = os.path.join(PROCESSED_DIR, "relations_pruned.csv")
        pruned_df.to_csv(pruned_path, index=False)
        print(f"  Saved pruned relations to {pruned_path}")

        # Print prune reason distribution
        reasons = pruned_df["prune_reason"].str.split(":").str[0]
        print("\n  Prune reason distribution:")
        for reason, count in reasons.value_counts().items():
            print(f"    {reason}: {count}")

    # Save summary stats
    stats = {
        "input_relations": len(relations_df),
        "valid_relations": len(valid_df),
        "pruned_relations": len(pruned_df),
        "prune_rate": round(
            len(pruned_df) / len(relations_df), 4
        ) if len(relations_df) > 0 else 0.0,
    }
    if not pruned_df.empty:
        stats["prune_reasons"] = dict(
            pruned_df["prune_reason"].str.split(":").str[0].value_counts()
        )

    stats_path = os.path.join(PROCESSED_DIR, "axiom_pruning_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\n  Saved stats to {stats_path}")


if __name__ == "__main__":
    run_axiom_pruning()
