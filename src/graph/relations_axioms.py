"""
relations_axioms.py - OWL Axiom-based Relation Pruning

Uses the axioms of the reference ontology to prune logically inconsistent
relations extracted by relations.py.

Pruning checks:
  1. Domain/range constraint violations (transitive subClassOf)
  2. Disjointness violations (the MMO file declares none, so this check is
     currently inactive)

Entity classes come from the pipeline's own typing decisions
(predicted_types.json); gold NER types are used only when ORACLE_TYPES is
set. Constraint classes that are not defined in the loaded ontology (e.g.
prov:Entity, the domain of alignedWith) are treated as unknown instead of as
violations, since no MMO class can be shown to be a subclass of them.

Reads relations_raw.csv (so re-running is idempotent) and always writes
relations.csv (valid relations), relations_pruned.csv and
axiom_pruning_stats.json.
"""

import os
import json

import pandas as pd

from config import (GOLD_TYPES_FILE, ORACLE_TYPES, PREDICTED_TYPES_BASE_FILE,
                    PREDICTED_TYPES_FILE, PROCESSED_DIR)
from ontology_utils import load_ontology, local_name


def load_term_classes(onto):
    """Term -> class URI: predicted types, or gold majority types in ORACLE mode."""
    if ORACLE_TYPES:
        path = os.path.join(PROCESSED_DIR, GOLD_TYPES_FILE)
        with open(path) as f:
            gold_types = json.load(f)
        print("  [ORACLE] using gold NER types for entity classes")
        return {t: onto.resolve(max(sorted(c), key=c.get))
                for t, c in gold_types.items() if c}
    for name in (PREDICTED_TYPES_FILE, PREDICTED_TYPES_BASE_FILE):
        path = os.path.join(PROCESSED_DIR, name)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                preds = json.load(f)
            print(f"  Entity classes from {name}")
            return {t: p.get('class_uri') for t, p in preds.items()}
    print("  No typing decisions found; domain/range checks are skipped")
    return {}


def get_property_constraints(onto):
    """Lower-cased local property name -> (domain URIs, range URIs)."""
    constraints = {}
    for uri, info in onto.properties.items():
        constraints[local_name(uri).lower()] = (info['domain'], info['range'])
    return constraints


def resolve_entity_classes(entity, term_classes, onto):
    """Class URIs of an extracted entity (its typing decision or own label)."""
    uris = set()
    cls = term_classes.get(entity)
    if cls in onto.classes:
        uris.add(cls)
    own = onto.resolve(entity)
    if own:
        uris.add(own)
    return uris


def is_compatible(entity_uris, constraint_uris, onto):
    """True unless every known constraint class is incompatible."""
    known = {c for c in constraint_uris if c in onto.classes}
    if not known or not entity_uris:
        return True  # no constraint / unknown type / constraint outside ontology
    for e_uri in entity_uris:
        anc = onto.ancestors(e_uri, include_self=True, include_root=True)
        if anc & known:
            return True
    return False


def prune_relations(relations_df, term_classes, onto, disjoint_pairs=()):
    """Remove relations that violate OWL axioms.

    Returns:
        valid_df: DataFrame of valid relations
        pruned_df: DataFrame of pruned relations (with reason)
    """
    constraints = get_property_constraints(onto)
    disjoint_pairs = set(disjoint_pairs)
    print(f"  Property constraints loaded: {len(constraints)}")
    print(f"  Disjoint pairs found: {len(disjoint_pairs)}")

    valid, pruned = [], []
    for _, row in relations_df.iterrows():
        rel_type = str(row['rel']).replace(" ", "").replace("_", "").lower()
        subj, obj = str(row['subj']).strip(), str(row['obj']).strip()
        subj_uris = resolve_entity_classes(subj, term_classes, onto)
        obj_uris = resolve_entity_classes(obj, term_classes, onto)

        prune_reason = None
        if rel_type in constraints:
            domain, range_ = constraints[rel_type]
            if not is_compatible(subj_uris, domain, onto):
                prune_reason = f"domain_violation: {subj} not in domain of {row['rel']}"
            elif not is_compatible(obj_uris, range_, onto):
                prune_reason = f"range_violation: {obj} not in range of {row['rel']}"

        if not prune_reason and disjoint_pairs and subj_uris and obj_uris:
            for s in subj_uris:
                for o in obj_uris:
                    s_anc = onto.ancestors(s, include_self=True, include_root=True)
                    o_anc = onto.ancestors(o, include_self=True, include_root=True)
                    if any(tuple(sorted((a, b))) in disjoint_pairs
                           for a in s_anc for b in o_anc):
                        prune_reason = (f"disjoint_violation: {subj} and {obj} "
                                        f"belong to disjoint classes")

        record = row.to_dict()
        if prune_reason:
            record['prune_reason'] = prune_reason
            pruned.append(record)
        else:
            valid.append(record)

    columns = list(relations_df.columns)
    return (pd.DataFrame(valid, columns=columns),
            pd.DataFrame(pruned, columns=columns + ['prune_reason']))


def get_disjoint_pairs(ttl_path):
    """owl:disjointWith / owl:AllDisjointClasses pairs of the ontology file."""
    from rdflib import Graph, OWL, RDF, URIRef

    g = Graph()
    g.parse(ttl_path, format="turtle")
    pairs = set()
    for s, _, o in g.triples((None, OWL.disjointWith, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            pairs.add(tuple(sorted([str(s), str(o)])))
    for adc in g.subjects(RDF.type, OWL.AllDisjointClasses):
        for member_list in g.objects(adc, OWL.members):
            classes = [c for c in g.items(member_list) if isinstance(c, URIRef)]
            for i, c1 in enumerate(classes):
                for c2 in classes[i + 1:]:
                    pairs.add(tuple(sorted([str(c1), str(c2)])))
    return pairs


def run_axiom_pruning():
    """Main entry point: load relations, prune, save results."""
    from ontology_utils import DEFAULT_TTL_PATH

    print("Loading ontology...")
    onto = load_ontology()
    raw_path = os.path.join(PROCESSED_DIR, "relations_raw.csv")
    relations_path = os.path.join(PROCESSED_DIR, "relations.csv")
    source = raw_path if os.path.exists(raw_path) else relations_path
    if not os.path.exists(source):
        print(f"ERROR: No relations found at {raw_path}. Run relations.py first.")
        return

    relations_df = pd.read_csv(source)
    print(f"Input relations: {len(relations_df)} ({os.path.basename(source)})")
    term_classes = load_term_classes(onto)

    print("\nPruning relations with OWL axiom checks...")
    valid_df, pruned_df = prune_relations(
        relations_df, term_classes, onto, get_disjoint_pairs(DEFAULT_TTL_PATH))
    print(f"\nResults:\n  Valid relations: {len(valid_df)}\n"
          f"  Pruned relations: {len(pruned_df)}")

    valid_df.to_csv(relations_path, index=False)
    pruned_df.to_csv(os.path.join(PROCESSED_DIR, "relations_pruned.csv"), index=False)
    print(f"  Saved valid relations to {relations_path}")

    stats = {
        "input_relations": len(relations_df),
        "valid_relations": len(valid_df),
        "pruned_relations": len(pruned_df),
        "prune_rate": round(len(pruned_df) / len(relations_df), 4)
        if len(relations_df) else 0.0,
        "entity_classes": "gold (ORACLE)" if ORACLE_TYPES else "predicted",
    }
    if len(pruned_df):
        reasons = pruned_df["prune_reason"].str.split(":").str[0].value_counts()
        stats["prune_reasons"] = {k: int(v) for k, v in reasons.items()}
        print("\n  Prune reason distribution:")
        for reason, count in reasons.items():
            print(f"    {reason}: {count}")
    with open(os.path.join(PROCESSED_DIR, "axiom_pruning_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    run_axiom_pruning()
