"""
extract_gold_relations.py - Extract Gold Standard Relations from Ontology

Parses the ontology TTL file to extract gold standard relations using:
  1. Direct property assertions between classes
  2. OWL restrictions (someValuesFrom, allValuesFrom)
  3. Property domain/range as soft gold relations
  4. SubProperty hierarchy relations

Outputs:
  - data/processed/gold_relations.csv
  - data/processed/gold_relations_stats.json
"""

import os
import json
from collections import Counter

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
PROV = Namespace("http://www.w3.org/ns/prov#")
PMD_CO = Namespace("https://w3id.org/pmd/co/")
DC_TERMS = Namespace("http://purl.org/dc/terms/")

# The 17 ontology object properties we care about (+ composedOf from PMDco)
KNOWN_PROPERTIES = {
    str(MMO.causeOf): "causeOf",
    str(MMO.necessaryCauseOf): "necessaryCauseOf",
    str(MMO.sufficientCauseOf): "sufficientCauseOf",
    str(MMO.correlatedWith): "correlatedWith",
    str(MMO.associatedWith): "associatedWith",
    str(MMO.isPartOf): "isPartOf",
    str(MMO.constrains): "constrains",
    str(MMO.hardConstrains): "hardConstrains",
    str(MMO.softConstrains): "softConstrains",
    str(MMO.measures): "measures",
    str(MMO.growsInto): "growsInto",
    str(MMO.initiatesAt): "initiatesAt",
    str(MMO.precedes): "precedes",
    str(MMO.alignedWith): "alignedWith",
    str(MMO.parallelTo): "parallelTo",
    str(MMO.spatiallyCoincidesWith): "spatiallyCoincidesWith",
    str(MMO.reliesOn): "reliesOn",
    # PMDco properties used in restrictions
    str(PMD_CO.composedOf): "composedOf",
    str(PMD_CO.characteristic): "hasCharacteristic",
    str(PMD_CO.input): "hasInput",
    str(PMD_CO.executedBy): "executedBy",
    str(DC_TERMS.isPartOf): "isPartOf",
    str(PROV.influenced): "influenced",
    str(PROV.generated): "generated",
    str(PROV.hadDerivation): "hadDerivation",
}


def get_label(g, uri):
    """Get human-readable label for a URI."""
    if not isinstance(uri, URIRef):
        return None

    label_props = [RDFS.label, SKOS.prefLabel, SKOS.altLabel,
                   MMO.altLabel, MMO.prefLabel]
    for prop in label_props:
        for label in g.objects(uri, prop):
            label_str = str(label).strip()
            if label_str and not label_str.startswith("http"):
                return label_str

    # Fallback: extract local name from URI
    uri_str = str(uri)
    fragment = uri_str.split("#")[-1].split("/")[-1]
    if fragment and fragment[0].isupper():
        return fragment
    return None


def get_local_name(uri):
    """Extract local name from a URI."""
    uri_str = str(uri)
    return uri_str.split("#")[-1].split("/")[-1]


def get_property_label(g, prop_uri):
    """Get the short relation name for a property URI."""
    prop_str = str(prop_uri)
    if prop_str in KNOWN_PROPERTIES:
        return KNOWN_PROPERTIES[prop_str]
    label = get_label(g, prop_uri)
    if label:
        return label
    return get_local_name(prop_uri)


def extract_gold_relations():
    """Extract all gold standard relations from ontology TTL."""
    print(f"Loading ontology from {TTL_PATH}")
    g = Graph()
    g.parse(TTL_PATH, format="turtle")
    print(f"  Loaded {len(g)} triples")

    gold_relations = []
    seen = set()

    def add_relation(subj_label, rel_label, obj_label, source):
        key = (subj_label.lower(), rel_label.lower(), obj_label.lower())
        if key not in seen:
            seen.add(key)
            gold_relations.append({
                "subject": subj_label,
                "relation": rel_label,
                "object": obj_label,
                "source": source,
            })

    # ---- Method 1: Direct property assertions between named classes ----
    print("  Method 1: Direct property assertions...")
    skip_preds = {
        RDF.type, RDFS.subClassOf, RDFS.label, RDFS.comment,
        RDFS.domain, RDFS.range, RDFS.subPropertyOf,
        OWL.equivalentClass, OWL.imports, OWL.versionIRI,
        OWL.onProperty, OWL.someValuesFrom, OWL.allValuesFrom,
        OWL.inverseOf,
    }
    skip_prefixes = (
        "http://www.w3.org/",
        "http://purl.org/dc/elements/",
        "http://purl.org/dc/terms/created",
        "http://purl.org/dc/terms/license",
    )
    m1_count = 0
    for s, p, o in g:
        if p in skip_preds:
            continue
        if any(str(p).startswith(prefix) for prefix in skip_prefixes):
            continue
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue

        s_label = get_label(g, s)
        p_label = get_property_label(g, p)
        o_label = get_label(g, o)

        if s_label and o_label and p_label:
            # Only include if subject or object is a class
            s_is_class = (s, RDF.type, OWL.Class) in g
            o_is_class = (o, RDF.type, OWL.Class) in g
            if s_is_class or o_is_class:
                add_relation(s_label, p_label, o_label, "direct_assertion")
                m1_count += 1
    print(f"    Found {m1_count} direct assertion relations")

    # ---- Method 2: OWL restrictions (someValuesFrom, allValuesFrom) ----
    print("  Method 2: OWL restrictions...")
    m2_count = 0
    for restriction in g.subjects(RDF.type, OWL.Restriction):
        on_property = list(g.objects(restriction, OWL.onProperty))
        some_values = list(g.objects(restriction, OWL.someValuesFrom))
        all_values = list(g.objects(restriction, OWL.allValuesFrom))

        if not on_property:
            continue

        # Handle inverse properties
        prop = on_property[0]
        is_inverse = False
        if not isinstance(prop, URIRef):
            # Check for owl:inverseOf blank node
            inv = list(g.objects(prop, OWL.inverseOf))
            if inv:
                prop = inv[0]
                is_inverse = True
            else:
                continue

        prop_label = get_property_label(g, prop)

        for val in some_values + all_values:
            if not isinstance(val, URIRef):
                continue
            val_label = get_label(g, val)
            if not val_label:
                continue

            # Find which class has this restriction as a superclass
            for cls in g.subjects(RDFS.subClassOf, restriction):
                if not isinstance(cls, URIRef):
                    continue
                cls_label = get_label(g, cls)
                if not cls_label:
                    continue

                if is_inverse:
                    add_relation(val_label, prop_label, cls_label,
                                 "owl_restriction")
                else:
                    add_relation(cls_label, prop_label, val_label,
                                 "owl_restriction")
                m2_count += 1
    print(f"    Found {m2_count} OWL restriction relations")

    # ---- Method 3: Property domain/range as soft gold ----
    print("  Method 3: Property domain/range...")
    m3_count = 0
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        prop_label = get_property_label(g, prop)
        domains = [d for d in g.objects(prop, RDFS.domain)
                   if isinstance(d, URIRef)]
        ranges = [r for r in g.objects(prop, RDFS.range)
                  if isinstance(r, URIRef)]

        for d in domains:
            for r in ranges:
                d_label = get_label(g, d)
                r_label = get_label(g, r)
                if d_label and r_label:
                    add_relation(d_label, prop_label, r_label,
                                 "domain_range")
                    m3_count += 1
    print(f"    Found {m3_count} domain/range relations")

    # ---- Method 4: SubProperty hierarchy (implies relation subsumption) ----
    print("  Method 4: SubProperty hierarchy...")
    m4_count = 0
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        prop_label = get_property_label(g, prop)
        for parent_prop in g.objects(prop, RDFS.subPropertyOf):
            if isinstance(parent_prop, URIRef):
                parent_label = get_property_label(g, parent_prop)
                if prop_label and parent_label:
                    add_relation(prop_label, "subPropertyOf", parent_label,
                                 "property_hierarchy")
                    m4_count += 1
    print(f"    Found {m4_count} property hierarchy relations")

    # Build DataFrame
    df = pd.DataFrame(gold_relations)
    print(f"\nTotal unique gold relations: {len(df)}")

    # Save CSV
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    csv_path = os.path.join(PROCESSED_DIR, "gold_relations.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved gold relations to {csv_path}")

    # Stats
    stats = {
        "total_relations": len(df),
        "by_relation_type": dict(Counter(df["relation"])),
        "by_source": dict(Counter(df["source"])),
        "unique_subjects": len(df["subject"].unique()),
        "unique_objects": len(df["object"].unique()),
        "unique_relations": len(df["relation"].unique()),
    }
    stats_path = os.path.join(PROCESSED_DIR, "gold_relations_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")

    # Print distribution
    print("\nRelation type distribution:")
    for rel_type, count in sorted(stats["by_relation_type"].items(),
                                   key=lambda x: -x[1]):
        print(f"  {rel_type}: {count}")

    print("\nSource distribution:")
    for source, count in sorted(stats["by_source"].items(),
                                 key=lambda x: -x[1]):
        print(f"  {source}: {count}")

    return df


if __name__ == "__main__":
    extract_gold_relations()
