"""
eval_relations.py - Class-level Evaluation of Extracted Relations

MaterioMiner has no relation annotations, so extracted (term, relation, term)
triples cannot be scored against text-level gold. What the ontology does
provide is class-level knowledge: its existential restrictions, e.g.
"FatigueTest subClassOf (input some FatigueTestSpecimen)". This script

  1. lifts each extracted triple to class level through the pipeline's typing
     decisions (predicted_types.json),
  2. maps relation names to ontology properties (hasProperty ->
     pmd:characteristic, influencedBy -> inverse of prov:influenced, MMO
     relations to themselves) and normalises part-whole relations
     (mmo:isPartOf, dcterms:isPartOf and the inverse of pmd:composedOf) to one
     canonical property,
  3. compares the lifted triples with the restriction triples; a predicted
     property also matches a gold super-property (causeOf matches
     prov:influenced because causeOf is declared a sub-property of it).

A diagnostic lifted through the gold majority types is also reported (gold
used for evaluation only). The old version compared term-level triples with
class-level gold strings, which could only match by accident.
"""

import os
import json

import pandas as pd

import conll
import metrics
from config import PREDICTED_TYPES_FILE, PREDICTED_TYPES_BASE_FILE, PROCESSED_DIR, RAW_DIR
from ontology_utils import MMO_NS, PMD_NS, PROV_NS, load_ontology, local_name

DCT_NS = "http://purl.org/dc/terms/"
PART_OF = MMO_NS + "isPartOf"   # canonical part-whole property

MMO_RELATIONS = [
    'causeOf', 'necessaryCauseOf', 'sufficientCauseOf', 'correlatedWith',
    'associatedWith', 'constrains', 'hardConstrains', 'softConstrains',
    'measures', 'growsInto', 'initiatesAt', 'precedes', 'alignedWith',
    'parallelTo', 'spatiallyCoincidesWith', 'reliesOn', 'isPartOf',
]
# Extractor relation name -> (property URI, inverse?)
REL_TO_PROPERTY = {name: (MMO_NS + name, False) for name in MMO_RELATIONS}
REL_TO_PROPERTY.update({
    'hasProperty': (PMD_NS + 'characteristic', False),
    'composedOf': (PMD_NS + 'composedOf', False),
    'influencedBy': (PROV_NS + 'influenced', True),
})


def canonical_triple(s, p, o):
    """Normalise part-whole properties: (part, PART_OF, whole)."""
    if p in (MMO_NS + 'isPartOf', DCT_NS + 'isPartOf'):
        return (s, PART_OF, o)
    if p == PMD_NS + 'composedOf':
        return (o, PART_OF, s)
    return (s, p, o)


def lift_triples(relations_df, term_classes, onto):
    """Map extracted term triples to canonical class-level triples."""
    lifted = set()
    unmapped = 0
    for subj, rel, obj in zip(relations_df['subj'], relations_df['rel'],
                              relations_df['obj']):
        mapping = REL_TO_PROPERTY.get(str(rel))
        if mapping is None:
            unmapped += 1
            continue
        prop, inverse = mapping
        s_cls, o_cls = term_classes.get(subj), term_classes.get(obj)
        if s_cls not in onto.classes or o_cls not in onto.classes or s_cls == o_cls:
            continue
        if inverse:
            s_cls, o_cls = o_cls, s_cls
        lifted.add(canonical_triple(s_cls, prop, o_cls))
    return lifted, unmapped


def load_predicted_classes():
    for name in (PREDICTED_TYPES_FILE, PREDICTED_TYPES_BASE_FILE):
        path = os.path.join(PROCESSED_DIR, name)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return {t: p.get('class_uri') for t, p in json.load(f).items()}
    return {}


def load_gold_majority_classes(onto):
    _, doc_entities, _ = conll.load_corpus(os.path.join(RAW_DIR, 'fine_grained_ner'))
    majority = conll.majority_types(doc_entities, conll.valid_terms(doc_entities))
    return {t: onto.resolve(ty) for t, ty in majority.items()}


def evaluate_relations(relations_csv=None):
    onto = load_ontology()
    relations_csv = relations_csv or os.path.join(PROCESSED_DIR, 'relations.csv')
    if not os.path.exists(relations_csv):
        print(f"WARNING: Predicted relations not found at {relations_csv}")
        return {}
    relations_df = pd.read_csv(relations_csv)
    gold = {canonical_triple(*t) for t in onto.restriction_triples()}

    def property_matches(p):
        return {PART_OF if q in (MMO_NS + 'isPartOf', DCT_NS + 'isPartOf') else q
                for q in onto.property_ancestors(p)}

    pred_lifted, unmapped = lift_triples(relations_df, load_predicted_classes(), onto)
    gold_lifted, _ = lift_triples(relations_df, load_gold_majority_classes(onto), onto)

    results = {
        'note': 'Class-level comparison with the ontology restrictions; the '
                'dataset has no text-level relation annotations.',
        'n_relations': int(len(relations_df)),
        'n_unmapped_relation_types': unmapped,
        'n_gold_restriction_triples': len(gold),
        'gold_properties': sorted({local_name(p) for _, p, _ in gold}),
        'predicted_typing': metrics.triple_metrics(pred_lifted, gold, property_matches),
        'diagnostic_gold_typing': metrics.triple_metrics(gold_lifted, gold, property_matches),
        'by_relation_type': {k: int(v) for k, v in relations_df['rel'].value_counts().items()},
    }

    print("\n" + "=" * 60)
    print("RELATION EVALUATION (class level, vs. ontology restrictions)")
    print("=" * 60)
    print(f"Extracted relations: {len(relations_df)} "
          f"({unmapped} with relation types that have no ontology property)")
    print(f"Gold restriction triples: {len(gold)}")
    for name in ('predicted_typing', 'diagnostic_gold_typing'):
        m = results[name]
        print(f"  {name:<24} P {m['precision']:.4f}  R {m['recall']:.4f}  "
              f"F1 {m['f1']:.4f}  ({m['n_predicted']} lifted triples, "
              f"{m['n_matched_gold']} gold matched)")

    out_path = os.path.join(PROCESSED_DIR, 'relation_eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to {out_path}")
    return results


if __name__ == "__main__":
    evaluate_relations()
