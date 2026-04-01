"""
eval_f1.py - Multi-level Ontology Alignment Evaluation

Evaluates alignment quality at three levels:
  1. Type-level: NER entity types mapped to ontology classes
  2. Term-level: Individual entity mentions aligned to ontology classes
  3. Concept-level: Cluster representatives aligned to ontology classes

Uses ALL classes from ontology.ttl as gold standard.
OAEI-standard P/R/F1 at thresholds 0.50-0.85.
Saves detailed results to eval_results.json.
"""

import os
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from config import DEFAULT_EMBEDDER_MODEL

# Get project root directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dataset")

ALIGNMENT_PATH = os.path.join(PROCESSED_DIR, 'ontology_alignment.csv')
ENTITY_TYPES_PATH = os.path.join(PROCESSED_DIR, 'entity_types.json')
CLUSTERS_PATH = os.path.join(PROCESSED_DIR, 'fine_auto_clusters.json')

EMBEDDER_MODEL = os.environ.get('EMBEDDER_MODEL', DEFAULT_EMBEDDER_MODEL)
EMBEDDER = SentenceTransformer(EMBEDDER_MODEL)


def load_gold_standard():
    """Parse ALL classes from ontology.ttl as gold standard.

    Returns:
        gold_labels: list of primary class labels
        all_labels: set of all labels (including alt/pref)
        label_to_uri: mapping label -> ontology URI
    """
    ttl_path = os.path.join(RAW_DIR, 'ontologies', 'ontology.ttl')
    if not os.path.exists(ttl_path):
        raise FileNotFoundError(f"Ontology not found: {ttl_path}")

    from rdflib import Graph, RDF, RDFS, OWL, Namespace

    g = Graph()
    g.parse(ttl_path, format='turtle')

    MMO = Namespace("https://w3id.org/pmd/materials-mechanics-ontology/")
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

    label_props = [RDFS.label, MMO.altLabel, MMO.prefLabel,
                   SKOS.prefLabel, SKOS.altLabel]

    gold_labels = []
    all_labels = set()
    label_to_uri = {}

    for cls in g.subjects(RDF.type, OWL.Class):
        cls_str = str(cls)
        if cls_str.startswith('http://www.w3.org/'):
            continue

        labels = []
        for prop in label_props:
            for label in g.objects(cls, prop):
                label_str = str(label).strip()
                if label_str and not label_str.startswith('http'):
                    labels.append(label_str)

        if not labels:
            fragment = cls_str.split('#')[-1].split('/')[-1]
            if fragment and fragment[0].isupper():
                labels.append(fragment)

        if labels:
            primary = labels[0]
            gold_labels.append(primary)
            for lab in labels:
                all_labels.add(lab.lower())
                label_to_uri[lab.lower()] = cls_str

    print(f"Gold standard: {len(gold_labels)} classes, "
          f"{len(all_labels)} labels")
    return gold_labels, all_labels, label_to_uri


def load_ner_types():
    """Load NER entity types from entity_types.json."""
    if not os.path.exists(ENTITY_TYPES_PATH):
        return {}
    with open(ENTITY_TYPES_PATH) as f:
        return json.load(f)


def load_clusters():
    """Load clusters from JSON."""
    if not os.path.exists(CLUSTERS_PATH):
        return {}
    with open(CLUSTERS_PATH) as f:
        return json.load(f)


def compute_f1(precision, recall):
    """Standard F1 from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def type_level_evaluation(gold_labels, all_gold_lower, ner_types):
    """Evaluate at NER type level: do NER types map to ontology classes?

    NER types (e.g., 'FatigueTest', 'Crack') should correspond to
    ontology classes. This measures how well the NER schema covers
    the ontology.
    """
    if not ner_types:
        return {'level': 'type', 'note': 'No entity_types.json available'}

    unique_types = set(ner_types.values())
    print(f"\nType-level evaluation: {len(unique_types)} NER types "
          f"vs {len(gold_labels)} ontology classes")

    # Embed NER types and gold labels
    type_list = sorted(unique_types)
    gold_list = sorted(gold_labels)

    type_embeds = EMBEDDER.encode(type_list)
    gold_embeds = EMBEDDER.encode(gold_list)

    results = {}
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    for thresh in thresholds:
        # For each NER type, find best-matching gold class
        matched_types = set()
        matched_gold = set()

        sim_matrix = util.cos_sim(type_embeds, gold_embeds).numpy()

        for i, ner_type in enumerate(type_list):
            best_j = np.argmax(sim_matrix[i])
            best_sim = sim_matrix[i, best_j]

            # Also check exact/substring match
            ner_lower = ner_type.lower()
            exact_match = any(
                ner_lower == g.lower() or ner_lower in g.lower() or g.lower() in ner_lower
                for g in gold_list
            )

            if best_sim >= thresh or exact_match:
                matched_types.add(ner_type)
                matched_gold.add(gold_list[best_j])

        prec = len(matched_types) / len(type_list) if type_list else 0
        rec = len(matched_gold) / len(gold_list) if gold_list else 0
        f1 = compute_f1(prec, rec)
        results[thresh] = {'precision': round(prec, 4),
                           'recall': round(rec, 4),
                           'f1': round(f1, 4),
                           'matched_types': len(matched_types),
                           'matched_gold': len(matched_gold)}

    return {'level': 'type', 'n_ner_types': len(unique_types),
            'n_gold_classes': len(gold_labels), 'thresholds': results}


def term_level_evaluation(gold_labels, all_gold_lower, alignment_df):
    """Evaluate at term level: individual discovered terms aligned to ontology.

    Uses the alignment CSV output.
    """
    if alignment_df is None or len(alignment_df) == 0:
        return {'level': 'term', 'note': 'No alignment data available'}

    print(f"\nTerm-level evaluation: {len(alignment_df)} alignments "
          f"vs {len(gold_labels)} gold classes")

    results = {}
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    for thresh in thresholds:
        subset = alignment_df[alignment_df['similarity'] >= thresh]

        if len(subset) == 0:
            results[thresh] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                               'n_alignments': 0}
            continue

        # Precision: fraction of alignments that match a real gold class
        refs_matched = subset['reference'].str.lower().isin(all_gold_lower)
        prec = refs_matched.mean() if len(subset) > 0 else 0

        # Recall: fraction of gold classes covered by at least one alignment
        covered_gold = set()
        for ref in subset['reference'].str.lower():
            if ref in all_gold_lower:
                covered_gold.add(ref)
        rec = len(covered_gold) / len(gold_labels) if gold_labels else 0

        f1 = compute_f1(prec, rec)
        results[thresh] = {'precision': round(prec, 4),
                           'recall': round(rec, 4),
                           'f1': round(f1, 4),
                           'n_alignments': len(subset),
                           'covered_gold': len(covered_gold)}

    return {'level': 'term', 'total_alignments': len(alignment_df),
            'thresholds': results}


def concept_level_evaluation(gold_labels, all_gold_lower, clusters,
                             alignment_df):
    """Evaluate at concept level: cluster representatives -> ontology classes.

    Each cluster's best-aligned term represents the cluster's concept.
    """
    if not clusters or alignment_df is None or len(alignment_df) == 0:
        return {'level': 'concept',
                'note': 'No clusters or alignment data available'}

    print(f"\nConcept-level evaluation: {len(clusters)} clusters "
          f"vs {len(gold_labels)} gold classes")

    # For each cluster, find the best alignment from its members
    cluster_alignments = []
    for cid, terms in clusters.items():
        best_sim = 0
        best_ref = None
        best_disc = None
        for t in terms:
            matches = alignment_df[alignment_df['discovered'] == t]
            if len(matches) > 0:
                top = matches.iloc[0]
                if top['similarity'] > best_sim:
                    best_sim = top['similarity']
                    best_ref = top['reference']
                    best_disc = t
        if best_ref:
            cluster_alignments.append({
                'cluster_id': cid,
                'representative': best_disc,
                'reference': best_ref,
                'similarity': best_sim,
            })

    if not cluster_alignments:
        return {'level': 'concept', 'note': 'No cluster alignments found'}

    ca_df = pd.DataFrame(cluster_alignments)

    results = {}
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    for thresh in thresholds:
        subset = ca_df[ca_df['similarity'] >= thresh]

        refs_matched = subset['reference'].str.lower().isin(all_gold_lower)
        prec = refs_matched.mean() if len(subset) > 0 else 0

        covered = set()
        for ref in subset['reference'].str.lower():
            if ref in all_gold_lower:
                covered.add(ref)
        rec = len(covered) / len(gold_labels) if gold_labels else 0

        f1 = compute_f1(prec, rec)
        results[thresh] = {'precision': round(prec, 4),
                           'recall': round(rec, 4),
                           'f1': round(f1, 4),
                           'n_concepts': len(subset),
                           'covered_gold': len(covered)}

    return {'level': 'concept', 'total_clusters': len(clusters),
            'aligned_clusters': len(cluster_alignments),
            'thresholds': results}


def run_evaluation():
    """Run multi-level evaluation and save results."""
    # Load gold standard
    gold_labels, all_gold_lower, label_to_uri = load_gold_standard()

    # Load alignment results
    alignment_df = None
    if os.path.exists(ALIGNMENT_PATH):
        alignment_df = pd.read_csv(ALIGNMENT_PATH)
        print(f"Loaded alignment: {len(alignment_df)} entries")

    # Load NER types and clusters
    ner_types = load_ner_types()
    clusters = load_clusters()

    # Run all evaluation levels
    type_results = type_level_evaluation(gold_labels, all_gold_lower, ner_types)
    term_results = term_level_evaluation(gold_labels, all_gold_lower,
                                         alignment_df)
    concept_results = concept_level_evaluation(gold_labels, all_gold_lower,
                                               clusters, alignment_df)

    # Compile results
    all_results = {
        'gold_standard': {
            'n_classes': len(gold_labels),
            'n_labels': len(all_gold_lower),
        },
        'type_level': type_results,
        'term_level': term_results,
        'concept_level': concept_results,
    }

    # Save detailed results
    results_path = os.path.join(PROCESSED_DIR, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed results to {results_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for level_name, level_data in [('Type', type_results),
                                    ('Term', term_results),
                                    ('Concept', concept_results)]:
        if 'thresholds' not in level_data:
            print(f"\n{level_name}: {level_data.get('note', 'N/A')}")
            continue

        print(f"\n{level_name}-level:")
        print(f"  {'Thresh':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        for thresh, metrics in sorted(level_data['thresholds'].items()):
            print(f"  {thresh:>8.2f} {metrics['precision']:>8.4f} "
                  f"{metrics['recall']:>8.4f} {metrics['f1']:>8.4f}")

    # Also save CSV summary for quick viewing
    rows = []
    for level_name, level_data in [('type', type_results),
                                    ('term', term_results),
                                    ('concept', concept_results)]:
        if 'thresholds' not in level_data:
            continue
        for thresh, metrics in level_data['thresholds'].items():
            rows.append({
                'level': level_name,
                'threshold': thresh,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
            })
    if rows:
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(os.path.join(PROCESSED_DIR, 'f1_ablation.csv'),
                          index=False)
        print(f"\nSaved summary CSV to f1_ablation.csv")


if __name__ == "__main__":
    run_evaluation()
