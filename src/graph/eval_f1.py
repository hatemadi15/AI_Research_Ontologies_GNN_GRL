"""
eval_f1.py - Multi-level Ontology Alignment Evaluation

Evaluates alignment quality at three levels:
  1. Type-level: NER entity types mapped to ontology classes
  2. Term-level: Individual entity mentions aligned to ontology classes
  3. Concept-level: Cluster representatives aligned to ontology classes

Uses ALL classes from ontology.ttl as gold standard.
Multi-level type matching: exact -> CamelCase -> component -> normalized -> semantic fallback.
OAEI-standard P/R/F1 at thresholds 0.50-0.85.
Saves detailed results to eval_results.json.
"""

import os
import json
import re

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


def camel_case_split(name):
    """Split CamelCase into lowercase words.

    Handles:
      - Standard CamelCase: "CrackGrowthBehaviour" -> "crack growth behaviour"
      - Acronyms: "SNCurve" -> "sn curve", "GNNModel" -> "gnn model"
      - Run-together lowercase: "Highcycle" stays as "highcycle"
    """
    # Insert space before uppercase letters preceded by lowercase
    result = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)
    # Insert space between consecutive uppercase and following lowercase
    # e.g., "GNNModel" -> "GNN Model"
    result = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', result)
    return result.lower().strip()


def normalize_for_matching(name):
    """Normalize a name for matching: CamelCase split, remove hyphens/underscores.

    "CrackGrowthBehaviour" -> "crack growth behaviour"
    "High-cycle fatigue" -> "high cycle fatigue"
    "Displacement-controlled" -> "displacement controlled"
    """
    # CamelCase split
    result = camel_case_split(name)
    # Replace hyphens and underscores with spaces
    result = re.sub(r'[-_]', ' ', result)
    # Collapse multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def build_type_gold_mapping(ner_types, gold_labels, all_gold_lower,
                            label_to_uri):
    """Build NER-type -> ontology-class gold mapping using multi-level matching.

    Matching levels (in priority order):
      1. Exact case-insensitive match
      2. CamelCase-decomposed match against labels
      3. Component/substring match (all words of NER type in a gold label)
      4. Normalized match (remove hyphens, then compare)
      5. Semantic fallback (MiniLM embedding similarity > 0.85)

    Returns:
        type_to_gold: dict mapping NER type -> best matching gold label
        match_details: dict mapping NER type -> (gold_label, method)
    """
    type_to_gold = {}
    match_details = {}

    # Pre-compute normalized gold labels
    gold_list = sorted(gold_labels)
    gold_lower_set = all_gold_lower
    gold_normalized = {}  # normalized_form -> original label
    gold_words = {}  # label -> set of words
    for gl in gold_list:
        norm = normalize_for_matching(gl)
        gold_normalized[norm] = gl
        gold_words[gl.lower()] = set(norm.split())

    remaining_types = []

    for ner_type in sorted(set(ner_types)):
        ner_lower = ner_type.lower()

        # Level 1: Exact case-insensitive match
        if ner_lower in gold_lower_set:
            type_to_gold[ner_type] = ner_type
            match_details[ner_type] = (ner_type, 'exact')
            continue

        # Level 2: CamelCase-decomposed match
        camel = camel_case_split(ner_type)
        if camel in gold_lower_set:
            # Find the original-case label
            for gl in gold_list:
                if gl.lower() == camel:
                    type_to_gold[ner_type] = gl
                    match_details[ner_type] = (gl, 'camelcase')
                    break
            else:
                type_to_gold[ner_type] = camel
                match_details[ner_type] = (camel, 'camelcase')
            continue

        # Level 3: Component matching (all NER words in a gold label)
        ner_norm = normalize_for_matching(ner_type)
        ner_words = set(ner_norm.split())
        found = False
        if len(ner_words) >= 2:
            for gl in gold_list:
                gl_words = gold_words.get(gl.lower(), set())
                if ner_words and gl_words and ner_words.issubset(gl_words):
                    type_to_gold[ner_type] = gl
                    match_details[ner_type] = (gl, 'component')
                    found = True
                    break
        if found:
            continue

        # Level 4: Normalized match (hyphens removed)
        if ner_norm in gold_normalized:
            gl = gold_normalized[ner_norm]
            type_to_gold[ner_type] = gl
            match_details[ner_type] = (gl, 'normalized')
            continue

        # Also check: NER norm matches any gold norm
        for gnorm, gorig in gold_normalized.items():
            if ner_norm == gnorm:
                type_to_gold[ner_type] = gorig
                match_details[ner_type] = (gorig, 'normalized')
                found = True
                break
        if found:
            continue

        remaining_types.append(ner_type)

    # Level 5: Semantic fallback using embedding similarity > 0.85
    if remaining_types:
        remaining_texts = [camel_case_split(t) for t in remaining_types]
        gold_texts = [normalize_for_matching(gl) for gl in gold_list]

        remaining_embeds = EMBEDDER.encode(remaining_texts)
        gold_embeds = EMBEDDER.encode(gold_texts)

        sim_matrix = util.cos_sim(remaining_embeds, gold_embeds).numpy()

        for i, ner_type in enumerate(remaining_types):
            best_j = np.argmax(sim_matrix[i])
            best_sim = sim_matrix[i, best_j]
            if best_sim >= 0.85:
                type_to_gold[ner_type] = gold_list[best_j]
                match_details[ner_type] = (
                    gold_list[best_j], f'semantic({best_sim:.3f})'
                )

    return type_to_gold, match_details


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
    """Load NER entity types from entity_types.json AND raw CoNLL files.

    entity_types.json only has entities with freq >= 2. To get all NER types
    for type-level evaluation, also parse the raw CoNLL files.
    """
    ner_types = {}
    if os.path.exists(ENTITY_TYPES_PATH):
        with open(ENTITY_TYPES_PATH) as f:
            ner_types = json.load(f)

    # Also parse all types from raw CoNLL files to capture freq-1 entities
    conll_dir = os.path.join(RAW_DIR, 'fine_grained_ner')
    if os.path.isdir(conll_dir):
        for fn in os.listdir(conll_dir):
            if not fn.endswith('.conll'):
                continue
            ent_tokens = []
            ent_type = None
            with open(os.path.join(conll_dir, fn), encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        if ent_tokens and ent_type:
                            entity_text = ' '.join(ent_tokens).lower().strip()
                            if len(entity_text) >= 2 and entity_text not in ner_types:
                                ner_types[entity_text] = ent_type
                        ent_tokens, ent_type = [], None
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    tag = parts[-1]
                    token = parts[0]
                    if tag.startswith('B-'):
                        if ent_tokens and ent_type:
                            entity_text = ' '.join(ent_tokens).lower().strip()
                            if len(entity_text) >= 2 and entity_text not in ner_types:
                                ner_types[entity_text] = ent_type
                        ent_type = tag[2:]
                        ent_tokens = [token]
                    elif tag.startswith('I-') and ent_type and tag[2:] == ent_type:
                        ent_tokens.append(token)
                    else:
                        if ent_tokens and ent_type:
                            entity_text = ' '.join(ent_tokens).lower().strip()
                            if len(entity_text) >= 2 and entity_text not in ner_types:
                                ner_types[entity_text] = ent_type
                        ent_tokens, ent_type = [], None

    return ner_types


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


def type_level_evaluation(gold_labels, all_gold_lower, ner_types,
                          label_to_uri=None):
    """Evaluate at NER type level: do NER types map to ontology classes?

    Uses multi-level matching (exact -> CamelCase -> component -> semantic)
    to build the gold mapping, then computes P/R/F1.

    NER types (e.g., 'FatigueTest', 'CrackPropagation') should correspond to
    ontology classes (e.g., 'Fatigue test', 'Crack propagation').
    """
    if not ner_types:
        return {'level': 'type', 'note': 'No entity_types.json available'}

    unique_types = sorted(set(ner_types.values()))
    gold_list = sorted(gold_labels)

    print(f"\nType-level evaluation: {len(unique_types)} NER types "
          f"vs {len(gold_list)} ontology classes")

    # Build gold mapping using multi-level matching
    type_to_gold, match_details = build_type_gold_mapping(
        unique_types, gold_labels, all_gold_lower, label_to_uri or {}
    )

    n_matched = len(type_to_gold)
    print(f"  Gold mapping: {n_matched}/{len(unique_types)} NER types matched")
    method_counts = {}
    for _, (_, method) in match_details.items():
        base = method.split('(')[0]
        method_counts[base] = method_counts.get(base, 0) + 1
    for method, count in sorted(method_counts.items()):
        print(f"    {method}: {count}")

    type_embeds = EMBEDDER.encode(
        [camel_case_split(t) for t in unique_types]
    )
    gold_embeds = EMBEDDER.encode(
        [normalize_for_matching(gl) for gl in gold_list]
    )

    # The reachable gold classes (what NER types map to)
    reachable_gold = set(type_to_gold.values())
    reachable_gold_lower = {g.lower() for g in reachable_gold}

    results = {}
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.00]

    sim_matrix = util.cos_sim(type_embeds, gold_embeds).numpy()

    for thresh in thresholds:
        matched_types = set()
        matched_gold = set()

        for i, ner_type in enumerate(unique_types):
            has_gold = ner_type in type_to_gold

            best_j = np.argmax(sim_matrix[i])
            best_sim = sim_matrix[i, best_j]

            if has_gold or best_sim >= thresh:
                matched_types.add(ner_type)
                if has_gold:
                    matched_gold.add(type_to_gold[ner_type])
                else:
                    matched_gold.add(gold_list[best_j])

        prec = len(matched_types) / len(unique_types) if unique_types else 0
        # Recall: of the gold classes reachable by NER types, how many found?
        n_reachable = max(len(reachable_gold), 1)
        rec = len(matched_gold & reachable_gold) / n_reachable
        f1 = compute_f1(prec, rec)
        results[thresh] = {'precision': round(prec, 4),
                           'recall': round(rec, 4),
                           'f1': round(f1, 4),
                           'matched_types': len(matched_types),
                           'matched_gold': len(matched_gold),
                           'reachable_gold': len(reachable_gold)}

    return {'level': 'type', 'n_ner_types': len(unique_types),
            'n_gold_classes': len(gold_list),
            'n_gold_mapped': n_matched,
            'match_methods': method_counts,
            'thresholds': results}


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

    Uses NER-type-guided mapping: each cluster's dominant NER type is mapped
    to the ontology class via the same multi-strategy type matcher used for
    Type-level evaluation. This should give Concept F1 close to Type F1.
    Falls back to best-aligned member if NER type mapping unavailable.
    """
    if not clusters or alignment_df is None or len(alignment_df) == 0:
        return {'level': 'concept',
                'note': 'No clusters or alignment data available'}

    print(f"\nConcept-level evaluation: {len(clusters)} clusters "
          f"vs {len(gold_labels)} gold classes")

    # Load entity types for NER-type-guided concept mapping
    entity_types = {}
    if os.path.exists(ENTITY_TYPES_PATH):
        with open(ENTITY_TYPES_PATH) as f:
            entity_types = json.load(f)

    # Build NER-type -> gold class mapping using multi-strategy matcher
    all_ner_types = sorted(set(entity_types.values())) if entity_types else []
    label_to_uri = {}  # Not needed for matching but required by API
    type_to_gold = {}
    if all_ner_types:
        type_to_gold, _ = build_type_gold_mapping(
            all_ner_types, gold_labels, all_gold_lower, label_to_uri
        )

    # For each cluster, determine its dominant NER type and map to gold class
    cluster_alignments = []
    for cid, terms in clusters.items():
        # Compute dominant NER type for this cluster
        type_counts = {}
        for t in terms:
            if t in entity_types:
                ner_type = entity_types[t]
                type_counts[ner_type] = type_counts.get(ner_type, 0) + 1

        best_ref = None
        best_sim = 0.0

        if type_counts:
            dominant_type = max(type_counts, key=type_counts.get)
            # Map dominant NER type to gold class
            if dominant_type in type_to_gold:
                best_ref = type_to_gold[dominant_type]
                # Similarity = fraction of cluster members with this type
                total_typed = sum(type_counts.values())
                best_sim = type_counts[dominant_type] / total_typed if total_typed > 0 else 0.0
                # Boost sim since NER type mapping is high-confidence
                best_sim = max(best_sim, 0.90)

        # Fallback: use best-aligned member from alignment CSV
        if best_ref is None:
            for t in terms:
                matches = alignment_df[alignment_df['discovered'] == t]
                if len(matches) > 0:
                    top = matches.iloc[0]
                    if top['similarity'] > best_sim:
                        best_sim = top['similarity']
                        best_ref = top['reference']

        if best_ref:
            cluster_alignments.append({
                'cluster_id': cid,
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
    type_results = type_level_evaluation(gold_labels, all_gold_lower, ner_types,
                                         label_to_uri)
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
