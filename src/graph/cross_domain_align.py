"""
cross_domain_align.py - Cross-Domain Alignment to EMMO/BFO Upper Ontologies

Maps the MaterioMiner domain ontology (428 classes) to upper-level ontologies:
  - EMMO (Elementary Multiperspective Material Ontology): materials science upper ontology
  - BFO (Basic Formal Ontology): foundational upper ontology

Uses a MINIMAL approach with manually-defined upper-level classes (well-known
EMMO/BFO categories) to avoid dependency on downloading full OWL files.

Alignment via sentence-transformer embedding similarity with MILA-style
bidirectional matching, Jaccard + edit distance lexical scoring, and
configurable similarity threshold.

Outputs per-ontology alignment CSVs and a combined quality report.
"""

import os
import re
import json
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from config import EMBEDDER_MODEL, PROCESSED_DIR
from ontology_utils import load_ontology

# Manually-defined upper-level ontology classes

EMMO_CLASSES = {
    'EMMO_Material': ['Material', 'Substance', 'Chemical substance'],
    'EMMO_Process': ['Process', 'Manufacturing', 'Experiment', 'Test', 'Measurement process'],
    'EMMO_Property': ['Property', 'Physical property', 'Mechanical property', 'Chemical property'],
    'EMMO_Measurement': ['Measurement', 'Quantity', 'Unit', 'Value'],
    'EMMO_Object': ['Object', 'Component', 'Specimen', 'Sample', 'Device'],
    'EMMO_Phenomenon': ['Phenomenon', 'Mechanism', 'Effect'],
    'EMMO_State': ['State', 'Phase', 'Condition'],
    'EMMO_Defect': ['Defect', 'Damage', 'Flaw', 'Crack', 'Void'],
    'EMMO_Microstructure': ['Microstructure', 'Grain', 'Crystal', 'Phase'],
    'EMMO_MechanicalLoading': ['Loading', 'Stress', 'Strain', 'Force', 'Fatigue'],
    'EMMO_ChemicalElement': ['Element', 'Atom', 'Ion'],
    'EMMO_Model': ['Model', 'Simulation', 'Computation'],
    'EMMO_Data': ['Data', 'Dataset', 'Result', 'Observation'],
    'EMMO_Geometry': ['Geometry', 'Shape', 'Size', 'Length', 'Area', 'Volume'],
}

BFO_CLASSES = {
    'BFO_Entity': ['Entity'],
    'BFO_Continuant': ['Continuant', 'Independent continuant'],
    'BFO_MaterialEntity': ['Material entity', 'Object', 'Object aggregate', 'Fiat object part'],
    'BFO_Quality': ['Quality', 'Property', 'Attribute'],
    'BFO_Disposition': ['Disposition', 'Tendency', 'Capability'],
    'BFO_Role': ['Role', 'Function'],
    'BFO_Occurrent': ['Occurrent', 'Process', 'Event'],
    'BFO_TemporalRegion': ['Temporal region', 'Time interval'],
    'BFO_SpatialRegion': ['Spatial region', 'Location', 'Site'],
    'BFO_GenericallyDependentContinuant': ['Information', 'Data', 'Pattern'],
    'BFO_SpecificallyDependentContinuant': ['Dependent continuant', 'State', 'Condition'],
}


def normalize_term(term):
    # Split CamelCase *before* lowercasing (lowercasing first made the split a no-op)
    term = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', term.strip()).lower()
    for prefix in ('the ', 'a ', 'an '):
        if term.startswith(prefix):
            term = term[len(prefix):]
    term = re.sub(r'\([^)]*\)', '', term).strip()
    return term


def jaccard_similarity(s1, s2):
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def edit_distance_similarity(s1, s2):
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def load_local_ontology():
    """Primary labels and all labels of the MaterioMiner (MMO) classes."""
    onto = load_ontology()
    primary_labels = [onto.primary_label(u) for u in onto.uris]
    all_labels = {lab for u in onto.uris for lab in onto.labels(u)}
    print(f"  MaterioMiner ontology: {len(primary_labels)} classes, "
          f"{len(all_labels)} total labels")
    return primary_labels, all_labels


def mila_alignment(local_classes, local_embeds, upper_labels, upper_embeds,
                   upper_concepts, threshold=0.4):
    n_local = len(local_classes)
    n_upper = len(upper_labels)

    score_matrix = np.zeros((n_local, n_upper))

    for i, lc in enumerate(local_classes):
        l_norm = normalize_term(lc)
        for j, ul in enumerate(upper_labels):
            u_norm = normalize_term(ul)

            embed_sim = float(util.cos_sim(
                local_embeds[i:i+1], upper_embeds[j:j+1]
            )[0][0])

            jacc = jaccard_similarity(l_norm, u_norm)
            edit_sim = edit_distance_similarity(l_norm, u_norm)

            exact = 0.0
            if l_norm == u_norm:
                exact = 1.0
            elif l_norm in u_norm or u_norm in l_norm:
                exact = 0.5

            score = (0.50 * embed_sim + 0.15 * jacc +
                     0.15 * edit_sim + 0.20 * exact)
            score_matrix[i, j] = score

    forward = {}
    for i in range(n_local):
        j = int(np.argmax(score_matrix[i]))
        forward[i] = (j, score_matrix[i, j])

    backward = {}
    for j in range(n_upper):
        i = int(np.argmax(score_matrix[:, j]))
        backward[j] = (i, score_matrix[i, j])

    mutual = set()
    for i, (j, _) in forward.items():
        if backward.get(j, (None,))[0] == i:
            mutual.add((i, j))

    used_local = set()
    used_upper = set()
    alignments = []

    for i, j in sorted(mutual, key=lambda x: -score_matrix[x[0], x[1]]):
        if score_matrix[i, j] >= threshold:
            alignments.append({
                'local_class': local_classes[i],
                'upper_concept': upper_concepts[j],
                'matched_label': upper_labels[j],
                'similarity': round(float(score_matrix[i, j]), 4),
                'embed_sim': round(float(util.cos_sim(
                    local_embeds[i:i+1], upper_embeds[j:j+1]
                )[0][0]), 4),
                'direction': 'mutual',
            })
            used_local.add(i)
            used_upper.add(j)

    remaining = [(i, j, s) for i, (j, s) in forward.items()
                 if i not in used_local and s >= threshold]
    for i, j, s in sorted(remaining, key=lambda x: -x[2]):
        if j not in used_upper:
            alignments.append({
                'local_class': local_classes[i],
                'upper_concept': upper_concepts[j],
                'matched_label': upper_labels[j],
                'similarity': round(float(s), 4),
                'embed_sim': round(float(util.cos_sim(
                    local_embeds[i:i+1], upper_embeds[j:j+1]
                )[0][0]), 4),
                'direction': 'forward',
            })
            used_local.add(i)
            used_upper.add(j)

    remaining_back = [(j, i, s) for j, (i, s) in backward.items()
                      if j not in used_upper and i not in used_local
                      and s >= threshold]
    for j, i, s in sorted(remaining_back, key=lambda x: -x[2]):
        alignments.append({
            'local_class': local_classes[i],
            'upper_concept': upper_concepts[j],
            'matched_label': upper_labels[j],
            'similarity': round(float(s), 4),
            'embed_sim': round(float(util.cos_sim(
                local_embeds[i:i+1], upper_embeds[j:j+1]
            )[0][0]), 4),
            'direction': 'backward',
        })
        used_local.add(i)
        used_upper.add(j)

    return alignments, score_matrix


def align_to_upper_ontology(local_classes, embedder, upper_classes_dict,
                            output_path, upper_name="upper", threshold=0.4):
    upper_labels = []
    upper_concepts = []
    for concept, labels in upper_classes_dict.items():
        for label in labels:
            upper_labels.append(label.lower())
            upper_concepts.append(concept)

    print(f"\nAligning {len(local_classes)} local -> "
          f"{len(upper_labels)} {upper_name} labels "
          f"({len(upper_classes_dict)} concepts)")

    local_normalized = [normalize_term(c) for c in local_classes]
    local_embeds = embedder.encode(local_normalized, convert_to_numpy=True)
    upper_embeds = embedder.encode(upper_labels, convert_to_numpy=True)

    alignments, score_matrix = mila_alignment(
        local_classes, local_embeds, upper_labels, upper_embeds,
        upper_concepts, threshold=threshold
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(alignments)
    if len(df) > 0:
        df = df.sort_values('similarity', ascending=False)
    df.to_csv(output_path, index=False)

    n_aligned = len(alignments)
    n_above_07 = sum(1 for a in alignments if a['similarity'] > 0.7)
    n_above_05 = sum(1 for a in alignments if a['similarity'] > 0.5)
    n_mutual = sum(1 for a in alignments if a['direction'] == 'mutual')
    n_forward = sum(1 for a in alignments if a['direction'] == 'forward')
    n_backward = sum(1 for a in alignments if a['direction'] == 'backward')
    avg_sim = float(df['similarity'].mean()) if len(df) > 0 else 0.0

    aligned_local = set(a['local_class'] for a in alignments)
    aligned_concepts = set(a['upper_concept'] for a in alignments)
    local_coverage = len(aligned_local) / len(local_classes) if local_classes else 0
    concept_coverage = len(aligned_concepts) / len(upper_classes_dict) if upper_classes_dict else 0

    stats = {
        'upper_ontology': upper_name,
        'n_local_classes': len(local_classes),
        'n_upper_labels': len(upper_labels),
        'n_upper_concepts': len(upper_classes_dict),
        'n_alignments': n_aligned,
        'n_above_0.7': n_above_07,
        'n_above_0.5': n_above_05,
        'n_mutual': n_mutual,
        'n_forward': n_forward,
        'n_backward': n_backward,
        'avg_similarity': round(avg_sim, 4),
        'local_coverage': round(local_coverage, 4),
        'concept_coverage': round(concept_coverage, 4),
        'threshold': threshold,
    }

    if len(df) > 0:
        stats['sim_quartiles'] = {
            'q25': round(float(df['similarity'].quantile(0.25)), 4),
            'q50': round(float(df['similarity'].quantile(0.50)), 4),
            'q75': round(float(df['similarity'].quantile(0.75)), 4),
        }

    concept_counts = {}
    for a in alignments:
        c = a['upper_concept']
        concept_counts[c] = concept_counts.get(c, 0) + 1
    stats['concept_distribution'] = concept_counts

    print(f"  Alignments: {n_aligned} "
          f"(mutual={n_mutual}, fwd={n_forward}, bwd={n_backward})")
    print(f"  Above 0.7 sim: {n_above_07}, above 0.5 sim: {n_above_05}")
    print(f"  Avg similarity: {avg_sim:.4f}")
    print(f"  Local coverage: {local_coverage:.1%} "
          f"({len(aligned_local)}/{len(local_classes)})")
    print(f"  Concept coverage: {concept_coverage:.1%} "
          f"({len(aligned_concepts)}/{len(upper_classes_dict)})")

    return alignments, stats


def generate_quality_report(emmo_stats, bfo_stats, output_path):
    lines = []
    lines.append("=" * 70)
    lines.append("CROSS-DOMAIN ONTOLOGY ALIGNMENT REPORT")
    lines.append("MaterioMiner Ontology -> EMMO / BFO")
    lines.append("=" * 70)

    for name, stats in [("EMMO", emmo_stats), ("BFO", bfo_stats)]:
        lines.append(f"\n{'─' * 50}")
        lines.append(f"Alignment to {name}")
        lines.append(f"{'─' * 50}")
        lines.append(f"  Local classes:     {stats['n_local_classes']}")
        lines.append(f"  {name} labels:     {stats['n_upper_labels']}")
        lines.append(f"  {name} concepts:   {stats['n_upper_concepts']}")
        lines.append(f"  Total alignments:  {stats['n_alignments']}")
        lines.append(f"    Mutual matches:  {stats['n_mutual']}")
        lines.append(f"    Forward-only:    {stats['n_forward']}")
        lines.append(f"    Backward-only:   {stats['n_backward']}")
        lines.append(f"  Above 0.7 sim:     {stats['n_above_0.7']}")
        lines.append(f"  Above 0.5 sim:     {stats['n_above_0.5']}")
        lines.append(f"  Avg similarity:    {stats['avg_similarity']:.4f}")
        lines.append(f"  Local coverage:    {stats['local_coverage']:.1%}")
        lines.append(f"  Concept coverage:  {stats['concept_coverage']:.1%}")
        lines.append(f"  Threshold:         {stats['threshold']}")

        if 'sim_quartiles' in stats:
            q = stats['sim_quartiles']
            lines.append(f"  Similarity Q25/Q50/Q75: "
                         f"{q['q25']:.4f} / {q['q50']:.4f} / {q['q75']:.4f}")

        if 'concept_distribution' in stats:
            lines.append(f"\n  {name} concept distribution:")
            for concept, count in sorted(stats['concept_distribution'].items(),
                                         key=lambda x: -x[1]):
                lines.append(f"    {concept}: {count} classes")

    lines.append(f"\n{'=' * 70}")
    lines.append("ANALYSIS SUMMARY")
    lines.append(f"{'=' * 70}")

    emmo_cov = emmo_stats['local_coverage']
    bfo_cov = bfo_stats['local_coverage']
    lines.append(f"\nLocal class alignment rate:")
    lines.append(f"  EMMO: {emmo_cov:.1%} of domain classes aligned")
    lines.append(f"  BFO:  {bfo_cov:.1%} of domain classes aligned")
    lines.append(f"\nUpper concept coverage:")
    lines.append(f"  EMMO: {emmo_stats['concept_coverage']:.1%} of concepts matched")
    lines.append(f"  BFO:  {bfo_stats['concept_coverage']:.1%} of concepts matched")

    lines.append(f"\nAlignment quality (by avg similarity):")
    lines.append(f"  EMMO: {emmo_stats['avg_similarity']:.4f}")
    lines.append(f"  BFO:  {bfo_stats['avg_similarity']:.4f}")

    better = "EMMO" if emmo_stats['avg_similarity'] > bfo_stats['avg_similarity'] else "BFO"
    lines.append(f"\n  -> {better} shows stronger average alignment similarity,")
    if better == "EMMO":
        lines.append("     as expected since EMMO is a materials-science-specific upper ontology")
    else:
        lines.append("     suggesting BFO's general categories map well to domain concepts")

    report = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")
    return report


def run_cross_domain_alignment(embedder_model=None):
    model_name = embedder_model or EMBEDDER_MODEL
    print(f"Cross-domain alignment using embedder: {model_name}")
    embedder = SentenceTransformer(model_name)

    print("\nLoading MaterioMiner domain ontology...")
    local_classes, _ = load_local_ontology()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("\n" + "=" * 50)
    print("EMMO Alignment (manual upper-level classes)")
    print("=" * 50)
    emmo_alignments, emmo_stats = align_to_upper_ontology(
        local_classes, embedder, EMMO_CLASSES,
        os.path.join(PROCESSED_DIR, 'cross_domain_emmo.csv'),
        upper_name="EMMO", threshold=0.4
    )

    print("\n" + "=" * 50)
    print("BFO Alignment (manual upper-level classes)")
    print("=" * 50)
    bfo_alignments, bfo_stats = align_to_upper_ontology(
        local_classes, embedder, BFO_CLASSES,
        os.path.join(PROCESSED_DIR, 'cross_domain_bfo.csv'),
        upper_name="BFO", threshold=0.4
    )

    report = generate_quality_report(
        emmo_stats, bfo_stats,
        os.path.join(PROCESSED_DIR, 'cross_domain_report.txt')
    )
    print("\n" + report)

    combined = {
        'emmo': emmo_stats,
        'bfo': bfo_stats,
        'embedder_model': model_name,
    }
    with open(os.path.join(PROCESSED_DIR, 'cross_domain_stats.json'), 'w') as f:
        json.dump(combined, f, indent=2)

    return combined


if __name__ == '__main__':
    run_cross_domain_alignment()
