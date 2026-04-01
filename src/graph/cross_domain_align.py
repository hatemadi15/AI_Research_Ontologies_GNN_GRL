"""
cross_domain_align.py - Cross-Domain Alignment to EMMO/BFO Upper Ontologies

Maps the MaterioMiner domain ontology (428 classes) to upper-level ontologies:
  - EMMO (Elementary Multiperspective Material Ontology): materials science upper ontology
  - BFO (Basic Formal Ontology): foundational upper ontology

Uses MILA-style bidirectional alignment with:
  - Sentence-BERT embeddings for semantic similarity
  - Jaccard + edit distance for lexical similarity
  - Structural heuristics from the domain ontology hierarchy
  - Configurable similarity threshold

Outputs per-ontology alignment CSVs and a combined quality report.
"""

import os
import re
import json
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dataset")
ONTOLOGY_DIR = os.path.join(RAW_DIR, "ontologies")

# Will be overridden by config if available
EMBEDDER_MODEL = os.environ.get('EMBEDDER_MODEL', 'all-MiniLM-L6-v2')


def normalize_term(term):
    """Normalize a term for comparison."""
    term = term.lower().strip()
    for prefix in ('the ', 'a ', 'an '):
        if term.startswith(prefix):
            term = term[len(prefix):]
    term = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', term).lower()
    term = re.sub(r'\([^)]*\)', '', term).strip()
    return term


def jaccard_similarity(s1, s2):
    """Token-level Jaccard similarity."""
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def edit_distance_similarity(s1, s2):
    """Normalized edit distance similarity."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def load_ontology_classes_rdf(filepath):
    """Load classes from an RDF/OWL/TTL ontology file using rdflib.

    Returns:
        classes: list of (uri, primary_label, all_labels) tuples
        label_list: flat list of primary labels
    """
    from rdflib import Graph, RDF, RDFS, OWL, Namespace

    g = Graph()
    fmt = 'turtle' if filepath.endswith('.ttl') else 'xml'
    g.parse(filepath, format=fmt)

    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    # Also check EMMO-specific and generic namespaces
    label_props = [
        RDFS.label,
        SKOS.prefLabel,
        SKOS.altLabel,
    ]

    classes = []
    for cls in g.subjects(RDF.type, OWL.Class):
        cls_str = str(cls)
        if cls_str.startswith('http://www.w3.org/'):
            continue  # Skip OWL built-in classes

        labels = []
        for prop in label_props:
            for label in g.objects(cls, prop):
                label_str = str(label).strip()
                if label_str and not label_str.startswith('http'):
                    if label_str not in labels:
                        labels.append(label_str)

        # Fallback: extract from URI fragment
        if not labels:
            fragment = cls_str.split('#')[-1].split('/')[-1]
            if fragment:
                # Clean up OBO-style IDs like BFO_0000001
                if re.match(r'^[A-Z]+_\d+$', fragment):
                    continue  # Skip numeric IDs without labels
                labels.append(fragment)

        if labels:
            classes.append((cls_str, labels[0], labels))

    label_list = [c[1] for c in classes]
    print(f"  Loaded {len(classes)} classes from {os.path.basename(filepath)}")
    return classes, label_list


def load_ontology_classes_dict(class_dict, name="dict"):
    """Load classes from a minimal dictionary representation.

    Args:
        class_dict: dict of {category: [class_labels]}
        name: name for display

    Returns:
        classes: list of (key, primary_label, all_labels) tuples
        label_list: flat list of primary labels
    """
    classes = []
    for category, labels in class_dict.items():
        primary = labels[0] if labels else category
        classes.append((category, primary, labels))

    label_list = [c[1] for c in classes]
    print(f"  Loaded {len(classes)} classes from {name}")
    return classes, label_list


def load_local_ontology():
    """Load the MaterioMiner domain ontology classes.

    Returns:
        classes: list of primary labels
        all_labels: set of all labels (including alternates)
    """
    ttl_path = os.path.join(ONTOLOGY_DIR, 'ontology.ttl')
    if not os.path.exists(ttl_path):
        raise FileNotFoundError(f"Domain ontology not found: {ttl_path}")

    from rdflib import Graph, RDF, RDFS, OWL, Namespace

    g = Graph()
    g.parse(ttl_path, format='turtle')

    MMO = Namespace("https://w3id.org/pmd/materials-mechanics-ontology/")
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

    label_props = [RDFS.label, MMO.altLabel, MMO.prefLabel,
                   SKOS.prefLabel, SKOS.altLabel]

    primary_labels = []
    all_labels = set()

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
            primary_labels.append(labels[0])
            for lab in labels:
                all_labels.add(lab)

    print(f"  MaterioMiner ontology: {len(primary_labels)} classes, "
          f"{len(all_labels)} total labels")
    return primary_labels, all_labels


def mila_alignment(local_classes, local_embeds, upper_classes, upper_embeds,
                   threshold=0.4):
    """MILA-style bidirectional alignment between two sets of classes.

    Args:
        local_classes: list of local class labels
        local_embeds: numpy array of local embeddings
        upper_classes: list of upper ontology class labels
        upper_embeds: numpy array of upper embeddings
        threshold: minimum combined score for a valid alignment

    Returns:
        alignments: list of dicts with alignment details
    """
    n_local = len(local_classes)
    n_upper = len(upper_classes)

    # Compute full score matrix (combined scoring)
    score_matrix = np.zeros((n_local, n_upper))

    for i, lc in enumerate(local_classes):
        l_norm = normalize_term(lc)
        for j, uc in enumerate(upper_classes):
            u_norm = normalize_term(uc)

            # Embedding similarity (50%)
            embed_sim = float(util.cos_sim(
                local_embeds[i:i+1], upper_embeds[j:j+1]
            )[0][0])

            # Jaccard similarity (15%)
            jacc = jaccard_similarity(l_norm, u_norm)

            # Edit distance similarity (15%)
            edit_sim = edit_distance_similarity(l_norm, u_norm)

            # Exact/substring match bonus (20%)
            exact = 0.0
            if l_norm == u_norm:
                exact = 1.0
            elif l_norm in u_norm or u_norm in l_norm:
                exact = 0.5

            score = (0.50 * embed_sim + 0.15 * jacc +
                     0.15 * edit_sim + 0.20 * exact)
            score_matrix[i, j] = score

    # Forward: each local -> best upper
    forward = {}
    for i in range(n_local):
        j = int(np.argmax(score_matrix[i]))
        forward[i] = (j, score_matrix[i, j])

    # Backward: each upper -> best local
    backward = {}
    for j in range(n_upper):
        i = int(np.argmax(score_matrix[:, j]))
        backward[j] = (i, score_matrix[i, j])

    # Mutual best matches
    mutual = set()
    for i, (j, _) in forward.items():
        if backward.get(j, (None,))[0] == i:
            mutual.add((i, j))

    # Build alignments with 1-to-1 constraint
    used_local = set()
    used_upper = set()
    alignments = []

    # Mutual matches first
    for i, j in sorted(mutual, key=lambda x: -score_matrix[x[0], x[1]]):
        if score_matrix[i, j] >= threshold:
            alignments.append({
                'local_class': local_classes[i],
                'upper_class': upper_classes[j],
                'similarity': round(float(score_matrix[i, j]), 4),
                'direction': 'mutual',
                'embed_sim': round(float(util.cos_sim(
                    local_embeds[i:i+1], upper_embeds[j:j+1]
                )[0][0]), 4),
            })
            used_local.add(i)
            used_upper.add(j)

    # Remaining forward matches
    remaining = [(i, j, s) for i, (j, s) in forward.items()
                 if i not in used_local and s >= threshold]
    for i, j, s in sorted(remaining, key=lambda x: -x[2]):
        if j not in used_upper:
            alignments.append({
                'local_class': local_classes[i],
                'upper_class': upper_classes[j],
                'similarity': round(float(s), 4),
                'direction': 'forward',
                'embed_sim': round(float(util.cos_sim(
                    local_embeds[i:i+1], upper_embeds[j:j+1]
                )[0][0]), 4),
            })
            used_local.add(i)
            used_upper.add(j)

    # Remaining backward matches
    remaining_back = [(j, i, s) for j, (i, s) in backward.items()
                      if j not in used_upper and i not in used_local
                      and s >= threshold]
    for j, i, s in sorted(remaining_back, key=lambda x: -x[2]):
        alignments.append({
            'local_class': local_classes[i],
            'upper_class': upper_classes[j],
            'similarity': round(float(s), 4),
            'direction': 'backward',
            'embed_sim': round(float(util.cos_sim(
                local_embeds[i:i+1], upper_embeds[j:j+1]
            )[0][0]), 4),
        })
        used_local.add(i)
        used_upper.add(j)

    return alignments, score_matrix


def align_to_upper_ontology(local_classes, embedder, upper_source, output_path,
                            upper_name="upper", threshold=0.4):
    """Align local ontology classes to an upper-level ontology.

    Args:
        local_classes: list of local class labels
        embedder: SentenceTransformer model
        upper_source: path to ontology file, or dict of {category: [labels]}
        output_path: CSV output path
        upper_name: name of upper ontology (for logging)
        threshold: alignment threshold

    Returns:
        alignments: list of alignment dicts
        stats: summary statistics dict
    """
    # Load upper ontology
    if isinstance(upper_source, dict):
        # Flatten dict to labels
        upper_label_list = []
        for category, labels in upper_source.items():
            for label in labels:
                if label not in upper_label_list:
                    upper_label_list.append(label)
    elif os.path.exists(upper_source):
        _, upper_label_list = load_ontology_classes_rdf(upper_source)
    else:
        raise FileNotFoundError(f"Upper ontology not found: {upper_source}")

    print(f"\nAligning {len(local_classes)} local -> {len(upper_label_list)} "
          f"{upper_name} classes")

    # Embed all class labels
    local_normalized = [normalize_term(c) for c in local_classes]
    upper_normalized = [normalize_term(c) for c in upper_label_list]

    local_embeds = embedder.encode(local_normalized, convert_to_numpy=True)
    upper_embeds = embedder.encode(upper_normalized, convert_to_numpy=True)

    # Run MILA-style bidirectional alignment
    alignments, score_matrix = mila_alignment(
        local_classes, local_embeds, upper_label_list, upper_embeds,
        threshold=threshold
    )

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(alignments)
    if len(df) > 0:
        df = df.sort_values('similarity', ascending=False)
    df.to_csv(output_path, index=False)

    # Compute statistics
    n_mutual = sum(1 for a in alignments if a['direction'] == 'mutual')
    n_forward = sum(1 for a in alignments if a['direction'] == 'forward')
    n_backward = sum(1 for a in alignments if a['direction'] == 'backward')
    avg_sim = float(df['similarity'].mean()) if len(df) > 0 else 0.0

    # Coverage analysis
    aligned_local = set(a['local_class'] for a in alignments)
    aligned_upper = set(a['upper_class'] for a in alignments)
    local_coverage = len(aligned_local) / len(local_classes) if local_classes else 0
    upper_coverage = len(aligned_upper) / len(upper_label_list) if upper_label_list else 0

    # Gap analysis: unaligned local classes
    unaligned_local = [c for c in local_classes if c not in aligned_local]
    unaligned_upper = [c for c in upper_label_list if c not in aligned_upper]

    stats = {
        'upper_ontology': upper_name,
        'n_local_classes': len(local_classes),
        'n_upper_classes': len(upper_label_list),
        'n_alignments': len(alignments),
        'n_mutual': n_mutual,
        'n_forward': n_forward,
        'n_backward': n_backward,
        'avg_similarity': round(avg_sim, 4),
        'local_coverage': round(local_coverage, 4),
        'upper_coverage': round(upper_coverage, 4),
        'n_unaligned_local': len(unaligned_local),
        'n_unaligned_upper': len(unaligned_upper),
        'threshold': threshold,
    }

    # Similarity distribution
    if len(df) > 0:
        stats['sim_quartiles'] = {
            'q25': round(float(df['similarity'].quantile(0.25)), 4),
            'q50': round(float(df['similarity'].quantile(0.50)), 4),
            'q75': round(float(df['similarity'].quantile(0.75)), 4),
        }

    print(f"  Alignments: {len(alignments)} "
          f"(mutual={n_mutual}, fwd={n_forward}, bwd={n_backward})")
    print(f"  Avg similarity: {avg_sim:.4f}")
    print(f"  Local coverage: {local_coverage:.1%} "
          f"({len(aligned_local)}/{len(local_classes)})")
    print(f"  Upper coverage: {upper_coverage:.1%} "
          f"({len(aligned_upper)}/{len(upper_label_list)})")

    return alignments, stats, unaligned_local, unaligned_upper


def generate_quality_report(emmo_stats, bfo_stats,
                            emmo_unaligned_local, emmo_unaligned_upper,
                            bfo_unaligned_local, bfo_unaligned_upper,
                            output_path):
    """Generate a cross-domain alignment quality report."""
    lines = []
    lines.append("=" * 70)
    lines.append("CROSS-DOMAIN ONTOLOGY ALIGNMENT REPORT")
    lines.append("MaterioMiner Ontology -> EMMO / BFO")
    lines.append("=" * 70)

    for name, stats, unaligned_local, unaligned_upper in [
        ("EMMO", emmo_stats, emmo_unaligned_local, emmo_unaligned_upper),
        ("BFO", bfo_stats, bfo_unaligned_local, bfo_unaligned_upper),
    ]:
        lines.append(f"\n{'─' * 50}")
        lines.append(f"Alignment to {name}")
        lines.append(f"{'─' * 50}")
        lines.append(f"  Local classes:     {stats['n_local_classes']}")
        lines.append(f"  {name} classes:    {stats['n_upper_classes']}")
        lines.append(f"  Total alignments:  {stats['n_alignments']}")
        lines.append(f"    Mutual matches:  {stats['n_mutual']}")
        lines.append(f"    Forward-only:    {stats['n_forward']}")
        lines.append(f"    Backward-only:   {stats['n_backward']}")
        lines.append(f"  Avg similarity:    {stats['avg_similarity']:.4f}")
        lines.append(f"  Local coverage:    {stats['local_coverage']:.1%}")
        lines.append(f"  {name} coverage:   {stats['upper_coverage']:.1%}")
        lines.append(f"  Threshold:         {stats['threshold']}")

        if 'sim_quartiles' in stats:
            q = stats['sim_quartiles']
            lines.append(f"  Similarity Q25/Q50/Q75: "
                         f"{q['q25']:.4f} / {q['q50']:.4f} / {q['q75']:.4f}")

        if unaligned_upper:
            lines.append(f"\n  Unaligned {name} classes ({len(unaligned_upper)}):")
            for c in sorted(unaligned_upper)[:15]:
                lines.append(f"    - {c}")
            if len(unaligned_upper) > 15:
                lines.append(f"    ... and {len(unaligned_upper) - 15} more")

        if unaligned_local:
            n_show = min(10, len(unaligned_local))
            lines.append(f"\n  Sample unaligned local classes "
                         f"({len(unaligned_local)} total, showing {n_show}):")
            for c in sorted(unaligned_local)[:n_show]:
                lines.append(f"    - {c}")

    lines.append(f"\n{'=' * 70}")
    lines.append("ANALYSIS SUMMARY")
    lines.append(f"{'=' * 70}")

    emmo_cov = emmo_stats['local_coverage']
    bfo_cov = bfo_stats['local_coverage']
    lines.append(f"\nLocal class alignment rate:")
    lines.append(f"  EMMO: {emmo_cov:.1%} of domain classes align to EMMO")
    lines.append(f"  BFO:  {bfo_cov:.1%} of domain classes align to BFO")
    lines.append(f"\nUpper ontology coverage:")
    lines.append(f"  EMMO: {emmo_stats['upper_coverage']:.1%} of EMMO classes matched")
    lines.append(f"  BFO:  {bfo_stats['upper_coverage']:.1%} of BFO classes matched")

    lines.append(f"\nAlignment quality (by avg similarity):")
    lines.append(f"  EMMO: {emmo_stats['avg_similarity']:.4f}")
    lines.append(f"  BFO:  {bfo_stats['avg_similarity']:.4f}")

    better = "EMMO" if emmo_stats['avg_similarity'] > bfo_stats['avg_similarity'] else "BFO"
    lines.append(f"\n  -> {better} shows stronger average alignment similarity,")
    if better == "EMMO":
        lines.append(f"     as expected since EMMO is a materials-science-specific upper ontology")
    else:
        lines.append(f"     suggesting BFO's general categories map well to domain concepts")

    report = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")
    return report


def run_cross_domain_alignment(embedder_model=None):
    """Run cross-domain alignment to EMMO and BFO."""
    model_name = embedder_model or EMBEDDER_MODEL
    print(f"Cross-domain alignment using embedder: {model_name}")
    embedder = SentenceTransformer(model_name)

    # Load domain ontology
    print("\nLoading MaterioMiner domain ontology...")
    local_classes, _ = load_local_ontology()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # --- EMMO alignment ---
    emmo_path = os.path.join(ONTOLOGY_DIR, 'emmo.ttl')
    emmo_alignments, emmo_stats, emmo_ul, emmo_uu = align_to_upper_ontology(
        local_classes, embedder, emmo_path,
        os.path.join(PROCESSED_DIR, 'cross_domain_emmo.csv'),
        upper_name="EMMO", threshold=0.4
    )

    # --- BFO alignment ---
    bfo_path = os.path.join(ONTOLOGY_DIR, 'bfo.owl')
    bfo_alignments, bfo_stats, bfo_ul, bfo_uu = align_to_upper_ontology(
        local_classes, embedder, bfo_path,
        os.path.join(PROCESSED_DIR, 'cross_domain_bfo.csv'),
        upper_name="BFO", threshold=0.4
    )

    # Generate quality report
    report = generate_quality_report(
        emmo_stats, bfo_stats,
        emmo_ul, emmo_uu, bfo_ul, bfo_uu,
        os.path.join(PROCESSED_DIR, 'cross_domain_report.txt')
    )
    print("\n" + report)

    # Save combined stats
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
