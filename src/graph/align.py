"""
align.py - MILA-style Bidirectional Ontology Alignment

Aligns discovered terms/clusters to reference ontology using:
  - rdflib parsing of ALL classes from ontology.ttl (rdfs:label + altLabel + prefLabel)
  - No hardcoded 8-class fallback
  - Term normalization (lowercase, lemmatize, strip articles)
  - Combined scoring: 45% embedding + 15% Jaccard + 15% edit distance + 20% exact + 5% structure
  - Bidirectional matching (forward + backward) with 1-to-1 constraint
  - GNN+text embedding fusion when available
  - Ablation flags for component toggling
"""

import os
import json
import pickle
from difflib import SequenceMatcher

import numpy as np
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from config import DEFAULT_EMBEDDER_MODEL

# Get project root directory (2 levels up from this script)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dataset")

TAXONOMY_PATH = os.path.join(PROCESSED_DIR, 'taxonomy_graph.pkl')
GNN_EMBED_PATH = os.path.join(PROCESSED_DIR, 'gnn_embeddings.npy')
GNN_MAP_PATH = os.path.join(PROCESSED_DIR, 'gnn_embed_map.json')

EMBEDDER_MODEL = os.environ.get('EMBEDDER_MODEL', DEFAULT_EMBEDDER_MODEL)
EMBEDDER = SentenceTransformer(EMBEDDER_MODEL)

# Ablation flags
USE_GNN_EMBEDDINGS = True
USE_BIDIRECTIONAL = True
USE_COMBINED_SCORING = True


def normalize_term(term):
    """Normalize a term: lowercase, strip articles, simple lemmatization."""
    term = term.lower().strip()
    # Strip leading articles
    for prefix in ('the ', 'a ', 'an '):
        if term.startswith(prefix):
            term = term[len(prefix):]
    # CamelCase splitting
    import re
    term = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', term).lower()
    # Remove parentheticals
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
    """Normalized edit distance similarity using SequenceMatcher."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def load_discovered():
    """Load taxonomy nodes as discovered classes."""
    if not os.path.exists(TAXONOMY_PATH):
        print("Warning: taxonomy_graph.pkl not found. Run taxonomy.py first.")
        return [], None
    with open(TAXONOMY_PATH, 'rb') as f:
        G = pickle.load(f)
    nodes = list(G.nodes)
    print(f"Discovered: {len(nodes)} classes ({G.number_of_edges()} isa edges)")
    return nodes, G


def load_reference():
    """Load ALL ontology classes from ontology.ttl using rdflib.

    Extracts: rdfs:label, altLabel, prefLabel for each owl:Class.
    Returns list of unique class labels.
    """
    ttl_path = os.path.join(RAW_DIR, 'ontologies', 'ontology.ttl')

    if not os.path.exists(ttl_path):
        raise FileNotFoundError(
            f"Ontology file not found: {ttl_path}. "
            "Place ontology.ttl in data/raw/dataset/ontologies/"
        )

    from rdflib import Graph, RDF, RDFS, OWL, Namespace

    g = Graph()
    g.parse(ttl_path, format='turtle')

    # Namespace for the ontology's custom annotation properties
    MMO = Namespace("https://w3id.org/pmd/materials-mechanics-ontology/")
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

    classes = {}  # URI -> list of labels
    label_props = [RDFS.label, MMO.altLabel, MMO.prefLabel,
                   SKOS.prefLabel, SKOS.altLabel]

    for cls in g.subjects(RDF.type, OWL.Class):
        cls_str = str(cls)
        if cls_str.startswith('http://www.w3.org/'):
            continue  # Skip OWL built-in classes

        labels = []
        for prop in label_props:
            for label in g.objects(cls, prop):
                label_str = str(label).strip()
                if label_str and not label_str.startswith('http'):
                    labels.append(label_str)

        # If no labels found, extract from URI fragment
        if not labels:
            fragment = cls_str.split('#')[-1].split('/')[-1]
            if fragment and fragment[0].isupper():
                labels.append(fragment)

        if labels:
            classes[cls_str] = labels

    # Flatten: collect all unique labels, with primary label first
    all_labels = set()
    primary_labels = []
    for uri, labels in classes.items():
        primary = labels[0]
        if primary not in all_labels:
            primary_labels.append(primary)
            all_labels.add(primary)
        for alt in labels[1:]:
            all_labels.add(alt)

    print(f"Loaded from ontology.ttl: {len(classes)} classes, "
          f"{len(all_labels)} total labels")
    return primary_labels, all_labels, classes


def load_gnn_embeddings():
    """Load GNN embeddings if available."""
    if os.path.exists(GNN_EMBED_PATH) and os.path.exists(GNN_MAP_PATH):
        gnn_embeds = np.load(GNN_EMBED_PATH)
        with open(GNN_MAP_PATH) as f:
            gnn_map = json.load(f)
        print(f"Loaded GNN embeddings: {gnn_embeds.shape}")
        return gnn_embeds, gnn_map
    return None, None


def fused_embedding(term, sbert_embed, gnn_embeds, gnn_map, alpha=0.5):
    """Fuse SBERT and GNN embeddings for a term.

    If GNN embedding available, returns alpha*GNN + (1-alpha)*SBERT (normalized).
    Otherwise returns SBERT embedding.
    """
    if gnn_embeds is not None and gnn_map is not None and term in gnn_map:
        gnn_idx = gnn_map[term]
        if gnn_idx < gnn_embeds.shape[0]:
            gnn_vec = gnn_embeds[gnn_idx]
            # Project to same dim if needed (pad shorter with zeros)
            sbert_vec = sbert_embed
            max_dim = max(len(gnn_vec), len(sbert_vec))
            gnn_padded = np.zeros(max_dim)
            sbert_padded = np.zeros(max_dim)
            gnn_padded[:len(gnn_vec)] = gnn_vec
            sbert_padded[:len(sbert_vec)] = sbert_vec
            fused = alpha * gnn_padded + (1 - alpha) * sbert_padded
            norm = np.linalg.norm(fused)
            if norm > 0:
                fused = fused / norm
            return fused
    return sbert_embed


def combined_score(disc_term, ref_term, disc_embed, ref_embed,
                   G=None, disc_in_degree=0):
    """Compute combined alignment score.

    Weights: 45% embedding + 15% Jaccard + 15% edit + 20% exact + 5% structure
    """
    if not USE_COMBINED_SCORING:
        # Fallback: pure embedding similarity
        sim = util.cos_sim(
            torch.tensor(disc_embed).unsqueeze(0),
            torch.tensor(ref_embed).unsqueeze(0)
        )[0][0].item()
        return sim

    # Normalize terms
    d_norm = normalize_term(disc_term)
    r_norm = normalize_term(ref_term)

    # 1. Embedding similarity (45%)
    embed_sim = util.cos_sim(
        torch.tensor(disc_embed).float().unsqueeze(0),
        torch.tensor(ref_embed).float().unsqueeze(0)
    )[0][0].item()

    # 2. Jaccard similarity (15%)
    jacc = jaccard_similarity(d_norm, r_norm)

    # 3. Edit distance similarity (15%)
    edit_sim = edit_distance_similarity(d_norm, r_norm)

    # 4. Exact match bonus (20%)
    exact = 1.0 if d_norm == r_norm else 0.0
    # Partial exact: if one is substring of other
    if not exact and (d_norm in r_norm or r_norm in d_norm):
        exact = 0.5

    # 5. Structure bonus (5%)
    struct = min(disc_in_degree / 10.0, 1.0) if disc_in_degree > 0 else 0.0

    score = (0.45 * embed_sim + 0.15 * jacc + 0.15 * edit_sim +
             0.20 * exact + 0.05 * struct)
    return score


def bidirectional_alignment(discovered, ref_classes, disc_embeds, ref_embeds,
                            G=None):
    """MILA-style bidirectional matching with 1-to-1 constraint.

    1. Forward: each discovered -> best reference
    2. Backward: each reference -> best discovered
    3. Combine: prefer mutual best matches, resolve conflicts by score
    """
    n_disc = len(discovered)
    n_ref = len(ref_classes)

    # Compute full score matrix
    score_matrix = np.zeros((n_disc, n_ref))
    for i, d in enumerate(discovered):
        d_deg = G.in_degree(d) if G and d in G else 0
        for j, r in enumerate(ref_classes):
            score_matrix[i, j] = combined_score(
                d, r, disc_embeds[i], ref_embeds[j], G, d_deg
            )

    if not USE_BIDIRECTIONAL:
        # Simple forward matching
        alignments = []
        for i, d in enumerate(discovered):
            j = np.argmax(score_matrix[i])
            alignments.append({
                'discovered': d,
                'reference': ref_classes[j],
                'similarity': score_matrix[i, j],
                'direction': 'forward',
            })
        return alignments

    # Forward: each discovered -> best reference
    forward = {}
    for i in range(n_disc):
        j = np.argmax(score_matrix[i])
        forward[i] = (j, score_matrix[i, j])

    # Backward: each reference -> best discovered
    backward = {}
    for j in range(n_ref):
        i = np.argmax(score_matrix[:, j])
        backward[j] = (i, score_matrix[i, j])

    # Mutual best matches (stable marriages)
    mutual = set()
    for i, (j, _) in forward.items():
        if backward.get(j, (None,))[0] == i:
            mutual.add((i, j))

    # Build final alignments with 1-to-1 constraint
    used_disc = set()
    used_ref = set()
    alignments = []

    # First: mutual matches (highest confidence)
    for i, j in sorted(mutual, key=lambda x: -score_matrix[x[0], x[1]]):
        alignments.append({
            'discovered': discovered[i],
            'reference': ref_classes[j],
            'similarity': score_matrix[i, j],
            'direction': 'mutual',
        })
        used_disc.add(i)
        used_ref.add(j)

    # Second: remaining forward matches (1-to-1)
    remaining = [(i, j, s) for i, (j, s) in forward.items()
                 if i not in used_disc]
    for i, j, s in sorted(remaining, key=lambda x: -x[2]):
        if j not in used_ref:
            alignments.append({
                'discovered': discovered[i],
                'reference': ref_classes[j],
                'similarity': s,
                'direction': 'forward',
            })
            used_disc.add(i)
            used_ref.add(j)

    # Third: remaining backward matches
    remaining_back = [(j, i, s) for j, (i, s) in backward.items()
                      if j not in used_ref and i not in used_disc]
    for j, i, s in sorted(remaining_back, key=lambda x: -x[2]):
        alignments.append({
            'discovered': discovered[i],
            'reference': ref_classes[j],
            'similarity': s,
            'direction': 'backward',
        })
        used_disc.add(i)
        used_ref.add(j)

    return alignments


def run_alignment():
    """Run the full MILA-style alignment pipeline."""
    discovered, G = load_discovered()
    if not discovered:
        print("No discovered classes. Run taxonomy.py first.")
        return

    primary_labels, all_labels, class_map = load_reference()
    ref_classes = primary_labels

    print(f"\nAligning {len(discovered)} discovered -> {len(ref_classes)} reference classes")

    # Load GNN embeddings for fusion
    gnn_embeds, gnn_map = None, None
    if USE_GNN_EMBEDDINGS:
        gnn_embeds, gnn_map = load_gnn_embeddings()

    # Compute embeddings for discovered terms
    disc_sbert = EMBEDDER.encode(discovered)
    if USE_GNN_EMBEDDINGS and gnn_embeds is not None:
        disc_embeds = np.array([
            fused_embedding(d, disc_sbert[i], gnn_embeds, gnn_map)
            for i, d in enumerate(discovered)
        ])
        print(f"Using fused GNN+SBERT embeddings (dim={disc_embeds.shape[1]})")
    else:
        disc_embeds = disc_sbert

    # Compute embeddings for reference classes
    ref_sbert = EMBEDDER.encode(ref_classes)
    # Reference classes don't have GNN embeddings, use SBERT
    # But we need same dimensionality
    if disc_embeds.shape[1] != ref_sbert.shape[1]:
        # Pad reference embeddings to match fused dimension
        ref_embeds = np.zeros((len(ref_classes), disc_embeds.shape[1]))
        ref_embeds[:, :ref_sbert.shape[1]] = ref_sbert
    else:
        ref_embeds = ref_sbert

    # Run bidirectional alignment
    alignments = bidirectional_alignment(
        discovered, ref_classes, disc_embeds, ref_embeds, G
    )

    # Create results DataFrame
    df = pd.DataFrame(alignments).sort_values('similarity', ascending=False)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(os.path.join(PROCESSED_DIR, 'ontology_alignment.csv'), index=False)

    # Stats
    n_mutual = sum(1 for a in alignments if a['direction'] == 'mutual')
    n_forward = sum(1 for a in alignments if a['direction'] == 'forward')
    n_backward = sum(1 for a in alignments if a['direction'] == 'backward')
    avg_sim = df['similarity'].mean() if len(df) > 0 else 0

    print(f"\nAlignment results:")
    print(f"  Total: {len(alignments)} alignments")
    print(f"  Mutual: {n_mutual}, Forward: {n_forward}, Backward: {n_backward}")
    print(f"  Avg similarity: {avg_sim:.4f}")
    print(f"  Saved to {os.path.join(PROCESSED_DIR, 'ontology_alignment.csv')}")

    # Save config for ablation tracking
    config = {
        'USE_GNN_EMBEDDINGS': USE_GNN_EMBEDDINGS,
        'USE_BIDIRECTIONAL': USE_BIDIRECTIONAL,
        'USE_COMBINED_SCORING': USE_COMBINED_SCORING,
        'n_discovered': len(discovered),
        'n_reference': len(ref_classes),
        'n_alignments': len(alignments),
        'n_mutual': n_mutual,
        'avg_similarity': round(avg_sim, 4),
    }
    with open(os.path.join(PROCESSED_DIR, 'alignment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    return df.head(20)


if __name__ == '__main__':
    top = run_alignment()
    if top is not None:
        print("\nTop Matches:")
        print(top.to_string())
