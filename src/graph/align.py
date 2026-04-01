"""
align.py - MILA-style Bidirectional Ontology Alignment

Aligns discovered terms/clusters to reference ontology using:
  - rdflib parsing of ALL classes from ontology.ttl (rdfs:label + altLabel + prefLabel)
  - No hardcoded 8-class fallback
  - Term normalization (CamelCase split, lowercase, lemmatize, strip articles)
  - Combined scoring: 45% embedding + 15% Jaccard + 15% edit distance + 20% exact + 5% structure
  - Bidirectional matching (forward + backward) with 1-to-1 constraint
  - GNN similarity computed SEPARATELY (64d vs 64d) then weighted with text sim
  - For reference classes without GNN embeddings, text-only similarity is used
  - Ablation flags for component toggling
"""

import os
import json
import re
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

# GNN fusion weight: alpha * text_sim + (1-alpha) * gnn_sim
GNN_FUSION_ALPHA = 0.7


def normalize_term(term):
    """Normalize a term: CamelCase split, remove hyphens, lowercase, strip articles.

    Handles compound NER types like "CrackGrowthBehaviour" -> "crack growth behaviour"
    and ontology labels like "High-cycle fatigue" -> "high cycle fatigue".
    """
    term = term.strip()
    # CamelCase splitting (handles "CrackGrowth" and "GNNModel")
    term = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', term)
    term = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', term)
    term = term.lower()
    # Replace hyphens and underscores with spaces
    term = re.sub(r'[-_]', ' ', term)
    # Strip leading articles
    for prefix in ('the ', 'a ', 'an '):
        if term.startswith(prefix):
            term = term[len(prefix):]
    # Remove parentheticals
    term = re.sub(r'\([^)]*\)', '', term)
    # Collapse multiple spaces
    term = re.sub(r'\s+', ' ', term).strip()
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


def combined_score(disc_term, ref_term, disc_sbert_embed, ref_sbert_embed,
                   G=None, disc_in_degree=0,
                   disc_gnn_embed=None, ref_gnn_embed=None,
                   alpha=GNN_FUSION_ALPHA):
    """Compute combined alignment score.

    FIXED: GNN and text similarities computed SEPARATELY in their native
    dimensions (text: 384d vs 384d, GNN: 64d vs 64d), then combined as:
        combined_sim = alpha * text_sim + (1-alpha) * gnn_sim

    For entities without GNN embeddings, text-only similarity is used.

    Weights: 45% embedding + 15% Jaccard + 15% edit + 20% exact + 5% structure
    """
    if not USE_COMBINED_SCORING:
        # Fallback: pure embedding similarity (text-only)
        text_sim = util.cos_sim(
            torch.tensor(disc_sbert_embed).float().unsqueeze(0),
            torch.tensor(ref_sbert_embed).float().unsqueeze(0)
        )[0][0].item()
        # Add GNN bonus if both have GNN embeddings
        if disc_gnn_embed is not None and ref_gnn_embed is not None:
            gnn_sim = util.cos_sim(
                torch.tensor(disc_gnn_embed).float().unsqueeze(0),
                torch.tensor(ref_gnn_embed).float().unsqueeze(0)
            )[0][0].item()
            return alpha * text_sim + (1 - alpha) * gnn_sim
        return text_sim

    # Normalize terms
    d_norm = normalize_term(disc_term)
    r_norm = normalize_term(ref_term)

    # 1. Embedding similarity (45%) - separate text and GNN sims
    text_sim = util.cos_sim(
        torch.tensor(disc_sbert_embed).float().unsqueeze(0),
        torch.tensor(ref_sbert_embed).float().unsqueeze(0)
    )[0][0].item()

    if disc_gnn_embed is not None and ref_gnn_embed is not None:
        gnn_sim = util.cos_sim(
            torch.tensor(disc_gnn_embed).float().unsqueeze(0),
            torch.tensor(ref_gnn_embed).float().unsqueeze(0)
        )[0][0].item()
        embed_sim = alpha * text_sim + (1 - alpha) * gnn_sim
    else:
        embed_sim = text_sim

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


def bidirectional_alignment(discovered, ref_classes,
                            disc_sbert, ref_sbert,
                            disc_gnn_map=None, ref_gnn_map=None,
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
        d_gnn = disc_gnn_map.get(d) if disc_gnn_map else None
        for j, r in enumerate(ref_classes):
            r_gnn = ref_gnn_map.get(r) if ref_gnn_map else None
            score_matrix[i, j] = combined_score(
                d, r, disc_sbert[i], ref_sbert[j], G, d_deg,
                disc_gnn_embed=d_gnn, ref_gnn_embed=r_gnn
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
    """Run the full MILA-style alignment pipeline.

    FIXED GNN fusion: computes text similarity (384d) and GNN similarity (64d)
    separately, then combines them with alpha weighting. Reference ontology
    classes have NO GNN embeddings (they're not in the co-occurrence graph),
    so text-only similarity is used for them.
    """
    discovered, G = load_discovered()
    if not discovered:
        print("No discovered classes. Run taxonomy.py first.")
        return

    primary_labels, all_labels, class_map = load_reference()
    ref_classes = primary_labels

    print(f"\nAligning {len(discovered)} discovered -> {len(ref_classes)} reference classes")

    # Compute SBERT embeddings for both sides (always in native 384d)
    disc_sbert = EMBEDDER.encode(discovered)
    ref_sbert = EMBEDDER.encode(ref_classes)

    # Load GNN embeddings separately (64d) - only discovered terms have them
    disc_gnn_map = None
    if USE_GNN_EMBEDDINGS:
        gnn_embeds, gnn_map = load_gnn_embeddings()
        if gnn_embeds is not None:
            disc_gnn_map = {}
            n_with_gnn = 0
            for d in discovered:
                if d in gnn_map:
                    gnn_idx = gnn_map[d]
                    if gnn_idx < gnn_embeds.shape[0]:
                        disc_gnn_map[d] = gnn_embeds[gnn_idx]
                        n_with_gnn += 1
            print(f"GNN embeddings available for {n_with_gnn}/{len(discovered)} "
                  f"discovered terms (dim={gnn_embeds.shape[1]})")
            print(f"Reference classes have NO GNN embeddings -> text-only for ref")
            print(f"Fusion: alpha={GNN_FUSION_ALPHA} (text-dominant)")

    # Reference classes don't have GNN embeddings (not in co-occurrence graph)
    # So ref_gnn_map stays None -> text-only similarity for ref classes

    # Run bidirectional alignment
    alignments = bidirectional_alignment(
        discovered, ref_classes, disc_sbert, ref_sbert,
        disc_gnn_map=disc_gnn_map, ref_gnn_map=None, G=G
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
        'GNN_FUSION_ALPHA': GNN_FUSION_ALPHA,
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
