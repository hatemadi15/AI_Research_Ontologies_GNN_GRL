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

from config import DEFAULT_EMBEDDER_MODEL, LLM_ALIGNMENT

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


def llm_augment_alignment(alignments, class_map):
    """FIX 4: LLM-Augmented Alignment.

    For entity↔class pairs with borderline similarity (0.35-0.65):
      - Query GPT-4o-mini: is the entity semantically related to the class?
      - If YES: boost score by +0.15
      - If NO: reduce by -0.15
    Limited to max 500 pairs, prioritized by closeness to threshold center (0.5).
    """
    if not LLM_ALIGNMENT:
        print("LLM_ALIGNMENT disabled, skipping augmentation")
        return alignments

    try:
        from llm_validator import _get_client, _load_cache, _save_cache, _cache_key
    except ImportError:
        print("llm_validator not available, skipping LLM augmentation")
        return alignments

    # Build class description lookup from class_map
    class_descriptions = {}
    for uri, labels in class_map.items():
        primary = labels[0]
        desc = ', '.join(labels[:3]) if len(labels) > 1 else primary
        class_descriptions[primary] = desc

    # Find borderline pairs (similarity 0.35-0.65)
    borderline = [(i, a) for i, a in enumerate(alignments)
                   if 0.35 <= a['similarity'] <= 0.65]

    # Sort by closeness to 0.5 (most ambiguous first)
    borderline.sort(key=lambda x: abs(x[1]['similarity'] - 0.5))

    # Limit to 500 pairs
    borderline = borderline[:500]

    if not borderline:
        print("No borderline pairs found for LLM augmentation")
        return alignments

    print(f"LLM-augmenting {len(borderline)} borderline alignment pairs...")

    cache = _load_cache()
    client = _get_client()
    llm_calls = 0
    boosted = 0
    reduced = 0

    for idx, alignment in borderline:
        entity = alignment['discovered']
        class_label = alignment['reference']
        class_desc = class_descriptions.get(class_label, class_label)

        key = _cache_key('llm_alignment', entity, class_label)
        if key in cache:
            answer = cache[key]
        else:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": (
                            f"In materials science, is the term '{entity}' "
                            f"semantically related to the ontology class "
                            f"'{class_label}' ({class_desc})? Answer YES or NO."
                        )
                    }],
                    max_tokens=5,
                    temperature=0.0,
                )
                answer = response.choices[0].message.content.strip().upper()
                cache[key] = answer
                llm_calls += 1
            except Exception as e:
                print(f"  LLM error for {entity}/{class_label}: {e}")
                continue

        if 'YES' in answer:
            alignments[idx]['similarity'] = min(1.0, alignment['similarity'] + 0.15)
            boosted += 1
        elif 'NO' in answer:
            alignments[idx]['similarity'] = max(0.0, alignment['similarity'] - 0.15)
            reduced += 1

    _save_cache(cache)
    print(f"  LLM alignment: {llm_calls} API calls, {boosted} boosted, {reduced} reduced")
    return alignments


def expand_ontology_coverage(ref_classes, discovered, disc_sbert, class_map):
    """FIX 1b: LLM-Based Ontology Class Expansion.

    For ontology classes with NO matching entity:
      - Query GPT-4o-mini for 3 synonym phrases
      - Compute SBERT similarity between synonyms and all entities
      - If any similarity > 0.55, create an alignment mapping
    Caches all LLM responses to data/processed/llm_class_expansions.json.
    """
    try:
        from llm_validator import _get_client, _load_cache, _save_cache, _cache_key
    except ImportError:
        print("llm_validator not available, skipping ontology expansion")
        return []

    # Load or create expansion cache
    expansion_cache_path = os.path.join(PROCESSED_DIR, 'llm_class_expansions.json')
    if os.path.exists(expansion_cache_path):
        with open(expansion_cache_path, 'r', encoding='utf-8') as f:
            expansion_cache = json.load(f)
    else:
        expansion_cache = {}

    # Find unmatched classes: compute SBERT similarity of each ref to all discovered
    ref_sbert = EMBEDDER.encode(ref_classes)
    from sentence_transformers import util as st_util
    sim_matrix = st_util.cos_sim(
        torch.tensor(ref_sbert).float(),
        torch.tensor(disc_sbert).float()
    ).numpy()

    # A class is "unmatched" if its best discovered similarity < 0.5
    unmatched = []
    for j, ref in enumerate(ref_classes):
        best_sim = sim_matrix[j].max()
        if best_sim < 0.5:
            unmatched.append(ref)

    print(f"Ontology expansion: {len(unmatched)}/{len(ref_classes)} classes unmatched")

    if not unmatched:
        return []

    # Limit total LLM calls (unmatched classes)
    unmatched = unmatched[:500]

    client = _get_client()
    cache = _load_cache()
    new_alignments = []
    llm_calls = 0

    for class_label in unmatched:
        key = _cache_key('expand_class', class_label)
        if key in cache:
            synonyms_str = cache[key]
        elif class_label in expansion_cache:
            synonyms_str = expansion_cache[class_label]
        else:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Generate 3 short synonym phrases for the materials "
                            f"science concept '{class_label}'. Return only the "
                            f"phrases, comma-separated."
                        )
                    }],
                    max_tokens=60,
                    temperature=0.0,
                )
                synonyms_str = response.choices[0].message.content.strip()
                cache[key] = synonyms_str
                expansion_cache[class_label] = synonyms_str
                llm_calls += 1
            except Exception as e:
                print(f"  LLM error for class '{class_label}': {e}")
                continue

        # Parse synonyms
        synonyms = [s.strip() for s in synonyms_str.split(',') if s.strip()]
        if not synonyms:
            continue

        # Compute SBERT similarity between synonyms and all discovered entities
        syn_embeds = EMBEDDER.encode(synonyms)
        syn_sims = st_util.cos_sim(
            torch.tensor(syn_embeds).float(),
            torch.tensor(disc_sbert).float()
        ).numpy()

        # Check if any synonym matches any discovered entity > 0.55
        for si in range(len(synonyms)):
            best_idx = syn_sims[si].argmax()
            best_sim = syn_sims[si, best_idx]
            if best_sim > 0.55:
                new_alignments.append({
                    'discovered': discovered[best_idx],
                    'reference': class_label,
                    'similarity': round(float(best_sim), 4),
                    'direction': 'llm_expansion',
                })
                break  # One match per class is enough

    # Save expansion cache
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(expansion_cache_path, 'w', encoding='utf-8') as f:
        json.dump(expansion_cache, f, indent=2, ensure_ascii=False)

    _save_cache(cache)
    print(f"  Expansion: {llm_calls} API calls, {len(new_alignments)} new alignments")
    return new_alignments


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

    # FIX 4: LLM-Augmented Alignment for borderline pairs
    if LLM_ALIGNMENT and os.getenv('OPENAI_API_KEY'):
        try:
            alignments = llm_augment_alignment(alignments, class_map)
        except Exception as e:
            print(f"LLM alignment augmentation failed: {e}")

    # FIX 1b: LLM-Based Ontology Class Expansion
    if LLM_ALIGNMENT and os.getenv('OPENAI_API_KEY'):
        try:
            expansion_alignments = expand_ontology_coverage(
                ref_classes, discovered, disc_sbert, class_map
            )
            if expansion_alignments:
                alignments.extend(expansion_alignments)
                print(f"  Added {len(expansion_alignments)} expansion alignments")
        except Exception as e:
            print(f"Ontology expansion failed: {e}")

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
