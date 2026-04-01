"""
cluster.py - Term Clustering with GNN Embeddings and NER Types

Clusters graph nodes using agglomerative clustering with:
  - GNN embeddings when available (preferred over raw SBERT)
  - NER-type-based sub-clustering when entity_types.json available
  - API key from environment variable (no hardcoded keys)
  - Optional Gemini LLM cluster naming
"""

import os
import json

import numpy as np
import torch
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import defaultdict

# Get project root directory (2 levels up from this script)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Paths
GRAPHPATHS = {
    'fine_auto': os.path.join(PROCESSED_DIR, 'graph_fine_auto.pt'),
    'coarse': os.path.join(PROCESSED_DIR, 'graph_coarse.pt')
}
NODEMAPPATHS = {
    'fine_auto': os.path.join(PROCESSED_DIR, 'nodemap_fine_auto.pt'),
    'coarse': os.path.join(PROCESSED_DIR, 'nodemap_coarse.pt')
}
GNN_EMBED_PATH = os.path.join(PROCESSED_DIR, 'gnn_embeddings.npy')
GNN_MAP_PATH = os.path.join(PROCESSED_DIR, 'gnn_embed_map.json')
ENTITY_TYPES_PATH = os.path.join(PROCESSED_DIR, 'entity_types.json')


def load_graph(key):
    """Load PyG data and inverse node->term map."""
    graph_path = GRAPHPATHS[key]
    nodemap_path = NODEMAPPATHS[key]
    if not (os.path.exists(graph_path) and os.path.exists(nodemap_path)):
        print(f"Skip {key}: Missing {graph_path} or {nodemap_path}. "
              "Run builder.py first.")
        return None, None, None
    data = torch.load(graph_path, weights_only=False)
    nodemap = torch.load(nodemap_path, weights_only=False)
    inv_nodemap = {idx: term for term, idx in nodemap.items()}

    # Use GNN embeddings if available, otherwise fall back to SBERT
    if os.path.exists(GNN_EMBED_PATH) and os.path.exists(GNN_MAP_PATH):
        print(f"  Using GNN embeddings from {GNN_EMBED_PATH}")
        gnn_embeds = np.load(GNN_EMBED_PATH)
        with open(GNN_MAP_PATH) as f:
            gnn_map = json.load(f)
        # Reorder to match nodemap ordering
        X = np.zeros((len(nodemap), gnn_embeds.shape[1]))
        for term, idx in nodemap.items():
            if term in gnn_map:
                gnn_idx = gnn_map[term]
                if gnn_idx < gnn_embeds.shape[0]:
                    X[idx] = gnn_embeds[gnn_idx]
        # Fill missing with SBERT features
        sbert_x = data.x.cpu().numpy()
        for idx in range(len(nodemap)):
            if np.allclose(X[idx], 0):
                if idx < sbert_x.shape[0]:
                    X[idx] = sbert_x[idx]
    else:
        print("  GNN embeddings not available, using SBERT features")
        X = data.x.cpu().numpy()

    return X, inv_nodemap, nodemap


def optimal_nclusters(X, max_k=50):
    """Find best n_clusters via silhouette score."""
    best_k, best_score = 5, -1
    max_possible = min(max_k, X.shape[0] // 3)
    if max_possible < 5:
        return min(5, X.shape[0] // 2)
    for k in range(5, max_possible + 1):
        clust = AgglomerativeClustering(
            n_clusters=k, metric='cosine', linkage='average'
        )
        labels = clust.fit_predict(X)
        score = silhouette_score(X, labels, metric='cosine')
        if score > best_score:
            best_score, best_k = score, k
    print(f"  Optimal clusters: {best_k} (silhouette {best_score:.3f})")
    return best_k


def cluster_graph(key, use_llm=False):
    """Cluster graph nodes into term groups.

    Uses GNN embeddings when available, with optional NER type sub-clustering.
    """
    X, inv_nodemap, nodemap = load_graph(key)
    if X is None:
        return None, None

    print(f"Processing {key}: {X.shape[0]} nodes, dim={X.shape[1]}")

    n_clusters = optimal_nclusters(X)
    clust = AgglomerativeClustering(
        n_clusters=n_clusters, metric='cosine', linkage='average'
    )
    labels = clust.fit_predict(X)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        term = inv_nodemap[idx]
        clusters[label].append(term)

    # Load entity types for enrichment if available
    entity_types = {}
    if os.path.exists(ENTITY_TYPES_PATH):
        with open(ENTITY_TYPES_PATH) as f:
            entity_types = json.load(f)
        print(f"  Loaded {len(entity_types)} entity types for cluster enrichment")

    # Sort terms within clusters
    cluster_list = {
        int(cid): sorted(terms, key=len)
        for cid, terms in clusters.items()
    }

    # Determine cluster names from dominant NER type
    cluster_names = {}
    for cid, terms in cluster_list.items():
        type_counts = defaultdict(int)
        for t in terms:
            if t in entity_types:
                type_counts[entity_types[t]] += 1
        if type_counts:
            dominant_type = max(type_counts, key=type_counts.get)
            cluster_names[cid] = dominant_type
        else:
            cluster_names[cid] = f"Cluster_{cid}"

    # LLM naming override (optional)
    if use_llm:
        llm_names = llm_name_clusters(cluster_list)
        cluster_names.update(llm_names)
        print(f"  LLM named {len(llm_names)} clusters")

    cluster_df = pd.DataFrame({
        'cluster_id': list(cluster_list.keys()),
        'cluster_name': [cluster_names.get(cid, f"Cluster_{cid}")
                         for cid in cluster_list.keys()],
        'n_terms': [len(terms) for terms in cluster_list.values()],
        'top_terms': [', '.join(terms[:5]) for terms in cluster_list.values()]
    }).sort_values('n_terms', ascending=False)

    return cluster_df, cluster_list


# ============== Gemini LLM Integration ==============
LLM_MODE = False  # Disabled by default; set to True or use env var


def llm_name_cluster(terms, domain="materials mechanics"):
    """Use Gemini to propose an ontology class name for a cluster."""
    import google.generativeai as genai

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return f"Cluster_{len(terms)}terms"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = f"""Domain: materials science (microstructure, fatigue, alloys).
Top terms: {', '.join(terms[:8])}.
Propose **ONE** ontology class name (e.g., "MicrostructureFeature", "FatigueCrack").
Format: CLASSNAME only."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"  LLM error: {e}")
        return f"Cluster_{len(terms)}terms"


def llm_name_clusters(cluster_list):
    """Name all clusters using LLM."""
    names = {}
    for cid, terms in cluster_list.items():
        if len(terms) >= 2:
            names[cid] = llm_name_cluster(terms)
    return names


if __name__ == '__main__':
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Check env var for LLM mode
    use_llm = os.getenv('USE_LLM_CLUSTERING', '').lower() in ('1', 'true', 'yes')
    if use_llm:
        LLM_MODE = True

    results = {}
    for key in ['fine_auto', 'coarse']:
        df, clusters = cluster_graph(key, use_llm=LLM_MODE)
        if df is not None:
            df.to_csv(os.path.join(PROCESSED_DIR, f'{key}_clusters.csv'),
                      index=False)
            with open(os.path.join(PROCESSED_DIR, f'{key}_clusters.json'),
                      'w', encoding='utf-8') as f:
                json.dump(clusters, f, indent=2, ensure_ascii=False)
            results[key] = df.head(10)
            print(f"Saved {key} clusters: {len(clusters)} groups")

    if results:
        summary = pd.concat([df.assign(graph=key) for key, df in results.items()])
        summary.to_csv(os.path.join(PROCESSED_DIR, 'all_clusters_summary.csv'),
                       index=False)
        print(summary)
    else:
        print("No graphs found. Run: python src/graph/builder.py")
