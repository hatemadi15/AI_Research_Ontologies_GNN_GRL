import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import defaultdict

# Get project root directory (2 levels up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Paths for graphs from builder.py (fine-auto primary)
GRAPHPATHS = {
    'fine_auto': os.path.join(PROCESSED_DIR, 'graph_fine_auto.pt'),
    'coarse': os.path.join(PROCESSED_DIR, 'graph_coarse.pt')
}
NODEMAPPATHS = {
    'fine_auto': os.path.join(PROCESSED_DIR, 'nodemap_fine_auto.pt'),
    'coarse': os.path.join(PROCESSED_DIR, 'nodemap_coarse.pt')
}

def load_graph(key):
    """Load PyG data and inverse node->term map."""
    graph_path = GRAPHPATHS[key]
    nodemap_path = NODEMAPPATHS[key]
    if not (os.path.exists(graph_path) and os.path.exists(nodemap_path)):
        print(f"Skip {key}: Missing {graph_path} or {nodemap_path}. Run builder.py first.")
        return None, None
    data = torch.load(graph_path, weights_only=False)
    nodemap = torch.load(nodemap_path, weights_only=False)
    X = data.x.cpu().numpy()
    inv_nodemap = {idx: term for term, idx in nodemap.items()}
    return X, inv_nodemap

def optimal_nclusters(X, max_k=50):
    """Find best n_clusters via silhouette score."""
    best_k, best_score = 5, -1
    for k in range(5, min(max_k, X.shape[0] // 3) + 1):
        clust = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        labels = clust.fit_predict(X)
        score = silhouette_score(X, labels, metric='cosine')
        if score > best_score:
            best_score, best_k = score, k
    print(f"Optimal clusters: {best_k} (silhouette {best_score:.3f})")
    return best_k

def cluster_graph(key, use_llm=False):
    """Cluster one graph's embeddings into term groups.
    
    Args:
        key: Graph key ('fine_auto' or 'coarse')
        use_llm: If True, use Gemini LLM to name clusters
    """
    X, inv_nodemap = load_graph(key)
    if X is None:
        return None, None
    print(f"Processing {key}: {X.shape[0]} nodes")
    n_clusters = optimal_nclusters(X)
    clust = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    labels = clust.fit_predict(X)
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        term = inv_nodemap[idx]
        clusters[label].append(term)
    # Sort: clusters by size, terms by freq (proxy via length)
    cluster_list = {int(cid): sorted(terms, key=len) for cid, terms in clusters.items()}
    
    # LLM cluster naming (optional)
    cluster_names = {}
    if use_llm:
        cluster_names = llm_name_clusters(cluster_list)
        print(f"LLM named {len(cluster_names)} clusters")
    
    cluster_df = pd.DataFrame({
        'cluster_id': list(cluster_list.keys()),
        'cluster_name': [cluster_names.get(cid, f"Cluster_{cid}") for cid in cluster_list.keys()],
        'n_terms': [len(terms) for terms in cluster_list.values()],
        'top_terms': [', '.join(terms[:5]) for terms in cluster_list.values()]
    }).sort_values('n_terms', ascending=False)
    return cluster_df, cluster_list


# ============== Gemini LLM Integration ==============
LLM_MODE = True  # Toggle: True to use Gemini for cluster naming, False for ablation

def llm_name_cluster(terms, domain="materials mechanics"):
    """Use Gemini to propose an ontology class name for a cluster."""
    import google.generativeai as genai
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return f"Cluster_{len(terms)}terms"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')  # Cheap/fast
    
    prompt = f"""Domain: materials science (microstructure, fatigue, alloys).
Top terms: {', '.join(terms[:8])}.
Propose **ONE** ontology class name (e.g., "MicrostructureFeature", "FatigueCrack").
Format: CLASSNAME only."""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return f"Cluster_{len(terms)}terms"


def llm_name_clusters(cluster_list):
    """Name all clusters using LLM."""
    names = {}
    for cid, terms in cluster_list.items():
        if len(terms) >= 2:  # Only name clusters with enough terms
            names[cid] = llm_name_cluster(terms)
    return names

if __name__ == '__main__':
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    results = {}
    for key in ['fine_auto', 'coarse']:
        df, clusters = cluster_graph(key, use_llm=LLM_MODE)
        if df is not None:
            df.to_csv(os.path.join(PROCESSED_DIR, f'{key}_clusters.csv'), index=False)
            with open(os.path.join(PROCESSED_DIR, f'{key}_clusters.json'), 'w', encoding='utf-8') as f:
                json.dump(clusters, f, indent=2, ensure_ascii=False)
            results[key] = df.head(10)
            print(f"Saved {key} clusters: {len(clusters)} groups")
    if results:
        summary = pd.concat([df.assign(graph=key) for key, df in results.items()])
        summary.to_csv(os.path.join(PROCESSED_DIR, 'all_clusters_summary.csv'), index=False)
        print(summary)
        print(f"Use {os.path.join(PROCESSED_DIR, 'fine_auto_clusters.json')} for taxonomy.py next.")
    else:
        print("No graphs found. Run: python src/graph/builder.py")
