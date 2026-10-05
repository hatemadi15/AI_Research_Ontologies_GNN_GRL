"""
cluster.py - Term Clustering with GNN Embeddings

Clusters graph nodes with agglomerative clustering (cosine, average linkage),
choosing the number of clusters by silhouette score:
  - GNN embeddings when available and USE_GNN_EMBEDDINGS is on (fine graph
    only; the GNN is trained on the fine graph), SBERT node features otherwise
  - Cluster names are cosmetic: the most frequent term of the cluster, or an
    LLM-proposed name (USE_LLM_CLUSTERING with Gemini / LLM_MODE)

NER-type-guided clustering (one cluster per gold NER type, sub-clustered by
embedding) uses the gold annotations, so it is only available as an oracle
component (NER_TYPE_CLUSTERING, which defaults to ORACLE_TYPES).
"""

import os
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from config import (GOLD_TYPES_FILE, LLM_MODE, NER_TYPE_CLUSTERING,
                    ORACLE_TYPES, PROCESSED_DIR, USE_GNN_EMBEDDINGS)

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
GOLD_TYPES_PATH = os.path.join(PROCESSED_DIR, GOLD_TYPES_FILE)
CORPUS_FREQ_PATH = os.path.join(PROCESSED_DIR, 'corpus_frequencies.json')


def load_graph(key):
    """Load node features and the inverse node->term map for a graph."""
    import torch

    graph_path = GRAPHPATHS[key]
    nodemap_path = NODEMAPPATHS[key]
    if not (os.path.exists(graph_path) and os.path.exists(nodemap_path)):
        print(f"Skip {key}: Missing {graph_path} or {nodemap_path}. "
              "Run builder.py first.")
        return None, None, None
    data = torch.load(graph_path, weights_only=False)
    nodemap = torch.load(nodemap_path, weights_only=False)
    inv_nodemap = {idx: term for term, idx in nodemap.items()}
    X = data.x.cpu().numpy()

    # The GNN is trained on the fine graph only; the coarse baseline always
    # uses its own SBERT node features.
    if (key == 'fine_auto' and USE_GNN_EMBEDDINGS
            and os.path.exists(GNN_EMBED_PATH) and os.path.exists(GNN_MAP_PATH)):
        gnn_embeds = np.load(GNN_EMBED_PATH)
        with open(GNN_MAP_PATH) as f:
            gnn_map = json.load(f)
        missing = [t for t in nodemap if t not in gnn_map]
        if not missing:
            print(f"  Using GNN embeddings from {GNN_EMBED_PATH}")
            X = np.stack([gnn_embeds[gnn_map[inv_nodemap[i]]]
                          for i in range(len(nodemap))])
        else:
            print(f"  GNN embeddings miss {len(missing)} terms, using SBERT features")
    else:
        print("  Using SBERT node features")
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


def ner_type_guided_clustering(X, inv_nodemap, max_subcluster_size=20):
    """ORACLE: group terms by their gold majority NER type, then sub-cluster.

    Uses the gold annotations (gold_term_types.json); only called when
    NER_TYPE_CLUSTERING is enabled.
    """
    if not os.path.exists(GOLD_TYPES_PATH):
        print("  No gold types available, falling back to silhouette clustering")
        return None, None
    with open(GOLD_TYPES_PATH) as f:
        gold_types = json.load(f)
    entity_types = {t: max(sorted(c), key=c.get) for t, c in gold_types.items() if c}

    print(f"  [ORACLE] NER-type-guided clustering with "
          f"{len(set(entity_types.values()))} gold types")

    type_groups = defaultdict(list)
    untyped = []
    for idx in range(X.shape[0]):
        term = inv_nodemap[idx]
        if term in entity_types:
            type_groups[entity_types[term]].append((idx, term))
        else:
            untyped.append((idx, term))
    if untyped:
        type_groups['Untyped'] = untyped

    cluster_list = {}
    cluster_names = {}
    cid = 0
    for ner_type, members in sorted(type_groups.items()):
        if len(members) <= max_subcluster_size:
            groups = [[term for _, term in members]]
        else:
            sub_X = X[[idx for idx, _ in members]]
            n_sub = max(2, len(members) // 12)
            sub_labels = AgglomerativeClustering(
                n_clusters=n_sub, metric='cosine', linkage='average'
            ).fit_predict(sub_X)
            sub_groups = defaultdict(list)
            for i, label in enumerate(sub_labels):
                sub_groups[label].append(members[i][1])
            groups = [sub_groups[k] for k in sorted(sub_groups)]
        for terms in groups:
            cluster_list[cid] = sorted(terms, key=lambda t: (len(t), t))
            cluster_names[cid] = ner_type
            cid += 1

    print(f"  NER-type clustering: {cid} clusters from {len(type_groups)} types")
    return cluster_list, cluster_names


def cluster_graph(key, use_llm=False):
    """Cluster graph nodes into term groups."""
    X, inv_nodemap, nodemap = load_graph(key)
    if X is None:
        return None, None

    print(f"Processing {key}: {X.shape[0]} nodes, dim={X.shape[1]}")

    cluster_list = cluster_names = None
    if NER_TYPE_CLUSTERING and key == 'fine_auto':
        if not ORACLE_TYPES:
            print("  WARNING: NER_TYPE_CLUSTERING uses gold types; results are "
                  "an oracle upper bound")
        cluster_list, cluster_names = ner_type_guided_clustering(X, inv_nodemap)

    if cluster_list is None:
        n_clusters = optimal_nclusters(X)
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, metric='cosine', linkage='average'
        ).fit_predict(X)
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[int(label)].append(inv_nodemap[idx])
        cluster_list = {cid: sorted(terms, key=lambda t: (len(t), t))
                        for cid, terms in sorted(clusters.items())}

        # Cosmetic names: the most frequent term of each cluster
        corpus_freq = {}
        if os.path.exists(CORPUS_FREQ_PATH):
            with open(CORPUS_FREQ_PATH) as f:
                corpus_freq = json.load(f)
        cluster_names = {
            cid: max(terms, key=lambda t: (corpus_freq.get(t, 0), -len(t)))
            for cid, terms in cluster_list.items()
        }

    # LLM naming override (optional - Gemini)
    if use_llm:
        llm_names = llm_name_clusters(cluster_list)
        cluster_names.update(llm_names)
        print(f"  LLM named {len(llm_names)} clusters")

    # LLM naming (llm_validator) for top-20 largest clusters (when LLM_MODE is enabled)
    if LLM_MODE:
        try:
            from llm_validator import name_cluster
            top_20 = sorted(cluster_list.items(), key=lambda x: -len(x[1]))[:20]
            for cid, terms in top_20:
                try:
                    cluster_names[cid] = name_cluster(terms)
                except Exception as e:
                    print(f"    LLM naming failed for cluster {cid}: {e}")
            print(f"  LLM named up to {len(top_20)} clusters")
        except ImportError as e:
            print(f"  LLM naming unavailable: {e}")

    import pandas as pd
    cluster_df = pd.DataFrame({
        'cluster_id': list(cluster_list.keys()),
        'cluster_name': [cluster_names.get(cid, f"Cluster_{cid}")
                         for cid in cluster_list.keys()],
        'n_terms': [len(terms) for terms in cluster_list.values()],
        'top_terms': [', '.join(terms[:5]) for terms in cluster_list.values()]
    }).sort_values(['n_terms', 'cluster_id'], ascending=[False, True])

    return cluster_df, cluster_list


# ============== Gemini LLM Integration ==============
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
    import pandas as pd

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    use_llm = os.getenv('USE_LLM_CLUSTERING', '').lower() in ('1', 'true', 'yes')

    results = {}
    for key in ['fine_auto', 'coarse']:
        df, clusters = cluster_graph(key, use_llm=use_llm)
        if df is not None:
            df.to_csv(os.path.join(PROCESSED_DIR, f'{key}_clusters.csv'),
                      index=False)
            with open(os.path.join(PROCESSED_DIR, f'{key}_clusters.json'),
                      'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in clusters.items()}, f,
                          indent=2, ensure_ascii=False)
            results[key] = df.head(10)
            sizes = Counter(len(v) for v in clusters.values())
            print(f"Saved {key} clusters: {len(clusters)} groups "
                  f"({sizes.get(1, 0)} singletons)")

    if results:
        summary = pd.concat([df.assign(graph=key) for key, df in results.items()])
        summary.to_csv(os.path.join(PROCESSED_DIR, 'all_clusters_summary.csv'),
                       index=False)
        print(summary)
    else:
        print("No graphs found. Run: python src/graph/builder.py")
