import os
import json
import re
import networkx as nx
import pandas as pd
from collections import defaultdict, Counter
import torch
from sentence_transformers import SentenceTransformer, util

# Get project root directory (2 levels up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

CLUSTERS_PATH = os.path.join(PROCESSED_DIR, 'fine_auto_clusters.json')
GRAPH_PATH = os.path.join(PROCESSED_DIR, 'graph_fine_auto.pt')
SENTENCES_PATH = os.path.join(PROCESSED_DIR, 'all_sentences.txt')  # From builder cache

embedder = SentenceTransformer('all-MiniLM-L6-v2')

HEARST_PATTERNS = [
    r'\b({hyponym})\s+(?:such as|like|including|especially)\s+({hypernym})\b',
    r'\b({hypernym})\s+(?:and|or)\s+other\s+({hyponym})\b',
    r'\b({hyponym})\s+(?:is\s+a|are)\s+({hypernym})\b',
    r'\b({hyponym})\s+(?:of\s+the|in\s+the)\s+({hypernym})\b'
]

def load_clusters():
    with open(CLUSTERS_PATH) as f:
        return json.load(f)

def distributional_hierarchy(clusters, X):
    """More specific (rarer) → more general (frequent) within clusters."""
    edges = []
    term_freq = Counter(t for c in clusters.values() for t in c)
    for terms in clusters.values():
        if len(terms) < 2: continue
        freqs = [(t, term_freq[t]) for t in terms]
        freqs.sort(key=lambda x: x[1])  # Rare → frequent
        for i in range(1, len(freqs)):
            child, parent = freqs[i-1][0], freqs[i][0]
            sim = util.cos_sim(X[0], X[0]).item()  # Placeholder
            edges.append((child, parent, {'type': 'distrib', 'score': 1.0}))
    return edges

def hearst_patterns(clusters, sentences):
    """Mine Hearst patterns from corpus."""
    all_terms = set(t for c in clusters.values() for t in c)
    edges = []
    for sent in sentences:
        for hyponym_tpl, hypernym_tpl in [('hyponym', 'hypernym')]:
            pattern = HEARST_PATTERNS[0].format(hyponym='|'.join(all_terms), 
                                               hypernym='|'.join(all_terms))
            matches = re.finditer(pattern, sent, re.I)
            for m in matches:
                edges.append((m.group(1).lower(), m.group(2).lower(), 
                             {'type': 'hearst', 'score': 1.0}))
    return edges

def build_taxonomy():
    clusters = load_clusters()
    data = torch.load(GRAPH_PATH, weights_only=False)
    X = data.x.cpu().numpy()
    
    distrib_edges = distributional_hierarchy(clusters, X)
    # hearst_edges = hearst_patterns(clusters, load_sentences())  # Add later
    
    G = nx.DiGraph()
    for child, parent, attrs in distrib_edges:
        G.add_edge(child, parent, **attrs)
    
    # Acyclic + transitive reduction
    G = nx.transitive_reduction(G)
    
    # Save taxonomy graph (using pickle since nx.write_gpickle is deprecated)
    import pickle
    with open(os.path.join(PROCESSED_DIR, 'taxonomy_graph.pkl'), 'wb') as f:
        pickle.dump(G, f)
    
    edges_df = pd.DataFrame([(u, v, d.get('type', 'distrib'), d.get('score', 1.0)) for u, v, d in G.edges(data=True)],
                           columns=['child', 'parent', 'type', 'score'])
    edges_df.to_csv(os.path.join(PROCESSED_DIR, 'taxonomy_edges.csv'), index=False)
    print(f"Taxonomy: {G.number_of_nodes()} nodes, {G.number_of_edges()} isa edges")

if __name__ == "__main__":
    build_taxonomy()
