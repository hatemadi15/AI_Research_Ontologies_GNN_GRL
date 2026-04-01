"""
taxonomy.py - Taxonomy Construction with Real Similarity & Hearst Patterns

Builds a directed taxonomy graph using:
  - Real pairwise cosine similarity between parent/child embeddings
  - Actual corpus frequency (from corpus_frequencies.json)
  - Enabled Hearst pattern extraction with correct hypo/hypernym direction
  - Sentences loaded from all_sentences.txt
"""

import os
import json
import re
import pickle
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from config import get_config

# Get project root directory (2 levels up from this script)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

CLUSTERS_PATH = os.path.join(PROCESSED_DIR, 'fine_auto_clusters.json')
GRAPH_PATH = os.path.join(PROCESSED_DIR, 'graph_fine_auto.pt')
NODEMAP_PATH = os.path.join(PROCESSED_DIR, 'nodemap_fine_auto.pt')
SENTENCES_PATH = os.path.join(PROCESSED_DIR, 'all_sentences.txt')
CORPUS_FREQ_PATH = os.path.join(PROCESSED_DIR, 'corpus_frequencies.json')
GNN_EMBED_PATH = os.path.join(PROCESSED_DIR, 'gnn_embeddings.npy')
GNN_MAP_PATH = os.path.join(PROCESSED_DIR, 'gnn_embed_map.json')

_config = get_config()
EMBEDDER_MODEL = os.environ.get('EMBEDDER_MODEL', _config.get('embedder_model', 'all-MiniLM-L6-v2'))
embedder = SentenceTransformer(EMBEDDER_MODEL)

# Hearst patterns: (hyponym, hypernym) extraction
# "X such as Y" => Y is-a X (Y=hyponym, X=hypernym)
# "X and other Y" => X is-a Y (X=hyponym, Y=hypernym)
# "X is a Y" => X is-a Y (X=hyponym, Y=hypernym)
HEARST_PATTERNS = [
    # "NP such as NP" -> second is hyponym of first
    (r'(\b[\w\s]+?)\s+such\s+as\s+([\w\s]+?)(?:\s*,|\s*and|\s*\.)', 'hyper', 'hypo'),
    (r'(\b[\w\s]+?)\s+(?:including|especially)\s+([\w\s]+?)(?:\s*,|\s*and|\s*\.)', 'hyper', 'hypo'),
    # "NP and other NP" -> first is hyponym of second
    (r'(\b[\w\s]+?)\s+and\s+other\s+([\w\s]+?)(?:\s*,|\s*\.)', 'hypo', 'hyper'),
    # "NP is a NP" -> first is hyponym of second
    (r'(\b[\w\s]+?)\s+(?:is\s+a|is\s+an|are)\s+([\w\s]+?)(?:\s*,|\s*\.|\s*that)', 'hypo', 'hyper'),
]


def load_clusters():
    with open(CLUSTERS_PATH) as f:
        return json.load(f)


def load_sentences():
    """Load sentences from all_sentences.txt (saved by builder.py)."""
    if not os.path.exists(SENTENCES_PATH):
        print(f"Warning: {SENTENCES_PATH} not found, Hearst patterns disabled")
        return []
    with open(SENTENCES_PATH, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(sentences)} sentences for Hearst extraction")
    return sentences


def load_corpus_frequencies():
    """Load actual corpus frequencies from corpus_frequencies.json."""
    if os.path.exists(CORPUS_FREQ_PATH):
        with open(CORPUS_FREQ_PATH) as f:
            return json.load(f)
    return {}


def get_term_embeddings(terms):
    """Get embeddings for terms, using GNN if available, else SBERT."""
    if os.path.exists(GNN_EMBED_PATH) and os.path.exists(GNN_MAP_PATH):
        gnn_embeds = np.load(GNN_EMBED_PATH)
        with open(GNN_MAP_PATH) as f:
            gnn_map = json.load(f)

        embeds = {}
        for t in terms:
            if t in gnn_map:
                idx = gnn_map[t]
                if idx < gnn_embeds.shape[0]:
                    embeds[t] = gnn_embeds[idx]
        if embeds:
            print(f"  Using GNN embeddings for {len(embeds)}/{len(terms)} terms")
            # Fill missing with SBERT
            missing = [t for t in terms if t not in embeds]
            if missing:
                sbert = embedder.encode(missing)
                for i, t in enumerate(missing):
                    embeds[t] = sbert[i]
            return embeds

    # Fallback: SBERT
    sbert = embedder.encode(terms)
    return {t: sbert[i] for i, t in enumerate(terms)}


def distributional_hierarchy(clusters, corpus_freq):
    """Build hierarchy: rarer terms are children of more frequent (general) terms.

    Uses actual corpus frequency and real pairwise cosine similarity.
    """
    edges = []
    all_terms = set(t for terms in clusters.values() for t in terms)

    # Get embeddings
    term_embeds = get_term_embeddings(list(all_terms))

    for cluster_id, terms in clusters.items():
        if len(terms) < 2:
            continue

        # Sort by corpus frequency (rare -> frequent)
        freqs = [(t, corpus_freq.get(t, 1)) for t in terms]
        freqs.sort(key=lambda x: x[1])

        for i in range(1, len(freqs)):
            child = freqs[i - 1][0]  # rarer = more specific
            parent = freqs[i][0]      # more frequent = more general

            # Compute REAL cosine similarity between child and parent
            if child in term_embeds and parent in term_embeds:
                child_vec = torch.tensor(term_embeds[child]).unsqueeze(0)
                parent_vec = torch.tensor(term_embeds[parent]).unsqueeze(0)
                sim = util.cos_sim(child_vec, parent_vec).item()
            else:
                sim = 0.0

            if sim > 0.1:  # Minimum similarity threshold
                edges.append((child, parent, {
                    'type': 'distrib',
                    'score': round(sim, 4),
                    'child_freq': corpus_freq.get(child, 1),
                    'parent_freq': corpus_freq.get(parent, 1),
                }))

    return edges


def hearst_patterns(clusters, sentences):
    """Extract hypernym-hyponym pairs from corpus using Hearst patterns."""
    all_terms = set(t for terms in clusters.values() for t in terms)
    # Build a lookup for fast matching
    term_set_lower = {t.lower(): t for t in all_terms}

    edges = []
    for sent in sentences:
        sent_lower = sent.lower()
        for pattern, role1, role2 in HEARST_PATTERNS:
            for m in re.finditer(pattern, sent_lower):
                g1 = m.group(1).strip().lower()
                g2 = m.group(2).strip().lower()

                # Match to known terms
                t1 = term_set_lower.get(g1)
                t2 = term_set_lower.get(g2)

                if t1 and t2 and t1 != t2:
                    # Assign hypo/hyper based on pattern roles
                    if role1 == 'hypo':
                        child, parent = t1, t2
                    else:
                        child, parent = t2, t1

                    edges.append((child, parent, {
                        'type': 'hearst',
                        'score': 1.0,
                        'pattern': pattern[:30],
                    }))

    # Deduplicate
    seen = set()
    unique_edges = []
    for child, parent, attrs in edges:
        key = (child, parent)
        if key not in seen:
            seen.add(key)
            unique_edges.append((child, parent, attrs))

    print(f"  Hearst patterns found: {len(unique_edges)} unique edges")
    return unique_edges


def build_taxonomy():
    """Build taxonomy graph from distributional hierarchy + Hearst patterns."""
    clusters = load_clusters()
    corpus_freq = load_corpus_frequencies()
    sentences = load_sentences()

    print(f"Building taxonomy from {len(clusters)} clusters...")

    # Distributional hierarchy
    distrib_edges = distributional_hierarchy(clusters, corpus_freq)
    print(f"  Distributional edges: {len(distrib_edges)}")

    # Hearst pattern edges
    hearst_edges = hearst_patterns(clusters, sentences)

    # Combine edges (Hearst gets priority for conflicts)
    G = nx.DiGraph()
    for child, parent, attrs in distrib_edges:
        G.add_edge(child, parent, **attrs)
    for child, parent, attrs in hearst_edges:
        G.add_edge(child, parent, **attrs)

    print(f"  Combined graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    # Break cycles before transitive reduction
    if not nx.is_directed_acyclic_graph(G):
        # Remove edges with lowest similarity to break cycles
        cycles = list(nx.simple_cycles(G))
        print(f"  Breaking {len(cycles)} cycles...")
        for cycle in cycles:
            if len(cycle) < 2:
                continue
            # Find weakest edge in cycle
            min_score = float('inf')
            min_edge = None
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                if G.has_edge(u, v):
                    score = G[u][v].get('score', 0)
                    if score < min_score:
                        min_score = score
                        min_edge = (u, v)
            if min_edge and G.has_edge(*min_edge):
                G.remove_edge(*min_edge)

    # Transitive reduction
    if nx.is_directed_acyclic_graph(G):
        G = nx.transitive_reduction(G)
    else:
        print("  Warning: graph still has cycles, skipping transitive reduction")

    # Save taxonomy graph
    with open(os.path.join(PROCESSED_DIR, 'taxonomy_graph.pkl'), 'wb') as f:
        pickle.dump(G, f)

    edges_df = pd.DataFrame(
        [(u, v, d.get('type', 'distrib'), d.get('score', 1.0))
         for u, v, d in G.edges(data=True)],
        columns=['child', 'parent', 'type', 'score']
    )
    edges_df.to_csv(os.path.join(PROCESSED_DIR, 'taxonomy_edges.csv'),
                    index=False)
    print(f"Taxonomy: {G.number_of_nodes()} nodes, {G.number_of_edges()} isa edges")


if __name__ == "__main__":
    build_taxonomy()
