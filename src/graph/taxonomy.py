"""
taxonomy.py - Taxonomy Construction with Embedding Similarity & Hearst Patterns

Builds a directed is-a graph (child -> parent) over the discovered terms:
  - Distributional hierarchy: inside each cluster, every term is attached to
    the most similar *more general* term of the cluster (higher corpus
    frequency; fewer words breaks ties), which yields a tree per cluster. The
    previous version chained the terms of a cluster in frequency order, so
    each cluster became a single path.
  - Hearst patterns (USE_HEARST_PATTERNS): known terms are located in the
    sentences with a word-boundary matcher, and the text *between* two
    consecutive mentions is matched against cue phrases ("X such as Y",
    "Y and other X", "Y is a X", ...). The old regexes captured the whole
    sentence prefix before the cue, so they almost never matched a term.
Embeddings: GNN embeddings when available and USE_GNN_EMBEDDINGS is on,
otherwise the sentence-transformer.
"""

import os
import re
import json
import pickle

import networkx as nx
import numpy as np
import pandas as pd

from config import EMBEDDER_MODEL, PROCESSED_DIR, USE_GNN_EMBEDDINGS, USE_HEARST_PATTERNS
from text_match import TermMatcher, normalize_text

CLUSTERS_PATH = os.path.join(PROCESSED_DIR, 'fine_auto_clusters.json')
SENTENCES_PATH = os.path.join(PROCESSED_DIR, 'all_sentences.txt')
CORPUS_FREQ_PATH = os.path.join(PROCESSED_DIR, 'corpus_frequencies.json')
GNN_EMBED_PATH = os.path.join(PROCESSED_DIR, 'gnn_embeddings.npy')
GNN_MAP_PATH = os.path.join(PROCESSED_DIR, 'gnn_embed_map.json')

# Minimum cosine similarity for a distributional is-a edge
MIN_DISTRIB_SIM = 0.1

# Cue phrases between two consecutive term mentions A ... B.
# role 'hyper_first': A is the hypernym (A such as B); 'hypo_first': B is.
HEARST_CUES = [
    (re.compile(r',?\s*(?:such as|including|especially|e\.g\.,?|like)\s*'), 'hyper_first'),
    (re.compile(r',?\s*(?:and|or) other\s*'), 'hypo_first'),
    (re.compile(r'\s*(?:is|are) (?:a|an|one)?\s*(?:type|kind|form|class)s? of\s*'), 'hypo_first'),
    (re.compile(r'\s*is (?:a|an)\s*'), 'hypo_first'),
]
# Separators that continue an enumeration after "such as" ("X such as A, B and C")
LIST_SEPARATOR = re.compile(r'\s*(?:,|and|or|, and|, or)\s*')
# A matched term must end its noun phrase: "crack tip, such as dislocation
# density" must not yield dislocation -> tip. The term is accepted only when
# it is followed by punctuation, the end of the sentence or a function word.
PHRASE_END = re.compile(
    r'\s*(?:$|[,.;:)\]]|(?:and|or|which|that|are|is|were|was|have|has|in|at|'
    r'on|with|for|to|by|from|under|during|as)\b)')

_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(EMBEDDER_MODEL)
    return _EMBEDDER


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
    """L2-normalised embeddings: GNN if enabled and available, else SBERT."""
    vectors = None
    if USE_GNN_EMBEDDINGS and os.path.exists(GNN_EMBED_PATH) and os.path.exists(GNN_MAP_PATH):
        gnn_embeds = np.load(GNN_EMBED_PATH)
        with open(GNN_MAP_PATH) as f:
            gnn_map = json.load(f)
        if all(t in gnn_map for t in terms):
            print(f"  Using GNN embeddings for {len(terms)} terms")
            vectors = np.stack([gnn_embeds[gnn_map[t]] for t in terms])
    if vectors is None:
        print(f"  Using SBERT embeddings for {len(terms)} terms")
        vectors = _get_embedder().encode(terms, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return {t: v for t, v in zip(terms, vectors / norms)}


def generality(term, corpus_freq):
    """Sort key: more frequent and shorter terms are more general."""
    return (corpus_freq.get(term, 1), -len(term.split()), -len(term), term)


def distributional_hierarchy(clusters, corpus_freq):
    """Attach each term to its most similar more-general term in its cluster."""
    all_terms = sorted(set(t for terms in clusters.values() for t in terms))
    term_embeds = get_term_embeddings(all_terms)
    edges = []
    for cluster_id in sorted(clusters, key=lambda c: int(c)):
        terms = clusters[cluster_id]
        if len(terms) < 2:
            continue
        for child in terms:
            child_key = generality(child, corpus_freq)
            parents = [p for p in terms if generality(p, corpus_freq) > child_key]
            if not parents:
                continue  # most general term of the cluster: a root
            sims = {p: float(term_embeds[child] @ term_embeds[p]) for p in parents}
            parent = max(parents, key=lambda p: (sims[p], p))
            if sims[parent] > MIN_DISTRIB_SIM:
                edges.append((child, parent, {
                    'type': 'distrib',
                    'score': round(sims[parent], 4),
                    'child_freq': corpus_freq.get(child, 1),
                    'parent_freq': corpus_freq.get(parent, 1),
                }))
    return edges


def hearst_patterns(clusters, sentences):
    """Extract (hyponym, hypernym) pairs between known terms."""
    all_terms = sorted(set(t for terms in clusters.values() for t in terms))
    matcher = TermMatcher(all_terms)
    found = {}
    for sent in sentences:
        text = normalize_text(sent)
        mentions = matcher.find(text)
        for i in range(len(mentions) - 1):
            a_start, a_end, a = mentions[i]
            b_start, b_end, b = mentions[i + 1]
            between = text[a_end:b_start]
            for cue, role in HEARST_CUES:
                if not cue.fullmatch(between):
                    continue
                if not PHRASE_END.match(text, b_end):
                    break  # b is only the start of a longer noun phrase
                if role == 'hyper_first':
                    # "a such as b, c and d": every listed item is a hyponym of a
                    pairs = [(b, a)]
                    prev_end = b_end
                    for c_start, c_end, c in mentions[i + 2:]:
                        if not LIST_SEPARATOR.fullmatch(text[prev_end:c_start]):
                            break
                        if not PHRASE_END.match(text, c_end):
                            break
                        pairs.append((c, a))
                        prev_end = c_end
                else:
                    pairs = [(a, b)]
                for child, parent in pairs:
                    if child != parent:
                        found.setdefault((child, parent), cue.pattern)
                break

    edges = [(c, p, {'type': 'hearst', 'score': 1.0, 'pattern': pat})
             for (c, p), pat in sorted(found.items())]
    print(f"  Hearst patterns found: {len(edges)} unique edges")
    return edges


def break_cycles(G):
    """Remove the weakest edge of each remaining cycle until G is a DAG."""
    removed = 0
    while True:
        try:
            cycle = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            break
        u, v = min(cycle, key=lambda e: (G[e[0]][e[1]].get('score', 0), e[0], e[1]))[:2]
        G.remove_edge(u, v)
        removed += 1
    if removed:
        print(f"  Broke {removed} cycle edges")
    return G


def build_taxonomy():
    """Build taxonomy graph from distributional hierarchy + Hearst patterns."""
    clusters = load_clusters()
    corpus_freq = load_corpus_frequencies()

    print(f"Building taxonomy from {len(clusters)} clusters...")
    distrib_edges = distributional_hierarchy(clusters, corpus_freq)
    print(f"  Distributional edges: {len(distrib_edges)}")

    hearst_edges = []
    if USE_HEARST_PATTERNS:
        hearst_edges = hearst_patterns(clusters, load_sentences())
    else:
        print("  Hearst patterns disabled (USE_HEARST_PATTERNS=false)")

    # Hearst edges take priority over distributional ones for the same pair
    G = nx.DiGraph()
    for child, parent, attrs in distrib_edges + hearst_edges:
        G.add_edge(child, parent, **attrs)
    print(f"  Combined graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    G = break_cycles(G)
    reduced = nx.transitive_reduction(G)
    reduced.add_nodes_from(G.nodes(data=True))
    reduced.add_edges_from((u, v, G.edges[u, v]) for u, v in reduced.edges)
    G = reduced

    with open(os.path.join(PROCESSED_DIR, 'taxonomy_graph.pkl'), 'wb') as f:
        pickle.dump(G, f)

    edges_df = pd.DataFrame(
        [(u, v, d.get('type', 'distrib'), d.get('score', 1.0))
         for u, v, d in sorted(G.edges(data=True))],
        columns=['child', 'parent', 'type', 'score']
    )
    edges_df.to_csv(os.path.join(PROCESSED_DIR, 'taxonomy_edges.csv'),
                    index=False)
    print(f"Taxonomy: {G.number_of_nodes()} nodes, {G.number_of_edges()} isa edges "
          f"({(edges_df['type'] == 'hearst').sum()} from Hearst patterns)")


if __name__ == "__main__":
    build_taxonomy()
