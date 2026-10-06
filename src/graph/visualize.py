"""
visualize.py - Full Knowledge Graph Visualization

Renders the knowledge graph with three views:
  1. Taxonomy (is-a hierarchy): tree/DAG colored by NER type, sized by frequency
  2. Relations: top-50 high-confidence non-taxonomic relations, colored by type
  3. Combined KG: taxonomy (solid) + relations (dashed) in one graph

Nodes are coloured by their predicted ontology class (predicted_types.json);
the palette is built from the classes that occur.

Outputs saved to data/processed/:
  - taxonomy_graph.png
  - relations_graph.png
  - knowledge_graph.png
"""

import os
import json
import pickle

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import PREDICTED_TYPES_FILE, PROCESSED_DIR

TAXONOMY_PATH = os.path.join(PROCESSED_DIR, 'taxonomy_graph.pkl')
RELATIONS_PATH = os.path.join(PROCESSED_DIR, 'relations.csv')
PREDICTED_TYPES_PATH = os.path.join(PROCESSED_DIR, PREDICTED_TYPES_FILE)
CORPUS_FREQ_PATH = os.path.join(PROCESSED_DIR, 'corpus_frequencies.json')

# Colours for the most frequent predicted classes; the rest are grey
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#7f7f7f']
DEFAULT_NODE_COLOR = '#aaaaaa'

# Relation type -> color mapping
REL_COLORS = {
    'causeOf': '#e41a1c',
    'necessaryCauseOf': '#b2182b',
    'sufficientCauseOf': '#d6604d',
    'hasProperty': '#4daf4a',
    'isPartOf': '#377eb8',
    'precedes': '#ff7f00',
    'growsInto': '#984ea3',
    'initiatesAt': '#a65628',
    'measures': '#f781bf',
    'associatedWith': '#999999',
    'correlatedWith': '#66c2a5',
    'constrains': '#fc8d62',
    'softConstrains': '#8da0cb',
    'hardConstrains': '#e78ac3',
    'reliesOn': '#a6d854',
    'parallelTo': '#ffd92f',
    'spatiallyCoincidesWith': '#e5c494',
    'alignedWith': '#b3b3b3',
}
DEFAULT_EDGE_COLOR = '#888888'


def _load_entity_types():
    """Term -> predicted class label (the pipeline's typing decision)."""
    if os.path.exists(PREDICTED_TYPES_PATH):
        with open(PREDICTED_TYPES_PATH, 'r', encoding='utf-8') as f:
            return {t: p.get('label', '') for t, p in json.load(f).items()}
    return {}


def _type_colors(nodes, entity_types):
    """Palette for the most frequent types among the drawn nodes."""
    counts = {}
    for n in nodes:
        t = entity_types.get(n, '')
        if t:
            counts[t] = counts.get(t, 0) + 1
    top = sorted(counts, key=lambda t: (-counts[t], t))[:len(PALETTE)]
    return {t: PALETTE[i] for i, t in enumerate(top)}


def _load_corpus_frequencies():
    if os.path.exists(CORPUS_FREQ_PATH):
        with open(CORPUS_FREQ_PATH, 'r') as f:
            return json.load(f)
    return {}


def _node_color(node, entity_types, colors):
    return colors.get(entity_types.get(node, ''), DEFAULT_NODE_COLOR)


def _node_size(node, corpus_freq, min_size=200, max_size=2000):
    freq = corpus_freq.get(node, 1)
    log_freq = np.log1p(freq)
    max_log = np.log1p(max(corpus_freq.values())) if corpus_freq else 1
    normalized = log_freq / max_log if max_log > 0 else 0.5
    return min_size + normalized * (max_size - min_size)


def plot_taxonomy():
    if not os.path.exists(TAXONOMY_PATH):
        print(f"Taxonomy not found: {TAXONOMY_PATH}. Run taxonomy.py first.")
        return

    with open(TAXONOMY_PATH, 'rb') as f:
        G = pickle.load(f)

    if G.number_of_nodes() == 0:
        print("Taxonomy graph is empty.")
        return

    entity_types = _load_entity_types()
    corpus_freq = _load_corpus_frequencies()

    print(f"Taxonomy: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    pos = nx.spring_layout(G, k=2, seed=42, iterations=50)

    colors = _type_colors(G.nodes(), entity_types)
    node_colors = [_node_color(n, entity_types, colors) for n in G.nodes()]
    node_sizes = [_node_size(n, corpus_freq) for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#555555', alpha=0.7,
                           arrows=True, arrowsize=10, width=1.0, style='solid')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')

    present_types = set()
    for n in G.nodes():
        t = entity_types.get(n, '')
        if t:
            present_types.add(t)
    legend_patches = [
        mpatches.Patch(color=colors[t], label=t)
        for t in sorted(present_types) if t in colors
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper left', fontsize=9)

    ax.set_title("Taxonomy Graph (is-a Hierarchy)", fontsize=16)
    ax.axis('off')

    output_path = os.path.join(PROCESSED_DIR, 'taxonomy_graph.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_relations():
    if not os.path.exists(RELATIONS_PATH):
        print(f"Relations not found: {RELATIONS_PATH}. Run relations.py first.")
        return

    df = pd.read_csv(RELATIONS_PATH)
    df_non_tax = df[df['rel'] != 'isPartOf'].copy()
    if 'sim' in df_non_tax.columns:
        df_non_tax = df_non_tax.dropna(subset=['sim'])
        df_non_tax = df_non_tax.sort_values('sim', ascending=False)
    df_top = df_non_tax.head(50)

    if len(df_top) == 0:
        print("No non-taxonomic relations found.")
        return

    print(f"Relations: plotting top {len(df_top)} edges")

    entity_types = _load_entity_types()
    corpus_freq = _load_corpus_frequencies()

    G = nx.DiGraph()
    edge_colors = []
    for _, row in df_top.iterrows():
        G.add_edge(row['subj'], row['obj'], rel=row['rel'])
        edge_colors.append(REL_COLORS.get(row['rel'], DEFAULT_EDGE_COLOR))

    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    pos = nx.spring_layout(G, k=2, seed=42, iterations=50)

    colors = _type_colors(G.nodes(), entity_types)
    node_colors = [_node_color(n, entity_types, colors) for n in G.nodes()]
    node_sizes = [_node_size(n, corpus_freq) for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, alpha=0.7,
                           arrows=True, arrowsize=12, width=1.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    edge_labels = nx.get_edge_attributes(G, 'rel')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=6, font_color='#333333')

    present_rels = set(df_top['rel'].unique())
    legend_patches = [
        mpatches.Patch(color=REL_COLORS.get(r, DEFAULT_EDGE_COLOR), label=r)
        for r in sorted(present_rels)
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper left', fontsize=8)

    ax.set_title("Top-50 Non-Taxonomic Relations", fontsize=16)
    ax.axis('off')

    output_path = os.path.join(PROCESSED_DIR, 'relations_graph.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_knowledge_graph():
    entity_types = _load_entity_types()
    corpus_freq = _load_corpus_frequencies()

    G = nx.DiGraph()
    edge_styles = {}
    edge_colors_map = {}

    if os.path.exists(TAXONOMY_PATH):
        with open(TAXONOMY_PATH, 'rb') as f:
            tax_G = pickle.load(f)
        for u, v in tax_G.edges():
            G.add_edge(u, v, rel='is-a')
            edge_styles[(u, v)] = 'solid'
            edge_colors_map[(u, v)] = '#555555'
        print(f"Taxonomy: {tax_G.number_of_edges()} edges")

    if os.path.exists(RELATIONS_PATH):
        df = pd.read_csv(RELATIONS_PATH)
        if 'sim' in df.columns:
            df_sorted = df.dropna(subset=['sim']).sort_values('sim', ascending=False)
        else:
            df_sorted = df
        df_top = df_sorted.head(50)
        for _, row in df_top.iterrows():
            edge = (row['subj'], row['obj'])
            if edge not in edge_styles:
                G.add_edge(row['subj'], row['obj'], rel=row['rel'])
                edge_styles[edge] = 'dashed'
                edge_colors_map[edge] = REL_COLORS.get(row['rel'], DEFAULT_EDGE_COLOR)
        print(f"Relations: {len(df_top)} edges added")

    if G.number_of_nodes() == 0:
        print("No data to plot. Run taxonomy.py and relations.py first.")
        return

    print(f"Combined KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    pos = nx.spring_layout(G, k=2, seed=42, iterations=50)

    colors = _type_colors(G.nodes(), entity_types)
    node_colors = [_node_color(n, entity_types, colors) for n in G.nodes()]
    node_sizes = [_node_size(n, corpus_freq) for n in G.nodes()]

    solid_edges = [e for e in G.edges() if edge_styles.get(e) == 'solid']
    solid_colors = [edge_colors_map.get(e, '#555555') for e in solid_edges]
    if solid_edges:
        nx.draw_networkx_edges(G, pos, edgelist=solid_edges, ax=ax,
                               edge_color=solid_colors, alpha=0.7,
                               arrows=True, arrowsize=10, width=1.0, style='solid')

    dashed_edges = [e for e in G.edges() if edge_styles.get(e) == 'dashed']
    dashed_colors = [edge_colors_map.get(e, DEFAULT_EDGE_COLOR) for e in dashed_edges]
    if dashed_edges:
        nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, ax=ax,
                               edge_color=dashed_colors, alpha=0.7,
                               arrows=True, arrowsize=10, width=1.2, style='dashed')

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    legend_patches = [mpatches.Patch(color='#555555', label='is-a (taxonomy)')]
    present_rels = set()
    if os.path.exists(RELATIONS_PATH):
        df_check = pd.read_csv(RELATIONS_PATH)
        if 'sim' in df_check.columns:
            df_check = df_check.dropna(subset=['sim']).sort_values('sim', ascending=False)
        for _, row in df_check.head(50).iterrows():
            present_rels.add(row['rel'])
    for r in sorted(present_rels):
        legend_patches.append(
            mpatches.Patch(color=REL_COLORS.get(r, DEFAULT_EDGE_COLOR), label=r))

    present_types = set()
    for n in G.nodes():
        t = entity_types.get(n, '')
        if t:
            present_types.add(t)
    for t in sorted(present_types):
        if t in colors:
            legend_patches.append(
                mpatches.Patch(color=colors[t], label=f"Node: {t}"))

    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper left', fontsize=7, ncol=2)

    ax.set_title("Combined Knowledge Graph (Taxonomy + Relations)", fontsize=16)
    ax.axis('off')

    output_path = os.path.join(PROCESSED_DIR, 'knowledge_graph.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("=== Taxonomy Visualization ===")
    plot_taxonomy()
    print("\n=== Relations Visualization ===")
    plot_relations()
    print("\n=== Combined Knowledge Graph ===")
    plot_knowledge_graph()
