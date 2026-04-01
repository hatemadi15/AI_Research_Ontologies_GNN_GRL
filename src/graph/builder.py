"""
builder.py - CoNLL BIO Tag Parser & Co-occurrence Graph Builder

Parses CoNLL files with BIO-tagged NER entities, builds a co-occurrence
graph with PMI-weighted edges, and produces typed node embeddings.

Outputs:
  - graph_fine_auto.pt: PyG Data with node features and PMI edges
  - nodemap_fine_auto.pt: term -> node_id mapping
  - entity_types.json: term -> NER type mapping
  - corpus_frequencies.json: term -> document frequency
  - all_sentences.txt: reconstructed sentences for downstream use
"""

import os
import json
import math
from itertools import combinations
from collections import Counter, defaultdict

import networkx as nx
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.utils import from_networkx

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def parse_conll_bio(filepath):
    """Parse a CoNLL file extracting BIO-tagged entities with their types.

    Returns:
        sentences: list of reconstructed sentence strings
        doc_entities: list of (entity_text, entity_type) per sentence
    """
    sentences = []
    doc_entities = []
    current_tokens = []
    current_entities = []

    # State for multi-token entity assembly
    ent_tokens = []
    ent_type = None

    def flush_entity():
        nonlocal ent_tokens, ent_type
        if ent_tokens and ent_type:
            entity_text = ' '.join(ent_tokens).lower().strip()
            if len(entity_text) >= 2:
                current_entities.append((entity_text, ent_type))
        ent_tokens = []
        ent_type = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                # Sentence boundary
                flush_entity()
                if current_tokens:
                    sentences.append(' '.join(current_tokens))
                    doc_entities.append(list(current_entities))
                    current_tokens = []
                    current_entities = []
                continue

            if line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            token = parts[0]
            tag = parts[-1]  # BIO tag is last column
            current_tokens.append(token)

            if tag.startswith('B-'):
                flush_entity()
                ent_type = tag[2:]
                ent_tokens = [token]
            elif tag.startswith('I-') and ent_type:
                tag_type = tag[2:]
                if tag_type == ent_type:
                    ent_tokens.append(token)
                else:
                    flush_entity()
            else:
                flush_entity()

    # Handle last sentence if file doesn't end with blank line
    flush_entity()
    if current_tokens:
        sentences.append(' '.join(current_tokens))
        doc_entities.append(list(current_entities))

    return sentences, doc_entities


def compute_pmi(cooccurrence_counts, term_doc_freq, total_docs):
    """Compute Pointwise Mutual Information for co-occurring term pairs."""
    pmi_edges = {}
    for (t1, t2), co_count in cooccurrence_counts.items():
        p_xy = co_count / total_docs
        p_x = term_doc_freq[t1] / total_docs
        p_y = term_doc_freq[t2] / total_docs
        denom = p_x * p_y
        if denom > 0 and p_xy > 0:
            pmi = math.log2(p_xy / denom)
            if pmi > 0:  # Only positive PMI (terms co-occur more than chance)
                pmi_edges[(t1, t2)] = pmi
    return pmi_edges


def build_graph(datadir, mode='auto', ner_type='fine'):
    """Build co-occurrence graph from CoNLL BIO-tagged files.

    Args:
        datadir: Directory containing .conll files
        mode: Extraction mode ('auto' uses BIO tags)
        ner_type: Label for this graph variant

    Returns:
        pyg_data: PyG Data object with embeddings and edges
        nodemap: dict mapping term -> node index
        entity_types: dict mapping term -> NER type
        corpus_freq: dict mapping term -> document frequency
        all_sentences: list of all reconstructed sentences
    """
    files = sorted([f for f in os.listdir(datadir) if f.endswith('.conll')])
    print(f"Processing {ner_type} ({mode}): {len(files)} files...")

    all_sentences = []
    all_doc_entities = []
    entity_type_map = {}  # term -> NER type (first seen)

    for filename in files:
        filepath = os.path.join(datadir, filename)
        sentences, doc_entities = parse_conll_bio(filepath)
        all_sentences.extend(sentences)
        all_doc_entities.extend(doc_entities)

        # Track entity types
        for sent_ents in doc_entities:
            for ent_text, ent_type in sent_ents:
                if ent_text not in entity_type_map:
                    entity_type_map[ent_text] = ent_type

    print(f"  Parsed {len(all_sentences)} sentences, "
          f"{sum(len(e) for e in all_doc_entities)} entity mentions")

    # Count document frequency (how many sentences each term appears in)
    term_doc_freq = Counter()
    for sent_ents in all_doc_entities:
        unique_terms = set(ent_text for ent_text, _ in sent_ents)
        for term in unique_terms:
            term_doc_freq[term] += 1

    # Filter: keep terms appearing in >= 2 sentences
    valid_terms = {t for t, c in term_doc_freq.items() if c >= 2}
    print(f"  Valid terms (freq >= 2): {len(valid_terms)}")

    # Build co-occurrence counts
    cooccurrence = Counter()
    for sent_ents in all_doc_entities:
        unique_in_sent = list(set(
            ent_text for ent_text, _ in sent_ents if ent_text in valid_terms
        ))
        if len(unique_in_sent) >= 2:
            for t1, t2 in combinations(sorted(unique_in_sent), 2):
                cooccurrence[(t1, t2)] += 1

    # Compute PMI-weighted edges
    total_docs = len(all_doc_entities)
    pmi_edges = compute_pmi(cooccurrence, term_doc_freq, total_docs)
    print(f"  PMI edges (positive): {len(pmi_edges)}")

    # Build node list and map
    # Include all valid terms that appear in at least one PMI edge
    terms_in_edges = set()
    for (t1, t2) in pmi_edges:
        terms_in_edges.add(t1)
        terms_in_edges.add(t2)

    # Also include valid terms not in edges (isolated but frequent)
    all_valid = sorted(valid_terms)
    nodemap = {term: i for i, term in enumerate(all_valid)}

    # Compute embeddings
    print(f"  Encoding {len(all_valid)} terms...")
    node_embeddings = embedder.encode(all_valid, show_progress_bar=True)
    x = torch.tensor(node_embeddings, dtype=torch.float)

    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(len(all_valid)))
    for (t1, t2), pmi_val in pmi_edges.items():
        if t1 in nodemap and t2 in nodemap:
            G.add_edge(nodemap[t1], nodemap[t2], weight=pmi_val)

    print(f"  {ner_type}-{mode} graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    # Convert to PyG
    pyg_data = from_networkx(G)
    pyg_data.x = x

    # Filter entity types to valid terms only
    entity_types = {t: entity_type_map[t] for t in all_valid
                    if t in entity_type_map}
    corpus_freq = {t: term_doc_freq[t] for t in all_valid}

    return pyg_data, nodemap, entity_types, corpus_freq, all_sentences


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    base_raw = os.path.join(PROJECT_ROOT, "data", "raw", "dataset")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Fine-grained NER (primary)
    data_fine, map_fine, ent_types, corp_freq, sentences = build_graph(
        os.path.join(base_raw, "fine_grained_ner"), 'auto', 'fine'
    )
    torch.save(data_fine, os.path.join(processed_dir, "graph_fine_auto.pt"))
    torch.save(map_fine, os.path.join(processed_dir, "nodemap_fine_auto.pt"))

    # Save entity types
    with open(os.path.join(processed_dir, "entity_types.json"), 'w') as f:
        json.dump(ent_types, f, indent=2)
    print(f"Saved entity_types.json: {len(ent_types)} typed entities")

    # Save corpus frequencies
    with open(os.path.join(processed_dir, "corpus_frequencies.json"), 'w') as f:
        json.dump(corp_freq, f, indent=2)
    print(f"Saved corpus_frequencies.json: {len(corp_freq)} terms")

    # Save all sentences
    with open(os.path.join(processed_dir, "all_sentences.txt"), 'w',
              encoding='utf-8') as f:
        for s in sentences:
            f.write(s + '\n')
    print(f"Saved all_sentences.txt: {len(sentences)} sentences")

    # Coarse-grained (baseline comparison)
    coarse_dir = os.path.join(base_raw, "coarse_grained_ner")
    if os.path.exists(coarse_dir) and os.listdir(coarse_dir):
        data_coarse, map_coarse, _, _, _ = build_graph(
            coarse_dir, 'auto', 'coarse'
        )
        torch.save(data_coarse,
                    os.path.join(processed_dir, "graph_coarse.pt"))
        torch.save(map_coarse,
                    os.path.join(processed_dir, "nodemap_coarse.pt"))

    print("Done: graph built from BIO-tagged NER entities.")
