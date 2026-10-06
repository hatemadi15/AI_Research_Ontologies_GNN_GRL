"""
builder.py - CoNLL BIO Tag Parser & Co-occurrence Graph Builder

Parses the CoNLL files with BIO-tagged NER entities, builds a co-occurrence
graph with PMI-weighted edges between terms, and encodes the terms with a
sentence-transformer as node features.

Nodes are the annotated entity texts occurring in >= 2 sentences
(conll.MIN_TERM_DF); they are the "discovered terms" of the pipeline and the
term universe of the evaluation.

Corpus augmentation (CORPUS_AUGMENT, default on): known terms are located in
the PubMed sentences with a word-boundary matcher, PMI is computed on PubMed
separately, and PubMed edges are added with weight AUGMENT_WEIGHT * PMI. Only
term pairs seen >= AUGMENT_MIN_PAIR_COUNT times are used, and very short or
purely numeric terms are not matched (they are mostly noise outside the
annotated papers). The node set is unchanged.

Outputs (in config.PROCESSED_DIR):
  - graph_fine_auto.pt / nodemap_fine_auto.pt: PyG graph and term -> node id
  - gold_term_types.json: term -> {NER type: mention count}. These are GOLD
    labels: only the evaluation and ORACLE_TYPES components may read them.
  - corpus_frequencies.json: term -> number of annotated sentences
  - conll_sentences.txt: the annotated sentences
  - all_sentences.txt: annotated sentences (+ PubMed sentences if augmented)
  - graph_coarse.pt / nodemap_coarse.pt: coarse-grained baseline graph
"""

import os
import re
import json
import math
from itertools import combinations
from collections import Counter

import conll
from config import (CORPUS_AUGMENT, CORPUS_AUGMENT_PATH, EMBEDDER_MODEL,
                    GOLD_TYPES_FILE, PROCESSED_DIR, PROJECT_ROOT, RAW_DIR)
from text_match import TermMatcher

# PubMed co-occurrence settings
AUGMENT_WEIGHT = 0.5
AUGMENT_MIN_PAIR_COUNT = 3
AUGMENT_MIN_TERM_LENGTH = 3

_EMBEDDER = None


def _get_embedder():
    """Load the sentence-transformer lazily (only when encoding)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {EMBEDDER_MODEL}")
        _EMBEDDER = SentenceTransformer(EMBEDDER_MODEL)
    return _EMBEDDER


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


def sentence_cooccurrence(sentence_terms, vocabulary):
    """Document frequencies and pair counts of terms per sentence."""
    term_df = Counter()
    pair_counts = Counter()
    for terms in sentence_terms:
        unique = sorted(set(t for t in terms if t in vocabulary))
        for t in unique:
            term_df[t] += 1
        for t1, t2 in combinations(unique, 2):
            pair_counts[(t1, t2)] += 1
    return term_df, pair_counts


def augment_with_corpus(corpus_path, terms,
                        min_pair_count=AUGMENT_MIN_PAIR_COUNT,
                        min_term_length=AUGMENT_MIN_TERM_LENGTH):
    """PMI edges between known terms computed on the PubMed sentences.

    Returns:
        pubmed_sentences: list of sentence strings
        pmi_edges: dict (t1, t2) -> PMI on PubMed (pairs seen >= min_pair_count)
        n_terms_found: number of known terms occurring in the corpus
    """
    if not os.path.exists(corpus_path):
        print(f"  Corpus augmentation: file not found: {corpus_path}")
        return [], {}, 0

    with open(corpus_path, 'r', encoding='utf-8') as f:
        pubmed_sentences = [line.strip() for line in f if line.strip()]
    print(f"  PubMed sentences: {len(pubmed_sentences)}")

    matchable = [t for t in terms
                 if len(t) >= min_term_length and re.search('[a-z]', t)]
    matcher = TermMatcher(matchable)
    sentence_terms = [matcher.terms_in(s) for s in pubmed_sentences]
    term_df, pair_counts = sentence_cooccurrence(sentence_terms, set(matchable))
    frequent_pairs = Counter({p: c for p, c in pair_counts.items()
                              if c >= min_pair_count})
    pmi_edges = compute_pmi(frequent_pairs, term_df, len(pubmed_sentences))
    print(f"  Corpus augmentation: {len(term_df)}/{len(terms)} terms found, "
          f"{len(pmi_edges)} PubMed PMI edges (pair count >= {min_pair_count})")
    return pubmed_sentences, pmi_edges, len(term_df)


def build_graph(datadir, ner_type='fine', pubmed_edges=None):
    """Build the co-occurrence graph from CoNLL BIO-tagged files.

    Args:
        datadir: Directory containing .conll files
        ner_type: Label for this graph variant
        pubmed_edges: optional dict (t1, t2) -> PubMed PMI to add (weighted)

    Returns:
        pyg_data: PyG Data object with embeddings and edges
        nodemap: dict mapping term -> node index
        gold_types: dict mapping term -> {NER type: mention count}
        corpus_freq: dict mapping term -> sentence frequency
        sentences: list of all reconstructed annotated sentences
    """
    import networkx as nx
    import torch
    from torch_geometric.utils import from_networkx

    sentences, doc_entities, _ = conll.load_corpus(datadir)
    n_mentions = sum(len(e) for e in doc_entities)
    print(f"Processing {ner_type}: {len(sentences)} sentences, "
          f"{n_mentions} entity mentions")

    term_df = conll.term_document_frequency(doc_entities)
    all_valid = conll.valid_terms(doc_entities)
    valid = set(all_valid)
    print(f"  Valid terms (freq >= {conll.MIN_TERM_DF}): {len(all_valid)}")

    _, cooccurrence = sentence_cooccurrence(
        [[t for t, _ in sent_ents] for sent_ents in doc_entities], valid)
    pmi_edges = compute_pmi(cooccurrence, term_df, len(doc_entities))
    print(f"  PMI edges (positive): {len(pmi_edges)}")

    edge_weights = dict(pmi_edges)
    if pubmed_edges:
        added = 0
        for pair, pmi in pubmed_edges.items():
            if pair[0] in valid and pair[1] in valid:
                if pair not in edge_weights:
                    added += 1
                edge_weights[pair] = edge_weights.get(pair, 0.0) + AUGMENT_WEIGHT * pmi
        print(f"  Augmented edges: +{added} PubMed-only edges "
              f"({len(pmi_edges)} edges from the annotated papers)")

    nodemap = {term: i for i, term in enumerate(all_valid)}

    print(f"  Encoding {len(all_valid)} terms...")
    node_embeddings = _get_embedder().encode(all_valid, show_progress_bar=False)
    x = torch.tensor(node_embeddings, dtype=torch.float)

    G = nx.Graph()
    G.add_nodes_from(range(len(all_valid)))
    for (t1, t2), weight in sorted(edge_weights.items()):
        G.add_edge(nodemap[t1], nodemap[t2], weight=weight)

    print(f"  {ner_type} graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    pyg_data = from_networkx(G)
    pyg_data.x = x

    type_counts = conll.term_type_counts(doc_entities)
    gold_types = {t: dict(sorted(type_counts[t].items())) for t in all_valid}
    corpus_freq = {t: term_df[t] for t in all_valid}
    return pyg_data, nodemap, gold_types, corpus_freq, sentences


def _write_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


if __name__ == "__main__":
    import torch

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    fine_dir = os.path.join(RAW_DIR, "fine_grained_ner")

    pubmed_sentences, pubmed_edges = [], None
    if CORPUS_AUGMENT:
        corpus_path = CORPUS_AUGMENT_PATH
        if not os.path.isabs(corpus_path):
            corpus_path = os.path.join(PROJECT_ROOT, corpus_path)
        print(f"Corpus augmentation enabled: {corpus_path}")
        _, fine_entities, _ = conll.load_corpus(fine_dir)
        pubmed_sentences, pubmed_edges, _ = augment_with_corpus(
            corpus_path, conll.valid_terms(fine_entities))

    # Fine-grained NER (primary)
    data_fine, map_fine, gold_types, corp_freq, sentences = build_graph(
        fine_dir, 'fine', pubmed_edges=pubmed_edges)
    torch.save(data_fine, os.path.join(PROCESSED_DIR, "graph_fine_auto.pt"))
    torch.save(map_fine, os.path.join(PROCESSED_DIR, "nodemap_fine_auto.pt"))

    with open(os.path.join(PROCESSED_DIR, GOLD_TYPES_FILE), 'w') as f:
        json.dump(gold_types, f, indent=2)
    print(f"Saved {GOLD_TYPES_FILE}: {len(gold_types)} terms (gold labels)")

    with open(os.path.join(PROCESSED_DIR, "corpus_frequencies.json"), 'w') as f:
        json.dump(corp_freq, f, indent=2)
    print(f"Saved corpus_frequencies.json: {len(corp_freq)} terms")

    _write_lines(os.path.join(PROCESSED_DIR, "conll_sentences.txt"), sentences)
    all_sentences = sentences + pubmed_sentences
    _write_lines(os.path.join(PROCESSED_DIR, "all_sentences.txt"), all_sentences)
    print(f"Saved all_sentences.txt: {len(all_sentences)} sentences "
          f"({len(pubmed_sentences)} from PubMed)")

    # Coarse-grained (baseline comparison, no augmentation)
    coarse_dir = os.path.join(RAW_DIR, "coarse_grained_ner")
    if os.path.exists(coarse_dir) and os.listdir(coarse_dir):
        data_coarse, map_coarse, _, _, _ = build_graph(coarse_dir, 'coarse')
        torch.save(data_coarse, os.path.join(PROCESSED_DIR, "graph_coarse.pt"))
        torch.save(map_coarse, os.path.join(PROCESSED_DIR, "nodemap_coarse.pt"))

    print("Done: graph built from BIO-tagged NER entities.")
