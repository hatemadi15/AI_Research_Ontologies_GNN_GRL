import os
from itertools import combinations
import networkx as nx
import spacy
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.utils import from_networkx

print("Loading NLP models...")
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_raw_text_from_conll(filepath):
    """Reconstructs raw sentences from CoNLL, ignoring labels."""
    sentences = []
    current_tokens = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    sentences.append(' '.join(current_tokens))
                    current_tokens = []
                continue
            parts = line.split()
            if parts:
                current_tokens.append(parts[0])  # Token only
        if current_tokens:
            sentences.append(' '.join(current_tokens))
    return sentences

def automated_candidate_extraction(sentences):
    """Noun chunk-based terms (fully automated)."""
    doc_entities = []
    global_candidates = set()
    print(f"Extracting terms from {len(sentences)} sentences...")
    for doc in nlp.pipe(sentences, batch_size=50):
        candidates = []
        for chunk in doc.noun_chunks:
            clean_term = chunk.text.lower().strip()
            if len(clean_term) < 2 or chunk.root.pos_ not in ['NOUN', 'PROPN']:
                continue
            candidates.append(clean_term)
        doc_entities.append(candidates)
        global_candidates.update(candidates)
    return doc_entities, global_candidates

def coarse_guided_extraction(sentences, datadir):
    """Optional: Coarse labels as baseline filter (e.g., MATERIAL, PROPERTY)."""
    # Placeholder: Implement if coarse CoNLL has broad tags (e.g., filter tokens labeled 'MATERIAL').
    # For now, falls back to automated.
    print("Coarse-guided skipped (implement tag filtering if needed).")
    return automated_candidate_extraction(sentences)

def build_graph(datadir, mode='fine_auto', ner_type='fine'):
    """Builds graph for given dataset/mode."""
    files = [f for f in os.listdir(datadir) if f.endswith('.conll')]
    print(f"Processing {ner_type} ({mode}): {len(files)} files...")
    all_sentences = []
    for filename in files:
        filepath = os.path.join(datadir, filename)
        sents = get_raw_text_from_conll(filepath)
        all_sentences.extend(sents)

    if mode == 'auto':
        doc_entities, unique_candidates = automated_candidate_extraction(all_sentences)
    else:  # 'coarse'
        doc_entities, unique_candidates = coarse_guided_extraction(all_sentences, datadir)

    term_counts = {t: 0 for ents in doc_entities for t in ents}
    for ents in doc_entities:
        for e in ents:
            term_counts[e] += 1
    valid_nodes = [t for t, c in term_counts.items() if c >= 2]
    nodemap = {term: i for i, term in enumerate(valid_nodes)}

    node_embeddings = embedder.encode(valid_nodes, show_progress_bar=True)
    x = torch.tensor(node_embeddings, dtype=torch.float)

    G = nx.Graph()
    G.add_nodes_from(range(len(valid_nodes)))
    for sent_ents in doc_entities:
        ids = list(set(nodemap[e] for e in sent_ents if e in nodemap))
        if len(ids) >= 2:
            for u, v in combinations(ids, 2):
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1.0)

    print(f"{ner_type}-{mode} graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    pyg_data = from_networkx(G)
    pyg_data.x = x
    return pyg_data, nodemap

if __name__ == "__main__":
    # Get the project root directory (2 levels up from this script)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    base_raw = os.path.join(PROJECT_ROOT, "data", "raw", "dataset")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Fine-grained auto (primary)
    data_fine_auto, map_fine_auto = build_graph(os.path.join(base_raw, "fine_grained_ner"), 'auto', 'fine')
    torch.save(data_fine_auto, os.path.join(processed_dir, "graph_fine_auto.pt"))
    torch.save(map_fine_auto, os.path.join(processed_dir, "nodemap_fine_auto.pt"))

    # Coarse-grained (baseline)
    data_coarse, map_coarse = build_graph(os.path.join(base_raw, "coarse_grained_ner"), 'coarse', 'coarse')
    torch.save(data_coarse, os.path.join(processed_dir, "graph_coarse.pt"))
    torch.save(map_coarse, os.path.join(processed_dir, "nodemap_coarse.pt"))

    print("Done: Compare fine_auto vs coarse graphs for quality.")
