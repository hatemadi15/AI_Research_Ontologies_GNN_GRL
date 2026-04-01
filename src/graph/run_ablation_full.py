"""
run_ablation_full.py - Full Pipeline Ablation Study

Runs the pipeline under different configurations and collects metrics
for each variant:
  1. full_pipeline    - All components enabled (NER + GNN + bidirectional)
  2. no_ner           - Noun chunks instead of NER entities
  3. no_gnn           - SBERT embeddings only, skip GNN training
  4. text_only        - No GNN, no bidirectional, text embeddings only
  5. no_bidirectional - Forward-only alignment
  6. baseline_sbert   - SBERT only, forward matching, embedding-only scoring

For each variant, records: nodes, edges, Type/Term/Concept F1, relations, runtime.
"""

import os
import sys
import json
import time
import shutil

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ABLATION_DIR = os.path.join(PROCESSED_DIR, "ablation_results")


def backup_processed():
    backup_dir = os.path.join(PROCESSED_DIR, "_backup")
    os.makedirs(backup_dir, exist_ok=True)
    for fn in os.listdir(PROCESSED_DIR):
        src = os.path.join(PROCESSED_DIR, fn)
        if os.path.isfile(src) and fn != "_backup":
            shutil.copy2(src, os.path.join(backup_dir, fn))
    return backup_dir


def restore_processed(backup_dir):
    for fn in os.listdir(backup_dir):
        src = os.path.join(backup_dir, fn)
        dst = os.path.join(PROCESSED_DIR, fn)
        shutil.copy2(src, dst)


def count_graph_stats():
    import torch
    graph_path = os.path.join(PROCESSED_DIR, "graph_fine_auto.pt")
    if not os.path.exists(graph_path):
        return 0, 0
    data = torch.load(graph_path, weights_only=False)
    return data.num_nodes, data.num_edges


def count_relations():
    rel_path = os.path.join(PROCESSED_DIR, "relations.csv")
    if not os.path.exists(rel_path):
        return 0
    df = pd.read_csv(rel_path)
    return len(df)


def get_eval_results():
    results_path = os.path.join(PROCESSED_DIR, "eval_results.json")
    if not os.path.exists(results_path):
        return {}
    with open(results_path) as f:
        return json.load(f)


def extract_f1(eval_results, level, threshold="1.0"):
    level_key = f"{level}_level"
    if level_key not in eval_results:
        return None
    level_data = eval_results[level_key]
    if 'thresholds' not in level_data:
        return None
    for key in [float(threshold), threshold, str(float(threshold))]:
        if key in level_data['thresholds']:
            return level_data['thresholds'][key].get('f1')
    return None


def run_full_pipeline_variant():
    import align
    import eval_f1

    align.USE_GNN_EMBEDDINGS = True
    align.USE_BIDIRECTIONAL = True
    align.USE_COMBINED_SCORING = True

    align.run_alignment()
    eval_f1.run_evaluation()


def run_no_gnn_variant():
    import align
    import eval_f1

    gnn_path = os.path.join(PROCESSED_DIR, "gnn_embeddings.npy")
    gnn_map = os.path.join(PROCESSED_DIR, "gnn_embed_map.json")
    gnn_bak = gnn_path + ".bak"
    gnn_map_bak = gnn_map + ".bak"

    if os.path.exists(gnn_path):
        os.rename(gnn_path, gnn_bak)
    if os.path.exists(gnn_map):
        os.rename(gnn_map, gnn_map_bak)

    try:
        align.USE_GNN_EMBEDDINGS = False
        align.USE_BIDIRECTIONAL = True
        align.USE_COMBINED_SCORING = True

        align.run_alignment()
        eval_f1.run_evaluation()
    finally:
        if os.path.exists(gnn_bak):
            os.rename(gnn_bak, gnn_path)
        if os.path.exists(gnn_map_bak):
            os.rename(gnn_map_bak, gnn_map)


def run_text_only_variant():
    import align
    import eval_f1

    gnn_path = os.path.join(PROCESSED_DIR, "gnn_embeddings.npy")
    gnn_map = os.path.join(PROCESSED_DIR, "gnn_embed_map.json")
    gnn_bak = gnn_path + ".bak"
    gnn_map_bak = gnn_map + ".bak"

    if os.path.exists(gnn_path):
        os.rename(gnn_path, gnn_bak)
    if os.path.exists(gnn_map):
        os.rename(gnn_map, gnn_map_bak)

    try:
        align.USE_GNN_EMBEDDINGS = False
        align.USE_BIDIRECTIONAL = False
        align.USE_COMBINED_SCORING = False

        align.run_alignment()
        eval_f1.run_evaluation()
    finally:
        if os.path.exists(gnn_bak):
            os.rename(gnn_bak, gnn_path)
        if os.path.exists(gnn_map_bak):
            os.rename(gnn_map_bak, gnn_map)


def run_no_bidirectional_variant():
    import align
    import eval_f1

    align.USE_GNN_EMBEDDINGS = True
    align.USE_BIDIRECTIONAL = False
    align.USE_COMBINED_SCORING = True

    align.run_alignment()
    eval_f1.run_evaluation()


def run_baseline_sbert_variant():
    import align
    import eval_f1

    gnn_path = os.path.join(PROCESSED_DIR, "gnn_embeddings.npy")
    gnn_map = os.path.join(PROCESSED_DIR, "gnn_embed_map.json")
    gnn_bak = gnn_path + ".bak"
    gnn_map_bak = gnn_map + ".bak"

    if os.path.exists(gnn_path):
        os.rename(gnn_path, gnn_bak)
    if os.path.exists(gnn_map):
        os.rename(gnn_map, gnn_map_bak)

    try:
        align.USE_GNN_EMBEDDINGS = False
        align.USE_BIDIRECTIONAL = False
        align.USE_COMBINED_SCORING = False

        align.run_alignment()
        eval_f1.run_evaluation()
    finally:
        if os.path.exists(gnn_bak):
            os.rename(gnn_bak, gnn_path)
        if os.path.exists(gnn_map_bak):
            os.rename(gnn_map_bak, gnn_map)


def run_no_ner_variant():
    import torch
    import networkx as nx
    from sentence_transformers import SentenceTransformer
    from torch_geometric.utils import from_networkx
    from itertools import combinations
    from collections import Counter
    import math

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    sentences_path = os.path.join(PROCESSED_DIR, "all_sentences.txt")
    if not os.path.exists(sentences_path):
        print("  No sentences file, skipping no_ner variant")
        return None

    with open(sentences_path) as f:
        sentences = [line.strip() for line in f if line.strip()]

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        print("  spaCy not available, skipping no_ner variant")
        return None

    doc_entities = []
    for sent in sentences:
        doc = nlp(sent)
        chunks = [(chunk.text.lower().strip(), "NounChunk")
                  for chunk in doc.noun_chunks
                  if len(chunk.text.strip()) >= 2]
        doc_entities.append(chunks)

    term_doc_freq = Counter()
    for sent_ents in doc_entities:
        for term, _ in set(sent_ents):
            term_doc_freq[term] += 1

    valid_terms = {t for t, c in term_doc_freq.items() if c >= 2}
    all_valid = sorted(valid_terms)
    nodemap = {term: i for i, term in enumerate(all_valid)}

    print(f"  Noun chunk graph: {len(all_valid)} terms")

    cooccurrence = Counter()
    for sent_ents in doc_entities:
        unique = list(set(t for t, _ in sent_ents if t in valid_terms))
        if len(unique) >= 2:
            for t1, t2 in combinations(sorted(unique), 2):
                cooccurrence[(t1, t2)] += 1

    total_docs = len(doc_entities)
    pmi_edges = {}
    for (t1, t2), co_count in cooccurrence.items():
        p_xy = co_count / total_docs
        p_x = term_doc_freq[t1] / total_docs
        p_y = term_doc_freq[t2] / total_docs
        denom = p_x * p_y
        if denom > 0 and p_xy > 0:
            pmi = math.log2(p_xy / denom)
            if pmi > 0:
                pmi_edges[(t1, t2)] = pmi

    node_embeddings = embedder.encode(all_valid, show_progress_bar=False)
    x = torch.tensor(node_embeddings, dtype=torch.float)

    G = nx.Graph()
    G.add_nodes_from(range(len(all_valid)))
    for (t1, t2), pmi_val in pmi_edges.items():
        if t1 in nodemap and t2 in nodemap:
            G.add_edge(nodemap[t1], nodemap[t2], weight=pmi_val)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"  Noun chunk graph: {n_nodes} nodes, {n_edges} edges")

    pyg_data = from_networkx(G)
    pyg_data.x = x

    orig_graph = os.path.join(PROCESSED_DIR, "graph_fine_auto.pt")
    orig_nodemap = os.path.join(PROCESSED_DIR, "nodemap_fine_auto.pt")
    orig_ent_types = os.path.join(PROCESSED_DIR, "entity_types.json")

    orig_graph_bak = orig_graph + ".bak"
    orig_nodemap_bak = orig_nodemap + ".bak"
    orig_ent_types_bak = orig_ent_types + ".bak"

    shutil.copy2(orig_graph, orig_graph_bak)
    shutil.copy2(orig_nodemap, orig_nodemap_bak)
    shutil.copy2(orig_ent_types, orig_ent_types_bak)

    torch.save(pyg_data, orig_graph)
    torch.save(nodemap, orig_nodemap)
    with open(orig_ent_types, 'w') as f:
        json.dump({}, f)

    try:
        import align
        import eval_f1

        gnn_path = os.path.join(PROCESSED_DIR, "gnn_embeddings.npy")
        gnn_map = os.path.join(PROCESSED_DIR, "gnn_embed_map.json")
        gnn_bak = gnn_path + ".bak2"
        gnn_map_bak = gnn_map + ".bak2"
        if os.path.exists(gnn_path):
            os.rename(gnn_path, gnn_bak)
        if os.path.exists(gnn_map):
            os.rename(gnn_map, gnn_map_bak)

        import taxonomy
        taxonomy.build_taxonomy()

        align.USE_GNN_EMBEDDINGS = False
        align.USE_BIDIRECTIONAL = True
        align.USE_COMBINED_SCORING = True
        align.run_alignment()
        eval_f1.run_evaluation()

        results = {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
        }
    finally:
        shutil.copy2(orig_graph_bak, orig_graph)
        shutil.copy2(orig_nodemap_bak, orig_nodemap)
        shutil.copy2(orig_ent_types_bak, orig_ent_types)
        os.remove(orig_graph_bak)
        os.remove(orig_nodemap_bak)
        os.remove(orig_ent_types_bak)
        if os.path.exists(gnn_bak):
            os.rename(gnn_bak, gnn_path)
        if os.path.exists(gnn_map_bak):
            os.rename(gnn_map_bak, gnn_map)

    return results


VARIANTS = [
    ("full_pipeline", "All components (NER + GNN + bidirectional + combined)",
     run_full_pipeline_variant),
    ("no_gnn", "No GNN embeddings (SBERT only)",
     run_no_gnn_variant),
    ("no_bidirectional", "Forward-only alignment",
     run_no_bidirectional_variant),
    ("text_only", "Text-only: no GNN, forward, embedding similarity",
     run_text_only_variant),
    ("baseline_sbert", "Baseline SBERT: no GNN, forward, embedding-only",
     run_baseline_sbert_variant),
    ("no_ner", "Noun chunks instead of NER entities",
     run_no_ner_variant),
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full ablation study")
    parser.add_argument('--variant', type=str, default=None,
                        help='Run a specific variant')
    parser.add_argument('--list', action='store_true',
                        help='List available variants')
    args = parser.parse_args()

    if args.list:
        print("Available ablation variants:")
        for name, desc, _ in VARIANTS:
            print(f"  {name}: {desc}")
        return

    os.makedirs(ABLATION_DIR, exist_ok=True)

    if args.variant:
        variants_to_run = [(n, d, f) for n, d, f in VARIANTS if n == args.variant]
        if not variants_to_run:
            print(f"Unknown variant: {args.variant}")
            print(f"Available: {[n for n, _, _ in VARIANTS]}")
            return
    else:
        variants_to_run = VARIANTS

    all_results = []

    base_nodes, base_edges = count_graph_stats()

    for name, desc, run_fn in variants_to_run:
        print(f"\n{'=' * 70}")
        print(f"ABLATION: {name}")
        print(f"  {desc}")
        print(f"{'=' * 70}")

        start_time = time.time()
        extra = None
        try:
            extra = run_fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        elapsed = time.time() - start_time

        eval_results = get_eval_results()
        n_relations = count_relations()

        if extra and isinstance(extra, dict):
            n_nodes = extra.get('n_nodes', base_nodes)
            n_edges = extra.get('n_edges', base_edges)
        else:
            n_nodes = base_nodes
            n_edges = base_edges

        type_f1 = extract_f1(eval_results, 'type', '1.0')
        term_f1 = extract_f1(eval_results, 'term', '0.5')
        concept_f1 = extract_f1(eval_results, 'concept', '0.5')

        type_data = eval_results.get('type_level', {})
        n_gold_mapped = type_data.get('n_gold_mapped', 0)
        n_ner_types = type_data.get('n_ner_types', 0)

        result = {
            'variant': name,
            'description': desc,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'n_ner_types': n_ner_types,
            'n_gold_mapped': n_gold_mapped,
            'type_f1': type_f1,
            'term_f1': term_f1,
            'concept_f1': concept_f1,
            'n_relations': n_relations,
            'runtime_sec': round(elapsed, 2),
            'full_results': eval_results,
        }
        all_results.append(result)

        result_path = os.path.join(ABLATION_DIR, f"{name}_results.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved: {result_path}")

        print(f"\n  Summary: nodes={n_nodes}, edges={n_edges}, "
              f"type_f1={type_f1}, term_f1={term_f1}, "
              f"concept_f1={concept_f1}, relations={n_relations}, "
              f"runtime={elapsed:.1f}s")

    if len(all_results) > 0:
        print(f"\n{'=' * 90}")
        print("ABLATION COMPARISON")
        print(f"{'=' * 90}")

        header = (f"{'Variant':<20} {'Nodes':>6} {'Edges':>6} "
                  f"{'TypeF1':>7} {'TermF1':>7} {'ConcF1':>7} "
                  f"{'Rels':>6} {'Time':>7}")
        print(header)
        print("-" * 90)
        for r in all_results:
            row = (f"{r['variant']:<20} {r['n_nodes']:>6} {r['n_edges']:>6} "
                   f"{r['type_f1'] or 0:>7.4f} {r['term_f1'] or 0:>7.4f} "
                   f"{r['concept_f1'] or 0:>7.4f} "
                   f"{r['n_relations']:>6} {r['runtime_sec']:>6.1f}s")
            print(row)

        summary_rows = [{
            'variant': r['variant'],
            'nodes': r['n_nodes'],
            'edges': r['n_edges'],
            'type_f1': r['type_f1'],
            'term_f1': r['term_f1'],
            'concept_f1': r['concept_f1'],
            'relations': r['n_relations'],
            'runtime_sec': r['runtime_sec'],
        } for r in all_results]
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(ABLATION_DIR, "ablation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved: {summary_path}")

    return all_results


if __name__ == "__main__":
    main()
