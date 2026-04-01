"""
rag_typing.py - RAG-Based Term Typing using GPT-4o-mini

For entities with low-confidence NER type assignments or no NER type:
  - Get top-5 candidate ontology classes by SBERT embedding similarity
  - Build prompt with entity context + candidate classes
  - Query GPT-4o-mini to select best matching class
  - Use the answer as type assignment

Inspired by SBU-NLP (#1 at LLMs4OL 2025).
Behind RAG_TYPING=true flag in config.py (default true).
Limited to max 300 LLM calls for cost control.
"""

import os
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from config import DEFAULT_EMBEDDER_MODEL, RAG_TYPING

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dataset")

EMBEDDER_MODEL = os.environ.get('EMBEDDER_MODEL', DEFAULT_EMBEDDER_MODEL)

MAX_LLM_CALLS = 300


def load_entities_and_types():
    """Load entities and their NER types from entity_types.json."""
    path = os.path.join(PROCESSED_DIR, 'entity_types.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_sentences():
    """Load sentences for context retrieval."""
    path = os.path.join(PROCESSED_DIR, 'all_sentences.txt')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return []


def load_reference_classes():
    """Load ontology class labels from ontology.ttl."""
    ttl_path = os.path.join(RAW_DIR, 'ontologies', 'ontology.ttl')
    if not os.path.exists(ttl_path):
        return []

    from rdflib import Graph, RDF, RDFS, OWL, Namespace

    g = Graph()
    g.parse(ttl_path, format='turtle')

    MMO = Namespace("https://w3id.org/pmd/materials-mechanics-ontology/")
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

    label_props = [RDFS.label, MMO.altLabel, MMO.prefLabel,
                   SKOS.prefLabel, SKOS.altLabel]

    labels = []
    for cls in g.subjects(RDF.type, OWL.Class):
        cls_str = str(cls)
        if cls_str.startswith('http://www.w3.org/'):
            continue

        cls_labels = []
        for prop in label_props:
            for label in g.objects(cls, prop):
                label_str = str(label).strip()
                if label_str and not label_str.startswith('http'):
                    cls_labels.append(label_str)

        if not cls_labels:
            fragment = cls_str.split('#')[-1].split('/')[-1]
            if fragment and fragment[0].isupper():
                cls_labels.append(fragment)

        if cls_labels:
            labels.append(cls_labels[0])

    return labels


def find_context(entity, sentences):
    """Find a sentence containing the entity for context."""
    entity_lower = entity.lower()
    for sent in sentences:
        if entity_lower in sent.lower():
            return sent
    return ""


def run_rag_typing():
    """Run RAG-based term typing.

    For entities with low-confidence or missing NER types:
    1. Get top-5 candidate ontology classes by SBERT similarity
    2. Query GPT-4o-mini with entity + context + candidates
    3. Update entity type assignments
    """
    if not RAG_TYPING:
        print("RAG_TYPING disabled, skipping")
        return

    if not os.getenv('OPENAI_API_KEY'):
        print("OPENAI_API_KEY not set, skipping RAG typing")
        return

    try:
        from llm_validator import _get_client, _load_cache, _save_cache, _cache_key
    except ImportError:
        print("llm_validator not available, skipping RAG typing")
        return

    entity_types = load_entities_and_types()
    sentences = load_sentences()
    ref_classes = load_reference_classes()

    if not entity_types or not ref_classes:
        print("No entities or reference classes found for RAG typing")
        return

    print(f"\nRAG typing: {len(entity_types)} entities, {len(ref_classes)} classes")

    # Load alignment to find entities with low-confidence alignments
    alignment_path = os.path.join(PROCESSED_DIR, 'ontology_alignment.csv')
    aligned_entities = set()
    high_conf_entities = set()
    if os.path.exists(alignment_path):
        import pandas as pd
        alignment_df = pd.read_csv(alignment_path)
        for _, row in alignment_df.iterrows():
            aligned_entities.add(row['discovered'])
            if row['similarity'] >= 0.6:
                high_conf_entities.add(row['discovered'])

    # Candidates for RAG typing: entities NOT in high-confidence alignments
    candidates = []
    for entity, ner_type in entity_types.items():
        if entity not in high_conf_entities:
            candidates.append(entity)

    # Sort by entity length (longer = more informative, prioritize)
    candidates.sort(key=lambda x: -len(x))
    candidates = candidates[:MAX_LLM_CALLS]

    print(f"  RAG typing candidates: {len(candidates)}")

    if not candidates:
        return

    # Compute SBERT embeddings
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    ref_embeds = embedder.encode(ref_classes)
    candidate_embeds = embedder.encode(candidates)

    # Compute similarity matrix
    sim_matrix = util.cos_sim(
        torch.tensor(candidate_embeds).float(),
        torch.tensor(ref_embeds).float()
    ).numpy()

    client = _get_client()
    cache = _load_cache()
    llm_calls = 0
    updated = 0
    rag_results = {}

    for i, entity in enumerate(candidates):
        if llm_calls >= MAX_LLM_CALLS:
            break

        # Get top-5 candidate classes
        top5_idx = np.argsort(sim_matrix[i])[-5:][::-1]
        top5_classes = [ref_classes[j] for j in top5_idx]
        top5_sims = [sim_matrix[i, j] for j in top5_idx]

        # Get sentence context
        context = find_context(entity, sentences)
        if not context:
            context = "(no context available)"

        # Build options string
        options = ', '.join([f"{c} ({s:.2f})" for c, s in zip(top5_classes, top5_sims)])

        key = _cache_key('rag_typing', entity, options)
        if key in cache:
            answer = cache[key]
        else:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Given the materials science entity '{entity}' "
                            f"appearing in this context: '{context[:200]}'. "
                            f"Which of these ontology classes is the best match? "
                            f"Options: {', '.join(top5_classes)}. "
                            f"Or NONE if none match. "
                            f"Answer with ONLY the class name or NONE."
                        )
                    }],
                    max_tokens=30,
                    temperature=0.0,
                )
                answer = response.choices[0].message.content.strip()
                cache[key] = answer
                llm_calls += 1
            except Exception as e:
                print(f"  LLM error for entity '{entity}': {e}")
                continue

        if answer and answer.upper() != 'NONE':
            # Find the matching class
            matched = None
            answer_lower = answer.lower()
            for cls in top5_classes:
                if cls.lower() == answer_lower or cls.lower() in answer_lower:
                    matched = cls
                    break

            if matched:
                rag_results[entity] = matched
                updated += 1

    _save_cache(cache)

    # Save RAG typing results
    rag_path = os.path.join(PROCESSED_DIR, 'rag_typing_results.json')
    with open(rag_path, 'w', encoding='utf-8') as f:
        json.dump(rag_results, f, indent=2, ensure_ascii=False)

    # Also update entity_types.json with RAG-typed entities
    if rag_results:
        for entity, cls in rag_results.items():
            entity_types[entity] = cls
        types_path = os.path.join(PROCESSED_DIR, 'entity_types.json')
        with open(types_path, 'w', encoding='utf-8') as f:
            json.dump(entity_types, f, indent=2, ensure_ascii=False)

    # Update alignment CSV with RAG-based alignments
    if rag_results and os.path.exists(alignment_path):
        import pandas as pd
        alignment_df = pd.read_csv(alignment_path)
        new_rows = []
        existing_pairs = set(
            zip(alignment_df['discovered'].str.lower(),
                alignment_df['reference'].str.lower())
        )
        for entity, cls in rag_results.items():
            if (entity.lower(), cls.lower()) not in existing_pairs:
                new_rows.append({
                    'discovered': entity,
                    'reference': cls,
                    'similarity': 0.70,  # RAG-typed confidence
                    'direction': 'rag_typed',
                })
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            alignment_df = pd.concat([alignment_df, new_df], ignore_index=True)
            alignment_df.to_csv(alignment_path, index=False)
            print(f"  Added {len(new_rows)} RAG-typed alignments to alignment CSV")

    print(f"  RAG typing: {llm_calls} API calls, {updated} entities typed")
    print(f"  Results saved to {rag_path}")


if __name__ == '__main__':
    run_rag_typing()
