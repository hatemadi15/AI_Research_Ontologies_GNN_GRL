"""
relations.py - Full Relation Extraction (17 Ontology Relation Types)

Extracts relations using:
  - Pattern matching for all 17 ontology relation types
  - 1-4 word span matching (not just 1-2)
  - Dependency parse extraction via spaCy
  - Fuzzy node matching (lemmatization, substring)
  - GNN co-occurrence relations
  - No bare except clauses
"""

import os
import json
import re
from collections import defaultdict

import numpy as np
import torch
import pandas as pd
import spacy
from sentence_transformers import util

# Absolute paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dataset")
GNN_EMBEDS_PATH = os.path.join(PROCESSED_DIR, 'gnn_embeds.pt')
NODEMAP_PATH = os.path.join(PROCESSED_DIR, 'nodemap_fine_auto.pt')
SENTENCES_PATH = os.path.join(PROCESSED_DIR, 'all_sentences.txt')
GRAPH_PATH = os.path.join(PROCESSED_DIR, 'graph_fine_auto.pt')

# Load NLP model
print("Loading NLP models...")
nlp = spacy.load('en_core_web_sm')

# All 17 ontology relation types with patterns (1-4 word spans)
SPAN = r'([\w]+(?:\s+[\w]+){0,3})'  # 1-4 word noun phrase

REL_PATTERNS = {
    'causeOf': [
        rf'{SPAN}\s+(?:causes?|leads?\s+to|induces?|results?\s+in)\s+{SPAN}',
        rf'{SPAN}\s+(?:is\s+(?:the\s+)?cause\s+of)\s+{SPAN}',
    ],
    'necessaryCauseOf': [
        rf'{SPAN}\s+(?:necessarily\s+causes?|is\s+necessary\s+for)\s+{SPAN}',
        rf'{SPAN}\s+(?:requires?|depends?\s+on)\s+{SPAN}',
    ],
    'sufficientCauseOf': [
        rf'{SPAN}\s+(?:is\s+sufficient\s+(?:for|to\s+cause))\s+{SPAN}',
        rf'{SPAN}\s+(?:guarantees?|ensures?)\s+{SPAN}',
    ],
    'correlatedWith': [
        rf'{SPAN}\s+(?:correlates?\s+with|is\s+correlated\s+(?:with|to))\s+{SPAN}',
        rf'{SPAN}\s+(?:is\s+proportional\s+to|varies?\s+with)\s+{SPAN}',
    ],
    'associatedWith': [
        rf'{SPAN}\s+(?:is\s+associated\s+with|relates?\s+to)\s+{SPAN}',
        rf'{SPAN}\s+(?:involves?|is\s+linked\s+to)\s+{SPAN}',
    ],
    'hasProperty': [
        rf'{SPAN}\s+(?:has|exhibits?|shows?|characterised\s+by|displays?)\s+{SPAN}',
        rf'{SPAN}\s+(?:possesses?|demonstrates?)\s+{SPAN}',
    ],
    'isPartOf': [
        rf'{SPAN}\s+(?:is\s+(?:a\s+)?part\s+of|belongs?\s+to)\s+{SPAN}',
        rf'{SPAN}\s+(?:consists?\s+of|comprises?|contains?)\s+{SPAN}',
    ],
    'constrains': [
        rf'{SPAN}\s+(?:constrains?|limits?|bounds?|restricts?)\s+{SPAN}',
    ],
    'hardConstrains': [
        rf'{SPAN}\s+(?:strictly\s+(?:constrains?|limits?))\s+{SPAN}',
    ],
    'softConstrains': [
        rf'{SPAN}\s+(?:softly\s+constrains?|partially\s+limits?)\s+{SPAN}',
    ],
    'measures': [
        rf'{SPAN}\s+(?:measures?|quantifies?|characterizes?)\s+{SPAN}',
    ],
    'growsInto': [
        rf'{SPAN}\s+(?:grows?\s+into|develops?\s+into|evolves?\s+into)\s+{SPAN}',
    ],
    'initiatesAt': [
        rf'{SPAN}\s+(?:initiates?\s+at|nucleates?\s+at|starts?\s+at)\s+{SPAN}',
    ],
    'precedes': [
        rf'{SPAN}\s+(?:precedes?|occurs?\s+before|is\s+followed\s+by)\s+{SPAN}',
    ],
    'alignedWith': [
        rf'{SPAN}\s+(?:is\s+aligned\s+with|aligns?\s+with)\s+{SPAN}',
    ],
    'parallelTo': [
        rf'{SPAN}\s+(?:is\s+parallel\s+to|runs?\s+parallel)\s+{SPAN}',
    ],
    'spatiallyCoincidesWith': [
        rf'{SPAN}\s+(?:coincides?\s+(?:spatially\s+)?with|overlaps?\s+with)\s+{SPAN}',
    ],
}

# spaCy dependency patterns for relation extraction
DEP_RELATIONS = {
    'nsubj+dobj': 'associatedWith',  # Subject-verb-object
    'nsubj+prep+pobj': 'associatedWith',
    'amod': 'hasProperty',  # Adjectival modifier
    'compound': 'isPartOf',  # Compound nouns
}


def load_nodemap():
    """Load nodemap from builder output."""
    if os.path.exists(NODEMAP_PATH):
        return torch.load(NODEMAP_PATH, weights_only=False)
    return {}


def load_gnn_data():
    """Load GNN embeddings if available."""
    if os.path.exists(GNN_EMBEDS_PATH):
        gnn_data = torch.load(GNN_EMBEDS_PATH, weights_only=False)
        return gnn_data['embeds'], gnn_data['nodemap']
    return None, None


def load_sentences():
    """Load sentences from cached file or reconstruct from CoNLL."""
    if os.path.exists(SENTENCES_PATH):
        with open(SENTENCES_PATH, 'r', encoding='utf-8') as f:
            sents = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(sents)} sentences from cache")
        return sents

    # Fallback: reconstruct from CoNLL
    raw_dir = os.path.join(RAW_DIR, 'fine_grained_ner')
    sents = []
    if not os.path.exists(raw_dir):
        print(f"ERROR: Directory {raw_dir} does not exist.")
        return []
    files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.conll')])
    for f in files:
        with open(os.path.join(raw_dir, f), encoding='utf-8') as fp:
            sent = []
            for line in fp:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if parts:
                        sent.append(parts[0])
                elif sent:
                    sents.append(' '.join(sent))
                    sent = []
            if sent:
                sents.append(' '.join(sent))
    print(f"Loaded {len(sents)} sentences from CoNLL files")
    return sents


def fuzzy_match(term, valid_terms, lemma_map):
    """Fuzzy matching: exact -> lemma -> substring."""
    term_clean = re.sub(r'^(the|a|an)\s+', '', term, flags=re.I).strip()

    # Exact match
    if term_clean in valid_terms:
        return term_clean

    # Lemma match
    lemma = lemma_map.get(term_clean, term_clean)
    if lemma in valid_terms:
        return lemma

    # Substring match (term is part of a known node or vice versa)
    for vt in valid_terms:
        if len(term_clean) >= 3 and term_clean in vt:
            return vt
        if len(vt) >= 3 and vt in term_clean:
            return vt

    return None


def extract_dep_relations(doc, valid_terms, lemma_map):
    """Extract relations from spaCy dependency parse."""
    rels = []
    for token in doc:
        # Subject-verb-object patterns
        if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
            subj = token.text.lower()
            # Find direct object
            for child in token.head.children:
                if child.dep_ in ('dobj', 'attr', 'oprd'):
                    obj = child.text.lower()
                    s_match = fuzzy_match(subj, valid_terms, lemma_map)
                    o_match = fuzzy_match(obj, valid_terms, lemma_map)
                    if s_match and o_match and s_match != o_match:
                        rels.append({
                            'subj': s_match,
                            'rel': 'associatedWith',
                            'obj': o_match,
                            'source': 'dep_parse',
                        })

        # Compound nouns -> isPartOf
        if token.dep_ == 'compound':
            part = token.text.lower()
            whole = token.head.text.lower()
            p_match = fuzzy_match(part, valid_terms, lemma_map)
            w_match = fuzzy_match(whole, valid_terms, lemma_map)
            if p_match and w_match and p_match != w_match:
                rels.append({
                    'subj': p_match,
                    'rel': 'isPartOf',
                    'obj': w_match,
                    'source': 'dep_parse',
                })

        # Adjectival modifiers -> hasProperty
        if token.dep_ == 'amod' and token.head.pos_ in ('NOUN', 'PROPN'):
            prop = token.text.lower()
            entity = token.head.text.lower()
            e_match = fuzzy_match(entity, valid_terms, lemma_map)
            if e_match and len(prop) > 2:
                # Check if the adjective itself maps to a property node
                p_match = fuzzy_match(prop, valid_terms, lemma_map)
                if p_match and p_match != e_match:
                    rels.append({
                        'subj': e_match,
                        'rel': 'hasProperty',
                        'obj': p_match,
                        'source': 'dep_parse',
                    })

    return rels


def extract_gnn_cooccurrence_relations(nodemap, gnn_embeds, threshold=0.90,
                                       min_cooccurrence=3,
                                       cooccurrence_counts=None):
    """Extract relations from GNN embedding similarity (co-occurrence).

    Args:
        threshold: Minimum cosine similarity (default 0.90, raised from 0.65)
        min_cooccurrence: Minimum co-occurrence count in sentences (default 3)
        cooccurrence_counts: dict of (t1, t2) -> count from builder
    """
    if gnn_embeds is None:
        return []

    inv_map = {idx: term for term, idx in nodemap.items()}
    n = gnn_embeds.shape[0]
    rels = []

    # Load co-occurrence counts if not provided
    if cooccurrence_counts is None:
        cooc_path = os.path.join(PROCESSED_DIR, 'corpus_frequencies.json')
        sentences = load_sentences()
        # Build sentence-level co-occurrence counts
        cooccurrence_counts = {}
        valid_terms = set(nodemap.keys())
        for sent in sentences:
            sent_lower = sent.lower()
            terms_in_sent = [t for t in valid_terms if t in sent_lower]
            for i, t1 in enumerate(terms_in_sent):
                for t2 in terms_in_sent[i+1:]:
                    pair = tuple(sorted([t1, t2]))
                    cooccurrence_counts[pair] = cooccurrence_counts.get(pair, 0) + 1

    # Compute pairwise similarity for high-sim pairs only
    embeds_t = torch.tensor(gnn_embeds, dtype=torch.float)
    batch_size = 100
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        batch = embeds_t[i:end_i]
        sims = util.cos_sim(batch, embeds_t)
        for bi in range(end_i - i):
            for j in range(i + bi + 1, n):
                sim_val = sims[bi, j].item()
                if sim_val >= threshold:
                    t1 = inv_map.get(i + bi, '')
                    t2 = inv_map.get(j, '')
                    if t1 and t2:
                        # Check minimum co-occurrence count
                        pair = tuple(sorted([t1, t2]))
                        cooc_count = cooccurrence_counts.get(pair, 0)
                        if cooc_count >= min_cooccurrence:
                            rels.append({
                                'subj': t1,
                                'rel': 'relatedTo',
                                'obj': t2,
                                'sim': round(sim_val, 4),
                                'source': 'gnn_cooccurrence',
                            })

    print(f"  GNN co-occurrence relations (threshold={threshold}, "
          f"min_cooc={min_cooccurrence}): {len(rels)}")
    return rels


def extract_rels():
    """Extract all relations using patterns, dep parse, and GNN."""
    sents = load_sentences()
    nodemap = load_nodemap()
    gnn_embeds, gnn_nodemap = load_gnn_data()

    if not nodemap:
        print("ERROR: No nodemap found. Run builder.py first.")
        return

    valid_terms = set(nodemap.keys())
    print(f"Valid terms: {len(valid_terms)}")

    # Build lemma map for fuzzy matching
    lemma_map = {}
    for term in valid_terms:
        doc = nlp(term)
        lemma = ' '.join([t.lemma_.lower() for t in doc])
        if lemma != term:
            lemma_map[lemma] = term
            lemma_map[term] = term

    # Blacklist generic/stopword terms
    BLACKLIST = {
        "the generation", "a result", "this process", "the material",
        "examples", "properties", "generation", "results", "studies",
        "effect", "effects", "increase", "decrease", "analysis",
        "method", "methods", "process", "value", "values",
    }

    rels = []

    print("Extracting relations...")
    print("  Phase 1: Pattern matching...")
    pattern_count = 0
    for doc in nlp.pipe(sents, batch_size=50):
        text_lower = doc.text.lower()

        for rel_type, patterns in REL_PATTERNS.items():
            for pattern in patterns:
                for m in re.finditer(pattern, text_lower):
                    groups = m.groups()
                    if len(groups) < 2:
                        continue

                    subj_raw = groups[0].strip()
                    obj_raw = groups[-1].strip()

                    # Clean articles
                    subj = re.sub(r'^(the|a|an)\s+', '', subj_raw,
                                  flags=re.I).strip()
                    obj = re.sub(r'^(the|a|an)\s+', '', obj_raw,
                                 flags=re.I).strip()

                    if subj.lower() in BLACKLIST or obj.lower() in BLACKLIST:
                        continue

                    # Fuzzy match to known nodes
                    s_match = fuzzy_match(subj, valid_terms, lemma_map)
                    o_match = fuzzy_match(obj, valid_terms, lemma_map)

                    if s_match and o_match and s_match != o_match:
                        rel_entry = {
                            'subj': s_match,
                            'rel': rel_type,
                            'obj': o_match,
                            'source': 'pattern',
                        }

                        # Add GNN similarity if available
                        if gnn_embeds is not None and gnn_nodemap is not None:
                            if s_match in gnn_nodemap and o_match in gnn_nodemap:
                                idx1 = gnn_nodemap[s_match]
                                idx2 = gnn_nodemap[o_match]
                                if idx1 < gnn_embeds.shape[0] and idx2 < gnn_embeds.shape[0]:
                                    vec1 = gnn_embeds[idx1]
                                    vec2 = gnn_embeds[idx2]
                                    sim = util.cos_sim(
                                        torch.tensor(vec1).unsqueeze(0),
                                        torch.tensor(vec2).unsqueeze(0)
                                    )[0][0].item()
                                    rel_entry['sim'] = round(sim, 4)

                        rels.append(rel_entry)
                        pattern_count += 1

    print(f"    Pattern relations: {pattern_count}")

    print("  Phase 2: Dependency parsing...")
    dep_count = 0
    dep_extracted = []
    for doc in nlp.pipe(sents, batch_size=50):
        dep_rels = extract_dep_relations(doc, valid_terms, lemma_map)
        dep_extracted.extend([(r, doc.text) for r in dep_rels])
        rels.extend(dep_rels)
        dep_count += len(dep_rels)
    print(f"    Dep parse relations: {dep_count}")

    # LLM classification for dep-parse extracted relations
    try:
        from config import LLM_MODE as _LLM_MODE
    except ImportError:
        _LLM_MODE = False
    if _LLM_MODE and dep_extracted:
        try:
            from llm_validator import classify_relation
            llm_classified = 0
            for rel_entry, context in dep_extracted:
                try:
                    llm_rel = classify_relation(
                        rel_entry['subj'], rel_entry['obj'], context
                    )
                    if llm_rel and llm_rel != 'NONE':
                        rel_entry['rel'] = llm_rel
                        rel_entry['source'] = 'dep_parse+llm'
                        llm_classified += 1
                except Exception as e:
                    pass  # Keep original dep-parse relation type
            print(f"    LLM reclassified {llm_classified} dep-parse relations")
        except ImportError as e:
            print(f"    LLM classification unavailable: {e}")

    print("  Phase 3: GNN co-occurrence relations...")
    if gnn_embeds is not None:
        try:
            from config import GNN_COOCCURRENCE_THRESHOLD, GNN_MIN_COOCCURRENCE_COUNT
        except ImportError:
            GNN_COOCCURRENCE_THRESHOLD = 0.90
            GNN_MIN_COOCCURRENCE_COUNT = 3
        gnn_rels = extract_gnn_cooccurrence_relations(
            gnn_nodemap or nodemap, gnn_embeds,
            threshold=GNN_COOCCURRENCE_THRESHOLD,
            min_cooccurrence=GNN_MIN_COOCCURRENCE_COUNT,
        )
        rels.extend(gnn_rels)

    # FIX 3b: LLM-Based Relation Classification for top GNN co-occurrence pairs
    try:
        from config import LLM_RELATIONS
    except ImportError:
        LLM_RELATIONS = False

    from config import LLM_API_KEY, LLM_MODEL
    if LLM_RELATIONS and LLM_API_KEY:
        try:
            from llm_validator import _get_client, _load_cache, _save_cache, _cache_key

            # Collect GNN co-occurrence relations and sort by similarity
            gnn_rels_for_llm = [r for r in rels
                                if r.get('source') == 'gnn_cooccurrence'
                                and r.get('sim', 0) > 0]
            gnn_rels_for_llm.sort(key=lambda x: -x.get('sim', 0))
            gnn_rels_for_llm = gnn_rels_for_llm[:300]

            if gnn_rels_for_llm:
                print(f"  Phase 4: LLM relation classification for "
                      f"{len(gnn_rels_for_llm)} top GNN pairs...")

                # Load relation cache
                rel_cache_path = os.path.join(PROCESSED_DIR,
                                              'llm_relation_cache.json')
                if os.path.exists(rel_cache_path):
                    with open(rel_cache_path, 'r', encoding='utf-8') as f:
                        rel_cache = json.load(f)
                else:
                    rel_cache = {}

                client = _get_client()
                cache = _load_cache()
                llm_classified = 0
                llm_calls = 0

                RELATION_CHOICES = (
                    "hasProperty, causeOf, influencedBy, isPartOf, composedOf, "
                    "constrains, measures, initiatesAt, precedes, relatedTo, "
                    "or NONE"
                )

                for rel_entry in gnn_rels_for_llm:
                    subj = rel_entry['subj']
                    obj = rel_entry['obj']
                    pair_key = f"{subj}|||{obj}"

                    key = _cache_key('llm_relation_v2', subj, obj)
                    if key in cache:
                        answer = cache[key]
                    elif pair_key in rel_cache:
                        answer = rel_cache[pair_key]
                    else:
                        try:
                            response = client.chat.completions.create(
                                model=LLM_MODEL,
                                messages=[{
                                    "role": "user",
                                    "content": (
                                        f"What is the semantic relationship "
                                        f"between '{subj}' and '{obj}' in "
                                        f"materials science? Choose exactly one: "
                                        f"{RELATION_CHOICES}."
                                    )
                                }],
                                max_tokens=20,
                                temperature=0.0,
                            )
                            answer = response.choices[0].message.content.strip()
                            cache[key] = answer
                            rel_cache[pair_key] = answer
                            llm_calls += 1
                        except Exception as e:
                            continue

                    # Update relation type if valid
                    answer_clean = answer.strip().rstrip('.')
                    valid_types = {
                        'hasproperty': 'hasProperty',
                        'causeof': 'causeOf',
                        'influencedby': 'influencedBy',
                        'ispartof': 'isPartOf',
                        'composedof': 'composedOf',
                        'constrains': 'constrains',
                        'measures': 'measures',
                        'initiatesat': 'initiatesAt',
                        'precedes': 'precedes',
                        'relatedto': 'relatedTo',
                    }
                    mapped = valid_types.get(answer_clean.lower())
                    if mapped and mapped != 'NONE':
                        rel_entry['rel'] = mapped
                        rel_entry['source'] = 'gnn_cooccurrence+llm'
                        llm_classified += 1

                # Save relation cache
                os.makedirs(PROCESSED_DIR, exist_ok=True)
                with open(rel_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(rel_cache, f, indent=2, ensure_ascii=False)
                _save_cache(cache)

                print(f"    LLM relation classification: {llm_calls} API calls, "
                      f"{llm_classified} reclassified")
        except Exception as e:
            print(f"  LLM relation classification failed: {e}")

    # Deduplicate
    seen = set()
    unique_rels = []
    for r in rels:
        key = (r['subj'], r['rel'], r['obj'])
        if key not in seen:
            seen.add(key)
            unique_rels.append(r)

    if unique_rels:
        out_path = os.path.join(PROCESSED_DIR, 'relations.csv')
        df = pd.DataFrame(unique_rels)
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} unique relations to {out_path}")

        # Stats by relation type
        type_counts = df['rel'].value_counts()
        print("\nRelation type distribution:")
        for rel_type, count in type_counts.items():
            print(f"  {rel_type}: {count}")

        # Stats by source
        source_counts = df['source'].value_counts()
        print("\nSource distribution:")
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
    else:
        print("No relations found.")


if __name__ == '__main__':
    extract_rels()
