"""
relations.py - Relation Extraction (17 Ontology Relation Types)

Extracts relations between discovered terms using:
  - Pattern matching for the ontology relation types (1-4 word spans around
    a cue phrase)
  - Dependency parse extraction via spaCy (USE_DEP_PARSING)
  - GNN co-occurrence relations: terms that co-occur in >= N sentences and
    are close in GNN embedding space (USE_GNN_EMBEDDINGS)
  - Optional LLM relation classification (LLM_RELATIONS; LLM_MODE for
    dependency-parse relations)

Entity resolution uses text_match.TermMatcher: the longest known term inside
a span, with word boundaries. (The previous fuzzy matcher took the first
substring hit while iterating a Python set, so results depended on the hash
seed, and raw substring tests matched terms inside other words.)

Writes relations_raw.csv (all extracted relations) and relations.csv (the
same; relations_axioms.py later replaces it with the axiom-consistent subset).
Both files are always written, also when empty.
"""

import os
import re
from itertools import combinations

import numpy as np
import pandas as pd

from config import (GNN_COOCCURRENCE_THRESHOLD, GNN_MIN_COOCCURRENCE_COUNT,
                    LLM_API_KEY, LLM_MODE, LLM_MODEL, LLM_RELATIONS,
                    PROCESSED_DIR, USE_DEP_PARSING, USE_GNN_EMBEDDINGS)
from text_match import TermMatcher

GNN_EMBEDS_PATH = os.path.join(PROCESSED_DIR, 'gnn_embeds.pt')
NODEMAP_PATH = os.path.join(PROCESSED_DIR, 'nodemap_fine_auto.pt')
SENTENCES_PATH = os.path.join(PROCESSED_DIR, 'all_sentences.txt')
OUTPUT_COLUMNS = ['subj', 'rel', 'obj', 'source', 'sim', 'context']

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
COMPILED_PATTERNS = [(rel, re.compile(p)) for rel, pats in REL_PATTERNS.items()
                     for p in pats]

# Generic terms that make patterns fire without meaning
BLACKLIST = {
    "the generation", "a result", "this process", "the material",
    "examples", "properties", "generation", "results", "studies",
    "effect", "effects", "increase", "decrease", "analysis",
    "method", "methods", "process", "value", "values",
}

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        print("Loading spaCy model...")
        _NLP = spacy.load('en_core_web_sm')
    return _NLP


def load_nodemap():
    """Load nodemap from builder output."""
    import torch
    if os.path.exists(NODEMAP_PATH):
        return torch.load(NODEMAP_PATH, weights_only=False)
    return {}


def load_gnn_data():
    """Load GNN embeddings if available and enabled."""
    import torch
    if USE_GNN_EMBEDDINGS and os.path.exists(GNN_EMBEDS_PATH):
        gnn_data = torch.load(GNN_EMBEDS_PATH, weights_only=False)
        return np.asarray(gnn_data['embeds']), gnn_data['nodemap']
    return None, None


def load_sentences():
    """Load sentences saved by builder.py."""
    if not os.path.exists(SENTENCES_PATH):
        print(f"ERROR: {SENTENCES_PATH} not found. Run builder.py first.")
        return []
    with open(SENTENCES_PATH, 'r', encoding='utf-8') as f:
        sents = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(sents)} sentences")
    return sents


def resolve(phrase, matcher):
    """Map a text span to the longest known term it contains (or None)."""
    phrase = re.sub(r'^(the|a|an)\s+', '', phrase.strip(), flags=re.I)
    if not phrase or phrase.lower() in BLACKLIST:
        return None
    return matcher.match_phrase(phrase)


def extract_pattern_relations(text, matcher):
    """Cue-phrase pattern relations in one sentence."""
    rels = []
    text_lower = text.lower()
    for rel_type, pattern in COMPILED_PATTERNS:
        for m in pattern.finditer(text_lower):
            s_match = resolve(m.group(1), matcher)
            o_match = resolve(m.group(2), matcher)
            if s_match and o_match and s_match != o_match:
                rels.append({'subj': s_match, 'rel': rel_type, 'obj': o_match,
                             'source': 'pattern'})
    return rels


def _phrase(token):
    """The token with its left modifiers ('fatigue crack' for head 'crack')."""
    doc = token.doc
    return doc[token.left_edge.i:token.i + 1].text


def extract_dep_relations(doc, matcher):
    """Extract relations from a spaCy dependency parse."""
    rels = []
    for token in doc:
        # Subject-verb-object patterns
        if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
            s_match = resolve(_phrase(token), matcher)
            for child in token.head.children:
                if child.dep_ in ('dobj', 'attr', 'oprd'):
                    o_match = resolve(_phrase(child), matcher)
                    if s_match and o_match and s_match != o_match:
                        rels.append({'subj': s_match, 'rel': 'associatedWith',
                                     'obj': o_match, 'source': 'dep_parse'})

        # Compound nouns -> isPartOf
        if token.dep_ == 'compound':
            p_match = resolve(token.text, matcher)
            w_match = resolve(token.head.text, matcher)
            if p_match and w_match and p_match != w_match:
                rels.append({'subj': p_match, 'rel': 'isPartOf', 'obj': w_match,
                             'source': 'dep_parse'})

        # Adjectival modifiers -> hasProperty
        if token.dep_ == 'amod' and token.head.pos_ in ('NOUN', 'PROPN') \
                and len(token.text) > 2:
            e_match = resolve(token.head.text, matcher)
            p_match = resolve(token.text, matcher)
            if e_match and p_match and p_match != e_match:
                rels.append({'subj': e_match, 'rel': 'hasProperty', 'obj': p_match,
                             'source': 'dep_parse'})
    return rels


def sentence_cooccurrence(sentences, matcher):
    """Pair counts (word-boundary matches) and a first context sentence."""
    counts = {}
    context = {}
    for sent in sentences:
        found = sorted(matcher.terms_in(sent))
        for pair in combinations(found, 2):
            counts[pair] = counts.get(pair, 0) + 1
            context.setdefault(pair, sent)
    return counts, context


def extract_gnn_cooccurrence_relations(nodemap, gnn_embeds, counts, contexts,
                                       threshold=0.90, min_cooccurrence=3):
    """Pairs that co-occur often and are close in GNN embedding space.

    Args:
        threshold: Minimum cosine similarity of the GNN embeddings
        min_cooccurrence: Minimum number of sentences with both terms
    """
    if gnn_embeds is None:
        return []
    Z = gnn_embeds / np.maximum(np.linalg.norm(gnn_embeds, axis=1, keepdims=True), 1e-12)
    rels = []
    for (t1, t2), count in sorted(counts.items()):
        if count < min_cooccurrence or t1 not in nodemap or t2 not in nodemap:
            continue
        sim = float(Z[nodemap[t1]] @ Z[nodemap[t2]])
        if sim >= threshold:
            rels.append({'subj': t1, 'rel': 'relatedTo', 'obj': t2,
                         'sim': round(sim, 4), 'source': 'gnn_cooccurrence',
                         'context': contexts.get((t1, t2), '')})
    print(f"  GNN co-occurrence relations (threshold={threshold}, "
          f"min_cooc={min_cooccurrence}): {len(rels)}")
    return rels


LLM_RELATION_TYPES = {
    'hasproperty': 'hasProperty', 'causeof': 'causeOf',
    'influencedby': 'influencedBy', 'ispartof': 'isPartOf',
    'composedof': 'composedOf', 'constrains': 'constrains',
    'measures': 'measures', 'initiatesat': 'initiatesAt',
    'precedes': 'precedes', 'relatedto': 'relatedTo',
}


def llm_classify_gnn_relations(rels, max_pairs=300):
    """Reclassify the top GNN co-occurrence pairs with the LLM (in place)."""
    from llm_validator import _get_client, _load_cache, _save_cache, _cache_key

    gnn_rels = sorted((r for r in rels if r.get('source') == 'gnn_cooccurrence'),
                      key=lambda r: (-r.get('sim', 0), r['subj'], r['obj']))[:max_pairs]
    if not gnn_rels:
        return
    print(f"  Phase 4: LLM relation classification for {len(gnn_rels)} top GNN pairs...")
    choices = ', '.join(LLM_RELATION_TYPES.values()) + ', or NONE'
    client = _get_client()
    cache = _load_cache()
    llm_calls = classified = 0
    for rel_entry in gnn_rels:
        subj, obj = rel_entry['subj'], rel_entry['obj']
        context = rel_entry.get('context', '')[:300]
        key = _cache_key('llm_relation_v3', subj, obj, context)
        if key in cache:
            answer = cache[key]
        else:
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": (
                        f"Sentence: '{context}'\n"
                        f"What is the semantic relationship between '{subj}' "
                        f"and '{obj}' in materials science? Choose exactly one: "
                        f"{choices}."
                    )}],
                    max_tokens=20,
                    temperature=0.0,
                )
                answer = (response.choices[0].message.content or '').strip()
                cache[key] = answer
                llm_calls += 1
            except Exception as e:
                print(f"    LLM error for {subj}/{obj}: {e}")
                continue
        mapped = LLM_RELATION_TYPES.get(answer.strip().rstrip('.').lower())
        if mapped:
            rel_entry['rel'] = mapped
            rel_entry['source'] = 'gnn_cooccurrence+llm'
            classified += 1
    _save_cache(cache)
    print(f"    LLM relation classification: {llm_calls} API calls, "
          f"{classified} reclassified")


def write_relations(rels):
    """Write relations_raw.csv and relations.csv (also when empty)."""
    df = pd.DataFrame(rels, columns=OUTPUT_COLUMNS)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for name in ('relations_raw.csv', 'relations.csv'):
        df.to_csv(os.path.join(PROCESSED_DIR, name), index=False)
    print(f"\nSaved {len(df)} unique relations to relations_raw.csv / relations.csv")
    if len(df):
        print("\nRelation type distribution:")
        for rel_type, count in df['rel'].value_counts().items():
            print(f"  {rel_type}: {count}")
        print("\nSource distribution:")
        for source, count in df['source'].value_counts().items():
            print(f"  {source}: {count}")


def extract_rels():
    """Extract all relations using patterns, dep parse, and GNN."""
    sents = load_sentences()
    nodemap = load_nodemap()
    gnn_embeds, gnn_nodemap = load_gnn_data()
    if not nodemap:
        print("ERROR: No nodemap found. Run builder.py first.")
        return

    matcher = TermMatcher(nodemap.keys())
    print(f"Valid terms: {len(nodemap)}")

    rels = []
    print("Extracting relations...")
    if USE_DEP_PARSING:
        print("  Phases 1+2: patterns and dependency parsing (one spaCy pass)...")
        docs = _get_nlp().pipe(sents, batch_size=100)
    else:
        print("  Phase 1: patterns (dependency parsing disabled)...")
        docs = (None for _ in sents)

    dep_extracted = []
    n_pattern = n_dep = 0
    for sent, doc in zip(sents, docs):
        for r in extract_pattern_relations(sent, matcher):
            r['context'] = sent
            rels.append(r)
            n_pattern += 1
        if doc is not None:
            for r in extract_dep_relations(doc, matcher):
                r['context'] = sent
                rels.append(r)
                dep_extracted.append(r)
                n_dep += 1
    print(f"    Pattern relations: {n_pattern}; dep parse relations: {n_dep}")

    # LLM classification of dep-parse relations (LLM_MODE)
    if LLM_MODE and dep_extracted and LLM_API_KEY:
        from llm_validator import classify_relation
        llm_classified = 0
        for rel_entry in dep_extracted:
            try:
                llm_rel = classify_relation(rel_entry['subj'], rel_entry['obj'],
                                            rel_entry['context'])
            except Exception as e:
                print(f"    LLM classification failed: {e}")
                break
            if llm_rel and llm_rel != 'NONE':
                rel_entry['rel'] = llm_rel
                rel_entry['source'] = 'dep_parse+llm'
                llm_classified += 1
        print(f"    LLM reclassified {llm_classified} dep-parse relations")

    print("  Phase 3: GNN co-occurrence relations...")
    if gnn_embeds is not None:
        counts, contexts = sentence_cooccurrence(sents, matcher)
        rels.extend(extract_gnn_cooccurrence_relations(
            gnn_nodemap or nodemap, gnn_embeds, counts, contexts,
            threshold=GNN_COOCCURRENCE_THRESHOLD,
            min_cooccurrence=GNN_MIN_COOCCURRENCE_COUNT,
        ))
        if LLM_RELATIONS and LLM_API_KEY:
            try:
                llm_classify_gnn_relations(rels)
            except Exception as e:
                print(f"  LLM relation classification failed: {e}")
    else:
        print("    GNN embeddings disabled or missing")

    # Deduplicate (keep the first occurrence of each triple)
    seen = set()
    unique_rels = []
    for r in rels:
        key = (r['subj'], r['rel'], r['obj'])
        if key not in seen:
            seen.add(key)
            unique_rels.append(r)
    write_relations(unique_rels)


if __name__ == '__main__':
    extract_rels()
