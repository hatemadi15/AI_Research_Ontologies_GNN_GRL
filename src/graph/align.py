"""
align.py - Term Typing and MILA-style Ontology Alignment

Every discovered term (graph node) is scored against every class of the
reference ontology (ontology.ttl, all labels incl. alternative labels):

  S[term, class] = max over the class labels of
      0.45 * embedding cosine + 0.15 * Jaccard + 0.15 * edit similarity
      + 0.20 * exact/substring match + 0.05 * taxonomy in-degree bonus
  (USE_COMBINED_SCORING=false: embedding cosine only)

GNN term-side smoothing (USE_GNN_EMBEDDINGS, GNN_SMOOTHING_BETA): classes have
no GNN vectors, so the GNN contributes by mixing in the class scores of a
term's nearest graph neighbours: S' = (1 - beta) * S + beta * W @ S, with W the
row-normalised top-k neighbour matrix by GNN cosine.

Outputs (config.PROCESSED_DIR):
  - predicted_types_base.json: per term the top-5 classes; the top-1 is the
    pipeline's typing decision (rag_typing.py may re-rank low-confidence ones)
  - ontology_alignment.csv: alignment rows (1-to-1 bidirectional matches,
    multi-class rows, structural/LLM expansion rows) with their score kind;
    these rows are an alignment artifact and are scored pairwise by eval_f1
  - alignment_config.json

Gold NER types are never read unless ORACLE_TYPES is set, in which case the
scores of each term's gold classes are raised to >= ORACLE_SCORE (upper bound).
"""

import os
import json
import re
import pickle
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from config import (EMBEDDER_MODEL, GNN_SMOOTHING_BETA, GNN_SMOOTHING_K,
                    GOLD_TYPES_FILE, LLM_ALIGNMENT, LLM_API_KEY, LLM_MODEL,
                    ORACLE_TYPES, PREDICTED_TYPES_BASE_FILE, PROCESSED_DIR,
                    USE_BIDIRECTIONAL, USE_COMBINED_SCORING, USE_GNN_EMBEDDINGS)
from ontology_utils import load_ontology

TAXONOMY_PATH = os.path.join(PROCESSED_DIR, 'taxonomy_graph.pkl')
NODEMAP_PATH = os.path.join(PROCESSED_DIR, 'nodemap_fine_auto.pt')
GNN_EMBED_PATH = os.path.join(PROCESSED_DIR, 'gnn_embeddings.npy')
GNN_MAP_PATH = os.path.join(PROCESSED_DIR, 'gnn_embed_map.json')

ORACLE_SCORE = 0.85        # score floor of gold classes in ORACLE mode
MULTI_CLASS_MIN_COSINE = 0.40
TOP_K = 5

_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(EMBEDDER_MODEL)
    return _EMBEDDER


def _encode(texts):
    vectors = np.asarray(_get_embedder().encode(list(texts), show_progress_bar=False),
                         dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def normalize_term(term):
    """Normalize a term: CamelCase split, remove hyphens, lowercase, strip articles.

    Handles compound NER types like "CrackGrowthBehaviour" -> "crack growth behaviour"
    and ontology labels like "High-cycle fatigue" -> "high cycle fatigue".
    """
    term = term.strip()
    # CamelCase splitting (handles "CrackGrowth" and "GNNModel")
    term = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', term)
    term = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', term)
    term = term.lower()
    # Replace hyphens and underscores with spaces
    term = re.sub(r'[-_]', ' ', term)
    # Strip leading articles
    for prefix in ('the ', 'a ', 'an '):
        if term.startswith(prefix):
            term = term[len(prefix):]
    # Remove parentheticals
    term = re.sub(r'\([^)]*\)', '', term)
    # Collapse multiple spaces
    term = re.sub(r'\s+', ' ', term).strip()
    return term


def jaccard_similarity(s1, s2):
    """Token-level Jaccard similarity."""
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def edit_distance_similarity(s1, s2):
    """Normalized edit distance similarity using SequenceMatcher."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def load_terms():
    """Discovered terms in node order (graph nodes from builder.py)."""
    import torch
    nodemap = torch.load(NODEMAP_PATH, weights_only=False)
    return sorted(nodemap, key=nodemap.get)


def load_taxonomy():
    """Taxonomy graph (for the in-degree structure bonus), or None."""
    if not os.path.exists(TAXONOMY_PATH):
        print("Warning: taxonomy_graph.pkl not found; no structure bonus")
        return None
    with open(TAXONOMY_PATH, 'rb') as f:
        G = pickle.load(f)
    print(f"Taxonomy: {G.number_of_nodes()} nodes, {G.number_of_edges()} isa edges")
    return G


def class_label_table(onto):
    """Flatten (class index, label) pairs over all labels of all classes."""
    owners, labels = [], []
    for j, uri in enumerate(onto.uris):
        for label in onto.labels(uri):
            owners.append(j)
            labels.append(label)
    return np.array(owners), labels


def score_matrix(terms, onto, G=None):
    """Combined term x class score matrix (max over class labels).

    Returns (S, E): S the scores used for typing, E the plain embedding cosine
    (max over labels), used for the multi-class alignment rows.
    """
    owners, labels = class_label_table(onto)
    term_emb = _encode(terms)
    label_emb = _encode(labels)
    cos = term_emb @ label_emb.T                      # terms x labels

    n_terms, n_classes = len(terms), len(onto.uris)
    E = np.full((n_terms, n_classes), -1.0)
    np.maximum.at(E.T, owners, cos.T)

    if not USE_COMBINED_SCORING:
        return E.copy(), E

    term_norm = [normalize_term(t) for t in terms]
    label_norm = [normalize_term(lab) for lab in labels]
    struct = np.zeros(n_terms)
    if G is not None:
        struct = np.array([min(G.in_degree(t) / 10.0, 1.0) if t in G else 0.0
                           for t in terms])

    combined = 0.45 * cos + 0.05 * struct[:, None]
    for i, d_norm in enumerate(term_norm):
        for k, r_norm in enumerate(label_norm):
            exact = 1.0 if d_norm == r_norm else (
                0.5 if (d_norm in r_norm or r_norm in d_norm) else 0.0)
            combined[i, k] += (0.15 * jaccard_similarity(d_norm, r_norm)
                               + 0.15 * edit_distance_similarity(d_norm, r_norm)
                               + 0.20 * exact)
    S = np.full((n_terms, n_classes), -1.0)
    np.maximum.at(S.T, owners, combined.T)
    return S, E


def gnn_neighbor_matrix(terms, k=GNN_SMOOTHING_K):
    """Row-normalised top-k neighbour weights by GNN cosine, or None."""
    if not (os.path.exists(GNN_EMBED_PATH) and os.path.exists(GNN_MAP_PATH)):
        return None
    gnn_embeds = np.load(GNN_EMBED_PATH)
    with open(GNN_MAP_PATH) as f:
        gnn_map = json.load(f)
    if not all(t in gnn_map for t in terms):
        print("GNN embeddings do not cover all terms; no smoothing")
        return None
    Z = np.stack([gnn_embeds[gnn_map[t]] for t in terms]).astype(float)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    sim = Z @ Z.T
    np.fill_diagonal(sim, -np.inf)
    W = np.zeros_like(sim)
    for i in range(len(terms)):
        idx = np.argsort(-sim[i], kind='stable')[:k]
        w = np.clip(sim[i, idx], 0.0, None)
        if w.sum() > 0:
            W[i, idx] = w / w.sum()
        else:
            W[i, i] = 1.0
    return W


def load_gold_class_indices(terms, onto):
    """ORACLE only: indices of each term's gold classes."""
    with open(os.path.join(PROCESSED_DIR, GOLD_TYPES_FILE)) as f:
        gold_types = json.load(f)
    index = {u: j for j, u in enumerate(onto.uris)}
    gold = {}
    for t in terms:
        uris = [onto.resolve(ty) for ty in gold_types.get(t, {})]
        gold[t] = [index[u] for u in uris if u in index]
    return gold


def bidirectional_alignment(S, terms, onto):
    """MILA-style bidirectional matching with a 1-to-1 constraint.

    1. Forward: each term -> best class; 2. Backward: each class -> best term;
    3. Mutual best matches first, then remaining forward, then backward.
    With USE_BIDIRECTIONAL=false every term keeps its forward match.
    """
    uris = onto.uris
    forward = {i: int(np.argmax(S[i])) for i in range(len(terms))}

    def row(i, j, direction):
        return {'discovered': terms[i], 'reference': onto.primary_label(uris[j]),
                'class_uri': uris[j], 'similarity': float(S[i, j]),
                'direction': direction, 'score_kind': 'typing'}

    if not USE_BIDIRECTIONAL:
        return [row(i, j, 'forward') for i, j in forward.items()]

    backward = {j: int(np.argmax(S[:, j])) for j in range(len(uris))}
    mutual = [(i, j) for i, j in forward.items() if backward[j] == i]
    used_terms, used_classes, rows = set(), set(), []
    for i, j in sorted(mutual, key=lambda p: -S[p]):
        rows.append(row(i, j, 'mutual'))
        used_terms.add(i)
        used_classes.add(j)
    for i, j in sorted(forward.items(), key=lambda p: -S[p]):
        if i not in used_terms and j not in used_classes:
            rows.append(row(i, j, 'forward'))
            used_terms.add(i)
            used_classes.add(j)
    for j, i in sorted(backward.items(), key=lambda p: -S[p[1], p[0]]):
        if j not in used_classes and i not in used_terms:
            rows.append(row(i, j, 'backward'))
            used_terms.add(i)
            used_classes.add(j)
    return rows


def multi_class_rows(E, terms, onto, existing):
    """All classes with embedding cosine >= MULTI_CLASS_MIN_COSINE per term."""
    rows = []
    for i, term in enumerate(terms):
        for j in np.argsort(-E[i], kind='stable'):
            if E[i, j] < MULTI_CLASS_MIN_COSINE:
                break
            uri = onto.uris[j]
            if (term, uri) in existing:
                continue
            existing.add((term, uri))
            rows.append({'discovered': term, 'reference': onto.primary_label(uri),
                         'class_uri': uri, 'similarity': float(E[i, j]),
                         'direction': 'multi_class', 'score_kind': 'cosine'})
    return rows


def expand_ontology_coverage_structural(terms, term_emb, onto, E):
    """Non-LLM coverage expansion for classes no term is close to.

    For each class whose best term cosine is < 0.5, the class's own labels and
    its skos:definition are embedded; the best-matching term (cosine >= 0.45)
    becomes an alignment row. Parent/child labels are no longer used as
    evidence: a term matching the label of a superclass ("microstructure")
    is not an instance of the subclass ("Bainite").
    """
    rows = []
    unmatched = [j for j in range(len(onto.uris)) if E[:, j].max() < 0.5]
    print(f"Structural expansion: {len(unmatched)}/{len(onto.uris)} classes to expand")
    for j in unmatched:
        uri = onto.uris[j]
        fragments = [normalize_term(lab) for lab in onto.labels(uri)]
        if onto.definition(uri):
            fragments.append(onto.definition(uri))
        frag_emb = _encode(fragments)
        sims = frag_emb @ term_emb.T
        best = float(sims.max())
        if best >= 0.45:
            _, t_idx = np.unravel_index(int(np.argmax(sims)), sims.shape)
            rows.append({'discovered': terms[t_idx],
                         'reference': onto.primary_label(uri), 'class_uri': uri,
                         'similarity': round(best, 4),
                         'direction': 'structural_expansion', 'score_kind': 'cosine'})
    print(f"  Structural expansion: {len(rows)} new alignment rows")
    return rows


def llm_augment_alignment(alignments, onto):
    """LLM-Augmented Alignment of borderline rows.

    For rows with similarity 0.35-0.65 the LLM is asked whether the term is
    related to the class (label + definition): YES -> +0.15, NO -> -0.15.
    Limited to 500 rows, most ambiguous first. Only alignment rows change;
    typing decisions are re-ranked by rag_typing.py instead.
    """
    try:
        from llm_validator import _get_client, _load_cache, _save_cache, _cache_key
    except ImportError:
        print("llm_validator not available, skipping LLM augmentation")
        return alignments

    borderline = [(i, a) for i, a in enumerate(alignments)
                  if 0.35 <= a['similarity'] <= 0.65]
    borderline.sort(key=lambda x: (abs(x[1]['similarity'] - 0.5),
                                   x[1]['discovered'], x[1]['class_uri']))
    borderline = borderline[:500]
    if not borderline:
        print("No borderline pairs found for LLM augmentation")
        return alignments

    print(f"LLM-augmenting {len(borderline)} borderline alignment pairs...")
    cache = _load_cache()
    client = _get_client()
    llm_calls = boosted = reduced = 0
    for idx, alignment in borderline:
        entity = alignment['discovered']
        uri = alignment['class_uri']
        class_label = onto.primary_label(uri)
        definition = onto.definition(uri)[:200] or ', '.join(onto.labels(uri))
        key = _cache_key('llm_alignment_v2', entity, uri)
        if key in cache:
            answer = cache[key]
        else:
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"In materials science, is the term '{entity}' an "
                            f"instance or example of the ontology class "
                            f"'{class_label}' ({definition})? Answer YES or NO."
                        )
                    }],
                    max_tokens=5,
                    temperature=0.0,
                )
                answer = (response.choices[0].message.content or '').strip().upper()
                cache[key] = answer
                llm_calls += 1
            except Exception as e:
                print(f"  LLM error for {entity}/{class_label}: {e}")
                continue
        if answer.startswith('YES'):
            alignments[idx]['similarity'] = min(1.0, alignment['similarity'] + 0.15)
            boosted += 1
        elif answer.startswith('NO'):
            alignments[idx]['similarity'] = max(0.0, alignment['similarity'] - 0.15)
            reduced += 1
    _save_cache(cache)
    print(f"  LLM alignment: {llm_calls} API calls, {boosted} boosted, {reduced} reduced")
    return alignments


def expand_ontology_coverage(terms, term_emb, onto, E):
    """LLM-Based Ontology Class Expansion (needs an LLM API key).

    For classes no term is close to (cosine < 0.5), the LLM proposes three
    synonyms; the best term matching a synonym (cosine > 0.55) becomes an
    alignment row.
    """
    try:
        from llm_validator import _get_client, _load_cache, _save_cache, _cache_key
    except ImportError:
        print("llm_validator not available, skipping ontology expansion")
        return []

    unmatched = [j for j in range(len(onto.uris)) if E[:, j].max() < 0.5][:500]
    print(f"LLM Ontology expansion: {len(unmatched)}/{len(onto.uris)} classes unmatched")
    if not unmatched:
        return []
    client = _get_client()
    cache = _load_cache()
    rows = []
    llm_calls = 0
    for j in unmatched:
        uri = onto.uris[j]
        class_label = onto.primary_label(uri)
        key = _cache_key('expand_class', class_label)
        if key in cache:
            synonyms_str = cache[key]
        else:
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": (
                        f"Generate 3 short synonym phrases for the materials "
                        f"science concept '{class_label}'. Return only the "
                        f"phrases, comma-separated."
                    )}],
                    max_tokens=60, temperature=0.0,
                )
                synonyms_str = (response.choices[0].message.content or '').strip()
                cache[key] = synonyms_str
                llm_calls += 1
            except Exception as e:
                print(f"  LLM error for class '{class_label}': {e}")
                continue
        synonyms = [s.strip() for s in synonyms_str.split(',') if s.strip()]
        if not synonyms:
            continue
        syn_sims = _encode(synonyms) @ term_emb.T
        for si in range(len(synonyms)):
            best_idx = int(np.argmax(syn_sims[si]))
            if syn_sims[si, best_idx] > 0.55:
                rows.append({'discovered': terms[best_idx], 'reference': class_label,
                             'class_uri': uri,
                             'similarity': round(float(syn_sims[si, best_idx]), 4),
                             'direction': 'llm_expansion', 'score_kind': 'cosine'})
                break
    _save_cache(cache)
    print(f"  LLM Expansion: {llm_calls} API calls, {len(rows)} new alignments")
    return rows


def predicted_types(S, terms, onto, source):
    """Top-k classes per term from the score matrix (many-to-one typing)."""
    out = {}
    for i, term in enumerate(terms):
        order = np.argsort(-S[i], kind='stable')[:TOP_K]
        top = [[onto.uris[j], onto.primary_label(onto.uris[j]), round(float(S[i, j]), 4)]
               for j in order]
        out[term] = {'class_uri': top[0][0], 'label': top[0][1], 'score': top[0][2],
                     'top5': top, 'source': source}
    return out


def run_alignment():
    """Type every discovered term and build the alignment rows."""
    terms = load_terms()
    G = load_taxonomy()
    onto = load_ontology()
    print(f"\nTyping/aligning {len(terms)} discovered terms -> "
          f"{len(onto.uris)} ontology classes "
          f"({'combined' if USE_COMBINED_SCORING else 'embedding-only'} scoring)")

    S, E = score_matrix(terms, onto, G)

    smoothing = 0.0
    if USE_GNN_EMBEDDINGS and GNN_SMOOTHING_BETA > 0:
        W = gnn_neighbor_matrix(terms)
        if W is not None:
            smoothing = GNN_SMOOTHING_BETA
            S = (1 - smoothing) * S + smoothing * (W @ S)
            print(f"GNN term-side smoothing: beta={smoothing}, k={GNN_SMOOTHING_K}")

    source = 'align'
    if ORACLE_TYPES:
        print("[ORACLE] raising the scores of gold classes to >= "
              f"{ORACLE_SCORE} (upper bound, not a model result)")
        for i, idx in enumerate(load_gold_class_indices(terms, onto).values()):
            if idx:
                S[i, idx] = np.maximum(S[i, idx], ORACLE_SCORE)
        source = 'align+oracle'

    preds = predicted_types(S, terms, onto, source)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, PREDICTED_TYPES_BASE_FILE), 'w',
              encoding='utf-8') as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    print(f"Saved {PREDICTED_TYPES_BASE_FILE}: {len(preds)} typed terms, "
          f"{len(set(p['class_uri'] for p in preds.values()))} distinct classes")

    # ---- Alignment rows (artifact; scored pairwise by eval_f1) ----
    alignments = bidirectional_alignment(S, terms, onto)
    existing = {(a['discovered'], a['class_uri']) for a in alignments}
    alignments.extend(multi_class_rows(E, terms, onto, existing))

    term_emb = _encode(terms)
    try:
        structural = expand_ontology_coverage_structural(terms, term_emb, onto, E)
        alignments.extend(structural)
    except Exception as e:
        print(f"Structural expansion failed: {e}")

    if LLM_ALIGNMENT and LLM_API_KEY:
        try:
            alignments = llm_augment_alignment(alignments, onto)
        except Exception as e:
            print(f"LLM alignment augmentation failed: {e}")
        try:
            alignments.extend(expand_ontology_coverage(terms, term_emb, onto, E))
        except Exception as e:
            print(f"LLM ontology expansion failed: {e}")

    df = pd.DataFrame(alignments).sort_values(
        ['similarity', 'discovered', 'class_uri'], ascending=[False, True, True])
    df.to_csv(os.path.join(PROCESSED_DIR, 'ontology_alignment.csv'), index=False)

    direction_counts = df['direction'].value_counts().to_dict()
    avg_sim = float(df['similarity'].mean()) if len(df) else 0.0
    print("\nAlignment rows:")
    print(f"  Total: {len(df)}")
    for direction, count in sorted(direction_counts.items()):
        print(f"  {direction}: {count}")
    print(f"  Saved to {os.path.join(PROCESSED_DIR, 'ontology_alignment.csv')}")

    config = {
        'USE_GNN_EMBEDDINGS': USE_GNN_EMBEDDINGS,
        'USE_BIDIRECTIONAL': USE_BIDIRECTIONAL,
        'USE_COMBINED_SCORING': USE_COMBINED_SCORING,
        'GNN_SMOOTHING_BETA': smoothing,
        'ORACLE_TYPES': ORACLE_TYPES,
        'n_terms': len(terms),
        'n_classes': len(onto.uris),
        'n_alignment_rows': len(df),
        'direction_counts': direction_counts,
        'avg_similarity': round(avg_sim, 4),
    }
    with open(os.path.join(PROCESSED_DIR, 'alignment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    return df.head(20)


if __name__ == '__main__':
    top = run_alignment()
    if top is not None:
        print("\nTop Matches:")
        print(top.to_string())
