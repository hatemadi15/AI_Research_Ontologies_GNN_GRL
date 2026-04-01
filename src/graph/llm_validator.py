"""
llm_validator.py - LLM Validation Layer using OpenRouter (OpenAI-compatible)

Provides LLM-backed validation for:
  - Cluster naming: propose ontology class names for term clusters
  - Alignment tie-breaking: select best match from candidates
  - Edge/relation disambiguation: classify relation types between entities

Uses config.LLM_API_KEY / config.LLM_BASE_URL / config.LLM_MODEL for backend.
Responses are cached to avoid redundant API calls.
"""

import os
import json
import hashlib

from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

# Absolute paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CACHE_PATH = os.path.join(PROCESSED_DIR, 'llm_cache.json')

# Lazy-initialized client
_client = None


def _get_client():
    """Get or create OpenAI-compatible client (defaults to OpenRouter)."""
    global _client
    if _client is None:
        api_key = LLM_API_KEY
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY (or OPENAI_API_KEY) environment variable not set. "
                "Set it before enabling LLM features."
            )
        _client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
    return _client


# ============== Cache ==============

def _load_cache():
    """Load LLM response cache from disk."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    """Save LLM response cache to disk."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _cache_key(func_name, *args):
    """Generate a deterministic cache key from function name and args."""
    raw = json.dumps([func_name] + list(args), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ============== Cluster Naming ==============

def name_cluster(terms, top_k=10):
    """Use GPT-4o-mini to name a cluster of related terms.

    Args:
        terms: List of terms in the cluster.
        top_k: Number of top terms to send to the LLM.

    Returns:
        A concise class name (1-3 words).
    """
    sample = terms[:top_k]
    cache = _load_cache()
    key = _cache_key('name_cluster', sample)
    if key in cache:
        return cache[key]

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "system",
            "content": (
                "You are an expert in materials science ontology. "
                "Given a list of related terms, provide a single concise "
                "class name (1-3 words) that best represents the group. "
                "Respond with ONLY the class name."
            )
        }, {
            "role": "user",
            "content": f"Terms: {', '.join(sample)}"
        }],
        max_tokens=20,
        temperature=0.0,
    )
    result = response.choices[0].message.content.strip()

    cache[key] = result
    _save_cache(cache)
    return result


# ============== Alignment Tie-Breaking ==============

def confirm_alignment(discovered_term, candidate_matches):
    """Use LLM to pick the best match from candidates.

    Args:
        discovered_term: The discovered term to align.
        candidate_matches: List of dicts with 'reference' and 'similarity' keys.

    Returns:
        The selected candidate number (1-indexed) as string, or 'NONE'.
    """
    cache = _load_cache()
    key = _cache_key('confirm_alignment', discovered_term, candidate_matches)
    if key in cache:
        return cache[key]

    candidates_str = '\n'.join([
        f"  {i+1}. {m['reference']} (sim={m['similarity']:.3f})"
        for i, m in enumerate(candidate_matches)
    ])

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "system",
            "content": (
                "You are an expert in materials science ontology alignment. "
                "Given a discovered term and candidate ontology class matches, "
                "select the BEST match or respond 'NONE' if no match is correct."
            )
        }, {
            "role": "user",
            "content": (
                f"Discovered term: '{discovered_term}'\n"
                f"Candidate matches:\n{candidates_str}\n\n"
                "Respond with ONLY the number of the best match, or 'NONE'."
            )
        }],
        max_tokens=5,
        temperature=0.0,
    )
    result = response.choices[0].message.content.strip()

    cache[key] = result
    _save_cache(cache)
    return result


# ============== Edge Disambiguation ==============

RELATION_TYPES = [
    'causeOf', 'hasProperty', 'isPartOf', 'precedes', 'growsInto',
    'initiatesAt', 'measures', 'associatedWith', 'correlatedWith',
    'constrains', 'softConstrains', 'hardConstrains', 'reliesOn',
    'parallelTo', 'spatiallyCoincidesWith', 'alignedWith', 'NONE'
]


def classify_relation(subject, object_, context_sentence):
    """Use LLM to classify the relation type between two entities.

    Args:
        subject: The subject entity.
        object_: The object entity.
        context_sentence: The sentence containing both entities.

    Returns:
        One of the RELATION_TYPES strings.
    """
    cache = _load_cache()
    key = _cache_key('classify_relation', subject, object_, context_sentence)
    if key in cache:
        return cache[key]

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "system",
            "content": (
                "You are an expert in materials science. Given two entities "
                "and their context, classify their relationship as one of: "
                f"{', '.join(RELATION_TYPES)}. "
                "Respond with ONLY the relation type."
            )
        }, {
            "role": "user",
            "content": (
                f"Subject: {subject}\n"
                f"Object: {object_}\n"
                f"Context: {context_sentence}"
            )
        }],
        max_tokens=20,
        temperature=0.0,
    )
    result = response.choices[0].message.content.strip()

    cache[key] = result
    _save_cache(cache)
    return result
