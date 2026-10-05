"""
rag_typing.py - RAG-Based Term Typing using the configured LLM

Re-ranks the typing decisions of align.py for low-confidence terms:
  - Candidates: the top-5 classes of predicted_types_base.json
  - Context: the first annotated sentence mentioning the term (PubMed
    sentences as fallback), found with a word-boundary matcher
  - The LLM (config.LLM_MODEL) sees the term, the context and the candidate
    classes with their skos:definition, and answers with a candidate number
    or NONE (NONE keeps the base decision)

Writes predicted_types.json (the final typing decisions). When RAG_TYPING is
off or no LLM key is set, the base predictions are copied unchanged. Gold
annotations are never read or written here.

Inspired by SBU-NLP (#1 at LLMs4OL 2025). Limited to MAX_LLM_CALLS terms.
"""

import os
import re
import json

from config import (LLM_API_KEY, LLM_MODEL, PREDICTED_TYPES_BASE_FILE,
                    PREDICTED_TYPES_FILE, PROCESSED_DIR, RAG_TYPING)
from ontology_utils import load_ontology
from text_match import TermMatcher

MAX_LLM_CALLS = 300
# Terms whose top-1 typing score is below this are re-ranked by the LLM
CONFIDENCE_THRESHOLD = 0.6


def load_base_predictions():
    path = os.path.join(PROCESSED_DIR, PREDICTED_TYPES_BASE_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run align.py first.")
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_predictions(preds):
    path = os.path.join(PROCESSED_DIR, PREDICTED_TYPES_FILE)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    print(f"Saved {PREDICTED_TYPES_FILE}: {len(preds)} terms")


def load_sentences(name):
    path = os.path.join(PROCESSED_DIR, name)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return []


def find_contexts(terms):
    """First sentence mentioning each term: annotated papers, then PubMed."""
    matcher = TermMatcher(terms)
    context = {}
    for name in ('conll_sentences.txt', 'all_sentences.txt'):
        for sent in load_sentences(name):
            for term in matcher.terms_in(sent):
                context.setdefault(term, sent)
            if len(context) == len(terms):
                return context
    return context


def parse_choice(answer, candidates):
    """Map an LLM answer to one of the candidate URIs, or None.

    Accepts a candidate number, or an exact candidate label. A label that is
    merely contained in the answer is not enough ("Fatigue" must not win when
    the answer is "Fatigue strength").
    """
    answer = (answer or '').strip()
    if not answer or answer.upper().startswith('NONE'):
        return None
    m = re.match(r'\D*(\d+)', answer)
    if m and 1 <= int(m.group(1)) <= len(candidates):
        return candidates[int(m.group(1)) - 1][0]
    cleaned = answer.strip(' ."\'').lower()
    for uri, label, _ in candidates:
        if label.lower() == cleaned:
            return uri
    return None


def run_rag_typing():
    """Re-rank low-confidence typing decisions with the LLM."""
    base = load_base_predictions()
    preds = {t: dict(p) for t, p in base.items()}

    if not RAG_TYPING:
        print("RAG_TYPING disabled, keeping the base typing decisions")
        save_predictions(preds)
        return
    if not LLM_API_KEY:
        print("LLM API key not set, keeping the base typing decisions")
        save_predictions(preds)
        return

    from llm_validator import _get_client, _load_cache, _save_cache, _cache_key

    onto = load_ontology()
    candidates = sorted((t for t, p in base.items() if p['score'] < CONFIDENCE_THRESHOLD),
                        key=lambda t: (base[t]['score'], t))[:MAX_LLM_CALLS]
    print(f"RAG typing: {len(candidates)}/{len(base)} terms below confidence "
          f"{CONFIDENCE_THRESHOLD}")
    contexts = find_contexts(candidates)

    client = _get_client()
    cache = _load_cache()
    llm_calls = changed = kept_none = 0
    for term in candidates:
        options = base[term]['top5']
        lines = []
        for n, (uri, label, _) in enumerate(options, 1):
            definition = onto.definition(uri)[:200] if uri in onto.classes else ''
            lines.append(f"{n}. {label}" + (f" - {definition}" if definition else ''))
        context = contexts.get(term, '(no context available)')
        key = _cache_key('rag_typing_v2', term, [o[0] for o in options], context)
        if key in cache:
            answer = cache[key]
        else:
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Given the materials science term '{term}' appearing "
                            f"in this context: '{context[:300]}'.\n"
                            f"Which ontology class does the term belong to?\n"
                            + '\n'.join(lines) +
                            "\nAnswer with ONLY the number of the best class, "
                            "or NONE if none fits."
                        )
                    }],
                    max_tokens=5,
                    temperature=0.0,
                )
                answer = (response.choices[0].message.content or '').strip()
                cache[key] = answer
                llm_calls += 1
            except Exception as e:
                print(f"  LLM error for term '{term}': {e}")
                continue

        choice = parse_choice(answer, options)
        if choice is None:
            preds[term]['llm_none'] = True
            kept_none += 1
            continue
        chosen = next(o for o in options if o[0] == choice)
        if choice != base[term]['class_uri']:
            changed += 1
        preds[term].update({'class_uri': chosen[0], 'label': chosen[1],
                            'score': chosen[2], 'source': 'rag',
                            'base_class_uri': base[term]['class_uri']})

    _save_cache(cache)
    save_predictions(preds)
    print(f"  RAG typing: {llm_calls} API calls, {changed} decisions changed, "
          f"{kept_none} answered NONE (base decision kept)")


if __name__ == '__main__':
    run_rag_typing()
