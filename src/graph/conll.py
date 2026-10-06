"""
conll.py - CoNLL BIO readers shared by the pipeline, evaluation and baselines.

The MaterioMiner files have one "token tag" pair per line and blank lines
between sentences. Two views of the same files are provided:

  - read_conll_sentences(): (tokens, tags) per sentence, for NER training
  - parse_conll_bio(): reconstructed sentence strings plus the
    (entity_text, entity_type) spans per sentence, for graph building and the
    term-typing gold standard

Span semantics (kept identical to the original builder.py parser): a B- tag
opens an entity, an I- tag of the same type extends it, and anything else
(O, an I- tag of another type, an I- tag with no open entity) closes it.
Entity texts are lowercased and must be at least 2 characters long.

This module only uses the standard library so it can be imported by tests and
CI without the heavy ML dependencies.
"""

import os
from collections import Counter, defaultdict

# Minimum number of sentences a term must occur in to become a graph node.
# Shared by builder.py (node set) and the evaluation (term universe).
MIN_TERM_DF = 2


def read_conll_sentences(filepath):
    """Read a CoNLL file into a list of (tokens, tags) sentences.

    The tag is the last whitespace-separated column; lines with fewer than two
    columns are skipped. There are no comment lines in these files: '#' is a
    real token (e.g. "# B-StrainAmplitude" in 10.3390_met10111458.conll), so
    it must not be treated as a comment marker.
    """
    sentences = []
    tokens, tags = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            tokens.append(parts[0])
            tags.append(parts[-1])
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def bio_spans(tokens, tags):
    """Extract (entity_text, entity_type) spans from one BIO-tagged sentence."""
    spans = []
    ent_tokens = []
    ent_type = None

    def flush():
        nonlocal ent_tokens, ent_type
        if ent_tokens and ent_type:
            entity_text = ' '.join(ent_tokens).lower().strip()
            if len(entity_text) >= 2:
                spans.append((entity_text, ent_type))
        ent_tokens = []
        ent_type = None

    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            flush()
            ent_type = tag[2:]
            ent_tokens = [token]
        elif tag.startswith('I-') and ent_type:
            if tag[2:] == ent_type:
                ent_tokens.append(token)
            else:
                flush()
        else:
            flush()
    flush()
    return spans


def parse_conll_bio(filepath):
    """Parse a CoNLL file extracting BIO-tagged entities with their types.

    Returns:
        sentences: list of reconstructed sentence strings
        doc_entities: list of (entity_text, entity_type) lists, one per sentence
    """
    sentences = []
    doc_entities = []
    for tokens, tags in read_conll_sentences(filepath):
        sentences.append(' '.join(tokens))
        doc_entities.append(bio_spans(tokens, tags))
    return sentences, doc_entities


def load_corpus(datadir):
    """Parse every .conll file in a directory (sorted by name).

    Returns:
        sentences: list of sentence strings
        doc_entities: list of (entity_text, entity_type) lists per sentence
        doc_ids: list with the source file name of each sentence
    """
    files = sorted(f for f in os.listdir(datadir) if f.endswith('.conll'))
    sentences, doc_entities, doc_ids = [], [], []
    for filename in files:
        sents, ents = parse_conll_bio(os.path.join(datadir, filename))
        sentences.extend(sents)
        doc_entities.extend(ents)
        doc_ids.extend([filename] * len(sents))
    return sentences, doc_entities, doc_ids


def term_document_frequency(doc_entities):
    """Number of sentences each entity text occurs in."""
    term_df = Counter()
    for sent_ents in doc_entities:
        for term in set(ent_text for ent_text, _ in sent_ents):
            term_df[term] += 1
    return term_df


def valid_terms(doc_entities, min_df=MIN_TERM_DF):
    """Sorted list of entity texts occurring in at least `min_df` sentences."""
    term_df = term_document_frequency(doc_entities)
    return sorted(t for t, c in term_df.items() if c >= min_df)


def term_type_counts(doc_entities):
    """Map entity text -> Counter of annotated types (mention counts)."""
    counts = defaultdict(Counter)
    for sent_ents in doc_entities:
        for ent_text, ent_type in sent_ents:
            counts[ent_text][ent_type] += 1
    return counts


def type_frequencies(doc_entities):
    """Corpus-wide mention count of each entity type."""
    return Counter(ent_type for sent_ents in doc_entities
                   for _, ent_type in sent_ents)


def majority_type(type_counter, global_freq=None):
    """Most frequent type of one term.

    Ties (15 of the 315 graph terms have one) are broken by the corpus-wide
    frequency of the type, then alphabetically, so the result is deterministic.
    """
    global_freq = global_freq or {}
    return min(type_counter.items(),
               key=lambda kv: (-kv[1], -global_freq.get(kv[0], 0), kv[0]))[0]


def majority_types(doc_entities, terms=None):
    """Map entity text -> majority annotated type (restricted to `terms`)."""
    counts = term_type_counts(doc_entities)
    global_freq = type_frequencies(doc_entities)
    keys = counts if terms is None else [t for t in terms if t in counts]
    return {t: majority_type(counts[t], global_freq) for t in keys}
