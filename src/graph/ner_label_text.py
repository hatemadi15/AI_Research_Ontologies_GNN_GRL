"""
ner_label_text.py - label descriptions for label-semantic NER (ner_baseline.py --label-semantics)

With --label-semantics the tags of a label X are scored with an embedding of a text
that describes X (ROADMAP §4, week-3 protocol, variant 2). This module builds those
texts:

  - Fine-grained labels are MMO classes (IRI MMO_NS + name). A label is described by
    the class name in words, ': ' and the class's skos:definition, its whitespace
    collapsed: "Crack Initation: The crack initiation stage is reached ...". The name
    keeps the class's own spelling. A class without a definition is described by its
    name in words alone.
  - Coarse-grained types (MAT, VAL, ...) have no definitions. A type is described by
    the names of the MMO classes annotated with it in the fine-grained files, as
    "<CODE>: <name>, <name>, ...". Only the sentences tokenised identically in the
    two files count (461 of 476); in them, every fine-grained entity is paired with
    the coarse-grained entity of exactly the same token span, if there is one. All
    fine-grained classes count, not only the top 27. The names are listed by
    decreasing number of paired entities, ties broken by class name, at most
    MAX_NAMES of them, each in words. A type with no paired class is described by its
    code alone.

Only the standard library and rdflib (through ontology_utils) are needed, so the
tests run in CI without torch.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict

from ontology_utils import MMO_NS, load_ontology

MAX_NAMES = 10      # class names in the description of a coarse-grained type


def split_name(name):
    """A class name in words, spelling kept: 'Crack Initation' for CrackInitation, 'SN Curve'."""
    return re.sub(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', ' ', name)


def mmo_definitions(names):
    """{name: skos:definition} of the MMO classes `names` ('' for a class without one)."""
    classes = load_ontology().classes
    return {name: classes.get(MMO_NS + name, {}).get('definition', '') for name in names}


def fine_descriptions(labels, definitions=None):
    """Descriptions of fine-grained labels, in the order of `labels`.

    definitions: {name: definition}, by default those of the MMO (mmo_definitions).
    """
    if definitions is None:
        definitions = mmo_definitions(labels)
    out = []
    for label in labels:
        definition = ' '.join((definitions.get(label) or '').split())
        out.append(f'{split_name(label)}: {definition}' if definition else split_name(label))
    return out


def entity_spans(tags):
    """(start, end, type) of the entities of an IOB2 tag sequence, end exclusive.

    An I-X continues an entity of type X that ends just before it; any other I-X
    starts an entity, as B-X does (seqeval's default mode).
    """
    spans = []
    for i, tag in enumerate(tags):
        if tag.startswith('I-') and spans and spans[-1][1:] == (i, tag[2:]):
            spans[-1] = (spans[-1][0], i + 1, tag[2:])
        elif tag != 'O':
            spans.append((i, i + 1, tag[2:]))
    return spans


def coarse_classes(fine, coarse):
    """{coarse-grained type: Counter of the fine-grained classes paired with it}.

    fine, coarse: the same sentences as (tokens, tags, ...) of each corpus, in the
    same order. Sentences tokenised differently are skipped; in the others, a
    fine-grained entity is paired with the coarse-grained entity of exactly the same
    token span.
    """
    classes = defaultdict(Counter)
    for (f_tokens, f_tags, *_), (c_tokens, c_tags, *_) in zip(fine, coarse, strict=True):
        if f_tokens != c_tokens:
            continue
        types = {(start, end): kind for start, end, kind in entity_spans(c_tags)}
        for start, end, name in entity_spans(f_tags):
            if (start, end) in types:
                classes[types[start, end]][name] += 1
    return classes


def coarse_descriptions(types, fine, coarse, max_names=MAX_NAMES):
    """Descriptions of coarse-grained types, in the order of `types` (rule: module docstring).

    fine, coarse: the corpora with all their classes, as for coarse_classes.
    """
    classes = coarse_classes(fine, coarse)
    out = []
    for code in types:
        ranked = sorted(classes.get(code, {}).items(), key=lambda kv: (-kv[1], kv[0]))
        names = ', '.join(split_name(name) for name, _ in ranked[:max_names])
        out.append(f'{code}: {names}' if names else code)
    return out


def descriptions_sha256(labels, descriptions):
    """SHA-256 of the [label, description] pairs in order, as UTF-8 JSON (stored with the results)."""
    pairs = [[label, text] for label, text in zip(labels, descriptions, strict=True)]
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode('utf-8')).hexdigest()
