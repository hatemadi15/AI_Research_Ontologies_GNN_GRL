"""
mulms.py - the MuLMS corpus as flat CoNLL files (docs/ROADMAP.md, "Candidates for week 10",
item 2)

MuLMS (Schrader et al., 2023; Hugging Face dataset timo-pierre-schrader/MuLMS, CC BY-SA 4.0)
annotates 13 entity types in the sentences of 50 open-access articles from seven subfields of
materials science. The release used is that of REVISION: three parquet files (FILES), one row
per sentence with
  - doc_id (e.g. pemfc_03, oastm_01), category (the subfield: electrolysis, graphene, pemfc,
    poly, semi, sofc or steel) and data_split (train1..train5, dev or test), the same for
    every sentence of a document;
  - tokens, and beginOffset, where the sentence begins in its document: the sentences are put
    in text order by int(beginOffset);
  - NER_labels, lists with one item per entity, of which two are used: 'value', the type, and
    'tokenIndices', the entity's first and last token (inclusive). Its character offsets
    ('begin', 'end') are strings and are not used.

The entities are nested (NUM and UNIT inside VALUE, MAT inside MAT) and a few cross; our
taggers are flat, one IOB2 tag per token. Each sentence's entities are therefore flattened by
a fixed rule, in this order:
  1. outermost: an entity strictly inside another of the sentence (first2 <= first,
     last <= last2, the spans not equal) is dropped ('nested');
  2. equal spans: of the entities left that share a span, the one whose type comes first in
     TYPE_ORDER is kept; the others, exact duplicates (same span and type) included, are
     dropped ('same_span');
  3. crossing: the entities left are taken by start, longer first, and one that overlaps an
     entity already kept is dropped ('crossing'), so of two crossing entities the one that
     starts first stays;
  4. IOB2: B-TYPE on the first token of a kept entity, I-TYPE on its other tokens, O elsewhere.
Step 1 looks at every entity of the sentence, so an entity inside a crossing entity that step 3
drops stays dropped (as nested): its tokens are O unless a kept entity covers them.

Some tokens are damaged by the text extraction: empty, a lone ' ' where a minus or degree sign
was lost, non-breaking spaces, fragments with spaces inside ('The cataly'), private-use symbol
characters, lone combining marks. A "token tag" line cannot hold whitespace, and an encoder
gives a word it erases no subword, so the word could get no tag. Before the IOB2 step each
sentence's tokens are therefore replaced by a second fixed rule (TOKEN_RULE, retokenize):
  5. each token becomes its whitespace-separated pieces (str.split), and a piece made only of
     characters that a BERT tokenizer erases (invisible: Unicode categories Cc, Cf, Co and Cs,
     i.e. control, format, private use and surrogate characters, and Mn, the combining marks
     that uncased models such as MatSciBERT strip with the accents; U+FFFD) is dropped. A kept
     entity moves to the pieces of its tokens, and one left without pieces is dropped
     ('no_visible_token'). A sentence left without pieces stops the conversion. The categories
     are those of Python's unicodedata, whose version the report records.

Outputs, written only when every check passes:
  - <out>/<doc_id>.conll: the document's sentences in text order, one "token tag" line per
    token, a blank line after each sentence;
  - <out>/split.json {doc_id: data_split} and <out>/subfields.json {doc_id: category};
  - the report (results/mulms_corpus.json): the source files and their SHA-256, the rules, the
    counts (entities in, kept and dropped by reason; tokens in, out and changed by the token
    rule; kept per type; per subfield), the SHA-256 of every output file ('outputs', {name in
    <out>: SHA-256}, which kaggle_jobs.py reads) and a combined digest ('outputs_sha256'), the
    SHA-256 of the lines "<name> <sha256>\n" sorted by name. It has no timestamps: a re-run
    gives the same report.
The counts of the corpus are checked first (EXPECTED: 50 documents, 10,184 sentences and 46,584
entities of 13 types).

The parquet files are downloaded once to <cache> and checked against the SHA-256 of FILES, in
the cache too; --offline never uses the network. Under share-alike, the converted files stay
out of git (data/processed/ is ignored) and only the report is committed. pyarrow is imported
by read_rows only, so that the tests run without it.

Usage:
  python src/graph/mulms.py [--cache DIR] [--out DIR] [--report PATH] [--offline]
"""

import argparse
import hashlib
import json
import os
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / 'data' / 'processed' / 'mulms_cache'
OUT_DIR = ROOT / 'data' / 'processed' / 'mulms'
REPORT = ROOT / 'results' / 'mulms_corpus.json'

DATASET = 'timo-pierre-schrader/MuLMS'
REVISION = 'bc1c97d63b194e1102ff2e80f1b1bd7379148d6a'
URL = 'https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}'
LICENSE = 'CC BY-SA 4.0'
# the parquet files at REVISION and their SHA-256, checked after a download and in the cache
FILES = {
    'MuLMS_Corpus/train-00000-of-00001.parquet':
        '802e019019c18e5543d9d96cb03f296a30f4ca8def35b368af1028430453add9',
    'MuLMS_Corpus/validation-00000-of-00001.parquet':
        '7c94e244ed2c74cf1bae443b4346ab7f9ac1d6807896a5e7e52eb71be886e536',
    'MuLMS_Corpus/test-00000-of-00001.parquet':
        '9ad56ac15c74faab3821a01e109abd462718d6d7dcb2dcdc748397dfc04eccf7',
}
COLUMNS = ['doc_id', 'category', 'data_split', 'tokens', 'beginOffset', 'NER_labels']
EXPECTED = {'documents': 50, 'sentences': 10184, 'entities': 46584, 'types': 13}
# of the entities that share a span, the one whose type comes first is kept
TYPE_ORDER = ('MEASUREMENT', 'VALUE', 'RANGE', 'SAMPLE', 'PROPERTY', 'TECHNIQUE', 'INSTRUMENT',
              'DEV', 'CITE', 'MAT', 'FORM', 'NUM', 'UNIT')
TYPE_RANK = {kind: rank for rank, kind in enumerate(TYPE_ORDER)}
RULE = ('per sentence: drop the entities strictly inside another; of equal spans keep the type '
        'first in type_order; of crossing entities keep the one that starts first; then IOB2')
INVISIBLE = ('Cc', 'Cf', 'Co', 'Cs', 'Mn')       # the Unicode categories a BERT tokenizer erases
TOKEN_RULE = ('per token: its whitespace-separated pieces (str.split), without the pieces made '
              'only of characters a BERT tokenizer erases (Unicode categories Cc, Cf, Co, Cs '
              'and Mn, U+FFFD); an entity moves to the pieces of its tokens and is dropped if '
              'they have none')
REASONS = ('nested', 'same_span', 'crossing', 'no_visible_token')
# the tokens changed by TOKEN_RULE: empty or whitespace only; only invisible pieces; two or
# more pieces; one piece that is not the token
TOKEN_CHANGES = ('dropped_blank', 'dropped_invisible', 'split', 'trimmed')
DOC_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]*')      # also the name of its .conll file
USER_AGENT = 'AI_Research_Ontologies_GNN_GRL mulms.py (research use)'
TIMEOUT = 120


class ConversionError(ValueError):
    """The corpus does not have the expected form."""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


# ---- the source files ----

def download(url, timeout=TIMEOUT):
    """The body of a GET (urllib follows Hugging Face's redirect to its file store)."""
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_sources(cache, offline=False, files=FILES, get=download):
    """{path in the dataset: local file} of the parquet files. A file is taken from <cache> when
    its SHA-256 is right, else downloaded (never when offline) and stored once its SHA-256 is."""
    cache, local = Path(cache), {}
    for path, digest in files.items():
        target = cache / path
        if not (target.is_file() and sha256(target.read_bytes()) == digest):
            if offline:
                state = 'has another SHA-256' if target.is_file() else 'is missing'
                raise ConversionError(f'offline: {target} {state}')
            data = get(URL.format(dataset=DATASET, revision=REVISION, path=path))
            if sha256(data) != digest:
                raise ConversionError(f'{path}: downloaded SHA-256 {sha256(data)}, not {digest}')
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + '.tmp')
            tmp.write_bytes(data)
            os.replace(tmp, target)
        local[path] = target
    return local


def read_rows(paths):
    """The rows of the parquet files, in order, as dicts of COLUMNS. Needs pyarrow."""
    import pyarrow.parquet as pq
    rows = []
    for path in paths:
        rows += pq.read_table(path, columns=COLUMNS).to_pylist()
    return rows


# ---- checks ----

def check_totals(rows, expected=EXPECTED):
    """ConversionError unless the rows hold the expected numbers of documents, sentences,
    entities and entity types."""
    found = {'documents': len({row['doc_id'] for row in rows}), 'sentences': len(rows),
             'entities': sum(len(row['NER_labels']['value']) for row in rows),
             'types': len({kind for row in rows for kind in row['NER_labels']['value']})}
    if found != expected:
        raise ConversionError(f'counts {found}, not {expected}')


def documents(rows):
    """{doc_id: its rows in text order, by int(beginOffset)}, sorted by doc_id. A document names
    one split and one subfield, and no two of its sentences begin at the same offset."""
    grouped = {}
    for row in rows:
        if not (isinstance(row['doc_id'], str) and DOC_ID.fullmatch(row['doc_id'])):
            raise ConversionError(f"document id {row['doc_id']!r}")
        grouped.setdefault(row['doc_id'], []).append(row)
    docs = {}
    for doc in sorted(grouped):
        found = sorted(grouped[doc], key=lambda row: int(row['beginOffset']))
        for key in ('data_split', 'category'):
            values = sorted({row[key] for row in found})
            if len(values) != 1:
                raise ConversionError(f'{doc}: {key} {values}')
        offsets = [int(row['beginOffset']) for row in found]
        if len(set(offsets)) < len(offsets):
            raise ConversionError(f'{doc}: two sentences begin at the same offset')
        docs[doc] = found
    return docs


def check_tokens(docs):
    """ConversionError if a sentence has no tokens."""
    for doc, rows in docs.items():
        for row in rows:
            if not row['tokens']:
                raise ConversionError(f"{doc}: the sentence at {row['beginOffset']} has no tokens")


def sentence_entities(doc, row):
    """[(first, last, type)] of a row's entities, its token indices as ints (inclusive)."""
    kinds, spans = row['NER_labels']['value'], row['NER_labels']['tokenIndices']
    if len(kinds) != len(spans):
        raise ConversionError(f'{doc}: {len(kinds)} entity types for {len(spans)} spans')
    entities = []
    for kind, span in zip(kinds, spans):
        if kind not in TYPE_RANK:
            raise ConversionError(f'{doc}: entity type {kind!r} is not in TYPE_ORDER')
        if len(span) != 2:
            raise ConversionError(f'{doc}: token indices {span}')
        first, last = int(span[0]), int(span[1])
        if not 0 <= first <= last < len(row['tokens']):
            raise ConversionError(f"{doc}: tokens {first}..{last} of a sentence of "
                                  f"{len(row['tokens'])}")
        entities.append((first, last, kind))
    return entities


# ---- the flattening rule ----

def strictly_inside(inner, outer):
    """The span of `inner` lies within that of `outer` and is not the same."""
    return outer[0] <= inner[0] and inner[1] <= outer[1] and inner[:2] != outer[:2]


def overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def flatten(entities):
    """(kept, dropped) of one sentence's entities [(first, last, type)], token spans inclusive:
    the entities the rule keeps, by start, and {reason: number dropped} (REASONS)."""
    outer = [e for e in entities if not any(strictly_inside(e, o) for o in entities)]
    best = {}                          # span -> its entity of the type first in TYPE_ORDER
    for e in outer:
        if e[:2] not in best or TYPE_RANK[e[2]] < TYPE_RANK[best[e[:2]][2]]:
            best[e[:2]] = e
    kept = []
    for e in sorted(best.values(), key=lambda e: (e[0], e[0] - e[1])):   # by start, longer first
        if not any(overlap(e, k) for k in kept):
            kept.append(e)
    dropped = {'nested': len(entities) - len(outer), 'same_span': len(outer) - len(best),
               'crossing': len(best) - len(kept)}
    return kept, dropped


# ---- the token rule ----

def invisible(piece):
    """The piece is made only of characters that a BERT tokenizer erases: categories Cc, Cf, Co
    and Cs (control, format, private use, surrogate) and Mn (combining marks, stripped with the
    accents by uncased models such as MatSciBERT), and U+FFFD. Unassigned characters (Cn) are
    kept, as BERT keeps them (its unknown token)."""
    return all(ch == '\ufffd' or unicodedata.category(ch) in INVISIBLE for ch in piece)


def pieces(token):
    """The words a token becomes (TOKEN_RULE): its whitespace-separated pieces, without the
    invisible ones."""
    return [piece for piece in token.split() if not invisible(piece)]


def retokenize(tokens, entities):
    """(words, entities, changes) of a sentence: its tokens replaced by their pieces, its
    entities [(first, last, type)] moved to the pieces of their tokens (dropped when these have
    none), and a Counter of the tokens changed (TOKEN_CHANGES)."""
    words, starts, changes = [], [], Counter()
    for token in tokens:
        found = pieces(token)
        starts.append(len(words))
        words += found
        if found != [token]:
            changes['split' if len(found) > 1 else 'trimmed' if found
                    else 'dropped_invisible' if token.split() else 'dropped_blank'] += 1
    starts.append(len(words))
    moved = [(starts[first], starts[last + 1] - 1, kind) for first, last, kind in entities
             if starts[last + 1] > starts[first]]
    return words, moved, changes


def iob2(n_tokens, entities):
    """The IOB2 tags of a sentence of n_tokens tokens with these non-overlapping entities
    [(first, last, type)]."""
    tags = ['O'] * n_tokens
    for first, last, kind in entities:
        if not 0 <= first <= last < n_tokens:
            raise ValueError(f'tokens {first}..{last} of a sentence of {n_tokens}')
        if any(tag != 'O' for tag in tags[first:last + 1]):
            raise ValueError(f'entities overlap at tokens {first}..{last}')
        tags[first:last + 1] = [f'B-{kind}'] + [f'I-{kind}'] * (last - first)
    return tags


def flatten_corpus(docs):
    """(sentences, counts) of documents(): {doc_id: [(tokens, tags)] in text order} and the
    counts of the report."""
    sentences, totals, types, subfields, splits = {}, Counter(), Counter(), {}, Counter()
    tokens = Counter()
    for doc, rows in docs.items():
        subfield = subfields.setdefault(rows[0]['category'], Counter())
        subfield['documents'] += 1
        splits[rows[0]['data_split']] += 1
        sentences[doc] = []
        for row in rows:
            entities = sentence_entities(doc, row)
            kept, dropped = flatten(entities)
            words, moved, changes = retokenize(row['tokens'], kept)
            if not words:
                raise ConversionError(f"{doc}: the sentence at {row['beginOffset']} has no "
                                      f"visible tokens")
            sentences[doc].append((words, iob2(len(words), moved)))
            totals.update(dropped, entities_in=len(entities), entities_kept=len(moved),
                          no_visible_token=len(kept) - len(moved))
            tokens.update(changes)
            tokens.update({'in': len(row['tokens']), 'out': len(words)})
            types.update(kind for _, _, kind in moved)
            subfield.update(sentences=1, entities_kept=len(moved))
    counts = {
        'totals': {'documents': len(docs),
                   'sentences': sum(len(found) for found in sentences.values()),
                   'entities_in': totals['entities_in'], 'entities_kept': totals['entities_kept'],
                   'dropped': {reason: totals[reason] for reason in REASONS}},
        'tokens': {key: tokens[key] for key in ('in', 'out', *TOKEN_CHANGES)},
        'kept_per_type': {kind: types[kind] for kind in sorted(TYPE_ORDER)},
        'subfields': {name: {key: counter[key] for key in ('documents', 'sentences',
                                                           'entities_kept')}
                      for name, counter in sorted(subfields.items())},
        'splits': dict(sorted(splits.items()))}
    return sentences, counts


def convert(rows, expected=EXPECTED):
    """(sentences, split, subfields, counts) of the corpus rows once every check passes: the flat
    sentences of each document, {doc_id: data_split}, {doc_id: category} and the counts."""
    check_totals(rows, expected)
    docs = documents(rows)
    check_tokens(docs)
    sentences, counts = flatten_corpus(docs)
    split = {doc: found[0]['data_split'] for doc, found in docs.items()}
    subfields = {doc: found[0]['category'] for doc, found in docs.items()}
    return sentences, split, subfields, counts


# ---- outputs ----

def conll_text(sentences):
    """One "token tag" line per token, a blank line after each sentence."""
    return ''.join(''.join(f'{w} {t}\n' for w, t in zip(words, tags)) + '\n'
                   for words, tags in sentences)


def write_outputs(out_dir, sentences, split, subfields):
    """Write <doc_id>.conll, split.json and subfields.json to out_dir; returns {name: SHA-256}."""
    files = {f'{doc}.conll': conll_text(found) for doc, found in sentences.items()}
    files['split.json'] = json.dumps(split, indent=2, sort_keys=True) + '\n'
    files['subfields.json'] = json.dumps(subfields, indent=2, sort_keys=True) + '\n'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in sorted(files):
        data = files[name].encode('utf-8')
        (out_dir / name).write_bytes(data)
        digests[name] = sha256(data)
    return digests


def combined_digest(digests):
    """The SHA-256 of the lines "<name> <sha256>\\n", sorted by name."""
    return sha256(''.join(f'{name} {digests[name]}\n' for name in sorted(digests)).encode())


def display_path(path):
    """A path relative to the repository when inside it (no machine-specific prefix)."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def build_report(counts, digests, out_dir):
    return {
        'script': 'src/graph/mulms.py',
        'source': {'dataset': DATASET, 'revision': REVISION, 'license': LICENSE,
                   'url': URL.format(dataset=DATASET, revision=REVISION, path='<file>'),
                   'files': dict(FILES)},
        'rule': {'type_order': list(TYPE_ORDER), 'description': RULE, 'tokens': TOKEN_RULE,
                 'unicode_version': unicodedata.unidata_version},
        **counts,
        'output_dir': display_path(out_dir),
        'outputs': digests,                      # read by kaggle_jobs.py (the bundle)
        'outputs_sha256': combined_digest(digests),
        'outputs_sha256_of': 'the lines "<name> <sha256>" of outputs, sorted by name, each '
                             'ending in a newline'}


def write_corpus(rows, out_dir=OUT_DIR, report_path=REPORT, expected=EXPECTED):
    """Convert the rows, write the outputs and the report; returns the report. Nothing is
    written unless every check passes."""
    sentences, split, subfields, counts = convert(rows, expected)
    digests = write_outputs(out_dir, sentences, split, subfields)
    report = build_report(counts, digests, out_dir)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='MuLMS as flat CoNLL files (IOB2)')
    parser.add_argument('--cache', default=str(CACHE_DIR), metavar='DIR',
                        help='the parquet files, downloaded once')
    parser.add_argument('--out', default=str(OUT_DIR), metavar='DIR')
    parser.add_argument('--report', default=str(REPORT), metavar='PATH')
    parser.add_argument('--offline', action='store_true',
                        help='never use the network: the cache must hold the files')
    args = parser.parse_args(argv)
    sources = fetch_sources(args.cache, args.offline)
    report = write_corpus(read_rows([sources[path] for path in FILES]), args.out, args.report)
    print(json.dumps({key: report[key] for key in ('totals', 'kept_per_type', 'subfields')},
                     indent=2))
    print(f"{args.report}\noutputs_sha256 {report['outputs_sha256']}")
    return report


if __name__ == '__main__':
    main()
