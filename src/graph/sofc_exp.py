"""
sofc_exp.py - the SOFC-Exp corpus as CoNLL files (ROADMAP P1 "Next, after week 4", item 7)

SOFC-Exp (Friedrich et al., ACL 2020;
https://github.com/boschresearch/sofc-exp_textmining_resources) annotates four entity
types (MATERIAL, VALUE, DEVICE, EXPERIMENT) in the sentences of 45 open-access articles on
solid oxide fuel cells that describe experiments. The annotations are CC BY 4.0; the texts
are those of the CC-BY articles.

The corpus is stand-off. Per document <doc>:
  - annotations/sentences/<doc>.csv: sentence id, experiment flag (1: annotated), begin and
    end in the text;
  - annotations/entity_types_and_slots/<doc>.csv: sentence id, token id, begin and end
    relative to the sentence's begin, the BIO entity tag and a slot tag;
  - texts/<doc>.txt: the text.

This script cuts every token out of the text and writes, per document, its annotated
sentences in text order as <doc>.conll: one "token tag" line per token, a blank line
between sentences (the format of the MaterioMiner files). It also writes split.json, the
official document split of SOFC-Exp-Metadata.csv (26 train, 8 dev, 11 test), and checks
the counts of the paper: 45 documents, 876 sentences and 4,590 entities.

context.json holds the whole text of each document for document context (ner_baseline.py
--context): {doc: {"sentences": every sentence as tokens, in text order, annotated or not
(annotations/tokens/<doc>.csv, offsets relative to the sentence), "annotated": the
positions among them of the sentences of <doc>.conll}}. The annotated sentences' tokens
must equal those of the entity annotations.

Usage:
  git clone https://github.com/boschresearch/sofc-exp_textmining_resources.git <dir>
  python src/graph/sofc_exp.py <dir>/sofc-exp-corpus [--out data/raw/dataset/sofc_exp]
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'dataset' / 'sofc_exp'
EXPECTED = {'documents': 45, 'sentences': 876, 'entities': 4590}
TAG = re.compile(r'O|[BI]-[A-Z]+')


class ConversionError(ValueError):
    """The corpus does not have the expected form."""


def read_rows(path):
    """The tab-separated rows of a stand-off file (blank lines skipped)."""
    return [line.split('\t') for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def convert_document(corpus, doc):
    """[(tokens, tags)] of the annotated sentences of one document, in text order."""
    corpus = Path(corpus)
    text = (corpus / 'texts' / f'{doc}.txt').read_text(encoding='utf-8')
    sentences = {}
    for sid, experiment, begin, _ in read_rows(corpus / 'annotations' / 'sentences' /
                                               f'{doc}.csv'):
        sentences[sid] = (experiment == '1', int(begin))
    tokens = {}
    for row in read_rows(corpus / 'annotations' / 'entity_types_and_slots' / f'{doc}.csv'):
        sid, _, begin, end, tag = row[:5]
        annotated, offset = sentences[sid]
        if not annotated:
            raise ConversionError(f'{doc}: sentence {sid} has entity tags but is not annotated')
        word = text[offset + int(begin):offset + int(end)]
        if not word or re.search(r'\s', word):
            raise ConversionError(f'{doc}: sentence {sid}: token {word!r} is empty or has '
                                  f'whitespace')
        if not TAG.fullmatch(tag):
            raise ConversionError(f'{doc}: sentence {sid}: tag {tag!r}')
        tokens.setdefault(sid, []).append((offset + int(begin), word, tag))
    out = []
    for sid in sorted(tokens, key=lambda s: sentences[s][1]):
        words = sorted(tokens[sid])
        tags = [t for _, _, t in words]
        for prev, tag in zip(['O'] + tags, tags):
            if tag.startswith('I-') and prev[2:] != tag[2:]:
                raise ConversionError(f'{doc}: sentence {sid}: {tag} does not continue {prev}')
        out.append(([w for _, w, _ in words], tags))
    return out


def document_context(corpus, doc, annotated):
    """Every sentence of a document as tokens, in text order, and the positions among them
    of its annotated sentences (`annotated`: convert_document's output, whose words they must
    match)."""
    corpus = Path(corpus)
    text = (corpus / 'texts' / f'{doc}.txt').read_text(encoding='utf-8')
    begins = {sid: int(begin) for sid, _, begin, _ in
              read_rows(corpus / 'annotations' / 'sentences' / f'{doc}.csv')}
    tagged = {row[0] for row in
              read_rows(corpus / 'annotations' / 'entity_types_and_slots' / f'{doc}.csv')}
    tokens = {}
    for sid, _, begin, end in (row[:4] for row in
                               read_rows(corpus / 'annotations' / 'tokens' / f'{doc}.csv')):
        word = text[begins[sid] + int(begin):begins[sid] + int(end)]
        if not word or re.search(r'\s', word):
            raise ConversionError(f'{doc}: sentence {sid}: token {word!r} is empty or has '
                                  f'whitespace')
        tokens.setdefault(sid, []).append((int(begin), word))
    order = sorted(tokens, key=lambda sid: begins[sid])
    sentences = [[w for _, w in sorted(tokens[sid])] for sid in order]
    positions = [k for k, sid in enumerate(order) if sid in tagged]
    if [sentences[k] for k in positions] != [words for words, _ in annotated]:
        raise ConversionError(f'{doc}: the token files and the entity annotations cut '
                              f'different words')
    return {'sentences': sentences, 'annotated': positions}


def read_split(corpus):
    """{document: 'train', 'dev' or 'test'} from SOFC-Exp-Metadata.csv."""
    with open(Path(corpus) / 'SOFC-Exp-Metadata.csv', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    return {row['name']: row['set'] for row in rows}


def convert(corpus, out_dir=OUT_DIR, expected=EXPECTED):
    """Write <doc>.conll for every document and split.json to out_dir; returns the counts."""
    corpus, out_dir = Path(corpus), Path(out_dir)
    split = read_split(corpus)
    docs = sorted(p.stem for p in (corpus / 'annotations' / 'entity_types_and_slots')
                  .glob('*.csv'))
    if sorted(split) != docs:
        raise ConversionError('the metadata and the annotation files name different documents')
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    context = {}
    for doc in docs:
        sentences = convert_document(corpus, doc)
        context[doc] = document_context(corpus, doc, sentences)
        lines = []
        for words, tags in sentences:
            lines += [f'{w} {t}' for w, t in zip(words, tags)] + ['']
            counts.update(t[2:] for t in tags if t.startswith('B-'))
        (out_dir / f'{doc}.conll').write_text('\n'.join(lines), encoding='utf-8')
        counts['sentences'] += len(sentences)
    (out_dir / 'split.json').write_text(json.dumps(split, indent=2, sort_keys=True) + '\n')
    (out_dir / 'context.json').write_text(json.dumps(context, sort_keys=True) + '\n',
                                          encoding='utf-8')
    stats = {'documents': len(docs), 'sentences': counts.pop('sentences'),
             'entities': sum(counts.values()), 'types': dict(sorted(counts.items())),
             'split': dict(sorted(Counter(split.values()).items()))}
    found = {k: stats[k] for k in expected}
    if found != expected:
        raise ConversionError(f'counts {found}, not those of the paper {expected}')
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description='SOFC-Exp as CoNLL files for ner_baseline.py')
    parser.add_argument('corpus', help='the sofc-exp-corpus directory of the repository')
    parser.add_argument('--out', default=str(OUT_DIR), metavar='DIR')
    args = parser.parse_args(argv)
    if not os.path.isdir(args.corpus):
        parser.error(f'not a directory: {args.corpus}')
    stats = convert(args.corpus, args.out)
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == '__main__':
    main()
