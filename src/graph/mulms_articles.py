"""
mulms_articles.py - which MuLMS documents Europe PMC indexes, found by title (week 12)

MuLMS ships no article identifiers, but each converted document (mulms.py) starts with its
title, after journal boilerplate. For every document this script searches Europe PMC's titles
for its title words and writes results/mulms_articles.json: per document, its subfield, the
title used, and the PMCID, PMID, DOI and title of the best hit with the Jaccard overlap of the
two titles' words (a document matches at MATCH_SCORE or more; a document without a match is
null). The matched articles are left out of the topic corpora (build_topic_corpus.py); the
documents that Europe PMC does not index cannot leak into them in the first place.

Titles: the first SEGMENTS blank-line separated sentences of a document, split at ' | ',
without the boilerplate segments (BOILERPLATE) and those with fewer than MIN_WORDS title
words (runs of letters and digits with a letter, 4 characters or more: 42CrMo4 counts). Two
queries per title:
its first 8 title words, then its 5 longest, each as TITLE:(... AND ...).

Usage:
  python src/graph/mulms_articles.py [--mulms-dir DIR] [--out results/mulms_articles.json]
"""

import argparse
import json
import os
import re

import build_fatigue_corpus as bfc
from config import DEFAULT_PROCESSED_DIR, PROJECT_ROOT

MULMS_DIR = os.path.join(DEFAULT_PROCESSED_DIR, 'mulms')
OUT = os.path.join(PROJECT_ROOT, 'results', 'mulms_articles.json')
SEGMENTS = 4
MIN_WORDS = 4
MATCH_SCORE = 0.5
PAGE_SIZE = 5
TITLE_WORD = re.compile(r'(?=[0-9]*[A-Za-z])[A-Za-z0-9]{4,}')
BOILERPLATE = re.compile(r'^(article|open article|research article|open access|abstract|'
                         r'journal of the electrochemical society open access|'
                         r'micromachines article|molecules article|a r t i c l e.*)$', re.I)


def segments(path, n=SEGMENTS):
    """The first n sentences of a converted document, as text."""
    words, out = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    out.append(' '.join(words))
                    words = []
                if len(out) >= n:
                    break
                continue
            words.append(line.split()[0])
    if words and len(out) < n:
        out.append(' '.join(words))
    return out


def title_words(text):
    """The words a title is matched on, in lower case."""
    return [w.lower() for w in TITLE_WORD.findall(text)]


def title_candidates(path):
    """The title candidates of a document: its segments split at ' | ', without boilerplate."""
    out = []
    for segment in segments(path):
        for part in re.split(r'\s\|\s', segment):
            part = part.strip(' .:|')
            if len(title_words(part)) >= MIN_WORDS and not BOILERPLATE.match(part):
                out.append(part)
    return out[:3]


def queries(title):
    """The two Europe PMC title queries of a title."""
    words = title_words(title)
    longest = sorted(set(words), key=lambda w: (-len(w), w))[:5]
    return ['TITLE:(' + ' AND '.join(f'"{w}"' for w in chosen) + ')'
            for chosen in (words[:8], longest)]


def score(title, matched):
    """Jaccard overlap of the two titles' words."""
    want, got = set(title_words(title)), set(title_words(matched or ''))
    return round(len(want & got) / max(1, len(want | got)), 2)


def best_match(path, search):
    """The best Europe PMC hit for a document's title candidates ({title, pmcid, pmid, doi,
    matched_title, score}), or None below MATCH_SCORE. `search(query)` gives the hits."""
    best = None
    for title in title_candidates(path):
        for q in queries(title):
            for h in search(q):
                s = score(title, h.get('title'))
                if best is None or s > best['score']:
                    best = {'title': title, 'pmcid': h.get('pmcid'), 'pmid': h.get('pmid'),
                            'doi': h.get('doi'), 'matched_title': h.get('title'), 'score': s}
            if best and best['score'] >= MATCH_SCORE:
                return best
    return best if best and best['score'] >= MATCH_SCORE else None


def europe_pmc_search(fetcher):
    """search(query) over Europe PMC's REST API, through a rate-limited Fetcher."""
    def search(q):
        page = fetcher.get(bfc.SEARCH_URL, params={'query': q, 'format': 'json',
                                                   'pageSize': PAGE_SIZE, 'resultType': 'lite'})
        return json.loads(page).get('resultList', {}).get('result', [])
    return search


def run(mulms_dir, out, search):
    with open(os.path.join(mulms_dir, 'subfields.json'), encoding='utf-8') as f:
        subfields = json.load(f)
    found = {}
    for doc in sorted(subfields):
        match = best_match(os.path.join(mulms_dir, f'{doc}.conll'), search)
        found[doc] = {'subfield': subfields[doc], **match} if match else None
        print(f"{doc:16s} {subfields[doc]:12s} "
              f"{(match or {}).get('pmcid') or 'no match':12s} "
              f"{(match or {}).get('matched_title') or ''}"[:110], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(found, f, indent=1, ensure_ascii=False)
    with_pmcid = sum(1 for v in found.values() if v and v.get('pmcid'))
    print(f'{with_pmcid} of {len(found)} documents have a PMCID; {out}')
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description="MuLMS's documents on Europe PMC, by title")
    parser.add_argument('--mulms-dir', default=MULMS_DIR)
    parser.add_argument('--out', default=OUT)
    args = parser.parse_args(argv)
    run(args.mulms_dir, args.out, europe_pmc_search(bfc.Fetcher()))


if __name__ == '__main__':
    main()
