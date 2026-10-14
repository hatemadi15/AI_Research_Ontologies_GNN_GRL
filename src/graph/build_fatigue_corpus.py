"""
build_fatigue_corpus.py - in-domain pre-training corpus of fatigue sentences from Europe PMC

Builds the corpus of the week-3 protocol (docs/ROADMAP.md, "Week 3 protocol", item 5): sentences
from open-access full texts whose title mentions fatigue and whose title or abstract names a
metallic material, for `dapt_mlm.py --corpus`. The selection is fixed in the constants below
before any download; the report (results/fatigue_corpus.json) records it, the counts at every
stage, the article IDs and the SHA-256 of the corpus.

  - Search: the Europe PMC REST API, QUERY (open access, full text, 'fatigue' in the title, one
    of MATERIAL_TERMS in the title or abstract), core records, cursorMark paging. The hit list
    is cached in <cache>/hits.json and reused, so that every run sees the same articles
    (--refresh-search asks again).
  - Local filters on the hits, in this order: a PMCID; a DOI that is not one of the four
    MaterioMiner papers (EXCLUDED_DOIS); language eng; the token 'fatigue' in the title; a
    material term in the title or abstract; no EXCLUDE_TITLE_TERMS in the title (clinical and
    nutritional words, added after reading the hits of QUERY). Terms match whole tokens after
    dapt_mlm.normalise (NFKC, lowercase, dashes -> '-').
  - Order: the PMCIDs sorted as strings, then random.Random(0).shuffle. Articles are taken in
    that order until the filtered sentence pool has TARGET_POOL sentences: the smallest prefix
    that does, found by checking every CHUNK articles (articles after it are not used, even if
    they were downloaded).
  - Download: <cache>/<PMCID>.xml, resumable (cached files are not requested again), at most
    one request per MIN_INTERVAL seconds, a timeout, retries with exponential backoff. A
    permanent HTTP error (404, ...) is remembered in <cache>/<PMCID>.err; other failures are
    retried by the next run. --offline never uses the network.
  - Text: the paragraphs of <body> (JATS). Dropped: reference list, tables, figures and their
    captions, display formulas, and sections such as Acknowledgements or Funding. Citations
    (xref bibr) are removed with the brackets they leave empty; figure and table references
    keep their text; sub is written inline and sup as ^x (10^7); an inline formula is the text
    of its MathML if it has at most MAX_INLINE_FORMULA characters. A display formula or a longer
    one cuts the paragraph: the sentence before it (if unfinished) and the one after it (if it
    starts in lowercase) are dropped.
  - Sentences: a rule-based splitter that does not break after ABBREVIATIONS or inside decimal
    numbers; sentences under MIN_WORDS or over MAX_WORDS words, or with fewer than
    MIN_ALPHA_FRACTION letters among their non-space characters, are dropped.
  - Leakage and duplicates: dapt_mlm.filter_corpus, exactly as in week 2: duplicates after
    normalisation, then every sentence sharing a word NGRAM-gram with a MaterioMiner sentence
    (fine- and coarse-grained CoNLL files).
  - Size: at most MAX_SENTENCES sentences, a random.Random(0) sample of the pool in a
    deterministic order (articles in the order above, sentences in text order), written in the
    sampled order.

Deterministic: given the cache, a re-run gives the same corpus and SHA-256 (the report differs in
its timestamps and request counts).

Usage:
  python src/graph/build_fatigue_corpus.py       # writes the corpus under data/processed/corpus
  python src/graph/build_fatigue_corpus.py --offline --out /tmp/again.txt --report /tmp/again.json
"""

import argparse
import datetime
import html
import http.client
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from functools import partial

import dapt_mlm
from config import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR

SEARCH_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
FULLTEXT_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'
USER_AGENT = 'AI_Research_Ontologies_GNN_GRL fatigue-corpus builder (research use, rate limited)'
DEFAULT_CACHE = os.path.join(PROCESSED_DIR, 'corpus', 'europepmc')
DEFAULT_CORPUS = os.path.join(PROCESSED_DIR, 'corpus', 'fatigue_sentences.txt')
DEFAULT_REPORT = os.path.join(PROJECT_ROOT, 'results', 'fatigue_corpus.json')
HITS_FILE = 'hits.json'

# ---- selection, fixed before any download ----

SEARCH_FILTERS = 'OPEN_ACCESS:y AND HAS_FT:y AND TITLE:fatigue'
# A metallic material named in the title or abstract. Single words are listed in the forms that
# occur (Europe PMC does not stem: alloy and alloys are different terms).
MATERIAL_TERMS = (
    'alloy', 'alloys', 'steel', 'steels', 'superalloy', 'superalloys', 'metal', 'metals',
    'metallic', 'titanium', 'aluminium', 'aluminum', 'magnesium', 'nickel', 'copper', 'iron',
    'zinc', 'cobalt', 'brass', 'bronze', 'solder', 'inconel', 'hastelloy', 'nitinol',
    'Ti-6Al-4V', 'Ti6Al4V', 'weld', 'welds', 'welded', 'welding', 'weldment', 'weldments',
    'powder bed fusion', 'selective laser melting', 'electron beam melting',
    'directed energy deposition', 'wire arc additive manufacturing')
# Biomedical senses of fatigue leak in through iron, magnesium, copper and zinc (nutrients) and
# 'metal': these words in the TITLE exclude a hit. Chosen by reading the titles of the hits of
# the query, then fixed. 'patients' only in the plural: patient-specific implants stay.
EXCLUDE_TITLE_TERMS = (
    'patients', 'muscle', 'muscles', 'muscular', 'cancer', 'cancers', 'syndrome', 'cognitive',
    'mental', 'sleep', 'depression', 'depressive', 'anxiety', 'covid', 'disease', 'diseases',
    'disorder', 'disorders', 'symptom', 'symptoms', 'supplement', 'supplements',
    'supplementation', 'deficiency', 'anemia', 'anaemia', 'vitamin', 'vitamins', 'serum',
    'exercise', 'sclerosis', 'nutritional', 'dietary', 'inflammatory', 'inflammation', 'mice',
    'mouse', 'rats', 'women', 'adults', 'children', 'elderly', 'athletes', 'workers',
    'quality of life',
    # corrections and retractions carry no text of their own
    'author correction', 'erratum', 'corrigendum', 'retraction')
# The four MaterioMiner papers (lowercase DOIs)
EXCLUDED_DOIS = ('10.1007/s11340-021-00758-x', '10.1038/ncomms7164',
                 '10.1038/s41467-020-19071-7', '10.3390/met10111458')
LANGUAGE = 'eng'
PAGE_SIZE = 500
RESULT_TYPE = 'core'        # core records carry the abstract


def build_query(terms):
    """The Europe PMC query: SEARCH_FILTERS and one of `terms` in the title or abstract."""
    group = ' OR '.join(f'"{term}"' for term in terms)
    return f'{SEARCH_FILTERS} AND (TITLE:({group}) OR ABSTRACT:({group}))'


QUERY = build_query(MATERIAL_TERMS)

# ---- corpus settings, fixed before any download ----

SEED = 0                    # order of the articles and sample of the pool
TARGET_POOL = 60_000        # download until the filtered pool has this many sentences
MAX_SENTENCES = 40_000      # sentences kept
CHUNK = 10                  # articles read between two checks of the pool size
MIN_WORDS = 5
MAX_WORDS = 120
MIN_ALPHA_FRACTION = 0.5    # letters among the non-space characters
MAX_INLINE_FORMULA = 30     # characters of MathML text kept in a sentence
NGRAM = 8                   # as dapt_mlm.py --ngram

# ---- politeness ----

MIN_INTERVAL = 0.25         # seconds between requests: at most 4 per second
TIMEOUT = 60                # seconds per request
RETRIES = 4                 # retries after the first attempt
BACKOFF = 2.0               # seconds before the first retry, doubled each time
RETRY_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_RETRY_AFTER = 120.0
MAX_CONSECUTIVE_FAILURES = 8    # stop (the cache keeps its files; re-run to resume)

# ---- text extraction ----

BREAK = chr(0xE000)        # private-use character: marks a removed formula in a paragraph
FLOATS = frozenset({'fig', 'fig-group', 'table-wrap', 'table-wrap-group', 'supplementary-material',
                    'graphic', 'media', 'caption', 'fn', 'fn-group', 'ref-list', 'glossary'})
CUTS = frozenset({'disp-formula', 'inline-graphic'})
BLOCKS = frozenset({'p', 'list', 'list-item', 'boxed-text', 'disp-quote', 'statement', 'sec',
                    'title'})
# Cross-references that are removed (citations, notes, affiliations); the others (fig, table, sec,
# media, ...) keep their text: "Fig. 3". A reference without a type counts as a citation; one
# with an empty type is kept (one publisher uses it for "Eq. (4)").
DROPPED_XREFS = frozenset({'bibr', 'ref', 'fn', 'table-fn', 'aff', 'author-notes', 'corresp',
                           'contrib', 'collab', 'award', 'kwd', 'bio', 'other'})
DASHES = '\N{EN DASH}\N{EM DASH}\N{MINUS SIGN}-'
CITATION_GAP = re.compile(r'[\s,;' + DASHES + r']*')     # between two citations: "1, 2", "1-3"
SKIPPED_SECTIONS = re.compile(
    r'^[\d.\s]*(acknowledg|funding|conflicts? of interest|competing interests?|'
    r'author contributions?|data availability|availability of data|ethic|abbreviations|'
    r'nomenclature|supplementary|references|declarations?|disclosure)', re.I)
INVISIBLE = dict.fromkeys([0x00AD, 0x200B, 0xFEFF])   # soft hyphen, zero-width space, BOM
# What removed citations leave behind, cleaned up in this order: empty brackets, spaces before
# punctuation, runs of commas ("work,,, shows"), a comma before a stop, a doubled period ("Ref..")
CLEANUPS = ((re.compile(r'\s*[(\[]\s*(?:[,;' + DASHES + r']\s*)*[)\]]'), ''),
            (re.compile(r'\s+([,.;)\]])'), r'\1'),
            (re.compile(r',(?:\s*,)+'), ','),
            (re.compile(r',\s*([.;:!?])'), r'\1'),
            (re.compile(r'(?<!\.)\.\.(?!\.)'), '.'))
TAG_RE = re.compile(r'</?[a-zA-Z][^<>]*>')

# ---- sentence splitting ----

ABBREVIATIONS = frozenset({'fig', 'figs', 'eq', 'eqs', 'ref', 'refs', 'vs', 'no', 'nos', 'ca',
                           'wt', 'at', 'cf', 'approx', 'sec', 'sect', 'vol', 'pp', 'resp'})
DOTTED = re.compile(r'(?:[A-Za-z]{1,2}\.)+[A-Za-z]{1,2}')     # e.g, i.e, a.u, U.S, Ph.D
CLOSERS = ')]"\'\N{RIGHT SINGLE QUOTATION MARK}\N{RIGHT DOUBLE QUOTATION MARK}'
OPENERS = '([{"\'\N{LEFT SINGLE QUOTATION MARK}\N{LEFT DOUBLE QUOTATION MARK}'
BOUNDARY = re.compile(r'([.!?])([' + re.escape(CLOSERS) + r']*)\s+')
TERMINAL = re.compile(r'[.!?][' + re.escape(CLOSERS) + r']*$')


class FetchError(Exception):
    """A request failed; `permanent` is true when asking again cannot help (HTTP 4xx)."""

    def __init__(self, message, permanent=False):
        super().__init__(message)
        self.permanent = permanent


def retry_delay(exc, default):
    """Seconds to wait after an HTTP error: its Retry-After (capped) if numeric, else default."""
    try:
        return min(float(exc.headers.get('Retry-After')), MAX_RETRY_AFTER)
    except (AttributeError, TypeError, ValueError):
        return default


class Fetcher:
    """HTTP GET with a rate limit, a timeout and retries with exponential backoff."""

    def __init__(self, min_interval=MIN_INTERVAL, timeout=TIMEOUT, retries=RETRIES,
                 backoff=BACKOFF, opener=urllib.request.urlopen, sleep=time.sleep,
                 clock=time.monotonic):
        self.min_interval, self.timeout, self.retries, self.backoff = (
            min_interval, timeout, retries, backoff)
        self.opener, self.sleep, self.clock = opener, sleep, clock
        self.requests = 0
        self.last = None

    def wait(self):
        """Sleep until min_interval has passed since the start of the previous request."""
        if self.last is not None:
            pause = self.last + self.min_interval - self.clock()
            if pause > 0:
                self.sleep(pause)
        self.last = self.clock()
        self.requests += 1

    def get(self, url, params=None):
        """Body of a GET. FetchError after an HTTP error that retrying cannot fix, or when the
        retries run out (timeouts, resets and 408/425/429/5xx are retried)."""
        if params:
            url += '?' + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        for attempt in range(self.retries + 1):
            self.wait()
            delay = self.backoff * 2 ** attempt
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                error = f'HTTP {exc.code}'
                if exc.code not in RETRY_CODES:
                    raise FetchError(error, permanent=True) from exc
                delay = retry_delay(exc, delay)
            except (OSError, http.client.HTTPException) as exc:
                error = f'{type(exc).__name__}: {exc}'
            if attempt < self.retries:
                self.sleep(delay)
        raise FetchError(f'{error} (after {self.retries + 1} attempts)')


def write_atomic(path, data):
    """Write bytes to <path>.tmp and rename over <path>."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ---- search and selection ----

def clean_text(text):
    """Title or abstract from the API as plain text: entities decoded, tags removed."""
    return ' '.join(TAG_RE.sub('', html.unescape(text or '')).split())


def term_patterns(terms):
    """(term, ' tok tok ') pairs, the tokens as dapt_mlm.normalise gives them."""
    return [(term, ' ' + ' '.join(dapt_mlm.normalise(term)) + ' ') for term in terms]


MATERIAL_PATTERNS = term_patterns(MATERIAL_TERMS)
EXCLUDE_PATTERNS = term_patterns(EXCLUDE_TITLE_TERMS)


def matches(text, patterns):
    """The terms of `patterns` that occur in `text` as whole-token sequences."""
    haystack = ' ' + ' '.join(dapt_mlm.normalise(text)) + ' '
    return [term for term, pattern in patterns if pattern in haystack]


def rejection(record):
    """Why a hit is not used ('no_pmcid', 'excluded_doi', 'not_english', 'no_fatigue_in_title',
    'no_material_term' or 'title_term:<term>'), or None when it passes the local filters."""
    if not record.get('pmcid'):
        return 'no_pmcid'
    if (record.get('doi') or '').lower() in EXCLUDED_DOIS:
        return 'excluded_doi'
    if record.get('language') != LANGUAGE:
        return 'not_english'
    title = clean_text(record.get('title'))
    if 'fatigue' not in dapt_mlm.normalise(title):
        return 'no_fatigue_in_title'
    if not matches(f"{title} {clean_text(record.get('abstract'))}", MATERIAL_PATTERNS):
        return 'no_material_term'
    found = matches(title, EXCLUDE_PATTERNS)
    return f'title_term:{found[0]}' if found else None


def select_articles(records):
    """(articles, rejected, excluded): the hits that pass the local filters, a Counter of the
    reasons for the others, and the titles removed by an exclusion term."""
    articles, rejected, excluded, seen = [], Counter(), [], set()
    for record in records:
        reason = rejection(record)
        if reason is None and record['pmcid'] in seen:
            reason = 'duplicate_pmcid'
        if reason is not None:
            rejected[reason] += 1
            if reason.startswith('title_term:'):
                excluded.append({'pmcid': record['pmcid'], 'term': reason.split(':', 1)[1],
                                 'title': clean_text(record['title'])})
            continue
        seen.add(record['pmcid'])
        articles.append(record)
    return articles, rejected, excluded


def order_articles(articles):
    """Sorted by PMCID (as strings), then shuffled with random.Random(SEED)."""
    ordered = sorted(articles, key=lambda article: article['pmcid'])
    random.Random(SEED).shuffle(ordered)
    return ordered


def slim(record):
    """The fields of a core search result that are used or reported."""
    journal = ((record.get('journalInfo') or {}).get('journal') or {}).get('title')
    return {'pmcid': record.get('pmcid'), 'doi': record.get('doi'), 'title': record.get('title'),
            'abstract': record.get('abstractText'), 'year': record.get('pubYear'),
            'journal': journal, 'license': record.get('license'),
            'language': record.get('language'), 'source': record.get('source')}


def search(fetcher, query=QUERY, page_size=PAGE_SIZE):
    """All core results of the query (cursorMark paging): (hit count, records)."""
    cursor, records = '*', []
    while True:
        page = json.loads(fetcher.get(SEARCH_URL, {
            'query': query, 'format': 'json', 'resultType': RESULT_TYPE,
            'pageSize': page_size, 'cursorMark': cursor}))
        batch = page['resultList']['result']
        records += batch
        previous, cursor = cursor, page.get('nextCursorMark')
        if not batch or not cursor or cursor == previous:
            return page['hitCount'], records


def load_hits(cache_dir, fetcher, refresh=False):
    """The cached search ({query, retrieved, api_hit_count, records}), or a new one, cached."""
    path = os.path.join(cache_dir, HITS_FILE)
    if os.path.exists(path) and not refresh:
        with open(path, encoding='utf-8') as f:
            hits = json.load(f)
        if hits['query'] != QUERY:
            raise SystemExit(f'{path} holds the results of another query; use --refresh-search '
                             f'(this changes the articles) or another --cache-dir')
        return hits
    if fetcher is None:
        raise SystemExit(f'--offline needs {path} (the cached search results)')
    count, records = search(fetcher)
    if count != len(records):
        print(f'  warning: the API reports {count} hits, {len(records)} were returned', flush=True)
    hits = {'query': QUERY, 'retrieved': utc_now(), 'api_hit_count': count,
            'records': [slim(r) for r in records]}
    write_atomic(path, json.dumps(hits, ensure_ascii=False).encode('utf-8'))
    return hits


# ---- full texts ----

def cached_xml(pmcid, cache_dir, fetcher):
    """(xml, error, downloaded): the XML of an article from the cache, else downloaded and cached.

    xml is None when it cannot be had; error says why. A permanent failure is written to
    <pmcid>.err so that later runs do not ask again; other failures leave nothing behind."""
    xml_path = os.path.join(cache_dir, f'{pmcid}.xml')
    err_path = os.path.join(cache_dir, f'{pmcid}.err')
    if os.path.exists(xml_path):
        with open(xml_path, 'rb') as f:
            return f.read(), None, False
    if os.path.exists(err_path):
        with open(err_path, encoding='utf-8') as f:
            return None, f.read().strip(), False
    if fetcher is None:
        return None, 'not cached (offline)', False
    try:
        data = fetcher.get(FULLTEXT_URL.format(pmcid=pmcid))
    except FetchError as exc:
        if exc.permanent:
            write_atomic(err_path, str(exc).encode('utf-8'))
        return None, str(exc), False
    write_atomic(xml_path, data)
    return data, None, True


def local(tag):
    """Tag name without its namespace."""
    return tag.rsplit('}', 1)[-1]


def mathml_text(el):
    """Text of a MathML element, with a space for every mspace."""
    parts = [' ' if local(el.tag) == 'mspace' else el.text or '']
    for child in el:
        parts += [mathml_text(child), child.tail or '']
    return ''.join(parts)


def formula_text(el):
    """An inline formula as plain text: its MathML, or BREAK when that is long or missing."""
    math = next((m for m in el.iter() if local(m.tag) == 'math'), None)
    text = ' '.join(mathml_text(math).split()) if math is not None else ''
    return text if 0 < len(text) <= MAX_INLINE_FORMULA else BREAK


def is_citation(el):
    """True for a cross-reference that is removed from the text (see DROPPED_XREFS)."""
    return local(el.tag) == 'xref' and el.get('ref-type', 'bibr') in DROPPED_XREFS


def child_text(el):
    """What a child element contributes to the text of its paragraph (not its tail)."""
    name = local(el.tag)
    if name in FLOATS or is_citation(el):
        return ''
    if name in CUTS:
        return BREAK
    if name in ('inline-formula', 'math'):
        return formula_text(el)
    if name == 'sup':
        text = inline_text(el)
        return f'^{text}' if re.search(r'\w', text) else ''
    if name == 'break':
        return ' '
    if name in BLOCKS:
        return f'{BREAK}{inline_text(el)}{BREAK}'
    return inline_text(el)


def inline_text(el):
    """The text of an element with citations dropped and formulas, sub and sup replaced.

    The separators between two citations ("1, 2", "1-3") go with them."""
    parts = [el.text or '']
    children = list(el)
    for i, child in enumerate(children):
        tail = child.tail or ''
        if (is_citation(child) and i + 1 < len(children) and is_citation(children[i + 1])
                and CITATION_GAP.fullmatch(tail)):
            tail = ''
        parts += [child_text(child), tail]
    return ''.join(parts)


def section_title(sec):
    title = next((c for c in sec if local(c.tag) == 'title'), None)
    return clean_text(inline_text(title)) if title is not None else ''


def body_paragraphs(root):
    """The text of every paragraph of <body>, in order, without floats and skipped sections.

    A display formula between two paragraphs puts a BREAK at the end of the first and at the
    start of the next, as one inside a paragraph does."""
    body = next((el for el in root if local(el.tag) == 'body'), None)
    found, after_formula = [], False

    def walk(parent):
        nonlocal after_formula
        for el in parent:
            name = local(el.tag)
            if name in CUTS:
                if found:
                    found[-1] += BREAK
                after_formula = True
            elif name in FLOATS or (name == 'sec' and SKIPPED_SECTIONS.match(section_title(el))):
                continue
            elif name == 'p':
                found.append((BREAK if after_formula else '') + inline_text(el))
                after_formula = False
            else:
                walk(el)

    if body is not None:
        walk(body)
    return found


def tidy(text):
    """Whitespace normalised and the leftovers of removed citations cleaned up (CLEANUPS)."""
    text = ' '.join(text.translate(INVISIBLE).split())
    for pattern, replacement in CLEANUPS:
        text = pattern.sub(replacement, text)
    return ' '.join(text.split())


# ---- sentences ----

def is_abbreviation(before):
    """True if `before`, the text up to a period, ends with an abbreviation (not a sentence end)."""
    word = re.search(r'[\w.]+$', before)
    if word is None:
        return False
    word = word.group(0)
    return (word.lower() in ABBREVIATIONS or bool(DOTTED.fullmatch(word))
            or bool(re.search(r'(?:^|\s)et al$', before, re.I)))


def split_sentences(text):
    """Rule-based sentence split: after . ! ? (and closing brackets or quotes) followed by a
    space and a capital letter or digit, except after an abbreviation. Decimal numbers have no
    space after their period."""
    sentences, start = [], 0
    for m in BOUNDARY.finditer(text):
        after = text[m.end():m.end() + 4].lstrip(OPENERS)
        if not after or not (after[0].isupper() or after[0].isdigit()):
            continue
        if m.group(1) == '.' and is_abbreviation(text[start:m.start()]):
            continue
        sentences.append(text[start:m.end(2)])
        start = m.end()
    sentences.append(text[start:])
    return [s.strip() for s in sentences if s.strip()]


def starts_lowercase(sentence):
    return next((c for c in sentence if c.isalpha()), 'A').islower()


def paragraph_sentences(paragraph):
    """(sentences, cut): the sentences of a paragraph, minus `cut` fragments next to a removed
    formula (BREAK): an unfinished sentence before it, a sentence in lowercase after it."""
    pieces = paragraph.split(BREAK)
    kept, cut = [], 0
    for k, piece in enumerate(pieces):
        found = split_sentences(piece)
        if k < len(pieces) - 1 and found and not TERMINAL.search(found[-1]):
            found.pop()
            cut += 1
        if k > 0 and found and starts_lowercase(found[0]):
            found.pop(0)
            cut += 1
        kept += found
    return kept, cut


def length_rejection(sentence):
    """'too_short', 'too_long' or 'non_alphabetic' for a sentence to drop, else None."""
    words = len(sentence.split())
    if words < MIN_WORDS:
        return 'too_short'
    if words > MAX_WORDS:
        return 'too_long'
    chars = [c for c in sentence if not c.isspace()]
    if sum(c.isalpha() for c in chars) < MIN_ALPHA_FRACTION * len(chars):
        return 'non_alphabetic'
    return None


def article_sentences(xml):
    """(sentences, counts) of one JATS article: its body sentences that pass the length filters,
    and a Counter of paragraphs, raw_sentences, cut_by_formula, too_short, too_long and
    non_alphabetic."""
    counts, sentences = Counter(), []
    for paragraph in map(tidy, body_paragraphs(ET.fromstring(xml))):
        if not paragraph.strip(BREAK + ' '):
            continue
        found, cut = paragraph_sentences(paragraph)
        counts['paragraphs'] += 1
        counts['raw_sentences'] += len(found) + cut
        counts['cut_by_formula'] += cut
        for sentence in found:
            reason = length_rejection(sentence)
            if reason:
                counts[reason] += 1
            else:
                sentences.append(sentence)
    return sentences, counts


def read_article(article, cache_dir, fetcher):
    """One article as an item: status 'ok', 'failed' (no usable XML) or 'no_text', its error,
    sentences, counts, and whether it was downloaded now."""
    xml, error, downloaded = cached_xml(article['pmcid'], cache_dir, fetcher)
    item = {'article': article, 'status': 'ok', 'error': None, 'sentences': [],
            'counts': Counter(), 'downloaded': downloaded}
    if xml is None:
        item.update(status='failed', error=error)
        return item
    try:
        item['sentences'], item['counts'] = article_sentences(xml)
    except ET.ParseError as exc:
        item.update(status='failed', error=f'invalid XML: {exc}')
    else:
        if not item['counts']['paragraphs']:
            item.update(status='no_text', error='no body paragraphs')
    return item


# ---- pool and sample ----

def pool_indices(sentences, reference):
    """(indices, dropped): the sentences that dapt_mlm.filter_corpus keeps, and what it drops."""
    _, dropped = dapt_mlm.filter_corpus(sentences, reference, NGRAM)
    gone = {d['index'] for kind in ('duplicate', 'overlap') for d in dropped[kind]}
    return [i for i in range(len(sentences)) if i not in gone], dropped


def collect(order, read, reference, target, chunk=CHUNK):
    """Read articles in order until the filtered pool has `target` sentences.

    Returns (items, reached): the items of the smallest prefix of `order` with that pool, or of
    all articles if the pool never gets there. The pool is checked after every `chunk`
    articles, and only once there are `target` sentences before the leakage filter. Stops with
    an error after MAX_CONSECUTIVE_FAILURES failed articles in a row (a network outage)."""
    items, sentences, owner, failures = [], [], [], 0
    for start in range(0, len(order), chunk):
        for article in order[start:start + chunk]:
            items.append(read(article))
            sentences += items[-1]['sentences']
            owner += [len(items) - 1] * len(items[-1]['sentences'])
            failures = failures + 1 if items[-1]['status'] == 'failed' else 0
            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise SystemExit(f'{failures} articles in a row failed (last: '
                                 f'{items[-1]["error"]}); re-run to resume from the cache')
        print(f'  {len(items)} articles read, {len(sentences):,} sentences before the leakage '
              f'filter', flush=True)
        if len(sentences) >= target:
            kept, _ = pool_indices(sentences, reference)
            if len(kept) >= target:
                return items[:owner[kept[target - 1]] + 1], True
    return items, False


def build_pool(items, reference):
    """(pool, owner, dropped): the filtered sentences of the items in order, the item index of
    each, and the dapt_mlm.filter_corpus report of what was dropped."""
    sentences = [s for item in items for s in item['sentences']]
    owner = [i for i, item in enumerate(items) for _ in item['sentences']]
    kept, dropped = pool_indices(sentences, reference)
    return [sentences[i] for i in kept], [owner[i] for i in kept], dropped


def sample_pool(pool, size=MAX_SENTENCES):
    """Indices into the pool of a random.Random(SEED) sample, in sampled order (all if fewer)."""
    return random.Random(SEED).sample(range(len(pool)), min(size, len(pool)))


# ---- report ----

def settings(args):
    return {'seed': SEED, 'target_pool': args.target_pool, 'max_sentences': args.max_sentences,
            'chunk': CHUNK, 'min_words': MIN_WORDS, 'max_words': MAX_WORDS,
            'min_alpha_fraction': MIN_ALPHA_FRACTION,
            'max_inline_formula_chars': MAX_INLINE_FORMULA, 'ngram': NGRAM, 'language': LANGUAGE, 'page_size': PAGE_SIZE,
            'min_request_interval_seconds': MIN_INTERVAL, 'timeout_seconds': TIMEOUT,
            'retries': RETRIES, 'backoff_seconds': BACKOFF,
            'abbreviations': sorted(ABBREVIATIONS) + ['et al'],
            'dotted_abbreviations': DOTTED.pattern, 'skipped_sections': SKIPPED_SECTIONS.pattern}


def selection():
    return {
        'source': 'Europe PMC REST API',
        'search_url': SEARCH_URL, 'fulltext_url': FULLTEXT_URL, 'result_type': RESULT_TYPE,
        'query': QUERY, 'search_filters': SEARCH_FILTERS, 'material_terms': list(MATERIAL_TERMS),
        'exclude_title_terms': list(EXCLUDE_TITLE_TERMS), 'excluded_dois': list(EXCLUDED_DOIS),
        'local_filters': ['a PMCID', 'DOI not in excluded_dois', f'language {LANGUAGE}',
                          "the token 'fatigue' in the title",
                          'a material term in the title or abstract',
                          'no exclusion term in the title'],
        'order': 'PMCIDs sorted as strings, then random.Random(0).shuffle'}


def stage_counts(hits, rejected, eligible, items, pool, dropped, kept, reached):
    """The counts at every stage, for the articles used (the prefix of the order)."""
    totals = sum((item['counts'] for item in items), Counter())
    status = Counter(item['status'] for item in items)
    length = {k: totals[k] for k in ('too_short', 'too_long', 'non_alphabetic')}
    after_filters = totals['raw_sentences'] - totals['cut_by_formula'] - sum(length.values())
    return {
        'hits': len(hits['records']), 'api_hit_count': hits['api_hit_count'],
        'hits_rejected': sum(rejected.values()), 'eligible_articles': eligible,
        'articles_used': len(items), 'articles_downloaded': status['ok'],
        'failed_downloads': status['failed'], 'articles_without_text': status['no_text'],
        'target_reached': reached, 'paragraphs': totals['paragraphs'],
        'raw_sentences': totals['raw_sentences'], 'cut_by_formula': totals['cut_by_formula'],
        'length_filtered': sum(length.values()), 'length_filtered_by': length,
        'after_length_filter': after_filters,
        'duplicates_dropped': len(dropped['duplicate']), 'overlap_dropped': len(dropped['overlap']),
        'pool': len(pool), 'kept': kept}


def article_entries(items, pool_owner, kept_owner):
    """The articles used (those with text): IDs, metadata and their sentences in pool and corpus."""
    in_pool, in_corpus = Counter(pool_owner), Counter(kept_owner)
    return [{'pmcid': item['article']['pmcid'], 'doi': item['article']['doi'],
             'year': item['article']['year'], 'journal': item['article']['journal'],
             'license': item['article']['license'], 'title': clean_text(item['article']['title']),
             'pool_sentences': in_pool[i], 'corpus_sentences': in_corpus[i]}
            for i, item in enumerate(items) if item['status'] == 'ok']


def leakage_report(dropped):
    """What the leakage filter did: reference size, overlaps per MaterioMiner file, examples."""
    docs = Counter(doc for d in dropped['overlap'] for doc in d['docs'])
    return {'materiominer_sentences': dropped['reference_sentences'],
            'materiominer_ngrams': dropped['reference_ngrams'], 'overlap_by_document': dict(docs),
            'overlap_examples': [{k: d[k] for k in ('sentence', 'ngram', 'docs')}
                                 for d in dropped['overlap'][:dapt_mlm.N_EXAMPLES]],
            'duplicate_examples': [{k: d[k] for k in ('sentence', 'duplicate_of')}
                                   for d in dropped['duplicate'][:dapt_mlm.N_EXAMPLES]]}


# ---- command line ----

def run(args, fetcher=None, reference=None):
    """Build the corpus and the report; `fetcher` and `reference` are for tests."""
    started, t0 = utc_now(), time.perf_counter()
    if fetcher is None and not args.offline:
        fetcher = Fetcher()
    hits = load_hits(args.cache_dir, fetcher, args.refresh_search)
    articles, rejected, excluded = select_articles(hits['records'])
    order = order_articles(articles)
    if reference is None:
        reference = dapt_mlm.materiominer_sentences(args.raw_dir)
    print(f'Query: {QUERY}\n{len(hits["records"]):,} hits ({hits["retrieved"]}), '
          f'{len(articles):,} pass the filters; rejected: {dict(rejected.most_common(8))}',
          flush=True)

    items, reached = collect(order, partial(read_article, cache_dir=args.cache_dir,
                                            fetcher=fetcher), reference, args.target_pool)
    pool, pool_owner, dropped = build_pool(items, reference)
    picked = sample_pool(pool, args.max_sentences)
    corpus = [pool[i] for i in picked]
    write_atomic(args.out, ''.join(f'{s}\n' for s in corpus).encode('utf-8'))

    report = {
        'script': 'src/graph/build_fatigue_corpus.py', 'protocol': 'docs/ROADMAP.md, Week 3 '
        'protocol, item 5 (In-domain pre-training)', 'selection': selection(),
        'settings': settings(args),
        'counts': stage_counts(hits, rejected, len(articles), items, pool, dropped, len(corpus),
                               reached),
        'rejected_by_reason': dict(rejected), 'excluded_by_title': excluded,
        'failed': [{'pmcid': it['article']['pmcid'], 'status': it['status'], 'error': it['error']}
                   for it in items if it['status'] != 'ok'],
        'articles': article_entries(items, pool_owner, [pool_owner[i] for i in picked]),
        'leakage': leakage_report(dropped),
        'corpus': {'path': dapt_mlm.display_path(args.out), 'sentences': len(corpus),
                   'sha256': dapt_mlm.file_sha256(args.out)},
        'run': {'search_retrieved': hits['retrieved'], 'started': started, 'finished': utc_now(),
                'seconds': round(time.perf_counter() - t0, 1),
                'used_articles_downloaded_now': sum(it['downloaded'] for it in items),
                'http_requests': fetcher.requests if fetcher else 0,
                'cache_dir': dapt_mlm.display_path(args.cache_dir)}}
    write_atomic(args.report, json.dumps(report, indent=2).encode('utf-8'))
    c = report['counts']
    print(f'{c["articles_used"]} articles used ({c["failed_downloads"]} failed), pool '
          f'{c["pool"]:,} ({c["duplicates_dropped"]} duplicates and {c["overlap_dropped"]} '
          f'overlaps dropped), {c["kept"]:,} sentences kept\n{args.out}\nsha256 '
          f'{report["corpus"]["sha256"]}\nreport {args.report}', flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='In-domain fatigue corpus from Europe PMC full texts (week-3 protocol, item 5)')
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE,
                        help='search results and full-text XML (resumable)')
    parser.add_argument('--out', default=DEFAULT_CORPUS, help='the corpus, one sentence per line')
    parser.add_argument('--report', default=DEFAULT_REPORT)
    parser.add_argument('--raw-dir', default=RAW_DIR,
                        help='MaterioMiner data, the reference of the leakage filter')
    parser.add_argument('--refresh-search', action='store_true',
                        help='search again instead of using <cache-dir>/hits.json')
    parser.add_argument('--offline', action='store_true',
                        help='never use the network: the cache must hold everything needed')
    fixed = parser.add_argument_group('pre-registered settings (do not tune)')
    fixed.add_argument('--target-pool', type=int, default=TARGET_POOL,
                       help='read articles until the filtered pool has this many sentences')
    fixed.add_argument('--max-sentences', type=int, default=MAX_SENTENCES)
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
