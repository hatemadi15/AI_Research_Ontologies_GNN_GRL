"""
build_topic_corpus.py - an unlabeled corpus per MuLMS subfield from Europe PMC (week 12)

Builds the in-topic corpora of the week-12 protocol (docs/ROADMAP.md §4): for a held-out MuLMS
subfield, sentences of open-access full texts on its topic, in which three CRF teachers find the
names that mention replacement swaps in (`ner_baseline.py --granularity mulms --self-train
<corpus> --augment-names`). It works as build_sofc_corpus.py, whose functions and those of
build_fatigue_corpus.py it uses, with these differences, fixed before any download:

  - Search: per topic, open-access full texts with one of TOPICS[topic] in the title (the hits
    are cached in <cache>/hits.json; a cache made for another query is refused).
  - Local filters, in this order: a PMCID; not one of MuLMS's articles
    (results/mulms_articles.json, mulms_articles.py: the documents of MuLMS that Europe PMC
    indexes); language eng; one of the topic's terms in the title (whole tokens after
    dapt_mlm.normalise); no EXCLUDE_TITLE_TERMS in the title.
  - Text: as build_sofc_corpus (superscripts inline).
  - Leakage and duplicates: dapt_mlm.filter_corpus against every sentence of MuLMS's 50
    documents (data/processed/mulms, mulms.py), then a second test that survives a different
    tokenization: a sentence sharing a run of SHINGLE characters of its text without spaces, in
    lower case, with any MuLMS sentence is dropped, and an article with ARTICLE_OVERLAP or more
    such sentences is dropped whole (a MuLMS document that Europe PMC holds under a PMCID the
    title search missed). SHINGLE is 50 characters, about the 8 words of the n-gram filter: at
    30 the runs were technical terms ("electrochemical impedance spectroscopy", "transmission
    electron microscopy") shared by nearly every article of a topic, which dropped 62 to 224
    articles per topic whole (18 October); at 50, 15 sentences and no article of the SOFC
    pool share a run.
  - Size and order: the settings of the fatigue corpus (TARGET_POOL, MAX_SENTENCES, SEED).
    The shingle test runs after the collector has reached TARGET_POOL with the n-gram filter
    alone, so what it drops comes off that pool; the margin over MAX_SENTENCES covers it.
  - Words: each sentence is cut into words with chem_tokenize.ChemWordTokenizer, as the SOFC
    corpus, then every ASCII hyphen inside a word becomes a word of its own, as MuLMS writes
    them (`4H - SiC`); one sentence per line, words separated by single spaces (ner_baseline.py
    reads it for --granularity mulms by splitting on spaces).

Deterministic: given the cache, a re-run gives the same corpus and SHA-256 (the report differs in
its timestamps and request counts).

Usage:
  python src/graph/build_topic_corpus.py --topic sofc
  python src/graph/build_topic_corpus.py --topic steel --offline --out /tmp/again.txt \\
      --report /tmp/again.json
"""

import argparse
import json
import os
import time
from collections import Counter
from functools import partial

import build_fatigue_corpus as bfc
import conll
import dapt_mlm
from chem_tokenize import ChemWordTokenizer
from config import DEFAULT_PROCESSED_DIR, PROCESSED_DIR, PROJECT_ROOT

MULMS_DIR = os.path.join(DEFAULT_PROCESSED_DIR, 'mulms')
ARTICLES = os.path.join(PROJECT_ROOT, 'results', 'mulms_articles.json')
CORPUS_DIR = os.path.join(PROCESSED_DIR, 'corpus')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'results', 'topic_corpora')

# ---- selection, fixed before any download ----

SEARCH_FILTERS = 'OPEN_ACCESS:y AND HAS_FT:y'
# per MuLMS subfield (mulms.py's category), the terms of which one must be in the title; single
# words in the forms that occur (Europe PMC does not stem)
TOPICS = {
    'electrolysis': ('electrolysis', 'electrolyzer', 'electrolyzers', 'electrolyser',
                     'electrolysers', 'oxygen evolution', 'hydrogen evolution', 'water splitting',
                     'electrocatalyst', 'electrocatalysts'),
    'graphene': ('graphene',),
    'pemfc': ('PEMFC', 'PEMFCs', 'PEM fuel cell', 'PEM fuel cells',
              'proton exchange membrane fuel', 'polymer electrolyte membrane fuel',
              'polymer electrolyte fuel', 'anion exchange membrane fuel',
              'membrane electrode assembly', 'membrane electrode assemblies'),
    'poly': ('thermoplastic', 'thermoplastics', 'epoxy', 'polyurethane', 'polyurethanes',
             'polypropylene', 'elastomer', 'elastomers', 'polymer melt', 'polymer melts',
             'injection molding', 'injection moulding', 'polymer composite',
             'polymer composites'),
    'semi': ('SiC', 'silicon carbide', 'GaN', 'gallium nitride', 'MOSFET', 'MOSFETs',
             'transparent conducting', 'silicon photovoltaic', 'silicon photovoltaics',
             'photovoltaic modules', 'epitaxial', 'epitaxy', 'chemical vapor deposition',
             'chemical vapour deposition', 'wafer', 'wafers'),
    'sofc': ('solid oxide', 'solid-oxide', 'SOFC', 'SOFCs', 'SOEC', 'SOECs'),
    'steel': ('steel', 'steels', 'stainless', 'martensitic', 'austenitic', 'bainitic',
              'ferritic'),
}
EXCLUDE_TITLE_TERMS = ('author correction', 'correction', 'erratum', 'corrigendum',
                       'retraction')
EXCLUDE_PATTERNS = bfc.term_patterns(EXCLUDE_TITLE_TERMS)
SUP = ''                    # superscripts inline, as in the SOFC corpus
SHINGLE = 50                # characters of text without spaces, lower case, shared: leakage
ARTICLE_OVERLAP = 5         # sentences sharing a shingle that drop an article whole


def query(topic):
    """The Europe PMC query of a topic: open-access full texts with one of its terms in the
    title."""
    group = ' OR '.join(f'"{term}"' for term in TOPICS[topic])
    return f'{SEARCH_FILTERS} AND TITLE:({group})'


def mulms_sentences(folder=MULMS_DIR):
    """(document, sentence) for every sentence of MuLMS's documents (mulms.py's CoNLL files):
    the reference of the leakage filters."""
    out = []
    for name in sorted(os.listdir(folder)):
        if name.endswith('.conll'):
            for tokens, _ in conll.read_conll_sentences(os.path.join(folder, name)):
                out.append((name[:-len('.conll')], ' '.join(tokens)))
    return out


def mulms_pmcids(path=ARTICLES):
    """The PMCIDs of MuLMS's documents that Europe PMC indexes (mulms_articles.py)."""
    with open(path, encoding='utf-8') as f:
        articles = json.load(f)
    return sorted({a['pmcid'] for a in articles.values() if a and a.get('pmcid')})


def rejection(record, excluded_pmcids, patterns):
    """Why a hit is not used ('no_pmcid', 'mulms_article', 'not_english', 'no_title_term' or
    'title_term:<term>'), or None when it passes the local filters."""
    if not record.get('pmcid'):
        return 'no_pmcid'
    if record['pmcid'] in excluded_pmcids:
        return 'mulms_article'
    if record.get('language') != bfc.LANGUAGE:
        return 'not_english'
    title = bfc.clean_text(record.get('title'))
    if not bfc.matches(title, patterns):
        return 'no_title_term'
    found = bfc.matches(title, EXCLUDE_PATTERNS)
    return f'title_term:{found[0]}' if found else None


def select_articles(records, excluded_pmcids, patterns):
    """(articles, rejected, excluded): as build_sofc_corpus.select_articles, with these
    filters."""
    articles, rejected, excluded, seen = [], Counter(), [], set()
    for record in records:
        reason = rejection(record, excluded_pmcids, patterns)
        if reason is None and record['pmcid'] in seen:
            reason = 'duplicate_pmcid'
        if reason is not None:
            rejected[reason] += 1
            if reason.startswith('title_term:'):
                excluded.append({'pmcid': record['pmcid'], 'term': reason.split(':', 1)[1],
                                 'title': bfc.clean_text(record['title'])})
            continue
        seen.add(record['pmcid'])
        articles.append(record)
    return articles, rejected, excluded


def shingles(text):
    """The runs of SHINGLE characters of a sentence's text without spaces, in lower case."""
    flat = ''.join(text.split()).lower()
    return {flat[i:i + SHINGLE] for i in range(len(flat) - SHINGLE + 1)}


def shingle_filter(pool, owner, reference):
    """(kept, dropped): the indices of the pool sentences that share no shingle with the
    reference sentences, less every sentence of an article (owner) with ARTICLE_OVERLAP or more
    sharing ones; dropped: the sharing sentences' indices, their count per article and the
    articles dropped whole."""
    index = set()
    for _, text in reference:
        index |= shingles(text)
    sharing = [i for i, text in enumerate(pool) if not index.isdisjoint(shingles(text))]
    by_article = Counter(owner[i] for i in sharing)
    articles = sorted(a for a, n in by_article.items() if n >= ARTICLE_OVERLAP)
    gone = set(sharing) | {i for i, a in enumerate(owner) if a in set(articles)}
    kept = [i for i in range(len(pool)) if i not in gone]
    return kept, {'sentences': sharing, 'by_article': dict(by_article), 'articles': articles}


def hyphen_split(words):
    """MuLMS's hyphens: every ASCII hyphen inside a word becomes a word of its own (4H-SiC: 4H -
    SiC); a word that is a lone hyphen stays, and Unicode dashes stay inside their words."""
    out = []
    for word in words:
        if len(word) > 1 and '-' in word:
            for k, part in enumerate(word.split('-')):
                if k:
                    out.append('-')
                if part:
                    out.append(part)
        else:
            out.append(word)
    return out


def words_line(sentence, tokenizer):
    """A sentence as written to the corpus: its words, single spaces between them."""
    return ' '.join(hyphen_split(tokenizer.tokenize(sentence)))


def selection(topic, excluded_pmcids):
    return {
        'source': 'Europe PMC REST API', 'search_url': bfc.SEARCH_URL,
        'fulltext_url': bfc.FULLTEXT_URL, 'result_type': bfc.RESULT_TYPE, 'topic': topic,
        'query': query(topic), 'title_terms': list(TOPICS[topic]),
        'exclude_title_terms': list(EXCLUDE_TITLE_TERMS), 'excluded_pmcids': excluded_pmcids,
        'local_filters': ['a PMCID', "not one of MuLMS's articles", f'language {bfc.LANGUAGE}',
                          'a title term in the title', 'no exclusion term in the title'],
        'order': 'PMCIDs sorted as strings, then random.Random(0).shuffle',
        'superscripts': 'inline, as in the SOFC corpus',
        'words': 'chem_tokenize.ChemWordTokenizer (ChemDataExtractor 1.3.0), then ASCII hyphens '
                 'inside a word made words of their own, space-separated'}


def leakage_report(dropped, shingled, items):
    """What the two leakage filters did."""
    docs = Counter(doc for d in dropped['overlap'] for doc in d['docs'])
    pmcid = [item['article']['pmcid'] for item in items]
    return {'mulms_sentences': dropped['reference_sentences'],
            'mulms_ngrams': dropped['reference_ngrams'], 'overlap_by_document': dict(docs),
            'overlap_examples': [{k: d[k] for k in ('sentence', 'ngram', 'docs')}
                                 for d in dropped['overlap'][:dapt_mlm.N_EXAMPLES]],
            'duplicate_examples': [{k: d[k] for k in ('sentence', 'duplicate_of')}
                                   for d in dropped['duplicate'][:dapt_mlm.N_EXAMPLES]],
            'shingle': {'characters': SHINGLE, 'article_overlap': ARTICLE_OVERLAP,
                        'sentences_dropped': len(shingled['sentences']),
                        'by_article': {pmcid[a]: n for a, n in shingled['by_article'].items()},
                        'articles_dropped': [pmcid[a] for a in shingled['articles']]}}


def run(args, fetcher=None, reference=None, excluded_pmcids=None):
    """Build the corpus and the report; `fetcher`, `reference` and `excluded_pmcids` are for
    tests."""
    started, t0 = bfc.utc_now(), time.perf_counter()
    if fetcher is None and not args.offline:
        fetcher = bfc.Fetcher()
    if excluded_pmcids is None:
        excluded_pmcids = mulms_pmcids(args.articles)
    if reference is None:
        reference = mulms_sentences(args.mulms_dir)
    topic_query = query(args.topic)
    hits = bfc.load_hits(args.cache_dir, fetcher, args.refresh_search, query=topic_query)
    patterns = bfc.term_patterns(TOPICS[args.topic])
    articles, rejected, excluded = select_articles(hits['records'], set(excluded_pmcids),
                                                   patterns)
    order = bfc.order_articles(articles)
    print(f'Query: {topic_query}\n{len(hits["records"]):,} hits ({hits["retrieved"]}), '
          f'{len(articles):,} pass the filters; rejected: {dict(rejected.most_common(8))}',
          flush=True)

    read = partial(bfc.read_article, cache_dir=args.cache_dir, fetcher=fetcher, sup=SUP)
    items, reached = bfc.collect(order, read, reference, args.target_pool)
    pool, pool_owner, dropped = bfc.build_pool(items, reference)
    kept, shingled = shingle_filter(pool, pool_owner, reference)
    pool, pool_owner = [pool[i] for i in kept], [pool_owner[i] for i in kept]
    picked = bfc.sample_pool(pool, args.max_sentences)
    tokenizer = ChemWordTokenizer()
    corpus = [words_line(pool[i], tokenizer) for i in picked]
    bfc.write_atomic(args.out, ''.join(f'{s}\n' for s in corpus).encode('utf-8'))

    counts = bfc.stage_counts(hits, rejected, len(articles), items, pool, dropped, len(corpus),
                              reached)
    counts['shingle_dropped'] = len(shingled['sentences'])
    counts['articles_dropped_by_shingles'] = len(shingled['articles'])
    report = {
        'script': 'src/graph/build_topic_corpus.py', 'protocol': 'docs/ROADMAP.md §4, week 12: '
        'mention replacement with names from in-topic text', 'topic': args.topic,
        'selection': selection(args.topic, excluded_pmcids),
        'settings': {**bfc.settings(args), 'splitter': 'build_fatigue_corpus.split_sentences'},
        'counts': counts, 'rejected_by_reason': dict(rejected), 'excluded_by_title': excluded,
        'failed': [{'pmcid': it['article']['pmcid'], 'status': it['status'], 'error': it['error']}
                   for it in items if it['status'] != 'ok'],
        'articles': bfc.article_entries(items, pool_owner, [pool_owner[i] for i in picked]),
        'leakage': leakage_report(dropped, shingled, items),
        'corpus': {'path': dapt_mlm.display_path(args.out), 'sentences': len(corpus),
                   'words': sum(len(s.split(' ')) for s in corpus),
                   'sha256': dapt_mlm.file_sha256(args.out)},
        'run': {'search_retrieved': hits['retrieved'], 'started': started,
                'finished': bfc.utc_now(), 'seconds': round(time.perf_counter() - t0, 1),
                'used_articles_downloaded_now': sum(it['downloaded'] for it in items),
                'http_requests': fetcher.requests if fetcher else 0,
                'cache_dir': dapt_mlm.display_path(args.cache_dir)}}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    bfc.write_atomic(args.report, json.dumps(report, indent=2).encode('utf-8'))
    c = report['counts']
    print(f'{c["articles_used"]} articles used ({c["failed_downloads"]} failed), pool '
          f'{c["pool"]:,} ({c["duplicates_dropped"]} duplicates, {c["overlap_dropped"]} '
          f'overlaps and {c["shingle_dropped"]} shingle overlaps dropped, '
          f'{c["articles_dropped_by_shingles"]} articles dropped whole), {c["kept"]:,} '
          f'sentences kept\n{args.out}\nsha256 {report["corpus"]["sha256"]}\nreport '
          f'{args.report}', flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="An unlabeled corpus of a MuLMS subfield's topic from Europe PMC full texts")
    parser.add_argument('--topic', required=True, choices=sorted(TOPICS))
    parser.add_argument('--cache-dir', default=None,
                        help='search results and full-text XML (resumable; default '
                             '<corpus dir>/europepmc_topic_<topic>)')
    parser.add_argument('--out', default=None,
                        help='the corpus, one sentence per line, words separated by spaces '
                             '(default <corpus dir>/topic_<topic>_sentences.txt)')
    parser.add_argument('--report', default=None,
                        help=f'default {dapt_mlm.display_path(REPORT_DIR)}/<topic>.json')
    parser.add_argument('--mulms-dir', default=MULMS_DIR,
                        help="MuLMS's converted documents (mulms.py): the reference of the "
                             'leakage filters')
    parser.add_argument('--articles', default=ARTICLES,
                        help="MuLMS's documents on Europe PMC (mulms_articles.py): left out")
    parser.add_argument('--refresh-search', action='store_true',
                        help='search again instead of using <cache-dir>/hits.json')
    parser.add_argument('--offline', action='store_true',
                        help='never use the network: the cache must hold everything needed')
    fixed = parser.add_argument_group('pre-registered settings (do not tune)')
    fixed.add_argument('--target-pool', type=int, default=bfc.TARGET_POOL,
                       help='read articles until the filtered pool has this many sentences')
    fixed.add_argument('--max-sentences', type=int, default=bfc.MAX_SENTENCES)
    args = parser.parse_args(argv)
    if args.cache_dir is None:
        args.cache_dir = os.path.join(CORPUS_DIR, f'europepmc_topic_{args.topic}')
    if args.out is None:
        args.out = os.path.join(CORPUS_DIR, f'topic_{args.topic}_sentences.txt')
    if args.report is None:
        args.report = os.path.join(REPORT_DIR, f'{args.topic}.json')
    return args


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
