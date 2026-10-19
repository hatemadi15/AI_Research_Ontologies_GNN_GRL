"""
build_sofc_corpus.py - an in-domain SOFC corpus from Europe PMC, tokenized like SOFC-Exp

Builds the corpus of the week-6 amendment "multi-task self-training on SOFC-Exp" (docs/ROADMAP.md
§4): sentences of open-access full texts about solid oxide fuel cells, for `ner_baseline.py
--granularity sofc_both --multi-task --self-train`. It works as build_fatigue_corpus.py, whose
functions it uses (search, download cache, JATS text, sentence split, leakage filter, sample),
with these differences, fixed before any download:

  - Search: QUERY, open-access full texts with "solid oxide", SOFC or SOFCs in the title (the
    hits are cached in <cache>/hits.json and reused).
  - Local filters, in this order: a PMCID; not one of SOFC-Exp's 45 articles (their PMCIDs are
    the file names of data/raw/dataset/sofc_exp); language eng; one of TITLE_TERMS in the title
    (whole tokens after dapt_mlm.normalise); no EXCLUDE_TITLE_TERMS in the title (corrections
    and retractions carry no text of their own).
  - Text: as build_fatigue_corpus, but a superscript is written inline (cm−2, Fe3+), as
    SOFC-Exp's texts write it, not after a caret.
  - Leakage and duplicates: dapt_mlm.filter_corpus, duplicates after normalisation, then every
    sentence sharing a word NGRAM-gram with any of the 9,466 sentences of SOFC-Exp's 45 articles
    (data/raw/dataset/sofc_exp/context.json), not only the annotated ones.
  - Size and order: the settings of the fatigue corpus (TARGET_POOL, MAX_SENTENCES, SEED);
    articles until the filtered pool reaches TARGET_POOL sentences, or all of them.
  - Words: each sentence is cut into words with chem_tokenize.ChemWordTokenizer
    (ChemDataExtractor 1.3.0), the tokenizer of SOFC-Exp's training and development articles,
    and written with a single space between words, one sentence per line. ner_baseline.py reads
    an SOFC corpus by splitting its lines on spaces.

Deterministic: given the cache, a re-run gives the same corpus and SHA-256 (the report differs in
its timestamps and request counts).

Usage:
  python src/graph/build_sofc_corpus.py       # writes the corpus under data/processed/corpus
  python src/graph/build_sofc_corpus.py --offline --out /tmp/again.txt --report /tmp/again.json
"""

import argparse
import json
import os
import time
from collections import Counter
from functools import partial

import build_fatigue_corpus as bfc
import dapt_mlm
from chem_tokenize import ChemWordTokenizer
from config import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR

DEFAULT_CACHE = os.path.join(PROCESSED_DIR, 'corpus', 'europepmc_sofc')
DEFAULT_CORPUS = os.path.join(PROCESSED_DIR, 'corpus', 'sofc_sentences.txt')
DEFAULT_REPORT = os.path.join(PROJECT_ROOT, 'results', 'sofc_corpus.json')
SOFC_EXP_DIR = os.path.join(RAW_DIR, 'sofc_exp')

# ---- selection, fixed before any download ----

SEARCH_FILTERS = 'OPEN_ACCESS:y AND HAS_FT:y'
TITLE_TERMS = ('solid oxide', 'solid-oxide', 'SOFC', 'SOFCs')
QUERY = f'{SEARCH_FILTERS} AND (TITLE:"solid oxide" OR TITLE:SOFC OR TITLE:SOFCs)'
EXCLUDE_TITLE_TERMS = ('author correction', 'correction', 'erratum', 'corrigendum',
                       'retraction')
TITLE_PATTERNS = bfc.term_patterns(TITLE_TERMS)
EXCLUDE_PATTERNS = bfc.term_patterns(EXCLUDE_TITLE_TERMS)
SUP = ''                    # superscripts inline, as in SOFC-Exp's texts


def sofc_exp_pmcids(sofc_dir=SOFC_EXP_DIR):
    """The PMCIDs of SOFC-Exp's articles (the names of its CoNLL files)."""
    return sorted(f[:-len('.conll')] for f in os.listdir(sofc_dir) if f.endswith('.conll'))


def sofc_exp_sentences(sofc_dir=SOFC_EXP_DIR):
    """(PMCID, sentence) for every sentence of SOFC-Exp's articles (context.json): the
    reference of the leakage filter."""
    with open(os.path.join(sofc_dir, 'context.json'), encoding='utf-8') as f:
        context = json.load(f)
    return [(doc, ' '.join(words)) for doc in sorted(context)
            for words in context[doc]['sentences']]


def rejection(record, excluded_pmcids):
    """Why a hit is not used ('no_pmcid', 'sofc_exp_article', 'not_english',
    'no_title_term' or 'title_term:<term>'), or None when it passes the local filters."""
    if not record.get('pmcid'):
        return 'no_pmcid'
    if record['pmcid'] in excluded_pmcids:
        return 'sofc_exp_article'
    if record.get('language') != bfc.LANGUAGE:
        return 'not_english'
    title = bfc.clean_text(record.get('title'))
    if not bfc.matches(title, TITLE_PATTERNS):
        return 'no_title_term'
    found = bfc.matches(title, EXCLUDE_PATTERNS)
    return f'title_term:{found[0]}' if found else None


def select_articles(records, excluded_pmcids):
    """(articles, rejected, excluded): as build_fatigue_corpus.select_articles, with these
    filters."""
    articles, rejected, excluded, seen = [], Counter(), [], set()
    for record in records:
        reason = rejection(record, excluded_pmcids)
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


def words_line(sentence, tokenizer):
    """A sentence as written to the corpus: its words, single spaces between them."""
    return ' '.join(tokenizer.tokenize(sentence))


def selection(excluded_pmcids):
    return {
        'source': 'Europe PMC REST API', 'search_url': bfc.SEARCH_URL,
        'fulltext_url': bfc.FULLTEXT_URL, 'result_type': bfc.RESULT_TYPE, 'query': QUERY,
        'title_terms': list(TITLE_TERMS), 'exclude_title_terms': list(EXCLUDE_TITLE_TERMS),
        'excluded_pmcids': excluded_pmcids,
        'local_filters': ['a PMCID', "not one of SOFC-Exp's articles", f'language {bfc.LANGUAGE}',
                          'a title term in the title', 'no exclusion term in the title'],
        'order': 'PMCIDs sorted as strings, then random.Random(0).shuffle',
        'superscripts': 'inline, as in the texts of SOFC-Exp',
        'words': 'chem_tokenize.ChemWordTokenizer (ChemDataExtractor 1.3.0), space-separated'}


def leakage_report(dropped):
    """What the leakage filter did: reference size, overlaps per SOFC-Exp article, examples."""
    docs = Counter(doc for d in dropped['overlap'] for doc in d['docs'])
    return {'sofc_exp_sentences': dropped['reference_sentences'],
            'sofc_exp_ngrams': dropped['reference_ngrams'], 'overlap_by_document': dict(docs),
            'overlap_examples': [{k: d[k] for k in ('sentence', 'ngram', 'docs')}
                                 for d in dropped['overlap'][:dapt_mlm.N_EXAMPLES]],
            'duplicate_examples': [{k: d[k] for k in ('sentence', 'duplicate_of')}
                                   for d in dropped['duplicate'][:dapt_mlm.N_EXAMPLES]]}


def run(args, fetcher=None, reference=None, excluded_pmcids=None):
    """Build the corpus and the report; `fetcher`, `reference` and `excluded_pmcids` are for
    tests."""
    started, t0 = bfc.utc_now(), time.perf_counter()
    if fetcher is None and not args.offline:
        fetcher = bfc.Fetcher()
    if excluded_pmcids is None:
        excluded_pmcids = sofc_exp_pmcids(args.sofc_dir)
    if reference is None:
        reference = sofc_exp_sentences(args.sofc_dir)
    hits = bfc.load_hits(args.cache_dir, fetcher, args.refresh_search, query=QUERY)
    articles, rejected, excluded = select_articles(hits['records'], set(excluded_pmcids))
    order = bfc.order_articles(articles)
    print(f'Query: {QUERY}\n{len(hits["records"]):,} hits ({hits["retrieved"]}), '
          f'{len(articles):,} pass the filters; rejected: {dict(rejected.most_common(8))}',
          flush=True)

    read = partial(bfc.read_article, cache_dir=args.cache_dir, fetcher=fetcher, sup=SUP)
    items, reached = bfc.collect(order, read, reference, args.target_pool)
    pool, pool_owner, dropped = bfc.build_pool(items, reference)
    picked = bfc.sample_pool(pool, args.max_sentences)
    tokenizer = ChemWordTokenizer()
    corpus = [words_line(pool[i], tokenizer) for i in picked]
    bfc.write_atomic(args.out, ''.join(f'{s}\n' for s in corpus).encode('utf-8'))

    report = {
        'script': 'src/graph/build_sofc_corpus.py', 'protocol': 'docs/ROADMAP.md §4, week-6 '
        'amendment: multi-task self-training on SOFC-Exp', 'selection': selection(excluded_pmcids),
        'settings': {**bfc.settings(args), 'splitter': 'build_fatigue_corpus.split_sentences'},
        'counts': bfc.stage_counts(hits, rejected, len(articles), items, pool, dropped,
                                   len(corpus), reached),
        'rejected_by_reason': dict(rejected), 'excluded_by_title': excluded,
        'failed': [{'pmcid': it['article']['pmcid'], 'status': it['status'], 'error': it['error']}
                   for it in items if it['status'] != 'ok'],
        'articles': bfc.article_entries(items, pool_owner, [pool_owner[i] for i in picked]),
        'leakage': leakage_report(dropped),
        'corpus': {'path': dapt_mlm.display_path(args.out), 'sentences': len(corpus),
                   'words': sum(len(s.split(' ')) for s in corpus),
                   'sha256': dapt_mlm.file_sha256(args.out)},
        'run': {'search_retrieved': hits['retrieved'], 'started': started,
                'finished': bfc.utc_now(), 'seconds': round(time.perf_counter() - t0, 1),
                'used_articles_downloaded_now': sum(it['downloaded'] for it in items),
                'http_requests': fetcher.requests if fetcher else 0,
                'cache_dir': dapt_mlm.display_path(args.cache_dir)}}
    bfc.write_atomic(args.report, json.dumps(report, indent=2).encode('utf-8'))
    c = report['counts']
    print(f'{c["articles_used"]} articles used ({c["failed_downloads"]} failed), pool '
          f'{c["pool"]:,} ({c["duplicates_dropped"]} duplicates and {c["overlap_dropped"]} '
          f'overlaps dropped), {c["kept"]:,} sentences kept\n{args.out}\nsha256 '
          f'{report["corpus"]["sha256"]}\nreport {args.report}', flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='In-domain SOFC corpus from Europe PMC full texts, tokenized like SOFC-Exp')
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE,
                        help='search results and full-text XML (resumable)')
    parser.add_argument('--out', default=DEFAULT_CORPUS,
                        help='the corpus, one sentence per line, words separated by spaces')
    parser.add_argument('--report', default=DEFAULT_REPORT)
    parser.add_argument('--sofc-dir', default=SOFC_EXP_DIR,
                        help="SOFC-Exp's CoNLL files and context.json (sofc_exp.py): the "
                             "articles left out and the reference of the leakage filter")
    parser.add_argument('--refresh-search', action='store_true',
                        help='search again instead of using <cache-dir>/hits.json')
    parser.add_argument('--offline', action='store_true',
                        help='never use the network: the cache must hold everything needed')
    fixed = parser.add_argument_group('pre-registered settings (do not tune)')
    fixed.add_argument('--target-pool', type=int, default=bfc.TARGET_POOL,
                       help='read articles until the filtered pool has this many sentences')
    fixed.add_argument('--max-sentences', type=int, default=bfc.MAX_SENTENCES)
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
