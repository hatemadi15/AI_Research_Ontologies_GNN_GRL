"""Topic corpus builder (week 12): the query per topic, the local filters, MuLMS's hyphen rule,
the shingle leakage test with its article-level drop, and the whole script offline.

No network: the search and the full texts come from the fake fetcher of the SOFC builder's
tests, and the leakage reference is a tiny fake MuLMS.
"""
import json
import os

import pytest

import build_fatigue_corpus as bf
import build_topic_corpus as bt
import dapt_mlm
from chem_tokenize import ChemWordTokenizer
from test_build_sofc_corpus import FakeEuropePMC, hit

# a fake MuLMS: one sentence the n-gram filter catches, one only the shingle test catches
# (two long words: fewer than 8 tokens, more than 30 characters without spaces)
NGRAM_LEAK = 'Anode supported cells were sintered at 1400 °C for 4 h in air before the tests.'
SHINGLE_LEAK = 'Thermogravimetric characterization of the SiO 2 layers followed.'
REFERENCE = [('sofc_01', NGRAM_LEAK.replace('before the tests.', 'and then reduced.')),
             ('semi_02', SHINGLE_LEAK)]


def test_query_and_local_filters():
    q = bt.query('sofc')
    assert q.startswith('OPEN_ACCESS:y AND HAS_FT:y AND TITLE:("solid oxide" OR ')
    assert '"SOEC"' in q and 'ABSTRACT' not in q
    assert set(bt.TOPICS) == {'electrolysis', 'graphene', 'pemfc', 'poly', 'semi', 'sofc',
                              'steel'}
    patterns = bf.term_patterns(bt.TOPICS['semi'])
    excluded = {'PMC9'}
    assert bt.rejection(hit('PMC1', 'A 4H-SiC MOSFET'), excluded, patterns) is None
    assert bt.rejection(hit('PMC1', 'Silicon carbide wafers'), excluded, patterns) is None
    assert bt.rejection({'title': 'SiC'}, excluded, patterns) == 'no_pmcid'
    assert bt.rejection(hit('PMC9', 'SiC'), excluded, patterns) == 'mulms_article'
    assert bt.rejection(hit('PMC2', 'SiC', 'ger'), excluded, patterns) == 'not_english'
    assert bt.rejection(hit('PMC2', 'Silicon'), excluded, patterns) == 'no_title_term'
    assert bt.rejection(hit('PMC2', 'Erratum: SiC'), excluded, patterns) == 'title_term:erratum'
    articles, rejected, excluded_by_title = bt.select_articles(
        [hit('PMC1', 'SiC'), hit('PMC1', 'SiC again'), hit('PMC3', 'Corrigendum to GaN')],
        set(), patterns)
    assert [a['pmcid'] for a in articles] == ['PMC1']
    assert rejected == {'duplicate_pmcid': 1, 'title_term:corrigendum': 1}
    assert excluded_by_title == [{'pmcid': 'PMC3', 'term': 'corrigendum',
                                  'title': 'Corrigendum to GaN'}]


def test_mulms_hyphen_rule():
    assert bt.hyphen_split(['4H-SiC', 'high-pressure', '-', 'Ti-6Al-4V', 'x-', 'Fe–N–C']) == [
        '4H', '-', 'SiC', 'high', '-', 'pressure', '-', 'Ti', '-', '6Al', '-', '4V', 'x', '-',
        'Fe–N–C']
    line = bt.words_line('The 4H-SiC wafer was etched at 750 °C.', ChemWordTokenizer())
    assert '4H - SiC' in line and not any('-' in w and len(w) > 1 for w in line.split(' '))


def test_shingle_filter_drops_sharing_sentences_and_leaking_articles():
    pool = ['Thermogravimetric characterization of the samples came first.',     # shares
            'A clean sentence about anodes.',
            'Then thermogravimetric characterization of each pellet.',           # shares
            *(f'Run {k}: thermogravimetric characterization of the SiO2 layers.'  # shares
              for k in range(5)),
            'Another clean sentence of the leaking article.']
    owner = [0, 0, 1, 2, 2, 2, 2, 2, 2]
    kept, dropped = bt.shingle_filter(pool, owner, REFERENCE)
    assert dropped['sentences'] == [0, 2, 3, 4, 5, 6, 7]
    assert dropped['by_article'] == {0: 1, 1: 1, 2: 5} and dropped['articles'] == [2]
    assert kept == [1]                      # the leaking article loses its clean sentence too
    # the n-gram filter would not have caught these: fewer than 8 tokens are shared
    kept_ngram, _ = dapt_mlm.filter_corpus(pool, REFERENCE, 8)
    assert len(kept_ngram) == len(pool)


def test_mulms_sentences_and_pmcids(tmp_path):
    folder = tmp_path / 'mulms'
    folder.mkdir()
    (folder / 'steel_01.conll').write_text('Steel B-MAT\nis O\nhard O\n\n42CrMo4 B-MAT\n\n')
    (folder / 'subfields.json').write_text('{}')
    assert bt.mulms_sentences(str(folder)) == [('steel_01', 'Steel is hard'),
                                               ('steel_01', '42CrMo4')]
    articles = tmp_path / 'articles.json'
    articles.write_text(json.dumps({'steel_01': {'pmcid': 'PMC7', 'score': 1.0},
                                    'steel_02': None, 'steel_03': {'pmcid': None}}))
    assert bt.mulms_pmcids(str(articles)) == ['PMC7']
    if os.path.isdir(bt.MULMS_DIR):            # the converted corpus is not in git
        reference = bt.mulms_sentences()
        assert len(reference) == 10184 and len({doc for doc, _ in reference}) == 50


def make_article(n, leak=()):
    """An article of 6 paragraphs of 4 sentences of its own, plus the leaking sentences."""
    paragraphs = [' '.join(f'Cell {n}-{p}-{s} with a Ni-YSZ anode gave {n + p + s}.5 W cm<sup>−2'
                           f'</sup> at 750 °C.' for s in range(4)) for p in range(6)]
    paragraphs += list(leak)
    body = ''.join(f'<p>{p}</p>' for p in paragraphs)
    return f'<article><body><sec><title>Results</title>{body}</sec></body></article>'.encode()


def run_script(tmp_path, fetcher, name, *flags):
    args = bt.parse_args(['--topic', 'sofc', '--cache-dir', str(tmp_path / 'cache'),
                          '--out', str(tmp_path / f'{name}.txt'),
                          '--report', str(tmp_path / f'{name}.json'), '--target-pool', '60',
                          '--max-sentences', '40', *flags])
    return bt.run(args, fetcher=fetcher, reference=REFERENCE, excluded_pmcids=['PMC1'])


def test_whole_script(tmp_path):
    records = [{'pmcid': f'PMC{n}', 'doi': f'10.1/{n}', 'title': f'Solid oxide fuel cell {n}',
                'abstractText': '', 'pubYear': '2020', 'language': 'eng', 'license': 'cc by',
                'journalInfo': {'journal': {'title': 'J'}}} for n in range(1, 9)]
    texts = {f'PMC{n}': make_article(n, [NGRAM_LEAK]) for n in range(1, 9)}
    # the first article read holds 5 sentences only the shingle test catches: it is dropped
    # whole (the target pool is reached after 3 articles, so a later one might go unread)
    leaker = bf.order_articles([r for r in records if r['pmcid'] != 'PMC1'])[0]['pmcid']
    texts[leaker] = make_article(int(leaker[3:]), [
        f'Run {k}: thermogravimetric characterization of SiO2.' for k in range(5)])
    fake = FakeEuropePMC(records, texts)
    first = run_script(tmp_path, fake, 'first')
    lines = (tmp_path / 'first.txt').read_text(encoding='utf-8').splitlines()
    counts = first['counts']
    # the target is reached before the shingle test, which then drops an article whole
    assert len(lines) == counts['kept'] == 40 and counts['target_reached']
    assert counts['pool'] == 48
    assert first['rejected_by_reason'] == {'mulms_article': 1}
    used = [a['pmcid'] for a in first['articles']]
    assert 'PMC1' not in used
    tok = ChemWordTokenizer()
    for line in lines:                       # the SOFC tokenizer's words, hyphens split out
        assert bt.hyphen_split(tok.tokenize(line)) == line.split(' ')
        assert ' Ni - YSZ anode gave ' in line and ' W cm−2 at 750 ° C .' in line
    assert counts['overlap_dropped'] >= 1
    entry = next(a for a in first['articles'] if a['pmcid'] == leaker)   # read, then dropped
    assert (entry['pool_sentences'], entry['corpus_sentences']) == (0, 0)
    assert first['leakage']['shingle']['articles_dropped'] == [leaker]
    assert first['leakage']['shingle']['by_article'][leaker] == 5
    assert counts['articles_dropped_by_shingles'] == 1
    assert not any('thermogravimetric' in line.lower() for line in lines)
    assert not any(line.startswith('Anode supported cells') for line in lines)
    assert first['selection']['excluded_pmcids'] == ['PMC1']
    assert first['selection']['topic'] == 'sofc' and first['topic'] == 'sofc'
    assert first['corpus']['sha256'] == dapt_mlm.file_sha256(str(tmp_path / 'first.txt'))
    # again from the cache, without the network: the same corpus
    second = run_script(tmp_path, None, 'second', '--offline')
    assert (tmp_path / 'second.txt').read_bytes() == (tmp_path / 'first.txt').read_bytes()
    assert second['counts'] == first['counts'] and second['run']['http_requests'] == 0


def test_cached_search_must_match_the_topic_query(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / bf.HITS_FILE).write_text(json.dumps({'query': bt.query('steel'), 'records': []}))
    with pytest.raises(SystemExit, match='another query'):
        bf.load_hits(str(cache), None, query=bt.query('sofc'))


def test_defaults_follow_the_topic():
    args = bt.parse_args(['--topic', 'steel'])
    assert args.cache_dir.endswith('europepmc_topic_steel')
    assert args.out.endswith('topic_steel_sentences.txt') and args.report.endswith('steel.json')
    assert args.target_pool == bf.TARGET_POOL and args.max_sentences == bf.MAX_SENTENCES
