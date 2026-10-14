"""Fatigue corpus builder: selection, sentence splitting, JATS extraction, leakage filter, sample.

No network: the search and the full texts come from a fake fetcher, and the leakage reference
is a tiny fake MaterioMiner.
"""
import json
import random
import urllib.error
import xml.etree.ElementTree as ET
from collections import Counter

import pytest

import build_fatigue_corpus as bf
import dapt_mlm

MATH = '<mml:math><mml:mi>{}</mml:mi></mml:math>'

JATS = f"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">
<front><article-meta><abstract><p>The abstract is not part of the body and must not be used.</p>
</abstract></article-meta></front>
<body>
<sec><title>1. Introduction</title>
<p>Fatigue limits of Ti-6Al-4V were measured<xref ref-type="bibr" rid="B1">1</xref>, see Fig.
<xref ref-type="fig" rid="F1">1</xref>. The stress ratio was R = <inline-formula><mml:math>
<mml:mo>&#8722;</mml:mo><mml:mn>1</mml:mn></mml:math></inline-formula> in all tests
[<xref ref-type="bibr" rid="B2">2</xref>&#8211;<xref ref-type="bibr" rid="B4">4</xref>]. Lives above
10<sup>7</sup> cycles and Al<sub>2</sub>O<sub>3</sub> inclusions were noted<sup><xref
ref-type="bibr" rid="B5">5</xref>,<xref ref-type="bibr" rid="B6">6</xref></sup>. Data are in
Supplementary Fig. <xref ref-type="media" rid="M1">2</xref> and Ref. <xref ref-type="bibr"
rid="B7">7</xref>.</p>
<fig id="F1"><label>Figure 1</label><caption><p>Caption sentence that must not be used.</p>
</caption></fig>
<p>Text before the figure is complete.<fig id="F2"><caption><title>Inner caption.</title>
<p>Caption inside a paragraph must be dropped.</p></caption></fig> Text after the figure goes on
here.</p>
<p>The relation is given by</p>
<disp-formula id="E1"><mml:math><mml:mi>x</mml:mi></mml:math></disp-formula>
<p>where the symbols are defined below. The long formula
<inline-formula>{MATH.format('x' * 40)}</inline-formula> is not kept. The last sentence of this
paragraph stays.</p>
<list list-type="bullet"><list-item>
<p>The first list item is a full sentence about alloys.</p></list-item></list>
<table-wrap id="T1"><caption><p>Table caption that must not be used.</p></caption><table><tr>
<td>cell cell cell cell cell cell</td></tr></table></table-wrap>
</sec>
<sec><title>Acknowledgements</title><p>This work was supported by a grant from nowhere in
particular.</p></sec>
</body>
<back><ref-list><ref><mixed-citation>Reference text that must not be used either.
</mixed-citation></ref></ref-list></back>
</article>"""

EXPECTED = [
    'Fatigue limits of Ti-6Al-4V were measured, see Fig. 1.',
    'The stress ratio was R = −1 in all tests.',
    'Lives above 10^7 cycles and Al2O3 inclusions were noted.',
    'Data are in Supplementary Fig. 2 and Ref.',
    'Text before the figure is complete.',
    'Text after the figure goes on here.',
    'The last sentence of this paragraph stays.',
    'The first list item is a full sentence about alloys.',
]

# the fake MaterioMiner: one sentence, tokens separated by spaces as in the CoNLL files
REFERENCE = [('mm.conll', 'Fatigue cracks initiated at the surface of the specimens under '
                          'constant amplitude loading .')]
LEAKING = 'We found that fatigue cracks initiated at the surface of the specimens in all tests.'
REPEATED = 'Crack growth rates were recorded for each specimen in the test series.'


def item(sentences, status='ok'):
    return {'article': {'pmcid': 'PMC0'}, 'status': status, 'error': None,
            'sentences': list(sentences), 'counts': {}, 'downloaded': False}


# ---- query and settings ----

def test_query_is_built_from_the_term_lists():
    assert bf.QUERY.startswith('OPEN_ACCESS:y AND HAS_FT:y AND TITLE:fatigue AND (TITLE:("alloy"')
    for term in bf.MATERIAL_TERMS:
        assert f'"{term}"' in bf.QUERY
    assert bf.QUERY.count('TITLE:(') == 1 and bf.QUERY.count('ABSTRACT:(') == 1
    assert bf.selection()['query'] == bf.QUERY
    assert set(bf.EXCLUDED_DOIS) == {'10.1007/s11340-021-00758-x', '10.1038/ncomms7164',
                                     '10.1038/s41467-020-19071-7', '10.3390/met10111458'}


def test_leakage_ngram_is_the_one_of_dapt_mlm():
    assert bf.NGRAM == dapt_mlm.parse_args([]).ngram == 8


# ---- selection ----

def hit(pmcid, title, abstract='', doi=None, language='eng'):
    return {'pmcid': pmcid, 'doi': doi, 'title': title, 'abstract': abstract,
            'language': language, 'year': '2020', 'journal': 'J', 'license': 'cc by'}


def test_local_filters():
    records = [
        hit('PMC1', 'Fatigue crack growth in a titanium alloy.'),
        hit('PMC2', 'Low-cycle fatigue behaviour', 'of Inconel 718 at 650 C'),
        hit('PMC3', 'Fatigue of Ti–6Al–4V welded joints'),            # en dashes
        hit('PMC4', 'Fatigue of HfO&lt;sub&gt;2&lt;/sub&gt; films', 'with metal electrodes'),
        hit(None, 'Fatigue of steel'),
        hit('PMC6', 'Fatigue of steel', doi='10.3390/MET10111458'),
        hit('PMC7', 'Fatigue of steel', language='ger'),
        hit('PMC8', 'Cyclic loading of steel'),
        hit('PMC9', 'Fatigue of concrete beams', 'reinforced with polymer fibres'),
        hit('PMC10', 'Ironic fatigue'),                                       # whole tokens only
        hit('PMC11', 'Iron deficiency and fatigue in patients'),
        hit('PMC12', 'Muscle fatigue and the magnesium status of athletes'),
        hit('PMC13', 'Author Correction: Fatigue of steel'),
        hit('PMC1', 'Fatigue crack growth in a titanium alloy.'),
        hit('PMC15', 'Patient-specific titanium implants and their fatigue life'),
    ]
    articles, rejected, excluded = bf.select_articles(records)
    assert [a['pmcid'] for a in articles] == ['PMC1', 'PMC2', 'PMC3', 'PMC4', 'PMC15']
    assert bf.rejection(records[4]) == 'no_pmcid'
    assert bf.rejection(records[5]) == 'excluded_doi'
    assert bf.rejection(records[6]) == 'not_english'
    assert bf.rejection(records[7]) == 'no_fatigue_in_title'
    assert bf.rejection(records[8]) == 'no_material_term'
    assert bf.rejection(records[9]) == 'no_material_term'
    assert dict(rejected) == {'no_pmcid': 1, 'excluded_doi': 1, 'not_english': 1,
                              'no_fatigue_in_title': 1, 'no_material_term': 2,
                              'title_term:patients': 1, 'title_term:muscle': 1,
                              'title_term:author correction': 1, 'duplicate_pmcid': 1}
    assert [(e['pmcid'], e['term']) for e in excluded] == [
        ('PMC11', 'patients'), ('PMC12', 'muscle'), ('PMC13', 'author correction')]


def test_order_is_sorted_then_shuffled_with_seed_zero():
    articles = [{'pmcid': f'PMC{n}'} for n in (9, 100, 25, 3, 7, 1000, 12)]
    expected = sorted(a['pmcid'] for a in articles)
    random.Random(0).shuffle(expected)
    ordered = bf.order_articles(articles)
    assert [a['pmcid'] for a in ordered] == expected
    assert bf.order_articles(list(reversed(articles))) == ordered      # input order is irrelevant
    assert [a['pmcid'] for a in ordered] != sorted(a['pmcid'] for a in articles)


def test_slim_keeps_the_used_fields():
    core = {'pmcid': 'PMC1', 'doi': '10.1/x', 'title': 'T', 'abstractText': 'A', 'pubYear': '2020',
            'journalInfo': {'journal': {'title': 'Journal'}}, 'license': 'cc by',
            'language': 'eng', 'source': 'MED', 'authorList': 'ignored'}
    assert bf.slim(core) == {'pmcid': 'PMC1', 'doi': '10.1/x', 'title': 'T', 'abstract': 'A',
                             'year': '2020', 'journal': 'Journal', 'license': 'cc by',
                             'language': 'eng', 'source': 'MED'}
    assert bf.slim({})['journal'] is None


# ---- sentences ----

def test_sentence_splitter_abbreviations_and_decimals():
    text = ('The specimens were tested according to ASTM E466 (Fig. 2). The ratio was R = 0.1 '
            'and the frequency was approx. 20 Hz. See Eq. 3 and Figs. 4 and 5. Smith et al. '
            'The results agree. The alloy (e.g. Ti-6Al-4V) was heat treated, i.e. Annealed at '
            '700 °C. The grain size was ca. 5 µm vs. 12 µm. Samples No. 5 and 6 failed early. '
            'We used 3.5 wt. % Cu and 2 at. % Zn (cf. Ref. 12). Is this a sentence? Yes! '
            'The value was 0.5. 10 specimens broke. (See Fig. 3.) The end. “Quoted.” Next one.')
    assert bf.split_sentences(text) == [
        'The specimens were tested according to ASTM E466 (Fig. 2).',
        'The ratio was R = 0.1 and the frequency was approx. 20 Hz.',
        'See Eq. 3 and Figs. 4 and 5.',
        'Smith et al. The results agree.',                      # never a break after "et al."
        'The alloy (e.g. Ti-6Al-4V) was heat treated, i.e. Annealed at 700 °C.',
        'The grain size was ca. 5 µm vs. 12 µm.',
        'Samples No. 5 and 6 failed early.',
        'We used 3.5 wt. % Cu and 2 at. % Zn (cf. Ref. 12).',
        'Is this a sentence?',
        'Yes!',
        'The value was 0.5.',
        '10 specimens broke.',
        '(See Fig. 3.)',
        'The end.',
        '“Quoted.”',
        'Next one.']
    assert bf.split_sentences('') == [] and bf.split_sentences('One sentence') == ['One sentence']
    # a period before a lowercase word is not a boundary, nor one inside a number
    lowercase = 'It was 2.5 mm thick. then 3.5 mm'
    assert bf.split_sentences(lowercase) == [lowercase]


def test_length_filters():
    assert bf.length_rejection('Too short for us.') == 'too_short'          # 4 words
    assert bf.length_rejection('Just long enough, this one.') is None        # 5 words
    assert bf.length_rejection(' '.join(['word'] * 121)) == 'too_long'
    assert bf.length_rejection(' '.join(['word'] * 120)) is None
    composition = 'Fe 0.25 C 1.2 Mn 0.3 Si 0.02 P 0.01 S 99.2 - 100 .'
    assert bf.length_rejection(composition) == 'non_alphabetic'
    assert bf.length_rejection('The alloy had 0.25 C and 1.2 Mn, 0.3 Si (wt.%).') is None


# ---- JATS extraction ----

def test_extraction_from_jats():
    sentences, counts = bf.article_sentences(JATS.encode('utf-8'))
    assert sentences == EXPECTED
    assert counts['paragraphs'] == 5
    # cut: 'The relation is given by' and 'where the symbols ...' (display formula between the
    # paragraphs), 'The long formula' and 'is not kept.' (a long inline formula)
    assert counts['cut_by_formula'] == 4
    assert counts['raw_sentences'] == len(EXPECTED) + 4 and counts['too_short'] == 0
    assert bf.BREAK not in ''.join(sentences)
    joined = ' '.join(sentences)
    for unused in ('abstract', 'Caption', 'Table caption', 'Reference text', 'grant'):
        assert unused not in joined


def test_formula_text_and_cuts():
    root = ET.fromstring(f'<p xmlns:mml="http://www.w3.org/1998/Math/MathML">a<inline-formula>'
                         f'{MATH.format("N")}</inline-formula>b</p>')
    assert bf.inline_text(root) == 'aNb'
    short = '<mml:math><mml:mi>d</mml:mi><mml:mspace/><mml:mi>N</mml:mi></mml:math>'
    root = ET.fromstring('<p xmlns:mml="http://www.w3.org/1998/Math/MathML">x '
                         f'<inline-formula>{short}</inline-formula> y '
                         '<inline-formula><inline-graphic/></inline-formula> z</p>')
    assert bf.inline_text(root) == f'x d N y {bf.BREAK} z'          # no MathML: cut
    # display formula in a paragraph: the unfinished sentence before and the lowercase after go
    text = f'It reads: {bf.BREAK} where N is the life. The next sentence is whole.'
    assert bf.paragraph_sentences(text) == (['The next sentence is whole.'], 2)
    assert bf.paragraph_sentences(f'A full sentence first. And {bf.BREAK} so on.') == (
        ['A full sentence first.'], 2)
    assert bf.paragraph_sentences(f'Whole before. {bf.BREAK} Whole after.') == (
        ['Whole before.', 'Whole after.'], 0)


def test_citation_clusters_and_xref_types():
    root = ET.fromstring(
        '<p>Resorption.<xref ref-type="ref" rid="a">1</xref>, <xref ref-type="ref" rid="b">2'
        '</xref> The next one. See Eq. (<xref ref-type="" rid="c">4</xref>) and Fig. '
        '<xref ref-type="fig" rid="d">3</xref>; also<xref rid="e">9</xref>.</p>')
    # citations go with the separators between them; an equation or figure reference stays
    expected = 'Resorption. The next one. See Eq. (4) and Fig. 3; also.'
    assert bf.tidy(bf.inline_text(root)) == expected


def test_citation_leftovers_are_tidied():
    assert bf.tidy('as shown [ – ] and ( ) here , and there .') == 'as shown and here, and there.'
    assert bf.tidy('previous work,,, the shape') == 'previous work, the shape'
    assert bf.tidy('measured at 5 K,.') == 'measured at 5 K.'
    dirty = 'soft' + chr(0xAD) + 'hyphen and' + chr(0x200B) + 'zero  width'   # soft hyphen, ZWSP
    assert bf.tidy(dirty) == 'softhyphen andzero width'


def test_xml_without_body_or_paragraphs():
    assert bf.article_sentences(b'<article><front/></article>') == ([], Counter())
    with pytest.raises(ET.ParseError):
        bf.article_sentences(b'<article><body><p>cut off')


# ---- leakage filter and pool ----

def test_pool_drops_duplicates_and_overlaps_with_materiominer():
    items = [item(['Crack growth rates were recorded for each specimen in the test series.',
                   LEAKING]),
             item(['Crack growth rates were recorded for each specimen in the test series.',
                   'A different sentence about welded joints and their residual stress fields.'])]
    pool, owner, dropped = bf.build_pool(items, REFERENCE)
    assert pool == ['Crack growth rates were recorded for each specimen in the test series.',
                    'A different sentence about welded joints and their residual stress fields.']
    assert owner == [0, 1]
    assert [d['sentence'] for d in dropped['overlap']] == [LEAKING]
    assert dropped['overlap'][0]['docs'] == ['mm.conll']
    assert len(dropped['duplicate']) == 1
    # the same as dapt_mlm's own filter, which the script imports instead of copying
    kept, _ = dapt_mlm.filter_corpus([s for it in items for s in it['sentences']], REFERENCE, 8)
    assert pool == kept
    # a 7-token overlap is not enough
    short = item(['Fatigue cracks initiated at the surface of a different part of it.'])
    assert bf.build_pool([short], REFERENCE)[0] == short['sentences']


def test_collect_takes_the_smallest_prefix_whatever_the_chunk():
    sents = [f'Sentence number {i} of the test corpus is here.' for i in range(1, 9)]
    items = {'A': item(sents[0:2]), 'B': item(sents[0:2]), 'C': item(sents[2:4]),
             'D': item(sents[4:6]), 'E': item(sents[6:8]), 'F': item(['Never read at all.'])}
    order = [{'pmcid': k} for k in 'ABCDEF']
    for chunk in (1, 2, 3, 6):
        used, reached = bf.collect(order, lambda a: items[a['pmcid']], [], target=5, chunk=chunk)
        # B repeats A, so the 5th pool sentence is the first of D
        assert reached and used == [items[k] for k in 'ABCD']
    used, reached = bf.collect(order, lambda a: items[a['pmcid']], [], target=100, chunk=2)
    assert not reached and len(used) == 6


def test_collect_stops_after_failures_in_a_row():
    order = [{'pmcid': f'PMC{n}'} for n in range(20)]
    with pytest.raises(SystemExit, match='in a row failed'):
        bf.collect(order, lambda a: item([], status='failed'), [], target=10)


def test_sample_is_fixed_and_in_sampled_order():
    pool = [f'Sentence {i} is long enough to count.' for i in range(100)]
    picked = bf.sample_pool(pool, 30)
    assert picked == random.Random(0).sample(range(100), 30) == bf.sample_pool(pool, 30)
    assert len(set(picked)) == 30 and picked != sorted(picked)
    assert sorted(bf.sample_pool(pool, 500)) == list(range(100))      # a small pool is kept whole


# ---- fetching ----

class Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


class Clock:
    """A clock that only moves when the fetcher sleeps."""

    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def scripted_fetcher(outcomes, **kwargs):
    """A Fetcher whose requests produce `outcomes` in turn (bytes, or an exception to raise)."""
    clock, calls = Clock(), []

    def opener(request, timeout):
        calls.append((clock.now, request.full_url, timeout))
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return Response(outcome)

    fetcher = bf.Fetcher(opener=opener, sleep=clock.sleep, clock=clock.monotonic, **kwargs)
    return fetcher, clock, calls


def http_error(code, headers=None):
    return urllib.error.HTTPError('http://x', code, 'err', headers or {}, None)


def test_fetcher_spaces_requests_and_sends_params():
    fetcher, clock, calls = scripted_fetcher([b'1', b'2', b'3'])
    bodies = [fetcher.get('http://x/search', {'q': 'a b'}) for _ in range(3)]
    assert bodies == [b'1', b'2', b'3']
    assert [t for t, _, _ in calls] == [0.0, 0.25, 0.5]               # at most 4 per second
    assert all(url == 'http://x/search?q=a+b' and timeout == bf.TIMEOUT
               for _, url, timeout in calls)
    assert bf.MIN_INTERVAL >= 0.2 and fetcher.requests == 3


def test_fetcher_retries_with_backoff():
    fetcher, clock, calls = scripted_fetcher([http_error(503), TimeoutError('slow'), b'done'])
    assert fetcher.get('http://x') == b'done'
    assert clock.sleeps == [2.0, 4.0] and len(calls) == 3
    fetcher, clock, calls = scripted_fetcher([http_error(429, {'Retry-After': '7'}), b'ok'])
    assert fetcher.get('http://x') == b'ok' and clock.sleeps == [7.0]


def test_fetcher_gives_up_and_distinguishes_permanent_errors():
    fetcher, clock, calls = scripted_fetcher([http_error(503)] * 5)
    with pytest.raises(bf.FetchError, match='HTTP 503') as info:
        fetcher.get('http://x')
    assert not info.value.permanent and len(calls) == 5
    assert clock.sleeps == [2.0, 4.0, 8.0, 16.0]
    fetcher, clock, calls = scripted_fetcher([http_error(404), b'never used'])
    with pytest.raises(bf.FetchError, match='HTTP 404') as info:
        fetcher.get('http://x')
    assert info.value.permanent and len(calls) == 1


class FakeEuropePMC:
    """Stands in for Fetcher: a search page and the full texts of a few articles."""

    def __init__(self, records, texts):
        self.records, self.texts, self.urls, self.requests = records, texts, [], 0

    def get(self, url, params=None):
        self.urls.append(url)
        self.requests += 1
        if url == bf.SEARCH_URL:
            first = params['cursorMark'] == '*'
            page = {'hitCount': len(self.records),
                    'resultList': {'result': self.records if first else []}}
            if first:
                page['nextCursorMark'] = 'next'
            return json.dumps(page).encode()
        pmcid = url.split('/')[-2]
        if pmcid not in self.texts:
            raise bf.FetchError('HTTP 404', permanent=True)
        return self.texts[pmcid]


def test_cached_xml_is_resumable(tmp_path):
    fake = FakeEuropePMC([], {'PMC1': b'<article/>'})
    assert bf.cached_xml('PMC1', str(tmp_path), fake) == (b'<article/>', None, True)
    assert bf.cached_xml('PMC1', str(tmp_path), fake) == (b'<article/>', None, False)
    assert len(fake.urls) == 1                                    # the second read is the cache
    assert bf.cached_xml('PMC2', str(tmp_path), fake) == (None, 'HTTP 404', False)
    assert bf.cached_xml('PMC2', str(tmp_path), fake) == (None, 'HTTP 404', False)
    assert len(fake.urls) == 2                                    # a permanent failure is kept
    assert bf.cached_xml('PMC3', str(tmp_path), None) == (None, 'not cached (offline)', False)
    assert not list(tmp_path.glob('*.tmp'))


def test_read_article_failures(tmp_path):
    fake = FakeEuropePMC([], {'PMC1': b'<article><body><p>cut off',
                              'PMC2': b'<article><body/></article>',
                              'PMC3': JATS.encode('utf-8')})
    bad, empty, good = (bf.read_article({'pmcid': f'PMC{n}'}, str(tmp_path), fake)
                        for n in (1, 2, 3))
    assert bad['status'] == 'failed' and bad['error'].startswith('invalid XML')
    assert empty['status'] == 'no_text' and empty['sentences'] == []
    assert good['status'] == 'ok' and good['sentences'] == EXPECTED and good['downloaded']
    missing = bf.read_article({'pmcid': 'PMC9'}, str(tmp_path), fake)
    assert missing['status'] == 'failed' and missing['error'] == 'HTTP 404'


# ---- the whole script ----

def make_article(n):
    """An article of 8 paragraphs of 4 sentences that are its own, plus two that every article
    has: one that leaks from MaterioMiner and one that is only repeated."""
    paragraphs = [' '.join(f'Specimen {n}-{p}-{s} of the alloy failed after {n + p + s} thousand '
                           f'cycles under load.' for s in range(4)) for p in range(8)]
    paragraphs += [LEAKING, REPEATED]
    body = ''.join(f'<p>{p}</p>' for p in paragraphs)
    return f'<article><body><sec><title>Results</title>{body}</sec></body></article>'.encode()


def run_script(tmp_path, fetcher, name, *flags):
    args = bf.parse_args(['--cache-dir', str(tmp_path / 'cache'),
                          '--out', str(tmp_path / f'{name}.txt'),
                          '--report', str(tmp_path / f'{name}.json'), '--target-pool', '100',
                          '--max-sentences', '80', *flags])
    return bf.run(args, fetcher=fetcher, reference=REFERENCE)


def test_whole_script_is_deterministic_and_leak_free(tmp_path):
    records = [{'pmcid': f'PMC{n}', 'doi': f'10.1/{n}', 'title': f'Fatigue of steel {n}',
                'abstractText': 'About a steel.', 'pubYear': '2020', 'language': 'eng',
                'license': 'cc by', 'journalInfo': {'journal': {'title': 'J'}}}
               for n in range(12)]
    records.append({'pmcid': 'PMC99', 'title': 'Fatigue in patients with steel pins',
                    'abstractText': '', 'language': 'eng'})
    fake = FakeEuropePMC(records, {f'PMC{n}': make_article(n) for n in range(12)})
    first = run_script(tmp_path, fake, 'first')

    out = (tmp_path / 'first.txt').read_bytes().decode('utf-8')
    lines = out.splitlines()
    counts = first['counts']
    assert len(lines) == counts['kept'] == 80 and len(set(lines)) == 80
    assert counts['hits'] == 13 and counts['eligible_articles'] == 12
    assert first['rejected_by_reason'] == {'title_term:patients': 1}
    assert counts['pool'] >= 100 and counts['target_reached']
    assert counts['pool'] == (counts['raw_sentences'] - counts['cut_by_formula']
                              - counts['length_filtered'] - counts['duplicates_dropped']
                              - counts['overlap_dropped'])
    # the leaking sentence is in every article: the first is an overlap, the rest duplicates of it
    assert counts['overlap_dropped'] == 1
    assert counts['duplicates_dropped'] == 2 * (counts['articles_used'] - 1)
    assert LEAKING not in lines and bf.BREAK not in out
    assert sum('Crack growth rates were recorded' in line for line in lines) <= 1
    assert first['corpus']['sha256'] == dapt_mlm.file_sha256(str(tmp_path / 'first.txt'))
    assert first['leakage']['overlap_by_document'] == {'mm.conll': 1}
    used = [a['pmcid'] for a in first['articles']]
    assert 0 < len(used) < 12 and 'PMC99' not in used
    assert sum(a['corpus_sentences'] for a in first['articles']) == 80
    assert dapt_mlm.read_sentences(str(tmp_path / 'first.txt')) == (lines, 0)   # accepted by dapt
    assert first['selection']['query'] == bf.QUERY
    assert first['selection']['excluded_dois'] == list(bf.EXCLUDED_DOIS)

    # again from the cache, without the network: the same corpus
    second = run_script(tmp_path, None, 'second', '--offline')
    assert (tmp_path / 'second.txt').read_bytes() == (tmp_path / 'first.txt').read_bytes()
    assert second['corpus']['sha256'] == first['corpus']['sha256']
    assert second['counts'] == first['counts'] and second['articles'] == first['articles']
    assert second['run']['http_requests'] == 0
    assert second['run']['used_articles_downloaded_now'] == 0
    assert first['run']['used_articles_downloaded_now'] >= len(used)


def test_offline_needs_the_cached_search(tmp_path):
    with pytest.raises(SystemExit, match='hits.json'):
        run_script(tmp_path, None, 'nothing', '--offline')


def test_cached_search_must_match_the_query(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / bf.HITS_FILE).write_text(json.dumps({'query': 'something else', 'records': []}))
    with pytest.raises(SystemExit, match='another query'):
        bf.load_hits(str(cache), None)
