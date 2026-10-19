"""SOFC corpus builder: selection, inline superscripts, SOFC-Exp's tokens, leakage filter.

No network: the search and the full texts come from a fake fetcher, and the leakage reference
is a tiny fake SOFC-Exp.
"""
import json

import pytest

import build_fatigue_corpus as bf
import build_sofc_corpus as bs
import dapt_mlm
from chem_tokenize import ChemWordTokenizer


class FakeEuropePMC:
    """Stands in for Fetcher: a search page and the full texts of a few articles."""

    def __init__(self, records, texts):
        self.records, self.texts, self.requests = records, texts, 0

    def get(self, url, params=None):
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


def hit(pmcid, title, language='eng'):
    return {'pmcid': pmcid, 'title': title, 'language': language}


def test_query_and_local_filters():
    assert bs.QUERY == ('OPEN_ACCESS:y AND HAS_FT:y AND (TITLE:"solid oxide" OR TITLE:SOFC OR '
                        'TITLE:SOFCs)')
    sofc_exp = {'PMC1'}
    cases = [({'title': 'Solid oxide fuel cells'}, 'no_pmcid'),
             (hit('PMC1', 'Solid oxide fuel cells'), 'sofc_exp_article'),
             (hit('PMC2', 'Solid oxide fuel cells', 'ger'), 'not_english'),
             (hit('PMC3', 'Proton conducting fuel cells'), 'no_title_term'),
             (hit('PMC4', 'Correction: anodes for SOFCs'), 'title_term:correction'),
             (hit('PMC5', 'A Ni-YSZ anode for solid-oxide fuel cells'), None),
             (hit('PMC6', 'SOFC-based power systems'), None),
             (hit('PMC7', 'Protonic SOFCs at 600 °C'), None)]
    for record, reason in cases:
        assert bs.rejection(record, sofc_exp) == reason, record
    articles, rejected, excluded = bs.select_articles([r for r, _ in cases] + [cases[5][0]],
                                                      sofc_exp)
    assert [a['pmcid'] for a in articles] == ['PMC5', 'PMC6', 'PMC7']
    assert rejected['duplicate_pmcid'] == 1 and rejected['sofc_exp_article'] == 1
    assert excluded == [{'pmcid': 'PMC4', 'term': 'correction',
                         'title': 'Correction: anodes for SOFCs'}]


SUP_XML = ('<article><body><p>The cell gave 0.5 W cm<sup>−2</sup> with Fe<sup>3+</sup> ions and '
           'a life of 10<sup>4</sup> h in all of the tests.</p></body></article>').encode()


def test_superscripts_are_inline_as_in_sofc_exps_texts():
    inline, _ = bf.article_sentences(SUP_XML, sup=bs.SUP)
    assert inline == ['The cell gave 0.5 W cm−2 with Fe3+ ions and a life of 104 h in all of '
                      'the tests.']
    caret, _ = bf.article_sentences(SUP_XML)          # the fatigue corpus keeps its carets
    assert caret == ['The cell gave 0.5 W cm^−2 with Fe^3+ ions and a life of 10^4 h in all of '
                     'the tests.']


def test_words_line():
    tok = ChemWordTokenizer()
    assert bs.words_line('The cell reached 0.5 W cm−2 at 800 °C with Ni-YSZ.', tok) == \
        'The cell reached 0.5 W cm−2 at 800 ° C with Ni - YSZ .'


def test_sofc_exp_articles_and_sentences_from_the_committed_files():
    pmcids = bs.sofc_exp_pmcids()
    assert len(pmcids) == 45 and pmcids[0] == 'PMC3564701'
    reference = bs.sofc_exp_sentences()
    assert len(reference) == 9466 and {doc for doc, _ in reference} == set(pmcids)


LEAKING = 'Anode supported cells were sintered at 1400 °C for 4 h in air before the tests.'
REFERENCE = [('PMC1', LEAKING.replace('before the tests.', 'and then reduced.'))]


def make_article(n):
    """An article of 6 paragraphs of 4 sentences of its own, and one that leaks from SOFC-Exp."""
    paragraphs = [' '.join(f'Cell {n}-{p}-{s} with a Ni-YSZ anode gave {n + p + s}.5 W cm<sup>−2'
                           f'</sup> at 750 °C.' for s in range(4)) for p in range(6)]
    paragraphs.append(LEAKING)
    body = ''.join(f'<p>{p}</p>' for p in paragraphs)
    return f'<article><body><sec><title>Results</title>{body}</sec></body></article>'.encode()


def run_script(tmp_path, fetcher, name, *flags):
    args = bs.parse_args(['--cache-dir', str(tmp_path / 'cache'),
                          '--out', str(tmp_path / f'{name}.txt'),
                          '--report', str(tmp_path / f'{name}.json'), '--target-pool', '60',
                          '--max-sentences', '50', *flags])
    return bs.run(args, fetcher=fetcher, reference=REFERENCE, excluded_pmcids=['PMC1'])


def test_whole_script(tmp_path):
    records = [{'pmcid': f'PMC{n}', 'doi': f'10.1/{n}', 'title': f'Solid oxide fuel cell {n}',
                'abstractText': '', 'pubYear': '2020', 'language': 'eng', 'license': 'cc by',
                'journalInfo': {'journal': {'title': 'J'}}} for n in range(1, 8)]
    fake = FakeEuropePMC(records, {f'PMC{n}': make_article(n) for n in range(1, 8)})
    first = run_script(tmp_path, fake, 'first')
    lines = (tmp_path / 'first.txt').read_text(encoding='utf-8').splitlines()
    counts = first['counts']
    assert len(lines) == counts['kept'] == 50 and counts['target_reached']
    assert first['rejected_by_reason'] == {'sofc_exp_article': 1}
    assert 'PMC1' not in [a['pmcid'] for a in first['articles']]
    tok = ChemWordTokenizer()
    for line in lines:                       # the words of SOFC-Exp's tokenizer, spaced
        assert tok.tokenize(line) == line.split(' ') and ' W cm−2 at 750 ° C .' in line
    assert counts['overlap_dropped'] == 1 and first['leakage']['overlap_by_document'] == \
        {'PMC1': 1}
    assert not any(line.startswith('Anode supported cells') for line in lines)
    assert first['selection']['excluded_pmcids'] == ['PMC1']
    assert first['corpus']['sha256'] == dapt_mlm.file_sha256(str(tmp_path / 'first.txt'))
    assert first['corpus']['words'] == sum(len(line.split(' ')) for line in lines)
    # again from the cache, without the network: the same corpus
    second = run_script(tmp_path, None, 'second', '--offline')
    assert (tmp_path / 'second.txt').read_bytes() == (tmp_path / 'first.txt').read_bytes()
    assert second['counts'] == first['counts'] and second['run']['http_requests'] == 0


def test_cached_search_must_match_the_sofc_query(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / bf.HITS_FILE).write_text(json.dumps({'query': bf.QUERY, 'records': []}))
    with pytest.raises(SystemExit, match='another query'):
        bf.load_hits(str(cache), None, query=bs.QUERY)
