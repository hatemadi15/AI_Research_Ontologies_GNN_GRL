"""MuLMS as flat CoNLL files (mulms.py): the flattening rule, IOB2, the order of the sentences,
the checks, deterministic outputs and report, and the parquet cache. Synthetic rows: no
network and no pyarrow (CI installs requirements-dev.txt only)."""
import hashlib
import importlib.util
import itertools
import json
import random
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

import conll
import mulms

NONE_DROPPED = {'nested': 0, 'same_span': 0, 'crossing': 0}
TOY = {'documents': 2, 'sentences': 4, 'entities': 12, 'types': 8}      # the counts of _corpus()


def _row(doc, begin, tokens, entities=(), split='train1', category='pemfc'):
    """A sentence as read_rows gives it, with entities [(first, last, type)] (the text and the
    character offsets of NER_labels are not used)."""
    n = len(entities)
    return {'doc_id': doc, 'category': category, 'data_split': split, 'tokens': list(tokens),
            'beginOffset': begin,
            'NER_labels': {'text': ['?'] * n, 'id': list(range(n)),
                           'value': [kind for _, _, kind in entities],
                           'begin': ['0'] * n, 'end': ['0'] * n,
                           'tokenIndices': [[first, last] for first, last, _ in entities]}}


def _corpus():
    """d1 (pemfc, train1): two sentences, the later one first, at offsets given as strings
    ('100' < '9' as text); d2 (steel, test): a sentence with every kind of drop, then one
    without entities."""
    return [
        _row('d1', '100', ['Pt', 'at', '80', '°C'],
             [(0, 0, 'MAT'), (2, 3, 'VALUE'), (2, 2, 'NUM'), (3, 3, 'UNIT')]),
        _row('d1', '9', ['A', 'PEMFC', 'stack'], [(1, 2, 'DEV'), (1, 1, 'MAT')]),
        _row('d2', 0, ['Steel', 'X', 'was', 'tested', 'by', 'XRD'],
             [(0, 1, 'MAT'), (0, 1, 'SAMPLE'), (1, 1, 'MAT'), (1, 2, 'FORM'),
              (5, 5, 'TECHNIQUE'), (5, 5, 'TECHNIQUE')], 'test', 'steel'),
        _row('d2', 40, ['No', 'entities', '.'], split='test', category='steel'),
    ]


# ---- the flattening rule ----

def test_flatten_keeps_the_outermost_entities():
    # NUM and UNIT inside VALUE, MAT inside MAT
    kept, dropped = mulms.flatten([(2, 3, 'NUM'), (2, 5, 'VALUE'), (4, 5, 'UNIT'),
                                   (0, 1, 'MAT'), (1, 1, 'MAT')])
    assert kept == [(0, 1, 'MAT'), (2, 5, 'VALUE')]
    assert dropped == {'nested': 3, 'same_span': 0, 'crossing': 0}
    assert mulms.flatten([]) == ([], NONE_DROPPED)


def test_flatten_equal_spans_keep_the_type_first_in_type_order():
    kept, dropped = mulms.flatten([(0, 1, 'NUM'), (0, 1, 'VALUE'), (3, 4, 'RANGE'),
                                   (3, 4, 'VALUE'), (6, 6, 'UNIT'), (6, 6, 'MEASUREMENT'),
                                   (6, 6, 'MAT')])
    assert kept == [(0, 1, 'VALUE'), (3, 4, 'VALUE'), (6, 6, 'MEASUREMENT')]
    assert dropped == {'nested': 0, 'same_span': 4, 'crossing': 0}
    assert mulms.flatten([(0, 1, 'VALUE'), (0, 1, 'NUM')])[0] == [(0, 1, 'VALUE')]


def test_flatten_exact_duplicates_count_once():
    kept, dropped = mulms.flatten([(0, 2, 'MAT'), (0, 2, 'MAT'), (0, 2, 'MAT')])
    assert kept == [(0, 2, 'MAT')]
    assert dropped == {'nested': 0, 'same_span': 2, 'crossing': 0}
    # duplicates inside another entity are nested: step 1 comes first
    kept, dropped = mulms.flatten([(0, 3, 'VALUE'), (0, 0, 'NUM'), (0, 0, 'NUM')])
    assert kept == [(0, 3, 'VALUE')]
    assert dropped == {'nested': 2, 'same_span': 0, 'crossing': 0}


def test_flatten_crossing_keeps_the_entity_that_starts_first():
    kept, dropped = mulms.flatten([(2, 4, 'FORM'), (0, 2, 'MAT')])
    assert kept == [(0, 2, 'MAT')]
    assert dropped == {'nested': 0, 'same_span': 0, 'crossing': 1}
    # the type order plays no part: the earlier entity stays though its type comes later
    assert mulms.flatten([(1, 2, 'MEASUREMENT'), (0, 1, 'UNIT')])[0] == [(0, 1, 'UNIT')]
    # a chain: the middle entity crosses both others, the last one only the dropped middle one
    kept, dropped = mulms.flatten([(0, 2, 'MAT'), (2, 4, 'FORM'), (4, 6, 'SAMPLE')])
    assert kept == [(0, 2, 'MAT'), (4, 6, 'SAMPLE')]
    assert dropped == {'nested': 0, 'same_span': 0, 'crossing': 1}


def test_an_entity_inside_a_dropped_crossing_entity_stays_dropped():
    """Step 1 looks at every entity of the sentence. UNIT (4..5) is inside VALUE (1..5), which
    crosses MAT (0..2) and is dropped by step 3; UNIT is not brought back, although it overlaps
    no kept entity: it counts as nested and its tokens are O."""
    kept, dropped = mulms.flatten([(0, 2, 'MAT'), (1, 5, 'VALUE'), (4, 5, 'UNIT')])
    assert kept == [(0, 2, 'MAT')]
    assert dropped == {'nested': 1, 'same_span': 0, 'crossing': 1}
    assert mulms.iob2(6, kept) == ['B-MAT', 'I-MAT', 'I-MAT', 'O', 'O', 'O']
    # inside the kept entity of the two, it is nested all the same
    kept, dropped = mulms.flatten([(0, 2, 'MAT'), (1, 5, 'VALUE'), (1, 2, 'NUM')])
    assert kept == [(0, 2, 'MAT')]
    assert dropped == {'nested': 1, 'same_span': 0, 'crossing': 1}


def test_flatten_invariants_on_random_sentences():
    rng = random.Random(0)
    for _ in range(500):
        n = rng.randint(1, 8)
        entities = []
        for _ in range(rng.randint(0, 7)):
            first = rng.randrange(n)
            entities.append((first, rng.randrange(first, n), rng.choice(mulms.TYPE_ORDER[:4])))
        kept, dropped = mulms.flatten(entities)
        assert len(kept) + sum(dropped.values()) == len(entities)
        assert kept == sorted(kept) and set(kept) <= set(entities)
        assert not any(mulms.overlap(a, b) for a, b in itertools.combinations(kept, 2))
        assert not any(mulms.strictly_inside(k, e) for k in kept for e in entities)
        for k in kept:
            assert all(mulms.TYPE_RANK[k[2]] <= mulms.TYPE_RANK[e[2]] for e in entities
                       if e[:2] == k[:2])
        assert len(mulms.iob2(n, kept)) == n


def test_the_rule_constants():
    assert mulms.TYPE_ORDER == ('MEASUREMENT', 'VALUE', 'RANGE', 'SAMPLE', 'PROPERTY',
                                'TECHNIQUE', 'INSTRUMENT', 'DEV', 'CITE', 'MAT', 'FORM', 'NUM',
                                'UNIT')
    assert mulms.EXPECTED == {'documents': 50, 'sentences': 10184, 'entities': 46584,
                              'types': len(set(mulms.TYPE_ORDER))}


# ---- IOB2 and the CoNLL format ----

def test_iob2_tags():
    assert mulms.iob2(6, [(0, 0, 'MAT'), (1, 2, 'MAT'), (4, 5, 'VALUE')]) == [
        'B-MAT', 'B-MAT', 'I-MAT', 'O', 'B-VALUE', 'I-VALUE']
    assert mulms.iob2(2, []) == ['O', 'O']
    with pytest.raises(ValueError, match='overlap'):
        mulms.iob2(4, [(0, 2, 'MAT'), (2, 3, 'NUM')])
    with pytest.raises(ValueError, match='of a sentence of 2'):
        mulms.iob2(2, [(1, 2, 'MAT')])
    # one "token tag" line per token, a blank line after each sentence
    text = mulms.conll_text([(['a', 'b'], ['B-MAT', 'I-MAT']), (['c'], ['O'])])
    assert text == 'a B-MAT\nb I-MAT\n\nc O\n\n'


# ---- order and checks ----

def test_sentences_are_in_text_order_by_int_begin_offset():
    """beginOffset sorts as a number: as text, '10' < '100' < '9'."""
    rows = [_row('d1', '10', ['b']), _row('d1', '100', ['c']), _row('d1', '9', ['a'])]
    assert [row['tokens'] for row in mulms.documents(rows)['d1']] == [['a'], ['b'], ['c']]
    sentences, *_ = mulms.convert(rows, {'documents': 1, 'sentences': 3, 'entities': 0,
                                         'types': 0})
    assert sentences == {'d1': [(['a'], ['O']), (['b'], ['O']), (['c'], ['O'])]}


@pytest.mark.parametrize('key', ['documents', 'sentences', 'entities', 'types'])
def test_the_totals_are_checked_before_anything_is_written(tmp_path, key):
    expected = dict(TOY, **{key: TOY[key] + 1})
    with pytest.raises(mulms.ConversionError, match=f"counts .*'{key}': {TOY[key]}"):
        mulms.write_corpus(_corpus(), tmp_path / 'out', tmp_path / 'report.json', expected)
    assert not (tmp_path / 'out').exists() and not (tmp_path / 'report.json').exists()


@pytest.mark.parametrize('rows, error', [
    ([_row('d1', 0, ['a']), _row('d1', 5, ['b'], split='dev')], r"d1: data_split \['dev', "),
    ([_row('d1', 0, ['a']), _row('d1', 5, ['b'], category='steel')], 'd1: category'),
    ([_row('d1', 0, ['a']), _row('d1', '0', ['b'])], 'begin at the same offset'),
    ([_row('../d1', 0, ['a'])], 'document id'),
])
def test_document_errors(rows, error):
    with pytest.raises(mulms.ConversionError, match=error):
        mulms.documents(rows)


@pytest.mark.parametrize('entities, error', [
    ([(0, 0, 'METAL')], "'METAL' is not in TYPE_ORDER"),
    ([(1, 2, 'MAT')], r'tokens 1\.\.2 of a sentence of 2'),
    ([(1, 0, 'MAT')], r'tokens 1\.\.0 of a sentence of 2'),
])
def test_entity_errors(entities, error):
    with pytest.raises(mulms.ConversionError, match=error):
        mulms.sentence_entities('d1', _row('d1', 0, ['a', 'b'], entities))


def test_malformed_entity_lists():
    row = _row('d1', 0, ['a', 'b'], [(0, 1, 'MAT')])
    assert mulms.sentence_entities('d1', row) == [(0, 1, 'MAT')]
    row['NER_labels']['tokenIndices'] = [[0, 1, 1]]
    with pytest.raises(mulms.ConversionError, match='token indices'):
        mulms.sentence_entities('d1', row)
    row['NER_labels']['tokenIndices'] = []
    with pytest.raises(mulms.ConversionError, match='1 entity types for 0 spans'):
        mulms.sentence_entities('d1', row)


def test_tokens_become_their_visible_whitespace_separated_pieces():
    """Rule 5: the str.split pieces of a token, without those made only of C* and Mn characters;
    an entity moves to the pieces of its tokens, and one left without pieces is dropped."""
    assert mulms.pieces('Pt') == ['Pt'] and mulms.pieces('The cataly') == ['The', 'cataly']
    assert mulms.pieces(' 8 of 14') == ['8', 'of', '14']
    assert mulms.pieces('') == mulms.pieces(' ') == mulms.pieces('\xa0' * 5) == []
    assert mulms.pieces('\uf0b0') == mulms.pieces('\u0307') == mulms.pieces('\ufeff') == []
    assert mulms.pieces('x \uf0b0') == ['x']
    assert mulms.pieces('\ufeffThe') == ['\ufeffThe'] and mulms.pieces('a\u0307') == ['a\u0307']
    # m2 g ' ' 1 and 80 ' ' C: a lost minus and a lost degree sign
    tokens = ['m2', 'g', ' ', '1', 'at', '80', ' ', 'C', '\uf0b0', 'c d']
    entities = [(0, 3, 'UNIT'), (5, 7, 'VALUE'), (8, 8, 'UNIT'), (9, 9, 'MAT')]
    words, moved, changes = mulms.retokenize(tokens, entities)
    assert words == ['m2', 'g', '1', 'at', '80', 'C', 'c', 'd']
    assert moved == [(0, 2, 'UNIT'), (4, 5, 'VALUE'), (6, 7, 'MAT')]
    assert changes == {'dropped_blank': 2, 'dropped_invisible': 1, 'split': 1}
    # an entity that starts or ends on a dropped token keeps the pieces of its other tokens
    assert mulms.retokenize([' ', 'a', 'b', ''], [(0, 1, 'MAT'), (2, 3, 'NUM')])[1] == [
        (0, 0, 'MAT'), (1, 1, 'NUM')]
    assert mulms.retokenize([' 8', 'x \uf0b0'], [])[2] == {'trimmed': 2}
    assert mulms.retokenize(['a', 'b'], [(0, 1, 'MAT')]) == (['a', 'b'], [(0, 1, 'MAT')], {})


def test_invisible_is_what_the_bert_normalizer_erases():
    """invisible() agrees with MatSciBERT's normalizer (BERT's, uncased: control characters
    removed, accents stripped) on characters of each kind: MuLMS's private-use symbols and lone
    combining marks, format and control characters, and visible ones, an unassigned one (BERT's
    unknown token) among them."""
    erased = ['\uf02b', '\uf0b0', '\uf8eb', '\u0307', '\u030c', '\ufeff', '\u200b', '\u00ad',
              '\x07', '\ufffd']
    kept = ['a', '\u00e9', '\u00c5', '\u00b5', '\u00b0', '\u2212', '\u0378', '\u1e8b', '\u0130',
            '\u4e2d']
    assert all(mulms.invisible(c) for c in erased) and not any(mulms.invisible(c) for c in kept)
    assert not any(c.isspace() for c in erased + kept)
    normalizers = pytest.importorskip('tokenizers.normalizers')
    bert = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True,
                                      strip_accents=None, lowercase=True)
    assert [bert.normalize_str(c).split() == [] for c in erased + kept] == (
        [True] * len(erased) + [False] * len(kept))


def test_write_corpus_applies_the_token_rule(tmp_path):
    """The words written hold no whitespace and read back as written; the report counts the
    tokens changed and the entities left without pieces; a sentence left without pieces stops
    the conversion, and nothing is written."""
    rows = [_row('d1', 0, ['80', ' ', 'C', '\uf0b0'], [(0, 2, 'VALUE'), (3, 3, 'UNIT')]),
            _row('d1', 7, ['', 'The cataly', 'st'], [(1, 2, 'MAT')])]
    expected = {'documents': 1, 'sentences': 2, 'entities': 3, 'types': 3}
    out, path = tmp_path / 'out', tmp_path / 'report.json'
    report = mulms.write_corpus(rows, out, path, expected)
    assert conll.read_conll_sentences(out / 'd1.conll') == [
        (['80', 'C'], ['B-VALUE', 'I-VALUE']),
        (['The', 'cataly', 'st'], ['B-MAT', 'I-MAT', 'I-MAT'])]
    assert report['totals']['entities_kept'] == 2
    assert report['totals']['dropped'] == {**NONE_DROPPED, 'no_visible_token': 1}
    assert report['tokens'] == {'in': 7, 'out': 5, 'dropped_blank': 2, 'dropped_invisible': 1,
                                'split': 1, 'trimmed': 0}
    assert report['rule']['tokens'] == mulms.TOKEN_RULE
    with pytest.raises(mulms.ConversionError, match='d1: the sentence at 0 has no visible'):
        mulms.write_corpus([_row('d1', 0, [' ', '\u0307'])], tmp_path / 'bad',
                           tmp_path / 'bad.json',
                           {'documents': 1, 'sentences': 1, 'entities': 0, 'types': 0})
    assert not (tmp_path / 'bad').exists() and not (tmp_path / 'bad.json').exists()
    with pytest.raises(mulms.ConversionError, match='d1: the sentence at 0 has no tokens'):
        mulms.check_tokens(mulms.documents([_row('d1', 0, [])]))


# ---- outputs and report ----

def test_write_corpus_on_a_toy_corpus(tmp_path):
    out, path = tmp_path / 'mulms', tmp_path / 'mulms_corpus.json'
    report = mulms.write_corpus(_corpus(), out, path, TOY)
    assert (out / 'd1.conll').read_text(encoding='utf-8') == (
        'A O\nPEMFC B-DEV\nstack I-DEV\n\nPt B-MAT\nat O\n80 B-VALUE\n°C I-VALUE\n\n')
    assert conll.read_conll_sentences(out / 'd2.conll') == [
        (['Steel', 'X', 'was', 'tested', 'by', 'XRD'],
         ['B-SAMPLE', 'I-SAMPLE', 'O', 'O', 'O', 'B-TECHNIQUE']),
        (['No', 'entities', '.'], ['O', 'O', 'O'])]
    assert json.loads((out / 'split.json').read_text()) == {'d1': 'train1', 'd2': 'test'}
    assert json.loads((out / 'subfields.json').read_text()) == {'d1': 'pemfc', 'd2': 'steel'}
    assert report['totals'] == {'documents': 2, 'sentences': 4, 'entities_in': 12,
                                'entities_kept': 5,
                                'dropped': {'nested': 4, 'same_span': 2, 'crossing': 1,
                                            'no_visible_token': 0}}
    assert report['tokens'] == {'in': 16, 'out': 16, 'dropped_blank': 0, 'dropped_invisible': 0,
                                'split': 0, 'trimmed': 0}
    assert len(report['kept_per_type']) == 13
    assert {kind: n for kind, n in report['kept_per_type'].items() if n} == {
        'DEV': 1, 'MAT': 1, 'SAMPLE': 1, 'TECHNIQUE': 1, 'VALUE': 1}
    assert report['subfields'] == {
        'pemfc': {'documents': 1, 'sentences': 2, 'entities_kept': 3},
        'steel': {'documents': 1, 'sentences': 2, 'entities_kept': 2}}
    assert report['splits'] == {'test': 1, 'train1': 1}
    assert report['source']['revision'] == mulms.REVISION
    assert report['source']['files'] == mulms.FILES
    assert report['rule'] == {'type_order': list(mulms.TYPE_ORDER), 'description': mulms.RULE,
                              'tokens': mulms.TOKEN_RULE,
                              'unicode_version': unicodedata.unidata_version}
    # outputs: {name in the output folder: SHA-256} of the files written; outputs_sha256 the
    # SHA-256 of their sorted "<name> <sha256>" lines
    names = sorted(p.name for p in out.iterdir())
    assert sorted(report['outputs']) == names == ['d1.conll', 'd2.conll', 'split.json',
                                                  'subfields.json']
    for name, digest in report['outputs'].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    lines = ''.join(f"{name} {report['outputs'][name]}\n" for name in names)
    assert report['outputs_sha256'] == hashlib.sha256(lines.encode()).hexdigest()
    assert json.loads(path.read_text()) == report


def test_outputs_and_report_are_deterministic(tmp_path):
    """The same rows, in any order, give the same bytes: no timestamps, sorted keys."""
    out, path = tmp_path / 'mulms', tmp_path / 'report.json'
    mulms.write_corpus(_corpus(), out, path, TOY)
    files = {p.name: p.read_bytes() for p in out.iterdir()}
    report = path.read_bytes()
    mulms.write_corpus(_corpus()[::-1], out, path, TOY)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == files
    assert path.read_bytes() == report
    again = mulms.write_corpus(_corpus(), tmp_path / 'again', tmp_path / 'again.json', TOY)
    assert {p.name: p.read_bytes() for p in (tmp_path / 'again').iterdir()} == files
    assert again['outputs_sha256'] == json.loads(report)['outputs_sha256']


# ---- the source files ----

def test_the_cache_is_checked_and_offline_never_downloads(tmp_path):
    data = b'parquet bytes'
    files = {'MuLMS_Corpus/x.parquet': hashlib.sha256(data).hexdigest()}
    urls = []

    def get(url):
        urls.append(url)
        return data

    with pytest.raises(mulms.ConversionError, match='is missing'):
        mulms.fetch_sources(tmp_path, offline=True, files=files, get=get)
    local = mulms.fetch_sources(tmp_path, files=files, get=get)
    assert local == {'MuLMS_Corpus/x.parquet': tmp_path / 'MuLMS_Corpus' / 'x.parquet'}
    assert urls == ['https://huggingface.co/datasets/timo-pierre-schrader/MuLMS/resolve/'
                    f'{mulms.REVISION}/MuLMS_Corpus/x.parquet']
    assert mulms.fetch_sources(tmp_path, offline=True, files=files, get=get) == local
    assert mulms.fetch_sources(tmp_path, files=files, get=get) == local and len(urls) == 1
    local['MuLMS_Corpus/x.parquet'].write_bytes(b'changed')
    with pytest.raises(mulms.ConversionError, match='has another SHA-256'):
        mulms.fetch_sources(tmp_path, offline=True, files=files, get=get)
    # online, a cached file with another SHA-256 is downloaded again
    assert mulms.fetch_sources(tmp_path, files=files, get=get) == local and len(urls) == 2
    assert local['MuLMS_Corpus/x.parquet'].read_bytes() == data
    # a download with another SHA-256 is not stored
    with pytest.raises(mulms.ConversionError, match='downloaded SHA-256'):
        mulms.fetch_sources(tmp_path / 'new', files=files, get=lambda url: b'other')
    assert not list((tmp_path / 'new').rglob('*.parquet*'))


def test_main_offline_needs_the_cached_files(tmp_path):
    with pytest.raises(mulms.ConversionError, match='offline: .* is missing'):
        mulms.main(['--cache', str(tmp_path / 'cache'), '--offline',
                    '--out', str(tmp_path / 'out'), '--report', str(tmp_path / 'report.json')])
    assert not (tmp_path / 'out').exists() and not (tmp_path / 'report.json').exists()


def test_mulms_imports_without_pyarrow(monkeypatch):
    """CI has no pyarrow: mulms.py imports it in read_rows only, never on import."""
    monkeypatch.setitem(sys.modules, 'pyarrow', None)           # import pyarrow fails
    monkeypatch.delitem(sys.modules, 'pyarrow.parquet', raising=False)
    spec = importlib.util.spec_from_file_location('mulms_without_pyarrow', mulms.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.flatten([(0, 1, 'MAT')]) == ([(0, 1, 'MAT')], NONE_DROPPED)
    with pytest.raises(ImportError):
        module.read_rows([])
    # a fresh interpreter: importing mulms loads no pyarrow module
    code = 'import sys, mulms; print(sorted(m for m in sys.modules if m.startswith("pyarrow")))'
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                          check=True, cwd=Path(mulms.__file__).parent)
    assert done.stdout.strip() == '[]'
