"""SOFC-Exp as CoNLL files (sofc_exp.py): the conversion on a toy corpus, and the committed
files. Pure Python."""
import json
from collections import Counter
from pathlib import Path

import pytest

import conll
import sofc_exp as se

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data' / 'raw' / 'dataset' / 'sofc_exp'


def _corpus(root, entity_rows=None):
    """Two documents. d1: 'Title text.' (not annotated), then 'Ni anode at 800 C.' (annotated,
    tokens at offsets relative to the sentence); d2: one annotated sentence."""
    c = root / 'sofc-exp-corpus'
    for sub in ('texts', 'annotations/sentences', 'annotations/entity_types_and_slots'):
        (c / sub).mkdir(parents=True)
    (c / 'SOFC-Exp-Metadata.csv').write_text('name\tset\nd1\ttrain\nd2\ttest\n')
    (c / 'texts' / 'd1.txt').write_text('Title text. Ni anode at 800 C.')
    (c / 'annotations' / 'sentences' / 'd1.csv').write_text('1\t0\t0\t11\n2\t1\t12\t30\n')
    rows = entity_rows or ['2\t1\t0\t2\tB-MATERIAL\tO', '2\t2\t3\t8\tB-DEVICE\tO',
                           '2\t3\t9\t11\tO\tO', '2\t4\t12\t15\tB-VALUE\tO',
                           '2\t5\t16\t17\tI-VALUE\tO', '2\t6\t17\t18\tO\tO']
    (c / 'annotations' / 'entity_types_and_slots' / 'd1.csv').write_text('\n'.join(rows) + '\n')
    (c / 'texts' / 'd2.txt').write_text('YSZ cells')
    (c / 'annotations' / 'sentences' / 'd2.csv').write_text('1\t1\t0\t9\n')
    (c / 'annotations' / 'entity_types_and_slots' / 'd2.csv').write_text(
        '1\t1\t0\t3\tB-MATERIAL\tO\n1\t2\t4\t9\tB-DEVICE\tO\n')
    return c


def test_conversion_of_a_toy_corpus(tmp_path):
    corpus = _corpus(tmp_path)
    assert se.convert_document(corpus, 'd1') == [
        (['Ni', 'anode', 'at', '800', 'C', '.'],
         ['B-MATERIAL', 'B-DEVICE', 'O', 'B-VALUE', 'I-VALUE', 'O'])]
    out = tmp_path / 'out'
    stats = se.convert(corpus, out, expected={'documents': 2, 'sentences': 2, 'entities': 5})
    assert stats == {'documents': 2, 'sentences': 2, 'entities': 5,
                     'types': {'DEVICE': 2, 'MATERIAL': 2, 'VALUE': 1},
                     'split': {'test': 1, 'train': 1}}
    assert conll.read_conll_sentences(out / 'd2.conll') == [
        (['YSZ', 'cells'], ['B-MATERIAL', 'B-DEVICE'])]
    assert json.loads((out / 'split.json').read_text()) == {'d1': 'train', 'd2': 'test'}
    with pytest.raises(se.ConversionError, match='not those of the paper'):
        se.convert(corpus, out)


@pytest.mark.parametrize('rows, error', [
    (['1\t1\t0\t5\tB-MATERIAL\tO'], 'not annotated'),                 # sentence 1 is not
    (['2\t1\t0\t5\tB-MATERIAL\tO'], 'whitespace'),                    # 'Ni an'
    (['2\t1\t0\t2\tI-MATERIAL\tO'], 'does not continue'),
    (['2\t1\t0\t2\tB-material\tO'], 'tag'),
])
def test_conversion_errors(tmp_path, rows, error):
    corpus = _corpus(tmp_path, rows)
    with pytest.raises(se.ConversionError, match=error):
        se.convert_document(corpus, 'd1')


def test_the_committed_files_have_the_counts_of_the_paper():
    files = sorted(DATA.glob('*.conll'))
    assert len(files) == 45
    sentences = [s for f in files for s in conll.read_conll_sentences(f)]
    assert len(sentences) == 876
    types = Counter(t[2:] for _, tags in sentences for t in tags if t.startswith('B-'))
    assert types == {'MATERIAL': 1492, 'VALUE': 1475, 'EXPERIMENT': 1090, 'DEVICE': 533}
    split = json.loads((DATA / 'split.json').read_text())
    assert sorted(split) == [f.stem for f in files]
    assert Counter(split.values()) == {'train': 26, 'dev': 8, 'test': 11}
