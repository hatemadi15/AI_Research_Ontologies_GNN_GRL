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
SLOTS = REPO / 'data' / 'raw' / 'dataset' / 'sofc_exp_slot'


def _corpus(root, entity_rows=None):
    """Two documents. d1: 'Title text.' (not annotated), then 'Ni anode at 800 C.' (annotated,
    tokens at offsets relative to the sentence); d2: one annotated sentence."""
    c = root / 'sofc-exp-corpus'
    for sub in ('texts', 'annotations/sentences', 'annotations/entity_types_and_slots',
                'annotations/tokens'):
        (c / sub).mkdir(parents=True)
    (c / 'SOFC-Exp-Metadata.csv').write_text('name\tset\nd1\ttrain\nd2\ttest\n')
    (c / 'texts' / 'd1.txt').write_text('Title text. Ni anode at 800 C.')
    (c / 'annotations' / 'sentences' / 'd1.csv').write_text('1\t0\t0\t11\n2\t1\t12\t30\n')
    rows = entity_rows or ['2\t1\t0\t2\tB-MATERIAL\tB-anode_material',
                           '2\t2\t3\t8\tB-DEVICE\tO', '2\t3\t9\t11\tO\tO',
                           '2\t4\t12\t15\tB-VALUE\tB-working_temperature',
                           '2\t5\t16\t17\tI-VALUE\tI-working_temperature',
                           '2\t6\t17\t18\tO\tO']
    (c / 'annotations' / 'entity_types_and_slots' / 'd1.csv').write_text('\n'.join(rows) + '\n')
    (c / 'texts' / 'd2.txt').write_text('YSZ cells')
    (c / 'annotations' / 'sentences' / 'd2.csv').write_text('1\t1\t0\t9\n')
    (c / 'annotations' / 'entity_types_and_slots' / 'd2.csv').write_text(
        '1\t1\t0\t3\tB-MATERIAL\tB-electrolyte_material\n1\t2\t4\t9\tB-DEVICE\tB-device\n')
    # every sentence's tokens, offsets relative to the sentence
    (c / 'annotations' / 'tokens' / 'd1.csv').write_text(
        '1\t1\t0\t5\n1\t2\t6\t10\n1\t3\t10\t11\n'
        '2\t1\t0\t2\n2\t2\t3\t8\n2\t3\t9\t11\n2\t4\t12\t15\n2\t5\t16\t17\n2\t6\t17\t18\n')
    (c / 'annotations' / 'tokens' / 'd2.csv').write_text('1\t1\t0\t3\n1\t2\t4\t9\n')
    return c


def test_conversion_of_a_toy_corpus(tmp_path):
    corpus = _corpus(tmp_path)
    assert se.convert_document(corpus, 'd1') == [
        (['Ni', 'anode', 'at', '800', 'C', '.'],
         ['B-MATERIAL', 'B-DEVICE', 'O', 'B-VALUE', 'I-VALUE', 'O'])]
    assert se.convert_document(corpus, 'd1', 'slot') == [
        (['Ni', 'anode', 'at', '800', 'C', '.'],
         ['B-anode_material', 'O', 'O', 'B-working_temperature', 'I-working_temperature', 'O'])]
    out = tmp_path / 'out'
    stats = se.convert(corpus, out, expected={'documents': 2, 'sentences': 2, 'entities': 5},
                       expected_slots=4)
    assert stats == {'documents': 2, 'sentences': 2, 'entities': 5,
                     'types': {'DEVICE': 2, 'MATERIAL': 2, 'VALUE': 1},
                     'slots': 4, 'slot_types': {'anode_material': 1, 'device': 1,
                                                'electrolyte_material': 1,
                                                'working_temperature': 1},
                     'split': {'test': 1, 'train': 1}}
    assert conll.read_conll_sentences(out / 'd2.conll') == [
        (['YSZ', 'cells'], ['B-MATERIAL', 'B-DEVICE'])]
    # the slot tags of the same sentences, in a directory of their own
    assert conll.read_conll_sentences(tmp_path / 'out_slot' / 'd2.conll') == [
        (['YSZ', 'cells'], ['B-electrolyte_material', 'B-device'])]
    with pytest.raises(se.ConversionError, match='4 slot mentions, not 3'):
        se.convert(corpus, out, expected={'documents': 2, 'sentences': 2, 'entities': 5},
                   expected_slots=3)
    assert json.loads((out / 'split.json').read_text()) == {'d1': 'train', 'd2': 'test'}
    # the whole texts, for document context: the title is context, not annotated
    assert json.loads((out / 'context.json').read_text()) == {
        'd1': {'sentences': [['Title', 'text', '.'], ['Ni', 'anode', 'at', '800', 'C', '.']],
               'annotated': [1]},
        'd2': {'sentences': [['YSZ', 'cells']], 'annotated': [0]}}
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


@pytest.mark.parametrize('rows, error', [
    (['2\t1\t0\t2\tB-MATERIAL'], 'tag None'),                         # no slot column
    (['2\t1\t0\t2\tB-MATERIAL\tB-Anode'], 'tag'),
    (['2\t1\t0\t2\tB-MATERIAL\tI-anode_material'], 'does not continue'),
])
def test_slot_conversion_errors(tmp_path, rows, error):
    corpus = _corpus(tmp_path, rows)
    with pytest.raises(se.ConversionError, match=error):
        se.convert_document(corpus, 'd1', 'slot')


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


def test_the_committed_slot_files_tag_the_same_sentences():
    """The slot level: the sentences of the entity files, 4,263 slot mentions of 19 types,
    every slot token inside an entity."""
    files = sorted(SLOTS.glob('*.conll'))
    assert [f.name for f in files] == [f.name for f in sorted(DATA.glob('*.conll'))]
    slots = Counter()
    for f in files:
        entity = conll.read_conll_sentences(DATA / f.name)
        slot = conll.read_conll_sentences(f)
        assert [w for w, _ in slot] == [w for w, _ in entity]
        for (_, s_tags), (_, e_tags) in zip(slot, entity):
            assert all(e != 'O' for s, e in zip(s_tags, e_tags) if s != 'O')
            slots.update(t[2:] for t in s_tags if t.startswith('B-'))
    assert sum(slots.values()) == 4263 and len(slots) == 19
    assert slots['experiment_evoking_word'] == 1090 and slots['interconnect_material'] == 1


def test_the_token_files_must_cut_the_annotated_words(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / 'annotations' / 'tokens' / 'd2.csv').write_text('1\t1\t0\t9\n')   # one token
    with pytest.raises(se.ConversionError, match='whitespace'):
        se.convert(corpus, tmp_path / 'out', expected={'documents': 2, 'sentences': 2,
                                                         'entities': 5})
    (corpus / 'annotations' / 'tokens' / 'd2.csv').write_text('1\t1\t0\t3\n')    # one word short
    with pytest.raises(se.ConversionError, match='cut different words'):
        se.convert(corpus, tmp_path / 'out', expected={'documents': 2, 'sentences': 2,
                                                         'entities': 5})


def test_the_committed_context_holds_every_article_sentence():
    files = sorted(DATA.glob('*.conll'))
    context = json.loads((DATA / 'context.json').read_text(encoding='utf-8'))
    assert sorted(context) == [f.stem for f in files]
    for f in files:
        entry = context[f.stem]
        annotated = [entry['sentences'][k] for k in entry['annotated']]
        assert annotated == [words for words, _ in conll.read_conll_sentences(f)]
        assert entry['annotated'] == sorted(entry['annotated'])
    assert sum(len(e['annotated']) for e in context.values()) == 876
    assert sum(len(e['sentences']) for e in context.values()) > 5000


def test_ner_baseline_reads_sofc_exp_as_granularity_sofc(capsys):
    """--granularity sofc: the converted files, their four types, the official split as the
    published split, and one held-out-paper fold per article."""
    from types import SimpleNamespace

    import ner_baseline as nb
    data, labels, tie = nb.granularity_data('sofc', SimpleNamespace(top_k=27))
    assert len(data) == 876 and len({d for _, _, d in data}) == 45
    assert labels == ['DEVICE', 'EXPERIMENT', 'MATERIAL', 'VALUE'] and tie == {}
    split = json.loads((DATA / 'split.json').read_text())
    parts, same = nb.published_split(data, 'sofc')
    assert same == data
    assert sorted(i for p in parts for i in p) == list(range(876))
    for part, name in zip(parts, ('train', 'dev', 'test')):
        assert {split[data[i][2].removesuffix('.conll')] for i in part} == {name}
    args = nb.parse_args(['--granularity', 'sofc', '--crf', '--leave-one-paper-out'])
    path = nb.results_path('out', 'sofc', args)
    assert path.endswith('ner_results_sofc_leave_one_paper_out_crf.json')
    assert nb.parse_result_name(path.rsplit('/', 1)[1]) == ('sofc', 'leave_one_paper_out', 'crf')
    assert len(nb.make_folds(data, args, indices=True)) == 45
    assert nb.PAPER_F1['sofc'] is None
    with pytest.raises(SystemExit):
        nb.parse_args(['--granularity', 'sofc', '--crf', '--multi-task'])
    assert '--granularity must be both' in capsys.readouterr().err


def test_ner_baseline_reads_the_slot_level_as_granularity_sofc_slot(capsys):
    """--granularity sofc_slot: the 17 slot types that SOFC-Exp's evaluation scores (the
    experiment-evoking word and interconnect_material are O), the same sentences and split
    as sofc; sofc_both names the two levels, and multi-task runs on the official split."""
    from types import SimpleNamespace

    import ner_baseline as nb
    args = SimpleNamespace(top_k=27)
    data, labels, tie = nb.granularity_data('sofc_slot', args)
    entity, _, _ = nb.granularity_data('sofc', args)
    assert labels == sorted(nb.SOFC_SLOT_TYPES) and len(labels) == 17 and tie == {}
    assert [(w, d) for w, _, d in data] == [(w, d) for w, _, d in entity]
    tags = Counter(t[2:] for _, g, _ in data for t in g if t.startswith('B-'))
    assert sum(tags.values()) == 4263 - 1090 - 1 and set(tags) == set(labels)
    assert nb.published_split(data, 'sofc_slot')[0] == nb.published_split(entity, 'sofc')[0]
    assert nb.PAPER_F1['sofc_slot'] is None
    for g in ('sofc', 'sofc_slot'):
        name = f'ner_results_{g}_published_split_crf_mt.json'
        assert nb.parse_result_name(name) == (g, 'published_split', 'crf_mt')
    args = nb.parse_args(['--granularity', 'sofc_both', '--published-split', '--crf',
                          '--multi-task'])
    assert nb.head_granularities(args) == {'fine': 'sofc_slot', 'coarse': 'sofc'}
    assert nb.granularities_of(args) == ['sofc_slot', 'sofc']
    assert nb.granularities_of(nb.parse_args(['--granularity', 'sofc'])) == ['sofc']
    assert nb.granularities_of(nb.parse_args([])) == ['fine', 'coarse']
    for argv, message in ((['--published-split', '--crf', '--multi-task'], 'not available'),
                          (['--granularity', 'sofc_slot', '--crf', '--label-semantics'],
                           'MaterioMiner types only'),
                          (['--granularity', 'sofc_both', '--crf', '--multi-task',
                            '--label-semantics'], 'MaterioMiner types only')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert message in capsys.readouterr().err


# ---- the test articles cut as the training articles are (week 7, part 2e) ----

def test_retokenize_sentence_carries_the_entities_over_by_offsets():
    from chem_tokenize import ChemWordTokenizer
    text = 'A (NiO)-yttria cell gave 0.5 W cm−2 at 800 °C with Pr0.5Ba0.5MnO3-δ.'
    entities = [(3, 6, 'MATERIAL'), (8, 14, 'MATERIAL'), (25, 35, 'VALUE'), (39, 45, 'VALUE'),
                (51, 67, 'MATERIAL')]
    words, tags, counts = se.retokenize_sentence(text, entities, ChemWordTokenizer())
    assert list(zip(words, tags)) == [
        ('A', 'O'), ('(NiO)-yttria', 'B-MATERIAL'), ('cell', 'O'), ('gave', 'O'),
        ('0.5', 'B-VALUE'), ('W', 'I-VALUE'), ('cm−2', 'I-VALUE'), ('at', 'O'),
        ('800', 'B-VALUE'), ('°', 'I-VALUE'), ('C', 'I-VALUE'), ('with', 'O'),
        ('Pr0.5Ba0.5MnO3-δ', 'B-MATERIAL'), ('.', 'O')]
    # NiO gets the word it shares with yttria (inexact), and yttria is left without one
    assert (counts['inexact'], counts['lost']) == (1, 1)
    assert se.entity_spans([(0, 3, 'B-X'), (4, 6, 'I-X'), (7, 9, 'O'), (10, 12, 'B-Y')]) == [
        (0, 6, 'X'), (10, 12, 'Y')]


def test_the_committed_retokenized_test_articles():
    folder = se.OUT_DIR.with_name('sofc_exp_test_cde')
    stats = json.loads((folder / 'stats.json').read_text())
    assert (stats['sentences'], stats['entities'], stats['inexact'], stats['lost']) == (
        173, 978, 14, 2)
    split = json.loads((se.OUT_DIR / 'split.json').read_text())
    test = sorted(d for d, part in split.items() if part == 'test')
    assert sorted(p.stem for p in folder.glob('*.conll')) == test == sorted(stats['documents'])
    entities = changed = 0
    for doc in test:
        new = conll.read_conll_sentences(folder / f'{doc}.conll')
        old = conll.read_conll_sentences(se.OUT_DIR / f'{doc}.conll')
        assert len(new) == len(old)
        for (words, tags), (old_words, _) in zip(new, old):
            assert ''.join(words) == ''.join(old_words)          # the same text, cut otherwise
            changed += words != old_words
            entities += sum(t.startswith('B-') for t in tags)
    assert entities == 978 - 2 and changed == 140


def test_ner_baseline_retokenized_test(capsys):
    """--retokenized-test: the test articles' sentences from sofc_exp_test_cde, the training
    and validation sentences and the split unchanged."""
    from types import SimpleNamespace

    import ner_baseline as nb
    plain, labels, _ = nb.granularity_data('sofc', SimpleNamespace(top_k=27))
    data, labels2, _ = nb.granularity_data('sofc', SimpleNamespace(top_k=27,
                                                                    retokenized_test=True))
    assert labels2 == labels
    parts, _ = nb.published_split(plain, 'sofc')
    parts2, _ = nb.published_split(data, 'sofc')
    assert [len(p) for p in parts2] == [len(p) for p in parts] == [568, 135, 173]
    for k in (0, 1):
        assert [data[i] for i in parts2[k]] == [plain[i] for i in parts[k]]
    test = [data[i] for i in parts2[2]]
    assert sum(w != plain[i][0] for (w, _, _), i in zip(test, parts[2])) == 140
    args = nb.parse_args(['--granularity', 'sofc', '--published-split', '--crf',
                          '--retokenized-test', '--tag', 'cdetest'])
    assert nb.variant_name(args) == 'crf_cdetest' and nb.stored_config(args)['retokenized_test']
    for argv, message in ((['--granularity', 'sofc', '--published-split', '--crf',
                            '--retokenized-test'], 'needs a --tag'),
                          (['--granularity', 'sofc_slot', '--published-split', '--crf',
                            '--retokenized-test', '--tag', 'cdetest'], 'is for --granularity'),
                          (['--granularity', 'sofc', '--crf', '--retokenized-test', '--tag',
                            'cdetest'], 'is for --granularity')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert message in capsys.readouterr().err
