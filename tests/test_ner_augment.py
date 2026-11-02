"""Mention replacement in ner_baseline.py (--crf --augment-mentions K; Dai and Adel, 2020).

The spans, the copies and the parser need no torch; the run with a tiny random model needs
torch, transformers, seqeval and scipy (skipped without them).
"""
import json

import pytest
from types import SimpleNamespace

import ner_baseline as nb

SENTENCES = [(['steel', 'alloy', 'was', 'tested'], ['B-MAT', 'I-MAT', 'O', 'O']),
             (['Ti64', 'fails', 'at', 'room', 'temperature'],
              ['B-MAT', 'O', 'O', 'B-COND', 'I-COND']),
             (['no', 'entity'], ['O', 'O']),
             (['steel', 'at', '20', 'K'], ['B-MAT', 'O', 'B-COND', 'I-COND'])]


def test_entity_spans_read_iob2_as_seqeval_does():
    assert nb.entity_spans(['B-A', 'I-A', 'O']) == [(0, 2, 'A')]
    assert nb.entity_spans(['B-A', 'B-A', 'I-A']) == [(0, 1, 'A'), (1, 3, 'A')]
    # an I- tag that continues nothing starts an entity, as does a change of type
    assert nb.entity_spans(['O', 'I-B', 'I-B', 'I-C']) == [(1, 3, 'B'), (3, 4, 'C')]
    assert nb.entity_spans(['O', 'O']) == []


def test_mention_replacement_keeps_types_and_contexts():
    copies = nb.mention_replacements(SENTENCES, 5, seed=0)
    assert copies == nb.mention_replacements(SENTENCES, 5, seed=0)         # seeded
    assert copies != nb.mention_replacements(SENTENCES, 5, seed=1)
    mentions = {('MAT', ('steel', 'alloy')), ('MAT', ('Ti64',)), ('MAT', ('steel',)),
                ('COND', ('room', 'temperature')), ('COND', ('20', 'K'))}
    originals = {tuple(w) for w, _ in SENTENCES}
    for words, tags in copies:
        assert len(words) == len(tags) and tuple(words) not in originals   # changed copies only
        spans = nb.entity_spans(tags)
        assert spans and all(t.startswith(('B-', 'I-')) or t == 'O' for t in tags)
        assert all((kind, tuple(words[a:b])) in mentions for a, b, kind in spans)
        # the words outside the mentions are those of a sentence of the corpus
        outside = [w for i, w in enumerate(words) if tags[i] == 'O']
        assert any(outside == [x for x, t in zip(w, t) if t == 'O'] for w, t in SENTENCES)
    assert nb.mention_replacements(SENTENCES, 3, seed=0, p=0.0) == []
    assert nb.mention_replacements([SENTENCES[2]], 3, seed=0, p=1.0) == []   # no entity


def test_augment_parser_and_names(capsys):
    args = nb.parse_args(['--crf', '--augment-mentions', '3', '--leave-one-paper-out',
                          '--tag', 'm100', '--seed-offset', '100'])
    assert nb.variant_name(args) == 'crf_mr_m100' and nb.reference_variant('crf_mr_m100') == 'crf'
    for argv, error in ((['--augment-mentions', '2'], 'needs --crf'),
                        (['--crf', '--augment-mentions', '0'], 'at least 1'),
                        (['--crf', '--augment-mentions', '2', '--label-semantics'],
                         'not available'),
                        (['--crf', '--augment-mentions', '2', '--context', '8'], 'not available'),
                        (['--crf', '--tag', 'mr'], 'would overwrite')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err
    # the existing options keep their stored configuration
    assert 'augment_mentions' not in vars(nb.parse_args(['--crf']))
    # with the multi-task model: its own variant, compared with the CRF head
    args = nb.parse_args(['--crf', '--multi-task', '--augment-mentions', '3', '--tag', 'm0'])
    assert nb.variant_name(args) == 'crf_mt_mr_m0'
    assert nb.reference_variant('crf_mt_mr_m0') == 'crf'


MT_EXAMPLES = [
    (['steel', 'alloy', 'was', 'tested'],
     {'fine': ['B-Alloy', 'I-Alloy', 'O', 'O'], 'coarse': ['B-MAT', 'I-MAT', 'O', 'O']}),
    (['Ti64', 'fails', 'at', 'room', 'temperature'],
     {'fine': ['B-Alloy', 'O', 'O', 'O', 'O'], 'coarse': ['B-MAT', 'O', 'O', 'B-COND', 'I-COND']}),
    (['copper', 'at', '20', 'K', 'and', 'room', 'temperature'],
     {'fine': ['O', 'O', 'B-Value', 'I-Value', 'O', 'O', 'O'],
      'coarse': ['B-MAT', 'O', 'B-COND', 'I-COND', 'O', 'B-COND', 'I-COND']}),
    (['steel', 'cracks'], {'fine': ['B-Alloy', 'O'], 'coarse': None}),       # one head only
    (['brass', 'bends'], {'fine': ['B-Alloy', 'I-Alloy'], 'coarse': ['B-MAT', 'O']})]  # no nest


def test_multitask_mention_replacement_keeps_both_levels():
    """A mention is replaced in both heads by one of the same coarse- and fine-grained type;
    examples labelled for one head only, or whose levels do not nest, are left alone."""
    copies = nb.mention_replacements_multitask(MT_EXAMPLES, 5, seed=0)
    assert copies == nb.mention_replacements_multitask(MT_EXAMPLES, 5, seed=0)    # seeded
    assert copies and copies != nb.mention_replacements_multitask(MT_EXAMPLES, 5, seed=1)
    mentions = {(('MAT', 'Alloy'), ('steel', 'alloy')), (('MAT', 'Alloy'), ('Ti64',)),
                (('MAT', None), ('copper',)), (('COND', None), ('room', 'temperature')),
                (('COND', 'Value'), ('20', 'K'))}
    sources = [[w for w, t in zip(words, tags['coarse']) if t == 'O']
               for words, tags in MT_EXAMPLES[:3]]
    for words, tags in copies:
        assert set(tags) == {'fine', 'coarse'}
        assert len(words) == len(tags['fine']) == len(tags['coarse'])
        fine = {(a, b): kind for a, b, kind in nb.entity_spans(tags['fine'])}
        coarse = nb.entity_spans(tags['coarse'])
        assert set(fine) <= {(a, b) for a, b, _ in coarse}                 # the levels nest
        assert all(((kind, fine.get((a, b))), tuple(words[a:b])) in mentions
                   for a, b, kind in coarse)
        # the words outside the mentions are those of one of the three usable examples
        assert [w for w, t in zip(words, tags['coarse']) if t == 'O'] in sources
    assert nb.mention_replacements_multitask(MT_EXAMPLES, 3, seed=0, p=0.0) == []
    assert nb.mention_replacements_multitask(MT_EXAMPLES[3:], 3, seed=0, p=1.0) == []
    # the single-task copies of the coarse level are not those of the multi-task model
    coarse_only = [(w, t['coarse']) for w, t in MT_EXAMPLES[:3]]
    assert [(w, t['coarse']) for w, t in copies] != nb.mention_replacements(coarse_only, 5, 0)


WORDS = ['steel', 'alloy', 'tensile', 'strength', 'yield', 'room', 'temperature', 'high']


def _toy_corpus(granularity):
    data = []
    for p, doc in enumerate(('a.conll', 'b.conll', 'c.conll', 'd.conll')):
        for k in range(6):
            extra = 6 * p + k
            data.append((['steel', 'alloy', 'tensile', 'strengths', 'yield', 'room',
                          'temperature'] + ['high'] * extra,
                         ['B-MAT', 'B-MAT', 'B-PROP', 'I-PROP', 'B-PROP', 'B-COND',
                          'I-COND'] + ['O'] * extra, doc))
    return data


def test_augmented_run(tmp_path, monkeypatch):
    """A tiny --augment-mentions run: its own results file, the copies in training only."""
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    path = tmp_path / 'tiny'
    path.mkdir()
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', *WORDS, '##s']
    (path / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    transformers.BertTokenizerFast(str(path / 'vocab.txt')).save_pretrained(path)
    torch.manual_seed(0)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64)).save_pretrained(path)
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', _toy_corpus)
    seen, train = [], nb.train_one

    def spy(seed, tr, va, te, *args, **kwargs):
        seen.append((seed, len(tr), kwargs.get('augmented')))
        return train(seed, tr, va, te, *args, **kwargs)
    monkeypatch.setattr(nb, 'train_one', spy)
    nb.run(nb.parse_args(['--granularity', 'fine', '--top-k', '3', '--model', str(path),
                          '--epochs', '1', '--threads', '1', '--device', 'cpu', '--crf',
                          '--augment-mentions', '2', '--leave-one-paper-out', '--seeds', '0',
                          '--papers', 'b', '--tag', 'm100', '--seed-offset', '100']))
    ((seed, n_train, augmented),) = seen
    assert seed == 100 and augmented and len(augmented) <= 2 * n_train
    res = json.loads((tmp_path / 'ner' / 'ner_results_fine_leave_one_paper_out_crf_mr2_m100.json')
                     .read_text())
    (run,) = res['runs']
    assert res['variant'] == 'crf_mr2_m100' and res['config']['augment_mentions'] == 2
    assert run['n_augmented'] == len(augmented) and run['n_train'] == n_train


def test_multitask_mention_replacement_takes_harvested_names():
    """Week 17: harvested names join the pool under their (coarse, fine) key; a name whose key
    no training mention has is never drawn; without names the copies are those of before."""
    names = [(('MAT', 'Alloy'), ('Inconel', '718')), (('MAT', None), ('nickel',)),
             (('DEV', None), ('cell',))]                        # DEV: no training mention
    with_names = nb.mention_replacements_multitask(MT_EXAMPLES, 20, seed=0, p=1.0, names=names)
    drawn = {tuple(words[a:b]) for words, tags in with_names
             for a, b, _ in nb.entity_spans(tags['coarse'])}
    assert ('Inconel', '718') in drawn and ('nickel',) in drawn and ('cell',) not in drawn
    for words, tags in with_names:
        fine = {(a, b): kind for a, b, kind in nb.entity_spans(tags['fine'])}
        for a, b, kind in nb.entity_spans(tags['coarse']):
            if tuple(words[a:b]) == ('Inconel', '718'):
                assert kind == 'MAT' and fine[(a, b)] == 'Alloy'
            if tuple(words[a:b]) == ('nickel',):
                assert kind == 'MAT' and (a, b) not in fine
    assert nb.mention_replacements_multitask(MT_EXAMPLES, 5, seed=0, names=()) == \
        nb.mention_replacements_multitask(MT_EXAMPLES, 5, seed=0)


def test_multitask_mention_filter_needs_both_heads_to_agree():
    """Week 17's filter: a coarse span every teacher tags alike, with the same fine type at
    exactly that span or none, every marginal of the span at the threshold on the coarse head
    and, with a fine type, on the fine head; a fine entity overlapping the span blocks it."""
    words = ['Inconel', '718', 'at', '600', 'C']
    def teacher(coarse, fine, c_marg, f_marg):
        return {'coarse': (coarse, c_marg), 'fine': (fine, f_marg)}
    high = [0.99] * 5
    t1 = teacher(['B-MAT', 'I-MAT', 'O', 'B-COND', 'I-COND'], ['B-Alloy', 'I-Alloy', 'O', 'O', 'O'],
                 high, high)
    t2 = teacher(['B-MAT', 'I-MAT', 'O', 'B-COND', 'I-COND'], ['B-Alloy', 'I-Alloy', 'O', 'O', 'O'],
                 high, high)
    out = nb.silver_mention_names_multitask([words], [[t1, t2]], 0.9)
    assert out == [(('MAT', 'Alloy'), ('Inconel', '718')), (('COND', None), ('600', 'C'))]
    # a low coarse marginal on one word of a span drops that span only
    low = [0.99, 0.99, 0.99, 0.5, 0.99]
    t3 = teacher(t1['coarse'][0], t1['fine'][0], low, high)
    assert nb.silver_mention_names_multitask([words], [[t1, t3]], 0.9) == \
        [(('MAT', 'Alloy'), ('Inconel', '718'))]
    # the teachers disagree on the fine type at the span: dropped; a low fine marginal too
    t4 = teacher(t1['coarse'][0], ['B-Value', 'I-Value', 'O', 'O', 'O'], high, high)
    assert nb.silver_mention_names_multitask([words], [[t1, t4]], 0.9) == \
        [(('COND', None), ('600', 'C'))]
    t5 = teacher(t1['coarse'][0], t1['fine'][0], high, [0.8, 0.99, 0.99, 0.99, 0.99])
    assert nb.silver_mention_names_multitask([words], [[t1, t5]], 0.9) == \
        [(('COND', None), ('600', 'C'))]
    # a fine entity overlapping but not equal to the coarse span blocks the key
    t6 = teacher(t1['coarse'][0], ['O', 'B-Value', 'O', 'O', 'O'], high, high)
    assert nb.silver_mention_names_multitask([words], [[t6, t6]], 0.9) == \
        [(('COND', None), ('600', 'C'))]
    assert nb.silver_mention_names_multitask([], [], 0.9) == []


def test_multitask_names_parser_and_variant(capsys, monkeypatch):
    """--augment-names with --multi-task needs --silver-mentions; the week-17 variants are
    crf_mt_mr6 (the control) and crf_mt_mrm6 (the names); --save-names and the feature,
    agreement and share options are single-task only."""
    monkeypatch.setattr(nb.os.path, 'isfile', lambda path: True)    # the corpus is not here
    control = nb.parse_args(['--leave-one-paper-out', '--crf', '--multi-task',
                             '--augment-mentions', '6', '--tag', 'm100', '--seed-offset', '100'])
    assert nb.variant_name(control) == 'crf_mt_mr6_m100'
    args = nb.parse_args(['--leave-one-paper-out', '--papers', '10.1038_ncomms7164', '--crf',
                          '--multi-task', '--self-train', 'data/processed/corpus/fatigue_sentences.txt',
                          '--self-train-sample', '0', '--silver-mentions', '--teachers', '3',
                          '--students', '10', '--augment-mentions', '6', '--augment-names'])
    assert nb.variant_name(SimpleNamespace(**{**vars(args), 'tag': 'm300'})) == 'crf_mt_mrm6_m300'
    base = ['--leave-one-paper-out', '--crf', '--multi-task', '--self-train',
            'data/processed/corpus/fatigue_sentences.txt', '--teachers', '3',
            '--augment-mentions', '6', '--augment-names']
    for extra, error in ((['--silver-mentions', '--save-names', 'x.json'], '--save-names'),
                         (['--silver-mentions', '--names-share', '0.5'], '--names-share'),
                         (['--silver-mentions', '--teachers-agree', '2'], '--teachers-agree'),
                         ([], 'needs --silver-mentions')):
        with pytest.raises(SystemExit):
            nb.parse_args(base + extra)
        assert error in capsys.readouterr().err
