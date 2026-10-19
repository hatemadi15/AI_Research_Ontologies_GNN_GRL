"""Mention replacement in ner_baseline.py (--crf --augment-mentions K; Dai and Adel, 2020).

The spans, the copies and the parser need no torch; the run with a tiny random model needs
torch, transformers, seqeval and scipy (skipped without them).
"""
import json

import pytest

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
