"""Document context in ner_baseline.py (--crf --context N; P1 "Next, after week 4" item 4).

The context selection and the parser need no torch; the encoding needs transformers, and
the run with a tiny random model torch, transformers, seqeval and scipy (skipped without).
"""
import json

import pytest

import ner_baseline as nb

# three papers: a has 4 sentences, b 3, c 1 (in corpus order)
DATA = [([f'{doc}{k}w{j}' for j in range(3)], ['O'] * 3, f'{doc}.conll')
        for doc, n in (('a', 4), ('b', 3), ('c', 1)) for k in range(n)]
WORDS = ['steel', 'alloy', 'tensile', 'strength', 'yield', 'room', 'temperature', 'high']


def test_contexts_are_the_same_paper_and_part_nearest_first():
    # positions: a0..a3 = 0..3, b0..b2 = 4..6, c0 = 7; the part is not in corpus order
    part = [2, 0, 3, 5, 7, 4]
    out = nb.part_contexts(DATA, part)
    words = {i: DATA[i][0] for i in range(len(DATA))}
    # a2: left a0 (a1 is in another part), right a3; a0: nothing left, right a2 then a3
    assert out[0] == ([words[0]], [words[3]])
    assert out[1] == ([], [words[2], words[3]])
    assert out[2] == ([words[2], words[0]], [])          # nearest first
    assert out[3] == ([words[4]], [])                    # b1: b0 left, b2 is not in the part
    assert out[4] == ([], [])                            # c0: alone in its paper
    assert out[5] == ([], [words[5]])


def test_context_parser_checks(capsys):
    args = nb.parse_args(['--crf', '--context', '64', '--leave-one-paper-out'])
    assert args.context == 64 and nb.variant_name(args) == 'crf_ctx'
    assert nb.reference_variant('crf_ctx_m100') == 'crf'
    tagged = nb.parse_args(['--crf', '--context', '64', '--tag', 'm100', '--seed-offset', '100'])
    assert nb.variant_name(tagged) == 'crf_ctx_m100'
    assert 'context' not in vars(nb.parse_args(['--crf']))      # other configs unchanged
    for argv, error in ((['--context', '64'], '--context needs --crf'),
                        (['--crf', '--context', '0'], 'at least 1 subword'),
                        (['--crf', '--context', '8', '--multi-task'], 'not available with'),
                        (['--crf', '--context', '8', '--label-semantics'], 'not available with'),
                        (['--crf', '--tag', 'ctx'], 'would overwrite')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err


@pytest.fixture(scope='module')
def tokenizer(tmp_path_factory):
    transformers = pytest.importorskip('transformers')
    path = tmp_path_factory.mktemp('tiny_vocab')
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', *WORDS, '##s']
    (path / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    return transformers.BertTokenizerFast(str(path / 'vocab.txt'))


def test_context_words_take_whole_words_up_to_the_budget(tokenizer):
    left = [['steel', 'alloy'], ['room', 'temperature', 'high']]     # nearest first
    right = [['strengths', 'yield'], ['tensile']]                     # strengths: 2 subwords
    # left: alloy and steel (the nearest sentence, from its end), then high: 3 subwords,
    # returned in reading order; right: strengths (2 subwords) and yield
    assert nb.context_words(tokenizer, left, right, 3) == (
        ['high', 'steel', 'alloy'], ['strengths', 'yield'])
    assert nb.context_words(tokenizer, left, right, 2) == (['steel', 'alloy'], ['strengths'])
    assert nb.context_words(tokenizer, left, right, 0) == ([], [])
    assert nb.context_words(tokenizer, left, right, 99) == (
        ['room', 'temperature', 'high', 'steel', 'alloy'], ['strengths', 'yield', 'tensile'])


def test_only_the_sentence_is_labelled(tokenizer):
    label2id = {'O': 0, 'B-MAT': 1, 'I-MAT': 2}
    sentence = (['steel', 'alloy'], ['B-MAT', 'I-MAT'])
    context = ([['high', 'yield']], [['tensile', 'strengths']])
    plain = nb.encode(tokenizer, [sentence], label2id)[0]
    with_ctx = nb.encode(tokenizer, [sentence], label2id, 512, [context], budget=8)[0]
    # [CLS] high yield steel alloy tensile strength ##s [SEP]
    assert tokenizer.convert_ids_to_tokens(with_ctx['input_ids']) == [
        '[CLS]', 'high', 'yield', 'steel', 'alloy', 'tensile', 'strength', '##s', '[SEP]']
    assert with_ctx['labels'] == [-100, -100, -100, 1, 2, -100, -100, -100, -100]
    assert [x for x in with_ctx['labels'] if x != -100] == [x for x in plain['labels'] if x != -100]
    # without contexts nothing changes; a full window leaves the sentence room
    assert plain['input_ids'] == tokenizer(['steel', 'alloy'], is_split_into_words=True)['input_ids']
    short = nb.encode(tokenizer, [sentence], label2id, 6, [context], budget=8)[0]
    assert len(short['input_ids']) <= 6 and [x for x in short['labels'] if x != -100] == [1, 2]


def _toy_ner_corpus():
    data = []
    for p, doc in enumerate(('a.conll', 'b.conll', 'c.conll', 'd.conll')):
        for k in range(6):
            extra = 6 * p + k
            data.append((['steel', 'alloy', 'tensile', 'strengths', 'yield', 'room',
                          'temperature'] + ['high'] * extra,
                         ['B-MAT', 'B-MAT', 'B-PROP', 'I-PROP', 'B-PROP', 'B-COND',
                          'I-COND'] + ['O'] * extra, doc))
    return data


def test_context_run(tmp_path, monkeypatch):
    """A tiny --context run: its own results file, the context in the stored config, and in
    every training input the sentence's words labelled, the context around them not."""
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    path = tmp_path / 'tiny'
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', *WORDS, '##s']
    path.mkdir()
    (path / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    transformers.BertTokenizerFast(str(path / 'vocab.txt')).save_pretrained(path)
    torch.manual_seed(0)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64)).save_pretrained(path)
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: _toy_ner_corpus())
    encoded, encode = [], nb.encode

    def spy(tok, sentences, label2id, max_length=512, contexts=None, budget=0):
        out = encode(tok, sentences, label2id, max_length, contexts, budget)
        encoded.append((sentences, contexts, out))
        return out
    monkeypatch.setattr(nb, 'encode', spy)
    nb.run(nb.parse_args(['--granularity', 'fine', '--top-k', '2', '--model', str(path),
                          '--epochs', '2', '--threads', '1', '--device', 'cpu', '--crf',
                          '--context', '8', '--leave-one-paper-out', '--seeds', '0',
                          '--papers', 'b', '--save-predictions']))
    res = json.loads((tmp_path / 'ner' / 'ner_results_fine_leave_one_paper_out_crf_ctx.json')
                     .read_text())
    assert res['variant'] == 'crf_ctx' and res['config']['context'] == 8
    (run,) = res['runs']
    assert run['run_id'] == 'b.conll' and len(run['test_pred']) == 6
    assert all(len(p.split()) == len(g.split()) for p, g in zip(run['test_pred'], run['test_gold']))
    assert len(encoded) == 3                                  # train, val, test
    for sentences, contexts, features in encoded:
        assert contexts is not None and len(contexts) == len(sentences)
        for (words, _), feature in zip(sentences, features):
            labelled = [x for x in feature['labels'] if x != -100]
            assert len(labelled) == len(words)               # every word once, no context word
            assert len(feature['input_ids']) <= 64
    # the test sentences (paper b) see each other: some input is longer than its sentence
    _, _, test_features = encoded[2]
    assert any(len(f['input_ids']) > sum(x != -100 for x in f['labels']) + 3
               for f in test_features)
