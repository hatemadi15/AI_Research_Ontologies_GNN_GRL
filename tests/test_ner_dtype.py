"""ner_baseline.py trains in float32 whatever dtype the checkpoint's weights have.

transformers 5 loads a model in the dtype of its weights; DeBERTa-v3-large's are float16, and
its week-6 smoke run trained with a NaN loss. Needs torch, transformers, seqeval and scipy
(skipped without them).
"""
import json
import math

import pytest

import ner_baseline as nb

WORDS = ['steel', 'alloy', 'tensile', 'strength', 'yield', 'room', 'temperature', 'high']


def _toy_corpus(granularity):
    data = []
    for p, doc in enumerate(('a.conll', 'b.conll', 'c.conll', 'd.conll')):
        for k in range(6):
            extra = 6 * p + k
            tags = (['B-MAT', 'B-MAT', 'B-PROP', 'I-PROP', 'B-PROP', 'B-COND', 'I-COND']
                    if granularity == 'fine' else ['B-M', 'B-M', 'B-P', 'I-P', 'B-P', 'B-C', 'I-C'])
            data.append((['steel', 'alloy', 'tensile', 'strengths', 'yield', 'room',
                          'temperature'] + ['high'] * extra, tags + ['O'] * extra, doc))
    return data


@pytest.fixture
def half_model(tmp_path):
    """A tiny BERT whose saved weights are float16."""
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    path = tmp_path / 'tiny16'
    path.mkdir()
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', *WORDS, '##s']
    (path / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    transformers.BertTokenizerFast(str(path / 'vocab.txt')).save_pretrained(path)
    torch.manual_seed(0)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64)).half().save_pretrained(path)
    return path


def test_a_float16_checkpoint_trains_in_float32(tmp_path, monkeypatch, half_model):
    torch = pytest.importorskip('torch')
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', _toy_corpus)
    dtypes, train = [], nb.train_one

    def spy(*args, **kwargs):
        result, tagger = train(*args, **{**kwargs, 'keep_tagger': True})
        dtypes.append({p.dtype for p in tagger.model.parameters()})
        return (result, tagger) if kwargs.get('keep_tagger') else result
    monkeypatch.setattr(nb, 'train_one', spy)
    nb.run(nb.parse_args(['--granularity', 'fine', '--top-k', '3', '--model', str(half_model),
                          '--epochs', '2', '--threads', '1', '--device', 'cpu', '--crf',
                          '--leave-one-paper-out', '--seeds', '0', '--papers', 'b']))
    assert dtypes == [{torch.float32}]
    res = json.loads((tmp_path / 'ner' / 'ner_results_fine_leave_one_paper_out_crf.json')
                     .read_text())
    (run,) = res['runs']
    assert all(math.isfinite(h['train_loss']) for h in run['history'])


def test_the_multitask_encoder_is_float32(half_model):
    torch = pytest.importorskip('torch')
    tagger = nb.make_multitask_tagger(str(half_model), {'fine': ['O', 'B-X', 'I-X'],
                                                        'coarse': ['O', 'B-Y', 'I-Y']})
    assert {p.dtype for p in tagger.parameters()} == {torch.float32}
