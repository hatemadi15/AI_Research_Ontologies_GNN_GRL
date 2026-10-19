"""Article draws in ner_baseline.py (--published-split --train-articles K; ROADMAP §4, week 9,
part 2): SOFC-Exp's official split with its training sentences cut to K of its training
articles, drawn several times. The folds and the parser need no torch; the run with a tiny
random model needs torch, transformers, seqeval and scipy (skipped without them)."""
import json
import random
from types import SimpleNamespace

import pytest

import ner_baseline as nb


def test_draws_cut_the_training_articles_only():
    """5 draws of 10 of SOFC-Exp's 26 training articles, shuffled with split seed + i; the
    official validation and test articles are kept."""
    data, _, _ = nb.granularity_data('sofc', SimpleNamespace(top_k=27, published_split=False,
                                                             retokenized_test=False))
    (train, val, test), _ = nb.published_split(data, 'sofc')
    args = nb.parse_args(['--granularity', 'sofc', '--published-split', '--train-articles',
                          '10', '--crf', '--tag', 'm0'])
    folds = nb.make_folds(data, args, indices=True, granularity='sofc')
    docs = sorted({data[i][2] for i in train})
    assert len(docs) == 26 and [name for name, *_ in folds] == [f'draw{i}' for i in range(5)]
    for i, (name, tr, va, te) in enumerate(folds):
        order = list(docs)
        random.Random(i).shuffle(order)
        assert {data[j][2] for j in tr} == set(order[:10])
        assert tr == [j for j in train if data[j][2] in set(order[:10])]
        assert (va, te) == (val, test)
    assert [len(tr) for _, tr, *_ in folds] == [182, 205, 248, 173, 156]
    seeded = nb.parse_args(['--granularity', 'sofc', '--published-split', '--train-articles',
                            '10', '--draws', '2', '--split-seed', '3', '--crf'])
    again = nb.make_folds(data, seeded, indices=True, granularity='sofc')
    assert [fold[1] for fold in again] == [fold[1] for fold in folds[3:5]]
    with pytest.raises(SystemExit, match='between 1 and the 26 training articles'):
        nb.article_draws(data, (train, val, test), 27, 1, 0)


def test_parser_and_names(capsys):
    args = nb.parse_args(['--granularity', 'sofc', '--published-split', '--train-articles', '10',
                          '--crf', '--augment-mentions', '3', '--tag', 'm100',
                          '--seed-offset', '100'])
    assert nb.protocol_name(args) == 'article_draws' and nb.held_out_protocol(args)
    assert nb.variant_name(args) == 'crf_mr_m100'
    config = nb.stored_config(args)
    assert (config['train_articles'], config['draws']) == (10, 5)
    for unchanged in (['--crf'], ['--granularity', 'sofc', '--published-split', '--crf']):
        config = nb.stored_config(nb.parse_args(unchanged))
        assert 'train_articles' not in config and 'draws' not in config
    assert nb.parse_result_name('ner_results_sofc_article_draws_crf_m0.json') == (
        'sofc', 'article_draws', 'crf_m0')
    for argv, error in ((['--granularity', 'sofc', '--train-articles', '10'], 'needs --published'),
                        (['--granularity', 'fine', '--published-split', '--train-articles', '10'],
                         'needs --published'),
                        (['--granularity', 'sofc', '--published-split', '--train-articles', '0'],
                         'at least 1 article'),
                        (['--granularity', 'sofc', '--published-split', '--draws', '3'],
                         '--draws needs --train-articles'),
                        (['--granularity', 'sofc_both', '--published-split', '--crf',
                          '--multi-task', '--train-articles', '10'], 'needs --published'),
                        (['--granularity', 'sofc', '--published-split', '--crf',
                          '--train-articles', '10', '--context', '8'], 'not available')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err


def test_a_run_on_article_draws(tmp_path, monkeypatch):
    """One file per variant, a run per draw (draw<i>), with the model seed of --seed-offset."""
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    words = ['steel', 'alloy', 'tensile', 'strength', 'yield', 'room', 'temperature', 'high']
    path = tmp_path / 'tiny'
    path.mkdir()
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', *words, '##s']
    (path / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    transformers.BertTokenizerFast(str(path / 'vocab.txt')).save_pretrained(path)
    torch.manual_seed(0)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64)).save_pretrained(path)
    # 4 training articles of 3 sentences, then 3 validation and 3 test sentences
    toy = [([w, 'alloy', 'high'] + ['high'] * k, ['B-MAT', 'I-MAT', 'O'] + ['O'] * k,
            f'{"abcdef"[k // 3]}.conll')
           for k, w in enumerate(['steel', 'yield'] * 9)]
    parts = [list(range(0, 12)), list(range(12, 15)), list(range(15, 18))]
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: toy)
    monkeypatch.setattr(nb, 'published_split', lambda data, granularity: (parts, list(data)))
    nb.run(nb.parse_args(['--granularity', 'sofc', '--model', str(path), '--epochs', '1',
                          '--threads', '1', '--device', 'cpu', '--published-split', '--crf',
                          '--train-articles', '2', '--draws', '2', '--augment-mentions', '2',
                          '--tag', 'm100', '--seed-offset', '100', '--save-predictions']))
    res = json.loads((tmp_path / 'ner' / 'ner_results_sofc_article_draws_crf_mr2_m100.json')
                     .read_text())
    assert res['protocol'] == 'article_draws' and res['variant'] == 'crf_mr2_m100'
    assert [r['run_id'] for r in res['runs']] == ['draw0', 'draw1']
    assert all(r['seed'] == 100 and r['n_train'] == 6 and r['test_index'] == parts[2]
               for r in res['runs'])
