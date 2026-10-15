"""Multi-task self-training in ner_baseline.py (--crf --multi-task --self-train CORPUS
--teachers N; P1 "Next, after week 4" item 2).

The silver filters and the parser need no torch; the weighted loss needs torch, and the run
with a tiny random model torch, transformers, seqeval and scipy (skipped without them).
"""
import json

import pytest

import ner_baseline as nb

E = ('B-X', 'O')          # tags with an entity
N = ('O', 'O')            # tags without


def test_filtered_tags():
    unanimous = [(E, 0.95), (E, 0.92), (E, 0.91)]
    assert nb.filtered_tags(unanimous, 3, 0.9) == list(E)
    assert nb.filtered_tags(unanimous, 3, 0.93) is None           # a teacher below it
    two = [(E, 0.95), (E, 0.92), (('B-Y', 'O'), 0.99)]
    assert nb.filtered_tags(two, 3, 0.9) is None
    assert nb.filtered_tags(two, 2, 0.9) == list(E)
    # only the teachers that agree count for the marginal
    assert nb.filtered_tags([(E, 0.95), (E, 0.92), (('B-Y', 'O'), 0.1)], 2, 0.9) == list(E)
    assert nb.filtered_tags([(N, 0.99)] * 3, 3, 0.9) is None      # no entity
    # with 3 teachers, the strictest filter is the week-4 rule
    for votes in (unanimous, two, [(E, 0.85)] * 3, [(N, 0.99)] * 3):
        assert (nb.filtered_tags(votes, *nb.SILVER_FILTERS[0]) is not None) == \
            nb.unanimous_silver(votes)


def test_choose_filter_takes_the_largest_yield_that_is_as_often_right():
    gold = [E, E, E, N]
    votes = [
        [(E, 0.95)] * 3,                                  # every filter, right
        [(E, 0.85)] * 3,                                  # from (3, 0.8) on, right
        [(E, 0.95), (E, 0.95), (('B-Y', 'O'), 0.95)],     # 2 of 3 only, right
        [(E, 0.95), (E, 0.95), (N, 0.95)],                # 2 of 3 only, wrong (gold: no entity)
    ]
    choice, table = nb.choose_filter(votes, gold)
    assert [(r['n'], r['right']) for r in table] == [(1, 1), (2, 2), (2, 2), (3, 2), (4, 3)]
    assert choice == (3, 0.8)                             # 2 of 2 right; the 2-of-3 ones are not
    # if the strictest selects nothing, it stays
    assert nb.choose_filter([[(E, 0.5)] * 3], [E])[0] == nb.SILVER_FILTERS[0]


def test_parser_and_names(tmp_path, capsys):
    corpus = tmp_path / 'c.txt'
    corpus.write_text('steel alloy\n')
    base = ['--crf', '--multi-task', '--self-train', str(corpus)]
    args = nb.parse_args(base + ['--teachers', '3', '--students', '4'])
    assert nb.variant_name(args) == 'crf_mt_st3' and nb.reference_variant('crf_mt_st3_m0') == 'crf'
    for argv, error in ((base, 'needs --teachers'),
                        (['--crf', '--label-semantics', '--self-train', str(corpus)],
                         'not available with --label-semantics')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err


def test_weighted_multitask_loss():
    torch = pytest.importorskip('torch')
    crfs = {'fine': nb.make_crf(['O', 'B-A', 'I-A']), 'coarse': nb.make_crf(['O', 'B-X', 'I-X'])}
    torch.manual_seed(0)
    logits = {h: torch.randn(3, 4, 3) for h in crfs}
    labels = {'fine': torch.tensor([[0, 1, 2, -100], [1, 0, -100, -100], [-100] * 4]),
              'coarse': torch.tensor([[1, 2, 0, -100], [0, 0, -100, -100], [1, 0, 0, 0]])}
    plain = nb.multitask_loss(logits, labels, crfs)
    ones = nb.multitask_loss(logits, labels, crfs, weights=torch.ones(3))
    assert all(torch.allclose(plain[h], ones[h]) for h in crfs)
    # row 1 at half weight: each head's mean over its labelled rows weighs it 0.5
    half = nb.multitask_loss(logits, labels, crfs, weights=torch.tensor([1.0, 0.5, 1.0]))
    for h in crfs:
        rows = (labels[h] != -100).any(1)
        nlls = [crfs[h].nll(*nb.word_view(logits[h][i:i + 1], labels[h][i:i + 1]))
                for i in range(3) if rows[i]]
        w = [1.0, 0.5, 1.0][:len(nlls)] if h == 'fine' else [1.0, 0.5, 1.0]
        expected = sum(n * x for n, x in zip(nlls, w)) / len(nlls)
        assert torch.allclose(half[h], expected, atol=1e-5)


TOY_WORDS = ['steel', 'alloy', 'tensile', 'strength', 'yield', 'room', 'temperature', 'high']


def _toy_corpora():
    """Fine and coarse versions of 24 sentences of four 'papers' (as test_ner_multitask)."""
    fine, coarse = [], []
    for p, doc in enumerate(('a.conll', 'b.conll', 'c.conll', 'd.conll')):
        for k in range(6):
            temp = k % 2 == 0
            tokens = ['steel', 'alloy', 'tensile', 'strengths', 'yield', 'room', 'temperature']
            f_tags = ['B-Steel', 'B-Alloy', 'B-Strength', 'I-Strength', 'B-Yield'] + (
                ['B-Temp', 'I-Temp'] if temp else ['O', 'O'])
            c_tags = ['B-Material', 'B-Material', 'B-Property', 'I-Property', 'B-Property'] + (
                ['B-Condition', 'I-Condition'] if temp else ['O', 'O'])
            extra = ['high'] * ((6 * p + k) % 5)
            coarse.append((tokens + extra, c_tags + ['O'] * len(extra), doc))
            fine.append((tokens + extra, f_tags + ['O'] * len(extra), doc))
    return {'fine': fine, 'coarse': coarse}


def test_multitask_self_training_run(tmp_path, monkeypatch):
    """3 multi-task teachers, a filter per head, and 4 students on the same silver data,
    each written to a fine and a coarse file; a second run trains nothing."""
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    path = tmp_path / 'tiny'
    path.mkdir()
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', *TOY_WORDS, '##s']
    (path / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    transformers.BertTokenizerFast(str(path / 'vocab.txt')).save_pretrained(path)
    torch.manual_seed(0)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64)).save_pretrained(path)
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('steel alloy yield\nhigh tensile strength\nroom temperature steel\n')
    corpora = _toy_corpora()
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: corpora[granularity])
    # every tagged sentence silver for both heads, with the first teacher's tags
    monkeypatch.setattr(nb, 'filtered_tags', lambda votes, agree, threshold: list(votes[0][0]))
    calls, train = [], nb.train_multitask

    def spy(seed, *args, **kwargs):
        calls.append((seed, kwargs.get('keep_tagger', False), kwargs.get('silver')))
        return train(seed, *args, **kwargs)
    monkeypatch.setattr(nb, 'train_multitask', spy)
    argv = ['--top-k', '4', '--model', str(path), '--epochs', '2', '--threads', '1',
            '--device', 'cpu', '--crf', '--multi-task', '--self-train', str(corpus),
            '--teachers', '3', '--students', '4', '--self-train-sample', '0', '--vary-split',
            '--seeds', '0', '--save-predictions']
    nb.run(nb.parse_args(argv))
    assert [(s, k) for s, k, _ in calls] == [(0, True), (100, True), (200, True),
                                            (0, False), (100, False), (200, False), (300, False)]
    silver = calls[3][2]
    assert len(silver) == 3 and all(c[2] == silver for c in calls[4:])
    assert all(set(tags) == {'fine', 'coarse'} for _, tags in silver)
    for off in (0, 100, 200, 300):
        for h in ('fine', 'coarse'):
            res = json.loads((tmp_path / 'ner' / f'ner_results_{h}_random_split_vary_'
                                                  f'crf_mt_st3_m{off}.json').read_text())
            assert res['variant'] == f'crf_mt_st3_m{off}' and res['config']['students'] == 4
            (run,) = res['runs']
            assert run['seed'] == off and 'test_pred' in run
            info = run['self_training']
            assert info['n_silver'] == {'fine': 3, 'coarse': 3}
            assert info['n_silver_sentences'] == 3
            assert [t['seed'] for t in info['teachers']] == [0, 100, 200]
            assert set(info['filters'][h]) == {'agree', 'threshold', 'validation'}
    n = len(calls)
    nb.run(nb.parse_args(argv))                       # finished runs are skipped
    assert len(calls) == n
