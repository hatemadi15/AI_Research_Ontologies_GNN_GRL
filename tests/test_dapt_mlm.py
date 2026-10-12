"""DAPT corpus filter, held-out split and batching; masking and resuming need torch."""
import random
import statistics
from types import SimpleNamespace

import pytest

import dapt_mlm as dm

REFERENCE = [('a.conll', 'The low - cycle fatigue ( LCF ) behaviour of the alloy was studied .'),
             ('b.conll', 'Fatigue tests .')]       # fewer than 8 tokens: no 8-grams


def test_normalise_lowercases_and_splits_punctuation():
    assert dm.normalise('Low-cycle fatigue (LCF), R = 0.1') == [
        'low', '-', 'cycle', 'fatigue', '(', 'lcf', ')', ',', 'r', '=', '0', '.', '1']
    # NFKC (no-break space, micro sign, ligature) and dashes
    mu = '\u03bcm'
    assert dm.normalise('5\u00a0\u00b5m \u2013 10 \u03bcm') == ['5', mu, '-', '10', mu]
    assert dm.normalise('R = \u22121, \ufb01ne') == ['r', '=', '-', '1', ',', 'fine']


def test_overlap_filter_and_duplicates():
    sentences = [
        'We studied the low-cycle fatigue (LCF) of steel.',       # shares an 8-gram
        'Low-cycle fatigue (LCF) of steel was studied.',          # shares only 7 tokens
        'THE LOW\u2013CYCLE FATIGUE (LCF) BEHAVIOUR was high.',   # case and dash differ
        'Fatigue cracks grow from notches in welded joints.',
        'fatigue  cracks grow from notches in welded joints .',  # duplicate of [3]
        'We studied the low-cycle fatigue (LCF) of steel .',      # duplicate of [0]
        'Fatigue tests.',                                         # short reference sentence
    ]
    kept, dropped = dm.filter_corpus(sentences, REFERENCE, n=8)
    assert kept == [sentences[1], sentences[3], sentences[6]]
    assert [(d['index'], d['duplicate_of']) for d in dropped['duplicate']] == [(4, 3), (5, 0)]
    assert [d['index'] for d in dropped['overlap']] == [0, 2]
    assert dropped['overlap'][0]['ngram'] == 'the low - cycle fatigue ( lcf )'
    assert dropped['overlap'][0]['docs'] == ['a.conll']
    assert dropped['reference_sentences'] == 2 and dropped['reference_ngrams'] == 8
    # a shorter n catches the 7-token overlap as well
    kept7, _ = dm.filter_corpus(sentences, REFERENCE, n=7)
    assert sentences[1] not in kept7
    assert dm.filter_corpus([], REFERENCE)[0] == []


def test_read_sentences_skips_blank_lines(tmp_path):
    path = tmp_path / 'corpus.txt'
    path.write_text('First sentence.\n\n  Second one.  \n\n', encoding='utf-8')
    assert dm.read_sentences(str(path)) == (['First sentence.', 'Second one.'], 2)


def test_heldout_split():
    sentences = [f's{i}' for i in range(1001)]
    train, held = dm.split_heldout(sentences, 0.05, seed=0)
    assert len(held) == 50 and len(train) == 951
    assert sorted(train + held) == sorted(sentences)
    assert dm.split_heldout(sentences, 0.05, seed=0) == (train, held)
    assert dm.split_heldout(sentences, 0.05, seed=1)[1] != held


def test_length_grouped_batches():
    rng = random.Random(1)
    lengths = [rng.randint(5, 128) for _ in range(1000)]
    batches = dm.length_grouped_batches(lengths, 32, seed=0, epoch=0, chunk_batches=5)
    assert batches == dm.length_grouped_batches(lengths, 32, seed=0, epoch=0, chunk_batches=5)
    assert batches != dm.length_grouped_batches(lengths, 32, seed=0, epoch=1, chunk_batches=5)
    assert batches != dm.length_grouped_batches(lengths, 32, seed=1, epoch=0, chunk_batches=5)
    assert sorted(i for b in batches for i in b) == list(range(1000))
    assert sorted(len(b) for b in batches) == [1000 % 32] + [32] * (1000 // 32)

    def padded(bs):
        return sum(len(b) * max(lengths[i] for i in b) for b in bs)
    order = list(range(1000))
    random.Random(0).shuffle(order)
    plain = [order[i:i + 32] for i in range(0, 1000, 32)]
    assert padded(batches) < 0.75 * padded(plain)
    # each chunk of 5 * 32 indices is sorted by length before it is cut into batches
    spreads = [max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches]
    assert statistics.mean(spreads) < 40


def test_derive_seed_is_stable():
    assert dm.derive_seed(0, 'mask', 0, 0) == dm.derive_seed(0, 'mask', 0, 0)
    seeds = {dm.derive_seed(0, 'mask', e, s) for e in range(3) for s in range(100)}
    assert len(seeds) == 300 and all(0 <= s < 2 ** 63 for s in seeds)
    assert dm.derive_seed(0, 'mask', 1, 2) != dm.derive_seed(0, 'dropout', 1, 2)


def test_time_budget_rule():
    # 300 s of steps plus a 10 s save after 3000 s: ends at 3310 s
    assert not dm.next_checkpoint_late(3000, 3400, 300.0, 10.0)
    assert dm.next_checkpoint_late(3000, 3300, 300.0, 10.0)
    assert dm.next_checkpoint_late(3000, 3330, 300.0, 10.0, eval_sec=30.0)
    assert not dm.next_checkpoint_late(10 ** 6, 0, 300.0, 10.0)        # no budget
    args = SimpleNamespace(max_steps=0, time_budget_min=10, save_every=100)
    padded = [1000] * 400 + [3000] * 100      # 500 batches; the last ones are long
    # the next checkpoint (step 200) is 50 steps of 1000 tokens away: 50 s at 1 ms/token;
    # the first step of a job always runs
    assert dm.stop_reason(args, 150, 150, padded, 0, 0.001, {'save': 5.0}, 590) is None
    assert dm.stop_reason(args, 150, 150, padded, 1, 0.001, {'save': 5.0}, 540) is None
    assert 'Time budget' in dm.stop_reason(args, 150, 150, padded, 1, 0.001, {'save': 5.0}, 550)
    # long batches count for more: 50 steps of 3000 tokens before step 500
    assert dm.stop_reason(args, 450, 450, padded, 1, 0.001, {'save': 5.0}, 440) is None
    assert dm.stop_reason(args, 450, 450, padded, 1, 0.001, {'save': 5.0}, 450)
    # near the epoch end the next checkpoint is the epoch-end one, with the evaluation
    timing = {'save': 5.0, 'eval': 60.0}
    assert dm.stop_reason(args, 190, 490, padded, 1, 0.001, timing, 505) is None
    assert dm.stop_reason(args, 190, 490, padded, 1, 0.001, timing, 506)
    capped = SimpleNamespace(max_steps=20, time_budget_min=0, save_every=100)
    assert dm.stop_reason(capped, 19, 19, padded, 5, 0.001, {}, 0) is None
    assert 'max-steps' in dm.stop_reason(capped, 20, 20, padded, 5, 0.001, {}, 0)


def test_resume_refuses_other_settings():
    settings = {'lr': 5e-5, 'seed': 0}
    state = {'settings': settings, 'settings_hash': dm.settings_hash(settings)}
    dm.check_resumable(state, settings, dm.settings_hash(settings), 'state.pt')
    other = {'lr': 1e-4, 'seed': 0}
    with pytest.raises(SystemExit, match='lr'):
        dm.check_resumable(state, other, dm.settings_hash(other), 'state.pt')


# ---- with torch ----

PAD, UNK, CLS, SEP, MASK = 0, 1, 2, 3, 4
SPECIAL = [PAD, UNK, CLS, SEP, MASK]


def _batch(torch, rows=256, width=128, vocab=1000):
    g = torch.Generator().manual_seed(123)
    ids = torch.randint(5, vocab, (rows, width), generator=g)
    lengths = torch.randint(8, width + 1, (rows,), generator=g)
    attention = (torch.arange(width)[None, :] < lengths[:, None]).long()
    ids[:, 0] = CLS
    ids[torch.arange(rows), lengths - 1] = SEP
    ids[attention == 0] = PAD
    return ids, attention


def test_masking_is_deterministic_and_bert_like():
    torch = pytest.importorskip('torch')
    ids, attention = _batch(torch)
    special = dm.special_mask(ids, attention, SPECIAL)

    def masked(*parts):
        gen = torch.Generator().manual_seed(dm.derive_seed(*parts))
        return dm.mask_tokens(ids, special, gen, MASK, 1000)
    inputs, labels = masked(0, 'mask', 0, 7)
    again = masked(0, 'mask', 0, 7)
    assert torch.equal(inputs, again[0]) and torch.equal(labels, again[1])
    assert not torch.equal(labels, masked(0, 'mask', 0, 8)[1])
    assert not torch.equal(labels, masked(0, 'mask', 1, 7)[1])

    selected = labels != -100
    assert not selected[special].any()                     # special tokens and padding
    assert torch.equal(inputs[special], ids[special])
    assert torch.equal(labels[selected], ids[selected])
    assert torch.equal(inputs[~selected], ids[~selected])
    rate = selected.sum().item() / (~special).sum().item()
    assert 0.14 < rate < 0.16
    n = selected.sum().item()
    to_mask = (inputs[selected] == MASK).sum().item() / n
    kept = (inputs[selected] == ids[selected]).sum().item() / n
    assert 0.77 < to_mask < 0.83 and 0.08 < kept < 0.12
    assert 0.08 < 1 - to_mask - kept < 0.12                  # random replacements


def test_heldout_masks_are_fixed():
    pytest.importorskip('torch')
    seqs = [[CLS] + list(range(5, 5 + n)) + [SEP] for n in (6, 30, 12, 50)]
    a = dm.heldout_examples(seqs, SPECIAL, PAD, MASK, 100, seed=0, mlm_prob=0.15)
    assert a == dm.heldout_examples(seqs, SPECIAL, PAD, MASK, 100, seed=0, mlm_prob=0.15)
    assert a != dm.heldout_examples(seqs, SPECIAL, PAD, MASK, 100, seed=1, mlm_prob=0.15)
    for seq, (inputs, labels) in zip(seqs, a):
        assert len(inputs) == len(labels) == len(seq)
        assert labels[0] == labels[-1] == -100 and inputs[0] == CLS and inputs[-1] == SEP


def test_state_is_saved_atomically(tmp_path):
    torch = pytest.importorskip('torch')
    path = tmp_path / 'state' / 'state.pt'
    dm.save_state(str(path), {'step': 3, 'w': torch.ones(2), 'log': [0.5, 0.25]})
    dm.save_state(str(path), {'step': 4, 'w': torch.zeros(2), 'log': [0.5, 0.25, 0.125]})
    state = dm.load_state(str(path))
    assert state['step'] == 4 and state['log'] == [0.5, 0.25, 0.125]
    assert torch.equal(state['w'], torch.zeros(2))
    assert [p.name for p in path.parent.iterdir()] == ['state.pt']
    assert dm.load_state(str(tmp_path / 'missing.pt')) is None


def _tiny_setup(torch, transformers):
    g = torch.Generator().manual_seed(0)
    seqs = [[CLS] + torch.randint(5, 60, (int(n),), generator=g).tolist() + [SEP]
            for n in torch.randint(3, 20, (44,), generator=g)]
    data = {'train': seqs[:36], 'special_ids': SPECIAL, 'pad_id': PAD, 'mask_id': MASK,
            'vocab_size': 60,
            'heldout': dm.heldout_examples(seqs[36:], SPECIAL, PAD, MASK, 60, 0, 0.15)}
    args = SimpleNamespace(seed=0, epochs=2, batch_size=8, lr=1e-3, weight_decay=0.01,
                           warmup_ratio=0.06, max_grad_norm=1.0, mlm_prob=0.15,
                           chunk_batches=2, save_every=3, log_every=0, max_steps=0,
                           time_budget_min=0)
    meta = {'settings': {'tiny': True}, 'settings_hash': 'x'}

    def model():
        torch.manual_seed(0)
        config = transformers.BertConfig(vocab_size=60, hidden_size=16, num_hidden_layers=1,
                                         num_attention_heads=2, intermediate_size=32,
                                         max_position_embeddings=32)
        return transformers.BertForMaskedLM(config)
    return data, args, meta, model


def test_resumed_training_matches_uninterrupted(tmp_path):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    data, args, meta, new_model = _tiny_setup(torch, transformers)
    full_model = new_model()
    full, done = dm.train(full_model, data, args, meta, None, str(tmp_path / 'a.pt'), 0.0)
    assert done and len(full['loss_log']) == 10 and len(full['heldout']) == 3
    assert full['epoch'] == 2 and len(full['train_loss']) == 2

    # stop at step 4 (between checkpoints) and 5 (epoch end), then finish
    path = str(tmp_path / 'b.pt')
    state = None
    for max_steps, n_steps in ((4, 4), (5, 5), (0, 10)):
        model = new_model()
        _, done = dm.train(model, data, SimpleNamespace(**{**vars(args), 'max_steps': max_steps}),
                           meta, state, path, 0.0)
        assert done == (max_steps == 0)
        state = dm.load_state(path)
        assert len(state['loss_log']) == n_steps
    assert len(state['jobs']) == 3
    assert state['loss_log'] == pytest.approx(full['loss_log'], abs=1e-6)
    assert [h['loss'] for h in state['heldout']] == pytest.approx(
        [h['loss'] for h in full['heldout']], abs=1e-6)
    for (name, a), b in zip(full_model.state_dict().items(), model.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6), name
