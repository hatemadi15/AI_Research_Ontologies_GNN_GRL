"""MuLMS protocols in ner_baseline.py (--granularity mulms; ROADMAP §4, week 10): held-out
subfields (--leave-one-subfield-out) and random splits of a training pool (--vary-split
--train-pool N), both with a fixed number of training and validation sentences. The folds and
the parser need no torch; the run with a tiny random model needs torch, transformers, seqeval
and scipy (skipped without them)."""
import json
import random

import pytest

import ner_baseline as nb


def toy_corpus():
    """6 documents of 3 subfields; document k has 3 + k sentences, its words name it."""
    data, subfields = [], {}
    for k, (doc, name) in enumerate([('a_01', 'alpha'), ('a_02', 'alpha'), ('b_01', 'beta'),
                                     ('b_02', 'beta'), ('c_01', 'gamma'), ('x_01', 'gamma')]):
        subfields[f'{doc}.conll'] = name
        data += [([doc, str(i)], ['B-MAT', 'O'], f'{doc}.conll') for i in range(3 + k)]
    return data, subfields


def test_subfield_folds_hold_out_each_subfield():
    data, subfields = toy_corpus()
    folds = nb.subfield_folds(data, subfields, split_seed=0, pool=5, indices=True)
    assert [name for name, *_ in folds] == ['alpha', 'beta', 'gamma']
    docs = sorted({d for _, _, d in data})
    for i, (name, train, val, test) in enumerate(folds):
        assert test == [j for j, x in enumerate(data) if subfields[x[2]] == name]
        order = [d for d in docs if subfields[d] != name]
        random.Random(i).shuffle(order)
        pool = [j for d in order for j, x in enumerate(data) if x[2] == d][:5]
        assert sorted(train + val) == sorted(pool) and len(val) == round(0.15 * 5)
        shuffled = list(range(5))
        random.Random(0).shuffle(shuffled)
        assert val == [pool[k] for k in shuffled[:1]]
        assert not set(train + val) & set(test)
    # without a pool every other document trains and validates
    whole = nb.subfield_folds(data, subfields, split_seed=0, indices=True)
    assert all(sorted(tr + va) == sorted(j for j, x in enumerate(data)
                                         if subfields[x[2]] != name)
               for name, tr, va, _ in whole)
    as_data = nb.subfield_folds(data, subfields, split_seed=0, pool=5)
    assert as_data[0][3] == [data[j] for j in folds[0][3]]
    with pytest.raises(SystemExit, match='hold only'):
        nb.subfield_folds(data, subfields, split_seed=0, pool=100)
    with pytest.raises(SystemExit, match='no subfield'):
        nb.subfield_folds(data, {d: n for d, n in subfields.items() if d != 'x_01.conll'}, 0)


def test_pool_split_trains_on_the_pool_and_tests_on_the_next_fifth():
    train, val, test = nb.pool_split(100, seed=3, pool=40)
    idx = list(range(100))
    random.Random(3).shuffle(idx)
    assert (train, val, test) == (idx[:34], idx[34:40], idx[40:60])
    with pytest.raises(SystemExit, match='--train-pool'):
        nb.pool_split(10, seed=0, pool=10)


def test_make_folds_reads_the_subfields(tmp_path, monkeypatch):
    data, subfields = toy_corpus()
    (tmp_path / 'subfields.json').write_text(json.dumps(
        {d.removesuffix('.conll'): n for d, n in subfields.items()}))
    monkeypatch.setattr(nb, 'MULMS_DIR', str(tmp_path))
    assert nb.mulms_subfields(str(tmp_path)) == subfields
    with pytest.raises(SystemExit, match='mulms.py'):
        nb.mulms_subfields(str(tmp_path / 'missing'))
    args = nb.parse_args(['--granularity', 'mulms', '--leave-one-subfield-out',
                          '--train-pool', '5', '--crf'])
    monkeypatch.setattr(nb, 'mulms_subfields', lambda folder=None: subfields)
    assert nb.make_folds(data, args, indices=True) == nb.subfield_folds(
        data, subfields, 0, 5, indices=True)
    pool = nb.parse_args(['--granularity', 'mulms', '--vary-split', '--train-pool', '5',
                          '--seeds', '0', '1', '--crf'])
    assert nb.make_folds(data, pool, indices=True) == [
        (f'seed{s}', *nb.pool_split(len(data), s, 5)) for s in (0, 1)]


def test_parser_and_names(capsys):
    args = nb.parse_args(['--granularity', 'mulms', '--leave-one-subfield-out', '--train-pool',
                          '400', '--crf', '--augment-mentions', '3', '--tag', 'm100',
                          '--seed-offset', '100'])
    assert nb.protocol_name(args) == 'leave_one_subfield_out' and nb.held_out_protocol(args)
    assert nb.variant_name(args) == 'crf_mr_m100'
    assert (nb.stored_config(args)['train_pool'],
            nb.stored_config(args)['leave_one_subfield_out']) == (400, True)
    pool = nb.parse_args(['--granularity', 'mulms', '--vary-split', '--train-pool', '400',
                          '--crf'])
    assert nb.protocol_name(pool) == 'random_split_pool' and not nb.held_out_protocol(pool)
    for unchanged in (['--crf'], ['--granularity', 'sofc', '--published-split', '--crf'],
                      ['--granularity', 'fine', '--leave-one-paper-out', '--crf']):
        config = nb.stored_config(nb.parse_args(unchanged))
        assert 'train_pool' not in config and 'leave_one_subfield_out' not in config
    assert nb.parse_result_name('ner_results_mulms_leave_one_subfield_out_crf_m0.json') == (
        'mulms', 'leave_one_subfield_out', 'crf_m0')
    assert nb.parse_result_name('ner_results_mulms_random_split_pool_crf_m0.json') == (
        'mulms', 'random_split_pool', 'crf_m0')
    for argv, error in ((['--granularity', 'mulms', '--crf'], 'needs --leave-one-subfield-out'),
                        (['--granularity', 'mulms', '--vary-split', '--crf'],
                         'needs --leave-one-subfield-out'),
                        (['--granularity', 'fine', '--leave-one-subfield-out'],
                         'are for --granularity mulms'),
                        (['--granularity', 'fine', '--vary-split', '--train-pool', '9'],
                         'are for --granularity mulms'),
                        (['--granularity', 'mulms', '--leave-one-subfield-out', '--vary-split'],
                         'a protocol of its own'),
                        (['--granularity', 'mulms', '--leave-one-subfield-out',
                          '--train-pool', '0'], 'at least 1 sentence'),
                        (['--granularity', 'mulms', '--leave-one-subfield-out', '--crf',
                          '--context', '8'], 'not available with --granularity mulms'),
                        (['--granularity', 'mulms', '--leave-one-subfield-out',
                          '--leave-one-paper-out'], 'not available with --granularity mulms')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err


def test_a_run_on_held_out_subfields(tmp_path, monkeypatch):
    """One file per variant, a run per subfield (named after it), with the model seed of
    --seed-offset and the test sentences of that subfield."""
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
    toy = [([w, 'alloy', 'high'] + ['high'] * (k % 3), ['B-MAT', 'I-MAT', 'O'] + ['O'] * (k % 3),
            f'{"aabbcc"[k // 3]}{k // 3}.conll')
           for k, w in enumerate(['steel', 'yield'] * 9)]
    subfields = {f'{"aabbcc"[i]}{i}.conll': {'a': 'alpha', 'b': 'beta', 'c': 'gamma'}["aabbcc"[i]]
                 for i in range(6)}
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: toy)
    monkeypatch.setattr(nb, 'mulms_subfields', lambda folder=None: subfields)
    nb.run(nb.parse_args(['--granularity', 'mulms', '--model', str(path), '--epochs', '1',
                          '--threads', '1', '--device', 'cpu', '--leave-one-subfield-out',
                          '--train-pool', '6', '--crf', '--augment-mentions', '2', '--tag',
                          'm100', '--seed-offset', '100', '--save-predictions']))
    res = json.loads((tmp_path / 'ner' / 'ner_results_mulms_leave_one_subfield_out_crf_mr_m100'
                                         '.json').read_text())
    assert res['protocol'] == 'leave_one_subfield_out' and res['variant'] == 'crf_mr_m100'
    assert res['paper_test_f1'] is None
    assert [r['run_id'] for r in res['runs']] == ['alpha', 'beta', 'gamma']
    folds = nb.subfield_folds(toy, subfields, 0, 6, indices=True)
    assert all(r['seed'] == 100 and r['n_train'] == 5 and r['n_val'] == 1
               and r['test_index'] == fold[3] for r, fold in zip(res['runs'], folds))


def test_silver_names_from_the_unused_sentences_of_some_subfields(tmp_path, monkeypatch):
    """Week 11 on MuLMS: --subfields runs only the folds named, and --self-train unused gives
    the teachers, per fold, the sentences in none of its parts (outside the pool and the test
    subfield), whose names join the students' pool (--augment-names)."""
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
    toy = [([w, 'alloy', 'high'] + ['high'] * (k % 3), ['B-MAT', 'I-MAT', 'O'] + ['O'] * (k % 3),
            f'{"aabbcc"[k // 3]}{k // 3}.conll')
           for k, w in enumerate(['steel', 'yield'] * 9)]
    subfields = {f'{"aabbcc"[i]}{i}.conll': {'a': 'alpha', 'b': 'beta', 'c': 'gamma'}["aabbcc"[i]]
                 for i in range(6)}
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: toy)
    monkeypatch.setattr(nb, 'mulms_subfields', lambda folder=None: subfields)
    samples = []
    train_with_teachers = nb.train_with_teachers

    def recording(seed, offsets, train, val, test, labels, args, device, sample, **kwargs):
        samples.append(sample)
        return train_with_teachers(seed, offsets, train, val, test, labels, args, device,
                                   sample, **kwargs)
    monkeypatch.setattr(nb, 'train_with_teachers', recording)
    nb.run(nb.parse_args(['--granularity', 'mulms', '--model', str(path), '--epochs', '1',
                          '--threads', '1', '--device', 'cpu', '--leave-one-subfield-out',
                          '--train-pool', '6', '--crf', '--teachers', '2', '--self-train',
                          'unused', '--augment-mentions', '2', '--augment-names',
                          '--subfields', 'beta', '--save-predictions']))
    [(_, tr, va, te)] = [f for f in nb.subfield_folds(toy, subfields, 0, 6, indices=True)
                         if f[0] == 'beta']
    used = set(tr) | set(va) | set(te)
    assert samples == [[toy[i][0] for i in range(len(toy)) if i not in used]]
    assert len(samples[0]) == 12 - 6                  # alpha and gamma, outside the pool
    for off in (0, 100):
        res = json.loads((tmp_path / 'ner' / f'ner_results_mulms_leave_one_subfield_out_'
                                              f'crf_mrs_m{off}.json').read_text())
        assert res['variant'] == f'crf_mrs_m{off}'
        assert (res['config']['self_train'], res['config']['self_train_sha256']) == \
            ('unused', None)
        assert 'subfields' not in res['config']
        [run] = res['runs']
        assert run['run_id'] == 'beta' and run['seed'] == off and run['test_index'] == te
        assert run['self_training']['n_sampled'] == 6
        assert run['self_training']['names']['n_test_entities'] == len(te)  # one MAT each
