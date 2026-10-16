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


# ---- SOFC-Exp's two levels (week-6 amendment) ----

def test_in_sofc_scope():
    exp, mat = (('B-EXPERIMENT', 'O'), 0.9), (('B-MATERIAL', 'O'), 0.99)
    tagged = [{'coarse': [exp, exp, mat], 'fine': []}, {'coarse': [exp, mat, mat], 'fine': []}]
    assert nb.in_sofc_scope(tagged, 3) == [True, False, False]     # every teacher, or none


def test_spaced_corpus(tmp_path):
    corpus = tmp_path / 'c.txt'
    corpus.write_text('The Ni - YSZ anode gave 0.5 W cm−2 .\n\nFe3+ and O2− ions\n',
                      encoding='utf-8')
    sample, config = nb.self_training_corpus(str(corpus), 0, spaced=True)
    assert sample == [['The', 'Ni', '-', 'YSZ', 'anode', 'gave', '0.5', 'W', 'cm−2', '.'],
                      ['Fe3+', 'and', 'O2−', 'ions']]
    assert config['self_train_words'] == 'spaces' and config['self_train_sample_size'] == 2
    plain, config = nb.self_training_corpus(str(corpus), 0)
    assert plain[1] == ['Fe3', '+', 'and', 'O2', '-', 'ions'] and 'self_train_words' not in config
    assert nb.spaced_corpus(nb.parse_args(['--granularity', 'sofc_both'])) and \
        not nb.spaced_corpus(nb.parse_args([]))


def test_student_offsets(tmp_path, capsys):
    corpus = tmp_path / 'c.txt'
    corpus.write_text('steel alloy\n')
    base = ['--crf', '--multi-task', '--self-train', str(corpus), '--teachers', '3',
            '--students', '10']
    assert nb.student_offsets_of(nb.parse_args(base)) == list(range(0, 1000, 100))
    args = nb.parse_args(base + ['--student-offsets', '500', '600', '700'])
    assert nb.student_offsets_of(args) == [500, 600, 700]
    assert 'student_offsets' not in nb.stored_config(args)          # a schedule, as --papers
    for argv, error in ((base + ['--student-offsets', '50'], 'multiples of 100'),
                        (base + ['--student-offsets', '100', '100'], 'distinct'),
                        (['--crf', '--self-train', str(corpus), '--student-offsets', '0'],
                         '--student-offsets needs --teachers')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err


def _tiny_model(tmp_path):
    import torch
    import transformers
    path = tmp_path / 'tiny'
    path.mkdir()
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', *TOY_WORDS, '##s']
    (path / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    transformers.BertTokenizerFast(str(path / 'vocab.txt')).save_pretrained(path)
    torch.manual_seed(0)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64)).save_pretrained(path)
    return path


def _toy_sofc():
    """SOFC-Exp's two levels on the toy sentences: slot tags from the fine ones, entity tags
    from the coarse ones (Condition as EXPERIMENT); papers a and b train, c is dev, d test."""
    toy = _toy_corpora()
    slot_of = {'Steel': 'anode_material', 'Alloy': 'cathode_material',
               'Strength': 'power_density', 'Yield': 'resistance', 'Temp': 'working_temperature'}
    entity_of = {'Material': 'MATERIAL', 'Property': 'VALUE', 'Condition': 'EXPERIMENT'}

    def rename(tags, names):
        return [t if t == 'O' else f'{t[:2]}{names[t[2:]]}' for t in tags]
    return {'sofc_slot': [(t, rename(tags, slot_of), d) for t, tags, d in toy['fine']],
            'sofc': [(t, rename(tags, entity_of), d) for t, tags, d in toy['coarse']]}


def test_multitask_self_training_on_sofc_exp(tmp_path, monkeypatch):
    """--granularity sofc_both: the corpus read as spaced words, silver only where every
    teacher tags an EXPERIMENT entity (in_sofc_scope), and --student-offsets: the teachers as
    always, then only the students asked for, a slot-level and an entity-level file each."""
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    path = _tiny_model(tmp_path)
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('steel alloy-yield room\nhigh tensile strength\nroom temperature steel\n')
    corpora = _toy_sofc()
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: corpora[granularity])
    raw = tmp_path / 'raw'
    (raw / 'sofc_exp').mkdir(parents=True)
    (raw / 'sofc_exp' / 'split.json').write_text(json.dumps(
        {'a': 'train', 'b': 'train', 'c': 'dev', 'd': 'test'}))
    monkeypatch.setattr(nb, 'RAW_DIR', str(raw))
    monkeypatch.setattr(nb, 'filtered_tags', lambda votes, agree, threshold: list(votes[0][0]))
    scopes = []

    def scope(tagged, n):                 # the second corpus sentence is out of scope
        scopes.append((len(tagged), n))
        return [True, False, True][:n]
    monkeypatch.setattr(nb, 'in_sofc_scope', scope)
    calls, train = [], nb.train_multitask

    def spy(seed, *args, **kwargs):
        calls.append((seed, kwargs.get('keep_tagger', False), kwargs.get('silver')))
        return train(seed, *args, **kwargs)
    monkeypatch.setattr(nb, 'train_multitask', spy)
    argv = ['--top-k', '4', '--model', str(path), '--epochs', '2', '--threads', '1',
            '--device', 'cpu', '--granularity', 'sofc_both', '--published-split', '--seeds', '0',
            '--crf', '--multi-task', '--self-train', str(corpus), '--teachers', '3',
            '--students', '4', '--student-offsets', '200', '300', '--self-train-sample', '0',
            '--save-predictions']
    nb.run(nb.parse_args(argv))
    assert [(s, k) for s, k, _ in calls] == [(0, True), (100, True), (200, True),
                                            (200, False), (300, False)]
    assert scopes == [(3, 3)]
    silver = calls[3][2]
    assert [words for words, _ in silver] == [['steel', 'alloy-yield', 'room'],
                                              ['room', 'temperature', 'steel']]
    assert calls[4][2] == silver
    assert not list((tmp_path / 'ner').glob('*_m0.json'))
    assert not list((tmp_path / 'ner').glob('*_m100.json'))
    for off in (200, 300):
        for g in ('sofc_slot', 'sofc'):
            res = json.loads((tmp_path / 'ner' / f'ner_results_{g}_published_split_'
                                                  f'crf_mt_st3_m{off}.json').read_text())
            assert (res['granularity'], res['variant']) == (g, f'crf_mt_st3_m{off}')
            config = res['config']
            assert config['students'] == 4 and 'student_offsets' not in config
            assert config['self_train_words'] == 'spaces'
            (run,) = res['runs']
            assert (run['run_id'], run['seed']) == ('seed0', off) and 'test_pred' in run
            info = run['self_training']
            assert (info['n_in_scope'], info['n_silver_sentences']) == (2, 2)
            assert info['n_silver'] == {'fine': 2, 'coarse': 2}
            assert [t['seed'] for t in info['teachers']] == [0, 100, 200]
    n = len(calls)
    nb.run(nb.parse_args(argv))                       # finished runs are skipped
    assert len(calls) == n
