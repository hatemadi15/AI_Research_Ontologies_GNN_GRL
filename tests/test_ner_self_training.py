"""Self-training in ner_baseline.py (--crf --self-train CORPUS).

The parser checks, the variant names, the corpus words and sample, the silver rule and the
resume isolation need no torch. The CRF marginal and loss tests need torch, and the
tiny-model tests also transformers, seqeval and scipy; they are skipped without them.
"""
import hashlib
import itertools
import json
import os
import random
from types import SimpleNamespace

import pytest

import conll
import ner_baseline as nb
from config import RAW_DIR

CORPUS_LINES = [
    'Steel alloy, tensile strengths.',
    'High yield strength at room temperature.',
    'The steel alloy (high strength) yield.',
    '',                                           # blank lines are no sentences
    'Room-temperature tensile strength of steel.',
    'Alloy steels yield high.',
    ' '.join(['high'] * 70),                      # longer than the tiny encoder (64 positions)
]


def _corpus(tmp_path, lines=CORPUS_LINES, name='corpus.txt'):
    path = tmp_path / name
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def _toy_ner_corpus():
    """24 sentences of four 'papers' with two MAT, two PROP and one COND entity each."""
    data = []
    for p, doc in enumerate(('a.conll', 'b.conll', 'c.conll', 'd.conll')):
        for k in range(6):
            extra = 6 * p + k
            data.append((['steel', 'alloy', 'tensile', 'strengths', 'yield', 'room',
                          'temperature'] + ['high'] * extra,
                         ['B-MAT', 'B-MAT', 'B-PROP', 'I-PROP', 'B-PROP', 'B-COND',
                          'I-COND'] + ['O'] * extra, doc))
    return data


# --- command line, names and resume isolation (no torch)

def test_self_train_parser_checks(tmp_path, capsys):
    corpus = str(_corpus(tmp_path))
    args = nb.parse_args(['--crf', '--self-train', corpus])
    assert args.self_train == corpus and nb.variant_name(args) == 'crf_st'
    args = nb.parse_args(['--crf', '--self-train', corpus, '--tag', 'm100', '--seed-offset', '100',
                          '--save-predictions', '--leave-one-paper-out'])
    assert nb.variant_name(args) == 'crf_st_m100' and nb.stored_config(args)['self_train'] == corpus
    # without the option there is no self_train field: the config other runs store is unchanged
    assert 'self_train' not in vars(nb.parse_args(['--crf']))
    for argv, error in ((['--self-train', corpus], '--self-train needs --crf'),
                        (['--class-weights', '--self-train', corpus], '--self-train needs --crf'),
                        # with --multi-task only as multi-task self-training (--teachers)
                        (['--crf', '--multi-task', '--self-train', corpus],
                         '--self-train with --multi-task needs --teachers'),
                        (['--crf', '--label-semantics', '--self-train', corpus],
                         '--self-train is not available with --label-semantics'),
                        (['--crf', '--self-train', str(tmp_path / 'missing.txt')],
                         '--self-train: no such file')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err
    # a CRF run tagged st or st_m0 would write to the files of the self-training runs
    for tag in ('st', 'st_m0'):
        with pytest.raises(SystemExit):
            nb.parse_args(['--crf', '--tag', tag])
        assert 'would overwrite the results of the crf_st' in capsys.readouterr().err
    assert nb.variant_name(nb.parse_args(['--crf', '--self-train', corpus, '--tag', 'st'])) == \
        'crf_st_st'


def test_teachers_parser_checks(tmp_path, capsys):
    corpus = str(_corpus(tmp_path))
    args = nb.parse_args(['--crf', '--self-train', corpus, '--teachers', '3',
                          '--self-train-sample', '0', '--leave-one-paper-out'])
    assert (args.teachers, args.self_train_sample) == (3, 0)
    assert nb.variant_name(args) == 'crf_st3' and nb.reference_variant('crf_st3_m100') == 'crf'
    # absent unless given: the config of the week-3 runs is unchanged
    args = vars(nb.parse_args(['--crf', '--self-train', corpus]))
    assert 'teachers' not in args and 'self_train_sample' not in args
    for argv, error in ((['--crf', '--teachers', '3'], '--teachers needs --self-train'),
                        (['--crf', '--self-train-sample', '0'],
                         '--self-train-sample needs --self-train'),
                        (['--crf', '--self-train', corpus, '--teachers', '1'], 'at least 2'),
                        (['--crf', '--self-train', corpus, '--teachers', '3', '--tag', 'm0'],
                         'no --tag or --seed-offset'),
                        (['--crf', '--self-train', corpus, '--self-train-sample', '-1'],
                         '--self-train-sample: 0')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err
    # a CRF run tagged st3 would write to the files of the 3-teacher runs
    with pytest.raises(SystemExit):
        nb.parse_args(['--crf', '--tag', 'st3_m0'])
    assert 'would overwrite the results of the crf_st3' in capsys.readouterr().err


def test_students_parser_checks(tmp_path, capsys):
    """--students N (week 5): a student per seed offset 0 to 100 (N - 1), N >= --teachers."""
    corpus = str(_corpus(tmp_path))
    st3 = ['--crf', '--self-train', corpus, '--teachers', '3']
    args = nb.parse_args(st3 + ['--students', '10'])
    assert (args.teachers, args.students, nb.variant_name(args)) == (3, 10, 'crf_st3')
    assert nb.teacher_offsets(args.students) == list(range(0, 1000, 100))
    # absent unless given: the config of the week-4 runs is unchanged
    assert 'students' not in vars(nb.parse_args(st3))
    for argv, error in ((['--crf', '--self-train', corpus, '--students', '4'],
                         '--students needs --teachers'),
                        (st3 + ['--students', '2'], 'at least as many as --teachers')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err


def test_unanimous_silver_and_teacher_seeds():
    sure = (('B-MAT', 'O'), 0.95)
    assert nb.unanimous_silver([sure, sure, sure])
    assert not nb.unanimous_silver([sure, sure, (('B-MAT', 'O'), 0.85)])   # one teacher unsure
    assert not nb.unanimous_silver([sure, sure, (('B-PROP', 'O'), 0.99)])  # one disagrees
    assert not nb.unanimous_silver([(('O', 'O'), 0.99)] * 3)               # no entity
    assert nb.teacher_offsets(3) == [0, 100, 200]


def test_sample_size_zero_is_the_whole_corpus(tmp_path):
    corpus = _corpus(tmp_path)
    sample, config = nb.self_training_corpus(str(corpus), 0)
    assert sample == [nb.corpus_words(line) for line in CORPUS_LINES if line]
    assert config['self_train_sample_size'] == 6
    # the default sample is unchanged
    assert nb.self_training_corpus(str(corpus))[1]['self_train_sample_size'] == 2000


def test_self_training_files_are_a_variant_of_their_own(tmp_path):
    args = nb.parse_args(['--crf', '--self-train', str(_corpus(tmp_path)), '--tag', 'm100',
                          '--seed-offset', '100', '--leave-one-paper-out'])
    assert os.path.basename(nb.results_path(str(tmp_path), 'coarse', args)) == (
        'ner_results_coarse_leave_one_paper_out_crf_st_m100.json')
    # compared with the CRF runs of the same granularity and protocol
    assert nb.reference_variant('crf_st') == nb.reference_variant('crf_st_m100') == 'crf'
    for protocol in ('random_split_fixed', 'random_split_vary', 'leave_one_paper_out'):
        for variant in ('crf_st', 'crf_st_m0', 'crf_st_m200'):
            fn = f'ner_results_fine{nb.result_suffix(protocol, variant)}.json'
            assert nb.parse_result_name(fn) == ('fine', protocol, variant)
    # the runs of a crf file are neither resumed nor replaced by a crf_st run, nor the reverse
    path = tmp_path / 'f.json'
    for stored, now, sha in (('crf', 'crf_st', 'a'), ('crf_st', 'crf', None),
                             ('crf_m100', 'crf_st_m100', 'a')):
        config = {'self_train_sha256': 'a'} if stored.startswith('crf_st') else {}
        path.write_text(json.dumps({'labels': ['A'], 'model': 'm', 'variant': stored,
                                    'config': config, 'runs': [{'run_id': 'p', 'test_pred': []}]}))
        assert nb.resumable_runs(str(path), ['A'], 'm', now, self_train_sha256=sha) == []
        with pytest.raises(nb.OverwriteError, match=f"variant: '{stored}' in the file"):
            nb.check_overwrite(str(path), ['A'], 'm', now, self_train_sha256=sha)


def test_runs_from_another_corpus_are_not_resumed(tmp_path):
    path = tmp_path / 'ner_results_fine_leave_one_paper_out_crf_st_m0.json'
    path.write_text(json.dumps({'labels': ['A'], 'model': 'm', 'variant': 'crf_st_m0',
                                'config': {'seed_offset': 0, 'self_train_sha256': 'a' * 64},
                                'runs': [{'run_id': 'a.conll'}]}))
    same = {'labels': ['A'], 'model': 'm', 'variant': 'crf_st_m0'}
    assert nb.resumable_runs(str(path), **same, self_train_sha256='a' * 64) == [
        {'run_id': 'a.conll'}]
    nb.check_overwrite(str(path), **same, self_train_sha256='a' * 64)
    assert nb.resumable_runs(str(path), **same, self_train_sha256='b' * 64) == []
    with pytest.raises(nb.OverwriteError, match=f"self_train_sha256: '{'a' * 64}' in the file, "
                                                f"'{'b' * 64}' now"):
        nb.check_overwrite(str(path), **same, self_train_sha256='b' * 64)
    # files of other variants store no corpus: unchanged for them
    assert nb.stored_settings({'labels': ['A'], 'model': 'm'}) == {
        'labels': ['A'], 'model': 'm', 'variant': 'baseline', 'seed_offset': 0,
        'self_train_sha256': None}


def test_run_resumes_only_runs_of_the_same_corpus(tmp_path, monkeypatch):
    """run() wiring, with the training replaced: config, resume and the overwrite guard."""
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: _toy_ner_corpus())
    monkeypatch.setattr(nb, 'torch_device', lambda args: 'cpu')
    monkeypatch.setattr(nb, 'train_one', lambda *args, **kwargs: pytest.fail('a plain CRF run'))
    calls = []

    def fake(seed, train, val, test, labels, args, device, sample, test_index=None):
        calls.append((seed, sample))
        return {'f1': 50.0, 'f1_strict': 49.0, 'best_epoch': 0, 'seed': seed,
                'per_class': {'macro avg': {'f1-score': 0.4}, 'weighted avg': {'f1-score': 0.5}},
                'self_training': {'n_silver': 1}}
    monkeypatch.setattr(nb, 'train_self_training', fake)
    corpus = _corpus(tmp_path)
    argv = ['--crf', '--self-train', str(corpus), '--granularity', 'fine', '--top-k', '2',
            '--vary-split', '--seeds', '0', '--tag', 'm100', '--seed-offset', '100']
    nb.run(nb.parse_args(argv))
    out = tmp_path / 'ner' / 'ner_results_fine_random_split_vary_crf_st_m100.json'
    res = json.loads(out.read_text())
    sample = [nb.corpus_words(line) for line in CORPUS_LINES if line]
    assert calls == [(100, sample)]
    assert res['variant'] == 'crf_st_m100' and res['runs'][0]['self_training'] == {'n_silver': 1}
    config = res['config']
    assert config['self_train'] == str(corpus)          # outside the project: the path as given
    assert config['self_train_sha256'] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert (config['self_train_sample_size'], config['self_train_silver_weight']) == (2000, 0.5)
    nb.run(nb.parse_args(argv))                          # the same corpus: resumed
    assert len(calls) == 1
    before = out.read_text()
    _corpus(tmp_path, CORPUS_LINES[:3])                  # another corpus: neither resumed ...
    with pytest.raises(nb.OverwriteError, match='self_train_sha256: '):
        nb.run(nb.parse_args(argv))
    assert len(calls) == 1 and out.read_text() == before          # ... nor replaced by accident
    nb.run(nb.parse_args(argv + ['--overwrite']))
    assert len(calls) == 2 and len(calls[1][1]) == 3
    assert json.loads(out.read_text())['config']['self_train_sha256'] == hashlib.sha256(
        corpus.read_bytes()).hexdigest()
    # an empty corpus stops the run before any training
    _corpus(tmp_path, ['', ' '])
    with pytest.raises(SystemExit, match='no sentences'):
        nb.run(nb.parse_args(argv + ['--overwrite']))
    assert len(calls) == 2


# --- corpus words and the sample (no torch)

def test_corpus_words_follow_the_materiominer_files():
    words = nb.corpus_words
    # hyphens join letters only; '.' letters or digits, ',' digits; '%' stays after a digit;
    # en dash (U+2013), em dash (U+2014) and minus (U+2212) are written '-'
    assert words('Displacement-controlled tests (DC) at 0.5% strain, i.e. 10\u201320 Hz.') == [
        'Displacement-controlled', 'tests', '(', 'DC', ')', 'at', '0.5%', 'strain', ',', 'i.e',
        '.', '10', '-', '20', 'Hz', '.']
    assert words('The 41NiCrMo7-3-2 steel with a 10-kN load cell at \u221235 \u00b0C '
                 '(R = \u22121).') == [
        'The', '41NiCrMo7', '-', '3', '-', '2', 'steel', 'with', 'a', '10', '-', 'kN', 'load',
        'cell', 'at', '-', '35', '\u00b0', 'C', '(', 'R', '=', '-', '1', ')', '.']
    assert words('Inconel 725\u2014an advanced nickel (Ni)-base alloy [6,14\u201319], e.g. '
                 'X-ray.') == [
        'Inconel', '725', '-', 'an', 'advanced', 'nickel', '(', 'Ni', ')', '-', 'base', 'alloy',
        '[', '6,14', '-', '19', ']', ',', 'e.g', '.', 'X-ray', '.']
    # NFKC (the fi ligature, the micro sign, subscripts, a no-break space), curly quotes, and
    # characters that the BERT tokenizer drops (a soft hyphen)
    assert words('Young\u2019s modulus of \ufb01ne fa\u00adtigue specimens: 1,000 \u00b5m; '
                 '\u2018H\u2082SO\u2084\u2019 Fig.1a') == [
        "Young's", 'modulus', 'of', 'fine', 'fatigue', 'specimens', ':', '1,000', '\u03bcm', ';',
        "'", 'H2SO4', "'", 'Fig', '.', '1a']
    assert words('\u03c3_a/2 = 300\u00a0MPa') == ['\u03c3', '_', 'a', '/', '2', '=', '300', 'MPa']
    # the MaterioMiner sentences, joined by spaces, split back into their own tokens, except
    # for a few words with a stray hyphen
    unsplit = set()
    for sub in ('fine_grained_ner', 'coarse_grained_ner'):
        folder = os.path.join(RAW_DIR, sub)
        for fn in sorted(os.listdir(folder)):
            for tokens, _ in conll.read_conll_sentences(os.path.join(folder, fn)):
                if words(' '.join(tokens)) != tokens:
                    unsplit |= {t for t in tokens if words(t) != [t]}
    assert unsplit == {'single-', 'in-', '-based', '-related', '-thick', '-e.g', '-which', '-a',
                       '-in'}


def test_the_sample_is_fixed_by_seed_0(tmp_path):
    lines = [f'Sentence {i} of the fatigue corpus.' for i in range(30)]
    path = _corpus(tmp_path, lines[:10] + ['', '  '] + lines[10:])
    random.seed(123)                       # the global random state plays no part
    sample, config = nb.self_training_corpus(str(path), size=12)
    order = list(range(30))
    random.Random(0).shuffle(order)
    assert sample == [nb.corpus_words(lines[i]) for i in sorted(order[:12])]   # in corpus order
    assert nb.self_training_corpus(str(path), size=12)[0] == sample
    assert config == {'self_train': str(path), 'self_train_sample_size': 12,
                      'self_train_silver_weight': 0.5,
                      'self_train_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    # the default: 2,000 sentences, or the whole of a smaller corpus
    full, config = nb.self_training_corpus(str(path))
    assert full == [nb.corpus_words(line) for line in lines]
    assert config['self_train_sample_size'] == nb.SELF_TRAIN_SAMPLE == 2000
    big = _corpus(tmp_path, [f'w{i}' for i in range(2500)], 'big.txt')
    assert len(nb.self_training_corpus(str(big))[0]) == 2000


def test_silver_sentences_have_an_entity_and_only_confident_tags():
    assert (nb.SILVER_THRESHOLD, nb.SILVER_WEIGHT) == (0.9, 0.5)
    assert nb.is_silver(['B-A', 'I-A', 'O'], [0.95, 0.9, 0.99])        # 0.9 is enough
    assert not nb.is_silver(['B-A', 'I-A', 'O'], [0.95, 0.89, 0.99])   # one word below
    assert not nb.is_silver(['O', 'O', 'O'], [1.0, 1.0, 1.0])          # no entity
    assert nb.is_silver(['O', 'B-B'], [0.97, 0.91])


# --- CRF marginals and the weighted loss (torch)
LABELS = ['O', 'B-A', 'I-A', 'B-B', 'I-B']


@pytest.fixture
def torch():
    return pytest.importorskip('torch')


def _crf(torch, seed=0):
    torch.manual_seed(seed)
    crf = nb.make_crf(LABELS)
    with torch.no_grad():
        for p in crf.parameters():
            p.normal_()
    return crf


def _brute_force(torch, crf, emissions):
    """log Z and the marginals (W, L) of one sentence, from every one of its tag paths."""
    trans, start, end = crf.scores()
    paths = list(itertools.product(range(len(LABELS)), repeat=emissions.size(0)))
    scores = []
    for path in paths:
        s = start[path[0]] + emissions[0, path[0]] + end[path[-1]]
        for t in range(1, len(path)):
            s = s + trans[path[t - 1], path[t]] + emissions[t, path[t]]
        scores.append(s)
    scores = torch.stack(scores)
    marginals = torch.zeros(emissions.shape)
    for path, p in zip(paths, torch.softmax(scores, 0)):
        for t, j in enumerate(path):
            marginals[t, j] += p
    return torch.logsumexp(scores, 0), marginals


def test_marginals_sum_to_one_and_match_brute_force(torch):
    crf = _crf(torch)
    long_em, short_em = torch.randn(4, len(LABELS)), torch.randn(2, len(LABELS))
    emissions = torch.stack([long_em, torch.cat([short_em, 50 * torch.randn(2, len(LABELS))])])
    mask = torch.tensor([[True] * 4, [True, True, False, False]])
    with torch.no_grad():                           # as when the teacher tags the corpus
        got = crf.marginals(emissions, mask)
        log_z = crf.log_partition(emissions, mask)
        for row, em in ((0, long_em), (1, short_em)):
            brute_log_z, brute = _brute_force(torch, crf, em)
            assert log_z[row].item() == pytest.approx(brute_log_z.item(), abs=1e-4)
            assert torch.allclose(got[row, :len(em)], brute, atol=1e-5)
    assert torch.allclose(got.sum(-1), mask.float(), atol=1e-5)    # 1 per word, 0 on padding
    assert torch.equal(got[1, 2:], torch.zeros(2, len(LABELS)))


def test_marginals_respect_iob2(torch):
    crf = nb.make_crf(LABELS)                       # zero transitions: only the constraints act
    a, b, i_a, i_b = (LABELS.index(t) for t in ('B-A', 'B-B', 'I-A', 'I-B'))
    emissions = torch.zeros(1, 5, len(LABELS))
    emissions[0, :, i_a] = 8.0                      # I-A favoured everywhere ...
    emissions[0, :, i_b] = 6.0
    emissions[0, 1, 0] = 30.0                       # ... but word 1 is O
    emissions[0, 3, a] = 30.0                       # and word 3 is B-A
    m = crf.marginals(emissions, torch.ones(1, 5, dtype=torch.bool))[0]
    assert torch.allclose(m.sum(-1), torch.ones(5), atol=1e-5)
    for t in (0, 2):                                # no I- tag at the start or after O
        assert m[t, i_a] < 1e-6 and m[t, i_b] < 1e-6
    assert m[4, i_b] < 1e-6 and m[4, i_a] > 0.9     # after B-A: I-A, never I-B
    assert m[2, a] + m[2, b] + m[2, 0] == pytest.approx(1.0, abs=1e-5)


def test_weighted_loss_is_the_weighted_sum_of_sentence_nlls(torch):
    crf = _crf(torch, 2)
    paths = [[1, 2, 0, 3, 4], [3, 0, 1], [0, 1, 2, 2]]
    emissions = torch.randn(3, 5, len(LABELS), requires_grad=True)
    tags = torch.tensor([p + [0] * (5 - len(p)) for p in paths])
    mask = torch.tensor([[t < len(p) for t in range(5)] for p in paths])
    weights = torch.tensor([1.0, 0.5, 0.5])                 # one gold and two silver sentences
    per_sentence = [crf.nll(emissions[i:i + 1, :len(p)], tags[i:i + 1, :len(p)],
                            mask[i:i + 1, :len(p)]) for i, p in enumerate(paths)]
    loss = crf.nll(emissions, tags, mask, weights=weights)
    expected = sum(w * nll for w, nll in zip(weights, per_sentence)) / 3
    assert loss.item() == pytest.approx(expected.item(), abs=1e-5)
    # gold sentences keep weight 1: then the loss is the unweighted mean, bit for bit
    assert torch.equal(crf.nll(emissions, tags, mask, weights=torch.ones(3)),
                       crf.nll(emissions, tags, mask))
    loss.backward()
    grad = emissions.grad.clone()
    emissions.grad = None
    expected.backward()
    assert torch.allclose(emissions.grad, grad, atol=1e-6)


# --- tagging and runs with a tiny random model (torch, transformers, seqeval and scipy)
TOY_WORDS = ['steel', 'alloy', 'tensile', 'strength', 'yield', 'room', 'temperature', 'high']


@pytest.fixture(scope='module')
def tiny_model(tmp_path_factory):
    """A tiny random BERT with a word-piece vocabulary ('strengths' is two subwords)."""
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    path = tmp_path_factory.mktemp('tiny_bert_st')
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', *TOY_WORDS, '##s']
    (path / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    transformers.BertTokenizerFast(str(path / 'vocab.txt')).save_pretrained(path)
    torch.manual_seed(0)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64)).save_pretrained(path)
    return str(path)


def test_long_sentences_are_skipped_and_words_without_subwords_dropped(tiny_model):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_model)
    sample = [['steel', '\u0301', 'alloy'],           # a lone combining mark has no subword
              ['high'] * 62,                          # 64 subwords with [CLS] and [SEP]: kept
              ['high'] * 63,                          # 65: skipped, not cut
              ['\ufffd']]                             # no word left: skipped
    kept, skipped = nb.fit_sample(tokenizer, sample, 64)
    assert kept == [['steel', 'alloy'], ['high'] * 62] and skipped == 2


def test_teacher_tags_and_silver_selection(tiny_model):
    torch = pytest.importorskip('torch')
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_model)
    label_list = ['O', 'B-MAT', 'I-MAT', 'B-PROP', 'I-PROP']
    vocab = tokenizer.get_vocab()
    table = torch.zeros(len(vocab), len(label_list))       # emissions per input id
    for word, tag, value in (('steel', 'B-MAT', 10), ('alloy', 'B-MAT', 10), ('high', 'O', 10),
                             ('[UNK]', 'O', 10), ('tensile', 'B-PROP', 10),
                             ('strength', 'I-PROP', 10), ('room', 'O', 5), ('room', 'B-MAT', 5)):
        table[vocab[word], label_list.index(tag)] = value

    class TableModel(torch.nn.Module):
        """A stand-in for a trained tagger: the emissions of each input id from `table`."""
        config = SimpleNamespace(max_position_embeddings=64)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=table[input_ids])

    tagger = SimpleNamespace(model=TableModel(), crf=nb.make_crf(label_list), tokenizer=tokenizer,
                             label2id={t: i for i, t in enumerate(label_list)},
                             id2label=dict(enumerate(label_list)))
    sentences = [['steel', 'high'], ['high', 'high', '.'], ['steel', 'room'],
                 ['tensile', 'strengths', '.']]
    tagged = nb.teacher_tags(tagger, sentences, 3, 'cpu')
    assert [tags for tags, _ in tagged] == [['B-MAT', 'O'], ['O', 'O', 'O'], ['B-MAT', 'O'],
                                            ['B-PROP', 'I-PROP', 'O']]
    marginals = [m for _, m in tagged]
    assert all(m > 0.999 for m in marginals[0] + marginals[1] + marginals[3])
    assert marginals[2][1] == pytest.approx(0.4966, abs=1e-3)   # room: O or B-MAT, a toss-up
    # silver: an entity, every tag confident; not the all-O sentence nor the uncertain one
    assert [nb.is_silver(tags, m) for tags, m in tagged] == [True, False, False, True]


@pytest.fixture
def tiny_run(tiny_model, tmp_path, monkeypatch):
    """run(*flags) on the toy corpus (MAT and PROP kept) in tmp_path/ner; returns the calls.

    Every train_one call gives {'seed', 'silver', 'augmented', 'teacher', 'result',
    'weights'}: weights are those of its CRF losses (None: unweighted), one list per batch.
    """
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: _toy_ner_corpus())
    calls, losses = [], []
    train_one, make_crf = nb.train_one, nb.make_crf

    def recording_crf(label_list):
        crf = make_crf(label_list)
        nll = crf.nll

        def spy(emissions, tags, mask, weights=None):        # the training loss only
            losses.append(None if weights is None else weights.tolist())
            return nll(emissions, tags, mask, weights=weights)
        crf.nll = spy
        return crf

    def recording(seed, *args, **kwargs):
        first = len(losses)
        out = train_one(seed, *args, **kwargs)
        calls.append({'seed': seed, 'silver': kwargs.get('silver'),
                      'augmented': kwargs.get('augmented'),
                      'teacher': kwargs.get('keep_tagger', False),
                      'result': out[0] if kwargs.get('keep_tagger') else out,
                      'weights': losses[first:]})
        return out
    monkeypatch.setattr(nb, 'make_crf', recording_crf)
    monkeypatch.setattr(nb, 'train_one', recording)

    def run(*flags):
        nb.run(nb.parse_args(['--granularity', 'fine', '--top-k', '2', '--model', tiny_model,
                              '--epochs', '2', '--threads', '1', '--device', 'cpu', *flags]))
        return calls
    return run


def _load(tmp_path, name):
    return json.loads((tmp_path / 'ner' / name).read_text())


RESULT_KEYS = ('f1', 'precision', 'recall', 'f1_strict', 'per_class', 'seed', 'best_epoch',
               'val_f1', 'n_train', 'n_val', 'n_test', 'history')


def test_self_training_run(tiny_run, tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(nb, 'is_silver', lambda tags, marginals: True)   # all tagged sentences
    flags = ('--vary-split', '--seeds', '0', '--tag', 'm100', '--seed-offset', '100')
    calls = tiny_run('--crf', '--self-train', str(corpus), *flags, '--save-predictions')
    res = _load(tmp_path, 'ner_results_fine_random_split_vary_crf_st_m100.json')
    assert res['variant'] == 'crf_st_m100' and res['config']['self_train'] == str(corpus)
    assert res['config']['self_train_sha256'] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    run = res['runs'][0]
    assert set(run) == {*RESULT_KEYS, 'run_id', 'runtime_sec', 'self_training', 'test_index',
                        'test_gold', 'test_pred'}
    # the teacher: a plain CRF run, without predictions; then the student, with the same seed
    teacher, student = calls
    assert (teacher['seed'], teacher['teacher'], teacher['silver']) == (100, True, None)
    assert (student['seed'], student['teacher']) == (100, False)
    assert 'test_pred' not in teacher['result']
    assert {k: run[k] for k in RESULT_KEYS} == {k: student['result'][k] for k in RESULT_KEYS}
    # 6 sentences sampled, the long one skipped; the others tagged and (here) all silver
    assert run['self_training'] == {'n_sampled': 6, 'n_skipped': 1, 'n_silver': 5,
                                    'marginal_threshold': 0.9,
                                    'teacher_val_f1': teacher['result']['val_f1'],
                                    'teacher_test_f1': teacher['result']['f1']}
    words = [nb.corpus_words(line) for line in CORPUS_LINES[:6] if line]
    assert [w for w, _ in student['silver']] == words[:5]
    assert all(len(w) == len(t) and set(t) <= {'O', 'B-MAT', 'I-MAT', 'B-PROP', 'I-PROP'}
               for w, t in student['silver'])
    # the teacher's loss is unweighted; the student's weighs each silver sentence 0.5
    n_gold = run['n_train']
    assert all(w is None for w in teacher['weights'])
    for epoch in range(2):                       # 16 gold + 5 silver sentences: 2 batches of 16
        batches = student['weights'][epoch * 2:(epoch + 1) * 2]
        assert sorted(sum(batches, [])) == [0.5] * 5 + [1.0] * n_gold
    assert len(student['weights']) == 4
    # the gold data, splits and validation are those of the CRF run
    assert run['n_train'] + run['n_val'] + run['n_test'] == 24
    assert run['test_index'] == nb.random_split(24, 0)[2]

    # the teacher equals a plain CRF run with the same seed
    tiny_run('--crf', *flags)
    plain = _load(tmp_path, 'ner_results_fine_random_split_vary_crf_m100.json')['runs'][0]
    assert {k: plain[k] for k in RESULT_KEYS} == {k: teacher['result'][k] for k in RESULT_KEYS}
    # finished runs are skipped
    assert len(tiny_run('--crf', '--self-train', str(corpus), *flags, '--save-predictions')) == 3


def test_without_silver_the_student_is_the_teacher(tiny_run, tmp_path, monkeypatch):
    monkeypatch.setattr(nb, 'is_silver', lambda tags, marginals: False)
    calls = tiny_run('--crf', '--self-train', str(_corpus(tmp_path)), '--leave-one-paper-out',
                     '--seeds', '0', '--max-train', '10')
    res = _load(tmp_path, 'ner_results_fine_leave_one_paper_out_crf_st_smoke10.json')
    assert [r['run_id'] for r in res['runs']] == ['a.conll', 'b.conll', 'c.conll', 'd.conll']
    assert len(calls) == 8
    for run, teacher, student in zip(res['runs'], calls[::2], calls[1::2]):
        assert (teacher['teacher'], student['silver']) == (True, [])
        assert {k: teacher['result'][k] for k in RESULT_KEYS} == {k: run[k] for k in RESULT_KEYS}
        assert (run['self_training']['n_silver'], run['self_training']['n_skipped']) == (0, 1)
        assert run['n_train'] == 10 and run['seed'] == 0


def test_self_training_with_three_teachers(tiny_run, tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(nb, 'unanimous_silver', lambda votes: True)   # every tagged sentence
    flags = ('--vary-split', '--seeds', '0')
    st3 = ('--crf', '--self-train', str(corpus), '--teachers', '3', '--self-train-sample', '0',
           *flags, '--save-predictions')
    record = tiny_run(*st3)
    calls = list(record)
    # three teachers (keep_tagger), then a student per teacher seed with the same silver data
    assert [(c['seed'], c['teacher']) for c in calls] == [
        (0, True), (100, True), (200, True), (0, False), (100, False), (200, False)]
    teachers, students = calls[:3], calls[3:]
    assert all(c['silver'] is None for c in teachers)
    words = [nb.corpus_words(line) for line in CORPUS_LINES[:6] if line]
    assert [w for w, _ in students[0]['silver']] == words[:5]          # the long one skipped
    assert students[1]['silver'] == students[2]['silver'] == students[0]['silver']
    for offset, student in zip((0, 100, 200), students):
        res = _load(tmp_path, f'ner_results_fine_random_split_vary_crf_st3_m{offset}.json')
        assert res['variant'] == f'crf_st3_m{offset}'
        config = res['config']
        assert (config['tag'], config['seed_offset'], config['teachers']) == (f'm{offset}', offset, 3)
        assert config['self_train_sample_size'] == 6 and config['self_train'] == str(corpus)
        run = res['runs'][0]
        assert {k: run[k] for k in RESULT_KEYS} == {k: student['result'][k] for k in RESULT_KEYS}
        info = run['self_training']
        assert (info['n_sampled'], info['n_skipped'], info['n_silver']) == (6, 1, 5)
        assert info['marginal_threshold'] == 0.9
        assert [t['seed'] for t in info['teachers']] == [0, 100, 200]
        assert [t['test_f1'] for t in info['teachers']] == [c['result']['f1'] for c in teachers]
        assert [t['val_f1'] for t in info['teachers']] == [c['result']['val_f1'] for c in teachers]
        assert 'test_pred' in run and run['seed'] == offset
    # each teacher is the plain CRF run of its seed
    for offset, teacher in zip((0, 100, 200), teachers):
        tiny_run('--crf', *flags, '--tag', f'm{offset}', '--seed-offset', str(offset))
        plain = _load(tmp_path, f'ner_results_fine_random_split_vary_crf_m{offset}.json')
        assert {k: plain['runs'][0][k] for k in RESULT_KEYS} == \
            {k: teacher['result'][k] for k in RESULT_KEYS}
    # finished runs are skipped
    n = len(record)
    assert len(tiny_run(*st3)) == n


def test_more_students_than_teachers(tiny_run, tmp_path, monkeypatch):
    """--students 4 with 3 teachers (week 5): the teachers and silver data of the 3-teacher
    run, a student for every seed offset 0 to 300, and the files of the students that a
    run already has left as they are."""
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(nb, 'unanimous_silver', lambda votes: True)   # every tagged sentence
    flags = ('--vary-split', '--seeds', '0')
    st3 = ('--crf', '--self-train', str(corpus), '--teachers', '3', '--self-train-sample', '0',
           *flags, '--save-predictions')
    name = 'ner_results_fine_random_split_vary_crf_st3_m{}.json'
    record = tiny_run(*st3)                      # the week-4 run: 3 teachers, 3 students
    first = list(record)
    assert len(first) == 6
    stored = {off: (tmp_path / 'ner' / name.format(off)).read_bytes() for off in (0, 100, 200)}
    # 4 students: the teachers again, and only the student that the files lack
    tiny_run(*st3, '--students', '4')
    calls = record[6:]
    assert [(c['seed'], c['teacher']) for c in calls] == [
        (0, True), (100, True), (200, True), (300, False)]
    for again, before in zip(calls[:3], first[:3]):      # the teachers repeat exactly
        assert {k: again['result'][k] for k in RESULT_KEYS} == \
            {k: before['result'][k] for k in RESULT_KEYS}
    assert calls[3]['silver'] == first[3]['silver']       # the same silver data
    for off, data in stored.items():                     # the week-4 files are untouched
        assert (tmp_path / 'ner' / name.format(off)).read_bytes() == data
    res = _load(tmp_path, name.format(300))
    assert res['variant'] == 'crf_st3_m300'
    assert (res['config']['seed_offset'], res['config']['teachers'],
            res['config']['students']) == (300, 3, 4)
    run = res['runs'][0]
    assert run['seed'] == 300 and run['run_id'] == 'seed0' and 'test_pred' in run
    assert {k: run[k] for k in RESULT_KEYS} == {k: calls[3]['result'][k] for k in RESULT_KEYS}
    info = run['self_training']
    assert [t['seed'] for t in info['teachers']] == [0, 100, 200]
    assert info['teachers'] == _load(tmp_path, name.format(0))['runs'][0]['self_training'][
        'teachers']
    # a student of seed 300 trained with 4 students from the start is the same student
    fresh = tmp_path / 'fresh'
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(fresh))
    tiny_run(*st3, '--students', '4')
    again = json.loads((fresh / 'ner' / name.format(300)).read_text())['runs'][0]
    assert {k: again[k] for k in RESULT_KEYS} == {k: run[k] for k in RESULT_KEYS}
    assert len(record) == 6 + 4 + 7
    # finished runs are skipped
    n = len(record)
    assert len(tiny_run(*st3, '--students', '4')) == n


# --- mention replacement with silver names (--augment-names; ROADMAP §4, week 11)

def test_the_names_join_the_pool_after_the_training_mentions():
    """names extend the pool of their type; without them the copies are those of week 7."""
    sentences = [(['steel', 'is', 'hard'], ['B-MAT', 'O', 'O']),
                 (['alloy', 'X', 'yields'], ['B-MAT', 'O', 'O'])]
    assert nb.mention_replacements(sentences, 3, 7) == \
        nb.mention_replacements(sentences, 3, 7, names=())
    names = [('MAT', ('Ti-6Al-4V',)), ('MAT', ('Inconel', '718')), ('PROP', ('hardness',))]
    copies = nb.mention_replacements(sentences, 20, 0, p=1.0, names=names)
    drawn = {tuple(w[:len(w) - 2]) for w, _ in copies}
    assert drawn == {('steel',), ('alloy',), ('Ti-6Al-4V',), ('Inconel', '718')}
    for words, tags in copies:                       # a two-word name is tagged B, I
        assert tags == ['B-MAT'] + ['I-MAT'] * (len(words) - 3) + ['O', 'O']
    # a type with names but no training mention is never copied into a sentence
    assert all('hardness' not in w for w, _ in copies)


def test_silver_mentions_and_unused_sentences():
    silver = [(['Ti-6Al-4V', 'at', '300', 'K'], ['B-MAT', 'O', 'B-COND', 'I-COND']),
              (['the', 'alloy'], ['O', 'B-MAT'])]
    assert nb.silver_mentions(silver) == [('MAT', ('Ti-6Al-4V',)), ('COND', ('300', 'K')),
                                          ('MAT', ('alloy',))]
    data = [([f'w{i}'], ['O'], 'd') for i in range(6)]
    assert nb.unused_sentences(data, ([4, 0], [2], [5])) == [['w1'], ['w3']]
    train = [(['steel', 'and', 'Ti-6Al-4V'], ['B-MAT', 'O', 'B-MAT'], 'a')]
    test = [(['Ti-6Al-4V', ',', 'alloy', 'and', 'Inconel'], ['B-MAT', 'O', 'B-MAT', 'O', 'B-MAT'],
             'b'), (['alloy'], ['B-PROP'], 'b')]
    names = [('MAT', ('alloy',)), ('MAT', ('alloy',)), ('PROP', ('Inconel',))]
    assert nb.name_coverage(names, train, test) == {
        'n_mentions': 3, 'n_distinct': 2, 'n_test_entities': 4,
        'n_test_unseen': 3,                      # alloy (MAT), Inconel (MAT), alloy (PROP)
        'n_test_unseen_in_silver': 1}            # only alloy as a MAT


def test_augment_names_parser_checks(tmp_path, capsys):
    corpus = str(_corpus(tmp_path))
    base = ['--leave-one-paper-out', '--crf', '--self-train', corpus, '--teachers', '3']
    args = nb.parse_args([*base, '--augment-mentions', '3', '--augment-names'])
    assert nb.variant_name(SimpleNamespace(**{**vars(args), 'tag': 'm100'})) == 'crf_mrs_m100'
    for argv, error in (
            ([*base, '--augment-names'], '--augment-names needs --augment-mentions'),
            (['--leave-one-paper-out', '--crf', '--augment-mentions', '3', '--augment-names'],
             '--augment-names needs --augment-mentions and --self-train'),
            ([*base, '--augment-mentions', '3'], '--augment-mentions is not available with '
                                                '--self-train'),
            (['--crf', '--multi-task', '--self-train', corpus, '--teachers', '3',
              '--augment-mentions', '3', '--augment-names'], 'not available with --multi-task'),
            (['--leave-one-paper-out', '--crf', '--self-train', 'unused', '--teachers', '3',
              '--augment-mentions', '3', '--augment-names'], 'needs --teachers and --train-pool'),
            (['--granularity', 'mulms', '--leave-one-subfield-out', '--train-pool', '400',
              '--crf', '--self-train', 'unused', '--teachers', '3', '--self-train-sample', '0',
              '--augment-mentions', '3', '--augment-names'], 'no --self-train-sample'),
            (['--granularity', 'mulms', '--leave-one-subfield-out', '--train-pool', '400',
              '--crf', '--self-train', corpus, '--teachers', '3'],
             '--self-train is not available with --granularity mulms'),
            (['--leave-one-paper-out', '--crf', '--subfields', 'steel'],
             '--subfields needs --leave-one-subfield-out')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err
    mulms = nb.parse_args(['--granularity', 'mulms', '--leave-one-subfield-out', '--train-pool',
                           '400', '--crf', '--self-train', 'unused', '--teachers', '3',
                           '--students', '10', '--augment-mentions', '3', '--augment-names',
                           '--subfields', 'steel', 'sofc'])
    assert nb.protocol_name(mulms) == 'leave_one_subfield_out'
    assert 'subfields' not in nb.stored_config(mulms)


def test_students_train_on_copies_with_the_silver_names(tiny_run, tmp_path, monkeypatch):
    """--augment-names: the teachers and silver data of --teachers 3, then each student on
    the gold data and its copies (with its own seed), whose pool holds the silver mentions;
    no silver sentence is trained on. The teachers are the plain CRF runs of their seeds."""
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(nb, 'unanimous_silver', lambda votes: True)   # every tagged sentence
    seen = []                                    # (silver sentences, their mentions)
    silver_mentions = nb.silver_mentions

    def recording_mentions(silver):
        seen.append((silver, silver_mentions(silver)))
        return seen[-1][1]
    monkeypatch.setattr(nb, 'silver_mentions', recording_mentions)
    flags = ('--vary-split', '--seeds', '0')
    mrs = ('--crf', '--self-train', str(corpus), '--teachers', '3', '--self-train-sample', '0',
           '--augment-mentions', '2', '--augment-names', *flags, '--save-predictions')
    calls = list(tiny_run(*mrs))
    [(silver, names)] = seen                     # once per run, for all its students
    words = [nb.corpus_words(line) for line in CORPUS_LINES[:6] if line]
    assert [w for w, _ in silver] == words[:5]   # the long one skipped
    assert names                                 # the random teachers tag a few entities
    assert [(c['seed'], c['teacher']) for c in calls] == [
        (0, True), (100, True), (200, True), (0, False), (100, False), (200, False)]
    teachers, students = calls[:3], calls[3:]
    assert all(c['silver'] is None and c['augmented'] is None for c in teachers)
    assert all(c['silver'] is None for c in students)
    data = nb.granularity_data('fine', nb.parse_args(['--granularity', 'fine', '--top-k', '2']))[0]
    tr, _, te = nb.random_split(len(data), 0)
    train = [(data[i][0], data[i][1]) for i in tr]
    for offset, student in zip((0, 100, 200), students):
        res = _load(tmp_path, f'ner_results_fine_random_split_vary_crf_mrs_m{offset}.json')
        assert res['variant'] == f'crf_mrs_m{offset}'
        assert (res['config']['augment_mentions'], res['config']['augment_names']) == (2, True)
        run = res['runs'][0]
        assert {k: run[k] for k in RESULT_KEYS} == {k: student['result'][k] for k in RESULT_KEYS}
        info = run['self_training']
        assert info['n_silver'] == 5 and [t['seed'] for t in info['teachers']] == [0, 100, 200]
        # the student's copies: mention replacement with its seed, the silver names joining
        # the pool of their types
        assert student['augmented'] == nb.mention_replacements(train, 2, offset, names=names)
        assert run['n_augmented'] == len(student['augmented'])
        assert info['names'] == nb.name_coverage(names, [data[i] for i in tr],
                                                 [data[i] for i in te])
        assert info['names']['n_mentions'] == len(names)
    # each teacher is the plain CRF run of its seed
    for offset, teacher in zip((0, 100, 200), teachers):
        tiny_run('--crf', *flags, '--tag', f'm{offset}', '--seed-offset', str(offset))
        plain = _load(tmp_path, f'ner_results_fine_random_split_vary_crf_m{offset}.json')
        assert {k: plain['runs'][0][k] for k in RESULT_KEYS} == \
            {k: teacher['result'][k] for k in RESULT_KEYS}


def test_without_silver_names_a_student_is_the_mention_replacement_run(tiny_run, tmp_path,
                                                                        monkeypatch):
    """With no silver sentence the pool is the training mentions', so each student equals
    the crf_mr run of its seed: the two variants differ only in the pool."""
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(nb, 'unanimous_silver', lambda votes: False)  # no silver sentence
    flags = ('--vary-split', '--seeds', '0')
    calls = list(tiny_run('--crf', '--self-train', str(corpus), '--teachers', '2',
                          '--self-train-sample', '0', '--augment-mentions', '2',
                          '--augment-names', *flags, '--save-predictions'))
    students = calls[2:]
    for offset, student in zip((0, 100), students):
        mrs = _load(tmp_path, f'ner_results_fine_random_split_vary_crf_mrs_m{offset}.json')
        assert mrs['runs'][0]['self_training']['names']['n_mentions'] == 0
        tiny_run('--crf', '--augment-mentions', '2', *flags, '--tag', f'm{offset}',
                 '--seed-offset', str(offset), '--save-predictions')
        mr = _load(tmp_path, f'ner_results_fine_random_split_vary_crf_mr_m{offset}.json')
        keys = (*RESULT_KEYS, 'test_pred', 'n_augmented')
        assert {k: mr['runs'][0][k] for k in keys} == {k: mrs['runs'][0][k] for k in keys}


# --- silver mentions and the names tag (week 12)

def test_silver_mention_names_need_every_teacher_confident_on_the_span():
    """A span counts when every teacher tags it as the same entity with every tag at 0.9 or
    more, whatever the rest of the sentence; a teacher with another span or a low marginal
    drops it."""
    sentences = [['Ti', 'alloy', 'at', '300', 'K'], ['steel', 'bar'], ['Ni', 'foam']]
    sure = (0.95, 0.99, 0.5, 0.97, 0.96)                    # 'at' uncertain: no silver sentence
    votes = [[(('B-MAT', 'I-MAT', 'O', 'B-COND', 'I-COND'), 0.5, sure)] * 3,
             # the third teacher ends the entity after 'steel'
             [(('B-MAT', 'I-MAT'), 0.95, (0.95, 0.95)), (('B-MAT', 'I-MAT'), 0.95, (0.95, 0.95)),
              (('B-MAT', 'O'), 0.95, (0.95, 0.95))],
             # the second teacher is not sure of 'foam'
             [(('B-MAT', 'I-MAT'), 0.9, (0.99, 0.9)), (('B-MAT', 'I-MAT'), 0.8, (0.99, 0.8)),
              (('B-MAT', 'I-MAT'), 0.9, (0.99, 0.9))]]
    assert nb.silver_mention_names(sentences, votes) == [('MAT', ('Ti', 'alloy')),
                                                         ('COND', ('300', 'K'))]
    # at a lower threshold the foam counts
    assert nb.silver_mention_names(sentences, votes, threshold=0.8)[-1] == ('MAT', ('Ni', 'foam'))


def test_silver_mentions_parser_and_variant(tmp_path, capsys):
    corpus = str(_corpus(tmp_path))
    base = ['--leave-one-paper-out', '--crf', '--self-train', corpus, '--teachers', '3',
            '--augment-mentions', '3']
    for argv, error in ((['--leave-one-paper-out', '--crf', '--silver-mentions'],
                         '--silver-mentions needs --augment-names'),
                        (['--leave-one-paper-out', '--crf', '--names-tag', 'topic'],
                         '--names-tag needs --augment-names'),
                        ([*base, '--augment-names', '--names-tag', 'Topic!'], 'invalid')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err
    args = nb.parse_args([*base, '--augment-names', '--silver-mentions', '--names-tag', 'topic'])
    assert nb.variant_name(SimpleNamespace(**{**vars(args), 'tag': 'm100'})) == 'crf_mrm_topic_m100'
    args = nb.parse_args([*base, '--augment-names', '--names-tag', 'fatigue'])
    assert nb.variant_name(SimpleNamespace(**{**vars(args), 'tag': 'm0'})) == 'crf_mrs_fatigue_m0'
    assert nb.spaced_corpus(SimpleNamespace(granularity='mulms'))
    assert not nb.spaced_corpus(SimpleNamespace(granularity='fine'))


def test_students_with_silver_mentions(tiny_run, tmp_path, monkeypatch):
    """--silver-mentions: the names are the silver mentions, which include every mention of a
    silver sentence; the record says which filter was used and the variant carries the names
    tag."""
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(nb, 'SILVER_THRESHOLD', 0.0)     # every agreed tag counts
    seen = {}
    for name in ('silver_mentions', 'silver_mention_names'):
        original = getattr(nb, name)

        def recording(*args, _name=name, _original=original, **kwargs):
            seen[_name] = _original(*args, **kwargs)
            return seen[_name]
        monkeypatch.setattr(nb, name, recording)
    flags = ('--vary-split', '--seeds', '0')
    common = ('--crf', '--self-train', str(corpus), '--teachers', '2', '--self-train-sample',
              '0', '--augment-mentions', '2', '--augment-names', *flags, '--save-predictions')
    calls = list(tiny_run(*common, '--silver-mentions', '--names-tag', 'toy'))
    assert 'silver_mentions' not in seen and 'silver_mention_names' in seen
    names = seen['silver_mention_names']
    for offset, student in zip((0, 100), calls[2:]):
        res = _load(tmp_path, f'ner_results_fine_random_split_vary_crf_mrm_toy_m{offset}.json')
        assert res['variant'] == f'crf_mrm_toy_m{offset}'
        assert (res['config']['silver_mentions'], res['config']['names_tag']) == (True, 'toy')
        info = res['runs'][0]['self_training']
        assert info['silver_filter'] == 'mentions' and info['names']['n_mentions'] == len(names)
        assert student['augmented'] is not None and student['silver'] is None
    # the sentence-level names (sentences whose every tag the teachers agree on) are among
    # the mention-level ones (spans they agree on)
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path / 'sentences'))
    calls = list(tiny_run(*common))
    assert set(seen['silver_mentions']) <= set(names)
    assert len(seen['silver_mentions']) <= len(names)
    res = _load(tmp_path / 'sentences', 'ner_results_fine_random_split_vary_crf_mrs_m0.json')
    assert res['runs'][0]['self_training']['silver_filter'] == 'sentences'
