"""Entity bias in ner_baseline.py (--entity-bias; recall calibration) and the inner folds
that choose it (--leave-two-papers-out). The folds and the parser need no torch; the decode
needs torch, the counts seqeval, and the runs with a tiny random model torch, transformers,
seqeval and scipy (skipped without)."""
import itertools
import json
import random
from types import SimpleNamespace

import pytest

import ner_baseline as nb

# four papers: a has 4 sentences, b 3, c 2, d 3 (in corpus order)
DATA = [([f'{doc}{k}w{j}' for j in range(3)], ['O'] * 3, f'{doc}.conll')
        for doc, n in (('a', 4), ('b', 3), ('c', 2), ('d', 3)) for k in range(n)]


def _old_lopo_folds(data, split_seed, indices):
    """The leave-one-paper-out folds as make_folds built them before paper_fold."""
    folds = []
    for doc in sorted({d for _, _, d in data}):
        rest = [i for i, x in enumerate(data) if x[2] != doc]
        order = list(range(len(rest)))
        random.Random(split_seed).shuffle(order)
        n_val = round(0.15 * len(rest))
        val = [rest[i] for i in order[:n_val]]
        train = [rest[i] for i in order[n_val:]]
        test = [i for i, x in enumerate(data) if x[2] == doc]
        if not indices:
            train, val, test = ([data[i] for i in part] for part in (train, val, test))
        folds.append((doc, train, val, test))
    return folds


def test_leave_one_paper_out_folds_do_not_change():
    data = [(w, t, d) for k in range(5) for w, t, d in DATA]       # 60 sentences
    for split_seed, indices in itertools.product((0, 1, 7), (False, True)):
        args = SimpleNamespace(leave_one_paper_out=True, split_seed=split_seed)
        assert nb.make_folds(data, args, indices) == _old_lopo_folds(data, split_seed, indices)


def test_two_papers_out_folds():
    data = [(w, t, d) for k in range(5) for w, t, d in DATA]
    args = SimpleNamespace(leave_one_paper_out=False, leave_two_papers_out=True, split_seed=0)
    folds = nb.make_folds(data, args, indices=True)
    names = [name for name, *_ in folds]
    assert names == ['a.conll+b.conll', 'a.conll+c.conll', 'a.conll+d.conll',
                     'b.conll+c.conll', 'b.conll+d.conll', 'c.conll+d.conll']
    for name, train, val, test in folds:
        out = set(name.split('+'))
        assert {data[i][2] for i in test} == out and test == sorted(test)
        assert {data[i][2] for i in train + val} == {'a.conll', 'b.conll', 'c.conll',
                                                    'd.conll'} - out
        assert len(val) == round(0.15 * len(train + val)) and not set(train) & set(val)
    # the model of a pair of training papers is the same for both papers it is tested on:
    # its training and validation sentences depend only on the two training papers
    lopo = SimpleNamespace(leave_one_paper_out=True, split_seed=0)
    three = [x for x in data if x[2] != 'a.conll']        # a dropped: b, c, d
    inner = {name: (train, val) for name, train, val, _ in nb.make_folds(three, lopo)}
    pairs = {name: (train, val) for name, train, val, _ in nb.make_folds(data, args)}
    assert inner['b.conll'] == pairs['a.conll+b.conll']


def test_parser_and_names(capsys):
    args = nb.parse_args(['--crf', '--entity-bias', '1', '0.5', '-0', '0', '--leave-one-paper-out',
                          '--tag', 'm100', '--seed-offset', '100'])
    assert nb.entity_biases(args) == [0.0, 0.5, 1.0]
    assert [nb.bias_key(b) for b in (0.0, 0.5, 1.0, -1.0, 2.25)] == ['0', '0.5', '1', '-1', '2.25']
    assert nb.variant_name(args) == 'crf_bias_m100' and nb.reference_variant('crf_bias_m100') == 'crf'
    assert nb.variant_name(nb.parse_args(['--crf', '--context', '8', '--entity-bias', '1'])) == \
        'crf_ctx_bias'
    assert nb.variant_name(nb.parse_args(['--crf', '--multi-task', '--entity-bias', '1'])) == \
        'crf_mt_bias'
    two = nb.parse_args(['--crf', '--leave-two-papers-out', '--entity-bias', '1'])
    assert nb.protocol_name(two) == 'leave_two_papers_out' and nb.held_out_protocol(two)
    assert nb.parse_result_name('ner_results_fine_leave_two_papers_out_crf_bias_m0.json') == \
        ('fine', 'leave_two_papers_out', 'crf_bias_m0')
    for argv, error in ((['--entity-bias', '1'], 'needs --crf'),
                        (['--crf', '--entity-bias', '1', '--self-train', __file__],
                         'not available with --self-train'),
                        (['--crf', '--leave-two-papers-out', '--vary-split'], 'protocol of its own'),
                        (['--crf', '--leave-two-papers-out', '--leave-one-paper-out'],
                         'protocol of its own'),
                        (['--crf', '--leave-two-papers-out', '--published-split'],
                         'protocol of its own'),
                        (['--crf', '--leave-two-papers-out', '--papers', 'a'],
                         'needs --leave-one-paper-out'),
                        (['--crf', '--tag', 'bias'], 'would overwrite')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err
    # the existing options keep their stored configuration and protocol names
    plain = nb.parse_args(['--crf', '--leave-one-paper-out'])
    assert not {'entity_bias', 'leave_two_papers_out'} & set(vars(plain))
    assert nb.protocol_name(plain) == 'leave_one_paper_out' and nb.entity_biases(plain) == []


def test_entity_counts_give_seqeval_micro_f1():
    pytest.importorskip('seqeval')
    rng = random.Random(0)
    tags = ['O', 'B-A', 'I-A', 'B-B', 'I-B']
    for _ in range(20):
        gold = [[rng.choice(tags) for _ in range(rng.randint(1, 8))] for _ in range(5)]
        pred = [[rng.choice(tags) for _ in g] for g in gold]
        correct, n_pred, n_gold = nb.entity_counts(gold, pred)
        f1 = 200 * correct / (n_pred + n_gold) if n_pred + n_gold else 0.0
        assert round(f1, 2) == nb.score(gold, pred)['f1']


LABELS = ['O', 'B-A', 'I-A', 'B-B', 'I-B']


class _FixedModel:
    """Logits from a fixed table per input id (a token classifier without context)."""

    def __init__(self, table):
        self.table = table

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.table[input_ids])


def _features(rng, n):
    """Encoded sentences: [CLS], words of 1-2 subwords (first one labelled), [SEP]."""
    features = []
    for _ in range(n):
        ids, labels = [2], [-100]
        for _ in range(rng.randint(1, 7)):
            ids.append(rng.randint(5, 39))
            labels.append(rng.randint(0, len(LABELS) - 1))
            if rng.random() < 0.3:
                ids.append(rng.randint(5, 39))
                labels.append(-100)
        ids.append(3)
        labels.append(-100)
        features.append({'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels})
    return features


def test_biased_decodes():
    torch = pytest.importorskip('torch')
    torch.manual_seed(0)
    crf = nb.make_crf(LABELS)
    with torch.no_grad():
        for p in crf.parameters():
            p.normal_()
    model = _FixedModel(3 * torch.randn(40, len(LABELS)))
    features = _features(random.Random(0), 30)
    id2label = dict(enumerate(LABELS))
    grid = [-50.0, -1.0, 0.0, 0.5, 1.0, 2.0, 50.0]
    out = nb.biased_decodes(model, features, 4, 0, id2label, 'cpu', crf, grid)
    assert out[0.0] == nb.predict(model, features, 4, 0, id2label, 'cpu', crf)
    # a path scores its emissions plus the bias once per word not tagged O, so the number of
    # such words never falls as the bias grows (whatever the emissions and transitions)
    for k in range(len(features)):
        entity_words = [sum(t != 'O' for t in out[b][k]) for b in grid]
        assert entity_words == sorted(entity_words)
        assert len(out[0.0][k]) == sum(x != -100 for x in features[k]['labels'])
    assert all(t == 'O' for k in range(len(features)) for t in out[-50.0][k])
    assert all(t != 'O' for k in range(len(features)) for t in out[50.0][k])
    assert any(out[1.0][k] != out[0.0][k] for k in range(len(features)))


def test_bias_fields():
    pytest.importorskip('seqeval')
    test = [(['x', 'y'], ['B-A', 'O'], 'p'), (['z'], ['B-B'], 'q'), (['w', 'v'], ['B-A', 'I-A'], 'p')]
    val = [(['u'], ['B-A'], 'r')]
    preds = {0.0: [['B-A', 'O'], ['O'], ['B-A', 'O']], 1.0: [['B-A', 'B-A'], ['B-B'], ['B-A', 'I-A']]}
    fields = nb.bias_fields([0.0, 1.0], test, preds, val, {0.0: [['O']], 1.0: [['B-A']]})
    assert fields == {'bias_test': {'0': {'p': [1, 2, 2], 'q': [0, 0, 1]},
                                    '1': {'p': [2, 3, 2], 'q': [1, 1, 1]}},
                      'bias_val': {'0': [0, 0, 1], '1': [1, 1, 1]}}
    saved = nb.bias_fields([0.0, 1.0], test, preds, val, {0.0: [['O']], 1.0: [['B-A']]},
                           save_predictions=True)
    assert saved['test_pred_bias'] == {'0': ['B-A O', 'O', 'B-A O'],
                                       '1': ['B-A B-A', 'B-B', 'B-A I-A']}


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
    if granularity == 'coarse':
        return [(w, [t.replace('MAT', 'M').replace('PROP', 'P').replace('COND', 'C')
                     for t in tags], d) for w, tags, d in data]
    return data


@pytest.fixture
def tiny_model(tmp_path, monkeypatch):
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
    return path


def _check_bias_run(run, docs, biases=('-1', '0', '2')):
    assert list(run['bias_test']) == list(biases) and list(run['bias_val']) == list(biases)
    for key in biases:
        assert list(run['bias_test'][key]) == docs
        assert sum(c[2] for c in run['bias_test'][key].values()) == \
            sum(nb.entity_counts([g.split() for g in run['test_gold']], [])[2:])
    # bias 0 is the run's own decode: its counts give the run's micro F1
    correct, n_pred, n_gold = (sum(c[i] for c in run['bias_test']['0'].values()) for i in range(3))
    assert round(200 * correct / (n_pred + n_gold), 2) == run['f1']
    assert run['test_pred_bias']['0'] == run['test_pred']


def test_bias_runs(tiny_model, tmp_path):
    """Tiny --entity-bias runs: leave-one-paper-out (one fold), the inner folds of
    --leave-two-papers-out (all six) and multi-task (both files)."""
    common = ['--top-k', '3', '--model', str(tiny_model), '--epochs', '1', '--threads', '1',
              '--device', 'cpu', '--crf', '--entity-bias', '2', '-1', '--seeds', '0',
              '--save-predictions']
    nb.run(nb.parse_args(['--granularity', 'fine', *common, '--leave-one-paper-out',
                          '--papers', 'b', '--tag', 'm100', '--seed-offset', '100']))
    res = json.loads((tmp_path / 'ner' / 'ner_results_fine_leave_one_paper_out_crf_bias_m100.json')
                     .read_text())
    (run,) = res['runs']
    assert res['variant'] == 'crf_bias_m100' and res['config']['entity_bias'] == [2.0, -1.0]
    assert run['run_id'] == 'b.conll' and run['seed'] == 100
    _check_bias_run(run, ['b.conll'])

    nb.run(nb.parse_args(['--granularity', 'fine', *common, '--leave-two-papers-out']))
    res = json.loads((tmp_path / 'ner' / 'ner_results_fine_leave_two_papers_out_crf_bias.json')
                     .read_text())
    assert res['protocol'] == 'leave_two_papers_out' and len(res['runs']) == 6
    for run in res['runs']:
        _check_bias_run(run, run['run_id'].split('+'))
        assert run['n_test'] == 12

    nb.run(nb.parse_args([*common, '--multi-task', '--leave-one-paper-out', '--papers', 'c']))
    for g in ('fine', 'coarse'):
        res = json.loads((tmp_path / 'ner' / f'ner_results_{g}_leave_one_paper_out_crf_mt_bias.json')
                         .read_text())
        (run,) = res['runs']
        assert res['variant'] == 'crf_mt_bias' and run['run_id'] == 'c.conll'
        _check_bias_run(run, ['c.conll'])
