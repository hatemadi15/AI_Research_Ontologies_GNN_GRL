"""Multi-task NER training in ner_baseline.py (--crf --multi-task).

The example building, the corpus check, the variant names and the parser need
no torch. The loss and batching tests need torch, and the tiny-model runs also
transformers, seqeval and scipy; they are skipped without them.
"""
import json
from types import SimpleNamespace

import pytest

import ner_baseline as nb


def _sent(tokens, tags, doc='p.conll'):
    return tokens.split(), tags.split(), doc


def test_examples_join_identical_sentences_and_split_the_others():
    fine = [_sent('steel is hard', 'B-Steel O B-Hardness'),
            _sent('The T dual role', 'O B-Temp O O'),
            _sent('cracks crack .', 'B-Crack I-Crack O')]
    coarse = [_sent('steel is hard', 'B-Material O B-Property'),
              _sent('The dual role', 'O O O'),
              _sent('cracks .', 'B-Defect O')]
    examples = nb.multitask_examples(fine, coarse)
    assert examples == [
        # same tokens: one example, each head with the tags of its own corpus
        (['steel', 'is', 'hard'], {'fine': ['B-Steel', 'O', 'B-Hardness'],
                                   'coarse': ['B-Material', 'O', 'B-Property']}),
        # other tokens: two examples, each labelled for one head only
        (['The', 'T', 'dual', 'role'], {'fine': ['O', 'B-Temp', 'O', 'O'], 'coarse': None}),
        (['The', 'dual', 'role'], {'fine': None, 'coarse': ['O', 'O', 'O']}),
        (['cracks', 'crack', '.'], {'fine': ['B-Crack', 'I-Crack', 'O'], 'coarse': None}),
        (['cracks', '.'], {'fine': None, 'coarse': ['B-Defect', 'O']})]
    with pytest.raises(ValueError):           # the corpora must hold the same sentences
        nb.multitask_examples(fine, coarse[:2])
    assert nb.multitask_examples([], []) == []


def test_corpora_must_align_by_length_and_paper():
    fine = [_sent('a', 'O', 'x.conll'), _sent('b c', 'O O', 'y.conll')]
    nb.check_aligned(fine, [_sent('a', 'O', 'x.conll'), _sent('b', 'B-Z', 'y.conll')])
    with pytest.raises(AssertionError, match='differ in length: 2 vs 1'):
        nb.check_aligned(fine, fine[:1])
    with pytest.raises(AssertionError, match=r'differ in paper at positions \[1\]'):
        nb.check_aligned(fine, [fine[0], _sent('b c', 'O O', 'z.conll')])


def test_materiominer_corpora_align_with_15_split_sentences():
    fine, coarse = nb.load_split_corpus('fine'), nb.load_split_corpus('coarse')
    nb.check_aligned(fine, coarse)
    examples = nb.multitask_examples(fine, coarse)
    split = [tags for _, tags in examples if None in tags.values()]
    assert (len(fine), len(examples), len(split)) == (476, 491, 30)
    assert sum(tags['fine'] is None for tags in split) == 15


def test_multitask_variant_names_and_files():
    def name(**kw):
        return nb.variant_name(SimpleNamespace(
            **{'class_weights': False, 'crf': True, 'multi_task': True, 'tag': None, **kw}))
    assert name() == 'crf_mt' and name(tag='x1') == 'crf_mt_x1'
    assert name(multi_task=False) == 'crf' and name(multi_task=False, tag='x1') == 'crf_x1'
    # compared with the single-task CRF runs of the same granularity
    assert nb.reference_variant('crf_mt') == nb.reference_variant('crf_mt_x1') == 'crf'
    assert nb.result_suffix('random_split_vary', 'crf_mt') == '_random_split_vary_crf_mt'
    for protocol in ('random_split_fixed', 'random_split_vary', 'leave_one_paper_out'):
        for variant in ('crf_mt', 'crf_mt_x1'):
            for granularity in ('fine', 'coarse'):
                fn = f'ner_results_{granularity}{nb.result_suffix(protocol, variant)}.json'
                assert nb.parse_result_name(fn) == (granularity, protocol, variant)


def test_multitask_parser_checks(capsys):
    args = nb.parse_args(['--crf', '--multi-task'])
    assert args.multi_task and args.granularity == 'both' and nb.variant_name(args) == 'crf_mt'
    args = nb.parse_args(['--crf', '--multi-task', '--tag', 'x1', '--seed-offset', '100',
                          '--save-predictions', '--leave-one-paper-out'])
    assert nb.variant_name(args) == 'crf_mt_x1' and args.seed_offset == 100
    # without the flag there is no multi_task field: the config other runs store is unchanged
    assert 'multi_task' not in vars(nb.parse_args(['--crf']))
    for argv, error in ((['--multi-task'], '--multi-task needs --crf'),
                        (['--multi-task', '--class-weights'], '--multi-task needs --crf'),
                        (['--crf', '--multi-task', '--granularity', 'fine'], 'must be both'),
                        (['--crf', '--multi-task', '--granularity', 'coarse'], 'must be both')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err
    # a single-task CRF run tagged mt would write to the files of the multi-task runs
    with pytest.raises(SystemExit):
        nb.parse_args(['--crf', '--tag', 'mt'])
    assert 'would overwrite the results of the crf_mt variant' in capsys.readouterr().err
    assert nb.variant_name(nb.parse_args(['--crf', '--multi-task', '--tag', 'mt'])) == 'crf_mt_mt'
    assert nb.variant_name(nb.parse_args(['--tag', 'mt'])) == 'mt'


def test_multitask_run_checks_both_files_before_training(tmp_path, monkeypatch):
    corpora = _toy_corpora()
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: corpora[granularity])
    monkeypatch.setattr(nb, 'train_multitask', lambda *args, **kwargs: pytest.fail('trained'))
    (tmp_path / 'ner').mkdir()
    coarse = tmp_path / 'ner' / 'ner_results_coarse_random_split_vary_crf_mt.json'
    coarse.write_text(json.dumps({'labels': ['Condition', 'Material', 'Property'],
                                  'model': nb.DEFAULT_MODEL, 'variant': 'crf_mt',
                                  'config': {'seed_offset': 100}, 'runs': [{'run_id': 'seed0'}]}))
    before = coarse.read_text()
    argv = ['--crf', '--multi-task', '--vary-split', '--seeds', '0', '--top-k', '4']
    with pytest.raises(nb.OverwriteError,
                       match=f'{coarse.name}: .*seed_offset: 100 in the file, 0 now'):
        nb.run(nb.parse_args(argv))
    # nothing written: no fine file, and the coarse file is left as it was
    assert coarse.read_text() == before and sorted(p.name for p in coarse.parent.iterdir()) == [
        coarse.name]


def test_compare_pairs_multitask_files_with_the_crf_files(tmp_path, capsys):
    def run(micro):
        return {'run_id': 'seed0', 'f1': micro, 'f1_strict': micro - 1,
                'per_class': {'macro avg': {'f1-score': (micro - 10) / 100},
                              'weighted avg': {'f1-score': (micro - 2) / 100}}}
    for granularity, crf, mt in (('fine', 70.0, 72.0), ('coarse', 75.0, 74.0)):
        for variant, micro in (('crf', crf), ('crf_mt', mt)):
            path = tmp_path / f'ner_results_{granularity}_random_split_vary_{variant}.json'
            path.write_text(json.dumps({'protocol': 'random_split_vary', 'runs': [run(micro)]}))
    comps = nb.compare_all(str(tmp_path))
    fine, coarse = ('ner_results_fine_random_split_vary_crf_mt',
                    'ner_results_coarse_random_split_vary_crf_mt')
    assert sorted(comps) == [coarse, fine]          # the crf files have no baseline file here
    assert comps[fine]['reference_file'] == 'ner_results_fine_random_split_vary_crf.json'
    assert comps[coarse]['reference_file'] == 'ner_results_coarse_random_split_vary_crf.json'
    assert (comps[fine]['reference'], comps[fine]['variant']) == ('crf', 'crf_mt')
    assert comps[fine]['micro']['per_run'] == [2.0] and comps[coarse]['micro']['per_run'] == [-1.0]
    assert 'no reference file' in capsys.readouterr().out


# --- loss masking and batching (torch)
FINE_LABELS = ['O', 'B-A', 'I-A']
COARSE_LABELS = ['O', 'B-X', 'I-X', 'B-Y', 'I-Y']


class SpyCRF:
    """A WordCRF that records the rows it is given."""

    def __init__(self, crf, calls, head):
        self.crf, self.calls, self.head = crf, calls, head

    def nll(self, emissions, tags, mask):
        self.calls.append((self.head, emissions.size(0), mask[:, 0].all().item(),
                           emissions.isnan().any().item()))
        return self.crf.nll(emissions, tags, mask)


def _crfs(torch):
    torch.manual_seed(0)
    crfs = {'fine': nb.make_crf(FINE_LABELS), 'coarse': nb.make_crf(COARSE_LABELS)}
    with torch.no_grad():
        for crf in crfs.values():
            for p in crf.parameters():
                p.normal_()
    return crfs


def test_loss_keeps_each_head_to_its_labelled_rows():
    torch = pytest.importorskip('torch')
    crfs = _crfs(torch)
    no = [-100] * 6
    # rows 0-1: both heads; row 2: fine only; row 3: coarse only (-100 off first subwords)
    labels = {'fine': torch.tensor([[-100, 1, 2, 0, -100, -100], [-100, 0, -100, 1, -100, -100],
                                    [-100, 1, 0, 0, 1, -100], no]),
              'coarse': torch.tensor([[-100, 1, 2, 3, -100, -100], [-100, 0, -100, 3, -100, -100],
                                      no, [-100, 3, 4, 0, 1, -100]])}
    clean = {'fine': torch.randn(4, 6, 3), 'coarse': torch.randn(4, 6, 5)}
    logits = {h: x.clone() for h, x in clean.items()}
    logits['fine'][3] = float('nan')            # rows a head has no labels for: never used
    logits['coarse'][2] = float('nan')
    for x in logits.values():
        x.requires_grad_()
    calls = []
    spies = {h: SpyCRF(crf, calls, h) for h, crf in crfs.items()}
    losses = nb.multitask_loss(logits, labels, spies)
    # each head's loss is its CRF's mean NLL over its labelled rows only
    for head, rows in (('fine', [0, 1, 2]), ('coarse', [0, 1, 3])):
        expected = crfs[head].nll(*nb.word_view(clean[head][rows], labels[head][rows]))
        assert losses[head].item() == pytest.approx(expected.item(), abs=1e-5)
    assert sorted(calls) == [('coarse', 3, True, False), ('fine', 3, True, False)]
    sum(losses.values()).backward()
    assert torch.equal(logits['fine'].grad[3], torch.zeros(6, 3))
    assert torch.equal(logits['coarse'].grad[2], torch.zeros(6, 5))
    assert logits['fine'].grad[:3].abs().sum() > 0 and logits['coarse'].grad[3].abs().sum() > 0

    # a batch without labels for a head: the head is left out and its CRF never called
    calls.clear()
    only_fine = nb.multitask_loss({h: x[2:3] for h, x in logits.items()},
                                  {h: lab[2:3] for h, lab in labels.items()}, spies)
    assert list(only_fine) == ['fine'] and calls == [('fine', 1, True, False)]
    expected = crfs['fine'].nll(*nb.word_view(clean['fine'][2:3], labels['fine'][2:3]))
    assert only_fine['fine'].item() == pytest.approx(expected.item(), abs=1e-5)


def test_batches_of_the_heads_line_up():
    pytest.importorskip('torch')
    features = {'fine': [], 'coarse': []}
    for i in range(7):                          # lengths 3..9, fine labels missing for 0, 3, 6
        ids = list(range(100 * i, 100 * i + 3 + i))
        mask = [1] * len(ids)
        features['fine'].append({'input_ids': ids, 'attention_mask': mask,
                                 'labels': [-100] * len(ids) if i % 3 == 0 else [i] * len(ids)})
        features['coarse'].append({'input_ids': ids, 'attention_mask': mask,
                                   'labels': [10 + i] * len(ids)})
    got = list(nb.multitask_batches(features, 3, 0, shuffle_seed=5))
    plain = list(nb.batches(features['fine'], 3, 0, shuffle_seed=5))
    assert [b[3] for b in got] == [b[3] for b in plain]          # the shuffle of batches()
    assert sorted(i for b in got for i in b[3]) == list(range(7))
    for ids, mask, labels, idx in got:
        assert set(labels) == {'fine', 'coarse'}
        for row, i in enumerate(idx):
            n = 3 + i
            assert ids[row, 0].item() == 100 * i and mask[row].sum().item() == n
            assert labels['coarse'][row, :n].tolist() == [10 + i] * n
            assert labels['fine'][row, :n].tolist() == [-100 if i % 3 == 0 else i] * n
            assert (labels['coarse'][row, n:] == -100).all() and (labels['fine'][row, n:] == -100).all()


# --- runs with a tiny random model (torch, transformers, seqeval and scipy; skipped without)
TOY_WORDS = ['steel', 'alloy', 'tensile', 'strength', 'yield', 'room', 'temperature', 'high']
RUN_KEYS = {'f1', 'precision', 'recall', 'f1_strict', 'per_class', 'seed', 'best_epoch', 'val_f1',
            'n_train', 'n_val', 'n_test', 'history', 'run_id', 'runtime_sec', 'multi_task'}
PREDICTION_KEYS = {'test_index', 'test_gold', 'test_pred'}
SPLIT_POSITIONS = (2, 9, 16, 23)        # one sentence per paper is tokenised differently


def _toy_corpora():
    """Fine and coarse versions of 24 sentences from four 'papers'.

    Fine classes Steel, Alloy, Strength and Yield occur in every sentence, Temp in
    every other one, so `--top-k 4` drops Temp; they map to the coarse types
    Material, Property and Condition. At SPLIT_POSITIONS the fine sentence has an
    extra word.
    """
    fine, coarse = [], []
    for p, doc in enumerate(('a.conll', 'b.conll', 'c.conll', 'd.conll')):
        for k in range(6):
            pos = 6 * p + k
            temp = k % 2 == 0
            tokens = ['steel', 'alloy', 'tensile', 'strengths', 'yield', 'room', 'temperature']
            f_tags = ['B-Steel', 'B-Alloy', 'B-Strength', 'I-Strength', 'B-Yield'] + (
                ['B-Temp', 'I-Temp'] if temp else ['O', 'O'])
            c_tags = ['B-Material', 'B-Material', 'B-Property', 'I-Property', 'B-Property'] + (
                ['B-Condition', 'I-Condition'] if temp else ['O', 'O'])
            extra = ['high'] * (pos % 5)
            coarse.append((tokens + extra, c_tags + ['O'] * len(extra), doc))
            if pos in SPLIT_POSITIONS:
                extra = extra + ['high']
            fine.append((tokens + extra, f_tags + ['O'] * len(extra), doc))
    return {'fine': fine, 'coarse': coarse}


def _toy_data():
    """The toy corpora as run_multitask sees them: labels selected, other tags O."""
    data = {}
    for g, corpus in _toy_corpora().items():
        labels, _ = nb.select_labels(corpus, 4 if g == 'fine' else None)
        data[g] = (labels, [(t, nb.restrict_tags(tags, set(labels)), d) for t, tags, d in corpus])
    return data


@pytest.fixture(scope='module')
def tiny_model(tmp_path_factory):
    """A tiny random BERT with a word-piece vocabulary ('strengths' is two subwords)."""
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    path = tmp_path_factory.mktemp('tiny_bert_mt')
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', *TOY_WORDS, '##s']
    (path / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    transformers.BertTokenizerFast(str(path / 'vocab.txt')).save_pretrained(path)
    torch.manual_seed(0)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64)).save_pretrained(path)
    return str(path)


@pytest.fixture
def tiny_run(tiny_model, tmp_path, monkeypatch):
    """run(*flags): ner_baseline.run on the toy corpora in tmp_path/ner; returns its calls.

    The calls are the model seeds that train_multitask was called with.
    """
    corpora = _toy_corpora()
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: corpora[granularity])
    calls = []
    train = nb.train_multitask

    def counting(seed, *args, **kwargs):
        calls.append(seed)
        return train(seed, *args, **kwargs)
    monkeypatch.setattr(nb, 'train_multitask', counting)

    def run(*flags):
        nb.run(nb.parse_args(['--top-k', '4', '--model', tiny_model, '--epochs', '2',
                              '--threads', '1', '--device', 'cpu', *flags]))
        return calls
    return run


def _load(tmp_path, name):
    return json.loads((tmp_path / 'ner' / name).read_text())


def _check_pair(fine, coarse, data, index):
    """Checks shared by every pair of fine and coarse records of one run."""
    for g, run in (('fine', fine), ('coarse', coarse)):
        assert set(run) == RUN_KEYS | PREDICTION_KEYS
        corpus = data[g][1]
        assert run['test_index'] == index and run['n_test'] == len(index)
        # gold tags from each corpus at the same positions, one prediction per word of it
        assert run['test_gold'] == [' '.join(corpus[i][1]) for i in index]
        assert [len(p.split()) for p in run['test_pred']] == [len(corpus[i][0]) for i in index]
        assert set(' '.join(run['test_pred']).split()) <= {
            'O', *(f'{b}-{t}' for t in data[g][0] for b in 'BI')}
        # history: the single-task fields (this head's val F1) plus both heads' losses and F1s
        hist = run['history']
        assert [h['epoch'] for h in hist] == list(range(len(hist)))
        for h in hist:
            assert h['val_f1'] == h[f'val_f1_{g}']
            assert h['train_loss'] == pytest.approx(
                h['train_loss_fine'] + h['train_loss_coarse'], abs=2e-4)
        f1s = [h['val_f1'] for h in hist]
        assert run['val_f1'] == max(f1s) and run['best_epoch'] == f1s.index(max(f1s))
    assert fine['history'] == [{**h, 'val_f1': h['val_f1_fine']} for h in coarse['history']]
    for key in ('run_id', 'seed', 'runtime_sec', 'n_train', 'n_val', 'n_test'):
        assert fine[key] == coarse[key]
    assert fine['multi_task']['other_best_epoch'] == coarse['best_epoch']
    assert coarse['multi_task']['other_best_epoch'] == fine['best_epoch']
    assert fine['multi_task']['n_examples'] == coarse['multi_task']['n_examples']


def test_multitask_run_writes_a_fine_and_a_coarse_file(tiny_run, tmp_path):
    calls = tiny_run('--crf', '--multi-task', '--vary-split', '--seeds', '0', '1', '--tag', 't1',
                     '--seed-offset', '100', '--save-predictions')
    assert calls == [100, 101]                                  # one model per run_id
    data = _toy_data()
    assert data['fine'][0] == ['Alloy', 'Steel', 'Strength', 'Yield']
    assert data['coarse'][0] == ['Condition', 'Material', 'Property']
    fine = _load(tmp_path, 'ner_results_fine_random_split_vary_crf_mt_t1.json')
    coarse = _load(tmp_path, 'ner_results_coarse_random_split_vary_crf_mt_t1.json')
    for g, res in (('fine', fine), ('coarse', coarse)):
        assert (res['granularity'], res['protocol'], res['variant']) == (
            g, 'random_split_vary', 'crf_mt_t1')
        assert res['labels'] == data[g][0] and res['paper_test_f1'] == nb.PAPER_F1[g]
        assert res['config']['multi_task'] and res['config']['crf']
        assert res['config']['seed_offset'] == 100 and 'granularity' not in res['config']
        assert res['summary']['run_ids'] == ['seed0', 'seed1']
    for run_f, run_c, split_seed in zip(fine['runs'], coarse['runs'], (0, 1)):
        train, val, test = nb.random_split(24, split_seed)
        _check_pair(run_f, run_c, data, test)
        assert (run_f['run_id'], run_f['seed']) == (f'seed{split_seed}', 100 + split_seed)
        assert (run_f['n_train'], run_f['n_val']) == (len(train), len(val))
        n_split = sum(i in SPLIT_POSITIONS for i in train)
        assert run_f['multi_task']['n_examples'] == len(train) + n_split
        assert run_f['multi_task']['n_split_examples'] == 2 * n_split
    assert any(i in SPLIT_POSITIONS for i in nb.random_split(24, 0)[0])   # the toy splits do split
    assert any(i in SPLIT_POSITIONS for i in nb.random_split(24, 1)[2])


def test_multitask_resume_needs_both_files(tiny_run, tmp_path):
    flags = ('--crf', '--multi-task', '--vary-split', '--seeds', '0', '1', '--save-predictions')
    assert tiny_run(*flags) == [0, 1]
    names = {g: f'ner_results_{g}_random_split_vary_crf_mt.json' for g in ('fine', 'coarse')}
    assert tiny_run(*flags) == [0, 1]                 # both files hold both runs: skipped
    # a run missing from one file is redone, and replaces its record in the other
    coarse = _load(tmp_path, names['coarse'])
    coarse['runs'] = coarse['runs'][:1]
    (tmp_path / 'ner' / names['coarse']).write_text(json.dumps(coarse))
    assert tiny_run(*flags) == [0, 1, 1]
    for g, name in names.items():
        assert [r['run_id'] for r in _load(tmp_path, name)['runs']] == ['seed0', 'seed1']
    # runs that cannot be resumed in either file stop the run before training, unless replaced
    # on purpose; then both files are rewritten
    fine = _load(tmp_path, names['fine'])
    (tmp_path / 'ner' / names['fine']).write_text(json.dumps({**fine, 'model': 'other'}))
    with pytest.raises(nb.OverwriteError, match="model: 'other' in the file"):
        tiny_run(*flags)
    assert tiny_run(*flags, '--overwrite') == [0, 1, 1, 0, 1]
    for g, name in names.items():
        res = _load(tmp_path, name)
        assert res['model'] != 'other' and [r['run_id'] for r in res['runs']] == ['seed0', 'seed1']
        assert 'overwrite' not in res['config']
    # a single-task CRF run neither reads nor writes the multi-task files, and stores no
    # multi_task setting
    before = {g: (tmp_path / 'ner' / n).read_text() for g, n in names.items()}
    tiny_run('--crf', '--granularity', 'coarse', '--vary-split', '--seeds', '0')
    single = _load(tmp_path, 'ner_results_coarse_random_split_vary_crf.json')
    assert 'multi_task' not in single['config'] and 'multi_task' not in single['runs'][0]
    assert {g: (tmp_path / 'ner' / n).read_text() for g, n in names.items()} == before


def test_multitask_leave_one_paper_out(tiny_run, tmp_path):
    assert tiny_run('--crf', '--multi-task', '--leave-one-paper-out', '--seeds', '3',
                    '--max-train', '10', '--save-predictions') == [3] * 4
    data = _toy_data()
    suffix = '_leave_one_paper_out_crf_mt_smoke10.json'
    fine, coarse = (_load(tmp_path, f'ner_results_{g}{suffix}') for g in ('fine', 'coarse'))
    papers = ['a.conll', 'b.conll', 'c.conll', 'd.conll']
    assert [r['run_id'] for r in fine['runs']] == [r['run_id'] for r in coarse['runs']] == papers
    for run_f, run_c in zip(fine['runs'], coarse['runs']):
        index = [i for i, x in enumerate(data['fine'][1]) if x[2] == run_f['run_id']]
        _check_pair(run_f, run_c, data, index)
        assert run_f['n_train'] == 10 and run_f['seed'] == 3


def test_multitask_mention_replacement_run(tiny_run, tmp_path, monkeypatch):
    """--multi-task --augment-mentions: one file per head (_crf_mt_mr), each run storing the
    number of copies, which train with both heads' tags. Every other sentence names its
    Strength 'yield strengths', so that a replacement can change a sentence."""
    corpora = _toy_corpora()
    for g in ('fine', 'coarse'):
        corpora[g] = [(tokens[:2] + ['yield'] + tokens[3:] if pos % 2 else tokens, tags, doc)
                      for pos, (tokens, tags, doc) in enumerate(corpora[g])]
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: corpora[granularity])
    assert tiny_run('--crf', '--multi-task', '--augment-mentions', '2', '--leave-one-paper-out',
                    '--seeds', '3', '--papers', 'b', '--tag', 'm100',
                    '--seed-offset', '100') == [103]
    suffix = '_leave_one_paper_out_crf_mt_mr2_m100.json'     # 2 copies: named after them
    fine, coarse = (_load(tmp_path, f'ner_results_{g}{suffix}') for g in ('fine', 'coarse'))
    assert fine['variant'] == 'crf_mt_mr2_m100' and fine['config']['augment_mentions'] == 2
    (run_f,), (run_c,) = fine['runs'], coarse['runs']
    assert run_f['run_id'] == 'b.conll' and run_f['n_augmented'] == run_c['n_augmented'] > 0
    assert run_f['multi_task']['n_examples'] == run_c['multi_task']['n_examples']


def _toy_sofc():
    """SOFC-Exp's two levels on the toy sentences (the coarse ones, so the levels align): slot
    tags from the fine tags (Temp as interconnect_material, a type the slot level drops) and
    entity tags from the coarse ones; papers a and b train, c is dev, d is test."""
    toy = _toy_corpora()
    slot_of = {'Steel': 'anode_material', 'Alloy': 'cathode_material',
               'Strength': 'power_density', 'Yield': 'resistance',
               'Temp': 'interconnect_material'}
    entity_of = {'Material': 'MATERIAL', 'Property': 'VALUE', 'Condition': 'EXPERIMENT'}

    def rename(tags, names):
        return [t if t == 'O' else f'{t[:2]}{names[t[2:]]}' for t in tags]
    slot = [(c[0], rename(f[1][:len(c[0])], slot_of), c[2])
            for f, c in zip(toy['fine'], toy['coarse'])]
    entity = [(t, rename(tags, entity_of), d) for t, tags, d in toy['coarse']]
    return {'sofc_slot': slot, 'sofc': entity}


def test_multitask_on_the_two_sofc_exp_levels(tiny_run, tmp_path, monkeypatch):
    """--granularity sofc_both --published-split: the slot head ('fine') and the entity head
    ('coarse') of one model per seed, on the official split by article, a file per level."""
    corpora = _toy_sofc()
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: corpora[granularity])
    raw = tmp_path / 'raw'
    (raw / 'sofc_exp').mkdir(parents=True)
    (raw / 'sofc_exp' / 'split.json').write_text(json.dumps(
        {'a': 'train', 'b': 'train', 'c': 'dev', 'd': 'test'}))
    monkeypatch.setattr(nb, 'RAW_DIR', str(raw))
    assert tiny_run('--crf', '--multi-task', '--granularity', 'sofc_both', '--published-split',
                    '--seeds', '0', '1', '--save-predictions') == [0, 1]
    slot = _load(tmp_path, 'ner_results_sofc_slot_published_split_crf_mt.json')
    entity = _load(tmp_path, 'ner_results_sofc_published_split_crf_mt.json')
    labels = {'fine': ['anode_material', 'cathode_material', 'power_density', 'resistance'],
              'coarse': ['EXPERIMENT', 'MATERIAL', 'VALUE']}
    data = {}
    for head, g, res in (('fine', 'sofc_slot', slot), ('coarse', 'sofc', entity)):
        assert (res['granularity'], res['protocol'], res['variant']) == (
            g, 'published_split', 'crf_mt')
        assert res['labels'] == labels[head] and res['paper_test_f1'] is None
        assert res['summary']['run_ids'] == ['seed0', 'seed1']
        data[head] = (labels[head], [(t, nb.restrict_tags(tags, set(labels[head])), d)
                                     for t, tags, d in corpora[g]])
    test = [i for i, x in enumerate(corpora['sofc']) if x[2] == 'd.conll']
    for run_s, run_e, seed in zip(slot['runs'], entity['runs'], (0, 1)):
        _check_pair(run_s, run_e, data, test)
        assert (run_s['run_id'], run_s['seed']) == (f'seed{seed}', seed)
        assert (run_s['n_train'], run_s['n_val']) == (12, 6)
        assert run_s['multi_task']['n_split_examples'] == 0       # the levels align
