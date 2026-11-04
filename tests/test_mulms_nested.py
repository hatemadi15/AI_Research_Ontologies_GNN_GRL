"""MuLMS's nested NER re-run (mulms_nested.py): the release's decoder and scorer on worked
cases, the configurations, the stacked-tag CRF against brute force, a smoke run on a tiny
encoder, and on the pinned release (when cached) its counts, configurations and the score of
its own gold tags."""
import itertools
import json
import math

import pytest

import mulms_nested as mn


def test_decode_follows_the_release():
    # a unit entity, a nested pair, and a multi-token entity with a unit one inside
    assert mn.decode(['U-MAT', 'O']) == {('MAT', 0, 0)}
    assert mn.decode(['B-VALUE+U-NUM', 'L-VALUE+U-UNIT']) == {
        ('VALUE', 0, 1), ('NUM', 0, 0), ('UNIT', 1, 1)}
    assert mn.decode(['B-MAT', 'I-MAT+U-NUM', 'L-MAT']) == {('MAT', 0, 2), ('NUM', 1, 1)}
    # an entity without its L- tag is dropped, and the I- tags it took go with it
    assert mn.decode(['B-MAT', 'I-MAT', 'O']) == set()
    # a stray I-, L- or a B- followed by another type is dropped
    assert mn.decode(['I-MAT', 'L-MAT']) == set()
    assert mn.decode(['B-MAT', 'L-NUM']) == set()
    # B- takes the I-/L- tags at the front of the following stacks only
    assert mn.decode(['B-MAT+B-VALUE', 'U-NUM+L-MAT+L-VALUE']) == {('NUM', 1, 1)}
    assert mn.decode(['O', 'O']) == set() and mn.decode([]) == set()


def test_score_micro_and_macro_over_the_types():
    pred = [{('MAT', 0, 0), ('NUM', 1, 1)}, {('MAT', 2, 3)}]
    gold = [{('MAT', 0, 0), ('UNIT', 1, 1)}, {('MAT', 2, 3), ('MAT', 5, 5)}]
    s = mn.score(pred, gold, types=('MAT', 'NUM', 'UNIT'))
    # TP 2 (MAT), FP 1 (NUM), FN 2 (UNIT, MAT): P 2/3, R 1/2, F1 4/7
    assert (s['TP'], s['FP'], s['FN']) == (2, 1, 2)
    assert s['micro_f1'] == round(100 * 4 / 7, 2)
    # macro: MAT P 1, R 2/3, F1 0.8; NUM and UNIT 0; the mean over the 3 types
    assert s['macro_f1'] == round(100 * 0.8 / 3, 2)
    assert s['per_type']['MAT']['f1'] == 80.0
    with pytest.raises(ValueError):
        mn.score(pred, gold[:1])


def sentence(doc, split, tags, gold=()):
    return {'doc_id': doc, 'split': split, 'tokens': [f'w{i}' for i in range(len(tags))],
            'tags': tags, 'gold': set(gold)}


def test_configurations_and_vocabulary():
    sents = [sentence(f'd{k}', f'train{k}', ['U-MAT']) for k in range(1, 6)]
    sents += [sentence('dv', 'dev', ['B-NUM', 'L-NUM']), sentence('ts', 'test', ['O'])]
    parts = mn.configuration(sents, 3)
    assert [s['doc_id'] for s in parts['tune']] == ['d3']
    assert [s['doc_id'] for s in parts['train']] == ['d1', 'd2', 'd4', 'd5']
    assert [s['doc_id'] for s in parts['dev']] == ['dv'] and len(parts['test']) == 1
    record = mn.configurations_record(sents)
    assert set(record) == {f'c{c}' for c in mn.CONFIGS}
    assert record['c1']['tune'] == ['d1'] and 'd1' not in record['c1']['train']
    assert mn.tag_vocabulary(sents) == ['O', 'B-NUM', 'L-NUM', 'U-MAT']
    with pytest.raises(ValueError):
        mn.configuration(sents, 6)
    rows = [{'doc_id': 'x', 'data_split': 'train1', 'tokens': ['a', 'b'],
             'NER_labels_BILOU': ['U-MAT', 'O'],
             'NER_labels': {'value': ['MAT', 'MAT'], 'tokenIndices': [[0, 0], [0, 0]]}}]
    assert mn.sentences_of(rows)[0]['gold'] == {('MAT', 0, 0)}     # duplicates collapse
    with pytest.raises(ValueError):
        mn.sentences_of([{**rows[0], 'NER_labels_BILOU': ['O']}])


def test_sqrt_schedule_and_settings():
    factor = mn.sqrt_schedule(4)
    assert factor(0) == 0.0
    assert math.isclose(factor(1), 1 / 4) and math.isclose(factor(4), 1.0)    # linear warm-up
    assert math.isclose(factor(16), 0.5)                                       # sqrt(4 / 16)
    # the paper's Table 13: MatSciBERT's CRF tagger at 1e-4 and 7e-3
    assert (mn.SETTINGS['lr'], mn.SETTINGS['lr_crf']) == (1e-4, 7e-3)
    assert mn.parse_args([]).lr_schedule == 'none'
    assert mn.parse_args(['--lr-schedule', 'sqrt']).lr_schedule == 'sqrt'


def test_stacked_crf_matches_brute_force():
    torch = pytest.importorskip('torch')
    torch.manual_seed(0)
    crf_class, _ = mn._define_modules()
    n, width = 3, 4
    crf = crf_class(n, init=1.0)
    emissions = torch.randn(2, width, n)
    mask = torch.tensor([[True] * width, [True, True, False, False]])

    def path_score(b, path):
        s = crf.start[path[0]] + emissions[b, 0, path[0]]
        for t in range(1, len(path)):
            s = s + crf.transitions[path[t - 1], path[t]] + emissions[b, t, path[t]]
        return s + crf.end[path[-1]]
    for b in range(2):
        length = int(mask[b].sum())
        paths = list(itertools.product(range(n), repeat=length))
        scores = torch.stack([path_score(b, p) for p in paths])
        assert math.isclose(float(crf.log_partition(emissions, mask)[b]),
                            float(torch.logsumexp(scores, 0)), rel_tol=1e-5)
        assert crf.decode(emissions, mask)[b] == list(paths[int(scores.argmax())])
    tags = torch.tensor([[0, 1, 2, 1], [2, 2, 0, 0]])
    nll = crf.nll(emissions, tags, mask)
    expected = sum(float(crf.log_partition(emissions, mask)[b])
                   - float(path_score(b, tags[b, :int(mask[b].sum())].tolist()))
                   for b in range(2))
    assert math.isclose(float(nll), expected, rel_tol=1e-5) and float(nll) > 0


@pytest.fixture(scope='module')
def tiny_model(tmp_path_factory):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    path = tmp_path_factory.mktemp('tiny_bert')
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'iron', 'steel', 'at', '5', 'k',
             'the', '##s']
    (path / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    transformers.BertTokenizerFast(str(path / 'vocab.txt')).save_pretrained(path)
    torch.manual_seed(0)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64)).save_pretrained(path)
    return str(path)


def toy_release():
    words = ['the', 'steel', 'at', '5', 'k']
    tags = ['O', 'U-MAT', 'O', 'B-VALUE+U-NUM', 'L-VALUE+U-UNIT']
    gold = {('MAT', 1, 1), ('VALUE', 3, 4), ('NUM', 3, 3), ('UNIT', 4, 4)}
    out = []
    for split in ('train1', 'train2', 'train3', 'train4', 'train5', 'dev', 'test'):
        for k in range(3):
            out.append({'doc_id': f'{split}_{k}', 'split': split, 'tokens': list(words),
                        'tags': list(tags), 'gold': set(gold)})
    # a sentence with a word the tokenizer erases (a lone non-breaking space)
    out.append({'doc_id': 'test_x', 'split': 'test', 'tokens': ['iron', ' ', 'steel'],
                'tags': ['U-MAT', 'O', 'U-MAT'], 'gold': {('MAT', 0, 0), ('MAT', 2, 2)}})
    return out


def test_smoke_run_on_a_tiny_encoder(tiny_model, tmp_path, monkeypatch):
    monkeypatch.setattr(mn, 'read_release', lambda cache=None, offline=False: toy_release())
    args = mn.parse_args(['--model', tiny_model, '--configs', '1', '2', '--epochs', '3',
                          '--patience', '2', '--out-dir', str(tmp_path), '--save-predictions',
                          '--device', 'cpu', '--tag', 'toy', '--seed-offset', '100'])
    results = mn.run(args)
    path = mn.results_path(tmp_path, 'toy')
    assert path.name == 'ner_results_mulms_nested_configs_bilou_crf_toy.json'
    stored = json.loads(path.read_text())
    assert [r['run_id'] for r in stored['runs']] == ['c1', 'c2']
    run = stored['runs'][0]
    assert run['seed'] == 100 and run['n_train'] == 12 and run['n_tune'] == 3
    assert run['n_test'] == 4 and len(run['test_pred']) == 4
    assert run['erased_word_sentences']['test'] == 1 and run['truncated_words'] == 0
    assert 1 <= run['best_epoch'] <= 3 and len(run['history']) >= run['best_epoch']
    assert 0 <= run['test']['micro_f1'] <= 100 and run['f1'] == run['test']['micro_f1']
    assert stored['config']['n_tags'] == len(mn.tag_vocabulary(toy_release()))
    assert stored['configurations']['c2']['tune'] == ['train2_0', 'train2_1', 'train2_2']
    # a second call skips the runs already in the file
    again = mn.run(args)
    assert [r['run_id'] for r in again['runs']] == ['c1', 'c2']
    assert results['test_micro_f1_mean'] == stored['test_micro_f1_mean']
    # the paper's schedule runs too, and is recorded
    args = mn.parse_args(['--model', tiny_model, '--configs', '3', '--epochs', '2',
                          '--out-dir', str(tmp_path), '--device', 'cpu', '--tag', 'sq',
                          '--lr-schedule', 'sqrt'])
    assert mn.run(args)['config']['lr_schedule'] == 'sqrt'


def test_micro_batches_give_the_batch_gradient(tiny_model):
    """The batch's summed loss over micro-batches has the full batch's gradient (dropout off)."""
    torch = pytest.importorskip('torch')
    from transformers import AutoModel, AutoTokenizer
    sents = toy_release()[:6]
    vocab = mn.tag_vocabulary(toy_release())
    index = {t: i for i, t in enumerate(vocab)}
    tok = AutoTokenizer.from_pretrained(tiny_model)
    feats = [mn.features(s, tok, index) for s in sents]
    _, tagger = mn._define_modules()
    torch.manual_seed(0)
    model = tagger(AutoModel.from_pretrained(tiny_model), len(vocab)).eval()
    model.pad_id = tok.pad_token_id

    def grads(micro):
        model.zero_grad()
        for m in range(0, len(feats), micro):
            batch = mn.collate(feats[m:m + micro], model.pad_id, 'cpu')
            model.crf.nll(model.emissions(batch), batch['tags'], batch['mask']).backward()
        return [p.grad.clone() for p in model.parameters() if p.grad is not None]
    full, parts = grads(6), grads(2)
    assert len(full) == len(parts) and all(torch.allclose(a, b, atol=1e-5)
                                           for a, b in zip(full, parts))


def release_or_skip():
    pytest.importorskip('pyarrow')
    try:
        return mn.read_release(offline=True)
    except (SystemExit, FileNotFoundError, OSError, ValueError) as err:
        pytest.skip(f'the pinned MuLMS release is not cached: {err}')


def test_the_pinned_release():
    sents = release_or_skip()
    assert len(sents) == 10184
    assert sum(len(s['gold']) for s in sents) > 45000
    vocab = mn.tag_vocabulary(sents)
    assert len(vocab) == 473 and vocab[0] == 'O'
    record = mn.configurations_record(sents)
    docs = {d for part in record['c1'].values() for d in part}
    assert len(docs) == 50
    for c in mn.CONFIGS:
        assert not set(record[f'c{c}']['tune']) & set(record[f'c{c}']['train'])
        assert record[f'c{c}']['dev'] == record['c1']['dev']
    # the release's own gold tags, decoded, against its gold entities: the ceiling the
    # three-layer tags leave
    test = [s for s in sents if s['split'] == 'test']
    ceiling = mn.score([mn.decode(s['tags']) for s in test], [s['gold'] for s in test])
    assert ceiling['FP'] == 0 and ceiling['micro_f1'] > 95
