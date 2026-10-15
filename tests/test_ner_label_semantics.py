"""Label-semantic NER head in ner_baseline.py (--crf --label-semantics).

The parser checks, the variant names and the resume isolation need no torch. The
head tests need torch, and the tiny-model runs also transformers, seqeval and scipy;
they are skipped without them.
"""
import json
import os
from types import SimpleNamespace

import pytest

import ner_baseline as nb
import ner_label_text as lt


def test_label_semantics_parser_checks(capsys):
    args = nb.parse_args(['--crf', '--label-semantics'])
    assert args.label_semantics and nb.variant_name(args) == 'crf_ls'
    args = nb.parse_args(['--crf', '--label-semantics', '--tag', 'm100', '--seed-offset', '100',
                          '--save-predictions', '--leave-one-paper-out'])
    assert nb.variant_name(args) == 'crf_ls_m100' and args.seed_offset == 100
    assert nb.stored_config(args)['label_semantics'] is True
    # without the flag there is no label_semantics field: the config other runs store is unchanged
    assert 'label_semantics' not in vars(nb.parse_args(['--crf']))
    for argv, error in ((['--label-semantics'], '--label-semantics needs --crf'),
                        (['--label-semantics', '--class-weights'], '--label-semantics needs --crf')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err
    # on the multi-task model (week 4): a variant of its own, compared with crf
    args = nb.parse_args(['--crf', '--multi-task', '--label-semantics', '--tag', 'm100',
                          '--seed-offset', '100'])
    assert nb.variant_name(args) == 'crf_mt_ls_m100' and nb.reference_variant('crf_mt_ls') == 'crf'
    assert nb.variant_name(nb.parse_args(['--crf', '--multi-task', '--label-semantics'])) == \
        'crf_mt_ls'
    # a multi-task run tagged ls would write to its files
    with pytest.raises(SystemExit):
        nb.parse_args(['--crf', '--multi-task', '--tag', 'ls'])
    assert 'would overwrite the results of the crf_mt_ls variant' in capsys.readouterr().err
    # a CRF run tagged ls would write to the files of the label-semantics runs
    with pytest.raises(SystemExit):
        nb.parse_args(['--crf', '--tag', 'ls'])
    assert 'would overwrite the results of the crf_ls variant' in capsys.readouterr().err
    assert nb.variant_name(nb.parse_args(['--crf', '--label-semantics', '--tag', 'ls'])) == 'crf_ls_ls'


def test_label_semantics_files_are_a_variant_of_their_own(tmp_path):
    args = nb.parse_args(['--crf', '--label-semantics', '--vary-split', '--tag', 'm100',
                          '--seed-offset', '100'])
    assert os.path.basename(nb.results_path(str(tmp_path), 'fine', args)) == (
        'ner_results_fine_random_split_vary_crf_ls_m100.json')
    # compared with the CRF runs of the same granularity and protocol
    assert nb.reference_variant('crf_ls') == nb.reference_variant('crf_ls_m100') == 'crf'
    for protocol in ('random_split_fixed', 'random_split_vary', 'leave_one_paper_out'):
        for variant in ('crf_ls', 'crf_ls_m100'):
            fn = f'ner_results_coarse{nb.result_suffix(protocol, variant)}.json'
            assert nb.parse_result_name(fn) == ('coarse', protocol, variant)
    # the runs of a crf file are neither resumed nor replaced by a crf_ls run, nor the reverse
    path = tmp_path / 'f.json'
    for stored, now in (('crf', 'crf_ls'), ('crf_ls', 'crf'), ('crf_m100', 'crf_ls_m100')):
        path.write_text(json.dumps({'labels': ['A'], 'model': 'm', 'variant': stored,
                                    'runs': [{'run_id': 'seed0', 'test_pred': ['O']}]}))
        assert nb.resumable_runs(str(path), ['A'], 'm', now, save_predictions=True) == []
        with pytest.raises(nb.OverwriteError, match=f"variant: '{stored}' in the file"):
            nb.check_overwrite(str(path), ['A'], 'm', now)
        assert len(nb.resumable_runs(str(path), ['A'], 'm', stored, save_predictions=True)) == 1


# --- the head (torch)
LABELS = ['A', 'B', 'C']
TAGS = ['O'] + [f'{p}-{t}' for t in LABELS for p in ('B', 'I')]     # the tag list of train_one


@pytest.fixture
def torch():
    return pytest.importorskip('torch')


def _head(torch, dim=8, seed=0):
    torch.manual_seed(seed)
    embeddings = 2.0 + 3.0 * torch.randn(len(LABELS), dim)
    config = SimpleNamespace(hidden_size=dim, initializer_range=0.02)
    return nb.make_label_head(embeddings, config), embeddings


def test_head_scores_follow_the_tag_list(torch):
    head, embeddings = _head(torch)
    with torch.no_grad():
        head.bias.normal_()                           # so that the order of the bias counts too
    hidden = torch.randn(2, 5, 8)
    logits = head(hidden)
    assert logits.shape == (2, 5, len(TAGS))
    # the stored embeddings are the given ones over one factor: a unit root-mean-square norm
    e = head.descriptions
    assert torch.allclose(e, embeddings / embeddings.pow(2).sum(1).mean().sqrt())
    assert e.pow(2).sum(1).mean().item() == pytest.approx(1.0)
    for j, tag in enumerate(TAGS):
        if tag == 'O':
            vector = head.v_o
        else:
            vector = (head.w_b if tag[0] == 'B' else head.w_i) @ e[LABELS.index(tag[2:])]
        assert torch.allclose(logits[..., j], hidden @ vector + head.bias[j], atol=1e-5)


def test_gradients_reach_the_maps_but_not_the_descriptions(torch):
    head, _ = _head(torch)
    before = head.descriptions.clone()
    assert {n for n, _ in head.named_parameters()} == {'w_b', 'w_i', 'v_o', 'bias'}
    assert [n for n, _ in head.named_buffers()] == ['descriptions']
    crf = nb.make_crf(TAGS)
    labels = torch.tensor([[-100, 1, 2, 0, 3, -100], [-100, 5, 6, 0, -100, -100]])
    loss = crf.nll(*nb.word_view(head(torch.randn(2, 6, 8)), labels))
    loss.backward()
    for p in (head.w_b, head.w_i, head.v_o, head.bias):
        assert p.grad is not None and p.grad.abs().sum() > 0
    assert not head.descriptions.requires_grad and head.descriptions.grad is None
    torch.optim.AdamW(head.parameters(), lr=0.1).step()
    assert torch.equal(head.descriptions, before)


def test_initial_logits_have_the_magnitude_of_the_default_classifier(torch):
    torch.manual_seed(0)
    hidden_size, n_labels = 64, 27
    # mean encoder states of different texts are alike: a large common part
    embeddings = 4.0 + torch.randn(n_labels, hidden_size)
    config = SimpleNamespace(hidden_size=hidden_size, initializer_range=0.02)
    head = nb.make_label_head(embeddings, config)
    default = torch.nn.Linear(hidden_size, 1 + 2 * n_labels)      # as initialised by BERT
    torch.nn.init.normal_(default.weight, std=0.02)
    torch.nn.init.zeros_(default.bias)
    hidden = 5.0 * torch.randn(500, hidden_size)
    with torch.no_grad():
        ratio = (head(hidden).std() / default(hidden).std()).item()
    assert 0.8 < ratio < 1.25


# --- runs with a tiny random model (torch, transformers, seqeval and scipy; skipped without)
TOY_WORDS = ['steel', 'alloy', 'tensile', 'strength', 'yield', 'room', 'temperature', 'high']


def _toy_corpora():
    """Fine and coarse versions of 24 sentences from four 'papers'.

    The fine classes Alloy (an MMO class with a definition), Steel and Strength map
    to the coarse types MAT and MEP. One sentence per paper is tokenised differently.
    """
    fine, coarse = [], []
    for p, doc in enumerate(('a.conll', 'b.conll', 'c.conll', 'd.conll')):
        for k in range(6):
            tokens = ['steel', 'alloy', 'tensile', 'strengths'] + ['high'] * k
            coarse.append((tokens, ['B-MAT', 'B-MAT', 'B-MEP', 'I-MEP'] + ['O'] * k, doc))
            extra = ['room'] if k == 3 else []
            fine.append((tokens + extra, ['B-Steel', 'B-Alloy', 'B-Strength', 'I-Strength']
                         + ['O'] * (k + len(extra)), doc))
    return {'fine': fine, 'coarse': coarse}


@pytest.fixture(scope='module')
def tiny_model(tmp_path_factory):
    """A tiny random BERT with a word-piece vocabulary ('strengths' is two subwords)."""
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    path = tmp_path_factory.mktemp('tiny_bert_ls')
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', *TOY_WORDS, '##s']
    (path / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    transformers.BertTokenizerFast(str(path / 'vocab.txt')).save_pretrained(path)
    torch.manual_seed(0)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=160)).save_pretrained(path)
    return str(path)


@pytest.fixture
def tiny_run(tiny_model, tmp_path, monkeypatch):
    """run(*flags): ner_baseline.run on the toy corpora in tmp_path/ner; returns its heads.

    The heads are (embeddings, head, initial state) of every make_label_head call.
    """
    corpora = _toy_corpora()
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: corpora[granularity])
    heads = []
    make = nb.make_label_head

    def recording(embeddings, config):
        head = make(embeddings, config)
        heads.append((embeddings, head, {k: v.clone() for k, v in head.state_dict().items()}))
        return head
    monkeypatch.setattr(nb, 'make_label_head', recording)

    def run(*flags):
        nb.run(nb.parse_args(['--model', tiny_model, '--epochs', '2', '--threads', '1',
                              '--device', 'cpu', *flags]))
        return heads
    return run


def _load(tmp_path, name):
    return json.loads((tmp_path / 'ner' / name).read_text())


def test_label_semantics_run(tiny_run, tiny_model, tmp_path):
    import torch
    import transformers
    flags = ('--crf', '--label-semantics', '--vary-split', '--seeds', '0', '--tag', 't1',
             '--seed-offset', '100', '--save-predictions')
    heads = tiny_run(*flags)
    assert len(heads) == 2                            # one run per granularity (both by default)
    texts = {}
    for g, (embeddings, head, initial) in zip(('fine', 'coarse'), heads):
        res = _load(tmp_path, f'ner_results_{g}_random_split_vary_crf_ls_t1.json')
        labels = res['labels']
        texts[g] = nb.label_descriptions(g, labels)
        assert res['variant'] == 'crf_ls_t1' and res['config']['label_semantics'] is True
        assert res['config']['label_descriptions_sha256'] == lt.descriptions_sha256(
            labels, texts[g])
        run = res['runs'][0]
        assert (run['run_id'], run['seed']) == ('seed0', 100) and 'test_pred' in run
        assert set(' '.join(run['test_pred']).split()) <= {
            'O', *(f'{b}-{t}' for t in labels for b in 'BI')}
        # a frozen embedding per label; the maps were trained
        assert embeddings.shape == (len(labels), 16)
        assert torch.equal(head.descriptions, initial['descriptions'])
        assert not torch.equal(head.w_b, initial['w_b'])
    assert texts['fine'][0].startswith('Alloy: An alloy is an inorganic material')
    assert texts['fine'][1:] == ['Steel', 'Strength']                # not MMO classes
    assert texts['coarse'] == ['MAT: Alloy, Steel', 'MEP: Strength']
    # each embedding is the mean last-layer state of the pre-trained encoder over the
    # description's own tokens (no [CLS] or [SEP])
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_model)
    encoder = transformers.AutoModel.from_pretrained(tiny_model).eval()
    for g, (embeddings, _, _) in zip(('fine', 'coarse'), heads):
        for text, row in zip(texts[g], embeddings):
            with torch.no_grad():
                states = encoder(**tokenizer(text, return_tensors='pt')).last_hidden_state
            assert torch.allclose(row, states[0, 1:-1].mean(0), atol=1e-5)
    # finished runs are skipped; a plain CRF run makes no label head and stores no setting
    assert len(tiny_run(*flags)) == 2
    tiny_run('--crf', '--granularity', 'fine', '--vary-split', '--seeds', '0')
    assert len(heads) == 2
    config = _load(tmp_path, 'ner_results_fine_random_split_vary_crf.json')['config']
    assert 'label_semantics' not in config and 'label_descriptions_sha256' not in config


def test_label_semantics_on_the_multitask_model(tiny_run, tiny_model, tmp_path):
    """--multi-task --label-semantics: one encoder, a label-semantic head per granularity."""
    import torch
    import transformers
    flags = ('--crf', '--multi-task', '--label-semantics', '--vary-split', '--seeds', '0',
             '--tag', 'm100', '--seed-offset', '100', '--save-predictions')
    heads = tiny_run(*flags)
    assert len(heads) == 2                            # one model, a head per granularity
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_model)
    encoder = transformers.AutoModel.from_pretrained(tiny_model).eval()
    sha = {}
    for g, (embeddings, head, initial) in zip(('fine', 'coarse'), heads):
        res = _load(tmp_path, f'ner_results_{g}_random_split_vary_crf_mt_ls_m100.json')
        labels = res['labels']
        texts = nb.label_descriptions(g, labels)
        assert res['variant'] == 'crf_mt_ls_m100'
        assert res['config']['multi_task'] and res['config']['label_semantics']
        sha[g] = res['config']['label_descriptions_sha256']
        assert sha[g] == lt.descriptions_sha256(labels, texts)
        run = res['runs'][0]
        assert (run['run_id'], run['seed']) == ('seed0', 100) and 'multi_task' in run
        assert set(' '.join(run['test_pred']).split()) <= {
            'O', *(f'{b}-{t}' for t in labels for b in 'BI')}
        # the descriptions of this granularity, embedded by the pre-trained encoder; frozen,
        # while the maps were trained
        assert embeddings.shape == (len(labels), 16)
        for text, row in zip(texts, embeddings):
            with torch.no_grad():
                states = encoder(**tokenizer(text, return_tensors='pt')).last_hidden_state
            assert torch.allclose(row, states[0, 1:-1].mean(0), atol=1e-5)
        assert torch.equal(head.descriptions, initial['descriptions'])
        assert not torch.equal(head.w_b, initial['w_b'])
    assert sha['fine'] != sha['coarse']
    # finished runs are skipped; a multi-task run without the flag makes no label head
    assert len(tiny_run(*flags)) == 2
    tiny_run('--crf', '--multi-task', '--vary-split', '--seeds', '0')
    assert len(heads) == 2
    config = _load(tmp_path, 'ner_results_coarse_random_split_vary_crf_mt.json')['config']
    assert 'label_semantics' not in config and 'label_descriptions_sha256' not in config


def test_descriptions_are_cut_to_the_token_limit(tiny_model):
    import torch
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_model)
    encoder = transformers.AutoModel.from_pretrained(tiny_model).eval()
    long_text = 'steel alloy ' * 40
    rows = nb.embed_descriptions(encoder, tokenizer, [long_text, 'steel'], 'cpu', max_tokens=10)
    assert rows.shape == (2, 16)
    with torch.no_grad():
        states = encoder(**tokenizer('steel alloy ' * 4, return_tensors='pt')).last_hidden_state
    assert torch.allclose(rows[0], states[0, 1:-1].mean(0), atol=1e-5)    # 8 words + 2 specials
    assert encoder.training is False
