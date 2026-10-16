"""The published split of MaterioMiner (ner_baseline.py --published-split; ROADMAP §4, week-5
amendment). The checks of the release files need no torch; the run needs torch,
transformers, seqeval and scipy (skipped without them)."""
import json
from types import SimpleNamespace

import pytest

import ner_baseline as nb


def _data(granularity, published=False):
    args = SimpleNamespace(top_k=27, published_split=published)
    return nb.granularity_data(granularity, args)[0]


@pytest.mark.parametrize('granularity, n_changed', [('fine', 6), ('coarse', 0)])
def test_the_release_partition_covers_the_corpus(granularity, n_changed):
    data = _data(granularity)
    parts, tagged = nb.published_split(data, granularity)
    assert [len(p) for p in parts] == [323, 57, 96]
    assert sorted(i for p in parts for i in p) == list(range(len(data))) == list(range(476))
    # the words are the corpus's; the tags differ in 6 fine-grained sentences (B- for I-)
    assert all(t[0] == d[0] and t[2] == d[2] for t, d in zip(tagged, data))
    changed = [i for i, (t, d) in enumerate(zip(tagged, data)) if t[1] != d[1]]
    assert len(changed) == n_changed
    for i in changed:
        for new, old in zip(tagged[i][1], data[i][1]):
            assert new == old or (new.startswith('B-') and old == 'I-' + new[2:])
    # granularity_data with --published-split gives the release's tags
    assert _data(granularity, published=True) == tagged


def test_the_two_granularities_have_their_own_partitions():
    fine = nb.published_split(_data('fine'), 'fine')[0]
    coarse = nb.published_split(_data('coarse'), 'coarse')[0]
    assert sorted(fine[2]) != sorted(coarse[2])


def test_the_release_files_are_checked(monkeypatch):
    sha = dict(nb.PUBLISHED_SPLIT_SHA256)
    sha['fine', 'val'] = '0' * 64
    monkeypatch.setattr(nb, 'PUBLISHED_SPLIT_SHA256', sha)
    with pytest.raises(SystemExit, match='SHA-256'):
        nb.published_split(_data('fine'), 'fine')
    # a corpus without one of the release's sentences does not match
    monkeypatch.undo()
    data = _data('coarse')
    with pytest.raises(SystemExit, match='not in the corpus'):
        nb.published_split(data[:-1], 'coarse')


def test_parser_names_and_folds(capsys):
    args = nb.parse_args(['--granularity', 'coarse', '--crf', '--published-split',
                          '--seeds', '0', '100'])
    assert nb.protocol_name(args) == 'published_split' and nb.variant_name(args) == 'crf'
    path = nb.results_path('out', 'coarse', args)
    assert path.endswith('ner_results_coarse_published_split_crf.json')
    assert nb.parse_result_name(path.rsplit('/', 1)[1]) == ('coarse', 'published_split', 'crf')
    assert nb.parse_result_name('ner_results_fine_published_split.json') == (
        'fine', 'published_split', 'baseline')
    assert 'published_split' not in vars(nb.parse_args(['--crf']))   # other configs unchanged
    data = _data('coarse', published=True)
    folds = nb.make_folds(data, args, indices=True, granularity='coarse')
    assert [f[0] for f in folds] == ['seed0', 'seed100']
    assert folds[0][1:] == folds[1][1:] == tuple(nb.published_split(data, 'coarse')[0])
    for argv, error in ((['--published-split', '--vary-split'], 'protocol of its own'),
                        (['--published-split', '--leave-one-paper-out'], 'protocol of its own'),
                        (['--crf', '--multi-task', '--published-split'], 'partitions differ')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err


def test_a_run_on_the_published_split(tmp_path, monkeypatch):
    """One file per variant, a run per model seed, all on the same partition."""
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
    toy = [(['steel', 'alloy', 'high'] + ['high'] * k, ['B-MAT', 'I-MAT', 'O'] + ['O'] * k,
            'a.conll') for k in range(20)]
    parts = [list(range(0, 12)), list(range(12, 15)), list(range(15, 20))]
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: toy)
    monkeypatch.setattr(nb, 'published_split', lambda data, granularity: (parts, list(data)))
    nb.run(nb.parse_args(['--granularity', 'coarse', '--model', str(path), '--epochs', '2',
                          '--threads', '1', '--device', 'cpu', '--published-split',
                          '--seeds', '0', '100', '--save-predictions']))
    res = json.loads((tmp_path / 'ner' / 'ner_results_coarse_published_split.json').read_text())
    assert res['protocol'] == 'published_split' and res['variant'] == 'baseline'
    assert res['config']['published_split'] is True
    assert [r['run_id'] for r in res['runs']] == ['seed0', 'seed100']
    assert [r['seed'] for r in res['runs']] == [0, 100]
    assert all(r['test_index'] == parts[2] and r['n_train'] == 12 for r in res['runs'])
