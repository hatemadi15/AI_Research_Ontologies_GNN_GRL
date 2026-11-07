"""The SOTA plan's P1 (ROADMAP §4, the screens): screen mode and round 1's changes in
ner_baseline.py and mulms_nested.py, the span head (span_ner.py), the data files, the cells,
checks and rounds of sota_p1.py, the timing table, and batches p1t and p1n."""
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

import kaggle_jobs as kj
import mulms_nested as mn
import ner_baseline as nb
import sota_p1 as p1
import span_ner


# ---------------------------------------------------------------------------------------------
# a tiny encoder and toy data
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    path = tmp_path_factory.mktemp('tiny_bert_p1')
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'iron', 'steel', 'at', '5', 'k',
             'the', '##s', 'cathode', 'of', 'cell']
    (path / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    transformers.BertTokenizerFast(str(path / 'vocab.txt')).save_pretrained(path)
    torch.manual_seed(0)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=128)).save_pretrained(path)
    return str(path)


def toy_part(n, seed, doc='p1'):
    """(words, IOB2 tags, doc) sentences: an entity sentence or a plain one, drawn by seed."""
    rng = random.Random(seed)
    out = []
    for k in range(n):
        if rng.random() < 0.5:
            words, tags = (['the', 'steel', 'at', '5', 'k', 'cathode'],
                           ['O', 'B-MAT', 'O', 'B-VAL', 'I-VAL', 'O'])
        else:
            words, tags = ['cathode', 'of', 'the', 'cell'], ['O'] * 4
        out.append((words, tags, f'{doc}_{k % 2}'))
    return out


def toy_args(model, **over):
    args = nb.parse_args(['--granularity', 'sofc', '--crf', '--published-split', '--seeds',
                          '0', '--model', model, '--epochs', '3', '--batch-size', '4'])
    for k, v in over.items():
        setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------------------------
# screen splits and data files
# ---------------------------------------------------------------------------------------------
def test_screen_split_parts():
    parts = [list(range(0, 323)), list(range(323, 380)), list(range(380, 476))]
    seen = set()
    for k in range(3):
        train, val = nb.screen_split_parts(parts, k)
        assert len(train) == 323 and len(val) == 57
        assert set(train) | set(val) == set(range(380)) and not set(train) & set(val)
        assert nb.screen_split_parts(parts, k) == (train, val)          # deterministic
        seen.add(tuple(val))
    assert len(seen) == 3


def test_data_files_are_current():
    record = json.loads(p1.SPLITS_FILE.read_text())
    assert record['sha256'] == p1.canonical_sha256(record['splits'])
    for level in p1.LEVELS:
        for k in range(p1.N_SCREEN_SPLITS):
            split = record['splits'][level][f'seed{k}']
            pool = [list(range(323)), list(range(323, 380))]
            assert (split['train'], split['val']) == tuple(nb.screen_split_parts(pool, k))
    shared = json.loads(p1.SHARED_FILE.read_text())
    assert shared['sha256'] == p1.canonical_sha256(shared['pairs'])
    assert {(x['mulms'], x['sofc_exp']) for x in shared['pairs']} == {
        ('sofc_01', 'PMC5700654'), ('sofc_02', 'PMC4772004'), ('sofc_03', 'PMC6158676'),
        ('sofc_04', 'PMC6247067')}
    assert shared['largest_other'] < p1.SHARED_MIN_SENTENCES
    assert p1.shared_mulms_documents() == {'sofc_01', 'sofc_02', 'sofc_03', 'sofc_04'}
    pytest.importorskip('pyarrow')
    try:
        fresh = p1.mm_screen_splits_record()
    except (SystemExit, FileNotFoundError, OSError) as err:
        pytest.skip(f'the MaterioMiner release is not here: {err}')
    assert record == json.loads(json.dumps(fresh))
    try:
        again = p1.shared_documents_record()
    except (SystemExit, FileNotFoundError, OSError) as err:
        pytest.skip(f'the MuLMS release or SOFC-Exp is not here: {err}')
    assert shared == json.loads(json.dumps(again))


# ---------------------------------------------------------------------------------------------
# round 1's changes
# ---------------------------------------------------------------------------------------------
def test_llrd_groups_and_the_stored_optimiser(tiny):
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(tiny)
    groups = nb.encoder_groups(encoder, 1e-4, 0.9)
    rates = [g['lr'] for g in groups]
    # embeddings (depth -1) lowest, then layer 0, then layer 1 (the top) at the base rate
    assert rates == pytest.approx([1e-4 * 0.9 ** 2, 1e-4 * 0.9, 1e-4])
    assert sum(len(g['params']) for g in groups) == len(list(encoder.parameters()))
    args = SimpleNamespace(lr=1e-4, llrd=0.0)
    opt = nb.optimizer_for(args, encoder, [], [])
    assert len(opt.param_groups) == 1 and opt.param_groups[0]['weight_decay'] == 0.01


def test_epoch_weights_average():
    torch = pytest.importorskip('torch')
    window = nb.EpochWeights(3)
    for epoch in range(5):
        window.add(epoch, {'m': {'w': torch.full((2,), float(epoch)),
                                 'n': torch.tensor([epoch])}}, best_epoch=3 if epoch >= 3 else
                   epoch)
    states, epochs = window.average(3)
    assert epochs == [1, 2, 3] and torch.equal(states['m']['w'], torch.full((2,), 2.0))
    assert int(states['m']['n']) == 3                    # not floating point: the best epoch's


def test_screen_trains_as_the_full_run(tiny):
    """Screen mode changes no training step: the same history and validation F1 as a run with
    a test set; its record has no test field."""
    train, val, test = toy_part(16, 0), toy_part(6, 1, 'p2'), toy_part(6, 2, 'p3')
    labels = ['MAT', 'VAL']
    full = nb.train_one(0, train, val, test, labels, toy_args(tiny), 'cpu')
    screen = nb.train_one(0, train, val, [], labels, toy_args(tiny, screen=True), 'cpu')
    assert screen['history'] == full['history'] and screen['val_f1'] == full['val_f1']
    assert not set(screen) & set(nb.TEST_FIELDS) and 'f1' in full
    averaged = nb.train_one(0, train, val, [], labels,
                            toy_args(tiny, screen=True, average_last=2, llrd=0.9,
                                     reinit_top=1, warmup=0.0), 'cpu')
    assert averaged['average']['epochs'][-1] == averaged['best_epoch']
    assert averaged['val_f1'] == averaged['average']['val_f1']


def test_multitask_scores_only_its_level(tiny):
    """--screen-level: the other head trains without its tags where they are dropped, and gets
    no record."""
    fine = toy_part(12, 0)
    coarse = [(w, [t.replace('MAT', 'M').replace('VAL', 'V') for t in g], d)
              for w, g, d in fine]
    val = toy_part(4, 3, 'p2')
    parts = {'fine': (fine, val, []),
             'coarse': ([(w, None if k % 3 == 0 else g, d)
                         for k, (w, g, d) in enumerate(coarse)], [], [])}
    args = nb.parse_args(['--granularity', 'both', '--multi-task', '--crf', '--screen',
                          '--screen-splits', '--screen-level', 'fine', '--seeds', '0',
                          '--model', tiny, '--epochs', '2', '--batch-size', '4'])
    records = nb.train_multitask(0, parts, {'fine': ['MAT', 'VAL'], 'coarse': ['M', 'V']},
                                 args, 'cpu', scored=('fine',))
    assert list(records) == ['fine']
    rec = records['fine']
    assert rec['multi_task']['other_tags_dropped'] == 4 and not set(rec) & set(nb.TEST_FIELDS)
    examples = nb.multitask_examples(fine, parts['coarse'][0])
    assert sum(1 for _, tags in examples if tags['coarse'] is None) == 4


def test_p1_flags_and_their_checks():
    ok = nb.parse_args(['--granularity', 'fine', '--crf', '--screen', '--screen-splits',
                        '--seeds', '0', '1', '--tag', 'p1a0'])
    assert nb.protocol_name(ok) == 'screen_split' and nb.variant_name(ok) == 'crf_p1a0'
    sofc = nb.parse_args(['--granularity', 'sofc', '--span-head', '--screen', '--published-split',
                          '--seeds', '0', '--tag', 'p1e'])
    assert nb.protocol_name(sofc) == 'published_screen' and nb.variant_name(sofc) == 'span_p1e'
    for bad in (['--granularity', 'fine', '--crf', '--screen', '--published-split'],
                ['--granularity', 'fine', '--crf', '--screen-splits'],
                ['--granularity', 'sofc', '--crf', '--screen', '--published-split',
                 '--save-predictions'],
                ['--granularity', 'both', '--multi-task', '--crf', '--screen',
                 '--screen-splits'],
                ['--granularity', 'fine', '--crf', '--joint-mulms', '--screen',
                 '--screen-splits'],
                ['--granularity', 'sofc', '--crf', '--span-head', '--published-split'],
                ['--granularity', 'sofc', '--crf', '--published-split', '--llrd', '1.5']):
        with pytest.raises(SystemExit):
            nb.parse_args(bad)
    assert nb.parse_result_name('ner_results_mulms_nested_screen_span_p1eu1.json') == (
        'mulms', 'nested_screen', 'span_p1eu1')


# ---------------------------------------------------------------------------------------------
# the span head
# ---------------------------------------------------------------------------------------------
def test_span_head_pieces(tiny):
    from transformers import AutoTokenizer
    sentences = [{'tokens': ['the', 'steel', 'at', '5', 'k'],
                  'entities': {('MAT', 1, 1), ('VAL', 3, 4), ('NUM', 3, 3)}}] * 199 + [
        {'tokens': ['the'] * 9, 'entities': {('MAT', 0, 8)}}]
    assert span_ner.width_cap(sentences) == 2                 # 99.5% of 598 entities
    assert span_ner.width_cap(sentences, 1.0) == 9
    tok = AutoTokenizer.from_pretrained(tiny)
    f = span_ner.encode(sentences[0], tok, {'MAT': 0, 'VAL': 1, 'NUM': 2}, 2)
    assert len(f['spans']) == 5 + 4 and f['unreachable'] == 0
    assert sorted(t for labels in f['labels'] for t in labels) == [0, 1, 2]
    g = span_ner.encode(sentences[-1], tok, {'MAT': 0, 'VAL': 1, 'NUM': 2}, 2)
    assert g['unreachable'] == 1
    assert span_ner.entities_to_iob2({('MAT', 1, 1), ('VAL', 3, 4)}, 5) == [
        'O', 'B-MAT', 'O', 'B-VAL', 'I-VAL']
    assert span_ner.iob2_entities(['O', 'B-MAT', 'O', 'B-VAL', 'I-VAL']) == {
        ('MAT', 1, 1), ('VAL', 3, 4)}
    torch = pytest.importorskip('torch')
    items = [{'spans': [(0, 0), (0, 1), (1, 1)], 'kept': [0, 1]}]
    logits = torch.tensor([[[3.0, -9.0], [2.0, 1.0], [-9.0, 4.0]]])
    nested = span_ner.decode(logits, items, ['A', 'B'], nested=True)[0]
    assert nested == {('A', 0, 0), ('A', 0, 1), ('B', 0, 1), ('B', 1, 1)}
    flat = span_ner.decode(logits, items, ['A', 'B'], nested=False)[0]
    assert flat == {('B', 1, 1), ('A', 0, 0)}            # greedy, most probable first


def test_span_and_joint_trainers_run(tiny):
    train, val = toy_part(12, 0), toy_part(4, 1, 'p2')
    labels = ['MAT', 'VAL']
    span = nb.train_span(0, train, val, [], labels, toy_args(tiny, screen=True), 'cpu')
    assert span['span_head']['width_cap'] == 2 and not set(span) & set(nb.TEST_FIELDS)
    other = [(['iron', 'at', '5', 'k'], ['B-X', 'O', 'B-Y', 'I-Y'])] * 20
    joint = nb.train_joint(0, train, val, [], labels, other, ['X', 'Y'],
                           toy_args(tiny, screen=True, average_last=2), 'cpu')
    assert joint['joint']['other_per_epoch'] == 12 and 'average' in joint


# ---------------------------------------------------------------------------------------------
# mulms_nested.py's P1 options
# ---------------------------------------------------------------------------------------------
def toy_release():
    words = ['the', 'steel', 'at', '5', 'k']
    tags = ['O', 'U-MAT', 'O', 'B-VALUE+U-NUM', 'L-VALUE+U-UNIT']
    gold = {('MAT', 1, 1), ('VALUE', 3, 4), ('NUM', 3, 3), ('UNIT', 4, 4)}
    return [{'doc_id': f'{split}_{k // 2}', 'split': split, 'tokens': list(words),
             'tags': list(tags), 'gold': set(gold)}
            for split in ('train1', 'train2', 'train3', 'train4', 'train5', 'dev', 'test')
            for k in range(4)]


def test_mulms_outermost_and_contexts():
    s = {'tokens': ['a', 'b', 'c', 'd'], 'gold': {('VALUE', 1, 2), ('NUM', 1, 1), ('MAT', 3, 3)}}
    assert mn.outermost_iob2(s) == ['O', 'B-VALUE', 'I-VALUE', 'B-MAT']
    group = toy_release()[:4]
    contexts = mn.part_contexts(group)
    assert contexts[0] == ([], [group[1]['tokens']]) and contexts[2] == ([], [group[3]['tokens']])


def test_mulms_screen_span_and_joint(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(mn, 'read_release', lambda cache=None, offline=False: toy_release())
    monkeypatch.setattr(mn, 'sofc_joint_sentences',
                        lambda: ([(['iron', 'at', '5', 'k'], ['B-X', 'O', 'B-Y', 'I-Y'])] * 6,
                                 ['X', 'Y']))
    base = ['--model', tiny, '--configs', '1', '--epochs', '2', '--out-dir', str(tmp_path),
            '--device', 'cpu']
    full = mn.run(mn.parse_args(base + ['--tag', 'full']))['runs'][0]
    screen = mn.run(mn.parse_args(base + ['--screen', '--tag', 'scr']))['runs'][0]
    assert screen['history'] == full['history'] and screen['dev'] == full['dev']
    assert not set(screen) & set(p1.TEST_FIELDS) and 'test' in full
    stored = json.loads(mn.results_path(tmp_path, 'scr', 0, True).read_text())
    assert stored['protocol'] == 'nested_screen' and stored['screen'] is True
    p1.check_screen_file(stored, 'scr')
    for k, extra in enumerate((['--span-head'], ['--joint-sofc'], ['--context', '16'],
                               ['--span-head', '--context', '16', '--lr-schedule', 'sqrt'])):
        tag = f'x{k}'
        run = mn.run(mn.parse_args(base + ['--screen', *extra, '--tag', tag]))['runs'][0]
        assert 'dev' in run and not set(run) & set(p1.TEST_FIELDS)
    with pytest.raises(SystemExit):
        mn.parse_args(base + ['--span-head', '--joint-sofc', '--tag', 'z'])


# ---------------------------------------------------------------------------------------------
# sota_p1: cells, checks, rounds and the selection
# ---------------------------------------------------------------------------------------------
def test_every_round1_job_parses_and_names_its_file():
    for benchmark in p1.BENCHMARKS:
        for name in p1.ROUND1[benchmark] + (list(p1.MULMS_B) if benchmark == 'mulms' else []):
            changes = frozenset() if name == 'a0' else frozenset([name])
            jobs = p1.jobs_for(benchmark, changes, p1.UNITS[benchmark])
            assert len(jobs) == (3 if benchmark == 'mulms' else 1)
            for job in jobs:
                name_ = p1.results_file(benchmark, job)
                assert nb.parse_result_name(name_)[1] in (
                    'screen_split', 'published_screen', 'nested_screen')
    job = p1.jobs_for('fine', {'c1'}, ['seed0'])[0]
    assert p1.results_file('fine', job) == 'ner_results_fine_screen_split_crf_mt_p1c1.json'
    job = p1.jobs_for('sofc', {'c1', 'c3'}, ['seed0'])[0]
    assert p1.results_file('sofc', job) == 'ner_results_sofc_published_screen_crf_mt_ctx_p1c1c3.json'
    job = p1.jobs_for('mulms', {'e'}, ['c2s100'])[0]
    assert p1.results_file('mulms', job) == 'ner_results_mulms_nested_screen_span_p1eu2s100.json'


ENV = {'device': 'Tesla T4', 'cuda': '12.8', 'torch': '2.11.0+cu128', 'transformers': '5.18.0'}


def fake_world(tmp_path, benchmark, means, a0_offsets=None, units=None):
    """Write screen files for a benchmark: per cell (changes) a value per unit, the A0 value
    plus the cell's mean gain plus a small per-unit wiggle."""
    rng = random.Random(1)
    units = units or p1.UNITS[benchmark]
    base = {u: 70.0 + rng.uniform(-2, 2) for u in units + p1.EXTRA_UNITS[benchmark]}
    for changes, gain in means.items():
        changes = frozenset(changes)
        for job in p1.jobs_for(benchmark, changes, units + p1.EXTRA_UNITS[benchmark]):
            args = nb.parse_args(job)
            # every cell's value is A0's plus its gain: the paired differences are exact
            runs = [{'run_id': u, 'seed': p1.unit_seed(u), 'val_f1': round(base[u] + gain, 2),
                     'history': [{'epoch': 0, 'val_f1': base[u]}], 'environment': ENV}
                    for u in units + p1.EXTRA_UNITS[benchmark]]
            data = {'screen': True, 'model': args.model, 'config': nb.stored_config(args),
                    'runs': runs}
            (tmp_path / p1.results_file(benchmark, job)).write_text(json.dumps(data))


def test_rounds_chain_and_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(p1, 'TIMING_FILE', tmp_path / 'none.json')
    monkeypatch.setattr(p1, '_REFERENCE_ENV', {'env': dict(p1.environment_of({
        'environment': ENV}))})
    # SOFC-Exp round 1: a2 and a5 gain, a1 a little, c3 and e gain, the rest nothing
    gains = {(): 0.0, ('a1',): 0.3, ('a2',): 0.8, ('a3',): -0.2, ('a4',): 0.0, ('a5',): 0.5,
             ('a6',): -1.0, ('a7',): 0.0, ('b1',): -0.5, ('b2',): -2.0, ('b3',): 0.0,
             ('c1',): 0.1, ('c2',): -0.3, ('c3',): 1.2, ('e',): 0.9}
    fake_world(tmp_path, 'sofc', gains)
    tables = p1.analyse([tmp_path])
    sofc = tables['sofc']
    # c1's 0.1 is in G; a cell with no gain is not (its gain does not exceed its SE of 0)
    assert set(sofc['G']) == {'a1', 'a2', 'a5', 'c1', 'c3', 'e'} and sofc['A*'] == 'a2a5'
    assert [p1.cell_name(c) for c, _ in sofc['next']] == ['a2a5']        # A* first
    # A* runs, gaining 1.1: A's best; then c3 (gain 1.2) is tried on it
    fake_world(tmp_path, 'sofc', {('a2', 'a5'): 1.1})
    sofc = p1.analyse([tmp_path])['sofc']
    assert sofc['A_best'] == 'a2a5' and [p1.cell_name(c) for c, _ in sofc['next']] == [
        'a2a5c3']
    # a2a5c3 gains 2.0 (kept); then e on it
    fake_world(tmp_path, 'sofc', {('a2', 'a5', 'c3'): 2.0})
    sofc = p1.analyse([tmp_path])['sofc']
    assert [p1.cell_name(c) for c, _ in sofc['next']] == ['a2a5c3e']
    # a2a5c3e gains 2.3: kept; c1 is skipped (E's head is in); 2.3 and 2.0 tie within 1 point
    fake_world(tmp_path, 'sofc', {('a2', 'a5', 'c3', 'e'): 2.3})
    sofc = p1.analyse([tmp_path])['sofc']
    assert sofc['chain'] == ['a2a5', 'a2a5c3', 'a2a5c3e']
    assert sofc['chain_end'] == 'a2a5c3e' and sofc['tie'] == ['a2a5c3e', 'a2a5c3']
    # the fake files already hold the extra units: the selection is made
    assert sofc['selected'] == 'a2a5c3e' and sofc['A_best'] == 'a2a5'
    assert 'mean' in sofc['cells']['a2a5c3e'] and sofc['cells']['a0']['gain'] == 0.0


def test_files_with_test_fields_or_other_settings_are_refused(tmp_path):
    job = p1.jobs_for('sofc', {'a1'}, ['seed0'])[0]
    args = nb.parse_args(job)
    good = {'screen': True, 'model': args.model, 'config': nb.stored_config(args),
            'runs': [{'run_id': 'seed0', 'seed': 0, 'val_f1': 70.0}]}
    p1.check_screen_file(good, 'g')
    p1.check_settings(good, job, 'g')
    with pytest.raises(ValueError, match='test fields'):
        p1.check_screen_file({**good, 'runs': [{'f1': 1.0}]}, 'x')
    with pytest.raises(ValueError, match='not a screen file'):
        p1.check_screen_file({**good, 'screen': False}, 'x')
    with pytest.raises(ValueError, match='settings'):
        p1.check_settings({**good, 'config': {**good['config'], 'lr': 5e-5}}, job, 'x')
    # P0's MuLMS files, by SHA-256: only their dev score is read
    assert p1.p0_dev(2) == json.loads((p1.GPU_DIR / p1.P0_MULMS_FILE.format(c=2)).read_text())[
        'runs'][0]['dev']['micro_f1']


def test_budget_cut_ladder():
    m = {'fine': {'a0': 0.12}, 'coarse': {'a0': 0.12}, 'sofc': {'a0': 0.36},
         'mulms': {'a0': 1.0}}
    small = p1.budget(m, {'b1': 100, 'b2': 40})
    assert small['cuts'] == [] and small['session_hours'] <= p1.CAP_SESSION_HOURS
    big = p1.budget({**m, 'mulms': {'a0': 20.0}}, {})
    assert big['cuts'][:3] == ['mulms: a2m', 'mulms: c3', 'mm: five_units']
    assert big['removed']['mulms'][:2] == ['a2m', 'c3'] and big['mm_five_units']


def test_p1_batches(tmp_path):
    lanes = kj.batch_lanes('p1t')
    flat = [j for lane in lanes for j in lane]
    assert lanes[0][0][-1] == 'smokebt' and lanes[0][1][0] == kj.DAPT_SCRIPT
    assert lanes[0][2][lanes[0][2].index('--model') + 1] == p1.B3_MODEL       # after the DAPT
    for job in flat:
        if job[0] == kj.DAPT_SCRIPT:
            continue
        if job[0] == p1.MULMS_NESTED_SCRIPT:
            mn.parse_args(job[1:])
        else:
            nb.parse_args(job)
    tags = [j[j.index('--tag') + 1] for j in flat if '--tag' in j]
    assert 'p1a0' in tags and 'p1a0u1' in tags
    assert all(t.startswith('p1t') for t in tags if t not in ('p1a0', 'p1a0u1', 'smokebt'))
    spec = kj.batch_spec('p1t')
    assert p1.SCIDEBERTA in spec['models'] and 'sofc_corpus_sha256' in spec
    entries = [{'benchmark': 'sofc', 'cell': 'b3', 'changes': ['b3'],
                'job': p1.jobs_for('sofc', {'b3'}, p1.UNITS['sofc'])[0]},
               {'benchmark': 'mulms', 'cell': 'e', 'changes': ['e'],
                'job': p1.jobs_for('mulms', {'e'}, ['c1s0'])[0]}]
    path = tmp_path / 'next.json'
    path.write_text(json.dumps(entries))
    lanes = kj.p1n_lanes(path)
    assert lanes[0][0][0] == kj.DAPT_SCRIPT and p1.B3_MODEL in lanes[0][1]
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec_ = importlib.util.spec_from_file_location('run_jobs', root / 'src/kaggle/run_jobs.py')
    rj = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(rj)
    assert rj.job_command(lanes[0][0])[1:] == lanes[0][0]           # the runner starts DAPT


def test_dapt_device_defaults_to_the_cpu():
    import dapt_mlm
    parser_args = dapt_mlm.parse_args(['--corpus', 'x.txt', '--out', 'y'])
    assert parser_args.device == 'cpu'


def test_p1t_checks(tmp_path):
    """p1t's checks pass on copies of the stored runs and fail on a changed one."""
    stored = json.loads((p1.GPU_DIR / p1.STORED_A0['sofc']).read_text())
    runs = [r for r in stored['runs'] if r['run_id'] in p1.UNITS['sofc']]
    name = p1.results_file('sofc', p1.jobs_for('sofc', frozenset(), p1.UNITS['sofc'])[0])
    (tmp_path / name).write_text(json.dumps({'runs': runs}))
    p0 = json.loads((p1.GPU_DIR / p1.P0_MULMS_FILE.format(c=1)).read_text())
    mulms = p1.results_file('mulms', p1.jobs_for('mulms', frozenset(), ['c1s0'])[0])
    (tmp_path / mulms).write_text(json.dumps(p0))
    smoke = json.loads(p1.SMOKE_REFERENCE.read_text())
    smoke['config']['tag'], smoke['variant'] = 'smokebt', 'crf_smokebt'
    smoke_path = tmp_path / 'ner_results_fine_random_split_vary_crf_smokebt.json'
    smoke_path.write_text(json.dumps(smoke))
    assert p1.p1t_checks(tmp_path) == {'smoke': [], 'sofc_a0': [], 'mulms_c1': []}
    runs[0] = {**runs[0], 'val_f1': runs[0]['val_f1'] + 0.01}
    (tmp_path / name).write_text(json.dumps({'runs': runs}))
    smoke['runs'][0]['f1'] += 1
    smoke_path.write_text(json.dumps(smoke))
    result = p1.p1t_checks(tmp_path)
    assert result['sofc_a0'] and result['smoke'] == ['runs'] and not result['mulms_c1']
