"""Tests of the SOTA plan's P2 (ROADMAP §4, the P2 protocol): its jobs, statistics, counts, votes,
bootstrap, file checks and decisions, and ner_baseline.py's two options for it (--score-level
with --published-split, --multi-task with --sofc-cv)."""
import json
import random
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest

import kaggle_jobs as kj
import mulms_nested as mn
import ner_baseline as nb
import sota_p0
import sota_p1
import sota_p2 as p2


# ---------------------------------------------------------------------------------------------
# the jobs
# ---------------------------------------------------------------------------------------------
def all_jobs(cv_system='a6c1'):
    return {b: [j for _, j in p2.batch_jobs(b, cv_system)] for b in ('p2a', 'p2m', 'p2b')}


def seeds_of(job):
    i = job.index('--seeds')
    out = []
    for x in job[i + 1:]:
        if x.startswith('--'):
            break
        out.append(int(x))
    return out


def test_jobs_seeds_and_systems():
    jobs = all_jobs()
    tags = {j[j.index('--tag') + 1]: j for b in jobs.values() for j in b}
    # the selected MaterioMiner systems run at 30 seeds, the references and plain at 10
    assert seeds_of(tags['p2c1']) == list(p2.VOTE_SEEDS) and len(p2.VOTE_SEEDS) == 30
    assert '--score-level' in tags['p2c1'] and '--published-split' in tags['p2c1']
    assert seeds_of(tags['p2a3a6']) == list(p2.VOTE_SEEDS)
    # the CRF head runs at the same 30 seeds (its votes are decided against S's); A6 on fine
    a0 = [j for b in jobs.values() for j in b if j[j.index('--tag') + 1] == 'p2a0']
    assert len(a0) == 2 and all(seeds_of(j) == list(p2.VOTE_SEEDS) for j in a0)
    a6 = [j for j in jobs['p2a'] if j[j.index('--tag') + 1] == 'p2a6']
    assert len(a6) == 2 and all(seeds_of(j) == list(p2.SEEDS) for j in a6)
    assert {'--reinit-top', '--epochs', '--patience'} <= set(tags['p2a3a6'])
    b1 = tags['p2b1']
    assert seeds_of(b1) == list(p2.SEEDS) and p2.DEBERTA in b1 and '1e-5' in b1
    assert '--gradient-checkpointing' in b1 and '8' in b1
    # every test-scored job runs at fresh seeds (1000 up); only the checks and A6's screen
    # extras do not
    for tag, job in tags.items():
        if tag == 'smokebv':
            continue
        if job[0] == p2.MULMS_NESTED_SCRIPT:
            assert int(job[job.index('--seed') + 1]) >= 1000
        elif '--sofc-cv' in job and tag != 'p2chkcv':
            assert int(job[job.index('--seed-offset') + 1]) >= 1000 and seeds_of(job) == [0]
        elif tag in ('p2chk', 'smokebu', 'p1a6', 'smokebv', 'p2chkcv'):
            assert '--screen' in job or tag != 'p1a6'
        else:
            assert min(seeds_of(job)) >= 1000, tag
    mulms = [j for j in jobs['p2m'] if j[0] == p2.MULMS_NESTED_SCRIPT and j != p2.MULMS_SMOKE_JOB]
    assert len(mulms) == 20
    assert sum('--lr' in j and '5e-5' in j for j in mulms) == 10
    assert len([j for j in jobs['p2b'] if '--sofc-cv' in j and j != p2.CV_CHECK_JOB]) == 20


def test_jobs_parse_and_write_distinct_files():
    """No two jobs write one file, within a batch or across batches (smoke4's repeat runs in
    each batch, a session of its own)."""
    every = []
    for batch, jobs in all_jobs().items():
        names = []
        for job in jobs:
            if job[0] == p2.MULMS_NESTED_SCRIPT:
                mn.parse_args(job[1:])
                names.append(p2.mulms_file(job))
            else:
                args = nb.parse_args(job)
                grans = (nb.head_granularities(args).values()
                         if getattr(args, 'multi_task', False) and not args.screen_level
                         else [args.screen_level or args.granularity])
                names += [p2.tagger_file(job, g) for g in grans]
        assert len(names) == len(set(names)), batch
        every += [n for n in names if 'smokebu' not in n]
    assert len(every) == len(set(every))


def test_cv_reference_and_batches():
    assert p2.cv_reference('a6') == 'a0'
    assert p2.cv_reference('a6c1') == 'a6' and p2.cv_reference('c1') == 'a6'
    with pytest.raises(SystemExit):
        p2.batch_jobs('p2b')
    cells = {j[j.index('--tag') + 1][:-5] for _, j in p2.batch_jobs('p2b', 'a6')
             if '--sofc-cv' in j and j != p2.CV_CHECK_JOB}
    assert cells == {'p2a6', 'p2a0'}
    # the checks have no minutes: kaggle_jobs puts them at the lanes' heads
    assert [j for m, j in p2.batch_jobs('p2a') if m == 0] == [
        p2.SMOKE_JOB, p2.CHECK_JOBS['fine'], p2.CHECK_JOBS['sofc']]
    assert [j for m, j in p2.batch_jobs('p2m') if m == 0] == [p2.SMOKE_JOB, p2.MULMS_SMOKE_JOB]
    assert [j for m, j in p2.batch_jobs('p2b', 'a6') if m == 0] == [p2.SMOKE_JOB,
                                                                 p2.CV_CHECK_JOB]


def test_p2_lanes(tmp_path):
    path = tmp_path / 'jobs.json'
    path.write_text(json.dumps([{'minutes': m, 'job': j} for m, j in p2.batch_jobs('p2a')]))
    lanes = kj.p2_lanes('p2a', path)
    assert lanes[0][0] == p2.SMOKE_JOB
    flat = [j for lane in lanes for j in lane]
    assert sorted(map(tuple, flat)) == sorted(tuple(j) for _, j in p2.batch_jobs('p2a'))
    assert 'p2a' in kj.BATCHES and kj.batch_extras('p2a') == kj.P1_PINS


def test_cv_system_tie_rule(monkeypatch):
    values = {('a6', 'c1'): 79.1, ('c1',): 79.1, ('a6',): 78.0}

    def fake(benchmark, changes, units, dirs):
        return {u: (values[tuple(sorted(changes))], None, None) for u in units}
    monkeypatch.setattr(sota_p1, 'read_cell', fake)
    record = p2.cv_system_record()
    # a tie in the mean goes to fewer changes
    assert record['cv_system'] == 'c1' and record['reference'] == 'a6'
    values[('a6',)] = 80.0
    record = p2.cv_system_record()
    assert record['cv_system'] == 'a6' and record['reference'] == 'a0'

    def missing(benchmark, changes, units, dirs):
        return {u: (78.0, None, None) for u in units[:5]}
    monkeypatch.setattr(sota_p1, 'read_cell', missing)
    with pytest.raises(SystemExit):
        p2.cv_system_record()


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------
def test_paired_welch_one_sample_against_scipy():
    stats = pytest.importorskip('scipy.stats')
    rng = random.Random(1)
    a = {k: 80 + rng.gauss(0, 1) for k in range(10)}
    b = {k: 79 + rng.gauss(0, 1) for k in range(10)}
    rec = p2.paired(a, b)
    ci = stats.ttest_rel(list(a.values()), list(b.values())).confidence_interval()
    assert rec['ci95'] == [round(ci.low, 2), round(ci.high, 2)] and rec['n'] == 10
    w = p2.welch(list(a.values()), list(b.values()))
    ci = stats.ttest_ind(list(a.values()), list(b.values()), equal_var=False).confidence_interval()
    assert w['ci95'] == [round(ci.low, 2), round(ci.high, 2)]
    o = p2.one_sample(list(a.values()), 79.5)
    ci = stats.ttest_1samp([v - 79.5 for v in a.values()], 0).confidence_interval()
    assert o['ci95'] == [round(ci.low, 2), round(ci.high, 2)]
    with pytest.raises(ValueError):
        p2.paired(a, {k: v for k, v in b.items() if k})


def test_against_published_uses_p0_conventions():
    values = [84.0, 84.5, 83.8, 84.9, 84.2, 84.1, 84.6, 83.9, 84.3, 84.4]
    rec = p2.against_published(values, 'sofc_official_10')
    assert rec['published']['sd_sample'] == round(0.9 * (5 / 4) ** 0.5, 3)
    assert rec['passed'] == (rec['ci95'][0] > 0) and 'borderline' in rec
    rec = p2.against_published(values * 5, 'sofc_cv')
    assert 'population_reading' in rec and rec['published']['sd_sample'] == 1.02


# ---------------------------------------------------------------------------------------------
# per-sentence counts, votes, bootstrap
# ---------------------------------------------------------------------------------------------
def random_tags(rng, n, types=('MAT', 'VAL')):
    return [rng.choice(['O', 'O', *(f'{p}-{t}' for t in types for p in 'BI')]) for _ in range(n)]


def test_tagger_counts_equal_seqeval():
    from seqeval.metrics import f1_score
    rng = random.Random(0)
    for _ in range(20):
        gold = [random_tags(rng, rng.randint(1, 12)) for _ in range(15)]
        pred = [random_tags(rng, len(g)) for g in gold]
        counts = p2.tagger_counts(gold, pred)
        assert round(p2.micro_f1(counts), 6) == round(100 * f1_score(gold, pred), 6)


def test_mulms_counts_equal_the_release_scorer():
    gold = [{('MAT', 0, 1), ('NUM', 2, 2)}, {('MAT', 0, 0)}, set()]
    pred = [[['MAT', 0, 1]], [['MAT', 0, 0], ['NUM', 1, 1]], [['MAT', 3, 3]]]
    run = {'run_id': 'c1', 'test_pred': pred,
           'test': mn.score([{tuple(e) for e in p} for p in pred], gold)}
    counts = p2.mulms_counts(run, gold)
    assert counts == [(1, 0, 1), (1, 1, 0), (0, 1, 0)]
    run['test']['micro_f1'] += 1
    with pytest.raises(ValueError):
        p2.mulms_counts(run, gold)


def test_votes_are_disjoint_majorities():
    gold = 'B-MAT O B-VAL'
    preds = {1000: 'B-MAT O O', 1100: 'B-MAT O B-VAL', 1200: 'O O B-VAL',
             1300: 'O O O', 1400: 'O O O', 1500: 'O O O'}
    runs = {s: {'test_pred': [p], 'test_gold': [gold]} for s, p in preds.items()}
    votes = p2.vote_runs(runs, 3)
    assert [v[0] for v in votes] == [[1000, 1100, 1200], [1300, 1400, 1500]]
    assert votes[0][2] == [['B-MAT', 'O', 'B-VAL']] and votes[1][2] == [['O', 'O', 'O']]
    assert len(p2.vote_runs({s: runs[s] for s in list(runs)[:5]}, 3)) == 1


def test_bootstrap_is_seeded_and_paired():
    rng = random.Random(3)
    a = [[(rng.randint(0, 4), rng.randint(0, 2), rng.randint(0, 2)) for _ in range(30)]
         for _ in range(4)]
    assert p2.bootstrap(a, a, n_boot=200) == [0.0, 0.0]
    b = [[(max(0, tp - 1), fp + 1, fn) for tp, fp, fn in run] for run in a]
    ci = p2.bootstrap(a, b, n_boot=200)
    assert ci == p2.bootstrap(a, b, n_boot=200) and ci[0] > 0
    with pytest.raises(ValueError):
        p2.bootstrap(a, [run[:10] for run in b], n_boot=10)


# ---------------------------------------------------------------------------------------------
# file checks
# ---------------------------------------------------------------------------------------------
def p2_file(job, run_ids, env):
    config = nb.stored_config(nb.parse_args(job))
    return {'config': config, 'runs': [{'run_id': r, 'environment': env, 'test_index': [1],
                                        'test_gold': ['O']} for r in run_ids]}


def test_check_p2_file():
    env = dict(json.loads((p2.GPU_DIR / sota_p1.P0_MULMS_FILE.format(c=1)).read_text())
               ['runs'][0]['environment'])
    job = p2.official_job('a6')
    ids = [f'seed{s}' for s in p2.SEEDS]
    ref = {'runs': [{'test_index': [1], 'test_gold': ['O']}]}
    assert set(p2.check_p2_file(p2_file(job, ids, env), job, 'f', ids, ref)) == set(ids)
    with pytest.raises(ValueError):
        p2.check_p2_file({**p2_file(job, ids, env), 'screen': True}, job, 'f', ids)
    with pytest.raises(ValueError):
        p2.check_p2_file(p2_file(job, ids[:-1], env), job, 'f', ids)
    with pytest.raises(ValueError):
        p2.check_p2_file(p2_file(p2.official_job('b1'), ids, env), job, 'f', ids)
    with pytest.raises(ValueError):
        p2.check_p2_file(p2_file(job, ids, {**env, 'device': 'Tesla P100'}), job, 'f', ids)
    with pytest.raises(ValueError):
        p2.check_p2_file(p2_file(job, ids, env), job, 'f', ids,
                         {'runs': [{'test_index': [2], 'test_gold': ['O']}]})


# ---------------------------------------------------------------------------------------------
# the decisions, on synthetic values
# ---------------------------------------------------------------------------------------------
def tagger_run(seed, f1):
    return {'run_id': f'seed{seed}', 'f1': f1, 'test_gold': ['B-MAT O'], 'test_pred': ['B-MAT O']}


def test_decide_counts_and_wordings(monkeypatch, tmp_path):
    rng = random.Random(5)
    monkeypatch.setattr(p2, 'official_values', lambda cell, dirs: {
        s: {'f1_11': (85.0 if cell == 'b1' else 83.5) + rng.gauss(0, .3),
            'f1_10': (85.0 if cell == 'b1' else 83.5) + rng.gauss(0, .3),
            'run': tagger_run(s, 100.0)} for s in p2.SEEDS})
    monkeypatch.setattr(p2, 'cv_values', lambda cell, dirs: {
        (f, o): {'f1': 84.0 + rng.gauss(0, .5), **tagger_run(o, 100.0)}
        for f in p2.CV_FOLDS for o in p2.CV_OFFSETS})
    monkeypatch.setattr(p2, 'mulms_values', lambda cell, dirs: {
        u: {'test': {'micro_f1': (80.5 if cell == 'a1m' else 79.0) + rng.gauss(0, .2)}}
        for u in p2.MULMS_UNITS})
    monkeypatch.setattr(p2, 'released_values', lambda level, cell, dirs: {
        s: {**tagger_run(s, 100.0), 'f1': 74.0 + rng.gauss(0, 1)}
        for s in p2.mm_seeds(level, cell)})
    cv = tmp_path / 'cv.json'
    cv.write_text(json.dumps({'cv_system': 'a6c1', 'reference': 'a6'}))
    monkeypatch.setattr(p2, 'CV_SYSTEM_FILE', cv)
    record = p2.decide(with_bootstrap=False)
    assert record['count']['live'] == 15 and record['count']['re_tests'] == 4
    assert [r['number'] for r in record['live']] == list(range(1, 16))
    assert record['count']['expected_false_passes_at_most'] == 0.375
    # B1 is set well above 83.4 but not above 86.0: decision 1 passes, gate 3 does not
    one, three = record['live'][0], record['live'][2]
    assert one['passed'] and not three['passed']
    assert one['wording'].startswith('above the best result with a stated protocol')
    assert set(record['selected_means']) >= {'MuLMS (A1m)', 'MaterioMiner fine (c1)'}
    words = {r['decision']: r['wording'] for r in record['live']}
    assert any('MuLMS, D-T' in d and w.startswith('above the published') for d, w in words.items())
    assert all(r['passed'] == (r['ci95'][0] > 0) for r in record['live'] + record['re_tests'])
    assert len(record['reported']['official_votes']['vote_of_3']) == 3


# ---------------------------------------------------------------------------------------------
# ner_baseline.py's options for P2
# ---------------------------------------------------------------------------------------------
def test_score_level_and_multitask_cv_flags():
    ok = nb.parse_args(['--granularity', 'both', '--multi-task', '--score-level', 'fine', '--crf',
                        '--published-split', '--seeds', '1000', '--tag', 'p2c1'])
    assert ok.screen_level == 'fine' and nb.protocol_name(ok) == 'published_split'
    for bad in (['--granularity', 'both', '--multi-task', '--crf', '--published-split'],
                ['--granularity', 'both', '--multi-task', '--score-level', 'fine', '--crf'],
                ['--granularity', 'sofc', '--multi-task', '--crf', '--sofc-cv'],
                ['--granularity', 'sofc_both', '--crf', '--sofc-cv']):
        with pytest.raises(SystemExit):
            nb.parse_args(bad + ['--seeds', '0'])
    cv = nb.parse_args(['--granularity', 'sofc_both', '--multi-task', '--crf', '--sofc-cv',
                        '--seeds', '0', '--seed-offset', '1000', '--tag', 'p2c1m1000'])
    assert nb.protocol_name(cv) == 'matscibert_cv'


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    path = tmp_path_factory.mktemp('tiny_bert_p2')
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


def test_one_level_multitask_scores_its_test_set(tiny):
    """--score-level outside screen mode: the level's head is tested, the other head's tags are
    dropped where they must be, and the other head gets no record."""
    fine = toy_part(12, 0)
    coarse = [(w, [t.replace('MAT', 'M').replace('VAL', 'V') for t in g], d) for w, g, d in fine]
    val, test = toy_part(4, 3, 'p2'), toy_part(4, 4, 'p3')
    parts = {'fine': (fine, val, test),
             'coarse': ([(w, None if k % 3 == 0 else g, d)
                         for k, (w, g, d) in enumerate(coarse)], [], [])}
    args = nb.parse_args(['--granularity', 'both', '--multi-task', '--crf', '--published-split',
                          '--score-level', 'fine', '--seeds', '0', '--model', tiny, '--epochs',
                          '2', '--batch-size', '4', '--save-predictions'])
    records = nb.train_multitask(0, parts, {'fine': ['MAT', 'VAL'], 'coarse': ['M', 'V']},
                                 args, 'cpu', test_index=list(range(4)), scored=('fine',))
    assert list(records) == ['fine']
    rec = records['fine']
    assert 'f1' in rec and rec['n_test'] == 4 and len(rec['test_pred']) == 4
    assert rec['multi_task']['other_tags_dropped'] == 4
    assert rec['multi_task']['other_best_epoch'] is None


def test_both_heads_record_unchanged(tiny):
    """Without --score-level the multi-task record keeps its old fields."""
    fine = toy_part(8, 0)
    coarse = [(w, [t.replace('MAT', 'M').replace('VAL', 'V') for t in g], d) for w, g, d in fine]
    val, test = toy_part(4, 3, 'p2'), toy_part(4, 4, 'p3')
    cval = [(w, [t.replace('MAT', 'M').replace('VAL', 'V') for t in g], d) for w, g, d in val]
    ctest = [(w, [t.replace('MAT', 'M').replace('VAL', 'V') for t in g], d) for w, g, d in test]
    parts = {'fine': (fine, val, test), 'coarse': (coarse, cval, ctest)}
    args = nb.parse_args(['--granularity', 'both', '--multi-task', '--crf', '--seeds', '0',
                          '--model', tiny, '--epochs', '2', '--batch-size', '4'])
    records = nb.train_multitask(0, parts, {'fine': ['MAT', 'VAL'], 'coarse': ['M', 'V']},
                                 args, 'cpu')
    assert set(records) == {'fine', 'coarse'}
    assert set(records['fine']['multi_task']) == {'n_examples', 'n_split_examples',
                                                  'other_best_epoch'}
    assert isinstance(records['fine']['multi_task']['other_best_epoch'], int)


def test_gated_wordings():
    assert p2.wording('gated', True, {'gate_ok': False, 'target': 86.0}).startswith('not taken')
    # the 86.0 wording needs decision 1 and its gate
    assert 'every reported result' in p2.wording('official_dt', True, {'gate_passed': True})
    assert 'every reported' not in p2.wording('official_dt', True, {'gate_passed': False})
    assert p2.wording('official_dt', False, {'gate_passed': True}).startswith('not shown')
    assert p2.wording('dp_tuning', True).startswith('tuning improves on our re-run')
    assert all(w.endswith('on this test set') for w in (
        p2.wording('dp', True), p2.wording('mm_live', False), p2.wording('vote', True, {'k': 3})))


def test_borderline_varies_mean_and_sd(monkeypatch):
    # a set of runs whose Welch interval against 79.6 just reaches 0: borderline
    rng = random.Random(2)
    values = [80.25 + rng.gauss(0, 0.3) for _ in range(10)]
    rec = p2.against_published(values, 'mulms_nested')
    lo = rec['ci95'][0]
    assert rec['borderline'] == (abs(lo) < 0.2) or rec['borderline'] in (True, False)
    far = p2.against_published([v + 5 for v in values], 'mulms_nested')
    assert far['passed'] and not far['borderline']


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_batch_checks(tmp_path):
    pulled = tmp_path / 'pulled'
    smoke = json.loads(sota_p1.SMOKE_REFERENCE.read_text())
    smoke['config']['tag'] = 'smokebu'
    smoke['variant'] = smoke['variant'].replace('smokebn', 'smokebu')
    write(pulled / 'ner' / 'ner_results_fine_random_split_vary_crf_smokebu.json', smoke)
    for g in ('fine', 'sofc'):
        stored = json.loads((p2.GPU_DIR / p2.STORED_A0[g]).read_text())
        write(pulled / 'ner' / f'ner_results_{g}_published_split_crf_p2chk.json',
              {'runs': [r for r in stored['runs'] if r['run_id'] == 'seed0']})
    result = p2.batch_checks(pulled, 'p2a')
    assert result == {'smoke': [], 'a0_repeat_fine': [], 'a0_repeat_sofc': []}
    bad = json.loads((pulled / 'ner' / 'ner_results_fine_published_split_crf_p2chk.json')
                     .read_text())
    bad['runs'][0]['f1'] += 0.01
    write(pulled / 'ner' / 'ner_results_fine_published_split_crf_p2chk.json', bad)
    assert p2.batch_checks(pulled, 'p2a')['a0_repeat_fine'] == ['seed0: f1 differs']
    assert p2.batch_checks(pulled, 'p2m')['mulms_smoke']
    assert p2.batch_checks(pulled, 'p2b')['cv_repeat']


def test_take_a6_screen_needs_all_nine_units(tmp_path):
    pulled = tmp_path / 'pulled'
    data = json.loads((p2.SCREEN_DIR / p2.A6_SCREEN_FILE).read_text())
    write(pulled / 'ner' / p2.A6_SCREEN_FILE, data)
    with pytest.raises(SystemExit):           # the stored file has the 5 base units only
        p2.take_a6_screen(pulled, tmp_path / 'screen')
