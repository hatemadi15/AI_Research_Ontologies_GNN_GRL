"""The SOTA plan's P0 (sota_p0.py, ner_baseline --sofc-cv): MatSciBERT's cross-validation
folds, the committed data files against their recomputation, the Welch test from summaries,
the gate's verdict, and the official split re-scored from the stored predictions."""
import json
import math
import random
import statistics

import pytest

import ner_baseline as nb
import sota_p0 as p0


def test_cv_folds_follow_matscibert():
    split = {f'A{i:02d}': 'train' for i in range(10)}
    split.update({f'B{i:02d}': 'dev' for i in range(3)})
    split.update({'T1': 'test', 'T2': 'test'})
    folds = nb.sofc_cv_folds(split)
    pool = sorted(d for d, p in split.items() if p != 'test')
    assert [name for name, *_ in folds] == ['cv1', 'cv2', 'cv3', 'cv4', 'cv5']
    for k, (name, train, dev, test) in enumerate(folds, 1):
        assert dev == pool[k - 1::5]                       # every 5th from position k - 1
        assert sorted(train + dev) == pool and test == ['T1', 'T2']
    assert sorted(d for *_, dev, _ in folds for d in dev) == pool    # each article once


def test_committed_data_files_are_current():
    folds = json.loads(p0.FOLDS_FILE.read_text())
    assert folds == json.loads(json.dumps(p0.cv_folds_record()))
    assert folds['sha256'] == p0.canonical_sha256(folds['folds'])
    record = folds['folds']
    assert all(len(f['test']) == 11 for f in record.values())
    assert sum(len(f['dev']) for f in record.values()) == 34
    configs = json.loads(p0.CONFIGS_FILE.read_text())
    assert configs['sha256'] == p0.canonical_sha256(configs['configurations'])
    for c in configs['configurations'].values():
        assert len(c['train']) + len(c['tune']) == 36 and len(c['dev']) == len(c['test']) == 7
    pytest.importorskip('pyarrow')
    try:
        fresh = p0.mulms_configurations_record()
    except (SystemExit, FileNotFoundError, OSError) as err:
        pytest.skip(f'the pinned MuLMS release is not cached: {err}')
    assert configs == json.loads(json.dumps(fresh))


def test_welch_from_summaries_matches_scipy():
    stats = pytest.importorskip('scipy.stats')
    rng = random.Random(1)
    a = [rng.gauss(84, 1) for _ in range(15)]
    b = [rng.gauss(85, 1.2) for _ in range(15)]
    summary = {x: {'mean': statistics.fmean(v), 'sd': statistics.stdev(v), 'n': len(v)}
               for x, v in (('a', a), ('b', b))}
    w = p0.welch_summary(summary['a'], summary['b'])
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert w['df'] == round(ref.df, 1)
    gap = statistics.fmean(a) - statistics.fmean(b)
    half = stats.t.ppf(0.975, ref.df) * gap / ref.statistic
    assert w['ci95'] == [round(gap - half, 2), round(gap + half, 2)]


def test_gate_fails_only_below():
    pytest.importorskip('scipy')
    published = {'mean': 80.0, 'sd_printed': 0.5, 'n': 5, 'decimals': 1,
                 'convention': 'population (np.std)'}
    # the printed population SD becomes a sample SD: 0.5 * sqrt(5 / 4)
    assert math.isclose(p0.sample_sd(published), 0.5 * math.sqrt(1.25))
    assert p0.sample_sd({**published, 'convention': 'sample (assumed)'}) == 0.5
    assert p0.gate([80.1, 79.9, 80.3, 79.7, 80.0], published)['verdict'] == 'passed'
    high = p0.gate([82.0, 82.5, 81.5, 82.2, 81.8], published)
    assert high['verdict'] == 'passed' and not high['borderline']
    low = p0.gate([77.0, 77.5, 76.5, 77.2, 76.8], published)
    assert low['verdict'] == 'failed' and low['welch']['ci95'][1] < 0
    assert low['published']['sd_sample'] == round(0.5 * math.sqrt(1.25), 3)
    # a verdict that flips within the printed SD's rounding (0.45 to 0.55) is borderline
    edge = None
    for shift in [x / 100 for x in range(50, 400)]:
        values = [80.0 - shift + d for d in (-0.2, -0.1, 0.0, 0.1, 0.2)]
        g = p0.gate(values, published)
        if g['borderline']:
            edge = g
            break
    assert edge is not None
    wide = p0.gate([75.0, 85.0, 70.0, 90.0, 80.0], published)
    assert wide['sd_warning'] and not p0.gate([80.0, 80.1, 79.9, 80.2, 79.8],
                                              published)['sd_warning']


def test_gate_inputs_are_checked():
    good = {'config': {'sofc_cv': True, 'crf': True, 'lr': 5e-5, 'batch_size': 16,
                       'epochs': 20, 'patience': 3, 'seed_offset': 100},
            'runs': [{'run_id': f'cv{k}', 'seed': 100} for k in range(1, 6)]}
    p0.check_cv_file(good, 100)
    with pytest.raises(ValueError, match='run_ids'):
        p0.check_cv_file({**good, 'runs': good['runs'][:4]}, 100)
    with pytest.raises(ValueError, match='lr'):
        p0.check_cv_file({**good, 'config': {**good['config'], 'lr': 3e-5}}, 100)
    with pytest.raises(ValueError, match='seed'):
        p0.check_cv_file(good, 200)
    import mulms_nested as mn
    config = {'lr': mn.SETTINGS['lr'], 'lr_crf': mn.SETTINGS['lr_crf'],
              'batch_size': mn.SETTINGS['batch_size'], 'epochs': mn.SETTINGS['epochs'],
              'patience': mn.SETTINGS['patience'], 'lr_schedule': 'none', 'max_train': 0,
              'seed': 0, 'seed_offset': 0, 'n_tags': 473}
    mulms = {'config': config, 'runs': [{'run_id': 'c2', 'seed': 0}]}
    p0.check_mulms_file(mulms, 2)
    with pytest.raises(ValueError):
        p0.check_mulms_file(mulms, 3)
    with pytest.raises(ValueError, match='lr_schedule'):
        p0.check_mulms_file({**mulms, 'config': {**config, 'lr_schedule': 'sqrt'}}, 2)


def test_smoke_comparison_ignores_run_times_and_tags():
    a = {'config': {'tag': 'smokebr', 'lr': 1e-4}, 'runs': [{'f1': 1.0, 'runtime_sec': 5}]}
    b = {'config': {'tag': 'smokebs', 'lr': 1e-4}, 'runs': [{'f1': 1.0, 'runtime_sec': 9}]}
    assert p0.smoke_identical(a, b) == []
    assert p0.smoke_identical(a, {**b, 'runs': [{'f1': 1.1, 'runtime_sec': 9}]}) == ['runs']


def test_sofc_cv_takes_one_seed_per_job():
    with pytest.raises(SystemExit):
        nb.parse_args(['--granularity', 'sofc', '--sofc-cv', '--crf', '--seeds', '0', '1'])
    args = nb.parse_args(['--granularity', 'sofc', '--sofc-cv', '--crf', '--seeds', '0'])
    assert nb.protocol_name(args) == 'matscibert_cv'


def test_official_split_rescored_from_stored_predictions():
    """The 11-article score re-computed from each run's saved predictions equals its stored
    F1 (rescore_official raises otherwise); the 10-article set drops one article's
    sentences."""
    pytest.importorskip('seqeval')
    if not (p0.GPU_DIR / p0.OFFICIAL_FILE).exists():
        pytest.skip('the stored official-split runs are not here')
    try:
        runs = p0.rescore_official()
    except SystemExit as err:
        pytest.skip(f'the SOFC-Exp corpus is not here: {err}')
    assert len(runs) == 10
    assert all(0 < r['n_sentences_10'] < 173 for r in runs)
    assert len({r['n_sentences_10'] for r in runs}) == 1


def test_p0a_batch():
    """P0's Kaggle batch: MuLMS's 5 configurations and SOFC-Exp's cross-validation at offsets
    0, 100 and 200, the smoke4 repeat first on lane 0, the two short MuLMS runs on different
    GPUs; every job parses with its script, and the runner starts mulms_nested.py itself."""
    import importlib.util
    from pathlib import Path

    import kaggle_jobs as kj
    import mulms_nested as mn
    lanes = kj.batch_lanes('p0a')
    flat = [j for lane in lanes for j in lane]

    def tag(job):
        return job[job.index('--tag') + 1]
    nested = [j for j in flat if j[0] == kj.MULMS_NESTED_SCRIPT]
    assert sorted(tag(j) for j in nested if '--max-train' not in j) == [
        'c1', 'c2', 'c3', 'c4', 'c5']
    assert sorted(tag(j) for j in flat if '--sofc-cv' in j) == ['m0', 'm100', 'm200']
    assert tag(lanes[0][0]) == 'smokebq' and tag(lanes[0][-1]) == 'smokebs'
    assert tag(lanes[1][0]) == 'smokebr'
    for job in nested:
        args = mn.parse_args(job[1:])
        assert args.tag == tag(job) and args.save_predictions
    for job in flat:
        if job[0] != kj.MULMS_NESTED_SCRIPT:
            nb.parse_args(job)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('run_jobs',
                                                  root / 'src' / 'kaggle' / 'run_jobs.py')
    rj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rj)
    assert rj.job_command(nested[0])[1:] == nested[0]
    assert nb.parse_result_name('ner_results_mulms_nested_configs_bilou_crf_c1.json') == (
        'mulms', 'nested_configs', 'bilou_crf_c1')
