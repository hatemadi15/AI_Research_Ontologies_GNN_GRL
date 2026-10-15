"""Kaggle GPU runs (kaggle_jobs.py, src/kaggle/run_jobs.py) and the CUDA settings of
ner_baseline.py. No network, no GPU and no torch: the Kaggle CLI is never called."""
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

import kaggle_jobs as kj
import ner_baseline as nb

REPO = Path(__file__).resolve().parents[1]
GPU_ENV = {'device': 'Tesla T4', 'cuda': '12.6', 'cudnn': 90100, 'torch': '2.9.0',
           'transformers': '5.18.0', 'deterministic': True, 'cublas_workspace_config': ':4096:8'}


def _run_jobs():
    spec = importlib.util.spec_from_file_location('run_jobs', REPO / 'src' / 'kaggle' / 'run_jobs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _written_files(job):
    """(granularity, protocol, variant) of every results file that a job writes."""
    args = nb.parse_args(job)
    protocol, variant = nb.protocol_name(args), nb.variant_name(args)
    granularities = ['fine', 'coarse'] if args.granularity == 'both' else [args.granularity]
    return {(g, protocol, variant) for g in granularities}


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------
def test_item1_is_the_crf_reference_and_multitask_on_both_protocols():
    lanes = kj.batch_lanes('item1')
    assert len(lanes) == kj.LANES == 2
    jobs = [j for lane in lanes for j in lane]
    files = [f for j in jobs for f in _written_files(j)]
    # every results file is written by one job only, so the lanes never write the same file
    assert len(files) == len(set(files))
    variants = {v for _, _, v in files}
    assert variants == {'crf_m0', 'crf_m100', 'crf_m200', 'crf_mt', 'crf_mt_m100', 'crf_mt_m200'}
    assert len(files) == 24          # 6 variants x 2 granularities x 2 protocols
    for j in jobs:
        args = nb.parse_args(j)
        assert args.crf and args.save_predictions and args.threads == 2
        assert args.seeds == [0, 1, 2, 3, 4]
        tag = args.tag or 'm0'
        assert args.seed_offset == int(tag[1:])
    # balanced: neither lane has more than 60% of the estimated time
    assert max(len(lane) for lane in lanes) - min(len(lane) for lane in lanes) <= 4


def test_item2_and_item12():
    lanes = kj.batch_lanes('item2')
    files = [f for lane in lanes for j in lane for f in _written_files(j)]
    assert len(files) == len(set(files)) == 24
    assert {v for _, _, v in files} == {'crf_ls_m0', 'crf_ls_m100', 'crf_ls_m200',
                                        'crf_mt_ls_m0', 'crf_mt_ls_m100', 'crf_mt_ls_m200'}
    # item12: every lane runs its item1 jobs first, then its item2 jobs
    both = kj.batch_lanes('item12')
    assert both == [a + b for a, b in zip(kj.batch_lanes('item1'), lanes)]
    files = [f for lane in both for j in lane for f in _written_files(j)]
    assert len(files) == len(set(files)) == 48


def test_item3_runs_self_training_by_granularity_lane(monkeypatch):
    """Week-3 self-training (3 seeds) and the 3-teacher runs paper by paper, a lane per
    granularity, so that the jobs that write a file share a lane."""
    monkeypatch.setattr(nb.os.path, 'isfile', lambda path: True)    # the corpus is not here
    lanes = kj.batch_lanes('item3')
    for g, lane in zip(('fine', 'coarse'), lanes):
        args = [nb.parse_args(j) for j in lane]
        assert {a.granularity for a in args} == {g}
        assert all(a.leave_one_paper_out and a.self_train == kj.CORPUS for a in args)
        assert [nb.variant_name(a) for a in args[:3]] == ['crf_st_m0', 'crf_st_m100', 'crf_st_m200']
        three = args[3:]
        assert [a.papers for a in three] == [[p] for p in kj.PAPERS]
        assert all(a.teachers == 3 and a.self_train_sample == 0 for a in three)
    lanes = kj.batch_lanes('item3r')
    for g, lane in zip(('fine', 'coarse'), lanes):
        (args,) = [nb.parse_args(j) for j in lane]
        assert args.vary_split and args.teachers == 3 and args.seeds == [0, 1, 2, 3, 4]
        assert args.granularity == g


def test_smoke1_repeats_one_crf_run_on_both_gpus():
    lanes = kj.batch_lanes('smoke1')
    crf = [j for lane in lanes for j in lane if '--multi-task' not in j
           and '--label-semantics' not in j]
    assert [j[j.index('--tag') + 1] for j in crf] == ['smokea', 'smokeb', 'smokec']
    strip = [[a for a in j if not a.startswith('smoke')] for j in crf]
    assert strip[0] == strip[1] == strip[2]       # the same run three times
    assert any('--multi-task' in j for j in lanes[0])
    assert any('--label-semantics' in j for j in lanes[1])
    for j in (j for lane in lanes for j in lane):
        assert 'smoke' in nb.variant_name(nb.parse_args(j))   # never collected


def test_week5_batches_add_seeds_300_to_900(monkeypatch):
    """Week 5: smoke4, then w5a (self-training, 10 students) and w5b (crf, crf_ls and
    crf_mt at seed offsets 300 to 900), each opened by a repeat of a week-4 CRF run."""
    monkeypatch.setattr(nb.os.path, 'isfile', lambda path: True)    # the corpus is not here
    assert kj.OFFSETS_W5 == (300, 400, 500, 600, 700, 800, 900)
    repeat = ['--granularity', 'fine', '--vary-split', '--seeds', '0', '--crf',
              '--save-predictions', '--threads', '2']
    # smoke4: the CRF run of split 0 (the stored crf_m0 run, if the commit repeats it) and a
    # tiny 3-teacher run with 4 students; neither is ever collected
    crf_lane, st_lane = kj.batch_lanes('smoke4')
    assert crf_lane == [repeat + ['--tag', 'smokee']]
    (tiny,) = [nb.parse_args(j) for j in st_lane]
    assert (tiny.teachers, tiny.students, tiny.self_train_sample, tiny.max_train) == (3, 4, 200, 32)
    for job in crf_lane + st_lane:
        args = nb.parse_args(job)
        assert 'smoke' in nb.results_path('', args.granularity, args)
    # w5a: a lane per granularity, the held-out papers first, every student seed 0 to 900
    lanes = kj.batch_lanes('w5a')
    for g, lane in zip(('fine', 'coarse'), lanes):
        args = [nb.parse_args(j) for j in lane]
        assert [nb.protocol_name(a) for a in args] == ['leave_one_paper_out', 'random_split_vary']
        assert all(a.granularity == g and a.teachers == 3 and a.students == 10 and
                   a.self_train_sample == 0 and a.self_train == kj.CORPUS and a.save_predictions
                   for a in args)
        assert all(nb.variant_name(a) == 'crf_st3' for a in args)
    # w5b: the repeat first, then every new seed of the three families, one file per job
    lanes = kj.batch_lanes('w5b')
    assert lanes[0][0] == repeat + ['--tag', 'smokef']
    jobs = [j for lane in lanes for j in lane][1:]
    assert len(jobs) == 7 * 2 * 2 * 2 + 7 * 2          # crf and crf_ls; crf_mt (both heads)
    files = [f for j in jobs for f in _written_files(j)]
    assert len(files) == len(set(files)) == 7 * 2 * 2 * 3
    assert {v for _, _, v in files} == {f'{s}_m{n}' for s in ('crf', 'crf_ls', 'crf_mt')
                                        for n in kj.OFFSETS_W5}
    for j in jobs:
        args = nb.parse_args(j)
        assert args.seed_offset == int(args.tag[1:]) >= 300 and args.save_predictions
    assert abs(len(lanes[0]) - len(lanes[1])) <= 4
    assert {'smoke4', 'w5a', 'w5b'} <= set(kj.BATCHES)


def test_balance_takes_the_longest_job_first_to_the_least_loaded_lane():
    # b (5) -> lane 0; c (3) -> lane 1; d (2) -> lane 1 (5 each); a (1) -> lane 0 (tie: lowest)
    lanes = kj.balance([(1, ['a']), (5, ['b']), (3, ['c']), (2, ['d'])])
    assert lanes == [[['b'], ['a']], [['c'], ['d']]]


def test_unknown_batch():
    with pytest.raises(ValueError):
        kj.batch_lanes('nope')


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
def test_render_script_puts_the_spec_into_run_jobs():
    spec = {'batch': 'x', 'gpu': kj.GPU, 'lanes': [[['--crf', "it's \"quoted\""]]]}
    text = kj.render_script(spec)
    namespace = {'__name__': 'run_jobs_test'}
    exec(compile(text, 'run_jobs.py', 'exec'), namespace)
    assert namespace['JOBS'] == spec
    with pytest.raises(ValueError):
        kj.render_script(spec, template='no placeholder here')


def test_kernel_metadata_is_a_private_t4_gpu_kernel_with_the_bundle():
    meta = kj.kernel_metadata('someone')
    assert meta['id'] == 'someone/ner-gpu-jobs' and meta['title'] == 'ner-gpu-jobs'
    assert meta['is_private'] and meta['enable_gpu'] and meta['enable_internet']
    assert not meta['enable_tpu']
    assert meta['machine_shape'] == 'NvidiaTeslaT4'
    assert meta['dataset_sources'] == ['someone/ner-gpu-bundle']
    assert meta['docker_image_pinning_type'] == 'original'
    assert meta['docker_image'] == kj.DOCKER_IMAGE and '@sha256:2757e0c7' in kj.DOCKER_IMAGE
    assert kj.kernel_metadata('someone', 'ner-gpu-jobs-b')['id'] == 'someone/ner-gpu-jobs-b'
    assert meta['code_file'] == 'run_jobs.py' and meta['kernel_type'] == 'script'
    assert kj.dataset_metadata('someone')['id'] == 'someone/ner-gpu-bundle'


def test_pins_match_the_cpu_runs():
    assert 'transformers==5.18.0' in kj.PIP
    # seqeval is source-only on PyPI: it goes into the bundle as a wheel, not through pip
    assert kj.WHEELS == ('seqeval==1.2.2',) and not any('seqeval' in p for p in kj.PIP)
    assert kj.batch_spec('smoke1')['wheels'] == ['seqeval-1.2.2-py3-none-any.whl']
    assert kj.MODELS == {'m3rg-iitd/matscibert': 'ced9d8f5f208712c4a90f98a246fe32155b29995'}
    assert kj.corpus_sha256() == ('7eaf1e79f7d43e52d7ef96c8afbeddede0c9b8ca71a3b1e9cf7a64d59eea3162')


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------
def test_build_bundle(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('One sentence.\nAnother one.\n')
    results = tmp_path / 'gpu'
    results.mkdir()
    (results / 'ner_results_fine_random_split_vary_crf_m0.json').write_text(
        json.dumps(_gpu_file('seed0')))
    # a vote made from GPU runs has no environment: not needed to resume runs, left out
    (results / 'ner_results_fine_random_split_vary_crf_ens3.json').write_text(
        json.dumps(_gpu_file('seed0', env=None)))
    (results / 'notes.txt').write_text('not a results file')
    wheel = tmp_path / 'seqeval-1.2.2-py3-none-any.whl'
    wheel.write_bytes(b'PK')
    out = tmp_path / 'bundle'
    info = kj.build_bundle('HEAD', out, results_dir=results, corpus=corpus,
                           expected_sha256=kj.file_sha256(corpus), wheels=[wheel])
    assert sorted(p.name for p in out.iterdir()) == [
        'BUNDLE.json', 'code.tar.gz', 'fatigue_sentences.txt',
        'gpu__ner_results_fine_random_split_vary_crf_m0.json', 'seqeval-1.2.2-py3-none-any.whl']
    assert info['wheels'] == ['seqeval-1.2.2-py3-none-any.whl']
    assert json.loads((out / 'BUNDLE.json').read_text()) == info
    assert len(info['commit']) == 40
    assert info['results'] == ['ner_results_fine_random_split_vary_crf_m0.json']
    with tarfile.open(out / 'code.tar.gz') as tar:
        assert 'src/graph/ner_baseline.py' in tar.getnames()
    with pytest.raises(SystemExit):
        kj.build_bundle('HEAD', out, results_dir=results, corpus=corpus, expected_sha256='0' * 64)


# ---------------------------------------------------------------------------
# Collecting results
# ---------------------------------------------------------------------------
def _gpu_file(*ids, env=GPU_ENV, f1=70.0):
    runs = [{'run_id': i, 'f1': f1} for i in ids]
    for r in runs:
        if env is not None:
            r['environment'] = dict(env)
    return {'runs': runs}


def test_check_collect():
    assert kj.check_collect(_gpu_file('seed0', 'seed1')) == []
    assert 'runs without an environment' in kj.check_collect(_gpu_file('seed0', env=None))[0]
    other = {**GPU_ENV, 'torch': '2.10.0'}
    key = kj.environment_key({'environment': other})
    assert 'more than one environment' in kj.check_collect(_gpu_file('seed0'), None, {key})[0]
    old = _gpu_file('seed0', 'seed1')
    assert kj.check_collect(_gpu_file('seed0', 'seed1', 'seed2'), old) == []
    assert kj.check_collect(_gpu_file('seed0'), old) == [
        'run seed1 of the file in results/gpu is missing or changed']
    assert kj.check_collect(_gpu_file('seed0', 'seed1', f1=71.0), old)[0].startswith('run seed0')


def test_collect_copies_gpu_results_and_skips_smoke_tests(tmp_path):
    pulled = tmp_path / 'pulled' / 'processed' / 'ner'
    pulled.mkdir(parents=True)
    good = 'ner_results_fine_random_split_vary_crf_m0.json'
    (pulled / good).write_text(json.dumps(_gpu_file('seed0')))
    (pulled / 'ner_results_fine_random_split_vary_crf_smokea.json').write_text(
        json.dumps(_gpu_file('seed0')))
    (pulled / 'ner_results_coarse_random_split_vary_crf_m0.json').write_text(
        json.dumps(_gpu_file('seed0', env=None)))
    dest = tmp_path / 'results_gpu'
    copied, problems = kj.collect(tmp_path / 'pulled', dest)
    assert copied == [good]
    assert list(problems) == ['ner_results_coarse_random_split_vary_crf_m0.json']
    assert sorted(p.name for p in dest.iterdir()) == [good]


# ---------------------------------------------------------------------------
# run_jobs.py (the kernel script)
# ---------------------------------------------------------------------------
def test_run_jobs_restores_gpu_results_only(tmp_path):
    rj = _run_jobs()
    assert rj.RESULTS_PREFIX == kj.RESULTS_PREFIX
    assert rj.JOBS is None and kj.PLACEHOLDER in (REPO / 'src' / 'kaggle' / 'run_jobs.py').read_text()
    inp = tmp_path / 'input' / 'datasets' / 'someone' / 'ner-gpu-bundle'
    inp.mkdir(parents=True)
    name = 'ner_results_fine_random_split_vary_crf_m0.json'
    (inp / f'gpu__{name}').write_text(json.dumps(_gpu_file('seed0')))
    # a CPU results file in the code tree is never picked up
    (inp / 'results').mkdir()
    (inp / 'results' / name).write_text(json.dumps(_gpu_file('seed0', env=None)))
    dest = tmp_path / 'processed' / 'ner'
    assert rj.restore_results(tmp_path / 'input', dest) == [name]
    assert json.loads((dest / name).read_text()) == _gpu_file('seed0')
    (inp / f'gpu__{name}').write_text(json.dumps(_gpu_file('seed0', env=None)))
    with pytest.raises(RuntimeError, match='not GPU runs'):
        rj.restore_results(tmp_path / 'input', tmp_path / 'other')


def test_run_jobs_unpacks_an_extracted_tree_or_the_archive(tmp_path):
    rj = _run_jobs()
    tree = tmp_path / 'in1' / 'code' / 'src' / 'graph'
    tree.mkdir(parents=True)
    (tree / 'ner_baseline.py').write_text('# code')
    dest = tmp_path / 'repo1'
    assert rj.unpack(tmp_path / 'in1', 'code.tar.gz', 'src/graph/ner_baseline.py', dest)
    assert (dest / 'src' / 'graph' / 'ner_baseline.py').read_text() == '# code'
    archive_dir = tmp_path / 'in2'
    archive_dir.mkdir()
    with tarfile.open(archive_dir / 'code.tar.gz', 'w:gz') as tar:
        data = b'# archived'
        info = tarfile.TarInfo('src/graph/ner_baseline.py')
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    dest = tmp_path / 'repo2'
    rj.unpack(archive_dir, 'code.tar.gz', 'src/graph/ner_baseline.py', dest)
    assert (dest / 'src' / 'graph' / 'ner_baseline.py').read_text() == '# archived'
    with pytest.raises(FileNotFoundError):
        rj.unpack(tmp_path / 'empty', 'code.tar.gz', 'src/graph/ner_baseline.py', tmp_path / 'x')


def test_run_jobs_gpu_check_and_job_environment():
    rj = _run_jobs()
    assert rj.check_gpus(['Tesla T4', 'Tesla T4'], 2, 'Tesla T4') is None
    assert 'need 2 GPUs' in rj.check_gpus(['Tesla T4'], 2, 'Tesla T4')
    assert 'not Tesla T4' in rj.check_gpus(['Tesla P100-PCIE-16GB'], 1, 'Tesla T4')
    env = rj.job_env(1, '/kaggle/working/processed')
    assert env['CUDA_VISIBLE_DEVICES'] == '1' and env['HF_HUB_OFFLINE'] == '1'
    assert env['PROCESSED_DIR'] == '/kaggle/working/processed'
    assert env['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert rj.job_command(['--crf'])[1:] == ['src/graph/ner_baseline.py', '--crf']


# ---------------------------------------------------------------------------
# ner_baseline.py on a CPU: unchanged
# ---------------------------------------------------------------------------
def test_cpu_runs_record_no_environment():
    assert nb.run_environment('cpu') is None


def test_torch_device_on_a_cpu_leaves_determinism_alone():
    torch = pytest.importorskip('torch')
    args = nb.parse_args(['--crf', '--device', 'cpu', '--threads', '1'])
    assert nb.torch_device(args) == 'cpu'
    assert not torch.are_deterministic_algorithms_enabled()
