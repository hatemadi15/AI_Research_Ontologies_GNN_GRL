"""Kaggle GPU runs (kaggle_jobs.py, src/kaggle/run_jobs.py) and the CUDA settings of
ner_baseline.py. No network, no GPU and no torch: the Kaggle CLI is never called."""
import argparse
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace
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


def test_w5c_runs_the_published_split_with_10_model_seeds():
    lanes = kj.batch_lanes('w5c')
    assert lanes[0][0][-2:] == ['--tag', 'smokeg']               # the repeat of a week-4 run
    jobs = [j for lane in lanes for j in lane][1:]
    args = [nb.parse_args(j) for j in jobs]
    assert all(a.published_split and a.seeds == list(range(0, 1000, 100)) and a.save_predictions
               for a in args)
    assert [(a.granularity, nb.variant_name(a)) for a in args] == [
        ('fine', 'baseline'), ('fine', 'crf'), ('coarse', 'baseline'), ('coarse', 'crf')]
    files = [f for j in jobs for f in _written_files(j)]
    assert len(files) == len(set(files)) == 4
    assert {p for _, p, _ in files} == {'published_split'}
    assert 'w5c' in kj.BATCHES


def test_smoke5_tries_the_week6_options_with_their_pins(monkeypatch):
    monkeypatch.setattr(nb.os.path, 'isfile', lambda path: True)    # the corpus is not here
    monkeypatch.setattr(kj, 'corpus_sha256', lambda: 'x' * 64)
    lanes = kj.batch_lanes('smoke5')
    args = [nb.parse_args(j) for lane in lanes for j in lane]
    repeat, *args = args                            # first the week-4 run again
    assert (repeat.tag, repeat.granularity, repeat.seeds, repeat.crf) == ('smokel', 'fine', [0],
                                                                         True)
    assert repeat.vary_split and repeat.max_train == 0 and repeat.model == nb.DEFAULT_MODEL
    assert all(a.max_train == 32 and a.epochs == 2 for a in args)
    deberta = [a for a in args if a.model == kj.DEBERTA]
    assert [a.tag for a in deberta] == ['smokeh', 'smokei']
    assert {nb.variant_name(a) for a in args} >= {'crf_ctx_smokej', 'crf_mt_st3', 'crf_smokek'}
    assert any(a.granularity == 'sofc' for a in args)
    for a in args:                                  # never collected
        assert 'smoke' in nb.results_path('', 'fine', a)
    spec = kj.batch_spec('smoke5')
    assert spec['models'][kj.DEBERTA] == '64a8c8eab3e352a784c658aef62be1662607476f'
    assert 'sentencepiece==0.2.2' in spec['pip'] and set(kj.PIP) <= set(spec['pip'])
    other = kj.batch_spec('w5c')                    # other batches keep their pins
    assert other['models'] == kj.MODELS and other['pip'] == list(kj.PIP)


def test_smoke6_runs_the_week6_options_at_full_size_outside_the_decision_splits(monkeypatch):
    monkeypatch.setattr(kj, 'corpus_sha256', lambda: 'x' * 64)
    repeat, *args = [nb.parse_args(j) for lane in kj.batch_lanes('smoke6') for j in lane]
    assert (repeat.tag, repeat.seeds, repeat.vary_split) == ('smokem', [0], True)
    assert [(a.granularity, a.tag) for a in args] == [('fine', 'smoken'), ('fine', 'smokeo'),
                                                      ('sofc', 'smokep')]
    assert all(a.published_split and a.seeds == [0] and a.crf and a.max_train == 0
               for a in args)
    assert [a.model for a in args] == [kj.DEBERTA, nb.DEFAULT_MODEL, nb.DEFAULT_MODEL]
    assert args[1].context == 64
    for a in args:                                  # never collected
        assert 'smoke' in nb.results_path('', a.granularity, a)
    assert kj.batch_spec('smoke6')['models'] == kj.batch_spec('smoke5')['models']


def _parsed(lanes):
    return [[nb.parse_args(j) for j in lane] for lane in lanes]


def _get(a, name, default=None):
    return getattr(a, name, default)                # options without a default are absent


def test_week6_screen_batches(monkeypatch):
    """w6a: DeBERTa-v3-large and document context, seeds 0-200 on the held-out papers; w6b:
    multi-task self-training per paper in one lane, and SOFC-Exp's official split."""
    monkeypatch.setattr(nb.os.path, 'isfile', lambda path: True)    # the corpus is not here
    monkeypatch.setattr(kj, 'corpus_sha256', lambda: 'x' * 64)
    a0, a1 = _parsed(kj.batch_lanes('w6a'))
    assert (a0[0].tag, a0[0].seeds, a0[0].vary_split) == ('smokeq', [0], True)
    screen = a0[1:] + a1
    assert all(a.leave_one_paper_out and not _get(a, 'papers') and a.crf for a in screen)
    assert sorted((nb.variant_name(a), a.granularity) for a in screen) == sorted(
        (f'{stem}_m{off}', g) for stem in ('crf_deberta', 'crf_ctx') for off in (0, 100, 200)
        for g in ('fine', 'coarse'))
    for a in screen:
        assert a.seed_offset == int(nb.variant_name(a).rsplit('_m', 1)[1])
        deberta = nb.variant_name(a).startswith('crf_deberta')
        assert a.model == (kj.DEBERTA if deberta else nb.DEFAULT_MODEL)
        assert _get(a, 'context', 0) == (0 if deberta else 64)
    assert kj.batch_spec('w6a')['models'][kj.DEBERTA] == kj.DEBERTA_PINS['models'][kj.DEBERTA]
    b0, b1 = _parsed(kj.batch_lanes('w6b'))
    assert b0[0].tag == 'smoker'
    assert [a.papers for a in b0[1:]] == [[p] for p in kj.PAPERS]
    assert all(_get(a, 'multi_task') and a.granularity == 'both' and _get(a, 'teachers') == 3
               and _get(a, 'students') == 3 and _get(a, 'self_train_sample') == 0
               and nb.variant_name(a) == 'crf_mt_st3' for a in b0[1:])
    assert [(a.granularity, a.published_split, a.crf) for a in b1] == [('sofc', True, False),
                                                                       ('sofc', True, True)]
    assert all(a.seeds == list(range(0, 1000, 100)) for a in b1)
    assert kj.batch_spec('w6b')['models'] == kj.MODELS


def test_week6_stage2_batches(monkeypatch, capsys):
    monkeypatch.setattr(nb.os.path, 'isfile', lambda path: True)
    monkeypatch.setattr(kj, 'corpus_sha256', lambda: 'x' * 64)
    lane0, lane1 = _parsed(kj.batch_lanes('w6c:deberta,ctx,mtst'))
    assert lane0[0].tag == 'smokes'
    mtst = [a for a in lane0 + lane1 if _get(a, 'multi_task')]
    assert lane1[:4] == mtst and all(_get(a, 'students') == 10 for a in mtst)   # 1 lane, first
    seeds = [a for a in lane0[1:] + lane1[4:]]
    assert sorted((nb.variant_name(a), a.granularity) for a in seeds) == sorted(
        (f'{stem}_m{off}', g) for stem in ('crf_deberta', 'crf_ctx')
        for off in range(300, 1000, 100) for g in ('fine', 'coarse'))
    assert kj.batch_extras('w6c:deberta,ctx,mtst') == kj.DEBERTA_PINS
    assert kj.batch_extras('w6c:ctx,mtst') == {}
    (repeat,), (sofc,) = _parsed(kj.batch_lanes('w6c:sofc-deberta'))
    assert (nb.variant_name(sofc), sofc.granularity, sofc.published_split, sofc.model) == (
        'crf_deberta', 'sofc', True, kj.DEBERTA)
    (_, again), (ctx,) = _parsed(kj.batch_lanes('w6c:sofc-ctx'))
    assert (nb.variant_name(ctx), _get(ctx, 'context'), ctx.seeds) == (
        'crf_ctx', 64, list(range(0, 1000, 100)))
    # the later commit of the amendment repeats the reference run of seed 0 (session b)
    assert (nb.variant_name(again), again.granularity, again.published_split, again.seeds,
            _get(again, 'context', 0)) == ('crf_smoket', 'sofc', True, [0], 0)
    assert len(kj.batch_lanes('w6c:sofc-deberta')[0]) == 1      # no repeat without sofc-ctx
    for bad in ('w6c:', 'w6c:foo', 'w6c:mtst,mtst', 'w6d'):
        with pytest.raises(SystemExit):
            kj.main(['--user', 'someone', 'push', '--batch', bad])
    assert 'w6c:<methods>' in capsys.readouterr().err
    assert kj.batch_name('w6c:mtst,deberta') == 'w6c:mtst,deberta'


def test_week7_bias_batch(monkeypatch):
    """w7a: the inner folds (seeds 0-200, counts only), the CRF head on the held-out papers
    again (seeds 0-900, with predictions) and on SOFC-Exp's official split, all with the
    bias grid; the repeat run first."""
    monkeypatch.setattr(kj, 'corpus_sha256', lambda: 'x' * 64)
    lane0, lane1 = _parsed(kj.batch_lanes('w7a'))
    assert lane0[0].tag == 'smokeu' and not _get(lane0[0], 'entity_bias')
    jobs = lane0[1:] + lane1
    assert all(a.crf and _get(a, 'entity_bias') == [float(b) for b in kj.BIAS_GRID]
               for a in jobs)
    assert 0.0 not in jobs[0].entity_bias and len(set(kj.BIAS_GRID)) == 11
    inner = [a for a in jobs if _get(a, 'leave_two_papers_out')]
    assert sorted((nb.variant_name(a), a.granularity, a.seed_offset) for a in inner) == sorted(
        (f'crf_bias_m{off}', g, off) for off in (0, 100, 200) for g in ('fine', 'coarse'))
    assert not any(a.save_predictions for a in inner)
    outer = [a for a in jobs if a.leave_one_paper_out]
    assert sorted((nb.variant_name(a), a.granularity, a.seed_offset) for a in outer) == sorted(
        (f'crf_bias_m{off}', g, off) for off in range(0, 1000, 100) for g in ('fine', 'coarse'))
    assert all(a.save_predictions and not _get(a, 'papers') for a in outer)
    (sofc,) = [a for a in jobs if _get(a, 'published_split')]
    assert (nb.variant_name(sofc), sofc.granularity, sofc.seeds, sofc.save_predictions) == (
        'crf_bias', 'sofc', list(range(0, 1000, 100)), True)
    assert len(jobs) == 6 + 20 + 1 and kj.batch_spec('w7a')['models'] == kj.MODELS
    assert kj.batch_name('w7a') == 'w7a'


def test_week7_sofc_levels_batch(monkeypatch):
    """w7b: three repeats first (the CRF run of smoke4, session b's SOFC-Exp CRF run of seed
    0, the MaterioMiner multi-task run of split 0), then SOFC-Exp's official split with 10
    model seeds: plain MatSciBERT and the CRF head on the slot level, and multi-task training
    on both levels."""
    monkeypatch.setattr(kj, 'corpus_sha256', lambda: 'x' * 64)
    lane0, lane1 = _parsed(kj.batch_lanes('w7b'))
    assert [a.tag for a in lane0[:2]] == ['smokev', 'smokew'] and lane1[0].tag == 'smokex'
    assert (lane0[1].granularity, lane0[1].seeds, _get(lane0[1], 'published_split')) == (
        'sofc', [0], True)
    assert lane1[0].multi_task and lane1[0].vary_split and lane1[0].seeds == [0]
    runs = sorted((a.granularity, nb.variant_name(a)) for a in lane0[2:] + lane1[1:])
    assert runs == [('sofc_both', 'crf_mt'), ('sofc_slot', 'baseline'), ('sofc_slot', 'crf')]
    for a in lane0[2:] + lane1[1:]:
        assert a.published_split and a.save_predictions and a.seeds == list(range(0, 1000, 100))
    assert kj.batch_name('w7b') == 'w7b' and kj.batch_spec('w7b')['models'] == kj.MODELS


def test_week7_mention_replacement_batches(monkeypatch):
    """w7c: the screen of mention replacement (3 copies), seeds 0-200 on the held-out papers;
    w7d: stage 2, seeds 300-900 there and SOFC-Exp's official split (S2). The repeat first."""
    monkeypatch.setattr(kj, 'corpus_sha256', lambda: 'x' * 64)
    for name, smoke, offsets in (('w7c', 'smokey', (0, 100, 200)),
                                 ('w7d', 'smokez', tuple(range(300, 1000, 100)))):
        lane0, lane1 = _parsed(kj.batch_lanes(name))
        assert lane0[0].tag == smoke and not _get(lane0[0], 'augment_mentions')
        jobs = lane0[1:] + lane1
        assert all(a.crf and a.augment_mentions == 3 and a.save_predictions for a in jobs)
        held_out = [a for a in jobs if a.leave_one_paper_out]
        assert sorted((nb.variant_name(a), a.granularity, a.seed_offset) for a in held_out) == (
            sorted((f'crf_mr_m{off}', g, off) for off in offsets for g in ('fine', 'coarse')))
        sofc = [a for a in jobs if _get(a, 'published_split')]
        assert [(nb.variant_name(a), a.granularity, a.seeds) for a in sofc] == (
            [('crf_mr', 'sofc', list(range(0, 1000, 100)))] if name == 'w7d' else [])
        assert kj.batch_name(name) == name


def test_a_refused_push_stops(monkeypatch, capsys):
    """The Kaggle CLI exits 0 when it refuses a push; push must not report the batch."""
    monkeypatch.setattr(kj, 'corpus_sha256', lambda: 'x' * 64)
    monkeypatch.setattr(kj, 'kaggle', lambda *a, **k: 'Kernel push error: Maximum batch GPU '
                                                       'session count of 2 reached.')
    args = SimpleNamespace(batch='w5c', kernel=kj.KERNEL, dataset=kj.DATASET, user='someone')
    with pytest.raises(SystemExit, match='not pushed'):
        kj.cmd_push(args)
    assert 'lane 0' not in capsys.readouterr().out


def test_bundle_and_push_use_the_chosen_dataset(monkeypatch, tmp_path):
    """--dataset names the bundle dataset that bundle versions and push attaches, so that a
    second bundle can wait in its own dataset while a queued session needs the first."""
    calls, pushed = [], {}

    def kaggle(*args, check=True):
        calls.append(args)
        if args[:2] == ('kernels', 'push'):
            pushed.update(json.loads(Path(args[3], 'kernel-metadata.json').read_text()))
            return 'Kernel version 1 successfully pushed.'
        return 'ready'

    monkeypatch.setattr(kj, 'corpus_sha256', lambda: 'x' * 64)
    monkeypatch.setattr(kj, 'kaggle', kaggle)
    monkeypatch.setattr(kj, 'WORK_DIR', tmp_path)
    monkeypatch.setattr(kj, 'build_wheels', lambda: [])
    monkeypatch.setattr(kj, 'build_bundle', lambda commit, folder, wheels: (
        folder.mkdir(parents=True, exist_ok=True),
        {'commit': 'c' * 40, 'corpus_sha256': 'x' * 64, 'results': []})[1])
    monkeypatch.setattr(kj.time, 'sleep', lambda seconds: None)
    kj.main(['--user', 'someone', '--dataset', 'ner-gpu-bundle-2', 'bundle', '--commit', 'HEAD'])
    meta = json.loads((tmp_path / 'bundle' / 'dataset-metadata.json').read_text())
    assert meta['id'] == 'someone/ner-gpu-bundle-2'
    assert ('datasets', 'status', 'someone/ner-gpu-bundle-2') in calls
    kj.main(['--user', 'someone', '--dataset', 'ner-gpu-bundle-2', '--kernel', 'k', 'push',
             '--batch', 'w5c'])
    assert pushed['id'] == 'someone/k'
    assert pushed['dataset_sources'] == ['someone/ner-gpu-bundle-2']
    kj.main(['--user', 'someone', 'push', '--batch', 'w5c'])
    assert pushed['dataset_sources'] == ['someone/ner-gpu-bundle']


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
                           expected_sha256=kj.file_sha256(corpus), wheels=[wheel],
                           sofc_report=tmp_path / 'no_sofc_corpus.json',
                           mulms_report=tmp_path / 'no_mulms.json',
                           topic_reports=tmp_path / 'no_topics')
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
        kj.build_bundle('HEAD', out, results_dir=results, corpus=corpus, expected_sha256='0' * 64,
                        sofc_report=tmp_path / 'no_sofc_corpus.json',
                           mulms_report=tmp_path / 'no_mulms.json',
                           topic_reports=tmp_path / 'no_topics')


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


def test_run_jobs_drops_the_restored_files_that_no_job_changed(tmp_path):
    rj = _run_jobs()
    inp = tmp_path / 'input' / 'datasets' / 'someone' / 'ner-gpu-bundle'
    inp.mkdir(parents=True)
    names = [f'ner_results_fine_random_split_vary_crf_m{k}.json' for k in (0, 100)]
    for name in names:
        (inp / f'gpu__{name}').write_text(json.dumps(_gpu_file('seed0')))
    dest = tmp_path / 'processed' / 'ner'
    assert rj.restore_results(tmp_path / 'input', dest) == names
    (dest / names[1]).write_text(json.dumps(_gpu_file('seed1')))      # a job extended it
    (dest / 'ner_results_fine_random_split_vary_crf_m200.json').write_text('{}')   # new
    assert rj.drop_unchanged(tmp_path / 'input', dest, names) == [names[0]]
    assert sorted(p.name for p in dest.iterdir()) == [
        names[1], 'ner_results_fine_random_split_vary_crf_m200.json']


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
def test_run_jobs_stops_a_job_at_the_time_limit(tmp_path, monkeypatch, capsys):
    """A job still running at the stop time is stopped and recorded as such; the lane goes
    on, and no job starts after the deadline."""
    rj = _run_jobs()
    monkeypatch.setattr(rj, 'REPO', tmp_path)
    monkeypatch.setattr(rj, 'MIN_JOB_SECONDS', 0.5)
    monkeypatch.setattr(rj, 'job_command', lambda job: [sys.executable, '-c', job[0]])
    manifest = rj.Manifest(tmp_path / 'manifest.json', {'jobs': []})
    now = time.time()
    rj.run_lane(0, [['import time; time.sleep(30)'], ['pass']], manifest, deadline=now + 60,
                logs=tmp_path, processed=tmp_path, stop_at=now + 1)
    first, second = manifest.data['jobs']
    assert first['status'] == 'stopped: time limit' and first['exit_code'] is None
    assert first['minutes'] < 0.25
    assert second['status'] == 'done' and second['exit_code'] == 0
    rj.run_lane(1, [['pass']], manifest, deadline=now - 1, logs=tmp_path, processed=tmp_path)
    assert manifest.data['jobs'][-1]['status'] == 'not started: deadline'
    assert 'lane 0 job 0: stopped: time limit' in capsys.readouterr().out


def test_run_jobs_limits_the_setup_commands(monkeypatch, capsys):
    """nvidia-smi runs with a time limit, so a hung driver ends the session's setup, and
    each setup step prints a line for the live log."""
    rj = _run_jobs()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd[0], kwargs.get('timeout')))
        return subprocess.CompletedProcess(cmd, 0, stdout='Tesla T4\nTesla T4\n', stderr='')
    monkeypatch.setattr(rj.subprocess, 'run', fake_run)
    assert rj.gpu_names() == ['Tesla T4', 'Tesla T4']
    assert calls == [('nvidia-smi', rj.TIMEOUTS['nvidia_smi'])]
    rj.setup_step(time.time() - 90, 'pip install')
    assert capsys.readouterr().out == 'setup: pip install (1.5 min)\n'


def test_environment_record_without_nvidia_smi_and_on_a_tpu(monkeypatch):
    """A TPU VM has no nvidia-smi: the record says so instead of ending the setup (the first
    TPU session died there); with device xla the command is not even tried."""
    rj = _run_jobs()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == 'nvidia-smi':
            raise FileNotFoundError(2, 'No such file or directory', 'nvidia-smi')
        return SimpleNamespace(stdout='2.9.0 None 5.18.0\n', stderr='', returncode=0)
    monkeypatch.setattr(rj.subprocess, 'run', fake_run)
    env = rj.environment()
    assert env['nvidia_smi'].startswith('not found') and env['torch_xla'] == '2.9.0 None 5.18.0'
    assert env['torch_cuda_transformers'] == '2.9.0 None 5.18.0' and 'nvidia-smi' in calls
    calls.clear()
    env = rj.environment('xla')
    assert env['nvidia_smi'] == 'skipped: xla' and 'nvidia-smi' not in calls


def test_cpu_runs_record_no_environment():
    assert nb.run_environment('cpu') is None


def test_torch_device_on_a_cpu_leaves_determinism_alone():
    torch = pytest.importorskip('torch')
    args = nb.parse_args(['--crf', '--device', 'cpu', '--threads', '1'])
    assert nb.torch_device(args) == 'cpu'
    assert not torch.are_deterministic_algorithms_enabled()


def test_sofc_mtst_batch(monkeypatch):
    """w6c:sofc-mtst (week-6 amendment): two repeats, then multi-task self-training on
    SOFC-Exp's two levels with the SOFC corpus, the 3 teachers in both lanes and the students
    split 0-400 / 500-900; only this batch's spec carries the SOFC corpus's SHA-256."""
    monkeypatch.setattr(nb.os.path, 'isfile', lambda path: True)    # the SOFC corpus is not here
    monkeypatch.setattr(kj, 'corpus_sha256',
                        lambda report=kj.CORPUS_REPORT: 'f' * 64 if report == kj.CORPUS_REPORT
                        else 's' * 64)
    name = 'w6c:sofc-mtst'
    assert kj.batch_name(name) == name
    lane0, lane1 = _parsed(kj.batch_lanes(name))
    assert [a.tag for a in lane0[:2]] == ['smokeaa', 'smokeab']
    assert (lane0[1].granularity, lane0[1].seeds, _get(lane0[1], 'published_split')) == (
        'sofc', [0], True)
    for args, offsets in ((lane0[2], [0, 100, 200, 300, 400]), (lane1[0], [500, 600, 700, 800,
                                                                         900])):
        assert (args.granularity, args.seeds, args.self_train) == ('sofc_both', [0],
                                                                   kj.SOFC_CORPUS)
        assert args.published_split and args.multi_task and args.crf and args.save_predictions
        assert (args.teachers, args.students, args.self_train_sample) == (3, 10, 0)
        assert args.student_offsets == offsets and nb.variant_name(args) == 'crf_mt_st3'
    assert len(lane0) == 3 and len(lane1) == 1
    spec = kj.batch_spec(name)
    assert (spec['corpus_sha256'], spec['sofc_corpus_sha256']) == ('f' * 64, 's' * 64)
    assert 'sofc_corpus_sha256' not in kj.batch_spec('w7b')
    with pytest.raises(argparse.ArgumentTypeError, match='alone'):
        kj.batch_name('w6c:ctx,sofc-mtst')


def test_build_bundle_with_the_sofc_corpus(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('One sentence.\n')
    sofc = tmp_path / 'sofc.txt'
    sofc.write_text('The Ni - YSZ anode .\n')
    report = tmp_path / 'sofc_corpus.json'
    report.write_text(json.dumps({'corpus': {'sha256': kj.file_sha256(sofc)}}))
    out = tmp_path / 'bundle'
    info = kj.build_bundle('HEAD', out, results_dir=tmp_path / 'none', corpus=corpus,
                           expected_sha256=kj.file_sha256(corpus), sofc_corpus=sofc,
                           sofc_report=report, mulms_report=tmp_path / 'no_mulms.json',
                           topic_reports=tmp_path / 'no_topics')
    assert (out / 'sofc_sentences.txt').read_bytes() == sofc.read_bytes()
    assert info['sofc_corpus_sha256'] == kj.file_sha256(sofc)
    report.write_text(json.dumps({'corpus': {'sha256': '0' * 64}}))
    with pytest.raises(SystemExit, match='SHA-256'):
        kj.build_bundle('HEAD', out, results_dir=tmp_path / 'none', corpus=corpus,
                        expected_sha256=kj.file_sha256(corpus), sofc_corpus=sofc,
                        sofc_report=report, mulms_report=tmp_path / 'no_mulms.json',
                           topic_reports=tmp_path / 'no_topics')
    info = kj.build_bundle('HEAD', out, results_dir=tmp_path / 'none', corpus=corpus,
                           expected_sha256=kj.file_sha256(corpus), sofc_report=tmp_path / 'no',
                           mulms_report=tmp_path / 'no_mulms.json',
                           topic_reports=tmp_path / 'no_topics')
    assert info['sofc_corpus_sha256'] is None and not (out / 'sofc_sentences.txt').exists()
    report.write_text(json.dumps({'corpus': {'sha256': kj.file_sha256(sofc)}}))
    with pytest.raises(SystemExit, match='build_sofc_corpus'):      # the report, no corpus
        kj.build_bundle('HEAD', out, results_dir=tmp_path / 'none', corpus=corpus,
                        expected_sha256=kj.file_sha256(corpus), sofc_corpus=tmp_path / 'gone',
                        sofc_report=report, mulms_report=tmp_path / 'no_mulms.json',
                           topic_reports=tmp_path / 'no_topics')


def test_run_jobs_places_the_sofc_corpus_only_when_asked(tmp_path):
    rj = _run_jobs()
    inputs, repo = tmp_path / 'input', tmp_path / 'repo'
    (inputs / 'bundle').mkdir(parents=True)
    corpus = inputs / 'bundle' / 'sofc_sentences.txt'
    corpus.write_text('The Ni - YSZ anode .\n')
    assert rj.place_sofc_corpus({}, inputs, repo) is None and not repo.exists()
    digest = rj.file_sha256(corpus)
    assert rj.place_sofc_corpus({'sofc_corpus_sha256': digest}, inputs, repo) == digest
    assert (repo / rj.SOFC_CORPUS).read_bytes() == corpus.read_bytes()
    with pytest.raises(RuntimeError, match='SHA-256'):
        rj.place_sofc_corpus({'sofc_corpus_sha256': '0' * 64}, inputs, repo)
    with pytest.raises(FileNotFoundError):
        rj.place_sofc_corpus({'sofc_corpus_sha256': digest}, tmp_path / 'empty', repo)


def test_week7_retokenized_test_batch(monkeypatch):
    """w7e (part 2e): two repeats, then the CRF head and plain MatSciBERT on SOFC-Exp's
    official split, 10 model seeds, scored on the retokenized test articles."""
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: 'x' * 64)
    lane0, lane1 = _parsed(kj.batch_lanes('w7e'))
    assert [a.tag for a in lane0[:2]] == ['smokeac', 'smokead']
    assert not _get(lane0[1], 'retokenized_test') and lane0[1].granularity == 'sofc'
    runs = lane0[2:] + lane1
    assert sorted(nb.variant_name(a) for a in runs) == ['cdetest', 'crf_cdetest']
    for a in runs:
        assert (a.granularity, a.seeds) == ('sofc', list(range(0, 1000, 100)))
        assert a.retokenized_test and a.published_split and a.save_predictions
    assert kj.batch_name('w7e') == 'w7e' and 'sofc_corpus_sha256' not in kj.batch_spec('w7e')


def test_week8_sofc_splits_batch(monkeypatch):
    """w8a (week 8, part 1): two repeats, then the CRF head on SOFC-Exp's random splits (lane
    0) and article folds (lane 1), model seed offsets 0 to 900, one results file each."""
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: 'x' * 64)
    lane0, lane1 = _parsed(kj.batch_lanes('w8a'))
    assert [a.tag for a in lane0[:2]] == ['smokeae', 'smokeaf']
    assert (lane0[1].granularity, nb.protocol_name(lane0[1])) == ('sofc', 'published_split')
    for lane, protocol in ((lane0[2:], 'random_split_vary'), (lane1, 'document_folds')):
        assert [a.tag for a in lane] == [f'm{off}' for off in range(0, 1000, 100)]
        assert [a.seed_offset for a in lane] == list(range(0, 1000, 100))
        for a in lane:
            assert (a.granularity, nb.protocol_name(a), nb.variant_name(a)) == (
                'sofc', protocol, f'crf_m{a.seed_offset}')
            assert a.crf and a.save_predictions and a.seeds == [0, 1, 2, 3, 4]
    assert all(a.document_folds == 5 for a in lane1)
    assert 'w8a' in kj.BATCHES and 'sofc_corpus_sha256' not in kj.batch_spec('w8a')


def test_week8_paper_diversity_batch(monkeypatch):
    """w8b (week 8, part 2): a repeat, then the CRF head with --two-paper-size on the held-out
    papers, seed offsets 0 to 200, one lane per granularity."""
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: 'x' * 64)
    lane0, lane1 = _parsed(kj.batch_lanes('w8b'))
    assert lane0[0].tag == 'smokeag' and len(lane0) == 4 and len(lane1) == 3
    for lane, granularity in ((lane0[1:], 'fine'), (lane1, 'coarse')):
        assert [a.seed_offset for a in lane] == [0, 100, 200]
        for a in lane:
            assert (a.granularity, nb.protocol_name(a), nb.variant_name(a)) == (
                granularity, 'leave_one_paper_out', f'crf_p3n2_m{a.seed_offset}')
            assert a.two_paper_size and a.save_predictions
    assert 'w8b' in kj.BATCHES


def test_week9_multi_task_mention_replacement_batch(monkeypatch):
    """w9a (week 9): two repeats (the CRF run of smoke4, the multi-task run of random split 0),
    then crf_mt_mr_m0 to _m900 on the held-out papers and S2 on SOFC-Exp's two levels."""
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: 'x' * 64)
    lane0, lane1 = _parsed(kj.batch_lanes('w9a'))
    assert lane0[0].tag == 'smokeah' and not _get(lane0[0], 'multi_task')
    assert lane1[0].tag == 'smokeai' and lane1[0].multi_task and lane1[0].vary_split
    assert not _get(lane1[0], 'augment_mentions')
    jobs = lane0[1:] + lane1[1:]
    sofc = [a for a in jobs if a.granularity == 'sofc_both']
    held_out = [a for a in jobs if a.granularity != 'sofc_both']
    assert len(sofc) == 1 and sofc[0].published_split
    assert sofc[0].seeds == list(range(0, 1000, 100))
    assert sorted(a.seed_offset for a in held_out) == list(range(0, 1000, 100))
    for a in jobs:
        assert a.crf and a.multi_task and a.augment_mentions == 3 and a.save_predictions
    for a in held_out:
        assert nb.protocol_name(a) == 'leave_one_paper_out'
        assert nb.variant_name(a) == f'crf_mt_mr_m{a.seed_offset}'
    assert nb.variant_name(sofc[0]) == 'crf_mt_mr'
    assert 'w9a' in kj.BATCHES


def test_week9_article_draws_batch(monkeypatch):
    """w9b (week 9, part 2): two repeats, then the CRF head and mention replacement on SOFC-Exp
    trained on 10 of its training articles, 5 draws, seed offsets 0 to 900."""
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: 'x' * 64)
    lane0, lane1 = _parsed(kj.batch_lanes('w9b'))
    assert [a.tag for a in lane0[:2]] == ['smokeaj', 'smokeak']
    assert (lane0[1].granularity, nb.protocol_name(lane0[1])) == ('sofc', 'published_split')
    jobs = lane0[2:] + lane1
    names = sorted(nb.variant_name(a) for a in jobs)
    assert names == sorted([f'crf_m{off}' for off in range(0, 1000, 100)]
                           + [f'crf_mr_m{off}' for off in range(0, 1000, 100)])
    for a in jobs:
        assert (a.granularity, nb.protocol_name(a)) == ('sofc', 'article_draws')
        assert (a.train_articles, a.draws) == (10, 5) and a.crf and a.save_predictions
    assert 'w9b' in kj.BATCHES


def test_week10_mulms_batch(monkeypatch, tmp_path):
    """w10a (week 10): the repeat, then on MuLMS the CRF head and mention replacement on the
    held-out subfields and the CRF head on the random pool splits, 400 training sentences,
    seed offsets 0 to 900; the spec lists the MuLMS files with their SHA-256."""
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: 'x' * 64)
    report = tmp_path / 'mulms_corpus.json'
    report.write_text(json.dumps({'outputs': {'a_01.conll': 'a' * 64, 'subfields.json': 'b' * 64}}))
    monkeypatch.setattr(kj, 'MULMS_REPORT', report)
    lane0, lane1 = _parsed(kj.batch_lanes('w10a'))
    assert lane0[0].tag == 'smokeal' and lane0[0].granularity == 'fine'
    jobs = lane0[1:] + lane1
    held = [a for a in jobs if nb.protocol_name(a) == 'leave_one_subfield_out']
    pool = [a for a in jobs if nb.protocol_name(a) == 'random_split_pool']
    offsets = range(0, 1000, 100)
    assert sorted(nb.variant_name(a) for a in held) == sorted(
        [f'crf_m{off}' for off in offsets] + [f'crf_mr_m{off}' for off in offsets])
    assert sorted(nb.variant_name(a) for a in pool) == sorted(f'crf_m{off}' for off in offsets)
    assert len(held) + len(pool) == len(jobs) == 30
    for a in jobs:
        assert (a.granularity, a.train_pool) == ('mulms', 400) and a.crf and a.save_predictions
    assert all(a.seeds == [0, 1, 2, 3, 4] and a.vary_split for a in pool)
    assert all(_get(a, 'augment_mentions') in (None, 3) for a in held)
    spec = kj.batch_spec('w10a')
    assert spec['mulms_files'] == {'a_01.conll': 'a' * 64, 'subfields.json': 'b' * 64}
    assert 'mulms_files' not in kj.batch_spec('w9b')
    assert 'w10a' in kj.BATCHES


def test_week11_silver_name_batches(monkeypatch, tmp_path):
    """w11a: on MuLMS each lane trains the teachers of all 7 subfields and half of the 10
    students (so the lanes write different files); w11b: on MaterioMiner one lane per
    granularity, a job per held-out paper; both start with the repeat."""
    monkeypatch.setattr(nb.os.path, 'isfile', lambda path: True)    # the corpus is not here
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: 'x' * 64)
    report = tmp_path / 'mulms_corpus.json'
    report.write_text(json.dumps({'outputs': {'a_01.conll': 'a' * 64}}))
    monkeypatch.setattr(kj, 'MULMS_REPORT', report)
    lane0, lane1 = _parsed(kj.batch_lanes('w11a'))
    assert lane0[0].tag == 'smokeam' and lane0[0].granularity == 'fine'
    for lane, offsets in ((lane0[1:], range(0, 500, 100)), (lane1, range(500, 1000, 100))):
        assert [a.subfields for a in lane] == [[s] for s in kj.MULMS_SUBFIELD_NAMES]
        for a in lane:
            assert (a.granularity, a.train_pool, nb.protocol_name(a)) == (
                'mulms', 400, 'leave_one_subfield_out')
            assert (a.self_train, a.teachers, a.students, a.augment_mentions) == (
                'unused', 3, 10, 3)
            assert a.augment_names and a.crf and a.save_predictions
            assert a.student_offsets == list(offsets)
    assert set(kj.MULMS_SUBFIELD_NAMES) == {'electrolysis', 'graphene', 'pemfc', 'poly', 'semi',
                                            'sofc', 'steel'}
    assert 'mulms_files' in kj.batch_spec('w11a')
    lane0, lane1 = _parsed(kj.batch_lanes('w11b'))
    assert lane0[0].tag == 'smokean'
    for lane, g in ((lane0[1:], 'fine'), (lane1, 'coarse')):
        assert [a.papers for a in lane] == [[p] for p in kj.PAPERS]
        for a in lane:
            assert (a.granularity, nb.protocol_name(a), a.self_train) == (
                g, 'leave_one_paper_out', kj.CORPUS)
            assert (a.teachers, a.students, a.self_train_sample, a.augment_mentions) == (
                3, 10, 0, 3)
            assert a.augment_names and not hasattr(a, 'student_offsets')
            view = nb.SimpleNamespace(**{**vars(a), 'tag': 'm300'})
            assert nb.variant_name(view) == 'crf_mrs_m300'
    assert 'mulms_files' not in kj.batch_spec('w11b')
    assert {'w11a', 'w11b'} <= set(kj.BATCHES)


def test_week12_topic_name_batches(monkeypatch, tmp_path):
    """w12a: the screen's five variants (amendment 1) on MuLMS, seed offsets 0 to 200, the
    jobs of a variant in one lane; w12b:<variant>: the winner's seed offsets 300 to 900, the
    teacher variants with each lane training the teachers and 3 or 4 students, the 6-copy
    jobs balanced."""
    monkeypatch.setattr(nb.os.path, 'isfile', lambda path: True)    # the corpora are not here
    monkeypatch.setattr(kj, 'MULMS_REPORT', tmp_path / 'no_mulms.json')
    monkeypatch.setattr(kj, 'mulms_files', lambda: {'a_01.conll': 'a' * 64})
    reports = tmp_path / 'topic_corpora'
    reports.mkdir()
    for topic in kj.MULMS_SUBFIELD_NAMES:
        (reports / f'{topic}.json').write_text(json.dumps({'corpus': {'sha256': topic * 8}}))
    monkeypatch.setattr(kj, 'TOPIC_REPORTS', reports)
    real_sha256 = kj.corpus_sha256
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: (
        real_sha256(report) if report is not None and Path(report).parent == reports else 'x' * 64))

    def kind(a):
        return nb.variant_name(nb.SimpleNamespace(**{**vars(a), 'tag': ''}))
    lane0, lane1 = _parsed(kj.batch_lanes('w12a'))
    assert lane0[0].tag == 'smokeao' and lane0[0].granularity == 'fine'
    assert len(lane0) == len(lane1) == 16
    kinds = {0: [kind(a) for a in lane0[1:]], 1: [kind(a) for a in lane1]}
    assert sorted(set(kinds[0])) == ['crf_mr6', 'crf_mrm_topic6', 'crf_mrs_topic']
    assert sorted(set(kinds[1])) == ['crf_mr6', 'crf_mrm_topic', 'crf_st3_topic']
    teacher_jobs = [a for a in lane0[1:] + lane1 if hasattr(a, 'teachers')]
    assert len(teacher_jobs) == 28
    for a in teacher_jobs:
        assert (a.granularity, a.train_pool, nb.protocol_name(a)) == (
            'mulms', 400, 'leave_one_subfield_out')
        assert a.self_train == f'data/processed/corpus/topic_{a.subfields[0]}_sentences.txt'
        assert (a.self_train_sample, a.teachers, a.students) == (0, 3, 10)
        assert a.crf and a.save_predictions and a.names_tag == 'topic'
        assert a.student_offsets == [0, 100, 200]
        name = kind(a)
        assert getattr(a, 'augment_names', False) == (name != 'crf_st3_topic')
        assert getattr(a, 'silver_mentions', False) == name.startswith('crf_mrm')
        assert getattr(a, 'augment_mentions', 0) == {
            'crf_mrs_topic': 3, 'crf_mrm_topic': 3, 'crf_mrm_topic6': 6, 'crf_st3_topic': 0}[name]
    for name in ('crf_mrs_topic', 'crf_mrm_topic6', 'crf_mrm_topic', 'crf_st3_topic'):
        assert [a.subfields for a in lane0[1:] + lane1 if kind(a) == name] == [
            [s] for s in kj.MULMS_SUBFIELD_NAMES]
    six = [a for a in lane0[1:] + lane1 if kind(a) == 'crf_mr6']
    assert [a.tag for a in six] == ['m0', 'm100', 'm200']
    for a in six:
        assert (a.granularity, a.train_pool, a.augment_mentions, a.crf) == ('mulms', 400, 6, True)
        assert nb.protocol_name(a) == 'leave_one_subfield_out' and not hasattr(a, 'teachers')
        assert nb.variant_name(a) == f'crf_mr6_{a.tag}'
    spec = kj.batch_spec('w12a')
    assert spec['topic_corpora'] == {t: t * 8 for t in kj.MULMS_SUBFIELD_NAMES}
    assert spec['mulms_files'] == {'a_01.conll': 'a' * 64}
    expected = {'mrs': 'crf_mrs_topic', 'mrm': 'crf_mrm_topic', 'mrm6': 'crf_mrm_topic6',
                'st3': 'crf_st3_topic'}
    for variant, name in expected.items():
        lane0, lane1 = _parsed(kj.batch_lanes(f'w12b:{variant}'))
        assert lane0[0].tag == 'smokeap'
        for lane, offsets in ((lane0[1:], [300, 400, 500]), (lane1, [600, 700, 800, 900])):
            assert [a.subfields for a in lane] == [[s] for s in kj.MULMS_SUBFIELD_NAMES]
            for a in lane:
                assert a.student_offsets == offsets and kind(a) == name
        assert kj.batch_spec(f'w12b:{variant}')['topic_corpora'] == spec['topic_corpora']
    lane0, lane1 = _parsed(kj.batch_lanes('w12b:mr6'))
    assert lane0[0].tag == 'smokeap'
    jobs = lane0[1:] + lane1
    assert sorted(a.tag for a in jobs) == [f'm{off}' for off in range(300, 1000, 100)]
    assert all(a.augment_mentions == 6 for a in jobs) and abs(len(lane0) - 1 - len(lane1)) <= 1
    assert 'topic_corpora' not in kj.batch_spec('w12b:mr6')
    with pytest.raises(ValueError, match='unknown week-12 variant'):
        kj.batch_lanes('w12b:mt')
    assert {'w12a', 'w12b:mrs', 'w12b:mrm', 'w12b:mr6', 'w12b:mrm6', 'w12b:st3'} <= set(kj.BATCHES)
    assert kj.batch_name('w12b:st3') == 'w12b:st3'


def test_tpu_smoke_batch_metadata_and_lane_environment(monkeypatch, tmp_path):
    """The TPU spike: tpu-smoke runs the same fine-grained CRF job on 8 lanes with --device
    xla; its spec names the device and the shorter deadline; the kernel is a TPU kernel; the
    runner gives each lane a chip instead of a GPU."""
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: 'x' * 64)
    lanes = _parsed(kj.batch_lanes('tpu-smoke'))
    assert len(lanes) == 8 and all(len(lane) == 1 for lane in lanes)
    assert [lane[0].tag for lane in lanes] == [f'tpu{c}' for c in 'abcdefgh']
    for lane in lanes:
        a = lane[0]
        assert (a.granularity, nb.protocol_name(a), a.device, a.crf) == (
            'fine', 'leave_one_paper_out', 'xla', True)
        assert nb.variant_name(a) == f'crf_{a.tag}'
    spec = kj.batch_spec('tpu-smoke')
    assert (spec['device'], spec['gpu'], spec['deadline_hours']) == ('xla', None, 7)
    assert 'device' not in kj.batch_spec('w9b') and kj.batch_spec('w9b')['gpu'] == 'Tesla T4'
    assert kj.is_tpu_batch('tpu-smoke') and not kj.is_tpu_batch('w12a')
    meta = kj.kernel_metadata('u', 'k', 'd', tpu=True)
    assert (meta['enable_tpu'], meta['enable_gpu']) == (True, False)
    assert 'machine_shape' not in meta and 'docker_image' not in meta
    assert (meta['kernel_type'], meta['code_file']) == ('notebook', 'run_jobs.ipynb')
    nb_json = json.loads(kj.notebook_cell('print(1)\n'))
    assert nb_json['cells'][0]['source'] == 'print(1)\n' and nb_json['nbformat'] == 4
    gpu = kj.kernel_metadata('u', 'k', 'd')
    assert (gpu['enable_tpu'], gpu['enable_gpu'], gpu['machine_shape']) == (
        False, True, kj.MACHINE)
    rj = _run_jobs()
    env = rj.job_env(3, tmp_path, 'xla')
    assert (env['TPU_VISIBLE_CHIPS'], env['PJRT_DEVICE'], env['TPU_PROCESS_BOUNDS']) == (
        '3', 'TPU', '1,1,1')
    assert 'CUDA_VISIBLE_DEVICES' not in env and env['HF_HUB_OFFLINE'] == '1'
    env = rj.job_env(1, tmp_path)
    assert env['CUDA_VISIBLE_DEVICES'] == '1' and 'TPU_VISIBLE_CHIPS' not in env
    assert 'tpu-smoke' in kj.BATCHES and kj.batch_name('tpu-smoke') == 'tpu-smoke'
    (lane,) = _parsed(kj.batch_lanes('tpu-smoke1'))
    assert len(lane) == 1 and lane[0].tag == 'tpu1' and lane[0].device == 'xla'
    spec1 = kj.batch_spec('tpu-smoke1')
    assert spec1['device'] == 'xla' and spec1['restore_results'] is False
    assert 'torch_xla==2.9.0' in spec1['pip'] and 'torch==2.9.0+cpu' in spec1['pip']
    assert 'libtpu==0.0.21.1' in spec1['pip'] and 'torchaudio==2.9.0+cpu' in spec1['pip']
    assert 'torch_xla==2.9.0' not in kj.batch_spec('w12a')['pip']
    assert 'restore_results' not in kj.batch_spec('w12a')


def test_build_bundle_with_the_mulms_files(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('One sentence.\n')
    mulms = tmp_path / 'mulms'
    mulms.mkdir()
    (mulms / 'a_01.conll').write_text('WO3 B-MAT\n\n')
    (mulms / 'subfields.json').write_text('{"a_01": "alpha"}')
    report = tmp_path / 'mulms_corpus.json'
    outputs = {name: kj.file_sha256(mulms / name) for name in ('a_01.conll', 'subfields.json')}
    report.write_text(json.dumps({'outputs': outputs}))
    out = tmp_path / 'bundle'
    info = kj.build_bundle('HEAD', out, results_dir=tmp_path / 'none', corpus=corpus,
                           expected_sha256=kj.file_sha256(corpus), sofc_report=tmp_path / 'no',
                           mulms_report=report, mulms_dir=mulms,
                           topic_reports=tmp_path / 'no_topics')
    assert info['mulms_files'] == outputs
    assert (out / 'mulms__a_01.conll').read_bytes() == (mulms / 'a_01.conll').read_bytes()
    (mulms / 'a_01.conll').write_text('changed B-MAT\n\n')
    with pytest.raises(SystemExit, match='mulms.py'):
        kj.build_bundle('HEAD', out, results_dir=tmp_path / 'none', corpus=corpus,
                        expected_sha256=kj.file_sha256(corpus), sofc_report=tmp_path / 'no',
                        mulms_report=report, mulms_dir=mulms,
                           topic_reports=tmp_path / 'no_topics')
    info = kj.build_bundle('HEAD', out, results_dir=tmp_path / 'none', corpus=corpus,
                           expected_sha256=kj.file_sha256(corpus), sofc_report=tmp_path / 'no',
                           mulms_report=tmp_path / 'no_mulms.json',
                           topic_reports=tmp_path / 'no_topics')
    assert info['mulms_files'] is None and not list(out.glob('mulms__*'))


def test_batch_spec_lists_the_topic_corpora_a_batch_reads(monkeypatch, tmp_path):
    """Week 12: a batch whose jobs read topic corpora lists them with their recorded SHA-256
    (results/topic_corpora/<topic>.json), in alphabetical order; other batches do not."""
    reports = tmp_path / 'topic_corpora'
    reports.mkdir()
    for topic, digest in (('sofc', 'c' * 64), ('steel', 'd' * 64)):
        (reports / f'{topic}.json').write_text(json.dumps({'corpus': {'sha256': digest}}))
    monkeypatch.setattr(kj, 'TOPIC_REPORTS', reports)
    real_sha256, real_lanes = kj.corpus_sha256, kj.batch_lanes
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: (
        real_sha256(report) if report is not None and Path(report).parent == reports else 'x' * 64))
    base = ['--granularity', 'mulms', '--leave-one-subfield-out', '--crf', '--train-pool', '400']
    jobs = [[*base, '--self-train', kj.TOPIC_CORPUS.format(topic='steel'), '--teachers', '3'],
            [*base, '--self-train', kj.TOPIC_CORPUS.format(topic='sofc'), '--teachers', '3']]
    monkeypatch.setattr(kj, 'batch_lanes',
                        lambda name: [jobs, [jobs[0]]] if name == 'w12x' else real_lanes(name))
    monkeypatch.setattr(kj, 'mulms_files', lambda: {'a_01.conll': 'a' * 64})
    spec = kj.batch_spec('w12x')
    assert spec['topic_corpora'] == {'sofc': 'c' * 64, 'steel': 'd' * 64}
    assert spec['mulms_files'] == {'a_01.conll': 'a' * 64}
    assert 'topic_corpora' not in kj.batch_spec('w9b')


def test_build_bundle_with_the_topic_corpora(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('One sentence.\n')
    folder = tmp_path / 'corpus_dir'
    folder.mkdir()
    (folder / 'topic_sofc_sentences.txt').write_text('A YSZ anode .\n')
    reports = tmp_path / 'topic_corpora'
    reports.mkdir()
    digest = kj.file_sha256(folder / 'topic_sofc_sentences.txt')
    (reports / 'sofc.json').write_text(json.dumps({'corpus': {'sha256': digest}}))
    common = dict(results_dir=tmp_path / 'none', corpus=corpus,
                  expected_sha256=kj.file_sha256(corpus), sofc_report=tmp_path / 'no',
                  mulms_report=tmp_path / 'no_mulms.json', topic_reports=reports,
                  topic_dir=folder)
    out = tmp_path / 'bundle'
    info = kj.build_bundle('HEAD', out, **common)
    assert info['topic_corpora'] == {'sofc': digest}
    assert (out / 'topic_sofc_sentences.txt').read_text() == 'A YSZ anode .\n'
    assert json.loads((out / 'BUNDLE.json').read_text())['topic_corpora'] == {'sofc': digest}
    (folder / 'topic_sofc_sentences.txt').write_text('changed\n')
    with pytest.raises(SystemExit, match='SHA-256'):
        kj.build_bundle('HEAD', out, **common)
    (folder / 'topic_sofc_sentences.txt').unlink()
    with pytest.raises(SystemExit, match='build_topic_corpus.py --topic sofc'):
        kj.build_bundle('HEAD', out, **common)
    info = kj.build_bundle('HEAD', out, **{**common, 'topic_reports': tmp_path / 'no'})
    assert info['topic_corpora'] is None and not list(out.glob('topic_*'))


def test_run_jobs_places_the_topic_corpora_only_when_asked(tmp_path):
    rj = _run_jobs()
    inputs, repo = tmp_path / 'input', tmp_path / 'repo'
    (inputs / 'bundle').mkdir(parents=True)
    corpus = inputs / 'bundle' / 'topic_sofc_sentences.txt'
    corpus.write_text('A YSZ anode .\n')
    assert rj.place_topic_corpora({}, inputs, repo) is None and not repo.exists()
    digest = rj.file_sha256(corpus)
    assert rj.place_topic_corpora({'topic_corpora': {'sofc': digest}}, inputs, repo) == {
        'sofc': digest}
    assert (repo / rj.TOPIC_CORPUS_DIR / 'topic_sofc_sentences.txt').read_bytes() == corpus.read_bytes()
    with pytest.raises(RuntimeError, match='SHA-256'):
        rj.place_topic_corpora({'topic_corpora': {'sofc': '0' * 64}}, inputs, repo)
    with pytest.raises(FileNotFoundError):
        rj.place_topic_corpora({'topic_corpora': {'steel': digest}}, inputs, repo)


def test_run_jobs_places_the_mulms_files_only_when_asked(tmp_path):
    rj = _run_jobs()
    inputs, repo = tmp_path / 'input', tmp_path / 'repo'
    (inputs / 'bundle').mkdir(parents=True)
    conll = inputs / 'bundle' / 'mulms__a_01.conll'
    conll.write_text('WO3 B-MAT\n\n')
    assert rj.place_mulms({}, inputs, repo) is None and not repo.exists()
    digest = rj.file_sha256(conll)
    assert rj.place_mulms({'mulms_files': {'a_01.conll': digest}}, inputs, repo) == 1
    assert (repo / rj.MULMS_DIR / 'a_01.conll').read_bytes() == conll.read_bytes()
    with pytest.raises(RuntimeError, match='SHA-256'):
        rj.place_mulms({'mulms_files': {'a_01.conll': '0' * 64}}, inputs, repo)
    with pytest.raises(FileNotFoundError):
        rj.place_mulms({'mulms_files': {'b_01.conll': digest}}, inputs, repo)


def test_week12_gazetteer_feature_batches(monkeypatch, tmp_path):
    """w12c (amendment 2): the match-feature variant's screen, one job per
    subfield over the two lanes, no copies; w12b:gzm: its decision over seed offsets 300 to
    900."""
    monkeypatch.setattr(nb.os.path, 'isfile', lambda path: True)
    monkeypatch.setattr(kj, 'MULMS_REPORT', tmp_path / 'no_mulms.json')
    monkeypatch.setattr(kj, 'mulms_files', lambda: {'a_01.conll': 'a' * 64})
    reports = tmp_path / 'topic_corpora'
    reports.mkdir()
    for topic in kj.MULMS_SUBFIELD_NAMES:
        (reports / f'{topic}.json').write_text(json.dumps({'corpus': {'sha256': topic * 8}}))
    monkeypatch.setattr(kj, 'TOPIC_REPORTS', reports)
    real_sha256 = kj.corpus_sha256
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: (
        real_sha256(report) if report is not None and Path(report).parent == reports else 'x' * 64))

    def kind(a):
        return nb.variant_name(nb.SimpleNamespace(**{**vars(a), 'tag': ''}))
    lane0, lane1 = _parsed(kj.batch_lanes('w12c'))
    assert lane0[0].tag == 'smokeaq' and lane0[0].granularity == 'fine'
    jobs = lane0[1:] + lane1
    assert (len(lane0), len(lane1)) == (8, 7)
    for name in ('crf_gzm_topic', 'crf_gzs_topic'):
        assert sorted(a.subfields for a in jobs if kind(a) == name) == sorted(
            [s] for s in kj.MULMS_SUBFIELD_NAMES)
    assert [kind(a) for a in lane0[1:]] == ['crf_gzm_topic'] * 7
    assert [kind(a) for a in lane1] == ['crf_gzs_topic'] * 7
    for a in jobs:
        assert a.names_feature and getattr(a, 'silver_mentions', False) == (
            kind(a) == 'crf_gzm_topic')
        assert not hasattr(a, 'augment_mentions') and not hasattr(a, 'augment_names')
        assert (a.teachers, a.students, a.student_offsets) == (3, 10, [0, 100, 200])
        assert a.self_train == f'data/processed/corpus/topic_{a.subfields[0]}_sentences.txt'
        assert a.crf and a.save_predictions and a.names_tag == 'topic'
    lane0, lane1 = _parsed(kj.batch_lanes('w12b:gzm'))
    assert lane0[0].tag == 'smokeap' and (len(lane0), len(lane1)) == (8, 7)
    for lane, offsets in ((lane0[1:], [300, 400, 500]), (lane1, [600, 700, 800, 900])):
        assert [a.subfields for a in lane] == [[s] for s in kj.MULMS_SUBFIELD_NAMES]
        for a in lane:
            assert kind(a) == 'crf_gzm_topic' and a.student_offsets == offsets
    assert 'w12c' in kj.BATCHES and {'w12b:gzm', 'w12b:gzs'} <= set(kj.BATCHES)
    # w12e: w12a's names variants without the 6-copy jobs, a variant's jobs in one lane
    lane0, lane1 = _parsed(kj.batch_lanes('w12e'))
    assert lane0[0].tag == 'smokeas' and (len(lane0), len(lane1)) == (15, 14)
    assert [kind(a) for a in lane0[1:]] == ['crf_mrs_topic'] * 7 + ['crf_mrm_topic6'] * 7
    assert [kind(a) for a in lane1] == ['crf_mrm_topic'] * 7 + ['crf_st3_topic'] * 7
    assert 'w12e' in kj.BATCHES
    for name, smoke, kinds in (('w12f', 'smokeat', ('crf_mrs_topic', 'crf_mrm_topic')),
                               ('w12g', 'smokeau', ('crf_mrm_topic6', 'crf_st3_topic'))):
        lane0, lane1 = _parsed(kj.batch_lanes(name))
        assert lane0[0].tag == smoke and (len(lane0), len(lane1)) == (8, 7)
        assert [kind(a) for a in lane0[1:]] == [kinds[0]] * 7
        assert [kind(a) for a in lane1] == [kinds[1]] * 7
        assert name in kj.BATCHES
    lane0, lane1 = _parsed(kj.batch_lanes('w12b:gzs'))
    assert all(kind(a) == 'crf_gzs_topic' and not hasattr(a, 'silver_mentions')
               for a in lane0[1:] + lane1)


def test_week12_analogue_batches(monkeypatch, tmp_path):
    """w12d:<variant>: the MaterioMiner analogue of the screen's winner, one lane per
    granularity, a job per held-out paper for the teacher variants (the mention-level silver
    names in the pool with 3 or 6 copies, or as a match feature) and a job per seed offset
    for crf_mr6; V1's analogue is w11b and V6's exists, so neither is a batch."""
    monkeypatch.setattr(nb.os.path, 'isfile', lambda path: True)    # the corpus is not here
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: 'x' * 64)

    def kind(a, tag='m300'):
        return nb.variant_name(nb.SimpleNamespace(**{**vars(a), 'tag': tag}))
    for variant, name, copies in (('mrm', 'crf_mrm', 3), ('mrm6', 'crf_mrm6', 6),
                                  ('gzm', 'crf_gzm', 0)):
        lane0, lane1 = _parsed(kj.batch_lanes(f'w12d:{variant}'))
        assert lane0[0].tag == 'smokear' and lane0[0].granularity == 'fine'
        for lane, g in ((lane0[1:], 'fine'), (lane1, 'coarse')):
            assert [a.papers for a in lane] == [[p] for p in kj.PAPERS]
            for a in lane:
                assert (a.granularity, nb.protocol_name(a), a.self_train) == (
                    g, 'leave_one_paper_out', kj.CORPUS)
                assert (a.teachers, a.students, a.self_train_sample) == (3, 10, 0)
                assert a.silver_mentions and a.crf and a.save_predictions
                assert getattr(a, 'augment_mentions', 0) == copies
                assert getattr(a, 'augment_names', False) == (variant != 'gzm')
                assert getattr(a, 'names_feature', False) == (variant == 'gzm')
                assert kind(a) == f'{name}_m300' and not hasattr(a, 'names_tag')
    lane0, lane1 = _parsed(kj.batch_lanes('w12d:mr6'))
    assert lane0[0].tag == 'smokear' and (len(lane0), len(lane1)) == (11, 10)
    for lane, g in ((lane0[1:], 'fine'), (lane1, 'coarse')):
        assert [a.tag for a in lane] == [f'm{off}' for off in range(0, 1000, 100)]
        for a in lane:
            assert (a.granularity, nb.protocol_name(a), a.augment_mentions) == (
                g, 'leave_one_paper_out', 6)
            assert not hasattr(a, 'self_train') and kind(a, a.tag) == f'crf_mr6_{a.tag}'
    for name in ('w12d:mrs', 'w12d:st3', 'w12d:x'):
        with pytest.raises(ValueError):
            kj.batch_lanes(name)
    assert {f'w12d:{v}' for v in ('mrm', 'mrm6', 'gzm', 'mr6')} <= set(kj.BATCHES)


def test_week13_sofc_folds_batch(monkeypatch):
    """w13a and w13c (week 13, 13.3): a repeat, then on SOFC-Exp's 5 article folds mention
    replacement (w13a, crf_mr_m<N>) or mention replacement on the multi-task model trained on
    both levels (w13c, sofc_both, crf_mt_mr_m<N>), seed offsets 0 to 900 split over the
    lanes (0-400, 500-900)."""
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: 'x' * 64)
    for name, smoke, granularity, variant in (('w13a', 'smokeav', 'sofc', 'crf_mr'),
                                              ('w13c', 'smokeax', 'sofc_both', 'crf_mt_mr')):
        lane0, lane1 = _parsed(kj.batch_lanes(name))
        assert lane0[0].tag == smoke and (len(lane0), len(lane1)) == (6, 5)
        assert [a.seed_offset for a in lane0[1:]] == [0, 100, 200, 300, 400]
        assert [a.seed_offset for a in lane1] == [500, 600, 700, 800, 900]
        for a in lane0[1:] + lane1:
            assert (a.granularity, nb.protocol_name(a), nb.variant_name(a)) == (
                granularity, 'document_folds', f'{variant}_m{a.seed_offset}')
            assert a.crf and a.augment_mentions == 3 and a.save_predictions
            assert a.document_folds == 5 and a.seeds == [0, 1, 2, 3, 4]
            assert _get(a, 'multi_task', False) == (granularity == 'sofc_both')
        assert name in kj.BATCHES
    # week 8's jobs are unchanged
    assert kj.sofc_split_job('document_folds', 300) == [
        '--granularity', 'sofc', '--document-folds', '5', '--crf', '--tag', 'm300',
        '--seed-offset', '300', '--save-predictions', '--threads', str(kj.THREADS)]
    assert 'w13a' in kj.BATCHES and 'sofc_corpus_sha256' not in kj.batch_spec('w13a')


def test_week13_folds_analogue_batch(monkeypatch):
    """w13b (13.4): a repeat, then the week-12 winner's analogue on SOFC-Exp's 5 article
    folds, one job per lane with the lane's student seed offsets (0-400, 500-900): 3 teachers
    tagging the whole SOFC corpus, mention replacement with the mention-level silver names
    and 6 copies (crf_mrm6); the batch carries the SOFC corpus's SHA-256."""
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: 'x' * 64)
    lane0, lane1 = _parsed(kj.batch_lanes('w13b'))
    assert lane0[0].tag == 'smokeaw' and (len(lane0), len(lane1)) == (2, 1)
    for a, offsets in ((lane0[1], [0, 100, 200, 300, 400]), (lane1[0], [500, 600, 700, 800, 900])):
        assert (a.granularity, nb.protocol_name(a), nb.variant_name(a)) == (
            'sofc', 'document_folds', 'crf_mrm6')
        assert a.document_folds == 5 and a.student_offsets == offsets
        assert a.self_train == kj.SOFC_CORPUS and a.self_train_sample == 0
        assert a.teachers == 3 and a.students == 10 and a.silver_mentions
        assert a.augment_mentions == 6 and a.augment_names and a.crf and a.save_predictions
    assert 'w13b' in kj.BATCHES and 'sofc_corpus_sha256' in kj.batch_spec('w13b')


def test_week14_copy_control_batch(monkeypatch):
    """w14a (week 14, 14.1): a repeat, then the 6-copy control without names on MuLMS's
    held-out subfields (seed offsets 300 to 900, one job per offset over the subfields) and
    on MaterioMiner's held-out papers at both granularities (offsets 0 to 900, one job per
    offset and granularity), the 27 jobs balanced over the lanes; no corpus in the spec."""
    monkeypatch.setattr(nb.os.path, 'isfile', lambda path: True)    # the corpus is not here
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: 'x' * 64)
    monkeypatch.setattr(kj, 'mulms_files', lambda: {'a_01.conll': 'a' * 64})
    lane0, lane1 = _parsed(kj.batch_lanes('w14a'))
    assert lane0[0].tag == 'smokeay' and lane0[0].granularity == 'fine'
    jobs = lane0[1:] + lane1
    assert len(jobs) == 27 and min(len(lane0) - 1, len(lane1)) >= 10
    mulms = [a for a in jobs if a.granularity == 'mulms']
    assert sorted(a.tag for a in mulms) == [f'm{off}' for off in range(300, 1000, 100)]
    for a in mulms:
        assert (a.train_pool, a.augment_mentions, a.crf, a.save_predictions) == (
            400, 6, True, True)
        assert nb.protocol_name(a) == 'leave_one_subfield_out' and not hasattr(a, 'teachers')
        assert nb.variant_name(a) == f'crf_mr6_{a.tag}'
        assert _get(a, 'seed_offset', 0) == int(a.tag[1:])
    for g in ('fine', 'coarse'):
        papers = [a for a in jobs if a.granularity == g]
        assert sorted(a.tag for a in papers) == sorted(f'm{off}' for off in range(0, 1000, 100))
        for a in papers:
            assert (nb.protocol_name(a), a.augment_mentions, a.crf) == (
                'leave_one_paper_out', 6, True)
            assert not hasattr(a, 'self_train') and nb.variant_name(a) == f'crf_mr6_{a.tag}'
            assert _get(a, 'seed_offset', 0) == int(a.tag[1:])
    spec = kj.batch_spec('w14a')
    assert spec['mulms_files'] == {'a_01.conll': 'a' * 64}
    assert 'topic_corpora' not in spec and 'sofc_corpus_sha256' not in spec
    assert 'w14a' in kj.BATCHES and kj.batch_name('w14a') == 'w14a'

def test_week14_mixed_corpus_and_ablation_batches(monkeypatch, tmp_path):
    """w14b (14.2): a repeat, then V4 with the mixed corpus of each held-out subfield, seed
    offsets 300 to 900 over w12b's lanes; w14c (14.3): V4's five ablations at seed offsets 0
    to 200, one job per subfield, each ablation's jobs in one lane; the spec lists the
    corpora a batch reads, the mixed ones by their own reports."""
    monkeypatch.setattr(nb.os.path, 'isfile', lambda path: True)    # the corpora are not here
    monkeypatch.setattr(kj, 'MULMS_REPORT', tmp_path / 'no_mulms.json')
    monkeypatch.setattr(kj, 'mulms_files', lambda: {'a_01.conll': 'a' * 64})
    reports = tmp_path / 'topic_corpora'
    reports.mkdir()
    sha = {}
    for topic in kj.MULMS_SUBFIELD_NAMES:
        for name in (topic, f'mixed_{topic}'):
            sha[name] = (name + '0' * 64)[:64]
            (reports / f'{name}.json').write_text(json.dumps({'corpus': {'sha256': sha[name]}}))
    monkeypatch.setattr(kj, 'TOPIC_REPORTS', reports)
    real_sha256 = kj.corpus_sha256
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: (
        real_sha256(report) if report is not None and Path(report).parent == reports else 'x' * 64))

    def kind(a):
        return nb.variant_name(nb.SimpleNamespace(**{**vars(a), 'tag': ''}))
    lane0, lane1 = _parsed(kj.batch_lanes('w14b'))
    assert lane0[0].tag == 'smokeaz' and lane0[0].granularity == 'fine'
    for lane, offsets in ((lane0[1:], [300, 400, 500]), (lane1, [600, 700, 800, 900])):
        assert [a.subfields for a in lane] == [[s] for s in kj.MULMS_SUBFIELD_NAMES]
        for a in lane:
            assert kind(a) == 'crf_mrm_mixed6' and a.student_offsets == offsets
            assert a.self_train == (f'data/processed/corpus/topic_mixed_{a.subfields[0]}'
                                    '_sentences.txt')
            assert (a.self_train_sample, a.teachers, a.students, a.augment_mentions) == (
                0, 3, 10, 6)
            assert a.silver_mentions and a.augment_names and a.names_tag == 'mixed'
            assert (a.granularity, a.train_pool, nb.protocol_name(a)) == (
                'mulms', 400, 'leave_one_subfield_out')
            assert not any(hasattr(a, o) for o in ('silver_threshold', 'teachers_agree',
                                                    'names_share'))
    spec = kj.batch_spec('w14b')
    assert spec['topic_corpora'] == {f'mixed_{t}': sha[f'mixed_{t}'] for t in kj.MULMS_SUBFIELD_NAMES}
    assert spec['mulms_files'] == {'a_01.conll': 'a' * 64}
    lane0, lane1 = _parsed(kj.batch_lanes('w14c'))
    assert lane0[0].tag == 'smokeba'
    expected = {'crf_mrm_topic10k6': {'self_train_sample': 10000},
                'crf_mrm_topic20k6': {'self_train_sample': 20000},
                'crf_mrm_topiclow6': {'silver_threshold': 0.7},
                'crf_mrm_topictwo6': {'teachers_agree': 2},
                'crf_mrm_topiccap6': {'names_share': 0.5}}
    assert [kind(a) for a in lane0[1:]] == [
        v for v in ('crf_mrm_topic10k6', 'crf_mrm_topic20k6', 'crf_mrm_topiclow6')
        for _ in kj.MULMS_SUBFIELD_NAMES]
    assert [kind(a) for a in lane1] == [v for v in ('crf_mrm_topictwo6', 'crf_mrm_topiccap6')
                                        for _ in kj.MULMS_SUBFIELD_NAMES]
    for a in lane0[1:] + lane1:
        assert a.student_offsets == [0, 100, 200] and a.augment_mentions == 6
        assert a.silver_mentions and a.augment_names and a.teachers == 3
        assert a.self_train == f'data/processed/corpus/topic_{a.subfields[0]}_sentences.txt'
        assert a.names_tag == kind(a)[len('crf_mrm_'):-1]
        settings = {o: getattr(a, o, None) for o in ('self_train_sample', 'silver_threshold',
                                                     'teachers_agree', 'names_share')}
        assert settings == {'self_train_sample': 0, 'silver_threshold': None,
                            'teachers_agree': None, 'names_share': None, **expected[kind(a)]}
    for name in ('crf_mrm_topic10k6', 'crf_mrm_topiccap6'):
        assert [a.subfields for a in lane0[1:] + lane1 if kind(a) == name] == [
            [s] for s in kj.MULMS_SUBFIELD_NAMES]
    spec = kj.batch_spec('w14c')
    assert spec['topic_corpora'] == {t: sha[t] for t in kj.MULMS_SUBFIELD_NAMES}
    assert {'w14b', 'w14c'} <= set(kj.BATCHES)
    assert kj.batch_spec('w12a')['topic_corpora'] == {t: sha[t] for t in kj.MULMS_SUBFIELD_NAMES}
