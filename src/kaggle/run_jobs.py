"""
run_jobs.py - runs NER jobs on a Kaggle GPU session (ROADMAP §4, week-4 protocol)

`kaggle_jobs.py push` writes a job spec into JOBS below and pushes this script as a
private Kaggle kernel with the bundle dataset attached (kaggle_jobs.py bundle: the code
of a frozen commit, the fatigue corpus and the GPU results so far). On Kaggle it:

  1. checks the GPUs: one per lane, each of the spec's type (`gpu`, e.g. Tesla T4); or, with
     `device` xla (a TPU batch), gives each lane one TPU chip (TPU_VISIBLE_CHIPS) and skips
     the GPU check;
  2. installs the spec's pinned packages (`pip`, from PyPI) and the bundle's wheels
     (`wheels`, packages that PyPI has as source only); torch is the Kaggle image's;
  3. unpacks the code, the corpus (its SHA-256 must be `corpus_sha256`), the SOFC corpus
     when the spec has `sofc_corpus_sha256` (bundle file sofc_sentences.txt, with that
     SHA-256), the MuLMS files when the spec lists them (`mulms_files`: bundle files
     mulms__<name>, each with its SHA-256), the topic corpora when the spec lists them
     (`topic_corpora`: bundle files topic_<topic>_sentences.txt, each with its SHA-256)
     and the GPU results so far (bundle files
     gpu__ner_results_*.json; every run must be a GPU run, i.e. have an `environment`), so
     that finished runs are skipped;
  4. downloads the pinned model snapshots (`models`: repository -> commit) and makes
     them the ones that `from_pretrained(<repository>)` loads; the jobs then run offline;
  5. runs the lanes at the same time, one per GPU (CUDA_VISIBLE_DEVICES=<lane>), and the
     jobs of a lane one after the other: `python src/graph/ner_baseline.py <job>`, with
     PROCESSED_DIR=/kaggle/working/processed. No job starts after `deadline_hours`;
  6. leaves in /kaggle/working: processed/ner/ (the results files that the jobs wrote or
     changed; the restored ones that no job changed are deleted at the end, so that a pull
     downloads only new files), logs/ (one log per job) and manifest.json (the bundle, the
     environment, and per job its command, exit code and times), rewritten after every job.

Nothing may hang (TIMEOUTS): nvidia-smi, pip and the environment's commands have time
limits, and a job still running `JOB_GRACE_HOURS` after `deadline_hours` is stopped, so
that a stuck session ends within minutes rather than at Kaggle's 12-hour limit. Each setup
step prints a line ("setup: ..."), so the live log (`kaggle kernels logs --follow`) shows
where a session is (a session of 17 October printed nothing for hours; ROADMAP §4, week 8).

It never exits with an error, so that Kaggle keeps the outputs of the jobs that ran.
Standard library only.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path

JOBS = None   # the job spec; kaggle_jobs.py push replaces this line

INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')
REPO = Path('/tmp/repo')
CORPUS = Path('data/processed/corpus/fatigue_sentences.txt')   # in the repository
SOFC_CORPUS = Path('data/processed/corpus/sofc_sentences.txt')  # with sofc_corpus_sha256
MULMS_DIR = Path('data/processed/mulms')                         # with mulms_files
MULMS_PREFIX = 'mulms__'           # the bundle names of the MuLMS files (kaggle_jobs.py)
TOPIC_CORPUS_DIR = Path('data/processed/corpus')                # with topic_corpora
ENV_KEYS = ('KAGGLE_KERNEL_RUN_TYPE', 'KAGGLE_DOCKER_IMAGE', 'KAGGLE_CONTAINER_NAME')
TIMEOUTS = {'nvidia_smi': 120, 'pip': 1800, 'environment': 300}   # seconds
JOB_GRACE_HOURS = 1.25   # kaggle_jobs.py pushes with a limit of deadline_hours + 1.5
MIN_JOB_SECONDS = 60     # a job started just before the stop still gets this long


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def find(root, name):
    """The first path called `name` under `root`, in sorted order (any mount layout)."""
    return next(iter(sorted(Path(root).rglob(name))), None)


def place_sofc_corpus(spec, inputs, repo):
    """The SOFC corpus into the repository when the spec asks for it (sofc_corpus_sha256):
    sofc_sentences.txt of the bundle, with that SHA-256. Returns the SHA-256, or None when
    the batch does not use it."""
    expected = spec.get('sofc_corpus_sha256')
    if not expected:
        return None
    corpus = find(inputs, 'sofc_sentences.txt')
    if corpus is None:
        raise FileNotFoundError(f'sofc_sentences.txt is not under {inputs}')
    digest = file_sha256(corpus)
    if digest != expected:
        raise RuntimeError(f'SOFC corpus SHA-256 {digest} is not {expected}')
    (repo / SOFC_CORPUS).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corpus, repo / SOFC_CORPUS)
    return digest


def place_topic_corpora(spec, inputs, repo):
    """The topic corpora (build_topic_corpus.py) into the repository when the spec lists them
    (topic_corpora: {topic: SHA-256}): bundle files topic_<topic>_sentences.txt, each with
    its SHA-256. Returns {topic: SHA-256}, or None when the batch uses none."""
    expected = spec.get('topic_corpora')
    if not expected:
        return None
    folder = repo / TOPIC_CORPUS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for topic, digest in sorted(expected.items()):
        name = f'topic_{topic}_sentences.txt'
        path = find(inputs, name)
        if path is None:
            raise FileNotFoundError(f'{name} is not under {inputs}')
        found = file_sha256(path)
        if found != digest:
            raise RuntimeError(f'topic corpus {topic}: SHA-256 {found} is not {digest}')
        shutil.copy2(path, folder / name)
    return dict(sorted(expected.items()))


def place_mulms(spec, inputs, repo):
    """The MuLMS files (mulms.py) into the repository when the spec lists them (mulms_files:
    {name: SHA-256}): bundle files mulms__<name>, each with its SHA-256. Returns how many, or
    None when the batch does not use them."""
    expected = spec.get('mulms_files')
    if not expected:
        return None
    folder = repo / MULMS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for name, digest in sorted(expected.items()):
        path = find(inputs, MULMS_PREFIX + name)
        if path is None:
            raise FileNotFoundError(f'{MULMS_PREFIX}{name} is not under {inputs}')
        found = file_sha256(path)
        if found != digest:
            raise RuntimeError(f'MuLMS file {name}: SHA-256 {found} is not {digest}')
        shutil.copy2(path, folder / name)
    return len(expected)


def unpack(root, archive, marker, dest):
    """Copy the tree that holds `marker` under `root` to `dest`, or extract `archive`.

    Kaggle may store an uploaded .tar.gz as it is or extract it; both are handled.
    `marker` is a path relative to the tree's top (src/graph/ner_baseline.py).
    """
    for found in sorted(Path(root).rglob(Path(marker).name)):
        if found.as_posix().endswith('/' + marker):
            top = Path(found.as_posix()[:-len(marker)])
            shutil.copytree(top, dest, dirs_exist_ok=True)
            return f'copied {top}'
    path = find(root, archive)
    if path is None or not path.is_file():
        raise FileNotFoundError(f'neither {marker} nor {archive} under {root}')
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as tar:
        tar.extractall(dest, filter='data')
    return f'extracted {path}'


RESULTS_PREFIX = 'gpu__'     # bundle name of a GPU results file: gpu__ner_results_<...>.json


def restore_results(root, dest):
    """Copy the bundle's GPU results files (gpu__ner_results_*.json) to dest, unprefixed.

    Every run in them must be a GPU run (have an `environment`): a CPU results file
    would make the GPU jobs skip its runs as done.
    """
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for path in sorted(Path(root).rglob(RESULTS_PREFIX + 'ner_results_*.json')):
        name = path.name[len(RESULTS_PREFIX):]
        runs = json.loads(path.read_text()).get('runs', [])
        cpu = [r.get('run_id') for r in runs if 'environment' not in r]
        if cpu:
            raise RuntimeError(f'{path.name}: runs {cpu} have no environment (not GPU runs)')
        shutil.copy2(path, dest / name)
        names.append(name)
    return names


def drop_unchanged(root, dest, names):
    """Delete the restored results files (`names`, in dest) that are still byte-identical to
    their bundle copy in root; returns the names deleted."""
    dropped = []
    for name in names:
        src = next(Path(root).rglob(RESULTS_PREFIX + name), None)
        out = Path(dest) / name
        if src is not None and out.is_file() and file_sha256(out) == file_sha256(src):
            out.unlink()
            dropped.append(name)
    return dropped


def gpu_names():
    out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                         capture_output=True, text=True, timeout=TIMEOUTS['nvidia_smi'])
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def check_gpus(names, lanes, wanted):
    """An error message if there are fewer GPUs than lanes or one is not `wanted`."""
    if len(names) < lanes:
        return f'{lanes} lanes need {lanes} GPUs; found {names}'
    wrong = [n for n in names[:lanes] if wanted and wanted not in n]
    if wrong:
        return f'GPUs {wrong} are not {wanted}'
    return None


def pin_models(models):
    """Download each repository at its pinned commit and point its `main` ref at it.

    from_pretrained(<repository>) then loads that snapshot with HF_HUB_OFFLINE=1.
    """
    from huggingface_hub import constants, snapshot_download
    pinned = {}
    for repo_id, commit in models.items():
        path = snapshot_download(repo_id, revision=commit)
        assert Path(path).name == commit, f'{repo_id}: got snapshot {path}, not {commit}'
        refs = Path(constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / 'refs'
        refs.mkdir(parents=True, exist_ok=True)
        (refs / 'main').write_text(commit)
        pinned[repo_id] = commit
    return pinned


def job_command(job):
    return [sys.executable, 'src/graph/ner_baseline.py', *job]


TPU_ENV = {'PJRT_DEVICE': 'TPU', 'TPU_PROCESS_BOUNDS': '1,1,1',
           'TPU_CHIPS_PER_PROCESS_BOUNDS': '1,1,1'}      # one chip per lane process


def job_env(lane, processed, device=None):
    """The environment of a lane's jobs: its GPU (CUDA_VISIBLE_DEVICES), or with device
    'xla' its TPU chip (TPU_VISIBLE_CHIPS, one chip per process), the processed directory
    and the offline switches."""
    env = dict(os.environ)
    if device == 'xla':
        env.update({**TPU_ENV, 'TPU_VISIBLE_CHIPS': str(lane)})
    else:
        env.update({'CUDA_VISIBLE_DEVICES': str(lane), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    env.update({'PROCESSED_DIR': str(processed), 'HF_HUB_OFFLINE': '1',
                'TRANSFORMERS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'})
    return env


class Manifest:
    """manifest.json, rewritten after every change (a killed session keeps what ran)."""

    def __init__(self, path, data):
        self.path, self.data, self.lock = path, data, threading.Lock()
        self.save()

    def update(self, fn):
        with self.lock:
            fn(self.data)
            self.save()

    def save(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)


def setup_step(started, step):
    """Print that a setup step is done, with the minutes since the start."""
    print(f'setup: {step} ({(time.time() - started) / 60:.1f} min)', flush=True)


def run_job(job, lane, processed, log, stop_at, device=None):
    """Run one job into `log`; its exit code, or None if it was stopped at `stop_at`."""
    with open(log, 'w') as out:
        try:
            return subprocess.run(job_command(job), cwd=REPO,
                                  env=job_env(lane, processed, device),
                                  stdout=out, stderr=subprocess.STDOUT,
                                  timeout=max(MIN_JOB_SECONDS, stop_at - time.time())).returncode
        except subprocess.TimeoutExpired:
            return None


def run_lane(lane, jobs, manifest, deadline, logs, processed, stop_at=None, device=None):
    """The jobs of a lane one after the other: none starts after `deadline`, and one still
    running at `stop_at` (default: JOB_GRACE_HOURS after the deadline) is stopped; device
    'xla' gives the lane a TPU chip (job_env)."""
    stop_at = deadline + 3600 * JOB_GRACE_HOURS if stop_at is None else stop_at
    for k, job in enumerate(jobs):
        record = {'lane': lane, 'index': k, 'args': job}
        if time.time() > deadline:
            record['status'] = 'not started: deadline'
            manifest.update(lambda d, r=record: d['jobs'].append(r))
            continue
        log = logs / f'lane{lane}_job{k:02d}.log'
        record.update({'log': log.name, 'start': time.strftime('%Y-%m-%dT%H:%M:%S')})
        begin = time.time()
        code = run_job(job, lane, processed, log, stop_at, device)
        record.update({'exit_code': code, 'minutes': round((time.time() - begin) / 60, 2),
                       'status': ('stopped: time limit' if code is None
                                  else 'done' if code == 0 else 'failed')})
        manifest.update(lambda d, r=record: d['jobs'].append(r))
        print(f'lane {lane} job {k}: {record["status"]} in {record["minutes"]} min', flush=True)


def environment(device=None):
    """The session's software: python, torch, transformers, torch_xla (a TPU: its device type
    and chips), nvidia-smi (a GPU; 'not found' without it), pip freeze and Kaggle's variables."""
    def run(cmd):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=TIMEOUTS['environment'])
        except subprocess.TimeoutExpired:
            return 'timed out'
        except OSError as exc:                     # no such command (nvidia-smi on a TPU VM)
            return f'not found: {exc}'
        return (out.stdout + out.stderr).strip() if out.returncode else out.stdout.strip()
    torch_info = run([sys.executable, '-c',
                      'import torch, transformers; print(torch.__version__, torch.version.cuda, '
                      'transformers.__version__)'])
    xla_info = run([sys.executable, '-c',
                    'import torch_xla, torch_xla.runtime as xr; print(torch_xla.__version__, '
                    'xr.device_type(), xr.addressable_device_count())'])
    return {'python': sys.version.split()[0], 'torch_cuda_transformers': torch_info,
            'torch_xla': xla_info or None,
            'nvidia_smi': 'skipped: xla' if device == 'xla' else run(['nvidia-smi']),
            'pip_freeze': run([sys.executable, '-m', 'pip', 'freeze']).splitlines(),
            **{k: os.environ.get(k) for k in ENV_KEYS}}


def main(spec):
    started = time.time()
    logs, processed = WORK / 'logs', WORK / 'processed'
    logs.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(WORK / 'manifest.json', {'spec': spec, 'jobs': [], 'setup': {}})
    setup_step(started, 'started')
    device = spec.get('device')
    try:
        if device == 'xla':
            manifest.update(lambda d: d['setup'].update({'tpu_lanes': len(spec['lanes'])}))
            setup_step(started, f"TPU, {len(spec['lanes'])} lanes (one chip each)")
        else:
            names = gpu_names()
            manifest.update(lambda d: d['setup'].update({'gpus': names}))
            error = check_gpus(names, len(spec['lanes']), spec.get('gpu'))
            if error:
                raise RuntimeError(error)
            setup_step(started, f'GPUs {names}')
        wheels = [find(INPUT, name) for name in spec.get('wheels', [])]
        if None in wheels:
            raise FileNotFoundError(f"wheels {spec.get('wheels')} not all under {INPUT}")
        pip = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *spec['pip'],
                              *map(str, wheels)], capture_output=True, text=True,
                             timeout=TIMEOUTS['pip'])
        manifest.update(lambda d: d['setup'].update({'pip': pip.returncode,
                                                     'pip_output': (pip.stdout + pip.stderr)[-12000:]}))
        if pip.returncode:
            raise RuntimeError('pip install failed')
        setup_step(started, 'pip install')
        how = unpack(INPUT, 'code.tar.gz', 'src/graph/ner_baseline.py', REPO)
        bundle = json.loads(find(INPUT, 'BUNDLE.json').read_text())
        corpus = find(INPUT, 'fatigue_sentences.txt')
        digest = file_sha256(corpus)
        if digest != spec['corpus_sha256']:
            raise RuntimeError(f'corpus SHA-256 {digest} is not {spec["corpus_sha256"]}')
        (REPO / CORPUS).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(corpus, REPO / CORPUS)
        sofc_digest = place_sofc_corpus(spec, INPUT, REPO)
        mulms = place_mulms(spec, INPUT, REPO)
        topics = place_topic_corpora(spec, INPUT, REPO)
        restored = restore_results(INPUT, processed / 'ner')
        setup_step(started, f'code, corpora and {len(restored)} results files in place')
        pinned = pin_models(spec['models'])
        setup_step(started, 'models pinned')
        manifest.update(lambda d: d['setup'].update({
            'code': how, 'bundle': bundle, 'corpus_sha256': digest,
            **({'sofc_corpus_sha256': sofc_digest} if sofc_digest else {}),
            **({'mulms_files': mulms} if mulms else {}),
            **({'topic_corpora': topics} if topics else {}), 'restored': restored,
            'models': pinned, 'environment': environment(device),
            'setup_minutes': round((time.time() - started) / 60, 2)}))
        setup_step(started, 'environment recorded; the lanes start')
        deadline = started + 3600 * spec.get('deadline_hours', 10)
        threads = [threading.Thread(target=run_lane,
                                    args=(lane, jobs, manifest, deadline, logs, processed, None,
                                          device))
                   for lane, jobs in enumerate(spec['lanes'])]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dropped = drop_unchanged(INPUT, processed / 'ner', restored)
        manifest.update(lambda d: d['setup'].update({'unchanged_dropped': dropped}))
    except Exception:
        message = traceback.format_exc()
        print(message, flush=True)
        manifest.update(lambda d: d['setup'].update({'error': message}))
    manifest.update(lambda d: d.update({'total_minutes': round((time.time() - started) / 60, 2)}))
    print(json.dumps({'jobs': [(j['lane'], j['index'], j['status']) for j in
                               manifest.data['jobs']], 'error': manifest.data['setup'].get('error')}),
          flush=True)


if __name__ == '__main__':
    main(JOBS)
