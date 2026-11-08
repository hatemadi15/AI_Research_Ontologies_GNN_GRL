"""
sota_p2.py - the SOTA plan's P2: the decisions (docs/ROADMAP.md §4, the P2 protocol;
docs/SOTA_PLAN.md §1, §2, §4 and §5)

P2 reads each final system's test set once, on fresh seeds (1000 up). It makes 15 live
decisions (2 of them gated on a D-T) and 4 re-tests (MaterioMiner's, whose outcome is known).
No P2 result changes a system, a setting, a rule or a wording. This script:
  - writes the batches' jobs: p2a (the checks, A6's 4 extra screen units on SOFC-Exp,
    SOFC-Exp's official split and MaterioMiner), p2m (MuLMS) and p2b (SOFC-Exp's
    cross-validation, whose system P1's tie rule picks without B1);
  - runs a pulled batch's checks, which read only the check files: smoke4's repeat in every
    batch, and a repeat of a known run on the batch's paths;
  - takes A6's screen file from p2a and commits the CV system before any test-scored file is
    opened;
  - checks and aggregates the files (results/gpu_p2), forms MaterioMiner's votes, computes
    the decisions, the paired bootstrap over test sentences and the wordings, and writes
    results/gpu/sota_p2.json.

Usage:
  python src/graph/sota_p2.py jobs p2a|p2m|p2b          # results/sota/p2_jobs_<batch>.json
  python src/graph/sota_p2.py checks <pulled> --batch B  # first, after every pull
  python src/graph/sota_p2.py cv-system <pulled p2a>     # results/sota/p2_cv_system.json
  python src/graph/sota_p2.py decide                     # results/gpu/sota_p2.json
"""
import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'graph'))

import sota_p0  # noqa: E402
import sota_p1  # noqa: E402

SOTA_DIR = ROOT / 'results' / 'sota'
GPU_DIR = ROOT / 'results' / 'gpu'
SCREEN_DIR = ROOT / 'results' / 'gpu_screen'
P2_DIR = ROOT / 'results' / 'gpu_p2'               # P2's files, apart from every other analysis
OUT = GPU_DIR / 'sota_p2.json'
MULMS_NESTED_SCRIPT = sota_p1.MULMS_NESTED_SCRIPT
DEBERTA = sota_p1.DEBERTA
THREADS = 2

# ---------------------------------------------------------------------------------------------
# Seeds and units
# ---------------------------------------------------------------------------------------------
SEEDS = tuple(range(1000, 2000, 100))              # single models: 10 fresh seeds
VOTE_SEEDS = tuple(range(1000, 4000, 100))         # MaterioMiner's votes: 30, the first 10 shared
MULMS_UNITS = tuple((c, s) for s in (1000, 1100) for c in range(1, 6))
CV_OFFSETS = SEEDS
CV_FOLDS = ('cv1', 'cv2', 'cv3', 'cv4', 'cv5')
LEVELS = ('fine', 'coarse')
VOTE_SIZES = (3, 5)
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 0

# ---------------------------------------------------------------------------------------------
# The systems: their flags (P1's settings), tags and the seeds they run at
# ---------------------------------------------------------------------------------------------
MM_FLAGS = {
    ('fine', 'c1'): ('--granularity', 'both', '--multi-task', '--score-level', 'fine', '--crf',
                     '--published-split'),
    ('coarse', 'a3a6'): ('--granularity', 'coarse', '--crf', '--published-split', '--epochs',
                         '40', '--patience', '5', '--reinit-top', '1'),
    ('fine', 'a0'): ('--granularity', 'fine', '--crf', '--published-split'),
    ('coarse', 'a0'): ('--granularity', 'coarse', '--crf', '--published-split'),
    ('fine', 'plain'): ('--granularity', 'fine', '--published-split'),
    ('fine', 'a6'): ('--granularity', 'fine', '--crf', '--published-split', '--reinit-top', '1'),
    ('coarse', 'plain'): ('--granularity', 'coarse', '--published-split'),
}
MM_SELECTED = {'fine': 'c1', 'coarse': 'a3a6'}
MM_THIRTY = ('c1', 'a3a6', 'a0')       # the systems whose votes are decided run at 30 seeds
MM_TARGETS = {'fine': 69.92, 'coarse': 72.32}     # Durmaz et al. 2024, no SD printed
OFFICIAL_FLAGS = {
    'b1': ('--granularity', 'sofc', '--crf', '--published-split', '--model', DEBERTA, '--lr',
           '1e-5', '--batch-size', '8', '--gradient-checkpointing'),
    'a6': ('--granularity', 'sofc', '--crf', '--published-split', '--reinit-top', '1'),
}
CV_FLAGS = {
    'a6c1': ('--granularity', 'sofc_both', '--multi-task', '--crf', '--sofc-cv', '--reinit-top',
             '1'),
    'c1': ('--granularity', 'sofc_both', '--multi-task', '--crf', '--sofc-cv'),
    'a6': ('--granularity', 'sofc', '--crf', '--sofc-cv', '--reinit-top', '1'),
    'a0': ('--granularity', 'sofc', '--crf', '--sofc-cv'),
}
CV_CANDIDATES = {'a6c1': ('a6', 'c1'), 'c1': ('c1',), 'a6': ('a6',)}   # P1's tie set without B1
MULMS_FLAGS = {'a1m': ('--lr', '5e-5'), 'a0': ()}
SMOKE_JOB = ['--granularity', 'fine', '--vary-split', '--seeds', '0', '--crf',
             '--save-predictions', '--threads', str(THREADS), '--tag', 'smokebu']
CHECK_JOBS = {   # A0 at seed 0 in P2's mode, which must equal the stored runs
    'fine': ['--granularity', 'fine', '--crf', '--published-split', '--seeds', '0',
             '--save-predictions', '--tag', 'p2chk', '--threads', str(THREADS)],
    'sofc': ['--granularity', 'sofc', '--crf', '--published-split', '--seeds', '0',
             '--save-predictions', '--tag', 'p2chk', '--threads', str(THREADS)],
}
STORED_A0 = {'fine': 'ner_results_fine_published_split_crf.json',
             'coarse': 'ner_results_coarse_published_split_crf.json',
             'sofc': 'ner_results_sofc_published_split_crf.json'}
A6_EXTRA_JOB = ['--granularity', 'sofc', '--crf', '--screen', '--published-split', '--reinit-top',
                '1', '--seeds', '500', '600', '700', '800', '--tag', 'p1a6', '--threads',
                str(THREADS)]
CV_SYSTEM_FILE = SOTA_DIR / 'p2_cv_system.json'
MULMS_SMOKE_JOB = [MULMS_NESTED_SCRIPT, '--configs', '1', '--tag', 'smokebv', '--max-train', '300',
                   '--epochs', '2', '--save-predictions', '--threads', str(THREADS)]
MULMS_SMOKE_REFERENCE = ROOT / 'results' / 'gpu_repeats' / \
    'ner_results_mulms_nested_configs_bilou_crf_smokebr_smoke300.json'
CV_CHECK_JOB = ['--granularity', 'sofc', '--sofc-cv', '--seeds', '0', '--crf', '--tag', 'p2chkcv',
                '--save-predictions', '--threads', str(THREADS)]
A6_SCREEN_FILE = 'ner_results_sofc_published_screen_crf_p1a6.json'

# minutes per run, measured in P1 (screen mode; 15% is added for the test sets), for the lanes
MINUTES = {'mm_c1': 1.69, 'mm_a3a6': 2.01, 'mm_a0': 1.33, 'mm_plain': 1.1, 'mm_a6': 1.29,
           'b1': 21.15,
           'a6': 3.5, 'a0_sofc': 3.3, 'mulms_a1m': 31.52, 'mulms_a0': 26.67, 'cv_a6c1': 5.91,
           'cv_c1': 5.16, 'cv_a6': 3.5, 'cv_a0': 3.3, 'smoke': 1.2}
TEST_OVERHEAD = 1.15


def seeds_args(seeds):
    return ['--seeds', *(str(s) for s in seeds)]


def mm_seeds(level, cell):
    return VOTE_SEEDS if cell in (MM_SELECTED[level], 'a0') else SEEDS


def mm_job(level, cell):
    seeds = mm_seeds(level, cell)
    return [*MM_FLAGS[(level, cell)], *seeds_args(seeds), '--save-predictions', '--tag',
            f'p2{cell}', '--threads', str(THREADS)]


def official_job(cell):
    return [*OFFICIAL_FLAGS[cell], *seeds_args(SEEDS), '--save-predictions', '--tag', f'p2{cell}',
            '--threads', str(THREADS)]


def cv_job(cell, offset):
    return [*CV_FLAGS[cell], '--seeds', '0', '--seed-offset', str(offset), '--save-predictions',
            '--tag', f'p2{cell}m{offset}', '--threads', str(THREADS)]


def mulms_job(cell, config, seed):
    return [MULMS_NESTED_SCRIPT, '--configs', str(config), '--seed', str(seed),
            *MULMS_FLAGS[cell], '--save-predictions', '--tag', f'p2{cell}u{config}s{seed}',
            '--threads', str(THREADS)]


def cv_reference(system):
    """The CV decision's reference: the tuned reference A6 when the system holds C1, the CRF
    head A0 when the system is A6 (which reads "tuning")."""
    return 'a0' if system == 'a6' else 'a6'


def batch_jobs(batch, cv_system=None):
    """[(minutes, job)] of a batch; the checks come first (minutes 0, kept at the lanes'
    heads by kaggle_jobs.p2_lanes)."""
    if batch == 'p2a':
        jobs = [(0.0, SMOKE_JOB), (0.0, CHECK_JOBS['fine']), (0.0, CHECK_JOBS['sofc']),
                (4 * MINUTES['a6'], A6_EXTRA_JOB)]
        for cell in ('b1', 'a6'):
            jobs.append((len(SEEDS) * MINUTES[cell] * TEST_OVERHEAD, official_job(cell)))
        for level in LEVELS:
            cells = (MM_SELECTED[level], 'a0', 'plain') + (('a6',) if level == 'fine' else ())
            for cell in cells:
                n = len(mm_seeds(level, cell))
                jobs.append((n * MINUTES['mm_' + cell] * TEST_OVERHEAD, mm_job(level, cell)))
        return jobs
    if batch == 'p2m':
        return [(0.0, SMOKE_JOB), (0.0, MULMS_SMOKE_JOB)] + [
            (MINUTES['mulms_' + cell] * TEST_OVERHEAD, mulms_job(cell, c, s))
            for cell in ('a1m', 'a0') for c, s in MULMS_UNITS]
    if batch == 'p2b':
        if cv_system is None:
            raise SystemExit('p2b needs the CV system (sota_p2.py cv-system, after p2a)')
        return [(0.0, SMOKE_JOB), (0.0, CV_CHECK_JOB)] + [
            (5 * MINUTES['cv_' + cell] * TEST_OVERHEAD, cv_job(cell, off))
            for cell in (cv_system, cv_reference(cv_system)) for off in CV_OFFSETS]
    raise SystemExit(f'unknown P2 batch {batch!r}')


def jobs_file(batch):
    return SOTA_DIR / f'p2_jobs_{batch}.json'


def write_jobs(batch):
    cv_system = None
    if batch == 'p2b':
        cv_system = json.loads(CV_SYSTEM_FILE.read_text())['cv_system']
    entries = [{'minutes': round(m, 1), 'job': job} for m, job in batch_jobs(batch, cv_system)]
    jobs_file(batch).write_text(json.dumps(entries, indent=1) + '\n')
    return entries


# ---------------------------------------------------------------------------------------------
# SOFC-Exp's CV system: P1's tie rule without B1
# ---------------------------------------------------------------------------------------------
def cv_system_record(dirs=(SCREEN_DIR,)):
    """P1's tie rule on SOFC-Exp's other cells: A6 with C1, C1 and A6, each over all 9 units
    (seeds 0-800), the highest mean chosen, a further tie to fewer changes, then the name."""
    units = sota_p1.UNITS['sofc'] + sota_p1.EXTRA_UNITS['sofc']
    means = {}
    for cell, changes in CV_CANDIDATES.items():
        values = sota_p1.read_cell('sofc', frozenset(changes), units, dirs)
        missing = [u for u in units if u not in values]
        if missing:
            raise SystemExit(f'{cell}: units {missing} have not run')
        means[cell] = round(statistics.fmean(values[u][0] for u in units), 2)
    order = sorted(means, key=lambda c: (-means[c], len(CV_CANDIDATES[c]), c))
    return {'note': ("SOFC-Exp's CV system (ROADMAP §4, P2): P1's tie rule without B1, which the "
                     "plan's fallback keeps out of the CV protocol"),
            'units': units, 'means_over_9_units': means, 'cv_system': order[0],
            'reference': cv_reference(order[0])}


# ---------------------------------------------------------------------------------------------
# The checks of a pulled batch
# ---------------------------------------------------------------------------------------------
def same_runs(new_runs, old_runs, keys=('f1', 'history', 'test_index', 'test_gold',
                                         'test_pred')):
    """Problems of a repeat: per run_id of the stored file, the keys that differ."""
    new = {r['run_id']: r for r in new_runs}
    out = []
    for old in old_runs:
        run = new.get(old['run_id'])
        if run is None:
            out.append(f"{old['run_id']} is missing")
            continue
        out += [f"{old['run_id']}: {k} differs" for k in keys if run.get(k) != old.get(k)]
    return out


def check_repeat(found, granularity, results_dir=GPU_DIR):
    """Problems of A0's seed-0 repeat in P2's mode (tag p2chk) against the stored run: its F1,
    history, test sentences, gold and predictions must be equal."""
    name = f'ner_results_{granularity}_published_split_crf_p2chk.json'
    if name not in found:
        return [f'{name} is missing']
    old = [r for r in json.loads((Path(results_dir) / STORED_A0[granularity]).read_text())['runs']
           if r['run_id'] == 'seed0']
    return same_runs(json.loads(found[name].read_text())['runs'], old)


def batch_checks(pulled, batch):
    """{check: problems} of a pulled P2 batch ([] when it passes): smoke4's repeat in every
    batch, and the batch's repeat of a known run (p2a: A0 at seed 0 on MaterioMiner
    fine-grained and SOFC-Exp's official split; p2m: P0's short MuLMS run smokebr; p2b: P0's
    CV run at offset 0). Only these files are read: no test-scored file of the batch is."""
    found = {p.name: p for p in Path(pulled).rglob('ner_results_*.json')}
    out = {}
    smoke = found.get('ner_results_fine_random_split_vary_crf_smokebu.json')
    if smoke is None:
        out['smoke'] = ['smokebu is missing']
    else:
        a = sota_p1.without(json.loads(smoke.read_text()), 'smokebu')
        b = sota_p1.without(json.loads(sota_p1.SMOKE_REFERENCE.read_text()), 'smokebn')
        out['smoke'] = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if batch == 'p2a':
        for granularity in ('fine', 'sofc'):
            out[f'a0_repeat_{granularity}'] = check_repeat(found, granularity)
    elif batch == 'p2m':
        name = 'ner_results_mulms_nested_configs_bilou_crf_smokebv_smoke300.json'
        if name not in found:
            out['mulms_smoke'] = [f'{name} is missing']
        else:
            out['mulms_smoke'] = sota_p0.smoke_identical(
                json.loads(found[name].read_text()),
                json.loads(MULMS_SMOKE_REFERENCE.read_text()))
    elif batch == 'p2b':
        name = 'ner_results_sofc_matscibert_cv_crf_p2chkcv.json'
        if name not in found:
            out['cv_repeat'] = [f'{name} is missing']
        else:
            old = json.loads((GPU_DIR / sota_p0.CV_FILE.format(offset=0)).read_text())['runs']
            out['cv_repeat'] = same_runs(json.loads(found[name].read_text())['runs'], old)
    else:
        raise SystemExit(f'unknown P2 batch {batch!r}')
    return out


def take_a6_screen(pulled, screen_dir=SCREEN_DIR):
    """Copy A6's screen file on SOFC-Exp (its 4 extra units, in p2a) from a pulled batch to the
    screen files, after checking that it is a screen file with all 9 units; returns its path.
    It is the only file of p2a read before the CV system is committed."""
    import shutil
    path = next(Path(pulled).rglob(A6_SCREEN_FILE), None)
    if path is None:
        raise SystemExit(f'{A6_SCREEN_FILE} is not in {pulled}')
    data = json.loads(path.read_text())
    sota_p1.check_screen_file(data, A6_SCREEN_FILE)
    sota_p1.check_settings(data, A6_EXTRA_JOB, A6_SCREEN_FILE)
    units = sota_p1.UNITS['sofc'] + sota_p1.EXTRA_UNITS['sofc']
    have = {r['run_id'] for r in data['runs']}
    if not set(units) <= have:
        raise SystemExit(f'{A6_SCREEN_FILE}: units {sorted(set(units) - have)} are missing')
    target = Path(screen_dir) / A6_SCREEN_FILE
    shutil.copy2(path, target)
    return target

# ---------------------------------------------------------------------------------------------
# The files: checks and values
# ---------------------------------------------------------------------------------------------
def load_file(name, dirs=(P2_DIR,)):
    path = next((Path(d) / name for d in dirs if (Path(d) / name).exists()), None)
    if path is None:
        raise FileNotFoundError(name)
    return json.loads(path.read_text())


def tagger_file(job, granularity):
    import ner_baseline as nb
    return Path(nb.results_path('', granularity, nb.parse_args(job))).name


def mulms_file(job):
    return sota_p1.results_file('mulms', job)


def check_p2_file(data, job, name, run_ids, reference=None):
    """Raise ValueError unless a P2 file is a full (test-scored) file with the job's settings, the
    expected runs, P1's environment, and (with `reference`, a stored file on the same protocol)
    the same test sentences and gold in every run."""
    if data.get('screen'):
        raise ValueError(f'{name}: a screen file')
    sota_p1.check_settings(data, job, name)
    runs = {r['run_id']: r for r in data['runs']}
    missing = [r for r in run_ids if r not in runs]
    if missing:
        raise ValueError(f'{name}: runs {missing} are missing')
    for run_id in run_ids:
        sota_p1.check_environment(runs[run_id], name)
    if reference is not None:
        ref = next(iter(reference['runs']))
        for run_id in run_ids:
            for key in ('test_index', 'test_gold'):
                if key in ref and runs[run_id].get(key) != ref[key]:
                    raise ValueError(f'{name}: {run_id} {key} differs from the stored runs')
    return runs


def released_values(level, cell, dirs=(P2_DIR,)):
    """{seed: run} of a MaterioMiner system on the released split, its file checked."""
    job = mm_job(level, cell)
    name = tagger_file(job, level)
    data = load_file(name, dirs)
    seeds = mm_seeds(level, cell)
    reference = json.loads((GPU_DIR / STORED_A0[level]).read_text())
    runs = check_p2_file(data, job, name, [f'seed{s}' for s in seeds], reference)
    return {s: runs[f'seed{s}'] for s in seeds}


def official_values(cell, dirs=(P2_DIR,)):
    """{seed: {'f1_11', 'f1_10', 'run'}} of a system on SOFC-Exp's official split: the 11 test
    articles (the stored F1, which the re-score must equal) and the 10 without PMC6247067."""
    from seqeval.metrics import f1_score
    import ner_baseline as nb
    job = official_job(cell)
    name = tagger_file(job, 'sofc')
    data = load_file(name, dirs)
    reference = json.loads((GPU_DIR / STORED_A0['sofc']).read_text())
    runs = check_p2_file(data, job, name, [f'seed{s}' for s in SEEDS], reference)
    args = nb.parse_args(['--granularity', 'sofc', '--published-split', '--crf'])
    corpus, _, _ = nb.granularity_data('sofc', args)
    out = {}
    for s in SEEDS:
        run = runs[f'seed{s}']
        docs = [corpus[i][2].removesuffix('.conll') for i in run['test_index']]
        gold = [g.split() for g in run['test_gold']]
        pred = [p.split() for p in run['test_pred']]
        keep = [k for k, d in enumerate(docs) if d != sota_p0.TEN_ARTICLES_DROP]
        eleven = round(100 * f1_score(gold, pred), 2)
        if eleven != run['f1']:
            raise ValueError(f'{name} seed{s}: re-scored {eleven}, stored {run["f1"]}')
        ten = round(100 * f1_score([gold[k] for k in keep], [pred[k] for k in keep]), 2)
        out[s] = {'f1_11': eleven, 'f1_10': ten, 'run': run, 'keep_10': keep}
    return out


def cv_values(cell, dirs=(P2_DIR,)):
    """{(fold, offset): run} of a system on SOFC-Exp's CV folds (the entity head's file for a
    multi-task system), every file checked against P0's CV file's test sentences."""
    out = {}
    reference = json.loads((GPU_DIR / sota_p0.CV_FILE.format(offset=0)).read_text())
    by_fold = {r['run_id']: r for r in reference['runs']}
    for off in CV_OFFSETS:
        job = cv_job(cell, off)
        name = tagger_file(job, 'sofc')
        data = load_file(name, dirs)
        runs = check_p2_file(data, job, name, list(CV_FOLDS))
        for fold in CV_FOLDS:
            run = runs[fold]
            if run.get('seed') != off:
                raise ValueError(f'{name} {fold}: seed {run.get("seed")}, not {off}')
            for key in ('test_index', 'test_gold'):
                if run.get(key) != by_fold[fold].get(key):
                    raise ValueError(f'{name} {fold}: {key} differs from P0 cv runs')
            out[(fold, off)] = run
    return out


def mulms_values(cell, dirs=(P2_DIR,)):
    """{(configuration, seed): run} of a MuLMS system, every file checked; the number of test
    sentences and predictions must be P0's."""
    p0 = json.loads((GPU_DIR / sota_p1.P0_MULMS_FILE.format(c=1)).read_text())['runs'][0]
    out = {}
    for c, s in MULMS_UNITS:
        job = mulms_job(cell, c, s)
        name = mulms_file(job)
        data = load_file(name, dirs)
        runs = check_p2_file(data, job, name, [f'c{c}'])
        run = runs[f'c{c}']
        if run.get('seed') != s or 'test' not in run:
            raise ValueError(f'{name}: seed {run.get("seed")} or no test score')
        if run.get('n_test') != p0['n_test'] or len(run.get('test_pred', [])) != \
                len(p0['test_pred']):
            raise ValueError(f'{name}: test sentences differ from P0\'s')
        out[(c, s)] = run
    return out


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------
def t_interval(values):
    """(mean, 95% t-interval) of a sample."""
    from scipy import stats
    n = len(values)
    mean = statistics.fmean(values)
    half = float(stats.t.ppf(0.975, n - 1)) * statistics.stdev(values) / math.sqrt(n)
    return mean, [mean - half, mean + half]


def rounded(record):
    return {k: (round(v, 2) if isinstance(v, float) else
                [round(x, 2) for x in v] if isinstance(v, list) and v and
                isinstance(v[0], float) else v) for k, v in record.items()}


def paired(a, b):
    """S minus R paired by unit: {estimate, ci95, n, sd}."""
    units = sorted(a)
    if sorted(b) != units:
        raise ValueError('paired systems need the same units')
    diffs = [a[u] - b[u] for u in units]
    mean, ci = t_interval(diffs)
    return rounded({'estimate': mean, 'ci95': ci, 'n': len(diffs),
                    'sd': statistics.stdev(diffs)})


def welch(a, b):
    """mean(a) minus mean(b) with Welch's 95% interval, from two samples (a zero-width
    interval when neither sample varies)."""
    if statistics.stdev(a) == 0 and statistics.stdev(b) == 0:
        gap = round(statistics.fmean(a) - statistics.fmean(b), 2)
        return {'estimate': gap, 'ci95': [gap, gap], 'df': None, 'n': [len(a), len(b)]}
    test = sota_p0.welch_summary(
        {'mean': statistics.fmean(a), 'sd': statistics.stdev(a), 'n': len(a)},
        {'mean': statistics.fmean(b), 'sd': statistics.stdev(b), 'n': len(b)})
    return {'estimate': test['gap'], 'ci95': test['ci95'], 'df': test['df'],
            'n': [len(a), len(b)]}


def one_sample(values, target):
    """mean minus a fixed target, with its 95% t-interval."""
    mean, ci = t_interval([v - target for v in values])
    return rounded({'estimate': mean, 'ci95': ci, 'n': len(values), 'target': target})


def against_published(values, key):
    """D-T: our runs against a published (mean, SD, n) by Welch's test, P0's conventions; the
    verdict passes when the interval lies above 0; borderline when it changes at either end
    of the printed SD's rounding range."""
    published = sota_p0.PUBLISHED[key]
    ours = {'mean': statistics.fmean(values), 'sd': statistics.stdev(values), 'n': len(values)}

    def at(printed, mean):
        test = sota_p0.welch_summary(ours, {'mean': mean,
                                            'sd': sota_p0.sample_sd(published, printed),
                                            'n': published['n']})
        return test, test['ci95'][0] > 0
    test, passed = at(published['sd_printed'], published['mean'])
    half = 0.5 * 10 ** -published['decimals']
    edges = {at(published['sd_printed'] + d, published['mean'] + m)[1]
             for d in (-half, 0, half) for m in (-half, 0, half)}
    record = {'estimate': test['gap'], 'ci95': test['ci95'], 'df': test['df'],
              'ours': {'mean': round(ours['mean'], 2), 'sd': round(ours['sd'], 2),
                       'n': ours['n']},
              'published': {**published, 'sd_sample': round(sota_p0.sample_sd(published), 3)},
              'passed': passed, 'borderline': edges != {passed}}
    if published['convention'] == 'sample (assumed)':
        other = {**published, 'convention': 'population (np.std)'}
        reading = sota_p0.welch_summary(ours, {'mean': published['mean'],
                                               'sd': sota_p0.sample_sd(other),
                                               'n': published['n']})
        record['population_reading'] = {'sd_sample': round(sota_p0.sample_sd(other), 3),
                                        'welch': reading, 'passed': reading['ci95'][0] > 0}
    return record


# ---------------------------------------------------------------------------------------------
# Per-sentence counts, votes and the paired bootstrap over test sentences
# ---------------------------------------------------------------------------------------------
def tagger_counts(gold_tags, pred_tags):
    """[(TP, FP, FN)] per sentence, entities read as seqeval's default mode does."""
    import ner_ensemble as ne
    out = []
    for gold, pred in zip(gold_tags, pred_tags):
        g = set(ne.tags_to_spans(gold)[0])
        p = set(ne.tags_to_spans(pred)[0])
        out.append((len(g & p), len(p - g), len(g - p)))
    return out


def micro_f1(counts):
    tp = sum(c[0] for c in counts)
    fp = sum(c[1] for c in counts)
    fn = sum(c[2] for c in counts)
    return 100 * 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def run_counts(run, keep=None):
    """A tagger run's per-sentence counts (optionally only the sentences in `keep`), checked
    against its stored F1."""
    gold = [g.split() for g in run['test_gold']]
    pred = [p.split() for p in run['test_pred']]
    counts = tagger_counts(gold, pred)
    if keep is None and round(micro_f1(counts), 2) != run['f1']:
        raise ValueError(f"{run['run_id']}: counts give {micro_f1(counts):.2f}, stored "
                         f"{run['f1']}")
    return counts if keep is None else [counts[k] for k in keep]


def mulms_gold():
    """The gold entity sets of MuLMS's test sentences, in the order of every run's test_pred."""
    import mulms_nested as mn
    sentences = mn.read_release(offline=True)
    parts = mn.configuration(sentences, 1)
    return [s['gold'] for s in parts['test']]


def mulms_counts(run, gold):
    pred = [{tuple(e) for e in p} for p in run['test_pred']]
    if len(pred) != len(gold):
        raise ValueError(f'{len(pred)} predictions for {len(gold)} sentences')
    counts = [(len(p & g), len(p - g), len(g - p)) for p, g in zip(pred, gold)]
    if round(micro_f1(counts), 2) != run['test']['micro_f1']:
        raise ValueError(f"{run.get('run_id')}: counts give {micro_f1(counts):.2f}, stored "
                         f"{run['test']['micro_f1']}")
    return counts


def vote_runs(runs, size):
    """The disjoint votes of `size` consecutive seeds: [(seeds, gold tags, vote tags)]."""
    import ner_ensemble as ne
    seeds = sorted(runs)
    out = []
    for i in range(0, len(seeds) - size + 1, size):
        group = seeds[i:i + size]
        members = [[p.split() for p in runs[s]['test_pred']] for s in group]
        gold = [g.split() for g in runs[group[0]]['test_gold']]
        for s in group[1:]:
            if runs[s]['test_gold'] != runs[group[0]]['test_gold']:
                raise ValueError(f'seed {s}: other test sentences')
        votes = [ne.vote_tags([m[k] for m in members])[0] for k in range(len(gold))]
        out.append((group, gold, votes))
    return out


def bootstrap(counts_a, counts_b, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """The paired bootstrap over test sentences: the 95% percentile interval of the difference
    between the systems' mean micro F1 over their runs. counts_*: per run, the per-sentence
    (TP, FP, FN); the same sentences in every run."""
    import numpy as np
    a = np.asarray(counts_a, dtype=float)      # (runs, sentences, 3)
    b = np.asarray(counts_b, dtype=float)
    n = a.shape[1]
    if b.shape[1] != n:
        raise ValueError('the systems are scored on different sentences')
    rng = random.Random(seed)

    def mean_f1(x, w):
        tp, fp, fn = np.tensordot(x, w, axes=([1], [0])).T
        denom = 2 * tp + fp + fn
        return float(np.mean(np.where(denom > 0, 200 * tp / np.maximum(denom, 1), 0.0)))
    diffs = []
    for _ in range(n_boot):
        w = np.bincount([rng.randrange(n) for _ in range(n)], minlength=n).astype(float)
        diffs.append(mean_f1(a, w) - mean_f1(b, w))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


# ---------------------------------------------------------------------------------------------
# The decisions
# ---------------------------------------------------------------------------------------------
def passed(record):
    return record['ci95'][0] > 0


ON = ', on this test set'
WORDINGS = {   # (passed, not passed) per kind (ROADMAP §4, P2's wordings)
    'mulms_dt': ('above the published result under its protocol' + ON,
                 'not shown above the published result under its protocol' + ON),
    'dp': ('the change improves on the tuned reference' + ON,
           'the change: not shown to improve on the tuned reference' + ON),
    'dp_tuning': ('tuning improves on our re-run of the published system' + ON,
                  'tuning: not shown to improve on our re-run of the published system' + ON),
    'mm_live': ('the selected system improves on the strongest known system' + ON,
                'the selected system: not shown to improve on the strongest known system' + ON),
    're-test': ('replicated', 'not replicated'),
}


def wording(kind, ok, extra=None):
    """The protocol's fixed wordings."""
    extra = extra or {}
    if kind == 'official_dt':
        if ok and extra.get('gate_passed'):
            return ("above every reported result on SOFC-Exp's official split, on this test set "
                    '(the 86.0 taken as fixed)')
        if ok:
            return ('above the best result with a stated protocol; not shown above the best '
                    'reported result, whose protocol we could not verify' + ON)
        return 'not shown above the best result with a stated protocol' + ON
    if kind == 'cv_dt':
        if ok and extra.get('gate_passed'):
            return ('above the best CV result with a stated protocol, and above the 85.71 '
                    'reported without a stated split, on this test set (the 85.71 taken as '
                    'fixed)')
        if ok:
            return ('above the best CV result with a stated protocol, on this test set; not '
                    'shown above the 85.71 reported without a stated split')
        return 'not shown above the best CV result with a stated protocol' + ON
    if kind == 'gated':
        if not extra.get('gate_ok'):
            return 'not taken: its gate did not pass'
        return (f"above {extra['target']}, taken as fixed (anti-conservative){ON}" if ok else
                f"not shown above {extra['target']}, taken as fixed{ON}")
    if kind == 'vote':
        k = extra['k']
        return (f"S's vote of {k} improves on the CRF head's vote of {k}{ON}" if ok else
                f"S's vote of {k}: not shown to improve on the CRF head's vote of {k}{ON}")
    return WORDINGS[kind][0 if ok else 1]


def decide(dirs=(P2_DIR,), with_bootstrap=True):
    """The record of results/gpu/sota_p2.json: the 15 live decisions (2 gated) and 4 re-tests
    with their estimates, intervals, verdicts and wordings; the reported comparisons; the
    count; per benchmark, S's mean with its seed interval (the "nothing clears" wording)."""
    live, retests, reported, best = [], [], {}, {}

    def boot(make):
        """The bootstrap of the (counts S, counts R) that `make` returns, when it is on."""
        return bootstrap(*make()) if with_bootstrap else None

    def add(rows, number, name, kind, record, extra=None, unit=None, test_set=None,
            boot_ci=None):
        record = {**record}
        record.setdefault('passed', passed(record))
        if kind == 'gated' and not (extra or {}).get('gate_ok'):
            record['passed'] = False
        row = {'number': number, 'decision': name, 'unit': unit, 'test_set': test_set, **record,
               'bootstrap_ci95': boot_ci,
               'wording': wording(kind, record['passed'], extra)}
        rows.append(row)
        return row

    def mean_ci(values):
        mean, ci = t_interval(values)
        return {'mean': round(mean, 2), 'ci95': [round(ci[0], 2), round(ci[1], 2)],
                'n': len(values)}

    # SOFC-Exp, official split: decisions 1-3
    s, r = official_values('b1', dirs), official_values('a6', dirs)
    eleven = [s[x]['f1_11'] for x in SEEDS]
    dt = against_published([s[x]['f1_10'] for x in SEEDS], 'sofc_official_10')
    gate = one_sample(eleven, 86.0)
    gate_ok = dt['passed']
    add(live, 1, 'SOFC-Exp official, D-T: B1 against 83.4 (10 articles)', 'official_dt', dt,
        {'gate_passed': gate_ok and passed(gate)}, 'seed', '10 test articles')
    dp = welch(eleven, [r[x]['f1_11'] for x in SEEDS])
    add(live, 2, 'SOFC-Exp official, D-P: B1 against A6', 'dp', dp, {}, 'seed (Welch)',
        '11 test articles',
        boot(lambda: ([run_counts(s[x]['run']) for x in SEEDS],
                      [run_counts(r[x]['run']) for x in SEEDS])))
    add(live, 3, 'SOFC-Exp official: B1 against 86.0 (11 articles; gated on 1)', 'gated', gate,
        {'gate_ok': gate_ok, 'target': 86.0}, 'seed (target fixed)', '11 test articles')
    best['SOFC-Exp official (B1, 11 articles)'] = mean_ci(eleven)
    reported['official_votes'] = {'note': "no decision; shown beside the CRF head's known 85.13"}
    official_runs = {x: s[x]['run'] for x in SEEDS}
    for k in VOTE_SIZES:
        votes = vote_runs(official_runs, k)
        reported['official_votes'][f'vote_of_{k}'] = [
            round(micro_f1(tagger_counts(g, v)), 2) for _, g, v in votes]

    # SOFC-Exp, CV: decisions 4-6
    cv = json.loads(CV_SYSTEM_FILE.read_text())
    system, ref_cell = cv['cv_system'], cv['reference']
    cs, cr = cv_values(system, dirs), cv_values(ref_cell, dirs)
    units = sorted(cs)
    values = [cs[u]['f1'] for u in units]
    dt = against_published(values, 'sofc_cv')
    gate = one_sample(values, 85.71)
    gate_ok = dt['passed']
    add(live, 4, f'SOFC-Exp CV, D-T: {system} against 84.77', 'cv_dt', dt,
        {'gate_passed': gate_ok and passed(gate)}, 'run (fold x seed)', '11 test articles')
    dp = paired({u: cs[u]['f1'] for u in units}, {u: cr[u]['f1'] for u in units})
    add(live, 5, f'SOFC-Exp CV, D-P: {system} against {ref_cell}',
        'dp_tuning' if system == 'a6' else 'dp', dp, {}, 'fold x seed (paired)',
        '11 test articles',
        boot(lambda: ([run_counts(cs[u]) for u in units], [run_counts(cr[u]) for u in units])))
    add(live, 6, f'SOFC-Exp CV: {system} against 85.71 (gated on 4)', 'gated', gate,
        {'gate_ok': gate_ok, 'target': 85.71}, 'run (target fixed)', '11 test articles')
    reported['cv_fold_means'] = {fold: round(statistics.fmean(cs[(fold, o)]['f1']
                                                             for o in CV_OFFSETS), 2)
                                 for fold in CV_FOLDS}
    best[f'SOFC-Exp CV ({system})'] = mean_ci(values)

    # MuLMS: decisions 7-8
    ms, mr = mulms_values('a1m', dirs), mulms_values('a0', dirs)
    values = [ms[u]['test']['micro_f1'] for u in MULMS_UNITS]
    dt = against_published(values, 'mulms_nested')
    add(live, 7, 'MuLMS, D-T: A1m against 79.6', 'mulms_dt', dt, {}, 'configuration x seed',
        '7 test documents')
    dp = paired({u: ms[u]['test']['micro_f1'] for u in MULMS_UNITS},
                {u: mr[u]['test']['micro_f1'] for u in MULMS_UNITS})
    gold = mulms_gold() if with_bootstrap else None
    add(live, 8, 'MuLMS, D-P: A1m against A0', 'dp_tuning', dp, {},
        'configuration x seed (paired)', '7 test documents',
        boot(lambda: ([mulms_counts(ms[u], gold) for u in MULMS_UNITS],
                      [mulms_counts(mr[u], gold) for u in MULMS_UNITS])))
    best['MuLMS (A1m)'] = mean_ci(values)

    # MaterioMiner: decisions 9-15 and the 4 re-tests
    numbers = {'fine': {'live': 9, 'dp': 10, 'votes': (12, 13)},
               'coarse': {'live': 11, 'votes': (14, 15)}}
    for level in LEVELS:
        cell = MM_SELECTED[level]
        sel, a0, plain = (released_values(level, c, dirs) for c in (cell, 'a0', 'plain'))
        first = {x: sel[x] for x in SEEDS}
        f1 = {x: first[x]['f1'] for x in SEEDS}
        add(live, numbers[level]['live'], f'MaterioMiner {level}: {cell} against the CRF head',
            'mm_live', paired(f1, {x: a0[x]['f1'] for x in SEEDS}), {}, 'seed (paired)',
            '96 released test sentences',
            boot(lambda: ([run_counts(first[x]) for x in SEEDS],
                          [run_counts(a0[x]) for x in SEEDS])))
        if level == 'fine':
            a6 = released_values(level, 'a6', dirs)
            add(live, numbers[level]['dp'], 'MaterioMiner fine, D-P: c1 against A6', 'dp',
                paired(f1, {x: a6[x]['f1'] for x in SEEDS}), {}, 'seed (paired)',
                '96 released test sentences',
                boot(lambda: ([run_counts(first[x]) for x in SEEDS],
                              [run_counts(a6[x]) for x in SEEDS])))
        for number, k in zip(numbers[level]['votes'], VOTE_SIZES):
            mine, theirs = vote_runs(sel, k), vote_runs(a0, k)
            record = welch([micro_f1(tagger_counts(g, v)) for _, g, v in mine],
                           [micro_f1(tagger_counts(g, v)) for _, g, v in theirs])
            add(live, number, f"MaterioMiner {level}: {cell}'s vote of {k} against the CRF "
                f"head's", 'vote', record, {'k': k}, 'vote (Welch)', '96 released test sentences',
                boot(lambda mine=mine, theirs=theirs: (
                    [tagger_counts(g, v) for _, g, v in mine],
                    [tagger_counts(g, v) for _, g, v in theirs])))
        target = MM_TARGETS[level]
        add(retests, None, f'MaterioMiner {level}, D-T (re-test): {cell} against {target}',
            're-test', one_sample(list(f1.values()), target), {}, 'seed (target fixed)',
            '96 released test sentences')
        add(retests, None, f'MaterioMiner {level}, D-P (re-test): {cell} against plain '
            'MatSciBERT', 're-test', paired(f1, {x: plain[x]['f1'] for x in SEEDS}), {},
            'seed (paired)', '96 released test sentences')
        best[f'MaterioMiner {level} ({cell})'] = mean_ci(list(f1.values()))
    live.sort(key=lambda row: row['number'])
    n_pass = sum(r['passed'] for r in live)
    return {'note': ("P2's decisions (ROADMAP §4, the P2 protocol): each final system's test set "
                     'read once, on fresh seeds 1000 up; a decision passes when its 95% interval '
                     'lies above 0; every interval is over seeds and does not cover sampling of '
                     'the test set'),
            'live': live, 're_tests': retests, 'reported': reported,
            'selected_means': best,
            'count': {'live': len(live), 'passed': n_pass,
                      'expected_false_passes_at_most': round(0.025 * len(live), 3),
                      're_tests': len(retests),
                      're_tests_passed': sum(r['passed'] for r in retests)},
            'cv_system': cv}


def main(argv=None):
    parser = argparse.ArgumentParser(description="The SOTA plan's P2: the decisions")
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('jobs')
    p.add_argument('batch', choices=('p2a', 'p2m', 'p2b'))
    p = sub.add_parser('cv-system')
    p.add_argument('pulled', help="p2a's pulled output: A6's screen file is taken from it")
    p = sub.add_parser('checks')
    p.add_argument('pulled')
    p.add_argument('--batch', choices=('p2a', 'p2m', 'p2b'), required=True)
    p = sub.add_parser('decide')
    p.add_argument('--results', nargs='+', default=[str(P2_DIR)])
    p.add_argument('--out', default=str(OUT))
    args = parser.parse_args(argv)
    if args.command == 'jobs':
        entries = write_jobs(args.batch)
        print(f'{len(entries)} jobs, about {sum(e["minutes"] for e in entries) / 120:.1f} '
              f'session-hours on two lanes, to {jobs_file(args.batch)}')
    elif args.command == 'cv-system':
        take_a6_screen(args.pulled)
        record = cv_system_record()
        CV_SYSTEM_FILE.write_text(json.dumps(record, indent=1) + '\n')
        print(f"CV system {record['cv_system']} (means {record['means_over_9_units']}); "
              f"reference {record['reference']}; wrote {CV_SYSTEM_FILE}")
    elif args.command == 'checks':
        result = batch_checks(args.pulled, args.batch)
        for check, problems in result.items():
            print(f"{check}: {'passed' if not problems else 'FAILED: ' + '; '.join(problems)}")
        if any(result.values()):
            raise SystemExit('a check failed: the batch is void; move its files, unread, to '
                             'results/gpu_p2_void/ (ROADMAP §4, P2)')
    else:
        record = decide(args.results)
        Path(args.out).write_text(json.dumps(record, indent=1) + '\n')
        for row in record['live'] + record['re_tests']:
            print(f"{row['number'] or 're-test'}. {row['decision']}: {row['estimate']:+.2f} "
                  f"{row['ci95']} -> {row['wording']}")
        print(f"{record['count']['passed']} of {record['count']['live']} live decisions pass "
              f"(at most about {record['count']['expected_false_passes_at_most']} by chance); "
              f"wrote {args.out}")


if __name__ == '__main__':
    main()
