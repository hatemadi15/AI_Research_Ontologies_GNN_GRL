"""
sota_p2.py - the SOTA plan's P2: the decisions (docs/ROADMAP.md §4, the P2 protocol;
docs/SOTA_PLAN.md §1, §2, §4 and §5)

P2 reads each final system's test set once, on fresh seeds (1000 up). It makes 12 live
decisions and 4 re-tests (MaterioMiner's, whose outcome is known). No P2 result changes a
system, a setting, a rule or a wording. This script:
  - writes the batches' jobs (p2a: the checks, A6's extra units on SOFC-Exp, SOFC-Exp's
    official split and MaterioMiner; p2m: MuLMS; p2b: SOFC-Exp's cross-validation, whose
    system P1's tie rule picks without B1 once A6's extra units are in);
  - checks a pulled batch (smoke4's repeat; A0 at seed 0 against the stored runs);
  - checks and aggregates the files (results/gpu_p2), forms MaterioMiner's votes, computes
    the decisions, the paired bootstrap over test sentences and the wordings, and writes
    results/gpu/sota_p2.json.

Usage:
  python src/graph/sota_p2.py jobs p2a|p2m|p2b     # results/sota/p2_jobs_<batch>.json
  python src/graph/sota_p2.py cv-system            # SOFC-Exp's CV system (after p2a)
  python src/graph/sota_p2.py checks <pulled>       # a pulled batch's checks
  python src/graph/sota_p2.py decide               # results/gpu/sota_p2.json
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
    ('coarse', 'plain'): ('--granularity', 'coarse', '--published-split'),
}
MM_SELECTED = {'fine': 'c1', 'coarse': 'a3a6'}
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

# minutes per run, measured in P1 (screen mode; 15% is added for the test sets), for the lanes
MINUTES = {'mm_c1': 1.69, 'mm_a3a6': 2.01, 'mm_a0': 1.33, 'mm_plain': 1.1, 'b1': 21.15,
           'a6': 3.5, 'a0_sofc': 3.3, 'mulms_a1m': 31.52, 'mulms_a0': 26.67, 'cv_a6c1': 5.91,
           'cv_c1': 5.16, 'cv_a6': 3.5, 'cv_a0': 3.3, 'smoke': 1.2}
TEST_OVERHEAD = 1.15


def seeds_args(seeds):
    return ['--seeds', *(str(s) for s in seeds)]


def mm_job(level, cell):
    seeds = VOTE_SEEDS if MM_SELECTED[level] == cell else SEEDS
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
            for cell in (MM_SELECTED[level], 'a0', 'plain'):
                key = 'mm_' + cell
                n = len(VOTE_SEEDS) if cell == MM_SELECTED[level] else len(SEEDS)
                jobs.append((n * MINUTES[key] * TEST_OVERHEAD, mm_job(level, cell)))
        return jobs
    if batch == 'p2m':
        return [(0.0, SMOKE_JOB)] + [(MINUTES['mulms_' + cell] * TEST_OVERHEAD, mulms_job(cell, c, s))
                                     for cell in ('a1m', 'a0') for c, s in MULMS_UNITS]
    if batch == 'p2b':
        if cv_system is None:
            raise SystemExit('p2b needs the CV system (sota_p2.py cv-system, after p2a)')
        return [(0.0, SMOKE_JOB)] + [(5 * MINUTES['cv_' + cell] * TEST_OVERHEAD, cv_job(cell, off))
                                     for cell in (cv_system, cv_reference(cv_system))
                                     for off in CV_OFFSETS]
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
def check_repeat(found, granularity, results_dir=GPU_DIR):
    """Problems of A0's seed-0 repeat (tag p2chk) against the stored run: F1, history, test
    sentences, gold and predictions must be equal."""
    name = f'ner_results_{granularity}_published_split_crf_p2chk.json'
    if name not in found:
        return [f'{name} is missing']
    new = {r['run_id']: r for r in json.loads(found[name].read_text())['runs']}
    old = {r['run_id']: r for r in json.loads(
        (Path(results_dir) / STORED_A0[granularity]).read_text())['runs']}
    a, b = new.get('seed0'), old.get('seed0')
    if a is None or b is None:
        return ['seed0 is missing']
    return [f'{k} differs' for k in ('f1', 'history', 'test_index', 'test_gold', 'test_pred')
            if a.get(k) != b.get(k)]


def batch_checks(pulled, batch=None):
    """{check: problems} of a pulled P2 batch ([] when it passes): smoke4's repeat in every
    batch; A0's seed-0 repeats in p2a."""
    found = {p.name: p for p in Path(pulled).rglob('ner_results_*.json')}
    out = {}
    smoke = found.get('ner_results_fine_random_split_vary_crf_smokebu.json')
    if smoke is None:
        out['smoke'] = ['smokebu is missing']
    else:
        a = sota_p1.without(json.loads(smoke.read_text()), 'smokebu')
        b = sota_p1.without(json.loads(sota_p1.SMOKE_REFERENCE.read_text()), 'smokebn')
        out['smoke'] = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if any('p2chk' in n for n in found) or batch == 'p2a':
        for granularity in ('fine', 'sofc'):
            out[f'a0_repeat_{granularity}'] = check_repeat(found, granularity)
    return out


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
    seeds = VOTE_SEEDS if MM_SELECTED[level] == cell else SEEDS
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
    """{(configuration, seed): run} of a MuLMS system, every file checked."""
    out = {}
    for c, s in MULMS_UNITS:
        job = mulms_job(cell, c, s)
        name = mulms_file(job)
        data = load_file(name, dirs)
        runs = check_p2_file(data, job, name, [f'c{c}'])
        run = runs[f'c{c}']
        if run.get('seed') != s or 'test' not in run:
            raise ValueError(f'{name}: seed {run.get("seed")} or no test score')
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
    """mean(a) minus mean(b) with Welch's 95% interval, from two samples."""
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

    def at(printed):
        test = sota_p0.welch_summary(ours, {'mean': published['mean'],
                                            'sd': sota_p0.sample_sd(published, printed),
                                            'n': published['n']})
        return test, test['ci95'][0] > 0
    test, passed = at(published['sd_printed'])
    half = 0.5 * 10 ** -published['decimals']
    edges = {at(published['sd_printed'] + d)[1] for d in (-half, half)}
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


def wording(kind, record, benchmark=None, extra=None):
    """The protocol's fixed wordings (ROADMAP §4, P2)."""
    extra = extra or {}
    ok = record['passed']
    if kind == 'official_dt':
        if extra.get('above_86'):
            return 'above every reported result on SOFC-Exp\'s official split, on this test set'
        if ok:
            return ('above the best result with a stated protocol; not shown above the best '
                    'reported result, whose protocol we could not verify (on this test set)')
        return 'not shown above the best result with a stated protocol (on this test set)'
    if kind == 'cv_dt':
        if ok and extra.get('above_85_71'):
            return 'above every reported cross-validation result, on this test set'
        if ok:
            return ('above the best cross-validation result with a stated protocol; not shown '
                    'above 85.71, whose protocol we could not verify (on this test set)')
        return ('not shown above the best cross-validation result with a stated protocol (on '
                'this test set)')
    if kind == 'mulms_dt':
        return ('above the published result under its protocol (on this test set)' if ok else
                'not shown above the published result under its protocol (on this test set)')
    if kind == 'dp':
        tuning = extra.get('tuning')
        if tuning:
            return 'tuning improves on the published recipe' if ok else \
                'tuning: not shown to improve on the published recipe'
        return 'the change improves on the tuned reference' if ok else \
            'the change: not shown to improve on the tuned reference'
    if kind == 'mm_live':
        base = 'the selected system improves on the strongest known system'
        text = base if ok else 'the selected system: not shown to improve on the strongest known ' \
                               'system'
        return text + (' (tuning)' if extra.get('tuning') else '')
    if kind == 'vote':
        k = extra['k']
        return (f'the vote of {k} improves on the strongest known single system' if ok else
                f'the vote of {k}: not shown to improve on the strongest known single system')
    if kind == 're-test':
        return 'replicated' if ok else 'not replicated'
    raise ValueError(kind)


def decide(dirs=(P2_DIR,), with_bootstrap=True):
    """The record of results/gpu/sota_p2.json: every decision with its estimate, interval,
    verdict and wording; the reported comparisons; the count."""
    live, retests, reported = [], [], {}
    def boot(make):
        """The bootstrap of the (counts S, counts R) that `make` returns, when it is on."""
        return bootstrap(*make()) if with_bootstrap else None

    def add(rows, name, kind, record, extra=None, unit=None, test_set=None, boot_ci=None):
        record = {**record}
        record.setdefault('passed', passed(record))
        rows.append({'decision': name, 'unit': unit, 'test_set': test_set, **record,
                     'bootstrap_ci95': boot_ci, 'wording': wording(kind, record, extra=extra)})

    # SOFC-Exp, official split
    s, r = official_values('b1', dirs), official_values('a6', dirs)
    dt = against_published([s[x]['f1_10'] for x in SEEDS], 'sofc_official_10')
    above = one_sample([s[x]['f1_11'] for x in SEEDS], 86.0)
    reported['official_vs_86_on_11'] = above
    add(live, 'SOFC-Exp official, D-T: B1 against 83.4 (10 articles)', 'official_dt', dt,
        {'above_86': above['ci95'][0] > 0}, 'seed', '10 test articles')
    dp = welch([s[x]['f1_11'] for x in SEEDS], [r[x]['f1_11'] for x in SEEDS])
    add(live, 'SOFC-Exp official, D-P: B1 against A6', 'dp', dp, {}, 'seed (Welch)',
        '11 test articles',
        boot(lambda: ([run_counts(s[x]['run']) for x in SEEDS],
                      [run_counts(r[x]['run']) for x in SEEDS])))
    reported['official_votes'] = {}
    official_runs = {x: s[x]['run'] for x in SEEDS}
    for k in VOTE_SIZES:
        votes = vote_runs(official_runs, k)
        reported['official_votes'][f'vote_of_{k}'] = [
            round(micro_f1(tagger_counts(g, v)), 2) for _, g, v in votes]

    # SOFC-Exp, CV
    cv = json.loads(CV_SYSTEM_FILE.read_text())
    system, ref_cell = cv['cv_system'], cv['reference']
    cs, cr = cv_values(system, dirs), cv_values(ref_cell, dirs)
    dt = against_published([cs[u]['f1'] for u in sorted(cs)], 'sofc_cv')
    above = one_sample([cs[u]['f1'] for u in sorted(cs)], 85.71)
    reported['cv_vs_85_71'] = above
    add(live, f'SOFC-Exp CV, D-T: {system} against 84.77', 'cv_dt', dt,
        {'above_85_71': above['ci95'][0] > 0}, 'run (fold x seed)', '11 test articles')
    dp = paired({u: cs[u]['f1'] for u in cs}, {u: cr[u]['f1'] for u in cr})
    add(live, f'SOFC-Exp CV, D-P: {system} against {ref_cell}', 'dp', dp,
        {'tuning': system == 'a6'}, 'fold x seed (paired)', '11 test articles',
        boot(lambda: ([run_counts(cs[u]) for u in sorted(cs)],
                      [run_counts(cr[u]) for u in sorted(cr)])))

    # MuLMS
    ms, mr = mulms_values('a1m', dirs), mulms_values('a0', dirs)
    dt = against_published([ms[u]['test']['micro_f1'] for u in MULMS_UNITS], 'mulms_nested')
    add(live, 'MuLMS, D-T: A1m against 79.6', 'mulms_dt', dt, {}, 'configuration x seed',
        '7 test documents')
    dp = paired({u: ms[u]['test']['micro_f1'] for u in MULMS_UNITS},
                {u: mr[u]['test']['micro_f1'] for u in MULMS_UNITS})
    gold = mulms_gold() if with_bootstrap else None
    add(live, 'MuLMS, D-P: A1m against A0', 'dp', dp, {'tuning': True},
        'configuration x seed (paired)', '7 test documents',
        boot(lambda: ([mulms_counts(ms[u], gold) for u in MULMS_UNITS],
                      [mulms_counts(mr[u], gold) for u in MULMS_UNITS])))

    # MaterioMiner, per level
    for level in LEVELS:
        cell = MM_SELECTED[level]
        sel, a0, plain = (released_values(level, c, dirs) for c in (cell, 'a0', 'plain'))
        first = {x: sel[x] for x in SEEDS}
        live_dp = paired({x: first[x]['f1'] for x in SEEDS}, {x: a0[x]['f1'] for x in SEEDS})
        add(live, f'MaterioMiner {level}: {cell} against the CRF head', 'mm_live', live_dp,
            {'tuning': cell != 'c1'}, 'seed (paired)', '96 released test sentences',
            boot(lambda: ([run_counts(first[x]) for x in SEEDS],
                          [run_counts(a0[x]) for x in SEEDS])))
        a0_f1 = [a0[x]['f1'] for x in SEEDS]
        for k in VOTE_SIZES:
            votes = vote_runs(sel, k)
            vote_f1 = [micro_f1(tagger_counts(g, v)) for _, g, v in votes]
            record = welch(vote_f1, a0_f1)
            add(live, f'MaterioMiner {level}: the vote of {k} of {cell} against the CRF head',
                'vote', record, {'k': k}, 'vote against run (Welch)',
                '96 released test sentences',
                boot(lambda votes=votes: ([tagger_counts(g, v) for _, g, v in votes],
                                          [run_counts(a0[x]) for x in SEEDS])))
        target = MM_TARGETS[level]
        add(retests, f'MaterioMiner {level}, D-T (re-test): {cell} against {target}', 're-test',
            one_sample([first[x]['f1'] for x in SEEDS], target), {}, 'seed (target fixed)',
            '96 released test sentences')
        add(retests, f'MaterioMiner {level}, D-P (re-test): {cell} against plain MatSciBERT',
            're-test', paired({x: first[x]['f1'] for x in SEEDS},
                              {x: plain[x]['f1'] for x in SEEDS}), {}, 'seed (paired)',
            '96 released test sentences')
    n_pass = sum(r['passed'] for r in live)
    return {'note': ("P2's decisions (ROADMAP §4, the P2 protocol): each final system's test set "
                     'read once, on fresh seeds 1000 up; a decision passes when its 95% interval '
                     'lies above 0'),
            'live': live, 're_tests': retests, 'reported': reported,
            'count': {'live': len(live), 'passed': n_pass,
                      'expected_false_passes': round(0.025 * len(live), 2),
                      're_tests': len(retests),
                      're_tests_passed': sum(r['passed'] for r in retests)},
            'cv_system': cv}


def main(argv=None):
    parser = argparse.ArgumentParser(description="The SOTA plan's P2: the decisions")
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('jobs')
    p.add_argument('batch', choices=('p2a', 'p2m', 'p2b'))
    sub.add_parser('cv-system')
    p = sub.add_parser('checks')
    p.add_argument('pulled')
    p.add_argument('--batch', choices=('p2a', 'p2m', 'p2b'), default=None)
    p = sub.add_parser('decide')
    p.add_argument('--results', nargs='+', default=[str(P2_DIR)])
    p.add_argument('--out', default=str(OUT))
    args = parser.parse_args(argv)
    if args.command == 'jobs':
        entries = write_jobs(args.batch)
        print(f'{len(entries)} jobs, about {sum(e["minutes"] for e in entries) / 120:.1f} '
              f'session-hours on two lanes, to {jobs_file(args.batch)}')
    elif args.command == 'cv-system':
        record = cv_system_record()
        CV_SYSTEM_FILE.write_text(json.dumps(record, indent=1) + '\n')
        print(f"CV system {record['cv_system']} (means {record['means_over_9_units']}); "
              f"reference {record['reference']}; wrote {CV_SYSTEM_FILE}")
    elif args.command == 'checks':
        result = batch_checks(args.pulled, args.batch)
        for check, problems in result.items():
            print(f"{check}: {'passed' if not problems else 'FAILED: ' + '; '.join(problems)}")
        if any(result.values()):
            raise SystemExit('a check failed: the batch counts for nothing (ROADMAP §4, P2)')
    else:
        record = decide(args.results)
        Path(args.out).write_text(json.dumps(record, indent=1) + '\n')
        for row in record['live'] + record['re_tests']:
            print(f"{row['decision']}: {row['estimate']:+.2f} {row['ci95']} -> {row['wording']}")
        print(f"{record['count']['passed']} of {record['count']['live']} live decisions pass "
              f"(about {record['count']['expected_false_passes']} expected by chance); wrote "
              f"{args.out}")


if __name__ == '__main__':
    main()
