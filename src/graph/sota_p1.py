"""
sota_p1.py - the SOTA plan's P1, the screens (the protocol in docs/ROADMAP.md §4)

No training. P1 selects, on validation data only, one system per benchmark for P2, and the
tuned reference. This module holds what the protocol fixes as code:
  - the data files, each with the SHA-256 of its canonical JSON: MaterioMiner's screen splits
    (results/sota/mm_screen_splits.json) and the documents that MuLMS and SOFC-Exp share
    (results/sota/shared_documents.json, by the 8-gram rule);
  - the cells: round 1's changes from the reference A0 (ROUND1, CHANGES), each cell's jobs
    and results files, and its units (UNITS; EXTRA_UNITS for the tie rule);
  - the checks of every file it reads: a screen file, no test field (TEST_FIELDS) outside the
    P0 files named by SHA-256 (P0_MULMS), the settings of its jobs, its units and seeds;
  - the analysis (`analyse`): per cell the mean, the mean of the last 3 validation scores, the
    gain over A0 paired by unit, its SD and standard error; G (the cells whose gain exceeds its
    standard error), A*, A's best cell, the chain, the next cells to run, and once nothing is
    left to run the selection with the tie rule;
  - the timing table (`timing`) from batch p1t's runs and the budget's cuts.

Usage:
  python src/graph/sota_p1.py data-files       # write the two data files
  python src/graph/sota_p1.py status [--results DIR ...]   # tables and the next cells
  python src/graph/sota_p1.py next-jobs [--out FILE]       # the next cells' jobs (batch p1n)
  python src/graph/sota_p1.py select [--out results/gpu/sota_p1.json]
  python src/graph/sota_p1.py timing <pulled dir> [--out results/sota/p1_timing.json]
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

SOTA_DIR = ROOT / 'results' / 'sota'
GPU_DIR = ROOT / 'results' / 'gpu'
SCREEN_DIR = ROOT / 'results' / 'gpu_screen'        # P1's screen files
SPLITS_FILE = SOTA_DIR / 'mm_screen_splits.json'
SHARED_FILE = SOTA_DIR / 'shared_documents.json'
TIMING_FILE = SOTA_DIR / 'p1_timing.json'
OUT = GPU_DIR / 'sota_p1.json'
MULMS_NESTED_SCRIPT = 'src/graph/mulms_nested.py'

# ---------------------------------------------------------------------------------------------
# The data files
# ---------------------------------------------------------------------------------------------
LEVELS = ('fine', 'coarse')
N_SCREEN_SPLITS = 14            # units 0-9 and the extra units 10-13
SHARED_MIN_SENTENCES = 10       # the 8-gram rule: at least 10 sentences of the document


def canonical_sha256(obj):
    """The SHA-256 of an object's canonical JSON: sorted keys, no spaces."""
    text = json.dumps(obj, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def mm_screen_splits_record():
    """{level: {seed<k>: {train, val}}}: each split's sentences as positions in the level's
    pool, the release's training file (0-322) then its validation file (323-379), in the order
    the run trains on them (ner_baseline.screen_split_parts), with the release files' SHA-256."""
    import ner_baseline as nb
    splits, sizes = {}, {}
    for level in LEVELS:
        data = nb.load_split_corpus(level)
        parts, _ = nb.published_split(data, level)
        n_train, n_val = len(parts[0]), len(parts[1])
        sizes[level] = {'train': n_train, 'val': n_val, 'test': len(parts[2])}
        pool = [list(range(n_train)), list(range(n_train, n_train + n_val))]
        splits[level] = {}
        for k in range(N_SCREEN_SPLITS):
            train, val = nb.screen_split_parts(pool, k)
            splits[level][f'seed{k}'] = {'train': train, 'val': val}
    return {'rule': ("the level's released training then validation sentences (pool positions "
                     '0-322 and 323-379) shuffled by random.Random(k); the first 57 are '
                     'validation, the other 323 training, in that order; no test sentence'),
            'release_sha256': {f'{level}_{part}': nb.PUBLISHED_SPLIT_SHA256[level, part]
                               for level in LEVELS for part in nb.PUBLISHED_PARTS},
            'sizes': sizes, 'splits': splits, 'sha256': canonical_sha256(splits)}


def shared_documents_record():
    """The MuLMS documents and SOFC-Exp articles that the 8-gram rule pairs: a pair is shared
    when at least SHARED_MIN_SENTENCES of the document's sentences share a word 8-gram with the
    article (dapt_mlm's normalisation), with where each lies in both corpora, and the largest
    count of any other pair."""
    from collections import defaultdict
    import dapt_mlm as dm
    import mulms_nested as mn
    import ner_baseline as nb
    context = json.loads((Path(nb.RAW_DIR) / 'sofc_exp' / 'context.json').read_text())
    index = defaultdict(set)
    for pmc, rec in context.items():
        for s in rec['sentences']:
            for gram in dm.ngrams(dm.normalise(' '.join(s)), 8):
                index[gram].add(pmc)
    sentences = mn.read_release(offline=True)
    counts = defaultdict(lambda: defaultdict(int))
    for s in sentences:
        hits = set()
        for gram in dm.ngrams(dm.normalise(' '.join(s['tokens'])), 8):
            hits |= index.get(gram, set())
        for pmc in hits:
            counts[s['doc_id']][pmc] += 1
    split = nb.sofc_split()
    folds = json.loads((SOTA_DIR / 'sofc_cv_folds.json').read_text())['folds']
    configs = json.loads((SOTA_DIR / 'mulms_configurations.json').read_text())['configurations']
    pairs, other = [], 0
    for doc in sorted(counts):
        for pmc, n in sorted(counts[doc].items()):
            if n < SHARED_MIN_SENTENCES:
                other = max(other, n)
                continue
            pairs.append({
                'mulms': doc, 'sofc_exp': pmc, 'sentences': n,
                'sofc_exp_part': split[pmc],
                'sofc_exp_cv_dev': [k for k, f in folds.items() if pmc in f['dev']],
                'mulms_parts': {c: [p for p in ('train', 'tune', 'dev', 'test') if doc in v[p]][0]
                                for c, v in configs.items()}})
    return {'rule': (f'at least {SHARED_MIN_SENTENCES} sentences of the MuLMS document share a '
                     "word 8-gram with the SOFC-Exp article, after dapt_mlm.py's normalisation"),
            'pairs': pairs, 'largest_other': other, 'sha256': canonical_sha256(pairs)}


def write_data_files():
    SOTA_DIR.mkdir(parents=True, exist_ok=True)
    for path, record in ((SPLITS_FILE, mm_screen_splits_record()),
                         (SHARED_FILE, shared_documents_record())):
        path.write_text(json.dumps(record, indent=1) + '\n')
        print(f"wrote {path} (sha256 {record['sha256'][:12]})")


def shared_pairs():
    record = json.loads(SHARED_FILE.read_text())
    if record['sha256'] != canonical_sha256(record['pairs']):
        raise SystemExit(f'{SHARED_FILE}: the SHA-256 does not match its pairs')
    return record['pairs']


def shared_mulms_documents():
    """The MuLMS documents that SOFC-Exp also holds (P1's C2 drops them from MuLMS's side)."""
    return {p['mulms'] for p in shared_pairs()}


def shared_sofc_articles():
    """The SOFC-Exp articles that MuLMS also holds (dropped from SOFC-Exp's side)."""
    return {p['sofc_exp'] for p in shared_pairs()}


# ---------------------------------------------------------------------------------------------
# The cells
# ---------------------------------------------------------------------------------------------
BENCHMARKS = ('fine', 'coarse', 'sofc', 'mulms')
UNITS = {'fine': [f'seed{k}' for k in range(10)], 'coarse': [f'seed{k}' for k in range(10)],
         'sofc': [f'seed{s}' for s in (0, 100, 200, 300, 400)],
         'mulms': ['c1s0', 'c2s0', 'c3s0']}
EXTRA_UNITS = {'fine': [f'seed{k}' for k in range(10, 14)],
               'coarse': [f'seed{k}' for k in range(10, 14)],
               'sofc': [f'seed{s}' for s in (500, 600, 700, 800)],
               'mulms': ['c4s0', 'c5s0', 'c1s100', 'c2s100']}
# the units if the budget's third cut applies (MaterioMiner at 5 units)
CUT_UNITS = {'fine': ([f'seed{k}' for k in range(5)], [f'seed{k}' for k in range(5, 9)]),
             'coarse': ([f'seed{k}' for k in range(5)], [f'seed{k}' for k in range(5, 9)])}
DEBERTA = 'microsoft/deberta-v3-large'
SCIDEBERTA = 'KISTI-AI/scideberta'
SCIDEBERTA_REVISION = '98fab84484cea48d2997348a18cebf2d4f419a14'
B3_MODEL = '/tmp/dapt_b3'       # B3's pre-trained model, written by its lane's DAPT job
B_RECIPE = ('--lr', '1e-5', '--batch-size', '8', '--gradient-checkpointing')
CHANGES = {
    'tagger': {                 # ner_baseline.py: MaterioMiner and SOFC-Exp
        'a1': ('--lr', '2e-5'), 'a2': ('--lr', '3e-5'), 'a3': ('--epochs', '40', '--patience', '5'),
        'a4': ('--warmup', '0'), 'a5': ('--llrd', '0.9'), 'a6': ('--reinit-top', '1'),
        'a7': ('--average-last', '3'),
        'b1': ('--model', DEBERTA, *B_RECIPE), 'b2': ('--model', SCIDEBERTA, *B_RECIPE),
        'b3': ('--model', B3_MODEL),
        'c1': (), 'c2': ('--joint-mulms',), 'c3': ('--context', '64'), 'e': ()},
    'mulms': {                  # mulms_nested.py
        'a1m': ('--lr', '5e-5'), 'a2m': ('--lr-schedule', 'sqrt', '--patience', '15'),
        'b1': ('--model', DEBERTA, '--lr', '1e-5', '--micro-batch', '4',
               '--gradient-checkpointing'),
        'b2': ('--model', SCIDEBERTA, '--lr', '1e-5', '--gradient-checkpointing'),
        'c2': ('--joint-sofc',), 'c3': ('--context', '64'), 'e': ('--span-head',)}}
ROUND1 = {'fine': ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'b1', 'b2', 'c1', 'c3'],
          'coarse': ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'b1', 'b2', 'c1', 'c3'],
          'sofc': ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'b1', 'b2', 'b3', 'c1', 'c2',
                   'c3', 'e'],
          'mulms': ['a0', 'a1m', 'a2m', 'c2', 'c3', 'e']}
MULMS_B = ('b1', 'b2')          # enter MuLMS by the protocol's rule (mulms_b_entries)
A_FAMILY = {'tagger': ('a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'), 'mulms': ('a1m', 'a2m')}
HEADS = ('c1', 'c2', 'e')       # each adds a head; at most one in a cell
LR_CHANGES = {'tagger': ('a1', 'a2'), 'mulms': ()}
THREADS = 2


def family(benchmark):
    return 'mulms' if benchmark == 'mulms' else 'tagger'


def cell_name(changes):
    """A cell's name and tag stem: 'a0', or its changes sorted and joined ('a2a5c3')."""
    return ''.join(sorted(changes)) if changes else 'a0'


def tag_of(changes):
    return 'p1' + cell_name(changes)


def unit_seed(unit):
    """A tagger unit's seed (seed<k>), or a MuLMS unit's (configuration, seed) (c<k>s<seed>)."""
    if unit.startswith('seed'):
        return int(unit[4:])
    c, s = unit[1:].split('s')
    return int(c), int(s)


def change_args(benchmark, changes):
    """The flags of a cell's changes, merged in name order."""
    out = []
    for name in sorted(changes):
        out += list(CHANGES[family(benchmark)][name])
    return out


def jobs_for(benchmark, changes, units):
    """The jobs of a cell on some of its units: one ner_baseline.py job per cell (its units are
    its --seeds); one mulms_nested.py job per MuLMS unit (a file each)."""
    changes = frozenset(changes)
    tag = tag_of(changes)
    extra = change_args(benchmark, changes)
    if benchmark == 'mulms':
        jobs = []
        for unit in units:
            c, seed = unit_seed(unit)
            jobs.append([MULMS_NESTED_SCRIPT, '--screen', '--configs', str(c), '--seed', str(seed),
                         *extra, '--tag', f"{tag}u{c}{f's{seed}' if seed else ''}",
                         '--threads', str(THREADS)])
        return jobs
    seeds = [str(unit_seed(u)) for u in units]
    if benchmark in LEVELS:
        base = (['--granularity', 'both', '--multi-task', '--screen-level', benchmark, '--crf']
                if 'c1' in changes else ['--granularity', benchmark, '--crf'])
        base += ['--screen', '--screen-splits']
    else:
        if 'c1' in changes:
            base = ['--granularity', 'sofc_both', '--multi-task', '--crf']
        elif 'e' in changes:
            base = ['--granularity', 'sofc', '--span-head']
        else:
            base = ['--granularity', 'sofc', '--crf']
        base += ['--screen', '--published-split']
    return [[*base, *extra, '--seeds', *seeds, '--tag', tag, '--threads', str(THREADS)]]


def results_file(benchmark, job):
    """The results file (name) that a job writes for its benchmark."""
    import mulms_nested as mn
    import ner_baseline as nb
    if job[0] == MULMS_NESTED_SCRIPT:
        args = mn.parse_args(job[1:])
        return mn.results_path('', args.tag, args.max_train, args.screen, args.span_head).name
    args = nb.parse_args(job)
    granularity = benchmark if benchmark in LEVELS else 'sofc'
    return Path(nb.results_path('', granularity, args)).name


# ---------------------------------------------------------------------------------------------
# Reading and checking the files
# ---------------------------------------------------------------------------------------------
# P0's MuLMS runs (A0 on MuLMS), by SHA-256 of the file: the one exception to the refusal of a
# test field; only their `dev` field is read
P0_MULMS = {1: 'c0d296b9de948b3c', 2: '96454fc298bdfbeb', 3: 'b28cb4c7ac0e1fe6',
            4: 'a61998b8717fb428', 5: '5aa81001eea19964'}
P0_MULMS_FILE = 'ner_results_mulms_nested_configs_bilou_crf_c{c}.json'
TEST_FIELDS = ('f1', 'f1_strict', 'precision', 'recall', 'macro_f1', 'weighted_f1', 'per_class',
               'n_test', 'test_index', 'test_gold', 'test_pred', 'test')
TOP_TEST_FIELDS = ('test_f1_mean', 'test_f1_std', 'test_f1_strict_mean', 'test_micro_f1_mean',
                   'test_micro_f1_sd')
# settings that do not change a run; a cell's file holds the runs of its jobs (the units of
# round 1, then the extra units), so its stored seeds are those of the last job
RUNTIME_KEYS = ('threads', 'device', 'seeds')


def check_screen_file(data, name):
    """Raise ValueError unless `data` is a screen file with no test field."""
    if not data.get('screen'):
        raise ValueError(f'{name}: not a screen file')
    found = [k for k in TOP_TEST_FIELDS if data.get(k) is not None]
    found += sorted({k for r in data.get('runs', []) for k in r if k in TEST_FIELDS})
    if found:
        raise ValueError(f'{name}: test fields {found}')


def check_settings(data, job, name):
    """Raise ValueError unless the file's stored settings are the job's."""
    import mulms_nested as mn
    import ner_baseline as nb
    if job[0] == MULMS_NESTED_SCRIPT:
        args = vars(mn.parse_args(job[1:]))
        wrong = {k: (v, args.get(k)) for k, v in data['config'].items()
                 if k in args and args[k] != v}
        if data.get('model') != args['model']:
            wrong['model'] = (data.get('model'), args['model'])
    else:
        want = nb.stored_config(nb.parse_args(job))
        wrong = {k: (data['config'].get(k), v) for k, v in want.items()
                 if k not in RUNTIME_KEYS and data['config'].get(k) != v}
    if wrong:
        raise ValueError(f'{name}: settings differ from its job (file, job): {wrong}')


def environment_of(run):
    env = run.get('environment') or {}
    return tuple(sorted((k, str(v)) for k, v in env.items() if k in (
        'device', 'cuda', 'torch', 'transformers')))


def p0_dev(c, results_dir=GPU_DIR):
    """A0's dev score on MuLMS configuration c at seed 0, from P0's file (SHA-256 checked)."""
    path = Path(results_dir) / P0_MULMS_FILE.format(c=c)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(P0_MULMS[c]):
        raise ValueError(f'{path.name}: SHA-256 {digest[:16]}, not {P0_MULMS[c]}')
    run = json.loads(path.read_text())['runs'][0]
    if run.get('run_id') != f'c{c}' or run.get('seed') != 0:
        raise ValueError(f'{path.name}: not configuration {c} at seed 0')
    return run['dev']['micro_f1']


def last3(run):
    """The mean of a run's last 3 validation scores (MaterioMiner and SOFC-Exp)."""
    values = [h['val_f1'] for h in run.get('history', []) if 'val_f1' in h]
    return statistics.fmean(values[-3:]) if values else None


def read_cell(benchmark, changes, units, dirs):
    """{unit: (value, last-3 mean or None, environment)} of the units whose run is found,
    every file checked. dirs: where the screen files are, in order."""
    out = {}
    if benchmark == 'mulms' and not changes:
        # A0 on MuLMS: c1 re-run in screen mode, c2-c5 at seed 0 from P0's files
        for unit in units:
            c, seed = unit_seed(unit)
            if seed == 0 and c != 1:
                out[unit] = (p0_dev(c), None, None)
    todo = [u for u in units if u not in out]
    jobs = jobs_for(benchmark, changes, todo)
    # MuLMS: one job (and file) per unit; the taggers: one job for all the cell's units
    pairs = list(zip(todo, jobs)) if benchmark == 'mulms' else [(None, job) for job in jobs]
    for unit, job in pairs:
        name = results_file(benchmark, job)
        path = next((Path(d) / name for d in dirs if (Path(d) / name).exists()), None)
        if path is None:
            continue
        data = json.loads(path.read_text())
        check_screen_file(data, name)
        check_settings(data, job, name)
        runs = {r['run_id']: r for r in data['runs']}
        if benchmark == 'mulms':
            c, seed = unit_seed(unit)
            run = runs.get(f'c{c}')
            if run is None:
                continue
            if run.get('seed') != seed:
                raise ValueError(f"{name}: seed {run.get('seed')}, not {seed}")
            check_environment(run, name)
            out[unit] = (run['dev']['micro_f1'], None, environment_of(run))
            continue
        for unit in todo:
            run = runs.get(unit)
            if run is None:
                continue
            if run.get('seed') != unit_seed(unit):
                raise ValueError(f"{name}: run {unit} has seed {run.get('seed')}")
            check_environment(run, name)
            out[unit] = (run['val_f1'], last3(run), environment_of(run))
    return out


_REFERENCE_ENV = {}


def check_environment(run, name, results_dir=GPU_DIR):
    """Raise ValueError unless a run's GPU, CUDA, torch and transformers are those of P0's runs
    (results/gpu, MuLMS configuration 1)."""
    if 'env' not in _REFERENCE_ENV:
        path = Path(results_dir) / P0_MULMS_FILE.format(c=1)
        _REFERENCE_ENV['env'] = dict(environment_of(json.loads(path.read_text())['runs'][0]))
    want, have = _REFERENCE_ENV['env'], dict(environment_of(run))
    wrong = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
    if wrong:
        raise ValueError(f'{name}: environment differs from P0 (run, P0): {wrong}')


def cell_record(values, a0_values, units):
    """A cell's summary on `units` (those it has): the mean, the mean of the last 3 validation
    scores, and the gain over A0 paired by unit, its SD and standard error."""
    have = [u for u in units if u in values]
    record = {'units': have, 'n': len(have),
              'values': {u: values[u][0] for u in have}}
    if not have:
        return record
    record['mean'] = round(statistics.fmean(values[u][0] for u in have), 2)
    l3 = [values[u][1] for u in have if values[u][1] is not None]
    record['last3_mean'] = round(statistics.fmean(l3), 2) if len(l3) == len(have) else None
    paired = [values[u][0] - a0_values[u][0] for u in have if u in a0_values]
    if len(paired) == len(have):
        record['gain'] = round(statistics.fmean(paired), 3)
        sd = statistics.stdev(paired) if len(paired) > 1 else 0.0
        record['gain_sd'] = round(sd, 3)
        record['gain_se'] = round(sd / math.sqrt(len(paired)), 3)
    return record


# ---------------------------------------------------------------------------------------------
# The rounds and the selection
# ---------------------------------------------------------------------------------------------
def mulms_b_entries(tables):
    """The B cells that enter MuLMS: each one's gain over A0 above 1.0 on SOFC-Exp and on
    MaterioMiner fine-grained or coarse-grained (round 1), with the budget table's leave.
    None while those round-1 cells are not all there."""
    fits = timing_allows_mulms_b()
    out = []
    for b in MULMS_B:
        rows = [tables[bm]['cells'].get(b) for bm in ('sofc', 'fine', 'coarse')]
        if any(r is None or 'gain' not in r for r in rows):
            if b in removed_cells('sofc') and b in removed_cells('fine') and \
                    b in removed_cells('coarse'):
                continue
            return None
        sofc, fine, coarse = (r['gain'] for r in rows)
        if sofc > 1.0 and (fine > 1.0 or coarse > 1.0) and fits.get(b, False):
            out.append(b)
    return out


def timing_allows_mulms_b():
    """{b1, b2: whether the budget table fits them on MuLMS} (False without a table)."""
    if not TIMING_FILE.exists():
        return {}
    return json.loads(TIMING_FILE.read_text()).get('mulms_b_fits', {})


def removed_cells(benchmark):
    """The round-1 cells that the budget's cuts take off a benchmark (results/sota/p1_timing)."""
    if not TIMING_FILE.exists():
        return set()
    cuts = json.loads(TIMING_FILE.read_text()).get('removed', {})
    return set(cuts.get(benchmark, []))


def benchmark_units(benchmark):
    """(units, extra units) of a benchmark, after the budget's cuts."""
    if TIMING_FILE.exists() and benchmark in LEVELS and json.loads(
            TIMING_FILE.read_text()).get('mm_five_units'):
        return CUT_UNITS[benchmark]
    return UNITS[benchmark], EXTRA_UNITS[benchmark]


def analyse(dirs=(SCREEN_DIR,)):
    """The state of P1 per benchmark: its cells' records, G, A*, A's best cell, the chain, the
    cells still to run (`next`: (changes, units)), and once none is left the selection."""
    tables = {}
    for benchmark in BENCHMARKS:
        tables[benchmark] = {'cells': {}, 'next': []}
    # round 1 on the taggers first: MuLMS's B entries depend on them
    for benchmark in ('fine', 'coarse', 'sofc', 'mulms'):
        table = tables[benchmark]
        units, extra = benchmark_units(benchmark)
        table['units'], table['extra_units'] = units, extra
        a0 = read_cell(benchmark, frozenset(), units + extra, dirs)
        table['a0_values'] = a0
        cells = [c for c in ROUND1[benchmark] if c not in removed_cells(benchmark)]
        if benchmark == 'mulms':
            entries = mulms_b_entries(tables)
            if entries is None:
                table['waiting'] = "MuLMS's B entries wait for round 1 on the taggers"
            else:
                table['mulms_b'] = entries
                cells += entries
        missing = []
        for name in cells:
            changes = frozenset() if name == 'a0' else frozenset([name])
            values = a0 if not changes else read_cell(benchmark, changes, units + extra, dirs)
            record = cell_record(values, a0, units)
            record.update({'changes': sorted(changes), 'round': 1})
            table['cells'][cell_name(changes)] = record
            table.setdefault('values', {})[cell_name(changes)] = values
            if record['n'] < len(units):
                missing.append((changes, [u for u in units if u not in values]))
        if missing or 'waiting' in table:
            table['next'] = missing
            continue
        round2(benchmark, table, dirs)
    return tables


def gain_ok(record):
    return 'gain' in record and record['gain'] > record['gain_se']


def add_cell(benchmark, table, changes, dirs, round_no=2):
    """Read a cell into the table; returns its record, or None (and queues it) if not run."""
    name = cell_name(changes)
    if name in table['cells'] and table['cells'][name]['n'] == len(table['units']):
        return table['cells'][name]
    values = read_cell(benchmark, changes, table['units'] + table['extra_units'], dirs)
    record = cell_record(values, table['a0_values'], table['units'])
    record.update({'changes': sorted(changes), 'round': round_no})
    table['cells'][name] = record
    table.setdefault('values', {})[name] = values
    if record['n'] < len(table['units']):
        table['next'].append((frozenset(changes),
                              [u for u in table['units'] if u not in values]))
        return None
    return record


def round2(benchmark, table, dirs):
    """Round 2 (A*, the chain) and then the selection with the tie rule, as far as the runs
    allow; what is left to run goes to table['next']."""
    fam = family(benchmark)
    cells = table['cells']
    g = sorted(name for name, r in cells.items() if r['round'] == 1 and name != 'a0'
               and gain_ok(r))
    table['G'] = g
    a_changes = [c for c in g if c in A_FAMILY[fam]]
    lrs = [c for c in a_changes if c in LR_CHANGES[fam]]
    if len(lrs) == 2:
        drop = min(lrs, key=lambda c: (cells[c]['mean'], c))
        a_changes = [c for c in a_changes if c != drop]
    a_star = frozenset(a_changes)
    table['A*'] = cell_name(a_star)
    if len(a_star) >= 2 and add_cell(benchmark, table, a_star, dirs) is None:
        return
    a_names = ['a0', *[c for c in A_FAMILY[fam] if c in cells], cell_name(a_star)]
    a_best = min(dict.fromkeys(a_names), key=lambda n: (-cells[n]['mean'], len(
        cells[n]['changes']), n))
    table['A_best'] = a_best
    b_in_g = [c for c in g if c.startswith('b')]
    current = frozenset(cells[a_best]['changes'])
    if b_in_g:
        best_b = min(b_in_g, key=lambda n: (-cells[n]['mean'], n))
        if cells[best_b]['mean'] > cells[a_best]['mean']:
            current = frozenset([best_b])
    chain = [cell_name(current)]
    if current == frozenset(['b3']) and a_star:
        step = current | a_star
        record = add_cell(benchmark, table, step, dirs)
        if record is None:
            table['chain'] = chain
            return
        chain.append(cell_name(step))
        if record['mean'] > cells[cell_name(current)]['mean']:
            current = step
    additions = sorted((c for c in g if c in ('c1', 'c2', 'c3', 'e')),
                       key=lambda c: (-cells[c]['gain'], 0 if c.startswith('c') else 1, c))
    for x in additions:
        if x in HEADS and current & set(HEADS):
            continue
        if x in current:
            continue
        step = current | {x}
        record = add_cell(benchmark, table, step, dirs)
        if record is None:
            table['chain'] = chain
            return
        chain.append(cell_name(step))
        if record['mean'] > cells[cell_name(current)]['mean']:
            current = step
    table['chain'] = chain
    table['chain_end'] = cell_name(current)
    select(benchmark, table, dirs)


def select(benchmark, table, dirs):
    """The selection with the tie rule: every cell within 1 point of the highest mean, at most
    the 3 highest, gets the extra units; the highest mean over all units is selected, a further
    tie going to fewer changes, then to the name."""
    cells = {n: r for n, r in table['cells'].items() if r['n'] == len(table['units'])}
    order = sorted(cells, key=lambda n: (-cells[n]['mean'], len(cells[n]['changes']), n))
    top = cells[order[0]]['mean']
    tied = [n for n in order if top - cells[n]['mean'] <= 1.0][:3]
    table['tie'] = tied
    if len(tied) < 2:
        table['selected'] = order[0]
        return
    all_units = table['units'] + table['extra_units']
    full = {}
    for name in tied:
        changes = frozenset(cells[name]['changes'])
        values = read_cell(benchmark, changes, all_units, dirs) if changes else \
            table['a0_values']
        missing = [u for u in table['extra_units'] if u not in values]
        if missing:
            table['next'].append((changes, missing))
            continue
        full[name] = round(statistics.fmean(values[u][0] for u in all_units), 2)
    if len(full) < len(tied):
        return
    table['tie_means'] = full
    table['selected'] = min(tied, key=lambda n: (-full[n], len(cells[n]['changes']), n))


def next_jobs(tables):
    """[(benchmark, cell, job)] of every cell still to run."""
    out = []
    for benchmark, table in tables.items():
        for changes, units in table['next']:
            for job in jobs_for(benchmark, changes, units):
                out.append((benchmark, sorted(changes), job))
    return out


def summary(tables):
    """The JSON record of P1 (results/gpu/sota_p1.json): per benchmark its cells, G, A*, A's best
    cell, the chain, the tie and the selection; no test number."""
    out = {'note': ('P1 screens (SOTA plan, ROADMAP §4): validation scores only; no claim and '
                    'no decision')}
    for benchmark, table in tables.items():
        out[benchmark] = {k: v for k, v in table.items()
                          if k not in ('values', 'a0_values', 'next')}
        out[benchmark]['next'] = [[cell_name(c), u] for c, u in table['next']]
    return out


# ---------------------------------------------------------------------------------------------
# The timing table (batch p1t) and the budget's cuts
# ---------------------------------------------------------------------------------------------
CAP_SESSION_HOURS = 24
# A0's stored runs: the mean best epoch and the patience, per benchmark (the table's epochs)
STORED_BEST_EPOCH = {'fine': 7.8, 'coarse': 10.5, 'sofc': 5.5, 'mulms': 6.0}
CUT_LADDER = (('mulms', 'a2m'), ('mulms', 'c3'), ('mm', 'five_units'), ('all', 'b2'),
              ('coarse', 'b1'))


def epochs_of(benchmark, changes):
    """A cell's expected epochs per run: the mean best epoch + the patience + 1, capped."""
    limit, patience = (100, 5) if benchmark == 'mulms' else (20, 3)
    if 'a3' in changes:
        limit, patience = 40, 5
    if 'a2m' in changes:
        patience = 15
    return min(limit, STORED_BEST_EPOCH[benchmark] + patience + 1)


def timing(minutes_per_epoch, removed=None, five_units=False):
    """The table: per benchmark and round-1 cell, its runs, minutes per run and lane-minutes;
    the reserves for round 2 (6 cells per benchmark at its mean cell) and the extra units (3
    cells); the session-hours on two lanes. minutes_per_epoch: {benchmark: {kind: minutes}},
    the kinds 'a0', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'e' (the A changes take A0's)."""
    removed = removed or {}
    table, total = {}, 0.0
    for benchmark in BENCHMARKS:
        units, extra = (CUT_UNITS[benchmark] if five_units and benchmark in LEVELS
                        else (UNITS[benchmark], EXTRA_UNITS[benchmark]))
        rows = {}
        for name in ROUND1[benchmark]:
            if name in removed.get(benchmark, ()):
                continue
            changes = frozenset() if name == 'a0' else frozenset([name])
            kind = name if name[0] in 'bce' else 'a0'
            per_epoch = minutes_per_epoch[benchmark].get(kind, minutes_per_epoch[benchmark]['a0'])
            minutes = per_epoch * epochs_of(benchmark, changes)
            runs = len(units)
            rows[name] = {'runs': runs, 'minutes_per_run': round(minutes, 2),
                          'lane_minutes': round(runs * minutes, 1)}
        cell_mean = statistics.fmean(r['minutes_per_run'] for r in rows.values())
        reserve2 = 6 * len(units) * cell_mean
        reserve3 = 3 * len(extra) * cell_mean
        lane = sum(r['lane_minutes'] for r in rows.values()) + reserve2 + reserve3
        table[benchmark] = {'cells': rows, 'round2_reserve': round(reserve2, 1),
                            'extra_reserve': round(reserve3, 1), 'lane_minutes': round(lane, 1)}
        total += lane
    return table, round(total / 2 / 60, 2)


def budget(minutes_per_epoch, mulms_b_minutes, cap=CAP_SESSION_HOURS):
    """The table after the cut ladder: cuts are made in order until the session-hours fit the
    cap. Returns the record for results/sota/p1_timing.json, with whether B1 and B2 would fit
    on MuLMS (`mulms_b_fits`: their 3 units' minutes on top of the table)."""
    removed, five, cuts = {}, False, []
    table, hours = timing(minutes_per_epoch, removed, five)
    for where, what in CUT_LADDER:
        if hours <= cap:
            break
        cuts.append(f'{where}: {what}')
        if what == 'five_units':
            five = True
        elif where == 'all':
            for benchmark in ('fine', 'coarse', 'sofc'):
                removed.setdefault(benchmark, []).append(what)
        else:
            removed.setdefault(where, []).append(what)
        table, hours = timing(minutes_per_epoch, removed, five)
    fits = {b: hours + 3 * mulms_b_minutes.get(b, 0.0) / 2 / 60 <= cap
            and b not in removed.get('sofc', []) for b in MULMS_B}
    return {'table': table, 'session_hours': hours, 'cap': cap, 'cuts': cuts,
            'removed': removed, 'mm_five_units': five, 'mulms_b_fits': fits}


STORED_A0 = {'fine': 'ner_results_fine_published_split_crf.json',
             'coarse': 'ner_results_coarse_published_split_crf.json',
             'sofc': 'ner_results_sofc_published_split_crf.json'}
DAPT_LIMIT_MINUTES = 60          # B3 leaves P1 if its pre-training takes longer


def per_epoch(path):
    """The mean minutes per epoch of a file's runs (run time over epochs trained)."""
    runs = json.loads(Path(path).read_text())['runs']
    return statistics.fmean(r['runtime_sec'] / 60 / len(r['history']) for r in runs)


def measured_minutes(pulled, results_dir=GPU_DIR):
    """({benchmark: {kind: minutes per epoch}}, {b1, b2: minutes of a MuLMS run}, the
    pre-training's minutes) from batch p1t's files under `pulled` (the 2-epoch runs, tags
    p1t<kind>) and A0's stored runs (results/gpu; MuLMS: P0's)."""
    pulled = Path(pulled)
    found = {p.name: p for p in pulled.rglob('ner_results_*.json')}
    out = {b: {} for b in BENCHMARKS}
    for benchmark in LEVELS + ('sofc',):
        out[benchmark]['a0'] = per_epoch(Path(results_dir) / STORED_A0[benchmark])
    out['mulms']['a0'] = statistics.fmean(
        per_epoch(Path(results_dir) / P0_MULMS_FILE.format(c=c)) for c in range(1, 6))
    for benchmark, kinds in (('fine', ('b1', 'b2', 'c1', 'c3')),
                             ('sofc', ('b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'e')),
                             ('mulms', ('c2', 'c3', 'e'))):
        for kind in kinds:
            import kaggle_jobs as kj
            name = results_file(benchmark, kj.timing_job(benchmark, kind))
            if name not in found:
                raise SystemExit(f'p1t: no timing file {name}')
            out[benchmark][kind] = per_epoch(found[name])
    out['coarse'].update({k: v * out['coarse']['a0'] / out['fine']['a0']
                          for k, v in out['fine'].items() if k != 'a0'})
    mulms_b = {b: out['mulms']['a0'] * out['sofc'][b] / out['sofc']['a0']
               * epochs_of('mulms', frozenset([b])) for b in MULMS_B}
    report = next(iter(pulled.rglob('dapt_b3.json')), None)
    dapt = None
    if report is not None:
        dapt = json.loads(report.read_text())['runtime']['wall_seconds'] / 60
    return out, mulms_b, dapt


def timing_record(pulled, results_dir=GPU_DIR):
    """The record of results/sota/p1_timing.json: the measured minutes, the table after the
    cut ladder, and whether B3's pre-training fits its hour."""
    minutes, mulms_b, dapt = measured_minutes(pulled, results_dir)
    record = budget(minutes, mulms_b)
    if dapt is None or dapt > DAPT_LIMIT_MINUTES:
        record['removed'].setdefault('sofc', []).append('b3')
        record['cuts'].append(f'sofc: b3 (its pre-training took {dapt} minutes)')
    record.update({'minutes_per_epoch': {b: {k: round(v, 3) for k, v in m.items()}
                                         for b, m in minutes.items()},
                   'mulms_b_minutes_per_run': {b: round(v, 1) for b, v in mulms_b.items()},
                   'dapt_minutes': None if dapt is None else round(dapt, 1)})
    return record


SMOKE_REFERENCE = ROOT / 'results' / 'gpu_repeats' / \
    'ner_results_fine_random_split_vary_crf_smokebn.json'


def without(data, tag):
    """A results file without what a repeat may change: its runs' run times, its tag, and the
    tag in its variant's name."""
    data = json.loads(json.dumps(data))
    data.get('config', {}).pop('tag', None)
    if data.get('variant', '').endswith(f'_{tag}'):
        data['variant'] = data['variant'][:-len(tag) - 1]
    for r in data.get('runs', []):
        r.pop('runtime_sec', None)
    return data


def p1t_checks(pulled, results_dir=GPU_DIR):
    """The checks of batch p1t ({check: problems, [] when it passes}): the smoke4 repeat
    smokebt equals smokebn but for its tag and run times; A0's screen runs on SOFC-Exp have the
    stored runs' val_f1 and history at seeds 0-400; MuLMS's c1 in screen mode has P0's dev
    scores and history."""
    found = {p.name: p for p in Path(pulled).rglob('ner_results_*.json')}
    out = {}
    smoke = found.get('ner_results_fine_random_split_vary_crf_smokebt.json')
    if smoke is None:
        out['smoke'] = ['smokebt is missing']
    else:
        a = without(json.loads(smoke.read_text()), 'smokebt')
        b = without(json.loads(SMOKE_REFERENCE.read_text()), 'smokebn')
        out['smoke'] = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    name = results_file('sofc', jobs_for('sofc', frozenset(), UNITS['sofc'])[0])
    stored = {r['run_id']: r for r in json.loads(
        (Path(results_dir) / STORED_A0['sofc']).read_text())['runs']}
    problems = []
    if name not in found:
        problems.append(f'{name} is missing')
    else:
        runs = {r['run_id']: r for r in json.loads(found[name].read_text())['runs']}
        for unit in UNITS['sofc']:
            r, s = runs.get(unit), stored.get(unit)
            if r is None or s is None:
                problems.append(f'{unit}: missing')
            elif r['val_f1'] != s['val_f1'] or r['history'] != s['history']:
                problems.append(f"{unit}: val F1 {r['val_f1']} against the stored {s['val_f1']}")
    out['sofc_a0'] = problems
    name = results_file('mulms', jobs_for('mulms', frozenset(), ['c1s0'])[0])
    problems = []
    if name not in found:
        problems.append(f'{name} is missing')
    else:
        run = json.loads(found[name].read_text())['runs'][0]
        p0 = json.loads((Path(results_dir) / P0_MULMS_FILE.format(c=1)).read_text())['runs'][0]
        if run['dev'] != p0['dev'] or run['history'] != p0['history']:
            problems.append(f"c1: dev micro F1 {run['dev']['micro_f1']} against P0's "
                            f"{p0['dev']['micro_f1']}")
    out['mulms_c1'] = problems
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="The SOTA plan's P1: the screens")
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('data-files')
    p = sub.add_parser('checks')
    p.add_argument('pulled', help="batch p1t's pulled output")
    p = sub.add_parser('timing')
    p.add_argument('pulled', help="batch p1t's pulled output (kaggle_jobs.py pull)")
    p.add_argument('--out', default=str(TIMING_FILE))
    for name in ('status', 'next-jobs', 'select'):
        p = sub.add_parser(name)
        p.add_argument('--results', nargs='+', default=[str(SCREEN_DIR)])
        if name == 'next-jobs':
            p.add_argument('--out', default=str(SOTA_DIR / 'p1_next_jobs.json'))
        if name == 'select':
            p.add_argument('--out', default=str(OUT))
    args = parser.parse_args(argv)
    if args.command == 'data-files':
        write_data_files()
        return
    if args.command == 'checks':
        result = p1t_checks(args.pulled)
        for check, problems in result.items():
            print(f"{check}: {'passed' if not problems else 'FAILED: ' + '; '.join(problems)}")
        if any(result.values()):
            raise SystemExit('a check failed: the batch counts for nothing (ROADMAP §4, P1)')
        return
    if args.command == 'timing':
        record = timing_record(args.pulled)
        Path(args.out).write_text(json.dumps(record, indent=1) + '\n')
        print(f"{record['session_hours']} session-hours (cap {record['cap']}); cuts "
              f"{record['cuts'] or 'none'}; wrote {args.out}")
        return
    tables = analyse(args.results)
    if args.command == 'status':
        for benchmark, table in tables.items():
            print(f"== {benchmark}: {len(table['cells'])} cells; G {table.get('G')}; "
                  f"A* {table.get('A*')}; A's best {table.get('A_best')}; chain "
                  f"{table.get('chain')}; selected {table.get('selected')}")
            for name, r in sorted(table['cells'].items()):
                print(f"   {name:10} n {r['n']:2} mean {r.get('mean')} gain {r.get('gain')} "
                      f"(SE {r.get('gain_se')}) last3 {r.get('last3_mean')}")
            for changes, units in table['next']:
                print(f'   next: {cell_name(changes)} on {units}')
    elif args.command == 'next-jobs':
        jobs = next_jobs(tables)
        Path(args.out).write_text(json.dumps(
            [{'benchmark': b, 'cell': cell_name(c), 'changes': c, 'job': j} for b, c, j in jobs],
            indent=1) + '\n')
        print(f'{len(jobs)} jobs to {args.out}')
    else:
        record = summary(tables)
        Path(args.out).write_text(json.dumps(record, indent=1) + '\n')
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
