"""
sota_p0.py - the reproduction gates of the SOTA plan's P0 (docs/SOTA_PLAN.md; the protocol
in docs/ROADMAP.md §4)

No training. P0 checks, before any candidate is screened, that our pipeline reproduces each
published number it will be compared with. A gate is passed unless our mean is below the
published one beyond chance: it fails when the Welch 95% interval of our mean minus the
published mean lies below 0, from the published mean, sample SD (PUBLISHED: Schrader et al.'s
printed SDs are population SDs) and number of runs. A verdict that changes within the printed
SD's rounding is flagged borderline, and our SD above twice the published one is flagged. What
a failed gate stops, and when its gap counts as explained, is fixed in the protocol.
  - SOFC-Exp, cross-validation: MatSciBERT-CRF, 84.77 +- 1.02 micro F1, the SD over 15 runs (5
    folds x 3 seeds; Gupta et al. 2022). Ours: the CRF head on the folds of MatSciBERT's code
    (ner_baseline.py --sofc-cv), 3 seeds, the same 15-run unit.
  - SOFC-Exp, official split: 83.4 +- 0.9 over 5 runs on 10 of the 11 test articles
    (Schrader et al. 2023, Table 11), the 11th being an article MuLMS also holds
    (TEN_ARTICLES_DROP: PMC6247067, MuLMS's sofc_04, in its dev set; matched by title,
    results/mulms_articles.json). Ours: the stored CRF head
    (results/gpu/ner_results_sofc_published_split_crf.json, 10 seeds), re-scored on the 10
    articles from its saved predictions, per seed.
  - MuLMS, nested: the MatSciBERT CRF tagger with stacked BILOU tags, 79.6 +- 0.4 over its 5
    training configurations at 1e-4/7e-3 (Schrader et al. 2023, Tables 9 and 11). Ours:
    mulms_nested.py, one run per configuration (seed 0), each in its own file
    (..._c<k>.json).
  - MaterioMiner: done (plain MatSciBERT 70.19 [69.03, 71.35] over 10 seeds on the released
    split, against 69.92), reported.

It also writes the data files that fix P0's partitions, each with the SHA-256 of its canonical
JSON (sorted keys, no spaces): results/sota/sofc_cv_folds.json (the article IDs of each fold,
from ner_baseline.sofc_cv_folds) and results/sota/mulms_configurations.json (the documents of
each configuration, from mulms_nested.configurations_record over the pinned release).

Usage:
  python src/graph/sota_p0.py data-files      # write the two data files
  python src/graph/sota_p0.py gates [--results results/gpu] [--out results/gpu/sota_p0.json]
  python src/graph/sota_p0.py smoke <file> <file>   # smokebr against smokebs
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
FOLDS_FILE = SOTA_DIR / 'sofc_cv_folds.json'
CONFIGS_FILE = SOTA_DIR / 'mulms_configurations.json'
OUT = GPU_DIR / 'sota_p0.json'
TEN_ARTICLES_DROP = 'PMC6247067'
OFFICIAL_FILE = 'ner_results_sofc_published_split_crf.json'
CV_OFFSETS = (0, 100, 200)
CV_FILE = 'ner_results_sofc_matscibert_cv_crf_m{offset}.json'
MULMS_FILE = 'ner_results_mulms_nested_configs_bilou_crf_c{config}.json'   # one per configuration
# the published numbers, micro F1: the mean, the SD as printed, the runs, and the SD's convention.
# Schrader et al.'s are the release's aggregate_cv_scores.py output, np.std (a population SD,
# ddof 0) rounded to 1 decimal, so the gate uses the sample SD, printed x sqrt(n / (n - 1)).
# Gupta et al. print no convention, and MatSciBERT's code computes no SD: 1.02 is taken as a
# sample SD (a population SD would make it 1.06, also reported).
PUBLISHED = {
    'sofc_cv': {'mean': 84.77, 'sd_printed': 1.02, 'n': 15, 'decimals': 2,
                'convention': 'sample (assumed)',
                'source': 'Gupta et al. 2022 (npj), MatSciBERT-CRF, 5 folds x 3 seeds'},
    'sofc_official_10': {'mean': 83.4, 'sd_printed': 0.9, 'n': 5, 'decimals': 1,
                         'convention': 'population (np.std)',
                         'source': 'Schrader et al. 2023, Table 11, SOFC-Exp 3e-4/7e-3, 5 runs, '
                                   '10 test articles'},
    'mulms_nested': {'mean': 79.6, 'sd_printed': 0.4, 'n': 5, 'decimals': 1,
                     'convention': 'population (np.std)',
                     'source': 'Schrader et al. 2023, Tables 9 and 11, MatSciBERT CRF tagger, '
                               'MuLMS 1e-4/7e-3, 5 training configurations'},
}
SD_WARNING = 2.0      # our SD above twice the published sample SD is flagged


def sample_sd(published, sd=None):
    """The published SD as a sample SD (ddof 1)."""
    sd = published['sd_printed'] if sd is None else sd
    if published['convention'].startswith('population'):
        return sd * math.sqrt(published['n'] / (published['n'] - 1))
    return sd


MATERIOMINER = {'ours': 70.19, 'ci95': [69.03, 71.35], 'n': 10, 'published': 69.92,
                'note': 'plain MatSciBERT on the released split, fine-grained (paper B §3.1)'}


def canonical_sha256(obj):
    """The SHA-256 of an object's canonical JSON: sorted keys, no spaces."""
    text = json.dumps(obj, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def cv_folds_record():
    """The folds of --sofc-cv: {cv<k>: {train, dev, test}} with their rule and hash."""
    import ner_baseline as nb
    folds = {name: {'train': train, 'dev': dev, 'test': test}
             for name, train, dev, test in nb.sofc_cv_folds(nb.sofc_split())}
    return {'rule': ("MatSciBERT's modify_cross_val_data_split (M3RG-IITD/MatSciBERT, "
                     'ner/ner_datasets.py): the 34 train and dev articles of the official split '
                     'sorted by name; fold k holds out every 5th from position k - 1 for '
                     'validation; every fold tests on the 11 official test articles'),
            'folds': folds, 'sha256': canonical_sha256(folds)}


def mulms_configurations_record(sentences=None):
    """The documents of MuLMS's 5 configurations, with their rule, release and hash."""
    import mulms
    import mulms_nested as mn
    sentences = mn.read_release(offline=True) if sentences is None else sentences
    configs = mn.configurations_record(sentences)
    return {'rule': ('configuration c trains on the train folds other than train<c>, stops '
                     "early on train<c>, and is scored on the release's dev ('validation') and "
                     'test sets'),
            'release': {'revision': mulms.REVISION, 'files': dict(mulms.FILES)},
            'configurations': configs, 'sha256': canonical_sha256(configs)}


def write_data_files():
    SOTA_DIR.mkdir(parents=True, exist_ok=True)
    for path, record in ((FOLDS_FILE, cv_folds_record()),
                         (CONFIGS_FILE, mulms_configurations_record())):
        path.write_text(json.dumps(record, indent=1) + '\n')
        print(f"wrote {path} (sha256 {record['sha256'][:12]})")


def welch_summary(a, b):
    """mean(a) - mean(b) with its Welch 95% interval, from (mean, sd, n) summaries."""
    from scipy import stats
    va, vb = a['sd'] ** 2 / a['n'], b['sd'] ** 2 / b['n']
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (a['n'] - 1) + vb ** 2 / (b['n'] - 1))
    gap = a['mean'] - b['mean']
    half = float(stats.t.ppf(0.975, df)) * se
    return {'gap': round(gap, 2), 'ci95': [round(gap - half, 2), round(gap + half, 2)],
            'df': round(df, 1)}


def gate(values, published):
    """A gate's record: our runs' mean, SD and n; the published summary with its sample SD; the
    Welch interval of the difference; the verdict (failed only when the interval lies below
    0); `borderline` when the verdict differs at either end of the printed SD's rounding
    range; `sd_warning` when our SD exceeds SD_WARNING times the published sample SD."""
    mean, sd = statistics.fmean(values), statistics.stdev(values)
    ours = {'mean': mean, 'sd': sd, 'n': len(values)}

    def verdict_at(printed):
        theirs = {'mean': published['mean'], 'sd': sample_sd(published, printed),
                  'n': published['n']}
        test = welch_summary(ours, theirs)
        return test, ('failed' if test['ci95'][1] < 0 else 'passed')
    test, verdict = verdict_at(published['sd_printed'])
    half = 0.5 * 10 ** -published['decimals']
    edges = {verdict_at(published['sd_printed'] + d)[1] for d in (-half, half)}
    return {'ours': {'mean': round(mean, 2), 'sd': round(sd, 2), 'n': len(values)},
            'published': {**published, 'sd_sample': round(sample_sd(published), 3)},
            'welch': test, 'verdict': verdict, 'borderline': edges != {verdict},
            'sd_warning': sd > SD_WARNING * sample_sd(published)}


def check_cv_file(data, offset):
    """A cross-validation file holds what gate 1 aggregates: runs cv1 to cv5 at seed `offset`,
    from --sofc-cv with the CRF head's recipe (lr 5e-5, batch 16, 20 epochs, patience 3)."""
    config = data.get('config', {})
    wanted = {'sofc_cv': True, 'crf': True, 'lr': 5e-5, 'batch_size': 16, 'epochs': 20,
              'patience': 3, 'seed_offset': offset}
    wrong = {k: config.get(k) for k, v in wanted.items() if config.get(k) != v}
    runs = data.get('runs', [])
    if sorted(r.get('run_id') for r in runs) != [f'cv{k}' for k in range(1, 6)]:
        wrong['run_ids'] = sorted(r.get('run_id') for r in runs)
    if any(r.get('seed') != offset for r in runs):
        wrong['seeds'] = sorted({r.get('seed') for r in runs})
    if wrong:
        raise ValueError(f'cross-validation file at offset {offset}: {wrong}')


def check_mulms_file(data, config_number):
    """A MuLMS file holds what gate 3 aggregates: run c<k> at seed 0 with the settings of
    mulms_nested.SETTINGS, no schedule, the full data and the 473 tags."""
    import mulms_nested as mn
    config = data.get('config', {})
    wanted = {'lr': mn.SETTINGS['lr'], 'lr_crf': mn.SETTINGS['lr_crf'],
              'batch_size': mn.SETTINGS['batch_size'], 'epochs': mn.SETTINGS['epochs'],
              'patience': mn.SETTINGS['patience'], 'lr_schedule': 'none', 'max_train': 0,
              'seed': 0, 'seed_offset': 0, 'n_tags': 473}
    wrong = {k: config.get(k) for k, v in wanted.items() if config.get(k) != v}
    runs = data.get('runs', [])
    if [r.get('run_id') for r in runs] != [f'c{config_number}'] or runs[0].get('seed') != 0:
        wrong['runs'] = [(r.get('run_id'), r.get('seed')) for r in runs]
    if wrong:
        raise ValueError(f'MuLMS file of configuration {config_number}: {wrong}')


def smoke_identical(a, b):
    """Whether two smoke files (mulms_nested.py on the two GPUs) are identical but for their
    runs' runtime_sec and their config's tag; returns the differing keys."""
    def strip(data):
        data = json.loads(json.dumps(data))
        data.get('config', {}).pop('tag', None)
        for r in data.get('runs', []):
            r.pop('runtime_sec', None)
        return data
    a, b = strip(a), strip(b)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def rescore_official(results_dir=GPU_DIR, drop=TEN_ARTICLES_DROP):
    """Per seed, the stored CRF head's micro F1 on the 11 official test articles (which must
    equal its stored F1) and on the 10 left without `drop`, from its saved predictions."""
    from seqeval.metrics import f1_score
    import ner_baseline as nb
    stored = json.loads((Path(results_dir) / OFFICIAL_FILE).read_text())
    args = nb.parse_args(['--granularity', 'sofc', '--published-split', '--crf'])
    data, _, _ = nb.granularity_data('sofc', args)
    out = []
    for run in stored['runs']:
        docs = [data[i][2].removesuffix('.conll') for i in run['test_index']]
        gold = [g.split() for g in run['test_gold']]
        pred = [p.split() for p in run['test_pred']]
        keep = [k for k, d in enumerate(docs) if d != drop]
        eleven = round(100 * f1_score(gold, pred), 2)
        if eleven != run['f1']:
            raise ValueError(f"run {run['run_id']}: re-scored {eleven}, stored {run['f1']}")
        ten = round(100 * f1_score([gold[k] for k in keep], [pred[k] for k in keep]), 2)
        out.append({'run_id': run['run_id'], 'seed': run['seed'], 'f1_11': eleven, 'f1_10': ten,
                    'n_sentences_10': len(keep)})
    return out


def gates(results_dir=GPU_DIR):
    """Every gate whose runs are there; a missing one says so."""
    results_dir = Path(results_dir)
    out = {'note': ('P0 reproduction gates (SOTA plan); no claim. A gate fails only when the '
                    'Welch 95% interval of our mean minus the published mean lies below 0.'),
           'materiominer': MATERIOMINER}
    official = rescore_official(results_dir)
    out['sofc_official_10'] = {**gate([r['f1_10'] for r in official],
                                      PUBLISHED['sofc_official_10']),
                               'dropped_article': TEN_ARTICLES_DROP, 'runs': official}
    cv_runs = []
    for offset in CV_OFFSETS:
        path = results_dir / CV_FILE.format(offset=offset)
        if path.exists():
            data = json.loads(path.read_text())
            check_cv_file(data, offset)
            cv_runs += data['runs']
    out['sofc_cv'] = (gate([r['f1'] for r in cv_runs], PUBLISHED['sofc_cv'])
                      if len(cv_runs) == 15 else {'missing': f'{len(cv_runs)} of 15 runs'})
    if out['sofc_cv'].get('ours'):
        folds = json.loads(FOLDS_FILE.read_text())
        out['sofc_cv']['folds_sha256'] = folds['sha256']
    runs = []
    for c in range(1, 6):
        path = results_dir / MULMS_FILE.format(config=c)
        if path.exists():
            data = json.loads(path.read_text())
            check_mulms_file(data, c)
            runs += data['runs']
    out['mulms_nested'] = (gate([r['test']['micro_f1'] for r in runs],
                                PUBLISHED['mulms_nested'])
                           if len(runs) == 5 else {'missing': f'{len(runs)} of 5 runs'})
    if runs:
        out['mulms_nested']['erased_word_test_sentences'] = sorted(
            {r['erased_word_sentences']['test'] for r in runs})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="The SOTA plan's P0 gates")
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('data-files')
    sm = sub.add_parser('smoke', help='compare the two GPUs\' short MuLMS runs')
    sm.add_argument('a')
    sm.add_argument('b')
    g = sub.add_parser('gates')
    g.add_argument('--results', default=str(GPU_DIR))
    g.add_argument('--out', default=str(OUT))
    args = parser.parse_args(argv)
    if args.command == 'data-files':
        write_data_files()
        return None
    if args.command == 'smoke':
        differ = smoke_identical(json.loads(Path(args.a).read_text()),
                                 json.loads(Path(args.b).read_text()))
        print('identical but for run times and tags' if not differ else f'DIFFER: {differ}')
        return differ
    out = gates(args.results)
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    for name in ('sofc_official_10', 'sofc_cv', 'mulms_nested'):
        g = out[name]
        if 'missing' in g:
            print(f"{name}: {g['missing']}")
            continue
        print(f"{name}: ours {g['ours']['mean']:.2f} (SD {g['ours']['sd']:.2f}, n "
              f"{g['ours']['n']}) against {g['published']['mean']} (n {g['published']['n']}): "
              f"{g['welch']['gap']:+.2f} {g['welch']['ci95']}, {g['verdict']}"
              f"{', borderline' if g['borderline'] else ''}"
              f"{', our SD is large' if g['sd_warning'] else ''}")
    print(f'wrote {args.out}')
    return out


if __name__ == '__main__':
    main()
