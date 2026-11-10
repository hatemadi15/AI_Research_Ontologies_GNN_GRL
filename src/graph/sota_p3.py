"""
sota_p3.py - the SOTA plan's P3: reported on the held-out-document protocols (docs/ROADMAP.md
§4, the P3 protocol; docs/SOTA_PLAN.md §4). No decision and no claim.

P2's selected systems run on papers A and B's held-out-document protocols, at the stored CRF
head's seed offsets (0-900), so that every run pairs with a stored run by fold and offset:
  - MaterioMiner coarse-grained, leave-one-paper-out: A3 with A6 (batch p3a);
  - SOFC-Exp, 5 document folds: the CV system, A6 with C1 (batch p3a);
  - MaterioMiner fine-grained, leave-one-paper-out: C1, plain multi-task training, whose runs
    are stored (paper B's crf_mt files); nothing runs.

Usage:
  python src/graph/sota_p3.py jobs              # results/sota/p3_jobs_p3a.json
  python src/graph/sota_p3.py checks <pulled>    # smoke4's repeat
  python src/graph/sota_p3.py report            # results/gpu/sota_p3.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'graph'))

import sota_p1  # noqa: E402
import sota_p2  # noqa: E402

SOTA_DIR = ROOT / 'results' / 'sota'
GPU_DIR = ROOT / 'results' / 'gpu'
P3_DIR = ROOT / 'results' / 'gpu_p3'
JOBS_FILE = SOTA_DIR / 'p3_jobs_p3a.json'
OUT = GPU_DIR / 'sota_p3.json'
THREADS = 2
OFFSETS = tuple(range(0, 1000, 100))
SMOKE_JOB = sota_p2.SMOKE_JOB

# the protocols that run: the system's flags, its cell, the granularity of its scored file, the
# minutes per run (P2's measured runs) and the stored CRF head's file at an offset
RUN = {
    'coarse_lopo': {'flags': ('--granularity', 'coarse', '--crf', '--leave-one-paper-out',
                              '--epochs', '40', '--patience', '5', '--reinit-top', '1'),
                    'cell': 'a3a6', 'granularity': 'coarse', 'minutes': 1.9, 'runs': 4,
                    'reference': 'ner_results_coarse_leave_one_paper_out_crf_m{off}.json'},
    'sofc_folds': {'flags': ('--granularity', 'sofc_both', '--multi-task', '--crf',
                             '--document-folds', '5', '--reinit-top', '1'),
                   'cell': 'a6c1', 'granularity': 'sofc', 'minutes': 5.74, 'runs': 5,
                   'reference': 'ner_results_sofc_document_folds_crf_m{off}.json'},
}
# the protocol whose selected system's runs are stored (paper B's plain multi-task runs)
STORED = {
    'fine_lopo': {'selected': 'ner_results_fine_leave_one_paper_out_crf_mt{m}.json',
                  'reference': 'ner_results_fine_leave_one_paper_out_crf_m{off}.json'},
}


def job(protocol, offset):
    spec = RUN[protocol]
    return [*spec['flags'], '--seeds', '0', *(('--seed-offset', str(offset)) if offset else ()),
            '--save-predictions', '--tag', f"p3{spec['cell']}m{offset}", '--threads', str(THREADS)]


def jobs():
    """[(minutes, job)] of batch p3a: smoke4's repeat first (minutes 0), then every run."""
    out = [(0.0, SMOKE_JOB)]
    for protocol, spec in RUN.items():
        for off in OFFSETS:
            out.append((spec['runs'] * spec['minutes'], job(protocol, off)))
    return out


def write_jobs():
    entries = [{'minutes': round(m, 1), 'job': j} for m, j in jobs()]
    JOBS_FILE.write_text(json.dumps(entries, indent=1) + '\n')
    return entries


def checks(pulled):
    """{check: problems} of a pulled p3a: smoke4's repeat, environment included."""
    found = {p.name: p for p in Path(pulled).rglob('ner_results_*.json')}
    smoke = found.get('ner_results_fine_random_split_vary_crf_smokebu.json')
    if smoke is None:
        return {'smoke': ['smokebu is missing']}
    a = sota_p1.without(json.loads(smoke.read_text()), 'smokebu')
    b = sota_p1.without(json.loads(sota_p1.SMOKE_REFERENCE.read_text()), 'smokebn')
    return {'smoke': sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))}


def paired_values(selected_runs, reference_runs, name):
    """{(run_id, offset): (S F1, reference F1)}, each pair's test sentences and gold equal."""
    out = {}
    for key, run in selected_runs.items():
        ref = reference_runs.get(key)
        if ref is None:
            raise ValueError(f'{name}: no stored CRF head run for {key}')
        for field in ('test_index', 'test_gold'):
            if field in ref and run.get(field) != ref[field]:
                raise ValueError(f'{name}: {key} {field} differs from the stored CRF head')
        out[key] = (run['f1'], ref['f1'])
    return out


def runs_at(path, offset):
    data = json.loads(Path(path).read_text())
    return data, {(r['run_id'], offset): r for r in data['runs']}


def protocol_record(pairs):
    """The reported numbers of one protocol: both means with their intervals, the paired
    difference and the per-fold means."""
    keys = sorted(pairs)
    s = {k: pairs[k][0] for k in keys}
    r = {k: pairs[k][1] for k in keys}

    def mean_ci(values):
        mean, ci = sota_p2.t_interval(values)
        return {'mean': round(mean, 2), 'ci95': [round(ci[0], 2), round(ci[1], 2)],
                'n': len(values)}
    folds = sorted({k[0] for k in keys})
    return {'selected': mean_ci(list(s.values())), 'crf_head': mean_ci(list(r.values())),
            'paired_difference': sota_p2.paired(s, r),
            'per_fold': {f: {'selected': round(statistics.fmean(s[k] for k in keys if k[0] == f),
                                               2),
                             'crf_head': round(statistics.fmean(r[k] for k in keys if k[0] == f),
                                               2)} for f in folds}}


def report(p3_dir=P3_DIR, gpu_dir=GPU_DIR):
    """The record of results/gpu/sota_p3.json (reported only)."""
    out = {'note': ("P3 (ROADMAP §4, the P3 protocol): P2's selected systems on the held-out-"
                    'document protocols, paired by fold and seed offset with the stored CRF '
                    'head; reported, no decision and no claim')}
    for protocol, spec in RUN.items():
        pairs = {}
        for off in OFFSETS:
            j = job(protocol, off)
            name = sota_p2.tagger_file(j, spec['granularity'])
            data, selected = runs_at(Path(p3_dir) / name, off)
            if data.get('screen'):
                raise ValueError(f'{name}: a screen file')
            sota_p1.check_settings(data, j, name)
            for run in selected.values():
                sota_p1.check_environment(run, name)
                if run.get('seed') != off:
                    raise ValueError(f"{name}: seed {run.get('seed')}, not {off}")
            if len(selected) != spec['runs']:
                raise ValueError(f'{name}: {len(selected)} runs, not {spec["runs"]}')
            _, reference = runs_at(Path(gpu_dir) / spec['reference'].format(off=off), off)
            pairs.update(paired_values(selected, reference, name))
        out[protocol] = {'system': spec['cell'], **protocol_record(pairs)}
    for protocol, spec in STORED.items():
        pairs = {}
        for off in OFFSETS:
            name = spec['selected'].format(m=f'_m{off}' if off else '')
            _, selected = runs_at(Path(gpu_dir) / name, off)
            _, reference = runs_at(Path(gpu_dir) / spec['reference'].format(off=off), off)
            pairs.update(paired_values(selected, reference, name))
        out[protocol] = {'system': 'c1 (stored plain multi-task runs)', **protocol_record(pairs)}
    out['mulms'] = 'not run: it waits for code that scores the nested system on held-out subfields'
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="The SOTA plan's P3, reported")
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('jobs')
    p = sub.add_parser('checks')
    p.add_argument('pulled')
    p = sub.add_parser('report')
    p.add_argument('--out', default=str(OUT))
    args = parser.parse_args(argv)
    if args.command == 'jobs':
        entries = write_jobs()
        print(f'{len(entries)} jobs, about {sum(e["minutes"] for e in entries) / 120:.1f} '
              f'session-hours on two lanes, to {JOBS_FILE}')
    elif args.command == 'checks':
        result = checks(args.pulled)
        for check, problems in result.items():
            print(f"{check}: {'passed' if not problems else 'FAILED: ' + '; '.join(problems)}")
        if any(result.values()):
            raise SystemExit('a check failed: the batch is void (ROADMAP §4, P3)')
    else:
        record = report()
        Path(args.out).write_text(json.dumps(record, indent=1) + '\n')
        for protocol in ('fine_lopo', 'coarse_lopo', 'sofc_folds'):
            r = record[protocol]
            d = r['paired_difference']
            print(f"{protocol} ({r['system']}): {r['selected']['mean']} against the CRF head's "
                  f"{r['crf_head']['mean']}: {d['estimate']:+.2f} {d['ci95']} (n {d['n']})")
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
