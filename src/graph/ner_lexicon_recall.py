"""Recall on the unseen test names by lexicon membership (week 15, post hoc; ROADMAP §4, the
week-15 analysis note): whether the harvested names help the tagger with the names the lexicon
holds, or shift it toward tagging more spans.

Inputs: the rebuilt lexicons of batch w15a (ner_baseline.py --students 0 --save-names, pulled by
`kaggle_jobs.py collect` to results/gpu_names/ner_names_*.json, which git ignores: the names
are never committed) and the saved test predictions of the systems in results/gpu. Output:
results/gpu/ner_lexicon_recall.json, aggregate counts only.

For each held-out unit (a MuLMS subfield at the decision seeds 300 to 900; a MaterioMiner
paper per granularity at seeds 0 to 900) the lexicon is the set of (type, words) the unit's
teachers harvested. A gold test entity is unseen when its (type, words) is no entity of the
training sentences (ner_mentions.py's rule); it is `in` the lexicon when the lexicon holds its
(type, words), else `out`. Per system and unit, pooled over the unit's seeds: recall on the
unseen entities in and out of the lexicon, and among the predicted entities whose (type, words)
is unseen, the share in the lexicon and the precision of each part. Paired over the units: a
method's recall gain over a reference on each part and the difference of the two gains (in
minus out), with 95% Student-t intervals (metrics.mean_ci).

Checks before any number is used: per unit, the rebuilt teachers' test F1 must equal the
stored teachers' of the names runs at every seed, and the rebuilt coverage counts the stored
ones (`self_training.names`); a unit that differs is listed in `checks` and left out.
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import metrics
import ner_mentions as nm
from ner_ensemble import tags_to_spans
from ner_seedmeans import DECISION_OFFSETS, LOPO, WEEK5_OFFSETS

NAMES_DIR = Path(__file__).resolve().parents[2] / 'results' / 'gpu_names'
OUTPUT = nm.RESULTS_DIR / 'ner_lexicon_recall.json'
PARTS = ('in', 'out')
MULMS_SYSTEMS = {'crf': 'CRF head', 'crf_mr': 'MR3', 'crf_mr6': 'MR6',
                 'crf_mrm_topic6': 'MR6 + names', 'crf_mrm_mixed6': 'MR6 + names (mixed corpus)'}
PAPER_SYSTEMS = {'crf': 'CRF head', 'crf_mr': 'MR3', 'crf_mr6': 'MR6', 'crf_mrm6': 'MR6 + names'}
COMPARISONS = {'names_vs_mr6': ('names', 'crf_mr6'), 'names_vs_crf': ('names', 'crf'),
               'mr6_vs_crf': ('crf_mr6', 'crf'), 'mr3_vs_crf': ('crf_mr', 'crf'),
               'mixed_vs_mr6': ('crf_mrm_mixed6', 'crf_mr6')}
SETTINGS = {'mulms': ('mulms', nm.SUBFIELDS, MULMS_SYSTEMS, 'crf_mrm_topic6', DECISION_OFFSETS,
                      'ner_names_mulms_leave_one_subfield_out_crf_mrm_topic6_*.json'),
            'fine': ('fine', LOPO, PAPER_SYSTEMS, 'crf_mrm6', WEEK5_OFFSETS,
                     'ner_names_fine_leave_one_paper_out_crf_mrm6.json'),
            'coarse': ('coarse', LOPO, PAPER_SYSTEMS, 'crf_mrm6', WEEK5_OFFSETS,
                       'ner_names_coarse_leave_one_paper_out_crf_mrm6.json')}


def load_stores(directory, pattern):
    """The runs of the names files matching `pattern` in `directory`, merged: {run_id: record}."""
    runs = {}
    for path in sorted(Path(directory).glob(pattern)):
        with open(path) as f:
            runs.update(json.load(f).get('runs', {}))
    return runs


def lexicon_of(record):
    """{(type, words)} of a names record ([type, 'w1 w2', count] rows)."""
    return {(kind, tuple(words.split(' '))) for kind, words, _ in record['names']}


def sentence_counts(words, gold_tags, pred_tags, forms, lexicon):
    """The Counter of one test sentence: ('gold', part), ('correct', part) over the unseen gold
    entities, ('pred', part), ('pred_correct', part) over the predicted entities whose
    (type, words) is unseen; part 'in' when the lexicon holds the (type, words), else 'out'."""
    counts = Counter()
    gold = set(tags_to_spans(gold_tags)[0])
    pred = set(tags_to_spans(pred_tags)[0])
    for start, end, kind in gold:
        if nm.kind_of(words[start:end], kind, forms) != 'unseen':
            continue
        part = 'in' if (kind, tuple(words[start:end])) in lexicon else 'out'
        counts['gold', part] += 1
        counts['correct', part] += (start, end, kind) in pred
    for start, end, kind in pred:
        if nm.kind_of(words[start:end], kind, forms) != 'unseen':
            continue
        part = 'in' if (kind, tuple(words[start:end])) in lexicon else 'out'
        counts['pred', part] += 1
        counts['pred_correct', part] += (start, end, kind) in gold
    return counts


def membership_counts(results, data, granularity, lexicons):
    """{run_id: Counter} of a parsed results file over the units in `lexicons`."""
    out = {}
    for run in results['runs']:
        if run['run_id'] not in lexicons:
            continue
        run_data, train, test = nm.run_split(results, run, data, granularity)
        forms = nm.entity_forms(run_data, train)
        counts = Counter()
        for i, gold_tags, pred_tags in zip(test, run['test_gold'], run['test_pred'],
                                           strict=True):
            counts.update(sentence_counts(run_data[i][0], gold_tags.split(), pred_tags.split(),
                                          forms, lexicons[run['run_id']]))
        out[run['run_id']] = counts
    return out


def unit_scores(counts):
    """Recall per part, precision per part among the unseen predictions, the lexicon's share of
    the unseen gold entities and of the unseen predictions, and the counts (percentages)."""
    def rate(num, den):
        return round(100 * num / den, 2) if den else None
    gold = {p: counts['gold', p] for p in PARTS}
    pred = {p: counts['pred', p] for p in PARTS}
    return {'recall_in': rate(counts['correct', 'in'], gold['in']),
            'recall_out': rate(counts['correct', 'out'], gold['out']),
            'precision_in': rate(counts['pred_correct', 'in'], pred['in']),
            'precision_out': rate(counts['pred_correct', 'out'], pred['out']),
            'gold_share_in': rate(gold['in'], gold['in'] + gold['out']),
            'pred_share_in': rate(pred['in'], pred['in'] + pred['out']),
            'gold': gold, 'correct': {p: counts['correct', p] for p in PARTS},
            'pred': pred, 'pred_correct': {p: counts['pred_correct', p] for p in PARTS}}


def system_units(files, data, granularity, lexicons):
    """{run_id: Counter} summed over a family's files (its seeds)."""
    total = {}
    for f in files:
        for run_id, counts in membership_counts(f, data, granularity, lexicons).items():
            total.setdefault(run_id, Counter()).update(counts)
    return total


def compare(method, reference, units):
    """Per unit the method's recall gain over the reference on each part and their difference
    (in minus out), with the mean paired differences and 95% CIs over the units."""
    rows = {}
    for u in units:
        m, r = unit_scores(method[u]), unit_scores(reference[u])
        if None in (m['recall_in'], m['recall_out'], r['recall_in'], r['recall_out']):
            continue
        gain_in = round(m['recall_in'] - r['recall_in'], 2)
        gain_out = round(m['recall_out'] - r['recall_out'], 2)
        rows[u] = {'gain_in': gain_in, 'gain_out': gain_out,
                   'in_minus_out': round(gain_in - gain_out, 2)}

    def interval(key):
        values = [rows[u][key] for u in rows]
        ci = metrics.mean_ci(values)
        return {'mean_diff': round(ci['mean'], 2), 'ci95': [round(x, 2) for x in ci['ci']],
                'n_units': len(values), 'n_positive': sum(v > 0 for v in values)}
    return {'per_unit': rows, **{k: interval(k) for k in ('gain_in', 'gain_out', 'in_minus_out')}}


def checks(stores, files):
    """Per unit, whether the rebuilt teachers' test F1 and the rebuilt coverage counts equal
    the stored ones of every names run (the files of the names system at the setting's seeds)."""
    out = {}
    for run_id, record in stores.items():
        rebuilt = [t['test_f1'] for t in record['teachers']]
        teachers, coverage = [], []
        for f in files or []:
            for run in f['runs']:
                if run['run_id'] != run_id:
                    continue
                stored = run['self_training']
                teachers.append([t['test_f1'] for t in stored['teachers']] == rebuilt)
                coverage.append(stored.get('names') == record['coverage'])
        out[run_id] = {'teachers': bool(teachers) and all(teachers),
                       'coverage': bool(coverage) and all(coverage),
                       'runs_checked': len(teachers), 'rebuilt_teachers_f1': rebuilt,
                       'n_distinct': len(record['names'])}
    return out


def analyse_setting(directory, names_dir, name):
    """One setting (SETTINGS): the checks, the pooled and per-unit scores of every system, and
    the comparisons; None without the names files."""
    granularity, protocol, systems, names_system, offsets, pattern = SETTINGS[name]
    stores = load_stores(names_dir, pattern)
    if not stores:
        return None
    data = nm.corpus(granularity)
    names_files = nm.family_files(directory, granularity, protocol, names_system, offsets)
    verdicts = checks(stores, names_files)
    units = sorted(u for u, v in verdicts.items() if v['teachers'] and v['coverage'])
    lexicons = {u: lexicon_of(stores[u]) for u in units}
    counts = {}
    for system in systems:
        files = nm.family_files(directory, granularity, protocol, system, offsets)
        if files:
            counts[system] = system_units(files, data, granularity, lexicons)
    pooled = {s: unit_scores(sum((c[u] for u in units), Counter())) for s, c in counts.items()}
    per_unit = {s: {u: unit_scores(c[u]) for u in units} for s, c in counts.items()}
    comparisons = {}
    for key, (method, reference) in COMPARISONS.items():
        method = names_system if method == 'names' else method
        if method in counts and reference in counts:
            comparisons[key] = compare(counts[method], counts[reference], units)
    return {'granularity': granularity, 'protocol': protocol, 'seeds': list(offsets),
            'systems': {s: systems[s] for s in counts}, 'units': units,
            'excluded': sorted(set(stores) - set(units)), 'checks': verdicts,
            'pooled': pooled, 'per_unit': per_unit, 'comparisons': comparisons}


def analyse(directory=nm.RESULTS_DIR, names_dir=NAMES_DIR):
    out = {name: analyse_setting(directory, names_dir, name) for name in SETTINGS}
    return {k: v for k, v in out.items() if v is not None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--results', default=str(nm.RESULTS_DIR))
    parser.add_argument('--names', default=str(NAMES_DIR))
    parser.add_argument('--out', default=str(OUTPUT))
    args = parser.parse_args(argv)
    out = analyse(args.results, args.names)
    if not out:
        raise SystemExit(f'no names files in {args.names} (kaggle_jobs.py collect after w15a)')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    for name, part in out.items():
        print(f"{name}: units {part['units']} (excluded {part['excluded']})")
        for key, c in part['comparisons'].items():
            print(f"  {key}: gain in {c['gain_in']['mean_diff']:+.2f} {c['gain_in']['ci95']}, "
                  f"out {c['gain_out']['mean_diff']:+.2f} {c['gain_out']['ci95']}, in - out "
                  f"{c['in_minus_out']['mean_diff']:+.2f} {c['in_minus_out']['ci95']}")
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
