"""
paper_tables.py - the tables of the paper plan (docs/PAPER.md) from the results files

No training: reads the committed comparisons and analyses and writes Markdown tables to
docs/tables/:
  table1_published_split.md (C1, C3, C13): per published split (MaterioMiner's fine and
      coarse levels, SOFC-Exp's official split), plain MatSciBERT and the CRF head over their
      10 model seeds under every F1 average, mean [95% CI], next to the published number; the
      CRF head minus plain MatSciBERT, paired by seed; and on SOFC-Exp each method minus the
      CRF head (ner_week5_seed_analysis.json, published_split).
  table2_decisions.md (C5-C8): every pre-registered decision and screen on MaterioMiner,
      weeks 3 to 7 (the comparisons files): its design, the mean paired difference of micro
      F1 with its 95% CI, and whether the CI lies above 0.
  table3_strict.md (C4): default against strict IOB2 scoring on the published splits, for
      plain MatSciBERT, the CRF head and their difference.
  table4_unseen.md (C17, C18): per setting, the share of the gold test entities whose names
      are unseen in the training sentences, the CRF head's recall on seen and unseen names
      (ner_mentions.json), and its F1 under the four matching criteria (ner_matching.json).
  tableA1_all_comparisons.md: every comparison of the comparisons files, the reported and
      descriptive ones too.
A table whose results are missing is skipped (and named on stdout).

Usage:
  python src/graph/paper_tables.py [--results results] [--out docs/tables]
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# (week, machine, seeds per split, comparisons file under the results directory)
COMPARISONS = (('3', 'CPU', '3', 'ner_week3_comparisons.json'),
               ('4', 'CPU', '3', 'ner_week4_comparisons.json'),
               ('4', 'GPU', '3', 'gpu/ner_week4_comparisons.json'),
               ('5', 'GPU', '10', 'gpu/ner_week5_comparisons.json'),
               ('6', 'GPU', '3', 'gpu/ner_week6_screen.json'),
               ('6', 'GPU', '10', 'gpu/ner_week6_comparisons.json'),
               ('7', 'GPU', '3', 'gpu/ner_week7_screen.json'),
               ('7', 'GPU', '10', 'gpu/ner_week7_comparisons.json'),
               ('9', 'GPU', '10', 'gpu/ner_week9_comparisons.json'))
DECIDED = ('decision', 'screen')
PROTOCOL_NAMES = {'random_split_vary': 'random splits', 'leave_one_paper_out': 'unseen papers'}
AVERAGES = (('micro', 'micro'), ('macro', 'macro'), ('weighted', 'weighted'),
            ('micro_strict', 'strict micro'))
SPLIT_NAMES = {'fine': 'MaterioMiner, fine-grained', 'coarse': 'MaterioMiner, coarse-grained',
               'sofc': 'SOFC-Exp', 'sofc_slot': 'SOFC-Exp, slot types'}
# the system whose published score a split's entry holds: plain MatSciBERT on MaterioMiner
# (Durmaz et al. 2024), MatSciBERT-CRF on SOFC-Exp (Gupta et al. 2022)
PUBLISHED_SYSTEM = {'fine': 'baseline', 'coarse': 'baseline', 'sofc': 'crf', 'sofc_slot': 'crf'}
UNSEEN_SETTINGS = (('fine_random_split_vary', 'fine, random splits'),
                   ('fine_leave_one_paper_out', 'fine, unseen papers'),
                   ('coarse_random_split_vary', 'coarse, random splits'),
                   ('coarse_leave_one_paper_out', 'coarse, unseen papers'),
                   ('sofc_random_split_vary', 'SOFC-Exp, random splits'),
                   ('sofc_document_folds', 'SOFC-Exp, article folds'),
                   ('sofc_published_split', 'SOFC-Exp, official split'),
                   ('sofc_slot_published_split', 'SOFC-Exp slot types, official split'))
CRITERIA = ('strict', 'exact', 'type', 'overlap')


def number(x, signed=False):
    """x with 2 decimals and a typographic minus; with signed, a + on positive numbers."""
    return (f'{x:+.2f}' if signed else f'{x:.2f}').replace('-', '\u2212')


def interval(entry, key='mean', signed=False):
    """'mean [lo, hi]' of a spread entry ({key, 'ci95'}), signed for differences."""
    lo, hi = entry['ci95']
    return f"{number(entry[key], signed)} [{number(lo, signed)}, {number(hi, signed)}]"


def markdown(header, rows):
    """A Markdown table."""
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines) + '\n'


def table1(published):
    """Table 1 from the published_split analysis {granularity: entry}."""
    rows = []
    for g, entry in published.items():
        name = SPLIT_NAMES.get(g, g)
        reference = number(entry['published'])
        if 'published_macro' in entry:
            reference += f" (macro {number(entry['published_macro'])})"
        for system, label in (('baseline', 'plain MatSciBERT'), ('crf', 'CRF head')):
            rows.append([name, label, *(interval(entry[system][a]) for a, _ in AVERAGES),
                         reference if system == PUBLISHED_SYSTEM.get(g, 'baseline') else ''])
        rows.append([name, 'CRF head − plain', *(interval(entry['crf_minus_baseline'][a],
                                                          'mean_diff', signed=True)
                                                 for a, _ in AVERAGES), ''])
        for method, m in entry.get('methods', {}).items():
            rows.append([name, f'`{method}` − CRF head',
                         *(interval(m['minus_crf'][a], 'mean_diff', signed=True)
                           for a, _ in AVERAGES), ''])
    header = ['split', 'system', *(label for _, label in AVERAGES), 'published']
    return ('Table 1. Published splits, 10 model seeds: mean F1 [95% CI]; differences '
            'paired by seed. The published score is plain MatSciBERT\'s on MaterioMiner '
            '(Durmaz et al. 2024) and MatSciBERT-CRF\'s on SOFC-Exp (Gupta et al. 2022).\n\n'
            + markdown(header, rows))


def comparison_rows(files, roles=None):
    """[(week, machine, seeds, entry)] of the comparisons files, in their order, only the
    entries whose role is in `roles` (all if None)."""
    rows = []
    for week, machine, seeds, data in files:
        for entry in data.values():
            if roles is None or entry.get('role') in roles:
                rows.append((week, machine, seeds, entry))
    return rows


def table2(rows, title='Table 2. Pre-registered decisions and screens on MaterioMiner'):
    """Table 2 from comparison_rows: one row per comparison."""
    out = []
    for week, machine, seeds, e in rows:
        decided = e.get('role') in DECIDED
        lo = e['micro']['ci95'][0]
        out.append([week, machine, seeds, e['granularity'],
                    PROTOCOL_NAMES.get(e['protocol'], e['protocol']),
                    f"`{e['variant']}` vs `{e['reference']}`", e.get('role', ''),
                    e['n_pairs'], interval(e['micro'], 'mean_diff', signed=True),
                    ('passes' if lo > 0 else 'no') if decided else ''])
    header = ['week', 'machine', 'seeds', 'granularity', 'protocol', 'comparison', 'role',
              'pairs', 'Δ micro F1 [95% CI]', 'CI above 0']
    return (f'{title}: mean paired difference of micro F1 over splits or held-out papers, '
            f'each side the mean over its seeds.\n\n' + markdown(header, out))


def table3(published):
    """Table 3: default against strict IOB2 scoring on the published splits."""
    rows = []
    for g, entry in published.items():
        for system, label in (('baseline', 'plain MatSciBERT'), ('crf', 'CRF head')):
            rows.append([SPLIT_NAMES.get(g, g), label, interval(entry[system]['micro']),
                         interval(entry[system]['micro_strict'])])
        d = entry['crf_minus_baseline']
        rows.append([SPLIT_NAMES.get(g, g), 'CRF head − plain',
                     interval(d['micro'], 'mean_diff', signed=True),
                     interval(d['micro_strict'], 'mean_diff', signed=True)])
    return ('Table 3. Default (conlleval) against strict IOB2 scoring, published splits, 10 '
            'model seeds: micro F1 [95% CI].\n\n'
            + markdown(['split', 'system', 'default', 'strict IOB2'], rows))


def table4(mentions, matching):
    """Table 4: unseen names and the matching criteria, CRF head, per setting."""
    rows = []
    for key, label in UNSEEN_SETTINGS:
        names = mentions.get('systems', {}).get(key, {}).get('crf')
        scores = matching.get('systems', {}).get(key, {}).get('crf')
        if not names or not scores:
            continue
        rows.append([label, f"{100 * names['unseen']['share']:.1f}",
                     f"{names['seen']['recall']:.1f}", f"{names['unseen']['recall']:.1f}",
                     *(number(scores[c]['mean']) for c in CRITERIA)])
    header = ['setting', 'unseen names (%)', 'recall, seen', 'recall, unseen',
              *(f'F1, {c}' for c in CRITERIA)]
    return ('Table 4. The CRF head (10 model seeds): the share of gold test entities whose '
            'words are no entity of their type in the training sentences, recall on seen and '
            'unseen names (pooled), and F1 under the matching criteria of SemEval-2013 Task '
            '9.1 (mean over splits, papers or seeds).\n\n' + markdown(header, rows))


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(description='The tables of the paper plan')
    parser.add_argument('--results', default=str(ROOT / 'results'))
    parser.add_argument('--out', default=str(ROOT / 'docs' / 'tables'))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    written = []

    def write(name, text):
        with open(os.path.join(args.out, name), 'w') as f:
            f.write(text)
        written.append(name)

    analysis = load(os.path.join(args.results, 'gpu', 'ner_week5_seed_analysis.json')) or {}
    published = analysis.get('published_split')
    if published:
        write('table1_published_split.md', table1(published))
        write('table3_strict.md', table3(published))
    else:
        print('tables 1 and 3: no published-split analysis')
    files = [(w, m, s, load(os.path.join(args.results, name)))
             for w, m, s, name in COMPARISONS]
    files = [f for f in files if f[3]]
    if files:
        write('table2_decisions.md', table2(comparison_rows(files, DECIDED)))
        write('tableA1_all_comparisons.md',
              table2(comparison_rows(files), 'Table A1. Every comparison on MaterioMiner'))
    else:
        print('table 2: no comparisons files')
    mentions = load(os.path.join(args.results, 'gpu', 'ner_mentions.json'))
    matching = load(os.path.join(args.results, 'gpu', 'ner_matching.json'))
    if mentions and matching:
        write('table4_unseen.md', table4(mentions, matching))
    else:
        print('table 4: no mention or matching analysis')
    print(f"wrote {', '.join(written) or 'nothing'} to {args.out}")
    return written


if __name__ == '__main__':
    main()
