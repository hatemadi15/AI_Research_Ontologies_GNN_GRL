"""
paper_figures.py - the figures of the paper plan (docs/PAPER.md) from the GPU results

No training: reads the 10-seed GPU results (results/gpu) and writes PNG files.
  fig1_random_vs_unseen.png (C2): per system with 10 seeds on both protocols, its mean micro
      F1 over the 5 random splits and over the 4 held-out papers, with 95% CIs, joined by a
      line, per granularity. A model scores the mean of its 10 runs per split, a vote the
      mean of its votes over the 3 seed triples. A third panel (week 8, part 1;
      ner_week5_seed_analysis.json, sofc_splits) puts SOFC-Exp's CRF head and its votes on
      the same axis: 5 random sentence splits against 5 folds of 9 held-out articles.
  fig2_seeds.png (C8): (a) the SD of the CRF head's micro F1 over its 10 seeds, per split
      and granularity; (b) the 120 three-seed mean differences of multi-task training
      against the CRF head on the random splits (fine-grained), against the 10-seed
      difference; (c) the same for document context on the held-out papers (fine-grained),
      with the week-6 screen's triple (seeds 0, 100 and 200) marked.
  fig3_types.png (C9): per coarse-grained type, the F1 of the reported system's votes
      (the multi-task vote) on the random splits against the type's test support per split.
  fig4_threshold.png (C15): the CRF head's bias grid (week 7, part 2a; the bias_test counts
      of the crf_bias runs): (a, b) per held-out paper, its mean micro F1 over the 10 seeds
      at each bias D, fine- and coarse-grained; (c) pooled over papers (articles) and seeds,
      the share of the entities added between two grid points that are right, against F1's
      break-even rate (F1/2 at D = 0), for both granularities and SOFC-Exp.
  fig5_mentions.png (C17; ner_mentions.json): (a) per setting (MaterioMiner's random splits,
      published split and held-out papers per granularity, SOFC-Exp's official split), the
      shares of the gold test entities whose words are an entity of the same type in the
      training sentences (seen), only of another type, or none (unseen); (b) the CRF head's
      recall on the seen and the unseen ones; (c) on the held-out papers, fine-grained, each
      method's F1 minus the CRF head's on seen and on unseen entities, with 95% CIs over the
      4 papers (votes against the CRF head's vote).
  fig6_gains.png (C5-C8, C19): on the 4 held-out papers, per granularity, each method's
      mean paired micro-F1 difference from the CRF head with its 95% CI over the papers:
      the 10-seed comparison where there is one (weeks 5-7), else the 3-seed screen (hollow
      markers); and a third training paper, the CRF head trained on three papers minus on
      two (paper_count, seeds 0-200).
A figure whose results are missing is skipped (and named on stdout).

Usage:
  python src/graph/paper_figures.py [--results results/gpu] [--out docs/figures]
"""

import argparse
import itertools
import json
import os
import statistics

import metrics
from ner_seedmeans import GRANULARITIES, LOPO, WEEK5_SYSTEMS, file_names, system_scores

RANDOM = 'random_split_vary'
ANALYSIS = 'ner_week5_seed_analysis.json'
# Figure 1: the systems with 10 seeds on both protocols (models, then their votes)
FIG1_SYSTEMS = {'crf': 'CRF head', 'crf_mt': 'multi-task', 'crf_ls': 'label semantics',
                'crf_st3': 'self-training', 'crf_ens3': 'CRF vote',
                'crf_mt_ens3': 'multi-task vote', 'crf_ls_ens3': 'label-semantics vote',
                'crf_st3_ens3': 'self-training vote'}
SEED_OFFSETS = tuple(range(0, 1000, 100))
# Figure 6: the methods against the CRF head on the held-out papers, in this order; the
# 10-seed comparisons first, the screens if a method has no 10-seed comparison
FIG6_METHODS = {'crf_ens3': 'seed vote', 'crf_mt': 'multi-task', 'crf_st3': 'self-training',
                'crf_ls': 'label semantics', 'crf_deberta': 'DeBERTa-v3-large',
                'crf_ctx': 'document context', 'crf_mt_st3': 'multi-task self-training',
                'crf_mr': 'mention replacement', 'crf_dc': 'consistency rule',
                'crf_cf': 'coarse-to-fine completion', 'crf_eb': 'recall calibration',
                'crf_mt_mr': 'multi-task + mention replacement'}
TEN_SEED_COMPARISONS = ('ner_week5_comparisons.json', 'ner_week6_comparisons.json',
                        'ner_week7_comparisons.json', 'ner_week9_comparisons.json')
SCREENS = ('ner_week6_screen.json', 'ner_week7_screen.json')
# Figure 4: the bias grid of the CRF head (week 7, part 2a)
SOFC_BIAS = 'ner_results_sofc_published_split_crf_bias.json'
SHORT_PAPER = {'10.1007_s11340-021-00758-x': 's11340', '10.1038_ncomms7164': 'ncomms7164',
               '10.1038_s41467-020-19071-7': 's41467', '10.3390_met10111458': 'met10111458'}
SCREEN_TRIPLE = (0, 1, 2)                 # seed offsets 0, 100 and 200 (the week-6 screen)
# Figure 5: seen and unseen test entities (ner_mentions.py)
MENTIONS = 'ner_mentions.json'
SCREEN_SEEDS = (0, 100, 200)
MENTION_SETTINGS = {'fine_random_split_vary': 'fine, random splits',
                    'fine_published_split': 'fine, published split',
                    'fine_leave_one_paper_out': 'fine, unseen papers',
                    'coarse_random_split_vary': 'coarse, random splits',
                    'coarse_published_split': 'coarse, published split',
                    'coarse_leave_one_paper_out': 'coarse, unseen papers',
                    'sofc_random_split_vary': 'SOFC-Exp, random splits',
                    'sofc_document_folds': 'SOFC-Exp, article folds',
                    'sofc_published_split': 'SOFC-Exp, official split',
                    'sofc_slot_published_split': 'SOFC-Exp slots, official split'}
MENTION_METHODS = {'crf_mt': 'multi-task', 'crf_mt_ens3': 'multi-task vote',
                   'crf_st3': 'self-training', 'crf_ls': 'label semantics',
                   'crf_ctx': 'document context', 'crf_ctx_ens3': 'context vote',
                   'crf_mt_st3': 'multi-task self-training', 'crf_mr': 'mention replacement',
                   'crf_mr_ens3': 'mention-replacement vote',
                   'crf_mt_mr': 'multi-task + mention replacement',
                   'crf_mt_mr_ens3': 'multi-task + mention-replacement vote',
                   'crf_mt_st3_ens3': 'multi-task self-training vote'}


def load(directory, names):
    """The parsed results files, or None if one is missing."""
    paths = [os.path.join(directory, n) for n in names]
    if not all(os.path.exists(p) for p in paths):
        return None
    out = []
    for p in paths:
        with open(p) as f:
            out.append(json.load(f))
    return out


def split_scores(files):
    """Per split (run_id), the system's micro F1: the mean over its files (seeds or votes)."""
    return [s['micro'] for s in system_scores(files).values()]


def mean_ci(values):
    """(mean, lower, upper) with the 95% Student-t CI over the values."""
    lo, hi = metrics.mean_ci(values)['ci']
    return statistics.mean(values), lo, hi


def protocol_points(directory, systems=FIG1_SYSTEMS):
    """{granularity: [(label, (random mean, lo, hi), (unseen mean, lo, hi))]} for the systems
    with all their files on both protocols."""
    out = {}
    for g in GRANULARITIES:
        rows = []
        for system, label in systems.items():
            sides = []
            for protocol in (RANDOM, LOPO):
                files = load(directory, file_names(g, protocol, system, WEEK5_SYSTEMS))
                if files is None:
                    break
                sides.append(mean_ci(split_scores(files)))
            if len(sides) == 2:
                rows.append((label, *sides))
        out[g] = rows
    return out


def family_files(directory, granularity, protocol, stem):
    """The 10 files of a seed family (<stem>_m0 to _m900; crf_mt's first has no tag)."""
    names = [f"ner_results_{granularity}_{protocol}_"
             f"{stem if stem == 'crf_mt' and off == 0 else f'{stem}_m{off}'}.json"
             for off in SEED_OFFSETS]
    return load(directory, names)


def triple_differences(variant_files, reference_files):
    """(the mean paired difference of each of the 120 seed triples, which of them pass the
    pre-registered rule, the 10-seed difference): member k of both families has seed offset
    100 k, and a triple's systems score the mean of their 3 members per split."""
    def diffs(members):
        v = system_scores([variant_files[k] for k in members])
        r = system_scores([reference_files[k] for k in members])
        return [v[i]['micro'] - r[i]['micro'] for i in v if i in r]
    means, passes = [], []
    for members in itertools.combinations(range(len(SEED_OFFSETS)), 3):
        d = diffs(members)
        means.append(statistics.mean(d))
        passes.append(metrics.mean_ci(d)['ci'][0] > 0)
    return means, passes, statistics.mean(diffs(range(len(SEED_OFFSETS))))


def sofc_points(analysis):
    """Figure 1's SOFC-Exp rows from the analysis's sofc_splits: [(label, (random mean, lo,
    hi), (article folds mean, lo, hi))] for the CRF head and its votes; [] without both."""
    splits = analysis.get('sofc_splits', {})
    if not all(p in splits for p in (RANDOM, 'document_folds')):
        return []
    rows = []
    for label, pick in (('CRF head', lambda e: e['micro']),
                        ('CRF vote', lambda e: e['crf_votes']['micro'])):
        sides = [pick(splits[p]) for p in (RANDOM, 'document_folds')]
        rows.append((label, *((m['mean'], *m['ci95']) for m in sides)))
    return rows


def figure1(points, path, sofc=()):
    import matplotlib.pyplot as plt
    panels = [(points[g], f'{g}-grained') for g in GRANULARITIES]
    if sofc:
        panels.append((sofc, 'SOFC-Exp (entity types)'))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.2), sharey=False)
    n_rows = max(len(rows) for rows, _ in panels)
    for ax, (rows, title) in zip(axes, panels):
        rows = rows[::-1]                            # the first system on top
        top = n_rows - len(rows)                     # a shorter panel's rows at the top
        for k, (label, (x, xlo, xhi), (y, ylo, yhi)) in enumerate(rows, start=top):
            ax.plot([y, x], [k, k], color='lightgrey', lw=2, zorder=1)
            ax.errorbar(x, k, xerr=[[x - xlo], [xhi - x]], fmt='o', color='tab:blue', ms=5,
                        capsize=2, label='5 random splits' if k == 0 else None)
            ax.errorbar(y, k, xerr=[[y - ylo], [yhi - y]], fmt='s', color='tab:red', ms=5,
                        capsize=2, label='4 unseen papers' if k == 0 else None)
            ax.annotate(f'{x - y:.1f}', ((x + y) / 2, k), textcoords='offset points',
                        xytext=(0, 4), fontsize=7, ha='center', color='dimgrey')
        ax.set_yticks(range(top, n_rows), [r[0] for r in rows], fontsize=8)
        ax.set(xlim=(30, 90), ylim=(-0.5, n_rows - 0.5), title=title, xlabel='micro F1')
        ax.grid(alpha=0.3, axis='x')
    handles, labels = axes[0].get_legend_handles_labels()
    if sofc:
        labels = [label.replace('4 unseen papers', 'unseen documents (MaterioMiner: 4 papers; '
                                'SOFC-Exp: 5 folds of 9 articles)') for label in labels]
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8)
    fig.suptitle('The same systems on random splits and on unseen documents (10 seeds, 95% '
                 'CIs; the number is the gap)', fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure2(spread, mt, ctx, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    labels = []
    for k, (key, sds) in enumerate(spread.items()):
        ax.scatter([k] * len(sds), sds, color='tab:blue', s=18)
        labels.append(key.replace('_random_split_vary', '\nrandom').replace(
            '_leave_one_paper_out', '\nunseen'))
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set(ylabel='SD of micro F1 over 10 seeds', title='(a) the CRF head, per split')
    ax.grid(alpha=0.3)
    for ax, data, title, mark in ((axes[1], mt, '(b) multi-task vs CRF head,\nrandom splits, '
                                   'fine-grained', None),
                                  (axes[2], ctx, '(c) document context vs CRF head,\nunseen '
                                   'papers, fine-grained', SCREEN_TRIPLE)):
        if data is None:
            ax.set_axis_off()
            continue
        means, passes, ten = data
        ax.hist([[m for m, p in zip(means, passes) if p],
                 [m for m, p in zip(means, passes) if not p]], bins=20, stacked=True,
                color=['tab:green', 'darkgrey'],
                label=[f'triple passes ({sum(passes)} of {len(passes)})',
                       'triple does not pass'])
        ax.axvline(ten, color='black', lw=1.2, label=f'10 seeds ({ten:+.2f})')
        ax.axvline(0, color='grey', lw=0.6, ls=':')
        if mark is not None:
            screen = means[list(itertools.combinations(range(len(SEED_OFFSETS)), 3)).index(mark)]
            ax.axvline(screen, color='tab:red', lw=1.2, ls='--',
                       label=f'the screen\'s triple ({screen:+.2f})')
        ax.set(xlabel='mean paired difference of a seed triple', ylabel='triples', title=title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure3(types, path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    low = sorted((t for t, row in types.items() if row['f1'] < 50),
                 key=lambda t: types[t]['support_per_split'])
    for t, row in types.items():
        ax.scatter(row['support_per_split'], row['f1'], color='tab:blue', s=16)
        if t in low:                       # alternate above and below: close types overlap
            ax.annotate(t, (row['support_per_split'], row['f1']), textcoords='offset points',
                        xytext=(4, 4 if low.index(t) % 2 == 0 else -10), fontsize=7)
    ax.set_xscale('log')
    ax.set_xticks([2, 3, 5, 10, 20, 50])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel='test entities per split (log scale)', ylabel='F1',
           title='Coarse-grained types: F1 of the multi-task vote against support\n'
                 '(5 random splits, 3 votes each)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bias_counts(files):
    """{run_id: {D: [[correct, predicted, gold] per file]}} of crf_bias results files: per run
    (a held-out paper, or a seed on SOFC-Exp), the entity counts at every bias D, summed over
    the run's test papers or articles, one entry per file (seed)."""
    out = {}
    for f in files:
        for run in f['runs']:
            for d, papers in run['bias_test'].items():
                totals = [sum(c[i] for c in papers.values()) for i in range(3)]
                out.setdefault(run['run_id'], {}).setdefault(float(d), []).append(totals)
    return out


def threshold_data(directory):
    """(curves, steps): curves[g][paper] = [(D, mean F1 over seeds)] for MaterioMiner's
    granularities; steps[name] = ([(D midpoint, share of the added entities that are right)],
    break-even F1/2 at D = 0), pooled over runs and seeds, for both granularities and SOFC-Exp.
    Missing files leave their entries out."""
    curves, steps, pooled = {}, {}, {}
    for g in GRANULARITIES:
        files = load(directory, [f'ner_results_{g}_leave_one_paper_out_crf_bias_m{off}.json'
                                 for off in SEED_OFFSETS])
        if files:
            counts = bias_counts(files)
            curves[g] = {paper.removesuffix('.conll'): [
                (d, statistics.mean(200 * c / (p + n) if p + n else 0.0 for c, p, n in runs))
                for d, runs in sorted(by_d.items())] for paper, by_d in counts.items()}
            pooled[f'{g}-grained'] = counts
    files = load(directory, [SOFC_BIAS])
    if files:
        pooled['SOFC-Exp'] = bias_counts(files)
    for name, counts in pooled.items():
        total = {}
        for by_d in counts.values():
            for d, runs in by_d.items():
                acc = total.setdefault(d, [0, 0, 0])
                for c in runs:
                    for i in range(3):
                        acc[i] += c[i]
        grid = sorted(total)
        points = [((a + b) / 2, (total[b][0] - total[a][0]) / (total[b][1] - total[a][1]))
                  for a, b in zip(grid, grid[1:]) if total[b][1] != total[a][1]]
        c, p, n = total[0.0]
        steps[name] = (points, 100 * c / (p + n))
    return curves, steps


def figure4(curves, steps, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, g in zip(axes[:2], GRANULARITIES):
        if g not in curves:
            ax.set_axis_off()
            continue
        for paper, points in sorted(curves[g].items()):
            ax.plot([d for d, _ in points], [f for _, f in points], marker='o', ms=3,
                    label=SHORT_PAPER.get(paper, paper))
        ax.axvline(0, color='grey', lw=0.6, ls=':')
        ax.set(xlabel='bias D on every entity tag', ylabel='micro F1 (mean of 10 seeds)',
               title=f'({"ab"[GRANULARITIES.index(g)]}) {g}-grained, per held-out paper')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    ax = axes[2]
    for (name, (points, breakeven)), color in zip(steps.items(),
                                                   ('tab:blue', 'tab:orange', 'tab:green')):
        ax.plot([d for d, _ in points], [100 * r for _, r in points], marker='o', ms=3,
                color=color, label=name)
        ax.axhline(breakeven, color=color, lw=1, ls='--')
    ax.axvline(0, color='grey', lw=0.6, ls=':')
    ax.set(xlabel='bias D (between two grid points)',
           ylabel='net % of the added entities right',
           title='(c) what the bias adds, pooled\n(dashed: break-even F1/2 at D = 0)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def mention_data(mentions):
    """([(label, shares by kind, recalls of seen and unseen, recall of all)], [(label, seen
    difference, unseen difference)]) for Figure 5, from ner_mentions.json; a difference is
    (mean, CI low, CI high). Settings and methods without results are left out."""
    settings = []
    for key, label in MENTION_SETTINGS.items():
        crf = mentions.get('systems', {}).get(key, {}).get('crf')
        if crf:
            settings.append((label, {k: crf[k]['share'] for k in ('seen', 'seen_other',
                                                                   'unseen')},
                             (crf['seen']['recall'], crf['unseen']['recall']),
                             crf['all']['recall']))
    methods = []
    found = mentions.get('methods', {}).get(f'fine_{LOPO}', {})
    for system, label in MENTION_METHODS.items():
        if system in found:
            m = found[system]
            label += ' (screen, 3 seeds)' if m.get('seeds') == list(SCREEN_SEEDS) else ''
            methods.append((label, *((m[k]['mean_diff'], *m[k]['ci95'])
                                     for k in ('seen', 'unseen'))))
    return settings, methods


def figure5(settings, methods, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2),
                             gridspec_kw={'width_ratios': [1, 1, 1.1]})
    labels = [s[0] for s in settings][::-1]
    rows = settings[::-1]
    ax = axes[0]
    left = [0.0] * len(rows)
    for kind, color, name in (('seen', 'tab:blue', 'seen, same type'),
                              ('seen_other', 'tab:grey', 'seen, other type only'),
                              ('unseen', 'tab:red', 'unseen')):
        widths = [100 * r[1][kind] for r in rows]
        ax.barh(range(len(rows)), widths, left=left, color=color, label=name, height=0.6)
        left = [a + b for a, b in zip(left, widths)]
    for k, r in enumerate(rows):
        ax.annotate(f"{100 * r[1]['unseen']:.0f}%", (100, k), xytext=(-3, 0),
                    textcoords='offset points', ha='right', va='center', fontsize=7,
                    color='white')
    ax.set_yticks(range(len(rows)), labels, fontsize=8)
    ax.set(xlim=(0, 100), xlabel='% of the gold test entities',
           title='(a) are the test names in the training data?')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax = axes[1]
    for k, r in enumerate(rows):
        seen, unseen = r[2]
        ax.plot([unseen, seen], [k, k], color='lightgrey', lw=2, zorder=1)
        ax.plot(seen, k, 'o', color='tab:blue', ms=6,
                label='seen, same type' if k == 0 else None)
        ax.plot(unseen, k, 's', color='tab:red', ms=6, label='unseen' if k == 0 else None)
        ax.plot(r[3], k, '|', color='black', ms=10, label='all' if k == 0 else None)
    ax.set_yticks(range(len(rows)), [''] * len(rows))
    ax.set(xlim=(0, 100), xlabel='recall of the CRF head (pooled over seeds and splits)',
           title='(b) how many of them the CRF head finds')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.grid(alpha=0.3, axis='x')
    ax = axes[2]
    if methods:
        rows = methods[::-1]
        for k, (label, seen, unseen) in enumerate(rows):
            for (mean, lo, hi), dy, color, marker, name in (
                    (seen, 0.15, 'tab:blue', 'o', 'seen, same type'),
                    (unseen, -0.15, 'tab:red', 's', 'unseen')):
                ax.errorbar(mean, k + dy, xerr=[[mean - lo], [hi - mean]], fmt=marker,
                            color=color, ms=5, capsize=2, label=name if k == 0 else None)
        ax.axvline(0, color='grey', lw=0.8)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
        ax.grid(alpha=0.3, axis='x')
    ax.set(xlabel='F1 minus the CRF head or its vote (95% CI, 4 papers)',
           title='(c) unseen papers, fine-grained: the gains')
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def gain_data(directory, analysis):
    """{granularity: [(label, mean, CI low, CI high, kind)]} for Figure 6, kind 'ten' (10
    seeds), 'screen' (3 seeds) or 'papers' (a third training paper); methods without a
    comparison on the held-out papers are left out."""
    def entries(names):
        out = []
        for name in names:
            found = load(directory, [name])
            if found:
                out += [e for e in found[0].values() if e['protocol'] == LOPO
                        and e['reference'] == 'crf']
        return out
    ten, screen = entries(TEN_SEED_COMPARISONS), entries(SCREENS)
    out = {}
    for g in GRANULARITIES:
        rows = []
        for variant, label in FIG6_METHODS.items():
            for kind, found in (('ten', ten), ('screen', screen)):
                hit = [e for e in found if e['granularity'] == g and e['variant'] == variant]
                if hit:
                    m = hit[0]['micro']
                    rows.append((label, m['mean_diff'], *m['ci95'], kind))
                    break
        papers = analysis.get('paper_count', {}).get(g, {}).get('three_minus_two')
        if papers:
            rows.append(('a third training paper', papers['mean_diff'], *papers['ci95'],
                         'papers'))
        if rows:
            out[g] = rows
    return out


def figure6(gains, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(gains), figsize=(6.2 * len(gains), 5.2), sharey=False,
                             squeeze=False)
    styles = {'ten': ('o', 'tab:blue', 'tab:blue', 'method, 10 seeds'),
              'screen': ('o', 'white', 'tab:blue', 'method, 3-seed screen'),
              'papers': ('D', 'tab:green', 'tab:green', 'a third training paper (3 seeds)')}
    for ax, (g, rows) in zip(axes[0], gains.items()):
        rows = rows[::-1]
        seen = set()
        for k, (label, mean, lo, hi, kind) in enumerate(rows):
            marker, face, edge, name = styles[kind]
            ax.errorbar(mean, k, xerr=[[mean - lo], [hi - mean]], fmt=marker, ms=6,
                        mfc=face, mec=edge, color=edge, capsize=2,
                        label=None if kind in seen else name)
            seen.add(kind)
        ax.axvline(0, color='grey', lw=0.8)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
        ax.set(xlabel='micro F1 minus the CRF head (95% CI over the 4 papers)',
               title=f'{g}-grained, unseen papers')
        ax.grid(alpha=0.3, axis='x')
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description='The figures of the paper plan')
    parser.add_argument('--results', default=os.path.join(
        os.path.dirname(__file__), '..', '..', 'results', 'gpu'))
    parser.add_argument('--out', default=os.path.join(
        os.path.dirname(__file__), '..', '..', 'docs', 'figures'))
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    os.makedirs(args.out, exist_ok=True)
    written = []
    points = protocol_points(args.results)
    analysis = (load(args.results, [ANALYSIS]) or [{}])[0]
    if all(points.values()):
        figure1(points, os.path.join(args.out, 'fig1_random_vs_unseen.png'),
                sofc_points(analysis))
        written.append('fig1_random_vs_unseen.png')
    else:
        print('fig1: no system with both protocols')
    spread = {key: list(entry['crf']['per_split'].values())
              for key, entry in analysis.get('spread', {}).items() if 'crf' in entry}

    def pair(g, protocol, stem):
        v, r = (family_files(args.results, g, protocol, s) for s in (stem, 'crf'))
        return triple_differences(v, r) if v and r else None
    mt, ctx = pair('fine', RANDOM, 'crf_mt'), pair('fine', LOPO, 'crf_ctx')
    if spread and (mt or ctx):
        figure2(spread, mt, ctx, os.path.join(args.out, 'fig2_seeds.png'))
        written.append('fig2_seeds.png')
    else:
        print('fig2: no seed spread or triples')
    types = analysis.get('types', {}).get('coarse_random_split_vary', {}).get('crf_mt_ens3')
    if types:
        figure3(types, os.path.join(args.out, 'fig3_types.png'))
        written.append('fig3_types.png')
    else:
        print('fig3: no per-type scores')
    curves, steps = threshold_data(args.results)
    if curves or steps:
        figure4(curves, steps, os.path.join(args.out, 'fig4_threshold.png'))
        written.append('fig4_threshold.png')
    else:
        print('fig4: no bias-grid results')
    mentions = (load(args.results, [MENTIONS]) or [{}])[0]
    settings, methods = mention_data(mentions)
    if settings:
        figure5(settings, methods, os.path.join(args.out, 'fig5_mentions.png'))
        written.append('fig5_mentions.png')
    else:
        print('fig5: no mention analysis')
    gains = gain_data(args.results, analysis)
    if gains:
        figure6(gains, os.path.join(args.out, 'fig6_gains.png'))
        written.append('fig6_gains.png')
    else:
        print('fig6: no comparisons on the held-out papers')
    print(f"wrote {', '.join(written) or 'nothing'} to {args.out}")
    return written


if __name__ == '__main__':
    main()
