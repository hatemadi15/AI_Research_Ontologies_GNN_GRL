"""
paper_figures.py - the figures of the paper plan (docs/PAPER.md) from the GPU results

No training: reads the 10-seed GPU results (results/gpu) and writes PNG files.
  fig1_random_vs_unseen.png (C2): per system with 10 seeds on both protocols, its mean micro
      F1 over the 5 random splits and over the 4 held-out papers, with 95% CIs, joined by a
      line, per granularity. A model scores the mean of its 10 runs per split, a vote the
      mean of its votes over the 3 seed triples.
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
# Figure 4: the bias grid of the CRF head (week 7, part 2a)
SOFC_BIAS = 'ner_results_sofc_published_split_crf_bias.json'
SHORT_PAPER = {'10.1007_s11340-021-00758-x': 's11340', '10.1038_ncomms7164': 'ncomms7164',
               '10.1038_s41467-020-19071-7': 's41467', '10.3390_met10111458': 'met10111458'}
SCREEN_TRIPLE = (0, 1, 2)                 # seed offsets 0, 100 and 200 (the week-6 screen)


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


def figure1(points, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, g in zip(axes, GRANULARITIES):
        rows = points[g][::-1]                       # the first system on top
        for k, (label, (x, xlo, xhi), (y, ylo, yhi)) in enumerate(rows):
            ax.plot([y, x], [k, k], color='lightgrey', lw=2, zorder=1)
            ax.errorbar(x, k, xerr=[[x - xlo], [xhi - x]], fmt='o', color='tab:blue', ms=5,
                        capsize=2, label='5 random splits' if k == 0 else None)
            ax.errorbar(y, k, xerr=[[y - ylo], [yhi - y]], fmt='s', color='tab:red', ms=5,
                        capsize=2, label='4 unseen papers' if k == 0 else None)
            ax.annotate(f'{x - y:.1f}', ((x + y) / 2, k), textcoords='offset points',
                        xytext=(0, 4), fontsize=7, ha='center', color='dimgrey')
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
        ax.set(xlim=(30, 90), title=f'{g}-grained', xlabel='micro F1')
        ax.grid(alpha=0.3, axis='x')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8)
    fig.suptitle('The same systems on random splits and on unseen papers (10 seeds, 95% CIs; '
                 'the number is the gap)', fontsize=10)
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
    if all(points.values()):
        figure1(points, os.path.join(args.out, 'fig1_random_vs_unseen.png'))
        written.append('fig1_random_vs_unseen.png')
    else:
        print('fig1: no system with both protocols')
    analysis = (load(args.results, [ANALYSIS]) or [{}])[0]
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
    print(f"wrote {', '.join(written) or 'nothing'} to {args.out}")
    return written


if __name__ == '__main__':
    main()
