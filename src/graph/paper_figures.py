"""
paper_figures.py - the figures of the paper plan (docs/PAPER.md) from the GPU results

No training: reads the 10-seed GPU results (results/gpu) and writes PNG files.
  fig1_random_vs_unseen.png (C2): per system with 10 seeds on both protocols, its mean micro
      F1 over the 5 random splits and over the 4 held-out papers, with 95% CIs, joined by a
      line, per granularity. A model scores the mean of its 10 runs per split, a vote the
      mean of its votes over the 3 seed triples. A third panel (week 8, part 1;
      ner_week5_seed_analysis.json, sofc_splits) puts SOFC-Exp's CRF head and its votes on
      the same axis: 5 random sentence splits against 5 folds of 9 held-out articles; a
      fourth (week 10; mulms) MuLMS's: 5 random splits of a 400-sentence pool against its 7
      held-out subfields, trained on 400 sentences of the other subfields.
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
      published split and held-out papers per granularity, SOFC-Exp's random splits, article
      folds and official split at both levels, MuLMS's random pool splits and held-out
      subfields), the shares of the gold test entities whose words are an entity of the same
      type in the training sentences (seen), only of another type, or none (unseen); (b) the
      CRF head's recall on the seen and the unseen ones; (c) on the held-out papers,
      fine-grained, each method's F1 minus the CRF head's on seen and on unseen entities,
      with 95% CIs over the 4 papers (votes against the CRF head's vote).
  fig6_gains.png (C5-C8, C19): on the 4 held-out papers, per granularity, each method's
      mean paired micro-F1 difference from the CRF head with its 95% CI over the papers:
      the 10-seed comparison where there is one (weeks 5-7), else the 3-seed screen (hollow
      markers); and a third training paper, the CRF head trained on three papers minus on
      two (paper_count, seeds 0-200).
  fig7_names.png (C26, paper A's Section 4.4; ner_mentions.json): per held-out setting
      (MuLMS's subfields, MaterioMiner's papers at both granularities), each system's recall
      and precision on the unseen test names, pooled over seeds and units, on the
      recall-precision plane with iso-F1 curves: the CRF head, mention replacement with 3
      copies, 6 copies without names, and 6 copies with the harvested names (the topic
      corpus's and the mixed corpus's on MuLMS, the fatigue corpus's on MaterioMiner), each
      joined to the system it adds to on the same seeds (the mixed corpus, on seeds 300-900
      only, is not joined to MR6's ten); the legend has one entry per marker.
  fig8_method.png (paper A's method overview, no data): the pipeline as boxes and arrows:
      three CRF teachers trained on the labeled sentences tag an unlabeled corpus; the
      mentions all three tag alike and confidently form the lexicon; mention replacement
      with six copies draws from the training mentions and the lexicon; the student trains
      on the gold sentences and their copies and is tested on the held-out document.
  fig9_units.png (paper A's main results figure; ner_week5_seed_analysis.json): per held-out
      unit, the paired micro-F1 difference of MR6 with harvested names against the CRF head
      and against MR6 without names: MuLMS's 7 subfields (seeds 300-900), MaterioMiner's 4
      papers at both granularities (10 seeds) and SOFC-Exp's 5 article folds (against the
      CRF head only), with the pooled 11-unit means and intervals of decision 13.2's units.
A figure whose results are missing is skipped (and named on stdout).

Usage:
  python src/graph/paper_figures.py [--results results/gpu] [--out docs/figures]
  fig10_recall_matched.png (paper A's Figure 4; ner_recall_matched.json, week 18): per MuLMS
      held-out subfield, the recall-matched control's recall at every bias D of the grid, on all
      test entities (the matching, pre-specified) and on the unseen test names the lexicon holds
      (post hoc), pooled over the decision seeds, with the names' values as dashed lines and the
      matched bias D* marked (flagged where the grid cannot reach the names' recall).
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
                'crf_mt_mr': 'multi-task + mention replacement',
                'crf_mrs': 'mention replacement + silver names'}
TEN_SEED_COMPARISONS = ('ner_week5_comparisons.json', 'ner_week6_comparisons.json',
                        'ner_week7_comparisons.json', 'ner_week9_comparisons.json',
                        'ner_week11_comparisons.json')
SCREENS = ('ner_week6_screen.json', 'ner_week7_screen.json')
# Figure 4: the bias grid of the CRF head (week 7, part 2a)
SOFC_BIAS = 'ner_results_sofc_published_split_crf_bias.json'
SHORT_PAPER = {'10.1007_s11340-021-00758-x': 's11340', '10.1038_ncomms7164': 'ncomms7164',
               '10.1038_s41467-020-19071-7': 's41467', '10.3390_met10111458': 'met10111458'}
SCREEN_TRIPLE = (0, 1, 2)                 # seed offsets 0, 100 and 200 (the week-6 screen)
# Figure 5: seen and unseen test entities (ner_mentions.py)
MENTIONS = 'ner_mentions.json'
MATCHED = 'ner_recall_matched.json'
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
                    'sofc_slot_published_split': 'SOFC-Exp slots, official split',
                    'mulms_random_split_pool': 'MuLMS, random splits',
                    'mulms_leave_one_subfield_out': 'MuLMS, held-out subfields'}
MENTION_METHODS = {'crf_mt': 'multi-task', 'crf_mt_ens3': 'multi-task vote',
                   'crf_st3': 'self-training', 'crf_ls': 'label semantics',
                   'crf_ctx': 'document context', 'crf_ctx_ens3': 'context vote',
                   'crf_mt_st3': 'multi-task self-training', 'crf_mr': 'mention replacement',
                   'crf_mr_ens3': 'mention-replacement vote',
                   'crf_mt_mr': 'multi-task + mention replacement',
                   'crf_mt_mr_ens3': 'multi-task + mention-replacement vote',
                   'crf_mt_st3_ens3': 'multi-task self-training vote',
                   'crf_mrs': 'mention replacement + silver names'}
# Figure 7: what the harvested names do to the unseen test names (paper A, Section 4.4)
NAMES_SETTINGS = {'mulms_leave_one_subfield_out': 'MuLMS, held-out subfields',
                  'fine_leave_one_paper_out': 'MaterioMiner fine-grained, held-out papers',
                  'coarse_leave_one_paper_out': 'MaterioMiner coarse-grained, held-out papers'}
NAMES_SYSTEMS = {'crf': 'CRF head', 'crf_mr': 'MR3 (3 copies)',
                 'crf_mr6': 'MR6 (6 copies, no names)',
                 'crf_mrm_topic6': 'MR6 + names (topic corpus)',
                 'crf_mrm6': 'MR6 + names (fatigue corpus)',
                 'crf_mrm_mixed6': 'MR6 + names (mixed corpus)'}
# the arrows join systems on the same seeds: the mixed corpus (seeds 300-900) has none to MR6 (0-900)
NAMES_PARENT = {'crf_mr': 'crf', 'crf_mr6': 'crf_mr', 'crf_mrm_topic6': 'crf_mr6',
                'crf_mrm6': 'crf_mr6'}
NAMES_BOTH = 'MR6 + names (MuLMS: topic corpus; MaterioMiner: fatigue corpus)'


def names_legend(shown):
    """[(system, label)] of Figure 7's legend for the systems shown, in NAMES_SYSTEMS order, one
    entry per marker: the names on MuLMS (topic corpus) and on MaterioMiner (fatigue corpus)
    share the red star and so one entry."""
    both = {'crf_mrm_topic6', 'crf_mrm6'} <= set(shown)
    return [(system, NAMES_BOTH if both and system == 'crf_mrm_topic6' else label)
            for system, label in NAMES_SYSTEMS.items()
            if system in shown and not (both and system == 'crf_mrm6')]


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


def mulms_points(analysis):
    """Figure 1's MuLMS rows from the analysis's mulms section: [(label, (random pool splits
    mean, lo, hi), (held-out subfields mean, lo, hi))] for the CRF head and, if both
    protocols have them, its votes; [] without both protocols."""
    m = analysis.get('mulms', {})
    pool, held = m.get('random_split_pool'), m.get('leave_one_subfield_out')
    if not pool or not held:
        return []
    pairs = [('CRF head', pool['micro'], held['crf']['micro'])]
    if 'crf_votes' in pool and 'votes' in held:
        pairs.append(('CRF vote', pool['crf_votes']['micro'], held['votes']['crf']['micro']))
    return [(label, *((e['mean'], *e['ci95']) for e in sides)) for label, *sides in pairs]


def figure1(points, path, sofc=(), mulms=()):
    import matplotlib.pyplot as plt
    panels = [(points[g], f'{g}-grained') for g in GRANULARITIES]
    unseen = [('MaterioMiner', '4 papers')]
    if sofc:
        panels.append((sofc, 'SOFC-Exp (entity types)'))
        unseen.append(('SOFC-Exp', '5 folds of 9 articles'))
    if mulms:
        panels.append((mulms, 'MuLMS (outermost entities)'))
        unseen.append(('MuLMS', '7 subfields'))
    # the axis of weeks 5-8 (30-90), wider only if a panel's intervals leave it
    ends = [end for rows, _ in panels for _, *sides in rows for _, lo, hi in sides
            for end in (lo, hi)]
    xlim = (30 if min(ends) >= 30 else min(ends) - 2, 90 if max(ends) <= 90 else max(ends) + 2)
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
        ax.set(xlim=xlim, ylim=(-0.5, n_rows - 0.5), title=title, xlabel='micro F1')
        ax.grid(alpha=0.3, axis='x')
    handles, labels = axes[0].get_legend_handles_labels()
    if len(unseen) > 1:
        documents = '; '.join(f'{corpus}: {held}' for corpus, held in unseen)
        labels = [label.replace('4 unseen papers', f'unseen documents ({documents})')
                  for label in labels]
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


def names_data(mentions):
    """{setting label: [(system, label, recall, precision, F1 on the unseen test names, F1 on
    the seen ones)]} for Figure 7, from ner_mentions.json, in NAMES_SYSTEMS order; a setting
    without the CRF head and a system without results are left out."""
    out = {}
    for key, setting in NAMES_SETTINGS.items():
        found = mentions.get('systems', {}).get(key, {})
        rows = []
        for system, label in NAMES_SYSTEMS.items():
            unseen = found.get(system, {}).get('unseen')
            if unseen:
                rows.append((system, label, unseen['recall'], unseen['precision'],
                             unseen['f1'], found[system].get('seen', {}).get('f1')))
        if rows and rows[0][0] == 'crf':
            out[setting] = rows
    return out


def figure7(rows_by_setting, path):
    import matplotlib.pyplot as plt
    n = len(rows_by_setting)
    if n == 3:                       # 7.2 inches wide: MuLMS on top, the two MaterioMiner panels below
        fig = plt.figure(figsize=(7.2, 9.0))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
        axes = [[fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]]
    else:
        fig, axes = plt.subplots(1, n, figsize=(min(7.2, 3.6 * n), 4.6), squeeze=False)
    styles = {'crf': ('o', 'black'), 'crf_mr': ('s', 'tab:blue'), 'crf_mr6': ('D', 'tab:cyan'),
              'crf_mrm_topic6': ('*', 'tab:red'), 'crf_mrm6': ('*', 'tab:red'),
              'crf_mrm_mixed6': ('P', 'tab:orange')}
    iso_labels, point_labels = [], []
    for k, (ax, (setting, rows)) in enumerate(zip(axes[0], rows_by_setting.items())):
        narrow = n == 3 and k > 0          # the lower panels: half the width, labels to the left
        at = {system: (recall, precision) for system, _, recall, precision, *_ in rows}
        xs, ys = [r[2] for r in rows], [r[3] for r in rows]
        x0, x1 = min(xs) - 3.5, max(xs) + 3.5
        y0, y1 = min(ys) - 3, max(ys) + 3
        for f in range(10, 100, 2):                          # iso-F1 curves: P = fR / (2R - f)
            curve = [(r, f * r / (2 * r - f)) for r in
                     [x0 + (x1 - x0) * k / 200 for k in range(201)] if 2 * r > f]
            curve = [(r, p) for r, p in curve if y0 <= p <= y1]
            if len(curve) > 1:
                ax.plot(*zip(*curve), color='lightgrey', lw=0.7, zorder=1)
                if curve[-1][0] >= x1 - (x1 - x0) / 100:      # the curve leaves on the right
                    iso_labels.append(ax.annotate(f'F1 {f}', curve[-1], xytext=(2, 0),
                                                  textcoords='offset points', fontsize=6.5,
                                                  color='grey', va='center'))
                elif curve[0][0] <= x0 + (x1 - x0) / 100:    # it enters on the left
                    iso_labels.append(ax.annotate(f'F1 {f}', curve[0], xytext=(2, 0),
                                                  textcoords='offset points', fontsize=6.5,
                                                  color='grey', va='center'))
                else:                                        # it enters at the top
                    iso_labels.append(ax.annotate(f'F1 {f}', curve[0], xytext=(0, 2),
                                                  textcoords='offset points', fontsize=6.5,
                                                  color='grey', ha='center', va='bottom'))
        for system, label, recall, precision, f1, seen in rows:
            parent = NAMES_PARENT.get(system)
            if parent in at:
                ax.annotate('', (recall, precision), at[parent],
                            arrowprops={'arrowstyle': '->', 'color': 'grey', 'lw': 1,
                                        'shrinkA': 6, 'shrinkB': 6}, zorder=2)
            marker, color = styles.get(system, ('o', 'tab:grey'))
            ax.plot(recall, precision, marker, color=color, ms=9 if marker == '*' else 6,
                    zorder=3)
            seen_text = f' (seen {seen:.1f})' if seen is not None else ''
            text = f'F1 {f1:.2f}{seen_text}' if narrow else f'{label}\nF1 {f1:.2f}{seen_text}'
            def near(other):
                return other in at and abs(at[other][0] - recall) < 2.5 and abs(at[other][1] - precision) < 2.5
            names_near = near('crf_mrm_topic6') or near('crf_mrm6')
            if system == 'crf_mr6':              # below-right when the CRF head crowds it, else below-left
                offset, align = ((7, -18), 'left') if near('crf') else ((-7, -14), 'right')
            elif system == 'crf_mr':             # away from a names point to its upper right
                offset, align = ((-7, 12), 'right') if names_near else ((7, 10), 'left')
            elif system == 'crf':
                offset, align = (-7, -14), 'right'
            elif system in ('crf_mrm_topic6', 'crf_mrm6'):
                offset, align = (7, 6), 'left'
            elif system == 'crf_mrm_mixed6':
                offset, align = (7, 14), 'left'
            else:
                offset, align = (7, -16), 'left'
            leader = ({'arrowstyle': '-', 'color': '0.6', 'lw': 0.5, 'shrinkA': 0, 'shrinkB': 3}
                      if narrow else None)
            point_labels.append(ax.annotate(
                text, (recall, precision), textcoords='offset points', fontsize=7, xytext=offset,
                ha=align, zorder=4, arrowprops=leader,
                bbox={'boxstyle': 'round,pad=0.12', 'fc': 'white', 'ec': 'none', 'alpha': 0.85}))
        ax.set(xlim=(x0 - (5 if narrow else 2), x1 + 4), ylim=(y0 - 1, y1 + 1))
        ax.set_title(setting, fontsize=9)
        if n > 1:                        # panel letters, as the journal asks for multi-panel figures
            ax.annotate('abcdefgh'[k] + ')', (0, 1), xycoords='axes fraction', xytext=(-30, 8),
                        textcoords='offset points', fontsize=11, fontweight='bold',
                        ha='left', va='bottom')
        ax.set_xlabel('recall on the unseen test names (%)', fontsize=8)
        ax.set_ylabel('precision on the unseen test names (%)', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    if n == 3:                       # the lower panels carry scores only: name the markers once
        from matplotlib.lines import Line2D
        shown = {system for rows in rows_by_setting.values() for system, *_ in rows}
        handles = [Line2D([], [], linestyle='', marker=styles[system][0], color=styles[system][1],
                          ms=8 if styles[system][0] == '*' else 6, label=label)
                   for system, label in names_legend(shown)]
        fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    else:
        fig.tight_layout()
    fig.canvas.draw()                # hide an iso-F1 label that a point's label would cover
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in point_labels]
    for t in iso_labels:
        if any(t.get_window_extent(renderer).overlaps(b) for b in boxes):
            t.set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def unit_data(analysis):
    """[(group, unit, difference against the CRF head, difference against MR6 or None)] for
    Figure 9: MuLMS's subfields on the decision's seeds (mulms.week12.<winner>.decision and
    week14.copies.mulms.decision), MaterioMiner's papers per granularity over 10 seeds
    (week14.copies.<granularity>: the analogue minus the control, and the control minus the
    CRF head, summed for the CRF head), SOFC-Exp's folds (week13.folds.crf_mrm6; no control).
    Empty without the MuLMS decision part."""
    winner = analysis.get('week13', {}).get('week12_winner', 'crf_mrm_topic6')
    w = analysis.get('mulms', {}).get('week12', {}).get(winner, {}).get('decision')
    copies = analysis.get('week14', {}).get('copies', {})
    c = copies.get('mulms', {}).get('decision')
    if not w or not c:
        return []
    rows = []
    units = list(w['subfields'])
    crf = w['minus_crf']['micro']['per_run']
    ctrl = c['minus_crf_mr6']['micro']['per_run']
    rows += [('MuLMS subfield (seeds 300–900)', u, a, b) for u, a, b in zip(units, crf, ctrl)]
    for g in ('fine', 'coarse'):
        part = copies.get(g)
        if not part:
            continue
        names = list(part['units'])
        vs_ctrl = part['minus_crf_mr6']['micro']['per_run']
        ctrl_vs_crf = part['control']['minus_crf']['micro']['per_run']
        rows += [(f'MaterioMiner {g}-grained paper (10 seeds)', SHORT_PAPER.get(u.replace('.conll', ''), u),
                  round(a + b, 2), a) for u, a, b in zip(names, vs_ctrl, ctrl_vs_crf)]
    folds = analysis.get('week13', {}).get('folds', {}).get('crf_mrm6')
    if folds:
        rows += [('SOFC-Exp article fold (10 seeds)', u, d, None)
                 for u, d in zip(folds['units'], folds['minus_crf']['micro']['per_run'])]
    return rows


def figure9(rows, path, registered=None):
    """Paper A's Figure 2 (module docstring). registered: (mean, low, high) of D5 as decided
    (MuLMS's units on all ten seeds), labelled beside the band of the units shown."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    groups = []
    for g, *_ in rows:
        if g not in groups:
            groups.append(g)
    y = 0
    ticks, labels, bands, labelled = [], [], [], False
    pooled_rows = []                 # the y of the 11 units' rows (MuLMS and MaterioMiner fine)
    for g in groups:
        members = [r for r in rows if r[0] == g]
        start = y                    # the group's header row, then its units
        y += 1
        for _, unit, vs_crf, vs_ctrl in members:
            if g.startswith('MuLMS') or 'fine-grained' in g:
                pooled_rows.append(y)
            ax.plot(vs_crf, y, 'o', color='tab:red', ms=6, zorder=3,
                    label=None if labelled else 'against the CRF head')
            if vs_ctrl is not None:
                ax.plot(vs_ctrl, y, 's', color='tab:blue', ms=5, zorder=3,
                        label=None if labelled else 'against MR6 (6 copies, no names)')
            labelled = True
            ticks.append(y)
            labels.append(unit)
            y += 1
        bands.append((g, start, y - 1))
        y += 0.5
    for g, start, end in bands:
        shade = '#f2f2f2' if bands.index((g, start, end)) % 2 else 'white'
        ax.axhspan(start - 0.5, end + 0.5, color=shade, zorder=0)
        ax.text(0.01, start, g, fontsize=8.5, va='center', ha='left', style='italic', zorder=4,
                transform=ax.get_yaxis_transform(), color='#333333',      # over the lines
                bbox={'boxstyle': 'square,pad=0.15', 'fc': shade, 'ec': 'none'})
    pooled = [r for r in rows if r[0].startswith('MuLMS') or 'fine-grained' in r[0]]
    notes = []                       # the lower right's labels, top to bottom
    if registered:
        m, lo, hi = registered
        notes.append(('tab:red', f'D5 vs CRF head as decided (MuLMS on all ten seeds): '
                                 f'{m:+.2f} [{lo:+.2f}, {hi:+.2f}]'))
    for values, color, label in (
            (lambda rs: [r[2] for r in rs], 'tab:red',
             'red band, D5\u2032 (the units shown; post hoc) vs CRF head'),
            (lambda rs: [r[3] for r in rs], 'tab:blue', 'blue band, D7 vs MR6')):
        v = values(pooled)
        if len(v) == 11:
            m, lo, hi = mean_ci(v)
            top_y, bottom_y = min(pooled_rows) - 1.5, max(pooled_rows) + 0.5
            ax.fill_betweenx([top_y, bottom_y], lo, hi, color=color, alpha=0.08, zorder=1, lw=0)
            ax.vlines(m, top_y, bottom_y, color=color, lw=1, ls='--', zorder=2)
            notes.append((color, f'{label}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}]'))
    for k, (color, text) in enumerate(notes):     # the empty lower right, clear of the rows
        ax.text(0.99, 0.015 + 0.04 * (len(notes) - 1 - k), text.replace('-', '\u2212'),
                fontsize=8.5, color=color, ha='right', va='bottom', transform=ax.transAxes,
                zorder=5, bbox={'boxstyle': 'square,pad=0.15', 'fc': 'white', 'ec': 'none'})
    ax.axvline(0, color='black', lw=0.8, zorder=2)
    ax.set_yticks(ticks, labels, fontsize=9)
    ax.set_ylim(-0.7, y + 2.4)              # room below the last row for the pooled labels
    ax.invert_yaxis()
    ax.set_xlabel('micro F1 of MR6 + harvested names minus the reference on the held-out unit '
                  '(percentage points)', fontsize=9)
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(alpha=0.3, axis='x')
    ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def figure10(matched, path):
    """Paper A's Figure 4 (module docstring): small multiples, one panel per subfield."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    units = matched['per_unit']
    curves = matched['post_hoc_curves']['units']
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.0))
    for k, (ax, (u, v)) in enumerate(zip(axes.flat, units.items())):
        ax.annotate('abcdefgh'[k] + ')', (0, 1), xycoords='axes fraction', xytext=(-22, 6),
                    textcoords='offset points', fontsize=10, fontweight='bold',
                    ha='left', va='bottom')
        grid = [float(g) for g in v['curve']]
        overall = [v['curve'][g]['recall'] for g in v['curve']]
        held = [curves[u]['control'][g]['recall_in'] for g in v['curve']]
        ax.plot(grid, held, '-o', color='tab:red', lw=1.5, ms=3.5, zorder=3)
        ax.plot(grid, overall, '-o', color='tab:blue', lw=1.5, ms=3.5, zorder=3)
        ax.axhline(curves[u]['names']['recall_in'], color='tab:red', lw=1.2, ls='--', zorder=2)
        ax.axhline(v['target_recall'], color='tab:blue', lw=1.2, ls='--', zorder=2)
        ax.axvline(v['d_star'], color='#555555', lw=0.9, ls=':', zorder=1)
        ax.text(v['d_star'], 0.03, ' D*', transform=ax.get_xaxis_transform(), fontsize=7.5,
                color='#333333', ha='left' if v['d_star'] < 4 else 'right', va='bottom')
        values = overall + held + [curves[u]['names']['recall_in'], v['target_recall']]
        pad = 0.08 * (max(values) - min(values))
        ax.set_ylim(min(values) - pad, max(values) + pad)
        ax.set_title(u + (' (flagged)' if v['flagged'] else ''), fontsize=9)
        ax.set_xticks([-2, 0, 2, 4], ['\u22122', '0', '2', '4'], fontsize=7.5)
        ax.tick_params(axis='y', labelsize=7.5)
        ax.grid(alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel('bias D on every entity tag', fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel('recall (%)', fontsize=8)
    legend = axes.flat[len(units)] if len(units) < axes.size else None
    if legend is not None:
        legend.axis('off')
        handles = [Line2D([], [], color='tab:blue', marker='o', ms=3.5, lw=1.5,
                          label='MR6 under the bias:\nall test entities'),
                   Line2D([], [], color='tab:blue', ls='--', lw=1.2,
                          label='MR6 + names: all test entities\n(the matching target)'),
                   Line2D([], [], color='tab:red', marker='o', ms=3.5, lw=1.5,
                          label='MR6 under the bias: unseen\nnames the lexicon holds'),
                   Line2D([], [], color='tab:red', ls='--', lw=1.2,
                          label='MR6 + names: unseen\nnames the lexicon holds'),
                   Line2D([], [], color='#555555', ls=':', lw=0.9, label='matched bias D*')]
        legend.legend(handles=handles, loc='center', fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def figure8(path):
    """Paper A's method overview: boxes and arrows, no data; 7.2 inches wide (a journal's
    double column) with 7.5-point type."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set(xlim=(-0.1, 14.6), ylim=(0, 11.2))
    ax.axis('off')
    top, bottom, h = 7.4, 1.4, 3.2
    boxes = {  # name: (x, y, w, h, text, face)
        'gold': (0.2, top, 2.6, h, 'Labeled\ntraining sentences\n(3 papers; 340\nMuLMS sentences;\nor 36 SOFC-Exp\narticles)',
                 '#dbe9f6'),
        'teachers': (3.1, top, 2.6, h, '3 CRF teachers\ntrained on them\n(seeds 0, 100\nand 200: the\nreference\'s runs)',
                     '#dbe9f6'),
        'corpus': (6.0, top, 2.6, h, 'Unlabeled corpus\nthe teachers tag:\n40,000 sentences\n(the held-out\ntopic, or a mix\nof others)',
                   '#fff2cc'),
        'filter': (8.9, top, 2.6, h, 'Mention filter:\nall 3 teachers\nagree on span\nand type, every\nmarginal \u2265 0.9',
                   '#fff2cc'),
        'lexicon': (11.8, top, 2.6, h, 'Lexicon:\n4k to 13k names\nper MuLMS subfield', '#fff2cc'),
        'pool': (1.5, bottom, 3.6, h, 'Replacement pool\ntraining mentions\n(601 to 1,309)\n+ the lexicon\n(at least 92% of entries)',
                 '#e2f0d9'),
        'mr': (5.7, bottom, 3.6, h, 'Mention replacement\n6 copies per sentence;\neach mention swapped\n(p = 0.5) for a\nsame-type name',
               '#e2f0d9'),
        'student': (9.9, bottom, 3.6, h, 'Student\nMatSciBERT + CRF,\nthe reference\'s recipe\nand seed; tested on\nthe held-out document',
                    '#e2f0d9')}
    for x, y, w, hh, text, face in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle='round,pad=0.02,rounding_size=0.15',
                                    facecolor=face, edgecolor='#555555', lw=0.8))
        ax.text(x + w / 2, y + hh / 2, text, ha='center', va='center', fontsize=7.2, linespacing=1.25)

    def arrow(a, b, text=None):
        xa, ya, wa, ha = boxes[a][:4]
        xb, yb, wb, hb = boxes[b][:4]
        start, end = (xa + wa, ya + ha / 2), (xb, yb + hb / 2)
        ax.annotate('', end, start, arrowprops={'arrowstyle': '->', 'color': '#333333', 'lw': 1,
                                                 'shrinkA': 1, 'shrinkB': 1})
        if text:
            ax.text((start[0] + end[0]) / 2, ya + ha + 0.2, text, ha='center', va='bottom',
                    fontsize=7, color='#333333')
    arrow('gold', 'teachers')
    arrow('teachers', 'corpus')
    arrow('corpus', 'filter')
    arrow('filter', 'lexicon')
    arrow('pool', 'mr')
    arrow('mr', 'student')
    # the lexicon's names and the training mentions feed the pool: down, across above the
    # lower row, then down into the pool
    pool_top, rail = boxes['pool'][1] + boxes['pool'][3], bottom + h + 0.7
    head = {'arrowstyle': '->', 'color': '#333333', 'lw': 1, 'shrinkA': 0, 'shrinkB': 1}
    for x_from, y_from, x_to in ((boxes['lexicon'][0] + 0.6, boxes['lexicon'][1],
                                  boxes['pool'][0] + boxes['pool'][2] - 0.6),
                                 (boxes['gold'][0] + 0.5, boxes['gold'][1], boxes['pool'][0] + 0.6)):
        ax.plot([x_from, x_from, x_to], [y_from, rail, rail], color='#333333', lw=1,
                solid_joinstyle='miter')
        ax.annotate('', (x_to, pool_top), (x_to, rail), arrowprops=head)
    ax.text(0.9, 6.1, 'training mentions', fontsize=7, color='#333333')
    ax.text(8.6, rail + 0.15, 'harvested names', fontsize=7, color='#333333', ha='center')
    ax.text(0.2, 0.15, 'No teacher-labeled sentence is trained on: the gold contexts stay, only the names change. Without the '
            'lexicon the student is\nmention replacement with 6 copies (the control); with 3 copies and no lexicon, '
            'MR3 (mention replacement at a fixed setting).', fontsize=7, color='#333333',
            va='bottom', linespacing=1.3)
    fig.savefig(path, dpi=300, bbox_inches='tight')
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
                sofc_points(analysis), mulms_points(analysis))
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
    names = names_data(mentions)
    if names:
        figure7(names, os.path.join(args.out, 'fig7_names.png'))
        written.append('fig7_names.png')
    else:
        print('fig7: no unseen-name scores of the names method')
    figure8(os.path.join(args.out, 'fig8_method.png'))
    written.append('fig8_method.png')
    units = unit_data(analysis)
    if units:
        winner = analysis.get('week13', {}).get('week12_winner', 'crf_mrm_topic6')
        d5 = analysis.get('week13', {}).get('pooled', {}).get(winner, {}).get('minus_crf')
        figure9(units, os.path.join(args.out, 'fig9_units.png'),
                registered=(d5['mean_diff'], *d5['ci95']) if d5 else None)
        written.append('fig9_units.png')
    else:
        print('fig9: no per-unit comparisons of the names method')
    matched = (load(args.results, [MATCHED]) or [{}])[0]
    if matched.get('per_unit') and matched.get('post_hoc_curves'):
        figure10(matched, os.path.join(args.out, 'fig10_recall_matched.png'))
        written.append('fig10_recall_matched.png')
    else:
        print('fig10: no recall-matched control')
    print(f"wrote {', '.join(written) or 'nothing'} to {args.out}")
    return written


if __name__ == '__main__':
    main()
