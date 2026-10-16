"""
ner_seedmeans.py - NER comparisons of seed means (ROADMAP §4, week-3 to week-5 protocols)

Week 3 scores a model trained with three seeds on a split by the mean of its three runs
(model seeds K, K + 100 and K + 200 on random split K; 0, 100 and 200 on a held-out
paper), an ensemble or a one-seed run by its one run, and compares these scores split
by split. No training: the script reads the results files of ner_baseline.py and
ner_ensemble.py.

  - Families of three seeds (the variants of their results files): crf = crf_m0,
    crf_m100, crf_m200 (the reference: the week-2 ensemble members); crf_mt = crf_mt,
    crf_mt_m100, crf_mt_m200; crf_ls, crf_st and crf_dapt2 = <family>_m0, _m100, _m200.
  - Single systems, scored once per split: crf_ens3, crf_mix3, crf_scibert and
    crf_pubmedbert.
  - A family's score for a run_id is the mean over its members' runs with that run_id,
    for micro, strict micro, macro and weighted F1. A run_id counts only if all three
    members have it.
  - Comparisons, in the protocol's reporting order: crf_mix3 vs crf_ens3 (the primary
    one of variant 1); crf_mix3, crf_ens3, crf_scibert and crf_pubmedbert vs crf; crf_ls,
    crf_st, crf_mt and crf_dapt2 vs crf. Each is made per granularity over the 5 random
    splits and the 4 held-out papers and paired by run_id: the per-run differences,
    their mean and the 95% Student-t CI, rounded as in ner_comparisons.json.
  - role: 'decision' on the protocol where the week-3 rule decides (crf_mix3 vs
    crf_ens3, crf_mt and crf_dapt2 over the random splits; crf_ls and crf_st over the
    held-out papers), 'reported' otherwise. complete: 5 pairs over the random splits,
    4 over the held-out papers.

A comparison is skipped while one of its results files is missing. The output is one
JSON object keyed like ner_comparisons.json:

    ner_results_<granularity>_<protocol>_<variant>[_vs_<reference>, unless it is crf]

An entry holds variant, reference, granularity, protocol, role, the files of both sides,
n_pairs, run_ids, complete, the per-split scores of both sides (variant_scores and
reference_scores: mean, 95% CI and per_run for every F1 average, in run_ids order), and
micro, micro_strict, macro and weighted, each with mean_diff, ci95 and per_run.

Week 4 (--week 4) makes the comparisons of the week-4 protocol with the same scores and
entries, written to ner_week4_comparisons.json:
  - New systems: the families crf_mt_ls (label semantics on the multi-task model) and
    crf_st3 (self-training with 3 teachers), = <family>_m0, _m100, _m200; and the votes
    crf_ls_ens3, crf_st_ens3 and crf_mt_ens3 (ner_ensemble.py over the three seeds of
    crf_ls, crf_st and crf_mt), scored once per split.
  - The CPU runs (results/): the label-semantics and self-training votes vs crf_ens3,
    decided over the held-out papers, and both votes vs crf (reported).
  - The GPU runs (--gpu, results/gpu/): crf_mt_ens3 vs crf_ens3 (decided over the random
    splits), crf_mt_ls vs crf (decided over the random splits and, separately, over the
    held-out papers) and crf_st3 vs crf (decided over the held-out papers); then, reported,
    the GPU repeats of the CPU votes (crf_ls_ens3 and crf_st_ens3 vs crf_ens3), crf_ens3,
    crf_mt_ens3, crf_mt, crf_ls and crf_st vs crf, crf_mt_ls vs crf_mt and crf_ls, and
    crf_st3 vs crf_st and crf_ens3. GPU runs are only compared with GPU runs, except in
    the descriptive entries `<key>_gpu_vs_cpu` (role 'descriptive'): a system's GPU scores
    minus its CPU scores (--cpu DIR, default the parent of the GPU directory), per split,
    for the systems run on both (HARDWARE_SYSTEMS): what the hardware alone changes.

Week 5 (--week 5 --gpu, results/gpu/) has 10 model seeds per split, with the same scores
and entries, written to ner_week5_comparisons.json:
  - Families of 10 seeds (seed offsets 0 to 900): crf, crf_mt (crf_mt, then
    crf_mt_m100 to _m900), crf_ls and crf_st3. A family scores the mean of its 10 runs.
  - Vote families: <family>_ens3 = the votes over the seed triples m0-m200, m300-m500
    and m600-m800 (<family>_ens3, <family>_ens3_m300, <family>_ens3_m600), scored by
    the mean of the 3 votes.
  - Decisions: crf_mt vs crf (random splits), crf_ls and crf_st3 vs crf (held-out
    papers), and the votes crf_mt_ens3, crf_ls_ens3 and crf_st3_ens3 vs crf_ens3 (both
    protocols: the reported system); reported: crf_ens3 vs crf.

Week 6 (--week 6 --gpu, results/gpu/; week-6 protocol) tests methods for unseen papers,
each a family crf_deberta (DeBERTa-v3-large), crf_ctx (document context) or crf_mt_st3
(multi-task self-training), on the held-out papers:
  - --screen: stage 1, seeds 0-200 (3-seed families, first votes), written to
    ner_week6_screen.json. Each method against crf has role 'screen': a method goes to
    stage 2 if its mean micro-F1 difference is above 0 at either granularity, which is
    printed. Reported: crf_mt_st3 vs crf_mt, and each method's first vote vs crf_ens3.
  - without --screen: stage 2, 10-seed families and vote families as in week 5, written
    to ner_week6_comparisons.json. Decisions: each method vs crf, and each method's vote
    vs the strongest system on unseen papers after week 5 (UNSEEN_BEST: crf_mt_ens3
    fine-grained, crf_ens3 coarse-grained); the vote against the other granularity's
    reference is reported. Reported: crf_mt_st3 vs crf_mt.

Week 7, part 1 (--week 7 --gpu; week-7 protocol) tests document-level label consistency
(ner_consistency.py: the variant <input>_dc, or _dcr with retyping) on the saved
predictions of the 10-seed runs:
  - Decisions: crf_dc vs crf on the held-out papers, and the rule on the strongest system
    on unseen papers (UNSEEN_BEST): crf_mt_ens3_dc vs crf_mt_ens3 fine-grained, crf_ens3_dc
    vs crf_ens3 coarse-grained. Reported: the random splits, the other granularity's vote,
    retyping (_dcr), and the rule on document context (crf_ctx_dc, crf_ctx_ens3_dc).

Usage:
  python src/graph/ner_seedmeans.py [--results results] [--out results/ner_week3_comparisons.json]
  python src/graph/ner_seedmeans.py --week 4                            # CPU votes
  python src/graph/ner_seedmeans.py --week 4 --gpu --results results/gpu
  python src/graph/ner_seedmeans.py --week 5 --gpu --results results/gpu
  python src/graph/ner_seedmeans.py --week 6 --gpu --results results/gpu [--screen]
  python src/graph/ner_seedmeans.py --week 7 --gpu --results results/gpu
"""

import argparse
import json
import os
import statistics
from pathlib import Path

import metrics
from ner_baseline import F1_AVERAGES, result_suffix

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results'
OUT_NAME = 'ner_week3_comparisons.json'
OUT_NAMES = {3: OUT_NAME, 4: 'ner_week4_comparisons.json', 5: 'ner_week5_comparisons.json',
             6: 'ner_week6_comparisons.json', 7: 'ner_week7_comparisons.json'}
SCREEN_NAME = 'ner_week6_screen.json'

GRANULARITIES = ('fine', 'coarse')
# pairs of a complete comparison: the 5 random splits and the 4 held-out papers
PROTOCOLS = {'random_split_vary': 5, 'leave_one_paper_out': 4}

# three model seeds per split: the variants of their results files
FAMILIES = {
    'crf': ('crf_m0', 'crf_m100', 'crf_m200'),
    'crf_mt': ('crf_mt', 'crf_mt_m100', 'crf_mt_m200'),
    'crf_ls': ('crf_ls_m0', 'crf_ls_m100', 'crf_ls_m200'),
    'crf_st': ('crf_st_m0', 'crf_st_m100', 'crf_st_m200'),
    'crf_dapt2': ('crf_dapt2_m0', 'crf_dapt2_m100', 'crf_dapt2_m200'),
}
# one run per split: the ensembles and the backbones of the mixed one
SINGLE = ('crf_ens3', 'crf_mix3', 'crf_scibert', 'crf_pubmedbert')
# week 4: label semantics on the multi-task model, self-training with 3 teachers, and the
# votes over the seeds of the week-3 changes that help
WEEK4_FAMILIES = {
    'crf_mt_ls': ('crf_mt_ls_m0', 'crf_mt_ls_m100', 'crf_mt_ls_m200'),
    'crf_st3': ('crf_st3_m0', 'crf_st3_m100', 'crf_st3_m200'),
}
WEEK4_SINGLE = ('crf_ls_ens3', 'crf_st_ens3', 'crf_mt_ens3')
SYSTEMS = {**FAMILIES, **WEEK4_FAMILIES,
           **{name: (name,) for name in SINGLE + WEEK4_SINGLE}}

# (variant, reference, protocol on which the week-3 rule decides; None: reported only),
# in the protocol's reporting order
COMPARISONS = (
    ('crf_mix3', 'crf_ens3', 'random_split_vary'),
    ('crf_mix3', 'crf', None),
    ('crf_ens3', 'crf', None),
    ('crf_scibert', 'crf', None),
    ('crf_pubmedbert', 'crf', None),
    ('crf_ls', 'crf', 'leave_one_paper_out'),
    ('crf_st', 'crf', 'leave_one_paper_out'),
    ('crf_mt', 'crf', 'random_split_vary'),
    ('crf_dapt2', 'crf', 'random_split_vary'),
)
# week 4, in the protocol's reporting order; a decision may be taken on both protocols
WEEK4_CPU_COMPARISONS = (
    ('crf_ls_ens3', 'crf_ens3', 'leave_one_paper_out'),
    ('crf_st_ens3', 'crf_ens3', 'leave_one_paper_out'),
    ('crf_ls_ens3', 'crf', None),
    ('crf_st_ens3', 'crf', None),
)
WEEK4_GPU_COMPARISONS = (
    ('crf_mt_ens3', 'crf_ens3', 'random_split_vary'),
    ('crf_mt_ls', 'crf', ('random_split_vary', 'leave_one_paper_out')),
    ('crf_st3', 'crf', 'leave_one_paper_out'),
    # the GPU repeats of the CPU votes (protocol amendment), reported
    ('crf_ls_ens3', 'crf_ens3', None),
    ('crf_st_ens3', 'crf_ens3', None),
    ('crf_ens3', 'crf', None),
    ('crf_mt_ens3', 'crf', None),
    ('crf_mt', 'crf', None),
    ('crf_ls', 'crf', None),
    ('crf_st', 'crf', None),
    ('crf_mt_ls', 'crf_mt', None),
    ('crf_mt_ls', 'crf_ls', None),
    ('crf_st3', 'crf_st', None),
    ('crf_st3', 'crf_ens3', None),
)


# systems run on the CPU and again on the GPU: their GPU minus CPU scores are described
HARDWARE_SYSTEMS = ('crf', 'crf_ens3', 'crf_mt', 'crf_ls', 'crf_st', 'crf_ls_ens3',
                    'crf_st_ens3')

# week 5: 10 model seeds per split, and votes over 3 disjoint seed triples
WEEK5_OFFSETS = tuple(range(0, 1000, 100))
WEEK5_TRIPLES = (0, 300, 600)        # the first seed offset of each voting triple


def seed_family(stem, untagged_zero=False, offsets=WEEK5_OFFSETS):
    """The variants of a seed family: <stem>_m0 to <stem>_m900 by default (crf_mt has no m0
    tag)."""
    return tuple(stem if untagged_zero and off == 0 else f'{stem}_m{off}' for off in offsets)


def vote_family(stem):
    """The votes over the triples of a family: <stem>_ens3, _ens3_m300, _ens3_m600."""
    return tuple(f'{stem}_ens3' + (f'_m{off}' if off else '') for off in WEEK5_TRIPLES)


WEEK5_SYSTEMS = {
    'crf': seed_family('crf'),
    'crf_mt': seed_family('crf_mt', untagged_zero=True),
    'crf_ls': seed_family('crf_ls'),
    'crf_st3': seed_family('crf_st3'),
    **{f'{stem}_ens3': vote_family(stem) for stem in ('crf', 'crf_mt', 'crf_ls', 'crf_st3')},
}
BOTH_PROTOCOLS = tuple(PROTOCOLS)
WEEK5_COMPARISONS = (
    # the claims so far, each on the protocol of its earlier decision
    ('crf_mt', 'crf', 'random_split_vary'),
    ('crf_ls', 'crf', 'leave_one_paper_out'),
    ('crf_st3', 'crf', 'leave_one_paper_out'),
    # the reported system, per protocol and granularity
    ('crf_mt_ens3', 'crf_ens3', BOTH_PROTOCOLS),
    ('crf_ls_ens3', 'crf_ens3', BOTH_PROTOCOLS),
    ('crf_st3_ens3', 'crf_ens3', BOTH_PROTOCOLS),
    # reported: what the vote adds to single models
    ('crf_ens3', 'crf', None),
)


# week 6: methods for unseen papers, screened with seeds 0-200, then decided with 10 seeds
WEEK6_METHODS = ('crf_deberta', 'crf_ctx', 'crf_mt_st3')
SCREEN_OFFSETS = (0, 100, 200)
LOPO = 'leave_one_paper_out'
# the strongest system on unseen papers after week 5, per granularity (week-5 decision 2)
UNSEEN_BEST = {'fine': 'crf_mt_ens3', 'coarse': 'crf_ens3'}
WEEK6_SCREEN_SYSTEMS = {
    'crf': seed_family('crf', offsets=SCREEN_OFFSETS),
    'crf_mt': seed_family('crf_mt', untagged_zero=True, offsets=SCREEN_OFFSETS),
    **{m: seed_family(m, offsets=SCREEN_OFFSETS) for m in WEEK6_METHODS},
    **{f'{stem}_ens3': (f'{stem}_ens3',) for stem in ('crf', 'crf_mt') + WEEK6_METHODS},
}
WEEK6_SCREEN_COMPARISONS = (
    *((m, 'crf', LOPO) for m in WEEK6_METHODS),
    ('crf_mt_st3', 'crf_mt', None),
    *((f'{m}_ens3', 'crf_ens3', None) for m in WEEK6_METHODS),
)
WEEK6_SYSTEMS = {
    **WEEK5_SYSTEMS,
    **{m: seed_family(m) for m in WEEK6_METHODS},
    **{f'{m}_ens3': vote_family(m) for m in WEEK6_METHODS},
}
WEEK6_COMPARISONS = (
    # decision 1: each method against the CRF head on the held-out papers
    *((m, 'crf', LOPO) for m in WEEK6_METHODS),
    # decision 2: each method's vote against the strongest system on unseen papers
    *((f'{m}_ens3', ref, {g: LOPO}) for m in WEEK6_METHODS for g, ref in UNSEEN_BEST.items()),
    # reported: what self-training adds to multi-task training
    ('crf_mt_st3', 'crf_mt', None),
)


# week 7, part 1: document-level label consistency (ner_consistency.py) on saved predictions
DC_RULES = ('_dc', '_dcr')                    # propagation; with retyping (reported)


def dc_family(variants, rule='_dc'):
    """The consistency rule's outputs of a family: _dc (or _dcr) before each seed tag."""
    out = []
    for v in variants:
        stem, sep, tag = v.rpartition('_m')
        out.append(f'{stem}{rule}_m{tag}' if sep and tag.isdigit() else f'{v}{rule}')
    return tuple(out)


WEEK7_SYSTEMS = {
    **WEEK6_SYSTEMS,
    **{f'{stem}{rule}': dc_family(WEEK6_SYSTEMS[stem], rule)
       for stem in ('crf', 'crf_ens3', 'crf_mt_ens3', 'crf_ctx', 'crf_ctx_ens3')
       for rule in DC_RULES},
}
WEEK7_COMPARISONS = (
    # decision 1: the rule on the CRF head (held-out papers; random splits reported)
    ('crf_dc', 'crf', LOPO),
    # decision 2: the rule on the strongest system on unseen papers, per granularity
    *((f'{ref}_dc', ref, {g: LOPO}) for g, ref in UNSEEN_BEST.items()),
    # reported: retyping, and the rule on document context
    ('crf_dcr', 'crf', None),
    *((f'{ref}_dcr', ref, None) for ref in UNSEEN_BEST.values()),
    ('crf_ctx_dc', 'crf_ctx', None),
    ('crf_ctx_ens3_dc', 'crf_ctx_ens3', None),
)


def comparison_list(week, gpu=False, screen=False):
    """The comparisons of a week's protocol: week 3, week 4 on the CPU or the GPU runs,
    week 5, or week 6 (its stage-1 screen with screen=True); weeks 5 and 6 GPU runs only."""
    if week == 3:
        if gpu:
            raise ValueError('the week-3 comparisons are made on the CPU runs only')
        return COMPARISONS
    if week in (5, 6, 7):
        if not gpu:
            raise ValueError(f'the week-{week} comparisons are made on the GPU runs only')
        if week == 7:
            return WEEK7_COMPARISONS
        if week == 6:
            return WEEK6_SCREEN_COMPARISONS if screen else WEEK6_COMPARISONS
        return WEEK5_COMPARISONS
    return WEEK4_GPU_COMPARISONS if gpu else WEEK4_CPU_COMPARISONS


def system_list(week, screen=False):
    """The systems (name: the variants of its results files) of a week's protocol."""
    if week == 7:
        return WEEK7_SYSTEMS
    if week == 6:
        return WEEK6_SCREEN_SYSTEMS if screen else WEEK6_SYSTEMS
    return WEEK5_SYSTEMS if week == 5 else SYSTEMS


def decides_on(decides, protocol, granularity=None):
    """Whether the protocol's rule decides a comparison on `protocol` (decides: None, a
    protocol, a tuple of protocols, or {granularity: protocol(s)} when only some
    granularities are decided)."""
    if decides is None:
        return False
    if isinstance(decides, dict):
        decides = decides.get(granularity, ())
    return protocol in ((decides,) if isinstance(decides, str) else decides)


def file_names(granularity, protocol, system, systems=None):
    """Results files of a system (a family's seeds, or one file) for one protocol."""
    systems = SYSTEMS if systems is None else systems
    return [f'ner_results_{granularity}{result_suffix(protocol, v)}.json' for v in systems[system]]


def comparison_key(granularity, protocol, variant, reference):
    """Key of a comparison: as in ner_comparisons.json, plus _vs_<reference> unless crf."""
    key = f'ner_results_{granularity}{result_suffix(protocol, variant)}'
    return key if reference == 'crf' else f'{key}_vs_{reference}'


def system_scores(files):
    """{run_id: {F1 average: score}} of a system from its parsed results files.

    One file gives the scores of its runs; a family's files give the mean over its
    members, for the run_ids that every member has (in the order of the first file).
    """
    by_id = [{r.get('run_id'): r for r in f['runs']} for f in files]
    ids = [i for i in by_id[0] if all(i in b for b in by_id[1:])]
    return {i: {name: statistics.fmean(get(b[i]) for b in by_id)
                for name, get in F1_AVERAGES.items()} for i in ids}


def spread(values, mean_key='mean'):
    """Mean, 95% Student-t CI and per-run values, rounded as in ner_comparisons.json."""
    return {mean_key: round(statistics.mean(values), 2),
            'ci95': [round(x, 2) for x in metrics.mean_ci(values)['ci']],
            'per_run': [round(v, 2) for v in values]}


def compare(variant, reference):
    """Scores of both sides and their differences (variant - reference), paired by run_id.

    variant, reference: {run_id: {F1 average: score}} as from system_scores, with at
    least one run_id in common. The pairs follow the order of the variant's run_ids.
    """
    ids = [i for i in variant if i in reference]
    out = {'run_ids': ids}
    for side, scores in (('variant_scores', variant), ('reference_scores', reference)):
        out[side] = {name: spread([scores[i][name] for i in ids]) for name in F1_AVERAGES}
    for name in F1_AVERAGES:
        out[name] = spread([variant[i][name] - reference[i][name] for i in ids], 'mean_diff')
    return out


def load_scores(directory, names):
    """system_scores of the results files `names` in `directory`."""
    files = []
    for name in names:
        with open(os.path.join(directory, name)) as f:
            files.append(json.load(f))
    return system_scores(files)


def week3_comparisons(directory, comparisons_list=COMPARISONS, systems=None,
                      decision_role='decision'):
    """Every comparison of `comparisons_list` (default: week 3) that the results files in
    `directory` allow, keyed as above; `systems` (default: SYSTEMS) names their files, and
    the comparisons that the rule decides get the role `decision_role`.

    Prints one line per comparison: its micro-F1 difference, or why it was skipped (a
    missing results file, or no run_id with scores on both sides).
    """
    comparisons = {}
    for variant, reference, decides in comparisons_list:
        for protocol, need in PROTOCOLS.items():
            for granularity in GRANULARITIES:
                key = comparison_key(granularity, protocol, variant, reference)
                files = {side: file_names(granularity, protocol, side, systems)
                         for side in (variant, reference)}
                missing = [fn for names in files.values() for fn in names
                           if not os.path.exists(os.path.join(directory, fn))]
                if missing:
                    print(f"  {key}: no {', '.join(missing)}, skipped")
                    continue
                scores = {side: load_scores(directory, names) for side, names in files.items()}
                if not any(i in scores[reference] for i in scores[variant]):
                    print(f"  {key}: no run_id with scores on both sides, skipped")
                    continue
                comp = compare(scores[variant], scores[reference])
                entry = {'variant': variant, 'reference': reference, 'granularity': granularity,
                         'protocol': protocol,
                         'role': (decision_role if decides_on(decides, protocol, granularity)
                                  else 'reported'),
                         'variant_files': files[variant], 'reference_files': files[reference],
                         'n_pairs': len(comp['run_ids']), 'run_ids': comp['run_ids'],
                         'complete': len(comp['run_ids']) >= need, **comp}
                comparisons[key] = entry
                d = entry['micro']
                print(f"  {key}: micro F1 difference {d['mean_diff']:+.2f} vs {reference}, "
                      f"95% CI {d['ci95']} over {entry['n_pairs']} paired runs ({entry['role']}"
                      f"{'' if entry['complete'] else ', incomplete'})")
    return comparisons


def hardware_comparisons(gpu_dir, cpu_dir):
    """The GPU scores of each HARDWARE_SYSTEMS system minus its CPU scores, per split.

    Descriptive only (role 'descriptive'): the two sides differ in hardware, not in method.
    Keyed <key of the system>_gpu_vs_cpu; skipped while a file is missing on either side.
    """
    out = {}
    for system in HARDWARE_SYSTEMS:
        for protocol, need in PROTOCOLS.items():
            for granularity in GRANULARITIES:
                names = file_names(granularity, protocol, system)
                if not all(os.path.exists(os.path.join(d, fn))
                           for d in (gpu_dir, cpu_dir) for fn in names):
                    continue
                gpu, cpu = load_scores(gpu_dir, names), load_scores(cpu_dir, names)
                if not any(i in cpu for i in gpu):
                    continue
                comp = compare(gpu, cpu)
                key = f'ner_results_{granularity}{result_suffix(protocol, system)}_gpu_vs_cpu'
                out[key] = {'variant': system, 'reference': system, 'granularity': granularity,
                            'protocol': protocol, 'role': 'descriptive',
                            'hardware': {'variant': 'gpu', 'reference': 'cpu'},
                            'variant_files': names, 'reference_files': names,
                            'n_pairs': len(comp['run_ids']), 'run_ids': comp['run_ids'],
                            'complete': len(comp['run_ids']) >= need, **comp}
                d = out[key]['micro']
                print(f"  {key}: GPU minus CPU micro F1 {d['mean_diff']:+.2f}, 95% CI "
                      f"{d['ci95']} over {out[key]['n_pairs']} splits (descriptive)")
    return out


def screen_verdicts(comparisons):
    """Per week-6 method with screen entries: 'goes to stage 2 (...)' if its mean micro-F1
    difference against crf over the held-out papers is above 0 at either granularity, else
    'stops (...)'; a method lacking a granularity's complete entry is 'incomplete'."""
    out = {}
    for method in WEEK6_METHODS:
        found = {g: comparisons.get(comparison_key(g, LOPO, method, 'crf'))
                 for g in GRANULARITIES}
        if not any(found.values()):
            continue
        if not all(c and c['complete'] for c in found.values()):
            out[method] = 'incomplete'
            continue
        diffs = ', '.join(f"{g} {c['micro']['mean_diff']:+.2f}" for g, c in found.items())
        passed = any(c['micro']['mean_diff'] > 0 for c in found.values())
        out[method] = f"{'goes to stage 2' if passed else 'stops'} ({diffs})"
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='NER comparisons of 3-seed means (no training): reads the results files '
                    'and writes ner_week<N>_comparisons.json')
    parser.add_argument('--results', default=str(RESULTS_DIR), metavar='DIR',
                        help='directory with the NER results files (default: results/ of the repo)')
    parser.add_argument('--week', type=int, choices=sorted(OUT_NAMES), default=3,
                        help='the protocol whose comparisons are made (default: 3)')
    parser.add_argument('--gpu', action='store_true',
                        help='with --week 4 or 5: the comparisons of the GPU runs (results/gpu)')
    parser.add_argument('--cpu', default=None, metavar='DIR',
                        help='with --gpu: the CPU results for the descriptive GPU-minus-CPU '
                             'entries (default: the parent of --results)')
    parser.add_argument('--screen', action='store_true',
                        help='with --week 6: the stage-1 screen (seeds 0-200), written to '
                             f'{SCREEN_NAME}')
    parser.add_argument('--out', default=None, metavar='PATH',
                        help='output file (default: DIR/ner_week<N>_comparisons.json)')
    args = parser.parse_args(argv)
    if not os.path.isdir(args.results):
        parser.error(f'results directory not found: {args.results}')
    if args.gpu and args.week == 3:
        parser.error('--gpu needs --week 4 or later: GPU runs started in week 4')
    if args.week >= 5 and not args.gpu:
        parser.error(f'--week {args.week} needs --gpu: it runs on the GPU only')
    if args.screen and args.week != 6:
        parser.error('--screen needs --week 6')
    comparisons = week3_comparisons(
        args.results, comparison_list(args.week, args.gpu, args.screen),
        system_list(args.week, args.screen),
        decision_role='screen' if args.screen else 'decision')
    if args.gpu and args.week == 4:
        cpu = args.cpu or os.path.dirname(os.path.abspath(args.results))
        comparisons.update(hardware_comparisons(args.results, cpu))
    if args.screen:
        for method, verdict in screen_verdicts(comparisons).items():
            print(f'  screen: {method} {verdict}')
    out = args.out or os.path.join(args.results,
                                   SCREEN_NAME if args.screen else OUT_NAMES[args.week])
    with open(out, 'w') as f:
        json.dump(comparisons, f, indent=2)
    complete = sum(c['complete'] for c in comparisons.values())
    print(f"Saved {out}: {len(comparisons)} comparisons, {complete} complete")
    return comparisons


if __name__ == '__main__':
    main()
