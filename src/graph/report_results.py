"""
report_results.py - regenerate the results sections of the docs from results/*.json

Rewrites two generated blocks and leaves the rest of each file alone:

    README.md         between <!-- RESULTS:START --> and <!-- RESULTS:END -->
    docs/ROADMAP.md   between <!-- STANDING:START --> and <!-- STANDING:END -->

Every number comes from the JSON files in results/; only the wording around the
numbers lives here. Standard library only (CI has no torch, seqeval or scipy),
and all paths are derived from this file's location.

Usage:
  python src/graph/report_results.py                   # rewrite both files in place
  python src/graph/report_results.py --check           # exit 1 and print a diff if out of date
  python src/graph/report_results.py --pending "NER coarse CRF" "NER ensemble"
  python src/graph/report_results.py --results DIR --readme PATH --roadmap PATH   # for tests

Exit codes: 0 = written / up to date, 1 = `--check` found a difference, 2 = bad
input (missing or invalid file, missing marker, bad arguments).

NER comparisons (results/ner_comparisons.json, written by `ner_baseline.py --compare`)
---------------------------------------------------------------------------------------
An entry compares one variant with a reference run on the same split and model
seed. Its `reference` field says which run: `baseline` (plain MatSciBERT) or
`crf` (the CRF head); entries without the field are older baseline comparisons.

  * "NER training variants vs the baseline": variants `cw` and `crf`, reference
    `baseline`. Also feeds the "At a glance" table and the ROADMAP standing table.
  * "NER week-2 variants vs the CRF head": variants `crf_mt` (multi-task), `crf_dapt`
    (domain-adaptive pre-training) and `crf_ens3` (3-model ensemble), reference `crf`.
    Its intro notes that these runs use 2 CPU threads and the CRF head 3; once the
    `crf_m0` comparisons are complete it also gives their difference to the CRF head,
    the effect of the thread count alone.
    Nothing from this table goes into "At a glance" or the standing table.

Which entries get a row or a verdict is decided by an allow-list of variants plus the
reference, so anything else in the file never gets one: ensemble members (`crf_m0`,
`crf_m100`, `crf_m200`; `crf_m0` only appears in the thread-count note above),
bare-tag variants, comparisons against another reference.
A row or verdict needs a complete comparison: 5 pairs for random splits (5 splits or
5 model seeds), 4 for leave one paper out. Until the first complete week-2 comparison
exists the generated text is identical to what it was before they were supported.

NER week 3 (results/ner_week3_comparisons.json, written by `ner_seedmeans.py`)
-------------------------------------------------------------------------------
Entries pair 3-seed means split by split (week-3 protocol, ROADMAP §4): reference `crf`
is the mean of `crf_m0`, `crf_m100` and `crf_m200`, `role` says whether the protocol
decides on that comparison or only reports it, and `variant_scores` holds the variant's
per-split scores.

  * "NER week-3 comparisons (3-seed means)": the complete comparisons of the allow-list
    WEEK3_COMPARISONS, in its order (the protocol's), each with a verdict bullet.
  * The reported NER system, a row in "At a glance" and in the standing table per the
    week-3 protocol: the 3-model seed ensemble `crf_ens3` (its own results files), or for
    a granularity the mixed-backbone ensemble `crf_mix3` once its complete comparison
    with `crf_ens3` over the 5 random splits has a micro-F1 CI above 0. The paper does
    not state its averaging, so both rows say under which averaging (micro, macro,
    weighted F1, as far as the summary has them) the CI over the 5 random splits lies
    above the published number.

NER week 4 (results/ner_week4_comparisons.json and results/gpu/ner_week4_comparisons.json,
written by `ner_seedmeans.py --week 4` and `--week 4 --gpu`)
-------------------------------------------------------------------------------------------
  * "NER week-4 comparisons": the complete comparisons of WEEK4_CPU_COMPARISONS (votes over
    the saved CPU predictions), then of WEEK4_GPU_COMPARISONS (GPU runs, compared only with
    GPU runs), in their order, with the machine in a Runs column. Each WEEK4_VERDICTS entry
    gets a verdict bullet, and the descriptive GPU-minus-CPU entries of the CRF head one
    bullet; they never get a row.
  * The reported system then has two slots per granularity (week-4 protocol). In
    distribution: the week-3 system, unless the GPU multi-task vote `crf_mt_ens3` improves
    on the GPU `crf_ens3` over the 5 random splits; it is then reported with its GPU
    results. On unseen papers: the week-3 system, unless a CPU vote (`crf_ls_ens3`,
    `crf_st_ens3`) improves on `crf_ens3` over the 4 held-out papers (if both do, the larger
    mean difference). Both rows name each slot's system, say "(week-4 protocol)" once a
    week-4 comparisons file exists, and the published check uses the in-distribution one.

NER week 5 (results/gpu/ner_week5_comparisons.json, written by `ner_seedmeans.py --week 5
--gpu`)
-------------------------------------------------------------------------------------------
  * "NER week-5 comparisons": the complete comparisons of WEEK5_COMPARISONS (10-seed means
    of GPU runs; a vote family scores the mean of its 3 votes), in their order, and a
    verdict bullet for each.
  * Once that file exists, the reported system of both tables is decided on it, per
    protocol (week5_system): crf_ens3, unless a vote of WEEK5_VOTES improves on it (the
    largest mean difference). Its scores are the vote family's, and a vote that holds only
    the unseen-paper slot is named. The published check is made against PUBLISHED_F1.
  * The published split (week-5 amendment) gets an "At a glance" row once
    results/gpu/ner_results_<granularity>_published_split[_crf].json hold 10 runs: plain
    MatSciBERT and the CRF head, and where the published number lies against plain
    MatSciBERT's CI over the seeds.
"""
import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

GRANULARITIES = ('fine', 'coarse')

# Variants shown in the tables and verdicts, with their display names. Only variants
# listed here are ever displayed (see the module docstring).
BASELINE_VARIANTS = {'cw': 'class-weighted loss', 'crf': 'CRF head'}
WEEK2_VARIANTS = {'crf_mt': 'multi-task (fine + coarse heads)',
                  'crf_dapt': 'domain-adaptive pre-training',
                  'crf_ens3': '3-model ensemble'}

# Week 3 (see the module docstring): the comparisons (variant, reference) in the protocol's
# reporting order, as in ner_seedmeans.COMPARISONS, and the display names of their sides
WEEK3_FILE = 'ner_week3_comparisons.json'
WEEK3_COMPARISONS = (('crf_mix3', 'crf_ens3'), ('crf_mix3', 'crf'), ('crf_ens3', 'crf'),
                     ('crf_scibert', 'crf'), ('crf_pubmedbert', 'crf'), ('crf_ls', 'crf'),
                     ('crf_st', 'crf'), ('crf_mt', 'crf'), ('crf_dapt2', 'crf'))
WEEK3_NAMES = {'crf': 'CRF head', 'crf_ens3': '3-model seed ensemble',
               'crf_mix3': 'mixed-backbone ensemble', 'crf_scibert': 'SciBERT backbone',
               'crf_pubmedbert': 'PubMedBERT backbone', 'crf_ls': 'label semantics',
               'crf_st': 'self-training', 'crf_mt': 'multi-task (fine + coarse heads)',
               'crf_dapt2': 'in-domain pre-training'}
# Week 4 (week-4 protocol, ROADMAP §4): the CPU votes in results/ and the GPU runs in
# results/gpu/, each file's comparisons in the protocol's reporting order, as in
# ner_seedmeans.WEEK4_CPU_COMPARISONS and WEEK4_GPU_COMPARISONS
WEEK4_FILE = 'ner_week4_comparisons.json'
GPU_PREFIX = 'gpu/'
WEEK4_CPU_COMPARISONS = (('crf_ls_ens3', 'crf_ens3'), ('crf_st_ens3', 'crf_ens3'),
                         ('crf_ls_ens3', 'crf'), ('crf_st_ens3', 'crf'))
WEEK4_GPU_COMPARISONS = (('crf_mt_ens3', 'crf_ens3'), ('crf_mt_ls', 'crf'), ('crf_st3', 'crf'),
                         ('crf_ls_ens3', 'crf_ens3'), ('crf_st_ens3', 'crf_ens3'),
                         ('crf_ens3', 'crf'), ('crf_mt_ens3', 'crf'), ('crf_mt', 'crf'),
                         ('crf_ls', 'crf'), ('crf_st', 'crf'), ('crf_mt_ls', 'crf_mt'),
                         ('crf_mt_ls', 'crf_ls'), ('crf_st3', 'crf_st'), ('crf_st3', 'crf_ens3'))
WEEK4_NAMES = {**WEEK3_NAMES, 'crf_ls_ens3': 'label-semantics vote',
               'crf_st_ens3': 'self-training vote', 'crf_mt_ens3': 'multi-task vote',
               'crf_mt_ls': 'label semantics on the multi-task model',
               'crf_st3': 'self-training with 3 teachers'}
# the week-4 comparisons that get a verdict bullet: the decisions, then the GPU repeats of
# the CPU votes and of the week-3 changes (comparisons file, variant, reference, runs)
WEEK4_VERDICTS = ((WEEK4_FILE, 'crf_ls_ens3', 'crf_ens3', 'CPU'),
                  (WEEK4_FILE, 'crf_st_ens3', 'crf_ens3', 'CPU'),
                  (GPU_PREFIX + WEEK4_FILE, 'crf_mt_ens3', 'crf_ens3', 'GPU'),
                  (GPU_PREFIX + WEEK4_FILE, 'crf_mt_ls', 'crf', 'GPU'),
                  (GPU_PREFIX + WEEK4_FILE, 'crf_st3', 'crf', 'GPU'),
                  (GPU_PREFIX + WEEK4_FILE, 'crf_ls_ens3', 'crf_ens3', 'GPU'),
                  (GPU_PREFIX + WEEK4_FILE, 'crf_st_ens3', 'crf_ens3', 'GPU'),
                  (GPU_PREFIX + WEEK4_FILE, 'crf_mt', 'crf', 'GPU'),
                  (GPU_PREFIX + WEEK4_FILE, 'crf_ls', 'crf', 'GPU'),
                  (GPU_PREFIX + WEEK4_FILE, 'crf_st', 'crf', 'GPU'))
# Week 5 (week-5 protocol, ROADMAP §4): 10-seed means of the GPU runs in results/gpu/, in the
# protocol's reporting order, as in ner_seedmeans.WEEK5_COMPARISONS
WEEK5_FILE = 'ner_week5_comparisons.json'
WEEK5_COMPARISONS = (('crf_mt', 'crf'), ('crf_ls', 'crf'), ('crf_st3', 'crf'),
                     ('crf_mt_ens3', 'crf_ens3'), ('crf_ls_ens3', 'crf_ens3'),
                     ('crf_st3_ens3', 'crf_ens3'), ('crf_ens3', 'crf'))
WEEK5_NAMES = {**WEEK4_NAMES, 'crf_st3_ens3': 'vote of self-training with 3 teachers'}
# the votes that may replace crf_ens3 as the reported system (week-5 protocol, decision 2)
WEEK5_VOTES = ('crf_mt_ens3', 'crf_ls_ens3', 'crf_st3_ens3')
# how the reported-system rows name the week-5 design
WEEK5_PROTOCOL = 'week-5 protocol; 10 seeds, GPU runs'
# Week 6 (week-6 protocol, ROADMAP §4): methods for unseen papers on the held-out papers, GPU
# runs; the screen (seeds 0-200) and the 10-seed decisions, as in ner_seedmeans
WEEK6_SCREEN_FILE = 'ner_week6_screen.json'
WEEK6_FILE = 'ner_week6_comparisons.json'
WEEK6_METHODS = ('crf_deberta', 'crf_ctx', 'crf_mt_st3')
WEEK6_SCREEN_COMPARISONS = (*((m, 'crf') for m in WEEK6_METHODS), ('crf_mt_st3', 'crf_mt'),
                            *((f'{m}_ens3', 'crf_ens3') for m in WEEK6_METHODS))
# the strongest system on unseen papers after week 5, which a method's vote must beat
UNSEEN_BEST = {'fine': 'crf_mt_ens3', 'coarse': 'crf_ens3'}
WEEK6_COMPARISONS = (*((m, 'crf') for m in WEEK6_METHODS),
                     *((f'{m}_ens3', ref) for m in WEEK6_METHODS for ref in UNSEEN_BEST.values()),
                     ('crf_mt_st3', 'crf_mt'))
WEEK6_NAMES = {**WEEK5_NAMES, 'crf_deberta': 'DeBERTa-v3-large', 'crf_ctx': 'document context',
               'crf_mt_st3': 'multi-task self-training',
               'crf_deberta_ens3': 'DeBERTa-v3-large vote', 'crf_ctx_ens3': 'document-context vote',
               'crf_mt_st3_ens3': 'multi-task self-training vote'}
# Week 7, part 1 (week-7 protocol): the consistency rule (ner_consistency.py) on the saved
# predictions of the 10-seed runs, as in ner_seedmeans.WEEK7_COMPARISONS; retyping (_dcr) and
# document context are reported only
WEEK7_FILE = 'ner_week7_comparisons.json'
WEEK7_NAMES = {'crf_dc': 'the consistency rule', 'crf_dcr': 'the consistency rule with retyping'}
# on SOFC-Exp both are decisions (the amendment to part 1, before SOFC-Exp's results)
WEEK7_REPORTED = ()
WEEK7_COMPARISONS = (('crf_dc', 'crf'), *((f'{ref}_dc', ref) for ref in UNSEEN_BEST.values()),
                     ('crf_dcr', 'crf'), *((f'{ref}_dcr', ref) for ref in UNSEEN_BEST.values()),
                     ('crf_ctx_dc', 'crf_ctx'), ('crf_ctx_ens3_dc', 'crf_ctx_ens3'))
# the week-7 decisions, which get a bullet each
WEEK7_DECISIONS = WEEK7_COMPARISONS[:3]
# Week 7, part 2a: recall calibration (ner_bias.py), the bias chosen per held-out paper on
# inner folds (crf_eb) or on the validation sentences (crf_ebv; reported on MaterioMiner, a
# decision on SOFC-Exp), as in ner_seedmeans.WEEK7_COMPARISONS
WEEK7A_COMPARISONS = (('crf_eb', 'crf'), *(('crf_eb_ens3', ref) for ref in UNSEEN_BEST.values()),
                      ('crf_ebv', 'crf'))
WEEK7A_DECISIONS = WEEK7A_COMPARISONS[:3]
# Week 7, part 2c: coarse-to-fine completion (ner_coarse_to_fine.py), fine-grained only; the
# first two decide, on the held-out papers
WEEK7C_COMPARISONS = (('crf_cf', 'crf'), ('crf_mt_ens3_cf', 'crf_mt_ens3'),
                      ('crf_mt_cf', 'crf_mt'))
WEEK7C_DECISIONS = WEEK7C_COMPARISONS[:2]
# Week 7, part 2b: mention replacement (--augment-mentions 3), screened and decided as the
# week-6 methods; its screen has a file of its own
WEEK7_SCREEN_FILE = 'ner_week7_screen.json'
WEEK7B_DECISIONS = (('crf_mr', 'crf'), *(('crf_mr_ens3', ref) for ref in UNSEEN_BEST.values()))
WEEK7_SYSTEM_NAMES = {**WEEK6_NAMES, **{
    f'{stem}{rule}': f'{WEEK6_NAMES[stem]} + {WEEK7_NAMES["crf" + rule]}'
    for stem in ('crf', 'crf_ens3', 'crf_mt_ens3', 'crf_ctx', 'crf_ctx_ens3')
    for rule in ('_dc', '_dcr')},
    'crf_eb': 'CRF head + recall calibration',
    'crf_ebv': 'CRF head + recall calibration chosen on validation',
    'crf_eb_ens3': 'vote of the CRF head + recall calibration',
    'crf_mr': 'mention replacement', 'crf_mr_ens3': 'mention-replacement vote',
    'crf_mt_mr': 'multi-task + mention replacement',
    'crf_mt_mr_ens3': 'multi-task + mention-replacement vote',
    'crf_mrs': 'mention replacement with silver names',
    'crf_mrs_ens3': 'silver-name mention-replacement vote',
    'crf_mrm': 'mention replacement with silver mentions',
    'crf_mrm_ens3': 'silver-mention mention-replacement vote',
    'crf_mrm6': 'mention replacement with silver mentions, 6 copies',
    'crf_mrm6_ens3': 'silver-mention 6-copy vote',
    'crf_gzm': 'CRF head + silver mentions as a match feature',
    'crf_gzm_ens3': 'match-feature vote',
    'crf_mr6': 'mention replacement, 6 copies', 'crf_mr6_ens3': '6-copy mention-replacement vote',
    'crf_st3_ens3': 'self-training vote',
    **{v: f'{WEEK6_NAMES[ref]} + coarse-to-fine completion'
       for v, ref in WEEK7C_COMPARISONS}}
# Week 9 (week-9 protocol): mention replacement on the multi-task model, as in
# ner_seedmeans.WEEK9_COMPARISONS; the first three decide
WEEK9_FILE = 'ner_week9_comparisons.json'
WEEK9_COMPARISONS = (('crf_mt_mr', 'crf_mt'),
                     *(('crf_mt_mr_ens3', ref) for ref in UNSEEN_BEST.values()),
                     ('crf_mt_mr', 'crf'), ('crf_mt_mr', 'crf_mr'))
WEEK9_DECISIONS = WEEK9_COMPARISONS[:3]
# Week 11 (week-11 protocol): mention replacement with the teachers' silver names in its pool,
# as in ner_seedmeans.WEEK11_COMPARISONS; the first decides (MuLMS: mulms_note)
WEEK11_FILE = 'ner_week11_comparisons.json'
WEEK11_COMPARISONS = (('crf_mrs', 'crf_mr'), ('crf_mrs', 'crf'), ('crf_mrs_ens3', 'crf_mr_ens3'),
                      *(('crf_mrs_ens3', ref)
                        for ref in dict.fromkeys(('crf_ens3', *UNSEEN_BEST.values()))))
WEEK11_DECISIONS = WEEK11_COMPARISONS[:1]
# Week 12 (week-12 protocol): the MaterioMiner analogue of the screen's winner, as in
# ner_seedmeans.WEEK12_COMPARISONS; the first five decide (the MuLMS screen and decisions
# are in mulms_note)
WEEK12_FILE = 'ner_week12_comparisons.json'
WEEK12_ANALOGUE_VARIANTS = ('crf_mrm', 'crf_mrm6', 'crf_gzm', 'crf_st3', 'crf_mr6')
WEEK12_COMPARISONS = (*((v, 'crf') for v in WEEK12_ANALOGUE_VARIANTS),
                      *((v, 'crf_mr') for v in WEEK12_ANALOGUE_VARIANTS),
                      *((f'{v}_ens3', 'crf_ens3') for v in WEEK12_ANALOGUE_VARIANTS),
                      *((f'{v}_ens3', 'crf_mr_ens3') for v in WEEK12_ANALOGUE_VARIANTS))
WEEK12_DECISIONS = WEEK12_COMPARISONS[:len(WEEK12_ANALOGUE_VARIANTS)]
# the label of a week-7 method in the SOFC-Exp bullet (both parts decide on SOFC-Exp)
WEEK7_SOFC_LABELS = {**{m: f"Week 7, {name} on the CRF head's predictions"
                        for m, name in WEEK7_NAMES.items()},
                     'crf_ebv': 'Week 7, part 2a, recall calibration of the CRF head, its bias '
                                'chosen on the validation articles, against the CRF head',
                     'crf_mt': 'Week 7, part 2d, multi-task training on both levels against the '
                               'CRF head',
                     'crf_mr': 'Week 7, part 2b, S2, mention replacement against the CRF head',
                     'crf_mt_mr': 'Week 9, S2, multi-task training with mention replacement on '
                                  'both levels against the CRF head',
                     'crf_cdetest': 'Week 7, part 2e, the CRF head on the test articles cut into '
                                    'words as the training articles are, against the CRF head on '
                                    'the released ones'}
# week 7, part 2e: plain MatSciBERT on the retokenized test articles (its decision 2)
RETOKENIZED_BASELINE_LABEL = ('Week 7, part 2e, plain MatSciBERT on the retokenized test '
                              'articles, against plain MatSciBERT on the released ones')
LOPO = 'leave_one_paper_out'
# the exploratory analyses of the week-5 runs (ner_seed_analysis.py), in results/gpu/
WEEK5_ANALYSIS = 'ner_week5_seed_analysis.json'
# each family against the CRF head on the protocol of its week-5 decision (decision 1)
WEEK5_CLAIMS = (('crf_mt', 'random_split_vary'), ('crf_ls', 'leave_one_paper_out'),
                ('crf_st3', 'leave_one_paper_out'))
PROTOCOL_PHRASES = {'random_split_vary': 'over the 5 random splits',
                    'leave_one_paper_out': 'over the 4 held-out papers'}
# the published F1 of each granularity (Durmaz et al. 2024), for the rows made from comparisons
PUBLISHED_F1 = {'fine': 69.92, 'coarse': 72.32}
# SOFC-Exp: MatSciBERT with a CRF on the official split (Gupta et al. 2022), the entity
# types and the slot types (SOFC-Slot)
SOFC_PUBLISHED = {'micro': 84.77, 'macro': 82.39}
SOFC_SLOT_PUBLISHED = {'micro': 72.85, 'macro': 65.35}
# SOFC-Exp's levels in the report: the bullet's lead, the name of the CRF head's comparison
# with plain MatSciBERT and whether it decides (S1 of week 6; the slot level, week 7 part 2d,
# reports it), the "At a glance" row's name and the published scores
SOFC_LEVELS = {
    'sofc': ("* NER on SOFC-Exp (week 6; its official split, 26 training, 8 development and 11 "
             "test articles; 10 model seeds, GPU runs; the 95% CIs are over seeds)",
             'S1, the CRF head', True, 'NER on SOFC-Exp', 'MatSciBERT-CRF', SOFC_PUBLISHED),
    'sofc_slot': ("* NER on SOFC-Exp's slot types (week 7, part 2d; the 17 slot types that its "
                  "evaluation scores, on the same sentences and official split; 10 model seeds, "
                  "GPU runs; the 95% CIs are over seeds)", 'The CRF head', False,
                  "NER on SOFC-Exp's slot types", 'MatSciBERT-CRF of SOFC-Slot',
                  SOFC_SLOT_PUBLISHED)}
# model seeds of a published-split comparison (0 to 900), all needed for its decision
PUBLISHED_SEEDS = 10
# `role` of a week-3 to week-5 comparison, as the verdict bullets word it
ROLE_WORDS = {'decision': 'the pre-registered decision', 'reported': 'reported, no decision',
              'screen': 'the screen'}
# averages the reported system is checked under against the published F1
AVERAGES = ('micro', 'macro', 'weighted')

PROTOCOL_NAMES = {'random_split_vary': 'new random split per seed',
                  'leave_one_paper_out': 'leave one paper out',
                  'random_split_fixed': 'fixed split',
                  'published_split': 'the published split'}
# Pairs a comparison needs before it is shown: the 5 splits (or model seeds) and the
# 4 held-out papers.
REQUIRED_PAIRS = {'random_split_vary': 5, 'random_split_fixed': 5, 'leave_one_paper_out': 4}
PROTOCOL_FILE_PART = {'random_split_fixed': '', 'random_split_vary': '_random_split_vary',
                      'leave_one_paper_out': '_leave_one_paper_out',
                      'published_split': '_published_split'}

NER_PROTOCOLS = [('', 'fixed split (seed 0), 5 model seeds'),
                 ('_random_split_vary', 'new random split per seed'),
                 ('_leave_one_paper_out', 'leave one paper out')]

COMPARISON_COLUMNS = ("| Variant | Protocol | Granularity | Pairs | Variant micro F1 [95% CI] | "
                      "Δ micro [95% CI] | Δ macro [95% CI] | Δ strict micro |")
WEEK3_COLUMNS = ("| Comparison | Protocol | Granularity | Pairs | Variant micro F1 [95% CI] | "
                 "Δ micro [95% CI] | Δ macro [95% CI] | Δ strict micro |")
WEEK4_COLUMNS = ("| Comparison | Runs | Protocol | Granularity | Pairs | Variant micro F1 [95% CI] | "
                 "Δ micro [95% CI] | Δ macro [95% CI] | Role |")


class ReportError(Exception):
    """An input problem the caller can fix: missing or invalid file, missing marker."""


class Results:
    """Cached reader for the JSON files of one results directory."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self._cache = {}

    def get(self, name):
        """Parsed content of <directory>/<name>, or None if the file does not exist."""
        if name not in self._cache:
            path = self.directory / name
            data = None
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding='utf-8'))
                except ValueError as exc:
                    raise ReportError(f"{path} is not valid JSON: {exc}") from exc
            self._cache[name] = data
        return self._cache[name]

    def require(self, name):
        data = self.get(name)
        if data is None:
            raise ReportError(f"missing results file {self.directory / name}")
        return data

    def comparisons(self, name='ner_comparisons.json'):
        """Entries of a comparisons file (default results/ner_comparisons.json; {} if absent)."""
        comps = self.get(name) or {}
        if not isinstance(comps, dict):
            raise ReportError(f"{self.directory / name} must hold an object "
                              "with one entry per results file")
        return {key: c for key, c in comps.items() if isinstance(c, dict)}


# --------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------
def fmt(x, digits=3):
    return '–' if x is None else f"{x:.{digits}f}"


def ci_text(c):
    return f"[{c[0]:.2f}, {c[1]:.2f}]"


def interval(m, signed=False):
    """'mean [lo, hi]' of a summary (mean, ci95) or of a paired difference (mean_diff, ci95)."""
    f = '{:+.2f}' if signed else '{:.2f}'
    return (f"{f.format(m['mean'] if 'mean' in m else m['mean_diff'])} "
            f"[{f.format(m['ci95'][0])}, {f.format(m['ci95'][1])}]")


def delta_cell(d):
    lo, hi = d['ci95']
    return f"{d['mean_diff']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def dig(d, *keys):
    """d[k1][k2]... or None as soon as a level is missing."""
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def ner_file(granularity, protocol, variant=None):
    """Name of a NER results file, e.g. ner_results_fine_random_split_vary_crf.json."""
    return (f"ner_results_{granularity}{PROTOCOL_FILE_PART[protocol]}"
            f"{'_' + variant if variant else ''}.json")


def has_runs(r, n):
    """True if the NER results file `r` summarises at least `n` runs."""
    return bool(r and r.get('summary') and r['summary']['n_runs'] >= n)


def baseline_ner(res):
    """Baseline results per granularity as (fixed split, varied splits, leave one paper out)."""
    return tuple({g: res.get(ner_file(g, protocol)) for g in GRANULARITIES}
                 for protocol in ('random_split_fixed', 'random_split_vary', 'leave_one_paper_out'))


# --------------------------------------------------------------------------------
# NER comparisons: which entries count, and what they say
# --------------------------------------------------------------------------------
def reference_of(c):
    """Run a comparison is measured against; entries without the field are baseline ones."""
    return c.get('reference') or 'baseline'


def is_complete(c):
    """True if a comparison has all its pairs (see REQUIRED_PAIRS)."""
    need = REQUIRED_PAIRS.get(c.get('protocol'))
    return need is not None and (c.get('n_pairs') or 0) >= need


def selected_comparisons(comps, variants, reference):
    """(key, entry) of the complete comparisons of `variants` against `reference`.

    Ordered as the tables are: by variant (in the order of `variants`), then
    protocol, then granularity. Everything else in `comps` is left out: other
    variants (ensemble members included), other references, incomplete comparisons.
    """
    wanted = [(key, c) for key, c in comps.items()
              if c.get('variant') in variants and reference_of(c) == reference
              and c.get('granularity') in GRANULARITIES and is_complete(c)]
    return sorted(wanted, key=lambda kc: (list(variants).index(kc[1]['variant']),
                                          list(PROTOCOL_NAMES).index(kc[1]['protocol']),
                                          GRANULARITIES.index(kc[1]['granularity'])))


def find_comparison(comps, granularity, protocol, variant, reference):
    """The complete comparison of one variant for one protocol and granularity, or None."""
    for c in comps.values():
        if (c.get('variant') == variant and c.get('granularity') == granularity
                and c.get('protocol') == protocol and reference_of(c) == reference
                and is_complete(c)):
            return c
    return None


def paired_verdict(c):
    """(word, numbers) of a paired micro-F1 comparison: improves, hurts or no significant change."""
    d, s = c['micro'], c['micro_strict']
    lo, hi = d['ci95']
    word = 'improves' if lo > 0 else ('hurts' if hi < 0 else 'no significant change')
    return word, (f"{d['mean_diff']:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}]; strict IOB2 "
                  f"{s['mean_diff']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]")


# --------------------------------------------------------------------------------
# Tables (the part of README.md between the glance table and "Reading the numbers")
# --------------------------------------------------------------------------------
def typing_row(name, gold, ty, note=''):
    return (f"| {name} | {gold} | {fmt(ty['accuracy'])} {ci_text(ty['accuracy_ci95'])}{note} | "
            f"{fmt(ty['set_f1'])} | {fmt(ty['macro_f1'])} | {fmt(ty['acc@5'])} | "
            f"{fmt(ty['h_f1'])} |")


def term_typing_tables(res):
    off = res.require('eval_pipeline_llm_off.json')
    on = res.get('eval_pipeline_llm_on.json')
    v5off, v5on = res.get('eval_v5_rescored_llm_off.json'), res.get('eval_v5_rescored_llm_on.json')
    abl = res.get('ablation_summary.json') or []
    oracle = next((a for a in abl if a['preset'] == 'oracle'), None)
    tt = res.get('term_typing_mindf2.json')

    out = ["#### Term typing: 315 graph terms, gold spans given\n",
           "| System | Gold types used? | Accuracy [95% CI] | Set F1 (LLMs4OL) | Macro-F1 | "
           "Acc@5 | Hier. F1 |",
           "|---|---|---|---|---|---|---|"]
    ref = off['reference_majority_class']['accuracy']
    out.append(f"| Majority-class reference | – | {fmt(ref)} | – | – | – | – |")
    out.append(typing_row('**Pipeline v6, LLM off**', 'no', off['typing']))
    if on:
        out.append(typing_row('**Pipeline v6, LLM on** (RAG re-ranking)', 'no', on['typing']))
    if oracle:
        out.append(f"| Pipeline v6, ORACLE (upper bound) | **yes** | "
                   f"{fmt(oracle['typing_accuracy'])} "
                   f"{ci_text(oracle['typing_accuracy_ci95'])} | {fmt(oracle['typing_set_f1'])} | "
                   f"{fmt(oracle['typing_macro_f1'])} | {fmt(oracle['typing_acc@5'])} | "
                   f"{fmt(oracle['typing_h_f1'])} |")
    if v5off:
        out.append(typing_row('v5 outputs re-scored, LLM off', '**yes** (leak)', v5off['typing'],
                              f" (coverage {v5off['typing']['coverage']:.2f})"))
    if v5on:
        out.append(typing_row('v5 outputs re-scored, LLM on', '**yes** (leak)', v5on['typing'],
                              f" (coverage {v5on['typing']['coverage']:.2f})"))

    if tt:
        out.append("\n#### Term-typing baselines (`term_typing_baseline.py`): 5-fold grouped CV, "
                   "mean ± sd over folds\n")
        out.append("| Method | Setting | Accuracy | Set F1 | Macro-F1 | Acc@5 | Hier. F1 | "
                   "Acc. on classes unseen in training |")
        out.append("|---|---|---|---|---|---|---|---|")
        for m, r in tt['methods'].items():
            mean, sd = r['mean_over_folds'], r['std_over_folds']
            runs = r['n_fold_runs']
            out.append(f"| {m} | {r['setting']} ({runs} folds) | {fmt(mean['accuracy'])} ± "
                       f"{fmt(sd['accuracy'])} | "
                       f"{fmt(mean['set_f1'])} | {fmt(mean['macro_f1'])} | {fmt(mean['acc@5'])} | "
                       f"{fmt(mean['h_f1'])} | {fmt(mean['acc_unseen_classes'])} |")
    return out


def other_levels_table(res):
    off, on = res.get('eval_pipeline_llm_off.json'), res.get('eval_pipeline_llm_on.json')
    v5off = res.get('eval_v5_rescored_llm_off.json')
    rel_off, rel_on = res.get('relation_eval_llm_off.json'), res.get('relation_eval_llm_on.json')
    return [
        "\n#### Other levels: old metric vs leakage-free metric\n",
        "| Level | Old v5 metric (invalid) | Leakage-free metric | v5 outputs | v6 LLM off | "
        "v6 LLM on |",
        "|---|---|---|---|---|---|",
        "| Type | Type F1 0.997 | – (dataset property: 177/177 labels are class names) "
        "| – | – | – |",
        f"| Term | Term F1@0.5 0.893 (0.935 LLM on) | alignment pairwise F1@0.5 | "
        f"{fmt(dig(v5off, 'alignment_rows', 'thresholds', '0.5', 'f1'))} | "
        f"{fmt(dig(off, 'alignment_rows', 'thresholds', '0.5', 'f1'))} | "
        f"{fmt(dig(on, 'alignment_rows', 'thresholds', '0.5', 'f1'))} |",
        f"| Concept | Concept F1 0.447 | clustering ARI / B-cubed F1 | "
        f"{fmt(dig(v5off, 'clustering', 'ari'))} / {fmt(dig(v5off, 'clustering', 'bcubed_f1'))} "
        f"(gold-type clusters) | "
        f"{fmt(dig(off, 'clustering', 'ari'))} / {fmt(dig(off, 'clustering', 'bcubed_f1'))} | "
        f"{fmt(dig(on, 'clustering', 'ari'))} / {fmt(dig(on, 'clustering', 'bcubed_f1'))} |",
        f"| Taxonomy | – | class-level edge F1 (precision vs closure; chance "
        f"{fmt(dig(off, 'taxonomy', 'random_precision_closure'))}) | "
        f"{fmt(dig(v5off, 'taxonomy', 'f1'))} "
        f"({fmt(dig(v5off, 'taxonomy', 'precision_closure'))}) | "
        f"{fmt(dig(off, 'taxonomy', 'f1'))} ({fmt(dig(off, 'taxonomy', 'precision_closure'))}) | "
        f"{fmt(dig(on, 'taxonomy', 'f1'))} ({fmt(dig(on, 'taxonomy', 'precision_closure'))}) |",
        f"| Relations | partial-entity recall 0.925 | class-level F1 vs 70 restrictions | – | "
        f"{fmt(dig(rel_off, 'predicted_typing', 'f1'))} | "
        f"{fmt(dig(rel_on, 'predicted_typing', 'f1'))} |",
    ]


def paired_delta(a):
    d, p = a.get('typing_diff_vs_base'), a.get('typing_mcnemar_p')
    if d is None or p is None:
        return '–'
    return f"{d:+.3f} ({'<0.001' if p < 0.001 else f'{p:.2f}'})"


def ablation_table(res):
    abl = res.get('ablation_summary.json') or []
    if not abl:
        return []
    out = ["\n#### Ablations (`run_ablation.py`, LLM off; oracle = upper bound)\n",
           "Δ is the paired typing-accuracy difference to `full_pipeline` on the same 315 terms; "
           "p is the exact McNemar test.\n",
           "| Preset | Typing acc. | Δ vs full (p) | Set F1 | Macro-F1 | Hier. F1 | Clust. ARI | "
           "Taxonomy P (closure) | Relation class-F1 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for a in abl:
        name = a['preset'] + (' *(oracle)*' if a['oracle'] else '')
        out.append(f"| {name} | {fmt(a['typing_accuracy'])} | {paired_delta(a)} | "
                   f"{fmt(a['typing_set_f1'])} | {fmt(a['typing_macro_f1'])} | "
                   f"{fmt(a['typing_h_f1'])} | {fmt(a['clustering_ari'])} | "
                   f"{fmt(a['taxonomy_precision_closure'])} | {fmt(a['relations_class_f1'])} |")
    return out


def ner_protocol_table(res):
    rows = []
    for suffix, label in NER_PROTOCOLS:
        for g in GRANULARITIES:
            r = res.get(f'ner_results_{g}{suffix}.json')
            if r and r.get('runs') and r.get('summary'):
                rows.append((suffix, label, g, r))
    if not rows:
        return []
    out = ["\n#### NER (MatSciBERT, MaterioMiner protocol, seqeval entity F1)\n",
           "Mean ± population SD over runs, and the 95% Student-t CI of the mean. Micro F1 is the "
           "primary metric: the authors' 2025 follow-up states micro averaging, while the 2024 "
           "paper only says \"averaging over the five random initializations and entity types\". "
           "The published numbers come from one unpublished random split and have no reported "
           "SD.\n",
           "| Protocol | Granularity | Runs | Micro F1 [95% CI] | Macro F1 | Weighted F1 | "
           "Strict micro | Published |",
           "|---|---|---|---|---|---|---|---|"]

    def cell(m):
        lo, hi = m['ci95']
        return f"{m['mean']:.2f} ± {m['pop_sd']:.2f} [{lo:.2f}, {hi:.2f}]"

    for suffix, label, g, r in rows:
        s = r['summary']
        n = s['n_runs']
        runs = f"{n} papers" if suffix == '_leave_one_paper_out' else f"{n} seeds"
        published = ('– (out of distribution)' if suffix == '_leave_one_paper_out'
                     else f"{r['paper_test_f1']:.2f}")
        out.append(f"| {label} | {g} | {runs} | {cell(s['micro'])} | {s['macro']['mean']:.2f} | "
                   f"{s['weighted']['mean']:.2f} | {s['micro_strict']['mean']:.2f} | "
                   f"{published} |")
    return out


def comparison_table(res, variants, reference, title, intro):
    """Paired-comparison table of `variants` against `reference` ([] if no row is complete)."""
    rows = selected_comparisons(res.comparisons(), variants, reference)
    if not rows:
        return []
    out = [title, intro, COMPARISON_COLUMNS, "|---|---|---|---|---|---|---|---|"]
    for key, c in rows:
        v = (res.get(f'{key}.json') or {}).get('summary', {}).get('micro')
        vcell = f"{v['mean']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]" if v else '–'
        out.append(f"| {variants[c['variant']]} | {PROTOCOL_NAMES[c['protocol']]} | "
                   f"{c['granularity']} | {c['n_pairs']} | {vcell} | {delta_cell(c['micro'])} | "
                   f"{delta_cell(c['macro'])} | {c['micro_strict']['mean_diff']:+.2f} |")
    return out


def ner_variants_table(res):
    """cw and crf against the baseline."""
    return comparison_table(
        res, BASELINE_VARIANTS, 'baseline',
        "\n#### NER training variants vs the baseline (paired by split and model seed)\n",
        "Δ = variant − baseline on the same split and model seed, with the 95% Student-t CI of "
        "the mean difference. Pre-registered rule: a variant improves on the baseline only if "
        "the micro-F1 interval over the 5 random splits lies entirely above 0.\n")


THREAD_NOTE = ("Week-2 runs use 2 CPU threads and the CRF head 3. With the same seeds, the "
               "thread count alone changes a run (floating-point summation order), which adds "
               "noise to each difference.")


def thread_note(res):
    """THREAD_NOTE, plus the measured effect once the `crf_m0` comparisons are complete.

    `crf_m0` is the CRF head re-run with 2 threads on the same splits and model seeds, so
    its difference to the CRF head is the effect of the thread count alone.
    """
    comps = res.comparisons()
    found = [(g, find_comparison(comps, g, 'random_split_vary', 'crf_m0', 'crf'))
             for g in GRANULARITIES]
    parts = [f"{g}-grained {delta_cell(c['micro'])}" for g, c in found if c]
    if not parts:
        return THREAD_NOTE
    return (f"{THREAD_NOTE} Re-running the CRF head with 2 threads (ensemble member `crf_m0`) "
            f"changes micro F1 by {' and '.join(parts)} over the 5 random splits.")


def ner_week2_table(res):
    """Multi-task, domain-adaptive pre-training and the 3-model ensemble against the CRF head."""
    return comparison_table(
        res, WEEK2_VARIANTS, 'crf',
        "\n#### NER week-2 variants vs the CRF head (paired by split and model seed)\n",
        "Δ = variant − CRF head on the same split and model seed, with the 95% Student-t CI of "
        "the mean difference. Pre-registered rule: a variant improves on the CRF head only if "
        f"the micro-F1 interval over the 5 random splits lies entirely above 0. "
        f"{thread_note(res)}\n")


WEEK3_INTRO = (
    "Δ = variant − reference on the same split, where a model trained with 3 model seeds (K, "
    "K + 100 and K + 200 on split K; 0, 100 and 200 on a held-out paper) scores their mean, the "
    "ensembles and the one-seed SciBERT and PubMedBERT runs score once, and the CRF head is the "
    "mean of `crf_m0`, `crf_m100` and `crf_m200`; "
    "pre-registered rule (week-3 protocol): a variant improves on its reference only if the 95% "
    "Student-t CI of the mean micro-F1 difference lies entirely above 0, over the 5 random splits "
    "for the mixed-backbone ensemble (against the 3-model seed ensemble), multi-task and "
    "in-domain pre-training, and over the 4 held-out papers for label semantics and "
    "self-training; every other row is reported without a decision.\n")


def week3_rows(comps):
    """The complete week-3 comparisons: by WEEK3_COMPARISONS, then protocol, then granularity."""
    rows = []
    for variant, reference in WEEK3_COMPARISONS:
        for protocol in ('random_split_vary', 'leave_one_paper_out'):
            for g in GRANULARITIES:
                c = find_comparison(comps, g, protocol, variant, reference)
                if c:
                    rows.append(c)
    return rows


def ner_week3_table(res):
    """Week-3 comparisons of 3-seed means ([] if none is complete)."""
    rows = week3_rows(res.comparisons(WEEK3_FILE))
    if not rows:
        return []
    out = ["\n#### NER week-3 comparisons (3-seed means)\n", WEEK3_INTRO, WEEK3_COLUMNS,
           "|---|---|---|---|---|---|---|---|"]
    for c in rows:
        v = dig(c, 'variant_scores', 'micro')
        out.append(f"| {WEEK3_NAMES[c['variant']]} vs {WEEK3_NAMES[c['reference']]} | "
                   f"{PROTOCOL_NAMES[c['protocol']]} | {c['granularity']} | {c['n_pairs']} | "
                   f"{interval(v) if v else '–'} | {delta_cell(c['micro'])} | "
                   f"{delta_cell(c['macro'])} | {c['micro_strict']['mean_diff']:+.2f} |")
    return out


WEEK4_INTRO = (
    "Δ = variant − reference on the same split, scored as in week 3: a model trained with 3 model "
    "seeds scores their mean, a vote over them scores once. Pre-registered rule (week-4 protocol): "
    "a change improves on its reference only if the 95% Student-t CI of the mean micro-F1 "
    "difference lies entirely above 0. The CPU rows are votes over the saved predictions of the "
    "week-3 runs, decided over the 4 held-out papers against the 3-model seed ensemble. The GPU "
    "rows are runs on Kaggle T4 GPUs, compared only with GPU runs: the multi-task vote against "
    "the GPU seed ensemble over the 5 random splits, label semantics on the multi-task model "
    "against the GPU CRF head over the 5 random splits and over the 4 held-out papers, and "
    "self-training with 3 teachers against the GPU CRF head over the 4 held-out papers. The "
    "other GPU rows are reported without a decision: the other protocol, comparisons with each "
    "change's parent and with the GPU seed ensemble, and repeats of the week-3 comparisons and "
    "of the CPU votes.\n")


def week4_rows(res):
    """(runs, entry) of the complete week-4 comparisons: the CPU file's, then the GPU file's,
    each by its comparison list, then protocol, then granularity (descriptive entries apart)."""
    rows = []
    for runs, name, comparisons in (('CPU', WEEK4_FILE, WEEK4_CPU_COMPARISONS),
                                    ('GPU', GPU_PREFIX + WEEK4_FILE, WEEK4_GPU_COMPARISONS)):
        comps = {k: c for k, c in res.comparisons(name).items() if c.get('role') != 'descriptive'}
        for variant, reference in comparisons:
            for protocol in ('random_split_vary', 'leave_one_paper_out'):
                for g in GRANULARITIES:
                    c = find_comparison(comps, g, protocol, variant, reference)
                    if c:
                        rows.append((runs, c))
    return rows


def ner_week4_table(res):
    """Week-4 comparisons ([] if none is complete)."""
    rows = week4_rows(res)
    if not rows:
        return []
    out = ["\n#### NER week-4 comparisons (3-seed means; CPU votes, GPU runs)\n", WEEK4_INTRO,
           WEEK4_COLUMNS, "|---|---|---|---|---|---|---|---|---|"]
    for runs, c in rows:
        v = dig(c, 'variant_scores', 'micro')
        out.append(f"| {WEEK4_NAMES[c['variant']]} vs {WEEK4_NAMES[c['reference']]} | {runs} | "
                   f"{PROTOCOL_NAMES[c['protocol']]} | {c['granularity']} | {c['n_pairs']} | "
                   f"{interval(v) if v else '–'} | {delta_cell(c['micro'])} | "
                   f"{delta_cell(c['macro'])} | {c.get('role', 'reported')} |")
    return out


WEEK5_INTRO = (
    "Δ = variant − reference on the same split, GPU runs only, with 10 model seeds per split: a "
    "model scores the mean of its 10 runs, a vote the mean of its votes over 3 disjoint seed "
    "triples (seeds 0–200, 300–500 and 600–800). Pre-registered rule (week-5 protocol): a change "
    "improves on its reference only if the 95% Student-t CI of the mean micro-F1 difference lies "
    "entirely above 0. Decisions: the claims of weeks 3 and 4 re-tested (multi-task over the 5 "
    "random splits; label semantics and self-training with 3 teachers over the 4 held-out "
    "papers), and each vote against the 3-model seed ensemble on both protocols (the reported "
    "system). The other rows are reported without a decision.\n")


def week5_rows(res):
    """The complete week-5 comparisons, by the protocol's order, then protocol, then
    granularity."""
    comps = res.comparisons(GPU_PREFIX + WEEK5_FILE)
    return [c for variant, reference in WEEK5_COMPARISONS
            for protocol in ('random_split_vary', 'leave_one_paper_out') for g in GRANULARITIES
            for c in [find_comparison(comps, g, protocol, variant, reference)] if c]


def ner_week5_table(res):
    """Week-5 comparisons ([] if none is complete)."""
    rows = week5_rows(res)
    if not rows:
        return []
    out = ["\n#### NER week-5 comparisons (10-seed means, GPU runs)\n", WEEK5_INTRO,
           WEEK3_COLUMNS.replace('Δ strict micro', 'Role'), "|---|---|---|---|---|---|---|---|"]
    for c in rows:
        v = dig(c, 'variant_scores', 'micro')
        out.append(f"| {WEEK5_NAMES[c['variant']]} vs {WEEK5_NAMES[c['reference']]} | "
                   f"{PROTOCOL_NAMES[c['protocol']]} | {c['granularity']} | {c['n_pairs']} | "
                   f"{interval(v) if v else '–'} | {delta_cell(c['micro'])} | "
                   f"{delta_cell(c['macro'])} | {c.get('role', 'reported')} |")
    return out


WEEK6_SCREEN_INTRO = (
    "Δ = method − reference on the same held-out paper, GPU runs only, with seeds 0, 100 and 200: "
    "a model scores the mean of its 3 runs, a vote over them scores once. Week-6 protocol, stage "
    "1: a method goes to stage 2 (10 seeds) if its mean micro-F1 difference against the CRF head "
    "is above 0 at either granularity. The screen decides nothing.\n")
WEEK6_INTRO = (
    "Δ = method − reference on the same held-out paper, GPU runs only, with 10 model seeds per "
    "paper: a model scores the mean of its 10 runs, a vote the mean of its votes over 3 disjoint "
    "seed triples. Pre-registered rule (week-6 protocol): a change improves on its reference only "
    "if the 95% Student-t CI of the mean micro-F1 difference lies entirely above 0. Decisions: "
    "each method against the CRF head, and each method's vote against the strongest system on "
    "unseen papers (the multi-task vote fine-grained, the 3-model seed ensemble "
    "coarse-grained).\n")


def ner_week6_table(res, name, title, intro, comparisons):
    """A week-6 table over the held-out papers ([] if no comparison is complete)."""
    comps = res.comparisons(GPU_PREFIX + name)
    rows = [c for variant, reference in comparisons for g in GRANULARITIES
            for c in [find_comparison(comps, g, LOPO, variant, reference)] if c]
    if not rows:
        return []
    out = [f"\n#### {title}\n", intro,
           WEEK3_COLUMNS.replace('Δ strict micro', 'Role').replace('| Protocol ', ''),
           "|---|---|---|---|---|---|---|"]
    for c in rows:
        v = dig(c, 'variant_scores', 'micro')
        out.append(f"| {WEEK6_NAMES[c['variant']]} vs {WEEK6_NAMES[c['reference']]} | "
                   f"{c['granularity']} | {c['n_pairs']} | {interval(v) if v else '–'} | "
                   f"{delta_cell(c['micro'])} | {delta_cell(c['macro'])} | "
                   f"{c.get('role', 'reported')} |")
    return out


WEEK7_INTRO = (
    "Δ = with the rule − without it, on the same split, GPU runs only, 10 model seeds: a model "
    "scores the mean of its 10 runs, a vote the mean of its votes over 3 disjoint seed triples. "
    "Week-7 protocol, part 1: the consistency rule (`ner_consistency.py`, fixed before it was "
    "applied) on the saved test predictions. Pre-registered rule: a change improves on its "
    "reference only if the 95% Student-t CI of the mean micro-F1 difference lies entirely above "
    "0. Decisions, on the 4 held-out papers: the rule on the CRF head, and on the strongest "
    "system on unseen papers (the multi-task vote fine-grained, the 3-model seed ensemble "
    "coarse-grained). The other rows are reported without a decision.\n")


WEEK7A_INTRO = (
    "Δ = with the bias − without it, on the same split, GPU runs only, 10 model seeds: a model "
    "scores the mean of its 10 runs, a vote the mean of its votes over 3 disjoint seed triples. "
    "Week-7 protocol, part 2a: recall calibration, a bias on the CRF head's emission of every "
    "tag but O before decoding, chosen per held-out paper on inner folds that never score that "
    "paper (`ner_bias.py`). Pre-registered rule: a change improves on its reference only if the "
    "95% Student-t CI of the mean micro-F1 difference lies entirely above 0. Decisions, on the "
    "4 held-out papers: the calibrated CRF head against the CRF head, and its vote against the "
    "strongest system on unseen papers after week 5 (the multi-task vote fine-grained, the "
    "3-model seed ensemble coarse-grained). The other rows are reported without a decision.\n")


WEEK7C_INTRO = (
    "Δ = with the rule − without it, fine-grained, on the same held-out paper, GPU runs only, "
    "10 model seeds: a model scores the mean of its 10 runs, a vote the mean of its votes over "
    "3 disjoint seed triples. Week-7 protocol, part 2c: coarse-to-fine completion "
    "(`ner_coarse_to_fine.py`, fixed before it was applied): a coarse-grained predicted entity "
    "of a type that maps to one fine-grained type in the fold's training papers becomes a "
    "fine-grained entity where the fine level predicts only O. Pre-registered rule: a change "
    "improves on its reference only if the 95% Student-t CI of the mean micro-F1 difference "
    "lies entirely above 0. Decisions: the rule on the CRF head and on the multi-task vote; "
    "the multi-task model is reported. Tested on MaterioMiner only (SOFC-Exp's slot level was "
    "not yet converted).\n")


WEEK9_INTRO = (
    "Δ = with the change − without it, on the same held-out paper, GPU runs only, 10 model "
    "seeds: a model scores the mean of its 10 runs, a vote the mean of its votes over 3 "
    "disjoint seed triples. Week-9 protocol: mention replacement on the multi-task model "
    "(`--multi-task --augment-mentions 3`; a mention is replaced in both heads' tags by a "
    "training mention of the same coarse- and fine-grained type). Pre-registered rule: a "
    "change improves on its reference only if the 95% Student-t CI of the mean micro-F1 "
    "difference lies entirely above 0. Decisions, on the 4 held-out papers: the combination "
    "against the multi-task model, and its vote against the strongest system on unseen papers "
    "(the multi-task vote fine-grained, the 3-model seed ensemble coarse-grained). The other "
    "rows are reported without a decision.\n")


WEEK11_INTRO = (
    "Δ = with the change − without it, on the same held-out paper, GPU runs only, 10 model "
    "seeds: a model scores the mean of its 10 runs, a vote the mean of its votes over 3 "
    "disjoint seed triples. Week-11 protocol: mention replacement whose replacement pool also "
    "holds the mentions of the sentences that 3 CRF teachers tag alike and confidently in the "
    "40,000-sentence fatigue corpus (`--augment-names`). Pre-registered rule: a change improves "
    "on its reference only if the 95% Student-t CI of the mean micro-F1 difference lies "
    "entirely above 0. Decision on MaterioMiner, per granularity on the 4 held-out papers: "
    "against mention replacement (do the silver names add?); its MuLMS decisions are in the "
    "week-10 bullet. The other rows are reported without a decision.\n")

WEEK12_INTRO = (
    "Δ = with the change − without it, on the same held-out paper, GPU runs only, 10 model "
    "seeds: a model scores the mean of its 10 runs, a vote the mean of its votes over 3 "
    "disjoint seed triples. Week-12 protocol, the MaterioMiner analogue of the screen's "
    "winner: the 3 CRF teachers tag the 40,000-sentence fatigue corpus and the names of the "
    "spans they tag alike and confidently (`--silver-mentions`) enter the student's training "
    "in the winner's way (the replacement pool, 3 or 6 copies; or a match feature of the CRF "
    "head). Pre-registered rule: a change improves on its reference only if the 95% Student-t "
    "CI of the mean micro-F1 difference lies entirely above 0. Decision per granularity on the "
    "4 held-out papers: against the CRF head; the MuLMS screen and decisions are in the "
    "week-10 bullet. The other rows are reported without a decision.\n")


def ner_week7_table(res, comparisons=WEEK7_COMPARISONS,
                    title='part 1 (the consistency rule, 10-seed means, GPU runs)',
                    intro=WEEK7_INTRO, file=WEEK7_FILE, week='7'):
    """Week-7 comparisons of one part ([] if none is complete): the held-out papers first.
    file and week serve the later weeks."""
    comps = res.comparisons(GPU_PREFIX + file)
    rows = [c for variant, reference in comparisons
            for protocol in (LOPO, 'random_split_vary') for g in GRANULARITIES
            for c in [find_comparison(comps, g, protocol, variant, reference)] if c]
    if not rows:
        return []
    out = [f"\n#### NER week-{week} comparisons, {title}\n", intro,
           WEEK3_COLUMNS.replace('Δ strict micro', 'Role'), "|---|---|---|---|---|---|---|---|"]
    for c in rows:
        v = dig(c, 'variant_scores', 'micro')
        out.append(f"| {WEEK7_SYSTEM_NAMES[c['variant']]} vs "
                   f"{WEEK7_SYSTEM_NAMES[c['reference']]} | {PROTOCOL_NAMES[c['protocol']]} | "
                   f"{c['granularity']} | {c['n_pairs']} | {interval(v) if v else '–'} | "
                   f"{delta_cell(c['micro'])} | {delta_cell(c['macro'])} | "
                   f"{c.get('role', 'reported')} |")
    return out


def build_tables(res):
    """All tables between the "At a glance" table and "Reading the numbers"."""
    out = (term_typing_tables(res) + other_levels_table(res) + ablation_table(res)
           + ner_protocol_table(res) + ner_variants_table(res) + ner_week2_table(res)
           + ner_week3_table(res) + ner_week4_table(res) + ner_week5_table(res)
           + ner_week6_table(res, WEEK6_SCREEN_FILE,
                             'NER week-6 screen (3-seed means, held-out papers, GPU runs)',
                             WEEK6_SCREEN_INTRO, WEEK6_SCREEN_COMPARISONS)
           + ner_week6_table(res, WEEK6_FILE,
                             'NER week-6 comparisons (10-seed means, held-out papers, GPU runs)',
                             WEEK6_INTRO, WEEK6_COMPARISONS)
           + ner_week7_table(res)
           + ner_week7_table(res, WEEK7A_COMPARISONS,
                             'part 2a (recall calibration, 10-seed means, GPU runs)',
                             WEEK7A_INTRO)
           + ner_week7_table(res, WEEK7C_COMPARISONS,
                             'part 2c (coarse-to-fine completion, 10-seed means, GPU runs)',
                             WEEK7C_INTRO)
           + ner_week7_table(res, WEEK9_COMPARISONS,
                             'mention replacement on the multi-task model (10-seed means, GPU '
                             'runs)', WEEK9_INTRO, file=WEEK9_FILE, week='9')
           + ner_week7_table(res, WEEK11_COMPARISONS,
                             'mention replacement with silver names (10-seed means, GPU runs)',
                             WEEK11_INTRO, file=WEEK11_FILE, week='11')
           + ner_week7_table(res, WEEK12_COMPARISONS,
                             "the MaterioMiner analogue of the screen's winner (10-seed means, "
                             'GPU runs)', WEEK12_INTRO, file=WEEK12_FILE, week='12'))
    return '\n'.join(out).strip()


# --------------------------------------------------------------------------------
# "Reading the numbers" notes
# --------------------------------------------------------------------------------
def pending_note(pending):
    """Note for runs that were still going when the snapshot was taken ('' if none)."""
    if not pending:
        return ''
    return ('\n\n> **Still running when this snapshot was taken:** ' + ', '.join(pending) +
            '. These rows will be added when the runs finish.')


def ablation_note(res):
    a = {row['preset']: row for row in (res.get('ablation_summary.json') or [])}
    if not a:
        return ''
    return (
        f"* Ablations: no component changes typing significantly. The GNN adds "
        f"{a['full_pipeline']['typing_accuracy'] - a['no_gnn']['typing_accuracy']:.3f} "
        f"({a['full_pipeline']['typing_accuracy']:.3f} vs {a['no_gnn']['typing_accuracy']:.3f}, "
        f"McNemar p = {a['no_gnn']['typing_mcnemar_p']:.2f}), and the whole pipeline beats the "
        f"plain-SBERT baseline by {-a['baseline_sbert']['typing_diff_vs_base']:.3f} "
        f"(p = {a['baseline_sbert']['typing_mcnemar_p']:.2f}). The GNN embeddings *hurt* "
        f"clustering "
        f"(ARI {a['full_pipeline']['clustering_ari']:.3f} vs {a['no_gnn']['clustering_ari']:.3f} "
        f"with plain SBERT). The oracle preset (gold types allowed) reaches "
        f"{a['oracle']['typing_accuracy']:.3f}, which is what the v5 evaluation was effectively "
        f"reporting.\n")


def verdict_line(res, comps, variant, name, reference, versus, sota=False):
    """One "Reading the numbers" bullet for a variant ('' if no comparison is complete).

    Verdicts are the paired micro-F1 differences (improves / hurts / no significant
    change, with strict IOB2 numbers) over the 5 random splits and leaving one paper
    out. `versus` completes "vs ..." in the bullet. `sota` appends the pre-registered
    check against the published number, which only makes sense against the baseline.
    """
    split_parts, ood_parts, beaten = [], [], []
    for g in GRANULARITIES:
        c = find_comparison(comps, g, 'random_split_vary', variant, reference)
        if c:
            word, num = paired_verdict(c)
            split_parts.append(f"{g}-grained {word} ({num})")
            v = res.get(ner_file(g, 'random_split_vary', variant))
            if (sota and word == 'improves' and v
                    and v['summary']['micro']['ci95'][0] > v['paper_test_f1']):
                sm = v['summary']['micro']
                beaten.append(f"{g}-grained {sm['mean']:.2f} [{sm['ci95'][0]:.2f}, "
                              f"{sm['ci95'][1]:.2f}] vs {v['paper_test_f1']:.2f}")
        c = find_comparison(comps, g, 'leave_one_paper_out', variant, reference)
        if c:
            word, num = paired_verdict(c)
            ood_parts.append(f"{g}-grained {word} ({num})")
    if not (split_parts or ood_parts):
        return ''
    line = f"* NER, {name} vs {versus} (paired micro-F1 difference): "
    line += ('; '.join(split_parts) + ' across the 5 random splits' if split_parts else '')
    if ood_parts:
        line += ('. Leaving one paper out: ' if split_parts else 'leaving one paper out: ')
        line += '; '.join(ood_parts)
    if beaten:
        line += ('. Pre-registered SOTA check (a paired gain over the baseline and a 95% CI '
                 'above the published number): met for ' + '; '.join(beaten)
                 + ', if the paper reports micro F1')
    return line + '.\n'


def mean_precision_recall(result):
    n = len(result['runs'])
    return (sum(r['precision'] for r in result['runs']) / n,
            sum(r['recall'] for r in result['runs']) / n)


def crf_explanation(res, comps, vary):
    """Why the CRF helps (exploratory): default seqeval scoring vs strict IOB2."""
    parts = []
    for g in GRANULARITIES:
        c = find_comparison(comps, g, 'random_split_vary', 'crf', 'baseline')
        b, v = vary[g], res.get(ner_file(g, 'random_split_vary', 'crf'))
        if not (c and b and v):
            continue
        (bp, br), (vp, vr) = mean_precision_recall(b), mean_precision_recall(v)
        bs, st = b['summary']['micro_strict'], c['micro_strict']
        slo, shi = st['ci95']
        if slo > 0:
            reading = (f"under strict IOB2 scoring the CRF still gains {st['mean_diff']:+.2f} "
                       f"[{slo:+.2f}, {shi:+.2f}], so part of the gain is better recognition")
        else:
            reading = (f"under strict IOB2 scoring the two are level ({st['mean_diff']:+.2f} "
                       f"[{slo:+.2f}, {shi:+.2f}]), so the gain is span validity alone. Dropping "
                       "invalid tags from the baseline's output gives the same benefit for free: "
                       f"its strict F1 is {bs['mean']:.2f} [{bs['ci95'][0]:.2f}, "
                       f"{bs['ci95'][1]:.2f}] against the published {b['paper_test_f1']:.2f}")
        parts.append(f"{g.capitalize()}-grained: precision {bp:.1f} → {vp:.1f}, recall "
                     f"{br:.1f} → {vr:.1f}; {reading}")
    if not parts:
        return ''
    return ("* Why the CRF helps (exploratory, not pre-registered): it never predicts an invalid "
            "I- tag, while seqeval's default (conlleval) scoring counts the baseline's stray I- "
            "tags as extra, mostly wrong entities; strict IOB2 scoring ignores them. "
            + '. '.join(parts) + ".\n")


def ner_notes(res):
    """The NER bullets of "Reading the numbers"."""
    fixed, vary, lopo = baseline_ner(res)
    comps = res.comparisons()
    note = ''
    if all(has_runs(fixed[g], 1) for g in GRANULARITIES):
        f_, c_ = fixed['fine']['summary'], fixed['coarse']['summary']
        note += (
            f"* NER, fixed split: plain MatSciBERT reaches micro F1 "
            f"{f_['micro']['mean']:.2f} (fine) "
            f"and {c_['micro']['mean']:.2f} (coarse) over 5 seeds, against the published 69.92 / "
            f"72.32. Macro F1 is much lower ({f_['macro']['mean']:.2f} / "
            f"{c_['macro']['mean']:.2f}), "
            f"so the comparison favours this repo only if the paper's \"averaging over ... entity "
            f"types\" means micro averaging, as in the authors' 2025 follow-up.\n")
    if all(has_runs(vary[g], 5) for g in GRANULARITIES):
        verdicts = []
        for g in GRANULARITIES:
            s, paper = vary[g]['summary']['micro'], vary[g]['paper_test_f1']
            lo, hi = s['ci95']
            if lo > paper:
                v = f"holds: {s['mean']:.2f}, 95% CI [{lo:.2f}, {hi:.2f}], above {paper:.2f}"
            elif hi < paper:
                v = f"reverses: {s['mean']:.2f}, 95% CI [{lo:.2f}, {hi:.2f}], below {paper:.2f}"
            else:
                v = (f"is not significant: {s['mean']:.2f}, 95% CI [{lo:.2f}, {hi:.2f}] "
                     f"includes {paper:.2f}")
            verdicts.append(f"{g}-grained the gain {v}")
        note += ("* NER across splits (`--vary-split`, a new random split per seed, micro F1; "
                 "pre-registered rule: the gain holds if the 95% CI lies above the published "
                 "number): " + '; '.join(verdicts) + ". Macro F1 across splits is "
                 f"{vary['fine']['summary']['macro']['mean']:.2f} (fine) and "
                 f"{vary['coarse']['summary']['macro']['mean']:.2f} (coarse).\n")
    else:
        note += ("* NER across splits: the check with a new random split per seed "
                 "(`--vary-split`) and with one paper held out (`--leave-one-paper-out`) is "
                 "running. Until it finishes, the fixed-split gain is not a SOTA claim.\n")
    if all(has_runs(lopo[g], 4) for g in GRANULARITIES):
        parts = []
        for g in GRANULARITIES:
            s = lopo[g]['summary']['micro']
            parts.append(f"{s['mean']:.2f} {g}-grained (per paper "
                         f"{min(s['per_run']):.1f}–{max(s['per_run']):.1f})")
        note += ("* NER out of distribution (`--leave-one-paper-out`: train on three papers, "
                 "test on the fourth): micro F1 " + ' and '.join(parts) + ". These numbers are "
                 "not comparable with 69.92 / 72.32.\n")

    for var, name in BASELINE_VARIANTS.items():
        note += verdict_line(res, comps, var, name, 'baseline', 'the baseline', sota=True)
    note += crf_explanation(res, comps, vary)
    for var, name in WEEK2_VARIANTS.items():
        note += verdict_line(res, comps, var, name, 'crf', 'the CRF head')
    note += ensemble_member_note(res)
    week3 = res.comparisons(WEEK3_FILE)
    for variant, reference in WEEK3_COMPARISONS:
        note += week3_verdict_line(week3, variant, reference)
    for name, variant, reference, runs in WEEK4_VERDICTS:
        note += week4_verdict_line(res.comparisons(name), variant, reference, runs)
    note += hardware_note(res)
    week5 = res.comparisons(GPU_PREFIX + WEEK5_FILE)
    for variant, reference in WEEK5_COMPARISONS:
        note += week4_verdict_line(week5, variant, reference, 'GPU', week=5)
    analysis = res.get(GPU_PREFIX + WEEK5_ANALYSIS) or {}
    note += (published_split_note(analysis) + seed_triples_note(analysis)
             + rare_types_note(res, analysis))
    screen = res.comparisons(GPU_PREFIX + WEEK6_SCREEN_FILE)
    for method in WEEK6_METHODS:
        note += week6_screen_line(screen, method)
    week6 = res.comparisons(GPU_PREFIX + WEEK6_FILE)
    for variant, reference in WEEK6_COMPARISONS:
        note += week6_verdict_line(week6, variant, reference, '10 seeds')
    week7 = res.comparisons(GPU_PREFIX + WEEK7_FILE)
    for variant, reference in WEEK7_DECISIONS:
        note += week6_verdict_line(week7, variant, reference,
                                   'the consistency rule on the saved predictions of 10 seeds',
                                   week='7, part 1', names=WEEK7_SYSTEM_NAMES)
    for variant, reference in WEEK7A_DECISIONS:
        note += week6_verdict_line(week7, variant, reference,
                                   'recall calibration, its bias chosen on inner folds, 10 seeds',
                                   week='7, part 2a', names=WEEK7_SYSTEM_NAMES)
    for variant, reference in WEEK7C_DECISIONS:
        note += week6_verdict_line(week7, variant, reference,
                                   'coarse-to-fine completion on the saved predictions of 10 '
                                   'seeds', week='7, part 2c', names=WEEK7_SYSTEM_NAMES)
    screen7 = res.comparisons(GPU_PREFIX + WEEK7_SCREEN_FILE)
    note += week6_screen_line(screen7, 'crf_mr', week='7, part 2b', names=WEEK7_SYSTEM_NAMES)
    for variant, reference in WEEK7B_DECISIONS:
        note += week6_verdict_line(week7, variant, reference, 'mention replacement, 10 seeds',
                                   week='7, part 2b', names=WEEK7_SYSTEM_NAMES)
    week9 = res.comparisons(GPU_PREFIX + WEEK9_FILE)
    for variant, reference in WEEK9_DECISIONS:
        note += week6_verdict_line(week9, variant, reference,
                                   'mention replacement on the multi-task model, 10 seeds',
                                   week='9', names=WEEK7_SYSTEM_NAMES)
    week11 = res.comparisons(GPU_PREFIX + WEEK11_FILE)
    for variant, reference in WEEK11_DECISIONS:
        note += week6_verdict_line(week11, variant, reference,
                                   'mention replacement with silver names, 10 seeds',
                                   week='11', names=WEEK7_SYSTEM_NAMES)
    week12 = res.comparisons(GPU_PREFIX + WEEK12_FILE)
    for variant, reference in WEEK12_DECISIONS:
        note += week6_verdict_line(week12, variant, reference,
                                   "the MaterioMiner analogue of the screen's winner, 10 seeds",
                                   week='12', names=WEEK7_SYSTEM_NAMES)
    return note + sofc_note(analysis) + mulms_note(analysis) + week13_note(analysis)


WEEK13_CORPORA = {'mulms': "MuLMS's 7 held-out subfields",
                  'fine': "MaterioMiner's 4 held-out papers, fine-grained",
                  'coarse': "MaterioMiner's 4 held-out papers, coarse-grained",
                  'sofc_folds': "SOFC-Exp's 5 article folds"}
WEEK13_INPUTS = {'crf': 'the CRF head', 'crf_mr': 'mention replacement',
                 'crf_mt_mr': 'mention replacement on the multi-task model',
                 'crf_mrm6': "mention replacement with the SOFC corpus's silver mentions and 6 "
                             'copies (the week-12 winner\'s analogue)'}
WEEK13_POOLED = {'crf_mr': 'mention replacement',
                 'crf_mt_mr': 'mention replacement on the multi-task model'}


def week13_note(analysis):
    """The week-13 bullet (ROADMAP §4, week 13) from the week-5 analyses file ('' without its
    week13 section): the consistency rule's outputs against their inputs on the held-out
    documents of every corpus that has them (a verdict where the comparison is a decision,
    reported otherwise), the decisions 13.3a and 13.3b on SOFC-Exp's article folds (with
    the multi-task combination's slot level and its difference against mention replacement,
    reported), the week-12 screen's winner, and the pooled estimates across corpora."""
    w = analysis.get('week13') or {}
    consistency = w.get('consistency') or {}
    if not consistency:
        return ''
    line = ("* NER week 13 (the consistency rule of week 7 on the saved 10-seed predictions of "
            "held-out documents, no training; the 95% CIs are over the subfields, papers or "
            "folds)")
    for corpus, entries in consistency.items():
        for variant, e in entries.items():
            system, _, rule = variant.rpartition('_')
            what = WEEK13_INPUTS.get(system, system)
            if rule == 'dcr':
                what += ' with retyping'
            word, numbers = paired_verdict(e['minus_input'])
            verdict = (f"decision {e['decision']}, {word} ({numbers})" if 'decision' in e
                       else f"{numbers} (reported; no decision)")
            line += (f". {WEEK13_CORPORA.get(corpus, corpus)}, on {what}: "
                     f"{interval(e['output']['micro'])} against {interval(e['input']['micro'])}, "
                     f"{e['n_added_per_seed']:.0f} entities added per seed; {verdict}")
    for method, e in (w.get('folds') or {}).items():
        word, numbers = paired_verdict(e['minus_crf'])
        line += (f". SOFC-Exp's {e['n_units']} article folds, decision {e['decision']}, "
                 f"{WEEK13_INPUTS.get(method, method)} against the CRF head, paired by fold: "
                 f"{interval(e['system']['micro'])} against {interval(e['crf']['micro'])}; "
                 f"{word} ({numbers})")
        if 'minus_crf_mr' in e:
            line += (f"; against mention replacement {delta_cell(e['minus_crf_mr']['micro'])} "
                     "(reported; no decision)")
        if 'slot' in e:
            line += f"; its slot level {interval(e['slot']['micro'])} (reported)"
    winner = w.get('week12_winner')
    if winner:
        line += f". The week-12 screen's winner: {WEEK12_NAMES.get(winner, winner)} ({winner})"
    for method, e in (w.get('pooled') or {}).items():
        parts = '; '.join(f"{WEEK13_CORPORA.get(c, c).split(',')[0]} "
                          f"{delta_cell(v['minus_crf'])}{' (loses)' if v['loses'] else ''}"
                          for c, v in e['corpora'].items())
        label = WEEK13_POOLED.get(method, WEEK12_NAMES.get(method, method))
        status = ('decision 13.2' if e['status'] == 'decision'
                  else "reported; 13.3's pooled estimate, no decision"
                  if 'sofc_folds' in e['corpora']
                  else 'reported; post hoc for plain mention replacement until 13.3 runs'
                  if method == 'crf_mr' else 'reported; no decision')
        line += (f". Pooled over {e['n_units']} held-out units, {label} against the CRF head: "
                 f"{delta_cell(e['minus_crf'])} ({parts}), "
                 f"{'passes' if e['passes'] else 'no'}"
                 f"{'' if e['no_corpus_loses'] else ', a corpus loses'} ({status})")
        for key, w in e.items():
            if not key.startswith('with_'):
                continue
            corpus = key[len('with_'):]
            extra = w[corpus]
            line += (f"; with {WEEK13_CORPORA.get(corpus, corpus)} ({extra['variant']}, "
                     f"{delta_cell(extra['minus_crf'])}{' (loses)' if extra['loses'] else ''}): "
                     f"{w['n_units']} units, {delta_cell(w['minus_crf'])}, "
                     f"{'passes' if w['passes'] else 'no'}"
                     f"{'' if w['no_corpus_loses'] else ', a corpus loses'} (reported; the "
                     "three-corpus estimate of 13.4)")
    return line + ".\n"


def mulms_note(analysis):
    """The week-10 bullet (MuLMS, ROADMAP §4) from the week-5 analyses file ('' until the
    held-out subfields have all their runs): its 2 decisions over the held-out subfields, each
    the mean of its 10 seeds, and the CRF head on the random pool splits (descriptive)."""
    m = analysis.get('mulms') or {}
    held = m.get('leave_one_subfield_out')
    if not held:
        return ''
    word, numbers = paired_verdict(held['crf_mr_minus_crf'])
    line = ("* NER week 10 (MuLMS, a third corpus: its outermost entities of 13 types; GPU runs, "
            f"10 model seeds; each of its {held['n_subfields']} materials subfields held out in "
            "turn, 340 training and 60 validation sentences from the others; the 95% CIs are "
            f"over the subfields): the CRF head {interval(held['crf']['micro'])}, mention "
            f"replacement {interval(held['crf_mr']['micro'])}. Decision 1, mention replacement "
            f"against the CRF head, paired by subfield: {word} ({numbers})")
    votes = held.get('votes')
    if votes:
        word, numbers = paired_verdict(votes['crf_mr_minus_crf'])
        line += (". Decision 2, its votes over the 3 seed triples against the CRF votes "
                 f"({interval(votes['crf_mr']['micro'])} against "
                 f"{interval(votes['crf']['micro'])}), paired by subfield: {word} ({numbers})")
    pool = m.get('random_split_pool')
    if pool:
        line += (f". The CRF head on {pool['n_splits']} random splits of the same size "
                 f"(descriptive): {interval(pool['micro'])}")
        if 'gap' in m:
            line += f", {m['gap']['micro']:.2f} points above the held-out subfields"
    names = m.get('silver_names')
    if names:
        line += (". Week 11, mention replacement with the teachers' silver names in its pool "
                 f"(the fold's sentences outside its pool and test set): "
                 f"{interval(names['crf_mrs']['micro'])}")
        for k, reference in ((1, 'the CRF head'), (2, 'mention replacement')):
            word, numbers = paired_verdict(names['minus_crf' if k == 1 else 'minus_crf_mr'])
            line += (f". Week-11 decision {k}, against {reference}, paired by subfield: "
                     f"{word} ({numbers})")
        if names.get('unseen_in_silver') is not None:
            line += (f". {100 * names['unseen_in_silver']:.1f}% of the unseen test names are "
                     "silver names")
    line += week12_sentences(m.get('week12') or {})
    return line + ".\n"


WEEK12_NAMES = {'crf_mrs_topic': 'silver sentences', 'crf_mrm_topic': 'silver mentions',
                'crf_mrm_topic6': 'silver mentions with 6 copies',
                'crf_st3_topic': 'self-training on the silver sentences', 'crf_mr6': '6 copies',
                'crf_gzm_topic': 'silver mentions as a match feature',
                'crf_gzs_topic': 'silver sentences as a match feature'}


def week12_sentences(week12):
    """The week-12 sentences of the MuLMS bullet (ROADMAP §4, week 12): per variant its
    3-seed screen against mention replacement (no verdict word: the screen decides nothing)
    and, where run, its decision on seeds 300 to 900 against the CRF head (1) and mention
    replacement (2), with the share of the unseen test names among its silver names."""
    out = ''
    for variant, parts in week12.items():
        name = WEEK12_NAMES.get(variant, variant)
        screen = parts.get('screen')
        if screen:
            d = screen['minus_crf_mr']['micro']
            out += (f". Week 12, names from the held-out subfield's topic corpus, {name} "
                    f"({variant}), the 3-seed screen against mention replacement: "
                    f"{delta_cell(d)}")
        decision = parts.get('decision')
        if decision:
            out += (f". Week-12 decision for {name} on seeds 300 to 900: "
                    f"{interval(decision[variant]['micro'])}")
            for k, reference in ((1, 'the CRF head'), (2, 'mention replacement')):
                word, numbers = paired_verdict(decision['minus_crf' if k == 1 else 'minus_crf_mr'])
                out += f"; decision {k}, against {reference}, paired by subfield: {word} ({numbers})"
        entry = decision or screen
        if entry and entry.get('unseen_in_silver') is not None:
            out += (f". {100 * entry['unseen_in_silver']:.1f}% of the unseen test names are "
                    f"its silver names")
    return out


def week6_screen_line(comps, method, week='6', names=WEEK6_NAMES):
    """One bullet for a method's week-6 screen ('' until both granularities are complete):
    whether it goes to stage 2 (its mean micro-F1 difference against the CRF head above 0 at
    either granularity, as in ner_seedmeans.screen_verdicts), the 3-seed differences, and the
    screen's comparisons reported alongside. No verdict words: the screen decides nothing.
    week and names serve the week-7 screen."""
    found = {g: find_comparison(comps, g, LOPO, method, 'crf') for g in GRANULARITIES}
    if not all(found.values()):
        return ''
    passed = any(c['micro']['mean_diff'] > 0 for c in found.values())
    others = [(method, 'crf', f"against the {names['crf']}"),
              (f'{method}_ens3', 'crf_ens3', f"its vote against the {names['crf_ens3']}")]
    if method == 'crf_mt_st3':
        others.append((method, 'crf_mt', f"against {WEEK6_NAMES['crf_mt']}"))
    parts = []
    for variant, reference, words in others:
        diffs = [f"{g}-grained {delta_cell(c['micro'])}" for g in GRANULARITIES
                 if (c := find_comparison(comps, g, LOPO, variant, reference))]
        if diffs:
            parts.append(f"{words} {', '.join(diffs)}")
    return (f"* NER week {week}, the screen (GPU runs, seeds 0, 100 and 200, leaving one paper "
            "out; mean paired micro-F1 difference [95% CI]; it decides only which methods get "
            f"10 seeds): {names[method]} {'goes to stage 2' if passed else 'stops'}; "
            + '; '.join(parts) + ".\n")


def week6_verdict_line(comps, variant, reference, design, week='6', names=WEEK6_NAMES):
    """One bullet for a week-6 comparison over the held-out papers, each granularity with
    its role ('' if none is complete); week and names serve the week-7 comparisons."""
    parts = []
    for g in GRANULARITIES:
        c = find_comparison(comps, g, LOPO, variant, reference)
        if c:
            word, num = paired_verdict(c)
            parts.append(f"{g}-grained {word} ({num}; "
                         f"{ROLE_WORDS.get(c.get('role'), ROLE_WORDS['reported'])})")
    if not parts:
        return ''
    sep = '; ' if week != '6' else ', '
    return (f"* NER week {week} (GPU runs{sep}{design}), {names[variant]} vs the "
            f"{names[reference]} (paired micro-F1 difference, leaving one paper out): "
            + '; '.join(parts) + ".\n")


def sofc_note(analysis):
    """Bullets on SOFC-Exp's official split from the week-5 analyses file, one per level
    (SOFC_LEVELS) that has its results; '' without them. S1 (entity types) and the methods are
    decisions under the week-6 rule, over the 10 model seeds (seed_decision)."""
    return ''.join(sofc_level_note(analysis, g) for g in SOFC_LEVELS)


def sofc_level_note(analysis, g):
    """The bullet of one SOFC-Exp level (see sofc_note); a method's votes, where the analysis
    has them, are reported against the CRF votes, paired by seed triple."""
    e = dig(analysis, 'published_split', g)
    if not e:
        return ''
    lead, s1, s1_decides, _, published, _ = SOFC_LEVELS[g]
    base, crf = e['baseline'], e['crf']
    line = (f"{lead}: plain MatSciBERT {interval(base['micro'])}, the CRF head "
            f"{interval(crf['micro'])} (macro {crf['macro']['mean']:.2f}), against the "
            f"published {published} {e['published']:.2f} micro and "
            f"{e.get('published_macro', 0):.2f} macro. {s1} against plain MatSciBERT, paired "
            "by seed: " + seed_decision(e['crf_minus_baseline'], decides=s1_decides))
    if 'crf_votes' in e:
        line += (f"; the CRF votes over the 3 seed triples score "
                 f"{e['crf_votes']['micro']['mean']:.2f}")
    for method, m in e.get('methods', {}).items():
        if method in WEEK7_SOFC_LABELS:
            label = WEEK7_SOFC_LABELS[method]
        else:
            label = f"S2, {WEEK6_NAMES[method]} against the CRF head"
        line += (f". {label}, paired by seed: {interval(m['micro'])}, "
                 + seed_decision(m['minus_crf'], decides=method not in WEEK7_REPORTED))
        if 'votes' in m:
            line += (f"; its votes over the 3 seed triples score {m['votes']['micro']['mean']:.2f}"
                     ", against the CRF votes, paired by triple: "
                     + seed_decision(m['votes_minus_crf_votes'], decides=False))
    if 'baseline_retokenized_test' in e:
        m = e['baseline_retokenized_test']
        line += (f". {RETOKENIZED_BASELINE_LABEL}, paired by seed: {interval(m['micro'])}, "
                 + seed_decision(m['minus_baseline']))
    return line + ".\n"


def seed_decision(c, decides=True):
    """A paired difference over the model seeds of one split, worded as its decision (the
    week-6 rule) when it is one and has all PUBLISHED_SEEDS seeds; otherwise its numbers,
    reported only."""
    word, numbers = paired_verdict(c)
    n = len(c['micro']['per_run'])
    if not decides:
        return f"{numbers} (reported; no decision)"
    if n == PUBLISHED_SEEDS:
        return f"{word} ({numbers})"
    return f"{numbers} over {n} of the {PUBLISHED_SEEDS} seeds (reported; no decision)"


def published_split_note(analysis):
    """Bullet on the published split (week-5 amendment) from the week-5 analyses; '' unless
    both granularities have it."""
    entries = analysis.get('published_split') or {}
    if not all(g in entries for g in GRANULARITIES):
        return ''
    fine, coarse = (entries[g] for g in GRANULARITIES)

    def both(get):
        return f"{get(fine)} fine-grained and {get(coarse)} coarse-grained"
    lies = [e['baseline']['published_lies'] for e in (fine, coarse)]
    if lies == ['within', 'within']:
        where = 'Both published numbers lie within these intervals'
    else:
        where = (f"The published {fine['published']:.2f} lies {lies[0]} the fine-grained "
                 f"interval, {coarse['published']:.2f} {lies[1]} the coarse-grained one")
    votes = (f", and its votes over the 3 seed triples score "
             f"{fine['crf_votes']['micro']['mean']:.2f} and "
             f"{coarse['crf_votes']['micro']['mean']:.2f}"
             if all('crf_votes' in e for e in (fine, coarse)) else '')
    return ("* NER, the published split (week-5 amendment: the release's partition and tags, "
            "micro F1 as its code scores it; 10 model seeds, GPU runs; the 95% CIs are over the "
            "seeds of one split): plain MatSciBERT, the paper's model, scores "
            + both(lambda e: interval(e['baseline']['micro'])) + f". {where}. The CRF head adds "
            + both(lambda e: interval(e['crf_minus_baseline']['micro'], signed=True))
            + ", paired by seed (strict IOB2 "
            + ' and '.join(interval(e['crf_minus_baseline']['micro_strict'], signed=True)
                           for e in (fine, coarse))
            + f"){votes}. Macro F1 is lower: "
            + ' and '.join(f"{e['baseline']['macro']['mean']:.2f}" for e in (fine, coarse))
            + " for plain MatSciBERT, "
            + ' and '.join(f"{e['crf']['macro']['mean']:.2f}" for e in (fine, coarse))
            + " for the CRF head.\n")


def seed_triples_note(analysis):
    """Bullet on how far a 3-seed design misleads (week 5, exploratory): for each family on
    the protocol of its decision, the share of the 120 seed triples that pass the rule and
    that have the opposite sign to the 10-seed difference; and the seed spread of the CRF
    head. '' without the analyses."""
    triples, spread = analysis.get('triples') or {}, analysis.get('spread') or {}
    parts = []
    for fam, protocol in WEEK5_CLAIMS:
        cells = []
        for g in GRANULARITIES:
            t = dig(triples, f'{g}_{protocol}', fam)
            if t:
                ten = t['ten_seed']
                cells.append(f"{g}-grained ({ten['mean_diff']:+.2f} "
                             f"[{ten['ci95'][0]:+.2f}, {ten['ci95'][1]:+.2f}] with 10 seeds) "
                             f"{100 * t['share_above_0']:.0f}% pass, "
                             f"{100 * t['share_opposite_sign']:.0f}% opposite sign")
        if cells:
            name = WEEK5_NAMES[fam]
            parts.append(f"{name[0].upper()}{name[1:]} {PROTOCOL_PHRASES[protocol]}: "
                         + '; '.join(cells))
    if not parts:
        return ''
    sds = []
    for p, phrase in PROTOCOL_PHRASES.items():
        found = [dig(spread, f'{g}_{p}', 'crf', 'mean_sd') for g in GRANULARITIES]
        if None not in found:
            sds.append(f"{found[0]:.2f} {'fine-grained' if not sds else 'fine'} and "
                       f"{found[1]:.2f} {'coarse-grained' if not sds else 'coarse'} {phrase}")
    tail = (". The SD of a split's micro F1 over the 10 seeds of the CRF head averages "
            + ', and '.join(sds)) if sds else ''
    return ("* NER, how far a 3-seed design misleads (week 5, exploratory): for each family "
            "against the CRF head on the protocol of its decision, the share of the 120 triples "
            "of the 10 seeds (the same triple on both sides) whose 3-seed comparison passes the "
            "rule, and whose mean difference has the opposite sign to the 10-seed one. "
            + '. '.join(parts) + tail + ".\n")


def rare_types_note(res, analysis, lowest=5):
    """Bullet on the coarse-grained types that keep macro F1 down (week 5, exploratory): the
    `lowest` types by F1 of the reported system's votes over the 5 random splits; '' without
    them."""
    found = week5_system(res, 'coarse', 'random_split_vary')
    system = found[0] if found else 'crf_ens3'
    types = dig(analysis, 'types', 'coarse_random_split_vary', system)
    if not types:
        return ''
    worst = list(types.items())[:lowest]
    first, t = worst[0]
    return ("* NER, the coarse-grained types that keep macro F1 down (week 5, exploratory; the "
            f"3 votes of the reported system `{system}` over the 5 random splits, F1 and test "
            "entities per split): "
            + ', '.join(f"{name} {v['f1']:.1f} ({v['support_per_split']:.1f})"
                        for name, v in worst)
            + f". Macro F1 counts each type alike: without {first} alone it would be "
            f"{t['macro_without']:.2f} instead of {t['macro_with']:.2f} on the splits that "
            "have it.\n")


def ensemble_member_note(res):
    """Exploratory bullet: the 3-model vote against the mean of its own members.

    Read from `summary.vs_member_mean` of the crf_ens3 results files that have all
    their runs ('' if none has).
    """
    parts = []
    for protocol, phrase in (('random_split_vary', 'across the 5 random splits'),
                             ('leave_one_paper_out', 'leaving one paper out')):
        found = []
        for g in GRANULARITIES:
            r = res.get(ner_file(g, protocol, 'crf_ens3'))
            gain = dig(r, 'summary', 'vs_member_mean')
            if gain and has_runs(r, REQUIRED_PAIRS[protocol]):
                found.append(f"{g}-grained {delta_cell(gain)}")
        if found:
            parts.append(f"{'; '.join(found)} {phrase}")
    if not parts:
        return ''
    return ("* NER, 3-model ensemble vs the mean of its own members on the same split "
            "(exploratory, not pre-registered; no single reference run enters): "
            + '; '.join(parts) + ".\n")


def week3_verdicts(comps, variant, reference, protocol):
    """(role words, verdicts) of a week-3 comparison for one protocol; None if none is complete."""
    verdicts, role = [], 'reported'
    for g in GRANULARITIES:
        c = find_comparison(comps, g, protocol, variant, reference)
        if c:
            word, num = paired_verdict(c)
            verdicts.append(f"{g}-grained {word} ({num})")
            role = c.get('role', role)
    if not verdicts:
        return None
    return ROLE_WORDS.get(role, ROLE_WORDS['reported']), '; '.join(verdicts)


def week3_verdict_line(comps, variant, reference):
    """One "Reading the numbers" bullet for a week-3 comparison ('' if none is complete).

    The verdicts of verdict_line over the 3-seed means, each protocol marked with its role:
    the one the week-3 rule decides on, or reported without a decision.
    """
    split = week3_verdicts(comps, variant, reference, 'random_split_vary')
    ood = week3_verdicts(comps, variant, reference, 'leave_one_paper_out')
    if not (split or ood):
        return ''
    line = (f"* NER week 3, {WEEK3_NAMES[variant]} vs the {WEEK3_NAMES[reference]} "
            "(3-seed means, paired micro-F1 difference): ")
    if split:
        line += f"{split[1]} across the 5 random splits ({split[0]})"
    if ood:
        line += (f". Leaving one paper out ({ood[0]}): " if split
                 else f"leaving one paper out ({ood[0]}): ") + ood[1]
    return line + '.\n'


def week4_verdict_line(comps, variant, reference, runs, week=4):
    """One "Reading the numbers" bullet for a week-4 comparison ('' if none is complete).

    As week3_verdict_line, naming the machine of both sides (CPU votes, GPU runs). With
    week=5, a week-5 comparison (10-seed means of GPU runs).
    """
    comps = {k: c for k, c in comps.items() if c.get('role') != 'descriptive'}
    split = week3_verdicts(comps, variant, reference, 'random_split_vary')
    ood = week3_verdicts(comps, variant, reference, 'leave_one_paper_out')
    if not (split or ood):
        return ''
    names = WEEK5_NAMES if week == 5 else WEEK4_NAMES
    design = f"{runs} runs, 10 seeds" if week == 5 else f"{runs} runs"
    line = (f"* NER week {week} ({design}), {names[variant]} vs the "
            f"{names[reference]} (paired micro-F1 difference): ")
    if split:
        line += f"{split[1]} across the 5 random splits ({split[0]})"
    if ood:
        line += (f". Leaving one paper out ({ood[0]}): " if split
                 else f"leaving one paper out ({ood[0]}): ") + ood[1]
    return line + '.\n'


def hardware_note(res):
    """Descriptive bullet: the GPU CRF reference minus the CPU one, same seeds and splits."""
    comps = res.comparisons(GPU_PREFIX + WEEK4_FILE)
    parts = []
    for protocol, phrase in (('random_split_vary', 'over the 5 random splits'),
                             ('leave_one_paper_out', 'over the 4 held-out papers')):
        found = [f"{g}-grained {delta_cell(c['micro'])}" for g in GRANULARITIES
                 for c in [find_comparison(comps, g, protocol, 'crf', 'crf')]
                 if c and c.get('role') == 'descriptive']
        if found:
            parts.append(f"{'; '.join(found)} {phrase}")
    if not parts:
        return ''
    return ("* NER, GPU minus CPU (descriptive, not a comparison of methods): the CRF head with "
            "the same seeds and splits, 3-seed means, " + '; '.join(parts) + ". This is why GPU "
            "runs are compared only with GPU runs.\n")


# --------------------------------------------------------------------------------
# "At a glance" table and the generated blocks
# --------------------------------------------------------------------------------
def improves(comps, g, protocol, variant, reference):
    """The complete comparison if its micro-F1 CI lies above 0, else None."""
    c = find_comparison({k: c for k, c in comps.items() if c.get('role') != 'descriptive'},
                        g, protocol, variant, reference)
    return c if c and c['micro']['ci95'][0] > 0 else None


def reported_system(res, g):
    """(variant, random-split results, held-out-paper results, prefix) of the reported NER
    system in distribution.

    Week 3: the 3-model seed ensemble crf_ens3, unless the mixed-backbone ensemble improves on
    it for the granularity (a complete comparison over the 5 random splits with a micro-F1 CI
    above 0). Week 4: the GPU multi-task vote crf_mt_ens3 replaces it if it improves on the GPU
    crf_ens3 over the 5 random splits; its results are then those of results/gpu/ (prefix
    'gpu/', else ''). The results are None for files that do not exist.
    """
    var = 'crf_mix3' if improves(res.comparisons(WEEK3_FILE), g, 'random_split_vary',
                                 'crf_mix3', 'crf_ens3') else 'crf_ens3'
    prefix = ''
    if improves(res.comparisons(GPU_PREFIX + WEEK4_FILE), g, 'random_split_vary', 'crf_mt_ens3',
                'crf_ens3'):
        var, prefix = 'crf_mt_ens3', GPU_PREFIX
    return (var, res.get(prefix + ner_file(g, 'random_split_vary', var)),
            res.get(prefix + ner_file(g, 'leave_one_paper_out', var)), prefix)


def unseen_system(res, g):
    """(variant, held-out-paper results) of the reported NER system on unseen papers.

    The week-3 reported system (crf_ens3, or crf_mix3), unless a week-4 CPU vote
    (crf_ls_ens3, crf_st_ens3) improves on crf_ens3 over the 4 held-out papers; if both do,
    the one with the larger mean difference.
    """
    var = 'crf_mix3' if improves(res.comparisons(WEEK3_FILE), g, 'random_split_vary',
                                 'crf_mix3', 'crf_ens3') else 'crf_ens3'
    passed = [c for vote in ('crf_ls_ens3', 'crf_st_ens3')
              for c in [improves(res.comparisons(WEEK4_FILE), g, 'leave_one_paper_out', vote,
                                 'crf_ens3')] if c]
    if passed:
        var = max(passed, key=lambda c: c['micro']['mean_diff'])['variant']
    return var, res.get(ner_file(g, 'leave_one_paper_out', var))


def week5_system(res, g, protocol):
    """(variant, scores, crf_ens3 scores) of the week-5 reported system on one protocol; None
    without the week-5 comparisons.

    crf_ens3, unless a vote of WEEK5_VOTES improves on it (the one with the largest mean
    difference). scores: the vote family's per-average mean, CI and per-split scores (the mean
    of its 3 votes), from the week-5 comparison that has it. crf_ens3 scores: those of the vote
    it replaced when another vote is the system (None if unknown), else None.
    """
    comps = res.comparisons(GPU_PREFIX + WEEK5_FILE)
    base = find_comparison(comps, g, protocol, 'crf_ens3', 'crf')
    base = base['variant_scores'] if base else next(
        (c['reference_scores'] for vote in WEEK5_VOTES
         for c in [find_comparison(comps, g, protocol, vote, 'crf_ens3')]
         if c and c.get('reference_scores')), None)
    passed = [c for vote in WEEK5_VOTES
              for c in [improves(comps, g, protocol, vote, 'crf_ens3')] if c]
    if passed:
        c = max(passed, key=lambda c: c['micro']['mean_diff'])
        return c['variant'], c['variant_scores'], c.get('reference_scores') or base
    return ('crf_ens3', base, None) if base else None


def week6_unseen(res, g):
    """(variant, scores, replaced scores, replaced system) of a week-6 vote that passes
    decision 2 on unseen papers for granularity g (the largest mean difference); None if
    none passes."""
    comps = {k: c for k, c in res.comparisons(GPU_PREFIX + WEEK6_FILE).items()
             if c.get('role') == 'decision'}
    passed = [c for m in WEEK6_METHODS
              for c in [improves(comps, g, LOPO, f'{m}_ens3', UNSEEN_BEST[g])] if c]
    if not passed:
        return None
    c = max(passed, key=lambda c: c['micro']['mean_diff'])
    return c['variant'], c['variant_scores'], c.get('reference_scores'), UNSEEN_BEST[g]


def reported_protocol(res):
    """How the reported-system rows name their design: week 5, and week 6 once one of its
    votes holds an unseen-paper slot."""
    if any(week6_unseen(res, g) for g in GRANULARITIES):
        return 'week-5 and week-6 protocols; 10 seeds, GPU runs'
    return WEEK5_PROTOCOL


def week5_cells(res, g, phrase):
    """The week-5 reported system of a granularity; None without its random-split comparisons.

    Returns (in distribution, on unseen papers, verdict). A slot is (name, cell): the system
    and its micro F1, over the 5 random splits or on the held-out papers (`phrase`), followed
    by the score of `crf_ens3` in parentheses when a vote replaced it. The unseen-paper slot is
    None without its comparisons; the verdict is that of the random splits against the
    published number.
    """
    def slot(found, score):
        var, scores, base, *replaced = found
        cell = score(scores['micro'])
        if base:
            cell += f" (`{replaced[0] if replaced else 'crf_ens3'}` {base['micro']['mean']:.2f})"
        return f"{WEEK6_NAMES[var]} `{var}`", cell

    found = week5_system(res, g, 'random_split_vary')
    if not found:
        return None
    unseen = week6_unseen(res, g) or week5_system(res, g, 'leave_one_paper_out')
    inside = slot(found, lambda m: f"micro F1 {interval(m)} over 5 random splits")
    outside = slot(unseen, lambda m: f"{m['mean']:.2f} {phrase}") if unseen else None
    return inside, outside, published_check(found[1], PUBLISHED_F1[g])


def published_split_row(res, g):
    """The "At a glance" row of the published split (week-5 amendment); '' until both its
    files in results/gpu/ hold 10 runs."""
    base = res.get(GPU_PREFIX + ner_file(g, 'published_split'))
    crf = res.get(GPU_PREFIX + ner_file(g, 'published_split', 'crf'))
    if not (has_runs(base, 10) and has_runs(crf, 10)):
        return ''
    paper = PUBLISHED_F1[g]

    def where(res):
        lo, hi = res['summary']['micro']['ci95']
        return 'within' if lo <= paper <= hi else 'below' if paper < lo else 'above'
    crf_where = where(crf) + (' (**beaten**)' if where(crf) == 'below' else '')
    return (f"| NER, {g}-grained, the published split (the release's partition, micro F1), 10 "
            f"seeds (GPU runs) | plain MatSciBERT {interval(base['summary']['micro'])}; CRF head "
            f"{interval(crf['summary']['micro'])} | published {paper:.2f}: {where(base)} the 95% "
            f"CI of plain MatSciBERT{' (reproduced)' if where(base) == 'within' else ''}, "
            f"{crf_where} that of the CRF head; the CIs are over seeds only |")


def sofc_row(res, g='sofc'):
    """The "At a glance" row of a SOFC-Exp level on its official split (week 6; the slot
    types, week 7 part 2d); '' until plain MatSciBERT and the CRF head each have 10 runs."""
    base = res.get(GPU_PREFIX + ner_file(g, 'published_split'))
    crf = res.get(GPU_PREFIX + ner_file(g, 'published_split', 'crf'))
    if not (has_runs(base, 10) and has_runs(crf, 10)):
        return ''
    _, _, _, name, published, scores = SOFC_LEVELS[g]
    lo, hi = crf['summary']['micro']['ci95']
    paper = scores['micro']
    where = 'within' if lo <= paper <= hi else 'below' if paper < lo else 'above'
    return (f"| {name}, the official split, 10 seeds (GPU runs) | plain MatSciBERT "
            f"{interval(base['summary']['micro'])}; CRF head {interval(crf['summary']['micro'])} "
            f"(macro {crf['summary']['macro']['mean']:.2f}) | published {published} "
            f"{paper:.2f} micro, {scores['macro']:.2f} macro (Gupta et al. 2022): "
            f"{where} the CRF head's 95% CI (over seeds only) |")


def system_name(var, prefix, uvar):
    """The reported system's name: `var` in distribution, then `uvar` on unseen papers if other."""
    name = f"{WEEK4_NAMES[var]} `{var}`" + (' (GPU runs)' if prefix else '')
    if uvar != var:
        name += f"; on unseen papers the {WEEK4_NAMES[uvar]} `{uvar}`"
    return name


def unseen_cell(res, g, var, ood, uvar, uood, phrase, named=False):
    """', <score> <phrase>' of the system on unseen papers ('' without complete results).

    If another system than `var` holds that slot, `named` names it first, and when a week-4
    vote replaced crf_ens3 there, crf_ens3's score follows in parentheses.
    """
    if uvar == var:
        return f", {ood['summary']['micro']['mean']:.2f} {phrase}" if has_runs(ood, 4) else ''
    if not has_runs(uood, 4):
        return ''
    lead = f"; on unseen papers the {WEEK4_NAMES[uvar]} `{uvar}`, " if named else ', '
    cell = f"{lead}{uood['summary']['micro']['mean']:.2f} {phrase}"
    base = res.get(ner_file(g, 'leave_one_paper_out', 'crf_ens3'))
    if uvar != 'crf_ens3' and has_runs(base, 4):
        cell += f" (`crf_ens3` {base['summary']['micro']['mean']:.2f})"
    return cell


def system_protocol(res):
    """The protocol whose rules chose the reported system: week 5, 4 or 3, the latest whose
    comparisons exist."""
    if res.comparisons(GPU_PREFIX + WEEK5_FILE):
        return reported_protocol(res)
    week4 = res.comparisons(WEEK4_FILE) or res.comparisons(GPU_PREFIX + WEEK4_FILE)
    return 'week-4 protocol' if week4 else 'week-3 protocol'


def published_check(summary, paper):
    """The reported system against the published F1, whose averaging the paper does not state.

    Returns ('beaten under <averages> averaging' or 'not significantly better', the other
    averages as 'name mean [lo, hi]'): an average is beaten when its 95% CI lies above the
    published number. Averages missing from the summary are left out.
    """
    present = [m for m in AVERAGES if m in summary]
    above = [m for m in present if summary[m]['ci95'][0] > paper]
    rest = [f"{m} {interval(summary[m])}" for m in present if m not in above]
    if not above:
        return 'not significantly better', rest
    names = ' and '.join(above) if len(above) < 3 else f"{', '.join(above[:-1])} and {above[-1]}"
    return f"beaten under {names} averaging", rest


def crf_gain_note(comps, g):
    """Short reading of a CRF gain for the at-a-glance table."""
    c = find_comparison(comps, g, 'random_split_vary', 'crf', 'baseline')
    if not c:
        return ''
    st = c['micro_strict']
    if st['ci95'][0] > 0:
        return f"; {st['mean_diff']:+.2f} of the gain survives strict IOB2 scoring"
    return "; the gain comes from valid spans only (see below)"


def glance_table(res):
    """The "At a glance" table with its trailing blank line ('' if it has no row).

    The baseline variants (cw, crf) and the reported system (week-3 protocol) appear here;
    the week-2 and week-3 comparisons do not.
    """
    _, vary, lopo = baseline_ner(res)
    comps = res.comparisons()
    off, on = res.get('eval_pipeline_llm_off.json'), res.get('eval_pipeline_llm_on.json')
    tt = res.get('term_typing_mindf2.json')
    llm = tt['methods'].get('hybrid+llm', {}).get('mean_over_folds', {}) if tt else {}

    glance = ["**At a glance** (95% CIs in brackets; details below)", "",
              "| Task | Result in this repo | Reference |", "|---|---|---|"]
    if all(has_runs(vary[g], 5) for g in GRANULARITIES):
        for g in GRANULARITIES:
            s, paper = vary[g]['summary']['micro'], vary[g]['paper_test_f1']
            if s['ci95'][0] > paper:
                ref = f"published {paper:.2f}: **beaten** if the paper used micro F1"
            elif s['ci95'][1] < paper:
                ref = f"published {paper:.2f}: below"
            else:
                ref = f"published {paper:.2f}: not significantly better"
            glance.append(f"| NER, {g}-grained, 5 random splits (micro F1) | {interval(s)} | "
                          f"{ref} |")
    if all(has_runs(lopo[g], 4) for g in GRANULARITIES):
        glance.append(f"| NER on an unseen paper (leave one paper out) | "
                      f"{lopo['fine']['summary']['micro']['mean']:.2f} fine, "
                      f"{lopo['coarse']['summary']['micro']['mean']:.2f} coarse | "
                      "no published equivalent |")
    for var, vname in BASELINE_VARIANTS.items():
        found = {g: find_comparison(comps, g, 'random_split_vary', var, 'baseline')
                 for g in GRANULARITIES}
        cells = [f"{g} {interval(found[g]['micro'], signed=True)}"
                 for g in GRANULARITIES if found[g]]
        if cells:
            glance.append(f"| NER with a {vname}: paired gain over the baseline | "
                          f"{'; '.join(cells)} | counts only if the CI lies above 0 |")
        for g in GRANULARITIES:
            c = found[g]
            v = res.get(ner_file(g, 'random_split_vary', var))
            if (c and c['micro']['ci95'][0] > 0
                    and v and v['summary']['micro']['ci95'][0] > v['paper_test_f1']):
                glance.append(f"| NER, {g}-grained with the {vname}, 5 random splits (micro F1) | "
                              f"{interval(v['summary']['micro'])} | published "
                              f"{v['paper_test_f1']:.2f}: **beaten** if the paper used micro F1"
                              f"{crf_gain_note(comps, g) if var == 'crf' else ''} |")
    for g in GRANULARITIES:
        row = published_split_row(res, g)
        if row:
            glance.append(row)
    for g in SOFC_LEVELS:
        row = sofc_row(res, g)
        if row:
            glance.append(row)
    for g in GRANULARITIES:
        week5 = week5_cells(res, g, 'on an unseen paper')
        if week5:
            (name, cell), unseen, (verdict, rest) = week5
            if unseen:
                same = unseen[0] == name
                name += '' if same else f"; on unseen papers the {unseen[0]}"
                cell += f"{', ' if same else '; '}{unseen[1]}"
            if verdict.startswith('beaten'):
                verdict = '**beaten**' + verdict[len('beaten'):]
                if rest:
                    verdict += (f"; {', '.join(rest)} {'is' if len(rest) == 1 else 'are'} "
                                "not significantly above")
            glance.append(f"| NER, {g}-grained, reported system ({reported_protocol(res)}): {name} | "
                          f"{cell} | published {PUBLISHED_F1[g]:.2f} (none for an unseen "
                          f"paper): {verdict} |")
            continue
        var, v, ood, prefix = reported_system(res, g)
        uvar, uood = unseen_system(res, g)
        if has_runs(v, 5):
            cell = (f"micro F1 {interval(v['summary']['micro'])} over 5 random splits"
                    + unseen_cell(res, g, var, ood, uvar, uood, 'on an unseen paper'))
            verdict, rest = published_check(v['summary'], v['paper_test_f1'])
            if verdict.startswith('beaten'):
                verdict = '**beaten**' + verdict[len('beaten'):]
                if rest:
                    verdict += (f"; {', '.join(rest)} {'is' if len(rest) == 1 else 'are'} "
                                "not significantly above")
            glance.append(f"| NER, {g}-grained, reported system ({system_protocol(res)}): "
                          f"{system_name(var, prefix, uvar)} | {cell} | published "
                          f"{v['paper_test_f1']:.2f} (none for an unseen paper): {verdict} |")
    if off:
        glance.append(f"| Term typing, 315 terms, pipeline (zero-shot) | accuracy "
                      f"{off['typing']['accuracy']:.3f}"
                      + (f" ({on['typing']['accuracy']:.3f} with LLM re-ranking)" if on else '')
                      + f" | supervised + LLM baseline {llm.get('accuracy', 0):.3f}; "
                      "majority class 0.083 |")
        glance.append(f"| Clustering / taxonomy / relations, class level | ARI "
                      f"{off['clustering']['ari']:.3f} / edge F1 {off['taxonomy']['f1']:.3f} / "
                      "F1 0.000 | near chance; see the roadmap |")
    return '\n'.join(glance) + '\n\n' if len(glance) > 4 else ''


def readme_block(res, note=''):
    """The generated block of README.md, markers included."""
    return f"""<!-- RESULTS:START -->
All numbers are leakage-free and come from the JSON files in [`results/`](results/)
(CPU, `SEED=0`, except the week-4 and week-5 runs on Kaggle T4 GPUs in
[`results/gpu/`](results/gpu/), marked GPU; LLM = `openai/gpt-4o-mini` via OpenRouter). Gold term spans
are given to the pipeline and the typing baselines. "Gold types used" marks
outputs whose predictions had access to the gold labels: those rows are
reported only to show the size of the leak.

{glance_table(res)}{build_tables(res)}{note}

**Reading the numbers**
* The old "Term F1 0.91" corresponds to *0.12* pairwise alignment F1. The
  v5 outputs score higher (0.27, and 0.57–0.74 typing accuracy) only because
  their top-1 rows were the gold classes injected by `type_match`.
* The pipeline's zero-shot typing (0.38; 0.43 with LLM re-ranking) beats plain
  label matching (0.36) but stays well below the supervised hybrid (0.50) and
  hybrid + LLM (0.65) baselines, which can also use labelled terms. Closing
  that gap is the first item of the roadmap.
* Taxonomy and relations are near zero at class level. The term-level
  taxonomy and relation extractors were never designed to recover MMO
  `subClassOf` edges or restrictions; see the roadmap.
{ablation_note(res)}{ner_notes(res)}<!-- RESULTS:END -->"""


def standing_block(res, note=''):
    """The generated block of docs/ROADMAP.md (the "where we stand" table), markers included."""
    off = res.require('eval_pipeline_llm_off.json')
    on = res.get('eval_pipeline_llm_on.json')
    tt = res.get('term_typing_mindf2.json')
    fixed, vary, lopo = baseline_ner(res)

    ner_cells = []
    for g in GRANULARITIES:
        if not has_runs(fixed[g], 1):
            continue
        s = fixed[g]['summary']
        cell = (f"{g}: micro {s['micro']['mean']:.2f} [{s['micro']['ci95'][0]:.2f}, "
                f"{s['micro']['ci95'][1]:.2f}] (macro {s['macro']['mean']:.2f}) on the fixed split")
        if has_runs(vary[g], 5):
            v = vary[g]['summary']['micro']
            cell += f", {v['mean']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] over 5 new splits"
        if has_runs(lopo[g], 4):
            cell += f", {lopo[g]['summary']['micro']['mean']:.2f} leaving one paper out"
        ner_cells.append(cell)

    crf_cells = []
    for g in GRANULARITIES:
        v = res.get(ner_file(g, 'random_split_vary', 'crf'))
        out_of_dist = res.get(ner_file(g, 'leave_one_paper_out', 'crf'))
        if v and v.get('summary', {}).get('n_runs', 0) >= 5:
            m = v['summary']['micro']
            cell = (f"{g}: micro {m['mean']:.2f} [{m['ci95'][0]:.2f}, {m['ci95'][1]:.2f}] "
                    "over 5 new splits")
            if out_of_dist and out_of_dist.get('summary'):
                cell += (f", {out_of_dist['summary']['micro']['mean']:.2f} "
                         "leaving one paper out")
            crf_cells.append(cell)

    reported_cells, reported_refs = [], []
    for g in GRANULARITIES:
        week5 = week5_cells(res, g, 'leaving one paper out')
        if week5:
            (name, cell), unseen, (verdict, _) = week5
            cell = f"{g}: {name}, {cell.replace('over 5 random splits', 'over 5 new splits')}"
            if unseen:
                cell += (f", {unseen[1]}" if unseen[0] == name
                         else f"; on unseen papers the {unseen[0]}, {unseen[1]}")
            reported_cells.append(cell)
            reported_refs.append(f"{g}-grained {verdict}")
            continue
        var, v, ood, prefix = reported_system(res, g)
        uvar, uood = unseen_system(res, g)
        if has_runs(v, 5):
            cell = (f"{g}: {system_name(var, prefix, var)}, micro "
                    f"{interval(v['summary']['micro'])} over 5 new splits"
                    + unseen_cell(res, g, var, ood, uvar, uood, 'leaving one paper out',
                                  named=True))
            reported_cells.append(cell)
            reported_refs.append(f"{g}-grained "
                                 f"{published_check(v['summary'], v['paper_test_f1'])[0]}")

    hyb = tt['methods']['hybrid']['mean_over_folds'] if tt else {}
    llm = tt['methods'].get('hybrid+llm', {}).get('mean_over_folds', {}) if tt else {}
    lines = [
        "| Task | Best leakage-free result in this repo | Reference |",
        "|---|---|---|",
        "| NER, fine / coarse (MatSciBERT, MaterioMiner protocol) | "
        + ('; '.join(ner_cells) if ner_cells else 'run pending (`ner_baseline.py`)')
        + " | 69.92 / 72.32 (Durmaz et al. 2024; micro F1 in the release code) |",
    ]
    if crf_cells:
        lines.append(f"| NER with the CRF head (`--crf`) | {'; '.join(crf_cells)} | same; the "
                     "in-distribution fine-grained gain over the baseline disappears under strict "
                     "IOB2 scoring |")
    if reported_cells:
        lines.append(f"| NER, reported system ({system_protocol(res)}) | "
                     f"{'; '.join(reported_cells)} | same; {', '.join(reported_refs)} |")
    lines += [
        f"| Term typing, 315 terms, pipeline (zero-shot) | accuracy {off['typing']['accuracy']:.3f}"
        + (f" (LLM on: {on['typing']['accuracy']:.3f})" if on else '')
        + " | majority class 0.083 |",
        f"| Term typing, 5-fold CV baselines | hybrid {hyb.get('accuracy', 0):.3f}; hybrid + LLM "
        f"{llm.get('accuracy', 0):.3f} (set F1 {llm.get('set_f1', 0):.3f}) | LLMs4OL 2025 MatOnto "
        "best F1 0.667 (different dataset) |",
        f"| Clustering vs gold classes | ARI {off['clustering']['ari']:.3f}, B-cubed F1 "
        f"{off['clustering']['bcubed_f1']:.3f} (all-singletons baseline "
        f"{off['clustering']['baseline_singletons']['bcubed_f1']:.3f}) | – |",
        f"| Taxonomy, class level | edge F1 {off['taxonomy']['f1']:.3f}; precision vs closure "
        f"{off['taxonomy']['precision_closure']:.3f} (chance "
        f"{off['taxonomy']['random_precision_closure']:.3f}) "
        "| LLMs4OL 2025 MatOnto best F1 0.662 (different dataset) |",
        "| Relations, class level vs 70 restrictions | F1 0.000 | no text-level gold exists |",
    ]
    return ("<!-- STANDING:START -->\n" + '\n'.join(lines) + note
            + "\n\nFull tables: [README](../README.md#results).\n<!-- STANDING:END -->")


# --------------------------------------------------------------------------------
# Rewriting the documents
# --------------------------------------------------------------------------------
def replace_block(text, name, block):
    """Replace the generated block `name` ('RESULTS' or 'STANDING') of `text` with `block`.

    The block runs from the START marker to the END marker, each alone at the start of
    its line. A document that has only the bare placeholder `<!-- NAME -->` gets the
    block in its place (first use). Anything else is an error: silently doing nothing
    would let `--check` pass on a document that is no longer generated.
    """
    start, end = f'<!-- {name}:START -->', f'<!-- {name}:END -->'
    found = list(re.finditer(rf'^{re.escape(start)}.*?^{re.escape(end)}', text, flags=re.S | re.M))
    if len(found) > 1:
        raise ReportError(f"more than one {start} ... {end} block")
    if found:
        return text[:found[0].start()] + block + text[found[0].end():]
    placeholder = f'<!-- {name} -->'
    if placeholder in text:
        return text.replace(placeholder, block)
    raise ReportError(f"no {start} ... {end} block (each marker alone at the start of a line) "
                      f"and no {placeholder} placeholder")


def render(res, readme_text, roadmap_text, pending=()):
    """New contents of README.md and docs/ROADMAP.md, as (readme, roadmap)."""
    note = pending_note(pending)
    return (replace_block(readme_text, 'RESULTS', readme_block(res, note)),
            replace_block(roadmap_text, 'STANDING', standing_block(res, note)))


def read_document(path):
    try:
        return Path(path).read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise ReportError(f"cannot read {path}: {getattr(exc, 'strerror', None) or exc}") from exc


def unified_diff(path, old, new):
    return '\n'.join(difflib.unified_diff(old.split('\n'), new.split('\n'),
                                          fromfile=f'{path} (current)',
                                          tofile=f'{path} (generated)', lineterm=''))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog='report_results.py', allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Regenerate the results block of README.md and the standing table of "
                    "docs/ROADMAP.md from the JSON files in results/. By default both files "
                    "are rewritten in place.",
        epilog="exit codes: 0 written or up to date, 1 --check found a difference, "
               "2 bad input")
    p.add_argument('--check', action='store_true',
                   help="write nothing; print a unified diff and exit 1 if either file would "
                        "change")
    p.add_argument('--pending', nargs='+', metavar='TEXT', default=[],
                   help="names of runs still going; adds a 'Still running when this snapshot "
                        "was taken' note below the tables and the standing table")
    p.add_argument('--results', type=Path, default=REPO_ROOT / 'results', metavar='DIR',
                   help="directory with the result JSON files (default: results/ of the repo)")
    p.add_argument('--readme', type=Path, default=REPO_ROOT / 'README.md', metavar='PATH',
                   help="README to update (default: README.md of the repo)")
    p.add_argument('--roadmap', type=Path, default=REPO_ROOT / 'docs' / 'ROADMAP.md',
                   metavar='PATH', help="roadmap to update (default: docs/ROADMAP.md of the repo)")
    return p.parse_args(argv)


def main(argv=None):
    """Command-line entry point; returns the exit code (see the module docstring)."""
    args = parse_args(argv)
    try:
        if not args.results.is_dir():
            raise ReportError(f"results directory not found: {args.results}")
        paths = [args.readme, args.roadmap]
        old = [read_document(path) for path in paths]
        new = list(render(Results(args.results), old[0], old[1], args.pending))
        updates = list(zip(paths, old, new))
        if not args.check:
            for path, before, text in updates:
                if text == before:
                    print(f"unchanged {path}")
                    continue
                try:
                    path.write_text(text, encoding='utf-8')
                except OSError as exc:
                    raise ReportError(f"cannot write {path}: {exc.strerror or exc}") from exc
                print(f"updated {path}")
            return 0
    except ReportError as exc:
        print(f"report_results: {exc}", file=sys.stderr)
        return 2
    stale = [path for path, before, text in updates if text != before]
    for path, before, text in updates:
        if text != before:
            print(unified_diff(path, before, text))
    if stale:
        print("report_results: " + ', '.join(str(path) for path in stale)
              + " out of date; run `python src/graph/report_results.py` to rewrite",
              file=sys.stderr)
        return 1
    print("up to date: " + ', '.join(str(path) for path in paths))
    return 0


if __name__ == '__main__':
    for _stream in (sys.stdout, sys.stderr):
        # The diff holds characters such as Δ and −; do not crash on a legacy console encoding.
        if hasattr(_stream, 'reconfigure'):
            _stream.reconfigure(errors='replace')
    try:
        code = main()
        sys.stdout.flush()
    except BrokenPipeError:
        # The reader closed early (`--check | head`). Point stdout at devnull so that
        # Python does not report a second error while shutting down.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = 1
    sys.exit(code)
