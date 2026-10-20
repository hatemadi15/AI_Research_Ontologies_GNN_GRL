# Roadmap: from honest baselines to state of the art

This document says where the project stands after the evaluation was made
leakage-free, what "state of the art" means for this data, and which
experiments are most likely to move each metric. Numbers marked *(est.)* are
expectations from the literature, not results.

## 1. Where we stand (measured, leakage-free)

<!-- STANDING:START -->
| Task | Best leakage-free result in this repo | Reference |
|---|---|---|
| NER, fine / coarse (MatSciBERT, MaterioMiner protocol) | fine: micro 72.99 [70.86, 75.12] (macro 67.99) on the fixed split, 73.91 [69.36, 78.45] over 5 new splits, 43.51 leaving one paper out; coarse: micro 75.57 [74.14, 77.01] (macro 67.14) on the fixed split, 74.57 [73.15, 75.98] over 5 new splits, 54.34 leaving one paper out | 69.92 / 72.32 (Durmaz et al. 2024; micro F1 in the release code) |
| NER with the CRF head (`--crf`) | fine: micro 76.36 [71.00, 81.73] over 5 new splits, 49.44 leaving one paper out; coarse: micro 78.15 [76.61, 79.69] over 5 new splits, 59.82 leaving one paper out | same; the in-distribution fine-grained gain over the baseline disappears under strict IOB2 scoring |
| NER, reported system (week-5 protocol; 10 seeds, GPU runs) | fine: 3-model seed ensemble `crf_ens3`, micro F1 79.68 [76.72, 82.65] over 5 new splits; on unseen papers the multi-task vote `crf_mt_ens3`, 52.47 leaving one paper out (`crf_ens3` 48.81); coarse: multi-task vote `crf_mt_ens3`, micro F1 79.43 [77.17, 81.69] over 5 new splits (`crf_ens3` 78.63); on unseen papers the 3-model seed ensemble `crf_ens3`, 61.10 leaving one paper out | same; fine-grained beaten under micro, macro and weighted averaging, coarse-grained beaten under micro and weighted averaging |
| Term typing, 315 terms, pipeline (zero-shot) | accuracy 0.384 (LLM on: 0.429) | majority class 0.083 |
| Term typing, 5-fold CV baselines | hybrid 0.496; hybrid + LLM 0.654 (set F1 0.598) | LLMs4OL 2025 MatOnto best F1 0.667 (different dataset) |
| Clustering vs gold classes | ARI 0.035, B-cubed F1 0.183 (all-singletons baseline 0.475) | – |
| Taxonomy, class level | edge F1 0.006; precision vs closure 0.027 (chance 0.013) | LLMs4OL 2025 MatOnto best F1 0.662 (different dataset) |
| Relations, class level vs 70 restrictions | F1 0.000 | no text-level gold exists |

Full tables: [README](../README.md#results).
<!-- STANDING:END -->

## 2. What "SOTA" means here

| Task | Published reference | Comparable today? |
|---|---|---|
| **NER on MaterioMiner** (span + class, seqeval entity F1) | MatSciBERT 69.92 fine-grained (top-27 classes) / 72.32 coarse-grained, random 65/15/20 split, 5 seeds. The paper does not state its averaging; its released code uses micro F1 in seqeval's default mode ([Durmaz et al., Sci Data 2024](https://www.nature.com/articles/s41597-024-03926-5)). Out-of-distribution (different protocol): roughly 60 micro F1 for MatSciBERT (read from a figure), 36% above GPT-4 in-context learning ([Sci Rep 2025](https://www.nature.com/articles/s41598-025-03619-y)) | **Yes**, via `ner_baseline.py --published-split`: on the split the release ships, plain MatSciBERT reproduces both numbers with 10 seeds (week 5). Over other splits, compare means with 95% CIs (§1). |
| **Term typing**, gold spans given (MaterioMiner → MMO) | No published number. Closest public task: LLMs4OL 2025 Task B on MatOnto, best F1 0.667 (IRIS) ([overview](https://www.tib-op.org/ojs/index.php/ocp/article/view/2913)) | Our baselines establish the first numbers. MatOnto is a different dataset. |
| **Taxonomy discovery** | LLMs4OL 2025 Task C on MatOnto, best F1 0.662 (SBU-NLP) | Not yet: the pipeline builds term-level is-a edges, which is a different task. |
| **Relation extraction** | No text-level gold in MaterioMiner | No. Only class-level checks against 70 ontology restrictions are possible. |

The previous claim of "Type F1 0.979 exceeds OAEI 2024 SOTA (~0.95)" compared
a property of the dataset (NER labels are ontology class names) with a
different benchmark (OAEI, which matches ontologies to ontologies). It should
not be reused.

## 3. Priorities

### P0: Keep the evaluation honest (partly done in this PR)
- Done: leakage-free typing, alignment, clustering, taxonomy and relation
  metrics; the gold file is isolated behind `ORACLE_TYPES`; a unit test
  enforces that; deterministic runs; CI; paired exact McNemar tests
  (`metrics.mcnemar_exact`) for every ablation preset.
- Done for NER: mean ± SD and 95% CIs over seeds, splits and held-out
  papers, and paired differences by split and model seed
  (`ner_baseline.py --compare`).
- Next: report every other headline number as mean ± std over seeds or folds
  plus a CI (the test sets are tiny: 96 NER test sentences, 315 terms). Use
  paired tests (McNemar for typing, paired differences by split and seed for
  NER) whenever claiming an improvement.
- Next: write one results JSON per run (git SHA, config, seed, model ids) and
  never tune thresholds or prompts on test folds. `FIXED_THRESHOLD` and the
  baseline hyper-parameters are fixed in advance for this reason.

### P1: Beat the published NER numbers (69.92 / 72.32)
The pipeline currently consumes gold spans. A real end-to-end system needs NER,
and NER is the only task with a published SOTA on this data.

Status: on the paper's own split (shipped with the release, scored like
its code), plain MatSciBERT reproduces the published numbers with 10 seeds,
and the CRF head (item 4) beats both (week 5; the CIs are over seeds only):
- fine-grained: plain 70.19 [69.03, 71.35], CRF head 73.31 [72.28, 74.35],
  published 69.92;
- coarse-grained: plain 72.52 [71.88, 73.16], CRF head 75.48
  [74.23, 76.72], published 72.32.

Across 5 random splits the CRF head also passes the pre-registered check at
both granularities: 76.36 [71.00, 81.73] fine-grained and 78.15
[76.61, 79.69] coarse-grained.

Week 2 tested three changes against the CRF head: multi-task training
(item 2), domain-adaptive pre-training on the PubMed sentences (item 5) and
a 3-model ensemble (item 8). None passes the pre-registered rule. The
ensemble scores highest (79.76 fine-grained, 79.45 coarse-grained), beats
the mean of its own members on every split, and is the only change that
improves on unseen papers (coarse-grained, +1.60 [+0.52, +2.69]). But one
reference model varies by several F1 points from run to run, which hides
gains of 1–3 points in a comparison against a single run per split.

Week 3 compared 3-seed means split by split (protocol in §4). The protocol
expected this to detect gains of about 2 points; in practice it detected two
gains of about 1 point, both coarse-grained:
- multi-task training on the random splits: +0.75 [+0.10, +1.39] (item 2);
- label semantics on unseen papers: +1.17 [+0.94, +1.41] (item 6).

Mixing backbones adds nothing to the seed ensemble (item 8). Self-training
(item 7) and in-domain pre-training (item 5) do not pass. The reported
system therefore stays the seed ensemble `crf_ens3`.

The paper does not state its averaging (its released code uses micro F1),
so the comparison is made under each:
- fine-grained, the 95% CI over the random splits lies above 69.92 under
  micro, macro and weighted F1;
- coarse-grained, it lies above 72.32 under micro and weighted F1, but not
  under macro F1 (73.03 [67.79, 78.27]).

The largest remaining gap is out of distribution (item 9). On an unseen
paper `crf_ens3` scores 49.55 fine-grained and 61.42 coarse-grained, 18–30
points below its scores on the random splits. The largest single-model
gains there are fine-grained:
- multi-task, reported: +3.39 [+0.45, +6.34];
- self-training, the decision: +2.91 [−0.73, +6.56], positive on every
  paper.

Week 4 (protocol in §4) ran items 1, 2, 3 and 5 of the list below. Apart
from the votes over saved CPU predictions, all of its training ran on Kaggle
T4 GPUs, and GPU runs are compared only with GPU runs.
- **The votes over the saved CPU predictions** (item 1, no training).
  - Label semantics, coarse-grained, on unseen papers: the vote improves on
    `crf_ens3` by +1.63 [+0.11, +3.16], positive on every paper. It
    therefore replaces `crf_ens3` there (63.05 against 61.42).
  - Label semantics, fine-grained: no significant change (+0.36
    [−2.53, +3.24]). On the random splits (reported) the vote is below
    `crf_ens3` at both granularities (−1.16 fine, −0.95 coarse), so
    `crf_ens3` stays the system in distribution.
  - Self-training, fine-grained, misses again: +3.68 [−1.15, +8.52],
    positive on every paper. Coarse-grained: −0.09.
- **The multi-task vote** (item 1, GPU) does not improve on the GPU
  `crf_ens3` over the random splits: +1.26 [−1.08, +3.59] fine-grained,
  −0.07 [−2.02, +1.87] coarse-grained. `crf_ens3` stays the system in
  distribution.
- **Label semantics on the multi-task model** (item 2, GPU) passes neither
  protocol against the CRF head:
  - random splits: +0.76 [−0.57, +2.10] fine-grained, +0.65 [−0.63, +1.93]
    coarse-grained;
  - unseen papers: +4.08 [−1.99, +10.14] fine-grained, −0.01 [−2.64, +2.63]
    coarse-grained.
- **Self-training with 3 teachers** (item 3, GPU) improves fine-grained on
  unseen papers: +1.07 [+0.33, +1.80] over the CRF head, positive on every
  paper. Coarse-grained: +0.85 [−0.58, +2.29].
  - On the random splits (reported) it also improves fine-grained:
    +1.22 [+0.60, +1.84], positive on every split. Coarse-grained: +0.71
    [−0.96, +2.38].
  - Tagging all 40,000 corpus sentences gives 607–1,256 silver sentences
    per paper fine-grained (395–1,305 per random split), against 60–240 per
    run in week 3 (out of 2,000 tagged). That is about 6 times more on
    average, not the 20 times item 3 expected: few sentences get the same
    tags from all 3 teachers with every marginal at 0.9 or more.
  - The students stay below their teachers' vote: −1.01 [−1.63, −0.39]
    against the GPU `crf_ens3`, negative on every paper, and −1.08
    [−2.14, −0.03] on the random splits. Only a vote can change the
    reported system, so it is unchanged.
- **The GPU runner** (item 5): `kaggle_jobs.py` uploads a frozen commit
  and runs the jobs on two T4 GPUs. A CRF run takes 1.2 minutes (13–27 on
  2 CPU threads).
  - A run repeats exactly, on either GPU, across code commits and across
    sessions: the 54 teachers trained inside the self-training runs (24 on
    the papers, 30 on the random splits) score exactly as the GPU `crf_m*`
    runs of another session.
  - All 294 GPU runs share one environment (Tesla T4, CUDA 12.8, torch
    2.11.0, transformers 5.18.0).
  - Week 4 used 7.5 GPU-hours, smoke jobs included, a quarter of Kaggle's
    weekly 30.

The GPU runs do not confirm the coarse-grained gains of week 3 or the CPU
label-semantics vote. These are the GPU repeats, which are reported without
a decision:

| Change, coarse-grained | CPU | GPU |
|---|---|---|
| multi-task, random splits | +0.75 [+0.10, +1.39] | −0.08 [−1.71, +1.55] |
| label semantics, unseen papers | +1.17 [+0.94, +1.41] | +0.07 [−2.47, +2.61] |
| label-semantics vote vs `crf_ens3`, unseen papers | +1.63 [+0.11, +3.16] | +0.40 [−1.36, +2.16] |

Each GPU interval still contains the CPU estimate, so the GPU runs do not
refute these gains either. The gains are about 1 point. The hardware alone
moves the 3-seed mean of the CRF head by 1.2 points per split on average
(up to 2.7, with the same seeds and splits). Three seeds per split cannot
settle effects of this size. The CPU decisions stand, as the protocol fixed.
`crf_ls_ens3` therefore remains the coarse-grained system on unseen papers,
with this caveat.

The changes that repeat on both machines are fine-grained, on unseen
papers:
- multi-task: +3.39 [+0.45, +6.34] on the CPU (week 3) and +3.87
  [−2.25, +9.99] on the GPU. Its GPU vote is +4.09 [−0.89, +9.07] against
  the GPU `crf_ens3`.
- self-training: +2.91 [−0.73, +6.56] on the CPU (week 3) and +2.14
  [−1.02, +5.29] on the GPU. With 3 teachers it is +1.07 [+0.33, +1.80].

Week 5 (protocol in §4) re-ran four families with 10 model seeds per split
on the GPU: the CRF reference, multi-task, label semantics and self-training
with 3 teachers. The week-4 runs give seeds 0–200 and week 5 added 300–900;
each vote is the mean over 3 disjoint seed triples. The runs repeat
exactly: every other week-5 session first repeated a week-4 run, and the 54
teachers retrained in the self-training session (3 per split or paper and
granularity) score exactly as the separately trained CRF runs. Week 5 used
10.7 GPU-hours, smoke sessions included.
- **The earlier claims, with 10 seeds** (decision 1):
  - Multi-task on the random splits: fine-grained improves, +1.83
    [+0.32, +3.34], positive on every split. Coarse-grained, the week-3
    claim does not stand: +0.71 [−0.15, +1.56].
  - Label semantics on unseen papers: neither granularity passes.
    Coarse-grained, the week-3 claim gives −0.36 [−1.66, +0.94];
    fine-grained +1.23 [−0.96, +3.42].
  - Self-training with 3 teachers on unseen papers: neither passes.
    Fine-grained, the week-4 claim gives +1.74 [−1.16, +4.64];
    coarse-grained +1.56 [−0.88, +4.00]. On the random splits (reported)
    it improves at both granularities: +1.05 [+0.10, +2.00] and +1.04
    [+0.16, +1.91].
- **The reported system** (decision 2). 2 of the 12 vote comparisons pass,
  where about 0.3 would pass by chance if no vote helped. Both are the
  multi-task vote:
  - coarse-grained on the random splits: +0.79 [+0.17, +1.42] over
    `crf_ens3`, positive on every split (79.43 against 78.63);
  - fine-grained on unseen papers: +3.65 [+0.49, +6.82], positive on every
    paper (52.47 against 48.81).

  `crf_ens3` keeps the other two slots: fine-grained on the random splits
  (79.68) and coarse-grained on unseen papers (61.10). There it replaces the
  label-semantics vote chosen on the CPU in week 4, which scores −0.63
  [−1.72, +0.46] against it on the GPU. The self-training vote gains +0.93
  to +1.12 in all four slots, but no CI excludes 0.
- **The published split** (week-5 amendment; numbers above). The CRF head
  adds +3.12 [+1.62, +4.62] fine-grained and +2.96 [+1.30, +4.61]
  coarse-grained over plain MatSciBERT, paired by seed; under strict IOB2
  scoring +1.70 [+0.21, +3.19] and +1.07 [−0.82, +2.97].
- **How far 3 seeds mislead** (exploratory). For each family against the
  CRF head on the protocol of its decision, a 3-seed comparison over the
  same splits would pass the rule for:
  - 41% of the 120 seed triples for fine-grained multi-task, the only one
    of these gains that passes with 10 seeds (+1.83);
  - 2–18% of them for label semantics and self-training on unseen papers,
    which do not pass with 10 seeds.

  Up to 37% of the triples give the opposite sign to the 10-seed difference
  (coarse-grained label semantics). On one split, the CRF head's micro F1
  varies over the 10 seeds with an SD of 1.5–1.7 points on the random splits
  and 2.2–2.5 on unseen papers. The week-3 design therefore detected a
  2-point gain less than half of the time, and its two coarse-grained passes
  (+0.75 and +1.17) do not hold with 10 seeds.
- **Rare coarse-grained types** (exploratory, item 5). In the reported
  system's votes on the random splits, SDE and MIP score about 18 F1, and
  PRP, MOD and CPR 40–42, with 2–8 test entities per split. Macro F1 counts
  each of them as much as MEC, with 53. They keep coarse-grained macro F1
  at 70.83 [64.49, 77.17], below 72.32 though not significantly.
- **The week-6 options at full size** (no decision; one run each, seed 0, on
  splits that no week-6 decision uses):
  - DeBERTa-v3-large with the CRF head scores 74.38 on the published split,
    fine-grained (MatSciBERT with the same seed: 73.49). It first trained
    with a NaN loss: transformers 5 loads a model in the dtype of its
    weights, and DeBERTa-v3's are float16. Models now load in float32,
    which leaves the MatSciBERT runs unchanged (the repeat run matches);
  - document context (64 subwords on each side) scores 74.56 on the same
    split;
  - on SOFC-Exp's official split, MatSciBERT with the CRF head scores micro
    84.98 and macro 82.31, against the published 84.77 and 82.39 (Gupta et
    al., 2022): the converted corpus behaves like the original.

**Week 6** (§4; GPU runs on the 4 held-out papers; done on 17 October: no
week-6 method passes a decision):
- **The screen** (seeds 0, 100 and 200, 3-seed means against the CRF head).
  DeBERTa-v3-large stops: −4.38 [−7.32, −1.45] fine-grained, −8.01
  [−11.72, −4.30] coarse-grained, with similar validation scores on the
  training papers. Document context goes to stage 2: +2.66 [+0.91, +4.40]
  fine-grained, −0.50 [−3.82, +2.83] coarse-grained. Multi-task
  self-training goes to stage 2 with the largest screen gain so far: +6.06
  [+1.05, +11.06] fine-grained, +0.25 [−1.12, +1.62] coarse-grained; its
  first vote +5.73 [+0.70, +10.77] against the seed vote, and +2.19
  [−1.61, +5.98] against multi-task training alone (reported). Document
  context's screen shrank with 10 seeds (below), so this one is decided the
  same way.
- **Document context with 10 seeds (stage 2)** passes neither decision.
  - Decision 1, against the CRF head: +1.14 [−0.94, +3.22] fine-grained
    and +0.12 [−2.03, +2.27] coarse-grained.
  - Decision 2, its vote against the strongest system on unseen papers:
    −1.44 [−3.80, +0.91] against the multi-task vote (fine-grained) and
    +0.76 [−1.79, +3.31] against the seed vote (coarse-grained).
  - Reported only: its vote against the seed vote fine-grained, +2.21
    [+0.01, +4.41].
- **Multi-task self-training with 10 seeds (stage 2)** passes none of its
  4 decisions (session b of 17 October, `w6c:mtst`, from bundle 9e990f0
  with session b's screen results; 7.2 hours on one lane, since the four
  papers' jobs extend the same files; the repeat `smokes` is identical at
  77.95, and the 168 multi-task teacher records it trained again equal the
  committed multi-task runs).
  - Decision 1, against the CRF head: fine-grained +5.47 [−0.74, +11.68]
    and coarse-grained +0.90 [−1.18, +2.98]. Fine-grained, three papers
    gain 4–9 points (+8.86, +8.31, +4.21), but 10.1038_s41467-020-19071-7
    gains +0.51. With 3 degrees of freedom the interval reaches −0.74.
  - Decision 2, its vote against the strongest system on unseen papers:
    +1.54 [−1.73, +4.82] against the multi-task vote (fine-grained) and
    +1.08 [−1.40, +3.57] against the seed vote (coarse-grained).
  - Reported: against multi-task training alone, +1.75 [−2.16, +5.67] and
    +1.46 [−0.99, +3.91]; its vote against the seed vote fine-grained,
    +5.20 [−0.59, +10.99]; on unseen names fine-grained, +6.35 [−0.13,
    +12.84] (seen +3.50 [−5.87, +12.86]).
  - Unlike document context's, its screen held up: the 7 seeds it did not
    see gain about +5.2 against +6.06 on the screen's 3. It has the largest
    mean gain of any method on unseen papers, and it still makes no claim.
    With 4 papers a gain must be both large and the same on every paper.
  - S2 on SOFC-Exp (`w6c:sofc-mtst`, the amendment below) does not run:
    it needed a pass on MaterioMiner.
  - *Why one paper gains little (exploratory, `ner_mentions.py`,
    `corpus_coverage`).* About half of a held-out paper's unseen names occur
    verbatim in the 40,000 corpus sentences that self-training tags: 49–62%
    of the unseen fine-grained mentions, 49–65% coarse-grained. The premise
    of self-training holds. But the paper that gains least,
    10.1038_s41467-020-19071-7, has the highest share (62.4%; the others
    49.5–58.4%). So the corpus's coverage of a paper's names does not
    explain the gain, at least over 4 papers.
- **The screen was optimistic, as week 5 predicted.** On the 7 seeds the
  screen did not use (300 to 900), context gains +0.49 [−3.10, +4.07]
  fine-grained, against +2.66 on the screen's 3. One paper
  (10.1038_ncomms7164) goes from +4.29 to −2.59. A screen that lets a
  method through on a noisy 3-seed estimate picks the methods that drew
  lucky seeds; the 10-seed decision is what corrects this.
- **SOFC-Exp, S1** (session b, model seeds 0 to 900): the CRF head scores
  83.75 [82.85, 84.65] (macro 81.11) and plain MatSciBERT 82.53 [81.84,
  83.23]. The CRF head improves on it, +1.22 [+0.50, +1.93] paired by seed
  (strict IOB2 +0.73 [−0.04, +1.51]): the week-5 finding repeats on a
  second dataset. The published MatSciBERT-CRF (84.77 micro) lies above the
  CRF head's CI over seeds (one seed scored 84.98 in the smoke test); the
  CRF votes over the 3 seed triples score 85.13.
- **SOFC-Exp, S2 for document context** (session a of 17 October,
  `w6c:sofc-ctx`, from the amendment's commit; its repeats of the week-4 run
  and of session b's SOFC-Exp CRF run of seed 0 are identical): 84.34 [83.91,
  84.76] against the CRF head's 83.75, +0.58 [−0.26, +1.42] paired by seed
  (macro +0.85 [−0.20, +1.90]): no significant change. Document context
  passes neither decision on MaterioMiner nor S2 on SOFC-Exp, so it stays out
  of the methods' combinations.
- **SOFC-Exp, S2 for DeBERTa-v3-large did not run.** It ran out of the T4's
  15 GB within 31 seconds: batch size 16 on SOFC-Exp's long sentences
  (session a, `w6c:ctx,sofc-deberta`). This is a deviation from the
  protocol, which planned S2 for DeBERTa whether or not it reached stage 2.
  Since DeBERTa stopped at the screen, it has not been run again (doing so
  would need gradient accumulation or a smaller batch, i.e. an amendment).
- **Checks.** Both sessions' repeats of the week-4 run are identical (micro
  F1 77.95), and the 72 multi-task teachers that session b trained again
  equal the committed multi-task runs. The week-7 consistency rule on document context, reported only:
  +0.10 [−0.67, +0.88] fine-grained and +0.06 [−0.09, +0.21]
  coarse-grained, and +0.45 [−0.46, +1.36] and +0.04 [−0.19, +0.28] on
  its votes.
- **Where the fine-grained gains come from** (exploratory; seeds 0 to 200,
  the 4 held-out papers). Multi-task training mostly raises precision (CRF
  head 52.1 precision, 42.7 recall; multi-task 58.8, 44.8), and
  self-training on top raises both (61.7, 46.3). Coarse-grained, all three
  sit near 59 and 59. Recall stays the bottleneck on unseen papers, which is
  what week 7's recall calibration targets; the two may add up.
- **Per type** (exploratory, 10 seeds × 4 papers): context's largest
  changes are on rare types, which a few entities swing (fine-grained
  MicrostructureEntity +18.9 and StrainRate −18.3; coarse-grained ODE −19.1
  and CPR −17.8).

**Seen and unseen names** (exploratory, 16 October; `ner_mentions.py` on
the saved predictions, no GPU; `results/gpu/ner_mentions.json`, claim C17,
Figure 5). Following Augenstein et al. (2017), a gold test entity is seen if
its exact words are an entity of the same type in the run's training
sentences, and unseen if they are no entity there. Each run's split is
rebuilt and checked against its stored test sentences, and the F1 over all
entities equals every run's stored micro F1.
- **The mix.** 81% of a held-out paper's fine-grained entities are unseen
  (79% coarse-grained; 72–83% per paper), against 35% (40%) on random splits
  and 40% (39%) on the published split. On SOFC-Exp's official split, 49%.
- **Recall per kind.** The CRF head finds 88% of the seen and 59% of the
  unseen fine-grained entities on random splits, 71% and 35% on held-out
  papers; coarse-grained 89/63 and 82/52; SOFC-Exp 89/82. The mix and the
  lower recall within each kind each account for 40–60% of the recall gap
  between the protocols (fine-grained 77.8 to 40.6; with the random splits'
  mix the held-out recalls per kind would give 58.5).
- **Per type.** Patterned names transfer to a new paper (unseen Value 76%,
  Alloy and MaterialClass 72%); descriptive ones do not (StateDescriptor
  30%, Mechanism 33%, GrainBoundary 3%, TestingMachine 6%).
- **Where the gains are.** Paired over the 4 held-out papers, fine-grained
  F1 on unseen names: multi-task +4.02 [+1.53, +6.51], its vote +3.60
  [+2.32, +4.88], the context vote +2.68 [+0.84, +4.53], multi-task
  self-training's screen +6.44 [+2.25, +10.62]; none is significant on the
  seen names, which are few. The CRF head's gain over plain MatSciBERT is
  on unseen names too (+5.94 on the published split, +3.02 on SOFC-Exp;
  −0.47 and −0.52 on seen names), and so is self-training's on the random
  splits (+2.61 and +2.26; seen +0.10 and +0.00).
- **What it means for the methods.** A method for unseen papers has to raise
  F1 on names the training papers do not contain. Mention replacement (part
  2b) targets this directly; self-training adds such names through its
  silver sentences.
- **Reported alongside the pending runs** (fixed now, before stage 2 of
  multi-task self-training, part 2b's screen and part 2d run; no decision):
  `ner_mentions.py` on their files, i.e. each method's F1 on seen and
  unseen names against the CRF head (its vote against the seed vote),
  paired over papers or seeds.

**Looser matching** (exploratory, 16 October; `ner_matching.py` on the
saved predictions, no GPU; `results/gpu/ner_matching.json`, claim C18). The
criteria of SemEval-2013 Task 9.1 (Segura-Bedmar et al., 2013): strict (span
and type, seqeval's micro F1), exact (span, any type), type (overlap, same
type) and overlap (any overlap, any type), each gold entity matched at most
once.
- **Coarse-grained,** the CRF head's gap between random splits and held-out
  papers is 18.4 points strict, 9.7 exact and 2.6 under overlap (held-out
  overlap F1 90.74 [89.17, 92.32] against 93.38). The model finds almost
  every entity of a new paper; it loses on boundaries and types.
- **Fine-grained,** 31.0 strict, 25.0 exact, 18.3 under overlap (70.58
  against 88.83): there entities are missed outright, most of them unseen
  names.
- **Near misses** on held-out papers: 31% of the fine-grained predictions
  (12% on random splits) overlap a gold entity with another span or type;
  10.8% lie inside a longer gold entity ("multi-GB cracks", "EBSD data",
  "smooth samples" where the model tags the noun), and 4.2% have an MMO
  ancestor or descendant of the gold type ("samples" as TestSpecimen or
  FatigueTestSpecimen, by paper).
- **SOFC-Exp:** 83.75 strict, 88.80 under overlap.
- **In the gold annotations** (`conventions`): 35 fine-grained names are
  entities in two or more papers; 7 take different types, 6 of them a class
  and its MMO subclass (DataSet/MechanicalTestDataSet,
  TestSpecimen/FatigueTestSpecimen, MechanicalProperty/TensileYieldStrength).
  Coarse-grained, 15 of 83. DataSet spans average 1.9 words in one paper,
  1.0–1.1 in the others.
- Strict F1 stays the decision metric; the looser criteria are reported
  alongside, as SemEval-2013 did.

**A third training paper** (exploratory, 16 October; `ner_seed_analysis.py`,
`paper_count`, no GPU; claim C19). Week 7's inner folds (part 2a) trained
the CRF head on two papers; their counts at D = 0 give its F1 on each
held-out paper with two training papers, against the leave-one-paper-out
runs with three (seeds 0–200 on both sides):
- fine-grained 39.02 → 46.38, +7.36 [+1.98, +12.74]; coarse-grained 50.37 →
  59.27, +8.90 [+5.90, +11.90]; every paper gains (+3.2 to +11.6);
- a third paper also brings more sentences, so this is a paper with its
  sentences, not diversity alone;
- it matches the largest method gain on unseen papers so far (multi-task
  self-training's screen, +6.06 fine-grained).
- week 8, part 2 separates the papers from the sentences (protocol in §4).
- *An outside reference point* (17 October, from the article, no runs).
  Potu et al. (2025, Fig. 2) fine-tune MatSciBERT on 80% of a corpus of 47
  other fatigue papers, annotated by MaterioMiner's curator with the same
  28 classes. They test on all of MaterioMiner and report 66.74 F1, against
  78.56 on their own validation sentences (mean of 5 seeds; GPT-4 in
  context 49.07 and 58.89). Our models trained on three MaterioMiner
  papers score 46–52 on the fourth. The setups differ: a token classifier
  without a CRF, 28 classes instead of 27, and a test set of all 4 papers.
  So with 47 training papers the gap to unseen papers is still about 12
  points, against 28–31 with three. Their training corpus is not public:
  the articles may not be redistributed.

Next, after week 5, in this order. The aim is a methods paper (§4, week 6),
with an evaluation paper if every option fails:
1. **Methods for unseen papers: screen, then decide with 10 seeds.**
   DeBERTa-v3-large, document context and multi-task self-training (items
   2–4 below), each against the CRF head on the 4 held-out papers. Week 5
   shows that 3 seeds cannot decide gains of 1–2 points, so the 3-seed
   screen only selects which methods get 10 seeds, and it lets almost every
   real gain through. A method's vote must then beat the strongest system
   on unseen papers: the multi-task vote fine-grained, `crf_ens3`
   coarse-grained. Multi-task self-training builds on the two changes that
   help: multi-task training, whose vote leads on unseen papers
   fine-grained, and self-training, which improves on the random splits.
2. **A second dataset.** SOFC-Exp is converted and checked (above). Its
   official split holds out 11 of its 45 articles, so it measures unseen
   documents. With 10 seeds there: the CRF head against plain MatSciBERT,
   and each method that passes item 1 against the CRF head. A methods paper
   needs its gain on both datasets.
3. **Combine the methods that pass** (week 7), for example multi-task
   training with a larger encoder or with document context. Screen a
   further method for unseen papers alongside: mention replacement (Dai and
   Adel, 2020; `--augment-mentions K`, prepared in week 6 without training).
   It adds K copies of each training sentence with its mentions swapped for
   other training mentions of the same type, so that the model learns the
   contexts rather than the names, which change from paper to paper.
   *Week 7, part 1 (§4): document-level label consistency, a post-processing
   rule that tags a paper's untagged occurrences of a term the model tags in
   most of them; no training.*
4. **Rare coarse-grained types** (item 5 below): silver sentences selected
   for SDE, MIP, PRP, MOD and CPR, or a class-balanced loss over them,
   judged by macro F1.
5. **Write the paper** (`docs/PAPER.md`): a methods paper if a change passes
   items 1 and 2, otherwise an evaluation paper on the reproduction, the gap
   to unseen papers, the CRF head's span validity and what 10 seeds change.

Week 5 closes label semantics: with 10 seeds, neither the model nor its vote
improves on either protocol.

After week 4 the list was, in this order (week 5 ran item 1 and prepared
items 2–7; protocol in §4):
1. **Ten seeds per split before the next claim.** Two week-3 gains and the
   CPU label-semantics vote did not repeat on the GPU, so 3 seeds per split
   cannot settle gains of about 1 point. A GPU run takes about a minute.
   Ten seeds on the 5 random splits and the 4 papers, both granularities,
   therefore take about 2 hours of a 2-GPU Kaggle session per family
   (self-training takes longer).
   - Re-run four families this way under the week-4 rule: the CRF
     reference, multi-task, label semantics, and self-training with 3
     teachers.
   - Decide the reported system on these 10-seed results.

   *Week 5: done. Fine-grained multi-task on the random splits passes; the
   coarse-grained claims of week 3 and the self-training claim of week 4 do
   not (above).*
2. **Multi-task and self-training together, fine-grained, on unseen
   papers.** These are the two changes whose fine-grained gains repeat on
   both machines:
   - multi-task: +3.39 on the CPU and +3.87 on the GPU;
   - self-training: +2.91 on the CPU and +2.14 on the GPU.

   They work differently, so their gains may add up. The design:
   - 3 multi-task teachers tag the corpus. This gives silver data for both
     granularities at once.
   - 3 multi-task students are trained, and their vote is compared with
     `crf_ens3`. Single students stay below the teachers' vote.
   - The silver filter is looser: 2 of 3 teachers, or a lower marginal,
     chosen on the validation sets. The unanimous filter keeps few
     sentences.
3. **A larger encoder.** Every encoder tried so far is base-sized:
   MatSciBERT, SciBERT and PubMedBERT. DeBERTa-v3-large (435M parameters)
   is the obvious candidate on a 16 GB T4. Screen it with 3 seeds, then
   use the 10-seed design if it helps.

   *Week 5 check (no training):*
   - *Snapshot.* `microsoft/deberta-v3-large` at 64a8c8e has no
     `tokenizer.json`, so the fast tokenizer is built from `spm.model`.
     That needs `sentencepiece` (0.2.2) and `protobuf`, which the Kaggle
     batch must pin.
   - *Tokenizer.* With them it splits pre-tokenised words as our encoder
     expects: one word ID per word, `▁` on a word's first piece.
   - *Still open.* The weights come as `pytorch_model.bin`. Memory and
     deterministic training on a T4 still need a GPU smoke test.
   - *Smoke sessions.* A run fits on a T4 and repeats exactly. It trains
     after the float32 fix: 74.38 with seed 0 on the published split
     (above).
4. **Document context.** Each sentence is tagged on its own, although the
   CoNLL files keep each paper's sentences in reading order. Give the
   encoder the neighbouring sentences as context and tag only the middle
   one (FLERT; Schweter and Akbik, 2020). This targets unseen papers, where
   a term is often introduced once and then reused. *Week 5: `--context N`;
   74.56 with seed 0 on the published split (above).*
5. **Rare coarse-grained types.** The coarse-grained claim fails only
   under macro F1 (73.03 [67.79, 78.27] vs 72.32), where rare types count
   as much as frequent ones. The results files keep per-class F1. Find the
   types that pull macro F1 down and target them, for example with silver
   sentences selected for those types. *Week 5: SDE, MIP, PRP, MOD and CPR
   (above).*
6. **Pin down the comparison with the paper** (item 4 below, still open):
   ask the authors for their split and averaging.

   *Week 5: the release ships the split of the paper's experiments, and its
   code scores micro F1 in seqeval's default mode. On that split, scored
   that way, plain MatSciBERT reproduces both published numbers (above). An
   email asking the authors to confirm the split, the scoring and the
   training settings is drafted for the owner to send.*
7. **A second materials NER dataset.** Every NER result so far rests on
   the 4 MaterioMiner papers. Repeating the main comparisons on another
   corpus would show which findings hold beyond them: the CRF head and its
   vote, the 10-seed design, and the gap between random splits and unseen
   documents. Candidates: SOFC-Exp (Friedrich et al., 2020) and MatScholar
   (Weston et al., 2019). Check availability, licence and format first.

   *Week 5 survey: SOFC-Exp is the choice.*
   - *Licence and texts.* Its annotations are CC BY 4.0, and the texts of its
     45 CC-BY articles ship with them.
   - *Splits.* Every document has an ID, so the leave-one-document-out
     protocol carries over. There is an official split by document (26
     train, 8 dev, 11 test).
   - *Published baselines.* MatSciBERT with a CRF reaches micro 84.77 and
     macro 82.39 (Gupta et al., 2022).
   - *Caveats.* It has 4 entity types (material, value, device, experiment),
     annotated only in the sentences that describe experiments, and a
     stand-off format, so the tokens must be rebuilt from character offsets.
   - *MatScholar.* It is larger (800 abstracts, 7 types, CC BY 4.0), but its
     released files lose the abstract boundaries, so no document-level
     protocol is possible.
   - *Converted* (`sofc_exp.py`, `--granularity sofc`): the counts of the
     paper, and on the official split the published MatSciBERT-CRF scores
     (above).

Week 4 closes label semantics on the multi-task model, and the multi-task
vote in distribution.

After week 3 the list was, in this order (week 4 ran items 1, 2, 3 and 5;
protocol in §4):
1. **Vote over the changes that help.** A seed ensemble gains about 2
   points over the mean of its members, and each week-3 gain is about 1
   point over a single CRF model. Vote over the 3 seeds of each, and
   compare the vote with `crf_ens3` under the week-3 rule:
   - label semantics, coarse-grained;
   - multi-task;
   - self-training, fine-grained, on unseen papers (the near-miss).

   The label-semantics and self-training runs saved their test
   predictions, so their votes need no training. The 27 multi-task runs
   must be repeated with `--save-predictions`. The reported system changes
   only for a vote that improves on `crf_ens3`. *Week 4: the
   label-semantics vote improves on `crf_ens3` coarse-grained on unseen
   papers on the CPU, which the GPU repeat does not confirm; the
   self-training vote misses again; the multi-task vote does not pass.*
2. **Combine the two changes that use the label hierarchy.** Multi-task
   training shares one encoder between the fine and coarse heads. Coarse
   label semantics describes each type by its fine-grained classes. Today
   `--label-semantics` is not available with `--multi-task`; the code
   change is a label-semantic head on each head of the multi-task model.
   *Week 4: done (`--multi-task --label-semantics`); no significant change
   on either protocol.*
3. **Self-training at scale, fine-grained.** The near-miss used 60–240
   silver sentences per run, out of 2,000 tagged. Two changes would give
   about 20 times more silver data:
   - tag all 40,000 corpus sentences;
   - use the ensemble as the teacher.

   The student would then take hours per run on CPU, so this needs a GPU
   (item 5). *Week 4: with 3 teachers, unanimous, fine-grained improves on
   unseen papers (+1.07 [+0.33, +1.80]). The silver data grew only about 6
   times, and the students stay below the teachers' vote.*
4. **Pin down the comparison with the paper.** Ask the authors for their
   split and averaging. The fine-grained claim holds under any averaging.
   The coarse-grained claim fails under macro F1, where rare types count as
   much as frequent ones. Rare types are also where the mixed-backbone vote
   lost (macro −4.11).
5. **A GPU.** A run takes 13–27 minutes on 2 CPU threads and about a
   minute on one GPU. That makes more seeds, larger backbones
   (DeBERTa-v3-large) and item 3 practical. *Week 4: done, on Kaggle's
   free T4 GPUs (§4).*

Domain-adaptive pre-training is closed: neither PubMed (week 2) nor
in-domain text (week 3) helps, although both lower the held-out perplexity.

After week 2 the list was, in this order (week 3 ran items 1–5; protocol in
§4):
1. **A comparison that can detect 1–3 points.** Compare the mean of 3 model
   seeds per split for the variant against the same for the reference,
   instead of single runs. The ensemble members `crf_m0`, `crf_m100` and
   `crf_m200` already give such a CRF reference on the 5 random splits and
   the 4 papers. Fix the design (seeds, splits, rule) before the runs, as
   in weeks 1 and 2. *Week 3: done; it detected gains of about 1 point.*
2. **Ensembles for the reported system.** A seed ensemble only costs
   training and inference time, and it gained on every split against its
   members. Next, mix backbones (SciBERT, PubMedBERT, DeBERTa-v3), whose
   errors should differ more than those of seeds. *Week 3: mixing
   backbones adds nothing to mixing seeds (item 8).*
3. **Out of distribution.** Leaving one paper out costs 18–27 points. Of
   the week-2 changes only the ensemble narrows it, by 1.6 points
   coarse-grained. Label semantics (item 6) and semi-supervised data
   (item 7) target it directly. *Week 3: label semantics improves
   coarse-grained; self-training does not pass, though fine-grained comes
   close.*
4. **Multi-task again, under the new design.** Its fine-grained difference
   was positive on all five splits (+2.46 on average) but not significant.
   *Week 3: coarse-grained improves on the random splits; fine-grained does
   not.*
5. **Domain-adaptive pre-training only on in-domain text.** PubMed did not
   help; the open-access fatigue papers of item 5 are untested. Low
   priority for its cost. *Week 3: no change at either granularity.*
6. **A GPU.** A run takes 13–27 minutes on 2 CPU threads; on one GPU it
   takes about a minute, which makes more seeds and larger backbones
   (DeBERTa-v3-large) practical. *Still open: item 5 of the list above.*

The P1 items, with what has been measured so far:

1. **Check that the reproduction holds across splits before claiming SOTA.**
   Result with a new random split per seed (`--vary-split`, 5 splits, micro
   F1). The rule, fixed before the runs: the gain holds only if the 95% CI
   lies above the published number.
   - Coarse-grained: 74.57, 95% CI [73.15, 75.98] vs 72.32. The gain holds.
   - Fine-grained: 73.91, 95% CI [69.36, 78.45] vs 69.92. Not significant;
     single splits range from 70.9 to 80.1.

   Caveats:
   - The 2024 paper does not say how its F1 is averaged. The authors' 2025
     follow-up uses micro F1, as we do. Our macro F1 is far below the
     published number for coarse (65.5 across splits) and close to it for
     fine (71.2).
   - Each test split has only about 96 sentences.

   Steps:
   - Done: 5 seeds × 2 granularities on the fixed split and on 5 new splits,
     with micro, macro and weighted F1 and 95% CIs (`summary` in the results
     JSON).
   - Done: `--leave-one-paper-out` (§1). Micro F1 averaged over the four
     held-out papers is 43.5 fine-grained and 54.3 coarse-grained, 20–30
     points below the random splits.
   - Done: paired comparison with this baseline on the same splits and model
     seeds (`--compare`, `results/ner_comparisons.json`). A fine-grained SOTA
     claim needs a significant gain over the baseline, not just a higher
     mean; the CRF head has one (item 4).

   A CPU run takes 9–25 minutes; on one GPU the whole protocol takes minutes.
2. **Hierarchy-aware multi-task training.** Coarse labels are the fine labels
   propagated up the MMO taxonomy, so a joint fine + coarse head (or a loss
   over the label hierarchy) shares signal with the 85 fine classes that have
   ≤ 3 examples. *(est. +1–3 F1 fine-grained)*

   Week 2, measured with `--crf --multi-task`: one MatSciBERT encoder with a
   CRF head per granularity, against the CRF head on the same splits and
   model seeds, one run each. No significant change in any of the four
   paired comparisons, so it was not adopted.

   | Micro F1 Δ [95% CI] | Fine-grained | Coarse-grained |
   |---|---|---|
   | 5 random splits | +2.46 [−0.91, +5.84] | +0.24 [−2.54, +3.02] |
   | Leaving one paper out | +0.11 [−8.56, +8.79] | −1.30 [−4.80, +2.20] |

   All five fine-grained differences across random splits are positive
   (+0.25 to +7.17), but one split (+7.17) gives most of the mean and the
   interval includes 0. On unseen papers the fine-grained differences range
   from −6.21 to +4.89.

   Week 3, with 3 seeds per split (the week-2 runs plus seed offsets 100
   and 200) against the 3-seed mean of the CRF head: coarse-grained improves
   under the pre-registered rule, fine-grained does not.

   | Micro F1 Δ [95% CI], 3-seed means | Fine-grained | Coarse-grained |
   |---|---|---|
   | 5 random splits (the decision) | +1.87 [−2.25, +6.00] | +0.75 [+0.10, +1.39] |
   | Leaving one paper out (reported) | +3.39 [+0.45, +6.34] | −0.60 [−4.55, +3.34] |

   The coarse-grained gain is small but positive on all five splits; macro
   F1 does not move (−0.50 [−2.63, +1.64]). On unseen papers the picture
   flips: fine-grained gains on all four papers (50.90 against 47.51 for
   the CRF head), coarse-grained does not change.
3. **Class-balanced or focal loss** for the minority classes the paper names
   as the main weakness. *(est. +0.5–2)*

   Measured with `--class-weights` (inverse square-root label frequency): no
   significant change in any of the four paired comparisons.

   | Micro F1 Δ [95% CI] | Fine-grained | Coarse-grained |
   |---|---|---|
   | 5 random splits | −0.24 [−3.09, +2.61] | +0.41 [−2.31, +3.13] |
   | Leaving one paper out | +1.81 [−4.28, +7.89] | +0.91 [−5.15, +6.98] |

   Macro F1, which gives rare classes more weight, does not change
   significantly either. Not adopted. A focal loss targets the same
   imbalance, so it now ranks below items 2 and 5.
4. **CRF or span-based decoding** to cut span errors, which are about 10% of
   errors in the 2025 study. *(est. +0.5–1.5)*

   Measured with `--crf`: a linear-chain CRF over first-subword emissions
   whose transitions forbid invalid IOB2 sequences. It improves all four
   paired comparisons, by more than the estimate.

   | Δ [95% CI] | Fine-grained | Coarse-grained |
   |---|---|---|
   | Micro F1, 5 random splits | +2.46 [+0.39, +4.52] | +3.58 [+2.43, +4.74] |
   | Strict IOB2 micro F1, 5 random splits | +0.10 [−2.08, +2.29] | +1.91 [+1.17, +2.65] |
   | Micro F1, leaving one paper out | +5.94 [+2.70, +9.18] | +5.47 [+0.43, +10.52] |
   | Strict IOB2 micro F1, leaving one paper out | +4.40 [+2.09, +6.72] | +3.03 [−1.00, +7.07] |

   Under strict IOB2 scoring, the in-distribution fine-grained gain vanishes.
   It comes from no longer emitting stray I- tags, which seqeval's default
   scoring counts as entities. The baseline's own strict F1 is already 76.26
   [72.52, 80.01]. The coarse-grained gain across random splits and the
   fine-grained gain on an unseen paper survive strict scoring.

   Next:
   - Use `--crf` for every new NER experiment, and judge items 2 and 5–8
     against the CRF results rather than the plain baseline.
   - Report strict micro F1 next to the default.
   - For models without a CRF, constrained decoding (repair or drop stray
     I- tags) gives the span-validity part of the gain for free.
   - Span-based decoding remains untested.
5. **Domain-adaptive pre-training.** Continue MLM on the 19k PubMed sentences
   already in the repo, plus open-access fatigue papers, before fine-tuning.
   *(est. +1–3)*

   Pre-training is done (`dapt_mlm.py`, PubMed sentences only): 3 epochs on
   18,353 sentences, after dropping 1 duplicate and 13 sentences that share a
   word 8-gram with MaterioMiner. Held-out MLM perplexity fell from 12.17 to
   5.51 (`results/dapt_mlm.json`).

   Fine-tuned with `--crf` from this checkpoint, against the CRF head on the
   same splits and model seeds: no significant change in any of the four
   paired comparisons, so it is not adopted.

   | Micro F1 Δ [95% CI] | Fine-grained | Coarse-grained |
   |---|---|---|
   | 5 random splits | +0.68 [−2.55, +3.90] | −0.07 [−3.01, +2.86] |
   | Leaving one paper out | −0.91 [−3.13, +1.31] | −1.06 [−3.25, +1.12] |

   The lower perplexity on PubMed text did not carry over to MaterioMiner;
   on unseen papers 6 of the 8 differences are negative. Week 2 did not use
   the second corpus named above, open-access fatigue papers.

   Week 3 builds that corpus (`build_fatigue_corpus.py`: 40,000 sentences
   from open-access fatigue papers in Europe PMC, `results/fatigue_corpus.json`)
   and repeats the pre-training on it: 3 epochs on 38,000 sentences, with
   2,000 held out. No sentence is a duplicate or shares a word 8-gram with
   MaterioMiner. Held-out MLM perplexity fell from 10.75 to 4.43
   (`results/dapt_mlm_fatigue.json`). The CRF head is then fine-tuned from
   the new checkpoint with 3 seeds per split, and the decision is taken on
   the random splits.

   | Micro F1 Δ [95% CI], 3-seed means | Fine-grained | Coarse-grained |
   |---|---|---|
   | 5 random splits (the decision) | +0.11 [−2.74, +2.97] | −0.15 [−1.06, +0.76] |
   | Leaving one paper out (reported) | +0.14 [−2.60, +2.88] | −1.54 [−4.34, +1.26] |

   In-domain pre-training does not help either granularity, so it is not
   adopted:
   - fine-grained, 77.40 against 77.29 for the CRF head, with per-split
     differences from −2.22 to +3.53 and macro F1 −0.16 [−3.69, +3.37];
   - coarse-grained, 77.50 against 77.65, with macro F1 −0.63 [−3.29,
     +2.04].

   It does not help on unseen papers either: +0.14 fine-grained, and −1.54
   coarse-grained, where all four papers are negative. As with PubMed in
   week 2, a lower held-out perplexity does not carry over to MaterioMiner
   NER.
6. **Label semantics.** Bi-encoder span–type matching (BINDER, GLiNER-style)
   with the 420 MMO `skos:definition`s as type descriptions. This helps rare
   and unseen classes, and it supports the full 177-class setting, not just
   the top 27.

   Week 3 scores each tag against an embedding of its MMO description
   (`--label-semantics`: the dot product of the token state with a learned
   map of the description embedding; protocol in §4). There are 3 seeds per
   split, compared with the 3-seed mean of the CRF head, and the decision
   is taken on unseen papers.

   | Micro F1 Δ [95% CI], 3-seed means | Fine-grained | Coarse-grained |
   |---|---|---|
   | Leaving one paper out (the decision) | −0.16 [−2.13, +1.80] | +1.17 [+0.94, +1.41] |
   | 5 random splits (reported) | −0.58 [−3.41, +2.24] | −0.77 [−1.74, +0.21] |

   Coarse-grained, label semantics improves on unseen papers under the
   pre-registered rule. It scores 60.21 against 59.03 for the CRF head, and
   the gain is between +1.05 and +1.38 on each of the four papers. Macro F1
   gains +1.20 [−1.38, +3.78], which is not significant. Coarse types have
   no MMO definitions, so their descriptions are the names of their
   fine-grained classes (§4).

   Fine-grained, it does not help: 47.35 against 47.51 for the CRF head.
   The per-paper differences range from −1.70 to +1.27, and macro F1
   changes by −1.17 [−4.73, +2.38].

   On the random splits it helps neither granularity (reported): 76.71
   against 77.29 fine-grained, and 76.89 against 77.65 coarse-grained,
   where four of the five splits are lower. The coarse-grained gain is
   specific to unseen papers.

   One label-semantics model (60.21) stays below the reported seed ensemble
   `crf_ens3` on unseen papers (61.42, coarse-grained). An ensemble of
   label-semantics models was not part of the protocol.
7. **Semi-supervised data.** Self-train on PubMed with a confidence threshold.
   Generate LLM silver labels with ontology-definition prompts, filter them by
   agreement with the teacher model, and add them as weak supervision.

   Week 3 self-trains on the open-access fatigue corpus (`--self-train`;
   protocol in §4):
   - The teacher, the CRF head, tags 2,000 in-domain sentences.
   - A sentence becomes silver data when every tag has a CRF marginal of at
     least 0.9.
   - A student trains on gold + silver data, with the silver loss weighted
     0.5.
   - The runs cover unseen papers only, with 3 seeds per paper, compared
     with the 3-seed mean of the CRF head. The decision is taken on the 4
     papers.

   | Micro F1 Δ [95% CI], 3-seed means | Fine-grained | Coarse-grained |
   |---|---|---|
   | Leaving one paper out (the decision) | +2.91 [−0.73, +6.56] | −0.12 [−1.97, +1.73] |

   Neither granularity improves under the pre-registered rule. Every teacher
   reproduces the CRF member with the same seed exactly, so each student is
   paired with its own teacher.

   Fine-grained, the gain is not significant but points one way: 50.42
   against 47.51 for the CRF head, positive on all four papers (+1.00 to
   +6.10), and 9 of the 12 students beat their teachers. Macro F1 changes
   by +2.17 [−3.18, +7.53]. With 4 papers, the 95% interval is wide.

   Coarse-grained, self-training does not help: 58.92 against 59.03, and
   macro F1 changes by −0.21 [−2.37, +1.96]. Against their teachers,
   students change by −4.89 to +2.12 points.

   The confidence filter keeps few sentences: 60 to 240 of 2,000 per run
   fine-grained, 29 to 345 coarse-grained.
8. **Ensembles.** Average across seeds and backbones (MatSciBERT, SciBERT,
   PubMedBERT, DeBERTa-v3-large); this usually gives a reliable +1–2.

   Measured with `ner_ensemble.py`: an entity-level majority vote of three
   CRF models (model seeds K, K + 100, K + 200), against the CRF head on the
   same splits. No significant change across the 5 random splits, so under
   the pre-registered rule it does not improve on the CRF head.

   | Micro F1 Δ [95% CI] | Fine-grained | Coarse-grained |
   |---|---|---|
   | 5 random splits | +3.39 [−1.35, +8.14] | +1.30 [−1.01, +3.62] |
   | Leaving one paper out | +0.11 [−1.99, +2.20] | +1.60 [+0.52, +2.69] |

   The ensemble has the highest scores so far: 79.76 [76.14, 83.38]
   fine-grained and 79.45 [77.82, 81.09] coarse-grained. The CRF head it is
   compared with is one model per split, and one model is noisy: re-running
   it with 2 threads moves single fine-grained splits by −3.28 to +6.35.
   Against the mean of its own three members on the same split, which no
   single run dominates, the vote gains +2.47 [+1.17, +3.77] fine-grained
   and +1.80 [+0.91, +2.69] coarse-grained, on every split (exploratory,
   not pre-registered).

   On unseen papers the ensemble reaches 49.55 fine-grained and 61.42
   coarse-grained. Coarse-grained, it is the only week-2 change that
   improves on the CRF head there, and it beats its own members on all
   four papers (+2.39 [+1.56, +3.21]). Fine-grained it gains +2.04
   [−0.23, +4.32] over its members, also positive on every paper.

   Week 3 mixes backbones: one SciBERT and one PubMedBERT CRF model per
   split vote with the MatSciBERT member `crf_m0` (`crf_mix3`). The
   pre-registered comparison is against the seed ensemble `crf_ens3`.

   | Micro F1 Δ [95% CI], mixed vs seed ensemble | Fine-grained | Coarse-grained |
   |---|---|---|
   | 5 random splits (the decision) | +0.03 [−1.80, +1.86] | −0.87 [−2.09, +0.35] |
   | Leaving one paper out (reported) | −1.34 [−4.76, +2.09] | −1.45 [−5.74, +2.84] |

   Mixing backbones adds nothing to mixing seeds, so `crf_ens3` stays the
   reported system.
   - Fine-grained, both votes score 79.8 and gain about 2.5 points on the
     3-seed CRF reference: `crf_ens3` +2.47 [+1.17, +3.77], `crf_mix3`
     +2.50 [+0.07, +4.92].
   - Coarse-grained, the mixed vote is lower on four of the five splits and
     loses macro F1 (−4.11 [−7.87, −0.36]): the weaker backbones outvote
     MatSciBERT on rare classes.
   - On unseen papers the mixed vote is also below the seed ensemble:
     48.21 against 49.55 fine-grained, 59.97 against 61.42 coarse-grained.

   Neither new backbone beats MatSciBERT on its own (one run per split or
   paper, against the 3-seed mean), and both fall further behind on unseen
   papers:

   | Micro F1 Δ [95% CI] vs the CRF head | Fine-grained | Coarse-grained |
   |---|---|---|
   | SciBERT, 5 random splits | −0.38 [−2.63, +1.87] | −0.99 [−1.71, −0.26] |
   | SciBERT, leaving one paper out | −4.83 [−6.66, −3.00] | −4.52 [−8.17, −0.87] |
   | PubMedBERT, 5 random splits | −1.42 [−5.21, +2.37] | −2.99 [−4.67, −1.32] |
   | PubMedBERT, leaving one paper out | −4.94 [−13.22, +3.34] | −3.75 [−8.41, +0.90] |
9. **Track out-of-distribution results** with `--leave-one-paper-out` for
   every change. It is cheap and closer to real use than random sentence
   splits. The baseline drops from about 74 to 44 (fine-grained) and 54
   (coarse-grained) on an unseen paper; the CRF head lifts these to 49 and
   60, and the 3-model ensemble to 50 and 61. That gap is larger than any in-distribution gain on this list, so
   items 5–7 should be judged on this protocol as well.

### P2: Term typing (LLMs4OL Task B style)
The cross-validated hybrid + LLM baseline is far ahead of the pipeline's
zero-shot typing (§1). Next steps:
1. **Make `align.py` use the baseline's components**: definition scoring, and
   kNN over labelled seed terms when any are available. Keep the zero-shot
   path for unseen ontologies.
2. **Context-aware term embeddings**: encode the term inside its sentence
   (MatSciBERT mean-pooled span) rather than the bare string. Many errors are
   short ambiguous strings ("data", "fatigue", "diameter").
3. **Contrastive bi-encoder, then cross-encoder.** Train on (term + context) ↔
   (label + definition) pairs, with hard negatives taken from sibling classes
   in the MMO tree. This was the IRIS recipe, which won MatOnto in 2025.
4. **Hierarchical decoding**: pick the coarse class first, then a child.
   Hierarchical F1 is already reported, which makes "near misses" visible.
5. **Better LLM prompting**:
   - retrieve demonstrations from labelled terms only;
   - ensemble several prompts and models (DREAM-LLMs style deliberation);
   - abstain on low agreement;
   - note the contamination caveat: 213 of the 420 MMO definitions were
     generated by GPT-4 (163 of them with manual adjustments).
6. **Abstention.** The risk–coverage curve shows precision > 0.8 at about 15%
   coverage. Calibrate an abstention threshold on validation folds for
   high-precision ontology population.

### P3: Make the graph learning matter
Diagnosis from this PR:
- The link-prediction GNN had collapsed to rank-1 embeddings. Its cosine
  similarities were meaningless; they drove clustering, taxonomy and 2,207
  "GNN relations" in v5.
- After the fix (residual projection, lr 1e-3, chosen on validation
  link-prediction loss), the GNN contributes only through term-side score
  smoothing, and the ablations show that this does not matter:
  - typing accuracy is 0.384 with it and 0.378 without (McNemar p = 0.69);
  - clustering on the GNN embeddings is far worse than on plain SBERT
    (ARI 0.035 vs 0.195).
- Link-prediction embeddings encode which terms co-occur, not which class
  they belong to. The steps below train the graph model for the typing
  objective instead.

Next:
1. **Heterogeneous term–class graph.**
   - Edges: term–term PMI edges, class–class `subClassOf` edges, and term–class
     edges for training-fold terms.
   - Train for node classification or link prediction to classes (R-GCN,
     GATv2 with edge weights) and evaluate inside the same cross-validation
     folds as the baselines. This is the natural "GNN for ontology learning"
     experiment.
2. **Use the PMI weights** in message passing (`SAGEConv` ignores them today).
   Compare co-occurrence windows: sentence, paragraph, document.
3. **Pre-train contrastively** (GraphCL/DGI), then fine-tune for typing.

### P4: Taxonomy discovery (class level)
The current term-level taxonomy (most similar more-general term plus Hearst
patterns) is not designed to recover MMO `subClassOf`, and it scores near
chance at class level. For a Task C-style result:
1. Predict hypernym pairs among the MMO classes that occur in the corpus.
   Inputs are label, definition and a few typed example terms; use an LLM or a
   fine-tuned cross-encoder, cross-validated over edges.
2. Decode a consistent hierarchy with a maximum spanning arborescence or a
   transitive-reduction constraint.
3. Report edge F1, ancestor F1 and the inverted-edge rate (already
   implemented in `metrics.taxonomy_edge_metrics`).

### P5: Relations
1. Annotate a relation layer for the four papers. The MaterioMiner authors
   plan this using the MMO object properties. Even 200 relation mentions would
   allow a real text-level relation-extraction benchmark.
2. Meanwhile, use distant supervision from the 70 class-level restrictions:
   - prompt an LLM with the property inventory and OWL domain/range;
   - score at class level (`eval_relations.py`).
3. Drop relation types without an ontology property, such as `relatedTo`, or
   map them to `associatedWith`.

### P6: Engineering
- YAML or Hydra configs instead of environment variables. Cache embeddings
  across stages and presets.
- Run the large-embedder presets (`domain_specter`, `domain_scibert`) and the
  NER script on GPU.
- Use the real `emmo.ttl` and `bfo.owl` files in `cross_domain_align.py`
  instead of hand-written upper-level class lists, with a gold mapping through
  PMDco where it exists.

## 4. Suggested order

| Week | Work | Exit criterion |
|---|---|---|
| 1 | Done: class weights (no significant change) and CRF (significant gain in all four comparisons), P1 items 3–4 | Met by the CRF head |
| 2 | Done: multi-task NER (fine + coarse heads), domain-adaptive pre-training and a 3-model ensemble, all with `--crf` (P1 items 2, 5 and 8; protocol below) | Not met: no 95% CI lies above 0 against the CRF head on the random splits; the ensemble comes closest (+3.39 fine-grained, +1.30 coarse-grained) and is the only change that improves on unseen papers (coarse-grained) |
| 3 | Done: NER, P1 "Next, after week 2" items 1–5: comparisons of 3-seed means, a mixed-backbone ensemble, label semantics and self-training for unseen papers, multi-task again, in-domain pre-training (protocol below) | Met for two changes, both coarse-grained: multi-task on the random splits (+0.75 [+0.10, +1.39]) and label semantics on unseen papers (+1.17 [+0.94, +1.41]). The mixed-backbone vote, self-training and in-domain pre-training do not pass; the reported system stays `crf_ens3` |
| 4 | Done for NER: P1 "Next, after week 3" items 1, 2, 3 and 5, i.e. votes over the week-3 changes that help, label semantics on the multi-task model, self-training at scale, on Kaggle T4 GPUs (protocol below). Not started: context-aware bi-encoder/cross-encoder typing + definition kNN in `align.py`; heterogeneous GNN | Met for NER by two changes, both on unseen papers. Coarse-grained, the label-semantics vote improves on `crf_ens3` on the CPU (+1.63 [+0.11, +3.16]); the GPU repeat does not confirm it (+0.40 [−1.36, +2.16]). Fine-grained, self-training with 3 teachers improves on the CRF head (GPU, +1.07 [+0.33, +1.80]; +1.22 [+0.60, +1.84] on the random splits, reported). The multi-task vote and label semantics on the multi-task model do not pass. Typing: not started |
| 5 | Done: NER, P1 "Next, after week 4" item 1, i.e. 10 seeds per split for the CRF reference, multi-task, label semantics and self-training with 3 teachers, on Kaggle T4 GPUs, and the published split (amendment); items 2–7 and a paper outline prepared, and the week-6 options run once each at full size (protocol below). The original week-5 work moves to week 7 | Met. Of the claims re-tested with 10 seeds, fine-grained multi-task on the random splits passes (+1.83 [+0.32, +3.34]); the coarse-grained claims of week 3 and the self-training claim of week 4 do not. The reported system: the multi-task vote coarse-grained on the random splits and fine-grained on unseen papers, `crf_ens3` elsewhere. On the published split, plain MatSciBERT reproduces 69.92 and 72.32 and the CRF head beats both |
| 6 | Done: NER, P1 "Next, after week 5", towards a methods paper: methods for unseen papers (DeBERTa-v3-large, document context, multi-task self-training) screened with 3 seeds and decided with 10, and a second dataset (SOFC-Exp, 10 seeds); protocol below. Done (P1, "Week 6"): DeBERTa-v3-large stops at the screen; document context passes the screen but neither 10-seed decision nor S2 on SOFC-Exp (+0.58 [−0.26, +1.42]); multi-task self-training passes the screen (+6.06 fine-grained) but none of its 4 decisions (+5.47 [−0.74, +11.68] fine-grained against the CRF head), so its S2 on SOFC-Exp does not run; on SOFC-Exp the CRF head improves on plain MatSciBERT (S1) | A method whose vote improves on the strongest system on unseen papers under the 10-seed design, and on SOFC-Exp |
| 7 | In progress: NER, part 1, document-level label consistency for unseen papers, a post-processing rule on the saved predictions (no GPU; protocol below); part 2a, recall calibration (a bias on the CRF head's emissions, chosen on inner folds; done, no claim: the missed entities are not near the threshold); part 2b, mention replacement (screen, then 10 seeds if it passes; protocol below; done: it improves on the CRF head at both granularities, +2.77 [+1.51, +4.03] fine-grained and +1.28 [+0.06, +2.51] coarse-grained, but its vote does not beat the strongest systems and S2 on SOFC-Exp gives −0.29), and later the combinations of the methods that pass week 6, fixed after the week-6 decisions; part 2c, coarse-to-fine completion (done, no claim); part 2d, multi-task training on SOFC-Exp's two levels (done, no claim: −0.78 [−2.39, +0.84] on the slot level, −0.15 [−1.36, +1.05] on the entity level); part 2e, SOFC-Exp's test articles cut into words as its training articles are (protocol below; done: both decisions pass, the shift costs the CRF head +0.86 [+0.39, +1.33] and plain MatSciBERT +1.91 [+1.33, +2.49], the CRF head then scores 84.61 [83.89, 85.34], which covers the published MatSciBERT-CRF's 84.77, and its gain over plain MatSciBERT falls to +0.17 [−0.37, +0.70]). Then class-level taxonomy (LLM or cross-encoder + arborescence) and a relation annotation pilot | Part 1: the rule improves on the strongest system on unseen papers and on SOFC-Exp's CRF head. Taxonomy: first class-level F1 above chance; relation guidelines |
| 8 | Part 1 done: SOFC-Exp beyond its official split, the CRF head with 10 seeds on 5 random sentence splits and on 5 folds of 9 held-out articles (`--document-folds 5`; protocol below, descriptive): 86.27 against 82.79, a gap of 3.48 points (28–31 on MaterioMiner); part 2 done (its first session hung and was cancelled; it ran after week 9), more papers or more sentences: three training papers cut to the size of two against two papers (`--two-paper-size`; protocol below). Both decisions pass: at the size of two papers, a third paper adds +1.18 [+0.22, +2.15] fine-grained and +3.59 [+1.59, +5.58] coarse-grained, and its sentences add +6.17 and +5.31 more | Part 1: the gap between random splits and unseen documents measured on a second corpus. Part 2: whether a third paper helps at the same number of training sentences |
| 9 | Part 1 done: mention replacement on the multi-task model (`crf_mt_mr`, 10 seeds on the held-out papers, S2 on SOFC-Exp's two levels; protocol below). 3 of 6 decisions pass: against the multi-task model +3.01 [+0.42, +5.60] fine-grained and +2.30 [+0.49, +4.10] coarse-grained, and its vote against the multi-task vote +2.86 [+0.18, +5.54] fine-grained, now the strongest fine-grained system on unseen papers (55.33); coarse-grained its vote does not beat the seed vote, and S2 fails at both levels, so no methods claim (C21). Part 2 done: on SOFC-Exp cut to MaterioMiner's size (10 of its 26 training articles, 5 draws), mention replacement scores +0.50 [−0.47, +1.46] against the CRF head and does not pass, so its MaterioMiner gain is not a matter of training size | Part 1: the combination's vote improves on the strongest system on unseen papers, and on SOFC-Exp. Part 2: mention replacement improves on the CRF head on a MaterioMiner-sized SOFC-Exp |
| 10 | Done: MuLMS as a third corpus (7 materials subfields, its outermost entities; `mulms.py`), mention replacement against the CRF head on held-out subfields with 400 training sentences, 10 seeds, and the CRF head on random splits of the same size; protocol committed before any run (a08356f; the batch `w10a` ran from that commit), protocol and outcome below | Mention replacement improves on the CRF head on held-out subfields (decision 1), and its votes on the CRF votes (decision 2). Decision 1 not met: +2.10 [−0.31, +4.51], six of the seven subfields gaining. Decision 2 met: +2.71 [+0.22, +5.20]. By the protocol, mention replacement and its combination (C20, C21) stay results for MaterioMiner |
| 11 | Protocol committed before any run: mention replacement whose replacement pool also holds the names in the sentences that 3 CRF teachers tag alike and confidently in unlabeled in-domain text (`--augment-names`, variant `crf_mrs`), on MuLMS's 7 held-out subfields (`w11a`) and MaterioMiner's 4 held-out papers (`w11b`), 10 seeds; protocol below | It improves on the CRF head on MuLMS's held-out subfields (decision 1) and on mention replacement (decisions 2–4: MuLMS, MaterioMiner fine- and coarse-grained); a methods claim needs decision 1 and one of decisions 2–4. MuLMS (`w11a`): decisions 1 (+2.04 [−0.85, +4.92]) and 2 (−0.06 [−1.09, +0.96]) not met, so no methods claim; MaterioMiner (`w11b`, 19–20 October): decisions 3 (fine-grained, +0.23 [−0.69, +1.14]) and 4 (coarse-grained, +0.61 [−0.96, +2.18]) not met either; the method is closed on both corpora |
| 12 | Protocol committed before any run: mention replacement whose replacement pool holds the names that 3 CRF teachers find in a corpus of the held-out subfield's own topic (40,000 open-access sentences per subfield, `build_topic_corpus.py`; phase 0's gate: they hold 55–81% of the unseen test names), on MuLMS's 7 held-out subfields; a 3-seed screen of seven variants (`w12a`: silver sentences `crf_mrs_topic`, silver mentions `crf_mrm_topic`, the same with 6 copies `crf_mrm_topic6`, 6 copies without names `crf_mr6`, self-training on the topic corpus `crf_st3_topic`; `w12c`: the silver mentions or sentences as a gazetteer match feature, `crf_gzm_topic` and `crf_gzs_topic`; amendments 1–3), then the 7-seed decision for the screen's winner on seeds the screen did not see (`w12b:<variant>`); protocol below, plan in `docs/METHODS_PLAN.md` | Screen: the variant with the largest mean gain over mention replacement goes on if it is above 0. Decision 1: it improves on the CRF head (paired by subfield); decision 2: on mention replacement. A methods claim needs decision 1 on MuLMS and the winner's MaterioMiner analogue passing against the CRF head. Screen done (20 October; 3 seeds, 7 subfields; outcome below): V4 `crf_mrm_topic6`, the mention-level filter with 6 copies, 52.09, +1.45 [−1.35, +4.25] over mention replacement and +3.16 [−1.96, +8.28] over the CRF head, is the winner (V2 +0.97, V1 +0.80, V6 +0.16; V3, V7 and V8 below mention replacement). Decision `w12b:mrm6` (seeds 300–900) after the reset of 24 October, then `w12d:mrm6` (`crf_mrm6` on MaterioMiner's held-out papers) if decision 1 passes, and 13.2's pooled decision |
| 13 | Protocol committed before any run (20 October, on the user's directive to push for a methods paper; the plan's stop rule of a third protocol on MuLMS lifted, the count reported): 13.1 the document consistency rule of week 7 on MuLMS's saved 10-seed predictions (no training; amended the same day with SOFC-Exp's article folds, reported, and mention replacement on MaterioMiner, decisions); 13.2 a pooled decision over MuLMS's 7 subfields and MaterioMiner's 4 papers for the week-12 winner; 13.3 mention replacement with and without multi-task training on SOFC-Exp's 5 article folds, 10 seeds (`w13a`); protocol below | Revised methods criterion: a pre-registered pass on held-out documents of two corpora, or the pooled decision over two or more corpora with no corpus losing; claims scoped to what passed. 13.1 done: on the CRF head the rule passes on MuLMS (13.1a, +0.82 [+0.18, +1.46], every subfield gaining), its second corpus after coarse-grained MaterioMiner (week 7, +0.12), and loses on SOFC-Exp's folds (13.1c, −0.44 [−0.72, −0.16]); it adds nothing to mention replacement (13.1b +0.57 [−0.12, +1.25]; 13.1d +0.19 [−0.34, +0.73] fine-grained, +0.11 [−0.08, +0.31] coarse-grained): a scoped claim of two corpora, under a point, with a boundary condition. 13.2 (for V4 `crf_mrm_topic6`, the screen's winner, with its MaterioMiner analogue `w12d:mrm6`) and 13.3 after the reset of 24 October |

### Week 2 protocol (fixed before any run)

The reference is the CRF head (`--crf`) on the same splits and model seeds:
the 5 random splits (`--vary-split`) and the 4 held-out papers. Each variant
changes one thing; every other setting is that of the CRF runs.

1. **Multi-task training** (`--crf --multi-task`, P1 item 2).
   - One MatSciBERT encoder with a CRF head per granularity. The loss is the
     fine NLL plus the coarse NLL, weighted equally.
   - Each head trains on exactly the sentences and tags of its single-task
     run. The fine and coarse files tokenise 15 of the 476 sentences
     differently; those enter as two examples, each labelled for one head.
   - Each granularity's test score uses the epoch with its best validation
     F1, as in the single-task runs. One run gives both granularities.
2. **Domain-adaptive pre-training** (`dapt_mlm.py`, then
   `--crf --model <checkpoint>`, P1 item 5).
   - Continued masked-LM training of MatSciBERT on
     `data/raw/dataset/pubmed_sentences.txt`. Duplicates and every sentence
     that shares a word 8-gram with MaterioMiner text are dropped first, so
     no evaluation text is seen.
   - 15% masking (80/10/10), lr 5e-5, batch 32, at most 128 tokens,
     3 epochs; the last checkpoint is used. 5% of the sentences are held out
     to report MLM perplexity.
3. **Ensembles** (`ner_ensemble.py`, P1 item 8).
   - Three CRF models per split: model seeds K, K + 100 and K + 200 on
     split K, and seeds 0, 100 and 200 for each held-out paper.
   - Entity-level majority vote: an entity is kept if at least 2 of the 3
     models predict the same span and type.

Decision rule, as in week 1: a variant improves on the CRF head for a
granularity only if the 95% CI of the mean paired micro-F1 difference over
the 5 random splits lies above 0. The same comparison over the 4 held-out
papers is reported as out-of-distribution robustness, and strict IOB2 and
macro F1 are reported alongside. Results are reported in the order above;
runs may overlap when CPU is free.

Deviation, noted after the first runs: two jobs share the 4 CPUs, so the
week-2 runs use 2 threads where the CRF results used 3. The thread count
alone changes a run with the same seeds (fine-grained split 0: 78.76 with 3
threads, 76.37 with 2), which adds noise to every difference. The `crf_m0`
ensemble members re-run the CRF head with 2 threads and measure the effect.

### Week 3 protocol (fixed before any run)

Week 3 runs items 1–5 of P1 "Next, after week 2". Every run uses `--crf`
and 2 CPU threads, and the week-2 code paths stay unchanged: new options only
add code.

**Design (item 1).**
- Every model is trained three times per split: model seeds K, K + 100 and
  K + 200 on random split K, and seeds 0, 100 and 200 on each held-out
  paper. A single model scores the mean micro F1 of its three runs on that
  split; an ensemble scores once.
- The reference is the CRF head scored the same way: the mean of the week-2
  runs `crf_m0`, `crf_m100` and `crf_m200`.
- Decision rule: a variant improves on its comparison if the 95% Student-t
  CI of the mean paired difference over the 5 random splits lies above 0.
  The same statistic over the 4 held-out papers is reported as
  out-of-distribution robustness. For the out-of-distribution methods
  (variants 2 and 3 below) the decision is taken on the 4 held-out papers
  instead, and the random splits are reported alongside.
- Strict IOB2 and macro F1 are reported with every comparison.
- With the week-2 spread between seeds (about 1.6 F1 per run,
  fine-grained), this design should detect gains of about 2 F1 points; one
  run per split could not detect gains below about 5.

**Variants**, reported in this order:

1. **Mixed-backbone ensemble** (P1 item 8).
   - CRF models from SciBERT (`allenai/scibert_scivocab_uncased`, results
     `crf_scibert`) and PubMedBERT
     (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`,
     `crf_pubmedbert`), with the MatSciBERT settings and one model seed per
     split (K on split K, 0 on each paper).
   - `crf_mix3`: the entity-level majority vote of MatSciBERT (`crf_m0`),
     SciBERT and PubMedBERT.
   - Primary comparison: `crf_mix3` against the MatSciBERT seed ensemble
     `crf_ens3` on the same splits, i.e. does mixing backbones add to mixing
     seeds? Both are also reported against the CRF reference.
   - The reported NER system (README, §1) is `crf_ens3`, unless `crf_mix3`
     improves on it under the rule above.
2. **Label semantics** (P1 item 6), `crf_ls_m0`, `crf_ls_m100`,
   `crf_ls_m200`.
   - The emission score of tags B-X and I-X is the dot product of the
     token's encoder state with a learned linear map (one for B, one for I)
     of the description embedding of X; O has its own learned vector. The
     CRF and all settings are unchanged.
   - Descriptions: for fine-grained labels, the class name split into words
     and its MMO `skos:definition`; for coarse-grained types, which have no
     definitions, the names of the MMO classes annotated with that type in
     the fine-grained files.
   - Each description is embedded once before training, as the mean
     last-layer state of the pre-trained encoder over its tokens; the linear
     maps are trained.
   - 3 seeds per split and per paper; decision on the held-out papers.
3. **Self-training** (P1 item 7), `crf_st_m0`, `crf_st_m100`, `crf_st_m200`.
   - Teacher: the CRF head trained on the run's training data.
   - It tags a fixed random sample (seed 0) of 2,000 sentences from the
     in-domain corpus of variant 5. MaterioMiner text is never used.
   - Silver data: the tagged sentences that contain at least one entity and
     in which every word's tag has a CRF marginal probability of at least
     0.9.
   - Student: same initialisation and seed, trained on gold + silver with
     the silver loss weighted 0.5; its epoch is chosen on the gold
     validation set.
   - Leaving one paper out only (3 seeds × 4 papers × 2 granularities), to
     bound CPU time; decision on the held-out papers.
4. **Multi-task again** (P1 item 2): `--crf --multi-task` with seed offsets
   100 and 200 (`crf_mt_m100`, `crf_mt_m200`), added to the week-2 runs
   (`crf_mt`, offset 0); compared with the CRF reference.
5. **In-domain pre-training** (P1 item 5), `crf_dapt2_m0`, `crf_dapt2_m100`,
   `crf_dapt2_m200`.
   - Corpus: open-access full texts from Europe PMC whose title mentions
     fatigue and whose title or abstract names a metallic material; the four
     MaterioMiner papers are excluded by DOI. The query and filters are
     fixed in code before any download, and the corpus report records the
     article IDs and the corpus SHA-256.
   - Sentences are deduplicated, and every sentence sharing a word 8-gram
     with MaterioMiner text is dropped, as in week 2. At most 40,000
     sentences are kept (a fixed random sample, seed 0).
   - MLM settings as in week 2 (`dapt_mlm.py`), then the CRF head from the
     new checkpoint, 3 seeds per split and paper; compared with the CRF
     reference.

Execution: multi-task (4) and the SciBERT and PubMedBERT runs (1) start
first, since they need no new code; label semantics (2), the corpus and
pre-training (5), and self-training (3) follow as their code is ready. Each
item is reported when its runs finish. The whole plan takes about 75 CPU
hours, about 2 days on two job slots.

### Week 4 protocol (fixed before any run)

Week 4 runs items 1, 2, 3 and 5 of P1 "Next, after week 3". The week-3
design and rule stand:
- 3 model seeds per split: K, K + 100 and K + 200 on random split K; 0, 100
  and 200 on each held-out paper. A model scores the mean of its 3 runs on a
  split; an ensemble scores once.
- A change improves on its comparison, for a granularity, if the 95%
  Student-t CI of the mean paired micro-F1 difference lies above 0 over the
  protocol named for it: the 5 random splits or the 4 held-out papers. The
  other protocol, strict IOB2 and macro F1 are reported alongside.
- Every run uses `--crf`. New options only add code; the code paths of
  weeks 1–3 stay unchanged.

**Votes over the CPU runs (item 1, no training).**
- `crf_ls_ens3`: the entity-level majority vote (2 of 3) of `crf_ls_m0`,
  `crf_ls_m100` and `crf_ls_m200`.
- `crf_st_ens3`: the same over `crf_st_m0`, `crf_st_m100` and
  `crf_st_m200`, on the held-out papers only, like their runs.
- Each vote is compared with `crf_ens3`, the reported system. The decision
  is on the 4 held-out papers, the target of both methods; the
  label-semantics vote on the random splits is reported.

**GPU runs (items 1, 2, 3 and 5).** Everything else trains on Kaggle:
- `NvidiaTeslaT4` sessions (2 T4 GPUs): one run per GPU, two at a time.
  `src/graph/kaggle_jobs.py` uploads the code and the corpus, starts the
  jobs and downloads the results.
- One frozen commit and one Kaggle image for all runs. `transformers` and
  the MatSciBERT checkpoint are pinned to those of the CPU runs.
- Deterministic CUDA: PyTorch deterministic algorithms, deterministic
  cuDNN and a fixed cuBLAS workspace. The first job trains one CRF run
  twice; GPU runs count as repeatable only if both give identical scores.
  Every GPU run records its environment (GPU, CUDA, torch, transformers).
- GPU runs are never compared with CPU runs. The hardware changes a run
  with the same seeds, as the thread count did in week 2. Every GPU
  comparison is between GPU runs, and the GPU results go to `results/gpu/`
  under the CPU file names.

GPU families, each with 3 seeds on the 5 random splits and the 4 held-out
papers, with `--save-predictions`:
1. The CRF reference `crf_m0`, `crf_m100`, `crf_m200` (both
   granularities) and its vote `crf_ens3`.
2. Multi-task: `crf_mt`, `crf_mt_m100`, `crf_mt_m200` (one run gives both
   granularities) and its vote `crf_mt_ens3` (item 1).
3. Label semantics: `crf_ls_m0`, `crf_ls_m100`, `crf_ls_m200` (both
   granularities).
4. Label semantics on the multi-task model (item 2): `crf_mt_ls_m0`,
   `crf_mt_ls_m100`, `crf_mt_ls_m200`. Each head of the multi-task model is
   the week-3 label-semantic head over its own granularity's descriptions,
   embedded by the pre-trained encoder before fine-tuning. Everything else
   is multi-task training.
5. Self-training at scale (item 3): `crf_st3_m0`, `crf_st3_m100`,
   `crf_st3_m200`. One run per split and granularity:
   - it trains the split's 3 CRF models as teachers, whose test scores must
     equal those of the GPU `crf_m*` runs;
   - each teacher tags all 40,000 sentences of the week-3 fatigue corpus
     (SHA-256 `7eaf1e79…`; no MaterioMiner text);
   - silver data: the sentences with an entity on which the 3 teachers give
     every word the same tag, each with a CRF marginal of at least 0.9 under
     every teacher;
   - 3 students, one per teacher seed, trained as in week 3: the teacher's
     initialisation and seed, gold + silver with the silver NLL weighted
     0.5, the epoch chosen on the gold validation set.

   Held-out papers first; the random splits if the quota allows.

Decisions, GPU runs only:
- item 1: `crf_mt_ens3` vs `crf_ens3` over the 5 random splits, where
  multi-task helped;
- item 2: `crf_mt_ls` vs `crf` over the 5 random splits and, separately,
  over the 4 held-out papers, since each parent helped on one of them;
- item 3: `crf_st3` vs `crf` over the 4 held-out papers.

Reported alongside:
- `crf_mt` and `crf_ls` vs `crf`: do the week-3 gains repeat on other
  hardware?
- `crf_mt_ls` vs `crf_mt` (random splits) and vs `crf_ls` (held-out
  papers);
- `crf_st3` vs `crf_ens3`;
- the GPU `crf` runs next to the CPU ones, descriptive only.

**Reported system.** For each granularity, `crf_ens3` stays unless a vote
passes:
- in distribution, only `crf_mt_ens3` can replace it, and it would be
  reported with its GPU scores;
- on unseen papers, `crf_ls_ens3` or `crf_st_ens3` can; if both pass, the
  one with the larger mean difference.

**Amendment, before any GPU run (the owner's request).** All remaining
week-4 work runs on the GPU; nothing trains on the CPU.
- The GPU families add week-3 self-training: `crf_st_m0`, `crf_st_m100`,
  `crf_st_m200` on the held-out papers, as on the CPU.
- The GPU votes `crf_ls_ens3` (both protocols) and `crf_st_ens3` (held-out
  papers) are compared with the GPU `crf_ens3`, repeating the CPU votes.
  The CPU decisions stand, since they were taken before this amendment; the
  GPU repeats are reported next to them.
- The new options (label semantics on the multi-task model, self-training
  with 3 teachers) get their smoke tests on the GPU, before their runs.
  Only the unit tests, with tiny models, run on the CPU.

Execution, in this order: the CPU votes; the GPU smoke job; the GPU
reference and multi-task (item 1); label semantics and the combination
(item 2); self-training, the week-3 design and then at scale (item 3). Each
item is reported when its runs finish.

Scheduling note, before the runs: Kaggle took 15–50 minutes to start each
smoke session, so items 1 and 2 share one GPU session (each GPU runs its
item-1 jobs first), and item 3 runs in a second session at the same time,
with the same pinned image and commit. Nothing else changes; the order
above is the order in which the items are reported. The smoke jobs found
two packages missing on Kaggle (seqeval, rdflib); both are now installed
there, pinned to the CPU versions. A CRF run should take 1–2 minutes on
a T4 (13–27 on 2 CPU threads), so the plan needs about 17 GPU-hours, about
half of Kaggle's weekly quota of 30.

Outcome: every planned run finished, the random splits of item 3 included,
in 7.5 GPU-hours (smoke jobs included).

### Week 5 protocol (fixed before any run)

Week 5 runs item 1 of P1 "Next, after week 4" on the GPU: 10 model seeds per
split, so that gains of about 1 point can be settled. It also prepares
items 2–7 and a paper without training (below). The NER results are the
closest to a paper, so this week replaces the original week-5 plan
(class-level taxonomy, relation pilot), which moves to week 7.

**Design.**
- 10 model seeds per split: K + 100j for j = 0, …, 9 on random split K, and
  100j on each held-out paper. Seed tags are m0 to m900; the multi-task run
  of offset 0 keeps its name, `crf_mt`.
- Families, each with 10 runs per split, at both granularities, on the 5
  random splits and the 4 held-out papers:
  1. the CRF reference `crf_m*`;
  2. multi-task `crf_mt*`;
  3. label semantics `crf_ls_m*`;
  4. self-training with 3 teachers `crf_st3_m*`. The teachers and the
     silver data stay those of week 4: the CRF runs of seed offsets 0, 100
     and 200, the whole corpus, tags that all teachers give alike with every
     marginal at least 0.9. Each of the 10 seeds trains a student on that
     silver data, as the week-4 students did (new option `--students 10`).
     The teachers are retrained in every run, and their scores must equal
     those of the GPU `crf_m*` runs.
- The week-4 GPU runs (offsets 0, 100 and 200) are kept; week 5 adds
  offsets 300 to 900. Every run is a GPU run in the week-4 environment
  (Kaggle T4, the pinned image, `transformers` and the MatSciBERT snapshot),
  from one new frozen commit, with `--crf` and `--save-predictions`. New
  options only add code. The first job checks that the new commit repeats a
  week-4 run exactly.
- Votes: the entity-level majority vote of 3, as before, over each of 3
  disjoint seed triples: m0–m200, m300–m500 and m600–m800
  (`<family>_ens3`, `<family>_ens3_m300` and `<family>_ens3_m600`). m900 is
  in no vote.
- Scores: on a split, a family scores the mean of its 10 runs, and a vote
  the mean of its 3 triples.

**Rule**, as in weeks 3 and 4: a change improves on its comparison, for a
granularity, if the 95% Student-t CI of the mean paired micro-F1 difference
lies above 0 over the protocol named for it (the 5 random splits or the 4
held-out papers). The other protocol, strict IOB2, macro and weighted F1 are
reported alongside. GPU runs are compared only with GPU runs.

**Decisions.**
1. The three claims so far, each re-tested on the protocol of its earlier
   decision, at both granularities:
   - multi-task vs `crf` over the 5 random splits (week 3: +0.75
     coarse-grained on the CPU; the GPU repeat gave −0.08);
   - label semantics vs `crf` over the 4 held-out papers (week 3: +1.17
     coarse-grained on the CPU; the GPU repeat gave +0.07);
   - self-training with 3 teachers vs `crf` over the 4 held-out papers
     (week 4: +1.07 fine-grained).

   A claim stands only if its 10-seed comparison passes. The 3-seed results
   stay in the record.
2. The reported system, for each protocol and granularity: the vote of the
   CRF head, `crf_ens3`, unless `crf_mt_ens3`, `crf_ls_ens3` or
   `crf_st3_ens3` improves on it; if several do, the one with the largest
   mean difference. This replaces the week-4 slots, including the CPU
   decision for `crf_ls_ens3`. With 12 comparisons at the 95% level, about
   0.3 false passes are expected if no vote helps; the write-up states the
   number of comparisons.

**Reported alongside (no decision):**
- each family vs `crf` on the other protocol, and `crf_ens3` vs `crf`;
- the reported system against the published numbers (69.92 and 72.32)
  under micro, macro and weighted F1, as in week 3;
- exploratory, for the paper:
  - the spread of each family over its 10 seeds, per split;
  - for each family against `crf`, the share of the 120 seed triples (the
    same triple on both sides) whose 3-seed comparison passes the rule, and
    the share whose difference has the opposite sign to the 10-seed one:
    how far a 3-seed design can mislead;
  - per coarse-grained type, the F1 and support of the reported system: the
    types that keep macro F1 below 72.32 (item 5).

**Order and budget.**
- Session b, existing options only: the check that a week-4 run repeats,
  then `crf`, `crf_ls` and `crf_mt` at offsets 300 to 900 (315 runs; a
  multi-task run gives both granularities), about 4 hours on its 2 GPUs.
- Session a: first a smoke session (a tiny `--teachers 3 --students 4` run:
  32 training sentences, 2 epochs, 200 corpus sentences). Then the
  3-teacher runs, one lane per granularity: per split the teachers again,
  the tagging and 7 new students, about 7 hours.
- In all about 12 of the 22.5 GPU-hours left in Kaggle's weekly quota (until
  17 October).

**Prepared without training** (run in week 6, under a protocol fixed before
its runs; the plan is to screen each new method with 3 seeds and confirm it
with 10):
- item 2, multi-task self-training: multi-task teachers tag the corpus once
  for both granularities, multi-task students train on it, the silver
  filter is chosen on the validation sets, and the students' vote is
  compared with `crf_ens3`;
- item 3, a larger encoder (DeBERTa-v3-large): its tokenizer and its memory
  on a T4;
- item 4, document context: the neighbouring sentences in the encoder
  input, only the middle one tagged (FLERT);
- item 5, rare coarse-grained types, from the per-type analysis above;
- item 6, the split and averaging of the paper: an email to the
  MaterioMiner authors, which the owner sends;
- item 7, a second materials NER dataset: availability, licence and format
  of the candidates, then a loader;
- the paper: an outline in `docs/PAPER.md`, with each claim and the result
  that supports it.

**Amendment, before any run of it: the published split.** The MaterioMiner
release (Durmaz et al., 2024; Fordatis, release v1.0.1, CC BY 4.0) ships the
train, validation and test sentences of its experiments
(`examples/data/*_grained_training`: 323, 57 and 96 sentences, a different
partition per granularity). Each of them is one of the 476 corpus sentences,
and the release code scores with seqeval's defaults: micro F1 in the default
mode, our `f1`. Week 5 therefore also runs the published split, so that the
comparison with 69.92 and 72.32 is like for like:
- the release's partition and tags, copied to
  `data/raw/dataset/published_split/` with their SHA-256. The tags differ
  from the corpus files in 7 tokens of 6 fine-grained sentences, where the
  release starts an entity with B-;
- plain MatSciBERT (the paper's model) and the CRF head, 10 model seeds each
  (0, 100, …, 900), at both granularities (new option `--published-split`);
- reported, no decision:
  - the mean over the 10 seeds, with its 95% CI, against 69.92 and 72.32,
    under micro, macro and weighted F1 and strict IOB2. On one split, this
    CI measures the seed variance only;
  - the CRF head minus plain MatSciBERT, paired by seed;
  - the CRF votes over the 3 seed triples.

These runs use a later frozen commit than the other week-5 runs; their
session again first repeats a week-4 run exactly. About 1 GPU-hour.

### Week 6 protocol (fixed before any run)

Week 6 runs items 1 and 2 of P1 "Next, after week 5", towards a methods
paper: three methods for unseen papers on MaterioMiner, and the second
dataset, SOFC-Exp. Every run is a GPU run in the week-5 environment (Kaggle
T4, the pinned image, `transformers` and the MatSciBERT snapshot,
deterministic CUDA) with `--crf` and `--save-predictions`, from one new
frozen commit, the one that adds this protocol. That commit loads every
model in float32 (week 5), which leaves the MatSciBERT runs unchanged; each
session first repeats a week-4 run to show it.

**Methods.** Each changes one thing against the CRF head; every other
setting is that of the CRF runs (learning rate 5e-5, batch 16, 20 epochs,
the best epoch on validation F1).
1. DeBERTa-v3-large (`--model microsoft/deberta-v3-large`, snapshot
   64a8c8e, with `sentencepiece` 0.2.2 for its tokenizer): a general-domain
   encoder of 435M parameters, against MatSciBERT's 110M. Variant
   `crf_deberta_m<N>`.
2. Document context (`--context 64`): up to 64 subwords of the neighbouring
   sentences of the same paper and part on each side, only the sentence
   itself labelled (FLERT). Variant `crf_ctx_m<N>`.
3. Multi-task self-training (`--multi-task --self-train <fatigue corpus>
   --teachers 3 --self-train-sample 0`). The teachers are 3 multi-task
   models with the seeds of `crf_mt`, `crf_mt_m100` and `crf_mt_m200`,
   retrained in every run; their scores must equal those stored runs. They
   tag the whole corpus once for both granularities. Per granularity, the
   silver filter is chosen on the validation sentences from 5 candidates
   (3 of 3 teachers agree with every marginal at least 0.9, 0.8 or 0.7; 2 of
   3 at 0.9 or 0.8): the one with the most sentences among those right at
   least as often as the strictest. Each student is a multi-task model
   trained on gold plus silver sentences, silver weight 0.5. Variant
   `crf_mt_st3_m<N>`; one run gives both granularities.

**Stage 1, the screen.** Seeds 0, 100 and 200 on the 4 held-out papers, at
both granularities: each method against the CRF head (`crf_m0` to
`crf_m200`, GPU), 3-seed means. A method goes to stage 2 if its mean
micro-F1 difference is above 0 at either granularity. The screen decides
nothing. Week 5 showed that 3 seeds miss most real gains of 1–2 points, but
rarely give a gain the wrong sign (0–8% of the triples for the gains that
passed or came close), so this rule lets almost every real gain through.
A method that the screen stops is reported with its 3 seeds.

**Stage 2, the decisions.** Seed offsets 300 to 900 are added, giving 10
seeds per paper; votes are taken over the 3 disjoint seed triples, as in
week 5. For each method in stage 2:
1. the method against the CRF head on the 4 held-out papers, per
   granularity (10-seed means);
2. the method's vote against the strongest system on unseen papers:
   fine-grained the multi-task vote `crf_mt_ens3` (52.47), coarse-grained
   `crf_ens3` (61.10).

With 3 methods, 2 granularities and 2 decisions there are 12 comparisons:
about 0.3 false passes are expected if no method helps, and the write-up
states the number. A method claim for the paper needs decision 2 and its
gain to repeat on SOFC-Exp.

**SOFC-Exp** (`--granularity sofc` on its official split: 26 training, 8
development and 11 test articles; model seeds 0 to 900):
- S1: the CRF head against plain MatSciBERT, paired by seed: the week-5
  finding on a second dataset. Reported alongside: the published
  MatSciBERT-CRF scores (micro 84.77, macro 82.39; Gupta et al., 2022),
  strict IOB2, and the CRF votes over the 3 seed triples.
- S2: DeBERTa-v3-large against the CRF head on the same split, paired by
  seed, whether or not it reaches stage 2. Document context gets S2 only if
  it reaches stage 2: SOFC-Exp annotates only some sentences of each
  article, so its context must first take the unannotated sentences from
  the article texts. Multi-task self-training does not apply (one
  granularity, and the corpus is about fatigue).
- On one split, the CIs are over the 10 seeds only, as on the published
  MaterioMiner split.

**Rule**, as before: a change improves on its comparison if the 95%
Student-t CI of the mean paired micro-F1 difference lies above 0, over the 4
held-out papers or, on SOFC-Exp, over the 10 seeds.

**Reported alongside (no decision):** strict IOB2, macro and weighted F1;
multi-task self-training against multi-task (`crf_mt`); each screen; each
method's first vote (seeds 0–200) in the screen; the per-type F1 of a method
that passes.

**Order and budget** (Kaggle's weekly quota: 11.8 hours left until 17
October, then 30 hours):
- Session a (`w6a`): the repeat run, then the DeBERTa and document-context
  screens, balanced over its 2 GPUs (12 jobs of 4 runs, about 2 hours).
- Session b (`w6b`): the repeat run and the multi-task self-training screen,
  one job per paper in one lane, since they extend the same files (about 3
  hours); in the other lane SOFC-Exp, plain MatSciBERT and the CRF head (20
  runs, about 1.5 hours).
- Then stage 2 (`w6c:<methods>`) for the methods that pass the screen, and
  S2 (`w6c:sofc-deberta`), as the quota allows: about 4 GPU-hours for
  DeBERTa, 2.5 for document context and 4.5 for multi-task self-training on
  the papers, and about 3.5 for DeBERTa on SOFC-Exp.

**Amendment, after the screen and before any run of it: document context
on SOFC-Exp.** The screen sends document context to stage 2 (fine-grained
+2.66 against the CRF head), so it gets S2. SOFC-Exp annotates 876 of the
9,466 sentences of its 45 articles, so the context comes from the article
texts, as fixed above:
- `data/raw/dataset/sofc_exp/context.json` (`sofc_exp.py`): every sentence
  of each article, cut from the release's token files, and the positions of
  its annotated sentences, whose words must equal those of the corpus;
- `--context 64` takes an article's context from its whole text when all its
  annotated sentences are in the part (train, validation or test), which
  holds for every article of the official split; otherwise, as on
  MaterioMiner, from the annotated sentences of the same part.

S2 for document context: `--context 64` against the CRF head of session b
on the official split, model seeds 0 to 900, paired by seed, under the rule
below. The code for the article texts came after the protocol, so these
runs (`w6c:sofc-ctx`) use a later frozen commit, the one that adds this
amendment. That commit changes nothing else that a run executes (it also
adds mention replacement, an option for week 7, off unless asked for). Its
session first repeats the week-4 run, then session b's SOFC-Exp CRF run of
seed 0 (`smoket`); both must be identical. Every other week-6 run keeps the
frozen week-6 commit. About 1.5 GPU-hours.

**Later (not week 6):** the random splits for the methods that pass,
combinations of the methods that pass (week 7), and the rare coarse-grained
types.

### Week 7 protocol, part 1: document-level label consistency (fixed before any run)

Week 7 starts with a method for unseen papers that needs no training: label
consistency within a document, applied to the saved test predictions of the
10-seed GPU runs (`ner_consistency.py`). The rule was fixed in the commit
that adds the tool, before it was applied to any model's predictions:
- Within a document (a MaterioMiner paper, a SOFC-Exp article; only the
  run's test sentences of it), each predicted entity gives a surface form,
  its exact words.
- An occurrence of a form is a place where its words appear. It is tagged
  if a predicted entity has exactly that span, untagged if each of its words
  is predicted O, and otherwise left alone.
- A form with more tagged than untagged occurrences, and one most frequent
  type among its tagged ones, gives that type to each untagged occurrence.
- Longer forms go first, and an occurrence becomes an entity only while all
  its words are still O, so no prediction is removed or changed.

Variant `<input>_dc`, e.g. `crf_dc_m300` from `crf_m300`.

**Why it may help.** On MaterioMiner, 60% of the gold mentions have a
surface form that occurs more than once in their paper, 83% (fine) and 88%
(coarse) of those with a single type, and 3% of the occurrences of an
entity's words are untagged. A model that tags a new paper's term in some
sentences and misses it in others can then be corrected from its own
majority ("one sense per discourse", Gale et al., 1992; non-local
consistency in NER, Krishnan and Manning, 2006). **Its cost** with perfect
predictions: applied to the gold tags of the 4 papers, the rule adds 40
fine-grained and 17 coarse-grained entities that the annotators left
untagged, i.e. 1.52 and 0.36 points of F1; on SOFC-Exp's official test set
it adds 30 (1.51 points), since SOFC-Exp leaves 14% of these occurrences
untagged.

**Decisions** (the decision rule as before: the 95% CI of the mean paired
micro-F1 difference above 0, over the 4 held-out papers):
1. The rule on the CRF head: `crf_dc_m<N>` against `crf_m<N>`, N = 0 to
   900, 10-seed means per paper, per granularity.
2. The rule on the strongest system on unseen papers: fine-grained the
   multi-task vote (`crf_mt_ens3_dc` against `crf_mt_ens3`), coarse-grained
   the seed vote (`crf_ens3_dc` against `crf_ens3`), each over its 3 seed
   triples. The votes are made again with `ner_ensemble.py
   --save-predictions` (scores unchanged), then the rule is applied.

**SOFC-Exp:** the rule on session b's CRF head on the official split,
paired by seed (10 seeds). A claim needs decision 2 and SOFC-Exp, as in
week 6. That makes 5 comparisons: about 0.13 false passes are expected if
the rule does nothing.

**Reported alongside (no decision):**
- the rule on the other granularity's vote (fine-grained `crf_ens3`,
  coarse-grained `crf_mt_ens3`);
- the rule on the random splits (the CRF head, 10 seeds);
- the number of entities added, and precision and recall;
- `--retype` (`_dcr`), which also gives the tagged occurrences of a
  propagated form its type: on the gold tags it retypes 13 fine-grained and
  27 coarse-grained entities.

Once week 6's stage 2 is in, the rule on document context (`crf_ctx_dc`
against `crf_ctx`, and on its votes) is also reported, without a decision;
combining the two is for part 2.

**Budget:** no GPU. Scoring takes seconds per file.

Part 2 of week 7 (combinations of the methods that pass, mention
replacement) is fixed after the week-6 decisions, before its runs.

**Outcome of part 1: no claim; the rule hurts on SOFC-Exp.** On SOFC-Exp's
official split (session b's CRF head, 10 seeds, paired by seed) the rule
adds 325 entities over the 10 runs and scores −0.42 [−0.61, −0.22]; with
retyping, −0.38 [−0.60, −0.15]. Both CIs lie below 0. On MaterioMiner it
passed only the coarse-grained CRF head (+0.12 [+0.01, +0.24]), not its
votes. This matches the rule's cost on gold tags: SOFC-Exp's annotators
leave 14% of the occurrences of a tagged term untagged, so filling them in
counts as error. On document context (reported) it adds +0.10 [−0.67,
+0.88] fine-grained and +0.06 [−0.09, +0.21] coarse-grained.

**Amendment, after MaterioMiner's results and before SOFC-Exp's.** On the
held-out papers the rule passes one of its 4 decisions (the CRF head,
coarse-grained, +0.12 [+0.01, +0.24]); its votes do not pass, so there is
no claim. Retyping, reported only, does better coarse-grained: +0.64
[+0.29, +1.00] on the CRF head and +0.83 [+0.32, +1.34] on the seed vote.
This fits where the CRF head errs on unseen papers (exploratory, its 10
seeds): coarse-grained, 16% of the gold entities get the right span but the
wrong type and 6% are missed; fine-grained, 36% are missed. A post-hoc
result decides nothing, so retyping is tested where it has not been seen:
on SOFC-Exp, `crf_dcr` against the CRF head (paired by seed, 10 seeds) is
a decision, alongside `crf_dc`. That makes 6 comparisons in part 1, about
0.15 false passes if neither rule does anything. The MaterioMiner results
of retyping stay reported, as fixed above.

**Prepared for part 2 (tools and tests only; not a protocol yet): recall
calibration.** On the held-out papers the CRF head predicts too few
entities fine-grained: precision 52.7 and recall 42.8 (10 seeds, the 4
papers; recall below precision on 3 of them), against 77.2 and 77.9 on the
random splits. Coarse-grained the two are equal (58.4 and 58.8). For a
calibrated model the F1-optimal threshold is half the attainable F1 (Lipton
et al., 2014). Here that is about 0.25–0.30, while a Viterbi decode tags an
entity only when it is the most probable reading. So a bias D added to the
emission of every tag but O before decoding may raise F1 on unseen papers at
both granularities, more where F1 is lower. The tools:
- `ner_baseline.py --entity-bias D ...` decodes the test and validation
  sentences again for each D. A run's own scores stay those of D = 0. Per D
  it stores the entity counts of each test paper and of the validation set,
  and with `--save-predictions` the test predictions.
- `--leave-two-papers-out` trains on two papers and tests on the other two,
  for each of the 6 pairs.
- `ner_bias.py --choose inner` gives the held-out paper P the D with the
  highest mean F1 on the inner folds that hold out P and another paper Q,
  scored on Q: models trained on two of P's three training papers. P's own
  labels are never read.
- `ner_bias.py --choose val` takes each run's best D on its validation set
  (SOFC-Exp's validation articles are unseen articles; MaterioMiner's
  validation sentences come from the training papers).

Prior work, found after the protocol below was fixed: Minkov et al. (2006)
already traded precision for recall in a CRF extractor with a single weight
on the O label. What this protocol adds is the choice of that weight for
unseen documents on inner folds, and its test under the 10-seed design.

The output (`crf_eb_m<N>`) pairs with the CRF head by seed. Its D = 0 scores
must equal the committed `crf_m<N>` runs, which checks determinism. Rough
cost: 36 inner runs on two papers (3 seeds, 6 pairs, 2 granularities) and
the CRF head again with the grid (80 runs), about 5 GPU-hours; a 3-seed
screen first takes about 2.5. Whether it enters part 2, and its decisions,
are fixed with part 2's protocol after the week-6 decisions.

### Week 7 protocol, part 2a: recall calibration (fixed before any run)

Part 2a tests the recall calibration prepared above on the CRF head. It does
not depend on week 6, so it is fixed now. Part 2b (combinations of the
methods that pass week 6) is fixed after the week-6 decisions.

**Runs.** One Kaggle session (`w7a`) from the commit that adds this
protocol; it first repeats the week-4 run (`smokeu`). Every run decodes the
bias grid D = −1, −0.5, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5 and 3, and 0:
- the inner folds: `--leave-two-papers-out --crf --entity-bias <grid>`,
  model seeds 0, 100 and 200 (`crf_bias_m0/m100/m200`), both
  granularities; 36 runs on two papers each, entity counts only;
- the CRF head on the held-out papers again, with its predictions:
  `--leave-one-paper-out --crf --entity-bias <grid> --save-predictions`,
  model seeds 0 to 900 (`crf_bias_m<N>`), both granularities; 80 runs;
- SOFC-Exp's official split: the CRF head with the grid and its
  predictions, model seeds 0 to 900 (`crf_bias`); 10 runs.

The D = 0 scores and predictions of every run on the held-out papers and on
SOFC-Exp must equal those of the committed `crf_m<N>` runs and of session
b's `crf` runs. A run that differs is not used, and the difference is
reported.

**The choice of D** (`ner_bias.py`, committed before any run):
- MaterioMiner: `--choose inner`. Per held-out paper and granularity, D has
  the highest mean F1 on the inner folds that hold out that paper and
  another paper Q, scored on Q, over the 3 seeds. The held-out paper's
  labels are never read. The result is `crf_eb_m<N>`.
- SOFC-Exp: `--choose val`, each run's D from its validation articles,
  which are unseen articles. The result is `crf_ebv`.

**Decisions** (the rule as before: the 95% CI of the mean paired micro-F1
difference above 0):
1. `crf_eb_m<N>` against `crf_m<N>`, 10-seed means per held-out paper, per
   granularity.
2. Its votes (`crf_eb_ens3` over the 3 seed triples, `ner_ensemble.py`)
   against the strongest system on unseen papers of week 5: `crf_mt_ens3`
   fine-grained and `crf_ens3` coarse-grained. If a week-6 vote passes its
   decision 2, the comparison with that vote is reported.
3. SOFC-Exp: `crf_ebv` against `crf`, paired by seed.

A claim needs decision 2 and SOFC-Exp, as in week 6. That makes 5
comparisons; about 0.13 false passes are expected if the bias does nothing.

**Reported alongside (no decision):**
- `--choose val` on MaterioMiner (`crf_ebv_m<N>` against `crf_m<N>`). Its
  validation sentences come from the training papers, so this shows whether
  in-distribution validation finds the bias.
- The D chosen per paper, the inner curves, and the held-out papers' own
  curves (F1 per D, never used to choose).
- Precision, recall and the number of predicted entities; strict, macro
  and weighted F1.

**Expected before the runs.** If the model's probabilities are roughly
calibrated, the chosen D is positive and larger fine-grained (F1 about 47
on unseen papers) than coarse-grained (about 59). The inner models train on
two papers, not three, so their F1 is lower and D may come out somewhat
larger than is best for the models trained on three.

**Budget:** about 1.5 hours of one session's two GPUs (week 5's CRF jobs
took about 1.2 minutes per run).

**Outcome of part 2a: no claim; none of the 5 decisions passes.** The
session (`w7a`, 96 minutes) ran all 116 runs. Its repeat of the week-4 run
is identical, and all 90 runs on the held-out papers and SOFC-Exp equal
their committed CRF runs at D = 0: scores, predictions and training
history.
- Decision 1, the calibrated CRF head (`crf_eb`) against the CRF head,
  10-seed means on the 4 held-out papers: fine-grained −0.59 [−1.75,
  +0.57], coarse-grained +0.17 [−0.14, +0.48].
- Decision 2, its vote against the strongest system on unseen papers:
  fine-grained −3.53 [−8.10, +1.05] against the multi-task vote,
  coarse-grained −0.09 [−0.75, +0.58] against the seed vote.
- SOFC-Exp, D chosen on the validation articles (`crf_ebv`) against the
  CRF head, paired by seed: 83.55 against 83.75, −0.21 [−0.58, +0.17].
- Reported: D chosen on MaterioMiner's validation sentences, +0.32 [−0.83,
  +1.46] fine-grained and +0.09 [−0.24, +0.43] coarse-grained.

The inner folds chose D = +1.25, +1.25, −1 and +0.25 fine-grained (papers
s11340, ncomms7164, s41467, met10111458) and 0, −1, −1, −1 coarse-grained.
The expectation before the runs (D positive, larger fine-grained) held for
3 of 4 papers fine-grained, and the gain it predicted did not follow. Each
paper's own curve (mean test F1 per D over 10 seeds, reported and never
used to choose) is flat. Even the best D per paper in hindsight, which no
rule could pick, would gain +0.56 fine-grained on average and +0.18
coarse-grained. Its sign differs by paper (−1, −0.5, +0.25 and +1
fine-grained), and the inner folds got the sign right on 1 of the 4
papers.

*Why (exploratory, the bias grid's entity counts pooled over the 4 papers
and 10 seeds):* the CRF head does not score its missed entities just below
the threshold.
- **Fine-grained.** At D = 0 it finds 5,386 of 13,280 gold entities
  (recall 40.6, precision 53.7).
  - Raising D to +0.25 adds 358 predicted entities, and the number of
    correct ones rises by 78 (22%). The break-even rate for F1 is F1/2 =
    23%.
  - D = +3 raises the odds of a one-word entity by a factor of e³ ≈ 20. It
    adds 4,044 predicted entities and recovers 570 of the 7,894 missed ones
    (7%), with 6.1 wrong entities per right one; precision falls to 42.3
    and recall rises to 44.8.
- **Coarse-grained.** The head already predicts about as many entities as
  there are (21,963 for 21,940). D = +3 recovers 74 of its 9,348 misses
  (0.8%). Its errors are wrong types and boundaries (C10), which one bias
  on every entity tag cannot fix.
- **SOFC-Exp.** Over the 10 runs the head predicts 10,161 entities for 9,780
  gold ones. D = +3 adds 1,212 predictions and 4 correct entities. The test
  curve peaks at D = −0.5 (+0.19, reported), while the validation articles
  chose D > 0 in 7 of 10 runs.

So the recall gap on unseen papers is not a decision-threshold problem.
The entities a bias of +0.25 adds are right at or below the rate F1 needs
to break even: 22% against 23% fine-grained, 19% against 29%
coarse-grained, 21% against 42% on SOFC-Exp. The predictions above the
threshold are right 53.7%, 57.3% and 82.2% of the time. Most missed
entities stay out of reach at any bias on the grid. The premise of the
expectation, roughly calibrated probabilities, under which the entities
just below the threshold would be right more often than the break-even
rate, does not hold on unseen papers. A fix has to change what the model scores, not how its
scores are decoded. This fits C7: the change that helps fine-grained on
unseen papers (the multi-task vote) mostly buys precision, not recall.

**A candidate for part 2b, from the gold annotations only (no protocol
yet): coarse-to-fine completion.** On the held-out papers the CRF head
misses 36% of the fine-grained entities but only 6% of the coarse-grained
ones (exploratory, above). The two levels nest: 1,262 of the 1,265
fine-grained entities are coarse-grained entities with the same span, and
each fine-grained type has a single coarse-grained parent. The reverse holds
only for some coarse-grained types, so a fold would have to learn it from
its training papers:
- VAL, STD and MAC spans are fine-grained entities of a single type (Value,
  StateDescriptor, MaterialClass) in every paper (97–100%), and MAT spans
  mostly are (Alloy, 50–100% by paper).
- EQU, CHP, MFP, DAM and LOC spans are fine-grained O in some papers and a
  fine-grained type in others, as their subclasses fall in or outside the
  top-27 fine-grained classes.

A rule could add a fine-grained entity where the coarse-grained prediction
has an entity of a type that maps to one fine-grained type in the training
papers (say 90% of at least 10 spans), and the fine-grained prediction has
only O. It would take the saved predictions of both granularities (two CRF
heads of the same seed, or the two heads of a multi-task run), so it needs
no GPU. SOFC-Exp has one level of types, so a gain could not be repeated
there; under the week-6 rule for a claim, this would be a result on
MaterioMiner only.

### Week 7 protocol, part 2c: coarse-to-fine completion (fixed before any run)

Part 2c tests the candidate above, which the gold annotations motivate:
fine-grained recall on unseen papers is the bottleneck, the coarse-grained
level misses far fewer entities, and the two levels nest. It needs no
training: a rule on the saved test predictions of both granularities
(`ner_coarse_to_fine.py`, committed with this protocol, before it is
applied to any predictions). Held-out papers only (leave-one-paper-out).

**The rule.** For the fold that holds out paper P, a mapping is taken from
the gold annotations of the other three papers (all their sentences, both
granularities):
- a coarse-grained type C maps to the fine-grained type F if C has at least
  10 spans there and at least 90% of them are fine-grained entities of type
  F with the same span.

In each test sentence of P, a coarse-grained predicted entity of a mapped
type whose words are all predicted O at the fine-grained level becomes a
fine-grained entity of the mapped type. No fine-grained prediction is
removed or changed. The fold's mapping is stored with its output.

**Inputs and names** (`_cf` before the seed tag): the fine-grained and
coarse-grained predictions of the same seed and kind of system:
- the CRF head, `crf_m<N>` of both granularities → `crf_cf_m<N>`;
- the multi-task model's two heads, `crf_mt` (and `_m<N>`) → `crf_mt_cf`;
- the multi-task vote of both granularities, `crf_mt_ens3` (and `_m300`,
  `_m600`), made again with `ner_ensemble.py --save-predictions` (its
  scores must equal the stored votes) → `crf_mt_ens3_cf`.

**Decisions** (the rule as before, over the 4 held-out papers,
fine-grained; the rule leaves the coarse level as it is):
1. `crf_cf_m<N>` against `crf_m<N>`, 10-seed means per paper.
2. `crf_mt_ens3_cf` against `crf_mt_ens3`, the strongest fine-grained system
   on unseen papers after week 5, over its 3 seed triples.

**Reported (no decision):** `crf_mt_cf` against `crf_mt`; the mapping of
each fold; the number of entities added; precision and recall.

**Scope.** SOFC-Exp has one level of types, so a gain cannot be repeated
there. Under week 6's rule for a claim, a pass would be a result on
MaterioMiner only, about a type hierarchy. That makes 2 comparisons:
about 0.05 false passes are expected if the rule does nothing.

**Expected before applying it** (from the gold annotations only): the
mapped types are VAL → Value, STD → StateDescriptor and MAC →
MaterialClass in every fold, and MAT → Alloy where the training papers
reach 90%.

*Clarification, made when the tool first met the data and before any
predictions were used:* 15 sentences are tokenised differently in the
fine-grained and coarse-grained files (as the multi-task runs already
handle). They cannot be aligned word by word, so they count for neither
the mapping nor the completion, and their fine-grained predictions stay as
they are. Each run stores how many test sentences this concerns
(`n_unaligned`).

**Budget:** no GPU; seconds per file.

**Outcome of part 2c: no significant change, no claim.** The folds mapped
VAL, STD and MAC everywhere, MAT where the training papers reached 90%, and
ETP when the paper holding all of it was a training paper. Fine-grained, on
the 4 held-out papers:
- on the CRF head, +0.34 [−0.91, +1.59] (decision 1);
- on the multi-task vote, +0.12 [−1.76, +2.01] (decision 2);
- on the multi-task model, +0.07 [−1.40, +1.54] (reported).

The rule adds 539 entities over the CRF head's 40 runs (about 13 per run),
and 39% of them are right; on the multi-task model 457, 35% right. That is
above the break-even precision of about F1/2, so they help a little, but
too few to show over 4 papers. The fine-grained misses are mostly not of
the types the coarse level can vouch for.

*Correction, 16 October, after the outcome:* the scope notes above (and
the week-6 protocol's exclusion of multi-task self-training from SOFC-Exp)
say that SOFC-Exp has one level of types. It has two: besides its 4 entity
types, every entity token can carry a slot type, a finer level that our
conversion had left out. Part 2c's result stands as a MaterioMiner result;
the second level is described below.

### Prepared for part 2d: SOFC-Exp's second level (slot types)

SOFC-Exp's annotation files give each token an entity tag and a slot tag
(`annotations/entity_types_and_slots`). The slot level, read from the
corpus (no training):
- 4,263 slot mentions of 19 types on the same 876 sentences; every slot
  token lies inside an entity, and every slot tag continues validly (IOB2).
- Each slot type falls within one entity type, mention by mention: the
  materials (anode, cathode, electrolyte, interlayer, support material,
  fuel) are MATERIAL entities;
  the values (working temperature, power density, resistance, thickness,
  current density, conductivity, open circuit voltage, time of operation,
  voltage, degradation rate) are VALUE entities; `device` refines DEVICE;
  `experiment_evoking_word` (1,090) is the EXPERIMENT entity under another
  name.
- SOFC-Exp's own evaluation (the authors' `evaluate_cross_validation.py`)
  scores 17 of them: all but `experiment_evoking_word` and
  `interconnect_material` (one mention), which its code maps to O. The
  17 hold 3,172 mentions: 2,066 / 404 / 702 in the train, dev and test
  articles.
- Published reference: MatSciBERT's paper reports "SOFC-Slot" with the
  same official test set, MatSciBERT-CRF micro F1 72.85 ± 1.11 and macro
  F1 65.35 ± 2.73 (3 seeds × 5 cross-validation splits; Supplementary
  Table 3 and Table 1), trained with the experiment-evoking word as an
  extra type.

What this changes: SOFC-Exp has the structure that multi-task training
uses (two nested levels on the same sentences, like MaterioMiner's fine
and coarse levels), so the one training change that holds on MaterioMiner
(C7) can be repeated on a second dataset, and so can multi-task
self-training if it passes its 10-seed decisions (it would also need an
in-domain corpus for SOFC-Exp).

Prepared (tests only, no training): `sofc_exp.py` writes the slot tags to
`data/raw/dataset/sofc_exp_slot/` (the entity files unchanged, byte for
byte); `ner_baseline.py --granularity sofc_slot` reads them with the 17
evaluated types (the other two become O), on the same official split;
`--granularity sofc_both --published-split --crf --multi-task` trains one
encoder with a slot head and an entity head, as `--multi-task` does with
MaterioMiner's levels, and writes `ner_results_sofc_slot_*` and
`ner_results_sofc_*` files. On a tiny random model (CPU), the code before
and after this change writes byte-identical results files, runtimes aside,
for every MaterioMiner option it touches:
- the CRF head;
- multi-task training, also with a tag and seed offset, label semantics or
  document context;
- multi-task self-training.

### Week 7 protocol, part 2d: multi-task training on SOFC-Exp's two levels (fixed before any run)

Multi-task training is the one training change that holds on MaterioMiner
(C7): fine-grained +1.83 [+0.32, +3.34] over the CRF head on the random
splits, and its vote fine-grained +3.65 [+0.49, +6.82] over the seed vote on
unseen papers. A methods claim needs a gain on a second dataset. Part 2d
repeats the comparison on SOFC-Exp, whose slot level (above) stands to its
entity types as MaterioMiner's fine level to its coarse one.

**Runs** (batch `w7b`, one Kaggle session with two lanes, from the commit
of this protocol; GPU, deterministic, 10 model seeds 0 to 900 on the
official split):
- first, three repeats, each of which must equal its stored run, or the
  batch is not used: the CRF run of smoke4 (`smokev`), session b's SOFC-Exp
  CRF run of seed 0 (`smokew`: the entity-level reference below comes from
  that session) and the MaterioMiner multi-task run of random split 0
  (`smokex`: this commit changes how the multi-task code finds its data);
- plain MatSciBERT and the CRF head on the slot level (`sofc_slot`,
  variants `baseline` and `crf`);
- multi-task training with a slot head and an entity head (`--granularity
  sofc_both --crf --multi-task`, variant `crf_mt`, a file per level).

The entity-level reference is the CRF head of week 6, S1 (session b, 10
seeds).

**Decisions** (week-6 rule: the 95% Student-t CI of the mean paired
micro-F1 difference above 0; the pairs are the 10 model seeds, 9 degrees of
freedom):
1. slot level: `crf_mt` against `crf`;
2. entity level: `crf_mt` against `crf`.

If multi-task training does nothing, about 0.05 false passes are expected.

**Reported (no decision):**
- the CRF head against plain MatSciBERT on the slot level, paired by seed
  (S1 on the second level);
- both against the published SOFC-Slot numbers. MatSciBERT-CRF scores 72.85
  micro and 65.35 macro, MatSciBERT-Linear 71.30 and 63.82. Their setup
  differs: 3 seeds × 5 cross-validation splits, and the experiment-evoking
  word trained as an 18th type;
- the votes over the 3 seed triples: `crf_mt` votes against the CRF votes,
  per level, paired by triple (2 degrees of freedom);
- strict IOB2, macro and weighted F1, and per-type F1 on the slot level.

**What a pass means.** Decision 1 repeats C7's fine-grained gain on a
second dataset's finer level, which a methods paper needs. Decision 2 would
go beyond MaterioMiner, where the coarse-grained gain did not hold.

**Not tested here.** Multi-task self-training on SOFC-Exp needs its
10-seed decisions on MaterioMiner first (week 6, stage 2) and an in-domain
SOFC corpus. Recall calibration of the multi-task model needs part 2a's
outcome first.

*Notes for an SOFC corpus (exploratory, 16 October; nothing downloaded
beyond a search):*
- Europe PMC has 377 open-access full texts with "solid oxide" or SOFC in
  the title, 29 of them SOFC-Exp's own articles (which a corpus would
  leave out, with any sentence sharing an 8-gram with SOFC-Exp).
- SOFC-Exp was tokenised with Stanford CoreNLP. `corpus_words`, made for
  MaterioMiner, cuts 4,526 of SOFC-Exp's 9,466 article sentences (48%) the
  same way. The differences are dashes, which CoreNLP keeps as they are
  (–, −); abbreviations it keeps with their period (Fig., al., e.g.);
  hyphenated materials it splits (Ni-YSZ); and charges and oxygen
  deficiencies it keeps on the formula (O3−δ, Fe3+). Silver sentences
  would need a splitter closer to CoreNLP's.
- SOFC-Exp annotates only sentences that describe experiments, and its
  slots are roles in them (an anode material of the reported cell). A
  general SOFC sentence names materials in other roles, so slot-level
  silver data would be noisier than MaterioMiner's.

*Correction, 16 October, after these notes:* the tokens are not CoreNLP's,
whatever the release's README says:
- the training and development articles were cut with ChemDataExtractor
  1.3.0, the test articles with Java's `BreakIterator`;
- the SOFC corpus uses ChemDataExtractor's tokenizer;
- only sentences in which the teachers tag an experiment can be silver.

All three are in the week-6 amendment at the end of this section. Recall
calibration of the multi-task model is dropped: part 2a found that a bias
does not recover the missed entities.

**Budget:** about 1.2 hours of one session's two GPUs. Lane 0 runs two
repeats, then plain MatSciBERT and the CRF head on the slots, about 70
minutes; lane 1 runs the multi-task repeat and the multi-task runs, about
60 minutes. It runs after the weekly quota resets, on the kernel that
`w6c:sofc-ctx` leaves free.

**Outcome of part 2d: neither decision passes, no claim.** Session a of 17
October (`w7b`, from the protocol's commit). The three repeats are
identical: `smokev` 77.95, `smokew` 84.98, `smokex` 78.59 fine-grained and
79.95 coarse-grained. On the official split, over 10 model seeds:
- decision 1, slot level: multi-task training 71.11 [69.87, 72.34] against
  the CRF head's 71.88 [71.01, 72.75], −0.78 [−2.39, +0.84] paired by seed;
- decision 2, entity level: 83.60 [83.00, 84.19] against 83.75 [82.85,
  84.65], −0.15 [−1.36, +1.05].

C7's fine-grained gain on MaterioMiner does not repeat on SOFC-Exp's finer
level. Multi-task training stays a MaterioMiner result.

Reported (no decision):
- **The votes** do not differ either: −0.86 [−3.67, +1.95] slot level
  (73.46 against the CRF votes' 74.32) and +0.08 [−2.85, +3.01] entity
  level (85.21 against 85.13), paired by triple.
- **S1 on the slot level.** The CRF head against plain MatSciBERT (70.20
  [68.82, 71.58]): +1.68 [−0.12, +3.49] paired by seed, not significant,
  unlike the entity level's +1.22 [+0.50, +1.93]. Under strict IOB2 it is
  −0.06 [−1.70, +1.58]: plain MatSciBERT scores 71.94 strict, more than
  under the default scoring. Its invalid tags are mostly wrong entities,
  which strict scoring drops (C4). Macro −0.03 [−1.91, +1.85], weighted
  +0.45 [−1.49, +2.38].
- **The published SOFC-Slot numbers** lie above the CRF head's CIs:
  MatSciBERT-CRF 72.85 micro and 65.35 macro, against 71.88 and 61.83
  [60.61, 63.04]. Their setup differs (5 cross-validation splits, the
  experiment word as an 18th type). The CRF votes score 74.32 [73.07,
  75.57].
- **Seen and unseen names** (`ner_mentions.py`). 59% of the slot level's
  test entities are unseen names, against 49% of the entity level's. The
  CRF head's gain over plain MatSciBERT is on them again: +2.83 [+0.56,
  +5.10], seen −0.23 [−1.69, +1.24]. Multi-task training gives −1.52
  [−3.65, +0.62] on the unseen names of the slot level and −0.45 [−2.27,
  +1.37] on those of the entity level.
- **Per type** (slot level, mean F1 over the 10 seeds; CRF head, then
  multi-task; test entities in brackets): the measured slots score high,
  e.g. working temperature 94.2 and 94.3 (138) and power density 91.7 and
  90.4 (70). The material roles score low: electrolyte 47.9 and 46.6
  (120), cathode 27.9 and 22.6 (37), anode 32.4 and 20.3 (26). A material
  slot names a material's role in the reported cell, which its words
  alone do not show. Multi-task training loses most on the anode (−12.1)
  and on open-circuit voltage (77.9 to 71.9, 25 entities). Current density
  is 95.2 with plain MatSciBERT and 72.6 with the CRF head (17).
- **Right span, wrong slot** (`ner_matching.py`). 11.7% of the CRF head's
  slot-level predictions have a gold entity's span and another slot type,
  against 0.3% at the entity level. Most are roles taken for one another:
  an electrolyte material tagged as a cathode material 27 times per seed
  (of 120), as an anode material 7 times, an anode as a cathode 7 times;
  voltage as open-circuit voltage 7 times. Scored on the span alone (exact
  matching), the CRF head's slot-level F1 is 83.44 instead of 71.88. On
  the slot level the model finds the entities; it is their role that it
  gets wrong.

### Week 7 protocol, part 2b: mention replacement (fixed before any run)

The further method for unseen papers that the plan names ("Next, after
week 5", item 3), prepared in week 6 without training. Mention replacement
(Dai and Adel, 2020; `--augment-mentions 3`) adds 3 copies of each
training sentence with an entity. In each copy, each mention is replaced
with probability 0.5 by a mention of the same type drawn from the training
sentences, frequent ones more often. Copies that change nothing are
dropped, and the validation and test sentences are unchanged. The model
sees each type's contexts with other names, which change from paper to
paper. Variant `crf_mr_m<N>`; the copies are drawn with the run's model
seed.

**Settings.** 3 copies and p = 0.5 are fixed here, not tuned. Dai and Adel
tune both on development data (1, 3, 6 or 10 copies; p from 0.1 to 0.7),
but the validation sentences of the training papers say little about
unseen papers (part 2a).

**Prior.** Dai and Adel's Table 3 reports a transformer on the materials
dataset MaSciP: +2.4 F1 with 50 training sentences, +0.6 with 150, −0.6
with 500 and +0.2 on the full set. The training papers of a fold hold
234–370 training sentences, so a large gain is unlikely. The screen costs
about one GPU-hour, so it is run anyway.

**Stage 1, the screen** (batch `w7c`, from the commit of this protocol).
Seeds 0, 100 and 200 on the 4 held-out papers, at both granularities,
against the CRF head (`crf_m0` to `crf_m200`), 3-seed means. Mention
replacement goes to stage 2 if its mean micro-F1 difference is above 0 at
either granularity, the week-6 screen rule. The screen decides nothing; its
differences and the first vote (`crf_mr_ens3` against `crf_ens3`) are
reported (`ner_seedmeans.py --week 7 --gpu --screen`).

**Stage 2** (batch `w7d`, only if the screen passes it). Seed offsets 300 to
900 are added on the held-out papers (10 seeds), with votes over the 3
disjoint seed triples. The decisions are those of the week-6 methods:
1. `crf_mr` against `crf` on the 4 held-out papers, per granularity;
2. its vote against the strongest system on unseen papers: `crf_mt_ens3`
   fine-grained, `crf_ens3` coarse-grained.

S2 on SOFC-Exp is also a decision, since a methods claim needs it:
`crf_mr` against the CRF head on the official split, 10 model seeds, paired
by seed. That makes 5 decisions, so about 0.13 false passes are expected if
mention replacement does nothing.

**Reported:** strict IOB2, macro and weighted F1; the number of copies per
run (`n_augmented`); per-type F1 if it passes.

**Budget:**
- the screen: 24 runs of about 4 times a CRF run, about 1 hour on one
  session's two GPUs;
- stage 2: 56 runs on the held-out papers and 10 on SOFC-Exp (about 12
  minutes each), about 3.3 hours.

It runs on kernel a after `w7b`.

**Outcome of the screen: mention replacement goes to stage 2.** Session a
of 17 October (`w7c`, from the protocol's commit 494cac1; the repeat of
smoke4's CRF run, `smokey`, is identical at 77.95; 34 minutes). Against
the CRF head, 3-seed means over the 4 held-out papers:
- fine-grained +3.25 [−2.33, +8.82] (per paper +2.85, +7.95, +2.72,
  −0.53); coarse-grained +0.25 [−2.21, +2.72]. Above 0 at one
  granularity, so stage 2 runs (`w7d`, pushed to kernel a at 04:17);
- the first vote against the seed vote (reported): fine-grained +5.00
  [−0.50, +10.51] (per paper +5.04, +9.12, +5.20, +0.65), coarse-grained
  +1.16 [−2.11, +4.43];
- on unseen names (`ner_mentions.py`, reported): fine-grained +3.48
  [−0.18, +7.15], seen names +1.30 [−10.44, +13.05]; coarse-grained +0.80
  and −1.45;
- copies per run: 414–731 fine-grained and 588–963 coarse-grained, on
  234–370 training sentences.

A 3-seed screen is optimistic (C8): document context passed with +2.66 and
gained +1.14 with 10 seeds. Stage 2 adds the seeds 300 to 900 and SOFC-Exp's
S2, and decides.

**Outcome of stage 2: decision 1 passes at both granularities; decision 2
and S2 do not.** Session a of 17 October (`w7d`, from the protocol's
commit 494cac1; the repeat `smokez` is identical at 77.95; 2.5 hours).
10 seeds on the 4 held-out papers:
- decision 1, against the CRF head: **fine-grained +2.77 [+1.51, +4.03]**
  (every paper gains: +3.48, +3.24, +1.69, +2.69) and **coarse-grained
  +1.28 [+0.06, +2.51]** (+1.61, +2.10, +0.29, +1.14). Both pass;
- decision 2, its vote against the strongest system on unseen papers:
  fine-grained −0.04 [−2.09, +2.01] against the multi-task vote,
  coarse-grained +2.26 [−0.12, +4.64] against the seed vote. Neither
  passes;
- S2 on SOFC-Exp's official split: 83.46 [83.08, 83.85] against the CRF
  head's 83.75, −0.29 [−1.04, +0.47] paired by seed. It does not pass.

2 of the 5 decisions pass, where about 0.13 would pass if mention
replacement did nothing. It is the first method to improve on the CRF
head on unseen papers at both granularities (the consistency rule passed
coarse-grained only, +0.12). It does not improve on the strongest system,
and it does not repeat on SOFC-Exp, so it is not the methods claim on its
own.

Reported:
- its vote against the seed vote: fine-grained +3.61 [+1.71, +5.52], about
  as much as the multi-task vote (+3.65); against the multi-task vote,
  coarse-grained +2.57 [−0.19, +5.34];
- strict IOB2 equals the default scoring (the CRF head's tags are valid);
  macro F1 +2.34 [−0.99, +5.66] fine-grained and +2.79 [+0.04, +5.55]
  coarse-grained, weighted +3.62 [+1.63, +5.62] and +1.86 [+0.17, +3.55];
- on unseen names (`ner_mentions.py`): fine-grained +3.19 [+1.66, +4.72],
  seen names +1.49 [−1.31, +4.30]; coarse-grained +1.58 [−0.02, +3.17]
  and +0.12 [−0.83, +1.06]. The gain lies on the names the training
  papers lack, as C17 predicts;
- copies per run: 414–760 fine-grained (mean 580) and 588–979
  coarse-grained (788), on 234–370 training sentences; on SOFC-Exp
  1,551–1,586 on 568.

On SOFC-Exp's 568 longer training sentences it gives nothing, in line with
Dai and Adel, whose gains for transformers shrank as the training set grew.
On MaterioMiner it adds names in new contexts, the multi-task model adds a
second level of types, and each improves a single model on unseen papers
fine-grained. The combination prepared below tests whether the two add up.

*Per type (exploratory, 17 October, before any run of week 9;
`ner_seed_analysis.py`, `week6_types` now also holds `crf_mt` and `crf_mr`;
10 seeds × 4 held-out papers, against the CRF head).* The two methods
improve different types.
- Coarse-grained, their per-type changes run against each other
  (correlation −0.72 over the 26 types; 17 change in opposite
  directions). Mention replacement gains on every one of the rare types
  that pull macro F1 down (week 5): SDE +7.74, MIP +8.13, PRP +11.71, MOD
  +17.88 and CPR +17.57, and on ODE (+7.13) and CGE (+4.85). Multi-task
  training loses on all seven (−2.85 to −10.67). This is where mention
  replacement's macro gain (+2.79) comes from.
- Fine-grained they overlap more (correlation 0.20; 15 of 25 types gain
  with both). Mention replacement gains most on Mechanism (+12.53, a
  frequent descriptive type), MicrostructureEntity (+16.19),
  MechanicalTestDataSet (+18.09) and StrainRate (+8.33); multi-task
  training on StateDescriptor (+11.88), MicrostructureEntity (+12.96),
  FatigueLifetime (+9.19), TestingMachine (+6.75) and Alloy (+5.96), where
  mention replacement loses on StateDescriptor (−3.43) and TestingMachine
  (−7.92).
- Rare types have 2–9 test entities per paper, so each of these numbers is
  noisy; they carry no decision. They agree with the week-9 premise that
  the two change different things.

*Prepared for a combination (17 October, tests only, no runs).* Mention
replacement on the multi-task model (`--multi-task --augment-mentions K`,
variant `crf_mt_mr`, a file per head). A copy replaces a coarse-grained
mention, in both heads' tags at once, by a training mention of the same
coarse-grained and fine-grained type (the type of the fine-grained entity
with its span, or none). The levels nest, so this keeps both tag sequences
valid. Examples labelled for one head only (the 15 sentences tokenised
differently) are left alone. Each run stores its number of copies. On a
tiny random model, every existing option writes byte-identical results
files (`runtime_sec` aside), single-task mention replacement included.
Whether it runs is fixed in a protocol after stage 2's decisions.

### Week 6 amendment, before stage 2's results: multi-task self-training on SOFC-Exp (fixed before any run)

The week-6 protocol left multi-task self-training out of SOFC-Exp for two
reasons: one level of types, and a corpus about fatigue. Neither holds any
more. SOFC-Exp has a second level (part 2d), and an SOFC corpus is built
here. This amendment is fixed on 16 October, before stage 2 of multi-task
self-training on MaterioMiner (`w6c:mtst`, after midnight UTC) gives any
result. A methods claim needs a method's gain to repeat on SOFC-Exp, so
without this the method could not make one.

**SOFC-Exp's tokens (found while building the corpus; exploratory).** The
release's README says Stanford CoreNLP cut the texts into tokens; the paper
names ChemDataExtractor. On the release's texts and token offsets:
- ChemDataExtractor 1.3.0's word tokenizer (`ChemWordTokenizer`) gives the
  tokens of every sentence of the 26 training and 8 development articles,
  all 7,630 of them. Stanford CoreNLP (3.9.2, 4.0.0 or 4.5.7, with the
  options tried) gives at most 48.6% of all sentences.
- The 11 test articles were cut otherwise. Java's word `BreakIterator` (US
  locale, presumably INCEpTION's default) gives 1,833 of their 1,836
  sentences; the other 3 differ at a combining dot accent. ChemDataExtractor
  gives 41%.

So the test articles are cut into words differently from the articles a
model trains on: the test articles have `Fig .`, `cm − 2`, `O2 −`,
`high-temperature` and `3%`, where training has `Fig.`, `cm−2`, `O2−`,
`high - temperature` and `3 %`. Every result on SOFC-Exp's official split
shares this, the published ones included, and the test set is used as
released. Sentences were split by Java's sentence `BreakIterator` in all
articles (the README).

**The corpus** (`build_sofc_corpus.py`; `results/sofc_corpus.json`):
- Search: Europe PMC's open-access full texts with "solid oxide", SOFC or
  SOFCs in the title, 377 hits. 29 are SOFC-Exp's own articles and 3 are
  corrections, so 345 are eligible. They are read in a fixed random order
  until the filtered pool reaches 60,000 sentences: 292 articles.
- Text: the fatigue corpus's rules, except that superscripts are written
  inline (`cm−2`, `Fe3+`), as SOFC-Exp's texts write them.
- Filters: 104 duplicates are dropped, and 3,208 sentences that share a
  word 8-gram with any of the 9,466 sentences of SOFC-Exp's 45 articles.
- Kept: 40,000 sentences (a seed-0 sample of the pool, median 26 words),
  cut into words with ChemDataExtractor 1.3.0's tokenizer, the tokenizer of
  the training articles. It is vendored as `src/graph/chem_tokenize.py`
  (MIT licence) and gives the original's spans on all 9,466 sentences.
- SHA-256 `1b6a00d2…`.

**The method** (`--granularity sofc_both --published-split --crf
--multi-task --self-train <SOFC corpus> --teachers 3 --students 10
--self-train-sample 0`) is that of MaterioMiner (week 6, method 3), with
the slot level as the fine head and the entity level as the coarse one:
- Teachers: 3 multi-task models, model seeds 0, 100 and 200 on the official
  split (568 training and 135 validation sentences). They must equal part
  2d's `crf_mt` runs of those seeds.
- Silver filter: each head's, chosen on the validation articles from the
  same 5 candidates.
- One change, fixed now: SOFC-Exp annotates only sentences that describe
  experiments (875 of its 876 annotated sentences have an EXPERIMENT
  entity). So a corpus sentence can be silver, for either head, only if
  every teacher's entity head tags an EXPERIMENT entity in it.
- Students: 10, model seeds 0 to 900 (`crf_mt_st3_m<N>`), each trained on
  the gold training sentences plus the silver ones at weight 0.5, with a
  slot-level and an entity-level file. The corpus is read as space-separated
  words.

**Decisions** (S2 for multi-task self-training; paired by seed over the 10
seeds; the rule as before):
1. Slot level: its slot head against the slot-level CRF head (`crf` on
   `sofc_slot`, part 2d).
2. Entity level: its entity head against the CRF head (`crf` on `sofc`,
   session b).

A methods claim for multi-task self-training needs its decision 2 on
MaterioMiner at one granularity and S2 at the matching level: fine-grained
with the slot level, coarse-grained with the entity level. 2 decisions:
about 0.05 false passes.

**Reported (no decision):**
- against multi-task training on the same levels (part 2d's `crf_mt`,
  paired by seed);
- the votes over the 3 seed triples against the CRF votes;
- strict IOB2, macro and weighted F1;
- the silver data: sentences in scope, silver per head, the chosen
  filters;
- the teachers against part 2d's `crf_mt` runs.

**When it runs:** only if multi-task self-training passes at least one of
its 4 decisions on MaterioMiner. Without one, no claim could follow, and it
is reported as not run. *Outcome, 17 October: not run.* Multi-task
self-training passes none of its 4 decisions (P1, "Week 6"). It also needs part 2d's results (`w7b`) for
decision 1 and for the check of the teachers.
- Batch: `w6c:sofc-mtst`, from the commit of this amendment.
- Repeats first: the CRF run of smoke4 (`smokeaa`) and session b's SOFC-Exp
  CRF run of seed 0 (`smokeab`).
- Lanes: each GPU trains the 3 teachers and 5 of the 10 students
  (`--student-offsets`), so the two lanes' teachers and silver data must be
  identical.

**Budget:** about 2 hours of one session. MaterioMiner's multi-task
self-training took 51–91 minutes per paper for 3 teachers and 3 students;
here each GPU trains 3 teachers and 5 students.

**Prepared (tests only, no training):**
- `chem_tokenize.py` and `build_sofc_corpus.py`;
- `ner_baseline.py` reads an SOFC corpus as spaced words, applies the scope
  rule with `sofc_both`, and has `--student-offsets` (not stored in the
  config);
- `kaggle_jobs.py` puts the SOFC corpus in the bundle (SHA-256 checked) and
  has the batch, and the kernel installs the corpus only for a batch that
  uses it.

Two checks that nothing else changed:
- On a tiny random model, the MaterioMiner options write byte-identical
  results files, runtimes aside, before and after: the CRF head,
  multi-task training, label semantics, document context, self-training
  with one or two teachers and extra students, and multi-task
  self-training.
- The fatigue corpus rebuilds with the same SHA-256.

### Week 7 protocol, part 2e: SOFC-Exp's test tokens (fixed before any run)

**The question.** How much does the shift between SOFC-Exp's training and
test tokens cost on its official split? The test articles are cut into
words by Java's BreakIterator, the training articles by ChemDataExtractor
(C16; the amendment above).

**The data.** `sofc_exp.py --retokenized-test` writes
`data/raw/dataset/sofc_exp_test_cde/`:
- the 173 annotated sentences of the 11 test articles, the same text cut
  into words by ChemDataExtractor's tokenizer; 140 of them are cut
  differently from the release;
- the entity tags carried over by character offsets: 978 entities;
- 14 of them inexact: a boundary falls inside a word, so the entity takes
  the whole words;
- 2 of them lost: a word holds two entities, as `(PBMCo)-LSGM` does, and
  only the first keeps it. The retokenized test has 976 gold entities.

**Runs** (batch `w7e`, from the commit of this protocol): the CRF head
(`--crf --retokenized-test --tag cdetest`) and plain MatSciBERT
(`--retokenized-test --tag cdetest`), model seeds 0 to 900.
- Training and validation are those of session b's runs, so the models are
  session b's. Every run's training (best epoch and validation F1 per epoch)
  must equal its session-b run: then only the test words differ.
- The session first repeats the CRF run of smoke4 (`smokeac`) and session
  b's SOFC-Exp CRF run of seed 0 (`smokead`).

**Decisions** (the rule as before): per model, micro F1 on the retokenized
test articles minus micro F1 on the released ones, paired by seed over the
10 seeds:
1. the CRF head: `crf_cdetest` against `crf`;
2. plain MatSciBERT: `cdetest` against `baseline`.

A CI above 0 means the shift costs that model F1 on the official split; a
CI below 0, that the released words suit it better. 2 comparisons.

**Reported:** strict, macro and weighted F1; per-type F1. The 2 lost
entities: dropping 2 of 978 gold entities changes F1 by about 0.2 points
at most.

**Expected before the runs:** the words that differ are mostly inside
MATERIAL and VALUE entities: formulas with charges, units and hyphenated
names. A model trained on ChemDataExtractor's words should find their
boundaries at least as well when the test articles are cut the same way,
so the difference should be at least 0.

**What a result changes.** Every SOFC-Exp comparison in this project
pairs systems on the same test words, so no decision changes either way.
A cost would mean that results on the official split, the published ones
included, understate what the models do on consistently tokenized text.

**Budget:** 20 runs of about 3.3 minutes, 10 per GPU: about 40 minutes of
one session. It runs on kernel a after part 2b's batches.

**Outcome of part 2e: both decisions pass. The shift costs both models F1,
and on test articles cut as the training articles are, the CRF head
reproduces the published MatSciBERT-CRF score.** Session `w7e` on
`ner-gpu-jobs-d` (version 3, bundle 4f5dbda; 37.6 minutes from 20:10 on 17
October). The repeats are identical (`smokeac` 77.95, `smokead` 84.98).
All 20 runs trained exactly as their session-b runs (seed, sizes, best
epoch, validation F1 and its history), so only the test words differ.
Paired by seed over the 10 seeds:
- decision 1, the CRF head: 84.61 [83.89, 85.34] on the retokenized test
  articles against 83.75 on the released ones, **+0.86 [+0.39, +1.33]**.
  It passes;
- decision 2, plain MatSciBERT: 84.45 [83.95, 84.94] against 82.53,
  **+1.91 [+1.33, +2.49]**. It passes.

As expected, the shift costs both models F1 on the official split, and
plain MatSciBERT twice as much as the CRF head. The published
MatSciBERT-CRF score (84.77; Gupta et al., 2022) lies above the CRF head's
CI on the released words, but inside it on the retokenized ones. The gains
fall mostly on unseen names (48.7% of the 976 gold entities). Plain
MatSciBERT gains +3.50 [+2.33, +4.68] F1 on them and +0.32 [+0.21, +0.42]
on seen ones. The CRF head gains +1.20 [+0.30, +2.10] and +0.47 [+0.22,
+0.71] (`ner_mentions.py`).

Reported, no decision: on the retokenized test articles the CRF head's gain
over plain MatSciBERT, paired by seed, is +0.17 [−0.37, +0.70] (strict
−0.13 [−0.71, +0.44]), against +1.22 [+0.50, +1.93] on the released words.
On SOFC-Exp the CRF head's gain (C13) was mostly a repair of the shift:
given the training articles' words, plain MatSciBERT finds the same
entities. No other decision changes, since every comparison on SOFC-Exp
pairs systems on the same test words. The retokenization's own loss of 2
of 978 entities moves F1 by about 0.2 points at most.

### Week 8 protocol, part 1: SOFC-Exp beyond its official split (fixed before any run)

**Why.** Claim C2 (random splits hide a large gap) rests on MaterioMiner's
4 papers, and every SOFC-Exp result on one split, its official split with 11
test articles. SOFC-Exp has 45 articles: enough to repeat the comparison of
random sentence splits with held-out documents on a second corpus. C17 and
C18 suggest the gap may be smaller there: half of SOFC-Exp's test names are
unseen, and the CRF head finds 82% of them.

**Design.** SOFC-Exp's entity level (4 types, `--granularity sofc`), all 876
annotated sentences with the corpus's tags, the CRF head (`--crf`), model
seed offsets 0 to 900 (`crf_m<N>`), predictions saved:
- **random splits:** 5 random 65/15/20 sentence splits, split seeds 0 to 4
  (`--vary-split`), as on MaterioMiner;
- **article folds:** `--document-folds 5`. The 45 articles, sorted and
  shuffled with split seed 0, are dealt into 5 groups of 9. Fold i tests on
  the sentences of group i; the other 36 articles' sentences give the
  validation set (15%, shuffled with the split seed) and the training set,
  as `--leave-one-paper-out` does.

That is 100 runs. The option `--document-folds` is committed with this
protocol; every existing option writes byte-identical results files with
it (the tiny-model check), and every earlier Kaggle batch is unchanged.

**Reported (descriptive, no decision):**
1. the CRF head's micro F1 per protocol: the mean over the 5 splits or
   folds of its 10-seed means, with a 95% CI over them, next to the
   official split's 83.75; and random splits minus article folds. The
   protocols test different sentences, so this is not a paired comparison,
   as C2 is not on MaterioMiner;
2. the same for the CRF head's votes over the seed triples;
3. macro, weighted and strict F1 (`ner_seed_analysis.py`, `sofc_splits`);
4. on both protocols, seen and unseen names (`ner_mentions.py`) and the
   looser matching criteria (`ner_matching.py`), as for C17 and C18.

**Expected before the runs:** a gap of a few points, far below
MaterioMiner's 28–31. SOFC-Exp's articles share one topic and one
annotation scheme, and the CRF head's score on unseen articles (83.75) is
already close to what in-distribution scores usually are.

**What a result changes.** A small gap would make C2 a finding about
corpora whose papers differ in topic and annotation conventions, as
MaterioMiner's do (C18), rather than about random splits alone. A large
one would make it hold on two corpora.

**Checks.** The session first repeats the CRF run of smoke4 (`smokeae`)
and session b's SOFC-Exp CRF run of seed 0 (`smokeaf`); both must equal
their stored runs.

**Budget:** 100 runs of about 3.3 minutes, 10 jobs of 5 runs per GPU:
about 2.8 hours of one session (`w8a`). It runs after the queued batches of
weeks 6 and 7, on the next free kernel.

**Outcome of part 1 (descriptive): random splits score 3.5 points above
held-out articles on SOFC-Exp, against 28–31 on MaterioMiner.** Session a of
17 October (`w8a`, bundle 04352cd, 06:55 to 10:05); both repeats are
identical (`smokeae` 77.95, `smokeaf` 84.98).
1. **The CRF head** (micro F1, the mean over 5 splits or folds of their
   10-seed means, 95% CI over them): 86.27 [84.82, 87.71] on random
   sentence splits (84.81 to 87.76 per split) and 82.79 [78.33, 87.24] on
   article folds (79.77 to 88.76 per fold), against 83.75 on the official
   split. Random splits minus article folds: +3.48 micro, +3.04 macro,
   +3.50 weighted; strict IOB2 equals the default scoring.
2. **Its votes** over the seed triples: 87.18 against 83.62 (+3.56).
3. **Seen and unseen names** (`ner_mentions.py`): 29.7% of the test names
   are unseen on random splits, 46.4% on article folds (49.3% on the
   official split). Recall is 91.3 on seen and 82.0 on unseen names on
   random splits, and 89.3 and 79.7 on article folds. Of the 3.6-point
   recall gap (88.4 against 84.8), the mix of names accounts for about 1.5
   points in either order of the decomposition (86.9 and 86.3), and the
   lower recall within each kind for the rest.
4. **Looser matching** (`ner_matching.py`): the 3.48-point strict gap is
   3.46 when only the span must match, 1.41 when an overlap of the same
   type counts, and 1.35 under any overlap (90.16 against 88.81). About
   two of the three and a half points are boundaries: 5.2% of the
   predictions on article folds lie inside a longer gold entity, 2.7% on
   random splits.
5. **Which articles are held out** matters about as much as the protocol:
   the article folds range over 9 points (79.77 to 88.76), the random
   splits over 3. *Exploratory (per fold, pooled over the 10 seeds):* the
   share of unseen names does not explain it. The best fold has 47.8%
   unseen names and the worst 45.4%; they differ in recall within each
   kind (seen 92.8 against 86.6, unseen 89.0 against 78.3; over the folds
   86.6–92.8 and 76.4–89.0). Some articles are harder for the model on
   every kind of name.

As expected before the runs, the gap is a few points, far below
MaterioMiner's. What the protocol said this changes: C2 is a finding about
corpora whose papers differ in topic and annotation conventions, as
MaterioMiner's do (C18), not about random splits alone. Random splits still
overstate SOFC-Exp's score on new articles, by 3.5 points; there a new
article brings fewer new names (46% against 81% fine-grained on
MaterioMiner), and the CRF head finds 80% of them (35% on MaterioMiner).
Files: `gpu/ner_results_sofc_random_split_vary_crf_m<N>.json` and
`gpu/ner_results_sofc_document_folds_crf_m<N>.json` (N = 0 to 900, with
predictions); `sofc_splits` in `gpu/ner_week5_seed_analysis.json`, the
SOFC-Exp settings of `gpu/ner_mentions.json` and `gpu/ner_matching.json`,
Table 4 and Figure 1's third panel.

### Week 8 protocol, part 2: more papers or more sentences? (fixed before any run)

**Why.** A third training paper adds +7.36 F1 fine-grained and +8.90
coarse-grained on an unseen paper (C19, exploratory), but a third paper also
brings more sentences. Whether the gain comes from the papers or from the
sentences decides what a materials-NER corpus should collect next: more
papers, or more sentences of the same papers.

**Design.** `--two-paper-size` (committed with this protocol): for each
held-out paper P and each pair {Q, R} of the other three papers, P's
leave-one-paper-out fold trained on all three other papers, its training
sentences cut to as many as the pair alone gives: the training sentences of
the `--leave-two-papers-out` fold that holds out P and the third paper
(fold `P|Q+R`, variant `crf_p3n2`). The cut keeps the first of the fold's
shuffled training sentences, a random sample of all three papers; the
validation and test sentences are P's fold's. 12 folds per granularity, the
CRF head, seed offsets 0, 100 and 200: 72 runs, with predictions. The
two-paper side is week 7's inner folds (part 2a, the same seeds, counts at
D = 0), which train the same CRF head on the pair.

**Decision** (per granularity, the rule as before): diversity, F1 on P with
three papers at the size of two minus F1 with the two papers alone, each
side the mean over P's 3 pairs and the 3 seeds, paired over the 4 held-out
papers. A CI above 0 means a third paper helps at the same number of
training sentences. 2 decisions; about 0.05 false passes each if it does
not.

**Reported:** size, F1 with all three papers (the `crf` runs, seeds 0 to
200) minus F1 with three papers at the size of two, over the 4 papers; and
per pair. Diversity and size add up to C19's gain. The two sides of the
decision validate on different sentences (15% of two papers or of three);
validation only picks the epoch.

**Expected before the runs:** diversity above 0 on both levels. A new paper
brings new names and contexts (C17), which more sentences of the same
papers do not.

**Checks.** The session first repeats the CRF run of smoke4 (`smokeag`).
Every existing option writes byte-identical results files with the new
option (the tiny-model check), and every earlier Kaggle batch is unchanged.

**Budget:** 72 runs on at most 301 training sentences, about 40 minutes of
one session (`w8b`), one lane per granularity. It runs on the next free
kernel after the queued batches.

*Clarification, made before any run of week 8:* both week-8 batches run from
the commit that adds part 2 (`04352cd`), one bundle for both. Part 1's runs
execute the same code there as at its own commit (`c06fc5e`):
`--two-paper-size` is off unless given, and every other option writes
byte-identical results files (the tiny-model check).

*Session note (17 October, 09:35; operations, not a deviation from the
design).* Part 2's session (`w8b`, kernel b, pushed 07:24, expected to take
about 40 minutes) still reported "running" after two hours. Its live log
(`kaggle kernels logs --follow`) held nothing, not even the model download
that every session prints during setup, and later answered with a server
error. Part 1's session on kernel a, from the same bundle, logged its setup
and 17 of its 22 jobs. The GPU quota rose by 2 hours per hour, so the stuck
session was being charged. It stopped before the first output line, in
`nvidia-smi`, `pip install` or the unpacking, none of which had a time
limit. The Kaggle API can cancel a session only by an ID that none of its
endpoints returns, so stopping it takes the Kaggle website or deleting the
kernel. The runner now limits `nvidia-smi` (2 minutes), `pip` (30 minutes)
and the environment's commands (5 minutes), stops a job still running 1.25
hours after the deadline, and prints a line after each setup step. A stuck
session therefore ends within minutes, and its live log shows where it
stopped. The runner is written at push time and does not touch the
training code, so every bundle's results are unchanged.
- *Outcome.* The session was cancelled on the Kaggle website at about 09:40.
  Its output was empty (a log of `[]`, no manifest), so its script never
  printed a line. Kaggle did not charge it: the used quota fell from 17.15
  to 15.39 hours after the cancel.
- *A likely cause.* The next bundle (536cbe7, for week 9) was uploaded at
  07:28, four minutes after this session was pushed. The session already
  reported "running", but may still have been mounting its inputs. From now
  on a new bundle is uploaded only after the running session's live log
  shows its inputs in place (the runner's line "setup: code, corpora and
  ... in place").
- Part 2 runs again from its own bundle (04352cd), on the next free kernel
  after week 9's sessions; nothing in its design changes.
- *Then a stuck queue.* Week 9, part 1, pushed to kernel b at 09:55, and a
  5-minute test batch pushed to the idle kernel a at 10:31 both stayed
  "queued" for an hour; a further push was refused with "Maximum batch GPU
  session count of 2 reached": queued sessions count towards Kaggle's limit
  of 2. Both were cancelled on the website at about 11:45 (neither ran, so
  neither was charged), and week 9, part 1 started at once when pushed
  again (11:49, kernel b, version 10). Kernel b's live log still answers
  with a server error, so its setup could not be read there; part 2's
  bundle (9468ae7) was uploaded after version 10 had run for seven minutes,
  well past its setup, and part 2 went to kernel a at 12:03.

**Outcome of part 2: both decisions pass. A third paper helps at the same
number of training sentences, but most of its gain is its sentences.**
Session `w8b` on `ner-gpu-jobs-c` (version 2, bundle 04352cd; 40.0 minutes
from 20:26 on 17 October). The repeat is identical (`smokeag` 77.95).
Means over the 4 held-out papers, each the mean over its 3 pairs and seeds
0–200:

| | Two papers | Three at the size of two | Three papers |
|---|---|---|---|
| Fine-grained | 39.02 | 40.20 | 46.38 |
| Coarse-grained | 50.37 | 53.96 | 59.27 |

- decision 1, fine-grained diversity: **+1.18 [+0.22, +2.15]** (per paper
  +1.16, +0.57, +0.99, +2.01). It passes;
- decision 2, coarse-grained diversity: **+3.59 [+1.59, +5.58]** (+2.72,
  +2.52, +3.88, +5.24). It passes.

Reported, size (all three papers minus three at the size of two): +6.17
[+1.10, +11.24] fine-grained (+2.09, +5.40, +9.41, +7.81) and +5.31 [+4.13,
+6.50] coarse-grained (+4.76, +5.26, +4.85, +6.38). Diversity and size add
up to C19's +7.36 and +8.90. Diversity accounts for about a sixth of the
third paper's gain fine-grained and two fifths coarse-grained. Every paper
gains from both at both levels. For a materials-NER corpus, a new paper is
worth more than the same number of sentences from the papers at hand, but
most of what a paper brings is its sentences.

### Week 9 protocol: mention replacement on the multi-task model, and on a smaller SOFC-Exp (fixed before any run)

Fixed on 17 October, after the decisions of weeks 6 and 7 and before any
run of week 9. Week 6 left no method with a claim, and in week 7 mention
replacement passed decision 1 at both granularities (C20). Two questions
follow.

**Part 1: does mention replacement add to the multi-task model?**

*Why.* Mention replacement improves on the CRF head on unseen papers at
both granularities: +2.77 [+1.51, +4.03] fine-grained and +1.28 [+0.06,
+2.51] coarse-grained over 10 seeds. Its vote does not beat the strongest
systems, though: fine-grained it ties the multi-task vote (−0.04 [−2.09,
+2.01]). Multi-task training also improves on the CRF head fine-grained on
unseen papers (+3.72 [+0.57, +6.87], reported in week 5), and its vote is
the strongest fine-grained system. Multi-task self-training, which builds
on it, passed no decision (week 6). The two change different things.
Mention replacement shows the model each type's contexts with other names,
and multi-task training adds the second level of types. If their gains add
up, the combination's vote beats the multi-task vote.

*Method* (prepared and tested on 17 October, commits 205cf96 and 1f021a7):
`--multi-task --augment-mentions 3`, variant `crf_mt_mr`. A copy replaces
each coarse-grained mention, with probability 0.5, in both heads' tags at
once, by a training mention of the same coarse- and fine-grained type.
Week 7's settings are kept: 3 copies and p = 0.5.

*Runs* (batch `w9a`, one Kaggle session from this protocol's commit; GPU,
deterministic):
- two repeats first, each of which must equal its stored run or the batch
  is not used: the CRF run of smoke4 (`smokeah`) and the multi-task run of
  random split 0 (`smokeai`), since the multi-task code gained mention
  replacement;
- seed offsets 0 to 900 on the 4 held-out papers (`crf_mt_mr_m<N>`, one
  model per seed for both granularities), and the votes over the 3
  disjoint seed triples;
- SOFC-Exp's official split, model seeds 0 to 900, both levels
  (`--granularity sofc_both`).

*Decisions.* The week-6 rule applies: the 95% Student-t CI of the mean
paired micro-F1 difference must lie above 0. On MaterioMiner the pairs are
the 4 held-out papers (3 degrees of freedom):
1. `crf_mt_mr` against `crf_mt`, per granularity: does mention replacement
   add to the multi-task model?
2. its vote against the strongest system on unseen papers: the multi-task
   vote fine-grained, the seed vote coarse-grained.

S2 on SOFC-Exp, paired by the 10 seeds: `crf_mt_mr` against the CRF head on
each level (slot types: `crf` on `sofc_slot`; entity types: `crf` on
`sofc`).

That is 6 decisions, so about 0.15 false passes are expected if the
combination does nothing. A methods claim needs decision 2 at a
granularity and S2 at the matching level: fine-grained with the slot
level, coarse-grained with the entity level.

*Reported (no decision):*
- against the CRF head and against mention replacement alone, on the
  held-out papers;
- the votes against the seed vote;
- strict IOB2, macro and weighted F1;
- unseen names (`ner_mentions.py`);
- the number of copies per run.

*Expected.* On SOFC-Exp's official split neither part helps alone:
multi-task training scores −0.78 (slot level) and −0.15 (entity level),
mention replacement −0.29. So S2 is unlikely to pass, and a pass on
MaterioMiner alone would be a result for few annotated papers, not a
methods claim. Part 2 asks whether training size explains this.

*Budget:* about 3.5 hours of one session. The 10 held-out jobs take about
17 minutes each, the SOFC-Exp job about 3 hours, balanced over the lanes.

**Part 2: is mention replacement's gain a matter of training size?**

*Why.* On MaterioMiner a held-out paper's model trains on 234–370
sentences with 606–1,078 fine-grained entities. SOFC-Exp's official split
trains on 568 sentences of 26 articles, with 2,962 entities: 40 words and
5.2 entities per sentence, against 25.5 and 2.8. Dai and Adel
found mention replacement's gains for transformers to shrink as training
data grows (+2.4 F1 with 50 sentences of MaSciP, +0.6 with 150, −0.6 with
500). Mention replacement's S2 on SOFC-Exp (−0.29 [−1.04, +0.47]) is
therefore no test of it with a MaterioMiner-sized training set. Part 2
gives SOFC-Exp that size.

*Design.* SOFC-Exp's entity level. From the 26 official training articles,
5 draws of 10 articles each, sorted and shuffled with split seeds 0 to 4:
156–248 training sentences and 820–1,276 entities per draw, as many
entities as on MaterioMiner. The
official 8 development articles validate and the 11 test articles test,
unchanged. On each draw, model seeds 0 to 900:
- the CRF head;
- mention replacement (`--augment-mentions 3`).

*Decision* (the week-6 rule; the pairs are the 5 draws, each the mean of
its 10 seeds, 4 degrees of freedom): mention replacement against the CRF
head. 1 decision.

*Reported:* the CRF head on 10 articles against the CRF head on all 26
(the official split, the same seeds): what 16 more training articles are
worth on SOFC-Exp's test articles.

*Expected.* Mention replacement improves on the CRF head at this size, as
on MaterioMiner. If it does not, its MaterioMiner gain belongs to that
corpus, not to small training sets.

*Code.* A new option draws the training articles. It is prepared and
checked (the tiny-model identity check; every earlier batch unchanged)
before part 2 runs, and nothing in this design changes after part 1's
results. *Budget:* 100 runs on 156–248 training sentences; the mention
replacement runs train on about 4 times as many. About 2 hours of one
session on two lanes.

*Prepared for part 2 (17 October, after the protocol and before any run;
tests only).* `--train-articles K` with `--published-split` on SOFC-Exp's
entity or slot level trains on K of the official training articles,
`--draws` times (default 5, folds `draw<i>`). The articles are sorted and
shuffled with the split seed + i, and the official validation and test
articles are kept (protocol `article_draws`; the model seed comes from
`--seed-offset`). Its 5 draws of 10 articles hold 182, 205, 248, 173 and
156 training sentences. Batch `w9b` repeats the CRF run of smoke4
(`smokeaj`) and session b's SOFC-Exp CRF run of seed 0 (`smokeak`), then
runs the CRF head and mention replacement with seed offsets 0 to 900, one
job per system and seed. `ner_seed_analysis.py` (`sofc_draws`) makes the
decision and the reported comparison of 10 articles against all 26. On a
tiny random model every existing option writes byte-identical results
files, and every earlier batch is unchanged. Part 1 runs from the
protocol's commit, part 2 from this one.

*Clarification, made before any result of week 9 (17 October, 12:30).*
Part 1's session on kernel b (version 10, pushed 11:49) hung before its
script started, as week 8, part 2's had (version 8): no line in its log
after 35 minutes, and after the cancel an empty log and no outputs. Part 1
therefore runs on a new notebook (`ner-gpu-jobs-c`), from part 2's bundle
(9468ae7), one bundle for both parts as in week 8. Between the protocol's
commit (536cbe7) and 9468ae7 the training code only gains `--train-articles`
(with `--draws`), which is off unless given; the batch `w9a` is the same at
both commits, and every existing option writes byte-identical results files
there (the tiny-model check of 9468ae7). The batch's repeats (`smokeah`, the
CRF run of smoke4; `smokeai`, the multi-task run of random split 0) check
this on the GPU. Nothing else in the design changes.
- *Still no GPU session (13:05).* Part 2 (kernel a, version 18, pushed
  12:03) and part 1 (`ner-gpu-jobs-c`, pushed 12:30) wait in Kaggle's queue.
  The pinned Kaggle image is not the cause: this morning's last good session
  ran it (Python 3.13, torch 2.11, CUDA 12.8, the generation of Kaggle's
  current default image), and two CPU probe notebooks started and finished
  within 30 seconds. Since 07:24 every GPU session has either queued or
  reported "running" without its script starting; queued sessions are not
  charged.
- *14:51: part 2 "runs"* (kernel a, version 18), and *15:26: part 1 runs*
  (`ner-gpu-jobs-c`). Part 1's live log showed every setup step within a
  minute, both repeats done (1.3 and 2.8 minutes) and then its jobs (15–23
  minutes each). Part 2's live log stayed empty or answered with a server
  error for two hours. Every session today that ran streamed its log from
  the first minute (week 8, part 1; week 9, part 1), and every session whose
  log stayed empty never printed a line (versions 8, 10 and 18), so an
  empty live log a few minutes after the start marks a stuck session.
  Kaggle refunded version 10's time after its cancel (used quota 15.82
  hours at 14:56).
- *17:14: part 2 runs again* on a new notebook (`ner-gpu-jobs-d`, from the
  same bundle 9468ae7) after version 18 was cancelled (its output: an empty
  log, no manifest). Its log showed every setup step within a minute and
  its first repeat (`smokeaj`) done in 1.2 minutes.
- *18:39: week 8, part 2 hangs again*, now on a new notebook
  (`ner-gpu-jobs-e`, from its bundle 04352cd, uploaded at 18:13 after both
  running sessions had logged their inputs in place): "running", but its
  live log is still empty after 5 minutes, while part 2's log on
  `ner-gpu-jobs-d` streams. It is to be cancelled on the website and pushed
  again. A new notebook does not prevent a hang.
- *18:48: week 8, part 2 pushed again*, to `ner-gpu-jobs-c` (version 2,
  after version 1's outputs were pulled), once version 1 of
  `ner-gpu-jobs-e` was cancelled; Kaggle queued it.
- *19:40: week 7, part 2e pushed* to `ner-gpu-jobs-d` (version 2, after
  part 2's outputs were pulled), from its own bundle 4f5dbda in a second
  dataset (`ner-gpu-bundle-2`; `kaggle_jobs.py --dataset`, e970571). A new
  version of the one bundle dataset would have had to wait until the queued
  week-8 session had logged its inputs in place. Its BUNDLE.json names
  4f5dbda, and its batch is the same at 4f5dbda and now. Kaggle queued it
  too.
- *19:43: week 7, part 2e hangs.* It reports "running" on
  `ner-gpu-jobs-d` (version 2), but its live log is still empty after 4
  minutes. It is to be cancelled.
- *A pattern (a hypothesis, untested).* The times between a session's push
  and the moment its bundle dataset got a version:

  | Session | Bundle version ready | Outcome |
  |---|---|---|
  | `w8a`, kernel a, 06:55 | 1 minute before | ran |
  | `w8b`, kernel b v8, 07:24 | a new version 4 minutes after | hung |
  | `w9a`, kernel b v10, 11:49 | a new version 7 minutes after | hung |
  | `w9b`, kernel a v18, 12:03 | 1 minute before | queued, then hung |
  | `w9a`, `ner-gpu-jobs-c` v1, 12:30 | 28 minutes before | queued, then ran |
  | `w9b`, `ner-gpu-jobs-d` v1, 17:14 | 5 hours before | ran |
  | `w8b`, `ner-gpu-jobs-e` v1, 18:39 | 19 minutes before | hung |
  | `w7e`, `ner-gpu-jobs-d` v2, 19:40 | under a minute before (a new dataset) | hung |

  Every session pushed within about 20 minutes of a new version of its
  bundle hung, except the morning's `w8a`; the two pushed 28 minutes or
  more after ran. From now on a session is pushed only once its bundle has
  been ready for 30 minutes, and a bundle dataset gets no new version within
  30 minutes after a session that attaches it was pushed. `w8b` on `ner-gpu-jobs-c` (version 2, pushed 28 minutes
  after its bundle was ready) is still queued.
- *20:10: week 7, part 2e runs.* Version 2 of `ner-gpu-jobs-d` was
  cancelled (its output: an empty log, no manifest) and part 2e was pushed
  again as version 3, 30 minutes after its bundle was ready. Its log showed
  every setup step within 1.2 minutes and its first job (the repeat) done in
  1.3 minutes: the first session pushed under the 30-minute rule runs.
- *20:22: week 10's bundle* (a08356f, the protocol's commit) is uploaded to
  a third dataset, `ner-gpu-bundle-3`, so that no dataset a session
  attaches gets a new version. Its BUNDLE.json names a08356f and the 52
  converted MuLMS files with the SHA-256 that `results/mulms_corpus.json`
  records. The dataset holds the 1,378 files of the bundle (the 977 files of
  a08356f, extracted, and 401 others whose sizes match the local bundle),
  although the CLI's reply to the upload was not JSON.
- *20:26: week 8, part 2 runs.* `w8b` on `ner-gpu-jobs-c` (version 2)
  left the queue after 98 minutes, although the two later sessions on
  `ner-gpu-jobs-d` had started at once. Its log showed every setup step
  within 1.7 minutes and its first job (the repeat) done in 1.2 minutes. A
  long wait in the queue is not a hang.
- *20:49: week 7, part 2e done* (37.6 minutes; outcome under its protocol).
  Its outputs are pulled and collected, and `ner-gpu-jobs-d` is free.
- *20:57: week 10 pushed* (`w10a`) to `ner-gpu-jobs-d` (version 4), from
  `ner-gpu-bundle-3` (a08356f), 31 minutes after the bundle was ready. It
  started at once next to `w8b`. Its log showed every setup step within 1.1
  minutes and its first job (the repeat `smokeal`) done in 1.3 minutes.
- *21:07: week 8, part 2 done* (40.0 minutes; outcome under its protocol).
  Its outputs are pulled and collected; week 10 runs alone.
- *21:37: week 10's first mention-replacement jobs done* (each the 7
  held-out subfields of one seed): 22.4 minutes on lane 1 and 21.8 on
  lane 0, with no error in the log. The cost model allowed about 29
  minutes, so the session should end around 23:45–00:00 rather than
  00:30–01:00.
- *21:55: week 10's post-processing checked before its results.*
  `ner_matching.py` keyed its random-against-unseen gap on MaterioMiner's
  protocol name, so it would have stopped on MuLMS; each corpus now uses its
  own two protocols (a test fails without the fix). Table 4 and Figures 1
  and 5 gain MuLMS's settings, and the seed analysis adds the CRF votes on
  the random pool splits (descriptive). A dry run on files from a tiny
  random model, trained on the real MuLMS folds and splits on the CPU,
  goes through the seed analysis, the mention and matching analyses, Table
  4 and both figures.
- *23:01: week 10's 10 mention-replacement jobs done* (19.5–26.3 minutes
  each, no error in the log). The first CRF job had run 10 minutes without
  finishing, longer than the cost model's 6: loading the model and scoring
  the held-out subfield weigh more than training on 340 sentences. The
  session should end around 00:15–00:40.
- *00:40 (18 October): week 10 done* (221.8 minutes; outcome under its
  protocol). Its 31 jobs ended without error: mention replacement took
  19.5–26.3 minutes per job, the CRF head 10.1–13.3 on the held-out
  subfields and 8.3–10.8 on the pool splits. Its outputs are pulled and
  collected; no session runs.
- *01:19: week 11's bundle* (427e34c, the protocol's commit) is a new version
  of `ner-gpu-bundle-3`, ready after the CLI retried four 429 replies.
  `w11a` goes to `ner-gpu-jobs-d` once the bundle has been ready for 30
  minutes and its BUNDLE.json names 427e34c. 3.95 GPU hours remain this week.
- *01:32: week 11 checked on the real data.* BUNDLE.json names 427e34c, the
  fatigue corpus's SHA-256 and the 52 MuLMS files. One `w11a` job (graphene)
  ran with its batch arguments on a tiny random model on the CPU, with two
  students. Its teachers tagged the fold's 9,433 sentences outside the pool and
  the test set (10,184 − 400 − 351), none skipped. The random model found no
  silver sentence, so the students trained on the training names, and the
  records count the test set's 1,549 entities, 1,264 of them unseen. The
  teacher of seed 0 reproduces week 10's tiny-model CRF run on graphene
  (1.11). One `w11b` job (coarse-grained, the paper 10.1007_s11340) passed
  the same check. Its teachers tagged all 40,000 fatigue sentences (the
  corpus's SHA-256 is the bundle's), and the record counts the paper's 173
  test entities, 129 of them unseen. Its 963 copies equal those of week 7's
  mention replacement run of seed 0 on that paper.
- *01:50: week 11, part `w11a` pushed* to `ner-gpu-jobs-d` (version 5), 31
  minutes after its bundle was ready. It started at once. Its log showed every
  setup step within 1.0 minute (383 results files in place) and the repeat
  `smokeam` done in 1.25 minutes.
- *02:31: the first held-out subfield of `w11a` done on both lanes*
  (electrolysis: 3 teachers, the tagging of the fold's unused sentences and 5
  students per lane, 22.4 and 22.9 minutes, no error in the log). At that
  pace the session ends around 04:35, within this week's quota.
- *04:44: week 11, part `w11a` done* (173 minutes; outcome under the week-11
  protocol). Its 15 jobs ended without error, 16.4–31.5 minutes per
  subfield; poly, the largest test set, was the slowest on both lanes. Its
  outputs are pulled and collected. About 1 GPU hour remains until the quota
  resets on 24 October; `w11b` runs after the reset.

**Outcome of part 1: 3 of the 6 decisions pass; the combination's vote is
the strongest fine-grained system on unseen papers, but S2 fails at both
levels, so no methods claim.** Session c of 17 October (`w9a` on
`ner-gpu-jobs-c`, from bundle 9468ae7; 3.2 hours from 15:26). The repeats
are identical: `smokeah` at 77.95, and `smokeai` at 78.59 fine-grained and
79.95 coarse-grained. 10 seeds on the 4 held-out papers:
- decision 1, against the multi-task model: **fine-grained +3.01 [+0.42,
  +5.60]** (every paper gains: +0.81, +3.47, +4.71, +3.05) and
  **coarse-grained +2.30 [+0.49, +4.10]** (+2.12, +2.85, +0.79, +3.42).
  Both pass;
- decision 2, its vote against the strongest system on unseen papers:
  **fine-grained +2.86 [+0.18, +5.54] against the multi-task vote** (+0.83,
  +4.48, +3.99, +2.16), which passes; coarse-grained +1.65 [−0.47, +3.78]
  against the seed vote (+1.07, +3.46, +0.34, +1.74), which does not;
- S2 on SOFC-Exp's official split, paired by seed: on the slot types 72.57
  [70.83, 74.31] against the CRF head's 71.88, +0.69 [−1.18, +2.57]; on
  the entity types 83.56 [82.82, 84.30] against 83.75, −0.19 [−1.27,
  +0.88]. Neither passes.

3 of the 6 decisions pass, where about 0.15 would pass if the combination
did nothing. On MaterioMiner the two methods add up. Against the CRF head
the combination gains +6.73 [+6.03, +7.43] fine-grained (every paper +6.25
to +7.16), against +3.72 for the multi-task model and +2.77 for mention
replacement alone (sum +6.49); coarse-grained +1.74 [−0.15, +3.62],
against −0.56 and +1.28. Its vote scores 55.33 fine-grained on unseen
papers, against 52.47 for the multi-task vote and 48.81 for the seed vote,
and coarse-grained 62.75 against the seed vote's 61.10. A methods claim
needed decision 2 and S2 at a matching level (fine-grained with the slot
types): decision 2 passes fine-grained, but S2 on the slot types does not.
The combination is therefore a result for few annotated papers (C21), not
the methods claim. Part 2 asks whether SOFC-Exp's larger training set
explains the difference. For later protocols, the strongest fine-grained
system on unseen papers is this vote (`crf_mt_mr_ens3`); coarse-grained it
remains the seed vote.

Reported:
- against the CRF head, as above; against mention replacement alone,
  +3.96 [+3.31, +4.60] fine-grained and +0.45 [−0.81, +1.72]
  coarse-grained;
- its vote against the seed vote: fine-grained +6.51 [+3.91, +9.12] (every
  paper +4.51 to +8.32), against the multi-task vote's +3.65; coarse-grained
  against the multi-task vote +1.97 [+0.22, +3.72];
- strict IOB2 equals the default scoring. Against the multi-task model,
  macro F1 +2.22 [−1.52, +5.97] fine-grained and +4.21 [+2.85, +5.58]
  coarse-grained, weighted F1 +3.11 [+1.02, +5.20] and +2.80 [+0.85,
  +4.74];
- on unseen names (`ner_mentions.py`, against the CRF head): fine-grained
  +7.10 [+6.43, +7.77], seen names +5.88 [+0.42, +11.34], the first
  fine-grained gain whose interval on seen names also lies above 0;
  coarse-grained +1.76 [−0.18, +3.69] and +1.12 [−0.96, +3.19]. The vote
  against the seed vote: +6.55 [+4.16, +8.93] on unseen names, +6.54
  [−6.20, +19.27] on seen ones;
- per type (exploratory, against the CRF head): fine-grained 18 of the 27
  types gain and 6 lose (TestingMachine −3.81 the most), the largest gains
  on MicrostructureEntity (+33.18), FatigueLifetime (+14.73),
  MechanicalTestDataSet (+14.08), Mechanism (+13.27) and StateDescriptor
  (+10.68). Coarse-grained 21 of the 27 gain. Of the seven types on which
  multi-task training lost, six now gain (SDE +1.84, MIP +0.07, PRP +2.73,
  MOD +11.82, CPR +23.70, CGE +5.65) and ODE loses (−1.86);
- copies per run: 551–914 (mean 734) on 234–370 training sentences, one
  set of copies for both heads; on SOFC-Exp 1,514–1,547 on 568;
- SOFC-Exp: the votes over the 3 seed triples score 75.16 on the slot types
  and 85.44 on the entity types, +0.84 [−3.00, +4.68] and +0.31 [−2.82,
  +3.44] against the CRF votes; on unseen names +1.54 [−1.48, +4.57] and
  +0.18 [−1.50, +1.86], on seen names −0.72 [−1.58, +0.13] and −0.47
  [−1.35, +0.41].

**Outcome of part 2: the decision does not pass; mention replacement's
gain on MaterioMiner is not a matter of training size.** Session d of 17
October (`w9b` on `ner-gpu-jobs-d`, from bundle 9468ae7; 2.2 hours from
17:15). The repeats are identical: `smokeaj` at 77.95, and `smokeak`
(session b's SOFC-Exp CRF run of seed 0) at 84.98. On SOFC-Exp's entity
types, trained on 10 of the 26 official training articles (5 draws, 156–248
sentences), 10 seeds per draw:
- the decision, mention replacement against the CRF head, paired by draw:
  81.12 [79.85, 82.40] against 80.62 [79.89, 81.36], +0.50 [−0.47, +1.46]
  (per draw +0.89, −0.42, +1.02, +1.25, −0.26). It does not pass.

Reported:
- macro F1 +0.58 [−0.17, +1.33] and weighted F1 +0.52 [−0.39, +1.44];
  strict IOB2 equals the default scoring;
- the CRF head on 10 articles against all 26 (the official split, the same
  seeds): −3.13 [−4.04, −2.21], paired by seed. 16 more training articles
  are worth about 3 points on SOFC-Exp's test articles;
- copies per run: 418–690 on 156–248 training sentences.

At MaterioMiner's size, mention replacement gains +0.50 on SOFC-Exp,
against +2.77 fine-grained and +1.28 coarse-grained on MaterioMiner's
held-out papers, where the protocol expected a gain. Its MaterioMiner gain
therefore belongs to that corpus, not to small training sets as such, and
C21 remains a result for MaterioMiner. One difference besides size
(post hoc, untested): 81% of a held-out paper's fine-grained names are
unseen and the CRF head finds 35% of them, against 49% unseen on SOFC-Exp's
official split, of which it finds 82% (C17). Mention replacement may help
where unseen names are many and hard to find.

### Candidates for week 10 (17 October; not a protocol)

Written down before any run of week 9, so the choice made after its
decisions can be checked against this list. None is fixed yet.
1. **Extra corpora, each with its own head.** C19 makes more papers the
   strongest lever, but outside evidence is against this way of adding
   them: in MuLMS (Schrader et al., 2023, Table 11), one MatSciBERT with
   a head per corpus, trained on MuLMS together with SOFC-Exp or a
   synthesis-procedure corpus (MSPT), changes micro F1 by −0.1 to −1.5 on
   each corpus's own test set. Those test sets are in distribution, with
   thousands of training sentences; ours are unseen papers with three
   training papers. A screen at most, and only if the others fail.
2. **MuLMS as a third corpus.** 50 CC-BY articles from 7 subfields
   (electrolysis, graphene, PEMFC, SOFC, polymers, semiconductors, steel),
   split 36/7/7 by document; 10,186 sentences and 46,351 entities of 13
   types; its MatSciBERT-CRF tagger scores 79.6 micro F1 on the test
   documents. Leaving one subfield out would test C2 across topics, and
   draws of a MaterioMiner-sized training set would test mention
   replacement at small size, as week 9, part 2 does on SOFC-Exp. The
   corpus is on Hugging Face (`timo-pierre-schrader/MuLMS`, CC BY-SA 4.0,
   parquet): sentences with their tokens, the document's subfield
   (`category`) and BILOU tags in which nested entities stack on a token
   (`B-DEV+U-MAT`). A flat version must therefore be chosen and checked
   first. Under share-alike, a build script with a SHA-256 check, as for
   the fatigue corpus, would keep converted files out of the repository.
   *Checked on 17 October (revision bc1c97d, no runs):* 50 documents,
   10,184 sentences and 46,584 entities. Per subfield: electrolysis 4
   documents, graphene 2, PEMFC 8, polymers 11, semiconductors 11, SOFC 4,
   steel 10; each has a test document. 32.9% of the entities lie inside
   another or share its span, mostly NUM or UNIT inside VALUE (5,461 and
   4,392), MAT inside MAT (2,035) and NUM inside RANGE (1,258); 769 pairs
   share a span (NUM and VALUE 481, RANGE and VALUE 165) and 30 pairs
   cross. *Corrected at 19:50:* the first count compared the character
   offsets, which the corpus stores as strings, as text (43.4% nested, 1,201
   crossing pairs); these counts use the token indices. Every subfield but
   graphene could be held out with at least 4 documents.
3. **A vote across methods.** Coarse-grained, multi-task training and
   mention replacement change different types (per-type correlation
   −0.72, above). A vote whose members come from different methods (the
   multi-task model, mention replacement and, after week 9, the two
   together) needs no training, since every member saved its predictions.
   It would be decided against the strongest system like any vote. The
   mixed-backbone vote of week 3 (three encoders) did not pass.
4. **Mention replacement with names from outside the training papers:**
   the silver names the teachers tag in the 40,000 corpus sentences, so
   that the copies hold more names than three papers do. It needs the
   teachers of each fold, as self-training does.

### Week 10 protocol: mention replacement on held-out subfields of a third corpus, MuLMS (fixed before any run)

Fixed on 17 October, after both parts of week 9 and before any run on MuLMS.
Chosen from the candidates above (item 2).

*Why.* Week 9 left mention replacement and its combination with multi-task
training as results for MaterioMiner. Neither gains on SOFC-Exp, on its
official split (C20, C21) or with its training set cut to MaterioMiner's size
(part 2). SOFC-Exp's 45 articles share one topic. 49% of its test names are
unseen there and the CRF head finds 82% of them, against 81% and 35% on a
held-out MaterioMiner paper (C17). Mention replacement may help only where
unseen names are many and hard to find (the post hoc reading of week 9, part
2). MuLMS (Schrader et al., 2023) covers seven materials subfields, so holding
one out makes the test documents differ in topic from the training documents,
as a new MaterioMiner paper does. Multi-task training needs two levels of
types and MuLMS has one, so only mention replacement is tested.

*Data.* MuLMS from Hugging Face (`timo-pierre-schrader/MuLMS`, revision
bc1c97d, CC BY-SA 4.0): 50 documents, 10,184 sentences and 46,584 entities of
13 types, 32.9% of them inside or on the span of another. A flat tagger needs
one level, so `mulms.py` applies three rules:
- it keeps the outermost entities, those that no other entity contains;
- of entities with the same span, it keeps the type that comes first in
  MEASUREMENT, VALUE, RANGE, SAMPLE, PROPERTY, TECHNIQUE, INSTRUMENT, DEV,
  CITE, MAT, FORM, NUM, UNIT (the larger unit first: VALUE over NUM);
- of two that cross, it keeps the one that starts first.

The rules keep 31,952 entities (68.6%). They drop 13,891 as nested, 721 with
the span of a kept one and 18 that cross a kept one. Kept per type: MAT
12,647, VALUE 4,628, PROPERTY 3,880, FORM 3,180, MEASUREMENT 2,106, CITE
1,704, SAMPLE 1,391, TECHNIQUE 920, DEV 757, INSTRUMENT 370, NUM 260, UNIT 91
and RANGE 18.

A fourth rule repairs the tokens that text extraction damaged:
- 261 tokens, in 121 sentences of 9 documents, are empty or hold whitespace.
  They include:
  - a lone space where a minus or degree sign was lost (`m2 g ' ' 1`,
    `80 ' ' C`);
  - non-breaking spaces;
  - fragments with spaces inside (`The cataly`);
  - 114 tokens in one garbled document (pemfc_14).
- 15 more tokens are private-use symbol characters or lone combining marks,
  which MatSciBERT's tokenizer erases.

A CoNLL line cannot hold whitespace, and a word without a subword cannot be
tagged, so scoring would stop on it. So each token becomes its
whitespace-separated pieces, and a piece made only of characters a BERT
tokenizer erases is dropped (Unicode categories Cc, Cf, Co, Cs and Mn). This
drops 218 tokens (203 blank, 15 invisible), splits 31 into pieces and trims
the surrounding whitespace of 27. 2 entities are left without tokens and are
dropped. All 10,184 sentences stay, with 292,513 words. Read back as
`ner_baseline.py` reads them, every word gets a MatSciBERT subword, and the
longest sentence needs 309 subwords (the limit is 512).

| Subfield | Documents | Sentences | Kept entities |
|---|---|---|---|
| electrolysis | 4 | 992 | 3,787 |
| graphene | 2 | 351 | 1,549 |
| pemfc | 8 | 1,638 | 7,238 |
| poly | 11 | 2,384 | 5,430 |
| semi | 11 | 2,040 | 5,400 |
| sofc | 4 | 928 | 3,854 |
| steel | 10 | 1,851 | 4,694 |

The converted files stay out of the repository (share-alike).
`results/mulms_corpus.json` records the input and output SHA-256 (the outputs
combined: afa79ed) and every count. The published MatSciBERT-CRF (79.6 micro F1) tags the nested entities
with stacked labels and is scored on all of them, so these scores cannot be
compared with it.

*Protocols* (`--granularity mulms`, about MaterioMiner's training size: 234–370
training sentences per held-out paper):
- *held-out subfields* (`--leave-one-subfield-out --train-pool 400`): 7
  folds, one per subfield, whose documents are the test set (351–2,384
  sentences). The other documents are sorted and shuffled with split seed 0
  plus the fold's index, then taken in that order until they hold 400
  sentences, the last one cut to its first sentences (about two documents).
  Of the 400, 15% validate (60) and 340 train;
- *random pool splits* (`--vary-split --train-pool 400`, descriptive): 5
  splits (split seeds 0–4). 340 training and 60 validation sentences are
  drawn from the whole corpus, and the next 20% of the sentences (2,037)
  test.

*Runs* (batch `w10a`, one Kaggle session from this protocol's commit; GPU,
deterministic): first the repeat of smoke4's CRF run (`smokeal`). Then, with
model seeds 0 to 900 (seed offsets) and predictions:
- on the held-out subfields, the CRF head (`crf_m<N>`) and mention
  replacement (`crf_mr_m<N>`, `--augment-mentions 3`, as in week 7);
- on the random pool splits, the CRF head.

The votes cover the 3 seed triples.

*Decisions* (the week-6 rule). The pairs are the 7 held-out subfields, each
the mean of its 10 seeds (6 degrees of freedom):
1. mention replacement against the CRF head;
2. its votes against the CRF votes, per subfield the mean of its 3 triples.

With 2 decisions, about 0.05 false passes are expected if mention
replacement does nothing.

*What a pass means.* If decision 1 passes, mention replacement's gain on
unseen documents holds on two corpora whose held-out documents differ in
topic (MaterioMiner's papers, MuLMS's subfields). It holds on none whose
articles share a topic (SOFC-Exp). That is a methods claim scoped to that
setting, reported together with SOFC-Exp's results. If decision 1 fails, C20
and C21 stay results for MaterioMiner.

*Reported:*
- the scores per subfield;
- macro, weighted and strict F1;
- the CRF head on the random pool splits against the held-out subfields (the
  gap of C2, on a third corpus);
- unseen names per setting and the gains on them (`ner_mentions.py`, once it
  reads MuLMS; no training);
- the number of copies per run.

*Expected.* On a held-out subfield most test names should be unseen, as on a
held-out MaterioMiner paper. If the reading of week 9, part 2 holds, mention
replacement gains there. MuLMS's most frequent kept types are materials and
values, whose patterned names transfer best on MaterioMiner (C17), so a
smaller gain than MaterioMiner's is plausible.

*Budget:* 20 jobs of 7 runs and 10 jobs of 5 runs, each on 340 training
sentences (mention replacement on about 4 times as many). That is about 3.5
hours of one session on two lanes. The session is pushed once its bundle has
been ready for 30 minutes (session notes of week 9). MuLMS runs' results
files hold MuLMS's tags (`test_gold`) and are CC BY-SA 4.0
(`results/README.md`).

*Prepared (17 October, tests only, before any run):*
- `mulms.py`;
- `ner_baseline.py --granularity mulms`, with `--leave-one-subfield-out
  --train-pool N` and `--vary-split --train-pool N` (protocols
  `leave_one_subfield_out` and `random_split_pool`). On a tiny random model
  every existing option writes byte-identical results files;
- `ner_seed_analysis.py` (`mulms`: both decisions, the votes and the gap);
- `kaggle_jobs.py`: batch `w10a`; the bundle ships the converted files as
  `mulms__<name>`, each checked against its recorded SHA-256 on Kaggle. Every
  earlier batch is unchanged.

**Outcome of week 10: decision 1 fails, decision 2 passes, so mention
replacement stays a MaterioMiner result.** Session d of 17–18 October
(`w10a` on `ner-gpu-jobs-d`, version 4, from bundle a08356f on
`ner-gpu-bundle-3`; 3.7 hours from 20:58). The repeat `smokeal` is
identical (77.95). On the 7 held-out subfields, 10 seeds each:
- decision 1, mention replacement against the CRF head: **+2.10 [−0.31,
  +4.51]** (50.76 against 48.66), which does not pass. Six subfields gain
  (electrolysis +1.42, graphene +1.66, pemfc +2.28, semi +2.52, sofc +7.39,
  steel +0.15) and poly loses 0.73;
- decision 2, its votes against the CRF votes: **+2.71 [+0.22, +5.20]**
  (53.25 against 50.54), which passes. Per subfield, in the same order:
  +2.12, +2.12, +3.18, −0.65 (poly), +3.38, +7.96 and +0.88.

About 0.05 false passes are expected from 2 decisions. The protocol makes
decision 1 the condition for a methods claim, so C20 and C21 stay results
for MaterioMiner.

Reported:
- macro F1 +2.12 [+1.14, +3.09] and weighted F1 +2.68 [+0.32, +5.04];
  strict IOB2 equals the default scoring;
- per subfield, the CRF head scores 32.49 (semi) to 56.82 (steel);
- the CRF head on 5 random splits of the same size: 65.72 [65.25, 66.19]
  (its votes 67.87), 17.06 points above the held-out subfields (macro
  15.06, weighted 18.74). On SOFC-Exp, held-out articles cost 3.48 points;
  on MaterioMiner a held-out paper costs 18–31;
- unseen names (`ner_mentions.py`): on a held-out subfield 88.0% of the
  test names are unseen in the training sentences and the CRF head finds
  39.4% of them (seen names: 66.5%). On the random splits 57.3% are unseen
  and it finds 54.1% (seen: 81.2%). The larger share of unseen names
  accounts for about a third of the 22.5-point recall gap (7.7–7.8
  points); lower recall within each kind accounts for the rest.
  Mention replacement gains +2.19 [−0.25, +4.63] on unseen names (seen
  +1.30 [−1.21, +3.81]), its votes +2.92 [+0.41, +5.44];
- looser matching (`ner_matching.py`): on the held-out subfields the CRF
  head scores 48.66 strict, 54.59 on the span alone and 69.06 under any
  overlap (random splits 65.72, 70.22 and 81.68), so the gap shrinks from
  17.06 to 12.62 under overlap. 13.1% of its predictions lie inside a
  longer gold entity (9.0% on the random splits);
- copies per run: 429–738 (mean 602).

The condition named after week 9, part 2 holds on MuLMS: unseen names are
many (88%) and hard to find (39%), as on a held-out MaterioMiner paper (81%
and 35%). Mention replacement's mean gain (+2.10) is close to its
fine-grained gain on MaterioMiner (+2.77), but the subfields differ (one
gains 7.4 points, one loses 0.7), and with 7 of them the interval reaches
below 0. Its votes pass. Across the two corpora the passes do not line up.
The single model passes on MaterioMiner at both granularities but not on
MuLMS. The vote against the seed vote passes on MuLMS but not
coarse-grained on MaterioMiner (+2.26 [−0.12, +4.64]); fine-grained it
gained +3.61 [+1.71, +5.52] there (reported, not a decision).

### Candidates for week 11 (18 October; not a protocol)

Written after week 10's decisions, from the list written before week 9 (its
items 1, 3 and 4, renumbered) and one new item.
1. **Extra corpora, each with its own head** (item 1 before). Outside
   evidence is against it (Schrader et al., 2023, Table 11). A screen at
   most, and only if the others fail.
2. **A vote across methods** (item 3 before). It needs no training, but it
   has no route to a repeat on a second corpus. MuLMS has one level of
   types, so it has no multi-task members. On SOFC-Exp mention replacement
   ran only at the entity level, where none of the members gains. Not
   chosen.
3. **Mention replacement with names from outside the training papers**
   (item 4 before): the names in sentences that three CRF teachers tag
   alike and confidently in unlabeled in-domain text join the replacement
   pool. Week 10 shows the setting it is meant for. On a held-out MuLMS
   subfield most names are unseen (88%) and hard to find (39%), and
   mention replacement with the training sentences' names gained +2.10
   without passing. On MaterioMiner the text is the 40,000-sentence
   fatigue corpus of self-training. On MuLMS it is the training subfields'
   sentences outside the 400-sentence pool, without their labels. It
   reuses the multi-teacher machinery of week 5. **Chosen for week 11.**
4. **Names from the test documents themselves** (new): the teachers tag the
   held-out paper's or subfield's own sentences, and their names join the
   pool. Not chosen: it uses the test sentences at training time, a
   different setting from every other week.

### Week 11 protocol: mention replacement with silver names (fixed before any run)

Fixed on 18 October, after week 10's decisions and before any run of week 11.
Chosen from the candidates above (item 3).

*Why.* Mention replacement improves on the CRF head on MaterioMiner's held-out
papers (C20), and on MuLMS's held-out subfields it gained +2.10 [−0.31,
+4.51] without passing (week 10). The names it swaps in are the training
sentences' own: 602–1,066 mentions per fold fine-grained and 1,034–1,713
coarse-grained from three papers, and 601–1,309 from 340 MuLMS sentences. A held-out paper or subfield is mostly new names (81% and 88%),
which the CRF head finds less than half of the time (35% and 39%). The names
of unlabeled in-domain text, tagged by teachers, give the copies many more
names than the training sentences hold. This week tests whether they add to
mention replacement where new names dominate.

*Method* (`--augment-names`, variant `crf_mrs`):
- **Teachers:** 3 CRF heads trained on the fold's training sentences with
  model seeds 0, 100 and 200. These repeat the runs `crf_m0`, `crf_m100` and
  `crf_m200` of weeks 5 (MaterioMiner) and 10 (MuLMS). GPU runs are
  deterministic, so their test scores must equal the stored ones (a check).
- **Unlabeled text:**
  - MaterioMiner: the whole fatigue corpus, 40,000 sentences of open-access
    fatigue papers (self-training's corpus; its SHA-256 is checked).
  - MuLMS: per fold, the sentences of the training subfields outside the
    400-sentence pool, without their tags (`--self-train unused`). The held-out
    subfield's sentences are never tagged.
- **Silver sentences:** as in weeks 4 and 5. All 3 teachers give a sentence
  the same tags, every tag has a CRF marginal of at least 0.9 under each
  teacher, and the sentence has an entity.
- **Silver names:** the mentions (type and words) of the silver sentences.
- **Students:** model seeds 0 to 900. Each trains on the gold sentences and
  their mention replacement copies, as `crf_mr_m<N>` does: 3 copies of each
  sentence with an entity, each mention replaced with probability 0.5, with
  the copy seed equal to the model seed. A mention's replacement is drawn from
  the training mentions and the silver mentions of its type, frequent ones
  more often. The silver sentences themselves are not trained on.
  - Without silver names a student equals the `crf_mr` run of its seed
    (tested), so `crf_mrs` and `crf_mr` differ only in the pool.
- **Recorded per run** (counts only; no name is stored, since the corpus is
  not redistributed):
  - the silver sentences, mentions and distinct names;
  - the unseen test entities (their type and words no entity of the training
    sentences) and how many of them are silver names.

*Decisions* (the week-6 rule: the 95% Student-t CI of the mean paired
difference of micro F1 above 0; each pair is a held-out document set, the mean
of its 10 seeds):
1. MuLMS, the 7 held-out subfields (6 degrees of freedom): `crf_mrs` against
   the CRF head (week 10's `crf`).
2. MuLMS: `crf_mrs` against mention replacement (week 10's `crf_mr`).
3. MaterioMiner fine-grained, the 4 held-out papers (3 degrees of freedom):
   `crf_mrs` against mention replacement (week 7's `crf_mr`).
4. MaterioMiner coarse-grained: the same.

With 4 decisions, about 0.1 false passes are expected.

*What a pass means.*
- **Decision 1 and one of decisions 2–4 pass:** the method improves on the CRF
  head on held-out subfields, where mention replacement alone did not pass,
  and the silver names add to mention replacement on at least one corpus. That
  is a methods claim scoped to unseen documents of a new topic.
- **Decision 1 passes but none of 2–4:** no claim. The gain may be mention
  replacement's own, whose week-10 interval reached +4.51.
- **Decision 1 fails:** mention replacement, with or without silver names,
  stays a MaterioMiner result.

*Reported:*
- `crf_mrs` against the CRF head on MaterioMiner;
- the votes over the 3 seed triples:
  - MuLMS: against the CRF votes and mention replacement's;
  - MaterioMiner: against mention replacement's vote, the seed vote and the
    multi-task vote;
- macro, weighted and strict F1, and the scores per paper and subfield;
- seen and unseen names (`ner_mentions.py`);
- the silver counts and the share of unseen test names that are silver names;
- copies per run;
- the teachers' identity with the stored CRF runs.

*Runs* (GPU, deterministic; each batch from this protocol's commit, the
repeat of smoke4's CRF run first):
- `w11a`, MuLMS (repeat `smokeam`). Each lane runs the 7 subfields, one job
  each (`--subfields`). A lane trains the 3 teachers of a fold and half of
  its students: seed offsets 0–400 on lane 0, 500–900 on lane 1. So the
  lanes write different files, and their teachers' scores and silver counts
  must be identical (a check). About 2.9 hours: 3.95 GPU hours remain this
  week.
- `w11b`, MaterioMiner (repeat `smokean`). One lane per granularity, one job
  per held-out paper (`--papers`). About 2.1 hours, after the weekly quota
  resets on 24 October.

The decisions on each corpus are made when its batch is in.

*Expected.* The silver names are names the teachers already find, so they may
help little with the names the model misses; the coverage count shows how many
of the unseen test names they hold. MaterioMiner's fatigue corpus shares its
domain; MuLMS's unused sentences come from the other subfields, so their names
may cover the held-out subfield less.

*Prepared (18 October, 50c07dc, tests only, before any run):*
- `ner_baseline.py`: `--augment-names`, `--self-train unused` and
  `--subfields`. On a tiny random model, every existing option that the change
  touches writes byte-identical results files: the CRF head, mention
  replacement on one model and multi-task, self-training with teachers, and
  MuLMS's held-out subfields and pool splits;
- `kaggle_jobs.py`: batches `w11a` and `w11b`; every earlier batch is
  unchanged;
- `ner_seedmeans.py --week 11` (decisions 3 and 4), `ner_seed_analysis.py`
  (`mulms`, `silver_names`: decisions 1 and 2), `report_results.py`,
  `ner_mentions.py`, and the paper's tables and figures.

**Outcome of part `w11a` (MuLMS): decisions 1 and 2 fail, so no methods
claim; decisions 3 and 4 (MaterioMiner, `w11b`) follow after the reset.**
Session d of 18 October (`w11a` on `ner-gpu-jobs-d`, version 5, from bundle
427e34c on `ner-gpu-bundle-3`; 2.9 hours from 01:51). The repeat `smokeam` is
identical (77.95). Per subfield, both lanes' 3 teachers and silver counts are
identical, and every teacher scores exactly as the stored week-10 CRF run of
its seed. On the 7 held-out subfields, 10 seeds each:
- decision 1, mention replacement with silver names against the CRF head:
  **+2.04 [−0.85, +4.92]** (50.69 against 48.66), which does not pass. Per
  subfield: electrolysis +2.64, graphene +0.38, pemfc +1.61, poly −0.28,
  semi +1.16, sofc +8.74, steel −0.00;
- decision 2, against mention replacement: **−0.06 [−1.09, +0.96]** (50.69
  against 50.76), which does not pass. Per subfield, in the same order:
  +1.21, −1.28, −0.67, +0.45, −1.36, +1.35 and −0.15.

Reported:
- macro F1 against the CRF head +2.37 [+1.27, +3.47], weighted +2.62
  [+0.02, +5.22]; against mention replacement +0.25 [−0.52, +1.02] and
  −0.06 [−1.19, +1.06]; strict IOB2 equals the default scoring;
- its votes score 53.02: +2.48 [−0.44, +5.40] against the CRF votes and
  −0.23 [−1.15, +0.68] against mention replacement's;
- the silver names: per fold the teachers tagged the 7,400–9,433 sentences
  outside the pool and the test set, and 181–419 of them were silver,
  holding 210–850 mentions (126–358 distinct names). They contain 4.4% of
  the held-out subfields' unseen test names (1.0% to 7.8% per subfield);
- seen and unseen names (`ner_mentions.py`): against the CRF head +2.13
  [−0.84, +5.09] on unseen names, +1.21 [−0.97, +3.39] on seen ones;
- copies per run: as mention replacement's within a few (433–722 per run
  at seed 0, against 439–713). A copy that changes nothing is dropped, and
  the pool decides which do.

The silver names barely touch the names the model misses. The teachers'
confident names come from the other subfields and hold 4.4% of a held-out
subfield's unseen test names. So the copies show the model hardly any new
names that matter, and the method scores as mention replacement does. On
MaterioMiner, 49% to 65% of a held-out paper's unseen test names occur
somewhere in the fatigue corpus (`ner_mentions.py`, corpus coverage), so
decisions 3 and 4 test the method where its names can reach the test
papers. By the protocol, though, a methods claim needed decision 1, so
mention replacement, with or without silver names, stays a MaterioMiner
result whatever `w11b` shows.

**Outcome of part `w11b` (MaterioMiner): decisions 3 and 4 fail as well;
the method is closed on both corpora.** Session d of 19–20 October (`w11b`
on `ner-gpu-jobs-d`, version 6, from bundle 77c37be on `ner-gpu-bundle-3`,
pushed at 21:06 as soon as the quota was raised; 3.4 hours, 9 jobs). The
repeat `smokean` is identical (77.95); per paper and granularity the 3
teachers score exactly as the stored week-10 CRF runs of their seeds, and
the two lanes' silver counts agree. On the 4 held-out papers, 10 seeds
each:
- decision 3, fine-grained, against mention replacement: **+0.23 [−0.69,
  +1.14]** (49.45 against 49.22), not met;
- decision 4, coarse-grained: **+0.61 [−0.96, +2.18]** (60.49 against
  59.88), not met.

Reported:
- against the CRF head +3.00 [+1.98, +4.02] fine-grained and +1.89 [−0.31,
  +4.10] coarse-grained, where mention replacement alone gains +2.77 and
  +1.28 (means); macro F1 +0.58 [−1.65, +2.81] and −0.24 [−2.47, +1.98]
  against mention replacement;
- its votes: +0.44 [−1.85, +2.73] and +0.24 [−1.55, +2.02] against mention
  replacement's votes; +4.05 [+3.03, +5.07] and +2.50 [+0.48, +4.51]
  against the CRF votes; +0.40 [−2.76, +3.55] and +2.81 [+0.95, +4.67]
  against the multi-task votes;
- the silver names: per paper the teachers tag the 40,000 fatigue
  sentences and find 607–1,256 silver sentences fine-grained (917–1,894
  mentions, 549–907 distinct names) and 328–1,443 coarse-grained (631–3,636
  mentions, 421–1,974 names), holding 1.9–9.3% (fine) and 5.3–7.7%
  (coarse) of the held-out paper's unseen test names, although the corpus
  holds 49–65% of them;
- seen and unseen names (`ner_mentions.py`): fine-grained +3.19 [+2.03,
  +4.35] on unseen names and +2.66 [+0.03, +5.29] on seen ones against
  the CRF head.

In-domain silver names do not add to mention replacement on MaterioMiner
either, for the reason seen on MuLMS: the sentence filter admits a small
fraction of the names the model misses (2–9% of a paper's unseen names,
against 49–65% in the corpus), as the CPU probe measures on MuLMS (the
sentence filter 8–26% at 0.9, the mention filter 18–34%). Week 11's
method is closed; week 12's topic corpora with the mention-level filter
and amendment 2's feature are the last pre-registered attempts.

### What follows week 11 (18 October; not a protocol)

*(Resolved on 19–20 October: `w11b` ran early on the raised quota and
failed decisions 3 and 4; the week-12 protocol below had already been
fixed on 18 October with phase 0's coverage as its gate, as item 1 of
"What follows" is superseded by.)*

Mention replacement and its silver-name variant have now failed the
methods criterion on MuLMS. The silver names failed because they hardly
reach the test documents: 4.4% of a held-out subfield's unseen names, since
the names came from the other subfields. That points to one remaining route,
with `w11b` as its gate:
1. **`w11b` first** (fixed by the week-11 protocol, after the reset on 24
   October). On MaterioMiner, 49–65% of a held-out paper's unseen test names
   occur in the fatigue corpus, so decisions 3 and 4 show whether names from
   matching unlabeled text add to mention replacement.
2. **Only if decision 3 or 4 passes: topic corpora for MuLMS.** Unlabeled
   open-access text of each held-out subfield's own topic (as the fatigue
   corpus is for MaterioMiner; an SOFC corpus exists), so that the names
   can reach the test documents without using them. That would be a new
   protocol, with decision 1 of week 11 asked again on MuLMS.
3. **Otherwise the evaluation paper.** The candidates left from week 11's
   list have no route to a methods claim:
   - extra corpora with heads, against which the outside evidence speaks;
   - a vote across methods, which cannot repeat on a second corpus;
   - names from the test documents themselves, which changes the setting
     and acts like the consistency rule of week 7, which did not pass.

### Week 12, phase 0 (18 October; not a protocol): can in-topic names reach a held-out subfield?

Measured before any protocol, on the CPU, to decide whether topic corpora
are worth the GPU hours. All from the saved week-10 predictions and the
existing corpora.

*Where MuLMS's gap lies, per type* (the CRF head, 10 seeds; pool splits
against held-out subfields; mention replacement minus the CRF head, mean
over the 7 subfields):

| Type | Entities | Pool F1 | Held-out F1 | Gap | MR − CRF | Unseen | Recall seen / unseen |
|---|---|---|---|---|---|---|---|
| MAT | 12,647 | 75.2 | 54.2 | 21.0 | +2.85 [−2.6, +11.4] | 85.6% | 70.5 / 43.5 |
| VALUE | 4,628 | 80.6 | 72.8 | 7.8 | +1.92 | 97.2% | 87.9 / 74.2 |
| PROPERTY | 3,880 | 37.4 | 28.7 | 8.7 | +2.62 | 96.1% | 39.2 / 24.0 |
| FORM | 3,180 | 59.9 | 26.0 | 33.9 | +3.48 | 87.2% | 54.9 / 18.4 |
| MEASUREMENT | 2,106 | 48.4 | 40.2 | 8.1 | +0.72 | 67.1% | 45.1 / 33.1 |
| CITE | 1,704 | 93.1 | 81.6 | 11.5 | +2.39 | 70.3% | 99.1 / 70.0 |
| SAMPLE | 1,391 | 44.3 | 4.0 | 40.3 | +1.75 | 98.6% | 7.5 / 2.6 |
| DEV | 757 | 56.3 | 14.7 | 41.5 | −1.47 | 97.1% | 37.8 / 8.9 |

MAT holds 40% of the entities and a 21-point gap, FORM a 34-point gap:
these are names, and mention replacement gains most on them (sofc: MAT
+11.4, FORM +8.8). SAMPLE and DEV are barely found on any held-out subfield.

*Coverage of a held-out subfield's unseen test names by the topic corpora*
(`mulms_coverage.py`, `results/mulms_coverage.json`; 18 October). Seven
corpora of 40,000 sentences, one per subfield (`build_topic_corpus.py`,
`results/topic_corpora/`), built from Europe PMC open-access full texts
whose titles carry the subfield's terms, with MuLMS's 16 identified articles
excluded by PMCID and every sentence sharing an 8-word n-gram or a
50-character shingle with a MuLMS sentence dropped. The share of a held-out
subfield's unseen test entities whose words occur in a row in a corpus
(exact; the corpus cut with MuLMS's hyphen rule) and ignoring word
boundaries and case (loose); the best of the other corpora for comparison
(exact), and as controls the SOFC corpus of the week-6 amendment (built the
same way from 383 hits, before the MuLMS exclusions) and the fatigue corpus
(fatigue papers, mostly steels, which is why it holds 56.5% of steel's
names). No sentence of a topic corpus leaks a MuLMS document (0 shingle
hits against the held-out documents in every cell):

| Held-out subfield | Unseen | Own topic corpus, exact | Loose | Best other topic corpus | SOFC corpus | Fatigue corpus |
|---|---|---|---|---|---|---|
| electrolysis | 83.3% | **70.7%** | 75.6% | pemfc 66.6% | 62.2% | 46.2% |
| graphene | 81.0% | 64.0% | 68.0% | electrolysis **67.3%** | 55.7% | 42.5% |
| pemfc | 86.8% | **73.9%** | 78.3% | electrolysis 69.5% | 59.8% | 46.1% |
| poly | 85.8% | **65.9%** | 69.9% | graphene 55.3% | 48.6% | 53.5% |
| semi | 95.4% | **69.3%** | 74.5% | graphene 62.8% | 55.7% | 50.1% |
| sofc | 93.1% | **80.8%** | 83.9% | electrolysis 71.2% | 80.6% | 49.1% |
| steel | 86.1% | **55.0%** | 63.0% | semi 48.1% | 45.7% | 56.5% |

Week 11's cross-subfield silver names held 4.4% of the unseen test names;
the fatigue corpus holds 49–65% of a held-out MaterioMiner paper's. Per
type, the own corpus holds 77–90% of the unseen MAT names and 85–96% of
the FORM names, the two types with the largest held-out gaps, 82–94% of
MEASUREMENT, 45–74% of PROPERTY and 43–57% of VALUE; SAMPLE, DEV and CITE
vary by subfield (sofc: 58%, 94%, 70%; steel: 9%, 8%, 0%). About 45–65
points of the coverage are generic: any materials corpus of 40,000 sentences
holds them (a steel subfield's names in a graphene corpus: 47.5%). The own
topic adds 4–11 points on six subfields and is 3 points behind the
electrolysis corpus on graphene (MuLMS's graphene documents are
electrochemical). So the names are there; whether the silver filter admits
them and whether they convert into F1 is week 12's question.

*Sources.* Europe PMC has open-access full texts for every subfield: 3,816
titles for electrolysis (electrolysis, electrolyzer, oxygen or hydrogen
evolution), 16,219 for graphene, 487 for PEMFC, 9,158 for polymers
(thermoplastic, epoxy, polyurethane, polypropylene, elastomer, polymer
melt, injection moulding), 5,161 for semiconductors (SiC, GaN, MOSFET,
transparent conducting, photovoltaic, epitaxial, chemical vapour
deposition), 383 for SOFC/SOEC and 7,980 for steel. The SOFC corpus reached
40,000 sentences from 383 hits, so every topic can. MuLMS ships no article
identifiers; 16 of its 50 documents are on Europe PMC (found by title), the
rest come from journals it does not index (MDPI *Metals*, ECS, IOP, Elsevier).
A topic corpus excludes those 16 by PMCID and every sentence that shares an
8-word n-gram or a 50-character shingle (its text without spaces, in lower
case; about 8 words, so the n-gram rule under another tokenization) with any
MuLMS sentence, and drops an article with 5 or more such sentences. The
shingle was first 30 characters, which matched technical terms
("electrochemical impedance spectroscopy", "transmission electron
microscopy", "yttria-stabilized zirconia (YSZ)") in 270–319 of each topic's
articles and dropped 62–224 articles per topic whole; at 50 characters, 15
sentences and no article of the SOFC pool share a run with MuLMS, at 60
none, so no MuLMS document is among the pool's articles.

*Tokens.* MuLMS writes a hyphen as a token of its own (2.2% of its tokens;
0.01% carry an inner hyphen), keeps Unicode dashes inside tokens, and 1.4%
of its tokens are subscript digits that PDF extraction split off (`SiO 2`),
which no tokenizer reproduces. The topic corpora are cut with
ChemDataExtractor's tokenizer, as the SOFC corpus, then ASCII hyphens are
split out.

*Reading.* The mechanism of a topic-corpus route is coherent: the gap is in
names of types that mention replacement already improves, and an in-topic
corpus holds most of those names, where cross-subfield text held almost
none. What is untested is whether names that reach the pool convert into
F1: `w11b` tests that on MaterioMiner, and the silver filter (every tag of
a sentence confident under three teachers) may admit few of them, so a
mention-level filter is a candidate for the screen.

### Session notes, 19 October (the plan, amendment 1, the probe, the TPU port)

- `docs/METHODS_PLAN.md` (09ac262): the plan toward a methods paper, with
  the compute (GPU, TPU, this environment's CPU), the schedule and the stop
  rules.
- Amendment 1 of the week-12 protocol (77c37be, before any run): the screen
  gains `crf_mrm_topic6` and `crf_st3_topic`. The bundle from 77c37be is the
  version of `ner-gpu-bundle-3` that `w11b`, `w12a` and `w12b:<variant>`
  run from after the reset.
- The CPU probe (`silver_lexicon.py`): its smoke run on sofc (teachers of 1
  epoch on 32 sentences, 300 corpus sentences) exercised every part in 11
  minutes; its lexicons were empty, as such teachers' must be, but the
  ceilings stand: on week 10's CRF predictions of sofc (10 seeds), the gold
  unseen names as a dictionary gain +15.16 F1 where the model tagged O and
  +27.06 overriding its tags; all gold test names +13.52 and +25.31. The real
  probe started at about 17:40 UTC on two CPU lanes (sofc, pemfc,
  electrolysis, graphene; steel, poly, semi), 2 threads each; a subfield
  takes about 3 hours (3 teachers of 15–25 minutes, 40,000 sentences tagged
  in about 40 minutes per teacher).
- The TPU port (66f4ef3, 068ad97, ca69981): `--device xla` with fixed-shape
  batching; on XLA over the CPU (torch 2.9, torch_xla 2.9, PJRT_DEVICE=CPU)
  a tiny-model CRF run repeats exactly and gives the CPU path's scores; the
  CPU path itself repeats the pre-port files byte for byte apart from
  `runtime_sec` on every tiny-model option. TPU batches (`tpu-smoke1`: one
  job on one chip; `tpu-smoke`: the same job on all 8 chips) are meant for
  Kaggle's TPU v5e-8 from a bundle of ca69981; their results go to
  `results/tpu/`, a family of their own. Four script sessions of
  `tpu-smoke1` on 19 October (kernel `ner-tpu-jobs`, `enable_tpu` with the
  machine shape stored as TpuV5E8) ran on a plain CPU machine: Kaggle's
  Python image, no `/dev/accel*` or `/dev/vfio/*` device, torch_xla seeing
  one CPU device, and 0 of the 20 weekly TPU hours counted. Along the way
  the runner learned to survive a machine without nvidia-smi, the batch
  installs torch 2.9.0 (the CPU wheel) with torch_xla 2.9.0 and libtpu
  0.0.21.1 (the `tpu` extra pins a libtpu PyPI no longer has) and
  torchaudio and torchvision to match, and the record lists the
  accelerator devices and variables. A fifth session runs the same batch
  as a notebook kernel (`ner-tpu-nb`).
- Pull request #11 (weeks 6–12) merged into `main` as b5ca78e after its CI
  passed (568 tests; the CI first found seqeval missing from the test
  requirements, a batch test that assumed the SOFC corpus file and a parser
  test that amendment 1 had made obsolete); the branch restarted from
  `main`.
- Related work for the names route (`docs/METHODS_PLAN.md` §2, the
  draft's §8, `docs/references.bib`): a second search found the pieces
  older than the plan's first check said. Zafarian and Asghari (NSURL
  2019) harvested a gazetteer from unlabeled news with a supervised tagger
  and a confidence threshold, as a feature; Kozareva (EACL 2006) built
  gazetteers from unlabeled text by patterns and, bootstrapping, kept an
  example only when two classifiers agreed; Song et al. (FLAIRS 2020)
  found that replacing annotated entities with random gazetteer entries
  gave no significant gain on OntoNotes, where test names are not the
  bottleneck. The novelty statement is narrowed to the combination and the
  setting (an unseen topic's lexicon from its own text, by three teachers'
  agreement, injected into gold contexts, held-out topics of a materials
  corpus), the dictionary rule becomes a reported baseline and a
  gazetteer match feature a required one if names pass (§4C).
- The fifth TPU session, the notebook kernel `ner-tpu-nb` (19:14–20:48
  UTC, 92 minutes): the same machine as the script kernels (no
  `/dev/accel*`, torch_xla sees one CPU device, no TPU variables in the
  environment); with PJRT_DEVICE=TPU the job spent 77 minutes in libtpu's
  metadata lookups (every `tpu-env` variable DEADLINE_EXCEEDED) and failed
  at the first XLA call ("TPU initialization failed: no device found");
  the quota still shows 0.00 of 20 TPU hours used, so Kaggle attached no
  TPU to any kernel pushed through the API, notebook or script. The spike
  stays open on the website's side: open `ner-tpu-nb` in Kaggle's editor,
  choose TPU v5e-8 in the accelerator menu, save and run it once; if the
  record then lists accelerator devices, `tpu-smoke` follows.
- Prepared, in no protocol: the names as a gazetteer match feature
  (`--names-feature`, `ner_baseline.py`): the students of a `--teachers`
  run keep the plain CRF head and add to the emissions of every verbatim
  match of a harvested name a term learned from zero (two columns per
  type, first word and later words, learning rate 1e-2, no weight decay);
  the lexicon is the same as V2's (the mention filter at 0.9), the index
  the probe's (at most 8 words, the most frequent type), and the record
  counts the names and the matches (`names_feature`). Variant
  `crf_gzm_topic`; batches `w12c` (its 3-seed screen over two lanes, about
  3.2 GPU hours) and `w12b:gzm`; the week-12 report names it; the probe's
  index moved into the baseline module. The paths the change does not
  touch were checked byte-identical to 77c37be on the tiny corpora
  (`crf`, `mr`, `mt_mr`, `st3`, `mrs`, the MuLMS pool and unused-corpus
  runs). **Amendment 2, proposed, not adopted**: V7 = `crf_gzm_topic`
  joins the screen under the same selection rule, which the related work
  (§2 of the plan) makes the baseline a reviewer will ask for; the GPU
  budget after the reset would then be about 29 of 30 hours (`w11b` 2.1,
  `w12a` 14.6, `w12c` 3.2, `w12b` 7, the MaterioMiner analogue 2.1), so
  the analogue would move to the following week. It is the user's call;
  nothing runs before 24 October either way.
- The GPU quota was raised to 45 hours a week on 19 October. The order
  changes to use it: `w11b` (bundle 77c37be, no dependency on the probe) is
  pushed at once; `w12a` and `w12c` follow as soon as the probe's
  seven-subfield reading fixes the filter and threshold (a third amendment
  if it changes them) and the bundle is rebuilt; then `w12b:<winner>` and,
  if decision 1 passes, the MaterioMiner analogue, all this week (Kaggle
  charges session hours: `w11b`'s 3.4-hour session cost 3.4 of them, and
  about 11 are needed for the rest of week 12 against 12.7 left).
- Prepared, 20 October, before the screen: the MaterioMiner analogue
  batches `w12d:<variant>` and the week-12 comparisons of
  `ner_seedmeans.py` and `report_results.py` (above, under the methods
  criterion), with tests (575 in all), so that the analogue can run the
  moment decision 1 passes.
- *02:35, 20 October: the screen pushed*, both sessions from bundle 47e2310
  (ready since 02:02): `w12a` to `ner-gpu-jobs-d` (version 7; 16 jobs per
  lane: the smoke repeat `smokeao`, V1, V4, V2, V6 and the three `crf_mr6`
  jobs) and `w12c` to `ner-gpu-jobs-c` (version 3; 8 and 7 jobs: the
  smoke repeat `smokeaq`, V7 and V8 over both lanes). The container of this
  session was restarted at about 02:20, which killed the graphene probe
  (its first teacher was tagging) and the waiters; the probe restarted at
  02:34 on its own and lands about 05:30, after which the seven-subfield
  reading is written. The six subfields in hand fixed amendment 3 and kept
  the threshold, so the screen did not wait for graphene.
- *03:56–04:10: the screen's first sessions fail on the resume check.*
  `w12c` completed in 59 minutes with 4 of its 14 topic jobs done and 10
  failed in 0.01 minutes: a results file (`crf_gzm_topic_m<N>.json`, a run
  per subfield) stores one corpus SHA-256, and each held-out subfield has
  its own topic corpus, so the second subfield's job found a file "from
  another corpus" and refused to resume it (the overwrite guard of week 3,
  right for every earlier protocol, where a file's runs share a corpus).
  `w12a`, still running, loses every names job but the first of each
  variant the same way; its `crf_mr6` jobs (no corpus) run. The fix
  (`ner_baseline.py`): with `--subfields`, a run records its own corpus
  (`self_train_sha256` per run); a file whose corpus differs keeps the runs
  of other subfields, resumes this subfield's run only from the same corpus
  and refuses another corpus for it as before; files of the other protocols
  are byte-identical to 77c37be (the tiny identity check, 8 configurations)
  and a tiny end-to-end run shows two subfields extending one file, a
  repeat resuming and another corpus refused. `w12c` also put each feature
  variant's jobs in both lanes, which would have made the lanes write the
  same files at once; a variant per lane now (`w12a`'s rule). `w12e` is
  `w12a` without the `crf_mr6` jobs. Both are pushed again from a bundle
  built on the fix. The probe's container was restarted three times in the
  night (02:20, 03:10, 04:00), killing the graphene probe each time; the
  probe now caches each teacher's votes locally and resumes from them.
- *04:20: the bundle rebuilt* from 3ac85a8 (the fix; 413 GPU results files,
  the 7 topic corpora, the 52 MuLMS files) as the new version of
  `ner-gpu-bundle-3`; `w12c` is pushed to `ner-gpu-jobs-c` 30 minutes
  later, `w12e` to `ner-gpu-jobs-d` once the first `w12a` session there
  completes and is pulled.
- *04:49: `w12c` pushed again* to `ner-gpu-jobs-c` (version 4) from bundle
  3ac85a8, V7's 7 jobs in lane 0 after the smoke repeat and V8's in lane 1.
- *05:03: the first `w12a` session done* (137.6 minutes, 2.3 quota hours):
  the smoke repeat, electrolysis for V1, V2, V4 and V6 (29–38 minutes
  each) and the three `crf_mr6` jobs (31–37 minutes each; valid, kept);
  pulled, and `w12e` pushed to `ner-gpu-jobs-d` (version 8) before the
  quota was checked against the real job times: a 3-student topic job
  takes about 30 minutes on a T4, a V6 job 38, so `w12e` needs about 8.1
  hours and `w12c` 3.1, against 9.14 hours left in the week with both
  sessions charged at once; Kaggle stops a session when the quota runs
  out and keeps none of its outputs. The user is asked to cancel version 8
  on the website; `w12f` (V1 in lane 0, V2 in lane 1, about 3.7 hours) is
  pushed in its place and `w12g` (V4 and V6) runs after the reset of 24
  October with the decision batch.
- *08:06: the feature screen done* (`w12c` on `ner-gpu-jobs-c`, version 4,
  bundle 3ac85a8; 196.7 minutes, setup 26.8 of them; all 15 jobs, 21–28
  minutes each). The repeat `smokeaq` is identical (77.95), so is `smokeao`
  of the first `w12a` session; every teacher of every job scores exactly
  as the stored week-10 CRF run of its seed, and every run records its own
  corpus. Collected: `crf_gzm_topic_m<0,100,200>`, `crf_gzs_topic_m<N>` and
  the three `crf_mr6_m<N>` files of the first `w12a` session (complete, 7
  subfields each). The screen so far (`ner_seed_analysis.py`, `week12`,
  3 seeds, 7 subfields; the decision rule is not applied to a screen):
  V7 `crf_gzm_topic` 48.91, −0.03 [−2.02, +1.97] against the CRF head and
  −1.73 [−4.13, +0.67] against mention replacement; V8 `crf_gzs_topic`
  48.79, −0.15 [−2.15, +1.86] and −1.85 [−3.59, −0.12]; V3 `crf_mr6`
  50.44, +1.50 [−1.65, +4.66] and −0.20 [−3.20, +2.80]. None is above
  mention replacement, so none of the three can be the screen's winner;
  V1, V2, V4 and V6 run after the reset. Per subfield (3-seed means):

| Subfield | CRF head | MR | V3 `crf_mr6` (Δ vs CRF, Δ vs MR) | V7 `crf_gzm_topic` | V8 `crf_gzs_topic` |
|---|---|---|---|---|---|
| electrolysis | 56.20 | 57.13 | 59.17 (+2.97, +2.04) | 57.87 (+1.66, +0.73) | 55.86 (-0.35, -1.28) |
| graphene | 49.20 | 50.83 | 51.46 (+2.25, +0.63) | 47.34 (-1.86, -3.49) | 49.40 (+0.20, -1.43) |
| pemfc | 50.37 | 52.39 | 54.16 (+3.79, +1.76) | 49.36 (-1.01, -3.03) | 49.77 (-0.60, -2.63) |
| poly | 53.47 | 49.48 | 50.02 (-3.45, +0.54) | 50.82 (-2.65, +1.34) | 49.91 (-3.56, +0.43) |
| semi | 31.52 | 37.59 | 30.25 (-1.27, -7.35) | 31.71 (+0.19, -5.89) | 32.04 (+0.52, -5.55) |
| sofc | 44.15 | 49.58 | 50.77 (+6.61, +1.19) | 47.83 (+3.68, -1.75) | 47.88 (+3.73, -1.70) |
| steel | 57.61 | 57.45 | 57.24 (-0.37, -0.21) | 57.43 (-0.19, -0.03) | 56.65 (-0.96, -0.80) |
| mean | 48.93 | 50.64 | 50.44 (+1.50, -0.20) | 48.91 (-0.03, -1.73) | 48.79 (-0.15, -1.85) |

  The feature's learned term stayed small: the largest absolute weight per
  run is 0.41–1.09 (mention lexicon) and 0.46–0.95 (sentence lexicon)
  after 20 epochs at learning rate 1e-2, against emissions of several
  units, with 850–8,660 (mention) and 356–3,668 (sentence) matched test
  words per subfield; the model learned to trust the lexicon little, and
  where the dictionary rule gained most (sofc, pemfc, semi) the feature
  gained nothing beyond the CRF head's seed noise. The quota at 08:10 is
  2.9 hours, which the running `w12e` session consumes until Kaggle stops
  it; no batch fits before the reset, so cancelling it no longer changes
  the plan: `w12e` runs after the reset of 24 October (about 8 hours),
  then `w12b:<winner>` if a names variant beats mention replacement.
- *13:23: `w12e` done after all* (`ner-gpu-jobs-d` version 8, bundle
  3ac85a8; 440 minutes, 28 jobs of 22–42 minutes). Kaggle did not stop the
  session at the quota: its line read 43.99 hours used at 10:10, 45.25 at
  11:25 with the session still running, and 46.23 of 45 at 13:23 with the
  session complete. A running session finishes past the week's limit and
  the overrun is charged; whether it comes off the next week shows at the
  reset. The repeat `smokeas` is identical (77.95); every teacher of every
  job equals the stored week-10 CRF run of its seed (`check_w12e.py`); all
  12 files hold the 7 subfields; collected. The screen is complete: V4
  `crf_mrm_topic6` is the winner (outcome under the week-12 protocol);
  `w12b:mrm6` and `w13a` go together at the reset, `w12d:mrm6` after
  `w12b` if decision 1 passes.
- *21:06: `w11b` pushed* to `ner-gpu-jobs-d` (version 6) from bundle
  77c37be, 5 jobs in lane 0 (the smoke4 repeat `smokean` and the 4
  fine-grained held-out papers) and 4 in lane 1 (the coarse ones), about
  2.1 GPU hours; its outputs go through `$SP/post_w11b.sh` for decisions 3
  and 4 of week 11.
- *22:41: the sixth TPU session*, a version of `ner-tpu-nb` committed from
  Kaggle's editor with "TPU VM v5e-8" chosen in the session options
  (21:12–22:41), ran on the same CPU-only image as the five API-pushed
  ones (`gcr.io/kaggle-images/python`, no `/dev/accel*`, no TPU variables,
  torch_xla's one CPU device; the job failed after the same 77 minutes of
  libtpu metadata timeouts), and Kaggle charged no TPU time for it (0.00 of
  20 hours after six sessions). The editor's draft session with the same
  setting stayed at "starting". The spike's cap is spent: the TPU is dropped
  from the schedule, and the ablations run on the GPU, whose quota is now
  45 hours a week; the XLA port stays in the code, tested on XLA-CPU, for a
  TPU that Kaggle may attach another day.

### Week 12, the CPU probe (19–20 October; not a protocol): what the silver filter admits, and the dictionary ceiling

`silver_lexicon.py` (`docs/METHODS_PLAN.md` §4B), per held-out subfield: the
fold's 3 CRF teachers trained here on the CPU (their numbers differ from
the GPU's), the subfield's topic corpus tagged, the lexicon of each filter
at thresholds 0.9, 0.7 and 0.5, its coverage of the subfield's unseen test
names (phase 0 measured what the corpus holds; this is the pool a student
would see), and the dictionary rule on week 10's saved CRF predictions (10
seeds): every verbatim match of a lexicon name tagged as its entity where
the model tagged O (fill) or whatever it tagged (override); the gold names
as the leaky ceilings. `results/silver_lexicons/<subfield>.json`; the names
stay local.

| Subfield | Teachers' test F1 (CPU) | Sentence filter 0.9: names, coverage, fill | Mention filter 0.9: names, coverage, fill | Mention filter 0.7: coverage, fill | Gold unseen names: fill / override |
|---|---|---|---|---|---|
| electrolysis | 56.3, 50.2, 57.2 | 300, 12.5%, +0.85 | 7,895, 29.9%, −0.77 | 35.5%, −3.78 | −6.85 / +4.47 |
| graphene | 46.7, 49.3, 49.0 | 1,221, 15.3%, −2.86 | 14,322, 28.8%, −5.07 | 32.8%, −6.51 | +9.11 / +24.32 |
| pemfc | 44.4, 51.1, 52.9 | 300, 19.2%, +3.16 | 6,259, 29.4%, +3.97 | 37.6%, −0.44 | −10.51 / −5.91 |
| poly | 45.6, 52.5, 53.8 | 854, 8.0%, −0.25 | 9,194, 18.3%, −3.31 | 23.4%, −9.91 | −3.63 / +2.22 |
| semi | 32.8, 35.0, 34.3 | 379, 4.6%, +1.20 | 5,713, 9.4%, +2.27 | 20.4%, −0.76 | +17.29 / +22.00 |
| sofc | 53.4, 51.9, 51.2 | 340, 25.8%, +5.79 | 4,344, 34.4%, +5.91 | 45.8%, +5.96 | +15.16 / +27.06 |
| steel | 55.2, 58.2, 58.6 | 667, 13.3%, +0.01 | 7,913, 25.0%, −10.03 | 32.5%, −13.18 | −1.82 / +4.13 |

*Reading, steel.* The corpus holds 55.0% of steel's unseen test names; the
mention filter at 0.9 admits 25.0% of them (7,913 names, 14,280 MAT and
11,224 VALUE mentions among 29,420), the sentence filter 13.3% (week 11's
cross-subfield names held 4.4%). SAMPLE and DEV are absent from every
lexicon (0 and 1 mentions at 0.9): the teachers do not find them in the
corpus either. As a dictionary the lexicon is harmful: filling the CRF
head's O tags with the 0.9 mention lexicon moves precision from 64.4 to
40.3 and recall from 50.9 to 55.9 (−10.0 F1), and even the gold unseen
names fill to 42.7 precision and 77.1 recall (−1.8; overriding, +4.1). A
string that is an entity somewhere in steel's documents is usually not one
where it occurs next, so no gazetteer can help there and a names method
must keep the context's say; whether mention replacement, which swaps
names only inside gold entity contexts, raises recall without that
precision loss is what the screen measures (its precision and recall will
be reported with it).

*Reading, sofc.* The opposite case. The corpus holds 80.8% of sofc's
unseen test names; the mention filter at 0.9 admits 34.4% (4,344 names;
15,553 MAT and 7,093 VALUE mentions among 24,460), the sentence filter
25.8% (340 names). The held-out CRF head is recall-starved there (10
seeds: precision 57.5, recall 35.3, F1 43.5), and the admitted names
recur as entities: filling its O tags with the 0.9 mention lexicon moves
recall to 46.5 for a precision of 52.8 (+5.9 F1, positive on all 10 seeds,
+1.9 to +9.1); the 0.9 sentence lexicon gives the same gain at less
precision cost (56.7 / 43.7, +5.8), the 0.7 sentence lexicon the largest
(+7.2 fill, +7.3 override), and only the 0.5 mention lexicon tips over
(+1.2 fill, −2.1 override). The gold unseen names fill to +15.2 and
override to +27.1. *So far.* Two subfields bracket the question: a lexicon
of names the teachers agree on is a strong gazetteer on sofc and poison on
steel, so the gain of any names method will vary by subfield as much as
the screen can detect, and on sofc the dictionary rule (+5.9 at the
protocol's filters) is the cheap baseline the week-12 variants must beat;
it will be reported next to them. Whether the protocol's threshold (0.9)
or filter should change is decided after the seven subfields are in.

*Reading, pemfc.* A third pattern. The corpus holds 73.9% of pemfc's
unseen test names; the mention filter at 0.9 admits 29.4% (6,259 names;
9,604 MAT and 10,248 VALUE mentions among 20,605), the sentence filter
19.2% (300 names). The held-out CRF head (10 seeds: precision 60.5, recall
42.9, F1 50.0) gains from the 0.9 lexicons as a fill (+4.0 mention, +3.2
sentence, on all 10 seeds) and loses from everything looser (mention 0.7
−0.4, 0.5 −4.1; every override negative), and loses most from the gold
names themselves (the unseen gold names fill to −10.5, override to −5.9):
in pemfc's documents the same string is an entity in one sentence and not
in the next, so the complete name list is poison while the teachers'
unanimous, confident names, the strings that are entities wherever they
occur, make a usable gazetteer. *So far (3 of 7).* The threshold 0.9 is
the right end of its range on every subfield seen (0.7 helps only on
sofc); the agreement filter selects what a gold list cannot; the two
filters are not yet separable (the mention filter at 0.9 is at least as
good as the sentence filter on sofc and pemfc, both are harmful on steel).

*Reading, poly.* Steel's pattern, with less of the names. The corpus holds
65.9% of poly's unseen test names, the lowest share but steel's; the
mention filter at 0.9 admits 18.3% (9,194 names; 17,217 MAT and
14,024 VALUE mentions among 32,236), the sentence filter 8.0%
(854 names). Every dictionary hurts the held-out CRF head (10 seeds:
precision 55.9, recall 48.7, F1 52.0): the 0.9 mention lexicon fills to
precision 47.4 and recall 50.1 (-3.3; 0 of 10 seeds up), the 0.9 sentence
lexicon -0.2, every looser lexicon and every override worse, and the gold
unseen names fill to -3.6 (override +2.2). *So far (4 of 7).* Two
subfields where a 0.9 lexicon helps as a dictionary (sofc +5.9, pemfc
+4.0), two where it hurts (steel −10.0, poly −3.3); on all four, 0.9 beats
0.7 and 0.5, and the sentence filter loses less where the lexicon hurts
and gains about the same where it helps.

*Reading, electrolysis.* Near neutral. The corpus holds 70.7% of
electrolysis's unseen test names; the mention filter at 0.9 admits 29.9%
(7,895 names; 27,267 MAT and 9,061 VALUE mentions among 36,756), the
sentence filter 12.5% (300 names). The held-out CRF head is the best
balanced of the seven (10 seeds: precision 57.3, recall 54.8, F1 55.9),
and a dictionary has little to add: the 0.9 sentence lexicon fills to
+0.9 (on all 10 seeds, +0.2 to +1.5), the 0.9 mention lexicon to −0.8 (on
none), every looser lexicon and every override worse (the 0.7 mention
lexicon −3.8); the gold unseen names fill to −6.9 and override to +4.5.
*So far (5 of 7).* The dictionary's value tracks the head's recall gap:
it helps where recall trails precision by 15 points or more (sofc 35
against 58, pemfc 43 against 61) and hurts or does nothing where the head
is nearly balanced (steel, poly, electrolysis); 0.9 beats 0.7 and 0.5
everywhere; as a dictionary the sentence filter is the safer (never below
−0.3 at 0.9) and the mention filter the richer (18–34% of the unseen
names against 8–26%).

*Reading, semi.* The weakest head and the thinnest lexicon. The corpus
holds 69.3% of semi's unseen test names, but the mention filter at 0.9
admits 9.4% (5,713 names; 7,783 of its 10,145 mentions are VALUE, 863
MAT) and the sentence filter 4.6% (379 names): the teachers, at 33–35
test F1, seldom agree confidently on semi's materials, so here the
teachers' recall, not the corpus, is the bottleneck. The held-out CRF head
(10 seeds: precision 54.0, recall 23.5, F1 32.5) gains from whatever is
admitted: the 0.9 mention lexicon fills to +2.3 (on all 10 seeds), the 0.9
sentence lexicon to +1.2 (all 10), the 0.7 sentence lexicon to +3.6 (all
10); the 0.7 and 0.5 mention lexicons and every override but the sentence
ones at 0.9 and 0.7 lose. The gold unseen names fill to +17.3 and
override to +22.0, the largest ceilings of the seven. *So far (6 of 7).*
The recall-gap rule holds (semi's gap is 30 points and the dictionary
helps); the 0.9 threshold holds for the mention filter everywhere, while
the sentence filter at 0.7 gains more than at 0.9 on sofc (+7.2), pemfc
(+2.4 against +3.2, less), semi (+3.6) and loses on poly (−2.5) and
electrolysis (−0.3): the sentence filter admits so few names that
loosening it costs little where the lexicon helps.

*Reading, graphene.* Every dictionary hurts, the gold one helps most. The
corpus holds 64.0% of graphene's unseen test names; the mention filter at
0.9 admits 28.8% (14,322 names; 34,118 MAT and 16,320 VALUE mentions
among 56,421, the largest lexicon of the seven), the sentence filter
15.3% (1,221 names). The held-out CRF head (10 seeds: precision 56.8,
recall 44.9, F1 50.0) loses with every silver lexicon on every seed (the
0.9 sentence lexicon −2.9, the 0.9 mention lexicon −5.1, looser ones and
overrides more), while the gold unseen names fill to +9.1 and override to
+24.3: graphene's names are in the corpus and the teachers find them, but
the strings they agree on (graphene itself, GO, its oxides and values)
occur as often outside entities as inside them in the test documents.

**The seven subfields** (the probe's reading, 20 October, 07:00 UTC;
`results/silver_lexicons/*.json`):
- *What the filters admit.* Of the unseen test names, which the corpora
  hold at 55–81%, the mention filter at 0.9 admits 9–34% (semi 9.4,
  poly 18.3, steel 25.0, graphene 28.8, pemfc 29.4, electrolysis 29.9,
  sofc 34.4) and the sentence filter 5–26% (semi 4.6, poly 8.0,
  electrolysis 12.5, steel 13.3, graphene 15.3, pemfc 19.2, sofc 25.8);
  week 11's cross-subfield names held 4.4%. On semi the teachers' recall,
  not the corpus, is the bottleneck.
- *As a dictionary, the lexicon's value tracks the held-out head's
  recall gap.* Filling the CRF head's O tags from the 0.9 mention
  lexicon gains where recall trails precision by 15 points or more (sofc
  +5.9, pemfc +4.0, semi +2.3; gaps of 22, 18 and 30) and loses where
  the head is nearer balance (steel −10.0, poly −3.3, graphene −5.1,
  electrolysis −0.8; gaps of 8, 7, 12 and 2). Over the seven subfields
  the 0.9 mention lexicon averages −1.0 as a fill and the 0.9 sentence
  lexicon +1.1 (never below −2.9, a gain on five); these are the
  dictionary baselines the week-12 variants must beat, reported per
  subfield beside them.
- *The threshold.* 0.9 is the right end of the range for both filters:
  at 0.7 and 0.5 every mention lexicon loses on five or more subfields
  and every override loses on most; the sentence filter at 0.7 gains
  more than at 0.9 only on sofc and semi and loses on poly, steel,
  graphene and electrolysis. No amendment of the threshold.
- *The agreement filter selects what a gold list cannot.* On pemfc the
  gold unseen names fill to −10.5 while the teachers' 0.9 names fill to
  +4.0: the names three teachers agree on at 0.9 are the strings that are
  entities wherever they occur, which a complete name list is not. On
  graphene even that selection fails (−5.1 against +9.1 for the gold
  names), on steel both fail.
- *What this says for the screen.* The lexicon alone can decide an
  unseen subfield for the worse as easily as for the better, so a names
  method must let the model weigh it: that is amendment 2's and 3's
  feature (V7, V8), with mention replacement (V1, V2, V4) keeping the
  context's say by construction. If no variant passes decision 1, the
  probe's table is the paper's explanation of why: the recall gap, the
  admission rates and the precision of a harvested lexicon vary by
  subfield more than any one injection can absorb.

### Week 13 protocol: the methods push (fixed before any run, 20 October)

*Directive.* On 20 October the user asked to push as hard as possible for
a methods paper. The plan's stop rule ("no fourth protocol on MuLMS",
`docs/METHODS_PLAN.md` §6) is lifted by that directive; this is the fourth
protocol that asks a method to improve on the CRF head on MuLMS's held-out
subfields (weeks 10, 11, 12 and 13), and the paper reports the count, as
before. Nothing already decided is re-decided; every analysis below is
fixed here before its numbers exist, and what is post hoc is labelled so.

*The methods criterion, revised.* A methods claim scoped to "unseen
documents" needs either a pre-registered pass against the CRF head on
held-out documents of two corpora (the per-corpus decisions as fixed in
their weeks), or a pre-registered pooled decision over the held-out units
of two or more corpora (13.2) with no corpus whose own interval lies below
0. The claim is scoped by what passed (the corpora, the granularity, the
mechanism); comparisons that did not pass are reported next to it. The
decision rule is the one of every week: the 95% Student-t interval of the
mean paired difference of micro F1 lies above 0, each unit the mean of 10
model seeds.

**13.1 The document consistency rule on MuLMS (CPU, no training).** The
rule of week 7, part 1 (`ner_consistency.py`, document majority
propagation without retyping; retyping reported), applied to the saved
test predictions of the CRF head (`crf_m<N>`) and of mention replacement
(`crf_mr_m<N>`), N = 0 to 900, on MuLMS's held-out subfields; a document
is a MuLMS CoNLL file. Decisions, paired over the 7 subfields: 13.1a
`crf_dc` against `crf`; 13.1b `crf_mr_dc` against `crf_mr`. Reported: the
retyping variants, the votes, per subfield. On MaterioMiner the rule gave
−0.13 to +0.14 (week 7); MuLMS's documents are whole articles with more
repeated mentions and more unseen names, so the rule has more to
propagate, but also more wrong predictions to propagate. Expected small.

*Outcome of 13.1a and 13.1b (20 October, 09:50 UTC).* 13.1a passes:
`crf_dc` against `crf`, +0.82 [+0.18, +1.46] (49.47 against 48.66; every
subfield gains: electrolysis +0.54, graphene +0.47, pemfc +1.35, poly
+0.07, semi +0.93, sofc +2.07, steel +0.30; 1,213 entities added per seed
over the 7 subfields). 13.1b does not: `crf_mr_dc` against `crf_mr`,
+0.57 [−0.12, +1.25] (51.32 against 50.76; six subfields gain, steel
−0.04). Reported: with retyping, +1.01 [+0.28, +1.74] on the CRF head and
+0.88 [−0.01, +1.77] on mention replacement. The rule's earlier results:
on MaterioMiner's held-out papers (week 7, part 1) it passed only the
coarse-grained CRF head, +0.12 [+0.01, +0.24] (fine-grained −0.13 to
+0.14); on SOFC-Exp's official split it hurts, −0.42 [−0.61, −0.22],
because SOFC-Exp's annotators leave 14% of a tagged term's occurrences
untagged.

**Amendment, after 13.1a and 13.1b and before any further number (20
October, 09:55 UTC).** Two more applications of the same rule to saved
predictions, neither seen:
- 13.1c, *reported, not a decision*: `crf_dc` and `crf_dcr` against `crf`
  on SOFC-Exp's 5 article folds (week 8's CRF head, 10 seeds, paired over
  the folds). The official split says the rule hurts there, and the folds
  carry the same annotation convention, so a loss is expected; it is
  reported as the rule's boundary condition.
- 13.1d, two decisions: `crf_mr_dc` against `crf_mr` on MaterioMiner's 4
  held-out papers, fine-grained and coarse-grained (week 7's mention
  replacement runs, 10 seeds, paired over the papers): does the rule add
  to mention replacement where the rule alone added +0.12 coarse-grained?
  Reported: `crf_mt_mr_dc` against `crf_mt_mr`; retyping.
What the rule's results would support: a decode-time method with a
boundary condition, "document-level propagation of a model's own
predictions improves NER on held-out documents of consistently annotated
corpora" (MuLMS +0.82, MaterioMiner coarse +0.12), which by the revised
criterion is a methods claim of two corpora, scoped as such; its combination
with mention replacement is reported against the CRF head on both corpora.

*Outcome of 13.1c and 13.1d (20 October, 10:40 UTC;
`ner_seed_analysis.py`, section `week13`, `consistency`).* 13.1c, reported:
on SOFC-Exp's 5 article folds the rule loses, −0.44 [−0.72, −0.16] (82.34
against 82.79; every fold loses, −0.17 to −0.75; 169 entities added per
seed over the 5 folds) and −0.43 [−0.68, −0.18] with retyping: the
boundary condition holds on held-out articles as on the official split.
13.1d: neither decision passes. Fine-grained, `crf_mr_dc` against `crf_mr`
+0.19 [−0.34, +0.73] (49.42 against 49.23; three papers gain, the fourth
loses 0.26; 54 entities added per seed over the 4 papers); coarse-grained
+0.11 [−0.08, +0.31] (60.00 against 59.89; every paper gains, +0.01 to
+0.27; 16 per seed). Reported: on the multi-task combination +0.31 [−0.47,
+1.09] fine-grained and +0.10 [−0.05, +0.25] coarse-grained; with retyping
+0.22 [−0.69, +1.12] and +0.63 [+0.13, +1.14] on mention replacement, +0.18
[−1.11, +1.47] and +0.60 [−0.21, +1.41] on the combination. Week 7's
figures recomputed from the same files agree (`crf_dc` against `crf` +0.30
[−0.35, +0.95] fine-grained and +0.12 [+0.01, +0.24] coarse-grained;
retyping +0.64 [+0.29, +1.00] coarse-grained).

*What 13.1 supports.* On the CRF head the rule passes two pre-registered
decisions on held-out documents of two corpora (MuLMS +0.82, MaterioMiner
coarse-grained +0.12) and loses on the third (SOFC-Exp: −0.42 on the
official split, −0.44 on the folds), so by the revised criterion
"document-level propagation of a model's own predictions improves NER on
held-out documents of corpora whose annotators tag repeated mentions
consistently" is a methods claim of two corpora, scoped as such, with its
boundary condition measured on the third. It does not add to mention
replacement on either corpus (13.1b, 13.1d). Retyping, never a decision, is
the larger effect wherever the rule helps (+1.01 on MuLMS; +0.64 and +0.63
coarse-grained on MaterioMiner, on the CRF head and on mention replacement)
and cannot be decided on a corpus that has not seen it: all three have. Of
the rule's 10 pre-registered decisions (6 in week 7, 4 here) 2 pass, about
0.25 expected by chance, on two corpora. The effect is under a point, so
the paper reports it as a decode-time rule with a measured boundary
condition beside mention replacement (13.2, 13.3), not as its main method.

**13.2 The pooled decision for the week-12 winner.** If the week-12 screen
names a winner (the largest mean gain over mention replacement, above 0,
among V1–V8 and V3), its decision batch `w12b:<winner>` (MuLMS, seeds
300–900) and its MaterioMiner analogue `w12d:<variant>` (fine-grained, 4
held-out papers, seeds 0–900; V1's analogue is `w11b`, V6's the week-4
`crf_st3` files) give 11 held-out units. Decision 13.2: the mean paired
difference against the CRF head over the 11 units (each unit its 10-seed
mean), 95% Student-t interval (10 degrees of freedom) above 0. The week-12
decisions 1 and 2 and the analogue's own decision stand as fixed and are
reported. *Post hoc, labelled as such:* the same pooled estimate for plain
mention replacement from the existing runs, +2.34 [+0.94, +3.75]
fine-grained (MaterioMiner's 4 papers +3.47, +3.24, +1.69, +2.69 and
MuLMS's 7 subfields +1.42, +1.66, +2.28, −0.73, +2.52, +7.39, +0.15) and
+1.80 [+0.39, +3.22] with the coarse-grained papers; both corpora's
results were known when it was computed (20 October), so it is a
secondary estimate, never a decision.

**13.3 SOFC-Exp's article folds (`w13a`).** The third corpus with held-out
documents: the 5 folds of 9 articles of week 8 (`--document-folds 5`),
10 model seeds (offsets 0 to 900), two methods: mention replacement
(`crf_mr`, entity level, `--augment-mentions 3`) and multi-task mention
replacement (`crf_mt_mr`, `--granularity sofc_both --multi-task`, the slot
and entity heads). Decisions, paired over the 5 folds against week 8's
CRF head on the folds (`sofc_document_folds_crf_m<N>`): 13.3a `crf_mr`
against `crf` on the entity level; 13.3b `crf_mt_mr` against `crf` on the
entity level. Reported: the slot level of `crf_mt_mr`; `crf_mt_mr` against
`crf_mr`; the pooled estimates over three corpora (mention replacement:
16 units; multi-task mention replacement: MaterioMiner fine-grained and
the SOFC-Exp entity level, 9 units). What a pass would mean: multi-task
mention replacement, the strongest MaterioMiner system (+6.73 [+6.03,
+7.43] fine-grained), improves on held-out documents of a second two-level
corpus; what the control says: on SOFC-Exp's official split, where test
names are mostly seen, neither method gains (weeks 7 and 9), and the
article folds hold out whole articles (the gap to random splits is 3.48
points, against 17 on MuLMS), so the expected gain is small and the folds
are a harder test than MaterioMiner's papers. Cost: about 4.5 lane hours
for `crf_mr` and 7.5 for `crf_mt_mr` (one session of both lanes), after
the reset of 24 October, beside `w12e`.

*Order after the reset.* `w12e` (kernel d) and `w13a` (kernel c) together;
then 13.1's results, already in hand, and the screen's selection; then
`w12b:<winner>` and `w12d:<variant>` if there is a winner; the pooled
decisions last. All analyses by `ner_seed_analysis.py` (`week13`
section) from the committed files.

### Week 12 protocol: names from a topic corpus (fixed before any run)

Fixed on 18 October, after phase 0's gate above and before any run of week
12. It changes the order of "What follows week 11": the gate of the topic
corpora is phase 0's coverage, measured on the CPU, not `w11b`, which cannot
run before the GPU reset on 24 October; `w11b` still runs first, and its
decisions 3 and 4 stand as the week-11 protocol fixed them.

*Why.* Mention replacement is the one method that improves on the CRF head
on unseen MaterioMiner papers (C20), and on MuLMS's held-out subfields it
gains +2.10 [−0.31, +4.51] without passing (C22). Its silver-name variant
added nothing there (C23) because the names came from the other subfields
and held 4.4% of a held-out subfield's unseen test names: the cross-subfield
sentences are few (7,400–9,433 per fold) and the sentence-level filter admits
181–419 of them. Phase 0 built a corpus of each subfield's own topic and
measured that it holds 55–81% of the held-out subfield's unseen test names
(64–84% ignoring word boundaries), 77–90% of the unseen MAT names and 85–96%
of the FORM names, the types with the largest held-out gaps. Two things are
untested: whether the silver filter admits those names, and whether names in
the pool turn into F1 on the held-out documents. The teachers never see the
held-out subfield's labeled sentences, and the corpus excludes MuLMS's
identified articles and every sentence that shares an 8-word n-gram or a
50-character run with a MuLMS sentence (phase 0), so the setting stays
"unseen documents of a new topic", with unlabeled text of that topic, as the
fatigue corpus is for MaterioMiner.

*Method* (`--augment-names`, as week 11, with these changes):
- **Unlabeled text:** the held-out subfield's topic corpus,
  `data/processed/corpus/topic_<subfield>_sentences.txt` (40,000 sentences,
  `results/topic_corpora/<subfield>.json`; SHA-256 checked in the bundle), all
  of it tagged (`--self-train-sample 0`). The variant tag is `topic`
  (`--names-tag topic`), so the runs are `crf_mrs_topic_m<N>`.
- **Three variants**, the screen's candidates:
  - V1 `crf_mrs_topic`: week 11's silver filter. A sentence is silver when
    the 3 teachers give it the same tags, every tag has a CRF marginal of at
    least 0.9 under each teacher, and it has an entity; its mentions join the
    pool.
  - V2 `crf_mrm_topic` (`--silver-mentions`): the mention-level filter. A
    mention joins the pool when the 3 teachers tag the same span with the same
    type and every tag of that span has a marginal of at least 0.9 under each
    teacher; the rest of its sentence need not be confident. Recorded as
    `silver_filter: mentions`.
  - V3 `crf_mr6` (`--augment-mentions 6`, no teachers): mention replacement
    with 6 copies of each sentence instead of 3. The control: if more
    copies alone gain as much, the names are not the mechanism.
- **Teachers and students:** as week 11. The 3 CRF teachers are the runs
  `crf_m0`, `crf_m100` and `crf_m200` of week 10 (their test scores must equal
  the stored ones; a check). Each student trains on the gold sentences and
  their copies, 3 copies with replacement probability 0.5, the copy seed equal
  to the model seed; without silver names a student equals the `crf_mr` run of
  its seed (tested). The silver sentences are not trained on.
- **Recorded per run:** the silver counts and the share of the unseen test
  entities that are silver names (as week 11), and the filter; no name is
  stored.

*Screen* (`w12a`; seed offsets 0, 100 and 200). Per variant and held-out
subfield, the mean of its 3 seeds; the screen statistic is the mean over the 7
subfields of the variant minus week 10's `crf_mr` at the same seeds (its 95%
CI is reported; the screen decides nothing). **Selection rule:** the variant
with the largest screen statistic goes to the decision if that statistic is
above 0. If none is, week 12 ends without a decision: no variant beats
mention replacement on the screen, and the methods route on MuLMS is
recorded as exhausted (the evaluation paper, as "What follows week 11"
states).

*Decision* (`w12b:<variant>`, the winner; seed offsets 300 to 900). The
screen's selection would bias a decision taken on the same seeds, so the
decision uses only seeds the screen did not see: 7 students per subfield,
and for both references the runs of the same seeds. The week-6 rule: the 95%
Student-t CI of the mean paired difference of micro F1 above 0, each pair a
held-out subfield (6 degrees of freedom), its value the mean over seeds 300
to 900.
1. The winner against the CRF head (week 10's `crf`).
2. The winner against mention replacement (week 10's `crf_mr`). A pass of
   decision 1 with a fail of decision 2 means the gain is mention
   replacement's, not the names'.
Reported, no decision: all 10 seeds (the screen's and the decision's) against
both references; the votes over the seed triples 300–500 and 600–800 (and
over the 3 triples with 10 seeds) against the CRF votes and mention
replacement's; macro and weighted F1; per subfield the counts of silver names
and the share of the unseen test names among them, against phase 0's
coverage (what the filter admits of what the corpus holds).

*The methods criterion* (week 10's): a methods claim scoped to "unseen
documents of a new topic" needs decision 1 on MuLMS plus a pass on
MaterioMiner against the CRF head. The MaterioMiner analogue of the winner
runs only if decision 1 passes: for V1 it is `w11b` (names from the fatigue
corpus with the sentence filter, decisions 3 and 4 of week 11); for V2 a
batch of the same shape with `--silver-mentions` (`crf_mrm`, 10 seeds, both
granularities); for V3 `crf_mr6` on the held-out papers. Each would be a
decision against week 5's CRF head, pre-registered when it is reached.
*(Prepared on 20 October, before the screen: the batches `w12d:<variant>`
for V2 `crf_mrm`, V4 `crf_mrm6`, V7 `crf_gzm` and V3 `crf_mr6` (one lane
per granularity, a job per held-out paper or per seed offset, the smoke
repeat `smokear`), V6's analogue being week 4's `crf_st3` files; the
decision against the CRF head per granularity and the reported
comparisons against mention replacement and the votes are
`ner_seedmeans.py --week 12` (`gpu/ner_week12_comparisons.json`), the
report's week-12 table and bullets.)*

*Attempts.* This is the third protocol that asks whether a method improves on
the CRF head on MuLMS's held-out subfields (weeks 10, 11 and 12), and the
screen chooses among three variants; the paper reports every attempt and the
count, and the decision seeds are disjoint from the screen's.

*What the design can detect.* With 7 subfields (6 degrees of freedom), the
paired differences of weeks 10 and 11 had between-subfield standard
deviations of 2.6 and 3.1 points, so decision 1 passes only for a mean gain
of about 2.4–2.9 points or more (the half-width of the 95% CI). A smaller
true gain cannot pass on MuLMS with this design, whatever the method; the
paper states this with the attempts.

*Budget* (T4; from `w11a`, 22 minutes per subfield and lane with 8,800
sentences tagged and 5 students): `w12a` about 4.6 hours in lane 0 (V1's 7
jobs, then two `crf_mr6` jobs) and 3.9 in lane 1 (V2's 7 jobs and one
`crf_mr6` job), about 8.5 GPU hours; `w12b` about 3.5 hours per lane (each
lane trains the teachers and 3 or 4 of the students), about 7 GPU hours, or
about 5 for `crf_mr6`. With `w11b` (2.1 hours), about 18 of the 30 weekly GPU
hours after the reset on 24 October. The jobs of a variant share a lane
because they extend the same files.

*Checks before any number is used:* the repeat of the smoke4 CRF run
(`smokeao`, `smokeap`) equals the stored run; each job's teachers equal week
10's `crf_m0`, `_m100` and `_m200` on their subfield (`check_w11.py`); the
bundle's corpora match the reports' SHA-256 (the runner checks them again);
every run records `silver_filter` and `names`. Done before the bundle: on a
tiny model and the tests' toy corpora, the committed code (40631ce) repeats
the week-11 code's runs byte for byte apart from `runtime_sec` for the CRF
head, mention replacement, multi-task mention replacement, self-training
and MuLMS's pool and held-out runs; the silver-name runs differ only by the
added `silver_filter` field.

**Outcome of the screen (20 October): V4 `crf_mrm_topic6` goes to the
decision.** The second part of the screen (`w12e`, `ner-gpu-jobs-d` version
8, bundle 3ac85a8: V1, V2, V4 and V6 over the 7 subfields, seeds 0–200;
checks as the first part: the repeat `smokeas` identical at 77.95, every
teacher equal to the stored week-10 CRF run of its seed, 7 subfields in
every file) completes the screen of seven variants (`ner_seed_analysis.py`,
`mulms`, `week12`; 3-seed means, the CI reported, no decision). Against
mention replacement (50.64): V4 `crf_mrm_topic6` 52.09, +1.45 [−1.35,
+4.25]; V2 `crf_mrm_topic` 51.60, +0.97 [−0.91, +2.84]; V1 `crf_mrs_topic`
51.44, +0.80 [−0.91, +2.51]; V6 `crf_st3_topic` 50.80, +0.16 [−2.31,
+2.63]; from the first part V3 −0.20, V7 −1.73 and V8 −1.85. Against the
CRF head (48.93): V4 +3.16 [−1.96, +8.28], V2 +2.67 [−1.25, +6.59], V1
+2.50 [−2.09, +7.09], V6 +1.87 [+0.20, +3.53]. By the selection rule V4 has
the largest screen statistic and it is above 0, so `w12b:mrm6` (its 7
students per subfield on seeds 300–900, both references at the same seeds)
takes decisions 1 and 2 after the reset of 24 October; `w12d:mrm6`
(`crf_mrm6`: the mention filter, 6 copies and the fatigue corpus on
MaterioMiner's held-out papers, both granularities, 10 seeds) runs if
decision 1 passes, and 13.2's pooled decision is V4's. Per subfield (3-seed
means):

| Subfield | CRF head | MR | V1 `crf_mrs_topic` (Δ vs CRF, Δ vs MR) | V2 `crf_mrm_topic` | V4 `crf_mrm_topic6` | V6 `crf_st3_topic` |
|---|---|---|---|---|---|---|
| electrolysis | 56.20 | 57.13 | 57.30 (+1.10, +0.17) | 55.71 (-0.49, -1.42) | 58.27 (+2.06, +1.13) | 58.71 (+2.50, +1.57) |
| graphene | 49.20 | 50.83 | 51.76 (+2.56, +0.93) | 51.79 (+2.59, +0.96) | 53.94 (+4.73, +3.11) | 52.38 (+3.17, +1.55) |
| pemfc | 50.37 | 52.39 | 53.28 (+2.91, +0.88) | 52.38 (+2.02, -0.01) | 52.15 (+1.79, -0.24) | 50.67 (+0.30, -1.73) |
| poly | 53.47 | 49.48 | 48.08 (-5.39, -1.40) | 51.33 (-2.14, +1.85) | 49.75 (-3.72, +0.27) | 54.44 (+0.97, +4.96) |
| semi | 31.52 | 37.59 | 38.34 (+6.82, +0.75) | 37.37 (+5.85, -0.22) | 37.91 (+6.39, +0.31) | 35.28 (+3.76, -2.32) |
| sofc | 44.15 | 49.58 | 54.13 (+9.98, +4.55) | 54.52 (+10.37, +4.94) | 57.00 (+12.85, +7.42) | 47.51 (+3.36, -2.07) |
| steel | 57.61 | 57.45 | 57.16 (-0.46, -0.30) | 58.12 (+0.51, +0.67) | 55.61 (-2.00, -1.84) | 56.61 (-1.00, -0.84) |
| mean | 48.93 | 50.64 | 51.44 (+2.50, +0.80) | 51.60 (+2.67, +0.97) | 52.09 (+3.16, +1.45) | 50.80 (+1.87, +0.16) |

What the screen says before the decision. The names move sofc most (+7.42
over mention replacement for V4; +4.55 and +4.94 for V1 and V2; the probe's
dictionary gained most there too) and graphene (+3.11), and lose on steel
(−1.84) and pemfc (−0.24) for V4; the between-subfield spread of V4's
differences (sd about 3.0 points) makes the 3-seed CI wide, and "What the
design can detect" above applies: decision 2 needs a mean gain near 2.4–2.9
on the 7 new seeds, more than the screen's 1.45, while decision 1's +3.16 is
above that bar on these seeds. The 6-copy control without names (V3, −0.20)
and the 3-copy mention filter (V2, +0.97) bracket V4: the copies alone add
nothing, the names with 3 copies about 1, with 6 copies about 1.5. Self-
training on the topic corpus (V6) stays at mention replacement's level
(+0.16) and is the one variant whose gain over the CRF head has a CI above
0 on the screen (+1.87 [+0.20, +3.53]; the screen decides nothing). The
mention filter admits 22.2% of the unseen test names, the sentence filter
11.4% (the corpus holds 55–81%), as the probe measured.

*Analysis.* `ner_seed_analysis.py` (`mulms`, `week12`: per variant the parts
`screen`, `decision` and `ten_seeds`, each present when its files exist),
`kaggle_jobs.py` batches `w12a` and `w12b:<variant>` (tested from the frozen
commit), `report_results.py` for the README bullet once the files exist.

**Amendment 1 (19 October, before any run; `docs/METHODS_PLAN.md` §4A).**
The screen gains two variants, with the same 3-seed design, the same
selection rule and the same decision for the winner:
- V4 `crf_mrm_topic6`: the mention-level filter with 6 copies of each
  sentence instead of 3 (`--augment-mentions 6`). With thousands of silver
  names and about 1,500 replacement slots per fold, 3 copies show most
  silver names never; 6 copies double the exposure, and V3 (6 copies
  without names) is its control.
- V6 `crf_st3_topic`: self-training on the topic corpus, week 4's method
  with 3 teachers (the students train on the gold sentences plus the silver
  sentences, weight 0.5), the baseline a names method must beat. If V6 wins
  the screen, the decision is V6's and the finding is "in-topic
  self-training", scoped as such.
The names of V4 and V6 and their parser rules are in `ner_baseline.py`
(`--names-tag` now also labels a self-training run's corpus); `w12a` puts
V1 and V4 in lane 0, V2 and V6 in lane 1, and the three `crf_mr6` jobs one
in lane 0 and two in lane 1 (about 7.5 hours per lane); `w12b:mrm6` and
`w12b:st3` are the decision batches of the new variants. The budget of the
week after the reset becomes about 26 of 30 GPU hours (`w11b` 2.1, `w12a`
14.6, `w12b` 7, the MaterioMiner analogue 2.1). Before the runs, the CPU
probe of `METHODS_PLAN.md` §4B measures what the silver filter admits and
the dictionary ceiling; if admission is the bottleneck, a second amendment
can change the threshold or the filter, still before any run.

**Amendment 2 (19 October, before any run; `docs/METHODS_PLAN.md` §4A).**
The screen gains a seventh variant, with the same 3-seed design, the same
selection rule and the same decisions for the winner:
- V7 `crf_gzm_topic`: the same harvested names as V2 (the mention-level
  filter at 0.9), injected not by mention replacement but as a gazetteer
  match feature of the plain CRF head (`--names-feature --silver-mentions`,
  no copies): every verbatim match of a name of the lexicon (the index of
  the CPU probe: at most 8 words, a name's most frequent type) adds to the
  emissions of its words a term learned from zero, two columns per type
  (first word, later words), learning rate 1e-2 without weight decay, the
  rest of the head as in `crf`; the record counts the names and the matches
  (`names_feature`). The related work of 19 October (`METHODS_PLAN.md` §2)
  makes the lexicon as a feature the baseline a reviewer will ask for, and
  the CPU probe shows the lexicon's value swinging by subfield (the
  dictionary rule −10.0 on steel, +5.9 on sofc), which a learned term can
  weigh and mention replacement cannot. If V7 wins the screen, the
  decision is V7's and the finding is "the harvested lexicon as a
  feature"; mention replacement with the lexicon (V1, V2, V4) is then its
  ablation.
V7's option, variant and parser rules are in `ner_baseline.py` (48f37d3);
its screen is the batch `w12c` (its 7 jobs over two lanes, about 3.2 GPU
hours, after `w11b` on the lane `w12a` leaves free), its decision batch
`w12b:gzm`. Kaggle charges session hours, not lane hours: from `w11a`'s
job times the screen costs about 5 session hours (`w12a`) plus 2 (`w12c`),
the decision about 2.5 and the MaterioMiner analogue about 1.1, within the
12.7 hours left this week after `w11b` (the quota was raised to 45 hours a
week on 19 October), so the whole of week 12 runs before the reset. The paths the change does not touch were checked
byte-identical to 77c37be on the tiny corpora (`crf`, `mr`, `mt_mr`, `st3`,
`mrs`, the MuLMS pool and unused-corpus runs), and the XLA path still
repeats. The bundle of every week-12 batch is rebuilt from the commit
that closes the CPU probe (a third amendment of the threshold or filter
if the probe calls for one), before the reset.

**Amendment 3 (20 October, before any run; on the CPU probe's first six
subfields, graphene pending).** The screen gains an eighth variant, V8
`crf_gzs_topic`: V7's match feature with the *sentence* filter's lexicon
(`--names-feature` without `--silver-mentions`). The probe prices the two
lexicons as dictionaries: at 0.9 the sentence filter's is the safer (fill
+1.8 on average over six subfields, never below −0.3, a gain on five), the
mention filter's the richer (18–34% of the unseen test names against
5–26%) but the more harmful where it harms (steel −10.0, poly −3.3; mean
−0.3). V7 learns how far to trust the richer lexicon; V8 is the same
mechanism on the safer one, for about 1.9 lane hours, so `w12c` carries
both (7 jobs per lane). The threshold stays 0.9 for both filters: at 0.7
and 0.5 every mention lexicon and every override lose on most subfields,
and the sentence filter's gains at 0.7 (sofc, semi) come with losses
elsewhere (poly, steel). Same 3-seed design, selection rule and decisions;
`w12b:gzs` is V8's decision batch, `w12d:gzs` its MaterioMiner analogue
is not prepared (the sentence filter's analogue on MaterioMiner is
`w11b`'s lexicon as a feature; it would be added if V8 won). If V7 or V8
wins, the finding is "the harvested lexicon as a feature", with the
filter named.

*Bundle.* Built from 40631ce on 18 October and uploaded as the new version
of `ner-gpu-bundle-3` (457 files: the code, the fatigue corpus, the 52
MuLMS files, the 7 topic corpora with their SHA-256 in `BUNDLE.json`, and
393 GPU results files); every upload succeeded at the first try. `w11b`,
`w12a` and `w12b:<variant>` were to run from it after the reset on 24 October;
amendment 2 replaces it by a bundle built after the probe: rebuilt from
47e2310 (amendments 1–3) on 20 October, ready at 02:02 UTC as the new version
of `ner-gpu-bundle-3` (413 GPU results files, the 7 topic corpora and the 52
MuLMS files with their SHA-256 in `BUNDLE.json`); every week-12 batch runs
from it.
