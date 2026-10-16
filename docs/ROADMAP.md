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

**Week 6 so far** (§4; GPU runs on the 4 held-out papers; stage 2 of
multi-task self-training and SOFC-Exp's S2 for document context are next):
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
| 6 | In progress: NER, P1 "Next, after week 5", towards a methods paper: methods for unseen papers (DeBERTa-v3-large, document context, multi-task self-training) screened with 3 seeds and decided with 10, and a second dataset (SOFC-Exp, 10 seeds); protocol below. So far (P1, "Week 6 so far"): DeBERTa-v3-large stops at the screen; document context passes the screen but neither 10-seed decision; multi-task self-training passes the screen (+6.06 fine-grained), stage 2 next; on SOFC-Exp the CRF head improves on plain MatSciBERT (S1) | A method whose vote improves on the strongest system on unseen papers under the 10-seed design, and on SOFC-Exp |
| 7 | In progress: NER, part 1, document-level label consistency for unseen papers, a post-processing rule on the saved predictions (no GPU; protocol below); part 2a, recall calibration (a bias on the CRF head's emissions, chosen on inner folds; protocol below); part 2b, combinations of the methods that pass week 6 and mention replacement, fixed after the week-6 decisions. Then class-level taxonomy (LLM or cross-encoder + arborescence) and a relation annotation pilot | Part 1: the rule improves on the strongest system on unseen papers and on SOFC-Exp's CRF head. Taxonomy: first class-level F1 above chance; relation guidelines |

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
