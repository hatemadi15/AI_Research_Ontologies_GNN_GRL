# Paper plan: named-entity recognition on MaterioMiner

This outline lists what a paper on the NER results could claim, the result
behind each claim, and what is still missing. It is updated as the weekly
protocols (ROADMAP §4) finish. All numbers come from the results files in
`results/` and the tables that `report_results.py` generates from them.

**Type.** The aim is a methods paper: a change that improves materials NER
on unseen papers over the strongest system so far (the seed vote, or the
multi-task vote fine-grained), decided under the 10-seed design and repeated
on a second dataset. Week 6 tests three candidates (DeBERTa-v3-large,
document context, multi-task self-training; ROADMAP P1, "Next, after week
5"), and later weeks combine the ones that pass. If none passes once these
options are exhausted, the paper is an evaluation and reproducibility paper.
Claims C1–C10 hold either way: in a methods paper they are its baselines and
evaluation protocol, in an evaluation paper its results. Document context
passed the week-6 screen (+2.66 [+0.91, +4.40] fine-grained) and is being
decided with 10 seeds; it is the leading candidate for a methods claim.

**Working titles.** Methods: an approach for materials NER on unseen papers,
named after the change that passes. Evaluation: "How much of a materials-NER
gain survives? Splits, scoring and seeds on MaterioMiner".

## Claims and evidence

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | The published numbers are reproduced on the split the release ships (323 / 57 / 96 sentences, a different partition per granularity, not described in the paper), scored like its code (micro F1, seqeval's default mode): plain MatSciBERT scores 70.19 [69.03, 71.35] fine-grained and 72.52 [71.88, 73.16] coarse-grained over 10 seeds, against 69.92 and 72.32. Single runs of plain MatSciBERT range over 4.6 points fine-grained (68.0–72.7) and 3.6 coarse-grained (70.4–74.1). | `results/gpu/ner_results_*_published_split.json`; README, the published-split rows | Done (GPU, 10 seeds); the authors are asked to confirm split and scoring (ROADMAP item 6) |
| C2 | Random splits hide a large gap (the paper says the out-of-distribution generalisation "can not be estimated" from four publications; leaving one paper out estimates it). The seed vote `crf_ens3` scores 79.68 fine-grained and 78.63 coarse-grained over 5 random splits, but 48.81 and 61.10 on a paper the model has never seen: 18–31 points lower. | `results/gpu/ner_week5_comparisons.json` (`crf_ens3` vs `crf`) | Done (GPU, 10 seeds) |
| C3 | A CRF head with IOB2 constraints beats the published numbers. On the published split it scores 73.31 [72.28, 74.35] and 75.48 [74.23, 76.72], +3.12 and +2.96 over plain MatSciBERT, paired by seed. Over 5 random splits, the reported system beats them fine-grained under micro, macro and weighted F1, and coarse-grained under micro and weighted F1 but not macro (70.83 [64.49, 77.17]). | README, the published-split and reported-system rows | Done (GPU, 10 seeds) |
| C4 | Much of the CRF head's gain is span validity. Under strict IOB2 scoring its gain on the published split shrinks from +3.12 to +1.70 [+0.21, +3.19] fine-grained and from +2.96 to +1.07 [−0.82, +2.97] coarse-grained. Over the random splits (CPU, 5 seeds) the fine-grained gain vanishes under strict scoring (+0.10 [−2.08, +2.29]), and dropping invalid tags from the baseline's output gives the same; coarse-grained, +1.91 [+1.17, +2.65] survives. | README, "Why the CRF helps"; `ner_week5_seed_analysis.json` (`published_split`) | Done |
| C5 | A vote over 3 seeds is the most reliable addition: +2.20 and +1.67 over the mean of single CRF runs on the random splits, +2.36 and +2.50 on unseen papers, every CI above 0. | `ner_week5_comparisons.json` (`crf_ens3` vs `crf`) | Done (GPU, 10 seeds) |
| C6 | Popular additions do not pass a pre-registered paired test with 10 seeds: label semantics (model or vote, either protocol), self-training with 3 teachers on unseen papers (+1.74 [−1.16, +4.64] fine-grained, the week-4 claim), and coarse-grained multi-task training (+0.71 [−0.15, +1.56], the week-3 claim). Earlier: class weights, domain-adaptive and in-domain pre-training, other backbones and a mixed-backbone vote. | ROADMAP P1 weeks 1–5; README week-3 to week-5 tables | Done |
| C7 | Multi-task training over both granularities is the one addition that holds: fine-grained +1.83 [+0.32, +3.34] over the CRF head on the random splits. Its vote improves on the seed vote coarse-grained on the random splits (+0.79 [+0.17, +1.42]) and fine-grained on unseen papers (+3.65 [+0.49, +6.82]): 2 of 12 vote comparisons, where about 0.3 would pass by chance. Self-training improves on the random splits (+1.05 and +1.04, reported, no decision). | `ner_week5_comparisons.json` | Done (GPU, 10 seeds) |
| C8 | How far a 3-seed design misleads, from the 120 seed triples of the 10-seed runs: the 1.8-point multi-task gain passes in 41% of the triples; label semantics and self-training on unseen papers in 2–18%; up to 37% of the triples give the opposite sign. A split's micro F1 varies over seeds with an SD of 1.5–1.7 points on random splits and 2.2–2.5 on unseen papers. The 1-point CPU gains of week 3 that passed with 3 seeds do not hold with 10. | `ner_week5_seed_analysis.json` (`triples`, `spread`); README, "how far a 3-seed design misleads" | Done (exploratory) |
| C9 | A few rare coarse-grained types keep macro F1 near the published number: in the reported system's votes on the random splits, SDE and MIP score about 18 F1 and PRP, MOD and CPR 40–42, with 2–8 test entities per split. | `ner_week5_seed_analysis.json` (`types`) | Done (exploratory); week 6 targets them |
| C10 | Where the CRF head fails on unseen papers: fine-grained it misses 36% of the gold entities outright (10% on random splits) and predicts 10,026 entities for 13,280 gold ones; 41% of these misses have a type seen fewer than 10 times in the three training papers. Coarse-grained it misses only 6%, but 16% get the right span and the wrong type, and 21% a wrong boundary. | `ner_week5_seed_analysis.json` (`errors`) | Done (exploratory, 10 seeds) |
| C11 | A larger general-domain encoder does not transfer to unseen papers: DeBERTa-v3-large (435M parameters) with the CRF head scores −4.38 [−7.32, −1.45] fine-grained and −8.01 [−11.72, −4.30] coarse-grained against MatSciBERT (110M) on the held-out papers, while its validation scores on the training papers are similar. | `gpu/ner_week6_screen.json` | Screen (3 seeds); SOFC-Exp S2 running |
| C12 | Document-level label consistency, a post-processing rule (a term the model tags in most of its occurrences in a paper is tagged in the rest), adds little: +0.30 [−0.35, +0.95] fine-grained and +0.12 [+0.01, +0.24] coarse-grained on the CRF head, and no significant change on the strongest votes. Retyping does better coarse-grained (+0.64 [+0.29, +1.00]), post hoc; it is decided on SOFC-Exp. | `gpu/ner_week7_comparisons.json` | Week 7 part 1: MaterioMiner done; SOFC-Exp pending |

## Figures and tables

- Figure 1: F1 on random splits against F1 on unseen papers, per system (C2).
- Figure 2: seed spread over 10 seeds per split, and the distribution of the
  3-seed differences over the 120 triples against the 10-seed one (C8).
- Figure 3: per-type F1 against test support, coarse-grained (C9).
- Table 1: the published split (plain MatSciBERT and the CRF head, 10 seeds)
  next to the published numbers, under micro, macro, weighted and strict F1
  (C1, C3, C4).
- Table 2: every pre-registered comparison of weeks 1–5, its protocol, its
  role (decision or reported) and its outcome, with the 3-seed and 10-seed
  results side by side (C5–C8).
- Table 3: default against strict IOB2 scoring for the baseline and the CRF
  head (C4).

## Reproducibility appendix

- The protocol of each week, committed before its runs (ROADMAP §4), and the
  decision rule: the 95% Student-t CI of the mean paired difference above 0.
- GPU runs: one frozen commit per batch, one Kaggle image pinned by digest,
  deterministic CUDA. Runs repeat exactly across GPUs, sessions and commits:
  the 54 teachers retrained in the week-5 self-training session equal the
  separately trained CRF runs, and every other session's first job repeats
  a week-4 run.
- Every run's test predictions are in `results/`, so any vote or score can be
  recomputed without training; the tables are generated and checked by the
  test suite (`report_results.py --check`).

## Still missing before submission

1. A second materials NER dataset (week 6), so the findings do not rest on
   4 papers. SOFC-Exp is converted (`sofc_exp.py`) and checked: MatSciBERT
   with the CRF head scores micro 84.98 and macro 82.31 on its official
   split with one seed, against the published 84.77 and 82.39. Its main
   comparisons need 10 seeds per split, on random splits and on folds
   grouped by article.
2. The authors' confirmation of the split and the scoring (item 6; the email
   is drafted). The reproduction (C1) already matches both numbers.
3. Related work:
   - MatSciBERT (Gupta et al., 2022) and MaterioMiner (Durmaz et al., 2024);
   - out-of-distribution materials NER (Sci Rep 2025);
   - seed variance and reporting score distributions (Reimers and Gurevych,
     2017; Dodge et al., 2020);
   - label semantics (Ma et al., 2022);
   - document context (FLERT; Schweter and Akbik, 2020);
   - self-training for NER;
   - SOFC-Exp (Friedrich et al., 2020).
4. The venue. Candidates are a materials-informatics journal or workshop, or
   a scholarly-document-processing workshop.
