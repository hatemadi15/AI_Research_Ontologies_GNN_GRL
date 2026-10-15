# Paper plan: named-entity recognition on MaterioMiner

This outline lists what a paper on the NER results could claim, the result
behind each claim, and what is still missing. It is updated as the weekly
protocols (ROADMAP §4) finish. All numbers come from the results files in
`results/` and the tables that `report_results.py` generates from them.

**Type.** An evaluation and reproducibility paper. It becomes a methods paper
only if a change beats the seed vote on unseen papers under the 10-seed design
(week 6, ROADMAP P1 "Next, after week 4" items 2–5).

**Working title.** "How much of a materials-NER gain survives? Splits,
scoring and seeds on MaterioMiner".

## Claims and evidence

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | The published numbers (69.92 fine, 72.32 coarse) come from one random split, which the release ships (323 / 57 / 96 sentences) without saying so in the paper. Over 5 new random splits, plain MatSciBERT scores 73.91 [69.36, 78.45] fine-grained and 74.57 [73.15, 75.98] coarse-grained. | README, "NER across splits"; `results/ner_results_*_random_split_vary.json` | Done (CPU). Week 5 runs the published split itself, with the release's scoring (micro F1, seqeval's default mode); the authors are asked to confirm it (item 6) |
| C2 | Random splits hide a large gap: on a paper the model has never seen, the seed vote `crf_ens3` scores 18–30 points lower than on random splits (49.55 fine, 61.42 coarse, against 79.76 and 79.45). | README, "NER out of distribution"; the `leave_one_paper_out` files | Done (CPU, 3 seeds); week 5 re-estimates it with 10 seeds |
| C3 | A CRF head with IOB2 constraints and a vote over 3 seeds beats the published numbers over 5 random splits: fine-grained under micro, macro and weighted F1; coarse-grained under micro and weighted F1, not macro (73.03 [67.79, 78.27]). | README, the reported-system rows | Done (CPU); week 5 decides the reported system on 10-seed votes (GPU) |
| C4 | The CRF head's fine-grained gain is span validity, not recognition: under strict IOB2 scoring it vanishes (+0.10 [−2.08, +2.29]), and dropping invalid tags from the baseline's output gives the same (strict F1 76.26). Coarse-grained, part of the gain is recognition (strict +1.91 [+1.17, +2.65]). | README, "Why the CRF helps" | Done (CPU) |
| C5 | Popular additions do not pass a pre-registered paired test: class weights, domain-adaptive and in-domain pre-training, other backbones, a mixed-backbone vote, label semantics, multi-task training and their combination. The 1-point coarse-grained gains of week 3 did not repeat on a GPU, and the hardware alone moves a 3-seed mean by 1.2 points per split (up to 2.7). | ROADMAP P1 weeks 1–4; README week-3 and week-4 tables | Done; week 5 re-tests the three claims with 10 seeds |
| C6 | Self-training with 3 teachers on an in-domain corpus of 40,000 sentences gives a consistent fine-grained gain: +1.07 [+0.33, +1.80] on unseen papers, +1.22 [+0.60, +1.84] on random splits. Its students stay below the seed vote. | `results/gpu/ner_week4_comparisons.json` | 3 seeds (GPU); week 5 re-tests it with 10 seeds |
| C7 | Exploratory: how far a 3-seed design misleads, from the 120 seed triples of the 10-seed runs. | `ner_seed_analysis.py` | Week 5 |
| C8 | Exploratory: a few rare coarse-grained types (2–8 test entities per split) keep macro F1 near the published number. A preview from one vote: SDE, MIP, PRP, MOD and CGE score 17–46 F1 on the random splits. | `ner_seed_analysis.py`, per-type scores | Preview; week 5 with 10 seeds |

## Figures and tables

- Figure 1: F1 on random splits against F1 on unseen papers, per system (C2).
- Figure 2: seed spread over 10 seeds per split, and the distribution of the
  3-seed differences over the 120 triples against the 10-seed one (C5, C7).
- Figure 3: per-type F1 against test support, coarse-grained (C8).
- Table 1: the reproduction over 5 random splits next to the published
  numbers, under micro, macro and weighted F1 (C1, C3).
- Table 2: every pre-registered comparison of weeks 1–5, its protocol, its
  role (decision or reported) and its outcome; a 10-seed column (C5, C6).
- Table 3: default against strict IOB2 scoring for the baseline and the CRF
  head (C4).

## Reproducibility appendix

- The protocol of each week, committed before its runs (ROADMAP §4), and the
  decision rule: the 95% Student-t CI of the mean paired difference above 0.
- GPU runs: one frozen commit per batch, one Kaggle image pinned by digest,
  deterministic CUDA. Runs repeat exactly across GPUs, sessions and commits:
  the 54 teachers of week 4 equal the separately trained runs, and week 5's
  first job repeats a week-4 run.
- Every run's test predictions are in `results/`, so any vote or score can be
  recomputed without training; the tables are generated and checked by the
  test suite (`report_results.py --check`).

## Still missing before submission

1. The week-5 results: 10 seeds per split, the re-tested claims and the
   reported system (in progress).
2. The authors' split and averaging (item 6). Without them, the comparison
   with 69.92 / 72.32 is stated as one over 5 random splits.
3. A second materials NER dataset (item 7, week 6), so the findings do not
   rest on 4 papers.
4. Related work:
   - MatSciBERT (Gupta et al., 2022) and MaterioMiner (Durmaz et al., 2024);
   - out-of-distribution materials NER (Sci Rep 2025);
   - seed variance and reporting score distributions (Reimers and Gurevych,
     2017; Dodge et al., 2020);
   - label semantics (Ma et al., 2022);
   - document context (FLERT; Schweter and Akbik, 2020);
   - self-training for NER.
5. The venue, chosen after week 5. Candidates are a materials-informatics
   journal or workshop, or a scholarly-document-processing workshop.
