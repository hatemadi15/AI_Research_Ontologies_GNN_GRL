# Plan: beating the published numbers properly (draft, 4 November 2026)

*Status: draft for an independent referee's check (a Fable-model referee) before
any run. It runs after the current work: the two exploratory GLiNER checks, the
post hoc checks over all 118 decisions, and paper B's typeset TACL version. The
published numbers come from an audit of the literature on 4 November, every
number read in its paper.*

## 1. What "properly" means

A claim to beat the state of the art (SOTA) on a benchmark needs all of these.
- **The benchmark's own protocol:** its split, entity set, nesting and scorer, exactly as
  the best published result used them.
- **A protocol fixed before any run,** with the decision rule, the target number and the
  seeds.
- **Selection on validation data only.** The test set is read once per final system, on
  10 seeds never used before (offsets 1000–1900).
- **Two decisions per benchmark:**
  - **D-T:** our 10-seed mean is above the target number. Where the paper gives a mean,
    an SD and a run count, this is a Welch test on those statistics; otherwise the target
    is treated as fixed, and the decision uses the one-sample 95% interval.
  - **D-P:** our system beats our own re-run of the published system, paired by seed on
    the same split, so that the gain belongs to the method and not to the pipeline.
- **Two numbers reported for ensembles:** a seed vote is an ensemble, so the single model
  and the vote are reported separately and claimed as such.
- **Every decision counted:** with the false passes expected by chance, as in papers A
  and B.

## 2. The targets (audit, 4 November)

| Benchmark and protocol | Best verified number | Higher claims, protocol unverified | Our runs so far (known, so not usable for selection) |
|---|---|---|---|
| MaterioMiner, released 323/57/96 sentence split, seqeval exact match, micro | 69.92 fine-grained, 72.32 coarse-grained: MatSciBERT, linear head, mean of 5 (Durmaz 2024) | none; no later paper uses this split | CRF head 73.31 [72.28, 74.35] and 75.48 [74.23, 76.72], 10 seeds (reported) |
| SOFC-Exp, 4 entity types, micro, conlleval, cross-validation over the 34 non-test articles (5 folds × 3 seeds, fixed 11 test articles) | 84.77 ± 1.02: MatSciBERT-CRF (Gupta 2022) | 85.71 ± 0.20 (MatSciBERT-MRC; best run 85.95; split unstated; Huang 2024); ≈86.0 (LLaMA-3.2-3B with QLoRA, official split, read off a bar chart; Lee 2025) | — |
| SOFC-Exp, official 26/8/11 article split | 83.4 ± 0.9: MatSciBERT-CRF, one test article dropped (MuLMS paper) | ≈86.0 (Lee 2025, as above) | CRF head 83.75 [82.85, 84.65], seed vote 85.13 |
| MuLMS, official 36/7/7-document split, 13 types, every nested entity scored, micro | 79.6 ± 0.4: MatSciBERT-CRF with nested BILOU (3 stacked layers), 5 runs (Schrader 2023) | none | none comparable: our runs keep only the outermost entities |

**Bars.** A claim must clear the highest credible number for its benchmark, not only the
best verified one:
- **SOFC-Exp:** about 86.0 on the official split. D-T is taken against 84.77 under
  cross-validation and against 83.4 on the official split, and the margin over 86.0 is
  reported as a separate line. A "SOTA" headline needs the lower bound of the official-split
  interval above 86.0.
- **MaterioMiner:** the published numbers are the bar.
- **MuLMS:** the published numbers are the bar.

## 3. Candidates

All candidates are screened on validation with 3 seeds, and none is tested.
- **A. Tuning the reference.** MatSciBERT-CRF with its learning rate (2e-5, 3e-5, 5e-5),
  epochs and warm-up chosen on validation. Published baselines are rarely tuned, so this
  is the cheapest gain and the honest baseline.
- **B. Larger encoders, tuned.**
  - DeBERTa-v3-large at 1e-5 and 2e-5, with gradient checkpointing to fit a T4. Our week-6
    screen ran it at MatSciBERT's 5e-5 and lost 4.4 and 8.0 points on unseen papers, so
    it has not been fairly tested.
  - A continued-pre-training RoBERTa (ManufactuBERT, 84.40 on SOFC-Exp), if its weights
    are public.
- **C. Training changes with prior support here:**
  - multi-task over the two type levels (MaterioMiner: +1.83 [+0.32, +3.34] on random
    splits) and over SOFC-Exp's entities and slots;
  - joint training across corpora, as Lee 2025 did. The shared SOFC-Exp/MuLMS article is
    dropped first.
- **D. Seed votes** of 3 and 5 models. They added 0.8 to 2.5 points wherever we measured
  them. They are reported as ensembles.
- **E. A span head for nested entities** (required for MuLMS):
  - stacked BILOU layers as in the MuLMS paper, or
  - a span classifier over MatSciBERT with a width cap.

  The span head is also tried flat on SOFC-Exp, where a reading-comprehension span model
  reached 85.7.
- **F. Optional: a fine-tuned 3B decoder** (LLaMA-3.2-3B or Qwen2.5-3B, 4-bit with LoRA,
  entity markers), the one approach published near 86 on SOFC-Exp. It runs only if
  A–E fall short of 86 on SOFC-Exp's validation, and only within the budget below.

## 4. Phases

- **P0, reproduction gates** (no claim). Our re-run of each published system must
  reproduce its number within that number's spread before anything is compared on that
  benchmark:
  - **MaterioMiner, plain MatSciBERT.** Done: 70.19 against 69.92.
  - **SOFC-Exp's cross-validation protocol.** First obtain Friedrich et al.'s fold
    assignment, from the original release or MatSciBERT's code; our data hold only the
    official split. If the folds cannot be found, the cross-validation target is dropped
    and only the official split is used. The gate is 15 runs.
  - **SOFC-Exp's official split.** Our CRF head's 83.75 is consistent with 83.4 ± 0.9.
  - **MuLMS, nested.** The MuLMS paper's model is implemented and scored on every nested
    entity, against 79.6 ± 0.4, over 5 runs.

  A failed gate stops that benchmark until the gap is explained.
- **P1, screens.** Candidates A–E on each benchmark, on validation, 3 seeds. The
  selection rule is fixed in advance: the highest mean validation micro F1, with ties
  going to the cheaper system. One system per benchmark, plus its vote, goes to P2.
- **P2, the decisions.** The selected system and our re-run baseline, 10 fresh seeds
  each, on the test set once. The decisions are D-T and D-P per benchmark and
  granularity:
  - MaterioMiner, fine-grained and coarse-grained: 4;
  - SOFC-Exp, cross-validation and official split: 4;
  - MuLMS: 2.

  That makes 10 decisions, with about 0.25 false passes expected.
- **P3, reported.** The selected systems also run on the held-out-document protocols of
  papers A and B. A system that beats a random-split number need not narrow the
  unseen-document gap, and the report says which it does.

## 5. Budget and order

- **GPU hours** (Kaggle's free T4s, about 30 GPU hours a week):
  - P0: about 5;
  - P1: 15–20;
  - P2: 10–15;
  - F, if it runs: 10–20 more.

  That is 2–3 weeks of quota without F.
- **Order:** P0 code (SOFC-Exp folds, the nested MuLMS head) → the protocol for P0–P2
  committed → the P0 runs after the 7 November quota reset → P1 → P2 → write-up.

## 6. Risks and limits

- **Test sets already seen.** Our CRF head's results on MaterioMiner's released split and
  on SOFC-Exp's official test articles are known. Selection uses validation only, and the
  P2 seeds are fresh; the paper discloses this.
- **Comparability.**
  - SOFC-Exp's test articles are tokenized unlike its training articles. The published
  numbers carry this, so we score on the released test files and report the re-cut files
  separately.
  - Cross-validation numbers average fold models; the official split trains one.
  - Macro and micro F1 are not mixed.
  - Best-run numbers (Huang 2024) are compared with means only as reported numbers.
- **MaterioMiner's split.** It is a random sentence split of 4 papers, so a new SOTA on it
  says little about new documents, which is papers A and B's point. P3 answers that.
- **Unverifiable bars.** The ≈86.0 on SOFC-Exp is read off a figure. We report our
  margin over it, and do not claim to beat a number we cannot verify.
- **Publication.** It goes in a short separate paper (paper C), or into paper B's
  discussion if the gains are small; this is decided after P2.
