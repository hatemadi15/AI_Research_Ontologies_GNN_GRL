# Plan: beating the published numbers properly (4 November 2026)

*Status: revised after an independent referee's check (a Fable-model referee,
4 November: "approve with required changes"). Its seven required changes,
R1–R7, are applied below, together with its missing candidates and its
design points. The plan runs after the current work: the two exploratory
GLiNER checks, the post hoc checks over all 118 decisions, and paper B's
typeset TACL version. A protocol in `docs/ROADMAP.md` §4 fixes it in detail
before any run. The published numbers come from a literature audit of
4 November, every number read in its paper.*

*P1, 7 November: its protocol is fixed in ROADMAP §4, after an independent
referee's check ("approve with required changes", all six applied). Its code
is written and tested (`ner_baseline.py --screen` and round 1's options,
`mulms_nested.py --screen`, `span_ner.py`, `sota_p1.py`, batches `p1t` and
`p1n`). The timing batch `p1t` passed every check (7 November). With its
table, the cut ladder takes A2m and C3 off MuLMS and screens MaterioMiner
on 5 units, for 21.19 session-hours. The table counts A0 rows that had
already run; without them MaterioMiner would have kept 10 units (disclosed
in ROADMAP §4). Round 1 ran on 8 November; its table is in ROADMAP §4
(validation only). A6 ran as A0 because of a bug, since fixed; it runs again
with MuLMS's extra units, and round 2 on MaterioMiner and SOFC-Exp waits for
it. Stage 2 ran on 8 November. MuLMS's selection is A1m, learning rate 5e-5:
78.62 against A0's 77.60 over 7 units, validation only. Round 2 on
MaterioMiner and SOFC-Exp is running in stages. Amendment 1
(before any run) has C1 on MaterioMiner run one level at a time: the two levels'
released splits are different partitions. It departs from this plan in four
places, each justified there:
- MaterioMiner is screened on 10 splits of the released training and
  validation sentences, not on 5 random splits that hold test sentences.
- Tokenization-robust training is not screened, because no validation set is
  cut like the test files.
- The tie rule covers up to 3 cells.
- P1's estimate from measured times is about 22 session-hours, capped at 24,
  where §4 below says 10–15. `p1t`'s table measured 31.1 before cuts.

The corpora share four documents, found by an 8-gram check. One of them is
MuLMS's test document `sofc_01`, an SOFC-Exp training article. Joint training
drops all four from the side it adds.*

*P0, 7 November: every gate passed (batch `p0a`; ROADMAP §4 has the record,
`results/gpu/sota_p0.json` the numbers). A pass means that no shortfall was
found, not that the numbers are shown equal. Our mean minus the published one:
- SOFC-Exp cross-validation: 84.16 against 84.77, −0.61 [−1.28, +0.07]. The
  interval only just reaches 0.
- SOFC-Exp's official split on 10 articles: 84.34 against 83.4, +0.94 [−0.41,
  +2.30].
- MuLMS, every nested entity: 79.04 against 79.6, −0.56 [−1.41, +0.29].

P1's protocol comes next.*

*P0, 4 November: the code is written and tested (`ner_baseline.py --sofc-cv`,
`mulms_nested.py`, `sota_p0.py`), and its protocol is fixed in ROADMAP §4. Four
findings:
- 84.77 is the micro F1 of the published (npj) version; the arXiv version prints
  84.39 micro and 81.92 macro.
- MatSciBERT's code picks its epoch by dev macro F1, while our recipe picks it by
  micro F1.
- The SOFC-Exp test article that MuLMS also holds is PMC6247067 (MuLMS's
  `sofc_04`), which is in MuLMS's dev set, not its training folds.
- MuLMS's MatSciBERT tagger used learning rates 1e-4/7e-3 (the paper's
  Table 13), not the release script's 5e-5, which is SciBERT's.

An independent referee approved P0 with six required changes, all applied.*

## 1. What "properly" means

- **The benchmark's own protocol.** We use the split, entity set, nesting and scorer of
  the result we compare with, and we name the test set of each decision.
- **A protocol fixed before any run.** It holds the decision rules, the targets, the
  seeds, the budget table and the outcome wordings (§5).
- **Selection on validation data only.**
  - The test set is read once per final system, on fresh seeds (offsets 1000–1900, and
    up to 1000–3900 for the votes).
  - MaterioMiner is screened on the validation sets of its 5 random splits, never on the
    57-sentence validation set of the released split.
  - The two best candidates within 1 point of each other on a benchmark get 4 more
    screen seeds.
  - The selection is made per benchmark and granularity.
- **D-T, against the target.**
  - Where the paper gives an SD and a run count, the test is Welch's, with the unit stated.
  - Otherwise, our one-sample 95% interval is set against the target taken as fixed. That
    is anti-conservative, and the paper says so.
  - Every D-T is worded "on this test set": the seed interval does not cover sampling of
    the test set (173 and 96 sentences).
- **D-P, against our own re-run.**
  - The reference is our re-run of the published system, or the tuned reference
    (candidate A) when the winner is another candidate, so that tuning is not credited to
    a method. If A itself wins, D-P reads "tuning".
  - The interval is paired by seed. Across architectures, seed pairing is nominal and the
    interval is unpaired (Welch).
  - A paired bootstrap over test sentences is reported beside it.
- **Single models are the headline.** A vote is an ensemble, and gets its own decisions
  only on 30 fresh seeds: 10 disjoint votes of 3, and 6 of 5, on SOFC-Exp's official split
  and MaterioMiner's released split, under 3 session-hours. Otherwise votes are reported
  without a decision.
- **Every decision is counted,** with the false passes expected by chance. Decisions
  whose outcome is already known are re-tests, in a count of their own (R3).

## 2. The targets, and what is already known

| Benchmark and protocol | Best verified number (± as printed) | Higher claims, protocol unverified | D-T unit |
|---|---|---|---|
| MaterioMiner, released 323/57/96 sentence split, seqeval exact match, micro | 69.92 fine-grained, 72.32 coarse-grained: MatSciBERT, linear head, mean of 5, no SD printed (Durmaz 2024) | none | one-sample, target fixed |
| SOFC-Exp, 4 entity types, micro, conlleval, cross-validation (CV) | 84.77 ± 1.02: MatSciBERT-CRF, SD over 15 fold × seed runs (Gupta 2022) | 85.71 ± 0.20 (MatSciBERT-MRC, best run 85.95, split unstated; Huang 2024) | per run, 5 folds × 10 seeds (n = 50), Welch against n = 15 runs, as P0 does |
| SOFC-Exp, official 26/8/11 article split | 83.4 ± 0.9 on 10 test articles: MatSciBERT-CRF, 5 seeds; one test article is also in MuLMS, in its dev set (Schrader 2023) | ≈86.0 on all 11 test articles (LLaMA-3.2-3B with QLoRA, read off a bar chart; Lee 2025) | per seed (n = 10), Welch against n = 5 on 10 articles; the margin over 86.0 is reported on 11 |
| MuLMS, official 36/7/7-document split, 13 types, every nested entity, micro | 79.6 ± 0.4: MatSciBERT-CRF with stacked BILOU layers, 5 runs over the 5 train-fold configurations (Schrader 2023) | none | per run (n = 10), Welch against n = 5 |

**The folds and test sets (R1, R2, R6).**
- **SOFC-Exp's CV folds** follow the release code (`modify_cross_val_data_split`): the 34
  train and dev article IDs sorted alphabetically, and fold k's dev set every 5th article
  from position k − 1. The test score is the mean of the per-fold scores, with no vote. P0
  checks this against MatSciBERT's repository and records the fold IDs as a data file
  with a hash.
- **SOFC-Exp's official split.** Every system is scored on both test sets from the same
  saved predictions: 11 articles (our history, Lee) and 10 (Schrader). Joint training with
  MuLMS drops the shared article from MuLMS's side, never from SOFC-Exp's test.
- **MuLMS** uses the release's tokens and evaluation script. Our loader alters 276 tokens
  and drops 2 entities, so it is not used here. The 5 fold configurations are recorded as
  a data file.

**Known test results (R3).** The candidate list was drawn after seeing them.
- **SOFC-Exp's official test articles:**
  - plain MatSciBERT; the CRF head, 83.75;
  - document context, 84.34;
  - multi-task;
  - mention replacement, alone and on the multi-task model;
  - label propagation and recall calibration;
  - the re-cut test articles;
  - GLiNER-medium.
- **MaterioMiner's released split:**
  - plain MatSciBERT, 70.19;
  - the CRF head, 73.31 and 75.48 (10 seeds);
  - the multi-task model where it was run.

So MaterioMiner's D-T against 69.92 and 72.32, and its D-P against plain MatSciBERT, are
re-tests of known passes, counted apart. The live MaterioMiner decision is the selected
system against the strongest known system, paired by seed: the CRF head, or the
multi-task CRF fine-grained.

## 3. Candidates

Every candidate is screened on validation, and none is tested.
- **A. The reference, tuned.**
  - MatSciBERT-CRF with its learning rate (2e-5, 3e-5, 5e-5), epochs and warm-up chosen
    on validation.
  - Fine-tuning stabilisers: layer-wise learning-rate decay, re-initialising the top
    layer, weight averaging over the last epochs (Zhang et al. 2021; Mosbach et al. 2021).
- **B. Encoders.**
  - DeBERTa-v3-large and SciDeBERTa: learning rate 1e-5, warm-up 10%, up to 20 epochs,
    gradient checkpointing, fp32, batch 8. Week 6's screen ran DeBERTa at 5e-5, which is
    likely what it measured. Its tokenizer maps 32 MuLMS words to `[UNK]`; this is
    disclosed.
  - MatSciBERT with continued pre-training on the 40,000-sentence SOFC topic corpus
    already built, under an hour of masked-language-model training.
- **C. Training.**
  - Multi-task over the two type levels (MaterioMiner, +1.83 [+0.32, +3.34] on random
    splits) and over SOFC-Exp's entities and slots.
  - Joint training across corpora (Lee 2025).
  - Document context (+0.58 on SOFC-Exp's official test articles, known, so disclosed).
  - Tokenization-robust training: training sentences also presented with a random re-cut
    around hyphens, superscripts and units. The shift costs the CRF head 0.86, and was
    found by inspecting the released test files; this is disclosed.
- **D. Seed votes** of 3 and 5, decided as in §1.
- **E. A head for nested entities**, required on MuLMS: the MuLMS paper's stacked BILOU
  layers, or a span classifier with a width cap. The span head is also tried flat on
  SOFC-Exp, where a reading-comprehension span model reached 85.7.
- **F. Optional: a fine-tuned 3B decoder** (Qwen2.5-3B, Apache-2.0, no gated download).
  It runs only if A–E fall short on SOFC-Exp's validation, and only within the budget.
  - Setup: 4-bit weights with LoRA r = 16, entity markers, greedy decoding, and
    hyperparameters from 3 seeds on validation.
  - Scoring: the generated text is aligned back to the released words, converted to IOB2
    and scored with the same scorer as every system. Unparsable outputs count as no
    entities, and their rate is reported.
  - Repeats: 4-bit kernels do not repeat bit-identically, so F's repeat check is a
    tolerance declared in advance (within 0.1 F1) instead of identity (R7).

## 4. Phases, budget and fallbacks

- **P0, reproduction gates** (no claim; all passed on 7 November). A failed gate stops
  that benchmark until the gap is explained.
  - MaterioMiner, plain MatSciBERT: done (70.19 against 69.92).
  - SOFC-Exp CV: 5 folds × 3 seeds against 84.77 ± 1.02.
  - SOFC-Exp official split: 83.75 on 11 articles, re-scored on 10 against 83.4 ± 0.9.
  - MuLMS, nested: the release's evaluation, 5 configurations against 79.6 ± 0.4.
- **P1, screens.** As in §1.
- **P2, decisions.**
  - The selected single model and its reference, 10 fresh seeds each (MuLMS: 5
    configurations × 2 seeds).
  - The votes on 30 fresh seeds where they are decided (§1).
  - The live decisions: SOFC-Exp CV and official split (D-T and D-P, 4); MuLMS (D-T and
    D-P, 2); MaterioMiner against its strongest known system (2); and votes on two
    benchmarks (up to 4).
  - The re-tests: MaterioMiner's 4, counted apart.
- **P3, reported.** The selected systems run on papers A and B's held-out-document
  protocols. On MuLMS this scores the nested system's outermost predictions.
- **Budget** (R5). The quota is now 45 session-hours a week, with two T4s per session.
  - Measured per run: MatSciBERT-CRF takes 3.3 min on SOFC-Exp's official split, 1.3 on
    MaterioMiner, and about 20–30 min on MuLMS's full training folds. Large encoders are
    4–5× slower.
  - Before P1, timed smoke runs fill a table of session-hours per candidate and benchmark,
    using both lanes.
  - At base size: P0 about 3 h, P1 10–15 h, P2 15–20 h, P3 about 5 h.
  - Large encoders add about 25 h for SOFC-Exp CV, 35–40 h for MuLMS P2 and 20 h for P3,
    and F adds 15–20 h.
- **Fallbacks, fixed now.**
  - Large encoders are screened on SOFC-Exp's official split and MaterioMiner first. They
    enter MuLMS and the CV protocol only if they lead on validation by more than 1 point
    and the budget table fits.
  - If a winner does not fit, its CV decision runs at 5 seeds × 5 folds.
  - Sessions resume finished runs, as `w21a` did.

## 5. Outcome wordings, fixed now

- **SOFC-Exp, official split, 11 articles.**
  - If the single-model interval's lower bound is above 86.0: "above every reported result
    on SOFC-Exp's official split, on this test set".
  - If D-T passes against 83.4 (10 articles) or 84.77 (CV) but the margin over 86.0 does
    not clear: "above the best result with a stated protocol; not shown above the best
    reported result, whose protocol we could not verify".
- **MuLMS.** If D-T passes: "above the published result under its protocol"; else "not
  shown above it".
- **MaterioMiner.** The re-tests are reported as replications. The live decision reads "the
  selected system improves on the strongest known system", or "not shown to".
- **Nothing clears:** "not shown to beat the published numbers; our best single model
  reaches X [interval]".
- **Expected.** The referee expects a single model near 85–85.5 on SOFC-Exp and a vote
  near 86–86.5, which would read "within the interval of the best reported number", not
  above it. The plan is worth running for MuLMS, for the CV protocol and for MaterioMiner
  even if SOFC-Exp's 86 is not cleared.

## 6. Risks, and publication

- **Comparability.**
  - The tokenization shift: the released test files are scored, and the re-cut files are
    reported apart.
  - Cross-validation means average fold models.
  - Macro and micro F1 are never mixed in a decision.
  - Best-run numbers are compared with means only as reported numbers.
- **MaterioMiner's split is random,** so a new number on it says little about new
  documents; P3 answers that.
- **Unverifiable bars.** The author may ask Lee et al. and Huang et al. for their exact
  numbers, split and scorer before P2.
- **Publication, fixed now.** A short paper C only if at least one live D-T passes on
  SOFC-Exp or MuLMS; otherwise a section of paper B's discussion.
