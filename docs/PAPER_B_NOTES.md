# Paper B: notes on the draft (not part of the submission)

## Review round 1 (3 November 2026, on `44f9677`)

Three reviewers ran independently:
- a referee, who recommended major revision;
- a number audit: about 965 items checked, 12 mismatches, none blocking;
  every verdict and the count reproduce;
- a consistency check against the submitted paper A (`325019a`), the
  record and the paper itself: 3 blocking items, 12 should-fix and 18
  minor.

Every finding was checked against the files before it was acted on.
Labels below: R = referee, A = audit, C = consistency.

**Verified and decisive.**
- **R2.** The abstract's lesson "the four comparisons of weeks 3 and 4 that
  passed with 3 seeds all failed with 10" changed hardware as well as
  seeds. Three of the four were CPU runs, and on the GPU the same seeds 0,
  100 and 200 already fail:

  | Comparison | CPU, seeds 0–200 | GPU, seeds 0–200 |
  |---|---|---|
  | label semantics, coarse | +1.17 [+0.94, +1.41] | +0.07 [−2.47, +2.61] |
  | multi-task, coarse, random splits | +0.75 [+0.10, +1.39] | −0.08 [−1.71, +1.55] |
  | label-semantics vote | +1.63 [+0.11, +3.16] | +0.40 [−1.36, +2.16] |

  Only self-training (GPU, week 4) failed because seeds were added. The
  lesson is reworded: a 3-seed interval is too narrow for run-to-run
  variation, and both seeds and hardware show it.
- **R1.** The model for units that share training documents (paper A's,
  `decision_tests.dependent_t`) widens a MaterioMiner interval by ×1.528.
  Applied to paper B's own passes, it removes the margin of several of
  them. Post hoc, and reported beside every verdict.
- **R5.** A held-out MuLMS subfield's pool is 2–3 whole articles
  (`subfield_folds`), while the random-split pool draws from all ~50
  articles (`pool_split`). The 17-point gap therefore mixes topic shift
  with document diversity at equal size.

**Dispositions.**
- **Post hoc reanalysis of released runs**
  (`src/graph/paper_b_checks.py`, `results/gpu/paper_b_checks.json`):
  - R1: dependence-adjusted intervals, with Nadeau and Bengio's correction
    on random splits;
  - R4: Benjamini–Hochberg over the 111 decisions, and Holm within each
    family;
  - R2: the hardware re-decisions, and the matched CPU–GPU runs;
  - R9: the detectable difference of every decision;
  - R3: unseen against seen gains;
  - R6: an interval for every gap.
- **New runs, week 20** (`docs/WEEK20_PLAN.md`, checked by an independent
  referee before any run):
  - R5: the MuLMS pool drawn by sentence (decision 20.1);
  - R10: week 8's diversity decisions on seeds 300–900 (20.2, 20.3).
- **Text:**
  - all of A1–A10 and C1–C33;
  - R2, R3, R6–R10, R12–R15, R17 and R19;
  - R18 in part: the definitions, and seed-only intervals marked;
  - R14 goes to the Limitations.
- **The author's choice:**
  - R11's LLM or generalist-NER baseline. Meanwhile the abstract is scoped
    to fine-tuned taggers, and the related work is extended.
  - R13's adjudication of 200 errors. Meanwhile the wording is softened.
  - R16's page plan, which depends on the venue (TACL or ARR).

**The post hoc reanalysis** (`paper_b_checks.py`, `62f9641`). Nothing
decides anything here. The 111 decisions and 24 passes reproduce from the
files, and every stored interval reproduces within 0.02.
- **Shared training documents.** Of the 15 passes over units with 7 or 10
  seeds, 5 keep their margin: MR fine-grained, D4, D5, D11 and 13.1a.
  Ten lose it. The 3 SOFC-Exp passes, seed-paired on one split, get no
  allowance.
- **Multiplicity.** Benjamini–Hochberg passes 3 at FDR 5% and 12 at 10%;
  Holm within each family keeps 6.
- **Hardware.** 186 matched CPU–GPU runs differ by a median 1.43 F1.
- **Gaps** with Welch intervals: SOFC-Exp +3.48 [−0.90, +7.86]; MuLMS
  +17.06 [+9.31, +24.81].
- **Power.** The single-model decisions on held-out papers detect 0.4 to
  3.8 points (multi-task self-training's fine-grained one 8.1).
- **Unseen minus seen** cannot be tested: the stored summaries hold no
  per-unit values.

These go into §7.7, §1, §5.2, lesson 1 and the abstract. The abstract is
rewritten at 307 words, from about 450. The page-plan cut waits on the
venue.

## The TACL version (3 November 2026)

- **`docs/PAPER_B_TACL.md`.** The anonymized TACL submission, condensed
  from this draft.
  - The main text is about 6,280 words with 4 figures and 4 tables.
  - Appendix A (replication material) is about 2,930 words; Appendix B
    (tables only) has 73 rows.
- **This draft stays the full record.**
- **Placeholders.** Every number that stage 21 sets is a ⟦GLINER⟧ or
  ⟦COUNT⟧ placeholder, 33 in all, until `w21a` is decided.
- **Changes from this draft:**
  - the abstract drops the post hoc margin count, the power range and
    the tokenization finding, which stay in the body;
  - "fine-tuned encoder taggers only" becomes a limitation naming
    DeBERTa's screen and GLiNER;
  - an anonymized statement of AI assistance is added, as in paper A.
- **Every main-text number** occurs in this draft, checked by a script,
  apart from stage 21's numbers, which come from its protocol.
- **Review round 2** comes after the placeholders are filled.
- **4 November: the 33 placeholders are filled** from week 21's analysis
  files (`ner_week5_seed_analysis.json`, `week21`; `ner_mentions.json`,
  `week21`). None of 21.1–21.4 passes, and 21.2 is below the CRF head.
  The count is 118 decisions, 27 passing, about 3.0 by chance, and 88 with
  7 or 10 seeds, 21 passing. `docs/PAPER_B.md` gains §7.7 on the
  generalist tagger, and its count moves to §7.8.
