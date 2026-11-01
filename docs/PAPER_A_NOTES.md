# Paper A: notes on the draft (not part of the submission)

`docs/PAPER_A.md` is the submission text of paper A; this file keeps what
a reader of the repository needs beside it.

**Tables and figures to generated files.** The paper is the main text
(`docs/PAPER_A.md`) and the Supplementary Information
(`docs/PAPER_A_SI.md`, Notes 1 to 9, Tables S1 to S7). Table 1 is written
by hand from the conversions in `src/graph/`; Table 2 to
`docs/tables/table6_held_out_decisions.md` and `table2_decisions.md` (D9
and D10 to `table11_week17.md`); Table 3 to `table9_robustness.md`; Table
4 to `results/gpu/ner_mentions.json` (and `table4_unseen.md` for the CRF
head); Table 5 to the paired-gains block of `table10_lexicon.md`, from
`results/gpu/ner_lexicon_recall.json` (`ner_lexicon_recall.py`; the
rebuilt lexicons themselves are not committed). Table S1 to the protocol
record in `docs/ROADMAP.md` §4 and `git log`; Table S2 to
`table7_ablations.md`; Table S3 to `table10_lexicon.md`; Table S4 to
`table8_types.md` (and `tableA2_types_fine.md` for MaterioMiner); Table
S5 to the week-12 screen in `docs/ROADMAP.md`; Table S6 to
`table11_week17.md`; Table S7 to `results/gpu/ner_robustness.json`
(`detectable`). The MaterioMiner paired unseen-name intervals of
Supplementary Note 5 are the `chain` block of
`results/gpu/ner_mentions.json`. Figure 1 is
`docs/figures/fig8_method.png`, Figure 2 `fig9_units.png`, Figure 3
`fig7_names.png` (`src/graph/paper_figures.py`). The worked example's
lexicon entries (Section 4) are the most frequent MAT and PROPERTY
entries of the electrolysis subfield's rebuilt lexicon in
`results/gpu_names/` (not committed); its MR6 replacement, Ni, is a MAT
mention of that fold's training sentences.

**Claims.** The paper rests on C2, C17, C20–C26, C28 and C29 of
`docs/PAPER.md`; citation keys refer to `docs/references.bib`.

**Journal structure.** Section 1 (with Box 1) is the Introduction,
Sections 5 and 6 the Results, Sections 7 to 9 the Discussion, Sections 2
to 4 the Methods, and the Supplementary Information the supplementary
material. The main text is about 9,400 words with its five tables; for npj
Computational Materials the Related work folds into the Introduction and
the Limitations into the Discussion, and the AI-use statement (Ethics and
broader impact) moves into the Methods; for ACL Rolling Review the paper
needs cutting to 8 pages and anonymising.

**Before submission (the author's items).**
- The affiliation on the title page.
- The venue's word and abstract limits (the abstract is 200 words).
- The companion study (`docs/PAPER_B.md`) posted as a preprint or
  uploaded as a related manuscript, and cited by title where the text
  says "companion study".
- The AI-use statement: the name and version of the AI assistant, which
  the repository does not record.
- The commit this paper describes tagged, and an archived copy with a DOI
  deposited and cited in the Code availability statement.
- Optionally: release the rebuilt lexicons (lists of terms; at least the
  part from CC BY articles), and have a materials scientist judge 100
  names per type to estimate the lexicon's precision.

**Review history.** The draft of 30 October went through six adversarial
review rounds on 31 October (ROADMAP, week-15 row). Round 7 (1 November,
after the week-17 follow-up) led to a new title and abstract, a shorter
Introduction and Section 3, and the follow-up labelled as such. Round 8
restructured the paper into a main text and a Supplementary Information
file. Round 9 (a number audit, a journal referee recommending minor
revision, and a consistency check) led to: a Box of terms; Section 3's
disclosure cut and the gate on D3 and D4 dated (lifted again at 18:42,
after the screen had chosen MR6 + names); Table 3 with a status column
(the MaterioMiner comparison against MR3 pre-specified on 20 October,
not 18); the lexicon split's reading rule quoted and read both ways
(mixed on point estimates, name knowledge on intervals); the rule for
classing predicted spans as seen or unseen; the eighteen-decision count
scoped to the protocols from 17 October 20:21; the follow-up's table,
the training-size subsample and the earlier protocols moved to the
Supplementary Information; the companion study's protocols added to
Table S1; and Figure 3's legend merged and its cross-seed arrow dropped.
ROADMAP's round notes give each round's findings.
