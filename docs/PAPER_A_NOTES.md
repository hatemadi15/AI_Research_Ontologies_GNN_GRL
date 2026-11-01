# Paper A: notes on the draft (not part of the submission)

`docs/PAPER_A.md` is the submission text of paper A; this file keeps what
a reader of the repository needs beside it.

**Tables and figures to generated files.** Table 1 is written by hand
from the conversions in `src/graph/`; Table 2 to
`docs/tables/table6_held_out_decisions.md` and `table2_decisions.md` (D9
and D10 to `table11_week17.md`); Table 3 to `table9_robustness.md`; Table
4 to `table7_ablations.md`; Table 5 to `results/gpu/ner_mentions.json`
(and `table4_unseen.md` for the CRF head); Table 6 to
`table10_lexicon.md`, from `results/gpu/ner_lexicon_recall.json`
(`ner_lexicon_recall.py`; the rebuilt lexicons themselves are not
committed); Table 7 to `table11_week17.md`; Table 8 to the week-12 screen
in `docs/ROADMAP.md`; Table A1 to the protocol record in
`docs/ROADMAP.md` §4 and `git log`; Table E1 to
`results/gpu/ner_robustness.json` (`detectable`); Table F1 to
`table8_types.md` (and `tableA2_types_fine.md` for MaterioMiner). Figure 1
is `docs/figures/fig8_method.png`, Figure 2 `fig9_units.png`, Figure 3
`fig7_names.png` (`src/graph/paper_figures.py`).

**Claims.** The paper rests on C2, C17, C20–C26, C28 and C29 of
`docs/PAPER.md`; citation keys refer to `docs/references.bib`.

**Journal structure.** Section 1 is the Introduction, Sections 5 and 6
the Results, Sections 7 to 9 the Discussion, and Sections 2 to 4 with the
appendices the Methods; for npj Computational Materials the AI-use
statement (Ethics and broader impact) moves into the Methods. Before
submission: the affiliation on the title page, the venue's abstract
limit (the abstract is 199 words), and the name and version of the AI
assistant if the venue asks for them.

**Review history.** The draft of 30 October went through six adversarial
review rounds on 31 October (ROADMAP, week-15 row) and a seventh on 1
November after the follow-up of Section 5.7 (week 17), whose reviews (a
number audit, a journal referee and a consistency check) led to the
revision of 1 November: the title and abstract, a shorter Introduction
and Section 3, Section 5.7 rewritten, the per-type table moved to
Appendix F, the follow-up labelled as such, and the statistical wording
corrected where an interval had been read as no effect.
