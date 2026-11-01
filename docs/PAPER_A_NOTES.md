# Paper A: notes on the draft (not part of the submission)

`docs/PAPER_A.md` is the submission text of paper A; this file keeps what
a reader of the repository needs beside it.

**Tables and figures to generated files.** The paper is the main text
(`docs/PAPER_A.md`) and the Supplementary Information
(`docs/PAPER_A_SI.md`, Notes 1 to 13, Tables S1 to S15). Table 1 is written
by hand from the conversions in `src/graph/`; Table 2 to
`docs/tables/table6_held_out_decisions.md` and `table2_decisions.md` (D9
and D10 to `table11_week17.md`, D11 and D12 to `table12_week18.md`, whose
levels are the subfields' means of `per_unit` in
`results/gpu/ner_recall_matched.json`); Table 3 to `table9_robustness.md`; Table
4 to `results/gpu/ner_mentions.json` (and `table4_unseen.md` for the CRF
head); Table 5 to the paired-gains block of `table10_lexicon.md`, from
`results/gpu/ner_lexicon_recall.json` (`ner_lexicon_recall.py`; the
rebuilt lexicons themselves are not committed); Table 6 to
`table12_week18.md` (from `results/gpu/ner_recall_matched.json`,
`ner_recall_matched.py`; the parenthesised values are its
`post_hoc_nearest` block, the held-name recalls its `post_hoc_curves`
block, the last two rows its `post_hoc_decomposition` and
`names_minus_best`). Table S1 to the protocol
record in `docs/ROADMAP.md` §4 and `git log`; Table S2 to
`table7_ablations.md`; Table S3 to `table10_lexicon.md`; Table S4 to
`table8_types.md` (and `tableA2_types_fine.md` for MaterioMiner); Table
S5 to the week-12 screen in `docs/ROADMAP.md`; Table S6 to
`table11_week17.md`; Table S7 to `results/gpu/ner_robustness.json`
(`detectable`); Tables S8, S9 and S10 to the second and fourth, sixth
and seventh parts of `table12_week18.md` (Table S9's last column is
`extra_names` in `paper_tables.py`); Tables S11 and S12 to the two
parts of `table13_names_route.md` (`names_route_record.py`, every
comparison of the names route in every measure; the pooled rows' macro
and weighted cells, post hoc, to `pooled_measures` in
`results/gpu/ner_robustness.json`); Tables S13, S14 and
S15 to `table14_chain.md`, `table15_lexicon_units.md` and
`table5_probe.md`; Supplementary Note 11
to the ROADMAP's session notes and `git log` (drafted with a source per
sentence, re-checked). The
push times of Supplementary Note 1 are `protocol_pushes.py`'s reading of
`results/github_push_activity.tsv` and `results/github_push_events.tsv`. The MaterioMiner paired unseen-name intervals of
Supplementary Note 5 are the `chain` block of
`results/gpu/ner_mentions.json`. Figure 1 is
`docs/figures/fig8_method.png`, Figure 2 `fig9_units.png`, Figure 3
`fig7_names.png`, Figure 4 `fig10_recall_matched.png`
(`src/graph/paper_figures.py`). The worked example's
lexicon entries (Section 4) are the most frequent MAT and PROPERTY
entries of the electrolysis subfield's rebuilt lexicon in
`results/gpu_names/` (not committed); its MR6 replacement, Ni, is a MAT
mention of that fold's training sentences.

**Claims.** The paper rests on C2, C17, C20–C26 and C28–C30 of
`docs/PAPER.md`; citation keys refer to `docs/references.bib`.

**Journal structure.** Section 1 (with Box 1) is the Introduction,
Sections 5 and 6 the Results, Sections 7 to 9 the Discussion, Sections 2
to 4 the Methods, and the Supplementary Information the supplementary
material. The main text is about 11,200 words to the Limitations with its six tables; for npj
Computational Materials the Related work folds into the Introduction and
the Limitations into the Discussion, and the AI-use statement (Ethics and
broader impact) moves into the Methods; for ACL Rolling Review the paper
needs cutting to 8 pages and anonymising.

**Before submission (the author's items).**
- The affiliation on the title page.
- The venue's word and abstract limits (the abstract is 150 words since
  round 13; the main text is about 11,200 words to the
  Limitations, and the round-11 referee advised 5,000 to 6,000 for an npj
  Article, moving detail to the SI). The round-12 referee's list of what
  could move without loss: Section 3's clock times into a short table
  (with three sentences of prose), Table 6's per-subfield rows (Table S8
  and Figure 4 hold them), Section 5.6 to about 120 words (Supplementary
  Note 8 holds the rest), Section 6's label-propagation sentence to
  Supplementary Note 7, and Tables S11 and S12 as a machine-readable data
  file with only the twelve decisions in every measure kept in the SI.
- The title: the round-11 referee advised keeping it (accurate for F1)
  and not adding "are learned", which the matched control cannot show on
  its own (Section 5.4). npj Computational Materials asks for at most 15
  words without punctuation (the current title has 19 and a colon);
  options in that form: "Names harvested from unlabeled papers are not
  shown to improve materials entity recognition" (13 words) or "A
  preregistered test of harvested names for materials named entity
  recognition" (11).
- The venue's format. npj Computational Materials (content types page,
  read 1 November): no main-text word or display-item limit; abstract at
  most 150 words without subheadings (now 150); at most 60 references,
  not strictly enforced (now 61); figure legends at most 350 words each;
  the sections Introduction (no subheadings), Results (subheadings),
  Discussion (no subheadings, and no limitations or conclusions
  sections), Methods (subheadings, after the Discussion), then Data and
  Code availability, Author contributions, Competing interests,
  References and Figure legends. For that venue Sections 2 to 4 become
  the Methods, Related work folds into the Introduction, Box 1 and
  Section 6 into the Results, the Limitations and the Conclusion into the
  Discussion, and the AI-use statement into the Methods; the citation
  keys are rendered as numbered references from `docs/references.bib`.
- The companion study (`docs/PAPER_B.md`) posted as a preprint or
  uploaded as a related manuscript, and cited by title where the text
  says "companion study".
- The AI-use statement: the name and version of the AI assistant, which
  the author states (each commit's session trailer identifies the
  session that made it); and, if true, that the author critically revised
  and approved the final text (Author contributions).
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
ROADMAP's round notes give each round's findings. Round 10 (1 November; a number audit,
a referee recommending minor revision, a consistency check) led, in part
before the week-18 runs, to: the abstract's MR3 interval on D5's seed
basis; Box 1's teachers and paired and pooled terms made exact; Section
3's second and third disclosures restated with times; Table 3's status
column completed; the SOFC-Exp paragraph made precise about what is not
shown; the AI-use and author-contribution statements; and, after the
week-18 result, the abstract's last sentence, the Conclusion's pooled
wording with its intervals, and Section 6 cut to its count, its detail
moved to Supplementary Note 7. The week-18 follow-up (the recall-matched
control, D11 and D12) added Table 6, Supplementary Note 10, Table S8 and
the activity-log push times (every protocol commit on GitHub 1 to 7
seconds after its commit time). Round 11 (a referee recommending major
revision, text only) reworded the D11 claim with its caveats and added
D11's decomposition, the best-bias comparison and the split within types
(post hoc); the completeness audit that followed added Supplementary
Notes 11 to 13 (the run record, every comparison in every measure, the
unseen-name trade, the lexicon split per unit and the probe). Round 12 (a
referee recommending minor revision, text only; a number audit; a
consistency check) led to: the abstract rewritten (the original criterion's
failure stated, D11 scoped to MuLMS and to its unmatched subfields);
Section 3's 20 October account with the 18:42 power reading, the 08:13
odds and the electrolysis runs, checked against the pulled files; the
pooled comparisons in macro and weighted F1 (post hoc, new; the names
against MR6 +0.83 [−0.01, +1.67] on the decision seeds, +1.12 [+0.21,
+2.03] on all ten); the mechanism read at D = 0 without "training copies";
D8's reading checked against its condition; four session times corrected
from the manifests; and Tables S9, S11 and S13 regenerated. Round 13 (a
referee recommending major revision of the presentation, not the
substance; a number audit) led to: the abstract cut to 150 words (npj's
limit) with the decision-seed basis named; the secondary measures given
on the decision seeds in both macro and weighted F1, with the pooled
macro gain over MR3 (+1.33 [+0.29, +2.36]) beside the micro statements
it qualifies; D11 stated with its four-subfield reading wherever it
appears, and the other unseen names' loss scoped (−0.88 [−2.63, +0.87]
on the four matched subfields); the Conclusion reordered, with the
criterion's caveat; Section 3's 20 October account as a dated list,
with the 18 October gate restored, the 13:35 reading and D6's widened
condition; D9 and D10's control noted as run in the same batch; the
internal reviews' role in the AI-use statement; the per-corpus
condition's weakness in the Limitations; BERT and AdamW cited, a
duplicate bibliography entry removed; session times sourced from the
manifests. The venue's format (npj: title, sections) is the author's
item above.
