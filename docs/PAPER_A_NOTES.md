# Paper A: notes on the draft (not part of the submission)

`docs/PAPER_A.md` is the submission text of paper A; this file keeps what
a reader of the repository needs beside it.

**Tables and figures to generated files.** The paper is the main text
(`docs/PAPER_A.md`) and the Supplementary Information
(`docs/PAPER_A_SI.md`, Notes 1 to 13, Tables S1 to S16). Table 1 is written
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
`table5_probe.md`; Table S16 and Table 2's P column to
`table16_decision_tests.md` (`decision_tests.py`, from
`results/gpu/ner_decision_tests.json`; reporting only, computed 1 November
for the journal's statistics rules); Supplementary Note 11
to the ROADMAP's session notes and `git log` (drafted with a source per
sentence, re-checked). The
push times of Supplementary Note 1 are `protocol_pushes.py`'s reading of
`results/github_push_activity.tsv` and `results/github_push_events.tsv`. The MaterioMiner paired unseen-name intervals of
Supplementary Note 5 are the `chain` block of
`results/gpu/ner_mentions.json`. Figure 1 is
`docs/figures/fig8_method.png`, Figure 2 `fig9_units.png`, Figure 3
`fig7_names.png`, Figure 4 `fig10_recall_matched.png`
(`src/graph/paper_figures.py`). The worked example's
lexicon entries (Methods, *Systems*) are the most frequent MAT and PROPERTY
entries of the electrolysis subfield's rebuilt lexicon in
`results/gpu_names/` (not committed); its MR6 replacement, Ni, is a MAT
mention of that fold's training sentences.

**Claims.** The paper rests on C2, C17, C20–C26 and C28–C30 of
`docs/PAPER.md`; citation keys refer to `docs/references.bib`.

**Journal structure.** Since 1 November the paper is in npj
Computational Materials' Article format (its content-types and
submission-guidelines pages, read 1 November): a title of 14 words
without punctuation, an abstract of 150 words, the Introduction (no
subheadings), the Results (subheadings), the Discussion (no subheadings;
it holds the former Limitations and Conclusion), the Methods
(subheadings), Data availability, Code availability, Acknowledgements,
Author contributions, Competing interests, References and Figure
legends. Earlier sections map as follows: Section 1 and Related work to
the Introduction; Box 1 to Methods, *Terms used throughout*; Sections 2,
3 and 4 to Methods, *Data*, *Evaluation protocol* and *Systems*;
Sections 5.1 to 5.6 to the first six Results subsections (*The
pre-registered decisions*, *Whose gain is it*, *The names' source and the
lexicon's settings*, *What the names do*, *SOFC-Exp: no gain shown*, *The
names on the strongest system*); Section 6 to Results, *What else was
tried, and the count*; Sections 7 to 9 to the Discussion; the Ethics and
broader impact statement to Methods, *Use of an AI assistant*. The review
history below keeps the section numbers of its time. The source files
keep citation keys; `python3 src/graph/paper_render.py` writes the
submission copies, `docs/submission/PAPER_A.md` and
`docs/submission/PAPER_A_SI.md`, with the keys numbered in order of first
citation and the references in Nature style from `docs/references.bib`
(more than five authors: the first, then et al.); rerun it after any edit
of the source, then `python3 src/graph/paper_pdf.py`, which prints the
submission PDFs (`docs/submission/PAPER_A.pdf` with each figure above its
legend and the tables at the end, `PAPER_A_SI.pdf`);
`docs/submission/README.md` lists the files and `COVER_LETTER.md` is a
draft for the author.

**Before submission (the author's items).**
- The affiliation on the title page.
- The venue's limits: npj Computational Materials sets no main-text word
  limit for an Article (its content-types page, read again on 1 November
  in round 15: title up to 15 words without punctuation, abstract up to
  150 words, 60 references "though not strictly enforced", legends up to
  350 words, panels labelled a), b), c), all methods in the main text).
  The abstract is 143 words since round 15 (150 if hyphenated words count
  twice, since round 16 added "three-copy"); the main text is long (about 7,300 words without the Methods),
  and the round-11 referee advised 5,000 to 6,000, moving detail to the
  SI. The round-12 referee's list of what
  could move without loss: Section 3's clock times into a short table
  (with three sentences of prose), Table 6's per-subfield rows (Table S8
  and Figure 4 hold them), Section 5.6 to about 120 words (Supplementary
  Note 8 holds the rest), Section 6's label-propagation sentence to
  Supplementary Note 7, and Tables S11 and S12 as a machine-readable data
  file with only the twelve decisions in every measure kept in the SI.
- The title (1 November, the author's choice of a positive title within
  npj's 15 words without punctuation): "Harvesting names from unlabeled
  papers to augment materials named entity recognition on unseen
  documents" (14 words). It names the method and its aim without
  claiming that the names improve recognition, which D2, D7, D9 and D10
  do not show (the round-11 referee also advised against "are learned").
  A positive alternative about what passed: "Preregistered tests show
  mention replacement improves materials named entity recognition on
  unseen papers" (13 words; D3, D4 and D5 are of MR6 + names, which
  contains mention replacement, and MR3's own pooled gain is post hoc, so
  this one needs care). The round-14 referee noted that an earlier option
  here ("Names harvested from unlabeled papers are not shown to improve
  materials entity recognition") contradicts D3 to D5: the names system
  does improve on the baseline; the negative result is the increment
  over mention replacement.
- The venue's format, done 1 November (Journal structure above), with
  the submission guidelines' rules (read 1 November): abstract 150 words;
  60 references (the limit, not strictly enforced); figure legends 64 to
  about 150 words (at most 350), each with a brief title, symbols written
  out and panels lettered; tables at the end of the submission copy (the
  renderer moves the blocks marked `<!-- table -->` with their notes);
  a Statistics and reproducibility subsection in the Methods with each
  decision's P value (Table 2, Table S16); no Supplementary Methods (the
  training settings, the corpus construction and the recall-matched
  control's design moved from Supplementary Notes 2, 3 and 10 into the
  Methods); the companion study cited as "H.M., in preparation"; author
  contributions by initials; competing interests and funding in the
  journal's forms. Still the author's: the cover letter (with the
  corresponding author's affiliation and contact, the related
  manuscript, suggested referees), the Reporting Summary and editorial
  policy checklist (encouraged at submission), and one PDF or Word file
  with the figures for the first submission.
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
item above. On 1 November the author chose npj Computational
Materials and a positive title, and asked that the headline stay
positive; the paper was converted to the journal's format (Journal
structure above). Round 14 (a referee recommending minor revision, a
number audit, a consistency check, all on the round-13 text) led to the
abstract leading with what passed (the pooled +3.58 and every
MaterioMiner paper at both granularities) while stating the registered
criterion in full, the criterion's mid-study adoption and the names'
unestablished F1 increment; the contribution statement without the post
hoc result credited to D11; D11 with its four-subfield reading
everywhere and Table 2's "passes (as registered)"; macro and weighted F1
together, with every decision's votes; the corpus-weighted Welch check;
the 05:03 cancel request; the practitioner advice scoped and given MR3's
pre-specified 16-unit estimate; two manifest times corrected; and the
figures' panel letters. ROADMAP's round-14 note lists every item.
Round 15 (a referee recommending major revision, a number audit and a
consistency check, all on `800a85a`, none finding a wrong number) led
to: an abstract that keeps the positive headline but gives mention
replacement's own +2.35, says the tests were pre-registered, calls the
first criterion "the original" (the pooled one was registered too) and
states the recall finding with D11, as the week-18 protocol fixed, with
the precision cost; the Introduction's contribution statement without
results and its results list as prose; the Discussion's opening and
advice scoped (not on MuLMS alone; F1 on the held names +1.13 [−1.84,
+4.11], post hoc, now in Tables 5 and S3); the Statistics subsection
with the design items, the pooled condition, D11's unit values, the
exceptions to the t intervals and a post hoc sensitivity analysis for
units sharing training documents (D3 [−0.13, +8.67]; D4, D5 and D11
stay above 0; Table S16's new column; Nadeau and Bengio cited, seqeval
moved to the text as a website); the other systems' methods moved from
the SI into the Methods; the registration disclosure corrected (which
references were new runs, the 05:06 plan, half-known decisions); D5′ and
D11 verdicts in Table S16; citations as superscripts; references
completed from their DOIs (Torres et al. now its 2026 journal version);
Data availability with the datasets' locations; the AI assistant's work
in the Methods only. ROADMAP's round-15 note lists every item.
Round 16 (a referee recommending major revision on substance, not on
accuracy; a number audit and a consistency check finding no wrong number)
led to: D7's seed basis in the abstract; F1 on the held names computed from
counts; the dictionary probe's recall described correctly; MR3's 16-unit
pool and the companion's MaterioMiner decisions given their true status
(11 of 16 units known; the screen seeds averaged in, +1.73 [−0.63, +4.08]
coarse-grained without them); where each pass of the dependence check
vanishes; P values for every inference drawn from a non-decision interval;
the votes, ablations and formulas defined in the Methods; American
spelling; Nature-style journal abbreviations; a fuller cover letter.
ROADMAP's two round-16 notes list every item.
Round 17 (a number audit and a consistency check of `0a8a150`, both
finding the builds reproducible) led to:
- D5's tipping correlation corrected to 0.45. The bisection had run on
  the rounded bound.
- Every comparison fixed with part of its results known labeled so: the
  13.3 and 13.4 pools, and the MuLMS comparisons over all ten seeds of
  29 October.
- P values at every remaining inference.
- D5′ and the subfields' short names defined at first use.
- SI Note 12 listing the post hoc readings of 1 November.
- The last British spellings removed.
- Three reference details fixed.

ROADMAP's round-17 note lists every item.
