# Harvested names are not shown to add to mention replacement for materials named-entity recognition on held-out documents: a pre-registered test

Hatem Madi (affiliation to be added; hatem.madi15@gmail.com)

## Abstract

Materials named-entity recognizers train on a few annotated papers but
must read new ones: our baseline scores 77 F1 on random splits of
MaterioMiner but 46 on a held-out paper, where 81% of the entities are
names no training sentence annotates. We test whether names harvested
from unlabeled papers help. Three baseline models tag 40,000 unlabeled
sentences; names they confidently agree on join the pool of mention
replacement, which swaps entity names in six copies of each training
sentence. Over 11 held-out subfields and papers of two corpora, this
gains 3.58 F1 points [95% CI 0.95, 6.21] over the baseline, under a
criterion adopted mid-study, which mention replacement alone already
met. Without harvested names, mention replacement gains 2.46 [1.00,
3.91] (post hoc); the pre-registered test of the names against a
six-copy control gives 0.80 [−0.66, 2.26], excluding gains above 2.3
points but not smaller ones. A follow-up registered afterwards, on
MaterioMiner's strongest system (multi-task training with mention
replacement), gives the names 0.96 [−0.01, 1.92] and 0.85 [−0.72, 2.42]
over a matched control at two type granularities, neither passing.
Mention replacement is worth using on new documents with few training
sentences; harvested names are not shown to add to it.

## 1 Introduction

Annotated materials corpora are small and come from few documents.
MaterioMiner (durmaz2024materiominer) annotates 4 papers; SOFC-Exp
(friedrich2020sofc) 45 articles; MuLMS (schrader2023mulms) 50 articles
from 7 subfields. A named-entity tagger trained on such a corpus is meant
for documents the corpus does not contain, and a random sentence split
does not measure that. Our baseline is MatSciBERT (gupta2022matscibert),
a BERT language model pre-trained on materials text, with a conditional
random field (CRF) output layer: the *CRF head* of Section 4. Scored by
micro F1 (the harmonic mean of precision and recall over entities matched
exactly in span and type, in percentage points; Section 3), it reaches
77.49 with MaterioMiner's fine-grained types (its 27 most frequent
classes; the coarse-grained level has 27 parent types, Table 1) over 5
random splits and 46.46 on a held-out paper; on MuLMS 65.72 over random
splits of the same size and 48.66 on a held-out subfield; on SOFC-Exp,
whose 45 articles share one topic and one annotation scheme, 86.27
against 82.79 on held-out articles. The gap is partly the names: 81% of
the entities of a held-out MaterioMiner paper are *unseen*, their words
annotated as no entity in any training sentence (88% on a held-out MuLMS
subfield, 46% on SOFC-Exp's held-out articles), and the CRF head finds
35% of them on MaterioMiner and 39% on MuLMS, against 71% and 66% of the
names it has seen. The held-out document's mix of names accounts for a
third to three fifths of the recall gap, by corpus and by the order of
the decomposition; the rest is lower recall on both kinds (companion
study).

This paper asks whether those names can be given to the tagger without
annotation, and answers with a negative result that we think is worth
publishing because of how it was obtained. Unlabeled materials text is
plentiful: a corpus of 40,000 sentences of open-access papers on a
held-out MuLMS subfield's topic holds 55–81% of its unseen test names.
Three CRF heads trained on the labeled sentences (the *teachers*) tag the
corpus; the names all three tag alike, each with every tag's marginal
probability at 0.9 or more, form a *lexicon*; and the lexicon enters
training through mention replacement (dai2020analysis), which copies each
training sentence six times with its entity mentions swapped for other
mentions of the same type, drawn from the training mentions and the
lexicon together (*MR6 + names*, Section 4). The model trained on the
augmented data, the *student*, never trains on a teacher-labeled
sentence: it sees only annotated sentences, some of whose names are
swapped for harvested ones.

Every test was fixed in a protocol committed to the repository before its
results (vanmiltenburg2021preregistering), with 7 to 10 training runs
(*seeds*) per held-out *unit*, a MuLMS subfield or a MaterioMiner paper,
and a decision rule stated in advance: a system improves on its reference
when the 95% interval of the mean paired difference in micro F1 over the
units lies above 0. A number in square brackets is that 95% Student-t
confidence interval, over the units named. The protocols are named after
the stage of the project, which the repository calls a week, in which
they were fixed; Appendix A dates each. The last four protocols of the
*names route*, the sequence of protocols that tested harvested names,
took ten such *decisions*, D1 to D10 (Table 2). The results, with the
qualifications Section 3 records:

- MR6 + names improves on the CRF head pooled over the 11 units of MuLMS
  and MaterioMiner (D5: +3.58 [+0.95, +6.21]; Section 5.1), under a pooled
  criterion adopted mid-study, when plain mention replacement was already
  known to meet it (Section 3). MuLMS alone does not pass (+3.19 [−1.13,
  +7.51]).
- That gain is not shown to be the names'. On the decision seeds, where
  MR6 + names also gains +3.58 (D5′, [+1.04, +6.13]), mention replacement
  with three copies and no names (MR3) gains +2.46 [+1.00, +3.91] (post
  hoc; +2.35 [+0.94, +3.75] on all ten seeds); three more copies add
  +0.33 [−0.25, +0.92] and the names +0.80 [−0.66, +2.26] (D7, the
  pre-registered test of the names), an interval that excludes a gain
  above 2.3 points but not a smaller one (Section 5.2).
- Neither the names' source nor the filter's settings changes F1
  detectably (Section 5.3). On MuLMS the names raise recall on unseen
  names (+2.56 [+0.76, +4.36], paired by subfield) and lower precision in
  pooled rates, a loss not established when paired; the recall gain falls
  mostly on the fifth of the unseen names the lexicon holds (post hoc,
  Section 5.4). On SOFC-Exp, where half the test names are seen, nothing
  helps (Section 5.5).
- A follow-up registered after these results tested the names on
  MaterioMiner's strongest system, multi-task training with mention
  replacement: over a six-copy matched control they add +0.96 [−0.01,
  +1.92] fine-grained and +0.85 [−0.72, +2.42] coarse-grained (D9, D10),
  neither passing (Section 5.7).

Three of the ten decisions pass, and they share MaterioMiner's four
papers (D3 and D4 at its two granularities, and D5, whose 11 units
include them); none would survive a Bonferroni correction (Section 5.1).
Section 6 lists what else was tried on the way.

## 2 Data

Table 1 gives the three corpora; the short names in parentheses are the
ones the figures and tables use. All three corpora are released under
open licences; our conversions and the held-out protocol are in the
repository.

**Table 1.** The three corpora: documents and sentences, annotation, the
held-out unit with its test size, and the training set per unit.

| Corpus | Documents, sentences | Annotation | Held-out unit (test size) | Training per unit |
|---|---|---|---|---|
| MaterioMiner (durmaz2024materiominer), release v1.0.1, CC BY 4.0 | 4 papers on fatigue of metals (s11340, ncomms7164, s41467, met10111458: 41, 81, 201, 153 sentences) | the classes of the dataset's materials mechanics ontology (MMO) at two levels: fine-grained, the 27 classes with most entities, ties broken by token count then name (the 27th and 28th tie at 24 entities), the other classes' entities relabelled O (1,328 entities; e.g. Alloy, Value, Mechanism, StateDescriptor, FatigueTestSpecimen, GrainBoundary); coarse-grained, 27 parent types (2,194 entities) | one paper (41–201 sentences) | the other three papers (275–435 sentences, of which 15% validate: 234–370 training sentences) |
| MuLMS (schrader2023mulms), Hugging Face `timo-pierre-schrader/MuLMS` at revision bc1c97d, CC BY-SA 4.0 | 50 articles from 7 materials subfields, 10,184 sentences: water electrolysis (electrolysis; 4 articles, 992 sentences), graphene (2, 351), polymer-electrolyte-membrane fuel cells (pemfc; 8, 1,638), polymers (poly; 11, 2,384), semiconductors (semi; 11, 2,040), solid oxide fuel cells (sofc; 4, 928), steel (10, 1,851) | 13 types (MAT, VALUE, PROPERTY, FORM, MEASUREMENT, CITE, SAMPLE, TECHNIQUE, DEV, INSTRUMENT, NUM, UNIT, RANGE); we keep the outermost of its nested entities, 31,952 of 46,584 | one subfield, every sentence of its articles (351–2,384) | 400 sentences of the other subfields' articles, taken in a shuffled order of those articles until 400 are reached (two or three articles, the last cut; a different set for each held-out subfield); 340 train and 60 validate, MaterioMiner's size by design |
| SOFC-Exp (friedrich2020sofc), `boschresearch/sofc-exp_textmining_resources`, CC BY 4.0 | 45 articles on solid oxide fuel cells; 876 of their 9,466 sentences carry entity annotation (the ones that describe experiments) | 4 types: MATERIAL, VALUE, DEVICE, EXPERIMENT | 9 articles, one of 5 folds (137–248 sentences) | the other 36 articles (534–628 training and 94–111 validation sentences) |

Notes to Table 1. MaterioMiner's papers are DOIs 10.1007/s11340-021-00758-x
(s11340), 10.1038/ncomms7164, 10.1038/s41467-020-19071-7 (s41467) and
10.3390/met10111458. MuLMS's kept entities by type: MAT 12,647, VALUE
4,628, PROPERTY 3,880, FORM 3,180, MEASUREMENT 2,106, CITE 1,704, SAMPLE
1,391, TECHNIQUE 920, DEV 757, INSTRUMENT 370, NUM 260, UNIT 91, RANGE 18.

Three details matter below. On MuLMS a flat tagger needs one level, so
31.4% of the entities are dropped with the conversion (14,632 of 46,584:
nested in a kept entity, sharing its span or crossing it), and text
extraction had damaged 276 tokens, which we repair; our MuLMS scores cannot be compared with the published nested
ones. On MuLMS the training pool is capped at 400 sentences on purpose,
so that MuLMS tests MaterioMiner's regime, a few hundred annotated
sentences from two or three documents (SOFC-Exp, trained on 36 articles,
does not; Section 5.5); the other subfields hold about
8,000 labeled sentences that no run here uses (Limitations). SOFC-Exp's official test articles
are tokenized differently from its training articles; the article folds
used here cut every article with the training articles' tokenizer.

**Unseen names.** A test entity is unseen when its words are annotated as
no entity, of any type, in any training sentence, and seen when some
training sentence annotates them with its type
(augenstein2017generalisation); entities whose words are annotated only
under another type, 0.0–2.1% of a corpus's test entities (up to 4.5% of
one paper's), count as neither. Recall
and precision are reported by kind of name, pooled over seeds and units.

## 3 Evaluation protocol

**Seeds and scoring.** Every system compared in a decision has 10 model
seeds per held-out unit (the random seed of the training run, offsets 0,
100, …, 900), except where a protocol reserved the screen's 3 seeds (0 to 200)
for selection and decided on the other 7 (300 to 900); the mixed-corpus
variant has those 7 seeds and the ablations the screen's 3. The score is micro F1 at the entity
level: an entity counts as found only with its exact span and type;
precision is the share of predicted entities that are correct, recall the
share of gold (annotated) entities found, F1 their harmonic mean, micro over all
entities of a test set (seqeval in its default mode, as the datasets' own
code scores), in percentage points.

**Terms.** A *decision* is a comparison with a pre-registered verdict; a
comparison labelled *reported* was fixed in a protocol but carries no
decision; *post hoc* means computed after its results were known. A
*screen* is a 3-seed comparison used to choose among variants; the
*decision seeds* are the 7 seeds (300 to 900) a protocol decided on
after a screen on the other 3. A *vote* is the majority vote, entity by
entity, of the three models of a seed triple; votes appear here only
where a protocol decided on them (Section 6). The *MaterioMiner analogue*
of a MuLMS system is the same system run on MaterioMiner's held-out
papers with the fatigue corpus. The *plan* is the project's methods plan,
a document of the repository that set the route and its stop rule before
the protocols. The *sign-flip test* is the exact permutation test of a
mean paired difference over all 2^n assignments of signs to the n units,
two-sided. Protocols are named by the stage of the project in which they
were fixed; the repository calls the stages weeks, though the protocols
of weeks 10 to 14 were fixed between 17 and 29 October 2026 and their
last runs ended on 31 October, and the *follow-up* protocol of week 17
was fixed on 31 October, after the results above had been written up,
and run that night (Appendix A).

**Decision rule.** A method improves on its reference when the 95%
Student-t interval of the mean paired micro-F1 difference lies above 0,
the pairs being the held-out units (7 subfields, 4 papers, 5 folds), each
side the mean of its seeds. With 4 papers the interval has 3 degrees of
freedom; with 7 subfields, 6. A pooled decision takes the units of two
corpora together (11 units, 10 degrees of freedom) and requires, in
addition, that neither corpus's own interval lie below 0, a guard that
trips only on a significant loss. Each decision is one-sided at 2.5%, so
of the ten decisions about 0.25 would pass if nothing worked (0.45 of the
eighteen counted in Section 6); the decisions are not independent
(Section 5.1). What the design can detect depends on
each comparison's spread. From the observed intervals, at 80% power and
the two-sided 5% level: the names against MR6 detect 2.0
points pooled over the 11 units and 2.8 on MuLMS alone; the names against
the CRF head 3.5 to 3.7 pooled and 5.9 on MuLMS alone (D1 failed at
+3.19); the
source decision 2.5; the 3-seed ablations 2.0 to 3.1 (Appendix E lists
every comparison).

**Pre-registration, and what changed.** Each protocol was committed to
the repository before its results, all but one before its runs, with its
decisions and the comparisons reported without a decision; amendments
are marked with their reason and time (Appendix A gives the commits and
the chronology). Three points must be disclosed.

First, the criterion for a methods claim was revised on 20 October, on
the author's own decision, recorded in the protocol, to push for a
methods paper. The week-12 protocol (18 October) had required a pass
against the CRF head on MuLMS alone plus a pass on MaterioMiner, and the
plan carried a stop rule, "no fourth protocol on MuLMS". The week-13
protocol added a pooled decision over both corpora (D5) as a second
route to the claim and lifted the stop rule. At that moment half of the
screen's results were known, but not those of MR6 + names, and the
protocol itself computed, from runs already made, that plain mention
replacement would pass the new pooled criterion (+2.34 [+0.94, +3.75],
labelled post hoc; +2.35 in Section 5.2 by the analysis's rounding). The
criterion under which MR6 + names passes was thus adopted knowing that
mention replacement already met it, before the names system's own pooled
result existed. Three amendments followed the same day, all before any
decision run of the names route (Appendix A). Under the week-12 criterion
MR6 + names has no claim, since its MuLMS decision failed; under the
revised one it passes.

Second, the pooled decision's protocol text named MuLMS's decision batch
(seeds 300 to 900) as the source of its units and, in the same
paragraph, "each unit its 10-seed mean"; the analysis as executed took
the 10-seed means, which include the screen's three seeds. We report
both readings, which agree (D5, +3.58 [+0.95, +6.21]; D5′, +3.58 [+1.04,
+6.13]), quote the executed one first, and use the 7-seed basis for
the post hoc decomposition, since those seeds the screen never saw.

Third, the protocol of D7 and D8 (week 14) was committed two minutes
after its first batch was pushed and five hours before any result
arrived, and that of D9 and D10 (week 17) after the results above had
been written up, before its runs (Section 5.7). Every comparison of the
names route not in Table 2 is either reported (fixed in a protocol
without a decision) or post hoc, and labelled so; the decisions of
earlier protocols quoted here (mention replacement's, the multi-task
model's, the label-propagation rule's) are labelled where they appear,
and the robustness checks of Sections 5.1 and 5.2 are post hoc.

**Runs.** Kaggle T4 GPUs with deterministic CUDA, one frozen commit per
batch, a pinned image and pinned libraries; model snapshots pinned by
commit. From week 5 on, every GPU session but two first repeated an
earlier run, and every repeat over weeks 4 to 17 was identical; of the
two, one week-5 batch ran none, and the second week-17 session's teachers
reproduced the stored multi-task runs' scores instead. Hyperparameters
and compute are in Appendix B.

## 4 Systems

Figure 1 sketches MR6 + names, the last of the four systems compared
throughout:

- **CRF head.** MatSciBERT (gupta2022matscibert) with a token classifier on
  the first sub-word of each word and a linear-chain conditional random
  field (CRF) over the words, constrained to the IOB2 tag grammar (a tag
  that begins an entity, continues it, or is outside one), trained by
  likelihood and decoded with Viterbi, the highest-scoring tag sequence
  (lafferty2001conditional;
  lample2016neural). AdamW at 5e-5 with
  linear warm-up, batch size 16, up to 20 epochs, early stopping on
  validation F1 with patience 3. It is the reference of every decision. It
  is not the strongest single model we know on every corpus: on
  MaterioMiner's held-out papers, multi-task training over both
  granularities gains +3.72 [+0.57, +6.87] over it and multi-task training
  with three-copy mention replacement +6.73 [+6.03, +7.43] fine-grained (our
  companion evaluation study); neither transfers to SOFC-Exp, and MuLMS
  has one type level, so the CRF head is the reference the three corpora
  share. The names were tested on that system, with six copies, in a
  follow-up (Section 5.7).
- **MR3.** Mention replacement (dai2020analysis): three copies of each
  training sentence with an entity, each mention replaced with
  probability 0.5 by a training mention of the same type; copies that
  change nothing are dropped. The student trains on the gold sentences and
  the copies with the CRF head's training settings and seed.
- **MR6.** The same with six copies: the matched control for the names
  (the control, below).
- **MR6 + names.** MR6 whose replacement pool also holds a lexicon
  harvested from unlabeled text:
  - *Teachers.* The CRF head's own runs of seeds 0, 100 and 200 on the
    fold's training sentences, so the teachers are the baseline, not a
    stronger model.
  - *Corpus.* 40,000 sentences of open-access full texts from Europe PMC
    found by title terms of the held-out subfield (for MaterioMiner, of
    fatigue; for SOFC-Exp, of solid oxide cells), with the datasets'
    own articles excluded by identifier and every sentence sharing a word
    8-gram with a dataset sentence dropped (for MuLMS also a 50-character
    run); 290–368 articles with text per corpus (Appendix C). A topic
    corpus holds 55–81% of a held-out MuLMS subfield's unseen test names,
    the fatigue corpus 50–62% of a held-out paper's fine-grained (49–65%
    of its coarse-grained), the SOFC corpus 36% of a fold's (measured before the runs and
    recorded in the week-13 protocol).
  - *Lexicon.* Each teacher tags the whole corpus and returns, per word,
    the probability the model assigns to the tag it chose (the
    forward–backward marginal of the Viterbi tag). A
    mention joins the lexicon when the three teachers tag the same span
    with the same type and every tag of the span has a marginal of at
    least 0.9 under each teacher, whatever the rest of the sentence. Per
    held-out MuLMS subfield this admits 8,328–50,869 mentions
    (4,380–13,395 distinct names), which hold 22.2% of the held-out
    subfield's unseen test names (7–36% per subfield), about a third of
    what the corpus holds. The lexicon's precision was not estimated and
    its names were not stored with the runs (Limitations).
  - *Pool.* The training mentions (601–1,309 per MuLMS unit) and the
    harvested mentions, one entry per occurrence in the corpus, so that a
    draw is uniform over entries and a frequent name is drawn more often;
    harvested entries are at least 92% of a unit's pool (601–1,309
    training mentions against 8,328–50,869 harvested), so most
    replacements are harvested names.

Every run stores its teachers' scores, the lexicon's counts and the share
of the unseen test names it holds, the corpus's hash and the filter. On a
T4 GPU a CRF-head run on MuLMS takes 1.6 minutes and an MR6 run 5; a
names student costs about 6 times the CRF head's run with the harvest
shared over its job's three or four students and about 14 times charged
to one student, not counting the corpus's construction, and the students
take two to three times as many updates as the CRF head (Appendix B).

**Figure 1.** MR6 + names. Three CRF-head teachers trained on the labeled
sentences tag a 40,000-sentence corpus; the mentions all three tag alike
with every marginal at 0.9 or more form the lexicon; the replacement pool
holds the training mentions and the lexicon; mention replacement writes
six copies of each training sentence with its mentions swapped; the
student trains on the gold sentences and the copies.

## 5 Results

### 5.1 The pre-registered decisions

Table 2 lists the ten decisions of the names route in the order they
were taken, with the seeds each used; the last two belong to the
follow-up of Section 5.7.

**Table 2.** The ten pre-registered decisions of the names route (the last
two the follow-up) in the order taken: corpus and units (with MuLMS's seeds), comparison, the
reference's and the system's mean micro F1, the mean paired difference
with its 95% CI, and the verdict.

| | Corpus, units (MuLMS seeds) | Comparison | Reference F1 | System F1 | Δ micro F1 [95% CI] | Verdict |
|---|---|---|---|---|---|---|
| D1 | MuLMS, 7 subfields (300–900) | MR6 + names vs CRF head | 48.54 | 51.73 | +3.19 [−1.13, +7.51] | no |
| D2 | MuLMS, 7 subfields (300–900) | MR6 + names vs MR3 | 50.81 | 51.73 | +0.92 [−1.63, +3.48] | no |
| D3 | MaterioMiner fine, 4 papers | MR6 + names vs CRF head | 46.46 | 50.72 | +4.27 [+1.39, +7.15] | passes |
| D4 | MaterioMiner coarse, 4 papers | MR6 + names vs CRF head | 58.60 | 60.38 | +1.77 [+0.67, +2.88] | passes |
| D5 | pooled: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR6 + names vs CRF head | | | +3.58 [+0.95, +6.21], neither corpus below 0 | passes |
| D5′ | the same with MuLMS seeds 300–900 (the protocol's batch; post hoc reading) | | | | +3.58 [+1.04, +6.13] | above 0 (post hoc) |
| D6 | SOFC-Exp, 5 article folds | MR6 + names vs CRF head | 82.79 | 82.75 | −0.04 [−1.21, +1.14] | no |
| D7 | pooled, 11 units (300–900) | MR6 + names vs MR6 | | | +0.80 [−0.66, +2.26] | no |
| D8 | MuLMS, 7 subfields (300–900) | own-topic corpus vs mixed corpus | 51.91 | 51.73 | −0.18 [−2.01, +1.65] | no |
| D9 | MaterioMiner fine, 4 papers (follow-up) | multi-task + MR6 + names vs multi-task + MR6 | 52.99 | 53.95 | +0.96 [−0.01, +1.92] | no |
| D10 | MaterioMiner coarse, 4 papers (follow-up) | multi-task + MR6 + names vs multi-task + MR6 | 60.40 | 61.25 | +0.85 [−0.72, +2.42] | no |

The reference F1 on MuLMS is the 7-seed mean (48.54); the 10-seed mean
quoted in Section 1 is 48.66. D5 and D5′ are two readings of one protocol
text, which named the decision batch's seeds as the units' source and, in
the same paragraph, "each unit its 10-seed mean" (Section 3): D5 is the
analysis as executed, D5′ the batch reading.

Three decisions pass, and they are not three pieces of evidence: D3 and
D4 are the same four papers at two granularities, and D5's eleven units
contain those four papers. The one independent corpus, MuLMS, fails every
single-model decision of MR6 + names (D1, D2), while its matched control
alone clears 0 there (MR6 − CRF head +2.68 [+0.13, +5.23] on the decision
seeds, a comparison reported in the week-14 protocol; Section 5.2); 6 of
its 7 subfields gain against the CRF head (sofc by 11.9, poly loses 2.4) and the differences have a
standard deviation of 4.7 points across subfields. On MaterioMiner every
paper gains at both granularities. The pooled estimate is stable (post
hoc): the mean of the two corpus means is 3.73; 9 of the 11 units are
positive on the executed 10-seed basis and 10 on the 7-seed basis; the
exact two-sided sign-flip permutation test gives p = 0.011 and 0.007; and
leaving any one unit out keeps the lower bound above 0 (lowest +0.44 and
+0.54, without semiconductors). Figure 2 shows every unit.

The decision rule carries no correction for the number of decisions.
Under a Bonferroni bound of 0.05/10 none of the three passes would hold:
the two-sided t p-values from the per-unit differences are 0.013 for D5
(0.011 for D5′), 0.018 for D3 and 0.014 for D4 (0.052 for D9 and 0.183
for D10). Mention replacement's post hoc pooled gain over the CRF head
(Section 5.2) has p = 0.004 on either seed basis, below 0.05/10, but as a
post hoc comparison it belongs to no corrected family.

**Figure 2.** Per held-out unit, the micro-F1 difference of MR6 + names
from the CRF head (circles) and from MR6 (squares): MuLMS's 7 subfields
on the decision seeds, MaterioMiner's 4 papers at both granularities and
SOFC-Exp's 5 folds, where MR6 was not run. The bands, drawn across the
11 units' rows, are the pooled intervals of D5′ and D7.

### 5.2 Whose gain is it

Table 3 gives every comparison among the four systems on the same 11
units, on both seed bases, and the MaterioMiner grids. In the first block
only D5 (10-seed basis) and D7 (decision seeds) are pre-registered, and
every other pooled entry is post hoc. In the second, D3 and D4 are
pre-registered, MR3 − CRF head holds the week-7 decisions, MR6 + names −
MR3 was reported in the week-12 protocol, and the other rows were
reported in the week-14 protocol.

**Table 3.** Every comparison among the four systems: over the 11
held-out units on both MuLMS seed bases (top) and over MaterioMiner's
four papers per granularity (bottom); the mean paired difference of micro
F1 with its 95% CI and, for the 11 units, the units above 0 and the
exact two-sided sign-flip p-value; provenance in the text above.

| Comparison, 11 units | MuLMS seeds 300–900: Δ micro F1 [95% CI] | units above 0 | sign-flip p (two-sided) | MuLMS seeds 0–900: Δ micro F1 [95% CI] | units above 0 | sign-flip p |
|---|---|---|---|---|---|---|
| MR3 − CRF head | +2.46 [+1.00, +3.91] | 11/11 | 0.001 | +2.35 [+0.94, +3.75] | 10/11 | 0.003 |
| MR6 − MR3 | +0.33 [−0.25, +0.92] | 8/11 | 0.23 | +0.22 [−0.30, +0.74] | 8/11 | 0.37 |
| MR6 − CRF head | +2.79 [+1.20, +4.38] | 10/11 | 0.002 | +2.56 [+0.94, +4.18] | 10/11 | 0.005 |
| MR6 + names − MR6 (D7) | +0.80 [−0.66, +2.26] | 7/11 | 0.24 | +1.02 [−0.56, +2.59] | 6/11 | 0.18 |
| MR6 + names − MR3 | +1.13 [−0.51, +2.76] | 7/11 | 0.16 | +1.23 [−0.22, +2.68] | 8/11 | 0.09 |
| MR6 + names − CRF head (D5′; D5) | +3.58 [+1.04, +6.13] | 10/11 | 0.007 | +3.58 [+0.95, +6.21] | 9/11 | 0.011 |

| MaterioMiner, 4 held-out papers (10 seeds) | fine-grained [95% CI] | coarse-grained [95% CI] |
|---|---|---|
| MR3 − CRF head | +2.77 [+1.51, +4.03] | +1.28 [+0.06, +2.51] |
| MR6 − MR3 | +0.20 [−1.47, +1.88] | −0.21 [−2.71, +2.29] |
| MR6 − CRF head | +2.98 [+0.09, +5.86] | +1.07 [−1.82, +3.97] |
| MR6 + names − MR6 | +1.29 [−2.51, +5.10] | +0.70 [−1.60, +3.00] |
| MR6 + names − MR3 | +1.50 [−1.76, +4.75] | +0.49 [+0.06, +0.92] |
| MR6 + names − CRF head (D3; D4) | +4.27 [+1.39, +7.15] | +1.77 [+0.67, +2.88] |

The +3.58 decomposes, on the decision seeds, into +2.46 from mention
replacement with three copies, +0.33 from three more copies and +0.80
from the harvested names; only the first term has an interval above 0,
and that post hoc (mention replacement's own pre-registered single-model
decisions passed on MaterioMiner and failed on MuLMS alone, where its
votes passed; Section 6). The
10-seed basis gives the same picture (+2.35, +0.22, +1.02). Against MR3, then,
the whole increment of MR6 + names is +1.13 [−0.51,
+2.76], and the pre-registered pass over the CRF head is a property that
mention replacement already has on these units: its own 11-unit gain,
computed post hoc from runs that predate the names route, is +2.35
[+0.94, +3.75] on the 10-seed basis, with 10 of 11 units positive. On
MuLMS alone, where D1 failed, MR6 gains +2.68 [+0.13, +5.23]
over the CRF head on the decision seeds (+2.33 [−0.27, +4.92] on the
10-seed basis) and +0.41 [−0.35, +1.17] over MR3, comparisons reported in
the week-14 protocol: on the decision seeds MR6 clears 0 on the one
independent corpus where MR6 + names did not. Per corpus the picture holds: on MaterioMiner, fine-grained, MR3 gains +2.77
[+1.51, +4.03] over the CRF head and MR6 + names +1.50 [−1.76, +4.75] over
MR3; coarse-grained, MR3 gains +1.28 [+0.06, +2.51] and MR6 + names +0.49
[+0.06, +0.92] over MR3 (reported in the week-12 protocol), an increment
that includes the three extra copies (the names' own increment over MR6
there is +0.70 [−1.60, +3.00], reported in the week-14 protocol).
That cell is one of three intervals in this paper comparing a names
variant with MR3 that lie above 0, with the 2-of-3-teachers
ablation on 3 seeds (+1.56 [+0.07, +3.05], Section 5.3) and SOFC-Exp
(+0.45 [+0.04, +0.86], Section 5.5); none is a pre-registered decision,
and the first has 3 degrees of freedom (Table 3, second block).

The names decision D7 was expected to be close in its protocol, which
foresaw a gain of +1.5 to +1.65 from the seeds in hand and, with a
between-unit spread near 3 points, an 11-unit half-width of about 2, and
put the chance of a pass at about 0.4; "not established" is thus the
designed outcome of a small effect, not evidence of none. The interval's
upper bound, +2.3, is nearly as large as mention replacement's whole gain
(+2.46); the comparison detects 2.0 points at 80% power. In the follow-up
on MaterioMiner's strongest system the names' increment is +0.96 [−0.01,
+1.92] fine-grained and +0.85 [−0.72, +2.42] coarse-grained (Section
5.7).

### 5.3 The names' source and the lexicon's settings

**Source (D8).** For each held-out subfield we built a mixed corpus of
40,000 sentences drawn equally from the six other subfields' corpora,
with the same source, pipeline and exclusions; it holds 49–70% of the
subfield's unseen test names against the topic corpus's 55–81%, and its
lexicon 18.3% against 22.2%. MR6 + names with the mixed corpus scores
51.91 against 51.73 with the topic's own, −0.18 [−2.01, +1.65] paired by
subfield (two subfields better with the own corpus, five with the mixed
one); the mixed variant gains +3.38 [−0.89, +7.64] over the CRF head and
+0.70 [−1.19, +2.58] over MR6 (both reported). No difference is detected; the interval runs
from −2.0 to +1.7 and the comparison detects 2.5 points (Appendix E).
The contrast is weaker than "source" suggests: the mixed corpus holds
most of the names the topic corpus holds (49–70% against 55–81% of the
unseen test names), so D8 tests the topic's marginal names, not an
off-topic source, and a null at this power was likely.

**Settings (reported, 3 seeds).** Each ablation (Table 4) changes one
setting and runs on the 7 subfields at the screen's seeds (0 to 200)
against the runs of MR6 + names on those seeds, which are the selected
variant's and so a slightly flattering reference.

**Table 4.** The lexicon's settings (reported, 3 seeds): distinct names,
the share of the unseen test names in the lexicon, micro F1 on the 7
subfields and the differences from MR6 + names, MR3 and the CRF head.

| Setting (the value in MR6 + names) | Distinct names (summed over the 7 subfields) | Unseen test names in the lexicon | Micro F1 | Δ vs MR6 + names [95% CI] | Δ vs MR3 | Δ vs CRF head |
|---|---|---|---|---|---|---|
| MR6 + names: 40,000 sentences, threshold 0.9, 3 of 3 teachers, a uniform pool | 54,095 | 22.2% | 52.09 | | +1.45 [−1.35, +4.25] | +3.16 [−1.96, +8.28] |
| corpus: its first 10,000 sentences | 19,171 | 16.4% | 51.94 | −0.15 [−1.78, +1.48] | +1.30 [−1.14, +3.74] | +3.01 [−1.21, +7.23] |
| corpus: its first 20,000 sentences | 32,461 | 20.1% | 51.93 | −0.16 [−2.45, +2.14] | +1.30 [−0.56, +3.15] | +3.00 [−0.49, +6.49] |
| filter: marginal threshold 0.7 | 77,879 | 31.5% | 51.93 | −0.16 [−1.62, +1.31] | +1.29 [−0.95, +3.53] | +3.00 [−1.85, +7.85] |
| filter: 2 of 3 teachers agree | 78,604 | 28.8% | 52.20 | +0.11 [−1.58, +1.81] | +1.56 [+0.07, +3.05] | +3.27 [−0.85, +7.38] |
| pool: harvested names in half of the draws (its uniform pool draws one in at least 92%) | 54,095 | 22.2% | 52.18 | +0.09 [−1.63, +1.80] | +1.54 [−0.74, +3.82] | +3.24 [−0.73, +7.22] |

Summed over the seven subfields, the lexicons range from 19 thousand to
79 thousand distinct names and from 16% to 32% of the unseen test names,
and micro F1 stays within 0.16
of the 52.09 of MR6 + names; every interval against it covers 0, and per
subfield the differences run from −3.2 to +2.9, the spread of 3-seed
means. These readings detect differences of 2 to 3 points: every
interval lies within −2.45 and +2.14, so larger effects are excluded and
smaller ones are not; the settings are not shown to be equal.
Within that limit, a quarter of the corpus (a quarter of the teachers'
tagging) with the strictest filter is not distinguishable from the rest, consistent with
Section 5.2: if the names add under a point, the settings
that govern how many names enter cannot show an effect. On the unseen
names the settings do move the trade of Section 5.4 (exploratory; the
screen's 3 seeds pooled over the subfields, against the screen runs of
MR6 + names at 48.1% recall and 50.0 precision, computed from their saved
predictions): smaller lexicons recall fewer at higher precision (10,000
sentences 46.9% at 52.3; 20,000 46.2% at 53.0; the names in half of the
draws 46.2% at 53.7), a lower threshold more at lower precision (0.7:
48.8% at 49.5), and 2 of 3 teachers a little more of both (48.3% at
50.6), while F1 on the unseen
names stays between 49.1 and 49.7.

### 5.4 What the names do

By kind of name, pooled over MuLMS's held-out subfields (Table 5; Figure 3
puts each system's recall and precision on the unseen names on one plane
with iso-F1 curves, curves of equal F1): the CRF head finds
39.4% of the unseen names at 55.8% precision; MR3 43.1% at 54.3; MR6
43.7% at 53.9; MR6 + names 47.3% at 50.9. Against the matched control the
names raise recall on unseen names by 3.6 points and lower precision by
3.0, a net +0.7 F1 on unseen names (48.3 to 49.0) and +0.2 on seen ones
(71.0 to 71.2); on the decision seeds alone the trade is 3.0 for 2.9 and
+0.5 (the note under Table 5); the point moves almost along an iso-F1
curve. Paired by subfield (post hoc, from the saved predictions), the
names' recall on unseen names is +2.56 [+0.76, +4.36] above MR6's on the
decision seeds and their precision −2.22 [−5.03, +0.60] below it, for
+0.56 [−1.44, +2.56] F1 (+3.24 [+1.36, +5.13] and −2.23 [−4.84, +0.38] over
all 10 seeds): the recall gain is established, the precision loss and the
net are not. Net of the predictions it drops, the names system makes 474
more unseen-name predictions per run than MR6 and 143 more correct ones,
a net rate of 30% (MR3's net additions over the CRF head, 43%; MR6's
over MR3, 33%); these are differences of pooled counts, not the precision
of the predictions one system makes and the other does not. With the
mixed corpus the figures are similar (46.7% at 52.4, against the topic
corpus's 46.9% at 51.3 on the same seeds). On MaterioMiner's
papers, fine-grained, the names add to both recall and precision on
unseen names (39.7 to 41.1 and 49.2 to 53.2 over MR6; F1 43.9 to 46.4),
and coarse-grained a little of each (53.0 to 54.2 and 52.4 to 53.2; F1
52.7 to 53.7).

**Table 5.** Recall, precision and F1 on the unseen test names, pooled
over seeds and MuLMS's seven held-out subfields.

| MuLMS, unseen names (88% of the test entities); 70 runs per row (seeds 0–900), 49 for the mixed corpus (seeds 300–900) | recall | precision | F1 |
|---|---|---|---|
| CRF head | 39.4 | 55.8 | 46.15 |
| MR3 | 43.1 | 54.3 | 48.10 |
| MR6 | 43.7 | 53.9 | 48.28 |
| MR6 + names, the topic corpus | 47.3 | 50.9 | 49.01 |
| MR6 + names, a mixed corpus | 46.7 | 52.4 | 49.35 |

On the decision seeds alone (49 runs per row) the topic corpus's row reads
46.9 / 51.3 / 49.0, MR6's 43.9 / 54.2 / 48.5 and the CRF head's 39.1 /
55.8 / 46.0, so the two corpora's rows compare on the same seeds.

**Figure 3.** Recall and precision on the unseen test names, pooled over
seeds and held-out units, for the CRF head, MR3, MR6 and MR6 + names
(with the topic corpus and a mixed corpus on MuLMS; with the fatigue
corpus on MaterioMiner), with iso-F1 curves; an arrow joins each system
to the one it builds on (the CRF head, MR3, MR6, then the names). A
point's label gives its F1 on the unseen names and, in parentheses, on
the seen ones; the MaterioMiner panels carry the scores only, and the
legend names the markers.

Per subfield, the lexicon's coverage
of the unseen test names runs from 7% (semiconductors) to 36%
(electrolysis), and the names' gain over MR6 on the decision seeds from
−2.3 (polymers) to +3.3 (solid oxide cells); the two correlate negatively
(Pearson r = −0.71, Spearman −0.61, 7 points), and the gain also
correlates with how weak the CRF head is on the subfield (r = −0.76 with
its F1; semiconductors at 32.5 and solid oxide cells at 43.5 gain most).
The two readings cannot be separated here. Coverage and the baseline's F1
are themselves correlated (r = 0.89); given F1, coverage keeps almost no
association with the gain (partial r = −0.10) while F1 given coverage
keeps some (−0.40), but with 7 points, 4 degrees of freedom for a partial
correlation, neither is distinguishable from 0, and the raw correlations
fall to −0.19 and −0.01 without the two subfields that drive them
(semiconductors and solid oxide cells). The raw correlation
with coverage runs against a simple dose–response in F1, though the
membership split below finds the recall gain on the held names in every
subfield; F1 also pays the precision the names cost. A mixed corpus
that covers fewer unseen names scores slightly higher on unseen-name F1
(49.35 against 49.0 on the same seeds) with lower recall (46.7 against
46.9) and higher precision (52.4 against 51.3).

**Which names the gain falls on (post hoc; the analysis and its reading
fixed before its run).** The lexicons were not stored with the runs. Week
15 rebuilt each from its unit's own teachers and corpus (the rebuilt
teachers' test F1 and the lexicon's coverage counts equal the stored
runs' in every unit) and split the unseen test names by whether the
lexicon holds them (Table 6). The reading was fixed in advance with three
outcomes: name knowledge if the names' recall gain over MR6 is larger on
the held names than on the others, the paired interval of the difference
above 0; a generic shift if the two gains are alike, the interval
covering 0 with both gains positive; mixed if the interval is above 0 but
the gain on the other names is also above 0. On MuLMS the lexicon holds
22% of the unseen test names pooled over the subfields (7% to 36% by
subfield). On those, MR6 + names recalls 82.4% against MR6's 75.0% (+7.29
[+3.27, +11.30], paired over the seven subfields); on the names it does
not hold, 36.8% against 35.0% (+1.12 [−0.78, +3.02]); the difference is
+6.17 [+2.44, +9.90], positive in every subfield: name knowledge by the
rule as fixed, which reads the mixed case only when the gain on the
other names clears 0. That gain, +1.12 [−0.78, +3.02], is not
established, and a smaller gain on the other names is not excluded; on
point estimates the case would read as mixed. The copies alone show no such split (MR6 over
the CRF head gains +3.39 on the held names and +5.22 on the others, a
difference of −1.84 [−5.27, +1.60]; MR3 −0.66 [−2.78, +1.45]), and the
mixed corpus's names give the same picture as the topic's (+4.86 [+1.50,
+8.21]). The precision side reads differently. By the pooled counts the
names add 8,336 predictions of held names, 39% of them right, and 12,206
of other names, 22% right, so two thirds of the wrong predictions they
add are names the lexicon does not hold, at about the rate of the
companion study's tagging bias (below); precision falls on both parts,
74.7% to 69.0% on the held names and 46.4% to 44.0% on the others. The
recall gain is name knowledge; the extra wrong predictions look like a
generic shift. On MaterioMiner the rule's letter gives a generic shift at
both granularities, both gains positive with the interval covering 0:
fine-grained +4.62 and +0.29, difference +4.33 [−5.33, +13.98];
coarse-grained +0.36 and +1.91, −1.54 [−3.60, +0.51]. We report that
verdict with one caveat, a post hoc judgement: the fine-grained interval,
−5 to +14, also covers every name-knowledge value MuLMS shows, so four
papers cannot tell the readings apart there. So on MuLMS the names do what
they were built to do, on the fifth of the unseen names the lexicon
holds, and the F1 increment stays within Section 5.2's noise because the
gain is confined to that fifth while the wrong predictions land on both
parts.

**Table 6.** Recall on the unseen test names by lexicon membership (post
hoc; lexicons rebuilt by the names method's own teachers; MuLMS on the
decision seeds, MaterioMiner on 10 seeds). Levels: per system, the share
of the unseen test names the lexicon holds (pooled over seeds and units),
recall on those and on the others, the share of the unseen-name
predictions that are lexicon names, and precision on each part. Paired
gains: per comparison, the recall gain on the held names and on the
others, the difference of the two gains, paired over the units with 95%
CIs, and the units where the gain on the held names is the larger.

**Levels**

| setting | system | unseen test names in the lexicon (%) | recall, in | recall, out | unseen predictions in the lexicon (%) | precision, in | precision, out |
|---|---|---|---|---|---|---|---|
| MuLMS, 7 subfields | CRF head | 22.2 | 71.8 | 29.7 | 30.7 | 74.3 | 47.6 |
| MuLMS, 7 subfields | MR3 | 22.2 | 75.6 | 34.1 | 28.4 | 74.1 | 46.3 |
| MuLMS, 7 subfields | MR6 | 22.2 | 75.0 | 35.0 | 27.5 | 74.7 | 46.4 |
| MuLMS, 7 subfields | MR6 + names | 22.2 | 82.4 | 36.8 | 29.0 | 69.0 | 44.0 |
| MuLMS, 7 subfields | MR6 + names (mixed corpus) | 22.2 | 81.6 | 36.7 | 28.8 | 70.7 | 45.0 |
| MaterioMiner fine, 4 papers | CRF head | 9.8 | 74.0 | 30.8 | 13.0 | 81.3 | 46.4 |
| MaterioMiner fine, 4 papers | MR3 | 9.8 | 83.1 | 34.0 | 13.0 | 82.7 | 46.3 |
| MaterioMiner fine, 4 papers | MR6 | 9.8 | 82.5 | 35.0 | 12.3 | 81.4 | 44.6 |
| MaterioMiner fine, 4 papers | MR6 + names | 9.8 | 86.6 | 36.1 | 13.6 | 81.2 | 48.8 |
| MaterioMiner coarse, 4 papers | CRF head | 18.2 | 79.6 | 45.4 | 17.7 | 82.2 | 45.5 |
| MaterioMiner coarse, 4 papers | MR3 | 18.2 | 79.1 | 47.2 | 17.3 | 82.8 | 46.6 |
| MaterioMiner coarse, 4 papers | MR6 | 18.2 | 80.5 | 46.9 | 17.8 | 81.5 | 46.1 |
| MaterioMiner coarse, 4 papers | MR6 + names | 18.2 | 80.7 | 48.3 | 17.5 | 82.4 | 47.0 |

**Paired gains**

| setting | comparison | recall gain on the names in the lexicon [95% CI] | recall gain on the names not in it [95% CI] | in − out [95% CI] | units with in > out |
|---|---|---|---|---|---|
| MuLMS, 7 subfields | MR6 + names − MR6 | +7.29 [+3.27, +11.30] | +1.12 [−0.78, +3.02] | +6.17 [+2.44, +9.90] | 7/7 |
| MuLMS, 7 subfields | MR6 + names − CRF head | +10.67 [+5.14, +16.20] | +6.34 [+2.60, +10.07] | +4.33 [+1.18, +7.48] | 6/7 |
| MuLMS, 7 subfields | MR6 − CRF head | +3.39 [+0.92, +5.85] | +5.22 [+1.92, +8.52] | −1.84 [−5.27, +1.60] | 3/7 |
| MuLMS, 7 subfields | MR3 − CRF head | +3.56 [+1.11, +6.01] | +4.22 [+0.87, +7.57] | −0.66 [−2.78, +1.45] | 3/7 |
| MuLMS, 7 subfields | MR6 + names (mixed corpus) − MR6 | +6.16 [+1.32, +11.01] | +1.31 [−1.07, +3.68] | +4.86 [+1.50, +8.21] | 7/7 |
| MaterioMiner fine, 4 papers | MR6 + names − MR6 | +4.62 [−5.27, +14.50] | +0.29 [−4.23, +4.81] | +4.33 [−5.33, +13.98] | 3/4 |
| MaterioMiner fine, 4 papers | MR6 + names − CRF head | +12.77 [−12.61, +38.15] | +6.00 [+2.84, +9.16] | +6.77 [−18.59, +32.13] | 2/4 |
| MaterioMiner fine, 4 papers | MR6 − CRF head | +8.15 [−7.65, +23.95] | +5.71 [+1.05, +10.37] | +2.44 [−15.96, +20.85] | 2/4 |
| MaterioMiner fine, 4 papers | MR3 − CRF head | +8.77 [−5.95, +23.49] | +4.17 [−0.17, +8.50] | +4.60 [−13.82, +23.03] | 3/4 |
| MaterioMiner coarse, 4 papers | MR6 + names − MR6 | +0.36 [−1.11, +1.84] | +1.91 [−1.46, +5.28] | −1.54 [−3.60, +0.51] | 0/4 |
| MaterioMiner coarse, 4 papers | MR6 + names − CRF head | +1.41 [−0.38, +3.19] | +3.62 [+1.18, +6.07] | −2.22 [−5.06, +0.62] | 0/4 |
| MaterioMiner coarse, 4 papers | MR6 − CRF head | +1.04 [−1.72, +3.80] | +1.72 [−0.37, +3.80] | −0.67 [−1.53, +0.19] | 0/4 |
| MaterioMiner coarse, 4 papers | MR3 − CRF head | −0.12 [−1.58, +1.34] | +2.32 [+0.88, +3.75] | −2.44 [−2.63, −2.24] | 0/4 |

A recall-matched control (a bias toward tagging) was not run against the
names; the nearest evidence is the
companion study's recall-calibrated bias on the CRF head's emissions,
not matched to the names and not run on MuLMS, which gave −0.59 [−1.75,
+0.57] fine-grained and +0.17 [−0.14, +0.48] coarse-grained on
MaterioMiner and −0.21 [−0.58, +0.17] on SOFC-Exp, the entities it added
being right 19–22% of the time, against the names' net rate of 30%
above. It is the first thing to do (Section 7).

### 5.5 Where nothing helps: SOFC-Exp

On SOFC-Exp's five folds of held-out articles MR6 + names with the SOFC
corpus scores 82.75 against the CRF head's 82.79, −0.04 [−1.21, +1.14]
(D6). It gains +0.45 [+0.04, +0.86] over MR3 on every fold (reported), but
MR3 itself does not gain there (−0.48 [−1.36, +0.39]; a week-13
decision, failed), nor on the official split (−0.29 [−1.04, +0.47]). Three things differ on this
corpus, in one place: only 46% of a fold's test names are unseen (81%
fine- and 79% coarse-grained on MaterioMiner's papers, 88% on MuLMS's
subfields) and the CRF head already finds 80% of them; the baseline is
at 82.8 F1 with a random-split gap of 3.5 points, so there is little
headroom; and the SOFC corpus holds 36% of the unseen test names
(27–43% per fold; measured before the runs and recorded in the week-13
protocol, not in a results file) against 55–81% on MuLMS, the filter admitting 20–33%.
Training size is not shown to explain mention replacement's null there: trained
on 10 of the official split's 26 training articles (5 draws of 156–248
sentences, MaterioMiner's size; 10 seeds each) and scored on its test
articles, MR3 gains +0.50 [−0.47, +1.46] over the CRF head, not
established (MR6 + names was not run at that size).
Pooled over the 16 units of the three corpora the estimate of MR6 + names is
+2.45 [+0.50, +4.40] (reported); the claim stays the two-corpus one, with
SOFC-Exp as its boundary condition.

### 5.6 Per entity type

Per type (exploratory, no decision; Appendix F), the pattern of Section
5.2 holds for most types. On MuLMS, MR6 + names gains over the CRF head
on the 12 types with a non-zero score, and MR6 holds more than half of
that gain on 9 of them; on coarse-grained MaterioMiner, MR6 + names gains
on 16 of 27 types and MR6 holds more than half of the gain on 13 of
those.

### 5.7 The names on the strongest system (follow-up)

An internal review of this draft (Ethics and broader impact) listed as
untested whether the names add to the strongest MaterioMiner system,
multi-task training with mention replacement (Section 4). A protocol for
that test was committed on 31 October at 19:15 UTC, after the results
above had been written up and before any run (Appendix A): two decisions
by the rule of Section 3, the names against a matched control on
MaterioMiner's four held-out papers over 10 seeds, fine-grained (D9) and
coarse-grained (D10); the protocol expected a pass to be unlikely. The
control is the multi-task model (one encoder, a CRF head per granularity)
with six-copy mention replacement; the names system adds a lexicon
harvested as in Section 4 but by three multi-task teachers (the stored
multi-task runs of seeds 0, 100 and 200, whose scores the jobs
reproduced), each tagging the fatigue corpus with both heads. A mention
is admitted when the three teachers agree on its coarse-grained span and
type and on its fine-grained type at exactly that span, or on its having
none, with every marginal at 0.9 or more on the coarse head and, where a
fine type is tagged, on the fine head; it joins the pool under its pair
of types, the key of multi-task mention replacement. Table 7 gives the
result.

**Table 7.** Follow-up: the names on the strongest MaterioMiner system.
Micro F1 over the four held-out papers (10 seeds; both granularities from
one model) with its 95% CI, the names against the matched control (D9,
D10) and both against multi-task training with three-copy mention
replacement, the multi-task model and the CRF head (reported).

| | System | micro F1 [95% CI] | Δ vs the control | Δ vs multi-task + MR3 | Δ vs multi-task | Δ vs CRF head |
|---|---|---|---|---|---|---|
| fine | multi-task + MR6 + names | 53.95 [47.52, 60.38] | +0.96 [−0.01, +1.92] (D9, no) | +0.76 [−0.15, +1.68] | +3.77 [+1.34, +6.21] | +7.49 [+5.96, +9.02] |
| fine | multi-task + MR6 (the control) | 52.99 [46.93, 59.06] | | −0.19 [−2.05, +1.67] | +2.82 [−0.00, +5.64] | +6.54 [+4.14, +8.94] |
| coarse | multi-task + MR6 + names | 61.25 [53.71, 68.80] | +0.85 [−0.72, +2.42] (D10, no) | +0.91 [−0.14, +1.97] | +3.21 [+0.46, +5.96] | +2.65 [−0.09, +5.38] |
| coarse | multi-task + MR6 (the control) | 60.40 [53.66, 67.15] | | +0.06 [−0.74, +0.86] | +2.36 [+1.15, +3.56] | +1.80 [+0.19, +3.40] |

Neither decision passes: fine-grained +0.96 [−0.01, +1.92] (two-sided t
p = 0.052; all four papers positive, 0.16 to 1.62, a sign-flip p of
0.125, the smallest four units allow), coarse-grained +0.85 [−0.72,
+2.42] (three papers positive). The names' whole increment over
multi-task training with three-copy mention replacement is +0.76 [−0.15,
+1.68] and +0.91 [−0.14, +1.97], and the three extra copies alone are not
shown to add (−0.19 [−2.05, +1.67] and +0.06 [−0.74, +0.86]): on this
system too what the names add is about a point and not established.
These comparisons detect 1.3 points fine-grained, tighter than the 3.4 to
3.8 the protocol expected from earlier comparisons on these papers,
because the four papers' differences lie close together (standard
deviation 0.6), and 2.1 coarse-grained, within the 1.4 to 2.4 it expected
(Appendix E). On the unseen test names the point estimates of recall and
precision both rise, fine-grained +0.58 [−2.93, +4.09] and +0.82 [−3.39,
+5.03], coarse-grained +1.45 [−0.69, +3.58] and +0.97 [−1.09, +3.03],
none established (reported; the paired intervals of Section 5.4); post
hoc, the fine-grained increment falls on the seen test names (F1 +2.43
[+0.40, +4.47], through precision) more than on the unseen ones (+0.43
[−1.57, +2.44]). Two caveats. The two-head filter admits lexicons of
16,455 to 42,066 entries per paper (3,746 to 6,587 distinct fine-typed
names) that hold fewer of the unseen test names than the single-task
lexicons of D3 and D4: pooled over the papers, 7.5% against 9.6%
fine-grained and 12.1% against 18.3% coarse-grained (lower in 7 of the 8
paper–granularity cells); whether that lowered the increment is not
known, and a looser filter was not tried. And a names job takes 82.5 to
122.7 minutes on a T4 (three teachers, their tagging of 40,000 sentences
with two heads and ten students): 10.0 minutes per student with the
harvest shared over the job's ten students and about 38 charged to one,
against 6.9 for a control run and about 2.4 for the two CRF-head runs
that give both granularities (Appendix B).

## 6 What else was tried, and the count

MR6 + names was reached through four protocols on MuLMS's held-out
subfields and their MaterioMiner analogues, each committed before its
runs, each failing or passing as recorded:

- *Mention replacement alone* (week 10): +2.10 [−0.31, +4.51] against the
  CRF head on MuLMS at 10 seeds, no; its votes +2.71 [+0.22, +5.20],
  passes. On MaterioMiner it had passed at both granularities (+2.77 and
  +1.28).
- *Names from the other subfields' unlabeled sentences* (week 11): the
  three teachers tag the labeled subfields' sentences outside the pool,
  with a sentence-level filter (a sentence is silver when all its tags
  agree at 0.9), whose names hold 4.4% of a held-out subfield's unseen
  test names (1–8% per subfield): +2.04 [−0.85, +4.92] against the CRF
  head and −0.06 [−1.09, +0.96] against MR3; on MaterioMiner, with the
  fatigue corpus and the same filter, +0.23 [−0.69, +1.14] and +0.61
  [−0.96, +2.18] against MR3 (2–9% of a paper's unseen names admitted).
  Four decisions, none passing; the route's reading at the time, that the
  filter and not the corpus was the bottleneck, is not borne out by
  Section 5.3.
- *Seven variants on the topic corpora* (week 12, a 3-seed screen on the
  7 subfields; Table 8, with the plan's labels, under which V5 was not
  assigned). The selection rule took the largest mean gain
  over MR3, which favours the noisiest variant; the screen decides
  nothing, and two of its losers deserve a 10-seed test that was not run:
  self-training (the student trains on the sentences the teachers labeled,
  not only on their names) on the corpus's silver sentences (V6) was the
  one variant
  whose interval against the CRF head cleared 0, by raising precision on
  unseen names (58.5 at 40.4% recall), and the lexicon as a gazetteer
  match feature of the CRF head, the classical use of a harvested
  lexicon, scored below MR3 on 3 seeds (V8's interval below 0, V7's
  covering it).

**Table 8.** The week-12 screen: seven variants on MuLMS's 7 held-out
subfields at 3 seeds, micro F1 and the differences from MR3 and the CRF
head (no decision).

| Variant (3 seeds, 7 subfields; no decision) | Micro F1 | Δ vs MR3 [95% CI] | Δ vs CRF head [95% CI] |
|---|---|---|---|
| MR6 + names, mention filter (V4, selected) | 52.09 | +1.45 [−1.35, +4.25] | +3.16 [−1.96, +8.28] |
| MR3 + names, mention filter (V2) | 51.60 | +0.97 [−0.91, +2.84] | +2.67 [−1.25, +6.59] |
| MR3 + names, sentence filter (V1) | 51.44 | +0.80 [−0.91, +2.51] | +2.50 [−2.09, +7.09] |
| self-training on the corpus's silver sentences (V6) | 50.80 | +0.16 [−2.31, +2.63] | +1.87 [+0.20, +3.53] |
| MR6, no names (V3) | 50.44 | −0.20 [−3.20, +2.80] | +1.50 [−1.65, +4.66] |
| lexicon as a gazetteer feature, mention filter (V7) | 48.91 | −1.73 [−4.13, +0.67] | −0.03 [−2.02, +1.97] |
| lexicon as a gazetteer feature, sentence filter (V8) | 48.79 | −1.85 [−3.59, −0.12] | −0.15 [−2.15, +1.86] |

- *The lexicon as a dictionary* (a CPU probe, no decision): applied to
  the CRF head's predictions as a rule, every verbatim match of a name
  tagged as its type where the head had tagged O, it helps where the
  head's recall trails its precision by 15 points or more (sofc +5.9,
  pemfc +4.0, semiconductors +2.3) and hurts elsewhere (steel −10.0,
  graphene −5.1), −1.0 on average over the seven subfields; its lexicons,
  from teachers trained on CPUs, differ from the runs' (25% of the unseen
  names on average against 23%), so it speaks to the dictionary rule
  only.
- *A second rule, label propagation within a document* (Appendix D): a
  term the model tags in most of its occurrences in a document is tagged
  in the rest. Two of its ten pre-registered decisions pass, on two
  corpora, both under a point; it is not shown to add to mention
  replacement (+0.57 [−0.12, +1.25] on MuLMS).

In all, the names route took ten pre-registered decisions in weeks 12
to 17, two of them in the follow-up, of which three pass, sharing one
corpus's four papers (Section 5.1). Counting also the six decisions of
weeks 10 and 11 and the two of week 13 on mention replacement without
names on SOFC-Exp's folds (13.3a, MR3, −0.48 [−1.36, +0.39]; 13.3b,
multi-task training with mention replacement, −0.11 [−0.69, +0.46]; both
failing), eighteen decisions tested the names or their
mention-replacement base, and four passed, all against the CRF head; of
the ten, none of the five against a mention-replacement control or on
the names' source passes. If no method worked, about 0.45 of eighteen
would be expected to pass, whatever their dependence, which only makes
chance passes come together, as the shared units of D3 to D5 do.

## 7 Discussion

**For practitioners.** With a few hundred annotated sentences and new
documents to read, use mention replacement with three copies (MR3): its gain
on held-out documents passed pre-registered on MaterioMiner and is above
0 post hoc over the 11 units of both corpora (+2.35 [+0.94, +3.75]); on
MuLMS alone its single-model decision failed and its votes passed
(Section 6). It costs twice the baseline's training time and no corpus.
The extra copies and the harvested names are not shown to add to it,
nor, in a follow-up, to MaterioMiner's strongest system (Section 5.7). Against MR3 the names cost three times its training
time with the harvest shared over a job's students and seven times
charged to one student, plus a 40,000-sentence corpus (Appendix B), for
an increment of +1.13 [−0.51, +2.76]; against the matched control the
increment is +0.8 [−0.7, +2.3], which on MuLMS buys 3.0 to 3.6 points of
recall on unseen names for 2.9 to 3.0 of precision in pooled rates, by
seed basis (paired by subfield on the decision seeds, +2.56 [+0.76, +4.36]
recall and −2.22 [−5.03, +0.60] precision). The absolute levels keep these
gains in proportion: on MuLMS's held-out subfields the CRF head scores
48.5, MR3 50.8, MR6 51.2 and MR6 + names 51.7, and the best of them still
finds only half of a subfield's entities (recall 50%). If recall on the names an
unlabeled corpus holds is what you need, the names buy it there, on the
fifth of the unseen names the lexicon covers, pooled over subfields
(Section 5.4), though
a decoding bias, untested against the names, might buy part of it; if F1 is, stop at mention
replacement. Expect nothing from either on a corpus like SOFC-Exp, where half
the test names are seen, the baseline is at 83 F1 and the corpus covers
few of the unseen names; which of the three matters is not separated
(Section 5.5).

**What would settle the names question.** Two analyses not run here: a
recall-matched control, a
bias toward tagging more spans calibrated on validation data, against
which the names' extra recall would have to show as precision (the
companion study's calibrated bias on the CRF head alone did not help,
Section 5.4); and a
training-size curve on MuLMS, whose six other subfields hold about 8,000
labeled sentences that the 340-sentence regime leaves unused, since
mention replacement's gains shrink with training size (dai2020analysis;
torres2024augmentation). Two baselines were dismissed on 3 seeds and
deserve 10: self-training on the same corpus and the lexicon as a
feature. And the lexicon's precision, never estimated, could be read
from a 100-name sample per type.

## 8 Related work

**Augmentation for NER.** Mention replacement, which swaps annotated
mentions for others of the same type, is one of the four simple
augmentations Dai and Adel (dai2020analysis) compared on a materials
synthesis corpus and a clinical one, where augmentation helped most when
the training sets were small; Torres et al.
(torres2024augmentation) find its gains shrink and reverse as training
sets grow, and Sharma et al. (sharma2023paraphrase) note that most
augmentations make the tagger memorise entities, which is why unseen
entities are scored separately here. The replacements otherwise come from
the training set (dai2020analysis; bartolini2024cosiner choose the one
that fits the context), from a masked language model (zhou2022melm), from
a generative model that writes text around an entity (hu2023entity), from
related sentences reached through Wikipedia anchors (cai2023graph) or
from predefined lists of the target domain's names (amalvy2022mention);
DAGA (ding2020daga) generates whole labeled sentences with a language
model trained on linearised annotations, and LADA (chen2020lada)
interpolates hidden states between a sentence and its neighbours for
semi-supervised NER. Chen et al.
(chen2021crossdomain) augment for cross-domain NER and Liu et al.
(liu2021crossner) benchmark it; both change domain, not document. We
change only where the names come from, keep every gold context, and test
on held-out documents of the same corpus.

**Lexicons from unlabeled text.** Gazetteers have been bootstrapped from
unlabeled text at least since Riloff and Jones (riloff1999learning), who
grew extraction dictionaries from seed words and their contexts; Kozareva
(kozareva2006bootstrapping) extracted
name lists with patterns, used them as features and, when bootstrapping,
added an example only when two classifiers agreed on it; Zafarian and
Asghari (zafarian2019gazetteer) decode an unlabeled corpus with a
supervised tagger, keep the confident entities as a gazetteer and retrain
with it as a feature; GEMNET (meng2021gemnet) gates gazetteer features
against the context; Song et al. (song2020gazetteers) measure Wikidata
match features at +0.5 F1 on OntoNotes and find no significant gain from
replacing annotated entities with gazetteer entries, which we read as the
same control on a corpus whose test names are mostly seen. AutoNER
(shang2018autoner) trains a tagger from a domain dictionary and unlabeled
text with no annotated sentence at all. On the teacher side, BOND
(liang2020bond) self-trains on dictionary-matched labels, Wu et al.
(wu2020teacher) train a student on unlabeled target-language text from
several source-language teachers (cross-lingual NER), and Qu et al.
(qu2023adaptive) use teachers' agreement in distantly supervised NER; our
self-training variant follows BOND's procedure on the same topic corpus,
where Du et al. (du2021selftraining) self-train on sentences retrieved
from the web. What this paper adds is not a new combination so much as the
test: the lexicon's contribution isolated against a matched control, its
source and its filter varied, on held-out documents, under protocols
fixed in advance, and the finding that the combination's gain is not
shown to exceed mention replacement's.

**Materials NER and unseen documents.** MatSciBERT (gupta2022matscibert)
is pre-trained on materials papers; ChemDataExtractor
(swain2016chemdataextractor) combines dictionaries, rules and a CRF
tagger for chemical text; Weston et al. (weston2019named)
trained a materials NER model on annotated abstracts and applied it at
scale; Mysore et al. (mysore2019materials) annotated synthesis
procedures, the corpus Dai and Adel augmented; MuLMS (schrader2023mulms),
SOFC-Exp (friedrich2020sofc) and MaterioMiner (durmaz2024materiominer)
are the corpora here; Trewartha et al. (trewartha2022quantifying) and
MatSci-NLP (song2023matscinlp) benchmark materials language models on
such tasks. Large language models now extract structured materials
information from text by fine-tuning (dagdelen2024structured) or by
prompting (polak2024extracting); neither is compared here (Limitations),
and both are scored on their own held-out documents, which is the
protocol this paper applies to a small tagger. Potu et al.
(potu2025ontology) train on 47 other fatigue papers annotated with
MaterioMiner's classes and report 66.74 F1 on MaterioMiner against 78.56
in distribution, a gap of about 12 points where our three-paper models
lose 28–31 (companion study); their corpus is not redistributed. Random
sentence splits leak names between training and test (babl2025random;
gorman2019standard and sogaard2021random take opposite sides on random
splits), taggers score far lower on mentions unseen in training
(augenstein2017generalisation; fu2020rethinking; in biomedical NER,
kim2022biomedical), their generalisation to unseen mentions rests on name
regularity more than on context and is inflated by mention coverage
between training and test (lin2020rigorous), they lean on the names
themselves (ghaddar2021context), contextual embeddings narrow but do not
close the gap on unseen mentions (taille2020contextualized), and
switching the entities of a test set for others of the same type exposes
it (agarwal2020entityswitched), which is the question of Section 5.4;
seed variance and significance testing on small test sets are treated by
Reimers and Gurevych (reimers2017reporting) and Dror et al.
(dror2018hitchhiker), whose advice the protocol follows, Bouthillier et
al. (bouthillier2021accounting) account for the variance of benchmarks
across seeds and data splits, and Card et al. (card2020power) show how underpowered typical NLP comparisons are, which
is why Appendix E reports what each comparison here could detect. Our
companion study measures the gap and the unseen-name shares on the three
corpora and finds that every gain that holds on held-out documents is a
gain on unseen names.

## 9 Conclusion

A tagger trained on a few annotated materials papers misses half or more
of a new paper's entity names. Giving it names harvested from unlabeled
text by three agreeing teachers, through mention replacement, improves on
the baseline on held-out documents of two corpora under a pre-registered
protocol, by a criterion adopted mid-study; but mention replacement
without the names gains most of that on the same units, and the names'
share, +0.8 points with an interval from −0.7 to +2.3, is not
established, nor is it in a follow-up on MaterioMiner's strongest system
(+0.96 [−0.01, +1.92] fine-grained, +0.85 [−0.72, +2.42]
coarse-grained). No dependence on the names' source or on the filter was
detected within about 2 to 3 points, and on MuLMS what the names buy is
recall on the unseen names the lexicon holds, a fifth of them, while
precision falls in pooled rates. We report the ten decisions, the
criterion revision, the amendments and the one deviation, and the
analyses that would settle the question. The practical advice is
mention replacement, whose own gain passed pre-registered on
MaterioMiner and holds post hoc over both corpora; the methodological
advice is to test a names method against a control with the same copies
and on documents the training data has never seen, with enough seeds and
units to detect a difference of a point or two between a method and its
control.

## Limitations

- **Few, heterogeneous units.** Seven subfields and four papers give 6
  and 3 degrees of freedom; the subfields differ by 24 points under the
  CRF head (32.49 to 56.82) and range from 2 to 11 articles, the papers
  from 41 to 201 sentences, and the pooled decision weights them equally
  as if drawn from one population. The leave-one-out and sign-flip
  checks (Section 5.1) are post hoc; a mixed model was not fitted, and
  with two corpora no between-corpus variance could be estimated, so the
  pooled intervals are conditional on these two corpora.
- **Three passes on one corpus's papers.** The MaterioMiner passes at two
  granularities and the pooled pass share four papers; MuLMS alone passes
  nothing single-model; none of the three passes survives a Bonferroni
  correction over the ten decisions (Section 5.1).
- **A revised criterion.** The criterion under which MR6 + names passes
  was adopted on 20 October on the author's decision to push for a
  methods claim,
  after the screen's first half and before its own screen
  result, and when mention replacement was already known to meet it (Section 3);
  under the earlier criterion there is no claim. The screen that chose
  MR6 + names and the MuLMS decisions share the same seven subfields:
  disjoint seeds remove the selection's seed noise, not its fit to these
  units.
- **MuLMS's training pool is two or three articles.** The 400-sentence
  pool is drawn from a shuffled order of the other subfields' articles
  until 400 sentences are reached, the last article cut, so each MuLMS
  model is trained on two or three articles, a different set per held-out
  subfield with some articles recurring across sets; the 88% share of
  unseen names and the 24-point spread between subfields reflect that.
- **One training regime.** Every positive result is at about 340 training
  sentences; MuLMS's full pool was never used, SOFC-Exp's larger training
  set is where nothing helps, and the literature predicts smaller gains
  for mention replacement as training grows. The claim is scoped to a few
  hundred annotated sentences.
- **One follow-up on the strongest system.** The names were run on
  multi-task training with six-copy mention replacement in a follow-up
  registered after the results above (Section 5.7): +0.96 [−0.01, +1.92]
  fine-grained and +0.85 [−0.72, +2.42] coarse-grained on one corpus's
  four papers, neither passing; the comparisons detect 1.3 and 2.1
  points. The two-head filter's lexicons hold fewer unseen names than the
  single-task ones (a third fewer coarse-grained, pooled); whether that
  lowered the increment is not known, and a looser filter was not tried.
- **The names' effect is only partly isolated.** The lexicon-membership
  split (Section 5.4) was run after the decisions, on rebuilt lexicons;
  it reads name knowledge on MuLMS and, by the rule's letter, a generic
  shift on MaterioMiner within four papers' wide intervals; no
  recall-matched control was run; the lexicon's precision was not
  estimated (the precision of the predictions of lexicon names, 69 to 75%
  on MuLMS and 81 to 83% on MaterioMiner, is the nearest figure); the
  per-subfield
  correlations of Section 5.4 are exploratory with 7 points and
  confounded; the recall–precision trade is reported as pooled rates and, post hoc,
  as paired per-subfield intervals (Section 5.4).
- **Untuned settings.** MR3's three copies and swap rate 0.5, MR6's six
  copies, the 0.9 threshold and the three-teacher agreement were fixed in
  advance, not tuned; Dai and Adel tune the copies (1 to 10) and the rate
  (0.1 to 0.7) on development data, and a tuned mention replacement could
  close or widen every gap in Table 3.
- **Null results are not equivalences.** The source decision and the
  ablations detect differences of about 2 to 3 points (Appendix E);
  smaller effects of the source or the settings are not excluded.
- **Baselines.** The gazetteer-feature and self-training variants were
  rejected on 3 seeds, and one encoder, MatSciBERT, was used. No large
  language model was compared, as an extractor, a teacher or a generator
  of same-type names (zhou2022melm; hu2023entity): the question here is
  whether names already present in unlabeled text add to mention
  replacement, and a generated pool is a different source that needs its
  own matched test; an open-weight model writing names for a held-out
  subfield, fed through the same pipeline, is the natural next
  comparison.
- **MuLMS as converted** keeps the outermost entities and repairs damaged
  tokens; its scores cannot be compared with the published nested ones.
- **Corpora and pre-registration.** The corpora come from Europe PMC's
  open-access subset, found by title terms, with journal concentration
  and licence mixes given in Appendix C, and are rebuilt from hashed
  reports rather than redistributed. Pre-registration rests on the
  repository's own dated commits, not on an independent registry.

## Ethics and broader impact

The work uses published corpora under their licences and open-access
full texts for the unlabeled corpora, which are not redistributed. No
personal data is involved. An AI coding assistant was used, under the
author's direction, to write code and drafts and to run the adversarial
internal reviews of the drafts cited in Section 5.7; the author is
responsible for the content. This paper shares data, baselines and protocol with a
companion evaluation study of the same programme (in preparation; its
draft is in the repository), which reports the random-split gap, seed
variance and the other additions tested. The two papers overlap in
Sections 2 to 4. The results that appear in both are the random-split
gap and the unseen-name shares of Section 1, mention replacement's
MaterioMiner decisions and the multi-task results of Section 4, and the
label-propagation rule of Appendix D, which the companion reports in its
survey; the companion summarises D1–D8 and the decomposition of Section
5.2 in one table and cites this paper for them.

## Code availability

The repository, github.com/hatemadi15/AI_Research_Ontologies_GNN_GRL
(Apache 2.0; an archived copy with a DOI will be deposited at
acceptance), holds the code, every protocol with its dates and
amendments, the analyses behind every number, the generated tables and
figures, and the Kaggle batch definitions that produced each results
file.

## Data availability

The same repository holds every results file with its seeds, environment
and saved test predictions, and the corpus reports with their queries,
article identifiers, licences and SHA-256 hashes. MaterioMiner's files
are included under CC BY 4.0; SOFC-Exp is converted from the authors'
repository (CC BY 4.0); MuLMS is read from its Hugging Face release at a
pinned revision (CC BY-SA 4.0). The unlabeled corpora are not
redistributed, 10–33% of their articles being under non-commercial or
no-derivative licences; they rebuild from their reports, with hash
checks, and the reports identify the CC BY articles. The runs did not store
the harvested lexicons; `ner_baseline.py --students 0 --save-names`
rebuilds each from its corpus (the week-15 analysis), and they are not
released, because they derive from corpora that are not redistributed.

## Author contributions, competing interests and funding

The author designed the study, wrote the code, ran the experiments and
wrote the paper. The author declares no competing interests. No external
funding was received; the GPU compute was Kaggle's free tier. The author
thanks the creators of MaterioMiner, SOFC-Exp and MuLMS for releasing
their corpora under open licences.

## Appendix A: the protocol record

**Table A1.** The protocol record of the names route: when each decision's
protocol was committed (commit times, UTC; the protocol text's own
written times are given below the table), when its runs ran, when its result was
recorded, and the amendments.

| Decision | Protocol committed (times UTC) | Runs | Result recorded | Amendments |
|---|---|---|---|---|
| the route's earlier protocols (Section 6): mention replacement on MuLMS (week 10); names from the other subfields' sentences (week 11) | 17 October 20:21 (`a08356f`); 18 October 01:13 (`427e34c`) | 17–20 October | week 11: 18 October (MuLMS, `14ed50f`) and 20 October (MaterioMiner, `04764ef`) | — |
| D1, D2 (week 12) | 18 October 11:00 (`40631ce`; refined 11:05, `bd20cec`): the screen on seeds 0–200, the decisions on seeds 300–900 so that the selection's seed noise does not enter them; the MaterioMiner analogue gated on D1; the methods criterion: D1 plus a MaterioMiner pass | 20–24 October | 24 October | three amendments before any run added variants V4, V6, V7–V8 to the screen |
| D3, D4 (week 12) | 18 October 11:00 (`40631ce`): the winner's MaterioMiner analogue decided against the CRF head per granularity, to run only if D1 passed; 20 October 00:51 (`ce0f592`): the analogue's batches, with its comparisons against MR3 and the votes reported; the gate lifted at 09:14 (`808ff4b`), restored at 13:53 (`5f7f827`), lifted again at 18:42 (`de62446`) | 24–25 October | 25 October (`3432988`) | — |
| D5 (week 13) | 20 October 09:14 (`808ff4b`): the pooled decision over both corpora as a second route to the claim, "each unit its 10-seed mean", the stop rule lifted; amended three times the same day, before any decision run of the route: 09:25 (`e94f352`), the label-propagation rule's two MaterioMiner decisions and an SOFC-Exp report added after its MuLMS results (Appendix D); 13:53 (`5f7f827`), eighteen minutes after the screen chose MR6 + names, D6 added conditional on D1 and the analogue, which the morning's text ran whenever the screen had a winner, gated on D1; 18:42 (`de62446`), D6's condition widened to D1 or D5 and the analogue's morning condition restored, the gating recorded as a mistake | 24–25 October | 25 October (`3432988`) | the analysis took MuLMS's 10-seed means; the 7-seed reading is D5′ |
| D6 (week 13) | 20 October 13:53 (`5f7f827`), conditional on D1; widened to D1 or D5 at 18:42 (`de62446`) | 25 October | 25 October | — |
| D7, D8 (week 14) | 29 October 12:07 (`2886267`), two minutes after the first batch was pushed and five hours before its results | 29 October | 29 October | — |
| ablations (week 14) | 29 October, as above | 31 October | 31 October | — |
| the lexicon-membership analysis (week 15; post hoc, no decision) | 31 October 16:13 (`2c60f33`): the rebuild, its checks and the reading fixed before the run | 31 October (`w15a`) | 31 October | — |
| D9, D10 (week 17; follow-up) | 31 October 19:15 (`f39447c`), after the results above had been written up, a pass expected to be unlikely; the code 19:41 (`013f9ac`) | 31 October 19:55 to 1 November 00:40 (`w17a`, `w17b`) | 1 November | — |

**Times.** All times are the commits' (UTC). When the pooled decision was
added at 09:14 on 20 October, the screen's first half was recorded (MR6
and the gazetteer variants, at 08:11) and its second half, with MR6 +
names, was not (13:35); the 18:42 amendment followed the spread between
subfields that showed the 7-subfield decision underpowered. The protocol
text's own written times differ from the commits': later by 6 to 53
minutes in the 09:25, 09:47 and 13:53 commits of 20 October (which carry
09:50 and 09:55, 10:40, and 14:00; the 09:47 commit recorded the
label-propagation rule's results on SOFC-Exp's folds and MaterioMiner),
earlier by about four hours in the 18:42 one (14:00 and 14:30), and
earlier by five minutes in the week-17 protocol (written 19:10, committed
19:15). Week 16 was the companion paper's write-up and has no protocol.

## Appendix B: hyperparameters and compute

MatSciBERT (110M parameters), token classification on the first sub-word,
a linear-chain CRF with IOB2 constraints; AdamW, learning rate 5e-5,
linear warm-up, batch size 16, up to 20 epochs, early stopping on
validation micro F1 with patience 3, the best epoch's weights evaluated;
the CRF layer at learning rate 5e-3 without weight decay, the encoder
with weight decay 0.01, warm-up over the first 10% of the steps, the
gradient norm clipped at 1.0, dropout 0.1, inputs truncated at 512
sub-words;
mention replacement with probability 0.5 per mention, 3 or 6 copies,
copies that change nothing dropped; teachers' marginals from
forward–backward, threshold 0.9, all three teachers agreeing on span and
type. On MuLMS's decision seeds a CRF head run takes 96 s on a T4 (the
best epoch after 8.1 epochs on average), an MR3 run 198 s (5.7) and an
MR6 run 296 s (5.1).
A names run's stored time is its job's: the three teachers, their tagging
of the corpus and the job's three or four students together, 2,099 s on
average over the 14 jobs (1,598–2,972 s; the best epoch after 5.6 epochs). Per student that
is 609 s (the job's time over its students, averaged over the 14 jobs),
6.4 times the CRF head's 95.8 s; the harvest alone, the job less
its students' own training at MR6's rate, is about 1,063 s, eleven CRF-head
runs, and a single student charged with the whole harvest costs about 14
times the CRF head's run. None of these figures counts building the corpus
(the search, download and filtering of about 300 articles). The students
see 2.8 to 4.6 times more sentences per epoch (601, 1,203 and 1,209
augmented sentences on average beside the 340 gold ones, for MR3, MR6 and
the names) and reach their best epoch after 5.7, 5.1 and 5.6 epochs
against the CRF head's 8.1, so up to it they take 2.0, 2.9 and 3.1 times
its updates. Runs
used Kaggle sessions of two T4 GPUs with deterministic CUDA, one frozen
commit per batch, a pinned image and pinned libraries; the three week-14
batches alone took 19 session hours (5.2, 4.3 and 9.7). The two week-17
sessions took 4.7 and 3.1 hours. A names job on the multi-task base
(three multi-task teachers, their tagging of the corpus with two heads,
ten students) took 82.5 to 122.7 minutes (mean 99.7), 10.0 minutes per
student with the harvest shared over its ten students; the harvest alone,
the job less ten students at the control's rate, is about 31 minutes, so
a student charged with all of it costs about 38. A multi-task six-copy
control run takes 6.9 minutes, a three-copy one 4.8, the multi-task model
2.0, and a CRF-head run 1.1 fine-grained and 1.3 coarse-grained on the
same papers; the students' best epoch came after 5.4 to 8.1 epochs on
average per paper and granularity.

Every interval in the paper is the analysis scripts' value, computed from
per-unit values (F1 means over seeds, or rates pooled over seeds)
rounded to two decimals; recomputed from the unrounded runs, no bound
moves by more than about 0.01 and no verdict changes.

## Appendix C: the unlabeled corpora

Each corpus is 40,000 sentences of open-access full texts found on Europe
PMC by a title query (for MuLMS's solid-oxide subfield: "solid oxide",
"solid-oxide", "SOFC", "SOEC" and their plurals; for SOFC-Exp's corpus "solid oxide", "SOFC" and
"SOFCs"; the other queries are in the repository's reports), with the dataset's own articles excluded by identifier, every
sentence sharing a word 8-gram with a dataset sentence dropped (and, for
MuLMS, any 50-character run), sentences of 5 to 120 words with at least
half alphabetic characters, and formulas longer than 30 characters cut; sentences are split by a
rule-based splitter that does not break after abbreviations or inside
decimal numbers, and words by a regular expression that separates
punctuation.
Per corpus: 290–368 articles with text (292–370 selected), published
between 1996 and 2026 (over 99% from 2009 on), 63–90% of them CC BY and
10–33% CC BY-NC or ND (0.3–4% without a licence field), the largest
journal holding 13–63% of the articles (MDPI Materials 52% of the steel
corpus and 63% of the fatigue corpus). Each
report records the query, the counts at every stage, the article
identifiers with licences and the corpus's SHA-256; a rebuild that
differs is detected by the hash.

## Appendix D: label propagation within a document

A post-processing rule on the saved predictions: a surface form that the
model tags in most of its occurrences in a document, with one most
frequent type, is tagged in its untagged occurrences too. On the CRF
head's 10-seed predictions it gains +0.82 [+0.18, +1.46] on MuLMS's
held-out subfields (pre-registered, every subfield gaining; 1,213
entities added per seed) and +0.12 [+0.01, +0.24] on coarse-grained
MaterioMiner (pre-registered in an earlier week; fine-grained +0.30
[−0.35, +0.95], no), and loses −0.44 [−0.72, −0.16] on SOFC-Exp's article
folds (reported), where the annotators leave 14% of a tagged term's
occurrences untagged on the official test set. It is not shown to add
to mention replacement (+0.57 [−0.12, +1.25] on MuLMS). Two of its ten
pre-registered decisions pass (six in week 7; four in week 13, two on
MuLMS registered at 09:14 on 20 October and two on MaterioMiner at 09:25,
after the MuLMS results, computed 22 minutes later from saved
predictions), where about 0.25 would by chance.

## Appendix E: detectable differences

The smallest true mean difference each comparison detects with 80% power
at the two-sided 5% level, from the half-width of its observed interval
and its degrees of freedom (`ner_robustness.py`, `detectable`).

**Table E1.** The difference each comparison could detect at 80% power
and the two-sided 5% level, from its stored interval's half-width and
degrees of freedom.

| Comparison | CI half-width | df | detectable at 80% power |
|---|---|---|---|
| D5: MR6 + names − CRF head, pooled, 10-seed basis | 2.63 | 10 | 3.67 |
| D5′: the same on the decision seeds | 2.54 | 10 | 3.55 |
| D7: MR6 + names − MR6, pooled | 1.46 | 10 | 2.04 |
| MR6 + names − MR3, pooled (post hoc) | 1.63 | 10 | 2.28 |
| MR6 − MR3, pooled (post hoc) | 0.58 | 10 | 0.82 |
| MR3 − CRF head, pooled (post hoc) | 1.46 | 10 | 2.03 |
| D1: MR6 + names − CRF head, MuLMS | 4.32 | 6 | 5.92 |
| D2: MR6 + names − MR3, MuLMS | 2.55 | 6 | 3.50 |
| MR6 + names − MR6, MuLMS (part of D7) | 2.02 | 6 | 2.77 |
| D8: own-topic − mixed corpus, MuLMS | 1.83 | 6 | 2.51 |
| D3: MR6 + names − CRF head, MaterioMiner fine | 2.88 | 3 | 3.77 |
| D4: MR6 + names − CRF head, MaterioMiner coarse | 1.10 | 3 | 1.44 |
| D9: multi-task + MR6 + names − multi-task + MR6, MaterioMiner fine (follow-up) | 0.96 | 3 | 1.26 |
| D10: the same, coarse (follow-up) | 1.57 | 3 | 2.05 |
| multi-task + MR6 + names − multi-task + MR3, MaterioMiner fine (reported) | 0.91 | 3 | 1.20 |
| the same, coarse (reported) | 1.05 | 3 | 1.38 |
| ablation, 10,000 sentences − MR6 + names (3 seeds) | 1.63 | 6 | 2.23 |
| ablation, 20,000 sentences − MR6 + names (3 seeds) | 2.29 | 6 | 3.14 |
| ablation, threshold 0.7 − MR6 + names (3 seeds) | 1.47 | 6 | 2.01 |
| ablation, 2 of 3 teachers − MR6 + names (3 seeds) | 1.70 | 6 | 2.32 |
| ablation, names in half of the draws − MR6 + names (3 seeds) | 1.71 | 6 | 2.35 |

## Appendix F: per entity type

Table F1 gives, per type, the F1 of the CRF head, MR6 and MR6 + names on
MuLMS's subfields, averaged over the (seed, unit) pairs whose test set
has the type (the MaterioMiner tables, both granularities, are in the
repository); these are exploratory and are not decided on (Section 5.6).
On MuLMS, MR6 + names gains over the CRF head on the 12 types with a
non-zero score (RANGE, 3 entities per unit, scores 0 under every
system), and MR6 holds more than half of that gain on 9 of them (not on
MEASUREMENT, TECHNIQUE and DEV, where the names' own increment is +1.7
to +2.8). On MaterioMiner, coarse-grained, MR6 + names gains over the
CRF head on 16 of 27 types and MR6 holds more than half of the gain on
13 of those; the names also lose against MR6 on several types, two where
MR6 had gained (MEP −8.0, STD −7.0) and one where it had not (MAT −5.8),
and most on types with a handful of test entities (PRP −10.2, CPR −9.2).

**Table F1.** Per entity type on MuLMS's 7 held-out subfields (decision
seeds): the test support per unit, the number of units with the type,
micro F1 of the CRF head, MR6 and MR6 + names, and the names method's
differences; types ordered by test support.

| type | test support per unit | units | CRF head | MR6 | MR6 + names | names − CRF head | names − MR6 |
|---|---|---|---|---|---|---|---|
| MAT | 1806.7 | 7 | 53.92 | 56.94 | 57.50 | +3.58 | +0.56 |
| VALUE | 661.1 | 7 | 72.74 | 74.53 | 75.47 | +2.72 | +0.94 |
| PROPERTY | 554.3 | 7 | 28.97 | 32.36 | 33.60 | +4.62 | +1.24 |
| FORM | 454.3 | 7 | 24.97 | 31.83 | 32.51 | +7.53 | +0.68 |
| MEASUREMENT | 300.9 | 7 | 40.11 | 41.40 | 43.11 | +2.99 | +1.71 |
| CITE | 243.4 | 7 | 82.05 | 85.15 | 84.56 | +2.51 | −0.59 |
| SAMPLE | 198.7 | 7 | 3.61 | 5.66 | 4.33 | +0.73 | −1.33 |
| TECHNIQUE | 131.4 | 7 | 17.34 | 19.73 | 22.49 | +5.15 | +2.76 |
| DEV | 108.1 | 7 | 13.87 | 16.18 | 18.69 | +4.83 | +2.51 |
| INSTRUMENT | 52.9 | 7 | 17.69 | 22.11 | 20.21 | +2.52 | −1.90 |
| NUM | 37.1 | 7 | 5.57 | 8.11 | 8.71 | +3.14 | +0.60 |
| UNIT | 13.0 | 7 | 12.78 | 21.48 | 21.00 | +8.21 | −0.48 |
| RANGE | 3.0 | 6 | 0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
