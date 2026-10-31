# Harvested names are not shown to add to mention replacement on held-out materials documents: a pre-registered test

Hatem Madi (affiliation to be added; hatem.madi15@gmail.com)

*Draft of 31 October 2026, revised after four adversarial review rounds. A
number in square brackets is a 95% Student-t confidence interval of a mean
paired difference over the held-out units named. The note at the end maps
this draft's tables and figures to the repository's generated files and its
sections to a journal's structure.*

## Abstract

Materials entity taggers are trained on a few annotated papers and scored
on sentences from those papers, which hides how they read a new one: our
baseline scores 77 F1 on random splits of MaterioMiner and 46 on a
held-out paper, because 81% of its entities are names absent from the
training sentences. We ask whether unlabeled papers can supply those names. Three
seeds of the baseline label 40,000 sentences of open papers; the names
they agree on confidently join the pool of mention replacement,
which swaps the names in six copies of each training sentence. Under
pre-registered protocols with 7–10 seeds per unit, the recipe gains 3.58
F1 points [0.95, 6.21] over the baseline across 11 held-out units of two
corpora. But mention replacement without any harvested name gains 2.46
[1.00, 3.91] on the same units (post hoc, on the seeds the screen did not
see); the extra copies add 0.33 [−0.25, 0.92] and the names 0.80 [−0.66,
2.26], neither established (the names comparison detects 2 points).
Neither the names' source nor the filter's settings changes the result
detectably, and where test names are mostly seen nothing helps. The names
trade precision for recall on unseen names on one corpus and raise both
on the other; which names they help with the recorded data cannot tell.
We report all eight decisions, the criterion's revision and a post hoc
robustness analysis. Mention replacement is worth using where most of a
new document's names are unseen and training is a few hundred sentences;
harvesting names, at six to fourteen times the training cost, is not
shown to add to it.


## 1 Introduction

Annotated materials corpora are small and come from few documents.
MaterioMiner (durmaz2024materiominer) annotates 4 papers; SOFC-Exp
(friedrich2020sofc) 45 articles; MuLMS (schrader2023mulms) 50 articles
from 7 subfields. A tagger trained on such a corpus is meant for documents
the corpus does not contain, and a random sentence split does not measure
that. On MaterioMiner our baseline, MatSciBERT with a CRF output layer
(the CRF head, Section 4), scores 77.49 fine-grained over 5 random splits
and 46.46 on a held-out paper; on MuLMS it scores 65.72 over random splits
of the same size and 48.66 on a held-out subfield; on SOFC-Exp, whose 45
articles share one topic and one annotation scheme, 86.27 against 82.79
on held-out articles. The gap is mostly names: 81% of the entities of a
held-out MaterioMiner paper are names that no training sentence
annotates with that type (88% on a held-out MuLMS subfield, 46% on
SOFC-Exp's held-out articles), and the CRF head finds 35% of them on
MaterioMiner and 39% on MuLMS, against 71% and 66% of the names it has
seen.

This paper asks whether those names can be given to the tagger without
annotation, and answers with a negative result that we think is worth
publishing because of how it was obtained. Unlabeled materials text is
plentiful, and a corpus of 40,000 sentences of open-access papers holds
55–81% of a held-out subfield's unseen test names. The recipe is simple:
three CRF taggers trained on the labeled sentences tag the corpus; the
names they tag alike, each with every tag's marginal probability at 0.9
or more, form a lexicon; the lexicon enters training through mention
replacement (dai2020analysis), which copies each training sentence six
times with its mentions swapped for other mentions of the same type,
drawn from the training mentions and the lexicon together. No
teacher-labeled sentence is trained on: the student sees only annotated
sentences, some of whose names are swapped for harvested ones.

Every decision was fixed in a protocol committed to the repository
before its results (vanmiltenburg2021preregistering; one protocol two
minutes after its first batch was pushed, Section 3), with 7 to 10 model
seeds per held-out unit and a decision rule stated in advance (the 95%
interval of the mean paired difference above 0); the screen that chose
the recipe among seven variants used three seeds that the MuLMS
decisions did not, and the two decisions that isolate the names'
contribution were registered after the main decision passed and before
their results. Section 3 records the revision of the criterion, its
amendments and the one deviation in the analysis. The results are:

- The recipe improves on the CRF head on held-out documents: +3.58
  [+0.95, +6.21] pooled over MuLMS's 7 held-out subfields and
  MaterioMiner's 4 held-out papers (D5; +3.58 [+1.04, +6.13] on the MuLMS
  seeds the screen did not see); post hoc, 9 or 10 of the 11 units
  positive by seed basis, a two-sided sign-flip permutation p of 0.011
  (0.007), every leave-one-unit-out interval above 0 (Sections 5.1 and
  5.2). MuLMS alone
  does not pass (+3.19 [−1.13, +7.51]).
- That gain is not shown to be the names'. Plain mention replacement with
  three copies and no lexicon gains +2.46 [+1.00, +3.91] over the CRF head
  on the same units and seeds (post hoc; its own pre-registered decisions
  passed on MaterioMiner and failed on MuLMS alone); the three extra
  copies add +0.33 [−0.25, +0.92] and the harvested names +0.80 [−0.66,
  +2.26] (D7, the pre-registered decision on the names, Section 5.2), in a
  comparison that detects about 2 points; the interval's upper bound,
  +2.3, is as large as mention replacement's whole gain.
- Neither the names' source nor the recipe's settings changes the result
  detectably, within about 2 to 3 points: a mixed corpus of six other
  materials topics gives −0.18 [−2.01, +1.65] against the held-out topic's
  own (pre-registered), and five settings of corpus size, filter and share
  move F1 by at most 0.2 points (Section 5.3).
- On MuLMS the names raise recall on unseen names and lower precision; on
  MaterioMiner they raise both. Whether they help with the names the
  lexicon holds cannot be told from the recorded data: per subfield their
  gain correlates negatively with the lexicon's coverage, but coverage and
  the baseline's strength are confounded (Section 5.4). On SOFC-Exp, where
  half the test names are seen, nothing helps (Section 5.5).

Section 6 lists what else was tried on the way and the count: eight
pre-registered decisions on the names route in its last three protocols,
three passing, which share one corpus's four papers.

## 2 Data

Table 1 gives the three corpora; the short names in parentheses are the
ones the figures and tables use. All three corpora are released under
open licences; our conversions and the held-out protocol are in the
repository.

| Corpus | Documents, sentences | Annotation | Held-out unit (test size) | Training per unit |
|---|---|---|---|---|
| MaterioMiner (durmaz2024materiominer), release v1.0.1, CC BY 4.0 | 4 papers on fatigue of metals (s11340, ncomms7164, s41467, met10111458: 41, 81, 201, 153 sentences) | the classes of the dataset's materials mechanics ontology (MMO) at two levels: fine-grained, the 27 most frequent classes (1,328 entities; e.g. Alloy, Value, Mechanism, StateDescriptor, FatigueTestSpecimen, GrainBoundary); coarse-grained, 27 parent types (2,194 entities) | one paper (41–201 sentences) | the other three papers (275–435 sentences, of which 15% validate: 234–370 training sentences) |
| MuLMS (schrader2023mulms), Hugging Face `timo-pierre-schrader/MuLMS` at revision bc1c97d, CC BY-SA 4.0 | 50 articles from 7 materials subfields, 10,184 sentences: water electrolysis (electrolysis; 4 articles, 992 sentences), graphene (2, 351), polymer-electrolyte-membrane fuel cells (pemfc; 8, 1,638), polymers (poly; 11, 2,384), semiconductors (semi; 11, 2,040), solid oxide fuel cells (sofc; 4, 928), steel (10, 1,851) | 13 types (MAT, VALUE, PROPERTY, FORM, MEASUREMENT, CITE, SAMPLE, TECHNIQUE, DEV, INSTRUMENT, NUM, UNIT, RANGE); we keep the outermost of its nested entities, 31,952 of 46,584 | one subfield, every sentence of its articles (351–2,384) | 400 sentences of the other subfields' articles, taken in a shuffled order of those articles until 400 are reached (about two articles, a different pair for each held-out subfield); 340 train and 60 validate, MaterioMiner's size by design |
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
under another type, 0.0–2.1% of a test set, count as neither. Recall
and precision are reported by kind of name, pooled over seeds and units.

## 3 Evaluation protocol

**Seeds and scoring.** Every system compared in a decision has 10 model
seeds per held-out unit (the random seed of the training run, offsets 0
to 900), except where a protocol reserved the screen's 3 seeds (0 to 200)
for selection and decided on the other 7 (300 to 900); the mixed-corpus
variant has those 7 seeds and the ablations the screen's 3. The score is micro F1 at the entity
level: an entity counts as found only with its exact span and type;
precision is the share of predicted entities that are correct, recall the
share of gold entities found, F1 their harmonic mean, micro over all
entities of a test set (seqeval in its default mode, as the datasets' own
code scores), in percentage points.

**Terms.** A *decision* is a comparison with a pre-registered verdict; a
comparison labelled *reported* was fixed in a protocol but carries no
decision; *post hoc* means computed after its results were known. A
*screen* is a 3-seed comparison used to choose among variants. A *vote*
is the majority vote, entity by entity, of the three models of a seed
triple, scored like a single model; votes are recorded in the repository beside the single
models they are built from and are not the systems compared here. The
*MaterioMiner analogue* of a MuLMS system is the same recipe run on
MaterioMiner's held-out papers with the fatigue corpus. The *plan* is
the project's methods plan, a document of the repository that set the
route and its stop rule before the protocols. The *sign-flip test* is
the exact permutation test of a mean paired difference over all 2^n
assignments of signs to the n units, two-sided. Protocols are named by
the week of the project in which they were fixed (weeks 10 to 14, 17 to
29 October 2026; Appendix A).

**Decision rule.** A method improves on its reference when the 95%
Student-t interval of the mean paired micro-F1 difference lies above 0,
the pairs being the held-out units (7 subfields, 4 papers, 5 folds), each
side the mean of its seeds. With 4 papers the interval has 3 degrees of
freedom; with 7 subfields, 6. A pooled decision takes the units of two
corpora together (11 units, 10 degrees of freedom) and requires, in
addition, that neither corpus's own interval lie below 0, a guard that
trips only on a significant loss. Each decision is one-sided at 2.5%, so
of 8 decisions about 0.2 would pass if nothing worked; the decisions are
not independent (Section 5.1). What the design can detect depends on
each comparison's spread. From the observed intervals, at 80% power and
the two-sided 5% level: the names against matched copies detect 2.0
points pooled over the 11 units and 2.8 on MuLMS alone; the names against
the CRF head 3.5 to 3.7 pooled and 5.9 on MuLMS alone (D1 failed at
+3.19); the
source decision 2.5; the 3-seed ablations 2.0 to 3.1 (Appendix E lists
every comparison).

**Pre-registration, and what changed.** Each protocol was committed to
the repository before its results, all but one before its runs, with its
decisions, the comparisons
reported without a decision and the expected number of false passes;
amendments are marked with their reason and time (Appendix A lists the
commits). Three points must be disclosed.

First, the criterion for a methods claim was revised on 20 October
(09:14), on the author's own decision, recorded in the protocol, to push
for a methods paper. The week-12
protocol (18 October) had required a pass against the CRF head on MuLMS
alone plus a pass on MaterioMiner and had gated the MaterioMiner runs on
the MuLMS pass, and the plan carried a stop rule, "no fourth protocol on
MuLMS". The week-13 protocol added a pooled decision over both corpora
as a second route to the claim and lifted the stop rule. At that moment
the screen's first half was known (the copy control and the gazetteer
variants, which use the names as a dictionary feature; recorded at
08:11) and its second half, with the recipe, was
not (recorded at 13:35); and the protocol itself computed, from runs
already made, that plain mention replacement would pass the new pooled
criterion (+2.34 [+0.94, +3.75] there, +2.35 in Table 3 by the
analysis's rounding; labelled post hoc because both corpora's results
were known). The criterion under which the recipe
passes was thus adopted knowing that its control already met it, before
the recipe's own pooled result existed. Two amendments followed the same
day, both before any decision run. The morning's text ran the
MaterioMiner analogue whenever the screen had a winner. At 13:53,
eighteen minutes after the screen had chosen the recipe, the SOFC-Exp
decision D6 was added, conditional on a pass of D1, and the analogue was
gated on D1 as well. At 18:42, after the recipe's spread between
subfields had shown the 7-subfield decision underpowered, D6's condition
was widened to a pass of D1 or D5 and the analogue's morning condition
was restored, the afternoon's gating being recorded as a mistake (the
amendment text carries the times 14:00 and 14:30). Under the
week-12 criterion the recipe has no claim, since the MuLMS decision
failed; under the revised one it passes.

Second, the pooled decision's protocol text named MuLMS's decision batch
(seeds 300 to 900) as the source of its units and, in the same paragraph,
"each unit its 10-seed mean"; the analysis as executed took the 10-seed
means, which include the screen's three seeds. We report both readings,
which agree (+3.58 [+0.95, +6.21] and +3.58 [+1.04, +6.13]), quote the
registered one first, and use the 7-seed basis for the post hoc
decomposition, since those seeds the screen never saw.

Third, the names-only and source decisions of week 14 used the 7 decision
seeds on MuLMS; their protocol was committed two minutes after the first
batch was pushed and five hours before any result arrived. Every
comparison not in Table 2 is either reported (fixed in a protocol without
a decision) or post hoc, and labelled so; the robustness checks of
Sections 5.1 and 5.2 are post hoc.

**Runs.** Kaggle T4 GPUs with deterministic CUDA, one frozen commit per
batch, a pinned image and pinned libraries; model snapshots pinned by
commit. Each GPU session first repeated one earlier run, and every
repeat over weeks 4 to 14 was identical. Hyperparameters and compute are
in Appendix B.

## 4 Systems

Figure 1 sketches the recipe, this paper's name for MR6 + names, the
last of the four systems compared throughout:

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
  with mention replacement +6.73 [+6.03, +7.43] fine-grained (our
  companion evaluation study); neither transfers to SOFC-Exp, and MuLMS
  has one type level, so the CRF head is the reference the three corpora
  share.
- **MR3.** Mention replacement (dai2020analysis): three copies of each
  training sentence with an entity, each mention replaced with
  probability 0.5 by a training mention of the same type; copies that
  change nothing are dropped. The student trains on the gold sentences and
  the copies with the CRF head's recipe and seed.
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
    of its coarse-grained), the SOFC corpus 36% of a fold's.
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
    harvested entries are at least about 85% of the pool (the unit's
    601–1,309 training mentions against 8,328–50,869 harvested), so most
    replacements are harvested names.

Every run stores its teachers' scores, the lexicon's counts and the share
of the unseen test names it holds, the corpus's hash and the filter. On a
T4 GPU a CRF head run takes 1.6 minutes and an MR6 run 5; a MR6 + names
job, which trains the three teachers, tags the corpus and trains three or
four students, takes 27 to 50 minutes, of which the harvest (teachers and
tagging) is about 18 minutes, eleven CRF-head runs: shared over a job's
students a names student costs about 6 times the CRF head's run and twice
MR6's, and charged to one student about 14 times, not counting the
corpus's construction. The students see 2.8 to 4.6 times more sentences
per epoch and reach their best epoch after 5 to 6 epochs against the CRF
head's 8, so they take about two to three times as many updates up to it
(2.0 for MR3, 2.9 for MR6, 3.1 for the names; Appendix B).

**Figure 1.** The recipe. Three CRF-head teachers trained on the labeled
sentences tag a 40,000-sentence corpus; the mentions all three tag alike
with every marginal at 0.9 or more form the lexicon; the replacement pool
holds the training mentions and the lexicon; mention replacement writes
six copies of each training sentence with its mentions swapped; the
student trains on the gold sentences and the copies.

## 5 Results

### 5.1 The pre-registered decisions

Table 2 lists the eight decisions of the names route in the order they
were taken, with the seeds each used.

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

The reference F1 on MuLMS is the 7-seed mean (48.54); the 10-seed mean
quoted in Section 1 is 48.66. D5 and D5′ are two readings of one protocol
text, which named the decision batch's seeds as the units' source and, in
the same paragraph, "each unit its 10-seed mean" (Section 3): D5 is the
analysis as executed, D5′ the batch reading.

Three decisions pass, and they are not three pieces of evidence: D3 and
D4 are the same four papers at two granularities, and D5's eleven units
contain those four papers. The one independent corpus, MuLMS, fails every
single-model decision (D1, D2), where 6 of its 7 subfields gain against
the CRF head (sofc by 11.9, poly loses 2.4) and the differences have a
standard deviation of 4.7 points across subfields. On MaterioMiner every
paper gains at both granularities. The pooled estimate is stable (post
hoc): the mean of the two corpus means is 3.73; 9 of the 11 units are
positive on the registered 10-seed basis and 10 on the 7-seed basis; the
exact two-sided sign-flip permutation test gives p = 0.011 and 0.007; and
leaving any one unit out keeps the lower bound above 0 (lowest +0.44 and
+0.54, without semiconductors). Figure 2 shows every unit.

**Figure 2.** Per held-out unit, the micro-F1 difference of MR6 + names
from the CRF head (circles) and from MR6 (squares): MuLMS's 7 subfields
on the decision seeds, MaterioMiner's 4 papers at both granularities and
SOFC-Exp's 5 folds, where MR6 was not run. The bands are the pooled
11-unit intervals of D5′ and D7.

### 5.2 Whose gain is it

Table 3 gives every comparison among the four systems on the same 11
units, on both seed bases, and the MaterioMiner grids; of its rows only
D5 (10-seed basis) and D7 (decision seeds) in the first block and D3 and
D4 in the second were pre-registered; the rest are post hoc.

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
replacement as published, +0.33 from the three extra copies and +0.80
from the harvested names; only the first term has an interval above 0,
and that post hoc (mention replacement's own pre-registered decisions
passed on MaterioMiner and failed on MuLMS alone). The
10-seed basis gives the same picture (+2.35, +0.22, +1.02). Against the
published method, then, the recipe's whole increment is +1.13 [−0.51,
+2.76], and the pre-registered pass over the CRF head is a property that
mention replacement already has on these units: its own 11-unit gain,
computed post hoc from runs that predate the names route, is +2.35
[+0.94, +3.75] on the 10-seed basis, with 10 of 11 units positive. Per
corpus the picture holds: on MaterioMiner, fine-grained, MR3 gains +2.77
[+1.51, +4.03] over the CRF head and MR6 + names +1.50 [−1.76, +4.75] over
MR3; coarse-grained, MR3 gains +1.28 [+0.06, +2.51] and MR6 + names +0.49
[+0.06, +0.92] over MR3, an increment that includes the three extra
copies (the names' own increment over MR6 there is +0.70 [−1.60, +3.00]).
That cell is one of three intervals in this paper comparing a names
variant with MR3 that lie above 0, with the 2-of-3-teachers
ablation on 3 seeds (+1.56 [+0.07, +3.05], Section 5.3) and SOFC-Exp
(+0.45 [+0.04, +0.86], Section 5.5); none is a pre-registered decision,
and the first has 3 degrees of freedom (Table 3, second block). On MuLMS MR3 gains +2.10 [−0.31, +4.51] over the CRF head
at 10 seeds and its votes pass (+2.71 [+0.22, +5.20]).

The names decision D7 was expected to be close in its protocol, which
foresaw a gain of +1.5 to +1.65 from the seeds in hand and, with a
between-unit spread near 3 points, an 11-unit half-width of about 2, and
put the chance of a pass at about 0.4; "not established" is thus the
designed outcome of a small effect, not evidence of none. The interval's
upper bound, +2.3, is as large as mention replacement's whole gain; the
comparison detects 2.0 points at 80% power.

### 5.3 The names' source and the recipe's settings

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

**Settings (reported, 3 seeds).** Each ablation changes one setting and
runs on the 7 subfields at the screen's seeds (0 to 200) against the
recipe's own runs on those seeds, which are the selected variant's and
so a slightly flattering reference. Table 4:

| Setting (the recipe's value) | Distinct names (summed over the 7 subfields) | Unseen test names in the lexicon | Micro F1 | Δ vs the recipe [95% CI] | Δ vs MR3 | Δ vs CRF head |
|---|---|---|---|---|---|---|
| the recipe: 40,000 sentences, threshold 0.9, 3 of 3 teachers, a uniform pool | 54,095 | 22.2% | 52.09 | | +1.45 [−1.35, +4.25] | +3.16 [−1.96, +8.28] |
| corpus: its first 10,000 sentences | 19,171 | 16.4% | 51.94 | −0.15 [−1.78, +1.48] | +1.30 [−1.14, +3.74] | +3.01 [−1.21, +7.23] |
| corpus: its first 20,000 sentences | 32,461 | 20.1% | 51.93 | −0.16 [−2.45, +2.14] | +1.30 [−0.56, +3.15] | +3.00 [−0.49, +6.49] |
| filter: marginal threshold 0.7 | 77,879 | 31.5% | 51.93 | −0.16 [−1.62, +1.31] | +1.29 [−0.95, +3.53] | +3.00 [−1.85, +7.85] |
| filter: 2 of 3 teachers agree | 78,604 | 28.8% | 52.20 | +0.11 [−1.58, +1.81] | +1.56 [+0.07, +3.05] | +3.27 [−0.85, +7.38] |
| pool: harvested names in half of the draws (at least 85% uniform) | 54,095 | 22.2% | 52.18 | +0.09 [−1.63, +1.80] | +1.54 [−0.74, +3.82] | +3.24 [−0.73, +7.22] |

Summed over the seven subfields, the lexicons range from 19 thousand to
79 thousand distinct names and from 16% to 32% of the unseen test names,
and micro F1 stays within 0.16
of the recipe's 52.09; every interval against the recipe covers 0, the
votes' too (+0.20, −0.32, +0.34, +0.45, +0.08), and per subfield the
differences run from −3.2 to +2.9, the spread of 3-seed means. These
readings detect differences of 2 to 3 points; what they show is the
absence of any effect of that size, not the equality of the settings.
Within that limit, a quarter of the corpus (a quarter of the teachers'
tagging) with the strictest filter does as well as the rest, consistent with
Section 5.2: if the names add under a point, the settings
that govern how many names enter cannot show an effect. On the unseen
names the settings do move the trade of Section 5.4 (exploratory; the
screen's 3 seeds pooled over the subfields, against the recipe's screen
runs at 48.1% recall and 50.0 precision, computed from their saved
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
(71.0 to 71.2); the point moves almost along an iso-F1 curve. With the
mixed corpus the figures are the same (46.7% at 52.4). On MaterioMiner's
papers, fine-grained, the names add to both recall and precision on
unseen names (39.7 to 41.1 and 49.2 to 53.2 over MR6; F1 43.9 to 46.4),
and coarse-grained a little of each (53.0 to 54.2 and 52.4 to 53.2; F1
52.7 to 53.7).

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
the seen ones; the MaterioMiner panels use the MuLMS panel's markers.

Whether this is name knowledge or a generic shift toward tagging more
spans, the recorded data do not say. Per subfield, the lexicon's coverage
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
(semiconductors and solid oxide cells). The sign of the raw correlation
with coverage, if anything, runs against name knowledge. A mixed corpus
that covers fewer unseen names scores slightly higher on unseen-name F1
(49.35 against 49.01) with lower recall (46.7 against 47.3) and higher
precision, its 49 runs being the decision seeds only. The
decisive analysis, recall on the unseen test names that are in the
lexicon against those that are not, was not run: the lexicons were not
stored with the runs, and rebuilding them takes a teacher pass per corpus
(the teachers are deterministic); nor was a recall-matched control (a
bias toward tagging) run against the names. Both are the first things to
do (Section 7).

### 5.5 Where nothing helps: SOFC-Exp

On SOFC-Exp's five folds of held-out articles the recipe with the SOFC
corpus scores 82.75 against the CRF head's 82.79, −0.04 [−1.21, +1.14]
(D6). It gains +0.45 [+0.04, +0.86] over MR3 on every fold (reported), but
MR3 itself does not gain there (−0.48 [−1.36, +0.39]), nor on the
official split (−0.29 [−1.04, +0.47]). Three things differ on this
corpus, in one place: only 46% of a fold's test names are unseen (81%
fine- and 79% coarse-grained on MaterioMiner's papers, 88% on MuLMS's
subfields) and the CRF head already finds 80% of them; the baseline is
at 82.8 F1 with a random-split gap of 3.5 points, so there is little
headroom; and the SOFC corpus holds 36% of the unseen test names
(27–43% per fold) against 55–81% on MuLMS, the filter admitting 20–33%.
Training size does not explain mention replacement's null there: trained
on 10 of the official split's 26 training articles (5 draws of 156–248
sentences, MaterioMiner's size; 10 seeds each) and scored on its test
articles, MR3 gains +0.50 [−0.47, +1.46] over the CRF head, still nothing
(the recipe was not run at that size).
Pooled over the 16 units of the three corpora the recipe's estimate is
+2.45 [+0.50, +4.40] (reported); the claim stays the two-corpus one, with
SOFC-Exp as its boundary condition.

### 5.6 Per entity type

Table 6 gives, per type, the F1 of the CRF head, MR6 and MR6 + names on
MuLMS's subfields, averaged over the (seed, unit) pairs whose test set
has the type (the MaterioMiner tables, both granularities, are in the
repository); these are exploratory and are not decided on. The pattern of
Section 5.2 holds for most types. On MuLMS, MR6 + names gains over the CRF
head on the 12 types with a non-zero score (RANGE, 3 entities per unit,
scores 0 under every system), and MR6 holds more than half of that gain on
9 of them (not on MEASUREMENT, TECHNIQUE and DEV, where the names' own
increment is +1.7 to +2.8). On MaterioMiner, coarse-grained, MR6 + names
gains over the CRF head on 16 of 27 types and MR6 holds more than half of
the gain on 13 of those; the names also lose against MR6 on several types,
two where MR6 had gained (MEP −8.0, STD −7.0) and one where it had not
(MAT −5.8), and most on types with a handful of test entities (PRP −10.2,
CPR −9.2).

**Table 6.** Per entity type on MuLMS's 7 held-out subfields (decision
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

## 6 What else was tried, and the count

The recipe was reached through four protocols on MuLMS's held-out
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
  7 subfields; Table 7, with the plan's labels, under which V5 was not
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

| Variant (3 seeds, 7 subfields; no decision) | Micro F1 | Δ vs MR3 [95% CI] | Δ vs CRF head [95% CI] |
|---|---|---|---|
| MR6 + names, mention filter (the recipe, V4) | 52.09 | +1.45 [−1.35, +4.25] | +3.16 [−1.96, +8.28] |
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
  pemfc +4.0) and hurts elsewhere (steel −10.0, graphene −5.1), −1.0 on
  average over the seven subfields. The subfields it helps are the ones
  where MR6 + names gains most (Section 5.4). The probe's teachers were
  trained on CPUs and its lexicons differ from the GPU runs' (the same
  filter admits 9–34% of a subfield's unseen names there against 7–36% in
  the runs, with a higher mean, 25% against 23%, and a different order of
  subfields), so the probe speaks to the dictionary rule, not to the runs' lexicons.
- *A second rule, label propagation within a document* (Appendix D): a
  term the model tags in most of its occurrences in a document is tagged
  in the rest. Two of its ten pre-registered decisions pass, on two
  corpora, both under a point; it adds nothing to mention replacement.

In all, the names route took eight pre-registered decisions in weeks 12
to 14, of which three pass, sharing one corpus's four papers (Section
5.1); with weeks 10 and 11, fourteen decisions on the route, four
passing; about 0.35 of fourteen would pass by chance under independence,
which these decisions do not have.

## 7 Discussion

**For practitioners.** With a few hundred annotated sentences and new
documents to read, use mention replacement as published (MR3): its gain
on held-out documents passed pre-registered on MaterioMiner and is above
0 post hoc over the 11 units of both corpora (+2.35 [+0.94, +3.75]); on
MuLMS alone its decision failed. It costs twice the baseline's training
time and no corpus. The extra copies and the harvested names are not
shown to add to it. Against MR3 the names cost three times its training
time with the harvest shared over a job's students and seven times
charged to one student, plus a 40,000-sentence corpus (Appendix B), for
an increment of +1.13 [−0.51, +2.76]; against the matched control the
increment is +0.8 [−0.7, +2.3], which on MuLMS buys 3.6 points of recall
on unseen names for 3.0 of precision. The absolute levels keep these
gains in proportion: on MuLMS's held-out subfields the CRF head scores
48.5, MR6 50.8 and MR6 + names 51.7, and the best of them still finds
only half of a subfield's entities (recall 50%). If recall on new names
is what you need, the names buy it there, though an untested decoding
bias might buy the same (Section 5.4); if F1 is, stop at mention
replacement. Expect nothing from either where most of a new document's
names are already in training, as on SOFC-Exp.

**What would settle the names question.** Three analyses not run here:
recall split by whether an unseen test name is in the lexicon, which
needs the lexicons rebuilt (one teacher pass per corpus); a recall-matched control, a
bias toward tagging more spans calibrated on validation data, against
which the names' extra recall would have to show as precision; and a
training-size curve on MuLMS, whose six other subfields hold about 8,000
labeled sentences that the 340-sentence regime leaves unused, since
mention replacement's gains shrink with training size (dai2020analysis;
torres2024augmentation). Two baselines were dismissed on 3 seeds and
deserve 10: self-training on the same corpus and the lexicon as a
feature. And the lexicon's precision, never estimated, could be read
from a 100-name sample per type.

**What this paper does not compare.** No large language model, as an
extractor, a teacher or a generator of same-type names (zhou2022melm;
hu2023entity), was run, for want of a budget; an open-weight model in a
few-shot setting on a held-out subfield is the natural first comparison.
A single encoder was used. The corpora come from one index, Europe PMC,
whose materials coverage is the PMC-deposited subset of the literature,
with 10–33% of each corpus under non-commercial or no-derivative
licences (Appendix C), which bears on redistribution, not on this test.

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

**Lexicons from unlabeled text.** Gazetteers have been generated from
unlabeled text since Kozareva (kozareva2006bootstrapping), who extracted
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
self-training variant follows BOND's recipe on the same topic corpus,
where Du et al. (du2021selftraining) self-train on sentences retrieved
from the web. What this paper adds is not a new combination so much as the
test: the lexicon's contribution isolated against a matched control, its
source and its filter varied, on held-out documents, under protocols
fixed in advance, and the finding that the combination's gain is mention
replacement's.

**Materials NER and unseen documents.** MatSciBERT (gupta2022matscibert)
is pre-trained on materials papers; Weston et al. (weston2019named)
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
(dror2018hitchhiker), whose advice the protocol follows, and Card et al.
(card2020power) show how underpowered typical NLP comparisons are, which
is why Appendix E reports what each comparison here could detect. Our
companion study measures the gap and the unseen-name shares on the three
corpora and finds that every gain that holds on held-out documents is a
gain on unseen names.

## 9 Conclusion

A tagger trained on a few annotated materials papers misses most of a
new paper's entity names. Giving it names harvested from unlabeled text
by three agreeing teachers, through mention replacement, improves on the
baseline on held-out documents of two corpora under a pre-registered
protocol; but mention replacement without the names gains most of it
on the same units, the names' share, +0.8 points with an interval from
−0.7 to +2.3, is not established, no dependence on their source or on the
filter was detected within about 2 to 3 points, and on MuLMS what they
buy is recall on unseen names at the price of precision, where the
baseline is weakest. We report the eight decisions, the criterion
revision, the amendments and the one deviation, and the analyses that
would settle the question. The practical advice is mention replacement; the
methodological advice is to test a names method against a matched copies
control and on documents the training data has never seen, with enough
seeds and units to detect a difference of a point or two between a
method and its control.

## Limitations

- **Few, heterogeneous units.** Seven subfields and four papers give 6
  and 3 degrees of freedom; the subfields differ by 24 points under the
  CRF head (32.49 to 56.82) and range from 2 to 11 articles, the papers
  from 41 to 201 sentences, and the pooled decision weights them equally
  as if drawn from one population. The leave-one-out and sign-flip
  checks (Section 5.1) are post hoc; a mixed model (random effects for
  units within corpora and for corpora, which would widen the interval by
  the between-corpus variance) was not fitted.
- **Three passes on one corpus's papers.** The MaterioMiner passes at two
  granularities and the pooled pass share four papers; MuLMS alone passes
  nothing single-model.
- **A revised criterion.** The criterion under which the recipe passes
  was adopted on 20 October on the author's decision to push for a
  methods claim,
  after the screen's first half and before the recipe's own screen
  result, and when its control was already known to meet it (Section 3);
  under the earlier criterion there is no claim.
- **MuLMS's training pool is about two articles.** The 400-sentence pool
  is drawn from a shuffled order of the other subfields' articles until
  400 sentences are reached, so each MuLMS model is trained on about two
  articles, a different pair per held-out subfield; the 88% share of
  unseen names and the 24-point spread between subfields reflect that.
- **One training regime.** Every positive result is at about 340 training
  sentences; MuLMS's full pool was never used, SOFC-Exp's larger training
  set is where nothing helps, and the literature predicts smaller gains
  for mention replacement as training grows. The claim is scoped to a few
  hundred annotated sentences.
- **The reference is not the strongest base.** On MaterioMiner,
  multi-task training with mention replacement beats the CRF head by more
  than the names route does (Section 4); the names were never run on that
  base, so whether they add to the best system is untested.
- **The names' effect is not isolated.** The lexicons were not stored
  with the runs, so recall was not split by lexicon membership (rebuilding
  them is a teacher pass per corpus); no recall-matched control was run;
  the lexicon's precision was not estimated; the per-subfield
  correlations of Section 5.4 are exploratory with 7 points and
  confounded; the recall–precision trade is reported as pooled rates, with
  no paired interval.
- **Null results are not equivalences.** The source decision and the
  ablations detect differences of about 2 to 3 points (Appendix E);
  smaller effects of the source or the settings are not excluded.
- **Baselines.** The gazetteer-feature and self-training variants were
  rejected on 3 seeds; no large language model was compared; one encoder,
  MatSciBERT, was used.
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
author's direction, to write code and drafts; the author is responsible
for the content. This paper shares data, baselines and protocol with a
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

| Decision | Protocol committed (times UTC) | Runs | Result recorded | Amendments |
|---|---|---|---|---|
| the route's earlier protocols (Section 6): mention replacement on MuLMS (week 10); names from the other subfields' sentences (week 11) | 17 October 20:21 (`a08356f`); 18 October 01:13 (`427e34c`) | 17–20 October | week 11: 18 October (MuLMS, `14ed50f`) and 20 October (MaterioMiner, `04764ef`) | — |
| D1, D2 (week 12) | 18 October 11:00 (`40631ce`; refined 11:05, `bd20cec`): the screen on seeds 0–200, the decisions on seeds 300–900 so that the selection cannot bias them; the MaterioMiner analogue gated on D1; the methods criterion: D1 plus a MaterioMiner pass | 20–24 October | 24 October | three amendments before any run added variants V4, V6, V7–V8 to the screen |
| D3, D4, D5 (week 13) | 20 October 09:14 (`808ff4b`): the pooled decision over both corpora as a second route to the claim, "each unit its 10-seed mean", the stop rule lifted; amended twice the same day, before any decision run: 13:53 (`5f7f827`), eighteen minutes after the screen chose the recipe, D6 added conditional on D1 and the analogue, which the morning's text ran whenever the screen had a winner, gated on D1; 18:42 (`de62446`), D6's condition widened to D1 or D5 and the analogue's morning condition restored, the gating recorded as a mistake | 24–25 October | 25 October (`3432988`) | the analysis took MuLMS's 10-seed means; the 7-seed reading is D5′ |
| D6 (week 13) | 20 October 13:53 (`5f7f827`), conditional on D1; widened to D1 or D5 at 18:42 (`de62446`) | 25 October | 25 October | — |
| D7, D8 (week 14) | 29 October 12:07 (`2886267`), two minutes after the first batch was pushed and five hours before its results | 29 October | 29 October | — |
| ablations (week 14) | 29 October, as above | 31 October | 31 October | — |

## Appendix B: hyperparameters and compute

MatSciBERT (110M parameters), token classification on the first sub-word,
a linear-chain CRF with IOB2 constraints; AdamW, learning rate 5e-5,
linear warm-up, batch size 16, up to 20 epochs, early stopping on
validation micro F1 with patience 3, the best epoch's weights evaluated;
mention replacement with probability 0.5 per mention, 3 or 6 copies,
copies that change nothing dropped; teachers' marginals from
forward–backward, threshold 0.9, all three teachers agreeing on span and
type. On MuLMS's decision seeds a CRF head run takes 96 s on a T4 (best
epoch 7.1 on average), an MR3 run 198 s (4.7) and an MR6 run 296 s (4.1).
A names run's stored time is its job's: the three teachers, their tagging
of the corpus and the job's three or four students together, 2,099 s on
average over the 14 jobs (1,598–2,972 s; the best epoch after 5.6 epochs). Per student that
is 609 s, 6.4 times the CRF head's run; the harvest alone, the job less
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
batches alone took 19 session hours (5.2, 4.3 and 9.7).

## Appendix C: the unlabeled corpora

Each corpus is 40,000 sentences of open-access full texts found on Europe
PMC by a title query (for MuLMS's solid-oxide subfield: "solid oxide",
"SOFC", "SOEC" and their plurals; for SOFC-Exp's corpus the same without
"SOEC"; the other queries are in the repository's reports), with the dataset's own articles excluded by identifier, every
sentence sharing a word 8-gram with a dataset sentence dropped (and, for
MuLMS, any 50-character run), sentences of 5 to 120 words with at least
half alphabetic characters, and formulas longer than 30 characters cut.
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
occurrences untagged on the official test set. It adds nothing to
mention replacement (+0.57 [−0.12, +1.25] on MuLMS). Two of its ten
pre-registered decisions (six in week 7, four in week 13) pass, where
about 0.25 would by chance.

## Appendix E: detectable differences

The smallest true mean difference each comparison detects with 80% power
at the two-sided 5% level, from the half-width of its observed interval
and its degrees of freedom (`ner_robustness.py`, `detectable`).

| Comparison | CI half-width | df | detectable at 80% power |
|---|---|---|---|
| D5: MR6 + names − CRF head, pooled, 10-seed basis | 2.63 | 10 | 3.7 |
| D5′: the same on the decision seeds | 2.54 | 10 | 3.5 |
| D7: MR6 + names − MR6, pooled | 1.46 | 10 | 2.0 |
| MR6 + names − MR3, pooled (post hoc) | 1.63 | 10 | 2.3 |
| MR6 − MR3, pooled (post hoc) | 0.58 | 10 | 0.8 |
| MR3 − CRF head, pooled (post hoc) | 1.46 | 10 | 2.0 |
| D1: MR6 + names − CRF head, MuLMS | 4.32 | 6 | 5.9 |
| D2: MR6 + names − MR3, MuLMS | 2.55 | 6 | 3.5 |
| MR6 + names − MR6, MuLMS (part of D7) | 2.02 | 6 | 2.8 |
| D8: own-topic − mixed corpus, MuLMS | 1.83 | 6 | 2.5 |
| D3: MR6 + names − CRF head, MaterioMiner fine | 2.88 | 3 | 3.8 |
| D4: MR6 + names − CRF head, MaterioMiner coarse | 1.10 | 3 | 1.4 |
| ablation, 10,000 sentences − the recipe (3 seeds) | 1.63 | 6 | 2.2 |
| ablation, 20,000 sentences − the recipe (3 seeds) | 2.29 | 6 | 3.1 |
| ablation, threshold 0.7 − the recipe (3 seeds) | 1.47 | 6 | 2.0 |
| ablation, 2 of 3 teachers − the recipe (3 seeds) | 1.70 | 6 | 2.3 |
| ablation, names in half of the draws − the recipe (3 seeds) | 1.71 | 6 | 2.4 |

## Note on this draft

This draft lives in the repository with the code and results. Its tables
and figures correspond to generated files: Table 2 to
`docs/tables/table6_held_out_decisions.md` and `table2_decisions.md`;
Table 3 to `table9_robustness.md`; Table 4 to `table7_ablations.md`; Table
5 to `results/gpu/ner_mentions.json` (and `table4_unseen.md` for the CRF
head); Table 6 to `table8_types.md` and `tableA2_types_fine.md`; Table 7
to the week-12 screen in `docs/ROADMAP.md`; Figure 1 to
`docs/figures/fig8_method.png`, Figure 2 to `fig9_units.png`, Figure 3 to
`fig7_names.png`; Appendix E to `results/gpu/ner_robustness.json`. The
claims it rests on are C2, C17 and C20–C26 of `docs/PAPER.md`; citation
keys refer to `docs/references.bib`. For a journal's structure, Section 1
is the Introduction, Sections 5 and 6 the Results, Sections 7 to 9 the
Discussion, and Sections 2 to 4 with the appendices the Methods.
