# Harvested names are not shown to add to mention replacement for materials named-entity recognition on held-out documents: a pre-registered test

Hatem Madi (affiliation to be added; hatem.madi15@gmail.com)

## Abstract

Materials named-entity recognizers train on a few annotated papers but
must read new ones: our baseline scores 77 F1 on random splits of
MaterioMiner but 46 on a held-out paper, where 81% of the entities are
names no training sentence annotates. We test whether harvested names
help. Three baseline models tag 40,000 unlabeled sentences; names they
confidently agree on join the pool of mention replacement, which swaps
the names in six copies of each training sentence. Over 11 held-out
subfields and papers of two corpora, this gains 3.58 F1 points [95% CI
0.95, 6.21] over the baseline, under a criterion adopted mid-study,
which mention replacement already met. On the same units three-copy
mention replacement without names gains 2.35 [0.94, 3.75] (post hoc),
and the pre-registered test of the names against a six-copy control
gives 0.80 [−0.66, 2.26], excluding gains above 2.3 points. A
later-registered follow-up on MaterioMiner's strongest system gives 0.96
[−0.01, 1.92] and 0.85 [−0.72, 2.42] over a matched control, neither
passing. Post hoc, on MuLMS the names raise recall by 7.3 points [3.3,
11.3] on the fifth of unseen names their lexicon holds. Mention
replacement, not the names, carries the gain where most test names are
unseen.

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

Every decision was fixed in a protocol committed to the repository
before its results (vanmiltenburg2021preregistering), with 7 to 10
training runs (*seeds*) per held-out *unit* (a MuLMS subfield, a
MaterioMiner paper or a SOFC-Exp fold of articles) and a decision rule
stated in advance: a system improves on its reference
when the 95% interval of the mean paired difference in micro F1 over the
units lies above 0. A number in square brackets is that 95% Student-t
confidence interval, over the units named; comparisons specified after
their results were known are labelled post hoc. Protocols are cited by
date, and Supplementary Note 1 lists each with its commit. The last four protocols of the
*names route*, the sequence of protocols that tested harvested names,
took ten such *decisions*, D1 to D10 (Table 2). The results, with the
qualifications Section 3 records:

- MR6 + names improves on the CRF head pooled over the 11 units of MuLMS
  and MaterioMiner (D5: +3.58 [+0.95, +6.21]; Section 5.1), under a pooled
  criterion adopted mid-study, when plain mention replacement was already
  known to meet it (Section 3). MuLMS alone does not pass (+3.19 [−1.13,
  +7.51]).
- That gain is not shown to be the names'. Mention replacement with
  three copies and no names (MR3) gains +2.35 [+0.94, +3.75] on the same
  units (post hoc). Decomposed on the seven seeds the screen did not see,
  three more copies add +0.33 [−0.25, +0.92] (post hoc) and the names
  +0.80 [−0.66, +2.26] (D7, the pre-registered test of the names), an
  interval that excludes a gain above 2.3 points but not a smaller one
  (Section 5.2).
- Neither the names' source nor the lexicon's settings changes F1
  detectably (Section 5.3). On MuLMS the names raise recall on unseen
  names (+2.56 [+0.76, +4.36], paired by subfield, post hoc) and lower
  precision in pooled rates, a loss not established when paired; the
  recall gain is established on the fifth of the unseen names the
  lexicon holds and not on the others (post hoc, Section 5.4). On
  SOFC-Exp, where half the test names are seen, no system is shown to
  improve on the baseline (Section 5.5).
- A follow-up registered after D1 to D8 had been written up tested the
  names on
  MaterioMiner's strongest system, multi-task training with mention
  replacement: over a six-copy matched control they add +0.96 [−0.01,
  +1.92] fine-grained and +0.85 [−0.72, +2.42] coarse-grained (D9, D10),
  neither passing (Section 5.6).

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
entities of a test set (seqeval, nakayama2018seqeval, in its default mode, as the
datasets' own code scores), in percentage points.

**Terms.** A *decision* is a comparison with a pre-registered verdict; a
comparison labelled *pre-specified* was fixed in a protocol but carries
no decision; *post hoc* means computed after its results were known. A
*screen* is a 3-seed comparison used to choose among variants; the
*decision seeds* are the 7 seeds (300 to 900) a protocol decided on
after a screen on the other 3. A *vote* is the majority vote, entity by
entity, of the three models of a seed triple; votes appear here only
where a protocol decided on them (Section 6). The *MaterioMiner analogue*
of a MuLMS system is the same system run on MaterioMiner's held-out
papers with the fatigue corpus. The *sign-flip test* is the exact
permutation test of a mean paired difference over all 2^n assignments of
signs to the n units, two-sided. The protocols of the names route were
fixed between 17 and 31 October 2026 and are cited by date; the last, of
31 October, is a *follow-up* registered after D1 to D8 had been written
up (Section 5.6; Supplementary Note 1).

**Decision rule.** A method improves on its reference when the 95%
Student-t interval of the mean paired micro-F1 difference lies above 0,
the pairs being the held-out units (7 subfields, 4 papers, 5 folds), each
side the mean of its seeds; with 4 papers the interval has 3 degrees of
freedom, with 7 subfields 6. A pooled decision takes the units of two
corpora together (11 units, 10 degrees of freedom) and requires, in
addition, that neither corpus's own interval lie below 0. Each decision
is one-sided at 2.5%, so of the ten decisions about 0.25 would pass if
nothing worked (0.45 of the eighteen counted in Section 6); the decisions
are not independent (Section 5.1). At 80% power the names against MR6
detect 2.0 points pooled over the 11 units and the names against the CRF
head 3.5 to 3.7 (Supplementary Note 9 gives every decision and the main
comparisons).

**Pre-registration, and what changed.** Each protocol was committed to
the repository before its results, all but one before its runs, with its
decisions and the comparisons pre-specified without a decision;
amendments are marked with their reason and time (Supplementary Note 1
gives the commits and the chronology). Three points must be disclosed.

First, the criterion for a methods claim was revised on 20 October, on
the author's own decision, recorded in the protocol, to push for a
methods paper. The protocol of 18 October had required a pass against the
CRF head on MuLMS alone plus a pass on MaterioMiner, and ran the
MaterioMiner analogue (D3, D4) only if D1 passed; the project's methods
plan carried a stop rule, "no fourth protocol on MuLMS". The protocol of
20 October added a pooled decision over both corpora (D5) as a second
path to the claim, lifted the stop rule and lifted the gate on D3 and D4
(re-imposed at 13:53, after the screen had chosen MR6 + names, and
lifted again at 18:42), so all three passes depend on that day's
changes. At that moment the
screen's results were known for three of its seven variants (MR6 and the
two gazetteer variants) but not for MR6 + names, and the protocol itself
computed, from runs already made, that plain mention replacement would
pass the new pooled criterion (+2.34 [+0.94, +3.75], labelled post hoc;
+2.35 in Section 5.2 by the analysis's rounding). The criterion under
which MR6 + names passes was thus adopted knowing that mention
replacement already met it, before the names system's own pooled result
existed. Three amendments followed the same day, all before any decision
run of the names route. Under the 18 October criterion MR6 + names has no
claim, since its MuLMS decision failed; under the revised one it passes.

Second, the pooled decision's protocol text named MuLMS's decision batch
(seeds 300 to 900) as the source of its units and, in the same
paragraph, "each unit its 10-seed mean"; the analysis as executed took
the 10-seed means, which include the screen's three seeds. We report
both readings, which agree (D5, +3.58 [+0.95, +6.21]; D5′, +3.58 [+1.04,
+6.13]), quote the executed one first, and use the 7-seed basis for the
post hoc decomposition, since those seeds the screen never saw.

Third, the protocol of D7 and D8 (29 October) was committed two minutes
after its first batch was pushed and five hours before any result
arrived, and that of D9 and D10 (31 October) after D1 to D8 had been
written up and before its runs (Section 5.6). Every comparison of the
names route not in Table 2 is either pre-specified or post hoc, and
labelled so; the decisions of earlier protocols quoted here (mention
replacement's, the multi-task model's, the label-propagation rule's) are
labelled where they appear, and the robustness checks of Sections 5.1
and 5.2 are post hoc.

**Runs.** Kaggle T4 GPUs with deterministic CUDA, one frozen commit per
batch, a pinned image and pinned libraries; model snapshots pinned by
commit. GPU sessions began by repeating an earlier run, and every repeat
was identical (Supplementary Note 2 lists the two sessions that did not).
Hyperparameters and compute are in Supplementary Note 2.

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
  validation F1 with patience 3. It is the reference of D1 and D3 to D6 and
  the baseline every system is measured from. It is not the strongest
  single model we know on every corpus: on
  MaterioMiner's held-out papers, multi-task training over both
  granularities gains +3.72 [+0.57, +6.87] over it and multi-task training
  with three-copy mention replacement +6.73 [+6.03, +7.43] fine-grained (our
  companion evaluation study); neither transfers to SOFC-Exp, and MuLMS
  has one type level, so the CRF head is the reference the three corpora
  share. The names were tested on that system, with six copies, in a
  follow-up (Section 5.6).
- **MR3.** Mention replacement (dai2020analysis): three copies of each
  training sentence with an entity, each mention replaced with
  probability 0.5 by a training mention of the same type; copies that
  change nothing are dropped. The student trains on the gold sentences and
  the copies with the CRF head's training settings and seed.
- **MR6.** The same with six copies: the matched control for the names.
- **MR6 + names.** MR6 whose replacement pool also holds a lexicon
  harvested from unlabeled text:
  - *Teachers.* The CRF head's own runs of seeds 0, 100 and 200 on the
    fold's training sentences, so the teachers are the baseline, not a
    stronger model.
  - *Corpus.* 40,000 sentences of open-access full texts from Europe PMC
    (europepmc2015europe) found by title terms of the held-out subfield (for MaterioMiner, of
    fatigue; for SOFC-Exp, of solid oxide cells), with the datasets'
    own articles excluded by identifier and every sentence sharing a word
    8-gram with a dataset sentence dropped (for MuLMS also a 50-character
    run); 290–368 articles with text per corpus (Supplementary Note 3). A topic
    corpus holds 55–81% of a held-out MuLMS subfield's unseen test names,
    the fatigue corpus 50–62% of a held-out paper's fine-grained (49–65%
    of its coarse-grained), the SOFC corpus 36% of a fold's (measured before the runs and
    recorded in the protocol of 20 October).
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
    harvested entries are at least 92% of a unit's pool, so most
    replacements are harvested names.

**An example.** Take a training sentence, constructed for illustration,
"The Pt catalyst lowers the overpotential", with Pt annotated MAT and
overpotential PROPERTY. A copy made by MR6 might read "The Pd catalyst
lowers the overpotential", the replacement drawn from the training
mentions; a copy made by MR6 + names for the held-out electrolysis
subfield might read "The MoS2 catalyst lowers the Tafel slope", both
replacements drawn from that subfield's lexicon. Its most frequent
entries are, as MAT, Ni,
Pt, hydrogen, water, electrode, Co and catalyst, and, as PROPERTY,
current density, scan rate and Tafel slope (the rebuilt lexicon of
Section 5.4); a materials scientist might not count electrode or catalyst
as materials, and the lexicon's precision was not estimated.

Every run stores its teachers' scores, the lexicon's counts and the share
of the unseen test names it holds, the corpus's hash and the filter. A
names student costs about 6 times the CRF head's run with the harvest
shared over its job's three or four students and about 14 times charged
to one student, not counting the corpus's construction (Supplementary
Note 2).

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
follow-up of Section 5.6.

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
seeds, pre-specified on 29 October; Section 5.2); 6 of
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
units, on both seed bases, and on MaterioMiner per granularity. In the
first block only D5 (10-seed basis) and D7 (decision seeds) are
pre-registered, and every other pooled entry is post hoc; in the second,
D3 and D4 are pre-registered, MR3 − CRF head holds mention replacement's
own decisions (companion study), MR6 + names − MR3 was pre-specified on
18 October, and the other rows on 29 October.

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
votes passed; Section 6). The 10-seed basis gives the same picture
(+2.35, +0.22, +1.02). Against MR3 the whole increment of MR6 + names is
+1.13 [−0.51, +2.76]: the pre-registered pass over the CRF head is a
property that mention replacement already has on these units. On MuLMS
alone, where D1 failed, MR6 gains +2.68 [+0.13, +5.23] over the CRF head
on the decision seeds (+2.33 [−0.27, +4.92] on all ten; pre-specified on
29 October). Three intervals in this paper comparing a names variant with
MR3 lie above 0, none of them a decision: MaterioMiner coarse-grained
(+0.49 [+0.06, +0.92], 3 degrees of freedom), the 2-of-3-teachers
ablation on 3 seeds (+1.56 [+0.07, +3.05], Supplementary Note 4) and
SOFC-Exp (+0.45 [+0.04, +0.86], Section 5.5). Per entity type the
pattern holds for most types (Supplementary Note 6).

D7's protocol foresaw a gain of +1.5 to +1.65 from the seeds in hand, an
11-unit half-width of about 2 and a chance of a pass of about 0.4; "not
established" is thus the designed outcome of a small effect, not evidence
of none. The interval excludes a gain above +2.26, nearly as large as
mention replacement's whole gain (+2.46). In the follow-up on
MaterioMiner's strongest system the names' increment is +0.96 [−0.01,
+1.92] fine-grained and +0.85 [−0.72, +2.42] coarse-grained (Section
5.6).

### 5.3 The names' source and the lexicon's settings

**Source (D8).** For each held-out subfield a mixed corpus of 40,000
sentences, drawn equally from the six other subfields' corpora and built
the same way, holds 49–70% of the subfield's unseen test names against
the topic corpus's 55–81%. MR6 + names with the topic's own corpus scores
51.73 against 51.91 with the mixed one, −0.18 [−2.01, +1.65] paired by
subfield, in a comparison that detects 2.5 points; since the mixed corpus
holds most of the same names, D8 tests the topic's marginal names, not an
off-topic source.

**Settings (pre-specified, 3 seeds).** Five ablations, of corpus size
(10,000 and 20,000 sentences), threshold (0.7), agreement (2 of 3
teachers) and the names' share of the pool, keep micro F1 within 0.16 of
MR6 + names on the same seeds, every interval against it within −2.45
and +2.14, while the lexicon ranges from 19 to 79 thousand distinct names
and from 16% to 32% of the unseen test names (Supplementary Note 4).
Effects beyond those bounds are excluded; smaller ones are not.

### 5.4 What the names do

On MuLMS, pooled over the held-out subfields (Table 4; Figure 3 puts each
system's recall and precision on the unseen names on one plane with
curves of equal F1), the CRF head finds 39.4% of the unseen names at
55.8% precision, MR3 43.1% at 54.3, MR6 43.7% at 53.9 and MR6 + names
47.3% at 50.9: against the matched control the names raise recall on
unseen names by 3.6 points and lower precision by 3.0, a net +0.7 F1 on
them, the point moving almost along a curve of equal F1. Paired by
subfield on the decision seeds (post hoc), the recall gain is +2.56
[+0.76, +4.36], the precision loss −2.22 [−5.03, +0.60] and the net
+0.56 [−1.44, +2.56]: the recall gain is established, the precision loss
and the net are not. On MaterioMiner's papers the pooled rates of both
recall and precision on unseen names rise over MR6 (fine-grained 39.7% to
41.1% and 49.2 to 53.2; coarse-grained 53.0% to 54.2% and 52.4 to 53.2),
within paired intervals that cover 0 (Supplementary Note 5).

**Table 4.** Recall, precision and F1 on the unseen test names, pooled
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
legend names the markers. The mixed-corpus point uses seeds 300–900
(49 runs), the others all ten.

**Which names the gain falls on (post hoc; the analysis and its reading
fixed before its run).** The lexicons were not stored with the runs; an
analysis of 31 October rebuilt each from its unit's own teachers and
corpus, reproducing the stored teachers' scores and coverage counts in
every unit, and split the unseen test names by whether the lexicon holds
them (Table 5; the full analysis is in Supplementary Note 5). The reading
was fixed in advance: *name knowledge* if the names' recall gain over MR6
is larger on the held names than on the others, the paired interval of
the difference above 0; a *generic shift* if the two gains are alike,
the interval covering 0 with both gains positive; *mixed* if the interval
is above 0 and the gain on the other names also clears 0. On MuLMS the
lexicon holds 22% of the unseen test names, pooled over the subfields (7%
to 36% by subfield). On those, MR6 + names recalls 82.4% against MR6's
75.0% (+7.29 [+3.27, +11.30], paired over the seven subfields); on the
others, 36.8% against 35.0% (+1.12 [−0.78, +3.02]); the difference is
+6.17 [+2.44, +9.90], positive in every subfield. Read on intervals, as
the rule's "clears 0" was written, that is name knowledge; read on point
estimates it is mixed, since the gain on the other names is positive but
not established, and a gain of up to +3.0 there is not excluded. The copies alone show no
detectable split (MR6 over the CRF head, −1.84 [−5.27, +1.60]). The
precision side differs: precision falls on both parts (74.7% to 69.0% on
the held names, 46.4% to 44.0% on the others), and two thirds of the
wrong predictions the names add are names the lexicon does not hold. On
MaterioMiner no reading is determined: on intervals none of the three
outcomes is met, every gain interval covering 0, and on point estimates
both granularities read as a generic shift (fine-grained difference
+4.33 [−5.33, +13.98]; coarse-grained −1.54 [−3.60, +0.51]). So on MuLMS
the names do what they were built to do on the fifth of the unseen names
the lexicon holds, where the recall gain is established; on the other
names it is not, and the wrong predictions land on both parts, which
keeps the F1 increment within Section 5.2's noise.

**Table 5.** Recall gains on the unseen test names by lexicon membership
(post hoc): per comparison, the gain on the names the lexicon holds and
on the others, their difference, paired over the units with 95% CIs, and
the units where the gain on the held names is the larger. The lexicon
holds 22.2% of MuLMS's unseen test names and 9.8% and 18.2% of
MaterioMiner's (fine- and coarse-grained). The mixed-corpus row is split
by the topic corpus's lexicon, which was the one rebuilt. MuLMS on the
decision seeds, MaterioMiner on 10 seeds.

| setting | comparison | recall gain, names in the lexicon [95% CI] | recall gain, names not in it [95% CI] | in − out [95% CI] | units with in > out |
|---|---|---|---|---|---|
| MuLMS, 7 subfields | MR6 + names − MR6 | +7.29 [+3.27, +11.30] | +1.12 [−0.78, +3.02] | +6.17 [+2.44, +9.90] | 7/7 |
| MuLMS, 7 subfields | MR6 − CRF head | +3.39 [+0.92, +5.85] | +5.22 [+1.92, +8.52] | −1.84 [−5.27, +1.60] | 3/7 |
| MuLMS, 7 subfields | MR6 + names (mixed corpus) − MR6 | +6.16 [+1.32, +11.01] | +1.31 [−1.07, +3.68] | +4.86 [+1.50, +8.21] | 7/7 |
| MaterioMiner fine, 4 papers | MR6 + names − MR6 | +4.62 [−5.27, +14.50] | +0.29 [−4.23, +4.81] | +4.33 [−5.33, +13.98] | 3/4 |
| MaterioMiner coarse, 4 papers | MR6 + names − MR6 | +0.36 [−1.11, +1.84] | +1.91 [−1.46, +5.28] | −1.54 [−3.60, +0.51] | 0/4 |

A recall-matched control, a bias toward tagging more spans calibrated on
validation data, was not run against the names; the nearest evidence, the
companion study's calibrated bias on the CRF head (not matched to the
names and not run on MuLMS), was not shown to help on MaterioMiner or SOFC-Exp
(Supplementary Note 5). It is the first thing to do (Section 7).

### 5.5 SOFC-Exp: no gain shown

On SOFC-Exp's five folds of held-out articles MR6 + names with the SOFC
corpus scores 82.75 against the CRF head's 82.79, −0.04 [−1.21, +1.14]
(D6). It gains +0.45 [+0.04, +0.86] over MR3 on every fold
(pre-specified), but MR3 itself is not shown to gain there (−0.48 [−1.36,
+0.39], a decision of 20 October; −0.29 [−1.04, +0.47] on the official
split, an earlier decision, also failed). Three things differ on this corpus, in one place: only 46% of a
fold's test names are unseen (81% fine- and 79% coarse-grained on
MaterioMiner's papers, 88% on MuLMS's subfields) and the CRF head already
finds 80% of them; the baseline is at 82.8 F1 with a random-split gap of
3.5 points, so there is little headroom; and the SOFC corpus holds 36% of
the unseen test names (27–43% per fold, measured before the runs) against
55–81% on MuLMS, its lexicon 20–33%. Training size is not shown to
explain mention replacement's null there: trained on 10 of the official
split's 26 training articles (5 draws of 156–248 sentences,
MaterioMiner's size; 10 seeds each) and scored on its test articles, MR3
gains +0.50 [−0.47, +1.46] over the CRF head (an earlier decision,
failed; MR6 + names was not run at that size). Pooled over the 16 units of the three
corpora the estimate of MR6 + names is +2.45 [+0.50, +4.40]
(pre-specified); the claim stays the two-corpus one, with SOFC-Exp as its
boundary condition.

### 5.6 The names on the strongest system (follow-up)

The names had been tested only on the single-task CRF head. After D1 to
D8 had been written up, a follow-up on MaterioMiner's strongest system,
multi-task training with mention replacement (Section 4), was committed
on 31 October at 19:15 UTC, before any of its runs (Supplementary Note
1): two decisions by the rule of Section 3, the names against a matched
control on MaterioMiner's four held-out papers over 10 seeds,
fine-grained (D9) and coarse-grained (D10); the protocol expected a pass
to be unlikely. The control is the multi-task model (one encoder, a CRF
head per granularity) with six-copy mention replacement; the names system
adds a lexicon harvested as in Section 4 by three multi-task teachers
(the stored multi-task runs of seeds 0, 100 and 200, whose scores the
jobs reproduced), each tagging the fatigue corpus with both heads. A
mention is admitted when the three teachers agree on its coarse-grained
span and type and on its fine-grained type at exactly that span, or on
its having none, with every marginal at 0.9 or more on the coarse head
and, where a fine type is tagged, on the fine head; it joins the pool
under its pair of types. Table 6 gives the result.

**Table 6.** Follow-up: the names on the strongest MaterioMiner system.
Micro F1 over the four held-out papers (10 seeds; both granularities from
one model) with its 95% CI; the names against the matched control
(decisions D9 and D10), and both against multi-task training with
three-copy mention replacement, the multi-task model and the CRF head
(pre-specified). A bound of −0.00 lies just below 0.

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
At the observed spreads D9 and D10 would detect about 1.3 and 2.1 points
(the protocol expected 3.4 to 3.8 and 1.4 to 2.4); the fine-grained
spread rests on 3 degrees of freedom, and the same comparison on the
single-task base had a far wider interval (+1.29 [−2.51, +5.10], Table
3), so the interval, not the 1.3, is the result. On the unseen test names the point estimates of recall and
precision both rise, none established (pre-specified; Supplementary Note
8). One caveat: the two-head filter's lexicons hold fewer of the unseen
test names than the single-task lexicons of D3 and D4, 7.5% against 9.6%
fine-grained and 12.1% against 18.3% coarse-grained pooled over the
papers, lower in 7 of the 8 paper–granularity cells (by the runs' stored
counts, which call a name unseen when its type and words never occur in
training; Table 5's 9.8% and 18.2% use Section 2's words-only rule).
Whether that lowered the increment is not known, and a looser filter was
not tried.

## 6 What else was tried, and the count

MR6 + names was reached through two earlier protocols and a screen, on
MuLMS's held-out subfields and their MaterioMiner analogues, each
committed before its runs (Supplementary Note 7 gives them in full):

- *Mention replacement alone* (17 October): +2.10 [−0.31, +4.51] against
  the CRF head on MuLMS at 10 seeds, no; its votes against the CRF head's
  votes, +2.71 [+0.22, +5.20], pass. On MaterioMiner it had passed at both
  granularities in an earlier protocol (+2.77 and +1.28; companion study).
- *Names from the other subfields' unlabeled sentences* (18 October): a
  sentence-level filter whose names hold 4.4% of a held-out subfield's
  unseen test names; four decisions on MuLMS and MaterioMiner, none
  passing.
- *Seven variants on the topic corpora* (the screen of the 18 October
  protocol, 3 seeds on the 7 subfields; Supplementary Table S5): the
  selection took the largest mean gain over MR3, which favours the
  noisiest variant, and decides nothing. Two losers deserve a 10-seed test
  that was not run: self-training on the corpus's teacher-labeled
  sentences, the one variant whose interval against the CRF head cleared
  0 (by raising precision on unseen names), and the lexicon as a
  gazetteer match feature of the CRF head, below MR3 on 3 seeds.

Supplementary Note 7 also reports the lexicon applied as a dictionary to
the CRF head's predictions (a probe without decision: it helps only where
the head's recall trails its precision by 15 points or more) and label
propagation within a document (two of its ten decisions pass, both under
a point; it is not shown to add to mention replacement).

In all, the names route took ten pre-registered decisions between 18 and
31 October, two of them in the follow-up, of which three pass, sharing
one corpus's four papers (Section 5.1). Counting also the six decisions
of 17 and 18 October and the two of 20 October on mention replacement
without names on SOFC-Exp's folds (MR3, −0.48 [−1.36, +0.39]; multi-task
training with mention replacement, −0.11 [−0.69, +0.46]; both failing),
eighteen decisions of the protocols of 17 to 31 October tested the names
or mention replacement (the label-propagation rule's four aside; the
companion study's earlier decisions on mention replacement and
multi-task training are not counted). Four passed: three against the CRF
head, and mention replacement's votes against the CRF head's votes. Of
the ten, none of the five against a mention-replacement control or on
the names' source passes. If no method worked, about 0.45 of eighteen
would be expected to pass, whatever their dependence, which only makes
chance passes come together, as the shared units of D3 to D5 do.

## 7 Discussion

**For practitioners.** With a few hundred annotated sentences and new
documents to read, use mention replacement with three copies (MR3): its
gain on held-out documents passed pre-registered on MaterioMiner
(companion study) and is above 0 post hoc over the 11 units of both
corpora (+2.46 [+1.00, +3.91] on the decision seeds, +2.35 [+0.94, +3.75]
on all ten); on MuLMS alone its single-model decision failed and its
votes passed (Section 6). It costs twice the baseline's training time and
no corpus. The extra copies and the harvested names are not shown to add
to it, nor, in a follow-up, to MaterioMiner's strongest system (Section
5.6). Against MR3 the names cost three times its training time with the
harvest shared over a job's students and seven times charged to one
student, plus a 40,000-sentence corpus (Supplementary Note 2), for an
increment of +1.13 [−0.51, +2.76] on the decision seeds; against the
matched control the increment is +0.80 [−0.66, +2.26], which on MuLMS
buys 3.0 to 3.6 points of recall on unseen names for 2.9 to 3.0 of
precision in pooled rates, by seed basis (paired by subfield on the
decision seeds, +2.56 [+0.76, +4.36] recall and −2.22 [−5.03, +0.60]
precision). The absolute levels keep these gains in proportion: on
MuLMS's held-out subfields, on the decision seeds, the CRF head scores
48.5, MR3 50.8, MR6 51.2 and MR6 + names 51.7, and the best of them still
finds only half of a subfield's entities (recall 50%). If recall on the
names an unlabeled corpus holds is what you need, the names are shown to
buy it there, on the fifth of the unseen names the lexicon covers
(Section 5.4), though a decoding bias, untested against the names, might
buy part of it; if F1 is, stop at mention replacement. Expect no gain to
be shown on a corpus like SOFC-Exp, where half the test names are seen,
the baseline is at 83 F1 and the corpus covers few of the unseen names;
which of the three matters is not separated (Section 5.5).

**What would settle the names question.** Two analyses not run here: a
recall-matched control, a bias toward tagging more spans calibrated on
validation data, against which the names' extra recall would have to
show as precision (the companion study's calibrated bias on the CRF head
alone was not shown to help; Supplementary Note 5); and a training-size
curve on MuLMS, whose six other subfields hold about 8,000 labeled
sentences that the 340-sentence regime leaves unused, since mention
replacement's gains shrink with training size (dai2020analysis;
torres2024augmentation). Two baselines were dismissed on 3 seeds and
deserve 10: self-training on the same corpus and the lexicon as a
feature. And the lexicon's precision, never estimated, could be read from
a sample of 100 names per type judged by a materials scientist; it would
tell whether the names fail to add because they are wrong or because
they are redundant with the training mentions.

## 8 Related work

**Augmentation for NER.** Mention replacement, which swaps annotated
mentions for others of the same type, is one of the simple augmentations
Dai and Adel (dai2020analysis) compared on a materials synthesis corpus
(mysore2019materials) and a clinical one; it helped most with small
training sets, and its gains shrink and reverse as training grows
(torres2024augmentation); most augmentations make the tagger memorise
entities (sharma2023paraphrase), which is why unseen entities are scored
separately here. The replacements otherwise come from the training set
(bartolini2024cosiner), a masked language model (zhou2022melm), a
generative model (hu2023entity), related sentences (cai2023graph) or
lists of the target domain's names (amalvy2022mention); DAGA
(ding2020daga) and LADA (chen2020lada) generate or interpolate labeled
examples, and cross-domain work (chen2021crossdomain; liu2021crossner)
changes domain, not document. We change only where the names come from,
keep every gold context, and test on held-out documents of the same
corpus.

**Lexicons and labels from unlabeled text.** Gazetteers have been
bootstrapped from unlabeled text since Riloff and Jones
(riloff1999learning) and Kozareva (kozareva2006bootstrapping), and
agreement between models as a filter on unlabeled examples goes back to
co-training and tri-training (blum1998combining; zhou2005tritraining).
Zafarian and Asghari (zafarian2019gazetteer) keep a supervised tagger's
confident entities as a gazetteer feature; GEMNET (meng2021gemnet) gates
gazetteer features against the context; Song et al. (song2020gazetteers)
find no significant gain from replacing annotated entities with
gazetteer entries, on a corpus whose test names are mostly seen. AutoNER
(shang2018autoner), BOND (liang2020bond) and weak supervision with
labelling functions (lison2020named) train taggers from dictionaries,
heuristics and unlabeled text; students learn from several teachers on
unlabeled target-language text (wu2020teacher), and teachers' agreement
selects labels in distantly supervised NER (qu2023adaptive). Our
self-training variant follows BOND on the topic corpus, where Du et al.
(du2021selftraining) retrieve sentences from the web. What this paper
adds is the test: the lexicon's contribution isolated against a matched
control, its source and settings varied, on held-out documents, under
protocols fixed in advance, with the finding that the combination's gain
is not shown to exceed mention replacement's.

**Materials text mining and unseen documents.** Text mining has become a
source of materials data (olivetti2020datadriven;
kononova2021opportunities; tshitoyan2019unsupervised). MatSciBERT
(gupta2022matscibert) is pre-trained on materials papers;
ChemDataExtractor (swain2016chemdataextractor) combines dictionaries,
rules and a CRF tagger; Weston et al. (weston2019named) applied a
materials NER model at scale; Trewartha et al. (trewartha2022quantifying)
and MatSci-NLP (song2023matscinlp) benchmark materials language models;
and large language models now extract structured materials information
by fine-tuning (dagdelen2024structured) or prompting
(polak2024extracting). Potu et al. (potu2025ontology) report 66.74 F1 on
MaterioMiner against 78.56 in distribution, a gap of about 12 points
where our three-paper models lose 28–31 (companion study). Random
sentence splits leak names between training and test (babl2025random;
gorman2019standard and sogaard2021random take opposite sides); taggers
score far lower on unseen mentions (augenstein2017generalisation;
fu2020rethinking; kim2022biomedical), lean on name regularity and the
names themselves (lin2020rigorous; ghaddar2021context), narrow but do
not close that gap with contextual embeddings (taille2020contextualized),
and are exposed when a test set's entities are switched
(agarwal2020entityswitched). Seed variance and significance on small
test sets are treated by Reimers and Gurevych (reimers2017reporting), Dror
et al. (dror2018hitchhiker) and Bouthillier et al.
(bouthillier2021accounting), and Card et al. (card2020power) show how
underpowered typical NLP comparisons are, which is why Supplementary Note
9 reports what each comparison here could detect.

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
detected within about 2 to 3 points, and on MuLMS what the names are shown
to buy is recall on the unseen names the lexicon holds, a fifth of them, while
precision falls in pooled rates. We report the ten decisions, the
criterion revision, the amendments and the two readings of D5's
protocol, and the analyses that would settle the question. The practical advice is
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
- **A revised criterion.** The criterion under which MR6 + names passes
  was adopted on 20 October on the author's decision to push for a
  methods claim, after part of the screen and before the names system's
  own screen result, and when mention replacement was already known to
  meet it (Section 3); under the earlier criterion there is no claim, and
  D3 and D4 would not have run. The screen that chose MR6 + names and the
  MuLMS decisions share the same seven subfields: disjoint seeds remove
  the selection's seed noise, not its fit to these units.
- **MuLMS's training pool is two or three articles.** The 400-sentence
  pool is drawn from a shuffled order of the other subfields' articles
  until 400 sentences are reached, the last article cut, so each MuLMS
  model is trained on two or three articles, a different set per held-out
  subfield with some articles recurring across sets; the 88% share of
  unseen names and the 24-point spread between subfields reflect that.
- **One training regime.** Every positive result is at about 340 training
  sentences; MuLMS's full pool was never used, SOFC-Exp's larger training
  set is where no gain is shown, and the literature predicts smaller
  gains for mention replacement as training grows. The claim is scoped to
  a few hundred annotated sentences.
- **The names' effect is only partly isolated.** The lexicon-membership
  split (Section 5.4) was run after the decisions, on rebuilt lexicons;
  it reads name knowledge on MuLMS by intervals (mixed on point
  estimates) and determines no reading on MaterioMiner; no recall-matched
  control was run; the lexicon's precision was not estimated (the
  precision of the predictions of lexicon names, 69 to 75% on MuLMS and 81
  to 83% on MaterioMiner, is the nearest figure).
- **Untuned settings.** MR3's three copies and swap rate 0.5, MR6's six
  copies, the 0.9 threshold and the three-teacher agreement were fixed in
  advance, not tuned; Dai and Adel tune the copies (1 to 10) and the rate
  (0.1 to 0.7) on development data, and a tuned mention replacement could
  close or widen every gap in Table 3. The follow-up's two-head filter
  admits fewer names than the single-task one, and a looser one was not
  tried.
- **Null results are not equivalences.** The source decision and the
  ablations detect differences of about 2 to 3 points (Supplementary Note
  9); smaller effects of the source or the settings are not excluded.
- **Baselines.** The gazetteer-feature and self-training variants were
  rejected on 3 seeds, and one encoder, MatSciBERT, was used. No large
  language model was compared, as an extractor, a teacher or a generator
  of same-type names (zhou2022melm; hu2023entity): the question here is
  whether names already present in unlabeled text add to mention
  replacement, and a generated pool is a different source that needs its
  own matched test; an open-weight model writing names for a held-out
  subfield, fed through the same pipeline, is the natural next
  comparison.
- **Corpora and pre-registration.** The corpora come from Europe PMC's
  open-access subset, found by title terms, with journal concentration
  and licence mixes given in Supplementary Note 3, and are rebuilt from
  hashed reports rather than redistributed. Pre-registration rests on the
  repository's dated commits, whose times the author sets, not on an
  independent registry; GitHub's records of the pushes are the nearest
  independent timestamps.

## Ethics and broader impact

The work uses published corpora under their licences and open-access
full texts for the unlabeled corpora, which are not redistributed. No
personal data is involved. An AI coding assistant was used, under the
author's direction, to write code and drafts and to run adversarial
internal reviews of the drafts; the author is responsible for the
content. This paper shares data, baselines and protocol with a
companion evaluation study of the same programme (in preparation; its
draft is in the repository), which reports the random-split gap, seed
variance and the other additions tested. The two papers overlap in
Sections 2 to 4. The results that appear in both are the random-split
gap and the unseen-name shares of Section 1, mention replacement's
MaterioMiner decisions and the multi-task results of Section 4, and the
label-propagation rule of Supplementary Note 7, which the companion reports in its
survey; the companion summarises D1–D8 and the decomposition of Section
5.2 in one table and cites this paper for them.

## Code availability

The repository, github.com/hatemadi15/AI_Research_Ontologies_GNN_GRL
(Apache 2.0; the commit this paper describes will be tagged, and an
archived copy with a DOI deposited, at submission), holds the code, every protocol with its dates and
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
rebuilds each from its corpus (the lexicon-split analysis of 31
October), and they are not
released, because they derive from corpora that are not redistributed.

## Author contributions, competing interests and funding

The author designed the study, wrote the code, ran the experiments and
wrote the paper. The author declares no competing interests. No external
funding was received; the GPU compute was Kaggle's free tier. The author
thanks the creators of MaterioMiner, SOFC-Exp and MuLMS for releasing
their corpora under open licences.
