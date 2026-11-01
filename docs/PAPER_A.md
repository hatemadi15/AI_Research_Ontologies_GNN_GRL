# Names mined from unlabeled papers are not shown to improve data augmentation for materials named-entity recognition: a pre-registered test

Hatem Madi (affiliation to be added; hatem.madi15@gmail.com)

## Abstract

Materials named-entity recognizers train on a few annotated papers but
must read new ones: our baseline scores 77 F1 on random splits of
MaterioMiner but 46 on a held-out paper, where 81% of the entities are
names no training sentence annotates. To test whether names mined from
unlabeled text help, three baseline models tag 40,000 unlabeled
sentences; the names they confidently agree on join the pool of mention
replacement, which swaps names in six copies of each training sentence.
Over 11 held-out subfields and papers of two corpora this gains 3.58 F1
points [95% CI 0.95, 6.21] over the baseline, under a criterion adopted
mid-study that mention replacement already met. Without the names,
three-copy mention replacement gains 2.35 [0.94, 3.75] on the same units
(post hoc); against a six-copy control the names add 0.80 [−0.66, 2.26]
on the decision seeds, and under a point on MaterioMiner's strongest
system, none passing. Against the six-copy control decoded with a bias
matched to its recall, the model with the names finds more of the unseen
names its lexicon holds (a follow-up decision that passes, but not
without the three of seven subfields no bias could match) but is not
shown to gain F1. On a third corpus,
with half the test names seen, nothing is shown to beat the baseline.
Only mention replacement's part of the gain has an interval above 0, and
that post hoc; the names' part does not.

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
corpus; the names all three tag alike, each tag with a probability (its
*marginal*) of 0.9 or more, form a *lexicon*; and the lexicon enters
training through mention replacement (dai2020analysis), which copies each
training sentence six times with its entity mentions swapped for other
mentions of the same type, drawn from the training mentions and the
lexicon together (*MR6 + names*, Section 4). The model trained on the
augmented data, the *student*, never trains on a teacher-labeled
sentence: it sees only annotated sentences, some of whose names are
swapped for harvested ones. Box 1 collects these and the other terms.

Every decision was fixed in a protocol committed to the repository
before its results (vanmiltenburg2021preregistering), with 7 to 10
training runs (*seeds*) per held-out *unit* (a MuLMS subfield, a
MaterioMiner paper or a SOFC-Exp fold of articles) and a decision rule
stated in advance: a system improves on its reference when the 95%
interval of the mean paired difference in micro F1 over the units lies
above 0. A number in square brackets is that 95% Student-t confidence
interval, over the units named; a comparison specified after the results
of the runs it analyses were known is labelled post hoc. Protocols are
cited by date, with the time where one day has two, and Supplementary
Note 1 lists each with its commit. The *names route* is the sequence of
protocols that led to and tested harvested names; five of them (18
October 11:00 to 1 November) took the twelve *decisions* on MR6 + names,
D1 to D12 (Table 2), beside others
counted in Section 6. The results, with the qualifications Section 3
records:

- MR6 + names improves on the CRF head pooled over the 11 units of MuLMS
  and MaterioMiner (D5; Section 5.1), under a pooled criterion adopted
  mid-study, when plain mention replacement was already known to meet it
  (Section 3). MuLMS alone does not pass (D1).
- That gain is not shown to be the names'. Mention replacement with three
  copies and no names (MR3) gains about two thirds of it on the same
  units (post hoc), and the pre-registered test of the names against
  six-copy mention replacement (MR6; D7) excludes an increment above 2.3
  points but not a smaller one (Section 5.2).
- Neither the names' source nor the lexicon's settings changes F1
  detectably (Section 5.3). On MuLMS the names raise recall on unseen
  names, more on the fifth of them the lexicon holds (post hoc). Against
  MR6 decoded with a bias chosen to match their overall recall, they find
  more of the names the lexicon holds (D11 passes as registered, though
  in three of seven subfields no bias reaches their recall and without
  those the interval covers 0) but are not shown to gain F1 (D12; Section
  5.4). On SOFC-Exp, where half the test names are
  seen, no system is shown to improve on the baseline (Section 5.5).
- Two follow-ups were registered after D1 to D8 had been written up: the
  recall-matched control (D11, D12) and a test of the names on
  MaterioMiner's strongest system, multi-task training (one encoder with
  a CRF head for each of MaterioMiner's two type levels) with mention
  replacement, where against a six-copy matched control neither of its
  two decisions passes (D9, D10; Section 5.6).

Four of the twelve decisions pass: three on F1, which share
MaterioMiner's four papers (D3 and D4 at its two granularities, and D5,
whose 11 units include them), and D11, on recall; none would survive a
Bonferroni correction (Section 5.1). Section 6 counts what else was
tried on the way.

**Box 1. Terms used throughout.**

- *Unit.* A held-out MuLMS subfield, MaterioMiner paper or SOFC-Exp fold
  of articles, on which a system trained without it is scored.
- *Seed.* The random seed of a training run: 10 per unit (offsets 0 to
  900), or the 7 *decision seeds* (300 to 900) where a protocol used the
  other 3 for a *screen*, a 3-seed comparison that chooses among variants
  and decides nothing.
- *CRF head.* The baseline, MatSciBERT with a conditional random field
  output layer (Section 4); *multi-task training* gives one encoder a CRF
  head for each of MaterioMiner's two type levels.
- *MR3, MR6.* Mention replacement with three or six copies of each
  training sentence, each mention swapped with probability 0.5 for one of
  the same type drawn from the *pool*: the training mentions, and for
  *MR6 + names* the lexicon too.
- *Teachers, marginal, lexicon, student.* Three CRF heads trained on the
  training sentences paired with a held-out unit (Table 1), never on the
  unit itself (seeds 0, 100 and 200), tag an unlabeled corpus; a tag's marginal is the probability a teacher gives it at its
  word; the lexicon is the mentions all three tag with the same span and
  type, every marginal at least 0.9; the student is the model trained on
  the gold sentences and their copies.
- *Decision, pre-specified, post hoc, follow-up.* A decision is a
  comparison whose verdict rule a protocol fixed before its results; a
  pre-specified comparison was fixed in a protocol without one; a post
  hoc one was specified after the results of the runs it analyses were
  known; the two *follow-ups* are the protocols of 31 October (19:15; D9,
  D10) and 1 November (10:51; D11, D12), registered after D1 to D8 had
  been written up. The *names route* is the protocols that led to and
  tested harvested names (17 October to 1 November).
- *Paired and pooled.* A paired difference is the mean over units of each
  unit's difference, each side the mean of its seeds (for a recall or
  precision, its rate over the seeds), with a 95% Student-t interval over
  the units. A pooled rate is a recall or precision from counts summed
  over seeds and units, without an interval; a pooled decision or estimate
  is one paired difference over the units of two or more corpora, with its
  interval.
- *Vote, sign-flip test.* A vote is the entity-level majority of the
  three models of a seed triple; the sign-flip test is the exact
  two-sided permutation test of a mean paired difference over all 2^n
  assignments of signs to the n units.
- *Decoding bias, recall-matched control.* A constant added at decoding
  to the score of every tag but O (the tag of a word outside any
  entity), so that a trained model tags more spans (or, below 0, fewer);
  the recall-matched control is MR6 decoded with the bias at which its
  recall over all test entities equals that of MR6 + names or, where no
  bias on the grid reaches it, at the grid's end (Section 5.4).
- *MaterioMiner analogue, gazetteer variant, label propagation.* A MuLMS
  system run on MaterioMiner's papers with a corpus on fatigue; the
  lexicon as a match feature of the CRF head instead of in the pool; a
  rule on saved predictions that tags a term in the rest of a document
  when the model tags it in most of its occurrences there (Supplementary
  Note 7).
- *Seen and unseen names* are defined in Section 2.

## 2 Data

Table 1 gives the three corpora; the short names in parentheses are the
ones the figures and tables use. All three corpora are released under
open licences; our conversions and the held-out protocol are in the
repository.

**Table 1.** The three corpora: documents and sentences, annotation, the
held-out unit with its test size, and the training set per unit.

| Corpus | Documents, sentences | Annotation | Held-out unit (test size) | Training per unit |
|---|---|---|---|---|
| MaterioMiner (durmaz2024materiominer), release v1.0.1, CC BY 4.0 | 4 papers on fatigue of metals (s11340, ncomms7164, s41467, met10111458: 41, 81, 201, 153 sentences) | the classes of the dataset's materials mechanics ontology (MMO) at two levels: fine-grained, its 27 classes with most entities, the other classes' entities relabelled O (1,328 entities; e.g. Alloy, Value, Mechanism, StateDescriptor, FatigueTestSpecimen, GrainBoundary); coarse-grained, 27 parent types (2,194 entities) | one paper (41–201 sentences) | the other three papers (275–435 sentences, of which 15% validate: 234–370 training sentences) |
| MuLMS (schrader2023mulms), Hugging Face `timo-pierre-schrader/MuLMS` at revision bc1c97d, CC BY-SA 4.0 | 50 articles from 7 materials subfields, 10,184 sentences: water electrolysis (electrolysis; 4 articles, 992 sentences), graphene (2, 351), polymer-electrolyte-membrane fuel cells (pemfc; 8, 1,638), polymers (poly; 11, 2,384), semiconductors (semi; 11, 2,040), solid oxide fuel cells (sofc; 4, 928), steel (10, 1,851) | 13 types (MAT, VALUE, PROPERTY, FORM, MEASUREMENT, CITE, SAMPLE, TECHNIQUE, DEV, INSTRUMENT, NUM, UNIT, RANGE); we keep the outermost of its nested entities, 31,952 of 46,584 | one subfield, every sentence of its articles (351–2,384) | 400 sentences of the other subfields' articles, taken in a shuffled order of those articles until 400 are reached (two or three articles, the last cut; a different set for each held-out subfield); 340 train and 60 validate, MaterioMiner's size by design |
| SOFC-Exp (friedrich2020sofc), `boschresearch/sofc-exp_textmining_resources`, CC BY 4.0 | 45 articles on solid oxide fuel cells; 876 of their 9,466 sentences carry entity annotation (the ones that describe experiments) | 4 types: MATERIAL, VALUE, DEVICE, EXPERIMENT | 9 articles, one of 5 folds (137–248 sentences) | the other 36 articles (534–628 training and 94–111 validation sentences) |

Supplementary Note 3 gives MaterioMiner's DOIs and its tie-break among
classes, and MuLMS's kept entities by type.

Three details matter below. On MuLMS a flat tagger needs one level, so
31.4% of the entities are dropped with the conversion (14,632 of 46,584:
nested in a kept entity, sharing its span or crossing it), and text
extraction had damaged 276 tokens, which we repair; our MuLMS scores
cannot be compared with the published nested ones. On MuLMS the training
pool is capped at 400 sentences on purpose, so that MuLMS tests
MaterioMiner's regime, a few hundred annotated sentences from two or
three documents (SOFC-Exp, trained on 36 articles, does not; Section
5.5); the other subfields hold about 8,000 labeled sentences that no run
here uses (Limitations). SOFC-Exp's official test articles are tokenized
differently from its training articles; the article folds used here cut
every article with the training articles' tokenizer.

**Unseen names.** A *name* is the word string of an entity of any type,
values such as "10 mA cm−2" included. A test entity is unseen when its
words are annotated as
no entity, of any type, in any training sentence, and seen when some
training sentence annotates them with its type
(augenstein2017generalisation); entities whose words are annotated only
under another type, 0.0–2.1% of a corpus's test entities (up to 4.5% of
one paper's), count as neither. A predicted entity is classed by the same
rule from its words and its predicted type, so a correct prediction has
its gold entity's kind: recall on unseen names is the share of the
unseen gold entities found, and precision the share of the predictions
classed unseen that are correct. A name, gold or predicted, is in a
lexicon when the lexicon holds its words with its type.

## 3 Evaluation protocol

**Seeds and scoring.** Every system compared in a decision has 10 seeds
per held-out unit (offsets 0, 100, …, 900), except where a protocol
reserved the screen's 3 seeds (0 to 200) for selection and decided on
the other 7; the mixed-corpus variant of D8 has those 7 seeds and the
lexicon-setting ablations the screen's 3 (Section 5.3). The score is micro F1 at the entity level: an
entity counts as found only with its exact span and type; precision is
the share of predicted entities that are correct, recall the share of
gold (annotated) entities found, F1 their harmonic mean, micro over all
entities of a test set (seqeval, nakayama2018seqeval, in its default
mode, as the datasets' own code scores), in percentage points.

**Decision rule.** A method improves on its reference when the 95%
Student-t interval of the mean paired micro-F1 difference lies above 0,
the pairs being the held-out units (7 subfields, 4 papers, 5 folds), each
side the mean of its seeds (D11 applies the rule to a recall; Section
5.4); with 4 papers the interval has 3 degrees of freedom, with 7
subfields 6. A pooled decision takes the units of two
corpora together (11 units, 10 degrees of freedom) and requires, in
addition, that neither corpus's own interval lie below 0. Each decision
is one-sided at 2.5%, so of the twelve decisions about 0.3 would pass if
nothing worked (0.5 of the twenty counted in Section 6); the decisions
are not independent (Section 5.1). At 80% power and the observed spreads
the names against MR6
detect 2.0 points pooled over the 11 units and the names against the CRF
head 3.5 to 3.7 (Supplementary Note 9 gives every decision and the main
comparisons).

**Pre-registration, and what changed.** Each protocol was committed to
the repository before its results, with its decisions and the
comparisons pre-specified without a decision, and all but three before
any of the runs they analyse had started (D5's, as executed, that of D7
and D8, and that of D11 and D12; below); amendments are marked with
their reason and time. Supplementary Note 1 gives the commits and the
chronology, with GitHub's own log of the pushes, which puts every
protocol commit on GitHub 1 to 7 seconds after its commit time. The
protocols of the names route were fixed between 17 October and 1
November 2026. Three points must be disclosed.

First, the criterion for a methods claim was revised on 20 October at
09:14, on the author's decision, recorded in the protocol, to push for a
methods paper. The protocol of 18 October 11:00 required a pass against
the CRF head on MuLMS alone plus one on MaterioMiner, and ran the
MaterioMiner analogue (D3, D4) only if D1 passed; the revision added a
pooled decision (D5) as a second route to the claim, lifted that gate
and lifted a stop rule (no fourth protocol on MuLMS). At 09:14 mention
replacement was known, from runs already made, to meet the pooled
criterion (post hoc; Table 3); the names system's pooled result did not
exist, and the screen had reported three of its seven variants, not MR6
+ names. The gate was re-imposed at 13:53, after the screen had chosen
MR6 + names, and lifted again at 18:42 (Supplementary Note 1). Under the
original criterion MR6 + names has no claim, since D1 failed, and D3 and
D4 would not have run.

Second, D5's protocol named MuLMS's decision seeds as the units' source
and, in the same paragraph, "each unit its 10-seed mean". The analysis as
executed (D5) averages in the screen's seeds 0 to 200, on which MR6 +
names was selected and whose runs were under way at 09:14, their results
not yet recorded; the decision-seed reading (D5′), which leaves them out,
agrees (Table 2), and the decomposition of Section 5.2 uses it.

Third, the protocol of D7 and D8 (29 October) was committed two minutes
after its first batch was pushed and five hours before that batch's
first result, but after MR6 + names' own results on the same units (D1,
D3) were known: only MR6 and the mixed-corpus variant were new runs. That
of D11 and D12 (1 November, 10:51), registered at the request of an
internal review of the draft (Ethics and broader impact) after D1 to D10
and the lexicon split of Section 5.4 were known, is of
the same kind: the names' side is the stored runs, and only the control,
MR6 run again (identically) and decoded under the bias, was new. That of
D9 and D10 (31 October, 19:15) followed the write-up of D1 to D8 and
preceded its runs. Every other comparison, here or from the
earlier protocols, is labelled pre-specified or post hoc where it
appears.

**Runs.** Kaggle T4 GPUs with deterministic CUDA, one frozen commit per
batch, a pinned image and pinned libraries; model snapshots pinned by
commit. From 15 October 16:00 UTC every GPU session but two began by
repeating an earlier run, and every repeat was identical; hyperparameters,
compute and the repeats are in Supplementary Note 2.

## 4 Systems

Figure 1 sketches MR6 + names, the last of the four systems compared
throughout:

- **CRF head.** MatSciBERT (gupta2022matscibert) with a token classifier on
  the first sub-word of each word and a linear-chain conditional random
  field (CRF) over the words, constrained to the IOB2 tag grammar (a tag
  that begins an entity, continues it, or is outside one), trained by
  likelihood and decoded with Viterbi, the highest-scoring tag sequence
  (lafferty2001conditional; lample2016neural), with early stopping on
  validation F1 (Supplementary Note 2). It is the reference of D1 and D3
  to D6 and the
  baseline every system is measured from. It is not the strongest single
  model we know on every corpus: on MaterioMiner's held-out papers,
  fine-grained, multi-task training gains +3.72 [+0.57, +6.87] over it
  and multi-task training with three-copy mention replacement +6.73
  [+6.03, +7.43] (pre-specified comparisons without a decision in the
  companion evaluation study; Supplementary Table S1); neither transfers
  to SOFC-Exp, and MuLMS has one type level, so the CRF head is the
  reference the three corpora share. The follow-up of 31 October tested the names on
  that system, with six copies (Section 5.6).
- **MR3.** Mention replacement (dai2020analysis): three copies of each
  training sentence with an entity, each mention replaced with
  probability 0.5 by a training mention of the same type; copies that
  change nothing are dropped. The student trains on the gold sentences and
  the copies with the CRF head's training settings and seed.
- **MR6.** The same with six copies: the matched control for the names.
- **MR6 + names.** MR6 whose replacement pool also holds a lexicon
  harvested from unlabeled text:
  - *Teachers.* The CRF head's own runs of seeds 0, 100 and 200 on the
    unit's training sentences, so the teachers are the baseline, not a
    stronger model.
  - *Corpus.* 40,000 sentences of open-access full texts from Europe PMC
    (europepmc2015europe) found by title terms of the held-out subfield
    (for MaterioMiner, fatigue in the title and a metal or process term in
    the title or abstract; for SOFC-Exp, solid oxide cells),
    with the datasets' own articles excluded by identifier and every
    sentence sharing a word 8-gram with a dataset sentence dropped (for
    MuLMS also a 50-character run); 290–368 articles with text per corpus
    (Supplementary Note 3). A topic corpus holds 55–81% of a held-out
    MuLMS subfield's unseen test names, the fatigue corpus 50–62% of a
    held-out paper's fine-grained (49–65% of its coarse-grained), the
    SOFC corpus 36% of a fold's (measured before the runs).
  - *Lexicon.* Each teacher tags the whole corpus and returns, per word,
    the probability the model assigns to the tag it chose (the
    forward–backward marginal of the Viterbi tag). A mention joins the
    lexicon when the three teachers tag the same span with the same type
    and every tag of the span has a marginal of at least 0.9 under each
    teacher, whatever the rest of the sentence. Per held-out MuLMS
    subfield this admits 8,328–50,869 mentions (4,380–13,395 distinct
    names), which hold 22.2% of the held-out subfield's unseen test names
    (7–36% per subfield), about a third of what the corpus holds. The
    lexicon's precision was not estimated and its names were not stored
    with the runs (Limitations).
  - *Pool.* The training mentions (601–1,309 per MuLMS unit) and the
    harvested mentions, one entry per occurrence in the corpus, so that a
    draw is uniform over entries and a frequent name is drawn more often;
    harvested entries are at least 92% of a unit's pool, so most
    replacements are harvested names.

**An example.** Take a training sentence, constructed for illustration,
"The Pt catalyst lowers the overpotential", with Pt annotated MAT and
overpotential PROPERTY. For the held-out electrolysis subfield, a copy
made by MR6 might read "The Ni catalyst lowers the overpotential", the
replacement drawn from the unit's training mentions, and a copy made by
MR6 + names "The MoS2 catalyst lowers the Tafel slope", both
replacements drawn from the subfield's lexicon. Its most frequent
entries are, as MAT, Ni, Pt, hydrogen, water, electrode, Co and catalyst,
and, as PROPERTY, current density and scan rate (each also in the
plural) and Tafel slope, and, as VALUE, 10 mA cm−2, 30 min, 2 h, 60 °C
and 1 M; values are 70% of its distinct names and 30% of its entries,
materials 23% and 65% (the rebuilt lexicon of Section 5.4; Supplementary
Note 10); a
materials scientist might not count
electrode or catalyst as materials, and the lexicon's precision was not
estimated.

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

Table 2 lists the twelve decisions of the names route in the order they
were taken, with the seeds each used; D9 and D10 belong to the follow-up
of Section 5.6, D11 and D12 to that of Section 5.4.

**Table 2.** The twelve pre-registered decisions of the names route (D9
to D12 the follow-ups) in the order taken: corpus and units (with
MuLMS's seeds), comparison, the reference's and the system's mean micro
F1 (for D11, recall), the mean paired difference with its 95% CI, and the
verdict.

| | Corpus, units (MuLMS seeds) | Comparison | Reference | System | Δ [95% CI] | Verdict |
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
| D11 | MuLMS, 7 subfields (300–900; follow-up) | MR6 + names vs MR6 at the recall-matched bias (3 of 7 subfields at the grid's end): recall on the unseen names the lexicon holds | 71.92 | 83.16 | +11.24 [+3.31, +19.18] | passes |
| D12 | MuLMS, 7 subfields (300–900; follow-up) | MR6 + names vs MR6 at the recall-matched bias (3 of 7 subfields at the grid's end): micro F1 | 48.47 | 51.73 | +3.26 [−0.79, +7.31] | no |

The reference F1 on MuLMS is the 7-seed mean (48.54); the 10-seed mean
quoted in Section 1 is 48.66. D5 and D5′ are the two readings of one
protocol text (Section 3). Every column is micro F1 except D11's. For
D11 and D12 the reference is MR6 decoded at the bias that matches the
names' overall recall; in three subfields no bias on the grid (−2 to 5)
reaches it, and the protocol matched them at its end, D = 5, where MR6's
recall is 1.7 to 2.9 points lower. D11's columns are recall on the
unseen test names the lexicon holds, the mean over the subfields of rates
pooled over the seeds (Section 5.4).

Three F1 decisions pass, and they are not three pieces of evidence: D3
and D4 are the same four papers at two granularities, and D5's eleven
units contain those four papers (the fourth pass, D11, is on recall;
Section 5.4). On MuLMS, the pooled corpus independent of
those papers, MR6 + names fails D1 and D2, while MR6 alone clears 0
against the CRF head on the decision seeds (+2.68 [+0.13, +5.23]; on all
ten +2.33 [−0.27, +4.92], covering 0; pre-specified on 29 October), with
a smaller
mean than MR6 + names' +3.19 but a narrower spread; with MR6 + names 6 of
the 7 subfields gain (sofc by 11.9, poly loses 2.4), and the differences
have a standard deviation of 4.7 points across subfields. On
MaterioMiner every paper gains at both granularities, but with four
units the sign-flip test cannot go below p = 0.125, so D3 and D4 rest on
the t interval with 3 degrees of freedom. The pooled estimate survives
leaving out any one unit (post hoc: the lowest lower bounds are +0.44
and +0.54, without semiconductors), and the other post hoc checks agree:
the mean of the two corpus means is 3.73; 9 of the 11 units are positive
on the executed 10-seed basis and 10 on the 7-seed basis; and the exact
sign-flip test gives p = 0.011 and 0.007. Figure 2 shows every unit.

The decision rule carries no correction for the number of decisions.
Under a Bonferroni bound of 0.05/12 none of the four passes would hold:
the two-sided t p-values from the per-unit differences are 0.013 for D5
(0.011 for D5′), 0.018 for D3, 0.014 for D4 and 0.013 for D11 (0.052 for
D9, 0.183 for D10 and 0.096 for D12). Mention replacement's post hoc
pooled gain over the CRF head (Section 5.2) has p = 0.004 on either seed
basis, below 0.05/12, but as a post hoc comparison it belongs to no
corrected family.

**Figure 2.** Per held-out unit, the micro-F1 difference of MR6 + names
from the CRF head (circles) and from MR6 (squares): MuLMS's 7 subfields
on the decision seeds, MaterioMiner's 4 papers at both granularities and
SOFC-Exp's 5 folds, where MR6 was not run. The bands, drawn across the
11 units' rows, are the pooled intervals of D5′ and D7.

### 5.2 Whose gain is it

Table 3 gives every comparison among the four systems on the same 11
units, on both seed bases, and on MaterioMiner per granularity, with the
status of each.

**Table 3.** Every comparison among the four systems: over the 11
held-out units on both MuLMS seed bases (top) and over MaterioMiner's
four papers per granularity (bottom); the mean paired difference of micro
F1 with its 95% CI, for the 11 units the units above 0 and the exact
two-sided sign-flip p-value, and the status of each entry.

| Comparison, 11 units | MuLMS seeds 300–900: Δ micro F1 [95% CI] | units above 0 | sign-flip p | MuLMS seeds 0–900: Δ micro F1 [95% CI] | units above 0 | sign-flip p | status |
|---|---|---|---|---|---|---|---|
| MR3 − CRF head | +2.46 [+1.00, +3.91] | 11/11 | 0.001 | +2.35 [+0.94, +3.75] | 10/11 | 0.003 | post hoc |
| MR6 − MR3 | +0.33 [−0.25, +0.92] | 8/11 | 0.23 | +0.22 [−0.30, +0.74] | 8/11 | 0.37 | post hoc (per corpus pre-specified, 29 October) |
| MR6 − CRF head | +2.79 [+1.20, +4.38] | 10/11 | 0.002 | +2.56 [+0.94, +4.18] | 10/11 | 0.005 | post hoc (per corpus pre-specified, 29 October) |
| MR6 + names − MR6 | +0.80 [−0.66, +2.26] | 7/11 | 0.24 | +1.02 [−0.56, +2.59] | 6/11 | 0.18 | D7 on seeds 300–900; post hoc on 0–900 |
| MR6 + names − MR3 | +1.13 [−0.51, +2.76] | 7/11 | 0.16 | +1.23 [−0.22, +2.68] | 8/11 | 0.09 | post hoc (on MuLMS alone, D2) |
| MR6 + names − CRF head | +3.58 [+1.04, +6.13] | 10/11 | 0.007 | +3.58 [+0.95, +6.21] | 9/11 | 0.011 | D5 on seeds 0–900; D5′ a post hoc reading |

| MaterioMiner, 4 held-out papers (10 seeds) | fine-grained [95% CI] | coarse-grained [95% CI] | status |
|---|---|---|---|
| MR3 − CRF head | +2.77 [+1.51, +4.03] | +1.28 [+0.06, +2.51] | companion decisions (protocol of 16 October, 14:39), both passing |
| MR6 − MR3 | +0.20 [−1.47, +1.88] | −0.21 [−2.71, +2.29] | pre-specified, 29 October |
| MR6 − CRF head | +2.98 [+0.09, +5.86] | +1.07 [−1.82, +3.97] | pre-specified, 29 October |
| MR6 + names − MR6 | +1.29 [−2.51, +5.10] | +0.70 [−1.60, +3.00] | pre-specified, 29 October (fine-grained: D7's per-corpus condition) |
| MR6 + names − MR3 | +1.50 [−1.76, +4.75] | +0.49 [+0.06, +0.92] | pre-specified, 20 October (00:51) |
| MR6 + names − CRF head | +4.27 [+1.39, +7.15] | +1.77 [+0.67, +2.88] | D3; D4 |

The +3.58 decomposes, on the decision seeds, into +2.46 from mention
replacement with three copies, +0.33 from three more copies and +0.80
from the harvested names; only the first term has an interval above 0,
and that post hoc (mention replacement's own pre-registered single-model
decisions passed on MaterioMiner and failed on MuLMS alone, where its
votes passed; Table 3 and Supplementary Note 7). The 10-seed basis gives
the same picture
(+2.35, +0.22, +1.02). Against MR3 the whole increment of MR6 + names,
+1.13, has an interval that covers 0: the pre-registered pass over the
CRF head is a property that mention replacement already has on these
units (post hoc). Three intervals in this paper comparing a names variant with MR3
lie above 0, none of them a decision: MaterioMiner coarse-grained (Table
3, 3 degrees of freedom), the 2-of-3-teachers ablation on 3 seeds
(Supplementary Note 4) and SOFC-Exp (Section 5.5). Per entity type the
pattern holds for most types (Supplementary Note 6).

D7's protocol foresaw a gain of +1.5 to +1.65 from the seeds in hand, an
11-unit half-width of about 2 and a chance of a pass of about 0.4; "not
established" is thus the likely outcome of a small effect under this
design, not evidence of none. The interval excludes a gain above +2.26,
nearly as large as mention replacement's whole gain (+2.46). The
follow-up on MaterioMiner's strongest system gives a similar increment
(Section 5.6).

### 5.3 The names' source and the lexicon's settings

**Source (D8).** For each held-out subfield a mixed corpus of 40,000
sentences, drawn equally from the six other subfields' corpora and built
the same way, holds 49–70% of the subfield's unseen test names against
the topic corpus's 55–81%. MR6 + names with the topic's own corpus scores
51.73 against 51.91 with the mixed one, −0.18 [−2.01, +1.65] paired by
subfield, in a comparison that detects 2.5 points; since the mixed corpus
holds most of the same names, D8 tests the names only the topic corpus
holds, not an off-topic source (a framing added after the result).
Against the CRF head the mixed-corpus variant gains +3.38 [−0.89, +7.64],
against MR3 +1.10 [−0.76, +2.97] and against MR6 +0.70 [−1.19, +2.58]
(pre-specified); its own lexicon holds 18.3% of the unseen test names.

**Settings (pre-specified, 3 seeds).** Five ablations, of corpus size
(10,000 and 20,000 sentences), threshold (0.7), agreement (2 of 3
teachers) and the names' share of the pool, keep micro F1 within 0.16 of
MR6 + names on the same seeds, every interval against it within −2.45
and +2.14, while the lexicon ranges from 19 to 79 thousand distinct
names, summed over the seven subfields, and from 16% to 32% of the
unseen test names (Supplementary Note 4). Effects beyond those bounds are
excluded; smaller ones are not.

### 5.4 What the names do

On MuLMS, pooled over the held-out subfields (post hoc, as is this
section up to the recall-matched control; Table 4; Figure 3 puts each system's recall and precision
on the unseen names on one plane with curves of equal F1), the CRF head
finds 39.4% of the unseen names at 55.8% precision, MR3 43.1% at 54.3,
MR6 43.7% at 53.9 and MR6 + names
47.3% at 50.9: against MR6, the copy-matched control, the names raise recall on
unseen names by 3.6 points and lower precision by 3.0, a net +0.7 F1 on
them, the point moving almost along a curve of equal F1. Paired by
subfield on the decision seeds, the recall gain is +2.56 [+0.76, +4.36],
the precision loss −2.22 [−5.03, +0.60] and the net
+0.56 [−1.44, +2.56]: only the recall gain's interval lies above 0. On MaterioMiner's papers the pooled rates of both
recall and precision on unseen names rise over MR6, within paired
intervals that cover 0 (Supplementary Note 5).

**Table 4.** Recall, precision and F1 on the unseen test names, pooled
over seeds and MuLMS's seven held-out subfields (post hoc).

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
(49 runs) and has no arrow, since MR6's point uses all ten seeds; on the
same seeds the two compare as in Table 4's note.

**Which names the gain falls on (post hoc; the analysis and its reading
fixed before its run).** The lexicons were not stored with the runs; an
analysis of 31 October rebuilt each from its unit's own teachers and
corpus, reproducing the stored teachers' scores and coverage counts in
every unit, and split the unseen test names by whether the lexicon holds
them (Table 5; Supplementary Note 5 gives the reading rule and the full
analysis of 31 October (16:13)). On MuLMS the lexicon holds 22% of the
unseen test names,
pooled over the subfields (7% to 36% by subfield). On those, MR6 + names
recalls 82.4% against MR6's 75.0%; on the others, 36.8% against 35.0%;
the gain on the held names is the larger in every subfield, and the
difference of the two gains has an interval above 0 (Table 5). The
reading rule fixed in advance (Supplementary Note 5) calls this *name
knowledge* if the other names' gain is read on its interval, which
covers 0, and *mixed* if on its point estimate (+1.12); MaterioMiner's
result is what the rule calls a *generic shift*, alike gains on both
parts, on point estimates, and none of its outcomes on intervals. The copies alone show no
detectable split (MR6 over the CRF head). Pooled precision falls on both
parts (74.7% to 69.0% on the held names, 46.4% to 44.0% on the others),
and two thirds of the wrong predictions the names add are names the
lexicon does not hold. Because the teachers are the baseline, the
lexicon holds the unseen names the baseline already finds most easily:
the CRF head recalls 71.8% of them and 29.7% of the others (Table S3).

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

**Against a recall-matched control (follow-up; D11, D12).** A recall
gain paid for in precision might be no more than a shift toward tagging
more spans, which a constant bias at decoding would also give. The
follow-up of 1 November tested this on MuLMS at the decision seeds
(Supplementary Note 10). MR6 was run again and decoded with a bias D
added to the score of every tag but O, over a grid from −2 to 5 (at D =
0 all 49 runs equal the stored ones in every score, in the training
history and in the predictions); per subfield the bias D* was found at
which MR6's recall over all test entities, pooled over the seeds, equals
that of MR6 + names, every quantity of the control at D* interpolated
linearly between the two grid values around it. At that bias the names
find more of the unseen test names their lexicon holds, +11.24 [+3.31,
+19.18], in every subfield (D11, passes; exact sign-flip p = 0.016, post
hoc), and fewer of the other unseen names, −1.73 [−3.02, −0.44]
(pre-specified); in micro F1 they are not shown to beat the control,
+3.26 [−0.79, +7.31] (D12), and in precision +5.78 [−1.22, +12.79]
(Table 6, Figure 4). In three subfields, electrolysis, pemfc and sofc,
no bias on the grid reaches the names' recall: MR6's recall peaks at D =
2.5 or 3, 0.2 to 0.5 points short, and falls beyond, where more spans
are tagged but fewer match exactly (Table S8). The protocol matched
these at the grid's end, D = 5, where MR6's recall is 1.7 to 2.9 points
below the names', and its F1 5.3 to 7.0 points and its recall on the
held names 6.9 to 11.5 points below its own at D = 0, which favours the
names in both decisions. It also required the decisions without them:
D11 +6.05 [−3.33, +15.44], all four subfields positive with 3 degrees of
freedom (an interval that could detect only about 12 points), and D12
−0.03 [−2.78, +2.72].

So +11.24 is the registered statistic, not the size of the effect. Per
subfield it is the names' gain over MR6 at D = 0 on the held names,
+7.29 on average (Table 5), plus MR6's own loss on them from D = 0 to
D*, +3.96 [−1.06, +8.98], nearly all of it in the three flagged
subfields (+0.14 on the other four; post hoc). Two post hoc readings
that do not depend on the protocol's matching of those three agree in
direction: with them matched at the grid value nearest the names'
recall, D11 is +8.63 [+3.35, +13.90] and D12 +1.37 [−0.66, +3.41]; and
against MR6 at whichever bias gives it its highest recall on the held
names, the names lead by +6.52 [+2.15, +10.89], in every subfield, while
on the other unseen names a bias overtakes them, −4.16 [−5.82, −2.50]
(Figure 4). No single decoding bias gives MR6 the names' recall on the
names the lexicon holds. A lexicon is mostly materials names and values
(83 to 99% of its distinct names; Supplementary Note 10), so a shift
toward those types could also favour the held names; within materials
names, though, the names gain on the held ones and not on the others
(+12.02 [+3.70, +20.34] against −1.19 [−5.59, +3.21] at D*; post hoc).
The gain is thus consistent with the student having learned names that
its training copies contain, chiefly materials names. At matched recall
it recalls fewer of the other unseen names, and it is not shown to gain
F1 over the bias.

**Table 6.** The names against the recall-matched control on MuLMS's
held-out subfields (follow-up; seeds 300–900): the names' recall over
all test entities pooled over the seeds, MR6's at bias 0, the matched
bias D*; recall on the unseen test names the lexicon holds for the
names, MR6 at D = 0 and the control at D*; and the names minus the
control at D* in that recall (D11), micro F1 (D12) and precision
(pre-specified), with the means over the subfields and their 95% CIs. In
a flagged subfield the names' recall exceeds MR6's at every bias of the
grid; it is matched at the grid's end, as the protocol fixed, and in
parentheses at the grid value of the nearest recall (post hoc). The last
two rows (post hoc) compare the names with MR6 at D = 0 and with MR6 at
whichever bias gives it its highest recall on the held names.

| Subfield | names' recall | MR6's recall at D = 0 | D* | recall on the held names: names / MR6 at 0 / control at D* | Δ recall, held names | Δ micro F1 | Δ precision |
|---|---|---|---|---|---|---|---|
| electrolysis | 60.13 | 58.17 | 5 (flagged; 2.5) | 90.02 / 83.62 / 76.73 | +13.29 (+8.76) | +7.13 (+2.30) | +10.28 (+3.75) |
| graphene | 47.45 | 48.79 | −1.08 | 84.91 / 84.43 / 84.68 | +0.22 | −1.31 | −3.05 |
| pemfc | 51.74 | 49.23 | 5 (flagged; 3) | 84.92 / 78.09 / 69.33 | +15.59 (+11.18) | +6.43 (+3.28) | +11.03 (+6.50) |
| poly | 53.52 | 50.61 | 1.35 | 71.22 / 66.58 / 69.29 | +1.93 | −1.48 | −2.55 |
| semi | 31.78 | 26.82 | 2.47 | 81.36 / 73.58 / 68.96 | +12.40 | +2.20 | +6.90 |
| sofc | 50.36 | 46.55 | 5 (flagged; 2.5) | 85.32 / 71.23 / 59.72 | +25.61 (+16.23) | +9.39 (+4.15) | +16.93 (+8.82) |
| steel | 56.37 | 53.83 | 1.92 | 84.39 / 73.61 / 74.72 | +9.67 | +0.47 | +0.93 |
| mean, 7 subfields | | | | | +11.24 [+3.31, +19.18] (D11, passes) | +3.26 [−0.79, +7.31] (D12, no) | +5.78 [−1.22, +12.79] |
| mean without the flagged, 4 subfields (pre-specified) | | | | | +6.05 [−3.33, +15.44] | −0.03 [−2.78, +2.72] | +0.56 [−6.74, +7.85] |
| mean, the flagged at the nearest recall, 7 subfields (post hoc) | | | | | +8.63 [+3.35, +13.90] | +1.37 [−0.66, +3.41] | +3.04 [−1.32, +7.41] |
| names − MR6 at D = 0, 7 subfields (post hoc) | | | | | +7.29 [+3.27, +11.30] | +0.51 [−1.51, +2.53] | |
| names − MR6 at its best bias for the held names, 7 subfields (post hoc) | | | | | +6.52 [+2.15, +10.89] | | |

**Figure 4.** The recall-matched control per MuLMS subfield (decision
seeds): MR6's recall at every bias D of the grid on all test entities
(blue; the matching, pre-specified) and on the unseen test names the
lexicon holds (red; post hoc), pooled over the seeds, with MR6 + names'
values as dashed lines of the same colours and the matched bias D*
dotted. In the three flagged subfields the blue curve peaks below its
dashed line; in every subfield the red curve stays below its dashed line.

### 5.5 SOFC-Exp: no gain shown

On SOFC-Exp's five folds of held-out articles MR6 + names with the SOFC
corpus scores 82.75 against the CRF head's 82.79, −0.04 [−1.21, +1.14]
(D6). It gains +0.45 [+0.04, +0.86] over MR3 on every fold
(pre-specified), but MR3 itself is not shown to gain there (−0.48 [−1.36,
+0.39], a decision of 20 October (09:14); −0.29 [−1.04, +0.47] on the
official split, paired by seed, an earlier decision, also failed). Two things differ on this
corpus, in one place: only 46% of a fold's test names are unseen (81%
fine- and 79% coarse-grained on MaterioMiner's papers, 88% on MuLMS's
subfields) and the CRF head already finds 80% of them; and the baseline
is at 82.8 F1 with a random-split gap of 3.5 points, so there is little
headroom. The SOFC corpus holds fewer of the unseen test names (36%,
27–43% per fold, measured before the runs) than MuLMS's topic corpora
(55–81%), but its lexicon holds 20–33% per fold, no less than MuLMS's or
MaterioMiner's, and against MR3 the names' interval lies above 0 there
(pre-specified, one of many comparisons without a decision): it is
mention replacement that is not shown to gain. Whether training size
explains that is open: at MaterioMiner's size MR3 gained +0.50 [−0.47,
+1.46], also not shown to differ from 0 (an earlier decision;
Supplementary Note 7). Pooled over the 16 units of the three corpora the
estimate of MR6 + names is +2.45 [+0.50, +4.40] and that of MR3 +1.46
[+0.28, +2.64], and over the 9 units of MaterioMiner and SOFC-Exp that
of multi-task training with MR3 +2.93 [+0.14, +5.72] (pre-specified,
MR3's and multi-task training's on 20 October at 09:14, the names' with
D6 at 13:53); the claim stays the two-corpus one, SOFC-Exp a corpus
where no gain is shown.

### 5.6 The names on the strongest system (follow-up)

The names had been tested only on the single-task CRF head. After D1 to
D8 had been written up, a follow-up protocol, committed on 31 October
before any of its runs and expecting a pass to be unlikely
(Supplementary Note 1), tested them on MaterioMiner's strongest system,
multi-task training with mention replacement: the names system against a
matched control, the multi-task model with six-copy mention replacement,
over 10 seeds on the four held-out papers, fine-grained (D9) and
coarse-grained (D10). Its lexicon is harvested as in Section 4 by three
multi-task teachers, with a filter over both heads (Supplementary Note
8). Neither decision passes (Table 2): fine-grained, all four papers are
positive (0.16 to 1.62; two-sided t p = 0.052, and a sign-flip p of
0.125, the smallest four units allow); coarse-grained, three are. Over
multi-task training with three-copy mention replacement the names' whole
increment is +0.76 [−0.15, +1.68] and +0.91 [−0.14, +1.97], and the three
extra copies alone are not shown to add (pre-specified; Supplementary
Table S6): on this system too what the names add is about a point and
not established. At the observed spreads D9 and D10 would detect about
1.3 and 2.1 points (the protocol expected 3.4 to 3.8 and 1.4 to 2.4), but
the fine-grained spread rests on 3 degrees of freedom and the same
comparison on the single-task base had a far wider interval (Table 3),
so the interval, not the 1.3, is the result. One caveat: the two-head
filter's lexicons hold fewer of the unseen test names than the
single-task lexicons of D3 and D4, by the runs' stored counts 7.5%
against 9.6% fine-grained and 12.1% against 18.3% coarse-grained pooled
over the papers, lower in 7 of
the 8 paper–granularity cells (Supplementary Note 8). Whether that
lowered the increment is not known, and a looser filter was not tried.

## 6 What else was tried, and the count

MR6 + names was reached through two earlier protocols and a screen of
seven variants on MuLMS's held-out subfields and their MaterioMiner
analogues, each committed before its runs, with two rules applied to
saved predictions (the lexicon as a dictionary, and label propagation
within a document); Supplementary Note 7 gives them in full, with the
count. In all, the protocols from 17 October (20:21) to 1 November took
twenty decisions on the names or on mention replacement: the twelve of
Table 2, the six of the two earlier protocols (four on MuLMS, two on
MaterioMiner) and two on SOFC-Exp's folds. Five passed: D3, D4, D5, D11
and mention replacement's votes against the CRF head's votes. Of the
seven decisions of Table 2 against a mention-replacement control or on
the names' source, only D11, on recall, passes. If no method worked,
about 0.5 of twenty would be expected to pass, whatever their
dependence, which only makes chance passes come together, as the shared
units of D3 to D5 do. The count leaves out the label-propagation rule, a
post-processing rule on saved predictions rather than a use of names:
with its four decisions of 20 October the window holds twenty-four
decisions, six passing, about 0.6 expected by chance; the companion
study's decisions, all earlier, are not counted.

## 7 Discussion

**For practitioners.** If you fine-tune an encoder tagger on a few
hundred annotated sentences to read new documents (no large language
model was compared), use mention replacement with three copies: its gain
on held-out documents passed pre-registered on MaterioMiner (companion
study) and is above 0 post hoc when the two corpora are pooled, while on
MuLMS alone only its votes passed (Supplementary Note 7). It costs about twice the
baseline's training time and needs no corpus. Neither three more copies
nor the harvested names are shown to add to it, here or on MaterioMiner's
strongest system, and a names student costs three to seven times an MR3
run, by how the harvest is charged, plus a 40,000-sentence corpus
(Supplementary Note 2). On MuLMS the names find more of the unseen names
their lexicon holds than a decoding bias tuned to the same overall
recall does (in three subfields no bias reached it), at the expense of
other unseen names, and are not shown to gain F1 over it (Section 5.4):
they are worth their cost only where those particular names, frequent
names of the corpus of which the baseline already finds 72% (Table S3),
matter more than F1.
Even the best system finds only half of a held-out MuLMS
subfield's entities. If F1 is the goal, stop at
mention replacement, and expect no gain to be shown on a corpus like
SOFC-Exp, where half the test names are seen and the baseline is at 83
F1 (Section 5.5).

**What remains open.** A training-size curve on MuLMS, whose six other
subfields hold about 8,000 labeled sentences that the 340-sentence
regime leaves unused, since mention replacement's gains shrink with
training size (dai2020analysis; torres2024augmentation). Two variants
dismissed on 3 seeds deserve 10: self-training on the corpus's
teacher-labeled sentences, the one variant of the screen whose interval
against the CRF head cleared 0, and the lexicon as a gazetteer feature
(Supplementary Note 7). A bias on the tags that begin an entity only, or
one per entity type, might match the three subfields that one bias on
every tag could not, since past D = 2.5 that bias lengthens and merges
spans (Section 5.4). And the
lexicon's precision, never estimated, could be read from a sample of 100
names per type judged by a materials scientist; it would tell whether
the names fail to add F1 because they are wrong, redundant with the
training mentions, or already within the baseline's reach.

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
control and its recall against a recall-matched decoding bias, its
source and settings varied, on held-out documents, under protocols fixed
in advance, with the finding that the names raise recall on the names
the lexicon holds beyond what a decoding bias gives, while the
combination's gain is not shown to exceed mention replacement's.

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
of a new paper's entity names. Giving it names mined from unlabeled text
by three agreeing teachers, through mention replacement, improves on the
baseline pooled over held-out documents of two corpora (+3.58 [+0.95,
+6.21]), by a criterion adopted mid-study, though not on MuLMS alone;
mention replacement without the names gains most of that on the same
units (post hoc), and the names' own increment, +0.80 [−0.66, +2.26] on
the decision seeds, is not established, nor on MaterioMiner's strongest
system. Against MR6 decoded with a bias chosen to match its overall
recall (reached in four of seven subfields), the model finds more of the
names its lexicon holds, consistent with its having learned names that
its training copies contain, but it is not shown to gain F1. Every
decision, the criterion's revision and the amendments are reported. The
practical advice is mention replacement. The methodological advice is to
test a names method against a control with the same copies, and its
recall against a decoding bias checked beforehand to reach the method's
recall, on documents the training data has never seen, with enough
seeds and units to detect a point or two.

## Limitations

- **Few, heterogeneous units.** Seven subfields and four papers give 6
  and 3 degrees of freedom; the subfields differ by 24 points under the
  CRF head (32.49 to 56.82) and range from 2 to 11 articles, the papers
  from 41 to 201 sentences, and the pooled decision weights them equally
  as if drawn from one population. The leave-one-out and sign-flip
  checks (Section 5.1) are post hoc, and with four units the sign-flip
  test cannot go below p = 0.125, so D3 and D4 rest on the t interval
  alone; a mixed model was not fitted, and with two corpora no
  between-corpus variance could be estimated, so the pooled intervals are
  conditional on these two corpora.
- **A revised criterion.** The criterion under which MR6 + names passes
  was adopted on 20 October on the author's decision to push for a
  methods claim, after part of the screen and before the names system's
  own screen result, and when mention replacement was already known to
  meet it (Section 3); under the earlier criterion there is no claim, and
  D3 and D4 would not have run. The gate on D3 and D4 was last lifted at
  18:42 that day, after the screen had chosen MR6 + names. The screen
  that chose MR6 + names and the MuLMS decisions share the same seven
  subfields: the decision seeds keep the selection's seed noise out of
  every MuLMS decision except D5 as executed, but not its fit to these
  units.
- **MuLMS's training pool is two or three articles.** Each MuLMS model
  trains on 400 sentences of two or three of the other subfields'
  articles (Table 1), a different set per held-out subfield with some
  articles recurring across sets; the 88% share of unseen names and the
  24-point spread between subfields reflect that.
- **One training regime.** Every positive result is at about 340 training
  sentences; MuLMS's full pool was never used, SOFC-Exp's larger training
  set is where no gain is shown, and the literature predicts smaller
  gains for mention replacement as training grows. The claim is scoped to
  a few hundred annotated sentences.
- **The names' effect is only partly isolated.** The lexicon-membership
  split (Section 5.4) was run after D1 to D8, on rebuilt lexicons, and
  its reading rule does not say whether a gain is read on its point
  estimate or its interval. The recall-matched control (D11, D12) was
  registered after D1 to D10 were known, with the names' side already
  stored, and run on MuLMS only. Its grid could not reach the names'
  recall in three of the seven subfields, which the protocol matched at
  the grid's end, to the control's disadvantage; without them D11's
  interval covers 0, and the readings that match them otherwise are post
  hoc (Section 5.4). It applies one bias to every tag, so it separates
  name knowledge from a shift toward the lexicon's types only through the
  post hoc split within types, and past D = 2.5 the bias lengthens spans;
  a bias on the tags that begin an entity was not tried. The lexicon's
  precision was not estimated (the precision of the predictions of
  lexicon names, 69 to 75% on MuLMS and 81 to 83% on MaterioMiner, is the
  nearest figure).
- **Untuned settings.** MR3's three copies and swap rate 0.5, MR6's six
  copies, the 0.9 threshold and the three-teacher agreement were fixed in
  advance, not tuned; Dai and Adel tune the copies (1 to 10) and the rate
  (0.1 to 0.7) on development data, and a tuned mention replacement could
  close or widen every gap in Table 3. The two-head filter of the follow-up of 31 October
  admits fewer names than the single-task one, and a looser one was not
  tried.
- **Null results are not equivalences.** The source decision and the
  ablations detect differences of about 2 to 3 points (Supplementary Note
  9); smaller effects of the source or the settings are not excluded.
- **Baselines.** The gazetteer-feature and self-training variants were
  rejected on 3 seeds, and one encoder, MatSciBERT, was used. No large
  language model was compared as an extractor, a teacher or a generator
  of same-type names (zhou2022melm; hu2023entity); a generated pool is a
  different source from names already present in unlabeled text and needs
  its own matched test.
- **Corpora and pre-registration.** The corpora come from Europe PMC's
  open-access subset, found by title terms, with journal concentration
  and licence mixes given in Supplementary Note 3, and are rebuilt from
  hashed reports rather than redistributed. Pre-registration rests on the
  repository's dated commits, whose times the author's machine sets, not
  on an independent registry, though GitHub's own log of the pushes,
  archived in the repository, puts every protocol commit of Table S1 (15
  October to 1 November) on GitHub 1 to 7 seconds after its commit time
  (Supplementary Note 1).

## Ethics and broader impact

The work uses published corpora under their licences and open-access
full texts for the unlabeled corpora, which are not redistributed. No
personal data is involved. An AI coding assistant was used, under the
author's direction, to draft the protocols, write the code, run the
experiments, draft the text of every section from the results files and
run adversarial internal reviews of the drafts; every number in the paper
was taken from the outputs of the repository's analysis scripts and
checked against the stored results files by those reviews, and the
author is responsible for the content. This paper shares
data, baselines and protocol with a companion evaluation study of the
same programme (in preparation; its draft is in the repository), which
reports the random-split gap, seed variance and the other additions
tested. The two papers overlap in Sections 2 to 4. The results that
appear in both are the random-split gap and the unseen-name shares of
Section 1, mention replacement's MaterioMiner decisions and the
multi-task results of Section 4, and the label-propagation rule of
Supplementary Note 7, which the companion reports in its survey; the
companion summarises D1–D8 and the decomposition of Section 5.2 in one
table and cites this paper for them.

## Code availability

The repository, github.com/hatemadi15/AI_Research_Ontologies_GNN_GRL
(Apache 2.0; the commit this paper describes will be tagged, and an
archived copy with a DOI deposited, at submission), holds the code, every
protocol with its dates and amendments, the analyses behind every number,
the generated tables and figures, and the Kaggle batch definitions that
produced each results file.

## Data availability

The same repository holds every results file with its seeds, environment
and saved test predictions, and the corpus reports with their queries,
article identifiers, licences and SHA-256 hashes. MaterioMiner's files
are included under CC BY 4.0; SOFC-Exp is converted from the authors'
repository (CC BY 4.0); MuLMS is read from its Hugging Face release at a
pinned revision (CC BY-SA 4.0). The unlabeled corpora are not
redistributed, 10–33% of their articles being under non-commercial or
no-derivative licences; they rebuild from their reports, with hash
checks, and the reports identify the CC BY articles. The runs did not
store the harvested lexicons; `ner_baseline.py --students 0 --save-names`
rebuilds each from its corpus (the lexicon-split analysis of 31
October), and they are not released, because they derive from corpora
that are not redistributed.

## Author contributions, competing interests and funding

The author conceived and directed the study; an AI assistant, under the
author's direction, drafted the protocols, wrote the code, ran the
experiments and drafted the paper (Ethics and broader impact). The author
is responsible for the content and declares no competing interests. No external
funding was received; the GPU compute was Kaggle's free tier. The author
thanks the creators of MaterioMiner, SOFC-Exp and MuLMS for releasing
their corpora under open licences.
