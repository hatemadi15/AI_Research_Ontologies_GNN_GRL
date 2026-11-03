# Paper B (evaluation): Materials NER on documents it has never read: a pre-registered evaluation on three corpora

*The first draft of 30 October (begun ahead of week 16 on the author's
go-ahead; `docs/METHODS_PLAN.md` §7), brought up to date on 3 November
with the record through week 18 by an audit against the results files
and the submitted paper A, then with week 19's screen (Section 7.6). It
is the evaluation paper of the two decided
on 30 October: the gap between random sentence splits and held-out
documents and its anatomy, seeds and votes, the reproduction, data against
method, and the pre-registered tests of the popular additions with their
counts. The names method is paper A's (`docs/PAPER_A.md`); A's decisions
appear here as rows of one table and its other results in a few
sentences, with a citation. The text was cut from `docs/PAPER_DRAFT.md`
(weeks 1 to 14) and extended from the claims table of `docs/PAPER.md`
(weeks 15 to 19), through which its numbers come from the committed
results (C1–C26, C28–C31). A number in square brackets is a 95%
Student-t confidence interval: after a difference, of the mean paired
difference over the units named (splits or held-out documents, or seeds
or draws where stated); after a score, of that score over its splits,
units or seeds. References are keys of
`docs/references.bib`. The length is open until the venue is chosen: an
8-page conference paper keeps Sections 1, 5, 6, 7.7, 8 and 9 and moves the
rest to an appendix; a journal paper keeps all of it.*

## Abstract

Materials-science named-entity recognition (NER) corpora are built from a few documents and scored
on random sentence splits of them, so a published score says how well a
model reads more sentences of the papers it was trained on, not how it
reads a new paper. We measure the difference on three corpora under
protocols committed before their results, with 7 or 10 model seeds per
unit (fewer in the early weeks) and paired confidence intervals over
held-out documents. On MaterioMiner, a model that scores 80 F1 on random
splits scores 49 on a held-out paper (fine-grained; 79 against 61
coarse-grained); on MuLMS a held-out
materials subfield costs 17 points; on SOFC-Exp, whose 45 articles share
one topic and one annotation scheme, held-out articles cost 3.5. The gap
is partly names: 81% of a new MaterioMiner paper's entities and 88% of a
new MuLMS subfield's are names no training sentence annotates, and the
model finds about a third of them, against 46% on SOFC-Exp, where it finds
four in five; the held-out mix of names accounts for 40 to 58% of
MaterioMiner's recall gap and a third of MuLMS's. Coarse-grained, the
model finds almost every entity of a new paper and loses on boundaries
and types, in part where the papers' annotators set them differently. We then test the additions a practitioner would reach
for: 111 pre-registered decisions over eighteen weeks, of which 24 pass.
The conditional random field (CRF) output layer helps on MaterioMiner and SOFC-Exp (decided on
SOFC-Exp only; MuLMS has no run without it), and seed votes add 0.8 to
2.5 points wherever measured (reported); multi-task training over two
type levels holds on MaterioMiner and fails on SOFC-Exp's two levels;
mention replacement holds on MaterioMiner, passes on MuLMS only by its
votes and is not shown to gain on SOFC-Exp; label semantics,
self-training, document context, recall calibration, a larger encoder and
coarse-to-fine completion fail. A names method built on mention
replacement passes a pooled decision on two corpora under a criterion
adopted mid-study, but three-copy mention replacement alone gives 2.35 of
its 3.58 points (post hoc), and the names are not shown to add to
six-copy mention replacement (paper A). Three lessons are methodological.
A 3-seed design passes a real 1.8-point gain in 41% of seed triples, and
the four comparisons of weeks 3 and 4 that passed with 3 seeds all failed
with 10. A third training paper adds 7 to 9 points, as much as the best
method. And a new document is worth more than as many sentences of known
ones: at equal sentence counts the third paper still adds 1.2 and 3.6
points (3 seeds), though most of its 7 to 9 points is its sentences.
The predictions of every GPU run behind a decision are released.

## 1 Introduction

Materials-science named-entity recognition (NER) corpora are small. They are built from a handful of
papers, and their models are scored on random sentence splits of those
papers. MaterioMiner (durmaz2024materiominer) annotates 476 sentences from
4 fatigue papers with an ontology of process, structure and property
classes. Its reported results are MatSciBERT (gupta2022matscibert) F1 of
69.92 (fine-grained, 27 classes) and 72.32 (coarse-grained) on one random
65/15/20 sentence split, averaged over 5 seeds. An extractor in use meets
new papers, not new sentences of known ones. Random splits leak the
papers' terms between training and test (babl2025random;
augenstein2017generalisation), and the dataset paper notes that four
publications cannot estimate out-of-distribution performance.

We ask five questions on MaterioMiner and take each to a second and a
third corpus where the data allow: SOFC-Exp (friedrich2020sofc), 45
articles on one topic, and MuLMS (schrader2023mulms), 50 articles from
seven materials subfields.
1. Do the published numbers reproduce, and on which split and scoring?
2. How much is lost on a document the model has never seen, and how does
   the loss depend on how much the documents differ?
3. Where is it lost: on names the training documents lack, on annotation
   conventions that differ between documents, or for want of training
   documents?
4. Which common additions improve materials NER on unseen documents once
   seeds are accounted for, and which repeat on a second corpus?
5. How far does a small-seed design mislead?

Every decision was fixed in a protocol committed to the repository
before its results; for a few, part of the runs came first (Section 3;
vanmiltenburg2021preregistering). It is decided by the 95% Student-t
interval of a paired difference over random splits or held-out units
(over seeds on SOFC-Exp's official split), each side the mean of its 7 or
10 model seeds (one in week 2, three in weeks 3, 4 and 8;
reimers2017reporting; dodge2020finetuning; bouthillier2021accounting).
From week 4 on, every decision rests on deterministic GPU runs with their
test predictions released (weeks 1 to 3 and week 4's votes ran on CPUs);
two probes without decisions trained on CPUs (Section 7.6).

The results are these. The published numbers reproduce (Section 5.1). A
held-out document costs 28 to 31 F1 points fine-grained on MaterioMiner,
17 on MuLMS and 3.5 on SOFC-Exp, and the size of the gap follows the
share of test names that training has not seen (Section 5.2). Most of the
fine-grained gap is unseen names the model misses outright; most of the
coarse-grained gap is boundaries and types, in part where the papers'
annotators set them differently (Sections 5.3 to 5.5), and the missed entities are not
near the decision threshold (Section 5.6). Seeds move one split's score by
1.5 to 2.5 points, and a 3-seed design passes a 1.8-point gain that 10
seeds establish in 41% of seed triples. The four comparisons of weeks 3
and 4 that passed with 3 seeds failed when re-tested with 10 on GPUs;
three of them had run on CPUs and already fail on GPUs at the same three
seeds, so a 3-seed interval misses the machine's run-to-run variation as
well as the seeds' (Section 6). Of the additions tested, the CRF output
layer passes on SOFC-Exp's official split, where most of its gain repairs
a tokenization shift, and seed votes add 0.8 to 2.5 points wherever
measured (reported); multi-task training passes on MaterioMiner's random
splits and, by its vote, fine-grained on held-out papers; mention
replacement passes on MaterioMiner's held-out papers (its votes also on
MuLMS's held-out subfields); document-level label propagation passes on
MuLMS and coarse-grained on MaterioMiner by under a point; harvested
names pass on MaterioMiner and, under a pooled criterion adopted
mid-study, pooled with MuLMS, without being shown to add to mention
replacement; the rest are not shown to help. 111 pre-registered
decisions, 24 passing, are counted in Section 7.7. A third annotated paper
adds as much on top of two papers as the best method adds on top of three
(exploratory); at equal sentence counts it keeps a sixth of its gain
fine-grained and two fifths coarse-grained (3 seeds), and the rest is its
sentences (Section 8).
Section 9 draws the lessons for anyone evaluating an extractor that will
meet new documents.

Paper A (`docs/PAPER_A.md`) tests the method the programme arrived at,
mention replacement with six copies and names harvested from unlabeled
text, and finds most of its gain to be three-copy mention replacement's,
the names' increment not established; its results appear here as summary
rows (Section 7.6).

## 2 Data

**MaterioMiner** (durmaz2024materiominer): 476 sentences, 12.2k tokens,
from 4 papers of 41, 81, 201 and 153 sentences.
- Fine-grained, the 27 most frequent classes are kept, as in the dataset
  paper (ties broken by token count, then name). They hold 1,328 of the
  2,189 fine-grained entities.
- Coarse-grained, all 27 types are kept: 2,194 entities.
- The two levels nest: a fine-grained entity is a coarse-grained entity
  with the same span, and each fine-grained class has one coarse-grained
  parent. 15 sentences are tokenized differently in the two files.
- The release ships a split of 323 / 57 / 96 sentences. The paper does
  not describe it, and it partitions the two granularities differently.

**SOFC-Exp** (friedrich2020sofc): 876 annotated sentences from 45 articles
on solid oxide fuel cells, 35.0k tokens, 4,590 entities of 4 types
(MATERIAL, VALUE, DEVICE, EXPERIMENT). Its official split holds out whole
articles: 26 / 8 / 11 articles, 568 / 135 / 173 sentences. Its test set
therefore measures unseen documents.
- A second level, slot types, refines the entity types on the same tokens
  (e.g. anode material or working temperature). Each slot type falls
  within one entity type. The 17 that the dataset's evaluation scores hold
  3,172 mentions (702 in the test articles).
- Its test articles are tokenized differently from its training articles
  (C16). ChemDataExtractor's tokenizer (swain2016chemdataextractor)
  reproduces the tokens of all 7,630 sentences of the 34 training and
  development articles. Java's word BreakIterator reproduces 1,833 of the
  1,836 sentences of the 11 test articles. A model trained on `cm−2`,
  `O2−` and `high - temperature` is tested on `cm − 2`, `O2 −` and
  `high-temperature`; every result on the official split, the published
  ones included, carries this. Scored on the test articles cut as the
  training articles are, the same models gain +0.86 [+0.39, +1.33] (the
  CRF head) and +1.91 [+1.33, +2.49] (plain MatSciBERT) over 10 seeds,
  mostly on unseen names (both pre-registered decisions pass).

**MuLMS** (schrader2023mulms): 50 articles from seven materials subfields
(in the release: electrolysis, graphene, pemfc, poly, semi, sofc and
steel; 2–11 articles each), 10,184 sentences and 46,584 entities of 13
types (CC BY-SA 4.0).
- 32.9% of the entities lie inside or on the span of another. A flat
  tagger needs one level, so we keep the outermost ones: 31,952 (68.6%).
  Of entities with the same span we keep the type first in a fixed order
  (the larger unit first: VALUE over NUM); of two that cross, the one
  that starts first. Most kept entities are materials (12,647) and values
  (4,628).
- Text extraction damaged 276 tokens: 261 are empty or hold whitespace,
  and 15 are private-use symbols or lone combining marks, which a BERT
  tokenizer erases. Each token becomes its whitespace-separated pieces,
  without the pieces a BERT tokenizer erases; 2 entities are left without
  tokens and dropped.
- The published MatSciBERT-CRF (79.6 micro F1) tags the nested entities
  with stacked labels and is scored on all of them, so our scores cannot
  be compared with it.

## 3 Evaluation protocol

**Splits.**
- The published split (MaterioMiner) and the official split (SOFC-Exp).
- Five random 65/15/20 sentence splits, one per split seed; on MuLMS,
  five random splits of a 400-sentence pool drawn from all subfields.
- Held-out documents: on MaterioMiner, train on three papers, with 15% of
  their sentences for validation, and test on the fourth; on SOFC-Exp,
  five folds of 9 held-out articles; on MuLMS, each of the 7 subfields
  held out in turn, the model trained on 340 sentences and validated on
  60 of a 400-sentence pool from two or three articles of the other
  subfields (a different set per held-out subfield, the articles taken
  whole; the random-split pool draws its sentences from all 50 articles),
  about MaterioMiner's training size (234–370 sentences per held-out
  paper), and tested on every sentence of the held-out subfield
  (351–2,384).

**Seeds.** From week 5 on, every system has 10 model seeds per split
(seed offsets 0 to 900), except the 3-seed screens and ablations, week
8's 3-seed design, and the MuLMS decisions on the seven seeds the screen
did not see (300–900, its decision seeds: paper A's decisions D1, D2, D8,
D11, D12 and D7's MuLMS
units; Section 7.6). Weeks 1 and 2 used one run per split, weeks 3 and 4
three seeds. A vote is an entity-level majority over 3 seeds; its score
is the mean over the 3 disjoint seed triples (0–200, 300–500, 600–800).

**Scoring.** seqeval (nakayama2018seqeval) entity-level micro F1 in its
default (conlleval) mode, as the dataset's code scores it
(tjongkimsang2003introduction). Strict IOB2 scoring (an I- tag must
continue an entity of its type, or the span does not count), macro and
support-weighted F1 are reported alongside.

**Decision rule.** A change improves on its reference only if the 95%
Student-t confidence interval of the mean paired micro-F1 difference lies
above 0. The pairs are splits or held-out units (papers, folds,
subfields), each side the mean of its seeds; on a single split (SOFC-Exp's
official split), the pairs are the seeds. With 4 papers the interval has 3
degrees of freedom, with 5 folds 4, with 7 subfields 6, so a gain must be
consistent across units to pass (dror2018hitchhiker). The interval treats
the units as independent, though a corpus's units share training
documents (Section 10); Section 7.7 reports each pass with a post hoc
allowance for it. A pooled
decision takes the units of two corpora together and requires, in
addition, that neither corpus's own interval lie below 0. Each protocol
states its decisions, the comparisons reported without a decision, and
the number of false passes expected if nothing works (2.5% of the
decisions, under the rule).

**Statuses.** A *decision* is a comparison its protocol decides by the
rule above, fixed before its results. A *screen* is a 3-seed comparison
that chooses what to decide, never a result. A *reported* comparison is
named in a protocol without a decision. A *pre-specified* analysis is
named in a protocol before its results, a *post hoc* one is chosen after
them, an *exploratory* one has no protocol, and a *descriptive* number
carries no inference. Intervals over seeds on one split (SOFC-Exp's
official split) are conditional on that split's articles.

**Pre-registration and runs.** A week is a stage of the programme in
which protocols were fixed, not a calendar week; weeks 1 to 19 ran from 5
October to 3 November 2026. Each protocol was committed to the repository
before its results, and before its runs except week 14's (two minutes
after its first batch was pushed) and D5 as executed, whose MuLMS means
include the screen's runs of seeds 0–200, under way when the pooled
decision was added on 20 October; week 5's re-tests kept week 4's seeds
0–200, whose results were known, and the names' side of weeks 14 and 18
was the stored runs. Amendments are marked as such, with their reason and
timing. The order of protocols and runs can be checked against GitHub's
own record of the pushes (`results/github_push_events.tsv`), not only the
commit dates, as paper A's Supplementary Table S1 does. Runs
use Kaggle T4 GPUs with deterministic CUDA, one frozen commit per batch
and a pinned image. From 15 October 16:00 UTC every completed GPU session
but two began by repeating an earlier run, and every repeat was
identical. Weeks 1 to 3 and week 4's votes ran on CPUs, week 4's training
on GPUs; the claims they passed were re-tested on GPUs with 10 seeds in
week 5 (changing the machine as well as the seeds; Section 6.2), and GPU
and CPU runs are never compared with each other in a decision.

## 4 Systems

- **Encoder:** MatSciBERT (gupta2022matscibert) with a token classifier on
  the first sub-word of each word. AdamW at 5e-5 with linear warm-up,
  batch size 16, up to 20 epochs, early stopping on validation F1
  (patience 3).
- **CRF head:** a linear-chain CRF over the words with IOB2 transition
  constraints, trained by likelihood and decoded with Viterbi
  (lafferty2001conditional; lample2016neural). It is the reference of
  every decision from week 5 on.
- **Seed vote:** the entities predicted by at least 2 of 3 seeds.
- **Multi-task:** one encoder with a fine-grained and a coarse-grained CRF
  head, trained on the sum of their losses; on SOFC-Exp, an entity head
  and a slot head.
- **Tested additions** (Section 7): class weights; domain-adaptive and
  in-domain pre-training; other encoders, SciBERT, PubMedBERT and
  DeBERTa-v3-large (he2023debertav3); label semantics (ma2022label);
  self-training on an in-domain corpus with 3 teachers (liang2020bond),
  alone and on the multi-task model; document context
  (schweter2020flert); document-level label propagation (gale1992one;
  krishnan2006effective); coarse-to-fine completion; recall calibration;
  mention replacement (dai2020analysis), alone, on the multi-task model,
  and with harvested names (paper A).

## 5 The gap and its anatomy

### 5.1 The published numbers reproduce (C1)

On the released split, scored like the dataset's code, plain MatSciBERT
scores 70.19 [69.03, 71.35] fine-grained and 72.52 [71.88, 73.16]
coarse-grained over 10 seeds. Both published numbers, 69.92 and 72.32,
lie inside these intervals. Single runs range over 4.6 points
fine-grained (68.0–72.7) and 3.6 coarse-grained (70.4–74.1), so a 5-seed
mean of one split is a noisy reference (Table 1).

### 5.2 The gap on three corpora (C2; Figure 1)

The seed vote of the CRF head scores 79.68 fine-grained and 78.63
coarse-grained over 5 random splits, but 48.81 and 61.10 on a held-out
paper. Every system with 10 seeds on both protocols loses 28–31 points
fine-grained and 17–20 coarse-grained. Figure 1 shows the gap system by
system; no addition with runs on both protocols narrows it by more than a
few points (mention replacement and its combinations were run on held-out
documents only).

The gap depends on how much the documents differ. SOFC-Exp's 45 articles
share one topic and one annotation scheme. There the CRF head scores
86.27 [84.82, 87.71] over 5 random sentence splits and 82.79 [78.33,
87.24] over 5 folds of 9 held-out articles (10 seeds each; Figure 1, third
panel): a gap of 3.48 points, against 28–31 on MaterioMiner. A held-out
SOFC-Exp article brings fewer new names (46% of its test names are
unseen, 30% on random splits; 81% on a held-out MaterioMiner paper,
fine-grained), and the CRF head finds 80% of them. About two of the three
and a half points are boundaries: under any overlap the gap is 1.35.
Which articles are held out matters as much as the protocol: the folds
range from 79.77 to 88.76. The share of new names does not explain this
(47.8% in the best fold, 45.4% in the worst); the folds differ in recall
on seen and unseen names alike (86.6–92.8 and 76.4–89.0), so some
articles are simply harder.

On MuLMS, whose held-out documents come from another materials subfield,
the models train on 340 sentences, about MaterioMiner's training size:
for a held-out subfield, from two or three whole articles of the others;
for a random split, from all 50 articles. The gap below therefore mixes
the change of topic with fewer training documents at equal size (Section
3; week 20 separates them). The CRF head scores 65.72 [65.25,
66.19] over 5 random splits of that size and 48.66 [40.91, 56.40] over
the 7 held-out subfields (10 seeds each; Figure 1, fourth panel): a gap
of 17.06 points. Per subfield it scores 32.49 (semi) to 56.82 (steel).
88% of a held-out subfield's test names are unseen, 57% on random
splits.

| Corpus | Held-out unit | Random splits | Held-out documents | Gap | Unseen test names, held out / random |
|---|---|---|---|---|---|
| MaterioMiner, fine-grained (CRF vote) | 1 of 4 papers | 79.68 | 48.81 | 30.9 | 81% / 34% |
| MaterioMiner, coarse-grained (CRF vote) | 1 of 4 papers | 78.63 | 61.10 | 17.5 | 79% / 40% |
| MuLMS (CRF head, 340 training sentences) | 1 of 7 subfields | 65.72 | 48.66 | 17.1 | 88% / 57% |
| SOFC-Exp (CRF head) | 9 of 45 articles | 86.27 | 82.79 | 3.5 | 46% / 30% |

MaterioMiner's rows are the CRF head's seed vote, the others the CRF head
itself; on MaterioMiner the CRF head alone loses as much (Figure 1).

### 5.3 Where unseen documents break the model (C9, C10; Figure 3)

- **Fine-grained,** the CRF head misses 36% of the gold entities outright
  on unseen papers (10% on random splits) and predicts 10,026 entities
  for 13,280 gold ones. 41% of these misses have a class seen fewer than
  10 times in the three training papers.
- **Coarse-grained,** it misses only 6%, but 16% of the gold entities get
  the right span and the wrong type, and 21% a wrong boundary.
- **Macro F1 on the random splits:** a few rare coarse-grained types (the
  dataset's labels SDE and MIP near 18 F1;
  PRP, MOD and CPR at 40–42, with 2–8 test entities per split) keep it
  near the published number.

### 5.4 Unseen documents are mostly unseen names (C17; Figure 5; Table 4)

Following augenstein2017generalisation, a gold test entity is *seen* if
its exact words are an entity of the same type in the run's training
sentences and *unseen* if they are no entity there; a third kind, seen
with another type only, holds 0–2% of the entities. Predictions are
split the same way, so recall, precision and F1 can be given per kind
(exploratory; all seeds and units pooled).
- **The mix.** On random splits, 34% of the fine-grained test entities are
  unseen (coarse-grained 40%; on the published split 40% and 39%): random
  sentence splits put a paper's names on both sides, as babl2025random
  find for other corpora. On a held-out paper, 81% are unseen
  (coarse-grained 79%), between 72% and 83% for each paper.
- **Unseen names in a new paper are harder.** Fine-grained, the CRF head
  finds 88% of the seen and 59% of the unseen entities of a random split,
  but 71% and 35% of those of a held-out paper (coarse-grained, 89% and
  63% against 82% and 52%).
- **Both effects count.** The CRF head's fine-grained recall falls from
  77.8 on random splits to 40.6 on held-out papers. With the random
  splits' mix of seen and unseen names, its recall per kind on held-out
  papers would give 58.5. The mix and the lower recall within each kind
  each account for 40–60% of the gap, in either order of the
  decomposition. The same holds coarse-grained, where recall falls from
  77.7 to 57.4.
- **Patterned names transfer, descriptive ones do not.** On held-out
  papers, the CRF head finds 76% of the unseen Value entities and 72% of
  the unseen Alloy and MaterialClass entities, whose names follow
  patterns (numbers with units, alloy designations). It finds 30% of the
  unseen StateDescriptor and 33% of the unseen Mechanism entities, and
  almost none of the unseen GrainBoundary, TestingMachine or
  StressAmplitude entities.
- **SOFC-Exp is easier.** 49% of its official test entities are unseen,
  and the CRF head finds 82% of them (89% of the seen). Its articles
  share one topic, its 4 types are broad, and its most frequent type,
  VALUE, is found at 95–97% whether seen or not.
- **MuLMS's held-out subfields are like MaterioMiner's held-out papers.**
  88% of their test entities are unseen (57% on random splits of the same
  size), and the CRF head finds 39% of them (66% of the seen). The larger
  share of unseen names accounts for about a third of its 22.5-point
  recall gap, the lower recall within each kind for the rest.
- **The gains show on unseen names.** Paired over the 4 held-out papers
  (fine-grained F1, against the CRF head, or a vote against the seed
  vote), every gain whose interval clears 0 overall also clears it on
  unseen names, but for six-copy mention replacement (+2.98 [+0.09,
  +5.86] overall, +3.65 [−0.15, +7.46] unseen): multi-task training +4.02
  [+1.53, +6.51] (seen +2.88 [−2.70, +8.47]), its vote +3.60 [+2.32,
  +4.88], mention replacement +3.19 [+1.66, +4.72] (seen +1.49 [−1.31,
  +4.30]) and its vote +4.20 [+1.60, +6.80]. Seen names are a fifth of a
  new paper's entities, so their intervals are wide: only mention
  replacement on the multi-task model (+5.88 [+0.42, +11.34]) and week
  11's names (+2.66 [+0.03, +5.29]) clear 0 there, and most other seen
  gains are a little smaller than their unseen ones by point estimate.
  Multi-task self-training, whose 3-seed screen gained +6.44 [+2.25,
  +10.62] on unseen names, gains +6.35 [−0.13, +12.84] there over 10
  seeds. The CRF head's own gain over plain MatSciBERT lies
  on unseen names as well: +5.94 [+3.11, +8.76] on the published split
  (fine-grained) and +3.02 [+1.99, +4.05] on SOFC-Exp, against −0.47 and
  −0.52 on seen names. On the random splits, self-training gains +2.61
  [+0.50, +4.73] (fine-grained) and +2.26 [+0.92, +3.60] (coarse-grained)
  on unseen names and nothing on seen ones. Mention replacement on the
  multi-task model (Section 7.5) gains on both: +7.10 [+6.43, +7.77] on
  unseen and +5.88 [+0.42, +11.34] on seen names.

A method for unseen documents therefore has to raise F1 on names the
training documents do not contain. That is where 81% of a new paper's
entities are and where every gain measured here shows; whether the gains
are larger there than on seen names, the seen names' wide intervals do
not tell. Paper A's method is built on that observation.

### 5.5 Found, but cut differently (C18)

seqeval counts a prediction only if its span and type are exactly those
of a gold entity. SemEval-2013 Task 9.1 (segurabedmar2013semeval) also
scored looser criteria; we use them with each gold entity matched at most
once and an overlap counted in full (exploratory):
- **Coarse-grained,** the CRF head scores 58.60 on held-out papers when
  span and type must match, 74.17 when only the span must, and 90.74
  [89.17, 92.32] when any overlap counts. On random splits it scores
  76.96, 83.91 and 93.38. The 18.4-point gap shrinks to 9.7 and then 2.6:
  on a new paper the model finds almost every entity, and loses mostly on
  where the boundaries go and which type applies.
- **Fine-grained,** the gap shrinks from 31.0 to 25.0 and 18.3 (overlap F1
  70.58 against 88.83). There the model misses entities outright
  (Section 5.3), most of them unseen names (Section 5.4).
- **Near misses.** On held-out papers, 31% of the fine-grained predictions
  overlap a gold entity but differ in span or type (13% on random
  splits). 10.8% lie inside a longer gold entity: the new paper tags
  "multi-GB cracks", "EBSD data" and "smooth samples" where the model tags
  the noun. 4.2% have a type that is an ancestor or descendant of the gold
one in the materials mechanics ontology (MMO) that MaterioMiner's classes
come from: "samples" is a TestSpecimen in one paper and a
  FatigueTestSpecimen in another.
- **SOFC-Exp:** 83.75 strict and 88.80 under overlap; types are almost
  never confused there (84.06 when only the span must match). Its slot
  types often are: 71.88 strict, 83.44 when only the span must match.
  11.7% of the CRF head's slot-level predictions have a gold entity's span
  and another slot, against 0.3% of its entity-level predictions. Most of
  them give a material the wrong role in the cell: an electrolyte material
  becomes a cathode material 27 times per seed (of 120), an anode material
  7 times. The model finds the materials; their role is what it misses.
- **The gold annotations differ too.** Only 35 fine-grained names are entities
  in two or more papers. 7 of them take different types, 6 of these a
  class and its MMO subclass (DataSet and MechanicalTestDataSet,
  TestSpecimen and FatigueTestSpecimen, MechanicalProperty and
  TensileYieldStrength). DataSet entities are 1.9 words long on average
  in one paper and 1.0–1.1 in the others.

Part of what looks like a failure to generalize is a difference in
annotation conventions between papers, which a model trained on other
papers cannot learn. We report strict F1 for comparison with published
numbers and the looser criteria alongside.

### 5.6 Not a threshold problem (C15; Figure 4)

On unseen papers the CRF head predicts too few entities (Section 5.3). A
natural repair adds a bias D to the emission score of every tag but O
before Viterbi decoding, trading precision for recall; for a calibrated
classifier the F1-optimal threshold is half the attainable F1
(lipton2014optimal), and a single weight on the O label has served that
purpose in CRF extractors (minkov2006ner). We chose D per held-out paper
on inner folds, without the paper's labels, and per run on the
validation articles of SOFC-Exp.
- *Result: no gain.* Against the CRF head on the 4 held-out papers it
  gives −0.59 [−1.75, +0.57] fine-grained and +0.17 [−0.14, +0.48]
  coarse-grained; its votes do not beat the strongest systems (−3.53
  [−8.10, +1.05] against the multi-task vote, −0.09 [−0.75, +0.58]
  against the seed vote); on SOFC-Exp it gives −0.21 [−0.58, +0.17]. The
  inner folds chose D = +1.25, +1.25, −1 and +0.25 fine-grained, but each
  paper's own curve is flat: the best D per paper in hindsight would gain
  only +0.56 on average (+0.18 coarse-grained), and its sign differs by
  paper.
- *Why (Figure 4):* the missed entities are not near the threshold.
  Pooled over the papers and seeds, the head finds 5,386 of 13,280
  fine-grained gold entities. D = +0.25 adds 358 predictions, of which 22%
  are right; the break-even rate for F1 is F1/2 = 23%, against 53.7% for
  the predictions above the threshold. D = +3, a factor of about 20 on the
  odds of a one-word entity, recovers 7% of the 7,894 misses, at 6 wrong
  entities per right one. Coarse-grained and on SOFC-Exp the added
  entities are right 19% and 21% of the time, below break-even (29% and
  42%).

The repair assumed roughly calibrated probabilities. Under that
assumption the entities just below the threshold would be right more
often than F1's break-even rate; on unseen papers they are not. The recall
gap comes from what the model scores, not how its scores are decoded.

## 6 Seeds and votes

### 6.1 Seed variance and the seed vote (C5, C8)

One split's micro F1 varies over the 10 seeds with an SD of 1.5–1.7 points
on random splits and 2.2–2.5 on unseen papers (Figure 2a), more than many
published differences. The seed vote is the most reliable addition we
measured: +2.20 and +1.67 over the mean single CRF run on the random
splits, +2.36 and +2.50 on unseen papers, every interval above 0; on
MuLMS the CRF votes score 50.54 against the head's 48.66, and on SOFC-Exp
85.13 against 83.75 on the official split. It costs three trainings and
nothing else.

### 6.2 How far a 3-seed design misleads (C8; Figure 2)

From the 120 seed triples of the 10-seed runs:
- the 1.83-point multi-task gain passes in 41% of the triples (49 of
  120);
- label semantics and self-training on unseen papers pass in 2–18%;
- up to 37% of triples give the sign opposite to the 10-seed difference.

The project's own history agrees, with a qualification. Weeks 3 and 4
took 22 decisions with 3 seeds; four passed: label semantics
coarse-grained on unseen papers (+1.17 [+0.94, +1.41]), multi-task
training coarse-grained on random splits (+0.75 [+0.10, +1.39]), the
label-semantics vote coarse-grained on unseen papers (+1.63 [+0.11,
+3.16]), all three on CPUs, and self-training fine-grained on unseen
papers (+1.07 [+0.33, +1.80]), on GPUs. Re-decided with 10 seeds on GPUs
in week 5, none of the four passed (−0.36 [−1.66, +0.94], +0.71 [−0.15,
+1.56], −0.63 [−1.72, +0.46] and +1.74 [−1.16, +4.64]; Table 2). For the
three CPU passes the re-test changed the machine as well as the seeds:
recomputed from the GPU runs of the same seeds 0, 100 and 200 (post hoc),
they already fail, +0.07 [−2.47, +2.61], −0.08 [−1.71, +1.55] and +0.40
[−1.36, +2.16]. Only self-training failed because seeds were added, its
mean rising from +1.07 to +1.74 while its interval widened. A 3-seed
interval on one machine thus misses run-to-run variation that another
machine, or more seeds, exposes. The week-6 screen
showed it prospectively: document context passed the 3-seed screen with
+2.66 [+0.91, +4.40] fine-grained, and on the 7 seeds the screen did not
use it gains +0.49 [−3.10, +4.07] (post hoc), one paper going from +4.29
to −2.59;
only 21 of its 120 triples pass. A screen on a noisy estimate selects
methods that drew lucky seeds. We kept 3-seed screens only to choose what
to run with 10 seeds; the decisions of weeks 6 and 7 still averaged in
the screen's three seeds, and from week 12 on, MuLMS's decisions on a
screened method used only the seven seeds the screen did not see, but for
the pooled D5 (Section 7.6); 13.1a and 13.1b, a rule applied to saved
predictions, used all ten.

## 7 The additions, tested

Each addition is decided under a protocol committed before its results,
against the CRF head on held-out documents, and its vote against the
strongest system on unseen papers (the multi-task vote fine-grained, the
seed vote coarse-grained). A methods claim also needs its gain to repeat
on a second corpus. Table 2 lists MaterioMiner's decisions and screens of
weeks 3 to 12 and Table 6 the decisions on held-out documents of weeks 7
to 14; week 2's six decisions, week 8's two on a third paper, SOFC-Exp's
official-split and training-size decisions and paper A's D9 to D12 are
not in them (Section 7.7 counts every decision).

### 7.1 The CRF head, span validity and a tokenization shift (C3, C4, C13, C16)

- **Published split:** the CRF head scores 73.31 [72.28, 74.35] and
  75.48 [74.23, 76.72], +3.12 and +2.96 over plain MatSciBERT, paired by
  seed, above both published numbers.
- **Span validity:** under strict IOB2 scoring the gain shrinks to +1.70
  [+0.21, +3.19] fine-grained and +1.07 [−0.82, +2.97] coarse-grained.
  Much of it comes from predicting only valid spans (Table 3).
- **SOFC-Exp:** the CRF head improves on plain MatSciBERT again, +1.22
  [+0.50, +1.93] over 10 seeds (83.75 against 82.53); strict, +0.73
  [−0.04, +1.51]. The published MatSciBERT-CRF score (84.77;
  gupta2022matscibert) lies above our interval over seeds. On test
  articles cut into words as the training articles are (Section 2), the
  CRF head scores 84.61 [83.89, 85.34], an interval that covers the
  published score, and its gain over plain MatSciBERT falls to +0.17
  [−0.37, +0.70]: on SOFC-Exp the CRF head mostly repairs the
  tokenization shift.
- **SOFC-Exp's slot types:** +1.68 [−0.12, +3.49] (71.88 against 70.20),
  not significant, and −0.06 [−1.70, +1.58] under strict scoring: plain
  MatSciBERT's invalid tags are mostly wrong entities there, so dropping
  them raises its score to 71.94. The published MatSciBERT-CRF score on
  SOFC-Slot (72.85, over 5 cross-validation splits) lies above our
  interval; the seed votes score 74.32.

### 7.2 Multi-task training, on MaterioMiner and on SOFC-Exp's two levels (C7)

- **MaterioMiner:** multi-task training over both granularities is the one
  training change that holds on the random splits, +1.83 [+0.32, +3.34]
  fine-grained (coarse-grained +0.71 [−0.15, +1.56], no). Its vote
  improves on the seed vote coarse-grained on the random splits (+0.79
  [+0.17, +1.42]) and fine-grained on unseen papers (+3.65 [+0.49,
  +6.82]); that is 2 of 12 vote comparisons, where about 0.3 would pass by
  chance. On unseen papers, fine-grained, it mostly buys precision (52.1
  to 58.8) rather than recall (42.7 to 44.8; seeds 0–200, exploratory).
- **SOFC-Exp's two levels,** where the slot types stand to the entity
  types as MaterioMiner's fine level to its coarse one: no gain. On the
  official split, over 10 seeds, multi-task training scores −0.78 [−2.39,
  +0.84] against the CRF head on the slot types (71.11 against 71.88) and
  −0.15 [−1.36, +1.05] on the entity types (83.60 against 83.75); its
  votes do not differ from the seed votes (−0.86 [−3.67, +1.95] and +0.08
  [−2.85, +3.01]). On unseen names it scores −1.52 [−3.65, +0.62] on the
  slot types and −0.45 [−2.27, +1.37] on the entity types, where on
  MaterioMiner's held-out papers it gained +4.02 [+1.53, +6.51]. Its
  largest losses are on the anode (32.4 to 20.3, 26 test entities) and
  cathode materials (27.9 to 22.6, 37) and on open-circuit voltage (77.9
  to 71.9, 25). One possible reason, not tested: a material slot is a
  role in the reported cell, which the entity level does not encode, so
  the entity head has little to add where the slot head errs (Section
  5.5). Multi-task training is therefore a MaterioMiner result.

### 7.3 What does not hold, and a rule that passes by under a point (C6, C11, C12, C14, C25)

Under the 10-seed rule, each of these fails:
- label semantics, as a model or a vote, on either protocol (+1.23
  [−0.96, +3.42] fine-grained on unseen papers);
- self-training with 3 teachers on unseen papers (+1.74 [−1.16, +4.64]
  fine-grained);
- document context (+1.14 [−0.94, +3.22] fine-grained; on SOFC-Exp
  +0.58 [−0.26, +1.42]);
- multi-task self-training (+5.47 [−0.74, +11.68] fine-grained, the
  largest mean gain among the methods that fail here (mention replacement
  on the multi-task model gains +6.73, Section 7.5), which one of the four
  papers does not share: three gain 4–9 points, +8.86, +8.31 and +4.21,
  and the fourth +0.51, so with 3 degrees of freedom the interval reaches
  below 0; coarse-grained +0.90 [−1.18, +2.98]; its screen, +6.06 [+1.05,
  +11.06], held at about +5.2 on the 7 seeds it did not see (post hoc), so this is a
  gain the design cannot confirm rather than one it inflated);
- recall calibration (Section 5.6);
- coarse-to-fine completion, which uses the coarse-grained level to
  complete the fine-grained one: +0.34 [−0.91, +1.59], the entities it
  adds right 39% of the time;
- earlier, with one run per split, class weights (week 1) and
  domain-adaptive pre-training (week 2); with 3 seeds, in-domain
  pre-training and a vote mixing SciBERT, PubMedBERT and MatSciBERT (week
  3; neither encoder was decided alone).

Two more results:
- **DeBERTa-v3-large** (a 3-seed screen of a general-domain encoder at
  MatSciBERT's learning rate, so size and domain are confounded): 435M
  parameters against MatSciBERT's 110M, it loses 4.4 and 8.0 points on
  unseen papers, with similar validation
  scores on the training papers. It did not fit a 15 GB GPU on SOFC-Exp at
  the protocol's batch size, so its planned SOFC-Exp comparison did not
  run.
- **Document-level label propagation:** a rule that tags a term in its
  untagged occurrences when the model tags it in most of them adds +0.30
  [−0.35, +0.95] and +0.12 [+0.01, +0.24] on MaterioMiner (the
  coarse-grained decision passes) and scores −0.42 [−0.61, −0.22] on
  SOFC-Exp's official split, whose annotators leave 14% of a term's
  repeated occurrences untagged. Whether "one sense per discourse" helps
  depends on how the annotation scheme treats repeated mentions.

Taken to held-out documents of the three corpora (C25), the rule on the
CRF head passes on MuLMS's held-out subfields, +0.82 [+0.18, +1.46] (49.47 against 48.66; every subfield
gains, +0.07 to +2.07; 1,213 entities added per seed), as it passed
coarse-grained on MaterioMiner (+0.12 [+0.01, +0.24]), and loses on
SOFC-Exp's held-out articles, −0.44 [−0.72, −0.16] (every fold loses;
reported, not a decision), as on the official split. It does not add to
mention replacement (+0.57 [−0.12, +1.25] on MuLMS; +0.19 [−0.34, +0.73]
and +0.11 [−0.08, +0.31] on MaterioMiner): mention replacement already
teaches the model the forms it sees again in a document, and the rule
propagates the model's errors as readily as its hits. 2 of its 10
pre-registered decisions pass, on two corpora, a two-corpus claim only
under the criterion revised in week 13 (week 7's protocol required
SOFC-Exp); the effect is under a point, with the third corpus's loss
explained by its convention on repeated mentions.

### 7.4 Mention replacement, on three corpora (C20, C22)

Mention replacement (dai2020analysis) makes three copies of each training
sentence with an entity, each mention replaced with probability 0.5 by a
training mention of the same type, so the model sees each type's contexts
with other names. On a materials dataset the original study found gains
for transformers mainly with 50–150 training sentences, so it was screened
with 3 seeds and decided with 10 only because the screen's mean was
positive (+3.25 [−2.33, +8.82] fine-grained, per paper +2.85, +7.95, +2.72
and −0.53).
- **MaterioMiner (C20): a gain at both granularities.** With 10 seeds it
  improves on the CRF head by +2.77 [+1.51, +4.03] fine-grained and +1.28
  [+0.06, +2.51] coarse-grained, and every held-out paper gains (+1.69 to
  +3.48 and +0.29 to +2.10): the first single-model addition to pass this
  decision at both granularities (the names method built on it, paper A's
  D3 and D4, is the second; Section 7.6; the decision averages in the screen's
  three seeds; on the other seven, post hoc, +2.57 [+0.10, +5.03]
  fine-grained and +1.73 [−0.63, +4.08] coarse-grained). The gain is on
  unseen names, +3.19 [+1.66, +4.72] fine-grained (seen +1.49 [−1.31,
  +4.30]). Per type
  (exploratory), coarse-grained, it gains on every rare type that pulls
  macro F1 down (the dataset's labels SDE, MIP, PRP, MOD and CPR, +7.7
  to +17.9), where multi-task training loses (−2.9 to −10.7); over the 27
  coarse-grained
  types the two methods' changes correlate at −0.72. Its vote gains +3.61
  [+1.71, +5.52] over the seed vote fine-grained, as much as the
  multi-task vote, but does not beat the strongest systems (−0.04 [−2.09,
  +2.01] against the multi-task vote fine-grained, +2.26 [−0.12, +4.64]
  against the seed vote coarse-grained).
- **SOFC-Exp: no.** −0.29 [−1.04, +0.47] against the CRF head on the
  official split, paired by seed, and −0.48 [−1.36, +0.39] over the five
  folds of held-out articles. At MaterioMiner's size it does not pass
  either: trained on 10 of the 26 training articles (156–248 sentences,
  about as many entities as MaterioMiner's training sets hold; 5 draws, 10
  seeds each), it gains +0.50 [−0.47, +1.46]
  over the CRF head, paired by draw. The 16 other articles are worth 3.13
  [2.21, 4.04] points to the CRF head there.
- **MuLMS (C22): the single model does not pass, its votes do.** Where a
  held-out subfield differs in topic from the training documents as a new
  MaterioMiner paper does, mention replacement scores 50.76 against the
  CRF head's 48.66: +2.10 [−0.31, +4.51] over 10 seeds, paired by
  subfield. Six subfields gain (electrolysis +1.42, graphene +1.66, pemfc
  +2.28, semi +2.52, sofc +7.39, steel +0.15) and poly loses 0.73, so with
  7 subfields the interval reaches below 0. Macro F1 gains +2.12 [+1.14,
  +3.09] and weighted F1 +2.68 [+0.32, +5.04] (reported). Its votes score
  53.25 against the CRF votes' 50.54: +2.71 [+0.22, +5.20], which passes.
  On unseen names it gains +2.19 [−0.25, +4.63] (seen +1.30 [−1.21,
  +3.81]), its votes +2.92 [+0.41, +5.44].
- **Pooled over the 16 held-out units of the three corpora:** +1.46
  [+0.28, +2.64] (specified on 20 October, when 11 of its 16 units'
  results were known; no decision).

The condition read off SOFC-Exp's results, that mention replacement helps
only where unseen names are many and hard to find, holds on MuLMS, and
the mean gain there is close to MaterioMiner's fine-grained +2.77, but
the single-model decision fails. Across the corpora the passes do not
line up: the single model passes on MaterioMiner, while the vote passes
on MuLMS but not coarse-grained on MaterioMiner. By the pre-registered
criterion, mention replacement stays a MaterioMiner result; it is the
base of paper A's method, and paper A credits it with most of that
method's pooled gain (Section 7.6).

### 7.5 Mention replacement on the multi-task model (C21)

As in Section 7.4, three copies of each training sentence, but a copy
replaces a coarse-grained mention, in both heads' tags at once, by a
training mention of the same coarse- and fine-grained type, so both
levels' tags stay valid.
- **MaterioMiner: the gains add up.** Against the multi-task model it
  gains +3.01 [+0.42, +5.60] fine-grained and +2.30 [+0.49, +4.10]
  coarse-grained over 10 seeds, and every held-out paper gains (+0.81 to
  +4.71 and +0.79 to +3.42). Against the CRF head it gains +6.73 [+6.03,
  +7.43] fine-grained, the sum of its parts (+3.72 and +2.77), and +1.74
  [−0.15, +3.62] coarse-grained, where multi-task training alone loses
  (−0.56). Its vote scores 55.33 fine-grained on unseen papers, +2.86
  [+0.18, +5.54] over the multi-task vote and +6.51 [+3.91, +9.12] over
  the seed vote, the strongest fine-grained system on unseen papers until
  week 17, whose multi-task models with six copies score 55.69 (no names)
  and 55.88 (names) by vote, neither shown to differ from it (+0.36
  [−1.78, +2.50] and +0.55 [−1.64, +2.74]; reported); coarse-grained it
  scores 62.75, +1.65 [−0.47, +3.78] over the seed vote, which does not
  pass. Fine-grained it gains +7.10 [+6.43, +7.77] on
  unseen names and +5.88 [+0.42, +11.34] on seen ones, the first gain
  whose interval on seen names lies above 0; per type (exploratory), 18
  of the 27 fine-grained types gain, most MicrostructureEntity (+33.2),
  FatigueLifetime (+14.7), MechanicalTestDataSet (+14.1) and Mechanism
  (+13.3).
- **SOFC-Exp: no.** +0.69 [−1.18, +2.57] on the slot types and −0.19
  [−1.27, +0.88] on the entity types against the CRF head on the official
  split, and −0.11 [−0.69, +0.46] over the five folds of held-out
  articles. Neither part gained there on its own either.

3 of its 7 decisions pass (13.3b on SOFC-Exp's folds included), where
about 0.18 would by chance. The pre-registered criterion for a methods
claim also needed the gain to repeat on SOFC-Exp at the matching level,
so the combination is a result
for training sets of a few papers with two type levels. There it gains
about as much as a third annotated paper (+7.36, Section 8).

### 7.6 Names from unlabeled text: paper A's results in summary, and a graph screen (C23, C24, C26, C28–C31)

Paper A asks whether the names a new document brings can be given to the
tagger without annotation: three CRF teachers (the CRF head's runs of
seeds 0, 100 and 200) tag 40,000 unlabeled sentences on the held-out
documents' topic; the mentions all three tag with the same span and
type, every marginal at least 0.9, form a lexicon, and
the lexicon enters training through mention replacement with six copies
(MR6; with three copies, as in Section 7.4, MR3). Its pre-registered
decisions (paper A's D1 to D12) and those of week 11 that led to them, in
the order taken, are summarized here; the mechanism, the ablations and
the discussion are A's.

| Comparison (paper A's label) | Units (MuLMS seeds) | Δ [95% CI] (micro F1 unless stated) | Verdict |
|---|---|---|---|
| names from the other subfields' unlabeled sentences vs CRF head / vs mention replacement, MuLMS (week 11) | 7 (0–900) | +2.04 [−0.85, +4.92] / −0.06 [−1.09, +0.96] | no / no |
| the same with the fatigue corpus vs mention replacement, MaterioMiner fine / coarse (week 11) | 4 | +0.23 [−0.69, +1.14] / +0.61 [−0.96, +2.18] | no / no |
| names from the held-out topic's corpus, 6 copies, vs CRF head / vs mention replacement, MuLMS (week 12; D1 / D2) | 7 (300–900) | +3.19 [−1.13, +7.51] / +0.92 [−1.63, +3.48] | no / no |
| the MaterioMiner analogue vs CRF head, fine / coarse (week 12; D3 / D4) | 4 | +4.27 [+1.39, +7.15] / +1.77 [+0.67, +2.88] | passes / passes |
| pooled over MuLMS and MaterioMiner fine vs CRF head (13.2; D5) | 11 (0–900) | +3.58 [+0.95, +6.21], neither corpus below 0 | passes |
| SOFC-Exp article folds vs CRF head (13.4; D6) | 5 | −0.04 [−1.21, +1.14] | no |
| the names vs 6 copies without names, pooled (14.1; D7) | 11 (300–900) | +0.80 [−0.66, +2.26] | no |
| the held-out topic's corpus vs a mixed corpus of six other topics, MuLMS (14.2; D8) | 7 (300–900) | −0.18 [−2.01, +1.65] | no |
| multi-task + MR6 + names vs multi-task + MR6, MaterioMiner fine / coarse (week 17; D9 / D10) | 4 | +0.96 [−0.01, +1.92] / +0.85 [−0.72, +2.42] | no / no |
| MR6 + names vs MR6 decoded at the bias that matches the names' recall over all test entities, MuLMS (week 18): recall on the unseen names the lexicon holds (D11) / micro F1 (D12) | 7 (300–900) | +11.24 [+3.31, +19.18] / +3.26 [−0.79, +7.31] | passes as registered (three subfields matched at the grid's end; +6.05 [−3.33, +15.44] on the four matched) / no |

Twelve decisions in weeks 12 to 18 (paper A's D1 to D12), four passing.
D3 and D4 are on MaterioMiner's four papers, and D5's 11 units include
them, under a pooled criterion adopted on 20 October when three-copy mention
replacement already met it; under the original criterion, a pass on
MuLMS (D1) and on MaterioMiner, there is no claim, and D3, D4 and D6
would not have run (their gate was last lifted that day, after the screen
had chosen the method). D11, on recall, passes as registered only with
three subfields that no decoding bias matched (+6.05 [−3.33, +15.44] on
the four it matched); its outcome was chosen after a post hoc split had
found the effect, so it tests a uniform shift toward tagging more spans,
not the effect afresh. Paper A's decomposition (its Table 3, post hoc but
for D7), its follow-ups (Table 2) and its lexicon split (Table 5, post
hoc) read as follows. On D5's units and seeds, three-copy mention
replacement alone gains +2.35 [+0.94, +3.75]. On the 11 units with MuLMS's
decision seeds, where D5′ (post hoc) reads +3.58 [+1.04, +6.13], it gains +2.46 [+1.00,
+3.91]; three more copies add +0.33 [−0.25, +0.92] (a comparison that
detects 0.8 points), and the names add +0.80 [−0.66, +2.26] (D7, which
detects 2.0). On MaterioMiner's strongest system the names add +0.96
[−0.01, +1.92] and +0.85 [−0.72, +2.42] (D9, D10). Against the
recall-matched bias they add +3.26 [−0.79, +7.31] in F1 (D12). On MuLMS
the names raise recall on the unseen names their lexicon holds (+7.29
[+3.27, +11.30] over MR6), at lower precision, so F1 on those names is
not shown to change (+1.13 [−1.85, +4.11]). On SOFC-Exp, where half the
test names are seen, no system is shown to improve on the CRF head; the
names gain +0.45 [+0.04, +0.86] over three-copy mention replacement there
(pre-specified).

D5 as executed takes MuLMS's 10-seed means, which include the screen's
seeds 0–200 on which the method was selected; its reading on the decision
seeds alone, D5′, is post hoc. Allowing for the training documents that
the units share (post hoc), D4, D5 and D11 stay above 0 and D3's interval
reaches −0.13 (paper A, Statistics and reproducibility).

The week-11 attempt failed; its sentence-level filter admitted 4.4% of a
held-out subfield's unseen names from the other subfields' sentences and
2–9% of a held-out paper's from the fatigue corpus (the reading at the
time, that the filter was the bottleneck, is not borne out by paper A's
ablations of the lexicon's settings); a harvested lexicon applied as a
dictionary rule to the CRF head's predictions helps where the head's
recall trails its precision by 15 points or more (sofc +5.9, pemfc +4.0)
and hurts elsewhere (steel −10.0, graphene −5.1), −1.0 on average (C24; Table 5).
The method's five ablations at the screen's seeds (3 seeds: a random
10,000 or 20,000 of the corpus's sentences, the threshold 0.7, 2 of 3
teachers, the names in half of the draws) lie within 0.16 points of it,
every interval within −2.45 and +2.14, while the lexicon's coverage of the
unseen names ranges from 16.4% to 31.5% (paper A, Supplementary Table S2).

A later screen, fixed before any run (week 19, C31), asked whether a
graph network chooses better names. A relational graph network (R-GCN;
schlichtkrull2018modeling) over every name the three teachers tag in the
unlabeled corpora, its relations sentence co-occurrence, character
similarity and a shared head word and its node features the teachers'
vote among others, was trained to type each unit's 82 to 253 training
names and ranked the new names by type probability. Its rivals at equal
lexicon size were the teachers' vote at threshold 0.7 (3 of 3) and the
same network without the graph (a multilayer perceptron, MLP). A
test-set gate over D7's 11 units decided whether the F1 decisions would
run: the graph's lexicon had to hold 5 points more of the unseen test
mentions than the vote and 2 more than the MLP. It holds 16.8% of them on
average, the vote 24.9% and the MLP 18.8%, below the vote in all 11 units
though with fewer of the names it holds under a wrong type (4.7% against
10.1%). The gate failed, the decisions did not run, and the screen enters
Section 7.7's count as one negative screen. In paper A's ablations,
lexicons holding 16% to 32% of the unseen names gave F1 within 0.16
points of each other, so the screen bears on how names are ranked, not
directly on F1. (The probe keeps one type per name and only names no
training sentence annotates, so on MuLMS, pooled, its vote holds 30.3% of
the unseen test mentions at 0.7 and 22.0% at 0.9, a little below paper
A's 31.5% and 22.2%.)

### 7.7 The count

Through week 19 the programme took 111 pre-registered decisions (weeks 2
to 18; weeks 15, 16 and 19 took none, and week 1's comparisons (class
weights and the CRF head) are left out, since they have no protocol in `docs/ROADMAP.md` §4), 15
three-seed screens (Table 2; paper A, Supplementary Table S5) and week
19's lexicon screen, which failed (Section 7.6). 24
decisions pass, where about 2.8 would if nothing worked. By design: 6
with one run per split (week 2), none passing; 24 with 3 seeds, 6
passing, namely the 22 of weeks 3 and 4 (4 passing, none of which
survived its 10-seed re-run) and week 8's 2 on a third paper at equal
size (both passing, never re-run); 81 with 7 or 10 seeds, 18 passing (75
with 10 seeds, 5 on MuLMS's 7 decision seeds, D7 on both). By addition,
with 7 or 10 seeds (8 of them on MaterioMiner's random splits):

| Addition | Decisions | Pass | Where it holds |
|---|---|---|---|
| CRF head over plain MatSciBERT | 1, on SOFC-Exp's official split, paired by seed (the published split reported without a decision) | 1 | MaterioMiner (reported) and SOFC-Exp, where most of the gain repairs a tokenization shift; not run on MuLMS; half the gain is span validity |
| seed vote over the single model | none with 7 or 10 seeds (reported in every setting); week 2's 2, against one run per split, fail | 0 | adds 0.8 to 2.5 points wherever measured |
| multi-task training (two type levels) | 6 on MaterioMiner (4 on random splits), 2 on SOFC-Exp | 3 / 0 | MaterioMiner only |
| mention replacement | 4 on MaterioMiner, 2 on MuLMS, 3 on SOFC-Exp (official split, folds, 10-article draws) | 2 / 1 (the votes) / 0 | MaterioMiner; MuLMS by vote |
| mention replacement on the multi-task model | 4 on MaterioMiner, 3 on SOFC-Exp (official split at two levels, folds) | 3 / 0 | MaterioMiner only |
| harvested names by mention replacement (week 11; paper A's D1 to D12) | 16: 7 on MuLMS, 6 on MaterioMiner, 2 pooled, 1 on SOFC-Exp | 4 (D3, D4, D5, D11) | MaterioMiner, and pooled with MuLMS under a criterion adopted mid-study; on MuLMS alone only on recall, with three subfields unmatched; not over six-copy mention replacement in F1; not SOFC-Exp |
| document-level label propagation | 10: 6 on MaterioMiner, 2 on MuLMS, 2 on SOFC-Exp | 2 | MuLMS and coarse-grained MaterioMiner; loses on SOFC-Exp |
| label semantics, self-training, document context, multi-task self-training, recall calibration, coarse-to-fine completion | 28 (6, 6, 5, 4, 5 and 2), 2 of them on SOFC-Exp | 0 | nowhere; DeBERTa-v3-large stopped at its screen, and its SOFC-Exp decision did not run |
| SOFC-Exp's test articles cut as its training articles are (a measurement) | 2 | 2 | — |

With 3 seeds, label semantics passes 2 of 8 decisions (4 of them on the
multi-task model), self-training 1 of 6, multi-task training 1 of 4, the
mixed-backbone vote 0 of 2, in-domain pre-training 0 of 2 and a third
paper at equal size 2 of 2; with one run per split, week 2's multi-task
training, domain-adaptive pre-training and seed vote pass none of 6.

Two findings of the count stand on their own. First, wherever a pass on
held-out documents was split by kind of name, its gain shows on unseen
names (Section 5.4; paper A), and on seen names it clears 0 only where
they are many or the gain is large: SOFC-Exp's two measurements (+0.47
and +0.32 on seen against +1.20 and +3.50 on unseen names) and mention
replacement on the multi-task model; the label-propagation rule's two
passes and the 3-seed passes were not split. Second,
no method for unseen documents that passed on one corpus is shown to
improve on the CRF head on SOFC-Exp, whose held-out articles bring fewer
new names (46%) and whose baseline leaves little headroom (82.8 F1); the
two cannot be separated here.

## 8 Data against method (C19; Figure 6)

The inner folds of Section 5.6 train the CRF head on two papers. On each
held-out paper, training on all three other papers instead adds +7.36
[+1.98, +12.74] F1 fine-grained (39.02 to 46.38) and +8.90 [+5.90,
+11.90] coarse-grained (50.37 to 59.27), over 4 papers and seeds 0–200
(exploratory);
every paper gains, from +3.2 to +11.6. A third paper also brings more
sentences. A pre-registered run separates the two by training on three
papers cut to the size of two. At the same number of sentences, the third
paper adds +1.18 [+0.22, +2.15] fine-grained (39.02 to 40.20) and +3.59
[+1.59, +5.58] coarse-grained (50.37 to 53.96), and every paper gains. Its
own sentences add +6.17 [+1.10, +11.24] and +5.31 [+4.13, +6.50].
By point estimate, diversity is thus about a sixth of the gain fine-grained and two fifths
coarse-grained.

A third paper matches the largest method gains measured on unseen papers
(mention replacement on the multi-task model, +6.73 fine-grained; with six
copies and harvested names, +7.49 [+5.96, +9.02], reported in week 17): for
materials NER on new papers, annotating more papers is the lever to beat.
An outside result points the same way. Potu et al. (potu2025ontology,
Fig. 2) train MatSciBERT on 47 other fatigue papers, annotated by
MaterioMiner's curator with the same classes, and report 66.74 F1 on
MaterioMiner, against 78.56 in distribution. Our models trained on three
of its papers score 46–54 on the fourth (up to 56 by vote). The setups
differ (no CRF, 28 classes, all 4 papers as the test set), and their two
scores come from different test sets, so their difference is not a gap
in our sense; their 66.74 is an outside score, 13 to 20 points above our
single models trained on three papers.
Their corpus cannot be redistributed. A method that makes better use of
the few annotated papers there are, and of unannotated ones, is what
remains; paper A tests one, names harvested from unannotated papers, and
their increment over mention replacement is not established.

## 9 Lessons for evaluating materials NER on unseen documents

1. **Hold out documents, and say which.** A random sentence split scores
   the reading of known papers; a held-out paper or subfield scores what
   an extractor will meet. Report both, and the share of test names the
   training data does not contain, which predicts the size of the gap
   (81% and 88% against 46%).
2. **Count unseen names and match criteria separately.** Fine-grained,
   the gap is names missed outright; coarse-grained, it is boundaries and
   types, in part where the papers' annotators set them differently, and the model finds
   almost everything under any overlap. A single strict F1 hides which.
3. **Ten seeds, paired over documents.** One split's score moves 1.5 to
   2.5 points with the seed. Three seeds pass a 1.8-point gain that ten
   establish 41% of the time, and passed four comparisons that ten seeds
   on GPUs rejected, three of them already at the same seeds once the
   machine changed. A 3-seed
   screen is a way to choose what to run, on seeds the decision will not
   use, never a result (this programme's weeks 6 and 7 and paper A's D5
   still decided on all ten seeds, and week 8's two 3-seed decisions were
   never re-run).
4. **Take the vote.** Three seeds' majority adds 1.3 to 2.5 points on
   MaterioMiner and MuLMS and 0.8 to 2.4 on SOFC-Exp (reported, not
   decided), at the cost of three trainings.
5. **Expect a method to fail to transfer.** Multi-task training, mention
   replacement and their combination each passed on MaterioMiner and
   were not shown to gain on SOFC-Exp, whose held-out articles bring
   fewer new names and whose baseline leaves less headroom; which of the
   two blocks them is not known, so check both before adopting a method.
6. **Pre-register and count.** Of 111 decisions, 24 pass; a paper that
   reported the 24 alone would read as a different programme. The count
   is part of the result.
7. **Annotate new documents.** A third paper adds 7 to 9 points on a new
   paper, as much as the best method; at equal sentence counts it still
   adds 1.2 and 3.6 points (3 seeds), but most of its gain is its
   sentences.

## 10 Limitations

- **Fine-tuned encoder taggers only.** Every system is a fine-tuned
  BERT-base tagger, but for DeBERTa-v3-large's screen. No large language model or generalist tagger, which
  would not depend on the training documents' names, was run on the
  held-out documents; whether one narrows the gap is open.
- **Model selection inside the training documents.** The epoch is chosen
  on a random 15% of the training documents' own sentences (on MuLMS, 60
  sentences of the pool), the kind of split this paper argues against;
  choosing it on a held-out training paper (the inner folds of Section
  5.6) might lower the validation scores' optimism, and was not tried for
  the reference systems.
- **Few units.** The unseen-paper decisions have 3 degrees of freedom,
  the fold decisions 4 and the subfield decisions 6, so only large,
  consistent gains pass, and the per-unit results vary widely (Figure 1's
  intervals; SOFC-Exp's folds alone range over 9 points, 79.77 to 88.76;
  MuLMS's subfields over 24, 32.49 to 56.82).
- **SOFC-Exp's methods on one split.** The gap of Section 5.2 is measured
  over 5 random splits and 5 folds, but most methods on SOFC-Exp are
  compared on its official split, with intervals over seeds only; mention
  replacement, its multi-task combination and paper A's method are the
  ones also decided on the folds.
- **MuLMS as converted.** Our MuLMS runs keep the outermost entities
  (68.6%) and repair 276 damaged tokens (Section 2), so their scores
  cannot be compared with the published nested ones; and the models train
  on 340 sentences by design, to match MaterioMiner's size, not on the
  corpus.
- **Methods tested on MaterioMiner only.** Self-training used a fatigue
  corpus, and coarse-to-fine completion was tested before SOFC-Exp's slot
  level was converted; their decisions rest on MaterioMiner alone (on
  MuLMS self-training was only screened with 3 seeds; paper A,
  Supplementary Table S5), as do
  recall calibration's beyond one SOFC-Exp split.
- **The released split** is undocumented. Its use reproduces the
  published numbers, but the authors have yet to confirm that it is
  theirs.
- **Annotation conventions differ between papers** (Section 5.5). Strict
  F1 on a held-out paper also counts the spans and types its annotators
  chose differently, which no model trained on the other papers can
  learn. The decisions use strict F1 for comparability; the looser
  criteria are reported, not decided on.
- **The count is of this programme's choices.** The 111 decisions test
  the additions we chose, in the order we chose them, with amendments
  marked; another programme would test others. About 2.8 would pass by
  chance whatever their dependence, which only makes chance passes come
  together. The votes share their members, the pooled decisions share
  units with the per-corpus ones, and the units share training documents
  (two MaterioMiner papers' training sets share two papers, two SOFC-Exp
  folds 27 of their 36 articles, 5 of the 21 MuLMS pairs an article),
  which also makes the intervals over units too narrow (paper A,
  Statistics and reproducibility); Section 7.7 gives every pass with a
  post hoc allowance for it.

## 11 Related work

**Materials NER.** Weston et al. (weston2019named) trained an NER model
on annotated materials abstracts and applied it to millions of abstracts.
MatSciBERT (gupta2022matscibert) is pre-trained on materials papers;
Trewartha et al. (trewartha2022quantifying) compare general, scientific
and materials pre-training on materials NER tasks, and MatSci-NLP
(song2023matscinlp) gathers materials NLP tasks, NER among them, into one
benchmark. MuLMS (schrader2023mulms) annotates 50 articles from seven
materials subfields; training one encoder with a head per corpus on it,
SOFC-Exp and a synthesis-procedure corpus did not improve NER on any of
them, which agrees with our multi-task result on SOFC-Exp's two levels
(Section 7.2). SOFC-Exp (friedrich2020sofc) holds out whole articles in
its official split, and its test articles are cut into words differently
from its training articles (Section 2; swain2016chemdataextractor).
MaterioMiner (durmaz2024materiominer) types its entities with the classes
of an ontology and ships one split that its paper does not describe.
Potu et al. (potu2025ontology) recognize ontology-conformal materials
entities and study a shift between two fatigue datasets; we study the
shift to an unseen paper of the same dataset. Large language models now
extract materials data by fine-tuning (dagdelen2024structured) or by
prompting (polak2024extracting); we evaluate fine-tuned encoder taggers
only (Section 10).

**Splits and unseen mentions.** Whether random splits are good enough is
contested: Gorman and Bedrick (gorman2019standard) argue for several
random splits over one standard split, and Søgaard et al.
(sogaard2021random) find that random splits overestimate performance on
new data. For NER, random splits leak entity names from training into
test and inflate scores (babl2025random), and performance is far lower on
mentions unseen in training (augenstein2017generalisation). Work since
splits test mentions by whether training saw them and finds the same
dependence: contextual embeddings help most on unseen mentions
(taille2020contextualized), coverage of test entities by training
explains much of a model's score (fu2020rethinking; lin2020rigorous),
models lean on the regularity of names (ghaddar2021context) and fail on
swapped ones (agarwal2020entityswitched), biomedical taggers generalize
poorly to novel entities (kim2022biomedical), and cross-domain benchmarks
make the shift explicit (liu2021crossner). We measure both effects on
three materials corpora by leaving whole documents out (Sections 5.2 and
5.4). Looser matching criteria, as in SemEval-2013 Task 9.1
(segurabedmar2013semeval), separate entities that are missed from those
found with another span or type (Section 5.5).

**Seeds, significance and pre-registration.** Scores of neural taggers
vary with the random seed enough that a comparison of single runs can
mislead (reimers2017reporting); in fine-tuning, seeds change both the
weight initialization and the data order (dodge2020finetuning), and
benchmarks should account for such variance (bouthillier2021accounting).
We decide with paired intervals over splits or documents
(dror2018hitchhiker), each side the mean of its seeds (10, or 7 on
MuLMS's decision seeds), under protocols committed before their results
(vanmiltenburg2021preregistering). Few units leave such decisions with
little power (card2020power), units that share training data make
intervals too narrow (nadeau2003inference), and many decisions call for
a false-discovery count (benjamini1995controlling) or a replicability
analysis across datasets (dror2017replicability); Section 7.7 reports
each. Section 6.2
shows what a 3-seed design would have concluded on the same runs, and
what it did conclude in our own first weeks.

**Methods for unseen documents.** The additions we test come from work on
low-resource and cross-document NER: CRF output layers
(lafferty2001conditional; lample2016neural); label names as type
representations (ma2022label); document context (schweter2020flert);
label consistency within a document (gale1992one; krishnan2006effective);
mention replacement (dai2020analysis), whose gains for transformers
shrink as training data grows (on SOFC-Exp they are absent even at
MaterioMiner's training size, Section 7.4), and generative augmentation
(ding2020daga; zhou2022melm), which we did not test; larger encoders (he2023debertav3); self-training with a teacher
and a student (liang2020bond), here with an in-domain corpus of
open-access papers, one per dataset, without the dataset's own papers or
any sentence that shares a word 8-gram with them (Du et al.,
du2021selftraining, retrieve unlabeled sentences from the web instead).
Decision thresholds: for a calibrated classifier the F1-optimal threshold
is half the optimal F1 (lipton2014optimal); a single weight on the O label
trades precision for recall in CRF extractors (minkov2006ner), CRFs can
be trained for F-score directly (suzuki2006training), and calibration
degrades under dataset shift (ovadia2019trust), which Section 5.6 finds.
Names harvested from unlabeled text, from Kozareva's bootstrapped
gazetteers (kozareva2006bootstrapping) to teachers tagging target text for
a student (wu2020teacher; qu2023adaptive), are paper A's related work.

## Reproducibility

The repository holds the protocols with their dates and amendments
(`docs/ROADMAP.md` §4), every results file with its seeds and
environment and, from week 4 on, its saved test predictions (`results/`,
`results/gpu/`), the analyses behind every number
(`src/graph/ner_seedmeans.py`, `ner_seed_analysis.py`,
`ner_mentions.py`, `ner_consistency.py`, `ner_matching.py`,
`ner_robustness.py`, `decision_tests.py`, `ner_lexicon_recall.py`,
`ner_recall_matched.py`; week 19's screen, `graph_names.py`), the
generated tables and figures
(`docs/tables/`, `docs/figures/`; `paper_tables.py`, `paper_figures.py`),
the corpus conversions (`sofc_exp.py`, `mulms.py`) and the Kaggle batches
that produced each file (`src/graph/kaggle_jobs.py`). Runs used Kaggle T4
GPUs with deterministic CUDA, a frozen commit per batch and a pinned
image; from 15 October 16:00 UTC every completed session but two began by
repeating an earlier run, and every repeat was identical. Table 1 gives
the published-split numbers, Table 2 the decisions and screens on
MaterioMiner of weeks 3 to 12, Table 3 default against strict scoring,
Table 4 the unseen-name shares and match criteria, Table 6 the decisions
on held-out documents of weeks 7 to 14; Section 7.7 counts the decisions
that are in neither.
