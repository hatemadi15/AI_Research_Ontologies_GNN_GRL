# Draft: materials NER on unseen papers

**Status.** A working draft of the sections that hold whichever paper this
becomes (`docs/PAPER.md`): a methods paper if a method passes its
pre-registered decisions and repeats on SOFC-Exp, otherwise an evaluation
paper. Sections 1–5, 7 and 8 rest on finished, committed results. In
Section 6 (methods for unseen papers), multi-task self-training (no
decision passes, C6), recall calibration (no gain, C15), multi-task
training on SOFC-Exp's two levels (no gain, C7) and mention replacement (a
gain on unseen papers at both granularities, C20) are decided, and so is
mention replacement on the multi-task model (the strongest fine-grained
system on unseen papers, but no gain on SOFC-Exp, C21). Mention replacement
gives no gain on SOFC-Exp even with its training set cut to MaterioMiner's
size (ROADMAP §4, week 9, part 2). On a third corpus, MuLMS, whose
held-out subfields differ in topic as MaterioMiner's papers do, its single
model does not pass and its votes do (C22), so by the pre-registered
criterion it stays a MaterioMiner result. The numbers come from `results/`
through the claims table of `docs/PAPER.md` (C1–C22). References are keys of `docs/references.bib`.

---

## Abstract (placeholder)

To be written once Section 6 is decided. The evaluation findings it will
carry either way:
- the published MaterioMiner numbers reproduce on the released split;
- a model trained on three papers loses 18–31 F1 points on the fourth;
  on SOFC-Exp, whose 45 articles share one topic and one annotation
  scheme, held-out articles cost 3.5 points, and on MuLMS a held-out
  materials subfield costs 17, so the gap grows with how much the
  documents differ;
- of the additions tested under a pre-registered 10-seed design, only the
  CRF head, seed votes and multi-task training hold, and multi-task
  training only on MaterioMiner: on SOFC-Exp's two levels it gains nothing;
- mention replacement is the one method that improves a single model on
  unseen papers at both granularities (+2.77 and +1.28 F1), all of it on
  names the training papers lack; on SOFC-Exp it gives nothing, even with
  its training set cut to MaterioMiner's size, and on MuLMS's held-out
  subfields, where unseen names are as many as in a new MaterioMiner
  paper, its single model gains +2.10 without passing while its vote
  passes;
- on the multi-task model the two gains add up: +6.73 F1 fine-grained over
  the CRF head, close to what a third training paper adds, and its vote is
  the strongest fine-grained system on unseen papers; on SOFC-Exp the
  combination gives nothing at either level;
- a 3-seed design passes a real 1.8-point gain in 41% of seed triples, and
  let through a 2.7-point gain that 10 seeds shrank to 1.1;
- the entities a model misses on unseen papers lie far below its decision
  threshold, so shifting the threshold does not recover them;
- 81% of a new paper's fine-grained entities are names that no training
  paper annotates (35% on random splits), the model finds 35% of them, and
  every gain that holds on unseen papers is a gain on these names;
- coarse-grained, the model finds almost every entity of a new paper (90.7
  F1 when any overlap counts, 93.4 on random splits); most of the strict
  gap lies in span boundaries and types that the papers' annotators set
  differently;
- a third training paper adds 7–9 F1 points on an unseen paper, as much as
  the best method. At the same number of training sentences it still adds
  1.2 (fine-grained) and 3.6 (coarse-grained) points, so most of a paper's
  value is its sentences.

## 1 Introduction

Materials-science NER corpora are small. They are built from a handful of
papers, and their models are scored on random sentence splits of those
papers. MaterioMiner (durmaz2024materiominer) annotates 476 sentences from
4 fatigue papers with an ontology of process, structure and property
classes. Its reported results are MatSciBERT (gupta2022matscibert) F1 of
69.92 (fine-grained, 27 classes) and 72.32 (coarse-grained) on one random
65/15/20 sentence split, averaged over 5 seeds. An extractor in use meets
new papers, not new sentences of known ones. Random splits leak the papers'
terms between training and test (babl2025random; augenstein2017generalisation),
and the dataset paper notes that four publications cannot estimate
out-of-distribution performance.

We ask five questions on MaterioMiner, with a second dataset, SOFC-Exp
(friedrich2020sofc), where possible:
1. Do the published numbers reproduce, and on which split and scoring?
2. How much is lost on a paper the model has never seen?
3. Where is it lost: on names the training papers lack, on annotation
   conventions that differ between papers, or for want of training papers?
4. Which common additions improve materials NER on unseen papers once
   seeds are accounted for?
5. How far does a small-seed design mislead?

Every comparison was fixed in a protocol committed before its runs
(vanmiltenburg2021preregistering). It is decided by the 95% Student-t
interval of a paired difference over splits or papers, each side the mean
of 10 model seeds (reimers2017reporting; dodge2020finetuning;
bouthillier2021accounting). All runs are deterministic GPU runs, and every
run's test predictions are released.

## 2 Data

**MaterioMiner** (durmaz2024materiominer): 476 sentences, 12.2k tokens,
from 4 papers of 41, 81, 201 and 153 sentences.
- Fine-grained, the 27 most frequent classes are kept, as in the dataset
  paper (ties broken by token count, then name). They hold 1,328 of the
  2,189 fine-grained entities.
- Coarse-grained, all 27 types are kept: 2,194 entities.
- The two levels nest: a fine-grained entity is a coarse-grained entity
  with the same span, and each fine-grained class has one coarse-grained
  parent. 15 sentences are tokenised differently in the two files.
- The release ships a split of 323 / 57 / 96 sentences. The paper does not
  describe it, and it partitions the two granularities differently.

**SOFC-Exp** (friedrich2020sofc): 876 annotated sentences from 45 articles
on solid oxide fuel cells, 35.0k tokens, 4,590 entities of 4 types
(MATERIAL, VALUE, DEVICE, EXPERIMENT). Its official split holds out whole
articles: 26 / 8 / 11 articles, 568 / 135 / 173 sentences. Its test set
therefore measures unseen documents.
- A second level, slot types, refines the entity types on the same tokens
  (e.g. anode material or working temperature). Each slot type falls within
  one entity type. The 17 that the dataset's evaluation scores hold 3,172
  mentions (702 in the test articles).
- Its test articles are tokenized differently from its training articles
  (C16). ChemDataExtractor's tokenizer (swain2016chemdataextractor)
  reproduces the tokens of all 7,630 sentences of the 34 training and
  development articles. Java's word BreakIterator reproduces 1,833 of the
  1,836 sentences of the 11 test articles. A model trained on `cm−2`,
  `O2−` and `high - temperature` is tested on `cm − 2`, `O2 −` and
  `high-temperature`; every result on the official split, the published
  ones included, carries this. Scored on the test articles cut as the
  training articles are, the same models gain +0.86 [+0.39, +1.33] (the CRF
  head) and +1.91 [+1.33, +2.49] (plain MatSciBERT) over 10 seeds, mostly
  on unseen names (week 7, part 2e; both pre-registered decisions pass).

**MuLMS** (schrader2023mulms): 50 articles from seven materials subfields
(in the release: electrolysis, graphene, pemfc, poly, semi, sofc and
steel; 2–11 articles each), 10,184 sentences and 46,584 entities of 13
types (CC BY-SA 4.0). Used in Section 6.6 only.
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
- Five random 65/15/20 sentence splits, one per split seed.
- Leaving one paper out: train on three MaterioMiner papers, with 15% of
  their sentences for validation, and test on the fourth.

**Seeds.** Every system has 10 model seeds per split (seed offsets 0 to
900). A vote is an entity-level majority over 3 seeds; its score is the
mean over the 3 disjoint seed triples (0–200, 300–500, 600–800).

**Scoring.** seqeval (nakayama2018seqeval) entity-level micro F1 in its
default (conlleval) mode, as the dataset's code scores it
(tjongkimsang2003introduction). Strict IOB2, macro and support-weighted F1
are reported alongside.

**Decision rule.** A change improves on its reference only if the 95%
Student-t confidence interval of the mean paired micro-F1 difference lies
above 0. The pairs are splits or held-out papers, each side the 10-seed
mean; on a single split (SOFC-Exp), the pairs are the seeds. With 4 papers
the interval has 3 degrees of freedom, so a gain must be large and
consistent across papers to pass (dror2018hitchhiker). Each protocol states
its decisions, the comparisons reported without a decision, and the number
of false passes expected if nothing works.

**Pre-registration and runs.** Each protocol was committed to the
repository before its runs; amendments are marked as such, with their
reason and timing. Runs use Kaggle T4 GPUs with deterministic CUDA, one
frozen commit per batch and a pinned image. Each GPU session first repeats
an earlier run, and every repeat was identical.

## 4 Systems

- **Encoder:** MatSciBERT (gupta2022matscibert) with a token classifier on
  the first sub-word of each word. AdamW at 5e-5 with linear warm-up,
  batch size 16, up to 20 epochs, early stopping on validation F1 (patience
  3).
- **CRF head:** a linear-chain CRF over the words with IOB2 transition
  constraints, trained by likelihood and decoded with Viterbi
  (lafferty2001conditional; lample2016neural).
- **Seed vote:** the entities predicted by at least 2 of 3 seeds.
- **Multi-task:** one encoder with a fine-grained and a coarse-grained CRF
  head, trained on the sum of their losses.
- **Tested additions:**
  - class weights;
  - domain-adaptive and in-domain pre-training;
  - label semantics (ma2022label);
  - self-training on an in-domain corpus with 3 teachers (liang2020bond);
  - DeBERTa-v3-large (he2023debertav3);
  - document context (schweter2020flert);
  - document-level label consistency (gale1992one; krishnan2006effective);
  - coarse-to-fine completion;
  - multi-task self-training, recall calibration, mention replacement and
    mention replacement on the multi-task model (Section 6).

## 5 Results

**5.1 Reproduction (C1).** On the released split, scored like the
dataset's code, plain MatSciBERT scores 70.19 [69.03, 71.35] fine-grained
and 72.52 [71.88, 73.16] coarse-grained over 10 seeds. Both published
numbers, 69.92 and 72.32, lie inside these intervals. Single runs range
over 4.6 points fine-grained (68.0–72.7) and 3.6 coarse-grained
(70.4–74.1), so a 5-seed mean of one split is a noisy reference.

**5.2 The gap to unseen papers (C2; Figure 1).** The seed vote of the CRF
head scores 79.68 fine-grained and 78.63 coarse-grained over 5 random
splits, but 48.81 and 61.10 on a held-out paper. Every system with 10 seeds
on both protocols loses 28–31 points fine-grained and 17–20 coarse-grained.
Figure 1 shows the gap system by system; no addition narrows it by more
than a few points.

The gap depends on how much the papers differ. SOFC-Exp's 45 articles share
one topic and one annotation scheme. There the CRF head scores 86.27
[84.82, 87.71] over 5 random sentence splits and 82.79 [78.33, 87.24] over
5 folds of 9 held-out articles (10 seeds each; Figure 1, third panel): a
gap of 3.48 points, against 28–31 on MaterioMiner. A held-out SOFC-Exp
article brings fewer new names (46% of its test names are unseen, 30% on
random splits; 81% on a held-out MaterioMiner paper, fine-grained), and the
CRF head finds 80% of them. About two of the three and a half points are
boundaries: under any overlap the gap is 1.35. Which articles are held out
matters as much as the protocol: the folds range from 79.77 to 88.76. The
share of new names does not explain this (47.8% in the best fold, 45.4%
in the worst); the folds differ in recall on seen and unseen names alike
(86.6–92.8 and 76.4–89.0), so some articles are simply harder.

On MuLMS (Section 2), whose held-out documents come from another materials
subfield, the gap is close to MaterioMiner's. The models train on 340
sentences, about MaterioMiner's training size. The CRF head scores 65.72
[65.25, 66.19] over 5 random splits of that size and 48.66 [40.91, 56.40]
over the 7 held-out subfields (10 seeds each; Figure 1, fourth panel): a
gap of 17.06 points. Per subfield it scores 32.49 (semi) to 56.82 (steel).
88% of a held-out subfield's test names are unseen, 57% on random splits.

**5.3 The CRF head, and a second dataset (C3, C4, C13).**
- **Published split:** the CRF head scores 73.31 [72.28, 74.35] and 75.48
  [74.23, 76.72], +3.12 and +2.96 over plain MatSciBERT, paired by seed.
- **Span validity:** under strict IOB2 scoring the gain shrinks to +1.70
  [+0.21, +3.19] fine-grained and +1.07 [−0.82, +2.97] coarse-grained. Much
  of it comes from predicting only valid spans.
- **SOFC-Exp:** the CRF head improves on plain MatSciBERT again, +1.22
  [+0.50, +1.93] over 10 seeds (83.75 against 82.53); strict, +0.73
  [−0.04, +1.51]. The published MatSciBERT-CRF score (84.77;
  gupta2022matscibert) lies above our interval over seeds. On test
  articles cut into words as the training articles are (Section 2), the CRF
  head scores 84.61 [83.89, 85.34], an interval that covers the published
  score, and its gain over plain MatSciBERT falls to +0.17 [−0.37, +0.70]:
  on SOFC-Exp the CRF head mostly repairs the tokenization shift.
- **SOFC-Exp's slot types:** +1.68 [−0.12, +3.49] (71.88 against 70.20),
  not significant, and −0.06 [−1.70, +1.58] under strict scoring: plain
  MatSciBERT's invalid tags are mostly wrong entities there, so dropping
  them raises its score to 71.94. The published MatSciBERT-CRF score on
  SOFC-Slot (72.85, over 5 cross-validation splits) lies above our
  interval; the seed votes score 74.32.

**5.4 Seed votes and multi-task training (C5, C7).**
- **The seed vote** is the most reliable addition: +2.20 and +1.67 over the
  mean single CRF run on the random splits, +2.36 and +2.50 on unseen
  papers, every interval above 0.
- **Multi-task training** over both granularities is the one training
  change that holds: +1.83 [+0.32, +3.34] fine-grained on the random
  splits. It does not repeat on SOFC-Exp, where a slot head and an entity
  head stand to each other as MaterioMiner's two levels (Section 6.5).
- **The multi-task vote** improves on the seed vote coarse-grained on the
  random splits (+0.79 [+0.17, +1.42]) and fine-grained on unseen papers
  (+3.65 [+0.49, +6.82]). That is 2 of 12 vote comparisons, where about 0.3
  would pass by chance.
- **Where the gain comes from:** on unseen papers, fine-grained, multi-task
  training mostly buys precision (52.1 → 58.8) rather than recall
  (42.7 → 44.8; seeds 0–200, exploratory).

**5.5 What does not hold (C6, C11, C12, C14).** Under the 10-seed rule,
each of these fails:
- label semantics, as a model or a vote, on either protocol;
- self-training with 3 teachers on unseen papers (+1.74 [−1.16, +4.64]
  fine-grained);
- coarse-grained multi-task training (+0.71 [−0.15, +1.56]);
- document context (+1.14 [−0.94, +3.22] fine-grained; on SOFC-Exp +0.58
  [−0.26, +1.42]);
- multi-task self-training (+5.47 [−0.74, +11.68] fine-grained, the
  largest mean gain, which one of the four papers does not share;
  Section 6.1);
- earlier, class weights, domain-adaptive and in-domain pre-training, and
  other backbones.

Three more results:
- **DeBERTa-v3-large:** 435M parameters against MatSciBERT's 110M, it loses
  4.4 and 8.0 points on unseen papers, with similar validation scores on
  the training papers.
- **Label consistency within a paper:** adds +0.30 and +0.12 on
  MaterioMiner and scores −0.42 [−0.61, −0.22] on SOFC-Exp, whose annotators
  leave 14% of a term's repeated occurrences untagged. Whether "one sense
  per discourse" helps depends on how the annotation scheme treats repeated
  mentions.
- **Coarse-to-fine completion:** using the coarse-grained level to complete
  the fine-grained one adds +0.34 [−0.91, +1.59]; the entities it adds are
  right 39% of the time.

**5.6 How far a 3-seed design misleads (C8; Figure 2).** From the 120 seed
triples of the 10-seed runs:
- the 1.83-point multi-task gain passes in 41% of the triples (49 of 120);
- label semantics and self-training on unseen papers pass in 2–18%;
- up to 37% of triples give the sign opposite to the 10-seed difference;
- one split's micro F1 varies over seeds with an SD of 1.5–1.7 points on
  random splits and 2.2–2.5 on unseen papers.

The week-6 screen showed the same prospectively. Document context passed
the 3-seed screen with +2.66 [+0.91, +4.40] fine-grained. On the 7 seeds
the screen did not use it gains +0.49 [−3.10, +4.07], and one paper goes
from +4.29 to −2.59. A screen on a noisy estimate selects methods that drew
lucky seeds; only 21 of the 120 triples of document context pass.

**5.7 Where unseen papers break the model (C9, C10; Figure 3).**
- **Fine-grained,** the CRF head misses 36% of the gold entities outright
  on unseen papers (10% on random splits) and predicts 10,026 entities for
  13,280 gold ones. 41% of these misses have a class seen fewer than 10
  times in the three training papers.
- **Coarse-grained,** it misses only 6%, but 16% of the gold entities get
  the right span and the wrong type, and 21% a wrong boundary.
- **Macro F1:** a few rare coarse-grained types (SDE and MIP near 18 F1;
  PRP, MOD and CPR at 40–42, with 2–8 test entities per split) keep it near
  the published number.
- **Not a threshold problem (C15; Section 6.2):** the missed entities are
  not near the decision threshold. Lowering it adds entities that are right
  only 22% of the time fine-grained.

**5.8 Unseen papers are mostly unseen names (C17; Figure 5).** Following
augenstein2017generalisation, a gold test entity is *seen* if its exact
words are an entity of the same type in the run's training sentences and
*unseen* if they are no entity there; a third kind, seen with another type
only, holds 0.5–2% of the entities. Predictions are split the same way, so
recall, precision and F1 can be given per kind (exploratory; all seeds and
splits pooled).
- **The mix.** On random splits, 35% of the fine-grained test entities are
  unseen (coarse-grained 40%; on the published split 40% and 39%): random
  sentence splits put a paper's names on both sides, as babl2025random find
  for other corpora. On a held-out paper, 81% are unseen (coarse-grained
  79%), between 72% and 83% for each paper.
- **Unseen names in a new paper are harder.** Fine-grained, the CRF head
  finds 88% of the seen and 59% of the unseen entities of a random split,
  but 71% and 35% of those of a held-out paper (coarse-grained, 89% and 63%
  against 82% and 52%).
- **Both effects count.** The CRF head's fine-grained recall falls from 77.8
  on random splits to 40.6 on held-out papers. With the random splits' mix
  of seen and unseen names, its recall per kind on held-out papers would
  give 58.5. The mix and the lower recall within each kind each account
  for 40–60% of the gap, in either order of the decomposition. The same
  holds coarse-grained, where recall falls from 77.7 to 57.4.
- **Patterned names transfer, descriptive ones do not.** On held-out
  papers, the CRF head finds 76% of the unseen Value entities and 72% of
  the unseen Alloy and MaterialClass entities, whose names follow patterns
  (numbers with units, alloy designations). It finds 30% of the unseen
  StateDescriptor and 33% of the unseen Mechanism entities, and almost
  none of the unseen GrainBoundary, TestingMachine or StressAmplitude
  entities.
- **SOFC-Exp is easier.** 49% of its test entities are unseen, and the CRF
  head finds 82% of them (89% of the seen). Its articles share one topic,
  its 4 types are broad, and its most frequent type, VALUE, is found at
  95–97% whether seen or not.
- **MuLMS's held-out subfields are like MaterioMiner's held-out papers.**
  88% of their test entities are unseen (57% on random splits of the same
  size), and the CRF head finds 39% of them (67% of the seen). The larger
  share of unseen names accounts for about a third of its 22.5-point
  recall gap, the lower recall within each kind for the rest.
- **The gains are on unseen names.** Paired over the 4 held-out papers
  (fine-grained F1), every gain is significant only on unseen names:
  multi-task training +4.02 [+1.53, +6.51] (seen +2.88 [−2.70, +8.47]),
  its vote against the seed vote +3.60 [+2.32, +4.88], the context vote
  +2.68 [+0.84, +4.53], and multi-task self-training's screen (3 seeds)
  +6.44 [+2.25, +10.62]. Seen names are few in a new paper, so their
  intervals are wide. The CRF head's own gain over plain MatSciBERT lies on
  unseen names as well: +5.94 [+3.11, +8.76] on the published split
  (fine-grained) and +3.02 [+1.99, +4.05] on SOFC-Exp, against −0.47 and
  −0.52 on seen names. On the random splits, self-training gains +2.61
  [+0.50, +4.73] (fine-grained) and +2.26 [+0.92, +3.60] (coarse-grained)
  on unseen names and nothing on seen ones. Mention replacement on the
  multi-task model (Section 6.4) is the one exception: +7.10 [+6.43,
  +7.77] on unseen and +5.88 [+0.42, +11.34] on seen names.

A method for unseen papers therefore has to raise F1 on names the training
papers do not contain. That is where 81% of a new paper's entities are, and
where every gain measured so far lands.

**5.9 Found, but cut differently (C18).** seqeval counts a prediction only
if its span and type are exactly those of a gold entity. SemEval-2013 Task
9.1 (segurabedmar2013semeval) also scored looser criteria; we use them with
each gold entity matched at most once and an overlap counted in full
(exploratory):
- **Coarse-grained,** the CRF head scores 58.60 on held-out papers when
  span and type must match, 74.17 when only the span must, and 90.74
  [89.17, 92.32] when any overlap counts. On random splits it scores 76.96,
  83.91 and 93.38. The 18.4-point gap shrinks to 9.7 and then 2.6: on a new
  paper the model finds almost every entity, and loses mostly on where the
  boundaries go and which type applies.
- **Fine-grained,** the gap shrinks from 31.0 to 25.0 and 18.3 (overlap F1
  70.58 against 88.83). There the model misses entities outright (C10),
  most of them unseen names (Section 5.8).
- **Near misses.** On held-out papers, 31% of the fine-grained predictions
  overlap a gold entity but differ in span or type (12% on random splits).
  10.8% lie inside a longer gold entity: the new paper tags "multi-GB
  cracks", "EBSD data" and "smooth samples" where the model tags the noun.
  4.2% have a type that is an MMO ancestor or descendant of the gold one:
  "samples" is a TestSpecimen in one paper and a FatigueTestSpecimen in
  another.
- **SOFC-Exp:** 83.75 strict and 88.80 under overlap; types are almost
  never confused there (84.06 when only the span must match). Its slot
  types often are: 71.88 strict, 83.44 when only the span must match. 11.7% of
  the CRF head's slot-level predictions have a gold entity's span and
  another slot, against 0.3% of its entity-level predictions. Most of them
  give a material the wrong role in the cell: an electrolyte material
  becomes a cathode material 27 times per seed (of 120), an anode material
  7 times. The model finds the materials; their role is what it misses.
- **The gold annotations agree.** Only 35 fine-grained names are entities
  in two or more papers. 7 of them take different types, 6 of these a class
  and its MMO subclass (DataSet and MechanicalTestDataSet, TestSpecimen and
  FatigueTestSpecimen, MechanicalProperty and TensileYieldStrength).
  DataSet entities are 1.9 words long on average in one paper and 1.0–1.1
  in the others.

Part of what looks like a failure to generalise is a difference in
annotation conventions between papers, which a model trained on other
papers cannot learn. We report strict F1 for comparison with published
numbers and the looser criteria alongside.

**5.10 What one more paper is worth (C19; Figure 6).** The inner folds of Section
6.2 train the CRF head on two papers. On each held-out paper, training on
all three other papers instead adds +7.36 [+1.98, +12.74] F1 fine-grained
(39.02 to 46.38) and +8.90 [+5.90, +11.90] coarse-grained (50.37 to 59.27),
over 4 papers and seeds 0–200; every paper gains, from +3.2 to +11.6. A
third paper also brings more sentences. A pre-registered run (week 8,
part 2) separates the two by training on three papers cut to the size of
two. At the same number of sentences, the third paper adds +1.18 [+0.22,
+2.15] fine-grained (39.02 to 40.20) and +3.59 [+1.59, +5.58]
coarse-grained (50.37 to 53.96), and every paper gains. Its own sentences
add +6.17 [+1.10, +11.24] and +5.31 [+4.13, +6.50]. Diversity is thus about
a sixth of the gain fine-grained and two fifths coarse-grained. It matches the largest method gain measured on unseen
papers (multi-task self-training's screen, +6.06 fine-grained): for
materials NER on new papers, annotating more papers is the lever to beat.
An outside result points the same way. Potu et al. (potu2025ontology,
Fig. 2) train MatSciBERT on 47 other fatigue papers, annotated by
MaterioMiner's curator with the same classes, and report 66.74 F1 on
MaterioMiner, against 78.56 in distribution. Our models trained on three
of its papers score 46–52 on the fourth. The setups differ (no CRF, 28
classes, all 4 papers as the test set), but the gap to unseen papers
shrinks from 28–31 points with three training papers to about 12 with 47.
Their corpus cannot be redistributed. A method that makes better use of
the few annotated papers there are, and of unannotated ones, is what
remains.

## 6 Methods for unseen papers (pending)

Each method is decided under a protocol committed before its runs. On
MaterioMiner it is decided against the CRF head on the 4 held-out papers,
and its vote against the strongest system on unseen papers (the multi-task
vote fine-grained, the seed vote coarse-grained). A methods claim also
needs its gain to repeat on a second corpus: SOFC-Exp, and for mention
replacement also MuLMS's held-out subfields (Section 6.6).

**6.1 Multi-task self-training.** Three multi-task teachers, trained with
three seeds on the gold data, tag 40,000 sentences of open-access fatigue
papers once, at both levels.
- *Silver data:* a sentence becomes silver data for a level when the
  teachers agree on its tags and every tag's CRF marginal is high. The
  agreement and threshold are chosen, per level, from 5 fixed candidates
  on the validation sentences.
- *Students:* each student is a multi-task model trained on gold plus
  silver sentences, the silver ones at weight 0.5 (liang2020bond).
- *Screen:* +6.06 [+1.05, +11.06] fine-grained against the CRF head.
- *Result (10 seeds): no decision passes.* Against the CRF head it gains
  +5.47 [−0.74, +11.68] fine-grained, the largest mean gain of any method
  on unseen papers, and +0.90 [−1.18, +2.98] coarse-grained. Three papers
  gain 4–9 points (+8.86, +8.31, +4.21) and the fourth +0.51, so with 3
  degrees of freedom the interval reaches below 0. Its votes gain +1.54
  [−1.73, +4.82] over the multi-task vote and +1.08 [−1.40, +3.57] over
  the seed vote coarse-grained. On unseen names it gains +6.35 [−0.13,
  +12.84] fine-grained. Unlike document context's (Section 5.6), its
  screen held up: about +5.2 on the 7 seeds the screen did not see.
- *SOFC-Exp:* its two levels (entity and slot types) allow a repeat. The
  corpus is 40,000 sentences of 292 open-access SOFC articles, with
  SOFC-Exp's articles and any sentence sharing an 8-gram with them left out.
  It is cut into words with the tokenizer of SOFC-Exp's training articles.
  As SOFC-Exp annotates only sentences that describe experiments, a corpus
  sentence can be silver only if every teacher tags an experiment in it.
  This repeat was fixed before the MaterioMiner decisions, to run if one of
  them passed; none did, so it did not run.

**6.2 Recall calibration.** On unseen papers the CRF head predicts too
few entities (Section 5.7), so a bias D is added to the emission score of
every tag but O before Viterbi decoding:

    e'_t(y) = e_t(y) + D · [y ≠ O]

- *Why it can help:* raising D trades precision for recall. For a
  calibrated classifier the F1-optimal threshold is half the attainable F1
  (lipton2014optimal). A single weight on the O label was used before to
  trade precision for recall in CRF extractors (minkov2006ner).
- *Choosing D for MaterioMiner:* D is chosen per held-out paper P from a
  fixed grid (−1 to 3), without P's labels. The inner folds leave out P
  and one other paper Q, train on the remaining two and score Q. D_P
  maximises the mean F1 over the three Q and three inner seeds.
- *Choosing D for SOFC-Exp:* D is chosen per run on the validation
  articles, which are unseen articles.
- *What is new:* the choice of the bias for unseen documents, and its test
  under the 10-seed design.
- *Result (C15): no gain.* Against the CRF head on the 4 held-out papers
  it gives −0.59 [−1.75, +0.57] fine-grained and +0.17 [−0.14, +0.48]
  coarse-grained. Its votes do not beat the strongest systems: −3.53
  [−8.10, +1.05] against the multi-task vote, −0.09 [−0.75, +0.58] against
  the seed vote. On SOFC-Exp it gives −0.21 [−0.58, +0.17]. The inner folds
  chose D = +1.25, +1.25, −1 and +0.25 fine-grained, but each paper's own
  curve is flat: the best D per paper in hindsight would gain only +0.56
  on average (+0.18 coarse-grained), and its sign differs by paper.
- *Why (Figure 4):* the missed entities are not near the threshold.
  Pooled over the papers and seeds, the head finds 5,386 of 13,280
  fine-grained gold entities.
  - D = +0.25 adds 358 predictions, of which 22% are right. The break-even
    rate for F1 is F1/2 = 23%, against 53.7% for the predictions above
    the threshold.
  - D = +3, a factor of about 20 on the odds of a one-word entity,
    recovers 7% of the 7,894 misses, at 6 wrong entities per right one.
  - Coarse-grained and on SOFC-Exp the added entities are right 19% and
    21% of the time, below break-even (29% and 42%).

  The method assumed roughly calibrated probabilities. Under that
  assumption the entities just below the threshold would be right more
  often than F1's break-even rate; on unseen papers they are not. The
  recall gap comes from what the model scores, not how its scores are
  decoded.

**6.3 Mention replacement** (dai2020analysis). Three copies of each
training sentence with an entity, each mention replaced with probability
0.5 by a training mention of the same type. The model sees each type's
contexts with other names, which change from paper to paper: 81% of a new
paper's fine-grained entities are names the training papers lack, and the
gains that hold are gains on them (Section 5.8). On a
materials dataset the original study found gains for transformers mainly
with 50–150 training sentences, so a large gain is not expected; it is
screened with 3 seeds and decided with 10 only if the screen passes.
- *Screen:* +3.25 [−2.33, +8.82] fine-grained (per paper +2.85, +7.95,
  +2.72, −0.53) and +0.25 [−2.21, +2.72] coarse-grained against the CRF
  head; its first vote +5.00 [−0.50, +10.51] fine-grained over the seed
  vote. The fine-grained gain lies on unseen names (+3.48 [−0.18, +7.15]).
  It goes to the 10-seed decisions, on MaterioMiner and on SOFC-Exp;
  screens are optimistic (Section 5.6).
- *Result (C20): a gain on unseen papers at both granularities.* With 10
  seeds it improves on the CRF head by +2.77 [+1.51, +4.03] fine-grained
  and +1.28 [+0.06, +2.51] coarse-grained, and every held-out paper gains
  (+1.69 to +3.48 and +0.29 to +2.10). It is the only method of Section 6
  to pass this decision at both granularities.
- *Where the gain is:* on unseen names, +3.19 [+1.66, +4.72] fine-grained
  (seen +1.49 [−1.31, +4.30]); coarse-grained +1.58 [−0.02, +3.17].
- *Per type (exploratory):* coarse-grained, it gains on every rare type
  that pulls macro F1 down (SDE, MIP, PRP, MOD and CPR, +7.7 to +17.9),
  where multi-task training loses (−2.9 to −10.7). Over the 26
  coarse-grained types the two methods' changes correlate at −0.72;
  fine-grained at 0.20. Rare types have 2–9 test entities per paper, so
  these are indications, not findings.
- *Its vote* gains +3.61 [+1.71, +5.52] over the seed vote fine-grained, as
  much as the multi-task vote, but it does not beat the strongest systems:
  −0.04 [−2.09, +2.01] against the multi-task vote fine-grained, +2.26
  [−0.12, +4.64] against the seed vote coarse-grained.
- *SOFC-Exp:* −0.29 [−1.04, +0.47] against the CRF head, paired by seed.
  Its 568 training sentences are longer and hold more entities, and the
  original study's gains for transformers shrank as the training set grew.
  Size is not the reason, though: trained on 10 of the 26 training
  articles (156–248 sentences, as many entities as on MaterioMiner; 5
  draws, 10 seeds each), it gains +0.50 [−0.47, +1.46] over the CRF head,
  paired by draw. The 16 other articles are worth 3.13 [2.21, 4.04] points
  to the CRF head there.

  Mention replacement and multi-task training each improve a single model
  on unseen papers fine-grained, in different ways: one shows the model
  names in new contexts, the other a second level of types. Section 6.4
  tests whether the two add up.

**6.4 Mention replacement on the multi-task model.** As in Section 6.3,
three copies of each training sentence, but a copy replaces a
coarse-grained mention, in both heads' tags at once, by a training mention
of the same coarse- and fine-grained type, so both levels' tags stay valid.
It is decided against the multi-task model (does mention replacement add
to it?), its vote against the strongest systems, and on SOFC-Exp's two
levels against the CRF head.
- *Result (C21): the gains add up on MaterioMiner.* Against the multi-task
  model it gains +3.01 [+0.42, +5.60] fine-grained and +2.30 [+0.49,
  +4.10] coarse-grained over 10 seeds, and every held-out paper gains
  (+0.81 to +4.71 and +0.79 to +3.42). Against the CRF head it gains +6.73
  [+6.03, +7.43] fine-grained, the sum of its parts (+3.72 and +2.77), and
  +1.74 [−0.15, +3.62] coarse-grained, where multi-task training alone
  loses (−0.56).
- *Its vote* scores 55.33 fine-grained on unseen papers: +2.86 [+0.18,
  +5.54] over the multi-task vote and +6.51 [+3.91, +9.12] over the seed
  vote, the strongest fine-grained system on unseen papers. Coarse-grained
  it scores 62.75, +1.65 [−0.47, +3.78] over the seed vote, which does not
  pass.
- *Where the gain is:* fine-grained +7.10 [+6.43, +7.77] on unseen names
  and +5.88 [+0.42, +11.34] on seen ones, the first gain whose interval on
  seen names lies above 0. Per type (exploratory), 18 of the 27
  fine-grained types gain, most MicrostructureEntity (+33.2),
  FatigueLifetime (+14.7), MechanicalTestDataSet (+14.1) and Mechanism
  (+13.3); coarse-grained 21 of 27, among them six of the seven rare types
  on which multi-task training alone lost.
- *SOFC-Exp:* +0.69 [−1.18, +2.57] on the slot types and −0.19 [−1.27,
  +0.88] on the entity types against the CRF head, paired by seed. Neither
  part gained there on its own either.

  3 of its 6 decisions pass, where about 0.15 would by chance. The
  pre-registered criterion for a methods claim also needed the gain to
  repeat on SOFC-Exp at the matching level, so the combination is a result
  for training sets of a few papers. There it gains about as much as a
  third annotated paper (+7.36, Section 5.10). Mention replacement alone
  does not help on SOFC-Exp at MaterioMiner's training size either
  (Section 6.3), so the combination's gain is, for now, a MaterioMiner
  result.

**6.5 Multi-task training on SOFC-Exp's two levels.** Not a new method:
the repeat of C7 on the second dataset, where the slot types stand to the
entity types as MaterioMiner's fine level to its coarse one.
- *Result: no gain.* On the official split, over 10 seeds, multi-task
  training scores −0.78 [−2.39, +0.84] against the CRF head on the slot
  types (71.11 against 71.88) and −0.15 [−1.36, +1.05] on the entity types
  (83.60 against 83.75). Its votes do not differ from the seed votes
  (−0.86 [−3.67, +1.95] and +0.08 [−2.85, +3.01]).
- *Unseen names:* −1.52 [−3.65, +0.62] on the slot types, −0.45 [−2.27,
  +1.37] on the entity types, where on MaterioMiner's held-out papers it
  gained +4.02 [+1.53, +6.51].
- *Per slot:* its largest losses are on the anode (32.4 to 20.3, 26 test
  entities) and cathode materials (27.9 to 22.6, 37) and on open-circuit
  voltage (77.9 to 71.9, 25).

Multi-task training is therefore a MaterioMiner result. One possible
reason, not tested: a material slot is a role in the reported cell, which
the entity level does not encode, so the entity head has little to add
where the slot head errs (Section 5.9).

**6.6 Mention replacement on held-out subfields of a third corpus.**
Neither mention replacement nor its combination with multi-task training
gains on SOFC-Exp (Sections 6.3 and 6.4), whose 45 articles share one
topic. On its official split's test articles 49% of the names are unseen
and the CRF head finds 82% of them; on a held-out MaterioMiner paper 81%
are unseen and it finds 35% (Section 5.8). Mention replacement may help
only where unseen names are many and hard to find, a reading made after
SOFC-Exp's results. MuLMS (Section 2) tests it: a held-out subfield
differs in topic from the training documents, as a new MaterioMiner paper
does. MuLMS has one level of types, so only mention replacement is tested,
with the settings of Section 6.3.
- *Design:* each of the 7 subfields is held out in turn, and its documents
  are the test set (351–2,384 sentences). The model trains on 340
  sentences of the other subfields' documents and validates on 60, about
  MaterioMiner's training size (234–370 sentences per held-out paper). As
  a reference, the CRF head also runs on 5 random splits of the same size,
  testing on 2,037 sentences.
- *Decisions,* over the 7 subfields, each the mean of its 10 seeds (6
  degrees of freedom): mention replacement against the CRF head, and its
  votes against the CRF votes. If the first passes, mention replacement's
  gain on unseen documents holds on two corpora whose held-out documents
  differ in topic and on none whose articles share one: a methods claim
  scoped to that setting.
- *Result (C22): the single model does not pass, its votes do.* Mention
  replacement scores 50.76 against the CRF head's 48.66: +2.10 [−0.31,
  +4.51] over 10 seeds, paired by subfield. Six subfields gain
  (electrolysis +1.42, graphene +1.66, pemfc +2.28, semi +2.52, sofc
  +7.39, steel +0.15) and poly loses 0.73, so with 7 subfields the
  interval reaches below 0. Macro F1 gains +2.12 [+1.14, +3.09] and
  weighted F1 +2.68 [+0.32, +5.04] (reported). Its votes score 53.25
  against the CRF votes' 50.54: +2.71 [+0.22, +5.20], which passes.
- *Unseen names:* 88% of a held-out subfield's test names are unseen, and
  the CRF head finds 39% of them, as on a held-out MaterioMiner paper (81%
  and 35%; Section 5.8). Mention replacement gains +2.19 [−0.25, +4.63] on
  them (seen +1.30 [−1.21, +3.81]), its votes +2.92 [+0.41, +5.44].
- *What it means:* the condition named after SOFC-Exp's results holds on
  MuLMS, and the mean gain is close to MaterioMiner's fine-grained +2.77,
  but the single-model decision fails. Across the corpora the passes do
  not line up: the single model passes on MaterioMiner, while the vote
  against the seed vote passes on MuLMS but not coarse-grained on
  MaterioMiner (+2.26 [−0.12, +4.64]). By the pre-registered criterion,
  mention replacement and its combination with multi-task training stay
  results for MaterioMiner (C20, C21).

## 7 Limitations

- **Four papers.** The unseen-paper decisions have 3 degrees of freedom,
  so only large, consistent gains pass, and the per-paper results vary
  widely (Figure 1's intervals).
- **One second dataset, and its methods on one split.** The gap of
  Section 5.2 is measured on SOFC-Exp over 5 random sentence splits and 5
  folds of held-out articles, but every method on SOFC-Exp is compared on
  its official split, with intervals over seeds only. The folds alone
  range over 9 points (79.77 to 88.76), so one split of held-out articles
  is a noisy estimate of a method's effect on new articles.
- **MuLMS as converted.** A flat tagger needs one level, so our MuLMS runs
  keep the outermost entities (68.6% of them) and repair 276 damaged tokens
  (Section 2). Their scores therefore cannot be compared with the published
  MuLMS numbers. Its 7 held-out subfields give 6 degrees of freedom, and the
  subfields differ widely (the CRF head scores 32.49 to 56.82). So a gain
  of 2 points, as mention replacement's (Section 6.6), passes only if the
  subfields agree closely.
- **Methods tested on MaterioMiner only.** Self-training used a fatigue
  corpus, which fits MaterioMiner's domain only, and coarse-to-fine
  completion was tested before SOFC-Exp's slot level was converted; their
  results rest on MaterioMiner alone.
- **The released split** is undocumented. Its use reproduces the published
  numbers, but the authors have yet to confirm that it is theirs.
- **A deviation:** DeBERTa-v3-large did not fit a 15 GB GPU on SOFC-Exp at
  the protocol's batch size, so its planned SOFC-Exp comparison did not
  run.
- **Annotation conventions differ between papers** (Section 5.9). Strict
  F1 on a held-out paper also counts the spans and types its annotators
  chose differently, which no model trained on the other papers can learn.
  The decisions use strict F1 for comparability; the looser criteria are
  reported, not decided on.

## 8 Related work

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
(Section 6.5). SOFC-Exp (friedrich2020sofc) holds out whole articles in its
official split, and its test articles are cut into words differently from
its training articles (Section 2; swain2016chemdataextractor). MaterioMiner
(durmaz2024materiominer) types its entities with the classes of an
ontology and ships one split that its paper does not describe. Potu et al. (potu2025ontology) recognise ontology-conformal
materials entities and study a shift between two fatigue datasets; we
study the shift to an unseen paper of the same dataset.

**Splits and unseen mentions.** Whether random splits are good enough is
contested: Gorman and Bedrick (gorman2019standard) argue for several
random splits over one standard split, and Søgaard et al.
(sogaard2021random) find that random splits overestimate performance on
new data. For NER, random splits leak entity names from training into
test and inflate scores (babl2025random), and performance is far lower on
mentions unseen in training (augenstein2017generalisation). We measure
both on a materials corpus by leaving whole papers out (Sections 5.2 and
5.8). Looser matching criteria, as in SemEval-2013 Task 9.1
(segurabedmar2013semeval), separate entities that are missed from those
found with another span or type (Section 5.9).

**Seeds, significance and pre-registration.** Scores of neural taggers
vary with the random seed enough that a comparison of single runs can
mislead (reimers2017reporting); in fine-tuning, seeds change both the weight
initialisation and the data order (dodge2020finetuning), and benchmarks
should account for such variance (bouthillier2021accounting). We decide
with paired intervals over splits or papers (dror2018hitchhiker), each
side the mean of 10 seeds, under protocols committed before the runs
(vanmiltenburg2021preregistering). Section 5.6 shows what a 3-seed design
would have concluded on the same runs.

**Methods for unseen documents.** The additions we test come from work on
low-resource and cross-document NER: CRF output layers
(lafferty2001conditional; lample2016neural); label names as type
representations (ma2022label); document context (schweter2020flert);
label consistency within a document (gale1992one; krishnan2006effective);
mention replacement (dai2020analysis), whose gains for transformers
shrink as training data grows; larger encoders (he2023debertav3); and
self-training with a teacher and a student (liang2020bond). Du et al.
(du2021selftraining) retrieve unlabeled sentences from the web; our
self-training uses an in-domain corpus of open-access papers, one per
dataset, without the dataset's own papers or any sentence that shares a
word 8-gram with them.

**Decision thresholds.** For a calibrated classifier the F1-optimal
threshold is half the optimal F1 (lipton2014optimal). A single weight on
the O label already trades precision for recall in CRF extractors
(minkov2006ner), and CRFs can be trained for F-score directly
(suzuki2006training). Calibration degrades under dataset shift
(ovadia2019trust). Our recall calibration chooses the weight for unseen
documents on inner folds; Section 6.2 shows why it does not help there.
