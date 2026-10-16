# Draft: materials NER on unseen papers

**Status.** A working draft of the sections that hold whichever paper this
becomes (`docs/PAPER.md`): a methods paper if a method passes its
pre-registered decisions and repeats on SOFC-Exp, otherwise an evaluation
paper. Sections 1–5 and 7 rest on finished, committed results. In
Section 6 (methods for unseen papers), recall calibration is decided (no
gain, C15). The others wait for their runs: stage 2 of multi-task
self-training, mention replacement and multi-task training on SOFC-Exp's
two levels (ROADMAP §4, weeks 6 and 7). The numbers come from `results/`
through the claims table of `docs/PAPER.md` (C1–C18). References are keys of `docs/references.bib`.

---

## Abstract (placeholder)

To be written once Section 6 is decided. The evaluation findings it will
carry either way:
- the published MaterioMiner numbers reproduce on the released split;
- a model trained on three papers loses 18–31 F1 points on the fourth;
- of the additions tested under a pre-registered 10-seed design, only the
  CRF head, seed votes and multi-task training hold;
- a 3-seed design passes a real 1.8-point gain in 41% of seed triples, and
  let through a 2.7-point gain that 10 seeds shrank to 1.1;
- the entities a model misses on unseen papers lie far below its decision
  threshold, so shifting the threshold does not recover them.
- 81% of a new paper's fine-grained entities are names that no training
  paper annotates (35% on random splits), the model finds 35% of them, and
  every gain that holds on unseen papers is a gain on these names.
- coarse-grained, the model finds almost every entity of a new paper (90.7
  F1 when any overlap counts, 93.4 on random splits); most of the strict
  gap lies in span boundaries and types that the papers' annotators set
  differently.

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

We ask four questions on MaterioMiner, with a second dataset, SOFC-Exp
(friedrich2020sofc), where possible:
1. Do the published numbers reproduce, and on which split and scoring?
2. How much is lost on a paper the model has never seen?
3. Which common additions improve materials NER on unseen papers once
   seeds are accounted for?
4. How far does a small-seed design mislead?

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
  ones included, carries this. What it costs is measured by scoring the
  same models on the test articles cut as the training articles are (week
  7, part 2e).

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
  - multi-task self-training and recall calibration (Section 6).

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

**5.3 The CRF head, and a second dataset (C3, C4, C13).**
- **Published split:** the CRF head scores 73.31 [72.28, 74.35] and 75.48
  [74.23, 76.72], +3.12 and +2.96 over plain MatSciBERT, paired by seed.
- **Span validity:** under strict IOB2 scoring the gain shrinks to +1.70
  [+0.21, +3.19] fine-grained and +1.07 [−0.82, +2.97] coarse-grained. Much
  of it comes from predicting only valid spans.
- **SOFC-Exp:** the CRF head improves on plain MatSciBERT again, +1.22
  [+0.50, +1.93] over 10 seeds (83.75 against 82.53). The published
  MatSciBERT-CRF score (84.77; gupta2022matscibert) lies above our interval
  over seeds.

**5.4 Seed votes and multi-task training (C5, C7).**
- **The seed vote** is the most reliable addition: +2.20 and +1.67 over the
  mean single CRF run on the random splits, +2.36 and +2.50 on unseen
  papers, every interval above 0.
- **Multi-task training** over both granularities is the one training
  change that holds: +1.83 [+0.32, +3.34] fine-grained on the random
  splits.
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
- document context (+1.14 [−0.94, +3.22] fine-grained);
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
  on unseen names and nothing on seen ones.

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
  never confused there (84.06 when only the span must match).
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

## 6 Methods for unseen papers (pending)

Each method is decided under a protocol committed before its runs. On
MaterioMiner it is decided against the CRF head on the 4 held-out papers,
and its vote against the strongest system on unseen papers (the multi-task
vote fine-grained, the seed vote coarse-grained). A methods claim also
needs its gain to repeat on SOFC-Exp.

**6.1 Multi-task self-training.** Three multi-task teachers, trained with
three seeds on the gold data, tag 40,000 sentences of open-access fatigue
papers once, at both levels.
- *Silver data:* a sentence becomes silver data for a level when the
  teachers agree on its tags and every tag's CRF marginal is high. The
  agreement and threshold are chosen, per level, from 5 fixed candidates
  on the validation sentences.
- *Students:* each student is a multi-task model trained on gold plus
  silver sentences, the silver ones at weight 0.5 (liang2020bond).
- *Status:* it passed the week-6 screen with +6.06 [+1.05, +11.06]
  fine-grained against the CRF head; its 10-seed decisions are next.
- *SOFC-Exp:* its two levels (entity and slot types) allow a repeat. The
  corpus is 40,000 sentences of 292 open-access SOFC articles, with
  SOFC-Exp's articles and any sentence sharing an 8-gram with them left out.
  It is cut into words with the tokenizer of SOFC-Exp's training articles.
  As SOFC-Exp annotates only sentences that describe experiments, a corpus
  sentence can be silver only if every teacher tags an experiment in it.
  This repeat is fixed before the MaterioMiner decisions, and runs if one
  of them passes.

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
contexts with other names, which change from paper to paper. On a
materials dataset the original study found gains for transformers mainly
with 50–150 training sentences, so a large gain is not expected; it is
screened with 3 seeds and decided with 10 only if the screen passes.

**6.4 Multi-task training on SOFC-Exp's two levels.** Not a new method:
the repeat of C7 on the second dataset, where the slot types stand to the
entity types as MaterioMiner's fine level to its coarse one.

## 7 Limitations

- **Four papers.** The unseen-paper decisions have 3 degrees of freedom,
  so only large, consistent gains pass, and the per-paper results vary
  widely (Figure 1's intervals).
- **One second dataset with one split.** On SOFC-Exp the intervals are
  over seeds only.
- **Methods tested on MaterioMiner only.** Self-training used a fatigue
  corpus, which fits MaterioMiner's domain only, and coarse-to-fine
  completion was tested before SOFC-Exp's slot level was converted; their
  results rest on MaterioMiner alone.
- **The released split** is undocumented. Its use reproduces the published
  numbers, but the authors have yet to confirm that it is theirs.
- **A deviation:** DeBERTa-v3-large did not fit a 15 GB GPU on SOFC-Exp at
  the protocol's batch size, so its planned SOFC-Exp comparison did not
  run.
