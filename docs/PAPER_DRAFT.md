# Draft: materials NER on unseen papers

**Status.** A working draft of the sections that hold whichever paper this
becomes (`docs/PAPER.md`): a methods paper if a method passes its
pre-registered decisions and repeats on SOFC-Exp, otherwise an evaluation
paper. Sections 1–5 and 7 rest on finished, committed results. In
Section 6 (methods for unseen papers), recall calibration is decided (no
gain, C15). The others wait for their runs: stage 2 of multi-task
self-training, mention replacement and multi-task training on SOFC-Exp's
two levels (ROADMAP §4, weeks 6 and 7). The numbers come from `results/`
through the claims table of `docs/PAPER.md` (C1–C15). References are keys of `docs/references.bib`.

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
- *SOFC-Exp:* its two levels (entity and slot types) allow a repeat,
  given an in-domain SOFC corpus tokenised as SOFC-Exp is.

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
- *Why:* the missed entities are not near the threshold. Pooled over the
  papers and seeds, the head finds 5,386 of 13,280 fine-grained gold
  entities.
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
