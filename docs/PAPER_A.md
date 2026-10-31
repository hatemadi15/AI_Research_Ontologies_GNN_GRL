# Paper A (methods): mention replacement with more copies and harvested names improves materials NER on unseen documents

*A first draft, 30 October (week 15 begins; `docs/METHODS_PLAN.md` §7);
second pass the same day: every number of Sections 1 to 6 checked
against the results files (three wordings corrected), the mechanism
figure added. Every number is taken from the committed results files
named in `docs/PAPER.md`'s claims table (C2, C17, C19, C20, C22–C26) and
the week-12 to week-14 outcomes in `docs/ROADMAP.md` §4; a number in
square brackets is a 95% Student-t confidence interval of a mean paired
difference over the held-out units named. The ablations of 14.3 (the
corpus's size, the filter's threshold and agreement, the silver share)
ran on 31 October and are in Section 4.3 (Table 2). Paper B, the evaluation paper, is
`docs/PAPER_DRAFT.md`; this paper cites it for the protocol and the gap.*

## Abstract

A materials NER model that scores 80 F1 on random sentence splits of its
corpus scores 49 on a paper it has never seen, and 49 instead of 66 on a
materials subfield it has never seen: most of a new document's entity
names are new to training, and the tagger finds a third of them. We give
the tagger more names without annotating anything. Three CRF taggers
trained on the labeled sentences tag a large unlabeled corpus of materials
papers; a mention that the three tag alike and confidently joins the pool
of mention replacement, which trains on six copies of each labeled
sentence with its mentions swapped for others of the same type. Under a
pre-registered protocol with 10 seeds per unit and paired intervals over
held-out units, the method improves on the CRF head by +3.58 [+0.95,
+6.21] micro F1 pooled over 11 held-out units of two corpora, MuLMS's 7
subfields and MaterioMiner's 4 papers (on the papers alone +4.27 [+1.39,
+7.15] fine-grained and +1.77 [+0.67, +2.88] coarse-grained), and does not
improve on SOFC-Exp's held-out articles, where half the test names are
seen in training. Two pre-registered ablations bound the claim: against
mention replacement with the same six copies and no names the gain is
+0.80 [−0.66, +2.26], so most of the gain is the extra augmentation and
the names' share is not established; and a mixed corpus of six other
materials topics does as well as the held-out topic's own (−0.18 [−2.01,
+1.65]), so the names' source does not matter. Five ablations of the corpus size,
the filter and the names' share of the draws move the result by at most
0.2 points: a quarter of the corpus and the strictest filter suffice.
What the names do is raise
recall on unseen names (39.4 to 47.3 on MuLMS) at a cost in precision. A
second, smaller result holds on the same two corpora: propagating a
model's own confident labels within a document adds +0.82 [+0.18, +1.46]
on MuLMS and +0.12 [+0.01, +0.24] on coarse-grained MaterioMiner, and
hurts on SOFC-Exp. We report every attempt: the method's eight
pre-registered decisions, of which three pass.

## 1 Introduction

Annotated materials corpora are small and come from few documents.
MaterioMiner annotates 4 papers; SOFC-Exp 45 articles; MuLMS 50 articles
from 7 subfields. A model trained on such a corpus is meant for documents
the corpus does not contain, and the usual random sentence split does not
measure that: on MaterioMiner the seed vote of the CRF head scores 79.68
fine-grained over 5 random splits and 48.81 on a held-out paper; on
MuLMS the CRF head scores 65.72 over random splits of the same size and
48.66 on a held-out subfield; on SOFC-Exp, whose 45 articles share one
topic and one annotation scheme, 86.27 against 82.79 on held-out
articles (paper B, Section 5.2). The gap is mostly names. On a held-out
MaterioMiner paper 81% of the gold entities are names that no training
sentence annotates with that type (88% on a held-out MuLMS subfield, 46%
on SOFC-Exp's held-out articles), and the CRF head finds 35% of them on
MaterioMiner and 39% on MuLMS, against 71% and 67% of the seen ones.

This paper asks whether the names can be given to the tagger without
annotation. Unlabeled materials text is plentiful, and a corpus of
40,000 sentences of open-access papers holds 55–81% of a held-out
subfield's unseen test names. The method is simple: three CRF taggers
trained on the labeled sentences tag the corpus; the mentions they tag
alike, each with every tag's marginal probability at 0.9 or more, form a
lexicon; the lexicon enters training through mention replacement
(dai2020analysis), which copies each training sentence six times with its
mentions swapped for other mentions of the same type, drawn from the
training mentions and the lexicon together. Nothing silver is trained on;
the gold contexts stay, only the names change.

Our contribution is the test as much as the method. Every comparison
below was fixed in a protocol committed to the repository before its runs,
with 10 model seeds per held-out unit and a decision rule (the 95%
interval of the mean paired difference above 0) stated in advance; the
screen that chose the method among seven variants used seeds that the
decisions did not; and the two ablations that bound the claim were
pre-registered after the main decision passed. The results are:

- The method improves on the CRF head on held-out documents of two
  corpora: +3.58 [+0.95, +6.21] pooled over MuLMS's 7 held-out subfields
  and MaterioMiner's 4 held-out papers, with neither corpus's own
  interval below 0 (Section 4.1).
- Most of that gain is the extra augmentation. Against mention
  replacement with the same six copies and no names, the method gains
  +0.80 [−0.66, +2.26] over the same units and seeds: not established
  (Section 4.2).
- The names' source does not matter: a mixed corpus of the six other
  materials subfields does as well as the held-out subfield's own
  (−0.18 [−2.01, +1.65]; Section 4.2).
- The recipe is insensitive to its settings: five ablations of the corpus
  size, the filter and the names' share of the draws move the result by
  at most 0.2 points (Section 4.3).
- The names raise recall on unseen names and lower precision; on
  SOFC-Exp, where half the test names are seen, the method does not help
  (Sections 4.4 and 4.5).
- Document-level propagation of a model's own labels is a second,
  smaller gain on the same two corpora, under a point (Section 4.6).

Section 5 lists what did not work on the way, with the counts a reader
needs to weigh eight decisions of which three pass.

## 2 Data and protocol

Paper B describes the corpora, their conversion and the protocol in
full; this section gives what the method's results depend on.

**Corpora and held-out units.**
- *MaterioMiner* (durmaz2024materiominer): 476 sentences from 4 papers on
  fatigue of metals, annotated with the classes of an ontology at two
  granularities (fine-grained, the 27 most frequent classes, 1,328
  entities; coarse-grained, 27 parent types, 2,194 entities). Each paper
  is held out in turn; the model trains on the other three, with 15% of
  their sentences for validation.
- *MuLMS* (schrader2023mulms): 50 articles from seven materials subfields
  (electrolysis, graphene, pemfc, poly, semi, sofc, steel), 10,184
  sentences; we keep the outermost of its nested entities (31,952 of
  46,584, 13 types). Each subfield is held out in turn; the model trains
  on a fixed pool of 400 sentences of the other six (340 for training, 60
  for validation), MaterioMiner's size, and is tested on every sentence
  of the held-out subfield.
- *SOFC-Exp* (friedrich2020sofc): 876 sentences from 45 articles on solid
  oxide fuel cells, 4 entity types. Five folds of 9 held-out articles.

**Tagger.** MatSciBERT (gupta2022matscibert) with a CRF output layer
constrained to valid IOB2 transitions, the strongest single model of
paper B on every corpus; 10 model seeds per held-out unit (seed offsets 0
to 900). Micro F1 at the entity level (seqeval, conlleval mode).

**Decision rule.** A method improves on its reference when the 95%
Student-t interval of the mean paired micro-F1 difference lies above 0,
the pairs being the held-out units (7 subfields, 4 papers, 5 folds), each
side the mean of its seeds. With 4 papers the interval has 3 degrees of
freedom; with 7 subfields, 6. A pooled decision takes the units of two
corpora together and requires, in addition, that neither corpus's own
interval lie below 0. The protocols, their amendments and the dates are in
the repository (`docs/ROADMAP.md` §4, weeks 10 to 14).

**Unseen names.** A test entity is unseen when no training sentence
annotates its words with its type (augenstein2017generalisation); recall
and precision are reported by kind of name, pooled over seeds and units.

## 3 The method

**Teachers.** The three CRF taggers of seeds 0, 100 and 200, trained on
the fold's training sentences: the same runs as the reference's first
three seeds, so the teachers are the baseline, not a stronger model.

**The corpus.** For a held-out MuLMS subfield, 40,000 sentences of
open-access full texts found on Europe PMC by the subfield's title terms,
with MuLMS's identified articles excluded by identifier and every sentence
that shares a word 8-gram or a 50-character run with a MuLMS sentence
dropped. For MaterioMiner, 40,000 sentences of open-access papers on
fatigue, without the dataset's papers or any sentence sharing a word
8-gram with them. For SOFC-Exp, 40,000 sentences of 292 open-access SOFC
articles under the same exclusions, cut into words with the tokenizer of
its training articles. A topic corpus holds 55–81% of a held-out MuLMS
subfield's unseen test names (exact match; 77–90% of the unseen MAT
names), the fatigue corpus 49–65% of a held-out paper's, the SOFC corpus
36% of a fold's.

**The lexicon.** Each teacher tags the whole corpus with Viterbi and
returns, per word, the marginal probability of its tag. A mention joins
the lexicon when the three teachers tag the same span with the same type
and every tag of the span has a marginal of at least 0.9 under each
teacher, whatever the rest of the sentence. Per held-out subfield this
admits 8,328–50,869 mentions (4,380–13,395 distinct names), which hold
22.2% of the held-out subfield's unseen test names (7–36% per subfield):
about a third of what the corpus holds.

**Mention replacement with the lexicon.** Each training sentence with an
entity is copied six times; in a copy, each mention is replaced with
probability 0.5 by a mention of the same type drawn uniformly from the
pool, which holds the training mentions (about 1,400 on MuLMS, 4 per
sentence) and the lexicon's; copies that change nothing are dropped. The
pool is 85–97% lexicon, so most replacements are harvested names. The
student trains on the gold sentences and their copies with the same
recipe and seed as the reference; the silver sentences are never trained
on. Without a lexicon the student is mention replacement with six copies,
the control of Section 4.2; with three copies and no lexicon it is Dai and
Adel's method, which paper B tests on its own.

**What is recorded.** Every run stores its teachers' scores, the
lexicon's counts and the share of the unseen test names it holds, the
corpus's hash and the filter; no name is stored, and the corpora are
rebuilt from their reports.

## 4 Results

### 4.1 The pre-registered decisions

Table 1 lists the decisions on held-out documents in the order they were
taken (paper B's Table 6 holds the full list).

| Corpus, units | Comparison | Decision | Δ micro F1 [95% CI] | Verdict |
|---|---|---|---|---|
| MuLMS, 7 subfields (seeds 300–900) | method vs CRF head | week 12, 1 | +3.19 [−1.13, +7.51] | no |
| MuLMS, 7 subfields (seeds 300–900) | method vs mention replacement | week 12, 2 | +0.92 [−1.63, +3.48] | no |
| MaterioMiner fine, 4 papers | method vs CRF head | week 12, analogue | +4.27 [+1.39, +7.15] | passes |
| MaterioMiner coarse, 4 papers | method vs CRF head | week 12, analogue | +1.77 [+0.67, +2.88] | passes |
| MuLMS 7 + MaterioMiner fine 4, pooled | method vs CRF head | 13.2 | +3.58 [+0.95, +6.21], no corpus losing | passes |
| SOFC-Exp, 5 article folds | method vs CRF head | 13.4 | −0.04 [−1.21, +1.14] | no |
| MuLMS 7 + MaterioMiner fine 4, pooled | method vs 6 copies without names | 14.1 | +0.80 [−0.66, +2.26] | no |
| MuLMS, 7 subfields (seeds 300–900) | own-topic corpus vs mixed corpus | 14.2 | −0.18 [−2.01, +1.65] | no |

On MuLMS the method scores 51.84 over 10 seeds against the CRF head's
48.66 and mention replacement's 50.76; on the decision's seeds 6 of the
seven subfields gain against the CRF head (sofc by 11.9, poly
loses 2.4) and the 7-unit
interval, with 6 degrees of freedom and a standard deviation of 4.7
points across subfields, does not clear 0. On MaterioMiner every paper gains at both
granularities. Pooled over the 11 units, the pre-registered decision
13.2 passes with neither corpus's interval below 0 (MuLMS +3.18 [−1.30,
+7.66], MaterioMiner +4.27). On the 7 MuLMS seeds the screen did not see,
the pooled estimate is the same (+3.58 [+1.04, +6.13], post hoc). The
method's votes over three seeds gain +4.99 [+3.27, +6.72] over the CRF
head's votes fine-grained on MaterioMiner.

### 4.2 Is it the names? Is it the topic?

**The names against matched copies (14.1).** Mention replacement with six
copies and no lexicon, on the same units and seeds, scores +2.68 [+0.13,
+5.23] over the CRF head on MuLMS's decision seeds and +2.98 [+0.09,
+5.86] on MaterioMiner's fine-grained papers, pooled over the 11 units
+2.56 [+0.94, +4.18] (reported; its gain over three copies is +0.41
[−0.35, +1.17] and +0.20 [−1.47, +1.88]). Against it the method gains
+0.51 [−1.51, +2.53] on MuLMS (three subfields lose), +1.29 [−2.51,
+5.10] on MaterioMiner fine-grained (one paper loses), +0.70 [−1.60,
+3.00] coarse-grained, and +0.80 [−0.66, +2.26] pooled: the decision
fails. Of the +3.58 over the CRF head, about +2.6 is the extra
augmentation and about +0.8 the names, a share this design cannot
establish. The 3-seed screen had read the control at −0.20 against
mention replacement; on the decision's seeds it is +0.41, which is why a
screen decides nothing here.

**The topic's role (14.2).** For each held-out subfield we built a mixed
corpus of 40,000 sentences drawn equally from the six other subfields'
corpora, with the same source, pipeline and exclusions; it holds 49–70% of
the subfield's unseen test names against the topic corpus's 55–81%, and
its lexicon 18.3% of them against 22.2%. The method with the mixed corpus
scores 51.91 against 51.73 with the topic's own, −0.18 [−2.01, +1.65]
paired by subfield (two subfields better with the own corpus, five with
the mixed one); the mixed variant gains +3.38 [−0.89, +7.64] over the CRF
head and +0.70 [−1.19, +2.58] over the copy control. The decision fails:
no difference is detected, and a large unlabeled corpus of other
materials topics gives the same gain.

So the method the evidence supports is mention replacement with six
copies whose pool also holds names harvested from any large materials
corpus; its pre-registered gain is over the CRF head, and the names' own
share beyond the copies is small.

### 4.3 Ablations (reported, 3 seeds)

Each ablation changes one setting of the method and runs on MuLMS's 7
held-out subfields at the screen's seeds (0 to 200), against the method's
own runs on those seeds. With 3 seeds and 7 subfields these are readings,
not decisions, and they detect only a difference of about 2 points or
more. Table 2 gives them, with the lexicon each setting admits (distinct
names, summed over the subfields) and the share of the held-out subfields'
unseen test names in it.

| Setting (the method's value) | Distinct names | Unseen test names in the lexicon | Micro F1 | Δ vs the method [95% CI] | Δ vs mention replacement | Δ vs CRF head |
|---|---|---|---|---|---|---|
| the method: 40,000 sentences, threshold 0.9, 3 of 3 teachers, a uniform pool | 54,095 | 22.2% | 52.09 | | +1.45 [−1.35, +4.25] | +3.16 [−1.96, +8.28] |
| corpus: its first 10,000 sentences | 19,171 | 16.4% | 51.94 | −0.15 [−1.78, +1.48] | +1.30 [−1.14, +3.74] | +3.01 [−1.21, +7.23] |
| corpus: its first 20,000 sentences | 32,461 | 20.1% | 51.93 | −0.16 [−2.45, +2.14] | +1.30 [−0.56, +3.15] | +3.00 [−0.49, +6.49] |
| filter: marginal threshold 0.7 | 77,879 | 31.5% | 51.93 | −0.16 [−1.62, +1.31] | +1.29 [−0.95, +3.53] | +3.00 [−1.85, +7.85] |
| filter: 2 of 3 teachers agree | 78,604 | 28.8% | 52.20 | +0.11 [−1.58, +1.81] | +1.56 [+0.07, +3.05] | +3.27 [−0.85, +7.38] |
| pool: harvested names in half of the draws (85–97% uniform) | 54,095 | 22.2% | 52.18 | +0.09 [−1.63, +1.80] | +1.54 [−0.74, +3.82] | +3.24 [−0.73, +7.22] |

Nothing moves. The lexicon ranges from 19 thousand to 79 thousand distinct
names and from 16% to 32% of the unseen test names, and micro F1 stays
within 0.16 of the method's 52.09; every interval covers 0, the votes' too
(+0.20, −0.32, +0.34, +0.45 and +0.08), and per subfield the differences
run from −3.2 to +2.9, the spread of 3-seed means. This is what 14.1
predicts: if the names add under a point, the settings that govern how
many names enter cannot show an effect, and none does. For a practitioner
the readings say that a quarter of the corpus (10,000 sentences, a quarter
of the teachers' tagging) with three teachers agreeing at 0.9 does as well
as the full corpus and as the loosened filters, and that letting the
harvested names take 85 to 97% of the draws, as the uniform pool does, is
no better than half. What the ablations cannot say is whether a much
smaller or a much larger lexicon would change the result: they bracket
the method's settings by a factor of four in names, no more.

On the unseen names the settings do move the trade that Section 4.4
describes (exploratory; the screen's 3 seeds pooled over the subfields,
the method's own screen runs at 48.1% recall and 50.0 precision): the
smaller lexicons find fewer of them at higher precision (10,000 sentences
46.9% at 52.3; 20,000 sentences 46.2% at 53.0; the names in half of the
draws 46.2% at 53.7) and the larger ones more at lower precision (the
threshold 0.7 48.8% at 49.5; 2 of 3 teachers 48.3% at 50.6), while F1 on
the unseen names stays between 49.1 and 49.7. More names buy recall and
cost precision in the same measure, which is why the settings do not show
in F1.

### 4.4 What the names do

By kind of name, pooled over the 70 runs of MuLMS's held-out subfields
(Table 3 and Figure 1, `docs/figures/fig7_names.png`, which puts each
system's recall and precision on the unseen names on one plane with
iso-F1 curves, for MuLMS and for MaterioMiner's papers at both
granularities): the CRF head finds 39.4% of the unseen names at 55.8%
precision; mention replacement 43.2% at 54.3; six copies without names
43.7% at 53.9; the method 47.3% at 50.9. The names raise recall on unseen
names by 3.6 points over the control and lower precision by 3.0, a net
+0.7 F1 on unseen names (48.3 to 49.0) and +0.2 on seen ones (71.0 to
71.2). With the mixed corpus the figures are the same (46.7% at 52.4).
On MaterioMiner's papers, fine-grained, the names add to both recall and
precision on unseen names (39.7 to 41.1 and 49.2 to 53.2 over the
control; F1 43.9 to 46.4), and coarse-grained a little of each (53.0 to
54.2 and 52.4 to 53.2; F1 52.7 to 53.7). The mechanism is the one
intended, more unseen names found, and its size is what 14.1 measured.

| MuLMS, unseen names (88% of the test entities) | recall | precision | F1 |
|---|---|---|---|
| CRF head | 39.4 | 55.8 | 46.2 |
| mention replacement, 3 copies | 43.2 | 54.3 | 48.1 |
| 6 copies, no names | 43.7 | 53.9 | 48.3 |
| 6 copies and the topic corpus's names (the method) | 47.3 | 50.9 | 49.0 |
| 6 copies and a mixed corpus's names | 46.7 | 52.4 | 49.4 |

### 4.5 Where it does not help: SOFC-Exp

On SOFC-Exp's five folds of held-out articles the method with the SOFC
corpus scores 82.75 against the CRF head's 82.79, −0.04 [−1.21, +1.14]
(decision 13.4, no). It gains +0.45 [+0.04, +0.86] over mention
replacement on every fold, but mention replacement itself is below the CRF
head there (−0.48 [−1.36, +0.39], decision 13.3a), as on the official
split. Only 46% of a fold's test names are unseen (81% fine-grained and 79%
coarse-grained on MaterioMiner's held-out papers, 88% on MuLMS's
subfields) and the corpus holds 36% of them; the method needs test names
that training has not seen. Pooled over the 16 units of the three corpora
the estimate is +2.45 [+0.50, +4.40] (reported); the claim is the
two-corpus one, with SOFC-Exp as its boundary condition.

### 4.6 A second result: label propagation within a document

A post-processing rule on the saved predictions (paper B, Section 6.8):
a term that a model tags in most of its occurrences in a document is
tagged in the rest. Pre-registered on the CRF head's 10-seed predictions,
it gains +0.82 [+0.18, +1.46] on MuLMS's held-out subfields (every
subfield gaining; 1,213 entities added per seed) and +0.12 [+0.01, +0.24]
on coarse-grained MaterioMiner, and loses −0.44 [−0.72, −0.16] on
SOFC-Exp's article folds, where the annotators leave 14% of a tagged
term's occurrences untagged. It adds nothing to mention replacement
(+0.57 [−0.12, +1.25] on MuLMS). Two of its ten decisions pass, on two
corpora: a scoped gain under a point, with a boundary condition the
annotation convention explains.

## 5 What did not work, and the count

The method was reached through three protocols on MuLMS's held-out
subfields and their MaterioMiner analogues, each committed before its
runs, each failing or passing as recorded:

- *Mention replacement alone* (week 10): +2.10 [−0.31, +4.51] against the
  CRF head on MuLMS, no; its votes +2.71 [+0.22, +5.20], passes. On
  MaterioMiner it had passed at both granularities (+2.77 and +1.28).
- *Names from the other subfields' unlabeled sentences* (week 11): the
  three teachers tag the labeled subfields' sentences outside the pool,
  whose names hold 4.4% of a held-out subfield's unseen test names:
  +2.04 [−0.85, +4.92] against the CRF head and −0.06 [−1.09, +0.96]
  against mention replacement; on MaterioMiner, with the fatigue corpus
  and a sentence-level filter, +0.23 and +0.61 against mention
  replacement. Four decisions, none passing: the filter, not the corpus,
  was the bottleneck (2–9% of the unseen names admitted).
- *Seven variants on the topic corpora* (week 12, a 3-seed screen against
  mention replacement): the sentence-level filter +0.80, the
  mention-level filter with three copies +0.97 and with six +1.45 (the
  winner), six copies without names −0.20, self-training on the corpus's
  silver sentences +0.16, and the lexicon as a gazetteer match feature of
  the CRF head −1.73 (mention filter) and −1.85 (sentence filter). The
  winner's two MuLMS decisions on new seeds failed (Table 1); its
  MaterioMiner analogue and the pooled decision passed.
- *The lexicon as a dictionary* (a CPU probe, no decision): applied to
  the CRF head's predictions as a rule, it helps where the head's recall
  trails its precision by 15 points or more (sofc +5.9, pemfc +4.0) and
  hurts elsewhere (steel −10.0, graphene −5.1), −1.0 on average.

In all, the names method took eight pre-registered decisions in weeks 12
to 14, of which three pass (the MaterioMiner analogue at both
granularities and the pooled decision); with weeks 10 and 11, fourteen
decisions on this route, four passing. The ablations of Sections 4.2 and 4.3
were registered after the pooled decision passed and before their runs.

## 6 Limitations

- **Few units.** Seven subfields and four papers give 6 and 3 degrees of
  freedom; the subfields differ by 24 points under the CRF head (32.49 to
  56.82), so a 3-point gain passes on MuLMS only if the subfields agree.
  The pooled decision is what the design could afford, and the claim is
  scoped to it.
- **The names' share is not established.** The method's pre-registered
  gain is over the CRF head; against the matched control it is +0.80
  [−0.66, +2.26]. A reader who wants "names help" should read "names add
  little to more augmentation, measurably on recall of unseen names".
- **The ablations are 3-seed readings** on 7 subfields: they detect a
  difference of about 2 points or more, and what they show is the absence
  of any effect of that size, not the equality of the settings.
- **MuLMS as converted** keeps the outermost entities and repairs damaged
  tokens; its scores cannot be compared with the published nested ones.
- **One corpus per dataset.** The corpora are open-access papers found by
  title terms; a corpus of another source may hold other names.
- **SOFC-Exp's folds** differ by 9 points (79.77 to 88.76 under the CRF
  head); a method's effect there is measured with little power.

## 7 Related work

*(Cut from paper B's Section 8; the novelty paragraph of
`docs/METHODS_PLAN.md` §2 gives the sources.)* Harvesting a gazetteer
from unlabeled text with a tagger is old: Kozareva (2006) generated
gazetteers from unlabeled news by patterns and added examples two
classifiers agreed on; Zafarian and Asghari (2019) decoded an unlabeled
corpus with a supervised tagger, kept the confident entities as a
gazetteer and retrained with it as a feature. Song et al. (2020) found
Wikidata match features worth +0.5 F1 on OntoNotes and that replacing
annotated entities with gazetteer entries gave no significant gain where
test names were not the bottleneck. Dai and Adel (2020) replace mentions
with the training set's own; Amalvy et al. (2022, 2023) adapt a tagger to
fantasy novels with predefined lists of target names and gain recall at a
precision cost, as we do on MuLMS; COSINER picks the replacement that fits
the context; Torres et al. (2024) find mention replacement helps small
training sets. On the teacher side, BOND (Liang et al., 2020) self-trains
on dictionary-matched labels, Wu et al. (2020) train a student on
unlabeled target text from several teachers, and Qu et al. (2023) use
teachers' agreement in distant supervision; GEMNET gates gazetteer
features against the context. None harvests the lexicon of an unseen
topic from that topic's unlabeled text by the agreement of several
teachers and injects it through mention replacement with the topic held
out, and our ablation says the topic need not be the held-out one. For
materials NER, MatSciBERT (Gupta et al., 2022), MuLMS (Schrader et al.,
2023), SOFC-Exp (Friedrich et al., 2020), MaterioMiner (Durmaz et al.,
2024) and Potu et al. (2025), who study a shift between two fatigue
datasets with the same classes, are the points of comparison; paper B
places our baselines against the published numbers.

## 8 Conclusion

Three CRF taggers can harvest a usable lexicon from unlabeled materials
text, and mention replacement can put it into training without training
on silver labels. Under a pre-registered protocol the combination improves
on the CRF head on held-out documents of two corpora, by +3.58 pooled;
its ablations say that most of the gain is mention replacement with more
copies, that the names add recall on unseen names at a cost in precision,
that the names may come from any large materials corpus, and that a
quarter of the corpus with the strictest filter does as well. On a corpus
whose test names are mostly seen, nothing helps. We think the honest
summary is useful to practitioners with a few annotated papers: use
mention replacement with more copies; add harvested names if recall on
new names is what you need; and measure on held-out documents, with seeds
and intervals, because random splits will not show any of this.

## Reproducibility

The repository holds the protocols (`docs/ROADMAP.md` §4), every results
file with its seeds and environment (`results/gpu/`), the analyses
(`src/graph/ner_seed_analysis.py`, `ner_mentions.py`), the corpus reports
with hashes (`results/topic_corpora/`, `results/fatigue_corpus.json`,
`results/sofc_corpus.json`) and the Kaggle batches that produced each
file (`src/graph/kaggle_jobs.py`); the corpora rebuild from their reports.
Runs used Kaggle T4 GPUs with deterministic CUDA, a frozen commit per
batch and a pinned image; each session first repeated an earlier run, and
every repeat was identical.
