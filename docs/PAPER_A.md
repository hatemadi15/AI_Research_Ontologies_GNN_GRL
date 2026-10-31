# Harvested names add little to mention replacement on held-out materials documents: a pre-registered test

*Repository draft, revised 31 October after a first adversarial review round
(`docs/ROADMAP.md`, week 15). Every number is taken from the committed
results files named in `docs/PAPER.md` (C2, C17, C19, C20, C22–C26),
`results/gpu/ner_robustness.json` and the generated tables in
`docs/tables/`; a number in square brackets is a 95% Student-t confidence
interval of a mean paired difference over the held-out units named.
Citation keys refer to `docs/references.bib`. Section numbering follows a
conference layout; the mapping to a journal's Introduction, Results,
Discussion and Methods is in `docs/REPRODUCIBILITY.md`.*

## Abstract

Materials entity taggers are trained on a few annotated papers and scored
on sentences from those papers, which hides how they read a new one: our
baseline scores 77 F1 on random splits of MaterioMiner and 46 on a
held-out paper, because 81% of a new paper's entity names never occur in
training. We ask whether unlabeled papers can supply those names. Three
copies of the baseline tagger label 40,000 open-access sentences; the
names they agree on confidently join the pool of mention replacement,
which swaps the names in six copies of each training sentence. Under
protocols fixed before the runs, with 7–10 seeds per unit, the recipe
gains 3.6 F1 points [1.0, 6.1] over the baseline across 11 held-out units
of two corpora. But
mention replacement without any harvested name, the published method we
started from, already gains 2.5 [1.0, 3.9] on the same units; the extra
copies add 0.3 [−0.3, 0.9] and the names 0.8 [−0.7, 2.3], neither
established, in a design that detects 2 points. The names buy
recall on unseen names with precision, their gain tracks the baseline's
weakness rather than the lexicon's coverage, and neither their source nor
the filter's settings changes the result detectably. Where test names are
mostly seen, nothing helps. We report the eight pre-registered decisions,
their amendments and a post hoc robustness analysis. Mention replacement
is worth using on new documents; harvesting names from unlabeled text, at
twenty times the cost, is not yet shown to add to it.


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
MaterioMiner and 39% on MuLMS, against 71% and 67% of the names it has
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

Every comparison was fixed in a protocol committed to the repository
before its runs (vanmiltenburg2021preregistering), with 7 to 10 model
seeds per held-out unit and a decision rule stated in advance (the 95%
interval of the mean paired difference above 0); the screen that chose
the recipe among seven variants used three seeds that the per-corpus
decisions did not, and the two decisions that isolate the names'
contribution were registered after the main decision passed and before
their runs. Section 3 records the one deviation and the one revision of
the criterion. The results are:

- The recipe improves on the CRF head on held-out documents: +3.58
  [+1.04, +6.13] pooled over MuLMS's 7 held-out subfields and
  MaterioMiner's 4 held-out papers, 10 of 11 units positive, a sign-flip
  permutation p of 0.007, every leave-one-unit-out interval above 0
  (Section 5.1 and 5.2). MuLMS alone does not pass (+3.19 [−1.13, +7.51]).
- That gain is not the names'. Plain mention replacement with three copies
  and no lexicon gains +2.46 [+1.00, +3.91] over the CRF head on the same
  units; the three extra copies add +0.33 [−0.25, +0.92] and the harvested
  names +0.80 [−0.66, +2.26] (the pre-registered decision on the names,
  Section 5.2), in a design that detects a difference of about 2 points.
- Neither the names' source nor the recipe's settings changes the result
  detectably: a mixed corpus of six other materials topics gives −0.18
  [−2.01, +1.65] against the held-out topic's own (pre-registered), and
  five settings of corpus size, filter and share move F1 by at most 0.2
  points (Section 5.3).
- The names raise recall on unseen names and lower precision, and their
  gain per subfield tracks how weak the baseline is there (r = −0.76), not
  how many unseen names the lexicon holds (r = −0.71) (Section 5.4). On
  SOFC-Exp, where half the test names are seen, nothing helps
  (Section 5.5).

Section 6 lists what else was tried on the way and the count: eight
pre-registered decisions on the names route in weeks 12 to 14, three
passing, of which the three passes share one corpus's four papers.

## 2 Data

Table D1 gives the three corpora. All three are released under open
licences; our conversions and the held-out protocol are in the
repository.

| Corpus | Documents, sentences | Annotation | Held-out unit (test size) | Training per unit |
|---|---|---|---|---|
| MaterioMiner (durmaz2024materiominer), release v1.0.1, CC BY 4.0 | 4 papers on fatigue of metals (41, 81, 201, 153 sentences; DOIs 10.1007/s11340-021-00758-x, 10.1038/ncomms7164, 10.1038/s41467-020-19071-7, 10.3390/met10111458) | the classes of the MMO ontology at two levels: fine-grained, the 27 most frequent classes (1,328 entities; e.g. Alloy, Value, Mechanism, StateDescriptor, FatigueTestSpecimen, GrainBoundary); coarse-grained, 27 parent types (2,194 entities) | one paper (41–201 sentences) | the other three papers (234–370 sentences), 15% of them for validation |
| MuLMS (schrader2023mulms), Hugging Face `timo-pierre-schrader/MuLMS` at a pinned revision, CC BY-SA 4.0 | 50 articles from 7 materials subfields, 10,184 sentences: water electrolysis (4 articles, 992 sentences), graphene (2, 351), polymer-electrolyte-membrane fuel cells (8, 1,638), polymers (11, 2,384), semiconductors (11, 2,040), solid oxide fuel cells (4, 928), steel (10, 1,851) | 13 types; we keep the outermost of its nested entities, 31,952 of 46,584: MAT 12,647, VALUE 4,628, PROPERTY 3,880, FORM 3,180, MEASUREMENT 2,106, CITE 1,704, SAMPLE 1,391, TECHNIQUE 920, DEV 757, INSTRUMENT 370, NUM 260, UNIT 91, RANGE 18 | one subfield, every sentence of its articles (351–2,384) | a fixed pool of 400 sentences of the other six subfields (340 training, 60 validation), MaterioMiner's size by design |
| SOFC-Exp (friedrich2020sofc), `boschresearch/sofc-exp_textmining_resources`, CC BY 4.0 | 45 articles on solid oxide fuel cells; 876 sentences carry entity annotation (the ones that describe experiments) | 4 types: MATERIAL, VALUE, DEVICE, EXPERIMENT | 9 articles, one of 5 folds (137–248 sentences) | the other 36 articles (534–628 sentences, 94–111 for validation) |

Three details matter below. On MuLMS a flat tagger needs one level, so
31.4% of the entities are dropped with the conversion (14,632 of 46,584:
nested in a kept entity, sharing its span or crossing it), and text
extraction had damaged 276 tokens, which we repair; our MuLMS scores cannot be compared with the published nested
ones. On MuLMS the training pool is capped at 400 sentences on purpose,
so that the three corpora test the same regime, a few hundred annotated
sentences; the six other subfields hold about 8,000 labeled sentences
that no run here uses (Limitations). SOFC-Exp's official test articles
are tokenized differently from its training articles; the article folds
used here cut every article with the training articles' tokenizer.

**Unseen names.** A test entity is unseen when no training sentence
annotates its words with its type (augenstein2017generalisation); recall
and precision are reported by kind of name, pooled over seeds and units.

## 3 Evaluation protocol

**Seeds and scoring.** Every system has 10 model seeds per held-out unit
(seed offsets 0 to 900), except where a protocol reserved the screen's 3
seeds (0 to 200) for selection and decided on the other 7 (300 to 900).
Micro F1 at the entity level, exact span and type (seqeval in its default
mode, as the datasets' own code scores), in percentage points.

**Decision rule.** A method improves on its reference when the 95%
Student-t interval of the mean paired micro-F1 difference lies above 0,
the pairs being the held-out units (7 subfields, 4 papers, 5 folds), each
side the mean of its seeds. With 4 papers the interval has 3 degrees of
freedom; with 7 subfields, 6. A pooled decision takes the units of two
corpora together (11 units, 10 degrees of freedom) and requires, in
addition, that neither corpus's own interval lie below 0, a guard that
trips only on a significant loss. Each decision is one-sided at 2.5%, so
of 8 decisions about 0.2 would pass if nothing worked; the decisions are
not independent (Section 5.1). A comparison labelled *reported* was fixed
in a protocol but carries no decision; *post hoc* means computed after
its results were known. Differences the design can detect: with the
observed spread, a pooled 11-unit comparison detects about 2.0 points at
80% power, a 7-subfield comparison about 2.8, a 3-seed 7-subfield reading
2.0 to 3.2.

**Pre-registration, and what changed.** Each protocol was committed to
the repository before its runs, with its decisions, the comparisons
reported without a decision and the expected number of false passes;
amendments are marked with their reason and time (Appendix A lists the
commits). Two points must be disclosed. First, the criterion for a
methods claim was revised on 20 October, after a 3-seed screen on MuLMS
and before any decision run: the week-12 protocol had required a pass
against the CRF head on MuLMS alone plus a pass on MaterioMiner, and had
gated the MaterioMiner runs on the MuLMS pass; the week-13 protocol added
the pooled decision over both corpora as a second route to the claim,
ran the MaterioMiner analogue regardless, and lifted the plan's stop rule
("no fourth protocol on MuLMS"). Under the week-12 criterion the recipe
has no claim, since the MuLMS decision failed; under the revised one it
passes. Second, the pooled decision's protocol text named MuLMS's decision
batch (seeds 300 to 900) as the source of its units and, in the same
paragraph, "each unit its 10-seed mean"; the analysis as executed took
the 10-seed means, which include the screen's three seeds. We report both
readings (they agree, +3.58 either way) and use the 7-seed one, which the
screen never saw, as the primary estimate in the text. The names-only and
source decisions of week 14 used the 7 decision seeds on MuLMS; their
protocol was committed two minutes after the first batch was pushed and
five hours before any result arrived.

**Runs.** Kaggle T4 GPUs with deterministic CUDA, one frozen commit per
batch, a pinned image and pinned libraries; model snapshots pinned by
commit. Each GPU session first repeated one earlier run, and every
repeat over weeks 4 to 14 was identical. Hyperparameters and compute are
in Appendix B.

## 4 Systems

Figure 1 (`docs/figures/fig8_method.png`) sketches the recipe; the four
systems compared throughout are:

- **CRF head.** MatSciBERT (gupta2022matscibert) with a token classifier on
  the first sub-word of each word and a linear-chain CRF over the words
  with IOB2 transition constraints, trained by likelihood and decoded with
  Viterbi (lafferty2001conditional; lample2016neural). AdamW at 5e-5 with
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
- **MR6.** The same with six copies: the matched control for the names.
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
    run); 290–370 articles per corpus (Appendix C). A topic corpus holds
    55–81% of a held-out MuLMS subfield's unseen test names, the fatigue
    corpus 49–62% of a held-out paper's fine-grained (49–65% of its
    coarse-grained), the SOFC corpus 36% of a fold's.
  - *Lexicon.* Each teacher tags the whole corpus and returns, per word,
    the forward–backward marginal probability of its Viterbi tag. A
    mention joins the lexicon when the three teachers tag the same span
    with the same type and every tag of the span has a marginal of at
    least 0.9 under each teacher, whatever the rest of the sentence. Per
    held-out MuLMS subfield this admits 8,328–50,869 mentions
    (4,380–13,395 distinct names), which hold 22.2% of the held-out
    subfield's unseen test names (7–36% per subfield), about a third of
    what the corpus holds. The lexicon's precision was not estimated and
    its names are not stored (Limitations).
  - *Pool.* The training mentions (601–1,309 per MuLMS fold) and the
    harvested mentions with their corpus frequencies; harvested mentions
    are about 86–99% of the pool's entries, so most replacements are
    harvested names.

Every run stores its teachers' scores, the lexicon's counts and the share
of the unseen test names it holds, the corpus's hash and the filter. A
MR6 + names job (three teachers, tagging, three students) takes 30 to 50
minutes on a T4 against 1.6 minutes for a CRF head run and 5 for an MR6
run; the students see 2.8 to 4.5 times more sentences per epoch and stop
at epoch 4 to 5 against the CRF head's 7, so they take about twice as
many updates (Appendix B).

## 5 Results

### 5.1 The pre-registered decisions

Table 1 lists the eight decisions of the names route in the order they
were taken, with the seeds each used.

| | Corpus, units (MuLMS seeds) | Comparison | Reference F1 | System F1 | Δ micro F1 [95% CI] | Verdict |
|---|---|---|---|---|---|---|
| D1 | MuLMS, 7 subfields (300–900) | MR6 + names vs CRF head | 48.54 | 51.73 | +3.19 [−1.13, +7.51] | no |
| D2 | MuLMS, 7 subfields (300–900) | MR6 + names vs MR3 | 50.81 | 51.73 | +0.92 [−1.63, +3.48] | no |
| D3 | MaterioMiner fine, 4 papers | MR6 + names vs CRF head | 46.46 | 50.72 | +4.27 [+1.39, +7.15] | passes |
| D4 | MaterioMiner coarse, 4 papers | MR6 + names vs CRF head | 58.60 | 60.38 | +1.77 [+0.67, +2.88] | passes |
| D5 | pooled: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR6 + names vs CRF head | | | +3.58 [+0.95, +6.21], neither corpus below 0 | passes |
| D5′ | the same with MuLMS seeds 300–900 (the protocol's batch; post hoc reading) | | | | +3.58 [+1.04, +6.13] | passes |
| D6 | SOFC-Exp, 5 article folds | MR6 + names vs CRF head | 82.79 | 82.75 | −0.04 [−1.21, +1.14] | no |
| D7 | pooled, 11 units (300–900) | MR6 + names vs MR6 | | | +0.80 [−0.66, +2.26] | no |
| D8 | MuLMS, 7 subfields (300–900) | own-topic corpus vs mixed corpus | 51.91 | 51.73 | −0.18 [−2.01, +1.65] | no |

Three decisions pass, and they are not three pieces of evidence: D3 and
D4 are the same four papers at two granularities, and D5's eleven units
contain those four papers. The one independent corpus, MuLMS, fails every
single-model decision (D1, D2), where 6 of its 7 subfields gain against
the CRF head (sofc by 11.9, poly loses 2.4) and the differences have a
standard deviation of 4.7 points across subfields. On MaterioMiner every
paper gains at both granularities. The pooled estimate is stable: the
stratified mean of the two corpus means is 3.73, 10 of the 11 units are
positive, the exact sign-flip permutation test gives p = 0.007 (0.011 on
the 10-seed reading), and leaving any one unit out keeps the lower bound
above 0 (lowest +0.54, without semi). Figure 2 (`docs/figures/fig9_units.png`)
shows every unit.

### 5.2 Whose gain is it

Table 2 (generated as `docs/tables/table9_robustness.md`) gives every
comparison among the four systems on the same 11 units, on both seed
bases; only D5 and D7 among its rows were pre-registered. On the decision
seeds:

| Comparison (11 units; MuLMS seeds 300–900) | Δ micro F1 [95% CI] | units above 0 | sign-flip p |
|---|---|---|---|
| MR3 − CRF head | +2.46 [+1.00, +3.91] | 11/11 | 0.001 |
| MR6 − MR3 | +0.33 [−0.25, +0.92] | 8/11 | 0.23 |
| MR6 − CRF head | +2.79 [+1.20, +4.38] | 10/11 | 0.002 |
| MR6 + names − MR6 (D7) | +0.80 [−0.66, +2.26] | 7/11 | 0.24 |
| MR6 + names − MR3 | +1.13 [−0.51, +2.76] | 7/11 | 0.16 |
| MR6 + names − CRF head (D5′) | +3.58 [+1.04, +6.13] | 10/11 | 0.007 |

The +3.58 decomposes, on this seed basis, into +2.46 from mention
replacement as published, +0.33 from the three extra copies and +0.80
from the harvested names, and only the first term is established. The
10-seed basis gives the same picture (+2.35, +0.22, +1.02). Against the
published method, then, the recipe's whole increment is +1.13 [−0.51,
+2.76], and the pre-registered pass over the CRF head is a property that
mention replacement already has on these units: its own 11-unit gain,
computed post hoc from runs that predate the names route, is +2.35
[+0.94, +3.75] on the 10-seed basis, with 10 of 11 units positive. Per
corpus the picture holds: on MaterioMiner, fine-grained, MR3 gains +2.77
[+1.51, +4.03] over the CRF head and MR6 + names +1.50 [−1.76, +4.75] over
MR3; coarse-grained, MR3 gains +1.28 [+0.06, +2.51] and MR6 + names +0.49
[+0.06, +0.92] over MR3, the one cell of the grid where the names' own
increment clears 0, with 3 degrees of freedom and no pre-registration
(the full grids are in Table 2's second block). On MuLMS MR3 gains +2.10
[−0.31, +4.51] over the CRF head at 10 seeds and its votes pass (+2.71
[+0.22, +5.20]).

The names decision D7 was declared underpowered in its protocol: with
the spread seen in the screen it expected a half-width near 2 for a
names share near 1, so "not established" is the designed outcome of a
small effect, not evidence of none. What the design can say is that the
names add less than about 2.3 points pooled.

### 5.3 The names' source and the recipe's settings

**Source (D8).** For each held-out subfield we built a mixed corpus of
40,000 sentences drawn equally from the six other subfields' corpora,
with the same source, pipeline and exclusions; it holds 49–70% of the
subfield's unseen test names against the topic corpus's 55–81%, and its
lexicon 18.3% against 22.2%. MR6 + names with the mixed corpus scores
51.91 against 51.73 with the topic's own, −0.18 [−2.01, +1.65] paired by
subfield (two subfields better with the own corpus, five with the mixed
one); the mixed variant gains +3.38 [−0.89, +7.64] over the CRF head and
+0.70 [−1.19, +2.58] over MR6. No difference is detected; the data are
compatible with differences of up to about 2 points either way.

**Settings (reported, 3 seeds).** Each ablation changes one setting and
runs on the 7 subfields at the screen's seeds (0 to 200) against the
recipe's own runs on those seeds, which are the selected variant's and
so a slightly flattering reference. Table 3 (`docs/tables/table7_ablations.md`):

| Setting (the recipe's value) | Distinct names | Unseen test names in the lexicon | Micro F1 | Δ vs the recipe [95% CI] | Δ vs MR3 | Δ vs CRF head |
|---|---|---|---|---|---|---|
| the recipe: 40,000 sentences, threshold 0.9, 3 of 3 teachers, a uniform pool | 54,095 | 22.2% | 52.09 | | +1.45 [−1.35, +4.25] | +3.16 [−1.96, +8.28] |
| corpus: its first 10,000 sentences | 19,171 | 16.4% | 51.94 | −0.15 [−1.78, +1.48] | +1.30 [−1.14, +3.74] | +3.01 [−1.21, +7.23] |
| corpus: its first 20,000 sentences | 32,461 | 20.1% | 51.93 | −0.16 [−2.45, +2.14] | +1.30 [−0.56, +3.15] | +3.00 [−0.49, +6.49] |
| filter: marginal threshold 0.7 | 77,879 | 31.5% | 51.93 | −0.16 [−1.62, +1.31] | +1.29 [−0.95, +3.53] | +3.00 [−1.85, +7.85] |
| filter: 2 of 3 teachers agree | 78,604 | 28.8% | 52.20 | +0.11 [−1.58, +1.81] | +1.56 [+0.07, +3.05] | +3.27 [−0.85, +7.38] |
| pool: harvested names in half of the draws (86–99% uniform) | 54,095 | 22.2% | 52.18 | +0.09 [−1.63, +1.80] | +1.54 [−0.74, +3.82] | +3.24 [−0.73, +7.22] |

The lexicon ranges from 19 thousand to 79 thousand distinct names and
from 16% to 32% of the unseen test names, and micro F1 stays within 0.16
of the recipe's 52.09; every interval against the recipe covers 0, the
votes' too (+0.20, −0.32, +0.34, +0.45, +0.08), and per subfield the
differences run from −3.2 to +2.9, the spread of 3-seed means. These
readings detect differences of 2 to 3 points; what they show is the
absence of any effect of that size, not the equality of the settings.
Within that limit, a quarter of the corpus (a quarter of the teachers'
tagging) with the strictest filter does as well as the rest, which is
what Section 5.2 predicts: if the names add under a point, the settings
that govern how many names enter cannot show an effect. On the unseen
names the settings do move the trade of Section 5.4 (exploratory; the
screen's 3 seeds pooled over the subfields, against the recipe's screen
runs at 48.1% recall and 50.0 precision, computed from their saved
predictions): smaller lexicons recall fewer at higher precision (10,000
sentences 46.9% at 52.3; 20,000 46.2% at 53.0; the names in half of the
draws 46.2% at 53.7), larger ones more at lower precision (threshold 0.7
48.8% at 49.5; 2 of 3 teachers 48.3% at 50.6), while F1 on the unseen
names stays between 49.1 and 49.7.

### 5.4 What the names do

By kind of name, pooled over MuLMS's held-out subfields (Table 4; Figure 3,
`docs/figures/fig7_names.png`, puts each system's recall and precision on
the unseen names on one plane with iso-F1 curves): the CRF head finds
39.4% of the unseen names at 55.8% precision; MR3 43.2% at 54.3; MR6
43.7% at 53.9; MR6 + names 47.3% at 50.9. Against the matched control the
names raise recall on unseen names by 3.6 points and lower precision by
3.0, a net +0.7 F1 on unseen names (48.3 to 49.0) and +0.2 on seen ones
(71.0 to 71.2); the point moves almost along an iso-F1 curve. With the
mixed corpus the figures are the same (46.7% at 52.4). On MaterioMiner's
papers, fine-grained, the names add to both recall and precision on
unseen names (39.7 to 41.1 and 49.2 to 53.2 over MR6; F1 43.9 to 46.4),
and coarse-grained a little of each (53.0 to 54.2 and 52.4 to 53.2; F1
52.7 to 53.7).

| MuLMS, unseen names (88% of the test entities); 70 runs per row, 49 for the mixed corpus | recall | precision | F1 |
|---|---|---|---|
| CRF head | 39.4 | 55.8 | 46.2 |
| MR3 | 43.2 | 54.3 | 48.1 |
| MR6 | 43.7 | 53.9 | 48.3 |
| MR6 + names, the topic corpus | 47.3 | 50.9 | 49.0 |
| MR6 + names, a mixed corpus | 46.7 | 52.4 | 49.4 |

Whether this is name knowledge or a generic shift toward tagging more
spans, the recorded data do not say, and what they say points to the
second. Per subfield, the lexicon's coverage of the unseen test names
runs from 7% (semiconductors) to 36% (electrolysis), and the names' gain
over MR6 on the decision seeds from −2.3 (polymers) to +3.3 (solid oxide
cells); the two correlate negatively (Pearson r = −0.71, Spearman −0.61),
while the gain correlates with how weak the CRF head is on the subfield
(r = −0.76 with its F1: semiconductors at 32.5 and solid oxide cells at
43.5 gain most). A mixed corpus that covers fewer unseen names scores at
least as well on every pooled measure. The decisive analysis, recall on
the unseen test names that are in the lexicon against those that are not,
cannot be run from the repository because the lexicons were not stored;
nor was a recall-matched control (a bias toward tagging) run against the
names. Both are the first things to do if the names question is taken up
again (Section 7).

### 5.5 Where nothing helps: SOFC-Exp

On SOFC-Exp's five folds of held-out articles the recipe with the SOFC
corpus scores 82.75 against the CRF head's 82.79, −0.04 [−1.21, +1.14]
(D6). It gains +0.45 [+0.04, +0.86] over MR3 on every fold (reported), but
MR3 itself is below the CRF head there (−0.48 [−1.36, +0.39]), as on the
official split (−0.29 [−1.04, +0.47]). Three things differ on this
corpus, in one place: only 46% of a fold's test names are unseen (81%
fine- and 79% coarse-grained on MaterioMiner's papers, 88% on MuLMS's
subfields) and the CRF head already finds 80% of them; the baseline is
at 82.8 F1 with a random-split gap of 3.5 points, so there is little
headroom; and the SOFC corpus holds 36% of the unseen test names
(27–43% per fold) against 55–81% on MuLMS, the filter admitting 20–33%.
Training size is not the explanation: trained on 10 of the official
split's 26 training articles (5 draws of 156–248 sentences, MaterioMiner's
size; 10 seeds each) and scored on its test articles, MR3 gains +0.50
[−0.47, +1.46] over the CRF head, still nothing.
Pooled over the 16 units of the three corpora the recipe's estimate is
+2.45 [+0.50, +4.40] (reported); the claim stays the two-corpus one, with
SOFC-Exp as its boundary condition.

### 5.6 Per entity type

Table 5 (`docs/tables/table8_types.md`) gives, per type, the F1 of the CRF
head, MR6 and MR6 + names on MuLMS's subfields and MaterioMiner's
coarse-grained papers, averaged over the (seed, unit) pairs whose test
set has the type; these are exploratory and are not decided on. The
pattern of Section 5.2 holds per type: where MR6 + names gains on a type
over the CRF head, most of the gain is already MR6's.

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
  Four decisions, none passing: the filter, not the corpus, was the
  bottleneck.
- *Seven variants on the topic corpora* (week 12, a 3-seed screen on the
  7 subfields; Table 6). The selection rule took the largest mean gain
  over MR3, which favours the noisiest variant; the screen decides
  nothing, and two of its losers deserve a 10-seed test that was not run:
  self-training on the corpus's silver sentences (V6) was the one variant
  whose interval against the CRF head cleared 0, by raising precision on
  unseen names (58.5 at 40.4% recall), and the lexicon as a gazetteer
  match feature of the CRF head, the classical use of a harvested
  lexicon, lost to MR3 on 3 seeds.

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
  where MR6 + names gains most (Section 5.4).
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
documents to read, use mention replacement: it is the one change here
that improves a single model on held-out documents of two corpora, at
twice the training cost of the baseline and no corpus to build. The
extra copies and the harvested names are not shown to add to it; the
names cost a 40,000-sentence corpus and three teacher passes, twenty
times the baseline's training time, for an increment of +0.8 [−0.7, +2.3]
that buys 3.6 points of recall on unseen names for 3.0 of precision. If
recall on new names is what you need, the names are a cheap way to buy
it; if F1 is, stop at mention replacement. Expect nothing from either
where most of a new document's names are already in training, as on
SOFC-Exp, and expect the most where the baseline is weakest.

**What would settle the names question.** Three analyses the present
data cannot support: recall split by whether an unseen test name is in
the lexicon, which needs the lexicons stored; a recall-matched control, a
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
mentions for others of the same type, is the simplest of the augmentations
Dai and Adel (dai2020analysis) compared, and the one that helped
transformers most on small training sets; Torres et al.
(torres2024augmentation) find its gains shrink and reverse as training
sets grow, and Sharma et al. (sharma2023paraphrase) note that most
augmentations make the tagger memorise entities, which is why unseen
entities are scored separately here. The replacements otherwise come from
the training set (dai2020analysis; bartolini2024cosiner choose the one
that fits the context), from a masked language model (zhou2022melm), from
a generative model that writes text around an entity (hu2023entity), from
related sentences reached through Wikipedia anchors (cai2023graph) or
from predefined lists of the target domain's names (amalvy2022mention).
Chen et al. (chen2021crossdomain) augment for cross-domain NER and Liu et
al. (liu2021crossner) benchmark it; both change domain, not document. We
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
match features at +0.5 F1 on OntoNotes and find that replacing annotated
entities with gazetteer entries gives no significant gain where test names
are not the bottleneck. On the teacher side, BOND (liang2020bond)
self-trains on dictionary-matched labels, Wu et al. (wu2020teacher) train
a student on unlabeled target text from several teachers, and Qu et al.
(qu2023adaptive) use teachers' agreement in distantly supervised NER; our
self-training variant (du2021selftraining) is the same idea on the same
corpus. What this paper adds is not a new combination so much as the
test: the lexicon's contribution isolated against a matched control, its
source and its filter varied, on held-out documents, under protocols
fixed in advance, and the finding that the combination's gain is mention
replacement's.

**Materials NER and unseen documents.** MatSciBERT (gupta2022matscibert)
is pre-trained on materials papers; Weston et al. (weston2019named)
trained a materials NER model on annotated abstracts and applied it at
scale; MuLMS (schrader2023mulms), SOFC-Exp (friedrich2020sofc) and
MaterioMiner (durmaz2024materiominer) are the corpora here, and Potu et
al. (potu2025ontology) study a shift between two fatigue datasets with
the same classes, where the gap to unseen papers shrinks from about 30
points with three training papers to about 12 with 47. Random sentence
splits leak names between training and test (babl2025random), and taggers
score far lower on mentions unseen in training
(augenstein2017generalisation); our companion study measures both on the
three corpora and finds that every gain that holds on held-out documents
is a gain on unseen names.

## 9 Conclusion

A tagger trained on a few annotated materials papers misses most of a
new paper's entity names. Giving it names harvested from unlabeled text
by three agreeing teachers, through mention replacement, improves on the
baseline on held-out documents of two corpora under a pre-registered
protocol; but the improvement is the one mention replacement makes on its
own, and the names' share of it, +0.8 points with an interval that
covers 0, is not established, does not depend on the names' source or on
the filter, and shows up as recall on unseen names bought with precision
where the baseline is weakest. We report the eight decisions, the one
criterion revision and the one deviation, and the analyses that would
settle the question. The practical advice is mention replacement; the
methodological advice is to test a names method against a matched copies
control and on documents the training data has never seen, with enough
seeds and units to detect the two points that separate a method from its
control.

## Limitations

- **Few, heterogeneous units.** Seven subfields and four papers give 6
  and 3 degrees of freedom; the subfields differ by 24 points under the
  CRF head (32.49 to 56.82) and range from 2 to 11 articles, the papers
  from 41 to 201 sentences, and the pooled decision weights them equally
  as if drawn from one population. The leave-one-out and sign-flip
  checks (Section 5.1) are post hoc; a mixed model over units nested in
  corpora was not fitted.
- **Three passes on one corpus's papers.** The MaterioMiner passes at two
  granularities and the pooled pass share four papers; MuLMS alone passes
  nothing single-model.
- **A revised criterion.** The methods criterion under which the recipe
  passes was revised before the decision runs but after a screen whose
  spread made the per-corpus test underpowered (Section 3); under the
  earlier criterion there is no claim.
- **One training regime.** Every positive result is at about 340 training
  sentences; MuLMS's full pool was never used, SOFC-Exp's larger training
  set is where nothing helps, and the literature predicts smaller gains
  for mention replacement as training grows. The claim is scoped to a few
  hundred annotated sentences.
- **The names' effect is not isolated.** The lexicons were not stored, so
  recall cannot be split by lexicon membership; no recall-matched control
  was run; the lexicon's precision was not estimated; the per-subfield
  correlations of Section 5.4 are exploratory with 7 points.
- **Null results are not equivalences.** The source decision and the
  ablations detect differences of about 2 to 3 points; smaller effects
  of the source or the settings are not excluded.
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
authors' direction, to write code and drafts; the authors are responsible
for the content. This paper shares data, baselines and protocol with a
companion evaluation study of the same programme, which reports the
random-split gap, seed variance and the other additions tested; the two
papers overlap in Sections 2 to 4 and in no result.

## Data and code availability

The repository holds the code, every protocol with its dates and
amendments, every results file with seeds, environment and saved test
predictions, the analyses behind every number, the generated tables and
figures, the corpus reports with queries, article identifiers and
SHA-256 hashes, and the Kaggle batches that produced each file, under the
Apache 2.0 licence. MaterioMiner's files are included under CC BY 4.0;
SOFC-Exp is converted from the authors' repository; MuLMS is read from
its Hugging Face release at a pinned revision (CC BY-SA 4.0). The
unlabeled corpora and the harvested lexicons are not released; the corpora
rebuild from their reports, with hash checks. Author contributions,
competing interests and acknowledgements are to be added for the venue.

## Appendix A: the protocol record

| Decision | Protocol committed | Runs | Result recorded | Amendments |
|---|---|---|---|---|
| D1, D2 (week 12) | 18 October (`bd20cec`): the screen on seeds 0–200, the decisions on seeds 300–900 so that the selection cannot bias them; the MaterioMiner analogue gated on D1; the methods criterion: D1 plus a MaterioMiner pass | 20–24 October | 24 October | three amendments before any run added variants V4, V6, V7–V8 to the screen |
| D3, D4, D5 (week 13) | 20 October 09:14 (`808ff4b`): the pooled decision over both corpora as a second route to the claim, "each unit its 10-seed mean", the stop rule lifted; amended the same day, after the screen had chosen the recipe and before any decision run (text timed 14:00–14:30, committed 18:42, `de62446`): the analogue runs regardless of D1, and D6 runs on a pass of D1 or D5 | 24–25 October | 25 October (`3432988`) | the analysis took MuLMS's 10-seed means; the 7-seed reading is D5′ |
| D6 (week 13) | 20 October, as above | 25 October | 25 October | — |
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
epoch 7.1 on average), an MR3 run 198 s (4.7), an MR6 run 296 s (4.1), an
MR6 + names run 2,109 s including its share of the teachers' tagging
(4.6); the students see 2.8 to 4.5 times more sentences per epoch. Runs
used Kaggle sessions of two T4 GPUs with deterministic CUDA, one frozen
commit per batch, a pinned image and pinned libraries; the three week-14
batches alone took 19 session hours (5.2, 4.3 and 9.7).

## Appendix C: the unlabeled corpora

Each corpus is 40,000 sentences of open-access full texts found on Europe
PMC by a title query (for solid oxide cells: "solid oxide", "SOFC",
"SOEC" and their plurals; the other queries are in the repository's
reports), with the dataset's own articles excluded by identifier, every
sentence sharing a word 8-gram with a dataset sentence dropped (and, for
MuLMS, any 50-character run), sentences of 5 to 120 words with at least
half alphabetic characters, and formulas longer than 30 characters cut.
Per corpus: 290–370 articles, published between 1976 and 2026 (over 99%
from 2009 on), 63–90% of them CC BY and 10–33% CC BY-NC or ND (1–4%
without a licence field), the largest journal holding 14–63% of the
articles (MDPI Materials 52% of the steel corpus and 63% of the fatigue
corpus). Each
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
