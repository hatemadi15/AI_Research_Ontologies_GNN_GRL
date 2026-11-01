# Supplementary Information

**Harvested names are not shown to add to mention replacement for materials named-entity recognition on held-out documents: a pre-registered test**

Hatem Madi

Supplementary Notes 1 to 9 give the protocol record, the hyperparameters,
compute and repeats, the unlabeled corpora, and the analyses the main
text summarises; Tables S1 to S6 are numbered in order of appearance.
Section, table and figure numbers without an S refer to the main text.

## Supplementary Note 1: the protocol record

The repository names its protocols by the stage of the project in which
they were fixed, a "week"; Table S1's first column gives that name with
each protocol's decisions, and the main text cites the protocols by date.

**Table S1.** The protocol record of the names route: when each decision's
protocol was committed (commit times, UTC; the protocol text's own
written times are given below the table), when its runs ran, when its result was
recorded, and the amendments.

| Decision | Protocol committed (times UTC) | Runs | Result recorded | Amendments |
|---|---|---|---|---|
| the earlier protocols (Section 6): mention replacement on MuLMS (week 10); names from the other subfields' sentences (week 11) | 17 October 20:21 (`a08356f`); 18 October 01:13 (`427e34c`) | 17–20 October | week 10: 18 October 00:48 (`c2ca3f6`); week 11: 18 October (MuLMS, `14ed50f`) and 20 October (MaterioMiner, `04764ef`) | — |
| D1, D2 (week 12) | 18 October 11:00 (`40631ce`; refined 11:05, `bd20cec`): the screen on seeds 0–200, the decisions on seeds 300–900 so that the selection's seed noise does not enter them; the MaterioMiner analogue gated on D1; the methods criterion: D1 plus a MaterioMiner pass | 20–24 October | 24 October | three amendments before any run added variants V4, V6, V7–V8 to the screen |
| D3, D4 (week 12) | 18 October 11:00 (`40631ce`): the winner's MaterioMiner analogue decided against the CRF head per granularity, to run only if D1 passed; 20 October 00:51 (`ce0f592`): the analogue's batches, with its comparisons against MR3 and the votes reported; the gate lifted at 09:14 (`808ff4b`), restored at 13:53 (`5f7f827`), lifted again at 18:42 (`de62446`) | 24–25 October | 25 October (`3432988`) | — |
| D5 (week 13) | 20 October 09:14 (`808ff4b`): the pooled decision over both corpora as a second route to the claim, "each unit its 10-seed mean", the stop rule lifted; amended three times the same day, before any decision run of the route: 09:25 (`e94f352`), the label-propagation rule's two MaterioMiner decisions and an SOFC-Exp report added after its MuLMS results (Supplementary Note 7); 13:53 (`5f7f827`), eighteen minutes after the screen chose MR6 + names, D6 added conditional on D1 and the analogue, which the morning's text ran whenever the screen had a winner, gated on D1; 18:42 (`de62446`), D6's condition widened to D1 or D5 and the analogue's morning condition restored, the gating recorded as a mistake | 24–25 October | 25 October (`3432988`) | the analysis took MuLMS's 10-seed means; the 7-seed reading is D5′ |
| D6 (week 13) | 20 October 13:53 (`5f7f827`), conditional on D1; widened to D1 or D5 at 18:42 (`de62446`) | 25 October | 25 October | — |
| D7, D8 (week 14) | 29 October 12:07 (`2886267`), two minutes after the first batch was pushed and five hours before its results | 29 October | 29 October | — |
| ablations (week 14) | 29 October, as above | 31 October | 31 October | — |
| the lexicon-membership analysis (week 15; post hoc, no decision) | 31 October 16:13 (`2c60f33`): the rebuild, its checks and the reading fixed before the run | 31 October (`w15a`) | 31 October | — |
| D9, D10 (week 17; follow-up) | 31 October 19:15 (`f39447c`), after D1 to D8 had been written up (the lexicon split's results arrived 26 minutes later), a pass expected to be unlikely; the code 19:41 (`013f9ac`) | 31 October 19:55 to 1 November 00:40 (`w17a`, `w17b`) | 1 November | — |

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
earlier by about six minutes in the week-17 protocol (written 19:10,
committed 19:15:57). Week 16 was the companion paper's write-up and has no protocol.

## Supplementary Note 2: hyperparameters, compute and repeats

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

**Repeats.** From week 5 on (15 October, 16:00 UTC), every GPU session
except two first repeated an earlier run, and every repeat was identical:
one batch of 15 October (23:14, `w5a`) ran none, and the follow-up's
second session (`w17b`) instead reproduced the stored multi-task runs'
scores through its teachers. The week-4 sessions of 15 October, which
wrote the CRF head's first GPU reference runs, followed a session that ran
one run twice, with identical results.

## Supplementary Note 3: the unlabeled corpora

Each corpus is 40,000 sentences of open-access full texts found on Europe
PMC (europepmc2015europe) by a title query (for MuLMS's solid-oxide subfield: "solid oxide",
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

## Supplementary Note 4: the lexicon's settings

**Settings (pre-specified, 3 seeds).** Each ablation (Table S2) changes one
setting and runs on the 7 subfields at the screen's seeds (0 to 200)
against the runs of MR6 + names on those seeds, which are the selected
variant's and so a slightly flattering reference.

**Table S2.** The lexicon's settings (pre-specified, 3 seeds): distinct names,
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

## Supplementary Note 5: what the names do, in detail

**Rates.** Over all ten seeds the names system makes, net of the
predictions it drops, 474 more unseen-name predictions per run than MR6
and 143 more correct ones, a net rate of 30% (MR3's net additions over
the CRF head, 43%; MR6's over MR3, 33%); these are differences of pooled
counts, not the precision of the predictions one system makes and the
other does not. On the decision seeds alone the pooled trade against MR6
is 3.0 points of recall for 2.9 of precision and +0.5 F1 on unseen names
(Table 4's note). The mixed corpus's figures are similar (46.7%
recall at 52.4 precision, against the topic corpus's 46.9% at 51.3 on the
same seeds), and its unseen-name F1 is slightly higher (49.35 against
49.0). On MaterioMiner's papers, fine-grained, the names add to both
pooled rates on unseen names (39.7% to 41.1% recall and 49.2 to 53.2
precision over MR6; F1 43.9 to 46.4), and coarse-grained a little of each
(53.0% to 54.2% and 52.4 to 53.2; F1 52.7 to 53.7), within paired
intervals that cover 0.

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
membership split of Section 5.4 finds the recall gain on the held names in every
subfield; F1 also pays the precision the names cost. 

**The lexicon split in full.** Table S3 gives the levels and every paired
comparison of the analysis of Section 5.4. On MuLMS's decision seeds
the names add 8,336 predictions of names the lexicon holds, 39% of them
right, and 12,206 of other names, 22% right (419 more unseen-name
predictions per run, 29% right), so two thirds of the wrong predictions
they add are names the lexicon does not hold, and 45% of the added correct
ones too; precision falls on both parts, 74.7% to 69.0% on the held names
and 46.4% to 44.0% on the others. The mixed-corpus system's own lexicon
holds 18.3% of the unseen test names, not the topic lexicon's 22.2% by
which its rows are split.

**Table S3.** Recall on the unseen test names by lexicon membership (post
hoc; lexicons rebuilt by the names method's own teachers; MuLMS on the
decision seeds, MaterioMiner on 10 seeds; a test name is unseen by main
Section 2's words-only rule; the mixed-corpus rows are split by the topic
corpus's lexicon, the one rebuilt). Levels: per system, the share
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

## Supplementary Note 6: per entity type

Table S4 gives, per type, the F1 of the CRF head, MR6 and MR6 + names on
MuLMS's subfields, averaged over the (seed, unit) pairs whose test set
has the type (the MaterioMiner tables, both granularities, are in the
repository); these are exploratory and are not decided on (Section 5.2).
On MuLMS, MR6 + names gains over the CRF head on the 12 types with a
non-zero score (RANGE, 3 entities per unit, scores 0 under every
system), and MR6 holds more than half of that gain on 9 of them (not on
MEASUREMENT, TECHNIQUE and DEV, where the names' own increment is +1.7
to +2.8). On MaterioMiner, coarse-grained, MR6 + names gains over the
CRF head on 16 of 27 types and MR6 holds more than half of the gain on
13 of those; the names also lose against MR6 on several types, two where
MR6 had gained (MEP −8.0, STD −7.0) and one where it had not (MAT −5.8),
and most on types with a handful of test entities (PRP −10.2, CPR −9.2).

**Table S4.** Per entity type on MuLMS's 7 held-out subfields (decision
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

## Supplementary Note 7: the earlier protocols, the screen and two other analyses

MR6 + names was reached through the protocols of weeks 10 and 11 and the
screen of week 12, on MuLMS's held-out subfields and their MaterioMiner
analogues, each committed before its runs and failing or passing as
recorded:

- *Mention replacement alone* (week 10, 17 October): +2.10 [−0.31, +4.51] against the
  CRF head on MuLMS at 10 seeds, no; its votes +2.71 [+0.22, +5.20],
  passes. On MaterioMiner it had passed at both granularities (+2.77 and
  +1.28).
- *Names from the other subfields' unlabeled sentences* (week 11, 18
  October): the
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
- *Seven variants on the topic corpora* (week 12, 18 October, a 3-seed screen
  on the 7 subfields; Table S5, with the plan's labels, under which V5
  was not assigned). The selection rule took the largest mean gain
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

**Table S5.** The week-12 screen: seven variants on MuLMS's 7 held-out
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
- *A second rule, label propagation within a document* (below): a
  term the model tags in most of its occurrences in a document is tagged
  in the rest. Two of its ten pre-registered decisions pass, on two
  corpora, both under a point; it is not shown to add to mention
  replacement (+0.57 [−0.12, +1.25] on MuLMS).

**Label propagation within a document.** A post-processing rule on the saved predictions: a surface form that the
model tags in most of its occurrences in a document, with one most
frequent type, is tagged in its untagged occurrences too. On the CRF
head's 10-seed predictions it gains +0.82 [+0.18, +1.46] on MuLMS's
held-out subfields (pre-registered, every subfield gaining; 1,213
entities added per seed) and +0.12 [+0.01, +0.24] on coarse-grained
MaterioMiner (pre-registered in an earlier week; fine-grained +0.30
[−0.35, +0.95], no), and loses −0.44 [−0.72, −0.16] on SOFC-Exp's article
folds (pre-specified), where the annotators leave 14% of a tagged term's
occurrences untagged on the official test set. It is not shown to add
to mention replacement (+0.57 [−0.12, +1.25] on MuLMS). Two of its ten
pre-registered decisions pass (six in week 7; four in week 13, two on
MuLMS registered at 09:14 on 20 October and two on MaterioMiner at 09:25,
after the MuLMS results, computed 22 minutes later from saved
predictions), where about 0.25 would by chance.

## Supplementary Note 8: the follow-up in detail

The two-head filter of the follow-up admitted lexicons of 16,455 to
42,066 entries per paper (3,746 to 6,587 distinct fine-typed names). On
the unseen test names the names' changes over the multi-task control are
+0.58 [−2.93, +4.09] recall and +0.82 [−3.39, +5.03] precision
fine-grained, and +1.45 [−0.69, +3.58] and +0.97 [−1.09, +3.03]
coarse-grained (pre-specified; the paired intervals of Section
5.4). Post hoc, F1 changes by +2.43 [+0.40, +4.47] on the seen and +0.43
[−1.57, +2.44] on the unseen test names fine-grained, and by +0.02
[−1.30, +1.34] and +1.20 [−0.82, +3.22] coarse-grained; no pattern is
read from these. A names job took 82.5 to 122.7 minutes on a T4 (three
teachers, their tagging of 40,000 sentences with two heads, and ten
students): 10.0 minutes per student with the harvest shared over the
job's ten students and about 38 charged to one, against 6.9 for a
control run and about 2.4 for the two CRF-head runs that give both
granularities (Supplementary Note 2).

## Supplementary Note 9: detectable differences

The smallest true mean difference each decision and main comparison
detects with 80% power
at the two-sided 5% level, from the half-width of its observed interval
and its degrees of freedom (`ner_robustness.py`, `detectable`).

**Table S6.** The difference each comparison could detect at 80% power
and the two-sided 5% level, from its stored interval's half-width and
degrees of freedom.

| Comparison | CI half-width | df | detectable at 80% power |
|---|---|---|---|
| D5: MR6 + names − CRF head, pooled, 10-seed basis | 2.63 | 10 | 3.67 |
| D5′: the same on the decision seeds | 2.54 | 10 | 3.55 |
| D7: MR6 + names − MR6, pooled | 1.46 | 10 | 2.04 |
| MR6 + names − MR3, pooled (post hoc, decision seeds) | 1.63 | 10 | 2.28 |
| MR6 − MR3, pooled (post hoc, decision seeds) | 0.58 | 10 | 0.82 |
| MR3 − CRF head, pooled (post hoc, decision seeds) | 1.46 | 10 | 2.03 |
| D1: MR6 + names − CRF head, MuLMS | 4.32 | 6 | 5.92 |
| D2: MR6 + names − MR3, MuLMS | 2.55 | 6 | 3.50 |
| MR6 + names − MR6, MuLMS (part of D7) | 2.02 | 6 | 2.77 |
| D8: own-topic − mixed corpus, MuLMS | 1.83 | 6 | 2.51 |
| D3: MR6 + names − CRF head, MaterioMiner fine | 2.88 | 3 | 3.77 |
| D4: MR6 + names − CRF head, MaterioMiner coarse | 1.10 | 3 | 1.44 |
| MR6 + names − MR6, MaterioMiner fine (pre-specified) | 3.80 | 3 | 4.97 |
| the same, coarse (pre-specified) | 2.30 | 3 | 3.01 |
| D6: MR6 + names − CRF head, SOFC-Exp, 5 folds | 1.17 | 4 | 1.57 |
| D9: multi-task + MR6 + names − multi-task + MR6, MaterioMiner fine (follow-up) | 0.96 | 3 | 1.26 |
| D10: the same, coarse (follow-up) | 1.57 | 3 | 2.05 |
| multi-task + MR6 + names − multi-task + MR3, MaterioMiner fine (pre-specified) | 0.91 | 3 | 1.20 |
| the same, coarse (pre-specified) | 1.05 | 3 | 1.38 |
| ablation, 10,000 sentences − MR6 + names (3 seeds) | 1.63 | 6 | 2.23 |
| ablation, 20,000 sentences − MR6 + names (3 seeds) | 2.29 | 6 | 3.14 |
| ablation, threshold 0.7 − MR6 + names (3 seeds) | 1.47 | 6 | 2.01 |
| ablation, 2 of 3 teachers − MR6 + names (3 seeds) | 1.70 | 6 | 2.32 |
| ablation, names in half of the draws − MR6 + names (3 seeds) | 1.71 | 6 | 2.35 |
