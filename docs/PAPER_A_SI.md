# Supplementary Information

**Names mined from unlabeled papers are not shown to improve data augmentation for materials named-entity recognition: a pre-registered test**

Hatem Madi

Supplementary Notes 1 to 11 give the protocol record, the hyperparameters,
compute and repeats, the corpora, and the analyses the main text
summarises; Tables S1 to S10 are numbered in order of appearance. Section,
table and figure numbers without an S refer to the main text.

## Supplementary Note 1: the protocol record

The repository names its protocols by the stage of the project in which
they were fixed, a "week"; Table S1's first column gives that name with
each protocol's decisions, and the main text cites the protocols by date.
The last rows give the companion study's protocols whose results the main
text quotes; their decisions are not among the twelve of Table 2 or the
twenty of Section 6.

**Table S1.** The protocol record of the names route, then of the
companion study's results quoted in the main text: when each protocol was
committed (commit times, UTC; the protocol text's own written times are
given below the table), when its runs ran, when its result was recorded,
and the amendments.

| Decision | Protocol committed (times UTC) | Runs | Result recorded | Amendments |
|---|---|---|---|---|
| the earlier protocols (Section 6): mention replacement on MuLMS (week 10); names from the other subfields' sentences (week 11) | 17 October 20:21 (`a08356f`); 18 October 01:13 (`427e34c`) | 17–20 October | week 10: 18 October 00:48 (`c2ca3f6`); week 11: 18 October (MuLMS, `14ed50f`) and 20 October (MaterioMiner, `04764ef`) | — |
| D1, D2 (week 12) | 18 October 11:00 (`40631ce`; refined 11:05, `bd20cec`): the screen on seeds 0–200, the decisions on seeds 300–900 so that the selection's seed noise does not enter them; the MaterioMiner analogue gated on D1; the methods criterion: D1 plus a MaterioMiner pass | 20–24 October | 24 October | three amendments before any run added variants V4, V6, V7–V8 to the screen |
| D3, D4 (week 12) | 18 October 11:00 (`40631ce`): the winner's MaterioMiner analogue decided against the CRF head per granularity, to run only if D1 passed; 20 October 00:51 (`ce0f592`): the analogue's batches, with its comparisons against MR3 and the votes pre-specified; the gate lifted at 09:14 (`808ff4b`), restored at 13:53 (`5f7f827`), lifted again at 18:42 (`de62446`) | 24–25 October | 25 October (`3432988`) | — |
| D5 (week 13) | 20 October 09:14 (`808ff4b`): the pooled decision over both corpora as a second route to the claim, "each unit its 10-seed mean", the stop rule lifted; amended three times the same day, before any decision run of the route: 09:25 (`e94f352`), the label-propagation rule's two MaterioMiner decisions and an SOFC-Exp comparison pre-specified after its MuLMS results (Supplementary Note 7); 13:53 (`5f7f827`), eighteen minutes after the screen chose MR6 + names, D6 added conditional on D1 and the analogue, which the morning's text ran whenever the screen had a winner, gated on D1; 18:42 (`de62446`), D6's condition widened to D1 or D5 and the analogue's morning condition restored, the gating recorded as a mistake | 24–25 October | 25 October (`3432988`) | the analysis took MuLMS's 10-seed means; the 7-seed reading is D5′ |
| D6 (week 13) | 20 October 13:53 (`5f7f827`), conditional on D1; widened to D1 or D5 at 18:42 (`de62446`) | 25 October | 25 October | — |
| D7, D8 (week 14) | 29 October 12:07 (`2886267`), two minutes after the first batch was pushed and five hours before its first result, after MR6 + names' own results on the same units | 29 October | 29 October | — |
| ablations (week 14) | 29 October, as above | 31 October | 31 October | — |
| the lexicon-membership analysis (week 15; post hoc, no decision) | 31 October 16:13 (`2c60f33`): the rebuild, its checks and the reading fixed before the run | 31 October (`w15a`) | 31 October | — |
| D9, D10 (week 17; follow-up) | 31 October 19:15 (`f39447c`), after D1 to D8 had been written up (the lexicon split's results arrived 26 minutes later), a pass expected to be unlikely; the code 19:41 (`013f9ac`) | 31 October 19:55 to 1 November 00:40 (`w17a`, `w17b`) | 1 November | — |
| D11, D12 (week 18; follow-up) | 1 November 10:51 (`e02c3be`), at the request of an internal review of the draft (main text, Ethics and broader impact), after D1 to D10 and the lexicon split were known; the names' side the stored decision runs of 24 October, only the control new; the code 10:58 (`7906996`) and the analysis 11:04 (`789f42e`), both before any run | 1 November 11:14 to 13:43 (`w18a`, pushed at 11:12, pulled at 13:58) | 1 November 14:06 (`13506a4`); the post hoc nearest-recall reading added after it, 14:16 (`ea8a92e`) | — |
| companion, week 5: multi-task training against the CRF head on MaterioMiner's held-out papers, pre-specified without decision (fine-grained +3.72 [+0.57, +6.87]; multi-task training's own decision was on the random splits) | 15 October 15:50 (`6f2dc11`) | 15 October | 15 October 23:20 (`1b30d82`) | — |
| companion, week 7 part 1: label propagation within a document, six decisions on saved predictions | 16 October 11:46 (`cc73d37`) | saved predictions | 16 October 11:53 (MaterioMiner, `93966bb`) and 13:54 (SOFC-Exp, `ceabb78`) | 11:57 (`e41aa44`): retyping made a decision on SOFC-Exp, after MaterioMiner's results and before SOFC-Exp's |
| companion, week 7 part 2a: recall calibration, a bias on the CRF head's emissions chosen on validation folds; five decisions, none passing (fine-grained −0.59 [−1.75, +0.57], coarse-grained +0.17 [−0.14, +0.48], SOFC-Exp −0.21 [−0.58, +0.17] paired by seed) | 16 October 13:11 (`61c9520`) | 16 October (`w7a`) | 16 October 15:33 (`503f690`) | — |
| companion, week 7 part 2b: MR3 against the CRF head on MaterioMiner's held-out papers, both passing (+2.77 [+1.51, +4.03] fine-grained, +1.28 [+0.06, +2.51] coarse-grained), and on SOFC-Exp's official split, failing (−0.29 [−1.04, +0.47] paired by seed) | 16 October 14:39 (`494cac1`): a 3-seed screen, then 10 seeds | 17 October (`w7c` 03:42–04:15, `w7d` 04:19–06:47) | 17 October 06:59 (`db7c78d`; the screen at 04:19, `7d55340`) | — |
| companion, week 8 part 1: SOFC-Exp's random sentence splits and article folds, the CRF head's 86.27 and 82.79 of Section 1 (pre-specified without decision) | 16 October 17:24 (`c06fc5e`) | 17 October | 17 October 10:11 (`8cd8fba`) | — |
| companion, week 9: multi-task training with MR3 against the CRF head, pre-specified without decision (fine-grained +6.73 [+6.03, +7.43]; its single-model decisions on MaterioMiner were against the multi-task model); MR3 on a MaterioMiner-sized SOFC-Exp, failing (+0.50 [−0.47, +1.46]) | 17 October 07:28 (`536cbe7`) | 17 October | 17 October 18:50 (`81134b7`) and 19:30 (`65c17b9`) | 12:30 (`fa69d9a`): part 1 runs from part 2's bundle, before any result |
| companion: seen and unseen test names and the decomposition of the recall gap (post hoc, on saved predictions; no protocol) | — | saved predictions | MaterioMiner 16 October 17:00 (`3c06c11`); MuLMS and SOFC-Exp added with their runs | — |

**Times.** Protocol and result times are the commits' (UTC); push, pull
and run times are the session notes' (week 18's run times are those its
Kaggle job wrote in its manifest, by the machine's clock), and GitHub's
records date only the pushes of commits. When the pooled decision was
added at 09:14 on 20 October, the screen's first half was recorded (MR6
and the gazetteer variants, at 08:11) and its second half, with MR6 +
names, was not (13:35); the 18:42 amendment followed the spread between
subfields that showed the 7-subfield decision underpowered. The protocol
text's own written times differ from the commits': later by 6 to 53
minutes in the 09:25, 09:47 and 13:53 commits of 20 October (which carry
09:50 and 09:55, 10:40, and 14:00; the 09:47 commit recorded the
label-propagation rule's results on SOFC-Exp's folds and MaterioMiner),
earlier by about four hours in the 18:42 one (14:30), and
earlier by about six minutes in the week-17 protocol (written 19:10,
committed 19:15:57). Week 16 was the companion paper's write-up and has no
protocol.

**Push times.** Two records kept by GitHub, archived in the repository
(`protocol_pushes.py`), date each push independently of the commit
times: the public event feed (`results/github_push_events.tsv`) and the
repository's activity log (`results/github_push_activity.tsv`), which
lists every push to a branch with the commits before and after it,
except merges made through pull requests.
By the activity log every protocol commit of Table S1, from the companion
study's of 15 October to the follow-up's of 1 November, reached GitHub 1
to 7 seconds after its commit time (for example `40631ce` at 11:00:59
on 18 October, `808ff4b` at 09:14:24 and `de62446` at 18:42:57 on 20
October, `2886267` at 12:07:40 on 29 October, `f39447c` at 19:15:59 on 31
October and `e02c3be` at 10:51:13 on 1 November). The event feed lacks
some pushes and held none after 30 October 22:51 when last archived; on
every push both records hold it agrees with the activity log within two
seconds, and within one on every protocol's push. The
activity log's times for the pushes of 1 November between 08:00 and
09:00 UTC, none of them a protocol's, read one hour late (the hour that
US Pacific time repeats when daylight saving ends that day): the commits
they carried had been built on by commits pushed before the times the
log gives them.

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
except two began by repeating an earlier run, and every repeat was
identical: one batch of 15 October (`w5a`, run 16:05 to 22:10 UTC) ran
none, and the second session of the follow-up of 31 October (`w17b`) instead reproduced the
stored multi-task runs' scores through its teachers. The week-4 sessions
of 15 October, which wrote the CRF head's first GPU reference runs, began
without a repeat after a smoke session that ran one CRF run three times,
with identical scores and predictions.

## Supplementary Note 3: the corpora

**The labeled corpora.** MaterioMiner's papers are DOIs
10.1007/s11340-021-00758-x (s11340), 10.1038/ncomms7164,
10.1038/s41467-020-19071-7 (s41467) and 10.3390/met10111458. Its
fine-grained level keeps the 27 classes with most entities, ties broken
by token count then name (the 27th and 28th classes tie at 24 entities),
and relabels the other classes' entities O. MuLMS's kept entities by
type: MAT 12,647, VALUE 4,628, PROPERTY 3,880, FORM 3,180, MEASUREMENT
2,106, CITE 1,704, SAMPLE 1,391, TECHNIQUE 920, DEV 757, INSTRUMENT 370,
NUM 260, UNIT 91, RANGE 18.

**The unlabeled corpora.** Each is 40,000 sentences of open-access full
texts found on Europe PMC (europepmc2015europe) by a query on titles (for
MaterioMiner's fatigue corpus: "fatigue" in the title and a metal or
process term, such as alloy, steel, titanium, weld or selective laser
melting, in the title or abstract; for MuLMS's solid-oxide subfield:
"solid oxide", "solid-oxide", "SOFC", "SOEC" and their plurals; for
SOFC-Exp's corpus "solid oxide", "SOFC" and "SOFCs"; the other queries
are in the repository's reports), with the dataset's own articles
excluded by identifier, every sentence sharing a word 8-gram with a
dataset sentence dropped (and, for MuLMS, any 50-character run),
sentences of 5 to 120 words with at least half alphabetic characters, and
formulas longer than 30 characters cut; sentences are split by a
rule-based splitter that does not break after abbreviations or inside
decimal numbers, and words by a regular expression that separates
punctuation. Per corpus: 290–368 articles with text (292–370 selected),
published between 1996 and 2026 (over 99% from 2009 on), 63–90% of them
CC BY and 10–33% CC BY-NC or ND (0.3–4% without a licence field), the
largest journal holding 13–63% of the articles (MDPI Materials 52% of the
steel corpus and 63% of the fatigue corpus). Each report records the
query, the counts at every stage, the article identifiers with licences
and the corpus's SHA-256; a rebuild that differs is detected by the hash.

## Supplementary Note 4: the lexicon's settings

**Settings (pre-specified, 3 seeds).** Each ablation (Table S2) changes one
setting and runs on the 7 subfields at the screen's seeds (0 to 200)
against the runs of MR6 + names on those seeds, which are the selected
variant's and so a slightly flattering reference.

**Table S2.** The lexicon's settings (pre-specified, 3 seeds): distinct
names, the share of the unseen test names in the lexicon (by the runs'
stored counts, which call a test name unseen when no training sentence
annotates its words with its type; for MR6 + names this gives the 22.2%
of main Section 2's rule too), micro F1 on the 7 subfields and the
differences from MR6 + names, MR3 and the CRF head.

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
names the settings do move the trade of Section 5.4 (post hoc; the
screen's 3 seeds pooled over the subfields, against the screen runs of
MR6 + names at 48.1% recall and 50.0 precision, computed from their saved
predictions): smaller lexicons recall fewer at higher precision (10,000
sentences 46.9% at 52.3; 20,000 46.2% at 53.0; the names in half of the
draws 46.2% at 53.7), a lower threshold more at lower precision (0.7:
48.8% at 49.5), and 2 of 3 teachers a little more of both (48.3% at
50.6), while F1 on the unseen
names stays between 49.1 and 49.7.

## Supplementary Note 5: what the names do, in detail

Everything in this Note is post hoc, except the companion study's
decisions quoted at its end.

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
(53.0% to 54.2% and 52.4 to 53.2; F1 52.7 to 53.7). Paired over the four
papers the changes are +0.86 [−3.38, +5.10] in recall and +1.32 [−6.62,
+9.25] in precision fine-grained, and +1.60 [−1.37, +4.57] and +1.16
[−2.24, +4.57] coarse-grained, every interval covering 0.

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
comparison of the analysis of Section 5.4. A test name is unseen by main
Section 2's words-only rule and in the lexicon when the lexicon holds its
words with its type; a prediction is classed the same way by its words
and predicted type, and the precision of each part is the share of its
unseen predictions that are correct. On MuLMS's decision seeds
the names add 8,336 predictions of names the lexicon holds, 39% of them
right, and 12,206 of other names, 22% right (419 more unseen-name
predictions per run, 29% right), so two thirds of the wrong predictions
they add are names the lexicon does not hold, and 45% of the added correct
ones too; precision falls on both parts, 74.7% to 69.0% on the held names
and 46.4% to 44.0% on the others. The mixed-corpus system's own lexicon
holds 18.3% of the unseen test names, not the topic lexicon's 22.2% by
which its rows are split.

**The reading rule.** Fixed before the run (31 October 16:13,
`2c60f33`): "Name knowledge: the names' recall gain over MR6 is larger on
the unseen names the lexicon holds than on those it does not, the paired
interval of the difference above 0. A generic shift: the two gains are
alike, the interval covering 0 with both gains positive. Mixed: the
interval above 0 but the gain on the names not held also above 0." The
rule does not say whether a gain is read on its point estimate or its
interval. On MuLMS the difference's interval is above 0 (+6.17 [+2.44,
+9.90]) and the gain on the names not held is +1.12 [−0.78, +3.02]:
mixed on its point estimate, name knowledge on its interval. On
MaterioMiner both gains are positive at both granularities and the
difference's interval covers 0 (fine-grained +4.33 [−5.33, +13.98],
coarse-grained −1.54 [−3.60, +0.51]): a generic shift on point
estimates, and none of the three outcomes on intervals, every gain
interval covering 0. The outcome note of 31 October read MuLMS as name
knowledge and MaterioMiner as a generic shift, the first on intervals and
the second on point estimates; read one way throughout, the rule gives
mixed and a generic shift, or name knowledge and none.

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

The recall-matched control of Supplementary Note 10, registered after
this analysis, tests one alternative to its reading, a uniform shift
toward tagging more spans. The companion study's earlier
recall calibration (Table S1, week 7 part 2a), a bias on the CRF head's
emissions chosen on validation folds and not matched to the names, gave
−0.59 [−1.75, +0.57] fine-grained and +0.17 [−0.14, +0.48]
coarse-grained on MaterioMiner and −0.21 [−0.58, +0.17] on SOFC-Exp,
paired by seed, the entities it added being right 19–22% of the time,
against the names' net rate of 30% above.

## Supplementary Note 6: per entity type

Table S4 gives, per type, the F1 of the CRF head, MR6 and MR6 + names on
MuLMS's subfields, averaged over the (seed, unit) pairs whose test set
has the type (the MaterioMiner tables, both granularities, are in the
repository); these are post hoc and not decided on (Section 5.2).
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

## Supplementary Note 7: the earlier protocols, the screen and other analyses

MR6 + names was reached through the protocols of weeks 10 and 11 and the
screen of week 12, on MuLMS's held-out subfields and their MaterioMiner
analogues, each committed before its runs and failing or passing as
recorded:

- *Mention replacement alone* (week 10, 17 October 20:21): +2.10 [−0.31,
  +4.51] against the CRF head on MuLMS at 10 seeds, no; its votes against
  the CRF head's votes, +2.71 [+0.22, +5.20], pass. On MaterioMiner it had
  passed at both granularities (+2.77 and +1.28; Table S1, week 7 part
  2b).
- *Names from the other subfields' unlabeled sentences* (week 11, 18
  October 01:13): the
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
- *Seven variants on the topic corpora* (week 12, 18 October 11:00, a
  3-seed screen on the 7 subfields; Table S5, with the plan's labels, under
  which V5 was not assigned). The selection rule took the largest mean gain
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

- *The lexicon as a dictionary* (a CPU probe planned on 19 October before
  it ran; no decision): applied to
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
pre-registered decisions pass (six of 16 October, in the companion
study's week 7; four of 20 October, in week 13, two on MuLMS registered
at 09:14 and two on MaterioMiner at 09:25, after the MuLMS results,
computed 22 minutes later from saved predictions), where about 0.25
would by chance.

**The count.** Section 6's twenty decisions are the twelve of Table 2;
the six of the protocols of 17 October (20:21) and 18 October (01:13),
on mention replacement alone on MuLMS (its single-model decision failed
and its votes passed against the CRF head's votes) and on names from the
other subfields' unlabeled sentences (four, none passing); and the two of
20 October (09:14) on mention replacement without names on SOFC-Exp's
folds (MR3, Section 5.5, and multi-task training with mention
replacement, −0.11 [−0.69, +0.46]; both failing). The label-propagation
rule's decisions are not counted, the rule being a post-processing rule
on saved predictions rather than a use of names; four of them fall in
the window (20 October, above), and with them the count is twenty-four,
six passing (13.1a among them), about 0.6 expected by chance. The
companion study's decisions, all earlier, are not counted either. Five passed: D3, D4 and D5 against the CRF head, D11 on
recall, and mention replacement's votes against the CRF head's votes. Of
the twelve of Table 2, the seven against a mention-replacement control
or on the names' source are D2, D7, D8, D9, D10, D11 and D12; only D11
passes.

**Training size on SOFC-Exp.** A decision of the companion study (17
October 07:28; Table S1, week 9) trained on 10 of the official split's 26
training articles (5 draws of 156–248 sentences, MaterioMiner's size; 10
seeds each) and scored on its test articles: MR3 gained +0.50 [−0.47,
+1.46] over the CRF head and did not pass. MR6 + names was not run at
that size.

## Supplementary Note 8: the follow-up on the strongest system in detail

**Design.** The control is the multi-task model (one encoder, a CRF head
per granularity) with six-copy mention replacement; the names system adds
a lexicon harvested by three multi-task teachers (the stored multi-task
runs of seeds 0, 100 and 200, whose scores the jobs reproduced), each
tagging the fatigue corpus with both heads. A mention is admitted when
the three teachers agree on its coarse-grained span and type and on its
fine-grained type at exactly that span, or on its having none, with every
marginal at 0.9 or more on the coarse head and, where a fine type is
tagged, on the fine head; it joins the pool under its pair of types. The
protocol expected detectable differences of about 3.4 to 3.8 points
fine-grained and 1.4 to 2.4 coarse-grained.

**Table S6.** The follow-up: the names on the strongest MaterioMiner
system. Micro F1 over the four held-out papers (10 seeds; both
granularities from one model) with its 95% CI; the names against the
matched control (decisions D9 and D10), and both against multi-task
training with three-copy mention replacement, the multi-task model and
the CRF head (pre-specified). A bound of −0.00 lies just below 0.

| | System | micro F1 [95% CI] | Δ vs the control | Δ vs multi-task + MR3 | Δ vs multi-task | Δ vs CRF head |
|---|---|---|---|---|---|---|
| fine | multi-task + MR6 + names | 53.95 [47.52, 60.38] | +0.96 [−0.01, +1.92] (D9, no) | +0.76 [−0.15, +1.68] | +3.77 [+1.34, +6.21] | +7.49 [+5.96, +9.02] |
| fine | multi-task + MR6 (the control) | 52.99 [46.93, 59.06] | | −0.19 [−2.05, +1.67] | +2.82 [−0.00, +5.64] | +6.54 [+4.14, +8.94] |
| coarse | multi-task + MR6 + names | 61.25 [53.71, 68.80] | +0.85 [−0.72, +2.42] (D10, no) | +0.91 [−0.14, +1.97] | +3.21 [+0.46, +5.96] | +2.65 [−0.09, +5.38] |
| coarse | multi-task + MR6 (the control) | 60.40 [53.66, 67.15] | | +0.06 [−0.74, +0.86] | +2.36 [+1.15, +3.56] | +1.80 [+0.19, +3.40] |

**Lexicons.** The two-head filter admitted lexicons of 16,455 to 42,066
entries per paper (3,746 to 6,587 distinct fine-typed names). By the
runs' stored counts, which call a test name unseen when no training
sentence annotates its words with its type, they hold 7.5% of the unseen
test names fine-grained and 12.1% coarse-grained, pooled over the papers,
against 9.6% and 18.3% for the single-task lexicons of D3 and D4, lower
in 7 of the 8 paper–granularity cells; Table 5's 9.8% and 18.2% count by
main Section 2's words-only rule.

**Unseen names.** On the unseen test names the names' changes over the
multi-task control are +0.58 [−2.93, +4.09] recall and +0.82 [−3.39,
+5.03] precision fine-grained, and +1.45 [−0.69, +3.58] and +0.97 [−1.09,
+3.03] coarse-grained (pre-specified; paired over the four papers). Post
hoc, F1 changes by +2.43 [+0.40, +4.47] on the seen and +0.43 [−1.57,
+2.44] on the unseen test names fine-grained, and by +0.02 [−1.30, +1.34]
and +1.20 [−0.82, +3.22] coarse-grained; no pattern is read from these.

**Cost.** A names job took 82.5 to 122.7 minutes on a T4 (three
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

**Table S7.** The difference each comparison could detect at 80% power
and the two-sided 5% level, from its stored interval's half-width and
degrees of freedom. These rest on the observed spread and are uncertain
themselves: the 95% interval of a standard deviation runs from 0.57 to
3.7 times its estimate with 3 degrees of freedom, 0.64 to 2.2 times with
6 and 0.70 to 1.8 times with 10.

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
| MR6 + names − MR6, MuLMS (D7's per-corpus condition) | 2.02 | 6 | 2.77 |
| D8: own-topic − mixed corpus, MuLMS | 1.83 | 6 | 2.51 |
| D3: MR6 + names − CRF head, MaterioMiner fine | 2.88 | 3 | 3.77 |
| D4: MR6 + names − CRF head, MaterioMiner coarse | 1.10 | 3 | 1.44 |
| MR6 + names − MR6, MaterioMiner fine (D7's per-corpus condition) | 3.80 | 3 | 4.97 |
| the same, coarse (pre-specified) | 2.30 | 3 | 3.01 |
| D6: MR6 + names − CRF head, SOFC-Exp, 5 folds | 1.17 | 4 | 1.57 |
| D9: multi-task + MR6 + names − multi-task + MR6, MaterioMiner fine (follow-up) | 0.96 | 3 | 1.26 |
| D10: the same, coarse (follow-up) | 1.57 | 3 | 2.05 |
| D11: recall on the held names, MR6 + names − the recall-matched control, MuLMS (follow-up) | 7.93 | 6 | 10.87 |
| D12: micro F1, the same (follow-up) | 4.05 | 6 | 5.55 |
| D11 without the flagged subfields (pre-specified) | 9.38 | 3 | 12.27 |
| D12 without the flagged subfields (pre-specified) | 2.75 | 3 | 3.60 |
| D11, the flagged subfields at the nearest recall (post hoc) | 5.28 | 6 | 7.23 |
| D12, the same (post hoc) | 2.04 | 6 | 2.79 |
| multi-task + MR6 + names − multi-task + MR3, MaterioMiner fine (pre-specified) | 0.91 | 3 | 1.20 |
| the same, coarse (pre-specified) | 1.05 | 3 | 1.38 |
| ablation, 10,000 sentences − MR6 + names (3 seeds) | 1.63 | 6 | 2.23 |
| ablation, 20,000 sentences − MR6 + names (3 seeds) | 2.29 | 6 | 3.14 |
| ablation, threshold 0.7 − MR6 + names (3 seeds) | 1.47 | 6 | 2.01 |
| ablation, 2 of 3 teachers − MR6 + names (3 seeds) | 1.70 | 6 | 2.32 |
| ablation, names in half of the draws − MR6 + names (3 seeds) | 1.71 | 6 | 2.35 |

## Supplementary Note 10: the recall-matched control in detail

**Design.** Registered on 1 November at 10:51 (`e02c3be`), after an
internal review of the draft asked whether the names' recall gain on the unseen names their
lexicon holds (Section 5.4) is name knowledge or a generic shift toward
tagging more spans. The control is MR6: its stored decision jobs (400
pool sentences, six copies, seeds 300 to 900 over the seven subfields)
run again, each run decoded at every bias D of the grid {−2, −1, −0.5,
0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5}, D added to the emission score of every
tag but O before Viterbi decoding, training untouched. The names' side
is the stored MR6 + names runs, unchanged. Per subfield the names'
recall over all test entities is pooled over the seven seeds (correct
entities summed over the seeds, divided by gold entities summed), and so
is the control's at each D; D* is the first crossing of the names'
recall by the control's, going from D = 0 toward it along the grid, by
linear interpolation between the two grid values that bracket it, and
every quantity of the control at D* is the same interpolation of its grid
values (counts pooled over the seeds for the pooled rates, each run's
micro F1 for the seed mean). A subfield whose target lies outside the
control's range over the grid is matched at the nearer end of the grid
and flagged; it stays in the decisions, which are also reported without
the flagged subfields. D11 is recall on the unseen test names the
lexicon holds (Section 5.4's split), the names minus the control at D*,
and D12 micro F1, both paired over the seven subfields under the
decision rule of Section 3. The protocol expected D12 near +0.7 to +1.3
(the names' F1 over MR6 at D = 0 on these units is +0.51 [−1.51,
+2.53], and on MaterioMiner a bias from 0 to 1 had cost the CRF head
about 0.2 F1 for 2 points of recall), a pass unlikely; it predicted no
number for D11.

**Checks.** The session's repeat of an earlier CRF-head run was
identical; at D = 0 all 49 control runs equal their stored MR6 runs in
F1, precision, recall, best epoch, training history and test
predictions; the lexicons are the rebuilt ones of Section 5.4, whose
checks pass.

**The matching.** Table S8 gives the control's recall at every bias. D*
is −1.08 for graphene, where the names' recall is below MR6's, and 1.35
to 2.47 for poly, steel and semi. In electrolysis, pemfc and sofc MR6's
recall rises to a peak at D = 2.5 or 3 (59.66, 51.54 and 49.93), 0.47,
0.20 and 0.43 points short of the names' (60.13, 51.74 and 50.36), and
falls beyond it: from D = 2.5 to 5 in electrolysis the bias adds about
9% more predicted entities while the correct ones fall by 4%, spans
growing or merging until they no longer match exactly. The protocol's
rule, which presumed recall to rise with D, matched these three at D =
5, where MR6's recall is 57.22, 50.03 and 47.57 and its F1 5.3 to 7.0
points below its own at D = 0.

**Further comparisons** (pre-specified, at D*, paired over the seven
subfields). Precision on all test entities +5.78 [−1.22, +12.79]; on the
unseen test names, recall +1.12 [+0.09, +2.15], precision +6.17 [−1.27,
+13.61] and F1 +3.54 [−0.58, +7.67]; recall on the unseen names the
lexicon does not hold −1.73 [−3.02, −0.44], positive in one subfield;
the difference of the two parts' differences +12.98 [+4.18, +21.77]; the
bias's own F1 change at D*, −2.75 [−5.77, +0.27]. Without the flagged
subfields D11 is +6.05 [−3.33, +15.44], D12 −0.03 [−2.78, +2.72] and
precision +0.56 [−6.74, +7.85].

**Cost.** The control's seven jobs took 31.9 to 39.4 minutes each on a
T4, a training run and twelve decodes per subfield and seed, in one
session of 151 minutes.

**Post hoc, after the result.** Four analyses were added after the
result, the first on the day of the result and the others after an
internal review of the draft asked for them; all are labelled post hoc
in `ner_recall_matched.py` and its results file. (1) With the flagged
subfields matched at the grid value of the nearest recall (2.5, 3 and
2.5; `match_nearest`), D11 is +8.63 [+3.35, +13.90], positive in every
subfield, D12 +1.37 [−0.66, +3.41], precision +3.04 [−1.32, +7.41],
recall on the unseen names the lexicon does not hold −2.07 [−3.68,
−0.46] and on all unseen names +0.25 [−0.08, +0.58]. (2) The rates of
the control at every bias on the unseen names in and out of the lexicon
(Table S8, Figure 4): MR6's recall on the held names is highest at D ≤
0.5 in five of the seven subfields and falls as D grows, while its
recall on the other unseen names rises; at whichever bias gives MR6 its
highest recall on the held names, the names lead by +6.52 [+2.15,
+10.89], in every subfield, and on the other unseen names a bias
overtakes them, −4.16 [−5.82, −2.50] (none of seven above 0). (3) The
decomposition of D11: per subfield, the names' gain over MR6 at D = 0 on
the held names (+7.29 [+3.27, +11.30], Table 5) plus MR6's own loss on
them from D = 0 to D* (+3.96 [−1.06, +8.98]); on the four matched
subfields the two parts are +5.92 [−1.10, +12.94] and +0.14 [−4.89,
+5.16], so there matching the recall leaves the gain where it was at D =
0. (4) The split by type. Values and materials names are 83 to 99% of a
subfield's distinct lexicon names and 83 to 99.6% of its entries (Table
S10), and they are 60 to 99.6% of the unseen test names a lexicon holds
but a minority of the others (41 to 76% of those are of other types), so
a shift toward those two types could favour the held names without any
name being learned. Within types (Table S9) the materials names the
lexicon holds gain +12.02 [+3.70, +20.34] at D* (+9.87 [+3.57, +16.16]
with the flagged subfields at the nearest recall), the other materials
names −1.19 [−5.59, +3.21]; values held +9.43 [+0.02, +18.83], the other
values +3.41 [−1.97, +8.80]; and unseen names of the other types not held
−3.91 [−7.32, −0.51]. The pattern fits gains specific to the materials
names the lexicon holds, with a smaller one on values and a shift away
from the types it barely holds. The exact two-sided sign-flip p over the
seven subfields is 0.016 for D11, the smallest seven units allow, and
0.14 for D12; at 80% power D11 detects 10.87 points (12.27 without the
flagged subfields, 7.23 post hoc) and D12 5.55 (3.60 and 2.79; Table
S7); the two-sided t p-values are 0.013 for D11 and 0.096 for D12.

**Table S8.** The recall-matched control's rates at every bias D, pooled over the decision seeds, beside MR6 + names' (MuLMS's seven held-out subfields; `docs/tables/table12_week18.md`): recall, precision and F1 over all test entities (pre-specified), then recall on the unseen test names the lexicon holds and on the other unseen names (post hoc).

| Subfield | rate | names | D = −2 | D = −1 | D = −0.5 | D = 0 | D = 0.5 | D = 1 | D = 1.5 | D = 2 | D = 2.5 | D = 3 | D = 4 | D = 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| electrolysis | recall | 60.13 | 54.35 | 56.44 | 57.37 | 58.17 | 58.78 | 59.20 | 59.46 | 59.61 | 59.66 | 59.59 | 58.81 | 57.22 |
| electrolysis | precision | 56.40 | 61.91 | 59.89 | 59.04 | 58.08 | 57.18 | 56.14 | 55.04 | 53.88 | 52.66 | 51.44 | 48.86 | 46.13 |
| electrolysis | F1 | 58.21 | 57.88 | 58.11 | 58.19 | 58.12 | 57.97 | 57.63 | 57.16 | 56.60 | 55.94 | 55.22 | 53.38 | 51.08 |
| graphene | recall | 47.45 | 46.25 | 47.56 | 48.22 | 48.79 | 49.39 | 49.90 | 50.41 | 50.66 | 50.93 | 51.19 | 51.42 | 51.64 |
| graphene | precision | 54.51 | 59.56 | 57.39 | 56.55 | 55.53 | 54.62 | 53.59 | 52.76 | 51.86 | 50.90 | 50.15 | 48.41 | 46.54 |
| graphene | F1 | 50.74 | 52.07 | 52.01 | 52.06 | 51.94 | 51.87 | 51.68 | 51.56 | 51.26 | 50.92 | 50.66 | 49.87 | 48.96 |
| pemfc | recall | 51.74 | 44.18 | 47.23 | 48.31 | 49.23 | 49.93 | 50.50 | 50.93 | 51.24 | 51.41 | 51.54 | 51.15 | 50.03 |
| pemfc | precision | 54.95 | 59.94 | 58.04 | 56.80 | 55.46 | 54.24 | 53.08 | 51.92 | 50.77 | 49.63 | 48.44 | 46.14 | 43.91 |
| pemfc | F1 | 53.30 | 50.87 | 52.08 | 52.21 | 52.16 | 51.99 | 51.76 | 51.42 | 51.00 | 50.50 | 49.94 | 48.51 | 46.77 |
| poly | recall | 53.52 | 45.69 | 48.39 | 49.53 | 50.61 | 51.83 | 52.85 | 53.80 | 54.39 | 54.93 | 55.36 | 55.19 | 53.98 |
| poly | precision | 45.22 | 58.40 | 55.36 | 53.65 | 52.02 | 50.57 | 48.98 | 47.29 | 45.66 | 44.03 | 42.49 | 39.29 | 36.46 |
| poly | F1 | 49.02 | 51.27 | 51.64 | 51.51 | 51.31 | 51.19 | 50.84 | 50.34 | 49.64 | 48.88 | 48.08 | 45.90 | 43.52 |
| semi | recall | 31.78 | 21.19 | 24.14 | 25.53 | 26.82 | 27.99 | 29.09 | 30.16 | 31.01 | 31.84 | 32.37 | 32.19 | 31.27 |
| semi | precision | 50.85 | 62.73 | 58.48 | 56.21 | 54.23 | 52.03 | 49.89 | 47.86 | 45.77 | 43.83 | 41.90 | 37.50 | 33.73 |
| semi | F1 | 39.11 | 31.68 | 34.17 | 35.11 | 35.89 | 36.40 | 36.75 | 37.00 | 36.97 | 36.88 | 36.52 | 34.64 | 32.45 |
| sofc | recall | 50.36 | 40.22 | 43.79 | 45.26 | 46.55 | 47.71 | 48.59 | 49.29 | 49.85 | 49.93 | 49.87 | 49.54 | 47.57 |
| sofc | precision | 60.82 | 63.37 | 61.02 | 59.77 | 58.50 | 57.28 | 56.12 | 54.89 | 53.56 | 52.00 | 50.42 | 47.45 | 43.89 |
| sofc | F1 | 55.10 | 49.21 | 50.99 | 51.51 | 51.84 | 52.06 | 52.08 | 51.94 | 51.64 | 50.95 | 50.14 | 48.47 | 45.65 |
| steel | recall | 56.37 | 49.78 | 51.99 | 52.92 | 53.83 | 54.57 | 55.30 | 55.87 | 56.46 | 57.00 | 57.46 | 58.08 | 58.00 |
| steel | precision | 57.10 | 66.83 | 64.29 | 62.98 | 61.77 | 60.29 | 58.88 | 57.35 | 55.95 | 54.52 | 53.08 | 50.24 | 47.29 |
| steel | F1 | 56.73 | 57.06 | 57.49 | 57.51 | 57.53 | 57.29 | 57.04 | 56.60 | 56.21 | 55.73 | 55.18 | 53.88 | 52.10 |
| electrolysis | recall, held names (post hoc) | 90.02 | 82.34 | 83.17 | 83.54 | 83.62 | 83.44 | 83.10 | 82.62 | 81.91 | 81.26 | 80.60 | 78.65 | 76.73 |
| electrolysis | recall, other unseen names (post hoc) | 38.23 | 33.14 | 36.19 | 37.61 | 38.84 | 39.97 | 40.96 | 41.66 | 42.25 | 42.80 | 43.19 | 43.12 | 42.01 |
| graphene | recall, held names (post hoc) | 84.91 | 84.00 | 84.75 | 84.69 | 84.43 | 84.26 | 84.00 | 83.94 | 83.46 | 82.98 | 82.76 | 81.90 | 81.15 |
| graphene | recall, other unseen names (post hoc) | 30.09 | 27.85 | 29.73 | 30.81 | 31.66 | 32.60 | 33.53 | 34.42 | 35.00 | 35.65 | 36.24 | 36.88 | 37.54 |
| pemfc | recall, held names (post hoc) | 84.92 | 75.86 | 77.63 | 77.95 | 78.09 | 77.74 | 77.19 | 76.39 | 75.58 | 74.55 | 73.74 | 71.89 | 69.33 |
| pemfc | recall, other unseen names (post hoc) | 38.19 | 31.47 | 34.86 | 36.18 | 37.26 | 38.27 | 39.16 | 39.92 | 40.51 | 41.03 | 41.43 | 41.42 | 40.71 |
| poly | recall, held names (post hoc) | 71.22 | 61.57 | 64.51 | 65.47 | 66.58 | 67.65 | 68.65 | 69.56 | 69.83 | 70.02 | 70.07 | 69.41 | 67.52 |
| poly | recall, other unseen names (post hoc) | 38.98 | 32.08 | 34.67 | 35.91 | 36.95 | 38.31 | 39.33 | 40.35 | 41.04 | 41.77 | 42.40 | 42.43 | 41.46 |
| semi | recall, held names (post hoc) | 81.36 | 67.15 | 71.58 | 73.32 | 73.58 | 73.35 | 73.12 | 72.16 | 70.50 | 68.85 | 66.88 | 62.03 | 57.37 |
| semi | recall, other unseen names (post hoc) | 26.87 | 16.35 | 19.22 | 20.63 | 22.02 | 23.29 | 24.52 | 25.76 | 26.78 | 27.75 | 28.49 | 28.59 | 27.93 |
| sofc | recall, held names (post hoc) | 85.32 | 67.13 | 69.86 | 70.82 | 71.23 | 71.38 | 70.97 | 70.49 | 70.10 | 69.09 | 67.59 | 64.89 | 59.72 |
| sofc | recall, other unseen names (post hoc) | 42.13 | 33.55 | 37.43 | 39.06 | 40.62 | 42.03 | 43.25 | 44.22 | 45.03 | 45.31 | 45.57 | 45.75 | 44.42 |
| steel | recall, held names (post hoc) | 84.39 | 70.65 | 72.69 | 73.29 | 73.61 | 73.96 | 74.25 | 74.45 | 74.77 | 74.88 | 75.01 | 74.69 | 73.98 |
| steel | recall, other unseen names (post hoc) | 43.70 | 39.06 | 41.14 | 42.04 | 43.02 | 43.85 | 44.72 | 45.32 | 45.97 | 46.61 | 47.18 | 48.19 | 48.40 |

**Table S9.** Recall on the unseen test names by type group and lexicon membership (post hoc): the names minus the control at D*, with the flagged subfields at the nearest grid value instead, and minus MR6 at D = 0, paired over the seven subfields with the subfields above 0; the last column gives the unseen test names per subfield in each cell.

| Type | unseen names | names − control at D* | the flagged at the nearest recall | names − MR6 at D = 0 | test names per subfield |
|---|---|---|---|---|---|
| MAT | held | +12.02 [+3.70, +20.34], 7/7 | +9.87 [+3.57, +16.16], 7/7 | +7.98 [+2.15, +13.82], 7/7 | 184–1022 |
| MAT | not held | −1.19 [−5.59, +3.21], 4/7 | −1.58 [−5.61, +2.46], 2/7 | +0.89 [−3.48, +5.27], 5/7 | 129–2223 |
| VALUE | held | +9.43 [+0.02, +18.83], 6/7 | +5.61 [+0.27, +10.94], 6/7 | +1.74 [−1.16, +4.65], 4/7 | 56–313 |
| VALUE | not held | +3.41 [−1.97, +8.80], 4/7 | +1.14 [−2.30, +4.58], 3/7 | −0.23 [−3.00, +2.53], 3/7 | 106–681 |
| other | held | +2.87 [−4.82, +10.55], 5/7 | +0.94 [−5.84, +7.72], 4/7 | +9.80 [+3.34, +16.25], 6/7 | 6–577 |
| other | not held | −3.91 [−7.32, −0.51], 1/7 | −3.58 [−6.49, −0.67], 1/7 | +1.38 [−0.82, +3.58], 6/7 | 754–2516 |

**Table S10.** The rebuilt lexicons of MuLMS's subfields by type: distinct names and, in parentheses, entries (occurrences in the corpus, which set how often a name is drawn; counts only, the names are not released).

| Subfield | VALUE | MAT | PROPERTY | MEASUREMENT | INSTRUMENT | TECHNIQUE | FORM | DEV | SAMPLE | CITE |
|---|---|---|---|---|---|---|---|---|---|---|
| electrolysis | 7,168 (13,635) | 2,392 (29,748) | 506 (1,507) | 121 (676) |  | 70 (102) | 17 (405) | 15 (63) |  | 1 (1) |
| graphene | 6,089 (11,328) | 1,553 (16,412) | 28 (46) | 72 (566) | 60 (60) | 2 (2) | 7 (36) |  | 2 (3) |  |
| pemfc | 4,928 (9,062) | 357 (3,811) | 10 (11) | 3 (4) |  | 2 (2) | 3 (18) |  |  | 18 (19) |
| poly | 7,478 (15,546) | 3,664 (26,517) | 1,359 (3,098) | 316 (3,528) | 236 (239) | 192 (386) | 84 (1,438) | 37 (80) | 29 (37) |  |
| semi | 4,580 (7,429) | 132 (666) | 96 (116) | 20 (115) |  |  | 1 (1) |  |  | 1 (1) |
| sofc | 4,011 (8,359) | 265 (5,167) | 38 (41) | 42 (153) | 3 (3) |  | 20 (678) |  |  | 1 (1) |
| steel | 5,797 (12,164) | 1,492 (15,248) | 439 (791) | 224 (1,485) | 23 (24) | 35 (41) | 48 (571) | 3 (3) |  | 5 (5) |

## Supplementary Note 11: checks, deviations, failures and timings of the names route

This Note gives, week by week, the checks the runs of the names route
passed, where the runs departed from what the protocols planned, which
sessions failed, and when each of these happened; commit times are the
commits' (UTC) and push, pull and run times the session notes', as in
Supplementary Note 1.

**Week 10.** The protocol of 17 October 20:21 (`a08356f`) listed the
analyses week 10 would report, and looser matching was not among them.
The looser-matching script had been introduced on 16 October as an
exploratory analysis. It was adapted to MuLMS's protocols on 17 October
at 21:43 (`6979420`), after `w10a` had been pushed at 20:57, and at
21:55, before the results, a dry run on files from a tiny random model
trained on MuLMS's real folds and splits went through the matching
analysis. Week 10's result was recorded on 18 October at 00:48
(`c2ca3f6`). On the held-out subfields the CRF head scored 48.66 with
span and type exact, 54.59 on the span alone and 69.06 under any
overlap, against 65.72, 70.22 and 81.68 on the random pool splits, so
the gap of 17.06 points fell to 12.62 under overlap. Of its predictions
on the held-out subfields, 13.1% lay inside a longer gold entity,
against 9.0% on the random splits.

**Week 11.** In `w11a` (MuLMS; pushed on 18 October at 01:50, done at
04:44), per subfield both lanes' three teachers and silver counts were
identical, and every teacher scored exactly as the stored week-10
CRF-head run of its seed. In `w11b` (MaterioMiner) each lane ran one
granularity, and per paper and granularity the three teachers scored
exactly as the stored CRF-head runs of their seeds. That a student
trained without silver names equals the `crf_mr` run of its seed was
shown by a test on a tiny model, committed with the week's preparation
on 18 October at 01:12 (`50c07dc`), not by a GPU run. At 01:32, before
the pushes, one `w11a` job (graphene) and one `w11b` job
(coarse-grained, paper s11340) ran on the real data with a tiny random
model on the CPU; in the `w11a` job the random model found no silver
sentence and the teacher of seed 0 reproduced week 10's tiny-model CRF
run on graphene (1.11), and the `w11b` job's 963 copies equalled those
of week 7's mention-replacement run of seed 0 on that paper. On a tiny
random model, every existing option that the week-11 change touched
wrote byte-identical results files.

The protocol of 18 October 01:13 (`427e34c`) ran `w11b` after the weekly
GPU quota's reset on 24 October, and `w11a` left about 1 GPU hour until
then. The quota was raised to 45 hours a week on 19 October, and `w11b`
was pushed that day at 21:06 to `ner-gpu-jobs-d` (version 6), before the
reset. It ran 3.4 hours (9 jobs), and its result, decisions 3 and 4 of
week 11 failing, was recorded on 20 October at 00:44 (`04764ef`). The
protocol named its own commit as every batch's source; `w11b` ran from
bundle `77c37be`, the commit of week 12's first amendment, and its files
carry the `silver_filter` field that the week-12 code added, which
`w11a`'s files lack. The week-12 code of 18 October 11:00 (`40631ce`)
had been checked to repeat the week-11 code's runs byte for byte apart
from `runtime_sec`, its silver-name runs differing only by that field;
no such check of `77c37be`'s own change to `ner_baseline.py` is
recorded.

On 18 October at 04:52 (`fa64ed6`), after `w11a`'s result had been
committed at 04:50, the plan made `w11b` the gate of the route: topic
corpora for MuLMS only if decision 3 or 4 of week 11 passed, and
otherwise the evaluation paper. Phase 0, measured on the CPU from 09:14
that day (`8b9dd59`), was to decide whether topic corpora were worth the
GPU hours. The week-12 protocol, committed at 11:00 (`40631ce`), made
phase 0's coverage the gate instead of `w11b`, which it said could not
run before the reset, and kept `w11b` first with its decisions as fixed.
When `w11b`'s failure was recorded, at 00:44 on 20 October, the week-12
protocol and its first two amendments were in place, and the week-12
screen was pushed at 02:35. Under the plan of 04:52 that failure would
have led to the evaluation paper.

**Week 12.** Phase 0's first text, on 18 October at 09:14 (`8b9dd59`),
tested only the SOFC corpus of the week-6 amendment, removed its overlap
with MuLMS's sofc documents by a 30-character shingle (665 sentences,
read then as some of those documents being among its articles), and put
about 45 points of the coverage down to generic materials names and
about 35 points to the topic. At 11:00 (`40631ce`), with the seven topic
corpora built, the generic part was 45–65 points and the own topic added
4–11 points on six subfields, while on graphene the own corpus was 3
points behind the electrolysis corpus. The 30-character shingle had
matched technical terms in 270–319 of each topic's articles and dropped
62–224 articles per topic whole; it was widened to 50 characters, at
which 15 sentences and no article of the SOFC pool shared a run with
MuLMS, and at 60 none. No sentence of a topic corpus shared a
50-character run with a held-out subfield's documents; the SOFC and
fatigue corpora, used as controls, shared 0 to 5 sentences with a
subfield's documents, which the counts left out. MuLMS ships no article
identifiers: 16 of its 50 documents were found on Europe PMC by title
and excluded by PMCID, the rest coming from journals Europe PMC does not
index, and every sentence sharing an 8-word n-gram or a 50-character run
with a MuLMS sentence was dropped as well. For MuLMS, therefore, the
exclusion "by identifier" of Section 4 and Supplementary Note 3 covered
16 of its 50 articles.

Amendment 1 was committed on 19 October at 17:15 (`77c37be`), before any
run, adding V4 (`crf_mrm_topic6`) and V6 (`crf_st3_topic`) to the
screen. Amendment 2, V7 (`crf_gzm_topic`), was proposed at 20:54
(`48f37d3`) and recorded as not adopted and the user's call. It was
adopted nine minutes later, at 21:03 (`8a18962`), before any run.
Amendment 3, V8 (`crf_gzs_topic`), was committed on 20 October at 01:50
(`47e2310`) on the CPU probe's first six subfields, graphene pending,
and kept the threshold at 0.9. The bundle rebuilt from `47e2310` was
ready at 02:02, and the screen was pushed at 02:35 without waiting for
graphene. Graphene's probe and the seven-subfield reading were committed
at 07:01 (`dcbff64`). On graphene the sentence filter's 0.9 lexicon
filled to −2.86, so that over the seven subfields it averaged +1.1 and
reached −2.9, where amendment 3 had priced it on six at +1.8 on average
and never below −0.3. The seven-subfield reading changed neither filter
nor threshold.

The protocol required, before any number was used, identical smoke
repeats, each job's teachers equal to week 10's `crf_m0`, `_m100` and
`_m200` on their subfield, the bundle's corpora matching the reports'
SHA-256 (checked again by the runner), and every run recording
`silver_filter` and `names`. Every teacher of every job equalled the
stored CRF-head run of its seed in `w12c` (version 4), in `w12e` and in
`w12b:mrm6`, and on MaterioMiner every teacher of `w12d:mrm6` equalled
the stored week-5 run of its seed on its paper. From the resume fix on,
every run recorded its own corpus's SHA-256. At the reset of 24 October
the bundle's `BUNDLE.json` named `3ac85a8`, the SOFC corpus's SHA-256
(1b6a00d2) and the seven topic corpora, and `ner_baseline.py`,
`ner_consistency.py` and `kaggle/` were unchanged since, so the decision
batches ran from it.

On a tiny model and the tests' toy corpora, the code of 18 October 11:00
(`40631ce`) repeated the week-11 code's runs byte for byte apart from
`runtime_sec` for the CRF head, MR3, multi-task MR3, self-training and
MuLMS's pool and held-out runs, and its silver-name runs differed only
by the added `silver_filter` field, as recorded at 11:05 (`bd20cec`). On
19 October the XLA port's CPU path repeated the pre-port files byte for
byte apart from `runtime_sec` on every tiny-model option. With amendment
2's feature, the paths it did not touch (`crf`, `mr`, `mt_mr`, `st3`,
`mrs`, the MuLMS pool and unused-corpus runs) were byte-identical to
`77c37be` on the tiny corpora. With the resume fix of 20 October
(`3ac85a8`), the files of the other protocols were byte-identical to
`77c37be` in 8 configurations, and a tiny end-to-end run showed two
subfields extending one file, a repeat resuming and another corpus
refused.

The screen was pushed on 20 October at 02:35 from bundle `47e2310`,
`w12a` to `ner-gpu-jobs-d` (version 7) and `w12c` to `ner-gpu-jobs-c`
(version 3). Between 03:56 and 04:10 both failed on the resume check: a
results file stored one corpus SHA-256, each held-out subfield had its
own topic corpus, and the second subfield's job refused to resume a file
"from another corpus". `w12c` completed in 59 minutes with 4 of its 14
topic jobs done and 10 failed in 0.01 minutes; `w12a` lost every names
job but the first of each variant, while its `crf_mr6` jobs ran. `w12c`
had also put each feature variant's jobs in both lanes, which would have
had the lanes write the same files at once. The fix, committed at 04:09
(`3ac85a8`), made a run with `--subfields` record its own corpus, kept
other subfields' runs in a file, resumed a subfield's run only from the
same corpus and put one variant per lane. The bundle was rebuilt from
`3ac85a8` at 04:20, and `w12c` was pushed again at 04:49 (version 4).
The first `w12a` session ended at 05:03 (137.6 minutes, 2.3 quota hours)
with the smoke repeat, electrolysis for V1, V2, V4 and V6 (29–38 minutes
each) and the three `crf_mr6` jobs, and was pulled. Of its files only
the three `crf_mr6` files were collected, at 08:06; V1, V2, V4 and V6
ran again on all seven subfields, electrolysis included, as `w12e`.
Electrolysis runs of V4, MR6 + names, on the screen's seeds had thus
been pulled, though not collected or recorded, before the criterion was
revised at 09:14.

`w12e` was pushed at 05:03 to `ner-gpu-jobs-d` (version 8) before the
quota was checked against the job times: it needed about 8.1 hours and
`w12c` 3.1, against 9.14 hours left in the week, and the notes expected
Kaggle to stop a session when the quota ran out and keep none of its
outputs. The user was asked to cancel version 8, and the plan, committed
at 05:06 (`9e9d906`), split the names variants into `w12f` (V1 and V2),
to run in its place, and `w12g` (V4 and V6), to run after the reset of
24 October with the decision batch. At 08:10 the quota left 2.9 hours,
the notes recorded that cancelling no longer changed the plan, and
`w12e` was to run after the reset. The week-13 protocol, committed at
09:14, also put `w12e` after the reset, beside `w13a`. `w12e` was not
cancelled and `w12f` was never pushed: the next sessions on either
kernel were those of 24 October, kernel d's version 9 and kernel c's
version 5. Kaggle did not stop the session: the quota read 43.99 hours
used at 10:10, 45.25 at 11:25 with the session running, and 46.23 of 45
at 13:23, when `w12e` completed (440 minutes, 28 jobs). The 1.23-hour
overrun was not carried into the next week. V4's screen result was
therefore recorded on 20 October at 13:35 (`5f844bd`), before the
amendments of 13:53 (`5f7f827`) and 18:42 (`de62446`); under the plans
of 05:03, 08:10 and 09:14 it would have come after the reset of 24
October.

The reset of 24 October reached the session at 12:16, 12 hours late,
because the container had restarted. `w12b:mrm6`, pushed at 12:22
(`ner-gpu-jobs-d`, version 9), failed without a trace at 13:44, after 82
minutes: status ERROR, an empty log, no manifest and no outputs, the
API's failure message empty, about 1.4 GPU hours charged. It was pushed
again unchanged at 13:49 (version 10), with its live log kept every 20
minutes, and completed at 18:09 (259.8 minutes). `w12d:mrm6`, pushed at
18:10 (version 11), still had an empty live log at 20:05 and was taken
to have hung before its setup. Version 11 ended in ERROR at 20:11 with
an empty log and no manifest; the quota then read 12.05 hours used, 1.8
fewer than at 20:04, so its hours were not charged. It was pushed again
at 20:12 (version 12) and completed on 25 October at 00:55 (276.1
minutes).

A session note headed 20:25 on 24 October, committed at 20:18
(`2c2409d`) and on GitHub at 20:18:40, before `w12d:mrm6`'s results,
worked out what D5 needed. V4's 10-seed MuLMS units against the CRF head
were +2.39, +1.75, +2.94, −2.76, +6.20, +12.16 and −0.40 (mean +3.18,
standard deviation 4.84). With the four MaterioMiner papers spread as
MR3's were there (standard deviation 0.8), the pooled 95% interval
cleared 0 when the analogue's mean gain over the CRF head was about +1.6
or more: at +1.77 the 11 units gave +2.67 [+0.09, +5.25], at +1.27,
+2.49 [−0.13, +5.11]. The note added that the decision was the
protocol's, whatever these scenarios said; the analogue gained +4.27.

The protocol of 18 October stated that, with the between-subfield
standard deviations of weeks 10 and 11 (2.6 and 3.1 points), D1 would
pass only for a mean gain of about 2.4–2.9 points or more. The screen's
reading as committed at 13:35 on 20 October applied that bar: D2 needed
a mean gain near 2.4–2.9, more than the screen's 1.45, while D1's +3.16
was above it. At 18:42 (`de62446`) the reading was redone with V4's own
spread: against the CRF head the standard deviation was 5.5 (sofc
+12.85, poly −3.72, steel −2.00), so the half-width over 7 subfields was
5.1 and D1 needed a mean near 5, and D2 (standard deviation 3.0,
half-width 2.8) a mean near 3, bars that the screen's +3.16 and +1.45
did not reach. The same reading put D5's half-width near 2.9 for a
pooled mean near 3.0 ([+0.1, +5.9] with the screen's values), with MR3's
spread on the four MaterioMiner papers (0.8) as a proxy.

**Week 13.** On 24 October, before `w13a` was pushed, the stored T4 runs
showed SOFC-Exp's article folds costing more than the protocol's 4.5 and
7.5 lane hours: the CRF head had taken 2.9 hours for its 10 seeds on the
5 folds, and on the official split MR3 took 3.1 times the CRF head's run
time and multi-task MR3 4.3 times, so about 8.8 and 12.5 lane hours. One
session of both lanes would have put the multi-task lane past the
runner's 10-hour deadline, after which no job starts and a running job
is stopped 1.25 hours later. `w13a` was therefore split: it ran MR3 over
both lanes, and a new batch, `w13c`, ran multi-task MR3, one after the
other on `ner-gpu-jobs-c`, with the runs, their files and the analysis
unchanged. The split was committed at 12:23:09 (`d6db00b`), in the
minute `w13a` was pushed (version 5); its note is headed 12:30. `w13a`
took 293.4 minutes and `w13c` 549.1, its multi-task jobs about 112
minutes each instead of the 75 estimated. In `w13b` (`ner-gpu-jobs-d`,
version 13; pushed on 25 October at 01:01, done at 11:24 after 619.8
minutes), in both lanes every fold's three teachers equalled the stored
week-8 CRF-head runs of their seeds, and the two lanes harvested the
same silver mentions per fold (21,376–48,943, 4,835–9,290 distinct,
20.3–33.4% of the fold's unseen test names). Lane 0's job took 615.2
minutes and lane 1's 591.6, inside the 11.25-hour stop, so no fold was
cut; amendment 13.4 had costed the batch at about 7.5 hours and the push
note at about 8.8.

**Week 14.** The protocol of 29 October 12:07 (`2886267`) ran `w14b`
after the reset of 31 October. With 6.5 of the week's 45 hours unused
after `w14a` and both kernels idle, `w14b` was pushed on 29 October at
18:00 (`ner-gpu-jobs-d`, version 14) from the bundle of `8919928`, the
notes recording a change of schedule only. It ran from 18:00 to 22:20
(4.3 session hours). In `w14b` every lane's teachers equalled week 10's
on their subfield, and every run carried its mixed corpus's SHA-256 and
the mention filter. In `w14c` (31 October, 00:14 to about 09:55, 9.7
session hours) each variant's teachers equalled week 10's on their
subfield, and every run carried its topic corpus's SHA-256, the mention
filter and the setting it changed. The three new options defaulted to
the earlier behaviour byte for byte on the tiny model.

The protocol listed what it would not run, and why: V4 on the multi-task
model (MaterioMiner only, the names typed at both levels, no second
corpus to repeat it on); the copy control on SOFC-Exp's article folds
(about 13 lane hours on a corpus where the method does not pass); the
names as a match feature (screened in week 12 as V7 and V8, below
mention replacement); and the names as a dictionary. The names on the
multi-task model were run after the write-up of D1 to D8, in the
follow-up of 31 October (D9, D10). MR6 was not run on SOFC-Exp's folds.

For a failing D8 the protocol fixed this reading: "A fail with the mixed
variant above the CRF head (reported) means a large materials corpus of
other topics does as well, a simpler method the paper then states, with
13.2 as its decision on the topic corpora and this as the reported
comparison; a fail with neither above the other means 7 units cannot
separate them, which the paper says". At 12:32 (`102144c`), before any
run of `w14b`, the protocol added the mixed corpora's coverage of the
unseen test names (49–70%, against the topic corpora's 55–81%) and the
sentence "So the mixed corpora hold most of what the topic corpora hold,
which is what 14.2 tests: whether the names the topic adds, and the
contexts they come in, are what the gain needs". D8 failed, −0.18
[−2.01, +1.65], with the mixed variant at +3.38 [−0.89, +7.64] against
the CRF head, and the outcome note of 22:38 read it by the first
reading: "the topic's text is not what the gain needs; a large unlabeled
materials corpus of other topics gives the same names-and-copies gain".
Section 5.3 now frames D8 thus: "since the mixed corpus holds most of
the same names, D8 tests the names only the topic corpus holds, not an
off-topic source (a framing added after the result)". That framing
entered the paper in review round 6 of 31 October, which cited the
protocol's sentence of 12:32.

**Week 17.** `smokebd`, a repeat of week 9's stored seed-0 `crf_mt_mr`
run on paper s11340, was identical on both heads (54.05 fine-grained,
62.29 coarse-grained, 887 augmented sentences, the same predictions), so
the week-17 code had left the plain multi-task and mention-replacement
paths as they were. Every names job's three teachers equalled the stored
`crf_mt` runs' test F1, validation F1 and best epoch on both heads. Two
fixes were made before any week-17 number was read: `collect --merge`
refused the files because each session's copy carried the means of its
own runs (`test_f1_mean` and the like), which are now recomputed with
the summary; and the `week17` analysis looked for the multi-task
reference's seed-0 file under a tag it does not carry (`crf_mt.json`),
so its section came out empty on the first run, and it now uses
`seed_family`, as `ner_mentions` does. The merge then united the two
sessions' student files, four papers per file, leaving the committed
`w17b` runs unchanged. The note of 1 November 00:30 first gave 49–65% as
the single-task lexicons' share of a paper's unseen test names; that was
the corpus's coverage, and the note was corrected on 1 November to
8.4–18.6% fine-grained and 15.5–23.3% coarse-grained.

**The odds stated in the methods plan.** The plan of 19 October 17:08
(`09ac262`) put D1 on MuLMS at about 0.45, D2 at about 0.3, the
MaterioMiner analogue given a MuLMS pass at about 0.6 and a methods
paper at about 0.35–0.4. On 20 October at 08:13 (`25690bc`), after the
CPU probe and the screen's first half and before the criterion was
revised at 09:14, it put D1 for one of V1, V2, V4 and V6 at about 0.3,
D2 at about 0.15 and a methods paper at about 0.2. The revision of 09:14
(`808ff4b`) changed no odds. At 13:35 (`5f844bd`), with V4 the screen's
winner, the plan put D1 at about 0.5, D2 at about 0.25, the analogue
given D1 at about 0.5 and a methods paper through the route at about
0.25, and at 13:53 (`5f7f827`) it added a pass on SOFC-Exp's folds given
D1 at about 0.3. At 18:42 (`de62446`) those figures were replaced: D1
about 0.3, D2 about 0.15, the analogue against the CRF head about 0.5,
D5 about 0.4, a methods paper through the route about 0.3, and
SOFC-Exp's folds about 0.3 given a pass of D1 or D5. The current text
keeps the figures of 19 October, of 08:13 and of 18:42; those of 13:35
and 13:53 survive only in the history. On 24 October at 18:15
(`89b7261`), after D1 and D2 had failed, the plan put D5 at about 0.5,
passing if the analogue gained about +2 or more, and a methods paper
through the names method at about 0.35. After D5 passed (25 October
01:11, `3432988`) it put a pass on SOFC-Exp's folds at about 0.3, and
after D7 failed (29 October 17:38, `8919928`) a pass of D8 at about 0.3.

**Sessions that failed, hung, lost jobs or were pushed again (weeks 10
to 18).**

1. 20 October, `w12a`, `ner-gpu-jobs-d` version 7 (pushed 02:35, bundle
   `47e2310`): lost every names job but the first of each variant to the
   resume check, and ended at 05:03 after 137.6 minutes; its names jobs
   were pushed again as `w12e`.
2. 20 October, `w12c`, `ner-gpu-jobs-c` version 3 (pushed 02:35, bundle
   `47e2310`): completed in 59 minutes with 4 of its 14 topic jobs done
   and 10 failed in 0.01 minutes; pushed again at 04:49 as version 4
   from bundle `3ac85a8`, which ran all 15 jobs by 08:06 (196.7
   minutes).
3. 20 October, `w12e`, `ner-gpu-jobs-d` version 8 (pushed 05:03, bundle
   `3ac85a8`): the second push of `w12a`'s names jobs, made before the
   quota was checked; the user was asked to cancel it, it was not
   cancelled, and it completed at 13:23 (440 minutes) at 46.23 of the
   week's 45 quota hours.
4. 24 October, `w12b:mrm6`, `ner-gpu-jobs-d` version 9 (pushed 12:22):
   failed without a trace at 13:44 after 82 minutes, about 1.4 GPU hours
   charged; pushed again unchanged at 13:49 as version 10, which
   completed at 18:09.
5. 24 October, `w12d:mrm6`, `ner-gpu-jobs-d` version 11 (pushed 18:10):
   hung before its setup and ended in ERROR at 20:11, uncharged; pushed
   again at 20:12 as version 12, which completed on 25 October at 00:55.

Every other GPU session of weeks 10 to 18 was pushed once and completed
its jobs: on `ner-gpu-jobs-d`, versions 4 (`w10a`), 5 (`w11a`), 6
(`w11b`), 13 (`w13b`), 14 (`w14b`), 15 (`w17a`) and 16 (`w18a`); on
`ner-gpu-jobs-c`, versions 5 (`w13a`), 6 (`w13c`), 7 (`w14a`), 8
(`w14c`), 9 (`w15a`) and 10 (`w17b`). The list leaves out the CPU probe,
whose container restarted at 02:20, 03:10 and 04:00 on 20 October,
killing the graphene probe each time, and the six TPU sessions of 19
October, to none of which a TPU was attached.
