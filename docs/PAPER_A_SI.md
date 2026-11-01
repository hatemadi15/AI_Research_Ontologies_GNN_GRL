# Supplementary Information

**Names mined from unlabeled papers are not shown to improve data augmentation for materials named-entity recognition: a pre-registered test**

Hatem Madi

Supplementary Notes 1 to 13 give the protocol record, the hyperparameters,
compute and repeats, the corpora, and the analyses the main text
summarises; Tables S1 to S15 are numbered in order of appearance. Section,
table and figure numbers without an S refer to the main text.

## Supplementary Note 1: the protocol record

The repository names its protocols by the stage of the project in which
they were fixed, a "week"; Table S1's first column gives that name with
each protocol's decisions, and the main text cites the protocols by date.
The last rows give the companion study's protocols whose results the main
text or these Notes quote; their decisions are not among the twelve of Table 2 or the
twenty of Section 6.

**Table S1.** The protocol record of the names route, then of the
companion study's results quoted in the main text or these Notes: when each protocol was
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
| companion, week 5: multi-task training against the CRF head on MaterioMiner's held-out papers, pre-specified without decision (fine-grained +3.72 [+0.57, +6.87]; multi-task training's own decision was on the random splits); and self-training on the fatigue corpus against the CRF head there, a decision that failed (+1.74 [−1.16, +4.64] fine-grained, +1.56 [−0.88, +4.00] coarse-grained; its votes too), which the names route reports again as V6's analogue (Table S11, rows 41–44) | 15 October 15:50 (`6f2dc11`) | 15 October | 15 October 23:20 (`1b30d82`) | — |
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
that completed, except two, began by repeating an earlier run, and every
repeat was identical (the sessions that failed or hung, Supplementary
Note 11, left no record of their jobs): one batch of 15 October (`w5a`, run 16:05 to 22:10 UTC) ran
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
excluded by identifier (for MuLMS, which ships none, the 16 of its 50
articles found on Europe PMC; Supplementary Note 11), every sentence
sharing a word 8-gram with a dataset sentence dropped (and, for MuLMS, any 50-character run),
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
comparison of the analysis of Section 5.4, and Supplementary Note 13 each
unit's gains and the unseen-name trade of every system against the one it
builds on, in F1, recall and precision. A test name is unseen by main
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
  only. Supplementary Note 13 gives it per subfield with the sentence
  filter's lexicon (+1.13 on average) and the gold unseen names as a
  ceiling (+2.68 filling the head's O tags, +11.18 overriding them).
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
replacement, −0.11 [−0.69, +0.46]; both failing). Of the twenty, five
passed: D3, D4 and D5 against the CRF head, D11 on recall, and mention
replacement's votes against the CRF head's votes. Of the twelve of Table
2, the seven against a mention-replacement control or on the names'
source are D2, D7, D8, D9, D10, D11 and D12; only D11 passes. The
label-propagation rule's decisions are not counted, the rule being a
post-processing rule on saved predictions rather than a use of names;
four of them fall in the window (20 October, above), and with them the
count is twenty-four, six passing (13.1a among them), about 0.6 expected
by chance. The companion study's decisions, all earlier, are not counted
either.

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
in 7 of the 8 paper–granularity cells (per paper 6.5 to 16.3% of the
unseen fine-grained test names and 6.7 to 24.6% of the coarse-grained,
against 8.4 to 18.6% and 15.5 to 23.3%); Table 5's 9.8% and 18.2% count
by main Section 2's words-only rule.

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
5, where MR6's recall is 57.22, 50.03 and 47.57 and its seed-mean F1 5.3
to 7.0 points below its own at D = 0 (5.4 to 7.0 in Table S8's pooled
F1).

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
(Table S8; Figure 4 shows those in the lexicon): MR6's recall on the held names is highest at D ≤
0.5 in five of the seven subfields and falls as D grows, while its
recall on the other unseen names rises; at whichever bias gives MR6 its
highest recall on the held names, the names lead by +6.52 [+2.15,
+10.89], in every subfield, and on the other unseen names a bias
overtakes them, −4.16 [−5.82, −2.50] (none of seven above 0). (3) The
decomposition of D11: per subfield, the names' gain over MR6 at D = 0 on
the held names (+7.29 [+3.27, +11.30], Table 5) plus MR6's own loss on
them from D = 0 to D* (+3.96 [−1.06, +8.98]); on the four matched
subfields the two parts are +5.92 [−1.10, +12.94] and +0.14 [−4.89,
+5.16], so there matching the recall leaves the gain where it was at
D = 0. (4) The split by type (Table S9). Values and materials names are 83
to 99% of a subfield's distinct lexicon names and 83 to 99.6% of its
entries (Table S10), and they are 60 to 99.6% of the unseen test names a
lexicon holds and 24 to 59% of the others (a minority in five of the
seven subfields, 46% pooled), so a shift toward those two types could
favour the held names without any name being learned. At D = 0, where no
matching is involved, the materials names the lexicon holds gain +7.98
[+2.15, +13.82] over MR6 (7 of 7 subfields) and the other materials
names +0.89 [−3.48, +5.27]; values held +1.74 [−1.16, +4.65], the other
values −0.23 [−3.00, +2.53]; names of the other types held +9.80 [+3.34,
+16.25], from cells of 6 to 577 names, and not held +1.38 [−0.82,
+3.58]. Of the held names recalled beyond MR6, 68% are materials names,
27% of the other types and 5% values (per seed, summed over the
subfields). At D* the same split reads +12.02 [+3.70, +20.34] for the
held materials names (+9.87 [+3.57, +16.16] with the flagged subfields
at the nearest recall), −1.19 [−5.59, +3.21] for the other materials
names, +9.43 [+0.02, +18.83] and +3.41 [−1.97, +8.80] for values, and
−3.91 [−7.32, −0.51] for the other types not held, the matched bias
adding its own loss on the held names. The pattern fits gains specific
to the names the lexicon holds, chiefly materials names, with little on
values; the shift away from the types the lexicon barely holds appears
only against the bias-shifted control. The exact two-sided sign-flip p over the
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

**Table S9.** Recall on the unseen test names by type group and lexicon membership (post hoc): the names minus the control at D*, with the flagged subfields at the nearest grid value instead, and minus MR6 at D = 0, paired over the seven subfields with the subfields above 0; then the unseen test names per subfield in each cell, and the names MR6 + names recalls beyond MR6 at D = 0 in each cell, per seed summed over the subfields (each subfield's gain times its names), with the cell's share of the held or the other names' total.

| Type | unseen names | names − control at D* | the flagged at the nearest recall | names − MR6 at D = 0 | test names per subfield | names recalled beyond MR6 at D = 0, per seed |
|---|---|---|---|---|---|---|
| MAT | held | +12.02 [+3.70, +20.34], 7/7 | +9.87 [+3.57, +16.16], 7/7 | +7.98 [+2.15, +13.82], 7/7 | 184–1022 | +316 (68%) |
| MAT | not held | −1.19 [−5.59, +3.21], 4/7 | −1.58 [−5.61, +2.46], 2/7 | +0.89 [−3.48, +5.27], 5/7 | 129–2223 | +175 (46%) |
| VALUE | held | +9.43 [+0.02, +18.83], 6/7 | +5.61 [+0.27, +10.94], 6/7 | +1.74 [−1.16, +4.65], 4/7 | 56–313 | +23 (5%) |
| VALUE | not held | +3.41 [−1.97, +8.80], 4/7 | +1.14 [−2.30, +4.58], 3/7 | −0.23 [−3.00, +2.53], 3/7 | 106–681 | −17 (−5%) |
| other | held | +2.87 [−4.82, +10.55], 5/7 | +0.94 [−5.84, +7.72], 4/7 | +9.80 [+3.34, +16.25], 6/7 | 6–577 | +126 (27%) |
| other | not held | −3.91 [−7.32, −0.51], 1/7 | −3.58 [−6.49, −0.67], 1/7 | +1.38 [−0.82, +3.58], 6/7 | 754–2516 | +222 (59%) |

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
04:44), per subfield both lanes' (a lane is one of a session's two T4
GPUs, each running its own list of jobs) three teachers and silver counts were
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
The first `w12a` session ended at 04:53 (137.6 minutes, 2.3 quota hours)
with the smoke repeat, electrolysis for V1, V2, V4 and V6 (29–38 minutes
each) and the three `crf_mr6` jobs, and was pulled at 05:03. Of its files only
the three `crf_mr6` files were collected, at 08:06; V1, V2, V4 and V6
ran again on all seven subfields, electrolysis included, as `w12e`.
Electrolysis runs of V4, MR6 + names, on the screen's seeds had thus
been pulled, though not collected or recorded, before the criterion was
revised at 09:14. The record does not show whether their scores were
read. Compared on 1 November with the pulled files, each of those twelve
runs (V1, V2, V4 and V6 on electrolysis, three seeds each) equals its
rerun in `w12e` and the committed file in every field but the run time
and environment, predictions included; V4's electrolysis gain over the
CRF head on those seeds is +2.06, below MR6's +2.97 (Table S12, rows 26
and 24).

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
at 13:23, when `w12e` was found complete; by its manifest it had ended
at 12:25 (441 minutes, 28 jobs after the repeat). The 1.23-hour
overrun was not carried into the next week. V4's screen result was
therefore recorded on 20 October at 13:35 (`5f844bd`), before the
amendments of 13:53 (`5f7f827`) and 18:42 (`de62446`); under the plans
committed at 05:06, 08:11 and 09:14 it would have come after the reset of 24
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
at 20:12 (version 12) and completed on 25 October at 00:48 (276.1
minutes; pulled at 00:56).

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
time and multi-task MR3 4.3 times, which the note put at about 8.8 and
12.5 lane hours. One
session of both lanes would have put the multi-task lane past the
runner's 10-hour deadline, after which no job starts and a running job
is stopped 1.25 hours later. `w13a` was therefore split: it ran MR3 over
both lanes, and a new batch, `w13c`, ran multi-task MR3, one after the
other on `ner-gpu-jobs-c`, with the runs, their files and the analysis
unchanged. The split was committed at 12:23:09 (`d6db00b`), in the
minute `w13a` was pushed (version 5); its note is headed 12:30. `w13a`
took 293.4 minutes and `w13c` 549.1, its multi-task jobs about 112
minutes each instead of the 75 estimated. In `w13b` (`ner-gpu-jobs-d`,
version 13; pushed on 25 October at 01:01, done at 11:21 after 619.8
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
A narrower framing, citing the protocol's sentence of 12:32, entered the
paper in review round 6 of 31 October as one added after the result;
Section 5.3 now reads the fail by both: the first reading's condition
holds on the point estimate but not on the interval, "does as well"
being an equivalence D8 cannot establish, and the amendment of 12:32,
made before any run, had already framed D8 as testing whether the names
the topic adds are what the gain needs.

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
   resume check, and ended at 04:53 after 137.6 minutes (pulled at
   05:03); its names jobs
   were pushed again as `w12e`.
2. 20 October, `w12c`, `ner-gpu-jobs-c` version 3 (pushed 02:35, bundle
   `47e2310`): completed in 59 minutes with 4 of its 14 topic jobs done
   and 10 failed in 0.01 minutes; pushed again at 04:49 as version 4
   from bundle `3ac85a8`, which ran all 15 jobs by 08:06 (196.7
   minutes).
3. 20 October, `w12e`, `ner-gpu-jobs-d` version 8 (pushed 05:03, bundle
   `3ac85a8`): the second push of `w12a`'s names jobs, made before the
   quota was checked; the user was asked to cancel it, it was not
   cancelled, and it ended at 12:25 (441 minutes), found complete at
   13:23 at 46.23 of the week's 45 quota hours.
4. 24 October, `w12b:mrm6`, `ner-gpu-jobs-d` version 9 (pushed 12:22):
   failed without a trace at 13:44 after 82 minutes, about 1.4 GPU hours
   charged; pushed again unchanged at 13:49 as version 10, which
   completed at 18:09.
5. 24 October, `w12d:mrm6`, `ner-gpu-jobs-d` version 11 (pushed 18:10):
   hung before its setup and ended in ERROR at 20:11, uncharged; pushed
   again at 20:12 as version 12, which completed on 25 October at 00:48.

Every other GPU session of weeks 10 to 18 was pushed once and completed
its jobs: on `ner-gpu-jobs-d`, versions 4 (`w10a`), 5 (`w11a`), 6
(`w11b`), 13 (`w13b`), 14 (`w14b`), 15 (`w17a`) and 16 (`w18a`); on
`ner-gpu-jobs-c`, versions 5 (`w13a`), 6 (`w13c`), 7 (`w14a`), 8
(`w14c`), 9 (`w15a`) and 10 (`w17b`). The list leaves out the CPU probe,
whose container restarted at 02:20, 03:10 and 04:00 on 20 October,
killing the graphene probe each time, and the six TPU sessions of 19
October, to none of which a TPU was attached.

## Supplementary Note 12: every comparison of the names route

Table S11 lists every comparison of the protocols of 17 October (20:21)
to 1 November that the results files hold, 162 rows in all (160 paired
comparisons, a gap between unpaired settings and a level), one each:
the twelve decisions of Table 2 and the other decisions of those
protocols, the screen, every pre-specified comparison and the post hoc
ones. Each row gives micro F1 (or the measure the comparison names:
recall, as in D11, precision, or F1 on the seen or unseen test names)
and, where the results hold them, macro F1 (every type weighted
equally), weighted F1 (types weighted by their test entities), strict
IOB2 scoring and the votes (the entity-level majority of the three
models of each seed triple). Only micro F1 decides, except D11, on
recall, week 10's decision on the votes (10.2) and, in the rows of V6's
analogue, the companion study's week-5 decision on the votes. The macro
and weighted cells of the 11-unit pooled rows are post hoc, recomputed
from the run files for this revision (`ner_robustness.py`,
`pooled_measures`, whose micro values equal the rows' to within 0.01).
A row's week is the protocol that named the comparison,
with the commit time of the protocol or amendment; a post hoc row gives
the protocol it re-reads. Strict IOB2 scoring gives the default
scoring's numbers in every row that stores both. Macro F1's interval
clears 0 where micro F1's does not in 26 rows: 23 per-corpus
comparisons, all on MuLMS and every one against the CRF head (13) or MR3
(10), 13 of them on the screen's three seeds, among them D1 (+3.78
[+2.47, +5.10]) and MR3's own decision of 17 October (+2.12 [+1.14,
+3.09]); and 3 pooled post hoc rows, the names against MR3 on both seed
bases (+1.33 [+0.29, +2.36] on the decision seeds, +1.73 [+0.84, +2.61]
on all ten) and against MR6 on all ten (+1.12 [+0.21, +2.03]). Over MR3
those pooled gains also count the three extra copies (MR6 over MR3,
+0.50 [−0.31, +1.30] and +0.61 [−0.15, +1.37]), and the screen's seeds
lift the all-ten readings (over MR3 on MuLMS, +1.72 [+0.44, +3.01] on all
ten against +1.10 [−0.46, +2.65] on the decision seeds). Weighted F1's
interval clears 0 where micro F1's does not in 20 rows (16 per-corpus on
MuLMS, among them D1 at +4.17 [+0.39, +7.95], 3 on MaterioMiner, and the
pooled names against MR3 on all ten); the converse, micro F1 above 0 and
macro F1 not, occurs in 8 rows, 7 on MaterioMiner or SOFC-Exp, among them
the names against MR3 on MaterioMiner's coarse-grained papers (micro
+0.49 [+0.06, +0.92], macro −0.39 [−2.14, +1.36]). Table S12 gives each
row's per-unit values and its votes' macro and weighted F1. Some numbers
quoted in the record are not in the tables: mention replacement's pooled
gain with MaterioMiner's coarse-grained papers, +1.80 [+0.39, +3.22],
which the protocol of 20 October (09:14) records (Section 3; from the
run files +1.80 [+0.39, +3.21]); the same protocol's +2.34 [+0.94,
+3.75] with the fine-grained papers, which row 75 gives as +2.35 from
per-unit values rounded differently; the reading of 20 October (18:42),
which put D5 at about [+0.1, +5.9] from the screen's values; and the two
scenarios of 24 October with an assumed MaterioMiner gain, +2.67 [+0.09,
+5.25] and +2.49 [−0.13, +5.11] (Supplementary Note 11). The tables leave out
the per-type analyses (Supplementary Note 6), the lexicon split
(Supplementary Note 5) and the companion study's own comparisons except
the two the names route reports again (label propagation and
self-training on MaterioMiner's papers). The protocol of 20 October
(09:14) also listed the label-propagation rule's votes as reported; they
were never computed. Generated by `names_route_record.py`
(`docs/tables/table13_names_route.md`).

**Table S11.** Every comparison of the names route: week and protocol
time, units, comparison, status (a decision with its verdict, the
screen, pre-specified, post hoc, or a comparison of an earlier protocol
reported again or recomputed), and the mean paired difference with its 95% CI (and,
for micro F1 and the votes, the units above 0) in micro F1 (or the
measure the comparison names), macro, weighted and strict F1 and the
votes; the macro and weighted cells of the 11-unit pooled rows are post
hoc. An empty cell: the results do not hold that quantity.

| # | week (protocol, UTC) | corpus, units (MuLMS seeds) | comparison (system − reference) | status | Δ micro F1 [95% CI], units above 0 | Δ macro F1 [95% CI] | Δ weighted F1 [95% CI] | Δ strict IOB2 micro F1 [95% CI] | votes: Δ micro F1 [95% CI], units above 0 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 (17 October 20:21) | MuLMS, 7 subfields (seeds 0–900) | MR3 − CRF head | decision 10.1: no; votes: decision 10.2: passes | +2.10 [−0.31, +4.51], 6/7 | +2.12 [+1.14, +3.09] | +2.68 [+0.32, +5.04] | +2.10 [−0.31, +4.51] | +2.71 [+0.22, +5.20], 6/7 |
| 2 | 10 (17 October 20:21) | MuLMS, 5 random pool splits against the 7 subfields (seeds 0–900; not paired) | CRF head on the random pool splits − CRF head on the held-out subfields | pre-specified | +17.06 | +15.06 | +18.74 | +17.06 |  |
| 3 | 10 (17 October 20:21) | MuLMS, 7 subfields (seeds 0–900) | MR3 − CRF head: F1 on the unseen test names | pre-specified | +2.19 [−0.25, +4.63] |  |  |  | +2.92 [+0.41, +5.44] |
| 4 | 10 (17 October 20:21) | MuLMS, 7 subfields (seeds 0–900) | MR3 − CRF head: F1 on the seen test names | pre-specified | +1.30 [−1.21, +3.81] |  |  |  | +1.17 [−1.69, +4.04] |
| 5 | 11 (18 October 01:13) | MuLMS, 7 subfields (seeds 0–900) | names from the other subfields' sentences − CRF head | decision 11.1: no | +2.04 [−0.85, +4.92], 5/7 | +2.37 [+1.27, +3.47] | +2.62 [+0.02, +5.22] | +2.04 [−0.85, +4.92] | +2.48 [−0.44, +5.40], 5/7 |
| 6 | 11 (18 October 01:13) | MuLMS, 7 subfields (seeds 0–900) | names from the other subfields' sentences − MR3 | decision 11.2: no | −0.06 [−1.09, +0.96], 3/7 | +0.25 [−0.52, +1.02] | −0.06 [−1.19, +1.06] | −0.06 [−1.09, +0.96] | −0.23 [−1.15, +0.68], 3/7 |
| 7 | 11 (18 October 01:13) | MaterioMiner fine, 4 papers | names from the fatigue corpus's sentences − MR3 | decision 11.3: no | +0.23 [−0.69, +1.14], 3/4 | +0.58 [−1.65, +2.81] | +0.34 [−0.62, +1.30] | +0.23 [−0.69, +1.14] | +0.44 [−1.85, +2.73], 2/4 |
| 8 | 11 (18 October 01:13) | MaterioMiner coarse, 4 papers | names from the fatigue corpus's sentences − MR3 | decision 11.4: no | +0.61 [−0.96, +2.18], 3/4 | −0.24 [−2.47, +1.98] | +0.66 [−1.28, +2.61] | +0.61 [−0.96, +2.18] | +0.24 [−1.55, +2.02], 2/4 |
| 9 | 11 (18 October 01:13) | MaterioMiner fine, 4 papers | names from the fatigue corpus's sentences − CRF head | pre-specified | +3.00 [+1.98, +4.02], 4/4 | +2.92 [−0.80, +6.64] | +3.96 [+1.72, +6.21] | +3.00 [+1.98, +4.02] | +4.05 [+3.03, +5.07], 4/4 |
| 10 | 11 (18 October 01:13) | MaterioMiner coarse, 4 papers | names from the fatigue corpus's sentences − CRF head | pre-specified | +1.89 [−0.31, +4.10], 4/4 | +2.55 [−1.00, +6.10] | +2.52 [−0.06, +5.10] | +1.89 [−0.31, +4.10] | +2.50 [+0.48, +4.51], 4/4 |
| 11 | 11 (18 October 01:13) | MaterioMiner fine, 4 papers | the votes: names from the fatigue corpus's sentences − multi-task | pre-specified | +0.40 [−2.76, +3.55], 3/4 | −0.68 [−7.69, +6.34] | +1.05 [−1.54, +3.64] | +0.40 [−2.76, +3.55] |  |
| 12 | 11 (18 October 01:13) | MaterioMiner coarse, 4 papers | the votes: names from the fatigue corpus's sentences − multi-task | pre-specified | +2.81 [+0.95, +4.67], 4/4 | +5.40 [+1.47, +9.33] | +3.62 [+1.10, +6.14] | +2.81 [+0.95, +4.67] |  |
| 13 | 11 (18 October 01:13) | MuLMS, 7 subfields (seeds 0–900) | names from the other subfields' sentences − CRF head: F1 on the unseen test names | pre-specified | +2.13 [−0.84, +5.09] |  |  |  | +2.58 [−0.39, +5.56] |
| 14 | 11 (18 October 01:13) | MuLMS, 7 subfields (seeds 0–900) | names from the other subfields' sentences − CRF head: F1 on the seen test names | pre-specified | +1.21 [−0.97, +3.39] |  |  |  | +1.49 [−0.90, +3.88] |
| 15 | 11 (18 October 01:13) | MaterioMiner fine, 4 papers | names from the fatigue corpus's sentences − CRF head: F1 on the unseen test names | pre-specified | +3.19 [+2.03, +4.35] |  |  |  | +4.26 [+2.31, +6.22] |
| 16 | 11 (18 October 01:13) | MaterioMiner fine, 4 papers | names from the fatigue corpus's sentences − CRF head: F1 on the seen test names | pre-specified | +2.66 [+0.03, +5.29] |  |  |  | +3.05 [−1.56, +7.65] |
| 17 | 11 (18 October 01:13) | MaterioMiner coarse, 4 papers | names from the fatigue corpus's sentences − CRF head: F1 on the unseen test names | pre-specified | +2.21 [−0.28, +4.70] |  |  |  | +2.79 [+0.35, +5.24] |
| 18 | 11 (18 October 01:13) | MaterioMiner coarse, 4 papers | names from the fatigue corpus's sentences − CRF head: F1 on the seen test names | pre-specified | +0.63 [−1.49, +2.75] |  |  |  | +1.54 [−0.58, +3.67] |
| 19 | 12 (18 October 11:00) | MuLMS, 7 subfields (seeds 0–200) | MR3 + names, sentence filter (V1) − MR3 | screen, its statistic (no decision) | +0.80 [−0.91, +2.51], 5/7 | +1.09 [−1.88, +4.06] | +1.08 [−0.90, +3.07] | +0.80 [−0.91, +2.51] | +0.10 [−1.68, +1.88], 3/7 |
| 20 | 12 (18 October 11:00) | MuLMS, 7 subfields (seeds 0–200) | MR3 + names, sentence filter (V1) − CRF head | screen (no decision) | +2.50 [−2.09, +7.09], 5/7 | +1.87 [−2.69, +6.43] | +3.03 [−2.30, +8.36] | +2.50 [−2.09, +7.09] | +2.91 [−2.04, +7.86], 5/7 |
| 21 | 12 (18 October 11:00) | MuLMS, 7 subfields (seeds 0–200) | MR3 + names, mention filter (V2) − MR3 | screen, its statistic (no decision) | +0.97 [−0.91, +2.84], 4/7 | +1.85 [+0.76, +2.94] | +1.52 [+0.20, +2.84] | +0.97 [−0.91, +2.84] | +0.17 [−1.51, +1.86], 3/7 |
| 22 | 12 (18 October 11:00) | MuLMS, 7 subfields (seeds 0–200) | MR3 + names, mention filter (V2) − CRF head | screen (no decision) | +2.67 [−1.25, +6.59], 5/7 | +2.63 [+0.59, +4.67] | +3.47 [−0.22, +7.16] | +2.67 [−1.25, +6.59] | +2.98 [−1.06, +7.02], 6/7 |
| 23 | 12 (18 October 11:00) | MuLMS, 7 subfields (seeds 0–200) | MR6 (V3) − MR3 | screen, its statistic (no decision) | −0.20 [−3.20, +2.80], 5/7 | +1.16 [−1.28, +3.61] | +0.14 [−3.21, +3.48] | −0.20 [−3.20, +2.80] | −0.96 [−3.87, +1.95], 4/7 |
| 24 | 12 (18 October 11:00) | MuLMS, 7 subfields (seeds 0–200) | MR6 (V3) − CRF head | screen (no decision) | +1.50 [−1.65, +4.66], 4/7 | +1.95 [−0.26, +4.15] | +2.08 [−0.72, +4.88] | +1.50 [−1.65, +4.66] | +1.85 [−1.68, +5.37], 4/7 |
| 25 | 12 (amendment 1, 19 October 17:15) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names (V4) − MR3 | screen, its statistic (no decision); selected | +1.45 [−1.35, +4.25], 5/7 | +3.19 [+1.62, +4.76] | +2.67 [+0.75, +4.59] | +1.45 [−1.35, +4.25] | +0.56 [−1.60, +2.72], 5/7 |
| 26 | 12 (amendment 1, 19 October 17:15) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names (V4) − CRF head | screen (no decision) | +3.16 [−1.96, +8.28], 5/7 | +3.97 [+0.69, +7.26] | +4.62 [+0.14, +9.09] | +3.16 [−1.96, +8.28] | +3.37 [−1.52, +8.26], 5/7 |
| 27 | 12 (amendment 1, 19 October 17:15) | MuLMS, 7 subfields (seeds 0–200) | self-training on the corpus's silver sentences (V6) − MR3 | screen, its statistic (no decision) | +0.16 [−2.31, +2.63], 3/7 | −0.65 [−2.59, +1.30] | −1.39 [−4.19, +1.41] | +0.16 [−2.31, +2.63] | −1.11 [−3.74, +1.52], 2/7 |
| 28 | 12 (amendment 1, 19 October 17:15) | MuLMS, 7 subfields (seeds 0–200) | self-training on the corpus's silver sentences (V6) − CRF head | screen (no decision) | +1.87 [+0.20, +3.53], 6/7 | +0.14 [−1.86, +2.13] | +0.56 [−1.77, +2.88] | +1.87 [+0.20, +3.53] | +1.70 [−0.44, +3.84], 5/7 |
| 29 | 12 (amendment 2, 19 October 21:03) | MuLMS, 7 subfields (seeds 0–200) | lexicon as a gazetteer feature, mention filter (V7) − MR3 | screen, its statistic (no decision) | −1.73 [−4.13, +0.67], 2/7 | −1.08 [−4.28, +2.12] | −1.88 [−4.87, +1.12] | −1.73 [−4.13, +0.67] | −2.48 [−4.97, +0.01], 1/7 |
| 30 | 12 (amendment 2, 19 October 21:03) | MuLMS, 7 subfields (seeds 0–200) | lexicon as a gazetteer feature, mention filter (V7) − CRF head | screen (no decision) | −0.03 [−2.02, +1.97], 3/7 | −0.30 [−3.08, +2.48] | +0.07 [−2.43, +2.57] | −0.03 [−2.02, +1.97] | +0.33 [−2.28, +2.94], 3/7 |
| 31 | 12 (amendment 3, 20 October 01:50) | MuLMS, 7 subfields (seeds 0–200) | lexicon as a gazetteer feature, sentence filter (V8) − MR3 | screen, its statistic (no decision) | −1.85 [−3.59, −0.12], 1/7 | −1.39 [−3.45, +0.66] | −2.11 [−4.12, −0.10] | −1.85 [−3.59, −0.12] | −2.49 [−4.27, −0.71], 0/7 |
| 32 | 12 (amendment 3, 20 October 01:50) | MuLMS, 7 subfields (seeds 0–200) | lexicon as a gazetteer feature, sentence filter (V8) − CRF head | screen (no decision) | −0.15 [−2.15, +1.86], 3/7 | −0.61 [−2.37, +1.14] | −0.16 [−2.63, +2.31] | −0.15 [−2.15, +1.86] | +0.31 [−1.97, +2.60], 5/7 |
| 33 | 12 (18 October 11:00) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − CRF head | D1: no | +3.19 [−1.13, +7.51], 6/7 | +3.78 [+2.47, +5.10] | +4.17 [+0.39, +7.95] | +3.19 [−1.13, +7.51] | +3.23 [−1.15, +7.61], 5/7 |
| 34 | 12 (18 October 11:00) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR3 | D2: no | +0.92 [−1.63, +3.48], 4/7 | +1.10 [−0.46, +2.65] | +1.17 [−1.09, +3.44] | +0.92 [−1.63, +3.48] | +0.57 [−2.22, +3.36], 5/7 |
| 35 | 12 (18 October 11:00) | MuLMS, 7 subfields (seeds 0–900) | MR6 + names − CRF head | pre-specified; D5's MuLMS part | +3.18 [−1.30, +7.66], 5/7 | +3.84 [+2.16, +5.52] | +4.30 [+0.47, +8.14] | +3.18 [−1.30, +7.66] | +3.28 [−1.16, +7.71], 6/7 |
| 36 | 12 (18 October 11:00) | MuLMS, 7 subfields (seeds 0–900) | MR6 + names − MR3 | pre-specified | +1.08 [−1.12, +3.28], 5/7 | +1.72 [+0.44, +3.01] | +1.62 [−0.05, +3.29] | +1.08 [−1.12, +3.28] | +0.56 [−1.58, +2.71], 3/7 |
| 37 | 12 (18 October 11:00) | MaterioMiner fine, 4 papers | MR6 + names − CRF head | D3: passes; D5's MaterioMiner part | +4.27 [+1.39, +7.15], 4/4 | +4.07 [+2.08, +6.05] | +5.00 [+1.44, +8.56] | +4.27 [+1.39, +7.15] | +4.99 [+3.27, +6.72], 4/4 |
| 38 | 12 (18 October 11:00) | MaterioMiner coarse, 4 papers | MR6 + names − CRF head | D4: passes | +1.77 [+0.67, +2.88], 4/4 | +2.40 [+0.81, +3.99] | +2.35 [+1.03, +3.67] | +1.77 [+0.67, +2.88] | +2.14 [+0.23, +4.05], 4/4 |
| 39 | 12 (the analogue batches, 20 October 00:51) | MaterioMiner fine, 4 papers | MR6 + names − MR3 | pre-specified | +1.50 [−1.76, +4.75], 3/4 | +1.73 [−0.47, +3.94] | +1.38 [−2.32, +5.07] | +1.50 [−1.76, +4.75] | +1.38 [−1.73, +4.49], 3/4 |
| 40 | 12 (the analogue batches, 20 October 00:51) | MaterioMiner coarse, 4 papers | MR6 + names − MR3 | pre-specified | +0.49 [+0.06, +0.92], 4/4 | −0.39 [−2.14, +1.36] | +0.49 [−0.03, +1.02] | +0.49 [+0.06, +0.92] | −0.12 [−1.65, +1.42], 2/4 |
| 41 | 12 (the analogue batches, 20 October 00:51) | MaterioMiner fine, 4 papers | self-training on the fatigue corpus's silver sentences (V6's analogue) − CRF head | reported again; the companion study's week-5 decision (15 October): no; votes: no | +1.74 [−1.16, +4.64], 3/4 | +1.90 [−1.11, +4.90] | +1.07 [−1.38, +3.52] | +1.74 [−1.16, +4.64] | +1.12 [−2.71, +4.96], 3/4 |
| 42 | 12 (the analogue batches, 20 October 00:51) | MaterioMiner coarse, 4 papers | self-training on the fatigue corpus's silver sentences (V6's analogue) − CRF head | reported again; the companion study's week-5 decision (15 October): no; votes: no | +1.56 [−0.88, +4.00], 3/4 | +1.55 [−0.79, +3.89] | +1.57 [−0.90, +4.04] | +1.56 [−0.88, +4.00] | +0.93 [−1.66, +3.53], 3/4 |
| 43 | 12 (the analogue batches, 20 October 00:51) | MaterioMiner fine, 4 papers | self-training on the fatigue corpus's silver sentences (V6's analogue) − MR3 | post hoc | −1.03 [−4.47, +2.41], 2/4 | −0.44 [−1.66, +0.78] | −2.55 [−4.65, −0.46] | −1.03 [−4.47, +2.41] | −2.49 [−5.74, +0.76], 0/4 |
| 44 | 12 (the analogue batches, 20 October 00:51) | MaterioMiner coarse, 4 papers | self-training on the fatigue corpus's silver sentences (V6's analogue) − MR3 | post hoc | +0.28 [−1.46, +2.01], 2/4 | −1.24 [−3.08, +0.60] | −0.29 [−1.39, +0.82] | +0.28 [−1.46, +2.01] | −1.32 [−1.85, −0.80], 0/4 |
| 45 | 13 (20 October 09:14) | MuLMS, 7 subfields (seeds 0–900) | CRF head + label propagation − CRF head | decision 13.1a: passes | +0.82 [+0.18, +1.46], 7/7 | +0.31 [+0.18, +0.44] | +0.95 [+0.30, +1.60] | +0.82 [+0.18, +1.46] |  |
| 46 | 13 (20 October 09:14) | MuLMS, 7 subfields (seeds 0–900) | CRF head + label propagation with retyping − CRF head | pre-specified | +1.01 [+0.28, +1.74], 7/7 | +0.36 [+0.20, +0.51] | +1.08 [+0.33, +1.83] | +1.01 [+0.28, +1.74] |  |
| 47 | 13 (20 October 09:14) | MuLMS, 7 subfields (seeds 0–900) | MR3 + label propagation − MR3 | decision 13.1b: no | +0.57 [−0.12, +1.25], 6/7 | +0.23 [+0.10, +0.35] | +0.70 [+0.08, +1.33] | +0.57 [−0.12, +1.25] |  |
| 48 | 13 (20 October 09:14) | MuLMS, 7 subfields (seeds 0–900) | MR3 + label propagation with retyping − MR3 | pre-specified | +0.88 [−0.01, +1.76], 7/7 | +0.28 [+0.09, +0.46] | +0.91 [+0.11, +1.70] | +0.88 [−0.01, +1.76] |  |
| 49 | 13 (amendment, 20 October 09:25) | MaterioMiner fine, 4 papers | CRF head + label propagation − CRF head | the companion study's week-7 decision (16 October), recomputed: no | +0.30 [−0.35, +0.95], 2/4 | +0.13 [−0.14, +0.39] | +0.54 [−0.40, +1.48] | +0.30 [−0.35, +0.95] |  |
| 50 | 13 (amendment, 20 October 09:25) | MaterioMiner fine, 4 papers | CRF head + label propagation with retyping − CRF head | the companion study's week-7 comparison (16 October; reported), recomputed | +0.31 [−0.73, +1.35], 3/4 | +0.08 [−1.00, +1.16] | +0.39 [−1.16, +1.94] | +0.31 [−0.73, +1.35] |  |
| 51 | 13 (amendment, 20 October 09:25) | MaterioMiner fine, 4 papers | MR3 + label propagation − MR3 | decision 13.1d: no | +0.19 [−0.34, +0.73], 3/4 | +0.07 [−0.15, +0.29] | +0.39 [−0.25, +1.04] | +0.19 [−0.34, +0.73] |  |
| 52 | 13 (amendment, 20 October 09:25) | MaterioMiner fine, 4 papers | MR3 + label propagation with retyping − MR3 | pre-specified | +0.22 [−0.69, +1.12], 3/4 | +0.05 [−1.00, +1.11] | +0.23 [−1.20, +1.66] | +0.22 [−0.69, +1.12] |  |
| 53 | 13 (amendment, 20 October 09:25) | MaterioMiner fine, 4 papers | multi-task + MR3 + label propagation − multi-task + MR3 | pre-specified | +0.31 [−0.47, +1.09], 3/4 | +0.07 [−0.02, +0.15] | +0.51 [−0.32, +1.33] | +0.31 [−0.47, +1.09] |  |
| 54 | 13 (amendment, 20 October 09:25) | MaterioMiner fine, 4 papers | multi-task + MR3 + label propagation with retyping − multi-task + MR3 | pre-specified | +0.18 [−1.11, +1.47], 2/4 | −0.18 [−0.99, +0.64] | +0.13 [−1.64, +1.90] | +0.18 [−1.11, +1.47] |  |
| 55 | 13 (amendment, 20 October 09:25) | MaterioMiner coarse, 4 papers | CRF head + label propagation − CRF head | the companion study's week-7 decision (16 October), recomputed: passes | +0.12 [+0.01, +0.24], 4/4 | +0.11 [−0.18, +0.40] | +0.18 [−0.08, +0.43] | +0.12 [+0.01, +0.24] |  |
| 56 | 13 (amendment, 20 October 09:25) | MaterioMiner coarse, 4 papers | CRF head + label propagation with retyping − CRF head | the companion study's week-7 comparison (16 October; reported), recomputed | +0.64 [+0.29, +1.00], 4/4 | +0.46 [−0.02, +0.95] | +0.70 [+0.28, +1.12] | +0.64 [+0.29, +1.00] |  |
| 57 | 13 (amendment, 20 October 09:25) | MaterioMiner coarse, 4 papers | MR3 + label propagation − MR3 | decision 13.1d: no | +0.11 [−0.08, +0.31], 4/4 | +0.10 [−0.14, +0.34] | +0.15 [−0.11, +0.41] | +0.11 [−0.08, +0.31] |  |
| 58 | 13 (amendment, 20 October 09:25) | MaterioMiner coarse, 4 papers | MR3 + label propagation with retyping − MR3 | pre-specified | +0.63 [+0.13, +1.14], 4/4 | +0.48 [+0.12, +0.83] | +0.67 [+0.26, +1.08] | +0.63 [+0.13, +1.14] |  |
| 59 | 13 (amendment, 20 October 09:25) | MaterioMiner coarse, 4 papers | multi-task + MR3 + label propagation − multi-task + MR3 | pre-specified | +0.10 [−0.05, +0.25], 3/4 | +0.08 [−0.06, +0.22] | +0.13 [−0.03, +0.29] | +0.10 [−0.05, +0.25] |  |
| 60 | 13 (amendment, 20 October 09:25) | MaterioMiner coarse, 4 papers | multi-task + MR3 + label propagation with retyping − multi-task + MR3 | pre-specified | +0.60 [−0.21, +1.41], 4/4 | +0.38 [−0.26, +1.02] | +0.62 [−0.18, +1.43] | +0.60 [−0.21, +1.41] |  |
| 61 | 13 (amendment, 20 October 09:25) | SOFC-Exp, 5 folds | CRF head + label propagation − CRF head | pre-specified (13.1c) | −0.44 [−0.72, −0.16], 0/5 | −0.50 [−0.92, −0.08] | −0.35 [−0.63, −0.07] | −0.44 [−0.72, −0.16] |  |
| 62 | 13 (amendment, 20 October 09:25) | SOFC-Exp, 5 folds | CRF head + label propagation with retyping − CRF head | pre-specified (13.1c) | −0.43 [−0.68, −0.18], 0/5 | −0.49 [−0.86, −0.11] | −0.34 [−0.59, −0.09] | −0.43 [−0.68, −0.18] |  |
| 63 | 13 (20 October 09:14) | SOFC-Exp, 5 folds | MR3 − CRF head | decision 13.3a: no | −0.48 [−1.36, +0.39], 1/5 | −0.50 [−1.31, +0.31] | −0.46 [−1.27, +0.34] | −0.48 [−1.36, +0.39] |  |
| 64 | 13 (20 October 09:14) | SOFC-Exp, 5 folds | multi-task + MR3 − CRF head | decision 13.3b: no | −0.11 [−0.69, +0.46], 1/5 | −0.16 [−0.59, +0.28] | −0.08 [−0.56, +0.40] | −0.11 [−0.69, +0.46] |  |
| 65 | 13 (20 October 09:14) | SOFC-Exp, 5 folds | multi-task + MR3 − MR3 | pre-specified | +0.37 [−0.11, +0.84], 5/5 | +0.34 [−0.46, +1.14] | +0.38 [−0.14, +0.90] | +0.37 [−0.11, +0.84] |  |
| 66 | 13 (20 October 09:14) | SOFC-Exp, 5 folds, slot level | multi-task + MR3, its slot level (no reference at that level) | pre-specified | 74.01 [67.89, 80.13] (its level) | 70.71 [64.62, 76.80] (its level) | 73.84 [67.95, 79.72] (its level) | 74.01 [67.89, 80.13] (its level) |  |
| 67 | 13 (amendment 13.4, 20 October 13:53) | SOFC-Exp, 5 folds | MR6 + names − CRF head | D6: no | −0.04 [−1.21, +1.14], 2/5 | −0.17 [−1.14, +0.81] | −0.04 [−1.12, +1.05] | −0.04 [−1.21, +1.14] |  |
| 68 | 13 (amendment 13.4, 20 October 13:53) | SOFC-Exp, 5 folds | MR6 + names − MR3 | pre-specified | +0.45 [+0.04, +0.86], 5/5 | +0.33 [−0.07, +0.73] | +0.43 [+0.03, +0.82] | +0.45 [+0.04, +0.86] |  |
| 69 | 13 (20 October 09:14) | pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR6 + names − CRF head | D5: passes (neither corpus below 0) | +3.58 [+0.95, +6.21], 9/11 | +3.92 [+2.87, +4.98] | +4.56 [+2.24, +6.88] |  |  |
| 70 | 13 (amendment 13.4, 20 October 13:53) | pooled, 16 units: MuLMS 7 (0–900) + MaterioMiner fine 4 + SOFC-Exp 5 | MR6 + names − CRF head | pre-specified (13.4) | +2.45 [+0.50, +4.40], 11/16 |  |  |  |  |
| 71 | 13 (20 October 09:14) | pooled, 16 units: MuLMS 7 (0–900) + MaterioMiner fine 4 + SOFC-Exp 5 | MR3 − CRF head | pre-specified (13.3) | +1.46 [+0.28, +2.64], 11/16 |  |  |  |  |
| 72 | 13 (20 October 09:14) | pooled, 9 units: MaterioMiner fine 4 + SOFC-Exp 5 | multi-task + MR3 − CRF head | pre-specified (13.3) | +2.93 [+0.14, +5.72], 5/9 |  |  |  |  |
| 73 | 13 (20 October 09:14) | pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4 | names from the other subfields' sentences (MuLMS) and the fatigue corpus's (MaterioMiner) − CRF head | post hoc | +2.39 [+0.71, +4.06], 9/11 |  |  |  |  |
| 74 | 13 (20 October 09:14) | pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR6 + names − MR3 | post hoc | +1.23 [−0.22, +2.68], 8/11 | +1.73 [+0.84, +2.61] | +1.53 [+0.26, +2.81] |  |  |
| 75 | 13 (20 October 09:14) | pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR3 − CRF head | post hoc | +2.35 [+0.94, +3.75], 10/11 | +2.19 [+1.25, +3.14] | +3.02 [+1.58, +4.47] |  |  |
| 76 | 13 (20 October 09:14) | pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4 | MR6 + names − CRF head | D5′, D5 on the decision seeds (post hoc reading): above 0 | +3.58 [+1.04, +6.13], 10/11 | +3.89 [+3.01, +4.76] | +4.47 [+2.17, +6.77] |  |  |
| 77 | 13 (20 October 09:14) | pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4 | MR6 + names − MR3 | post hoc | +1.13 [−0.51, +2.76], 7/11 | +1.33 [+0.29, +2.36] | +1.25 [−0.29, +2.78] |  |  |
| 78 | 13 (20 October 09:14) | pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4 | MR3 − CRF head | post hoc | +2.46 [+1.00, +3.91], 11/11 | +2.56 [+1.56, +3.55] | +3.22 [+1.88, +4.57] |  |  |
| 79 | 14 (29 October 12:07) | pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4 | MR6 + names − MR6 | D7: no (neither corpus below 0) | +0.80 [−0.66, +2.26], 7/11 | +0.83 [−0.01, +1.67] | +0.89 [−0.40, +2.18] |  |  |
| 80 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 | D7's per-corpus condition: its interval not below 0 | +0.51 [−1.51, +2.53], 4/7 | +0.53 [−0.56, +1.62] | +0.84 [−1.04, +2.71] | +0.51 [−1.51, +2.53] | +0.35 [−1.67, +2.36], 3/7 |
| 81 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–900) | MR6 + names − MR6 | pre-specified | +0.85 [−1.46, +3.17], 3/7 | +0.98 [−0.36, +2.32] | +1.35 [−0.82, +3.51] | +0.85 [−1.46, +3.17] | +0.74 [−1.47, +2.95], 3/7 |
| 82 | 14 (29 October 12:07) | MaterioMiner fine, 4 papers | MR6 + names − MR6 | D7's per-corpus condition: its interval not below 0 | +1.29 [−2.51, +5.10], 3/4 | +1.36 [−0.80, +3.51] | +0.99 [−2.20, +4.18] | +1.29 [−2.51, +5.10] | +0.94 [−1.60, +3.49], 3/4 |
| 83 | 14 (29 October 12:07) | MaterioMiner coarse, 4 papers | MR6 + names − MR6 | pre-specified | +0.70 [−1.60, +3.00], 3/4 | −0.43 [−3.87, +3.00] | +0.26 [−1.74, +2.25] | +0.70 [−1.60, +3.00] | +1.06 [−0.62, +2.74], 3/4 |
| 84 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 300–900) | MR6 − CRF head | pre-specified | +2.68 [+0.13, +5.23], 6/7 | +3.25 [+2.60, +3.90] | +3.33 [+1.02, +5.65] | +2.68 [+0.13, +5.23] | +2.88 [−0.03, +5.80], 6/7 |
| 85 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 300–900) | MR6 − MR3 | pre-specified | +0.41 [−0.35, +1.17], 5/7 | +0.56 [−0.40, +1.53] | +0.34 [−0.21, +0.88] | +0.41 [−0.35, +1.17] | +0.22 [−0.94, +1.38], 4/7 |
| 86 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–900) | MR6 − CRF head | pre-specified | +2.33 [−0.27, +4.92], 6/7 | +2.86 [+2.01, +3.71] | +2.96 [+0.65, +5.27] | +2.33 [−0.27, +4.92] | +2.54 [−0.39, +5.47], 6/7 |
| 87 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–900) | MR6 − MR3 | pre-specified | +0.23 [−0.39, +0.85], 5/7 | +0.74 [−0.07, +1.56] | +0.28 [−0.55, +1.10] | +0.23 [−0.39, +0.85] | −0.17 [−0.94, +0.59], 2/7 |
| 88 | 14 (29 October 12:07) | MaterioMiner fine, 4 papers | MR6 − CRF head | pre-specified | +2.98 [+0.09, +5.86], 4/4 | +2.71 [−0.80, +6.22] | +4.01 [+1.37, +6.64] | +2.98 [+0.09, +5.86] | +4.05 [+1.84, +6.27], 4/4 |
| 89 | 14 (29 October 12:07) | MaterioMiner fine, 4 papers | MR6 − MR3 | pre-specified | +0.20 [−1.47, +1.88], 3/4 | +0.37 [−2.17, +2.92] | +0.38 [−1.13, +1.90] | +0.20 [−1.47, +1.88] | +0.44 [−1.94, +2.82], 2/4 |
| 90 | 14 (29 October 12:07) | MaterioMiner coarse, 4 papers | MR6 − CRF head | pre-specified | +1.07 [−1.82, +3.97], 3/4 | +2.83 [−0.65, +6.32] | +2.09 [−0.55, +4.74] | +1.07 [−1.82, +3.97] | +1.08 [−1.68, +3.84], 4/4 |
| 91 | 14 (29 October 12:07) | MaterioMiner coarse, 4 papers | MR6 − MR3 | pre-specified | −0.21 [−2.71, +2.29], 2/4 | +0.04 [−2.14, +2.23] | +0.24 [−2.12, +2.59] | −0.21 [−2.71, +2.29] | −1.18 [−4.39, +2.03], 2/4 |
| 92 | 14 (29 October 12:07) | pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR6 − CRF head | post hoc | +2.56 [+0.94, +4.18], 10/11 | +2.81 [+1.87, +3.75] | +3.34 [+1.86, +4.82] |  |  |
| 93 | 14 (29 October 12:07) | pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR6 + names − MR6 | post hoc | +1.02 [−0.56, +2.59], 6/11 | +1.12 [+0.21, +2.03] | +1.22 [−0.21, +2.65] |  |  |
| 94 | 14 (29 October 12:07) | pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR6 − MR3 | post hoc | +0.22 [−0.30, +0.74], 8/11 | +0.61 [−0.15, +1.37] | +0.32 [−0.26, +0.90] |  |  |
| 95 | 14 (29 October 12:07) | pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4 | MR6 − CRF head | post hoc | +2.79 [+1.20, +4.38], 10/11 | +3.06 [+2.15, +3.96] | +3.58 [+2.12, +5.04] |  |  |
| 96 | 14 (29 October 12:07) | pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4 | MR6 − MR3 | post hoc | +0.33 [−0.25, +0.92], 8/11 | +0.50 [−0.31, +1.30] | +0.36 [−0.11, +0.82] |  |  |
| 97 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names, own-topic corpus − mixed corpus | D8: no | −0.18 [−2.01, +1.65], 2/7 | −0.24 [−1.56, +1.08] | −0.15 [−2.06, +1.75] | −0.18 [−2.01, +1.65] | −0.39 [−2.64, +1.86], 1/7 |
| 98 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 300–900) | the mixed-corpus variant − CRF head | pre-specified | +3.38 [−0.89, +7.64], 7/7 | +4.02 [+2.38, +5.66] | +4.32 [+0.18, +8.47] | +3.38 [−0.89, +7.64] | +3.62 [−0.71, +7.95], 7/7 |
| 99 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 300–900) | the mixed-corpus variant − MR3 | pre-specified | +1.10 [−0.76, +2.97], 5/7 | +1.33 [+0.02, +2.65] | +1.33 [−0.74, +3.39] | +1.10 [−0.76, +2.97] | +0.96 [−1.02, +2.93], 5/7 |
| 100 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 300–900) | the mixed-corpus variant − MR6 | pre-specified | +0.70 [−1.19, +2.58], 3/7 | +0.77 [−0.61, +2.15] | +0.99 [−1.08, +3.06] | +0.70 [−1.19, +2.58] | +0.74 [−0.98, +2.45], 4/7 |
| 101 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, corpus: its first 10,000 sentences − MR6 + names | pre-specified | −0.15 [−1.78, +1.48], 4/7 | −0.73 [−2.44, +0.98] | −0.76 [−2.29, +0.76] | −0.15 [−1.78, +1.48] | +0.20 [−1.06, +1.46], 5/7 |
| 102 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, corpus: its first 10,000 sentences − MR3 | pre-specified | +1.30 [−1.14, +3.74], 5/7 | +2.46 [+0.15, +4.78] | +1.91 [−0.57, +4.38] | +1.30 [−1.14, +3.74] | +0.76 [−1.71, +3.23], 3/7 |
| 103 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, corpus: its first 10,000 sentences − CRF head | pre-specified | +3.01 [−1.21, +7.23], 5/7 | +3.25 [+1.06, +5.44] | +3.85 [−0.17, +7.88] | +3.01 [−1.21, +7.23] | +3.57 [−1.12, +8.26], 6/7 |
| 104 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, corpus: its first 20,000 sentences − MR6 + names | pre-specified | −0.16 [−2.45, +2.14], 4/7 | −0.60 [−1.77, +0.57] | −0.77 [−2.55, +1.02] | −0.16 [−2.45, +2.14] | −0.32 [−2.07, +1.42], 4/7 |
| 105 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, corpus: its first 20,000 sentences − MR3 | pre-specified | +1.30 [−0.56, +3.15], 6/7 | +2.59 [+1.10, +4.08] | +1.90 [+0.03, +3.78] | +1.30 [−0.56, +3.15] | +0.24 [−1.90, +2.37], 3/7 |
| 106 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, corpus: its first 20,000 sentences − CRF head | pre-specified | +3.00 [−0.49, +6.49], 6/7 | +3.37 [+0.67, +6.08] | +3.85 [+0.51, +7.19] | +3.00 [−0.49, +6.49] | +3.05 [−1.07, +7.16], 6/7 |
| 107 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, filter: marginal threshold 0.7 − MR6 + names | pre-specified | −0.16 [−1.62, +1.31], 4/7 | −0.31 [−1.43, +0.81] | −0.18 [−1.90, +1.54] | −0.16 [−1.62, +1.31] | +0.34 [−1.07, +1.75], 4/7 |
| 108 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, filter: marginal threshold 0.7 − MR3 | pre-specified | +1.29 [−0.95, +3.53], 3/7 | +2.88 [+1.42, +4.34] | +2.49 [+0.85, +4.13] | +1.29 [−0.95, +3.53] | +0.90 [−1.16, +2.97], 4/7 |
| 109 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, filter: marginal threshold 0.7 − CRF head | pre-specified | +3.00 [−1.85, +7.85], 5/7 | +3.66 [+0.35, +6.98] | +4.43 [−0.13, +9.00] | +3.00 [−1.85, +7.85] | +3.71 [−1.38, +8.80], 6/7 |
| 110 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, filter: 2 of 3 teachers agree − MR6 + names | pre-specified | +0.11 [−1.58, +1.81], 4/7 | −0.39 [−1.95, +1.17] | +0.01 [−1.75, +1.76] | +0.11 [−1.58, +1.81] | +0.45 [−1.07, +1.97], 5/7 |
| 111 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, filter: 2 of 3 teachers agree − MR3 | pre-specified | +1.56 [+0.07, +3.05], 6/7 | +2.80 [+1.45, +4.15] | +2.68 [+1.44, +3.92] | +1.56 [+0.07, +3.05] | +1.01 [−0.87, +2.89], 4/7 |
| 112 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, filter: 2 of 3 teachers agree − CRF head | pre-specified | +3.27 [−0.85, +7.38], 5/7 | +3.58 [+0.46, +6.71] | +4.62 [+0.87, +8.37] | +3.27 [−0.85, +7.38] | +3.82 [−0.24, +7.88], 5/7 |
| 113 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, pool: harvested names in half of the draws − MR6 + names | pre-specified | +0.09 [−1.63, +1.80], 3/7 | −0.38 [−2.20, +1.45] | −0.27 [−1.84, +1.31] | +0.09 [−1.63, +1.80] | +0.08 [−1.25, +1.42], 5/7 |
| 114 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, pool: harvested names in half of the draws − MR3 | pre-specified | +1.54 [−0.74, +3.82], 6/7 | +2.81 [+0.70, +4.92] | +2.41 [−0.02, +4.83] | +1.54 [−0.74, +3.82] | +0.64 [−1.92, +3.21], 5/7 |
| 115 | 14 (29 October 12:07) | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, pool: harvested names in half of the draws − CRF head | pre-specified | +3.24 [−0.73, +7.22], 6/7 | +3.60 [+0.32, +6.87] | +4.35 [+0.35, +8.35] | +3.24 [−0.73, +7.22] | +3.45 [−1.20, +8.10], 6/7 |
| 116 | 17 (31 October 19:15) | MaterioMiner fine, 4 papers | multi-task + MR6 + names − multi-task + MR6 | D9: no | +0.96 [−0.01, +1.92], 4/4 | +1.41 [−1.16, +3.97] | +0.93 [−0.57, +2.44] | +0.96 [−0.01, +1.92] | +0.19 [−2.28, +2.66], 2/4 |
| 117 | 17 (31 October 19:15) | MaterioMiner fine, 4 papers | multi-task + MR6 + names − multi-task + MR3 | pre-specified | +0.76 [−0.15, +1.68], 4/4 | +1.96 [−1.83, +5.75] | +0.74 [−0.17, +1.65] | +0.76 [−0.15, +1.68] | +0.55 [−1.64, +2.74], 3/4 |
| 118 | 17 (31 October 19:15) | MaterioMiner fine, 4 papers | multi-task + MR6 + names − multi-task | pre-specified | +3.77 [+1.34, +6.21], 4/4 | +4.18 [+0.90, +7.46] | +3.85 [+2.39, +5.31] | +3.77 [+1.34, +6.21] | +3.41 [+0.76, +6.07], 4/4 |
| 119 | 17 (31 October 19:15) | MaterioMiner fine, 4 papers | multi-task + MR6 + names − CRF head | pre-specified | +7.49 [+5.96, +9.02], 4/4 | +7.94 [+3.37, +12.51] | +7.20 [+5.70, +8.70] | +7.49 [+5.96, +9.02] | +7.06 [+6.01, +8.12], 4/4 |
| 120 | 17 (31 October 19:15) | MaterioMiner fine, 4 papers | multi-task + MR6 − multi-task + MR3 | pre-specified | −0.19 [−2.05, +1.67], 1/4 | +0.55 [−1.93, +3.03] | −0.19 [−2.23, +1.84] | −0.19 [−2.05, +1.67] | +0.36 [−1.78, +2.50], 3/4 |
| 121 | 17 (31 October 19:15) | MaterioMiner fine, 4 papers | multi-task + MR6 − multi-task | pre-specified | +2.82 [−0.00, +5.64], 4/4 | +2.77 [−1.53, +7.08] | +2.92 [+0.92, +4.92] | +2.82 [−0.00, +5.64] | +3.22 [+1.23, +5.21], 4/4 |
| 122 | 17 (31 October 19:15) | MaterioMiner fine, 4 papers | multi-task + MR6 − CRF head | pre-specified | +6.54 [+4.14, +8.94], 4/4 | +6.53 [+3.58, +9.49] | +6.27 [+4.25, +8.29] | +6.54 [+4.14, +8.94] | +6.87 [+3.47, +10.28], 4/4 |
| 123 | 17 (31 October 19:15) | MaterioMiner coarse, 4 papers | multi-task + MR6 + names − multi-task + MR6 | D10: no | +0.85 [−0.72, +2.42], 3/4 | +0.49 [−1.48, +2.46] | +0.78 [−0.65, +2.20] | +0.85 [−0.72, +2.42] | +0.63 [−1.65, +2.91], 3/4 |
| 124 | 17 (31 October 19:15) | MaterioMiner coarse, 4 papers | multi-task + MR6 + names − multi-task + MR3 | pre-specified | +0.91 [−0.14, +1.97], 3/4 | +1.32 [−1.99, +4.63] | +1.07 [+0.09, +2.05] | +0.91 [−0.14, +1.97] | +1.27 [−0.89, +3.42], 3/4 |
| 125 | 17 (31 October 19:15) | MaterioMiner coarse, 4 papers | multi-task + MR6 + names − multi-task | pre-specified | +3.21 [+0.46, +5.96], 4/4 | +5.53 [+1.63, +9.44] | +3.87 [+1.01, +6.72] | +3.21 [+0.46, +5.96] | +3.23 [−0.32, +6.78], 3/4 |
| 126 | 17 (31 October 19:15) | MaterioMiner coarse, 4 papers | multi-task + MR6 + names − CRF head | pre-specified | +2.65 [−0.09, +5.38], 4/4 | +3.82 [−0.07, +7.71] | +3.33 [+0.39, +6.27] | +2.65 [−0.09, +5.38] | +2.92 [−0.61, +6.45], 3/4 |
| 127 | 17 (31 October 19:15) | MaterioMiner coarse, 4 papers | multi-task + MR6 − multi-task + MR3 | pre-specified | +0.06 [−0.74, +0.86], 3/4 | +0.83 [−1.41, +3.07] | +0.29 [−0.42, +1.01] | +0.06 [−0.74, +0.86] | +0.64 [−0.83, +2.11], 2/4 |
| 128 | 17 (31 October 19:15) | MaterioMiner coarse, 4 papers | multi-task + MR6 − multi-task | pre-specified | +2.36 [+1.15, +3.56], 4/4 | +5.04 [+2.55, +7.54] | +3.09 [+1.61, +4.57] | +2.36 [+1.15, +3.56] | +2.61 [+0.83, +4.38], 4/4 |
| 129 | 17 (31 October 19:15) | MaterioMiner coarse, 4 papers | multi-task + MR6 − CRF head | pre-specified | +1.80 [+0.19, +3.40], 4/4 | +3.33 [−0.09, +6.75] | +2.55 [+0.58, +4.52] | +1.80 [+0.19, +3.40] | +2.29 [+0.61, +3.97], 4/4 |
| 130 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: recall on the unseen test names the lexicon holds | D11: passes | +11.24 [+3.31, +19.18], 7/7 |  |  |  |  |
| 131 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: micro F1 | D12: no | +3.26 [−0.79, +7.31], 5/7 |  |  |  |  |
| 132 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: precision on all test entities | pre-specified | +5.78 [−1.22, +12.79], 5/7 |  |  |  |  |
| 133 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: recall on the unseen test names | pre-specified | +1.12 [+0.09, +2.15], 6/7 |  |  |  |  |
| 134 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: precision on the unseen test names | pre-specified | +6.17 [−1.27, +13.61], 5/7 |  |  |  |  |
| 135 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: F1 on the unseen test names | pre-specified | +3.54 [−0.58, +7.67], 5/7 |  |  |  |  |
| 136 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: recall on the unseen test names the lexicon does not hold | pre-specified | −1.73 [−3.02, −0.44], 1/7 |  |  |  |  |
| 137 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: the recall difference on the names the lexicon holds minus that on the others | pre-specified | +12.98 [+4.18, +21.77], 6/7 |  |  |  |  |
| 138 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900) | MR6 at matched recall − MR6 at bias 0: micro F1 | pre-specified | −2.75 [−5.77, +0.27], 2/7 |  |  |  |  |
| 139 | 18 (1 November 10:51) | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: recall on the unseen test names the lexicon holds | pre-specified (D11's reading without the flagged subfields) | +6.05 [−3.33, +15.44], 4/4 |  |  |  |  |
| 140 | 18 (1 November 10:51) | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: micro F1 | pre-specified (D12's reading without the flagged subfields) | −0.03 [−2.78, +2.72], 2/4 |  |  |  |  |
| 141 | 18 (1 November 10:51) | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: precision on all test entities | pre-specified | +0.56 [−6.74, +7.85], 2/4 |  |  |  |  |
| 142 | 18 (1 November 10:51) | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: recall on the unseen test names | pre-specified | +0.33 [−0.35, +1.01], 3/4 |  |  |  |  |
| 143 | 18 (1 November 10:51) | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: precision on the unseen test names | pre-specified | +0.56 [−6.84, +7.96], 2/4 |  |  |  |  |
| 144 | 18 (1 November 10:51) | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: F1 on the unseen test names | pre-specified | +0.20 [−2.58, +2.98], 2/4 |  |  |  |  |
| 145 | 18 (1 November 10:51) | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: recall on the unseen test names the lexicon does not hold | pre-specified | −0.88 [−2.63, +0.87], 1/4 |  |  |  |  |
| 146 | 18 (1 November 10:51) | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: the recall difference on the names the lexicon holds minus that on the others | pre-specified | +6.94 [−3.59, +17.46], 3/4 |  |  |  |  |
| 147 | 18 (1 November 10:51) | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 at matched recall − MR6 at bias 0: micro F1 | pre-specified | −0.23 [−1.85, +1.40], 2/4 |  |  |  |  |
| 148 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: recall on the unseen test names the lexicon holds | post hoc (D11's reading with the flagged at the nearest recall) | +8.63 [+3.35, +13.90], 7/7 |  |  |  |  |
| 149 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: micro F1 | post hoc (D12's reading with the flagged at the nearest recall) | +1.37 [−0.66, +3.41], 5/7 |  |  |  |  |
| 150 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: precision on all test entities | post hoc | +3.04 [−1.32, +7.41], 5/7 |  |  |  |  |
| 151 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: recall on the unseen test names | post hoc | +0.25 [−0.08, +0.58], 5/7 |  |  |  |  |
| 152 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: precision on the unseen test names | post hoc | +3.24 [−1.32, +7.80], 5/7 |  |  |  |  |
| 153 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: F1 on the unseen test names | post hoc | +1.54 [−0.45, +3.54], 5/7 |  |  |  |  |
| 154 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: recall on the unseen test names the lexicon does not hold | post hoc | −2.07 [−3.68, −0.46], 1/7 |  |  |  |  |
| 155 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: the recall difference on the names the lexicon holds minus that on the others | post hoc | +10.70 [+4.33, +17.07], 6/7 |  |  |  |  |
| 156 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 at matched recall − MR6 at bias 0: micro F1 | post hoc | −0.86 [−1.93, +0.21], 2/7 |  |  |  |  |
| 157 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at its best bias of the grid: recall on the unseen test names the lexicon holds | post hoc | +6.52 [+2.15, +10.89], 7/7 |  |  |  |  |
| 158 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at its best bias of the grid: recall on the unseen test names the lexicon does not hold | post hoc | −4.16 [−5.82, −2.50], 0/7 |  |  |  |  |
| 159 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at bias 0: recall on the unseen test names the lexicon holds | post hoc | +7.29 [+3.27, +11.30], 7/7 |  |  |  |  |
| 160 | 18 (1 November 10:51) | MuLMS, 7 subfields (seeds 300–900) | MR6 at bias 0 − MR6 at matched recall: recall on the unseen test names the lexicon holds | post hoc | +3.96 [−1.06, +8.98], 4/7 |  |  |  |  |
| 161 | 18 (1 November 10:51) | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at bias 0: recall on the unseen test names the lexicon holds | post hoc | +5.92 [−1.10, +12.94], 4/4 |  |  |  |  |
| 162 | 18 (1 November 10:51) | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 at bias 0 − MR6 at matched recall: recall on the unseen test names the lexicon holds | post hoc | +0.14 [−4.89, +5.16], 1/4 |  |  |  |  |

**Table S12.** Per row of Table S11 (#): the per-unit differences of its
micro (or named) comparison and of its votes (MuLMS's subfields;
MaterioMiner's papers s11340, ncomms7164, s41467 and met10111458;
SOFC-Exp's folds), and its votes' macro and weighted F1 differences with
their 95% CIs.

| # | corpus, units (MuLMS seeds) | comparison | per unit | votes per unit | votes: Δ macro F1 [95% CI] | votes: Δ weighted F1 [95% CI] |
|---|---|---|---|---|---|---|
| 1 | MuLMS, 7 subfields (seeds 0–900) | MR3 − CRF head | electrolysis +1.42, graphene +1.66, pemfc +2.28, poly −0.73, semi +2.52, sofc +7.39, steel +0.15 | electrolysis +2.12, graphene +2.12, pemfc +3.18, poly −0.65, semi +3.38, sofc +7.96, steel +0.88 | +3.22 [+1.63, +4.80] | +3.50 [+1.12, +5.88] |
| 5 | MuLMS, 7 subfields (seeds 0–900) | names from the other subfields' sentences − CRF head | electrolysis +2.64, graphene +0.38, pemfc +1.61, poly −0.28, semi +1.16, sofc +8.74, steel −0.00 | electrolysis +3.41, graphene +1.29, pemfc +2.33, poly −0.62, semi +2.26, sofc +8.86, steel −0.18 | +3.18 [+1.75, +4.60] | +3.16 [+0.55, +5.76] |
| 6 | MuLMS, 7 subfields (seeds 0–900) | names from the other subfields' sentences − MR3 | electrolysis +1.21, graphene −1.28, pemfc −0.67, poly +0.45, semi −1.36, sofc +1.35, steel −0.15 | electrolysis +1.29, graphene −0.83, pemfc −0.85, poly +0.03, semi −1.12, sofc +0.91, steel −1.06 | −0.04 [−0.75, +0.67] | −0.34 [−1.29, +0.60] |
| 7 | MaterioMiner fine, 4 papers | names from the fatigue corpus's sentences − MR3 | s11340 −0.50, ncomms7164 +0.12, s41467 +0.42, met10111458 +0.87 | s11340 −1.25, ncomms7164 +1.55, s41467 −0.27, met10111458 +1.72 | +0.86 [−3.22, +4.95] | +0.26 [−2.18, +2.69] |
| 8 | MaterioMiner coarse, 4 papers | names from the fatigue corpus's sentences − MR3 | s11340 −0.62, ncomms7164 +1.43, s41467 +0.24, met10111458 +1.38 | s11340 −1.17, ncomms7164 +1.35, s41467 +0.89, met10111458 −0.12 | +0.92 [−1.88, +3.72] | +0.36 [−1.78, +2.50] |
| 9 | MaterioMiner fine, 4 papers | names from the fatigue corpus's sentences − CRF head | s11340 +2.97, ncomms7164 +3.36, s41467 +2.11, met10111458 +3.55 | s11340 +4.09, ncomms7164 +4.19, s41467 +3.19, met10111458 +4.73 | +3.90 [+1.10, +6.71] | +4.39 [+2.06, +6.72] |
| 10 | MaterioMiner coarse, 4 papers | names from the fatigue corpus's sentences − CRF head | s11340 +0.99, ncomms7164 +3.53, s41467 +0.53, met10111458 +2.52 | s11340 +2.58, ncomms7164 +4.15, s41467 +1.10, met10111458 +2.15 | +4.20 [−0.11, +8.50] | +3.20 [+0.39, +6.01] |
| 11 | MaterioMiner fine, 4 papers | the votes: names from the fatigue corpus's sentences − multi-task | s11340 −2.30, ncomms7164 +0.36, s41467 +1.15, met10111458 +2.38 |  |  |  |
| 12 | MaterioMiner coarse, 4 papers | the votes: names from the fatigue corpus's sentences − multi-task | s11340 +3.52, ncomms7164 +3.93, s41467 +1.32, met10111458 +2.47 |  |  |  |
| 19 | MuLMS, 7 subfields (seeds 0–200) | MR3 + names, sentence filter (V1) − MR3 | electrolysis +0.17, graphene +0.93, pemfc +0.88, poly −1.40, semi +0.75, sofc +4.55, steel −0.30 | electrolysis −0.44, graphene −0.30, pemfc +1.33, poly −2.20, semi +0.03, sofc +3.67, steel −1.38 | +0.13 [−3.58, +3.84] | +0.20 [−1.94, +2.34] |
| 20 | MuLMS, 7 subfields (seeds 0–200) | MR3 + names, sentence filter (V1) − CRF head | electrolysis +1.10, graphene +2.56, pemfc +2.91, poly −5.39, semi +6.82, sofc +9.98, steel −0.46 | electrolysis +2.73, graphene +2.79, pemfc +3.31, poly −6.07, semi +6.95, sofc +10.95, steel −0.29 | +2.51 [−3.39, +8.41] | +3.43 [−2.22, +9.09] |
| 21 | MuLMS, 7 subfields (seeds 0–200) | MR3 + names, mention filter (V2) − MR3 | electrolysis −1.42, graphene +0.96, pemfc −0.01, poly +1.85, semi −0.22, sofc +4.94, steel +0.67 | electrolysis −2.07, graphene −0.21, pemfc +0.37, poly +0.98, semi −1.13, sofc +3.65, steel −0.37 | +1.13 [−0.54, +2.81] | +0.64 [−0.65, +1.92] |
| 22 | MuLMS, 7 subfields (seeds 0–200) | MR3 + names, mention filter (V2) − CRF head | electrolysis −0.49, graphene +2.59, pemfc +2.02, poly −2.14, semi +5.85, sofc +10.37, steel +0.51 | electrolysis +1.10, graphene +2.88, pemfc +2.35, poly −2.89, semi +5.79, sofc +10.93, steel +0.72 | +3.52 [+1.65, +5.38] | +3.87 [+0.22, +7.52] |
| 23 | MuLMS, 7 subfields (seeds 0–200) | MR6 (V3) − MR3 | electrolysis +2.04, graphene +0.63, pemfc +1.76, poly +0.54, semi −7.35, sofc +1.19, steel −0.21 | electrolysis +1.34, graphene −0.98, pemfc +1.74, poly +0.16, semi −7.55, sofc +0.32, steel −1.76 | +0.44 [−1.73, +2.61] | −0.72 [−4.00, +2.56] |
| 24 | MuLMS, 7 subfields (seeds 0–200) | MR6 (V3) − CRF head | electrolysis +2.97, graphene +2.25, pemfc +3.79, poly −3.45, semi −1.27, sofc +6.61, steel −0.37 | electrolysis +4.51, graphene +2.11, pemfc +3.72, poly −3.71, semi −0.63, sofc +7.60, steel −0.67 | +2.82 [−0.00, +5.64] | +2.51 [−0.62, +5.64] |
| 25 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names (V4) − MR3 | electrolysis +1.13, graphene +3.11, pemfc −0.24, poly +0.27, semi +0.31, sofc +7.42, steel −1.84 | electrolysis −1.63, graphene +1.50, pemfc +0.18, poly +0.24, semi +0.16, sofc +5.19, steel −1.72 | +1.70 [+0.25, +3.15] | +1.58 [+0.16, +3.00] |
| 26 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names (V4) − CRF head | electrolysis +2.06, graphene +4.73, pemfc +1.79, poly −3.72, semi +6.39, sofc +12.85, steel −2.00 | electrolysis +1.54, graphene +4.59, pemfc +2.16, poly −3.63, semi +7.08, sofc +12.47, steel −0.63 | +4.09 [+0.61, +7.56] | +4.82 [+0.55, +9.08] |
| 27 | MuLMS, 7 subfields (seeds 0–200) | self-training on the corpus's silver sentences (V6) − MR3 | electrolysis +1.57, graphene +1.55, pemfc −1.73, poly +4.96, semi −2.32, sofc −2.07, steel −0.84 | electrolysis −0.45, graphene +0.73, pemfc −2.62, poly +4.22, semi −2.99, sofc −3.88, steel −2.78 | −1.98 [−4.07, +0.11] | −2.64 [−5.49, +0.20] |
| 28 | MuLMS, 7 subfields (seeds 0–200) | self-training on the corpus's silver sentences (V6) − CRF head | electrolysis +2.50, graphene +3.17, pemfc +0.30, poly +0.97, semi +3.76, sofc +3.36, steel −1.00 | electrolysis +2.72, graphene +3.82, pemfc −0.64, poly +0.35, semi +3.93, sofc +3.40, steel −1.69 | +0.40 [−2.36, +3.17] | +0.59 [−2.18, +3.36] |
| 29 | MuLMS, 7 subfields (seeds 0–200) | lexicon as a gazetteer feature, mention filter (V7) − MR3 | electrolysis +0.73, graphene −3.49, pemfc −3.03, poly +1.34, semi −5.89, sofc −1.75, steel −0.03 | electrolysis −1.79, graphene −4.28, pemfc −3.08, poly +0.92, semi −7.24, sofc −1.38, steel −0.51 | −2.16 [−5.74, +1.41] | −2.76 [−5.84, +0.33] |
| 30 | MuLMS, 7 subfields (seeds 0–200) | lexicon as a gazetteer feature, mention filter (V7) − CRF head | electrolysis +1.66, graphene −1.86, pemfc −1.01, poly −2.65, semi +0.19, sofc +3.68, steel −0.19 | electrolysis +1.38, graphene −1.19, pemfc −1.10, poly −2.95, semi −0.32, sofc +5.90, steel +0.58 | +0.22 [−3.00, +3.44] | +0.47 [−2.60, +3.55] |
| 31 | MuLMS, 7 subfields (seeds 0–200) | lexicon as a gazetteer feature, sentence filter (V8) − MR3 | electrolysis −1.28, graphene −1.43, pemfc −2.63, poly +0.43, semi −5.55, sofc −1.70, steel −0.80 | electrolysis −2.88, graphene −2.00, pemfc −1.54, poly −0.10, semi −6.30, sofc −2.89, steel −1.75 | −2.27 [−4.83, +0.29] | −2.89 [−4.93, −0.85] |
| 32 | MuLMS, 7 subfields (seeds 0–200) | lexicon as a gazetteer feature, sentence filter (V8) − CRF head | electrolysis −0.35, graphene +0.20, pemfc −0.60, poly −3.56, semi +0.52, sofc +3.73, steel −0.96 | electrolysis +0.29, graphene +1.09, pemfc +0.44, poly −3.97, semi +0.62, sofc +4.39, steel −0.66 | +0.11 [−1.73, +1.96] | +0.34 [−2.32, +3.01] |
| 33 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − CRF head | electrolysis +2.52, graphene +0.47, pemfc +3.43, poly −2.35, semi +6.12, sofc +11.86, steel +0.28 | electrolysis +2.00, graphene −0.42, pemfc +4.03, poly −2.38, semi +7.29, sofc +11.30, steel +0.80 | +4.62 [+3.42, +5.82] | +4.30 [+0.49, +8.12] |
| 34 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR3 | electrolysis +0.89, graphene −1.21, pemfc +1.04, poly −3.02, semi +5.12, sofc +3.62, steel +0.00 | electrolysis +0.41, graphene −2.05, pemfc +0.25, poly −3.34, semi +5.68, sofc +3.00, steel +0.02 | +0.99 [−0.75, +2.72] | +0.67 [−1.96, +3.30] |
| 35 | MuLMS, 7 subfields (seeds 0–900) | MR6 + names − CRF head | electrolysis +2.39, graphene +1.75, pemfc +2.94, poly −2.76, semi +6.20, sofc +12.16, steel −0.40 | electrolysis +1.85, graphene +1.25, pemfc +3.40, poly −2.80, semi +7.22, sofc +11.69, steel +0.32 | +4.44 [+2.63, +6.26] | +4.48 [+0.72, +8.23] |
| 36 | MuLMS, 7 subfields (seeds 0–900) | MR6 + names − MR3 | electrolysis +0.96, graphene +0.09, pemfc +0.65, poly −2.03, semi +3.68, sofc +4.76, steel −0.55 | electrolysis −0.27, graphene −0.87, pemfc +0.22, poly −2.15, semi +3.84, sofc +3.73, steel −0.56 | +1.22 [−0.20, +2.65] | +0.98 [−0.77, +2.72] |
| 37 | MaterioMiner fine, 4 papers | MR6 + names − CRF head | s11340 +1.99, ncomms7164 +5.67, s41467 +3.64, met10111458 +5.78 | s11340 +4.36, ncomms7164 +4.91, s41467 +4.16, met10111458 +6.55 | +5.10 [+1.24, +8.95] | +5.61 [+1.83, +9.39] |
| 38 | MaterioMiner coarse, 4 papers | MR6 + names − CRF head | s11340 +1.74, ncomms7164 +2.56, s41467 +0.88, met10111458 +1.91 | s11340 +2.51, ncomms7164 +3.59, s41467 +0.77, met10111458 +1.70 | +2.94 [−0.81, +6.70] | +2.79 [+0.26, +5.32] |
| 39 | MaterioMiner fine, 4 papers | MR6 + names − MR3 | s11340 −1.49, ncomms7164 +2.43, s41467 +1.95, met10111458 +3.09 | s11340 −0.98, ncomms7164 +2.26, s41467 +0.70, met10111458 +3.53 | +2.06 [−1.06, +5.17] | +1.48 [−2.10, +5.05] |
| 40 | MaterioMiner coarse, 4 papers | MR6 + names − MR3 | s11340 +0.13, ncomms7164 +0.47, s41467 +0.59, met10111458 +0.77 | s11340 −1.25, ncomms7164 +0.80, s41467 +0.56, met10111458 −0.58 | −0.33 [−2.43, +1.76] | −0.05 [−1.79, +1.69] |
| 41 | MaterioMiner fine, 4 papers | self-training on the fatigue corpus's silver sentences (V6's analogue) − CRF head | s11340 +2.36, ncomms7164 −0.81, s41467 +1.93, met10111458 +3.49 | s11340 +2.35, ncomms7164 −2.48, s41467 +2.10, met10111458 +2.52 | +1.82 [−1.83, +5.48] | +0.36 [−2.85, +3.56] |
| 42 | MaterioMiner coarse, 4 papers | self-training on the fatigue corpus's silver sentences (V6's analogue) − CRF head | s11340 +2.65, ncomms7164 +1.79, s41467 −0.67, met10111458 +2.47 | s11340 +2.35, ncomms7164 +1.40, s41467 −1.42, met10111458 +1.42 | +0.91 [−2.82, +4.64] | +0.97 [−1.79, +3.73] |
| 43 | MaterioMiner fine, 4 papers | self-training on the fatigue corpus's silver sentences (V6's analogue) − MR3 | s11340 −1.11, ncomms7164 −4.04, s41467 +0.24, met10111458 +0.80 | s11340 −2.99, ncomms7164 −5.13, s41467 −1.35, met10111458 −0.49 | −1.22 [−2.14, −0.29] | −3.78 [−6.22, −1.34] |
| 44 | MaterioMiner coarse, 4 papers | self-training on the fatigue corpus's silver sentences (V6's analogue) − MR3 | s11340 +1.04, ncomms7164 −0.31, s41467 −0.96, met10111458 +1.33 | s11340 −1.41, ncomms7164 −1.40, s41467 −1.63, met10111458 −0.86 | −2.36 [−5.21, +0.48] | −1.87 [−2.57, −1.17] |
| 45 | MuLMS, 7 subfields (seeds 0–900) | CRF head + label propagation − CRF head | electrolysis +0.54, graphene +0.47, pemfc +1.35, poly +0.07, semi +0.93, sofc +2.07, steel +0.29 |  |  |  |
| 46 | MuLMS, 7 subfields (seeds 0–900) | CRF head + label propagation with retyping − CRF head | electrolysis +0.81, graphene +0.50, pemfc +1.69, poly +0.09, semi +1.13, sofc +2.37, steel +0.50 |  |  |  |
| 47 | MuLMS, 7 subfields (seeds 0–900) | MR3 + label propagation − MR3 | electrolysis +0.35, graphene +0.20, pemfc +0.61, poly +0.02, semi +0.68, sofc +2.13, steel −0.03 |  |  |  |
| 48 | MuLMS, 7 subfields (seeds 0–900) | MR3 + label propagation with retyping − MR3 | electrolysis +0.46, graphene +0.30, pemfc +1.27, poly +0.03, semi +1.22, sofc +2.74, steel +0.14 |  |  |  |
| 49 | MaterioMiner fine, 4 papers | CRF head + label propagation − CRF head | s11340 +0.52, ncomms7164 −0.00, s41467 +0.76, met10111458 −0.09 |  |  |  |
| 50 | MaterioMiner fine, 4 papers | CRF head + label propagation with retyping − CRF head | s11340 +0.52, ncomms7164 −0.41, s41467 +1.11, met10111458 +0.03 |  |  |  |
| 51 | MaterioMiner fine, 4 papers | MR3 + label propagation − MR3 | s11340 +0.52, ncomms7164 +0.17, s41467 +0.35, met10111458 −0.26 |  |  |  |
| 52 | MaterioMiner fine, 4 papers | MR3 + label propagation with retyping − MR3 | s11340 +0.52, ncomms7164 +0.05, s41467 +0.80, met10111458 −0.50 |  |  |  |
| 53 | MaterioMiner fine, 4 papers | multi-task + MR3 + label propagation − multi-task + MR3 | s11340 +0.41, ncomms7164 +0.31, s41467 +0.86, met10111458 −0.33 |  |  |  |
| 54 | MaterioMiner fine, 4 papers | multi-task + MR3 + label propagation with retyping − multi-task + MR3 | s11340 +0.41, ncomms7164 −0.41, s41467 +1.23, met10111458 −0.51 |  |  |  |
| 55 | MaterioMiner coarse, 4 papers | CRF head + label propagation − CRF head | s11340 +0.07, ncomms7164 +0.15, s41467 +0.06, met10111458 +0.21 |  |  |  |
| 56 | MaterioMiner coarse, 4 papers | CRF head + label propagation with retyping − CRF head | s11340 +0.42, ncomms7164 +0.90, s41467 +0.49, met10111458 +0.77 |  |  |  |
| 57 | MaterioMiner coarse, 4 papers | MR3 + label propagation − MR3 | s11340 +0.01, ncomms7164 +0.14, s41467 +0.03, met10111458 +0.27 |  |  |  |
| 58 | MaterioMiner coarse, 4 papers | MR3 + label propagation with retyping − MR3 | s11340 +0.30, ncomms7164 +0.89, s41467 +0.92, met10111458 +0.43 |  |  |  |
| 59 | MaterioMiner coarse, 4 papers | multi-task + MR3 + label propagation − multi-task + MR3 | s11340 +0.09, ncomms7164 +0.19, s41467 −0.03, met10111458 +0.15 |  |  |  |
| 60 | MaterioMiner coarse, 4 papers | multi-task + MR3 + label propagation with retyping − multi-task + MR3 | s11340 +0.03, ncomms7164 +1.10, s41467 +0.32, met10111458 +0.94 |  |  |  |
| 61 | SOFC-Exp, 5 folds | CRF head + label propagation − CRF head | fold0 −0.75, fold1 −0.30, fold2 −0.41, fold3 −0.17, fold4 −0.57 |  |  |  |
| 62 | SOFC-Exp, 5 folds | CRF head + label propagation with retyping − CRF head | fold0 −0.69, fold1 −0.30, fold2 −0.40, fold3 −0.19, fold4 −0.57 |  |  |  |
| 63 | SOFC-Exp, 5 folds | MR3 − CRF head | fold0 −0.39, fold1 −0.42, fold2 −0.81, fold3 −1.36, fold4 +0.56 |  |  |  |
| 64 | SOFC-Exp, 5 folds | multi-task + MR3 − CRF head | fold0 −0.12, fold1 −0.07, fold2 −0.65, fold3 −0.34, fold4 +0.61 |  |  |  |
| 65 | SOFC-Exp, 5 folds | multi-task + MR3 − MR3 | fold0 +0.27, fold1 +0.35, fold2 +0.16, fold3 +1.02, fold4 +0.04 |  |  |  |
| 67 | SOFC-Exp, 5 folds | MR6 + names − CRF head | fold0 −0.25, fold1 +0.03, fold2 −0.60, fold3 −0.88, fold4 +1.54 |  |  |  |
| 68 | SOFC-Exp, 5 folds | MR6 + names − MR3 | fold0 +0.13, fold1 +0.45, fold2 +0.21, fold3 +0.47, fold4 +0.97 |  |  |  |
| 69 | pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR6 + names − CRF head | electrolysis +2.39, graphene +1.75, pemfc +2.94, poly −2.76, semi +6.20, sofc +12.16, steel −0.40, s11340 +1.99, ncomms7164 +5.67, s41467 +3.64, met10111458 +5.78 |  |  |  |
| 70 | pooled, 16 units: MuLMS 7 (0–900) + MaterioMiner fine 4 + SOFC-Exp 5 | MR6 + names − CRF head | electrolysis +2.39, graphene +1.75, pemfc +2.94, poly −2.76, semi +6.20, sofc +12.16, steel −0.40, s11340 +1.99, ncomms7164 +5.67, s41467 +3.64, met10111458 +5.78, fold0 −0.25, fold1 +0.03, fold2 −0.60, fold3 −0.88, fold4 +1.54 |  |  |  |
| 71 | pooled, 16 units: MuLMS 7 (0–900) + MaterioMiner fine 4 + SOFC-Exp 5 | MR3 − CRF head | electrolysis +1.42, graphene +1.66, pemfc +2.28, poly −0.73, semi +2.52, sofc +7.39, steel +0.15, s11340 +3.48, ncomms7164 +3.24, s41467 +1.69, met10111458 +2.69, fold0 −0.39, fold1 −0.42, fold2 −0.81, fold3 −1.36, fold4 +0.56 |  |  |  |
| 72 | pooled, 9 units: MaterioMiner fine 4 + SOFC-Exp 5 | multi-task + MR3 − CRF head | s11340 +7.16, ncomms7164 +7.04, s41467 +6.25, met10111458 +6.46, fold0 −0.12, fold1 −0.07, fold2 −0.65, fold3 −0.34, fold4 +0.61 |  |  |  |
| 73 | pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4 | names from the other subfields' sentences (MuLMS) and the fatigue corpus's (MaterioMiner) − CRF head | electrolysis +2.64, graphene +0.38, pemfc +1.61, poly −0.28, semi +1.16, sofc +8.74, steel −0.00, s11340 +2.97, ncomms7164 +3.36, s41467 +2.11, met10111458 +3.55 |  |  |  |
| 74 | pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR6 + names − MR3 | electrolysis +0.96, graphene +0.09, pemfc +0.65, poly −2.03, semi +3.68, sofc +4.76, steel −0.55, s11340 −1.49, ncomms7164 +2.43, s41467 +1.95, met10111458 +3.09 |  |  |  |
| 75 | pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR3 − CRF head | electrolysis +1.42, graphene +1.66, pemfc +2.28, poly −0.73, semi +2.53, sofc +7.39, steel +0.15, s11340 +3.48, ncomms7164 +3.24, s41467 +1.69, met10111458 +2.69 |  |  |  |
| 76 | pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4 | MR6 + names − CRF head | electrolysis +2.52, graphene +0.47, pemfc +3.43, poly −2.35, semi +6.12, sofc +11.86, steel +0.28, s11340 +1.99, ncomms7164 +5.67, s41467 +3.64, met10111458 +5.78 |  |  |  |
| 77 | pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4 | MR6 + names − MR3 | electrolysis +0.89, graphene −1.21, pemfc +1.04, poly −3.02, semi +5.12, sofc +3.62, steel +0.00, s11340 −1.49, ncomms7164 +2.43, s41467 +1.95, met10111458 +3.09 |  |  |  |
| 78 | pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4 | MR3 − CRF head | electrolysis +1.63, graphene +1.68, pemfc +2.40, poly +0.67, semi +1.01, sofc +8.24, steel +0.28, s11340 +3.48, ncomms7164 +3.24, s41467 +1.69, met10111458 +2.69 |  |  |  |
| 79 | pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4 | MR6 + names − MR6 | electrolysis +0.13, graphene −1.24, pemfc +1.16, poly −2.28, semi +3.27, sofc +3.32, steel −0.77, s11340 −2.08, ncomms7164 +1.35, s41467 +3.28, met10111458 +2.63 |  |  |  |
| 80 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 | electrolysis +0.13, graphene −1.24, pemfc +1.16, poly −2.28, semi +3.27, sofc +3.32, steel −0.77 | electrolysis −0.30, graphene −1.78, pemfc +1.83, poly −2.27, semi +3.79, sofc +1.72, steel −0.56 | +0.92 [−0.21, +2.05] | +0.74 [−1.15, +2.64] |
| 81 | MuLMS, 7 subfields (seeds 0–900) | MR6 + names − MR6 | electrolysis −0.18, graphene −0.12, pemfc +0.21, poly −1.68, semi +4.59, sofc +4.20, steel −1.03 | electrolysis −1.19, graphene −0.36, pemfc +0.70, poly −1.49, semi +5.09, sofc +2.77, steel −0.36 | +1.04 [−0.13, +2.20] | +1.26 [−0.86, +3.38] |
| 82 | MaterioMiner fine, 4 papers | MR6 + names − MR6 | s11340 −2.08, ncomms7164 +1.35, s41467 +3.28, met10111458 +2.63 | s11340 −0.91, ncomms7164 +0.13, s41467 +2.06, met10111458 +2.48 | +1.04 [−0.43, +2.52] | +0.74 [−1.40, +2.88] |
| 83 | MaterioMiner coarse, 4 papers | MR6 + names − MR6 | s11340 +2.36, ncomms7164 −1.08, s41467 +0.37, met10111458 +1.16 | s11340 +2.23, ncomms7164 −0.08, s41467 +0.48, met10111458 +1.63 | −0.36 [−3.20, +2.47] | +0.86 [−1.16, +2.87] |
| 84 | MuLMS, 7 subfields (seeds 300–900) | MR6 − CRF head | electrolysis +2.40, graphene +1.71, pemfc +2.28, poly −0.07, semi +2.86, sofc +8.54, steel +1.05 | electrolysis +2.30, graphene +1.36, pemfc +2.20, poly −0.11, semi +3.51, sofc +9.58, steel +1.36 | +3.70 [+2.78, +4.62] | +3.56 [+1.07, +6.05] |
| 85 | MuLMS, 7 subfields (seeds 300–900) | MR6 − MR3 | electrolysis +0.77, graphene +0.03, pemfc −0.12, poly −0.74, semi +1.85, sofc +0.30, steel +0.77 | electrolysis +0.70, graphene −0.27, pemfc −1.58, poly −1.08, semi +1.89, sofc +1.28, steel +0.59 | +0.07 [−1.04, +1.17] | −0.07 [−1.11, +0.97] |
| 86 | MuLMS, 7 subfields (seeds 0–900) | MR6 − CRF head | electrolysis +2.57, graphene +1.87, pemfc +2.73, poly −1.08, semi +1.62, sofc +7.96, steel +0.62 | electrolysis +3.04, graphene +1.61, pemfc +2.70, poly −1.31, semi +2.13, sofc +8.92, steel +0.68 | +3.41 [+2.06, +4.76] | +3.21 [+0.70, +5.73] |
| 87 | MuLMS, 7 subfields (seeds 0–900) | MR6 − MR3 | electrolysis +1.15, graphene +0.21, pemfc +0.45, poly −0.35, semi −0.91, sofc +0.57, steel +0.47 | electrolysis +0.92, graphene −0.50, pemfc −0.48, poly −0.66, semi −1.26, sofc +0.96, steel −0.20 | +0.19 [−0.63, +1.01] | −0.29 [−1.15, +0.58] |
| 88 | MaterioMiner fine, 4 papers | MR6 − CRF head | s11340 +4.07, ncomms7164 +4.32, s41467 +0.36, met10111458 +3.15 | s11340 +5.27, ncomms7164 +4.77, s41467 +2.10, met10111458 +4.07 | +4.05 [−0.81, +8.92] | +4.87 [+1.71, +8.02] |
| 89 | MaterioMiner fine, 4 papers | MR6 − MR3 | s11340 +0.60, ncomms7164 +1.08, s41467 −1.33, met10111458 +0.46 | s11340 −0.08, ncomms7164 +2.13, s41467 −1.36, met10111458 +1.06 | +1.01 [−1.58, +3.60] | +0.73 [−1.54, +3.00] |
| 90 | MaterioMiner coarse, 4 papers | MR6 − CRF head | s11340 −0.62, ncomms7164 +3.65, s41467 +0.52, met10111458 +0.74 | s11340 +0.27, ncomms7164 +3.68, s41467 +0.29, met10111458 +0.07 | +3.30 [−1.69, +8.30] | +1.93 [−1.15, +5.01] |
| 91 | MaterioMiner coarse, 4 papers | MR6 − MR3 | s11340 −2.23, ncomms7164 +1.55, s41467 +0.23, met10111458 −0.40 | s11340 −3.48, ncomms7164 +0.88, s41467 +0.08, met10111458 −2.20 | +0.03 [−3.61, +3.67] | −0.91 [−4.32, +2.50] |
| 92 | pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR6 − CRF head | electrolysis +2.57, graphene +1.87, pemfc +2.73, poly −1.08, semi +1.62, sofc +7.96, steel +0.62, s11340 +4.07, ncomms7164 +4.32, s41467 +0.36, met10111458 +3.15 |  |  |  |
| 93 | pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR6 + names − MR6 | electrolysis −0.18, graphene −0.12, pemfc +0.21, poly −1.68, semi +4.59, sofc +4.20, steel −1.03, s11340 −2.08, ncomms7164 +1.35, s41467 +3.28, met10111458 +2.63 |  |  |  |
| 94 | pooled, 11 units: MuLMS 7 (0–900) + MaterioMiner fine 4 | MR6 − MR3 | electrolysis +1.15, graphene +0.21, pemfc +0.45, poly −0.35, semi −0.91, sofc +0.57, steel +0.47, s11340 +0.60, ncomms7164 +1.08, s41467 −1.33, met10111458 +0.46 |  |  |  |
| 95 | pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4 | MR6 − CRF head | electrolysis +2.40, graphene +1.71, pemfc +2.28, poly −0.07, semi +2.86, sofc +8.54, steel +1.05, s11340 +4.07, ncomms7164 +4.32, s41467 +0.36, met10111458 +3.15 |  |  |  |
| 96 | pooled, 11 units: MuLMS 7 (300–900) + MaterioMiner fine 4 | MR6 − MR3 | electrolysis +0.77, graphene +0.03, pemfc −0.12, poly −0.74, semi +1.85, sofc +0.30, steel +0.77, s11340 +0.60, ncomms7164 +1.08, s41467 −1.33, met10111458 +0.46 |  |  |  |
| 97 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names, own-topic corpus − mixed corpus | electrolysis +0.49, graphene −0.26, pemfc −0.63, poly −2.53, semi +3.73, sofc −1.54, steel −0.54 | electrolysis −0.33, graphene −0.77, pemfc −0.33, poly −3.09, semi +4.49, sofc −2.46, steel −0.23 | −0.24 [−1.99, +1.51] | −0.33 [−2.78, +2.11] |
| 98 | MuLMS, 7 subfields (seeds 300–900) | the mixed-corpus variant − CRF head | electrolysis +2.04, graphene +0.73, pemfc +4.06, poly +0.18, semi +2.39, sofc +13.40, steel +0.83 | electrolysis +2.33, graphene +0.35, pemfc +4.36, poly +0.72, semi +2.80, sofc +13.76, steel +1.03 | +4.86 [+3.14, +6.58] | +4.64 [+0.47, +8.81] |
| 99 | MuLMS, 7 subfields (seeds 300–900) | the mixed-corpus variant − MR3 | electrolysis +0.40, graphene −0.95, pemfc +1.67, poly −0.49, semi +1.39, sofc +5.16, steel +0.54 | electrolysis +0.73, graphene −1.28, pemfc +0.58, poly −0.25, semi +1.19, sofc +5.46, steel +0.26 | +1.23 [−0.64, +3.09] | +1.00 [−1.38, +3.39] |
| 100 | MuLMS, 7 subfields (seeds 300–900) | the mixed-corpus variant − MR6 | electrolysis −0.36, graphene −0.98, pemfc +1.79, poly +0.25, semi −0.46, sofc +4.86, steel −0.22 | electrolysis +0.03, graphene −1.01, pemfc +2.16, poly +0.83, semi −0.70, sofc +4.18, steel −0.33 | +1.16 [−0.61, +2.94] | +1.08 [−1.02, +3.17] |
| 101 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, corpus: its first 10,000 sentences − MR6 + names | electrolysis +0.36, graphene −2.43, pemfc +1.87, poly +1.36, semi −2.27, sofc −1.01, steel +1.07 | electrolysis +1.09, graphene −1.54, pemfc +1.06, poly +0.94, semi −2.01, sofc +1.09, steel +0.78 | −0.28 [−2.46, +1.90] | −0.28 [−1.86, +1.30] |
| 102 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, corpus: its first 10,000 sentences − MR3 | electrolysis +1.49, graphene +0.68, pemfc +1.63, poly +1.63, semi −1.96, sofc +6.42, steel −0.78 | electrolysis −0.54, graphene −0.04, pemfc +1.24, poly +1.18, semi −1.85, sofc +6.28, steel −0.94 | +1.42 [−1.48, +4.33] | +1.30 [−1.04, +3.65] |
| 103 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, corpus: its first 10,000 sentences − CRF head | electrolysis +2.42, graphene +2.31, pemfc +3.66, poly −2.36, semi +4.11, sofc +11.84, steel −0.94 | electrolysis +2.63, graphene +3.05, pemfc +3.22, poly −2.69, semi +5.07, sofc +13.56, steel +0.15 | +3.81 [+1.03, +6.59] | +4.53 [+0.11, +8.96] |
| 104 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, corpus: its first 20,000 sentences − MR6 + names | electrolysis +0.90, graphene −2.03, pemfc +2.30, poly +0.81, semi −2.79, sofc −3.19, steel +2.90 | electrolysis +0.59, graphene −1.84, pemfc +1.75, poly +0.14, semi −3.40, sofc −1.09, steel +1.59 | −0.60 [−1.24, +0.04] | −0.86 [−2.29, +0.58] |
| 105 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, corpus: its first 20,000 sentences − MR3 | electrolysis +2.04, graphene +1.07, pemfc +2.06, poly +1.08, semi −2.47, sofc +4.24, steel +1.05 | electrolysis −1.04, graphene −0.34, pemfc +1.93, poly +0.38, semi −3.24, sofc +4.10, steel −0.13 | +1.10 [−0.74, +2.95] | +0.73 [−1.33, +2.79] |
| 106 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, corpus: its first 20,000 sentences − CRF head | electrolysis +2.97, graphene +2.70, pemfc +4.09, poly −2.91, semi +3.60, sofc +9.66, steel +0.89 | electrolysis +2.13, graphene +2.75, pemfc +3.91, poly −3.49, semi +3.68, sofc +11.38, steel +0.96 | +3.49 [+0.10, +6.88] | +3.96 [+0.13, +7.79] |
| 107 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, filter: marginal threshold 0.7 − MR6 + names | electrolysis +0.62, graphene −3.15, pemfc +0.18, poly −0.47, semi +1.05, sofc −0.96, steel +1.62 | electrolysis +2.14, graphene −2.31, pemfc −0.53, poly −0.13, semi +1.79, sofc +0.32, steel +1.12 | +0.37 [−0.43, +1.18] | +0.45 [−1.31, +2.21] |
| 108 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, filter: marginal threshold 0.7 − MR3 | electrolysis +1.75, graphene −0.05, pemfc −0.06, poly −0.20, semi +1.36, sofc +6.47, steel −0.22 | electrolysis +0.51, graphene −0.81, pemfc −0.35, poly +0.11, semi +1.95, sofc +5.51, steel −0.60 | +2.07 [+0.44, +3.71] | +2.03 [+0.56, +3.51] |
| 109 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, filter: marginal threshold 0.7 − CRF head | electrolysis +2.68, graphene +1.58, pemfc +1.97, poly −4.19, semi +7.44, sofc +11.89, steel −0.38 | electrolysis +3.68, graphene +2.28, pemfc +1.63, poly −3.76, semi +8.87, sofc +12.79, steel +0.49 | +4.46 [+0.74, +8.18] | +5.27 [+0.56, +9.97] |
| 110 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, filter: 2 of 3 teachers agree − MR6 + names | electrolysis −0.33, graphene −1.96, pemfc +1.00, poly +1.54, semi +1.85, sofc −2.74, steel +1.41 | electrolysis +0.24, graphene −1.67, pemfc +0.83, poly +3.02, semi +1.61, sofc −1.45, steel +0.58 | +0.10 [−0.60, +0.80] | +0.40 [−1.27, +2.08] |
| 111 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, filter: 2 of 3 teachers agree − MR3 | electrolysis +0.80, graphene +1.14, pemfc +0.76, poly +1.81, semi +2.16, sofc +4.69, steel −0.43 | electrolysis −1.39, graphene −0.17, pemfc +1.01, poly +3.26, semi +1.77, sofc +3.74, steel −1.14 | +1.80 [+0.70, +2.90] | +1.99 [−0.01, +3.98] |
| 112 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, filter: 2 of 3 teachers agree − CRF head | electrolysis +1.73, graphene +2.77, pemfc +2.79, poly −2.18, semi +8.24, sofc +10.11, steel −0.59 | electrolysis +1.78, graphene +2.92, pemfc +2.99, poly −0.61, semi +8.69, sofc +11.02, steel −0.05 | +4.18 [+0.79, +7.57] | +5.22 [+1.60, +8.84] |
| 113 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, pool: harvested names in half of the draws − MR6 + names | electrolysis −0.97, graphene −0.59, pemfc +1.40, poly +1.45, semi −2.28, sofc −1.31, steel +2.90 | electrolysis −0.50, graphene +0.35, pemfc +0.83, poly +0.28, semi −2.77, sofc +0.55, steel +1.84 | −0.17 [−2.86, +2.52] | −0.12 [−1.78, +1.53] |
| 114 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, pool: harvested names in half of the draws − MR3 | electrolysis +0.16, graphene +2.52, pemfc +1.16, poly +1.72, semi −1.97, sofc +6.11, steel +1.06 | electrolysis −2.13, graphene +1.85, pemfc +1.01, poly +0.52, semi −2.61, sofc +5.74, steel +0.12 | +1.53 [−2.07, +5.13] | +1.46 [−1.08, +4.01] |
| 115 | MuLMS, 7 subfields (seeds 0–200) | MR6 + names, pool: harvested names in half of the draws − CRF head | electrolysis +1.09, graphene +4.15, pemfc +3.19, poly −2.27, semi +4.10, sofc +11.54, steel +0.90 | electrolysis +1.04, graphene +4.94, pemfc +2.99, poly −3.35, semi +4.31, sofc +13.02, steel +1.21 | +3.91 [−0.73, +8.55] | +4.69 [+0.04, +9.35] |
| 116 | MaterioMiner fine, 4 papers | multi-task + MR6 + names − multi-task + MR6 | s11340 +1.14, ncomms7164 +0.16, s41467 +1.62, met10111458 +0.90 | s11340 −1.21, ncomms7164 −1.08, s41467 +1.70, met10111458 +1.35 | +1.63 [−1.05, +4.31] | +0.54 [−1.80, +2.88] |
| 117 | MaterioMiner fine, 4 papers | multi-task + MR6 + names − multi-task + MR3 | s11340 +0.82, ncomms7164 +1.47, s41467 +0.07, met10111458 +0.69 | s11340 +0.14, ncomms7164 −0.62, s41467 +0.13, met10111458 +2.55 | +1.39 [−2.71, +5.49] | +0.56 [−1.92, +3.05] |
| 118 | MaterioMiner fine, 4 papers | multi-task + MR6 + names − multi-task | s11340 +1.63, ncomms7164 +4.95, s41467 +4.78, met10111458 +3.74 | s11340 +0.97, ncomms7164 +3.86, s41467 +4.12, met10111458 +4.70 | +4.04 [+0.60, +7.47] | +3.61 [+1.37, +5.86] |
| 119 | MaterioMiner fine, 4 papers | multi-task + MR6 + names − CRF head | s11340 +7.98, ncomms7164 +8.52, s41467 +6.32, met10111458 +7.15 | s11340 +7.36, ncomms7164 +7.70, s41467 +6.15, met10111458 +7.05 | +8.62 [−0.03, +17.26] | +6.96 [+6.17, +7.74] |
| 120 | MaterioMiner fine, 4 papers | multi-task + MR6 − multi-task + MR3 | s11340 −0.33, ncomms7164 +1.31, s41467 −1.54, met10111458 −0.20 | s11340 +1.35, ncomms7164 +0.46, s41467 −1.57, met10111458 +1.20 | −0.24 [−3.91, +3.42] | +0.02 [−2.71, +2.75] |
| 121 | MaterioMiner fine, 4 papers | multi-task + MR6 − multi-task | s11340 +0.48, ncomms7164 +4.78, s41467 +3.16, met10111458 +2.84 | s11340 +2.18, ncomms7164 +4.94, s41467 +2.41, met10111458 +3.35 | +2.41 [−1.75, +6.56] | +3.08 [+1.27, +4.88] |
| 122 | MaterioMiner fine, 4 papers | multi-task + MR6 − CRF head | s11340 +6.83, ncomms7164 +8.35, s41467 +4.71, met10111458 +6.26 | s11340 +8.57, ncomms7164 +8.77, s41467 +4.45, met10111458 +5.70 | +6.99 [+0.06, +13.92] | +6.42 [+3.43, +9.40] |
| 123 | MaterioMiner coarse, 4 papers | multi-task + MR6 + names − multi-task + MR6 | s11340 +0.79, ncomms7164 +1.39, s41467 −0.51, met10111458 +1.73 | s11340 +0.86, ncomms7164 +1.47, s41467 −1.46, met10111458 +1.64 | +0.12 [−2.31, +2.56] | +0.63 [−1.38, +2.63] |
| 124 | MaterioMiner coarse, 4 papers | multi-task + MR6 + names − multi-task + MR3 | s11340 +1.20, ncomms7164 +1.41, s41467 −0.06, met10111458 +1.09 | s11340 +2.69, ncomms7164 +1.46, s41467 −0.57, met10111458 +1.48 | +1.77 [−4.28, +7.83] | +1.39 [−0.78, +3.57] |
| 125 | MaterioMiner coarse, 4 papers | multi-task + MR6 + names − multi-task | s11340 +3.32, ncomms7164 +4.26, s41467 +0.73, met10111458 +4.51 | s11340 +4.70, ncomms7164 +4.70, s41467 −0.01, met10111458 +3.54 | +5.55 [+0.05, +11.05] | +4.11 [+0.33, +7.89] |
| 126 | MaterioMiner coarse, 4 papers | multi-task + MR6 + names − CRF head | s11340 +2.18, ncomms7164 +4.69, s41467 +0.59, met10111458 +3.13 | s11340 +3.76, ncomms7164 +4.92, s41467 −0.24, met10111458 +3.22 | +4.34 [−1.55, +10.24] | +3.69 [−0.16, +7.54] |
| 127 | MaterioMiner coarse, 4 papers | multi-task + MR6 − multi-task + MR3 | s11340 +0.41, ncomms7164 +0.02, s41467 +0.44, met10111458 −0.64 | s11340 +1.83, ncomms7164 −0.02, s41467 +0.89, met10111458 −0.16 | +1.65 [−3.57, +6.87] | +0.76 [−0.46, +1.99] |
| 128 | MaterioMiner coarse, 4 papers | multi-task + MR6 − multi-task | s11340 +2.53, ncomms7164 +2.87, s41467 +1.24, met10111458 +2.78 | s11340 +3.84, ncomms7164 +3.23, s41467 +1.45, met10111458 +1.90 | +5.42 [+1.16, +9.69] | +3.48 [+1.31, +5.65] |
| 129 | MaterioMiner coarse, 4 papers | multi-task + MR6 − CRF head | s11340 +1.40, ncomms7164 +3.30, s41467 +1.10, met10111458 +1.40 | s11340 +2.90, ncomms7164 +3.44, s41467 +1.23, met10111458 +1.58 | +4.22 [−1.07, +9.51] | +3.06 [+0.81, +5.31] |
| 130 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: recall on the unseen test names the lexicon holds | electrolysis +13.29, graphene +0.22, pemfc +15.59, poly +1.93, semi +12.40, sofc +25.61, steel +9.67 |  |  |  |
| 131 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: micro F1 | electrolysis +7.13, graphene −1.31, pemfc +6.43, poly −1.48, semi +2.20, sofc +9.39, steel +0.47 |  |  |  |
| 132 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: precision on all test entities | electrolysis +10.28, graphene −3.05, pemfc +11.03, poly −2.55, semi +6.90, sofc +16.93, steel +0.93 |  |  |  |
| 133 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: recall on the unseen test names | electrolysis +2.36, graphene +0.46, pemfc +1.34, poly −0.14, semi +0.14, sofc +2.80, steel +0.86 |  |  |  |
| 134 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: precision on the unseen test names | electrolysis +11.30, graphene −3.44, pemfc +11.58, poly −2.30, semi +6.92, sofc +18.08, steel +1.05 |  |  |  |
| 135 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: F1 on the unseen test names | electrolysis +7.52, graphene −1.10, pemfc +6.56, poly −1.37, semi +2.31, sofc +9.93, steel +0.96 |  |  |  |
| 136 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: recall on the unseen test names the lexicon does not hold | electrolysis −3.78, graphene +0.52, pemfc −2.52, poly −1.07, semi −0.81, sofc −2.29, steel −2.16 |  |  |  |
| 137 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at matched recall: the recall difference on the names the lexicon holds minus that on the others | electrolysis +17.07, graphene −0.30, pemfc +18.11, poly +3.00, semi +13.22, sofc +27.90, steel +11.83 |  |  |  |
| 138 | MuLMS, 7 subfields (seeds 300–900) | MR6 at matched recall − MR6 at bias 0: micro F1 | electrolysis −7.00, graphene +0.06, pemfc −5.28, poly −0.80, semi +1.07, sofc −6.06, steel −1.24 |  |  |  |
| 139 | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: recall on the unseen test names the lexicon holds | graphene +0.22, poly +1.93, semi +12.40, steel +9.67 |  |  |  |
| 140 | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: micro F1 | graphene −1.31, poly −1.48, semi +2.20, steel +0.47 |  |  |  |
| 141 | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: precision on all test entities | graphene −3.05, poly −2.55, semi +6.90, steel +0.93 |  |  |  |
| 142 | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: recall on the unseen test names | graphene +0.46, poly −0.14, semi +0.14, steel +0.86 |  |  |  |
| 143 | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: precision on the unseen test names | graphene −3.44, poly −2.30, semi +6.92, steel +1.05 |  |  |  |
| 144 | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: F1 on the unseen test names | graphene −1.10, poly −1.37, semi +2.31, steel +0.96 |  |  |  |
| 145 | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: recall on the unseen test names the lexicon does not hold | graphene +0.52, poly −1.07, semi −0.81, steel −2.16 |  |  |  |
| 146 | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at matched recall: the recall difference on the names the lexicon holds minus that on the others | graphene −0.30, poly +3.00, semi +13.22, steel +11.83 |  |  |  |
| 147 | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 at matched recall − MR6 at bias 0: micro F1 | graphene +0.06, poly −0.80, semi +1.07, steel −1.24 |  |  |  |
| 148 | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: recall on the unseen test names the lexicon holds | electrolysis +8.76, graphene +0.22, pemfc +11.18, poly +1.93, semi +12.40, sofc +16.23, steel +9.67 |  |  |  |
| 149 | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: micro F1 | electrolysis +2.30, graphene −1.31, pemfc +3.28, poly −1.48, semi +2.20, sofc +4.15, steel +0.47 |  |  |  |
| 150 | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: precision on all test entities | electrolysis +3.75, graphene −3.05, pemfc +6.50, poly −2.55, semi +6.90, sofc +8.82, steel +0.93 |  |  |  |
| 151 | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: recall on the unseen test names | electrolysis +0.22, graphene +0.46, pemfc −0.16, poly −0.14, semi +0.14, sofc +0.36, steel +0.86 |  |  |  |
| 152 | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: precision on the unseen test names | electrolysis +4.25, graphene −3.44, pemfc +6.73, poly −2.30, semi +6.92, sofc +9.48, steel +1.05 |  |  |  |
| 153 | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: F1 on the unseen test names | electrolysis +2.43, graphene −1.10, pemfc +3.22, poly −1.37, semi +2.31, sofc +4.36, steel +0.96 |  |  |  |
| 154 | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: recall on the unseen test names the lexicon does not hold | electrolysis −4.57, graphene +0.52, pemfc −3.24, poly −1.07, semi −0.81, sofc −3.18, steel −2.16 |  |  |  |
| 155 | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 + names − MR6 at matched recall: the recall difference on the names the lexicon holds minus that on the others | electrolysis +13.33, graphene −0.30, pemfc +14.41, poly +3.00, semi +13.22, sofc +19.41, steel +11.83 |  |  |  |
| 156 | MuLMS, 7 subfields (seeds 300–900; the flagged at the grid value of the nearest recall) | MR6 at matched recall − MR6 at bias 0: micro F1 | electrolysis −2.17, graphene +0.06, pemfc −2.13, poly −0.80, semi +1.07, sofc −0.82, steel −1.24 |  |  |  |
| 157 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at its best bias of the grid: recall on the unseen test names the lexicon holds | electrolysis +6.40, graphene +0.16, pemfc +6.83, poly +1.15, semi +7.78, sofc +13.94, steel +9.38 |  |  |  |
| 158 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at its best bias of the grid: recall on the unseen test names the lexicon does not hold | electrolysis −4.96, graphene −7.45, pemfc −3.24, poly −3.45, semi −1.72, sofc −3.62, steel −4.70 |  |  |  |
| 159 | MuLMS, 7 subfields (seeds 300–900) | MR6 + names − MR6 at bias 0: recall on the unseen test names the lexicon holds | electrolysis +6.40, graphene +0.48, pemfc +6.83, poly +4.64, semi +7.78, sofc +14.09, steel +10.78 |  |  |  |
| 160 | MuLMS, 7 subfields (seeds 300–900) | MR6 at bias 0 − MR6 at matched recall: recall on the unseen test names the lexicon holds | electrolysis +6.89, graphene −0.25, pemfc +8.76, poly −2.71, semi +4.62, sofc +11.51, steel −1.11 |  |  |  |
| 161 | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 + names − MR6 at bias 0: recall on the unseen test names the lexicon holds | graphene +0.48, poly +4.64, semi +7.78, steel +10.78 |  |  |  |
| 162 | MuLMS, 4 subfields (seeds 300–900; without the flagged electrolysis, pemfc, sofc) | MR6 at bias 0 − MR6 at matched recall: recall on the unseen test names the lexicon holds | graphene −0.25, poly −2.71, semi +4.62, steel −1.11 |  |  |  |

## Supplementary Note 13: further tables

Three analyses the Notes above summarise, in full. Table S13 gives the
unseen-name trade of Section 5.4 for every system against the one it
builds on (post hoc; `ner_mentions.py`, `chain`), Table S14 the lexicon
split of Section 5.4 per held-out unit (post hoc; `ner_lexicon_recall.py`),
and Table S15 the CPU probe of Supplementary Note 7 per subfield (no
decision; `silver_lexicon.py`).

**Table S13.** The unseen-name trade in full: per setting, each system
against the one it builds on, the mean paired difference over the units
in F1, recall and precision on the unseen and the seen test names and
over all entities, with 95% CIs, on all ten seeds and, on MuLMS, on the
decision seeds (300–900).

| setting | comparison | seeds | part | Δ F1 [95% CI] | Δ recall [95% CI] | Δ precision [95% CI] |
|---|---|---|---|---|---|---|
| MuLMS, 7 subfields | MR3 − CRF head | all ten | unseen names | +2.19 [−0.25, +4.63] | +3.76 [+0.73, +6.79] | −1.10 [−2.67, +0.47] |
| MuLMS, 7 subfields | MR3 − CRF head | all ten | seen names | +1.30 [−1.21, +3.81] | +5.06 [+1.90, +8.22] | −3.48 [−5.68, −1.29] |
| MuLMS, 7 subfields | MR3 − CRF head | all ten | all entities | +2.10 [−0.31, +4.51] | +3.89 [+0.91, +6.87] | −1.40 [−2.92, +0.13] |
| MuLMS, 7 subfields | MR3 − CRF head | decision seeds (300–900) | unseen names | +2.31 [−0.27, +4.89] | +4.05 [+0.94, +7.16] | −1.16 [−3.83, +1.51] |
| MuLMS, 7 subfields | MR3 − CRF head | decision seeds (300–900) | seen names | +1.83 [−0.76, +4.41] | +5.57 [+2.28, +8.85] | −2.86 [−5.42, −0.30] |
| MuLMS, 7 subfields | MR3 − CRF head | decision seeds (300–900) | all entities | +2.27 [−0.25, +4.79] | +4.21 [+1.13, +7.30] | −1.38 [−3.80, +1.04] |
| MuLMS, 7 subfields | MR6 − MR3 | all ten | unseen names | +0.22 [−0.45, +0.88] | +0.63 [−0.20, +1.46] | −0.26 [−1.31, +0.79] |
| MuLMS, 7 subfields | MR6 − MR3 | all ten | seen names | +0.18 [−0.98, +1.33] | +0.79 [−0.79, +2.37] | +0.08 [−1.58, +1.74] |
| MuLMS, 7 subfields | MR6 − MR3 | all ten | all entities | +0.23 [−0.39, +0.85] | +0.68 [−0.09, +1.44] | −0.22 [−1.21, +0.77] |
| MuLMS, 7 subfields | MR6 − MR3 | decision seeds (300–900) | unseen names | +0.43 [−0.32, +1.17] | +0.73 [+0.04, +1.42] | −0.05 [−2.04, +1.93] |
| MuLMS, 7 subfields | MR6 − MR3 | decision seeds (300–900) | seen names | +0.27 [−0.51, +1.04] | +0.52 [−0.61, +1.64] | +0.27 [−0.92, +1.46] |
| MuLMS, 7 subfields | MR6 − MR3 | decision seeds (300–900) | all entities | +0.41 [−0.35, +1.17] | +0.69 [+0.07, +1.30] | +0.01 [−1.85, +1.87] |
| MuLMS, 7 subfields | MR6 + names − MR6 | all ten | unseen names | +0.93 [−1.36, +3.22] | +3.24 [+1.36, +5.13] | −2.23 [−4.84, +0.38] |
| MuLMS, 7 subfields | MR6 + names − MR6 | all ten | seen names | +0.77 [−1.18, +2.72] | +2.43 [−1.04, +5.89] | −0.99 [−2.80, +0.81] |
| MuLMS, 7 subfields | MR6 + names − MR6 | all ten | all entities | +0.85 [−1.46, +3.17] | +3.12 [+1.23, +5.00] | −2.17 [−4.69, +0.35] |
| MuLMS, 7 subfields | MR6 + names − MR6 | decision seeds (300–900) | unseen names | +0.56 [−1.44, +2.56] | +2.56 [+0.76, +4.36] | −2.22 [−5.03, +0.60] |
| MuLMS, 7 subfields | MR6 + names − MR6 | decision seeds (300–900) | seen names | +0.47 [−1.39, +2.33] | +1.89 [−0.70, +4.48] | −0.96 [−3.61, +1.68] |
| MuLMS, 7 subfields | MR6 + names − MR6 | decision seeds (300–900) | all entities | +0.51 [−1.51, +2.53] | +2.48 [+0.67, +4.29] | −2.15 [−4.95, +0.66] |
| MuLMS, 7 subfields | MR6 + names (mixed corpus) − MR6 | decision seeds (300–900) | unseen names | +0.64 [−1.33, +2.62] | +2.47 [−0.09, +5.02] | −1.85 [−3.56, −0.15] |
| MuLMS, 7 subfields | MR6 + names (mixed corpus) − MR6 | decision seeds (300–900) | seen names | +0.99 [−0.70, +2.67] | +2.43 [−0.71, +5.58] | −0.02 [−2.48, +2.45] |
| MuLMS, 7 subfields | MR6 + names (mixed corpus) − MR6 | decision seeds (300–900) | all entities | +0.70 [−1.19, +2.58] | +2.51 [+0.04, +4.98] | −1.68 [−3.35, −0.02] |
| MaterioMiner fine, 4 papers | MR3 − CRF head | all ten | unseen names | +3.19 [+1.66, +4.72] | +4.51 [+2.27, +6.74] | +0.35 [−2.01, +2.71] |
| MaterioMiner fine, 4 papers | MR3 − CRF head | all ten | seen names | +1.49 [−1.31, +4.30] | +3.38 [−0.02, +6.77] | −0.49 [−6.98, +6.00] |
| MaterioMiner fine, 4 papers | MR3 − CRF head | all ten | all entities | +2.77 [+1.51, +4.03] | +4.11 [+2.97, +5.25] | +0.32 [−3.23, +3.87] |
| MaterioMiner fine, 4 papers | MR6 − MR3 | all ten | unseen names | +0.46 [−2.16, +3.09] | +1.25 [−0.88, +3.38] | −0.49 [−4.41, +3.43] |
| MaterioMiner fine, 4 papers | MR6 − MR3 | all ten | seen names | +0.56 [−0.37, +1.49] | +1.21 [−1.26, +3.68] | +0.16 [−0.96, +1.29] |
| MaterioMiner fine, 4 papers | MR6 − MR3 | all ten | all entities | +0.20 [−1.47, +1.88] | +0.93 [+0.09, +1.77] | −0.57 [−3.79, +2.64] |
| MaterioMiner fine, 4 papers | MR6 + names − MR6 | all ten | unseen names | +1.26 [−3.40, +5.91] | +0.86 [−3.38, +5.10] | +1.32 [−6.62, +9.25] |
| MaterioMiner fine, 4 papers | MR6 + names − MR6 | all ten | seen names | +1.03 [−0.77, +2.82] | +0.77 [−1.43, +2.97] | +1.02 [−1.12, +3.16] |
| MaterioMiner fine, 4 papers | MR6 + names − MR6 | all ten | all entities | +1.29 [−2.51, +5.10] | +0.88 [−2.82, +4.58] | +1.56 [−5.05, +8.17] |
| MaterioMiner fine, 4 papers | multi-task + MR3 − multi-task | all ten | unseen names | +3.09 [+0.99, +5.18] | +3.46 [+1.66, +5.27] | +2.58 [−1.00, +6.17] |
| MaterioMiner fine, 4 papers | multi-task + MR3 − multi-task | all ten | seen names | +3.00 [−0.85, +6.85] | +5.34 [−4.38, +15.06] | +0.62 [−7.19, +8.44] |
| MaterioMiner fine, 4 papers | multi-task + MR3 − multi-task | all ten | all entities | +3.01 [+0.42, +5.60] | +3.40 [+0.67, +6.13] | +2.17 [−0.43, +4.78] |
| MaterioMiner fine, 4 papers | multi-task + MR6 − multi-task + MR3 | all ten | unseen names | −0.00 [−1.75, +1.74] | −0.42 [−3.49, +2.64] | +0.14 [−2.17, +2.46] |
| MaterioMiner fine, 4 papers | multi-task + MR6 − multi-task + MR3 | all ten | seen names | −1.13 [−4.03, +1.76] | −0.11 [−4.08, +3.86] | −1.78 [−5.15, +1.60] |
| MaterioMiner fine, 4 papers | multi-task + MR6 − multi-task + MR3 | all ten | all entities | −0.19 [−2.05, +1.67] | −0.33 [−3.43, +2.78] | −0.29 [−1.78, +1.19] |
| MaterioMiner fine, 4 papers | multi-task + MR6 + names − multi-task + MR6 | all ten | unseen names | +0.43 [−1.57, +2.44] | +0.58 [−2.93, +4.09] | +0.82 [−3.39, +5.03] |
| MaterioMiner fine, 4 papers | multi-task + MR6 + names − multi-task + MR6 | all ten | seen names | +2.43 [+0.40, +4.47] | +0.88 [−3.39, +5.16] | +3.39 [+2.65, +4.12] |
| MaterioMiner fine, 4 papers | multi-task + MR6 + names − multi-task + MR6 | all ten | all entities | +0.96 [−0.01, +1.92] | +0.85 [−2.03, +3.73] | +1.46 [−1.21, +4.13] |
| MaterioMiner coarse, 4 papers | MR3 − CRF head | all ten | unseen names | +1.58 [−0.02, +3.17] | +1.86 [+0.44, +3.27] | +1.30 [−0.53, +3.12] |
| MaterioMiner coarse, 4 papers | MR3 − CRF head | all ten | seen names | +0.12 [−0.83, +1.06] | +0.73 [−2.72, +4.18] | −0.31 [−2.60, +1.98] |
| MaterioMiner coarse, 4 papers | MR3 − CRF head | all ten | all entities | +1.28 [+0.06, +2.51] | +1.55 [+0.48, +2.62] | +1.02 [−0.61, +2.65] |
| MaterioMiner coarse, 4 papers | MR6 − MR3 | all ten | unseen names | −0.64 [−3.44, +2.15] | −0.27 [−1.99, +1.45] | −1.00 [−4.87, +2.87] |
| MaterioMiner coarse, 4 papers | MR6 − MR3 | all ten | seen names | +0.51 [−1.18, +2.21] | +2.98 [+0.65, +5.30] | −1.62 [−2.34, −0.91] |
| MaterioMiner coarse, 4 papers | MR6 − MR3 | all ten | all entities | −0.21 [−2.71, +2.29] | +0.45 [−1.38, +2.27] | −0.82 [−3.97, +2.33] |
| MaterioMiner coarse, 4 papers | MR6 + names − MR6 | all ten | unseen names | +1.37 [−1.75, +4.50] | +1.60 [−1.37, +4.57] | +1.16 [−2.24, +4.57] |
| MaterioMiner coarse, 4 papers | MR6 + names − MR6 | all ten | seen names | −0.48 [−3.19, +2.23] | −3.39 [−7.96, +1.18] | +1.90 [−0.56, +4.36] |
| MaterioMiner coarse, 4 papers | MR6 + names − MR6 | all ten | all entities | +0.70 [−1.60, +3.00] | +0.47 [−1.09, +2.03] | +0.90 [−2.08, +3.88] |
| MaterioMiner coarse, 4 papers | multi-task + MR3 − multi-task | all ten | unseen names | +2.02 [−0.19, +4.22] | +2.36 [+0.31, +4.42] | +1.68 [−0.68, +4.05] |
| MaterioMiner coarse, 4 papers | multi-task + MR3 − multi-task | all ten | seen names | +2.11 [−0.02, +4.24] | +4.50 [+0.12, +8.89] | −0.01 [−4.00, +3.97] |
| MaterioMiner coarse, 4 papers | multi-task + MR3 − multi-task | all ten | all entities | +2.30 [+0.49, +4.10] | +2.73 [+0.90, +4.56] | +1.87 [+0.00, +3.74] |
| MaterioMiner coarse, 4 papers | multi-task + MR6 − multi-task + MR3 | all ten | unseen names | +0.28 [−0.80, +1.37] | +0.65 [−0.73, +2.03] | −0.07 [−1.25, +1.11] |
| MaterioMiner coarse, 4 papers | multi-task + MR6 − multi-task + MR3 | all ten | seen names | −0.56 [−1.03, −0.08] | −0.27 [−1.92, +1.37] | −0.83 [−1.80, +0.14] |
| MaterioMiner coarse, 4 papers | multi-task + MR6 − multi-task + MR3 | all ten | all entities | +0.06 [−0.74, +0.86] | +0.41 [−0.69, +1.52] | −0.28 [−1.26, +0.70] |
| MaterioMiner coarse, 4 papers | multi-task + MR6 + names − multi-task + MR6 | all ten | unseen names | +1.20 [−0.82, +3.22] | +1.45 [−0.69, +3.58] | +0.97 [−1.09, +3.03] |
| MaterioMiner coarse, 4 papers | multi-task + MR6 + names − multi-task + MR6 | all ten | seen names | +0.02 [−1.30, +1.34] | −0.96 [−2.09, +0.16] | +0.83 [−1.27, +2.93] |
| MaterioMiner coarse, 4 papers | multi-task + MR6 + names − multi-task + MR6 | all ten | all entities | +0.85 [−0.72, +2.42] | +0.88 [−0.81, +2.58] | +0.82 [−0.64, +2.27] |
| SOFC-Exp, 5 folds | MR3 − CRF head | all ten | unseen names | −0.43 [−1.66, +0.79] | +0.08 [−0.63, +0.79] | −0.93 [−2.96, +1.10] |
| SOFC-Exp, 5 folds | MR3 − CRF head | all ten | seen names | −0.50 [−1.27, +0.28] | −0.45 [−1.98, +1.09] | −0.51 [−1.68, +0.66] |
| SOFC-Exp, 5 folds | MR3 − CRF head | all ten | all entities | −0.48 [−1.36, +0.39] | −0.22 [−0.99, +0.56] | −0.72 [−2.30, +0.87] |

**Table S14.** The lexicon split per unit: per setting and comparison,
each held-out unit's recall gain on the unseen test names its lexicon
holds and on the others, and their difference (the paired means are
Table S3's).

| setting | comparison | unit | gain, names in the lexicon | gain, names not in it | in − out |
|---|---|---|---|---|---|
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | electrolysis | +6.40 | −0.61 | +7.01 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | graphene | +0.48 | −1.57 | +2.05 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | pemfc | +6.83 | +0.93 | +5.90 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | poly | +4.64 | +2.03 | +2.61 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | semi | +7.78 | +4.85 | +2.93 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | sofc | +14.09 | +1.51 | +12.58 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − MR6 | steel | +10.78 | +0.68 | +10.10 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | electrolysis | +10.80 | +3.04 | +7.76 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | graphene | +1.18 | +2.20 | −1.02 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | pemfc | +8.23 | +7.52 | +0.71 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | poly | +8.45 | +5.19 | +3.26 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | semi | +14.36 | +7.41 | +6.95 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | sofc | +20.67 | +14.30 | +6.37 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names − CRF head | steel | +11.01 | +4.71 | +6.30 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | electrolysis | +4.40 | +3.65 | +0.75 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | graphene | +0.70 | +3.77 | −3.07 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | pemfc | +1.40 | +6.59 | −5.19 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | poly | +3.81 | +3.16 | +0.65 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | semi | +6.58 | +2.56 | +4.02 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | sofc | +6.58 | +12.79 | −6.21 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 − CRF head | steel | +0.23 | +4.03 | −3.80 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | electrolysis | +3.02 | +3.81 | −0.79 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | graphene | −0.75 | +2.30 | −3.05 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | pemfc | +4.01 | +6.42 | −2.41 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | poly | +2.91 | +1.38 | +1.53 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | semi | +3.89 | +1.40 | +2.49 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | sofc | +8.31 | +11.43 | −3.12 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR3 − CRF head | steel | +3.52 | +2.82 | +0.70 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | electrolysis | +6.17 | −2.01 | +8.18 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | graphene | −0.81 | −1.25 | +0.44 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | pemfc | +7.21 | +1.39 | +5.82 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | poly | +4.53 | +2.62 | +1.91 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | semi | +3.43 | +0.86 | +2.57 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | sofc | +16.40 | +5.76 | +10.64 |
| MuLMS, 7 held-out subfields (seeds 300–900) | MR6 + names (mixed corpus) − MR6 | steel | +6.20 | +1.77 | +4.43 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.1007_s11340-021-00758-x | +1.25 | −3.14 | +4.39 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.1038_ncomms7164 | +13.48 | +0.61 | +12.87 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.1038_s41467-020-19071-7 | −0.45 | −0.08 | −0.37 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.3390_met10111458 | +4.19 | +3.78 | +0.41 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.1007_s11340-021-00758-x | +0.00 | +6.86 | −6.86 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.1038_ncomms7164 | +35.22 | +5.64 | +29.58 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.1038_s41467-020-19071-7 | +2.96 | +3.42 | −0.46 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.3390_met10111458 | +12.90 | +8.08 | +4.82 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.1007_s11340-021-00758-x | −1.25 | +10.00 | −11.25 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.1038_ncomms7164 | +21.74 | +5.03 | +16.71 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.1038_s41467-020-19071-7 | +3.41 | +3.50 | −0.09 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.3390_met10111458 | +8.71 | +4.30 | +4.41 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.1007_s11340-021-00758-x | +0.00 | +8.00 | −8.00 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.1038_ncomms7164 | +21.30 | +1.61 | +19.69 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.1038_s41467-020-19071-7 | +4.09 | +3.24 | +0.85 |
| MaterioMiner fine-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.3390_met10111458 | +9.68 | +3.81 | +5.87 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.1007_s11340-021-00758-x | +1.54 | +4.85 | −3.31 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.1038_ncomms7164 | −0.41 | −0.16 | −0.25 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.1038_s41467-020-19071-7 | +0.67 | +1.75 | −1.08 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − MR6 | 10.3390_met10111458 | −0.34 | +1.20 | −1.54 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.1007_s11340-021-00758-x | +1.15 | +5.84 | −4.69 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.1038_ncomms7164 | +3.06 | +3.48 | −0.42 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.1038_s41467-020-19071-7 | +0.67 | +2.55 | −1.88 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 + names − CRF head | 10.3390_met10111458 | +0.75 | +2.63 | −1.88 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.1007_s11340-021-00758-x | −0.39 | +0.99 | −1.38 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.1038_ncomms7164 | +3.47 | +3.64 | −0.17 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.1038_s41467-020-19071-7 | +0.00 | +0.80 | −0.80 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR6 − CRF head | 10.3390_met10111458 | +1.09 | +1.43 | −0.34 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.1007_s11340-021-00758-x | +0.38 | +2.97 | −2.59 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.1038_ncomms7164 | +0.82 | +3.12 | −2.30 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.1038_s41467-020-19071-7 | −1.26 | +1.19 | −2.45 |
| MaterioMiner coarse-grained, 4 held-out papers (10 seeds) | MR3 − CRF head | 10.3390_met10111458 | −0.41 | +1.99 | −2.40 |

**Table S15.** The CPU probe per held-out MuLMS subfield (no decision):
the share of the unseen test names the topic corpus holds and that the
sentence and mention filters admit at threshold 0.9; the CRF head's
precision and recall (10 seeds); and the change of its micro F1 when
every verbatim match of a lexicon name is tagged where it predicted O,
for each 0.9 lexicon and for the gold unseen names themselves, which
also override its tags in the last column.

| subfield | corpus holds (%) | sentence filter admits (%) | mention filter admits (%) | CRF precision | CRF recall | Δ F1, sentence lexicon | Δ F1, mention lexicon | Δ F1, gold names (fill) | Δ F1, gold names (override) |
|---|---|---|---|---|---|---|---|---|---|
| electrolysis | 70.7 | 12.5 | 29.9 | 57.3 | 54.7 | +0.85 | −0.77 | −6.85 | +4.47 |
| graphene | 64.0 | 15.3 | 28.8 | 56.8 | 44.9 | −2.86 | −5.07 | +9.11 | +24.32 |
| pemfc | 73.9 | 19.2 | 29.4 | 60.5 | 42.9 | +3.16 | +3.97 | −10.51 | −5.91 |
| poly | 65.9 | 8.0 | 18.3 | 55.9 | 48.7 | −0.25 | −3.31 | −3.63 | +2.22 |
| semi | 69.3 | 4.6 | 9.4 | 54.0 | 23.5 | +1.20 | +2.27 | +17.29 | +22.00 |
| sofc | 80.8 | 25.8 | 34.4 | 57.5 | 35.3 | +5.79 | +5.91 | +15.16 | +27.06 |
| steel | 55.0 | 13.3 | 25.0 | 64.4 | 50.9 | +0.01 | −10.03 | −1.82 | +4.13 |
| mean | 68.5 | 14.1 | 25.0 | 58.1 | 43.0 | +1.13 | −1.00 | +2.68 | +11.18 |
