# Paper plan: named-entity recognition on MaterioMiner

This outline lists what a paper on the NER results could claim, the result
behind each claim, and what is still missing. It is updated as the weekly
protocols (ROADMAP §4) finish. All numbers come from the results files in
`results/` and the tables that `report_results.py` generates from them.

**Type.** The aim is a methods paper: a change that improves materials NER
on unseen papers over the strongest system so far (the seed vote, or the
multi-task vote fine-grained), decided under the 10-seed design and repeated
on a second dataset. Week 6 tests three candidates (DeBERTa-v3-large,
document context, multi-task self-training; ROADMAP P1, "Next, after week
5"), and later weeks combine the ones that pass. If none passes once these
options are exhausted, the paper is an evaluation and reproducibility paper.
Claims C1–C10 hold either way: in a methods paper they are its baselines and
evaluation protocol, in an evaluation paper its results. Document context
passed the week-6 screen (+2.66 [+0.91, +4.40] fine-grained) but neither of
its 10-seed decisions (+1.14 [−0.94, +3.22] against the CRF head).
Multi-task self-training has the largest mean gain of any method,
+5.47 [−0.74, +11.68] fine-grained over 10 seeds, but passes none of its
4 decisions: three papers gain 4–9 points, the fourth +0.51. Mention
replacement improves on the CRF head on unseen papers at both
granularities with 10 seeds (C20), but its vote does not beat the
strongest systems and it gives nothing on SOFC-Exp. Combined with
multi-task training (week 9, C21) the two gains add up on unseen papers:
+6.73 [+6.03, +7.43] fine-grained over the CRF head, and its vote is the
strongest fine-grained system there (+2.86 [+0.18, +5.54] over the
multi-task vote). It gives nothing on SOFC-Exp's official split at either
level, so by the pre-registered criterion it is a result for few annotated
papers, not the methods claim. SOFC-Exp's larger training set is not the
reason: cut to MaterioMiner's size, mention replacement gains +0.50 [−0.47,
+1.46] there (week 9, part 2). Week 10 (pre-registered) tested it on a third
corpus whose held-out documents differ in topic, as MaterioMiner's papers do:
MuLMS's seven materials subfields, each held out in turn, with MaterioMiner's
training size (C22). There, as on a held-out MaterioMiner paper, most test
names are unseen (88%) and the CRF head finds few of them (39%). Mention
replacement gains +2.10 [−0.31, +4.51] over the CRF head, six of the seven
subfields gaining, which does not pass; its votes improve on the CRF votes
(+2.71 [+0.22, +5.20]). The methods claim needed the first decision, so
mention replacement and its combination stay results for MaterioMiner.
Unless one of the remaining candidates passes (ROADMAP §4), the paper is
the evaluation paper. Week 11 (pre-registered) tests the first of them,
mention replacement whose pool also holds the names that three CRF teachers
tag in unlabeled in-domain text (C23). On MuLMS's held-out subfields it
fails both decisions: +2.04 [−0.85, +4.92] over the CRF head and −0.06
[−1.09, +0.96] against mention replacement. The teachers' names hold 4.4%
of a held-out subfield's unseen test names. On MaterioMiner (decisions 3
and 4, 19–20 October) it adds +0.23 [−0.69, +1.14] fine-grained and +0.61
[−0.96, +2.18] coarse-grained to mention replacement: no gain there either,
with the teachers' names holding 2–9% of a held-out paper's unseen names. The methods claim needed the
first of these decisions, so it is ruled out for this method too; its
MaterioMiner decisions run after 24 October. Recall calibration (week 7, part 2a) passed none
of its 5 decisions, and showed why a decoding change cannot close the recall
gap on unseen papers (C15). Most entities of an unseen paper are names the
training papers never annotate (81% fine-grained), and every gain that holds
there is a gain on those names (C17), so that is what a method for unseen
papers has to improve. The week-6 protocol
excluded multi-task methods from SOFC-Exp as a one-level corpus, but SOFC-Exp
has a second level, 17 evaluated slot types that refine its 4 entity types
(ROADMAP, "Prepared for part 2d"). Multi-task training (C7) does not repeat
there: −0.78 [−2.39, +0.84] on the slot types and −0.15 [−1.36, +1.05] on
the entity types (week 7, part 2d). Multi-task self-training is repeated
there if it passes a decision on MaterioMiner.

**Working titles.** Methods: an approach for materials NER on unseen papers,
named after the change that passes. Evaluation: "How much of a materials-NER
gain survives? Splits, scoring and seeds on MaterioMiner".

## Claims and evidence

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | The published numbers are reproduced on the split the release ships (323 / 57 / 96 sentences, a different partition per granularity, not described in the paper), scored like its code (micro F1, seqeval's default mode): plain MatSciBERT scores 70.19 [69.03, 71.35] fine-grained and 72.52 [71.88, 73.16] coarse-grained over 10 seeds, against 69.92 and 72.32. Single runs of plain MatSciBERT range over 4.6 points fine-grained (68.0–72.7) and 3.6 coarse-grained (70.4–74.1). | `results/gpu/ner_results_*_published_split.json`; README, the published-split rows | Done (GPU, 10 seeds); the authors are asked to confirm split and scoring (ROADMAP item 6) |
| C2 | Random splits hide a large gap (the paper says the out-of-distribution generalisation "can not be estimated" from four publications; leaving one paper out estimates it). The seed vote `crf_ens3` scores 79.68 fine-grained and 78.63 coarse-grained over 5 random splits, but 48.81 and 61.10 on a paper the model has never seen: 18–31 points lower. The size of the gap depends on how much the papers differ: on SOFC-Exp, whose 45 articles share one topic and one annotation scheme, the CRF head scores 86.27 [84.82, 87.71] over 5 random sentence splits and 82.79 [78.33, 87.24] over 5 folds of 9 held-out articles, a gap of 3.48 (votes 87.18 and 83.62); 30% of its test names are unseen on random splits, 46% on held-out articles (81% on a held-out MaterioMiner paper, fine-grained). On MuLMS, whose held-out documents come from another materials subfield, the CRF head trained on 340 sentences scores 65.72 [65.25, 66.19] over 5 random splits and 48.66 [40.91, 56.40] over its 7 held-out subfields, a gap of 17.06 (votes 67.87 and 50.54); 57% of its test names are unseen on random splits, 88% on held-out subfields (C22). | `results/gpu/ner_week5_comparisons.json` (`crf_ens3` vs `crf`); SOFC-Exp: `gpu/ner_week5_seed_analysis.json` (`sofc_splits`), `gpu/ner_mentions.json`; MuLMS: `gpu/ner_week5_seed_analysis.json` (`mulms`) | Done (GPU, 10 seeds); SOFC-Exp: week 8, part 1 (descriptive); MuLMS: week 10 (descriptive) |
| C3 | A CRF head with IOB2 constraints beats the published numbers. On the published split it scores 73.31 [72.28, 74.35] and 75.48 [74.23, 76.72], +3.12 and +2.96 over plain MatSciBERT, paired by seed. Over 5 random splits, the reported system beats them fine-grained under micro, macro and weighted F1, and coarse-grained under micro and weighted F1 but not macro (70.83 [64.49, 77.17]). | README, the published-split and reported-system rows | Done (GPU, 10 seeds) |
| C4 | Much of the CRF head's gain is span validity. Under strict IOB2 scoring its gain on the published split shrinks from +3.12 to +1.70 [+0.21, +3.19] fine-grained and from +2.96 to +1.07 [−0.82, +2.97] coarse-grained. Over the random splits (CPU, 5 seeds) the fine-grained gain vanishes under strict scoring (+0.10 [−2.08, +2.29]), and dropping invalid tags from the baseline's output gives the same; coarse-grained, +1.91 [+1.17, +2.65] survives. On SOFC-Exp's official split strict scoring shrinks the gain from +1.22 to +0.73 [−0.04, +1.51] on the entity types and from +1.68 to −0.06 [−1.70, +1.58] on the slot types, where plain MatSciBERT scores higher strict (71.94) than under the default scoring (70.20). On the entity types' test articles cut as the training articles are (C16), the gain is +0.17 [−0.37, +0.70] (strict −0.13 [−0.71, +0.44]). | README, "Why the CRF helps"; `ner_week5_seed_analysis.json` (`published_split`) | Done |
| C5 | A vote over 3 seeds is the most reliable addition: +2.20 and +1.67 over the mean of single CRF runs on the random splits, +2.36 and +2.50 on unseen papers, every CI above 0. | `ner_week5_comparisons.json` (`crf_ens3` vs `crf`) | Done (GPU, 10 seeds) |
| C6 | Popular additions do not pass a pre-registered paired test with 10 seeds: label semantics (model or vote, either protocol), self-training with 3 teachers on unseen papers (+1.74 [−1.16, +4.64] fine-grained, the week-4 claim), coarse-grained multi-task training (+0.71 [−0.15, +1.56], the week-3 claim), and document context (FLERT) on unseen papers (+1.14 [−0.94, +3.22] fine-grained, +0.12 [−2.03, +2.27] coarse-grained; its vote −1.44 [−3.80, +0.91] against the multi-task vote) and on SOFC-Exp's unseen articles (+0.58 [−0.26, +1.42], paired by seed). Multi-task self-training, the largest mean gain on unseen papers, also fails: +5.47 [−0.74, +11.68] fine-grained (three papers +4 to +9, one +0.51) and +0.90 [−1.18, +2.98] coarse-grained against the CRF head; its votes +1.54 [−1.73, +4.82] against the multi-task vote and +1.08 [−1.40, +3.57] against the seed vote. Earlier: class weights, domain-adaptive and in-domain pre-training, other backbones and a mixed-backbone vote. | ROADMAP P1 weeks 1–6; README week-3 to week-6 tables | Done (document context and multi-task self-training: week 6) |
| C7 | Multi-task training over both granularities is the one addition that holds: fine-grained +1.83 [+0.32, +3.34] over the CRF head on the random splits. Its vote improves on the seed vote coarse-grained on the random splits (+0.79 [+0.17, +1.42]) and fine-grained on unseen papers (+3.65 [+0.49, +6.82]): 2 of 12 vote comparisons, where about 0.3 would pass by chance. Self-training improves on the random splits (+1.05 and +1.04, reported, no decision). Multi-task training does not repeat on SOFC-Exp's official split, with a slot head and an entity head: −0.78 [−2.39, +0.84] on the 17 slot types and −0.15 [−1.36, +1.05] on the 4 entity types against the CRF head, paired by seed; its votes −0.86 [−3.67, +1.95] and +0.08 [−2.85, +3.01] against the CRF votes. Across corpora the same holds outside our runs: one MatSciBERT with a head per corpus, trained on MuLMS with SOFC-Exp or a synthesis-procedure corpus, changes micro F1 by −0.1 to −1.5 (Schrader et al., 2023, Table 11). | `ner_week5_comparisons.json`; SOFC-Exp: `gpu/ner_week5_seed_analysis.json` (`published_split`, `methods`, `crf_mt`) | Done (GPU, 10 seeds); SOFC-Exp: week 7, part 2d, no repeat |
| C8 | How far a 3-seed design misleads, from the 120 seed triples of the 10-seed runs: the 1.8-point multi-task gain passes in 41% of the triples; label semantics and self-training on unseen papers in 2–18%; up to 37% of the triples give the opposite sign. A split's micro F1 varies over seeds with an SD of 1.5–1.7 points on random splits and 2.2–2.5 on unseen papers. The 1-point CPU gains of week 3 that passed with 3 seeds do not hold with 10. Week 6 shows it prospectively: document context passed the 3-seed screen with +2.66 [+0.91, +4.40] fine-grained, but gains +0.49 [−3.10, +4.07] on the 7 seeds the screen did not see (one paper from +4.29 to −2.59). Multi-task self-training's screen held up (+6.06 on the screen's 3 seeds, about +5.2 on the other 7), and it still fails with 4 papers, since one gains +0.51: a screen cannot tell which kind it has. | `ner_week5_seed_analysis.json` (`triples`, `spread`); `gpu/ner_week6_screen.json`, `gpu/ner_week6_comparisons.json`; README | Done (exploratory) |
| C9 | A few rare coarse-grained types keep macro F1 near the published number: in the reported system's votes on the random splits, SDE and MIP score about 18 F1 and PRP, MOD and CPR 40–42, with 2–8 test entities per split. | `ner_week5_seed_analysis.json` (`types`) | Done (exploratory); week 6 targets them |
| C10 | Where the CRF head fails on unseen papers: fine-grained it misses 36% of the gold entities outright (10% on random splits) and predicts 10,026 entities for 13,280 gold ones; 41% of these misses have a type seen fewer than 10 times in the three training papers. Coarse-grained it misses only 6%, but 16% get the right span and the wrong type, and 21% a wrong boundary. | `ner_week5_seed_analysis.json` (`errors`) | Done (exploratory, 10 seeds) |
| C11 | A larger general-domain encoder does not transfer to unseen papers: DeBERTa-v3-large (435M parameters) with the CRF head scores −4.38 [−7.32, −1.45] fine-grained and −8.01 [−11.72, −4.30] coarse-grained against MatSciBERT (110M) on the held-out papers, while its validation scores on the training papers are similar. | `gpu/ner_week6_screen.json` | Screen (3 seeds). SOFC-Exp S2 did not run: out of memory on a 15 GB T4 at batch size 16 (a deviation; ROADMAP P1, "Week 6 so far") |
| C12 | Document-level label consistency, a post-processing rule (a term the model tags in most of its occurrences in a paper is tagged in the rest), adds little on MaterioMiner and hurts on SOFC-Exp: +0.30 [−0.35, +0.95] fine-grained and +0.12 [+0.01, +0.24] coarse-grained on the CRF head, no significant change on the strongest votes, +0.10 / +0.06 on document context (reported); on SOFC-Exp −0.42 [−0.61, −0.22], and −0.38 [−0.60, −0.15] with retyping (which did better coarse-grained on MaterioMiner, +0.64 [+0.29, +1.00], post hoc). How an annotation scheme treats repeated mentions decides whether "one sense per discourse" helps. | `gpu/ner_week7_comparisons.json`; `gpu/ner_week5_seed_analysis.json` (`published_split`, `sofc`) | Done (week 7, part 1); week 13 applies the same rule to MuLMS's held-out subfields and SOFC-Exp's article folds (C25) |
| C13 | The CRF head's gain repeats on a second dataset: on SOFC-Exp's official split (11 unseen test articles, 10 seeds) it scores 83.75 [82.85, 84.65] against plain MatSciBERT's 82.53 [81.84, 83.23], +1.22 [+0.50, +1.93] paired by seed; its seed votes score 85.13. The published MatSciBERT-CRF (84.77) lies above the CI over seeds. On test articles cut into words as the training articles are (C16; week 7, part 2e) the CRF head scores 84.61 [83.89, 85.34], whose CI covers the published score, and its gain over plain MatSciBERT is +0.17 [−0.37, +0.70]: on SOFC-Exp the gain was mostly a repair of the tokenization shift. On the 17 slot types the gain is +1.68 [−0.12, +3.49] (71.88 against 70.20), not significant; the published MatSciBERT-CRF on SOFC-Slot (72.85, 5 cross-validation splits) lies above the CI, and the seed votes score 74.32. | `gpu/ner_results_sofc_published_split*.json`, `gpu/ner_results_sofc_slot_published_split*.json`; README, SOFC-Exp row | Done (week 6, S1; slot types: week 7, part 2d) |
| C14 | Using the type hierarchy after the fact adds little: completing the fine-grained predictions from the coarse-grained ones, where a coarse type maps to one fine type in the training papers, gives +0.34 [−0.91, +1.59] on the CRF head and +0.12 [−1.76, +2.01] on the multi-task vote (fine-grained, unseen papers). Its added entities are right 39% of the time. | `gpu/ner_week7_comparisons.json` (`_cf`) | Done (week 7, part 2c) |
| C15 | The recall gap on unseen papers is not a decision-threshold problem. A bias on the CRF head's emission of every entity tag, chosen per held-out paper on inner folds (recall calibration), gives −0.59 [−1.75, +0.57] fine-grained and +0.17 [−0.14, +0.48] coarse-grained against the CRF head, its votes do not beat the strongest systems, and on SOFC-Exp (chosen on the validation articles) it gives −0.21 [−0.58, +0.17]. Even the best bias per paper in hindsight would gain +0.56 and +0.18. The entities a small bias adds are right 22% (fine-grained), 19% (coarse-grained) and 21% (SOFC-Exp) of the time, at or below F1's break-even rate and far below the precision of the predictions above the threshold (53.7%, 57.3%, 82.2%). A bias of +3 recovers 7% of the missed fine-grained entities, with 6 wrong entities per right one. | `gpu/ner_week7_comparisons.json` (`_eb`, `_ebv`); `gpu/ner_entity_bias_*.json`; the `bias_test` counts of `gpu/ner_results_*_crf_bias*.json`; ROADMAP, "Outcome of part 2a" | Done (week 7, part 2a; the counts analysis is exploratory) |
| C16 | SOFC-Exp's test articles are cut into words differently from its training articles, against what its README states (Stanford CoreNLP). ChemDataExtractor 1.3.0's word tokenizer gives the tokens of all 7,630 sentences of the 26 training and 8 development articles, and Java's word BreakIterator gives 1,833 of the 1,836 sentences of the 11 test articles. CoreNLP (3.9.2, 4.0.0, 4.5.7) gives at most 48.6% of all sentences. Training has `Fig.`, `cm−2`, `O2−` and `high - temperature` where the test articles have `Fig .`, `cm − 2`, `O2 −` and `high-temperature`. Every result on the official split, the published ones included, is scored under this shift. What it costs (week 7, part 2e, 10 seeds; both pre-registered decisions pass): the same models scored on the test articles cut as the training articles are gain +0.86 [+0.39, +1.33] (the CRF head, 84.61) and +1.91 [+1.33, +2.49] (plain MatSciBERT, 84.45), mostly on unseen names (+1.20 and +3.50). | The release's texts and token offsets (scratch checks, ROADMAP §4, week-6 amendment); `tests/test_chem_tokenize.py` (7,630 of 7,630 on the committed tokens); `gpu/ner_results_sofc_published_split*_cdetest.json` | Done (the shift: exploratory; its cost: week 7, part 2e) |
| C17 | Unseen papers are mostly unseen names, and the gains that hold are gains on them. On a held-out paper, 81% of the fine-grained gold entities (79% coarse-grained; 72–83% per paper) are names that no training sentence annotates with that type, against 35% (40%) on random splits and 40% (39%) on the published split. The CRF head finds 59% of the unseen names on random splits but 35% in a new paper (seen names: 88% and 71%). With the random splits' mix of names its recall per kind on held-out papers would give 58.5 instead of 40.6: the mix of names and the lower recall within each kind each account for 40–60% of the recall gap, 37 points fine-grained and 20 coarse-grained (in either order of the decomposition). Patterned names transfer (unseen Value 76%, Alloy and MaterialClass 72%), descriptive ones do not (StateDescriptor 30%, Mechanism 33%, GrainBoundary 3%). On SOFC-Exp, 49% of the test names are unseen and the CRF head finds 82% of them; on its slot types 59%, of which it finds 62%. On MuLMS's held-out subfields 88% are unseen and it finds 39% of them (seen names 67%), against 57% and 54% on random splits of the same size; there the mix of names accounts for about a third of the 22.5-point recall gap (C22). On held-out papers every fine-grained gain before week 9 is significant only on unseen names: multi-task +4.02 [+1.53, +6.51] (seen +2.88 [−2.70, +8.47]), its vote +3.60 [+2.32, +4.88], the context vote +2.68 [+0.84, +4.53], multi-task self-training's screen +6.44 [+2.25, +10.62] (with 10 seeds +6.35 [−0.13, +12.84]), mention replacement +3.19 [+1.66, +4.72] (C20). The combination of the last two (C21) gains on both kinds: +7.10 [+6.43, +7.77] on unseen and +5.88 [+0.42, +11.34] on seen names. The CRF head's gain over plain MatSciBERT is on unseen names too: +5.94 [+3.11, +8.76] on the published split (fine-grained), +3.02 [+1.99, +4.05] on SOFC-Exp and +2.83 [+0.56, +5.10] on its slot types, against −0.47, −0.52 and −0.23 on seen ones. | `gpu/ner_mentions.json` (`ner_mentions.py`; its F1 over all names equals every run's stored micro F1) | Done (exploratory; the per-kind differences carry no decision) |
| C18 | On a new paper the model finds most entities, but not the exact span and type its annotators chose. Under the looser criteria of SemEval-2013 Task 9.1 (Segura-Bedmar et al., 2013), the CRF head's coarse-grained F1 on held-out papers is 58.60 when span and type must match, 74.17 when only the span must, and 90.74 [89.17, 92.32] when any overlap counts, against 76.96, 83.91 and 93.38 on random splits: the 18.4-point gap shrinks to 9.7 and then 2.6. Fine-grained it shrinks from 31.0 to 25.0 and 18.3 (overlap F1 70.58 against 88.83); there entities are missed outright (C10, C17). On held-out papers 31% of the fine-grained predictions overlap a gold entity with another span or type (12% on random splits): 10.8% lie inside a longer gold entity (the new paper tags "multi-GB cracks", "EBSD data", "smooth samples" where the model tags the noun), and 4.2% have a type that is an MMO ancestor or descendant of the gold one ("samples" is a TestSpecimen in one paper and a FatigueTestSpecimen in another). On SOFC-Exp the CRF head scores 83.75 strict and 88.80 under overlap; on its slot types 71.88 strict and 83.44 on the span alone: 11.7% of its slot-level predictions have a gold entity's span and another slot (0.3% at the entity level), mostly a material given the wrong role in the cell (an electrolyte material taken for a cathode material 27 times per seed, of 120). The gold annotations show the same conventions: only 35 fine-grained names are entities in two or more papers, 7 of them take different types, and 6 of these 7 a class and its MMO subclass (DataSet and MechanicalTestDataSet, TestSpecimen and FatigueTestSpecimen, MechanicalProperty and TensileYieldStrength); DataSet entities are 1.9 words long on average in one paper and 1.0–1.1 in the others. | `gpu/ner_matching.json` (`ner_matching.py`; its strict F1 equals every run's stored micro F1) | Done (exploratory) |
| C19 | A third annotated paper is worth as much as the best method. On each held-out paper, the CRF head trained on the other three papers scores +7.36 [+1.98, +12.74] fine-grained and +8.90 [+5.90, +11.90] coarse-grained over the same head trained on two of them (39.02 → 46.38 and 50.37 → 59.27, the mean over the 4 papers; seeds 0–200). Every paper gains (+3.2 to +11.6). Week 8, part 2 (pre-registered, both decisions pass) separates the paper from its sentences. Three papers cut to the size of two beat two papers by +1.18 [+0.22, +2.15] fine-grained and +3.59 [+1.59, +5.58] coarse-grained, and every paper gains. The third paper's own sentences add +6.17 [+1.10, +11.24] and +5.31 [+4.13, +6.50]. Diversity is thus about a sixth of the gain fine-grained and two fifths coarse-grained. The largest method gain on unseen papers is mention replacement on the multi-task model, +6.73 [+6.03, +7.43] fine-grained over 10 seeds (C21; multi-task self-training's 3-seed screen gave +6.06), 0.6 points below a third paper. On SOFC-Exp, training the CRF head on 10 instead of all 26 training articles costs 3.13 [2.21, 4.04] points on its test articles (week 9, part 2). An outside point agrees: MatSciBERT trained on 47 other fatigue papers, annotated by MaterioMiner's curator with the same classes, scores 66.74 on MaterioMiner and 78.56 in distribution (Potu et al., 2025, Fig. 2; 5 seeds, no CRF, 28 classes, all 4 papers as the test set), against 46–52 for our models trained on three of its papers. | `gpu/ner_week5_seed_analysis.json` (`paper_count`; the two-paper runs are the inner folds of week 7, part 2a; `paper_diversity`, week 8, part 2; SOFC-Exp: `sofc_draws`) | Done (exploratory; diversity: week 8, part 2, pre-registered) |
| C20 | Mention replacement (Dai and Adel, 2020; 3 copies of each training sentence, each mention swapped with probability 0.5 for a training mention of its type) is the first method to improve on the CRF head on unseen papers at both granularities: +2.77 [+1.51, +4.03] fine-grained and +1.28 [+0.06, +2.51] coarse-grained over 10 seeds, every paper gaining (+1.69 to +3.48 and +0.29 to +2.10). The gain lies on unseen names (+3.19 [+1.66, +4.72] fine-grained; seen +1.49 [−1.31, +4.30]). Its vote gains +3.61 [+1.71, +5.52] over the seed vote fine-grained, as much as the multi-task vote, but does not beat the strongest systems (−0.04 [−2.09, +2.01] against the multi-task vote; +2.26 [−0.12, +4.64] coarse-grained against the seed vote). On SOFC-Exp's 568 longer training sentences it gives −0.29 [−1.04, +0.47], and on 10 of its 26 training articles (156–248 sentences, MaterioMiner's size; 5 draws) +0.50 [−0.47, +1.46], so training size does not explain the difference (week 9, part 2). 2 of its 5 decisions pass (about 0.13 expected by chance). Per type (exploratory), coarse-grained it gains on every rare type that pulls macro F1 down (SDE, MIP, PRP, MOD, CPR: +7.7 to +17.9), where multi-task training loses (−2.9 to −10.7); over the 26 coarse types the two methods' changes correlate at −0.72, fine-grained at 0.20. | `gpu/ner_week7_comparisons.json` (`crf_mr`), `gpu/ner_week7_screen.json`; `gpu/ner_week5_seed_analysis.json` (`published_split`, `sofc`, `crf_mr`; per type `week6_types`); `gpu/ner_mentions.json` | Done (week 7, part 2b; decision 1 passes at both granularities, decision 2 and S2 do not); at MaterioMiner's size on SOFC-Exp: `gpu/ner_week5_seed_analysis.json` (`sofc_draws`), week 9, part 2, no pass; on MuLMS's held-out subfields: week 10, the single model does not pass, its votes do (C22) |
| C21 | Mention replacement on the multi-task model adds up on unseen papers. A copy replaces a coarse-grained mention, in both heads' tags at once, by a training mention of the same coarse- and fine-grained type. Against the multi-task model it gains +3.01 [+0.42, +5.60] fine-grained and +2.30 [+0.49, +4.10] coarse-grained over 10 seeds, every paper gaining. Against the CRF head it gains +6.73 [+6.03, +7.43] fine-grained (the multi-task model alone +3.72, mention replacement alone +2.77), close to a third training paper's +7.36 (C19), and +1.74 [−0.15, +3.62] coarse-grained. Its vote scores 55.33 fine-grained, +2.86 [+0.18, +5.54] over the multi-task vote, and is the strongest fine-grained system on unseen papers; coarse-grained it does not beat the seed vote (+1.65 [−0.47, +3.78]). The gain lies on unseen names (+7.10 [+6.43, +7.77]) and, unlike any earlier fine-grained gain, on seen ones too (+5.88 [+0.42, +11.34]). On SOFC-Exp's official split it gives nothing: +0.69 [−1.18, +2.57] on the slot types and −0.19 [−1.27, +0.88] on the entity types, paired by seed. 3 of its 6 decisions pass (about 0.15 expected by chance); a methods claim also needed S2 at the matching level, so this is a result for few annotated papers. | `gpu/ner_week9_comparisons.json`; `gpu/ner_week5_seed_analysis.json` (`published_split`, `sofc`, `crf_mt_mr`; per type `week6_types`); `gpu/ner_mentions.json` | Done (week 9, part 1; decision 1 at both granularities and decision 2 fine-grained pass, decision 2 coarse-grained and S2 do not); mention replacement alone does not help on a MaterioMiner-sized SOFC-Exp either (C20) |
| C22 | On a third corpus whose held-out documents differ in topic, mention replacement's single-model gain does not pass; its votes do. MuLMS (Schrader et al., 2023; its outermost entities, 13 types): each of its 7 materials subfields is held out in turn, and the models train on 340 sentences of the others, about MaterioMiner's size. Over 10 seeds mention replacement scores 50.76 against the CRF head's 48.66: +2.10 [−0.31, +4.51], paired by subfield. Six subfields gain (+0.15 to +7.39, the most on sofc) and poly loses 0.73. Its votes score 53.25 against the CRF votes' 50.54: +2.71 [+0.22, +5.20], which passes. Reported: macro F1 +2.12 [+1.14, +3.09], weighted F1 +2.68 [+0.32, +5.04]. As on a held-out MaterioMiner paper, most test names are unseen (88%) and the CRF head finds few of them (39%). On unseen names mention replacement gains +2.19 [−0.25, +4.63] (seen +1.30 [−1.21, +3.81]), its votes +2.92 [+0.41, +5.44]. 1 of its 2 decisions passes (about 0.05 expected by chance). The protocol made the first decision the condition for a methods claim, so C20 and C21 stay results for MaterioMiner. Across corpora the passes do not line up: the single model passes on MaterioMiner at both granularities, and the vote against the seed vote on MuLMS but not coarse-grained on MaterioMiner (+2.26 [−0.12, +4.64]). | `gpu/ner_week5_seed_analysis.json` (`mulms`); `gpu/ner_mentions.json`, `gpu/ner_matching.json`; the conversion: `results/mulms_corpus.json` | Done (week 10, pre-registered; decision 1 fails, decision 2 passes) |
| C23 | Names from unlabeled text add nothing to mention replacement on MuLMS. Three CRF teachers tag the training subfields' sentences outside the 400-sentence pool. The mentions of the sentences they tag alike and confidently (181–419 per fold, 126–358 distinct names) join the replacement pool. Over 10 seeds the method scores 50.69 on the held-out subfields: +2.04 [−0.85, +4.92] against the CRF head (decision 1) and −0.06 [−1.09, +0.96] against mention replacement (decision 2); its votes +2.48 [−0.44, +5.40] and −0.23 [−1.15, +0.68]. The teachers' names hold 4.4% of the unseen test names (1.0–7.8% per subfield), so the copies hardly show the model the names it misses. On MaterioMiner (decisions 3 and 4; 10 seeds, 4 held-out papers) it scores 49.45 fine-grained and 60.49 coarse-grained: +0.23 [−0.69, +1.14] and +0.61 [−0.96, +2.18] against mention replacement, so no gain there either, although 49–65% of a held-out paper's unseen test names occur in the fatigue corpus: the teachers' names hold 1.9–9.3% of them (fine) and 5.3–7.7% (coarse). Against the CRF head it gains +3.00 [+1.98, +4.02] fine-grained and +1.89 [−0.31, +4.10] coarse-grained, as mention replacement does (reported). | `gpu/ner_week5_seed_analysis.json` (`mulms`, `silver_names`); `gpu/ner_week11_comparisons.json` (decisions 3 and 4); `gpu/ner_mentions.json` (corpus coverage) | Done on both corpora (week 11, pre-registered): MuLMS decisions 1 and 2 and MaterioMiner decisions 3 and 4 all fail, so no methods claim. Week 12 (pre-registered, ROADMAP §4) takes the names from a corpus of the held-out subfield's own topic, 40,000 sentences per subfield, which holds 55–81% of its unseen test names (`results/mulms_coverage.json`, phase 0) |
| C24 | A lexicon harvested from the held-out topic's unlabeled text helps as a dictionary only where the held-out head's recall trails its precision by 15 points or more (sofc +5.9, pemfc +4.0, semi +2.3) and hurts elsewhere (steel −10.0, graphene −5.1, poly −3.3); the agreement filter at 0.9 admits 9–34% of the unseen names and picks the context-free ones (pemfc: silver +4.0 against gold −10.5). As a learned match feature of the CRF head it adds nothing on the 3-seed screen (−0.03 and −0.15 against the head, −1.73 and −1.85 against mention replacement); 6 copies without names add nothing to 3 (−0.20). | `silver_lexicons/<subfield>.json` (Table 5, `docs/tables/table5_probe.md`); `gpu/ner_week5_seed_analysis.json` (`mulms`, `week12`) | Week 12 (pre-registered; the probe is no decision): the feature variants and the copy control are screened; the four replacement-pool variants run after 24 October, then the decision for a winner, if any |
| C25 | Document-level propagation of a model's own predictions (C12's rule) improves NER on held-out documents of two corpora and hurts on the third, which its annotation convention explains. Pre-registered, on the CRF head's saved predictions: MuLMS, each of 7 subfields held out, +0.82 [+0.18, +1.46] over 10 seeds (49.47 against 48.66; every subfield gains, +0.07 to +2.07; 1,213 entities added per seed); MaterioMiner, coarse-grained, +0.12 [+0.01, +0.24] (week 7; fine-grained +0.30 [−0.35, +0.95]); SOFC-Exp, −0.42 [−0.61, −0.22] on its official split and −0.44 [−0.72, −0.16] on 5 folds of held-out articles (every fold loses), where the annotators leave 14% of a tagged term's occurrences untagged. It does not add to mention replacement: +0.57 [−0.12, +1.25] on MuLMS, +0.19 [−0.34, +0.73] fine-grained and +0.11 [−0.08, +0.31] coarse-grained on MaterioMiner. Retyping (reported, never a decision) is larger wherever the rule helps: +1.01 [+0.28, +1.74] on MuLMS, +0.64 [+0.29, +1.00] coarse-grained on MaterioMiner and +0.63 [+0.13, +1.14] on mention replacement there. 2 of the rule's 10 pre-registered decisions pass (about 0.25 expected by chance), on two corpora. | `gpu/ner_week5_seed_analysis.json` (`week13`, `consistency`; Table 6, `docs/tables/table6_held_out_decisions.md`); `gpu/ner_week7_comparisons.json` | Week 13 (pre-registered 20 October): 13.1a passes, 13.1b and 13.1d fail, 13.1c reported. By the revised methods criterion (ROADMAP §4, week 13) a scoped claim of two corpora, under a point, with its boundary condition; 13.2 and 13.3 (mention replacement pooled across corpora and on SOFC-Exp's folds) run after 24 October |

A draft of the sections that hold for either paper type is in
`docs/PAPER_DRAFT.md`.

## Figures and tables

Figures 1–6 are drawn from the GPU results by `src/graph/paper_figures.py`
into `docs/figures/` (no training; re-run it when results change):
- Figure 1 (`fig1_random_vs_unseen.png`): per system with 10 seeds on both
  protocols, micro F1 on the 5 random splits and on the 4 unseen papers,
  with 95% CIs and the gap: 28–31 points fine-grained, 17–20 coarse-grained;
  a third panel puts SOFC-Exp's CRF head and its votes on the same axis, 5
  random sentence splits against 5 folds of 9 held-out articles: 3.5 points
  (C2). A fourth panel does the same for MuLMS: 5 random splits of a
  400-sentence pool against its 7 held-out subfields, 17.1 points (C2,
  C22).
- Figure 2 (`fig2_seeds.png`): the CRF head's SD over 10 seeds per split;
  the 120 three-seed differences of multi-task training on the random splits
  (49 of 120 triples pass, against the 10-seed +1.83); and of document
  context on unseen papers (21 of 120 pass, against the 10-seed +1.14), with
  the week-6 screen's triple at +2.66 (C8).
- Figure 3 (`fig3_types.png`): per coarse-grained type, the multi-task
  vote's F1 against its test support per split on the random splits; SDE,
  MIP, PRP, MOD and CPR are the low ones (C9).
- Figure 4 (`fig4_threshold.png`): the CRF head's F1 per held-out paper at
  each bias on the entity tags. Fine- and coarse-grained, the curves are
  flat or fall, and their best bias differs in sign from paper to paper. A
  third panel shows, pooled, the net share of the entities each step of the
  bias adds that are right; it is at or below F1's break-even rate on
  MaterioMiner and SOFC-Exp alike (C15).
- Figure 5 (`fig5_mentions.png`): (a) per setting, the share of the gold
  test entities whose name is an entity of the same type in the training
  sentences, of another type only, or unseen: 34–40% unseen on random and
  published splits, 79–81% on held-out papers, 49% on SOFC-Exp (59% on its
  slot types), 57% on MuLMS's random splits and 88% on its held-out
  subfields; (b) the CRF
  head's recall on seen and unseen names; (c) on held-out papers,
  fine-grained, each method's F1 minus the CRF head's (a vote minus the
  seed vote) on seen and on unseen names, with 95% CIs over the 4 papers;
  only the combination of multi-task training and mention replacement lies
  above 0 on seen names too (C17, C21).
- Figure 6 (`fig6_gains.png`): on the 4 held-out papers, per granularity,
  each method's micro-F1 difference from the CRF head with its 95% CI over
  the papers: the 10-seed comparison where there is one, else the 3-seed
  screen (hollow markers: only DeBERTa-v3-large, which stopped there); and,
  last, a third training paper (+7.36 fine-grained, +8.90 coarse-grained;
  C19). Multi-task self-training has the largest mean (+5.47 fine-grained),
  but its interval crosses 0. With 10 seeds the seed vote,
  mention replacement at both granularities (C20), multi-task training
  fine-grained (+3.72 [+0.57, +6.87], reported), its combination with
  mention replacement fine-grained (+6.73 [+6.03, +7.43], C21) and the
  consistency rule coarse-grained (+0.12) lie above 0. A third paper (3
  seeds) has the largest mean at both granularities; fine-grained the
  combination lies 0.6 points below it, with a far narrower interval
  (C5–C8, C19–C21).
Tables 1–6 are generated into `docs/tables/` by `src/graph/paper_tables.py`
from the committed results (no training; re-run it when results change):
- Table 1 (`table1_published_split.md`): the published splits, MaterioMiner's
  two levels and SOFC-Exp's official split at its two levels (entity and
  slot types). Plain MatSciBERT and the CRF head with 10 seeds under micro,
  macro, weighted and strict F1, the published scores, the CRF head minus
  plain MatSciBERT, and on SOFC-Exp each method minus the CRF head (C1, C3,
  C4, C7, C13).
- Table 2 (`table2_decisions.md`): every pre-registered decision and screen
  on MaterioMiner, weeks 3–9: its machine, seeds per split, paired micro-F1
  difference with 95% CI, and whether the CI lies above 0 (C5–C8). The
  3-seed decisions of weeks 3 and 4 and their 10-seed repeats of week 5
  stand in the same table. `tableA1_all_comparisons.md` adds the reported
  and descriptive comparisons.
- Table 3 (`table3_strict.md`): default against strict IOB2 scoring on the
  published splits (C4).
- Table 4 (`table4_unseen.md`): per setting, the share of unseen names, the
  CRF head's recall on seen and unseen names, and its F1 under the four
  matching criteria (C17, C18), MuLMS's random splits and held-out
  subfields included (C22).
- Table 5 (`table5_probe.md`): per MuLMS subfield, the CPU probe of week 12:
  what the topic corpus holds of the unseen test names, what the two silver
  filters admit at 0.9, the held-out CRF head's precision and recall, and the
  dictionary rule's F1 change with each lexicon and with the gold unseen names
  (C24).
- Table 6 (`table6_held_out_decisions.md`): every pre-registered decision on
  held-out documents of the three corpora, from the seed analysis: MuLMS's
  held-out subfields (weeks 10–12), the consistency rule of weeks 7 and 13 on
  MuLMS and MaterioMiner, SOFC-Exp's article folds (13.3) and the pooled
  decision 13.2; its units, the paired micro-F1 difference with 95% CI and
  whether the CI lies above 0 (C22–C25). The methods route's evidence in one
  table; the reported comparisons stay in the README's bullets.

## Reproducibility appendix

- The protocol of each week, committed before its runs (ROADMAP §4), and the
  decision rule: the 95% Student-t CI of the mean paired difference above 0.
- GPU runs: one frozen commit per batch, one Kaggle image pinned by digest,
  deterministic CUDA. Runs repeat exactly across GPUs, sessions and commits:
  the 54 teachers retrained in the week-5 self-training session equal the
  separately trained CRF runs, and every other session's first job repeats
  a week-4 run.
- Every run's test predictions are in `results/`, so any vote or score can be
  recomputed without training; the tables are generated and checked by the
  test suite (`report_results.py --check`).

## Still missing before submission

1. A second materials NER dataset (week 6), so the findings do not rest on
   4 papers. SOFC-Exp is converted (`sofc_exp.py`) and checked: MatSciBERT
   with the CRF head scores micro 84.98 and macro 82.31 on its official
   split with one seed, against the published 84.77 and 82.39. Its main
   comparisons need 10 seeds per split, on random splits and on folds
   grouped by article. Week 8, part 1 (ROADMAP §4, fixed before its runs)
   ran the CRF head that way, 5 random sentence splits and 5 folds of 9
   held-out articles with 10 seeds each: 86.27 against 82.79, a gap of
   3.48 points (C2). A third corpus, MuLMS (week 10), holds out documents
   of other materials subfields: the gap is 17.06 points there (C2), and
   mention replacement's single model does not pass (C22). Week 12 tests
   names from a corpus of the held-out subfield's topic there (C23); if
   its decisions fail too, the methods route on MuLMS is exhausted and
   the paper is the evaluation paper with MaterioMiner's method results.
2. The authors' confirmation of the split and the scoring (item 6; the email
   is drafted). The reproduction (C1) already matches both numbers.
3. Related work. A draft is in `docs/PAPER_DRAFT.md`, Section 8 (17
   October). `docs/references.bib` holds 35 checked references (verified
   on 16 October against Crossref, publishers, the ACL Anthology and arXiv;
   MuLMS on 17 October against the ACL Anthology), with notes on their
   scope:
   - materials NER: MatSciBERT (Gupta et al., 2022); the MaterioMiner dataset
     (Durmaz et al., 2024; its title does not contain "MaterioMiner"); the
     closest prior work, Potu et al. (2025), whose domain shift is between
     two fatigue datasets rather than unseen papers of one; SOFC-Exp
     (Friedrich et al., 2020; its tokens come from ChemDataExtractor, Swain and
     Cole, 2016, and Java's BreakIterator, C16); Weston et al. (2019); Trewartha et al.
     (2022); MatSci-NLP (Song et al., 2023); MuLMS (Schrader et al., 2023: 50
     CC-BY articles from 7 materials subfields; a head per corpus with SOFC-Exp
     or a synthesis-procedure corpus gives no NER gain, C7);
   - evaluation: seed variance and score distributions (Reimers and
     Gurevych, 2017; Dodge et al., 2020; Bouthillier et al., 2021); splits
     (Gorman and Bedrick, 2019, for random splits; Søgaard et al., 2021,
     against them; Babl et al., 2025, on entity leakage in random NER
     splits); unseen mentions (Augenstein et al., 2017); looser matching
     criteria (Segura-Bedmar et al., 2013, SemEval-2013 Task 9.1); significance
     testing (Dror et al., 2018); pre-registration (van Miltenburg et al.,
     2021);
   - the methods tested: CRFs (Lafferty et al., 2001; Lample et al., 2016),
     label semantics (Ma et al., 2022, which embeds label names), document
     context (FLERT; Schweter and Akbik, 2020), self-training (BOND, Liang et
     al., 2020; Du et al., 2021 uses web text, not an in-domain corpus),
     DeBERTa-v3 (He et al., 2023), mention replacement (Dai and Adel, 2020),
     one sense per discourse (Gale et al., 1992; Krishnan and Manning, 2006);
   - recall calibration: the F1-optimal threshold of a calibrated classifier
     is half its F1 (Lipton et al., 2014); a single weight on the O label
     already trades precision for recall in CRF extractors (Minkov et al.,
     2006); training CRFs for F-score (Suzuki et al., 2006); calibration
     under shift (Ovadia et al., 2019). The paper's contribution there is
     the choice of the bias for unseen documents, on inner folds.
4. The venue. Candidates are a materials-informatics journal or workshop, or
   a scholarly-document-processing workshop.
