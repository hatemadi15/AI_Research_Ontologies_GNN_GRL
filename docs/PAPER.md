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
its 10-seed decisions (+1.14 [−0.94, +3.22] against the CRF head). The
candidate left is multi-task self-training (screen +6.06 [+1.05, +11.06]
fine-grained; 10-seed decisions next on MaterioMiner), with mention
replacement screened next. Recall calibration (week 7, part 2a) passed none
of its 5 decisions, and showed why a decoding change cannot close the recall
gap on unseen papers (C15). Most entities of an unseen paper are names the
training papers never annotate (81% fine-grained), and every gain that holds
there is a gain on those names (C17), so that is what a method for unseen
papers has to improve. The week-6 protocol
excluded multi-task methods from SOFC-Exp as a one-level corpus, but SOFC-Exp
has a second level, 17 evaluated slot types that refine its 4 entity types
(ROADMAP, "Prepared for part 2d"), so multi-task training (C7) and, if it
passes, multi-task self-training can be repeated there.

**Working titles.** Methods: an approach for materials NER on unseen papers,
named after the change that passes. Evaluation: "How much of a materials-NER
gain survives? Splits, scoring and seeds on MaterioMiner".

## Claims and evidence

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | The published numbers are reproduced on the split the release ships (323 / 57 / 96 sentences, a different partition per granularity, not described in the paper), scored like its code (micro F1, seqeval's default mode): plain MatSciBERT scores 70.19 [69.03, 71.35] fine-grained and 72.52 [71.88, 73.16] coarse-grained over 10 seeds, against 69.92 and 72.32. Single runs of plain MatSciBERT range over 4.6 points fine-grained (68.0–72.7) and 3.6 coarse-grained (70.4–74.1). | `results/gpu/ner_results_*_published_split.json`; README, the published-split rows | Done (GPU, 10 seeds); the authors are asked to confirm split and scoring (ROADMAP item 6) |
| C2 | Random splits hide a large gap (the paper says the out-of-distribution generalisation "can not be estimated" from four publications; leaving one paper out estimates it). The seed vote `crf_ens3` scores 79.68 fine-grained and 78.63 coarse-grained over 5 random splits, but 48.81 and 61.10 on a paper the model has never seen: 18–31 points lower. | `results/gpu/ner_week5_comparisons.json` (`crf_ens3` vs `crf`) | Done (GPU, 10 seeds) |
| C3 | A CRF head with IOB2 constraints beats the published numbers. On the published split it scores 73.31 [72.28, 74.35] and 75.48 [74.23, 76.72], +3.12 and +2.96 over plain MatSciBERT, paired by seed. Over 5 random splits, the reported system beats them fine-grained under micro, macro and weighted F1, and coarse-grained under micro and weighted F1 but not macro (70.83 [64.49, 77.17]). | README, the published-split and reported-system rows | Done (GPU, 10 seeds) |
| C4 | Much of the CRF head's gain is span validity. Under strict IOB2 scoring its gain on the published split shrinks from +3.12 to +1.70 [+0.21, +3.19] fine-grained and from +2.96 to +1.07 [−0.82, +2.97] coarse-grained. Over the random splits (CPU, 5 seeds) the fine-grained gain vanishes under strict scoring (+0.10 [−2.08, +2.29]), and dropping invalid tags from the baseline's output gives the same; coarse-grained, +1.91 [+1.17, +2.65] survives. | README, "Why the CRF helps"; `ner_week5_seed_analysis.json` (`published_split`) | Done |
| C5 | A vote over 3 seeds is the most reliable addition: +2.20 and +1.67 over the mean of single CRF runs on the random splits, +2.36 and +2.50 on unseen papers, every CI above 0. | `ner_week5_comparisons.json` (`crf_ens3` vs `crf`) | Done (GPU, 10 seeds) |
| C6 | Popular additions do not pass a pre-registered paired test with 10 seeds: label semantics (model or vote, either protocol), self-training with 3 teachers on unseen papers (+1.74 [−1.16, +4.64] fine-grained, the week-4 claim), coarse-grained multi-task training (+0.71 [−0.15, +1.56], the week-3 claim), and document context (FLERT) on unseen papers (+1.14 [−0.94, +3.22] fine-grained, +0.12 [−2.03, +2.27] coarse-grained; its vote −1.44 [−3.80, +0.91] against the multi-task vote). Earlier: class weights, domain-adaptive and in-domain pre-training, other backbones and a mixed-backbone vote. | ROADMAP P1 weeks 1–6; README week-3 to week-6 tables | Done (document context: week 6) |
| C7 | Multi-task training over both granularities is the one addition that holds: fine-grained +1.83 [+0.32, +3.34] over the CRF head on the random splits. Its vote improves on the seed vote coarse-grained on the random splits (+0.79 [+0.17, +1.42]) and fine-grained on unseen papers (+3.65 [+0.49, +6.82]): 2 of 12 vote comparisons, where about 0.3 would pass by chance. Self-training improves on the random splits (+1.05 and +1.04, reported, no decision). | `ner_week5_comparisons.json` | Done (GPU, 10 seeds) |
| C8 | How far a 3-seed design misleads, from the 120 seed triples of the 10-seed runs: the 1.8-point multi-task gain passes in 41% of the triples; label semantics and self-training on unseen papers in 2–18%; up to 37% of the triples give the opposite sign. A split's micro F1 varies over seeds with an SD of 1.5–1.7 points on random splits and 2.2–2.5 on unseen papers. The 1-point CPU gains of week 3 that passed with 3 seeds do not hold with 10. Week 6 shows it prospectively: document context passed the 3-seed screen with +2.66 [+0.91, +4.40] fine-grained, but gains +0.49 [−3.10, +4.07] on the 7 seeds the screen did not see (one paper from +4.29 to −2.59). | `ner_week5_seed_analysis.json` (`triples`, `spread`); `gpu/ner_week6_screen.json`, `gpu/ner_week6_comparisons.json`; README | Done (exploratory) |
| C9 | A few rare coarse-grained types keep macro F1 near the published number: in the reported system's votes on the random splits, SDE and MIP score about 18 F1 and PRP, MOD and CPR 40–42, with 2–8 test entities per split. | `ner_week5_seed_analysis.json` (`types`) | Done (exploratory); week 6 targets them |
| C10 | Where the CRF head fails on unseen papers: fine-grained it misses 36% of the gold entities outright (10% on random splits) and predicts 10,026 entities for 13,280 gold ones; 41% of these misses have a type seen fewer than 10 times in the three training papers. Coarse-grained it misses only 6%, but 16% get the right span and the wrong type, and 21% a wrong boundary. | `ner_week5_seed_analysis.json` (`errors`) | Done (exploratory, 10 seeds) |
| C11 | A larger general-domain encoder does not transfer to unseen papers: DeBERTa-v3-large (435M parameters) with the CRF head scores −4.38 [−7.32, −1.45] fine-grained and −8.01 [−11.72, −4.30] coarse-grained against MatSciBERT (110M) on the held-out papers, while its validation scores on the training papers are similar. | `gpu/ner_week6_screen.json` | Screen (3 seeds). SOFC-Exp S2 did not run: out of memory on a 15 GB T4 at batch size 16 (a deviation; ROADMAP P1, "Week 6 so far") |
| C12 | Document-level label consistency, a post-processing rule (a term the model tags in most of its occurrences in a paper is tagged in the rest), adds little on MaterioMiner and hurts on SOFC-Exp: +0.30 [−0.35, +0.95] fine-grained and +0.12 [+0.01, +0.24] coarse-grained on the CRF head, no significant change on the strongest votes, +0.10 / +0.06 on document context (reported); on SOFC-Exp −0.42 [−0.61, −0.22], and −0.38 [−0.60, −0.15] with retyping (which did better coarse-grained on MaterioMiner, +0.64 [+0.29, +1.00], post hoc). How an annotation scheme treats repeated mentions decides whether "one sense per discourse" helps. | `gpu/ner_week7_comparisons.json`; `gpu/ner_week5_seed_analysis.json` (`published_split`, `sofc`) | Done (week 7, part 1) |
| C13 | The CRF head's gain repeats on a second dataset: on SOFC-Exp's official split (11 unseen test articles, 10 seeds) it scores 83.75 [82.85, 84.65] against plain MatSciBERT's 82.53 [81.84, 83.23], +1.22 [+0.50, +1.93] paired by seed; its seed votes score 85.13. The published MatSciBERT-CRF (84.77) lies above the CI over seeds. | `gpu/ner_results_sofc_published_split*.json`; README, SOFC-Exp row | Done (week 6, S1) |
| C14 | Using the type hierarchy after the fact adds little: completing the fine-grained predictions from the coarse-grained ones, where a coarse type maps to one fine type in the training papers, gives +0.34 [−0.91, +1.59] on the CRF head and +0.12 [−1.76, +2.01] on the multi-task vote (fine-grained, unseen papers). Its added entities are right 39% of the time. | `gpu/ner_week7_comparisons.json` (`_cf`) | Done (week 7, part 2c) |
| C15 | The recall gap on unseen papers is not a decision-threshold problem. A bias on the CRF head's emission of every entity tag, chosen per held-out paper on inner folds (recall calibration), gives −0.59 [−1.75, +0.57] fine-grained and +0.17 [−0.14, +0.48] coarse-grained against the CRF head, its votes do not beat the strongest systems, and on SOFC-Exp (chosen on the validation articles) it gives −0.21 [−0.58, +0.17]. Even the best bias per paper in hindsight would gain +0.56 and +0.18. The entities a small bias adds are right 22% (fine-grained), 19% (coarse-grained) and 21% (SOFC-Exp) of the time, at or below F1's break-even rate and far below the precision of the predictions above the threshold (53.7%, 57.3%, 82.2%). A bias of +3 recovers 7% of the missed fine-grained entities, with 6 wrong entities per right one. | `gpu/ner_week7_comparisons.json` (`_eb`, `_ebv`); `gpu/ner_entity_bias_*.json`; the `bias_test` counts of `gpu/ner_results_*_crf_bias*.json`; ROADMAP, "Outcome of part 2a" | Done (week 7, part 2a; the counts analysis is exploratory) |
| C16 | SOFC-Exp's test articles are cut into words differently from its training articles, against what its README states (Stanford CoreNLP). ChemDataExtractor 1.3.0's word tokenizer gives the tokens of all 7,630 sentences of the 26 training and 8 development articles, and Java's word BreakIterator gives 1,833 of the 1,836 sentences of the 11 test articles. CoreNLP (3.9.2, 4.0.0, 4.5.7) gives at most 48.6% of all sentences. Training has `Fig.`, `cm−2`, `O2−` and `high - temperature` where the test articles have `Fig .`, `cm − 2`, `O2 −` and `high-temperature`. Every result on the official split, the published ones included, is scored under this shift. | The release's texts and token offsets (scratch checks, ROADMAP §4, week-6 amendment); `tests/test_chem_tokenize.py` (7,630 of 7,630 on the committed tokens) | Done (exploratory; to report as a dataset note) |
| C17 | Unseen papers are mostly unseen names, and the gains that hold are gains on them. On a held-out paper, 81% of the fine-grained gold entities (79% coarse-grained; 72–83% per paper) are names that no training sentence annotates with that type, against 35% (40%) on random splits and 40% (39%) on the published split. The CRF head finds 59% of the unseen names on random splits but 35% in a new paper (seen names: 88% and 71%). With the random splits' mix of names its recall per kind on held-out papers would give 58.5 instead of 40.6: the mix of names and the lower recall within each kind each account for 40–60% of the recall gap, 37 points fine-grained and 20 coarse-grained (in either order of the decomposition). Patterned names transfer (unseen Value 76%, Alloy and MaterialClass 72%), descriptive ones do not (StateDescriptor 30%, Mechanism 33%, GrainBoundary 3%). On SOFC-Exp, 49% of the test names are unseen and the CRF head finds 82% of them. On held-out papers every fine-grained gain is significant only on unseen names: multi-task +4.02 [+1.53, +6.51] (seen +2.88 [−2.70, +8.47]), its vote +3.60 [+2.32, +4.88], the context vote +2.68 [+0.84, +4.53], multi-task self-training's screen +6.44 [+2.25, +10.62]. The CRF head's gain over plain MatSciBERT is on unseen names too: +5.94 [+3.11, +8.76] on the published split (fine-grained) and +3.02 [+1.99, +4.05] on SOFC-Exp, against −0.47 and −0.52 on seen ones. | `gpu/ner_mentions.json` (`ner_mentions.py`; its F1 over all names equals every run's stored micro F1) | Done (exploratory; the per-kind differences carry no decision) |
| C18 | On a new paper the model finds most entities, but not the exact span and type its annotators chose. Under the looser criteria of SemEval-2013 Task 9.1 (Segura-Bedmar et al., 2013), the CRF head's coarse-grained F1 on held-out papers is 58.60 when span and type must match, 74.17 when only the span must, and 90.74 [89.17, 92.32] when any overlap counts, against 76.96, 83.91 and 93.38 on random splits: the 18.4-point gap shrinks to 9.7 and then 2.6. Fine-grained it shrinks from 31.0 to 25.0 and 18.3 (overlap F1 70.58 against 88.83); there entities are missed outright (C10, C17). On held-out papers 31% of the fine-grained predictions overlap a gold entity with another span or type (12% on random splits): 10.8% lie inside a longer gold entity (the new paper tags "multi-GB cracks", "EBSD data", "smooth samples" where the model tags the noun), and 4.2% have a type that is an MMO ancestor or descendant of the gold one ("samples" is a TestSpecimen in one paper and a FatigueTestSpecimen in another). On SOFC-Exp the CRF head scores 83.75 strict and 88.80 under overlap. | `gpu/ner_matching.json` (`ner_matching.py`; its strict F1 equals every run's stored micro F1) | Done (exploratory) |

A draft of the sections that hold for either paper type is in
`docs/PAPER_DRAFT.md`.

## Figures and tables

Figures 1–5 are drawn from the GPU results by `src/graph/paper_figures.py`
into `docs/figures/` (no training; re-run it when results change):
- Figure 1 (`fig1_random_vs_unseen.png`): per system with 10 seeds on both
  protocols, micro F1 on the 5 random splits and on the 4 unseen papers,
  with 95% CIs and the gap: 28–31 points fine-grained, 17–20 coarse-grained
  (C2).
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
  published splits, 79–81% on held-out papers, 49% on SOFC-Exp; (b) the CRF
  head's recall on seen and unseen names; (c) on held-out papers,
  fine-grained, each method's F1 minus the CRF head's (a vote minus the
  seed vote) on seen and on unseen names, with 95% CIs over the 4 papers
  (C17).
- Table 1: the published split (plain MatSciBERT and the CRF head, 10 seeds)
  next to the published numbers, under micro, macro, weighted and strict F1
  (C1, C3, C4).
- Table 2: every pre-registered comparison of weeks 1–5, its protocol, its
  role (decision or reported) and its outcome, with the 3-seed and 10-seed
  results side by side (C5–C8).
- Table 3: default against strict IOB2 scoring for the baseline and the CRF
  head (C4).

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
   grouped by article.
2. The authors' confirmation of the split and the scoring (item 6; the email
   is drafted). The reproduction (C1) already matches both numbers.
3. Related work. `docs/references.bib` holds 34 checked references (verified
   on 16 October against Crossref, publishers, the ACL Anthology and arXiv),
   with notes on their scope:
   - materials NER: MatSciBERT (Gupta et al., 2022); the MaterioMiner dataset
     (Durmaz et al., 2024; its title does not contain "MaterioMiner"); the
     closest prior work, Potu et al. (2025), whose domain shift is between
     two fatigue datasets rather than unseen papers of one; SOFC-Exp
     (Friedrich et al., 2020; its tokens come from ChemDataExtractor, Swain and
     Cole, 2016, and Java's BreakIterator, C16); Weston et al. (2019); Trewartha et al.
     (2022); MatSci-NLP (Song et al., 2023);
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
