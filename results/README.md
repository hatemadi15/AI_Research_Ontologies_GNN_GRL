# Results snapshot

Summary JSON files produced by the runs reported in the top-level README.
They were generated on CPU with `SEED=0`; LLM runs used `openai/gpt-4o-mini`
via OpenRouter. `data/processed/` is not versioned, so these copies let
reviewers check the reported numbers.

| File | Produced by | Content |
|---|---|---|
| `eval_pipeline_llm_off.json` | `run_pipeline.py` (LLM features off) | leakage-free typing / alignment / clustering / taxonomy metrics |
| `eval_pipeline_llm_on.json` | `run_pipeline.py` (LLM features on) | same, with LLM alignment, RAG typing and LLM relation labelling |
| `eval_v5_rescored_llm_off.json`, `eval_v5_rescored_llm_on.json` | `eval_f1.py --legacy-alignment-csv` on the v5 outputs | the old (leaky) v5 predictions scored with the new metrics |
| `relation_eval_llm_off.json`, `relation_eval_llm_on.json` | `eval_relations.py` | class-level relation evaluation against the ontology restrictions |
| `ablation_summary.json` | `run_ablation.py` (LLM off) | one row per preset, with the paired typing difference and exact McNemar p-value against `full_pipeline`; `oracle` rows are upper bounds |
| `term_typing_mindf2.json` | `term_typing_baseline.py --llm` | cross-validated term-typing baselines on the 315 graph terms (5 folds × 3 seeds; LLM on the first seed) |
| `term_typing_mindf1.json` | `term_typing_baseline.py --min-df 1` | the same baselines on all 1,055 annotated surfaces (no LLM) |
| `ner_results_fine.json`, `ner_results_coarse.json` | `ner_baseline.py` | MatSciBERT NER under the MaterioMiner protocol, per seed |
| `ner_results_{fine,coarse}_random_split_vary.json` | `ner_baseline.py --vary-split` | the same, with a new random split per seed (5 splits) |
| `ner_results_{fine,coarse}_leave_one_paper_out.json` | `ner_baseline.py --leave-one-paper-out` | trained on three papers, tested on the fourth (4 folds) |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_cw.json` | `ner_baseline.py --class-weights` with `--vary-split` or `--leave-one-paper-out` | class-weighted loss, on the same splits and model seeds as the baseline file of that protocol |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf.json` | `ner_baseline.py --crf` with `--vary-split` or `--leave-one-paper-out` | CRF head with IOB2 constraints, on the same splits and model seeds |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_mt.json` | `ner_baseline.py --crf --multi-task` with `--vary-split` or `--leave-one-paper-out` | multi-task: one MatSciBERT encoder with a CRF head per granularity, trained once for both; each file holds one head's runs, on the same splits and model seeds as the CRF files |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_dapt.json` | `ner_baseline.py --crf --tag dapt --model data/processed/dapt/matscibert_pubmed` with `--vary-split` or `--leave-one-paper-out` | the CRF head fine-tuned from the domain-adapted checkpoint (`dapt_mlm.json`), on the same splits and model seeds as the CRF files |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_{m0,m100,m200}.json` | `ner_baseline.py --crf --tag m<N> --seed-offset <N> --save-predictions` with `--vary-split` or `--leave-one-paper-out` | ensemble members: the CRF head with model seeds K + N on split K (seed N for each held-out paper), with their test predictions; `crf_m0` is the CRF head re-run with 2 CPU threads |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_ens3.json` | `ner_ensemble.py --members <the three member files>` | entity-level majority vote of the three members, per split or held-out paper; `summary.vs_member_mean` is the vote minus the mean of its members, per split |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_{scibert,pubmedbert}.json` | `ner_baseline.py --crf --save-predictions --tag scibert --model allenai/scibert_scivocab_uncased` (`--tag pubmedbert --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) with `--vary-split` or `--leave-one-paper-out` | week 3: the CRF head on a SciBERT or PubMedBERT encoder, one model seed per split (K on split K, 0 on each held-out paper), with test predictions for the mixed vote |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_mix3.json` | `ner_ensemble.py --variant crf_mix3 --members <the crf_m0, crf_scibert and crf_pubmedbert files>` | week 3: entity-level majority vote of three backbones (MatSciBERT `crf_m0`, SciBERT, PubMedBERT); `model` is `mixed` and `config.models` lists the members' models |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_mt_{m100,m200}.json` | `ner_baseline.py --crf --multi-task --tag m<N> --seed-offset <N>` with `--vary-split` or `--leave-one-paper-out` | week 3: multi-task with model seeds K + N; with `crf_mt` (offset 0) the three seeds of the multi-task family |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_ls_{m0,m100,m200}.json` | the label-semantics variant of `ner_baseline.py --crf` (week-3 protocol, variant 2), seed offsets 0, 100 and 200, with `--vary-split` or `--leave-one-paper-out` | week 3: emissions scored against embeddings of the MMO label descriptions, three model seeds per split and held-out paper |
| `ner_results_{fine,coarse}_leave_one_paper_out_crf_st_{m0,m100,m200}.json` | the self-training variant of `ner_baseline.py --crf --leave-one-paper-out` (week-3 protocol, variant 3), seed offsets 0, 100 and 200 | week 3: a student trained on gold plus silver-tagged in-domain sentences, three model seeds per held-out paper (no random splits); each run's `self_training` gives the sampled and silver sentence counts and the teacher's validation and test F1 |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_dapt2_{m0,m100,m200}.json` | `ner_baseline.py --crf --model <in-domain checkpoint> --tag dapt2_m<N> --seed-offset <N>` with `--vary-split` or `--leave-one-paper-out` | week 3: the CRF head fine-tuned from MatSciBERT further pre-trained on open-access fatigue papers (variant 5), three model seeds per split and held-out paper |
| `dapt_mlm.json` | `dapt_mlm.py` | domain-adaptive pre-training of MatSciBERT on the PubMed sentences: corpus filtering (duplicates, sentences sharing a word 8-gram with MaterioMiner), settings, held-out MLM perplexity before training and after each epoch, training loss per step |
| `dapt_mlm_fatigue.json` | `dapt_mlm.py --corpus data/processed/corpus/fatigue_sentences.txt --out data/processed/dapt/matscibert_fatigue` | week 3 (variant 5): the same pre-training of MatSciBERT on the in-domain fatigue corpus (`fatigue_corpus.json`), with the same filters, settings and report fields; the starting checkpoint of the `_crf_dapt2_m<N>` runs |
| `fatigue_corpus.json` | `build_fatigue_corpus.py` | the in-domain pre-training corpus of the week-3 protocol (variant 5): the Europe PMC query and filters, the counts at every stage, the article IDs (PMCID, DOI) and the SHA-256 of the 40,000 sentences (`data/processed/corpus/fatigue_sentences.txt`, not versioned; `--offline` rebuilds it from the cached XML) |
| `sofc_corpus.json` | `build_sofc_corpus.py` | the in-domain SOFC corpus of the week-6 amendment (multi-task self-training on SOFC-Exp): the Europe PMC query and filters (SOFC-Exp's 45 articles left out, every sentence sharing a word 8-gram with them dropped), the counts at every stage, the article IDs and the SHA-256 of the 40,000 sentences (`data/processed/corpus/sofc_sentences.txt`, not versioned; `--offline` rebuilds it from the cached XML), cut into words with ChemDataExtractor 1.3.0's tokenizer (`chem_tokenize.py`), as SOFC-Exp's training articles are |
| `ner_comparisons.json` | `ner_baseline.py --compare <results dir>` | each variant minus its reference (`reference`: the baseline for `cw` and `crf`, the CRF for `crf_*` variants), paired by split and model seed: per-run differences, their mean and 95% CI for micro, strict micro, macro and weighted F1 |
| `ner_week3_comparisons.json` | `ner_seedmeans.py` | week-3 comparisons of 3-seed means (ROADMAP §4): per split, a variant minus its reference, where a family of three seeds scores their mean (`crf` = `crf_m0`, `crf_m100`, `crf_m200`) and an ensemble scores once; each entry gives its `role` (`decision` or `reported`), whether all 5 splits or 4 papers are paired (`complete`), the per-split scores of both sides, and the per-run differences with their mean and 95% CI for micro, strict micro, macro and weighted F1. Self-training has no random-split runs (unseen papers only, by the protocol), so its two random-split comparisons are absent |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_ls_ens3.json`, `ner_results_{fine,coarse}_leave_one_paper_out_crf_st_ens3.json` | `ner_ensemble.py --members <the three member files>` | week-4 votes (ROADMAP §4, week-4 protocol) over the three seeds of label semantics (`crf_ls_m*`) and self-training (`crf_st_m*`, unseen papers only), made from their saved CPU predictions without training |
| `ner_week4_comparisons.json` | `ner_seedmeans.py --week 4` | week-4 comparisons of the CPU votes, as in `ner_week3_comparisons.json`: `crf_ls_ens3` and `crf_st_ens3` against `crf_ens3` (decided on the 4 unseen papers) and against the 3-seed CRF reference (reported) |

`ner_results_fine.json` and `ner_results_coarse.json` hold all 5 model seeds
per granularity (see `runs`), trained on one fixed random split (`--split-seed
0`). Their `summary` gives, for micro, strict micro, macro and
support-weighted F1, the mean, the population SD and the 95% Student-t CI over
runs. Files with a `_random_split_vary` suffix use a new random split per
seed; `_leave_one_paper_out` files train on three papers and test on the
fourth. A final `_cw` (class-weighted loss) or `_crf` (CRF head) marks a
training variant; files without one are the plain baseline. A week-2
variant built on the CRF head adds its tag: `_crf_mt` is multi-task
training, `_crf_dapt` starts from the domain-adapted checkpoint, `_crf_m<N>`
files are ensemble members and `_crf_ens3` their vote. These runs use 2 CPU
threads where the `_crf` runs used 3, so a run with the same seeds is not
bit-identical. In week 3 a trailing `_m<N>` marks one seed of a family
(`_crf_mt_m100`, `_crf_ls_m0`, `_crf_dapt2_m200`), `_crf_scibert` and
`_crf_pubmedbert` change the encoder, and `_crf_mix3` is the vote of the
three encoders. Week 4 adds `_crf_ls_ens3` and `_crf_st_ens3`, the votes over
the three label-semantics and self-training seeds.

`gpu/` holds the week-4 runs on Kaggle T4 GPUs (ROADMAP §4, week-4
protocol), under the same file names. Each run records its `environment`
(GPU, CUDA, cuDNN, torch, transformers, deterministic algorithms); all runs
share one environment and one frozen commit, and a run repeats exactly.
GPU runs are compared only with GPU runs: `gpu/ner_week4_comparisons.json`
(`ner_seedmeans.py --week 4 --gpu --results results/gpu`) holds those
comparisons, plus descriptive GPU-minus-CPU entries (`role` `descriptive`)
for the systems run on both machines. New in `gpu/`: `_crf_mt_ls_m<N>`
(label semantics on the multi-task model) and `_crf_st3_m<N>` (self-training
with 3 teachers, whose runs record each teacher's scores), and the vote
`_crf_mt_ens3`. `kaggle_jobs.py collect` copies the files of a GPU session
here after checking that every run is a GPU run with the shared environment.

Week 5 (week-5 protocol) adds model seeds 300 to 900 to the families `crf`,
`crf_mt`, `crf_ls` and `crf_st3` (`_m300` to `_m900`; the multi-task run of
seed 0 keeps the name `_crf_mt`), and the votes over the seed triples 300–500
and 600–800 (`_ens3_m300`, `_ens3_m600`, next to `_ens3` for 0–200).
`gpu/ner_week5_comparisons.json` (`ner_seedmeans.py --week 5 --gpu --results
results/gpu`) compares 10-seed means: a family scores the mean of its 10 runs
on a split, a vote the mean of its 3 votes. `gpu/ner_week5_seed_analysis.json`
(`ner_seed_analysis.py`) holds the exploratory analyses: the seed spread, the
3-seed comparisons of all 120 seed triples, the per-type scores of each vote
family, and the published split. The published split itself is in
`gpu/ner_results_{fine,coarse}_published_split.json` (plain MatSciBERT) and
`gpu/ner_results_{fine,coarse}_published_split_crf.json` (the CRF head):
`ner_baseline.py --published-split --seeds 0 100 ... 900`, the partition and
tags that the MaterioMiner release ships (`data/raw/dataset/published_split/`).

Week 6 (week-6 protocol) adds methods for unseen papers on the held-out
papers: `_crf_deberta_m<N>` (DeBERTa-v3-large), `_crf_ctx_m<N>` (document
context, `--context 64`) and `_crf_mt_st3_m<N>` (multi-task self-training),
their votes (`_ens3`, `_ens3_m300`, `_ens3_m600`), and SOFC-Exp's official
split (`gpu/ner_results_sofc_published_split*.json`: plain MatSciBERT, the
CRF head and the methods, model seeds 0 to 900). `gpu/ner_week6_screen.json`
(`ner_seedmeans.py --week 6 --gpu --screen`) holds the screen of seeds 0 to
200, and `gpu/ner_week6_comparisons.json` (without `--screen`) the 10-seed
decisions. `gpu/ner_week5_seed_analysis.json` adds SOFC-Exp to the published
splits, the per-type F1 of each week-6 method against the CRF head
(`week6_types`; later also multi-task training, mention replacement and,
from week 9, the two together) and the CRF head's kinds of error
(`errors`). Document
context has all 10 seeds (stage 2, session a of 16 October), and on SOFC-Exp
`gpu/ner_results_sofc_published_split_crf_ctx.json` (S2, session a of 17
October, model seeds 0 to 900); DeBERTa-v3-large
has only the screen's seeds 0 to 200, since it stopped there, and no SOFC-Exp
file (its S2 ran out of GPU memory). Multi-task self-training has all 10
seeds (`_crf_mt_st3_m0` to `_m900`, stage 2 in session b of 17 October:
3 multi-task teachers and 10 students per held-out paper, each student file
with its teachers' scores and the silver counts) and its votes; it has no
SOFC-Exp file, since it passed no decision on MaterioMiner.

Week 7, part 1 (week-7 protocol) applies the consistency rule
(`ner_consistency.py`) to saved predictions; it trains nothing. `_dc` marks
its output and `_dcr` its output with retyping, before the seed tag:
`gpu/ner_results_<granularity>_<protocol>_crf_dc_m<N>.json` for the CRF head
on both protocols, and `_crf_ens3_dc*` and `_crf_mt_ens3_dc*` for the votes
on the held-out papers, and `_crf_ctx_dc_m<N>` and `_crf_ctx_ens3_dc*` for
document context (reported). Part 2c (coarse-to-fine completion,
`ner_coarse_to_fine.py`) writes `_cf` the same way, fine-grained only:
`_crf_cf_m<N>`, `_crf_mt_cf*` and `_crf_mt_ens3_cf*`, each run with the
fold's type mapping (`mapping`), the entities added and the test sentences
it could not align (`n_unaligned`). The votes were made again with `ner_ensemble.py
--save-predictions` (their scores equal the stored votes) before the rule was
applied. The output files keep the scores, the number of entities added and
retyped, and the input's F1 (`input_f1`), but not the predictions, which the
rule regenerates in seconds. `gpu/ner_week7_comparisons.json`
(`ner_seedmeans.py --week 7 --gpu --results results/gpu`) holds the
comparisons.

Week 7, part 2a (recall calibration; Kaggle session `w7a` of 16 October):
- `gpu/ner_results_<granularity>_leave_one_paper_out_crf_bias_m<N>.json`
  and `gpu/ner_results_sofc_published_split_crf_bias.json` are the CRF head
  again with `--entity-bias` (model seeds 0 to 900). Its scores are those
  of D = 0 and equal the `crf_m<N>` and `crf` files. For every D of the
  grid, `bias_test` and `bias_val` hold the entity counts [correct,
  predicted, gold] per test paper and of the validation sentences. The
  predictions at each D (`test_pred_bias`) were dropped here and stay in
  the pulled session (`data/processed/kaggle/ner-gpu-jobs/20261016T152832`,
  not in git).
- `_leave_two_papers_out_crf_bias_m{0,100,200}` are the inner folds
  (`--leave-two-papers-out`, 6 pairs of papers), counts only.
- `ner_bias.py` writes the runs at the chosen D:
  - `_crf_eb_m<N>` (D from the inner folds, with predictions);
  - `_crf_ebv_m<N>` and `gpu/ner_results_sofc_published_split_crf_ebv.json`
    (D from each run's validation sentences);
  - each run keeps its D (`bias`), its D = 0 scores (`input_f1`) and the
    entity counts at 0 and at D (`entities`).
- `gpu/ner_entity_bias_<granularity>_<protocol>_<choice>.json` hold the
  chosen D, the inner curves and the test sets' own curves (reported,
  never used to choose).
- `_crf_eb_ens3*` are the votes over the seed triples.

Week 7, part 2b (mention replacement, `--augment-mentions 3`; Kaggle session
`w7c` of 17 October): `gpu/ner_results_<granularity>_leave_one_paper_out_crf_mr_m<N>.json`
are the CRF head trained with 3 copies of each training sentence with an
entity, each mention swapped with probability 0.5 for a training mention of
the same type (seeds 0 to 200 for the screen; each run stores its number of
copies, `n_augmented`), and `_crf_mr_ens3` their vote.
`gpu/ner_week7_screen.json` (`ner_seedmeans.py --week 7 --gpu --screen`)
holds the screen. Stage 2 (`w7d`, the same day) adds the seeds 300 to 900
(`_crf_mr_m300` to `_m900`; the votes `_crf_mr_ens3`, `_ens3_m300` and
`_ens3_m600`) and SOFC-Exp's official split with model seeds 0 to 900
(`gpu/ner_results_sofc_published_split_crf_mr.json`). Its decisions are in
`gpu/ner_week7_comparisons.json` and, for SOFC-Exp, in
`gpu/ner_week5_seed_analysis.json` (`published_split`, `sofc`, `crf_mr`).

Week 9, part 1 (mention replacement on the multi-task model, `--multi-task
--augment-mentions 3`; Kaggle session `w9a` of 17 October on
`ner-gpu-jobs-c`, from bundle 9468ae7):
`gpu/ner_results_<granularity>_leave_one_paper_out_crf_mt_mr_m<N>.json`, N =
0 to 900, are the multi-task model's fine- and coarse-grained heads (one
model per seed, a file per head), each run with its number of copies
(`n_augmented`, one set for both heads), and `_crf_mt_mr_ens3`,
`_ens3_m300` and `_ens3_m600` their votes over the 3 seed triples.
`gpu/ner_results_sofc_published_split_crf_mt_mr.json` and
`gpu/ner_results_sofc_slot_published_split_crf_mt_mr.json` are the same on
SOFC-Exp's official split (`--granularity sofc_both`, model seeds 0 to
900). The MaterioMiner decisions are in `gpu/ner_week9_comparisons.json`
(`ner_seedmeans.py --week 9 --gpu`), S2 in
`gpu/ner_week5_seed_analysis.json` (`published_split`, `sofc` and
`sofc_slot`, `crf_mt_mr`), and the per-type changes in its `week6_types`.

Week 9, part 2 (mention replacement on SOFC-Exp at MaterioMiner's size;
Kaggle session `w9b` of 17 October on `ner-gpu-jobs-d`, from bundle 9468ae7):
`gpu/ner_results_sofc_article_draws_crf_m<N>.json` and `_crf_mr_m<N>.json`,
N = 0 to 900, are the CRF head and mention replacement on SOFC-Exp's entity
types, trained on 10 of the 26 official training articles (`--published-split
--train-articles 10`; 5 draws, runs `draw0` to `draw4`, 156–248 training
sentences; the official validation and test articles), with predictions.
`gpu/ner_week5_seed_analysis.json` (`sofc_draws`) holds the decision (paired
by draw, each the mean of its 10 seeds) and the CRF head on 10 articles
against all 26 (`ten_minus_all_articles`, paired by seed).

Week 13, 13.1 (the consistency rule on held-out documents; protocol, amendment
and outcome in ROADMAP §4; no training): `ner_consistency.py`, with and without
`--retype`, on the saved 10-seed predictions of the CRF head and mention
replacement on MuLMS's held-out subfields
(`gpu/ner_results_mulms_leave_one_subfield_out_{crf,crf_mr}_{dc,dcr}_m<N>.json`,
run ids the subfields; CC BY-SA 4.0), of the CRF head on SOFC-Exp's 5 article
folds (`gpu/ner_results_sofc_document_folds_crf_{dc,dcr}_m<N>.json`, run ids
`fold0` to `fold4`) and of mention replacement and its multi-task combination
on MaterioMiner's held-out papers
(`gpu/ner_results_{fine,coarse}_leave_one_paper_out_{crf_mr,crf_mt_mr}_{dc,dcr}_m<N>.json`),
N = 0 to 900; each run keeps the entities added and retyped, the input's F1
and, unlike week 7's files, the new predictions (`--save-predictions`).
`gpu/ner_week5_seed_analysis.json` (`week13`, `consistency`) holds every
output against its input, paired over the subfields, papers or folds, with
its verdict where the comparison is a decision (13.1a, 13.1b, 13.1d, and
week 7's on the CRF head on MaterioMiner, recomputed), and (`week13`,
`pooled`) the pooled estimates across corpora of 13.2 and 13.3
(`week12_winner` names the screen's winner once the screen is complete).
Batches `w13a` and `w13c` (13.3) write
`gpu/ner_results_sofc_document_folds_crf_mr_m<N>.json` (`w13a`,
`ner-gpu-jobs-c` version 5, 24 October, bundle 3ac85a8; in hand, run ids
`fold0` to `fold4`, with predictions) and
`gpu/ner_results_sofc{,_slot}_document_folds_crf_mt_mr_m<N>.json` (`w13c`,
`ner-gpu-jobs-c` version 6, 24–25 October; in hand, a file per level); (`week13`,
`folds`) then holds the decisions 13.3a and 13.3b, each method paired over the
folds against week 8's CRF head on them, and the multi-task combination's slot
level and its difference against mention replacement (reported). Batch `w13b`
(13.4, conditional on a pass of week 12's decision 1 or of 13.2) will write
`gpu/ner_results_sofc_document_folds_crf_mrm6_m<N>.json`, the week-12
winner's analogue on the folds (the SOFC corpus's silver mentions, 6 copies):
(`week13`, `folds`, `crf_mrm6`) its decision 13.4 and (`week13`, `pooled`,
`crf_mrm_topic6`, `with_sofc_folds`) the three-corpus estimate (reported).

Week 11 (mention replacement with silver names; protocol and the outcome of part
`w11a` in ROADMAP §4; batch `w11a` from 427e34c, `w11b` from 77c37be): `w11a`
(Kaggle session of 18 October, `ner-gpu-jobs-d` version 5) wrote
`gpu/ner_results_mulms_leave_one_subfield_out_crf_mrs_m<N>.json` (MuLMS, run ids the
subfields; CC BY-SA 4.0); `w11b` (session of 19–20 October, `ner-gpu-jobs-d`
version 6) wrote `gpu/ner_results_{fine,coarse}_leave_one_paper_out_crf_mrs_m<N>.json`
(MaterioMiner), model seeds 0 to 900, with predictions, their votes
`..._crf_mrs_ens3[_m300|_m600].json` (`ner_ensemble.py`) and
`gpu/ner_week11_comparisons.json` (`ner_seedmeans.py --week 11`: decisions 3 and 4
and the reported comparisons). Each run's `self_training`
holds its 3 teachers' scores and the silver counts (`names`: mentions, distinct names,
and the unseen test entities that are silver names); no name is stored.
`gpu/ner_week5_seed_analysis.json` (`mulms`, `silver_names`) holds the MuLMS
decisions, and `gpu/ner_week11_comparisons.json` will hold the MaterioMiner ones.

Week 12, the MaterioMiner analogue of the screen's winner (`w12d:mrm6`,
`ner-gpu-jobs-d` version 12, 24–25 October, bundle 3ac85a8; outcome in
ROADMAP §4): `gpu/ner_results_{fine,coarse}_leave_one_paper_out_crf_mrm6_m<N>.json`,
N = 0 to 900 (run ids the held-out papers; each run records its teachers,
the mention filter and the silver names' counts), the votes of the three
seed triples `..._crf_mrm6_ens3{,_m300,_m600}.json` (`ner_ensemble.py`), and
`gpu/ner_week12_comparisons.json` (`ner_seedmeans.py --week 12 --gpu`): the
analogue decided against the CRF head per granularity, reported against
mention replacement and with its votes against both votes; V6's analogue
`crf_st3` (week 4's files) is reported there. The pooled decision 13.2 is in
`gpu/ner_week5_seed_analysis.json` (`week13`, `pooled`, `crf_mrm_topic6`).
Week 13, 13.4 (`w13b`, `ner-gpu-jobs-d` version 13, 25 October, bundle
3ac85a8): `gpu/ner_results_sofc_document_folds_crf_mrm6_m<N>.json`, N = 0 to
900, the analogue on SOFC-Exp's 5 article folds with the SOFC corpus (run ids
`fold0` to `fold4`; each run records its teachers, the mention filter and the
silver names' counts); (`week13`, `folds`, `crf_mrm6`) holds decision 13.4 and
(`week13`, `pooled`, `crf_mrm_topic6`, `with_sofc_folds`) the three-corpus
estimate.

Week 12, phase 0 (names from a topic corpus; the gate in ROADMAP §4, no
training): `mulms_articles.json` (`mulms_articles.py`) is the Europe PMC search
for MuLMS's 50 documents by title, 16 found with a PMCID; `topic_corpora/<topic>.json`
(`build_topic_corpus.py`, one per MuLMS subfield) is the report of a corpus of
40,000 sentences of open-access full texts whose titles carry the subfield's
terms (the query, the articles with their licences, every count, the leakage
filters against MuLMS: the 16 PMCIDs, 8-word n-grams and 50-character
shingles, and the corpus's SHA-256; the corpus itself,
`data/processed/corpus/topic_<topic>_sentences.txt`, is not versioned and the
report rebuilds it from the cache); `mulms_coverage.json` (`mulms_coverage.py`)
is the gate: per held-out subfield and corpus, the share of the unseen test
entities (and of their distinct names) whose words occur in a corpus sentence,
exact and ignoring word boundaries and case, per type, with the sentences
that share a shingle with the held-out documents left out (`leaked`, 0
everywhere). The week-12 batches `w12a`, `w12e` and `w12c` (the screen, seven variants
after amendments 1–3; all run on 20 October) and `w12b:mrm6` (the decision for
the screen's winner V4: `crf_mrm_topic6_m<300..900>`, `ner-gpu-jobs-d` version
10, 24 October; in hand), from a bundle rebuilt after the CPU
probe (`ner-gpu-bundle-3`), write
`gpu/ner_results_mulms_leave_one_subfield_out_<variant>_m<N>.json` for the
variants `crf_mrs_topic`, `crf_mrm_topic`, `crf_mrm_topic6`, `crf_st3_topic`,
`crf_gzm_topic` and `crf_gzs_topic` (amendments 2 and 3, batch `w12c`) and
`crf_mr6` (run ids the subfields; CC BY-SA 4.0);
`gpu/ner_week5_seed_analysis.json` (`mulms`, `week12`) holds the screen
and the decisions. In hand (20 October): `crf_gzm_topic_m<N>` and
`crf_gzs_topic_m<N>` (the feature screen, `ner-gpu-jobs-c` version 4, bundle
3ac85a8) and `crf_mr6_m<N>` (the first `w12a` session, `ner-gpu-jobs-d`
version 7, bundle 47e2310), and `crf_mrs_topic_m<N>`, `crf_mrm_topic_m<N>`,
`crf_mrm_topic6_m<N>` and `crf_st3_topic_m<N>` (`w12e`, `ner-gpu-jobs-d`
version 8, bundle 3ac85a8; the screen is complete), N in 0, 100, 200; each
run of a topic variant
records its corpus (`self_train_sha256`), its teachers and, for the feature
variants, `names_feature` (the lexicon's size, the matches and the largest
learned weight). `silver_lexicons/<subfield>.json` (`silver_lexicon.py`,
the CPU probe of `docs/METHODS_PLAN.md` §4B, no decision; all 7 subfields,
19–20 October) records per subfield the lexicons' sizes and their coverage of the unseen test names
under both silver filters at three thresholds, and the dictionary rule's
effect on week 10's saved predictions with the gold-name ceilings; the names
themselves stay local.

Week 10 (MuLMS, a third corpus; protocol and outcome in ROADMAP §4; Kaggle
session `w10a` of 17–18 October, `ner-gpu-jobs-d` version 4, bundle a08356f):
`mulms_corpus.json` is the report of `src/graph/mulms.py`. It records the
Hugging Face revision of MuLMS, the SHA-256 of its parquet files and of every
converted file, the flattening rule (the outermost entities), the token rule
(whitespace-separated pieces, without those a BERT tokenizer erases) and every
count.
The converted CoNLL files stay in `data/processed/mulms/`, outside git, and
`mulms.py` rebuilds them; the Kaggle bundle ships them. The runs wrote
`gpu/ner_results_mulms_leave_one_subfield_out_crf_m<N>.json` (the CRF head)
and `_crf_mr_m<N>.json` (mention replacement), run ids the 7 subfields, and
`gpu/ner_results_mulms_random_split_pool_crf_m<N>.json` (runs `seed0` to
`seed4`), model seeds 0 to 900, with predictions. `gpu/ner_week5_seed_analysis.json`
(`mulms`) holds both decisions, the votes and the gap; `gpu/ner_mentions.json`
and `gpu/ner_matching.json` hold MuLMS's unseen names and looser matching
(settings `mulms_leave_one_subfield_out` and `mulms_random_split_pool`). *Licence:*
MuLMS is CC BY-SA 4.0 (Schrader et al., 2023), so every results file with
`mulms` in its name, which holds MuLMS's tags (`test_gold`), is CC BY-SA 4.0
as well, unlike the rest of the repository (Apache 2.0). Since 31 October it also holds `chain`: each system's paired per-unit differences of recall, precision and F1 on the unseen names against the system it builds on (MR3 against the CRF head, MR6 against MR3, the names and the mixed variant against MR6), on the seeds both have and, under `decision`, on the decision seeds (post hoc; paper A, Section 5.4), and every `methods` entry carries `recall` and `precision` beside its F1.

Week 7, part 2e (SOFC-Exp's test articles cut into words as its training
articles are; Kaggle session `w7e` of 17 October, `ner-gpu-jobs-d` version 3,
bundle 4f5dbda): `gpu/ner_results_sofc_published_split_cdetest.json` (plain
MatSciBERT) and `_crf_cdetest.json` (the CRF head), model seeds 0 to 900 on
the retokenized test articles (`sofc_exp.py --retokenized-test`), with
predictions. Every run trained exactly as its run in
`ner_results_sofc_published_split.json` or `_crf.json`, so the pairs by seed
differ only in the test words. The decisions are in ROADMAP §4 (outcome of
part 2e), and the seen and unseen names in `gpu/ner_mentions.json`.

Week 7, part 2d (multi-task training on SOFC-Exp's two levels; Kaggle
session `w7b` of 17 October, model seeds 0 to 900 on the official split):
`gpu/ner_results_sofc_slot_published_split.json` and `_crf.json` are plain
MatSciBERT and the CRF head on the 17 slot types
(`data/raw/dataset/sofc_exp_slot/`), and
`gpu/ner_results_sofc_slot_published_split_crf_mt.json` and
`gpu/ner_results_sofc_published_split_crf_mt.json` the multi-task model's
slot and entity heads (`--granularity sofc_both --crf --multi-task`, one
file per level). `gpu/ner_week5_seed_analysis.json` (`published_split`,
`sofc_slot` and `sofc`) holds the decisions, the votes over the seed
triples and, per published split, each system's mean F1 per type over its
seeds (`types`).

`gpu/ner_week5_seed_analysis.json` also holds `paper_count` (claim C19): per
held-out paper, the CRF head's F1 trained on two papers (the inner folds'
counts at D = 0) and on three (leave-one-paper-out), seeds 0 to 200, and
their difference over the 4 papers; and `sofc_splits` (week 8, part 1): the CRF
head on SOFC-Exp's 5 random sentence splits
(`gpu/ner_results_sofc_random_split_vary_crf_m<N>.json`) and 5 folds of 9
held-out articles (`gpu/ner_results_sofc_document_folds_crf_m<N>.json`), N =
0 to 900, with predictions: per protocol the mean and CI over the splits of
every F1 average, the CRF votes over the seed triples, and the gap (random
splits minus article folds, +3.48 micro). `gpu/ner_mentions.json` and
`gpu/ner_matching.json` hold both protocols as SOFC-Exp settings.

Week 8, part 2 (more papers or more sentences; Kaggle session `w8b` of 17
October, `ner-gpu-jobs-c` version 2, bundle 04352cd):
`gpu/ner_results_{fine,coarse}_leave_one_paper_out_crf_p3n2_m{0,100,200}.json`.
Each held-out paper's leave-one-paper-out fold is trained on all three other
papers, its training sentences cut to as many as each pair of them gives (run
ids `P|Q+R`, 12 per granularity), with predictions.
`gpu/ner_week5_seed_analysis.json` (`paper_diversity`) holds per granularity
the F1 with two papers, three at the size of two and three papers, the 2
decisions (diversity) and the size difference, paired over the 4 papers.

Seen and unseen test names (exploratory, no training; claim C17 of
`docs/PAPER.md`): `gpu/ner_mentions.json` (`ner_mentions.py`) splits the gold
test entities of every stored run with predictions by whether their exact
words are an entity of the same type in the run's training sentences (seen),
only of another type, or none (unseen), each run's split rebuilt as
`ner_baseline.py` builds it. A prediction takes a kind the same way, so
recall, precision and F1 are given per kind (pooled over seeds and splits):
- `systems`: every family with all its files (10 seeds, or the screen's 3)
  and its votes (made again from the members' predictions) on both
  protocols, and plain MatSciBERT, the CRF head and the methods on the
  published splits (MaterioMiner and SOFC-Exp);
- `papers`, `types`: each held-out paper's shares of the kinds, and per type
  the CRF head's unseen share and its recall on seen and unseen entities;
- `gap`: the CRF head's recall on random splits and held-out papers, and the
  two counterfactual mixes of kinds;
- `methods`: each method against the CRF head (a vote against its vote),
  F1 per kind, paired over splits, papers or seeds, mean and 95% CI (no
  decision). Its F1 over all kinds equals the stored micro F1 of every run.
- `corpus_coverage`: per granularity and held-out paper, the unseen gold test
  mentions and the share of them, and of their distinct names, that occur in
  a sentence of the fatigue corpus of self-training (cut into words as
  self-training cuts it), with the corpus's SHA-256; written only where the
  corpus (not versioned) is present.

Figure 5 of the paper plan (`docs/figures/fig5_mentions.png`,
`paper_figures.py`) draws it.

Looser matching (exploratory, no training; claim C18): `gpu/ner_matching.json`
(`ner_matching.py`) scores the same predictions under the four criteria of
SemEval-2013 Task 9.1: strict (span and type; equal to the stored micro F1),
exact (span, any type), type (overlap, same type) and overlap (any overlap,
any type), each gold entity matched at most once. `systems` holds the mean
and 95% CI over splits, papers or seeds for the CRF head, the multi-task
model and their votes on both protocols, and for plain MatSciBERT and the
CRF head on the published splits; `gap` the random-split minus held-out
difference per criterion; `near_misses` the shares of the CRF head's
predictions that are right, of another type (fine-grained, also of an MMO
ancestor or descendant type), inside a longer gold entity, around a shorter
one, crossing one, or spurious, on both protocols and the published splits,
with the most frequent pairs of gold and predicted type among those of the
right span (`wrong_type_pairs`, per run); `conventions` what the gold annotations alone
show: the names that are entities in two or more papers, those with
different types, and the mean span length of each type per paper.

Week 14 (the names method's ablations; protocol in ROADMAP §4, week 14):
`topic_corpora/mixed_<subfield>.json` (`build_mixed_corpus.py`) is the report of
the mixed corpus of a held-out subfield, 40,000 sentences drawn from the six
other subfields' topic corpora (6,667 or 6,666 each, a seeded shuffle; the
sentences of the subfield's own corpus and those already taken skipped, 17–69
per corpus), with the sources' SHA-256, the counts and the corpus's SHA-256
(the corpus, `data/processed/corpus/topic_mixed_<subfield>_sentences.txt`, is
not versioned and the report rebuilds it from the topic corpora);
`mulms_coverage_mixed.json` (`mulms_coverage.py` with the seven mixed corpora)
is their coverage of the unseen test names by phase 0's rule: 49–70% exact
(57–74% loose) against the topic corpora's 55–81% (63–84%), measured before
any run of 14.2. The week-14 batches: `w14a` (14.1, the 6-copy control
`crf_mr6` on MuLMS's held-out subfields at seeds 300–900 and on MaterioMiner's
held-out papers at 10 seeds, both granularities; `ner-gpu-jobs-c` version 7,
29 October, 12:05–17:20 UTC, 5.2 session hours: the 27 files `mulms_leave_one_subfield_out_crf_mr6_m<300..900>` and
`{fine,coarse}_leave_one_paper_out_crf_mr6_m<N>`, with the control's seed votes
`crf_mr6_ens3`, `_ens3_m300` and `_ens3_m600` per granularity; 14.1 no, +0.80
[−0.66, +2.26] pooled), `w14b` (14.2, `crf_mrm_mixed6_m<300..900>`; `ner-gpu-jobs-d` version 14, 29 October
18:00–22:20 UTC, 4.3 session hours, pushed early on the week's remaining quota; 14.2 no, −0.18 [−2.01, +1.65]) and `w14c` (14.3,
`crf_mrm_topic10k6`, `crf_mrm_topic20k6`, `crf_mrm_topiclow6`,
`crf_mrm_topictwo6` and `crf_mrm_topiccap6` at seeds 0–200; `ner-gpu-jobs-c`
version 8, 31 October 00:14 to about 09:55 UTC, 9.7 session hours, the smoke
repeat identical, `check_w14c.py` all checks pass), the last after the reset of
31 October; their analyses in `gpu/ner_week5_seed_analysis.json` (`week14`:
`copies`, `topic`, `ablations`), Table 6 and Table 7
(`docs/tables/table7_ablations.md`, the ablations against the method).
`gpu/ner_robustness.json` (`ner_robustness.py`, 31 October, post hoc, for
paper A's first review round; Table 9, `docs/tables/table9_robustness.md`)
holds every comparison among the CRF head, mention replacement with 3 and
6 copies and the names method over the 11 units of 13.2, with MuLMS's
units as 10-seed means (as decided) and as means over seeds 300–900: the
paired intervals, the units above 0, exact sign-flip permutation p-values,
leave-one-unit-out intervals, the stratified mean, minimum detectable
effects at 80% power (`detectable`, per comparison), the MaterioMiner
grids at both granularities (the decisions' stored intervals), the
per-subfield coverage–gain correlation with its partial correlations and
the leave-two-out check, the stratified mean, and per system on MuLMS's
decision seeds the runtime, the best epoch (a 0-based index in the run
files, and the count of epochs up to it), the mean count of augmented
sentences and, against the CRF head, the ratio of sentences per epoch
and of updates up to the best epoch (`compute`; the names system's job
costs under `job_costs`). `pooled_measures` (added 1 November, post hoc,
for paper A's review round 12) recomputes the same pooled comparisons on
both seed bases from the run files in micro, macro and weighted F1; its
micro values equal `pooled`'s to within 0.01, and its macro and weighted
values fill the pooled rows of Table 13 (Supplementary Table S11). In
macro F1 the names against MR6 are +0.83 [−0.01, +1.67] on the decision
seeds and +1.12 [+0.21, +2.03] on all ten.
`gpu/ner_decision_tests.json` (`decision_tests.py`, 1 November, for the
journal's reporting rules; Table 16, `docs/tables/table16_decision_tests.md`,
paper A's Table S16 and Table 2's P column) gives, for each of paper A's
twelve decisions and D5′, the one-sample t test of the per-unit
differences the decision used (n, mean, SD, t, degrees of freedom, one-
and two-sided P) with the exact sign-flip P, and checks that the
recomputed interval equals the stored one (to 0.01) and that the one-sided
P is below 0.025 exactly where the interval clears 0; it changes no
decision. Each row also carries its verdict words (D5′ "above 0 (post
hoc)", D11 "passes (as registered)"). Post hoc (round 15 of paper A's
review): `post_hoc_dependence` repeats each test with the units correlated
where their training sets share documents (1/4 between MaterioMiner's
papers, 1/5 between SOFC-Exp's folds, 1/4 between the 5 MuLMS subfield
pairs whose training pools share an article, `mulms_pools`), the
correlation the corrected resampled t test assumes for k-fold designs;
`post_hoc_reported` gives the t test of MR3 over the CRF head on the 11
pooled units, both seed bases (P 0.0038 and 0.0040), and (round 16) the
P values of the reported comparisons the paper draws inferences from
(`inferences`; from per-unit values, or from the interval where the
results store none), the companion study's MR3 decisions on MaterioMiner
on the seeds its screen did not use (`mr3_materiominer_decision_seeds`,
with `mr3_materiominer_all_seeds` reproducing the decisions) and D5 with
the coarse-grained papers (`d5_coarse`); `post_hoc_dependence.tipping`
gives the correlation at which each pass's adjusted interval reaches 0
(round 17: found on the unrounded lower bound, which moved D5's from 0.44
to 0.45; `inferences` grew to 21 entries, adding week 18's readings, the
types at D = 0, the self-training votes against MR3's and multi-task
training's gains on MaterioMiner, the last from
`ner_week5_comparisons.json`; round 18 brought it to 41, adding the
comparisons the paper reads as not shown and those outside its Table 2,
F1 on the held names from `paper_tables.held_f1_gain`).
Table 10
(`table10_lexicon.md`, paper A's Table S3 and Table 5) adds, post hoc, F1
on each lexicon part and the paired F1 gain on the held names, from
`ner_lexicon_recall.json`'s pooled and per-unit recall and precision.
`gpu/ner_week5_seed_analysis.json` adds `week14.types`, the per-type F1 of
the names method and the 6-copy control against the CRF head on MuLMS's
decision seeds and MaterioMiner's papers (Tables 8 and A2,
`docs/tables/table8_types.md`, `tableA2_types_fine.md`).

Week 15 (the week-15 analysis note, ROADMAP §4; post hoc, no decision):
batch `w15a` rebuilds the lexicons of the names method with no students
(`ner_baseline.py --students 0 --save-names`): V4's on MuLMS's 7 held-out
subfields (the CRF head's teachers of seeds 0, 100 and 200, the topic corpus,
the mention filter at 0.9) and its analogue's on MaterioMiner's 4 papers at
both granularities (the fatigue corpus). The names files
(`ner_names_<granularity>_<protocol>_<variant>[_<subfield>].json`: the names
with their counts, the teachers' scores, the coverage counts) are pulled to
`results/gpu_names/`, which git ignores (the harvested names are never
committed). `gpu/ner_lexicon_recall.json` (`ner_lexicon_recall.py`) holds the
aggregate analysis: per unit and system, recall on the unseen test names the
lexicon holds and on those it does not, the unseen predictions' lexicon share
and precision, the paired gains over MR6 and the CRF head on each part and
their difference with 95% CIs, and the checks that the rebuilt teachers' F1
and coverage counts equal the stored ones. Run record: `w15a` pushed to
`ner-gpu-jobs-c` (version 9, bundle `2c60f33`) on 31 October at 16:30 UTC;
pulled at 19:41 (16 jobs done; the smoke repeat identical); every unit's
rebuilt teachers and coverage counts equal the stored runs'. Outcome (Table
10, paper A's Table 5 and, in full, Supplementary Table S3): on MuLMS the names' recall gain over MR6 is +7.29
[+3.27, +11.30] on the unseen names the lexicon holds and +1.12 [−0.78,
+3.02] on the others (difference +6.17 [+2.44, +9.90], 7 of 7 subfields):
mixed on point estimates and name knowledge on intervals, the reading
fixed in advance not saying which (paper A's round-9 revision, 1
November); on MaterioMiner a generic shift on point estimates, and none of
the three outcomes on intervals, within four papers' wide intervals
(fine +4.33 [−5.33, +13.98], coarse −1.54 [−3.60, +0.51]); the wrong
predictions the names add fall on both parts (two thirds on names the
lexicon does not hold, 22% right).

Week 17 (the week-17 protocol, ROADMAP §4; fixed 31 October 19:15 UTC
before any run): the harvested names on the strongest MaterioMiner base,
the multi-task model with 6-copy mention replacement: `crf_mt_mr6_m<N>`
(the matched control) and `crf_mt_mrm6_m<N>` (three multi-task teachers,
the fatigue corpus tagged with both heads, the mention filter on both
heads, 10 students), 10 seeds on the 4 held-out papers, a fine and a
coarse file per seed offset. Decisions 17.1 (fine) and 17.2 (coarse)
against the control; the comparisons with `crf_mt_mr`, `crf_mt` and the
CRF head reported (`ner_week5_seed_analysis.json`, `week17`). Run record:
`w17a` pushed to `ner-gpu-jobs-d` (version 15, bundle `013f9ac`) at
19:54:50 UTC and `w17b` to `ner-gpu-jobs-c` (version 10, the same bundle)
at 19:54:53 UTC on 31 October; the two sessions write the student files
for different papers, united by `kaggle_jobs.py collect --merge`. Outcome
(1 November): 17.1 fine-grained +0.96 [−0.01, +1.92], no (every paper
positive); 17.2 coarse-grained +0.85 [−0.72, +2.42], no; the names
against multi-task + MR3 +0.76 [−0.15, +1.68] and +0.91 [−0.14, +1.97],
the control against it −0.19 [−2.05, +1.67] and +0.06 [−0.74, +0.86];
levels 53.95/52.99 and 61.25/60.40; the two-head filter's lexicons hold
6.5–16.3% (fine) and 6.7–24.6% (coarse) of a paper's unseen test names
(`docs/tables/table11_week17.md`; the unseen-name trade in
`gpu/ner_mentions.json`, `chain`; the detectable differences in
`gpu/ner_robustness.json`). The `w17a` session's repeats (`smokebc`,
`smokebd`) identical to the stored runs (`w17b` ran none); the teachers equal the stored `crf_mt` runs.

Week 18 (the week-18 protocol, ROADMAP §4; fixed 1 November 10:51 UTC
before any run): the names against a recall-matched control on MuLMS's 7
held-out subfields at the decision seeds 300 to 900. `crf_mr6_bias_m<N>`
are the stored `crf_mr6` jobs run again with `--entity-bias` over the grid
−2 to 5 (training untouched; the test predictions per bias in
`test_pred_bias`), one file per seed offset holding the 7 subfields.
`gpu/ner_recall_matched.json` (`ner_recall_matched.py`, committed before
the result) holds the checks, per subfield the matched bias D* with its
flag and the control's recall, precision and F1 at every bias, the
decisions 18.1 (recall on the unseen names the lexicon holds) and 18.2
(micro F1) with the pre-specified comparisons, the same without the
flagged subfields (`without_flagged`), and a post hoc reading with the
flagged subfields matched at the nearest recall (`post_hoc_nearest`,
added after the result). Run record: `w18a` pushed to `ner-gpu-jobs-d`
(version 16, bundle `7906996`) at 11:12:34 UTC on 1 November, pulled at
13:58; `smokebe` identical to `smokebc`; at D = 0 all 49 runs equal their
stored `crf_mr6` runs. Outcome: 18.1 +11.24 [+3.31, +19.18], passes;
18.2 +3.26 [−0.79, +7.31], no; electrolysis, pemfc and sofc flagged
(without them +6.05 [−3.33, +15.44] and −0.03 [−2.78, +2.72]; post hoc
+8.63 [+3.35, +13.90] and +1.37 [−0.66, +3.41]); `docs/tables/table12_week18.md`.

Week 19 (the week-19 protocol, ROADMAP §4; fixed 3 November 00:32 UTC
before any run): graph-propagated names at matched lexicon size on the 11
units of D7. `week19/graph_names_probe.json` (`graph_names.py`, the probe;
counts and rates only) holds per unit the graph's size and homophily, the
vote grid, the ceiling, every ranker's coverage and wrong-type rate at both
sizes, the teachers and the checks, then the gate and the SHA-256 of the
Phase-1 lexicon files (git-ignored, like the candidate dumps of batch `w19a`
in `data/processed/week19/`); `graph_names.py --summary` prints the means and
pooled rates from its counts. Run record: `w19a` from the bundle of
`c449864` (kernel version 19), pushed 01:25 UTC and completed 03:51 UTC;
`smokebf` identical to `smokebe`; the teachers and the rebuilt names equal
the stored ones on every unit. Outcome: the gate fails (G 16.78%, V 24.86%,
M 18.80% coverage; wrong-type 4.65% against 10.14%), so Phase 1 (`w19b`) was
not run.

Week 20 (the week-20 protocol, ROADMAP §4; fixed 3 November 07:00 UTC before
any run): two controls for paper B's first referee. In `gpu/`:
- `crf_spool_m<N>`: MuLMS's held-out subfields with the 400-sentence pool
  drawn by sentence (`--pool-by-sentence`), seeds 0-900;
- `crf_p3n2_m<N>` and `crf_bias_m<N>` at seed offsets 300-900: week 8's
  two sides, three papers at the size of two and the two papers alone
  (the inner folds read at D = 0).

The analysis is the `week20` section of `gpu/ner_week5_seed_analysis.json`
(repeats, `spool`, `diversity`), and the `spool` section of
`gpu/ner_mentions.json` gives seen and unseen recall. `gpu_repeats/` holds
the versioned repeat files that the analysis reads by default: `smokebi`,
`smokebj` and `smokebk`, each identical to its stored run but for
`runtime_sec`. Run record: `w20a` from the bundle of `393b482` (kernel
version 20), pushed 08:38 UTC and completed 12:25 UTC; `smokebh` is
identical to `smokebf`. Outcome:
- 20.1: +9.51 [+1.12, +17.91], passes; the remaining gap is +7.55 (Welch
  [+4.05, +11.05]);
- 20.2: +4.03 [+1.06, +7.00], passes;
- 20.3: +3.52 [+1.32, +5.72], passes.

Paper B's post hoc checks (review round 1, 3 November, extended to weeks
20 and 21 on 4 November; none decides anything): `gpu/paper_b_checks.json`
(`paper_b_checks.py`) enumerates the 118 pre-registered decisions with
their per-unit differences. It checks that each stored interval
reproduces, that the totals match paper B's §7.8 by design and by
addition, and that weeks 20 and 21's dependence entries equal the readings
their protocols fixed. Per decision it gives:
- the one-sided P;
- the 80%-power detectable difference;
- the interval allowing for units that share training documents (paper
  A's model, `decision_tests.dependent_t`), or the corrected resampled
  test of Nadeau and Bengio on random splits.

Over the decisions it gives Benjamini–Hochberg q-values and Holm within
each addition's family. It also gives:
- week 3 and 4's four 3-seed passes recomputed on the other machine at the
  same seeds;
- matched CPU–GPU runs;
- each corpus's gap with a Welch interval.

It notes that unseen-minus-seen gains cannot be computed from the stored
summaries. Over the 118 decisions:
- the dependence allowance keeps 6 of the 18 passes over units with 7 or
  10 seeds (20.3 among them);
- Benjamini–Hochberg passes 3 at FDR 5% and all 27 passes at 10%, since
  the largest of their P values, 0.0223, is just under its threshold,
  0.0229 (12 of the 111 of weeks 2 to 18 passed at 10%);
- Holm within each family keeps 9: the 6 of weeks 2 to 18 and week 20's 3;
- 2.95 passes are expected by chance, 10.9% of the 27.

Week 21 (the week-21 protocol, ROADMAP §4; fixed 3 November 09:39 UTC
before any run): a free generalist NER baseline, GLiNER-medium
(`gliner_baseline.py`), on the CRF head's held-out units. In `gpu/`:
- `gliner_ft_m<N>`: fine-tuned, seed offsets 0-900, on MaterioMiner's 4
  papers at both levels, MuLMS's 7 subfields and SOFC-Exp's 5 folds;
- `gliner_zs_m0`: zero-shot, one run per unit, on MaterioMiner fine,
  MuLMS and SOFC-Exp (no coarse arm).

The analysis is the `week21` section of `gpu/ner_week5_seed_analysis.json`
(checks, repeats, decisions, `corpora`, `zero_shot`), and the `week21`
section of `gpu/ner_mentions.json` gives seen and unseen recall.
`gpu_repeats/` holds `smokebm`, identical to the paper-1 run of
`gliner_ft_m0` but for `runtime_sec`. Run record: `w21s` (kernel version
21, bundle of `ea8ac82`); `w21a` over two sessions, kernel version 22
(bundle of `c8df5c4`; the deadline left the 3 zero-shot jobs) and 23
(bundle of `3138ee5`); `smokebl` and both sessions' `smokebn` are
identical to `smokebh`. The second session rewrote 3 finished files with a
summary mean rounded the other way (51.055); their runs are identical, and
the first session's files are kept. Outcome (fine-tuned minus the CRF
head, 10 seeds):
- 21.1, MaterioMiner fine: +3.77 [−5.38, +12.92], not shown;
- 21.2, coarse: −10.12 [−15.68, −4.56], below the CRF head;
- 21.3, MuLMS: +1.58 [−1.92, +5.08], not shown;
- 21.4, SOFC-Exp folds: +0.27 [−1.84, +2.37], not shown.

Zero-shot minus the CRF head (reported): −19.87 [−30.73, −9.00]
(MaterioMiner fine), −32.44 [−38.71, −26.17] (MuLMS), −47.78 [−48.85,
−46.70] (SOFC-Exp).

Week 21's exploratory checks (batch `w21x`; ROADMAP §4, fixed 4 November
before the run; outside the count, no decision). In `gpu/`:
- `gliner_ftn_m<N>`: E1, `gliner_ft`'s recipe on MaterioMiner
  coarse-grained with each code given as its class name (`COARSE_NAMES`),
  seed offsets 0-900;
- `gliner_ftp_m<N>`: E2, the package's trainer (`gliner.training`,
  accelerate 1.15.0) with the same recipe, stopped after each seed's best
  epoch in `gliner_ft`, on MaterioMiner fine-grained paper 1, seed offsets
  0-900.

The analysis is the `week21x` section of `gpu/ner_week5_seed_analysis.json`.
Run record: kernel version 24 (bundle of `c764023`); `smokebo` is
identical to `smokebn`, and `smokebp` to E2's seed-0 run (not collected).
Outcome:
- E1: 54.64, minus the CRF head −3.97 [−9.15, +1.22] and minus the codes
  +6.16 [−0.35, +12.67] (higher on every paper);
- E2: 52.38 against our loop's 52.64, −0.25 [−3.93, +3.42] over 10 seed
  pairs.

The SOTA plan's P0, reproduction gates (batch `p0a`; ROADMAP §4, fixed 4
November before the run; no claim and no decision of paper B's count). In
`gpu/`:
- `sofc_matscibert_cv_crf_m<N>`: the CRF head on MatSciBERT's 5
  cross-validation folds of SOFC-Exp (`--sofc-cv`; the folds are in
  `sota/sofc_cv_folds.json`), runs `cv1` to `cv5`, seed offsets 0, 100
  and 200;
- `mulms_nested_configs_bilou_crf_c<k>`: `mulms_nested.py`, MuLMS's
  released tagger over every nested entity, configuration k = 1-5 at seed
  0 (the configurations are in `sota/mulms_configurations.json`);
- `sota_p0.json`: the three gates, from `python src/graph/sota_p0.py
  gates`.

`gpu_repeats/` holds `smokebr` and `smokebs`, two short MuLMS runs on the
two GPUs, identical but for `runtime_sec` and their tags. Run record:
kernel version 25 (bundle of `d0aa043`). `smokebq` is identical to
`smokebn` but for its tag (not collected). Outcome, our mean minus the
published one; every gate passed:
- SOFC-Exp cross-validation: 84.16 against 84.77 (Gupta et al. 2022),
  −0.61 [−1.28, +0.07];
- SOFC-Exp's official split on 10 articles: 84.34 against 83.4 (Schrader
  et al. 2023), +0.94 [−0.41, +2.30];
- MuLMS, every nested entity: 79.04 against 79.6 (Schrader et al. 2023),
  −0.56 [−1.41, +0.29].

The SOTA plan's P1, screens (ROADMAP §4, fixed 7 November before any P1
run; validation scores only, no claim and no decision):
- `sota/mm_screen_splits.json` and `sota/shared_documents.json`: the
  hashed data files (MaterioMiner's screen splits; the four documents the
  corpora share);
- `gpu_screen/`: the screen files, kept apart from every stored analysis.
  Batch `p1t`'s 2-epoch timing runs carry the tags `p1t<kind>`, and its A0
  reference runs carry `p1a0` (SOFC-Exp) and `p1a0u1` (MuLMS's c1);
- `sota/p1_timing.json`: the timing table and the budget's cuts, from
  `python src/graph/sota_p1.py timing`;
- `sota/p1t_dapt_b3.json`: B3's pre-training report, without the text of
  the sentences it dropped;
- `sota/p1_next_jobs.json`: the cells to run next, from `sota_p1.py
  next-jobs`.

Run record of `p1t`: kernel version 26 (bundle of `4ca09ee`). Every check
passed:
- the smoke4 repeat `smokebt` equals `smokebn` (not collected);
- A0 on SOFC-Exp in screen mode equals the stored `val_f1`;
- MuLMS's c1 in screen mode equals P0's dev score.

The table's cuts are: A2m and C3 leave MuLMS, and MaterioMiner goes to 5
units, for 21.19 session-hours of the 24-hour cap.

Round 1 (batch `p1n`, kernel version 27, bundle of `e559980`; every job
done):
- its screen files are in `gpu_screen/`, and its table is
  `sota/p1_round1.json`, from `sota_p1.py select`;
- A6's three files are in `gpu_screen_void/`. A bug made A6 run as A0, and
  the bug is fixed (ROADMAP §4). A6 runs again in the next stage.

Stage 2 (batch `p1n`, kernel version 28, bundle of `92078e6`; every job
done): A6 runs again after its fix, and MuLMS's extra units run.
`sota/p1_stage2.json` is the table. MuLMS's selection is A1m (validation
only).

Stage 3 (round 2's first cells; batch `p1n`, kernel version 29, bundle of
`75b3970`): `sota/p1_stage3.json` is the table.

Stage 4 (batch `p1n`, kernel version 30, bundle of `84c8186`):
`sota/p1_stage4.json` is the table. Coarse-grained's selection is A3 with A6,
and SOFC-Exp's is B1 (validation only).

Stage 5 (batch `p1n`, kernel version 31, bundle of `e99e300`) holds fine's
extra units. **P1's selections** are in `gpu/sota_p1.json`, from `python
src/graph/sota_p1.py select` (validation only, no claim):
- MaterioMiner fine: C1;
- coarse: A3 with A6;
- SOFC-Exp: B1;
- MuLMS: A1m.
