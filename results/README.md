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
