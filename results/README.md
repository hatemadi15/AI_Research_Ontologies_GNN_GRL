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
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_{scibert,pubmedbert}.json` | `ner_baseline.py --crf --save-predictions --tag scibert --model allenai/scibert_scivocab_uncased` (`--tag pubmedbert --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) with `--vary-split` or `--leave-one-paper-out` | week 3: the CRF head on a SciBERT or PubMedBERT encoder, one model seed per split (K on split K, 0 on each held-out paper), with test predictions for the mixed vote; random splits done, leave one paper out in progress |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_mix3.json` | `ner_ensemble.py --variant crf_mix3 --members <the crf_m0, crf_scibert and crf_pubmedbert files>` | week 3: entity-level majority vote of three backbones (MatSciBERT `crf_m0`, SciBERT, PubMedBERT); `model` is `mixed` and `config.models` lists the members' models; random splits done, leave one paper out follows its members |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_mt_{m100,m200}.json` | `ner_baseline.py --crf --multi-task --tag m<N> --seed-offset <N>` with `--vary-split` or `--leave-one-paper-out` | week 3: multi-task with model seeds K + N; with `crf_mt` (offset 0) the three seeds of the multi-task family |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_ls_{m0,m100,m200}.json` | the label-semantics variant of `ner_baseline.py --crf` (week-3 protocol, variant 2), seed offsets 0, 100 and 200, with `--vary-split` or `--leave-one-paper-out` | week 3: emissions scored against embeddings of the MMO label descriptions, three model seeds per split and held-out paper; leave one paper out done, random splits in progress |
| `ner_results_{fine,coarse}_leave_one_paper_out_crf_st_{m0,m100,m200}.json` | the self-training variant of `ner_baseline.py --crf --leave-one-paper-out` (week-3 protocol, variant 3), seed offsets 0, 100 and 200 | week 3: a student trained on gold plus silver-tagged in-domain sentences, three model seeds per held-out paper (no random splits); each run's `self_training` gives the sampled and silver sentence counts and the teacher's validation and test F1 |
| `ner_results_{fine,coarse}_{random_split_vary,leave_one_paper_out}_crf_dapt2_{m0,m100,m200}.json` | `ner_baseline.py --crf --model <in-domain checkpoint> --tag dapt2_m<N> --seed-offset <N>` with `--vary-split` or `--leave-one-paper-out` | week 3: the CRF head fine-tuned from MatSciBERT further pre-trained on open-access fatigue papers (variant 5), three model seeds per split and held-out paper |
| `dapt_mlm.json` | `dapt_mlm.py` | domain-adaptive pre-training of MatSciBERT on the PubMed sentences: corpus filtering (duplicates, sentences sharing a word 8-gram with MaterioMiner), settings, held-out MLM perplexity before training and after each epoch, training loss per step |
| `dapt_mlm_fatigue.json` | `dapt_mlm.py --corpus data/processed/corpus/fatigue_sentences.txt --out data/processed/dapt/matscibert_fatigue` | week 3 (variant 5): the same pre-training of MatSciBERT on the in-domain fatigue corpus (`fatigue_corpus.json`), with the same filters, settings and report fields; the starting checkpoint of the `_crf_dapt2_m<N>` runs |
| `fatigue_corpus.json` | `build_fatigue_corpus.py` | the in-domain pre-training corpus of the week-3 protocol (variant 5): the Europe PMC query and filters, the counts at every stage, the article IDs (PMCID, DOI) and the SHA-256 of the 40,000 sentences (`data/processed/corpus/fatigue_sentences.txt`, not versioned; `--offline` rebuilds it from the cached XML) |
| `ner_comparisons.json` | `ner_baseline.py --compare <results dir>` | each variant minus its reference (`reference`: the baseline for `cw` and `crf`, the CRF for `crf_*` variants), paired by split and model seed: per-run differences, their mean and 95% CI for micro, strict micro, macro and weighted F1 |
| `ner_week3_comparisons.json` | `ner_seedmeans.py` | week-3 comparisons of 3-seed means (ROADMAP §4): per split, a variant minus its reference, where a family of three seeds scores their mean (`crf` = `crf_m0`, `crf_m100`, `crf_m200`) and an ensemble scores once; each entry gives its `role` (`decision` or `reported`), whether all 5 splits or 4 papers are paired (`complete`), the per-split scores of both sides, and the per-run differences with their mean and 95% CI for micro, strict micro, macro and weighted F1. Comparisons are added as their runs finish |

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
three encoders.
