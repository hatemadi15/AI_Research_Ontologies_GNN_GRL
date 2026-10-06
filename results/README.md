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
| `ner_comparisons.json` | `ner_baseline.py --compare <results dir>` | each variant minus its baseline, paired by split and model seed: per-run differences, their mean and 95% CI for micro, strict micro, macro and weighted F1 |

`ner_results_fine.json` and `ner_results_coarse.json` hold all 5 model seeds
per granularity (see `runs`), trained on one fixed random split (`--split-seed
0`). Their `summary` gives, for micro, strict micro, macro and
support-weighted F1, the mean, the population SD and the 95% Student-t CI over
runs. Files with a `_random_split_vary` suffix use a new random split per
seed; `_leave_one_paper_out` files train on three papers and test on the
fourth. A final `_cw` (class-weighted loss) or `_crf` (CRF head) marks a
training variant; files without one are the plain baseline.
