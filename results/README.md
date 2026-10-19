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

Week 11 (mention replacement with silver names; protocol and the outcome of part
`w11a` in ROADMAP §4; batches `w11a` and `w11b` from 427e34c): `w11a` (Kaggle
session of 18 October, `ner-gpu-jobs-d` version 5) wrote
`gpu/ner_results_mulms_leave_one_subfield_out_crf_mrs_m<N>.json` (MuLMS, run ids the
subfields; CC BY-SA 4.0); `w11b`, after 24 October, will write
`gpu/ner_results_{fine,coarse}_leave_one_paper_out_crf_mrs_m<N>.json`
(MaterioMiner), model seeds 0 to 900, with predictions. Each run's `self_training`
holds its 3 teachers' scores and the silver counts (`names`: mentions, distinct names,
and the unseen test entities that are silver names); no name is stored.
`gpu/ner_week5_seed_analysis.json` (`mulms`, `silver_names`) holds the MuLMS
decisions, and `gpu/ner_week11_comparisons.json` will hold the MaterioMiner ones.

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
everywhere). The week-12 batches `w12a` (the screen, five variants after
amendment 1) and `w12b:<variant>` (the decision) run from 77c37be (bundle on
`ner-gpu-bundle-3`) after 24 October and will write
`gpu/ner_results_mulms_leave_one_subfield_out_<variant>_m<N>.json` for the
variants `crf_mrs_topic`, `crf_mrm_topic`, `crf_mrm_topic6`, `crf_st3_topic`
and `crf_mr6` (run ids the subfields; CC BY-SA 4.0);
`gpu/ner_week5_seed_analysis.json` (`mulms`, `week12`) will hold the screen
and the decisions. `silver_lexicons/<subfield>.json` (`silver_lexicon.py`,
the CPU probe of `docs/METHODS_PLAN.md` §4B, no decision) records per
subfield the lexicons' sizes and their coverage of the unseen test names
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
as well, unlike the rest of the repository (Apache 2.0).

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
