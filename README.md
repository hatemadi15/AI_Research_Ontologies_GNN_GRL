# Ontology Learning on MaterioMiner (GNN + LLM)

A research pipeline that learns ontology structure from the **MaterioMiner**
corpus: four materials-fatigue papers whose entities are annotated with the
classes of the **Materials Mechanics Ontology (MMO)**. It builds a term
co-occurrence graph, trains GNN embeddings, clusters terms, induces a
taxonomy, types every term against the MMO, extracts relations, and evaluates
all of it **without leaking gold labels**.

> **Note on earlier results.** Versions up to v5 (PR #3, commit `ef05d19`)
> reported *Type F1 0.997, Term F1 0.911/0.956, Concept F1 0.447* and claimed
> to beat SOTA. Those numbers are not valid. The gold NER labels (which *are*
> MMO class names) were fed into the predictions, and the metrics never
> checked whether a term received the right class.
> [PR #5](https://github.com/hatemadi15/AI_Research_Ontologies_GNN_GRL/pull/5)
> removed the leakage and replaced the metrics; the before/after comparison
> is in [Evaluation](#evaluation) and [docs/ROADMAP.md](docs/ROADMAP.md). All
> numbers below are leakage-free.

## Data

| | |
|---|---|
| Corpus | 4 papers, 476 sentences, 2,168 annotated mentions, 1,055 distinct entity texts |
| Labels | 177 fine-grained types (all are MMO class names) and 27 coarse-grained types (CoNLL BIO) |
| Graph terms | 315 entity texts that occur in ≥ 2 sentences; 47 of them carry more than one gold type |
| Ontology | `ontology.ttl` (MMO): 428 classes, 420 `skos:definition`s, 431 `subClassOf` axioms, 70 existential restrictions on MMO classes |
| Extra text | `pubmed_sentences.txt`, 19,333 unannotated PubMed sentences |

The dataset has **no relation annotations**.

## Pipeline

```
builder.py        CoNLL -> PMI co-occurrence graph (+ PubMed co-occurrences), SBERT node features
gnn.py            GraphSAGE link prediction (residual input projection) -> term embeddings
cluster.py        agglomerative clustering (silhouette-selected k)
taxonomy.py       is-a edges: most-similar more-general cluster member + Hearst patterns
align.py          term x class scores (embedding + lexical, all labels, GNN smoothing)
                  -> predicted_types_base.json (typing) + ontology_alignment.csv (alignment rows)
rag_typing.py     LLM re-ranks the top-5 classes of low-confidence terms -> predicted_types.json
relations.py      cue patterns + dependency parse + GNN co-occurrence (+ LLM labelling)
relations_axioms.py  OWL domain/range pruning
eval_f1.py        leakage-free typing / alignment / clustering / taxonomy metrics
eval_relations.py class-level relation evaluation against the ontology restrictions
```

Stand-alone baselines:

* `term_typing_baseline.py`: cross-validated term-typing baselines
  (label/definition matching, kNN, hybrid, LLM re-ranking).
* `ner_baseline.py`: MatSciBERT NER under the MaterioMiner protocol. This is
  the only setting with published SOTA numbers.
  - Split protocols: the fixed split, a new random split per seed
    (`--vary-split`), or training on three papers and testing on the fourth
    (`--leave-one-paper-out`).
  - Training variants: a class-weighted loss (`--class-weights`) or a CRF head
    with IOB2 constraints (`--crf`).
  - `--compare` pairs every variant run with the baseline run on the same
    split and seed.

## Repository layout

| Path | Content |
|---|---|
| `src/graph/` | pipeline stages, evaluators, baselines, `run_pipeline.py`, `run_ablation.py` |
| `data/raw/dataset/` | MaterioMiner CoNLL files (`fine_grained_ner/`, `coarse_grained_ner/`), the MMO and upper ontologies (`ontologies/`), PubMed sentences |
| `data/processed/` | outputs of pipeline and baseline runs (not versioned) |
| `results/` | versioned JSON snapshots behind every number in this README (see [results/README.md](results/README.md)) |
| `docs/ROADMAP.md` | where the project stands and the prioritised next steps |
| `tests/` | unit tests, run by CI |

## Results

<!-- RESULTS:START -->
All numbers are leakage-free and come from the JSON files in [`results/`](results/)
(CPU, `SEED=0`, except the week-4 and week-5 runs on Kaggle T4 GPUs in
[`results/gpu/`](results/gpu/), marked GPU; LLM = `openai/gpt-4o-mini` via OpenRouter). Gold term spans
are given to the pipeline and the typing baselines. "Gold types used" marks
outputs whose predictions had access to the gold labels: those rows are
reported only to show the size of the leak.

**At a glance** (95% CIs in brackets; details below)

| Task | Result in this repo | Reference |
|---|---|---|
| NER, fine-grained, 5 random splits (micro F1) | 73.91 [69.36, 78.45] | published 69.92: not significantly better |
| NER, coarse-grained, 5 random splits (micro F1) | 74.57 [73.15, 75.98] | published 72.32: **beaten** if the paper used micro F1 |
| NER on an unseen paper (leave one paper out) | 43.51 fine, 54.34 coarse | no published equivalent |
| NER with a class-weighted loss: paired gain over the baseline | fine -0.24 [-3.09, +2.61]; coarse +0.41 [-2.31, +3.13] | counts only if the CI lies above 0 |
| NER with a CRF head: paired gain over the baseline | fine +2.46 [+0.39, +4.52]; coarse +3.58 [+2.43, +4.74] | counts only if the CI lies above 0 |
| NER, fine-grained with the CRF head, 5 random splits (micro F1) | 76.36 [71.00, 81.73] | published 69.92: **beaten** if the paper used micro F1; the gain comes from valid spans only (see below) |
| NER, coarse-grained with the CRF head, 5 random splits (micro F1) | 78.15 [76.61, 79.69] | published 72.32: **beaten** if the paper used micro F1; +1.91 of the gain survives strict IOB2 scoring |
| NER, fine-grained, the published split (the release's partition, micro F1), 10 seeds (GPU runs) | plain MatSciBERT 70.19 [69.03, 71.35]; CRF head 73.31 [72.28, 74.35] | published 69.92: within the 95% CI of plain MatSciBERT (reproduced), below (**beaten**) that of the CRF head; the CIs are over seeds only |
| NER, coarse-grained, the published split (the release's partition, micro F1), 10 seeds (GPU runs) | plain MatSciBERT 72.52 [71.88, 73.16]; CRF head 75.48 [74.23, 76.72] | published 72.32: within the 95% CI of plain MatSciBERT (reproduced), below (**beaten**) that of the CRF head; the CIs are over seeds only |
| NER on SOFC-Exp, the official split, 10 seeds (GPU runs) | plain MatSciBERT 82.54 [81.84, 83.23]; CRF head 83.75 [82.85, 84.65] (macro 81.11) | published MatSciBERT-CRF 84.77 micro, 82.39 macro (Gupta et al. 2022): above the CRF head's 95% CI (over seeds only) |
| NER, fine-grained, reported system (week-5 protocol; 10 seeds, GPU runs): 3-model seed ensemble `crf_ens3`; on unseen papers the multi-task vote `crf_mt_ens3` | micro F1 79.68 [76.72, 82.65] over 5 random splits; 52.47 on an unseen paper (`crf_ens3` 48.81) | published 69.92 (none for an unseen paper): **beaten** under micro, macro and weighted averaging |
| NER, coarse-grained, reported system (week-5 protocol; 10 seeds, GPU runs): multi-task vote `crf_mt_ens3`; on unseen papers the 3-model seed ensemble `crf_ens3` | micro F1 79.43 [77.17, 81.69] over 5 random splits (`crf_ens3` 78.63); 61.10 on an unseen paper | published 72.32 (none for an unseen paper): **beaten** under micro and weighted averaging; macro 70.83 [64.49, 77.17] is not significantly above |
| Term typing, 315 terms, pipeline (zero-shot) | accuracy 0.384 (0.429 with LLM re-ranking) | supervised + LLM baseline 0.654; majority class 0.083 |
| Clustering / taxonomy / relations, class level | ARI 0.035 / edge F1 0.006 / F1 0.000 | near chance; see the roadmap |

#### Term typing: 315 graph terms, gold spans given

| System | Gold types used? | Accuracy [95% CI] | Set F1 (LLMs4OL) | Macro-F1 | Acc@5 | Hier. F1 |
|---|---|---|---|---|---|---|
| Majority-class reference | – | 0.083 | – | – | – | – |
| **Pipeline v6, LLM off** | no | 0.384 [0.33, 0.43] | 0.351 | 0.457 | 0.502 | 0.522 |
| **Pipeline v6, LLM on** (RAG re-ranking) | no | 0.429 [0.38, 0.48] | 0.392 | 0.478 | 0.502 | 0.591 |
| Pipeline v6, ORACLE (upper bound) | **yes** | 1.000 [1.00, 1.00] | 0.914 | 0.908 | 1.000 | 1.000 |
| v5 outputs re-scored, LLM off | **yes** (leak) | 0.571 [0.51, 0.62] (coverage 0.67) | 0.615 | 0.349 | 0.648 | 0.703 |
| v5 outputs re-scored, LLM on | **yes** (leak) | 0.740 [0.69, 0.79] (coverage 0.95) | 0.692 | 0.675 | 0.803 | 0.845 |

#### Term-typing baselines (`term_typing_baseline.py`): 5-fold grouped CV, mean ± sd over folds

| Method | Setting | Accuracy | Set F1 | Macro-F1 | Acc@5 | Hier. F1 | Acc. on classes unseen in training |
|---|---|---|---|---|---|---|---|
| majority | supervised-CV (15 folds) | 0.083 ± 0.006 | 0.076 | 0.003 | 0.083 | 0.076 | 0.000 |
| exact | zero-shot (15 folds) | 0.152 ± 0.031 | 0.223 | 0.172 | 0.152 | 0.306 | 0.466 |
| label | zero-shot (15 folds) | 0.355 ± 0.043 | 0.325 | 0.353 | 0.492 | 0.509 | 0.737 |
| definition | zero-shot (15 folds) | 0.321 ± 0.058 | 0.293 | 0.350 | 0.495 | 0.480 | 0.661 |
| label+def | zero-shot (15 folds) | 0.355 ± 0.042 | 0.325 | 0.372 | 0.511 | 0.509 | 0.778 |
| knn | supervised-CV (15 folds) | 0.416 ± 0.050 | 0.381 | 0.313 | 0.630 | 0.514 | 0.000 |
| hybrid | supervised-CV (15 folds) | 0.496 ± 0.041 | 0.454 | 0.430 | 0.771 | 0.612 | 0.455 |
| hybrid+llm | supervised-CV + LLM (5 folds) | 0.654 ± 0.084 | 0.598 | 0.606 | 0.778 | 0.742 | 0.786 |

#### Other levels: old metric vs leakage-free metric

| Level | Old v5 metric (invalid) | Leakage-free metric | v5 outputs | v6 LLM off | v6 LLM on |
|---|---|---|---|---|---|
| Type | Type F1 0.997 | – (dataset property: 177/177 labels are class names) | – | – | – |
| Term | Term F1@0.5 0.893 (0.935 LLM on) | alignment pairwise F1@0.5 | 0.273 | 0.121 | 0.131 |
| Concept | Concept F1 0.447 | clustering ARI / B-cubed F1 | 0.775 / 0.907 (gold-type clusters) | 0.035 / 0.183 | 0.035 / 0.183 |
| Taxonomy | – | class-level edge F1 (precision vs closure; chance 0.013) | 0.000 (0.069) | 0.006 (0.027) | 0.025 (0.031) |
| Relations | partial-entity recall 0.925 | class-level F1 vs 70 restrictions | – | 0.000 | 0.000 |

#### Ablations (`run_ablation.py`, LLM off; oracle = upper bound)

Δ is the paired typing-accuracy difference to `full_pipeline` on the same 315 terms; p is the exact McNemar test.

| Preset | Typing acc. | Δ vs full (p) | Set F1 | Macro-F1 | Hier. F1 | Clust. ARI | Taxonomy P (closure) | Relation class-F1 |
|---|---|---|---|---|---|---|---|---|
| full_pipeline | 0.384 | – | 0.351 | 0.457 | 0.522 | 0.035 | 0.027 | 0.000 |
| no_gnn | 0.378 | -0.006 (0.69) | 0.345 | 0.459 | 0.517 | 0.195 | 0.037 | 0.000 |
| no_bidirectional | 0.384 | +0.000 (1.00) | 0.351 | 0.457 | 0.522 | 0.035 | 0.027 | 0.000 |
| embedding_only | 0.365 | -0.019 (0.24) | 0.334 | 0.451 | 0.512 | 0.035 | 0.022 | 0.000 |
| no_hearst | 0.384 | +0.000 (1.00) | 0.351 | 0.457 | 0.522 | 0.035 | 0.029 | 0.000 |
| no_dep_parsing | 0.384 | +0.000 (1.00) | 0.351 | 0.457 | 0.522 | 0.035 | 0.027 | 0.000 |
| no_corpus_augment | 0.381 | -0.003 (1.00) | 0.348 | 0.456 | 0.518 | 0.082 | 0.010 | 0.000 |
| baseline_sbert | 0.356 | -0.029 (0.08) | 0.325 | 0.446 | 0.509 | 0.195 | 0.033 | 0.000 |
| oracle *(oracle)* | 1.000 | +0.616 (<0.001) | 0.914 | 0.908 | 1.000 | 0.863 | 0.136 | 0.002 |

#### NER (MatSciBERT, MaterioMiner protocol, seqeval entity F1)

Mean ± population SD over runs, and the 95% Student-t CI of the mean. Micro F1 is the primary metric: the authors' 2025 follow-up states micro averaging, while the 2024 paper only says "averaging over the five random initializations and entity types". The published numbers come from one unpublished random split and have no reported SD.

| Protocol | Granularity | Runs | Micro F1 [95% CI] | Macro F1 | Weighted F1 | Strict micro | Published |
|---|---|---|---|---|---|---|---|
| fixed split (seed 0), 5 model seeds | fine | 5 seeds | 72.99 ± 1.53 [70.86, 75.12] | 67.99 | 72.28 | 74.97 | 69.92 |
| fixed split (seed 0), 5 model seeds | coarse | 5 seeds | 75.57 ± 1.03 [74.14, 77.01] | 67.14 | 75.87 | 77.11 | 72.32 |
| new random split per seed | fine | 5 seeds | 73.91 ± 3.28 [69.36, 78.45] | 71.19 | 73.98 | 76.26 | 69.92 |
| new random split per seed | coarse | 5 seeds | 74.57 ± 1.02 [73.15, 75.98] | 65.53 | 74.46 | 76.24 | 72.32 |
| leave one paper out | fine | 4 papers | 43.51 ± 3.19 [37.65, 49.37] | 35.58 | 40.61 | 45.04 | – (out of distribution) |
| leave one paper out | coarse | 4 papers | 54.34 ± 3.50 [47.91, 60.78] | 43.14 | 53.73 | 56.78 | – (out of distribution) |

#### NER training variants vs the baseline (paired by split and model seed)

Δ = variant − baseline on the same split and model seed, with the 95% Student-t CI of the mean difference. Pre-registered rule: a variant improves on the baseline only if the micro-F1 interval over the 5 random splits lies entirely above 0.

| Variant | Protocol | Granularity | Pairs | Variant micro F1 [95% CI] | Δ micro [95% CI] | Δ macro [95% CI] | Δ strict micro |
|---|---|---|---|---|---|---|---|
| class-weighted loss | new random split per seed | fine | 5 | 73.67 [69.77, 77.56] | -0.24 [-3.09, +2.61] | +0.02 [-4.99, +5.02] | -0.64 |
| class-weighted loss | new random split per seed | coarse | 5 | 74.98 [71.87, 78.08] | +0.41 [-2.31, +3.13] | +2.18 [-3.08, +7.45] | +0.39 |
| class-weighted loss | leave one paper out | fine | 4 | 45.32 [36.99, 53.64] | +1.81 [-4.28, +7.89] | -0.73 [-5.84, +4.38] | +1.75 |
| class-weighted loss | leave one paper out | coarse | 4 | 55.26 [50.15, 60.36] | +0.91 [-5.15, +6.98] | +2.88 [-7.08, +12.84] | +0.76 |
| CRF head | new random split per seed | fine | 5 | 76.36 [71.00, 81.73] | +2.46 [+0.39, +4.52] | +3.06 [-0.07, +6.18] | +0.10 |
| CRF head | new random split per seed | coarse | 5 | 78.15 [76.61, 79.69] | +3.58 [+2.43, +4.74] | +4.90 [+2.37, +7.43] | +1.91 |
| CRF head | leave one paper out | fine | 4 | 49.44 [41.00, 57.89] | +5.94 [+2.70, +9.18] | +4.18 [+1.29, +7.08] | +4.40 |
| CRF head | leave one paper out | coarse | 4 | 59.82 [54.15, 65.49] | +5.47 [+0.43, +10.52] | +5.85 [+1.37, +10.32] | +3.03 |

#### NER week-2 variants vs the CRF head (paired by split and model seed)

Δ = variant − CRF head on the same split and model seed, with the 95% Student-t CI of the mean difference. Pre-registered rule: a variant improves on the CRF head only if the micro-F1 interval over the 5 random splits lies entirely above 0. Week-2 runs use 2 CPU threads and the CRF head 3. With the same seeds, the thread count alone changes a run (floating-point summation order), which adds noise to each difference. Re-running the CRF head with 2 threads (ensemble member `crf_m0`) changes micro F1 by fine-grained +0.71 [-4.23, +5.65] and coarse-grained -0.75 [-2.59, +1.08] over the 5 random splits.

| Variant | Protocol | Granularity | Pairs | Variant micro F1 [95% CI] | Δ micro [95% CI] | Δ macro [95% CI] | Δ strict micro |
|---|---|---|---|---|---|---|---|
| multi-task (fine + coarse heads) | new random split per seed | fine | 5 | 78.83 [74.09, 83.57] | +2.46 [-0.91, +5.84] | +2.52 [-3.89, +8.93] | +2.46 |
| multi-task (fine + coarse heads) | new random split per seed | coarse | 5 | 78.39 [75.24, 81.54] | +0.24 [-2.54, +3.02] | -1.08 [-5.32, +3.17] | +0.24 |
| multi-task (fine + coarse heads) | leave one paper out | fine | 4 | 49.56 [38.66, 60.46] | +0.11 [-8.56, +8.79] | -0.50 [-10.76, +9.76] | +0.11 |
| multi-task (fine + coarse heads) | leave one paper out | coarse | 4 | 58.52 [50.40, 66.63] | -1.30 [-4.80, +2.20] | -2.92 [-8.66, +2.83] | -1.30 |
| domain-adaptive pre-training | new random split per seed | fine | 5 | 77.04 [73.61, 80.47] | +0.68 [-2.55, +3.90] | -1.63 [-7.48, +4.21] | +0.68 |
| domain-adaptive pre-training | new random split per seed | coarse | 5 | 78.08 [75.82, 80.34] | -0.07 [-3.01, +2.86] | -1.63 [-6.72, +3.45] | -0.07 |
| domain-adaptive pre-training | leave one paper out | fine | 4 | 48.54 [41.61, 55.46] | -0.91 [-3.13, +1.31] | +0.47 [-1.82, +2.75] | -0.91 |
| domain-adaptive pre-training | leave one paper out | coarse | 4 | 58.75 [51.91, 65.60] | -1.06 [-3.25, +1.12] | -0.25 [-2.46, +1.95] | -1.06 |
| 3-model ensemble | new random split per seed | fine | 5 | 79.76 [76.14, 83.38] | +3.39 [-1.35, +8.14] | +2.91 [-3.70, +9.53] | +3.39 |
| 3-model ensemble | new random split per seed | coarse | 5 | 79.45 [77.82, 81.09] | +1.30 [-1.01, +3.62] | +2.60 [-0.51, +5.71] | +1.30 |
| 3-model ensemble | leave one paper out | fine | 4 | 49.55 [40.11, 58.99] | +0.11 [-1.99, +2.20] | +0.16 [-3.64, +3.96] | +0.11 |
| 3-model ensemble | leave one paper out | coarse | 4 | 61.42 [55.81, 67.03] | +1.60 [+0.52, +2.69] | +2.03 [-2.08, +6.14] | +1.60 |

#### NER week-3 comparisons (3-seed means)

Δ = variant − reference on the same split, where a model trained with 3 model seeds (K, K + 100 and K + 200 on split K; 0, 100 and 200 on a held-out paper) scores their mean, the ensembles and the one-seed SciBERT and PubMedBERT runs score once, and the CRF head is the mean of `crf_m0`, `crf_m100` and `crf_m200`; pre-registered rule (week-3 protocol): a variant improves on its reference only if the 95% Student-t CI of the mean micro-F1 difference lies entirely above 0, over the 5 random splits for the mixed-backbone ensemble (against the 3-model seed ensemble), multi-task and in-domain pre-training, and over the 4 held-out papers for label semantics and self-training; every other row is reported without a decision.

| Comparison | Protocol | Granularity | Pairs | Variant micro F1 [95% CI] | Δ micro [95% CI] | Δ macro [95% CI] | Δ strict micro |
|---|---|---|---|---|---|---|---|
| mixed-backbone ensemble vs 3-model seed ensemble | new random split per seed | fine | 5 | 79.79 [77.67, 81.90] | +0.03 [-1.80, +1.86] | -0.03 [-2.53, +2.46] | +0.03 |
| mixed-backbone ensemble vs 3-model seed ensemble | new random split per seed | coarse | 5 | 78.58 [77.93, 79.23] | -0.87 [-2.09, +0.35] | -4.11 [-7.87, -0.36] | -0.87 |
| mixed-backbone ensemble vs 3-model seed ensemble | leave one paper out | fine | 4 | 48.21 [36.53, 59.90] | -1.34 [-4.76, +2.09] | +1.12 [-7.40, +9.64] | -1.34 |
| mixed-backbone ensemble vs 3-model seed ensemble | leave one paper out | coarse | 4 | 59.97 [52.21, 67.73] | -1.45 [-5.74, +2.84] | -3.96 [-13.94, +6.03] | -1.45 |
| mixed-backbone ensemble vs CRF head | new random split per seed | fine | 5 | 79.79 [77.67, 81.90] | +2.50 [+0.07, +4.92] | +2.46 [-1.05, +5.97] | +2.50 |
| mixed-backbone ensemble vs CRF head | new random split per seed | coarse | 5 | 78.58 [77.93, 79.23] | +0.93 [-0.52, +2.38] | -1.72 [-4.99, +1.56] | +0.93 |
| mixed-backbone ensemble vs CRF head | leave one paper out | fine | 4 | 48.21 [36.53, 59.90] | +0.71 [-3.77, +5.18] | +2.94 [-4.67, +10.54] | +0.71 |
| mixed-backbone ensemble vs CRF head | leave one paper out | coarse | 4 | 59.97 [52.21, 67.73] | +0.94 [-3.20, +5.07] | -0.86 [-7.75, +6.03] | +0.94 |
| 3-model seed ensemble vs CRF head | new random split per seed | fine | 5 | 79.76 [76.14, 83.38] | +2.47 [+1.17, +3.77] | +2.49 [+1.47, +3.52] | +2.47 |
| 3-model seed ensemble vs CRF head | new random split per seed | coarse | 5 | 79.45 [77.82, 81.09] | +1.80 [+0.91, +2.69] | +2.40 [+1.75, +3.04] | +1.80 |
| 3-model seed ensemble vs CRF head | leave one paper out | fine | 4 | 49.55 [40.11, 58.99] | +2.04 [-0.23, +4.32] | +1.82 [-2.54, +6.18] | +2.04 |
| 3-model seed ensemble vs CRF head | leave one paper out | coarse | 4 | 61.42 [55.81, 67.03] | +2.39 [+1.56, +3.21] | +3.10 [-0.65, +6.84] | +2.39 |
| SciBERT backbone vs CRF head | new random split per seed | fine | 5 | 76.91 [73.61, 80.21] | -0.38 [-2.63, +1.87] | -0.00 [-2.52, +2.52] | -0.38 |
| SciBERT backbone vs CRF head | new random split per seed | coarse | 5 | 76.67 [75.01, 78.32] | -0.99 [-1.71, -0.26] | -3.41 [-6.58, -0.23] | -0.99 |
| SciBERT backbone vs CRF head | leave one paper out | fine | 4 | 42.67 [35.09, 50.26] | -4.83 [-6.66, -3.00] | -2.91 [-6.17, +0.36] | -4.83 |
| SciBERT backbone vs CRF head | leave one paper out | coarse | 4 | 54.52 [47.37, 61.66] | -4.52 [-8.17, -0.87] | -5.52 [-11.60, +0.57] | -4.52 |
| PubMedBERT backbone vs CRF head | new random split per seed | fine | 5 | 75.87 [73.38, 78.37] | -1.42 [-5.21, +2.37] | -1.98 [-6.29, +2.33] | -1.42 |
| PubMedBERT backbone vs CRF head | new random split per seed | coarse | 5 | 74.66 [72.25, 77.06] | -2.99 [-4.67, -1.32] | -7.42 [-13.86, -0.98] | -2.99 |
| PubMedBERT backbone vs CRF head | leave one paper out | fine | 4 | 42.57 [30.99, 54.14] | -4.94 [-13.22, +3.34] | -5.04 [-16.62, +6.55] | -4.94 |
| PubMedBERT backbone vs CRF head | leave one paper out | coarse | 4 | 55.28 [49.07, 61.49] | -3.75 [-8.41, +0.90] | -6.30 [-16.07, +3.46] | -3.75 |
| label semantics vs CRF head | new random split per seed | fine | 5 | 76.71 [73.20, 80.22] | -0.58 [-3.41, +2.24] | -0.20 [-3.76, +3.36] | -0.58 |
| label semantics vs CRF head | new random split per seed | coarse | 5 | 76.89 [74.60, 79.18] | -0.77 [-1.74, +0.21] | -1.96 [-5.62, +1.70] | -0.77 |
| label semantics vs CRF head | leave one paper out | fine | 4 | 47.35 [39.44, 55.26] | -0.16 [-2.13, +1.80] | -1.17 [-4.73, +2.38] | -0.16 |
| label semantics vs CRF head | leave one paper out | coarse | 4 | 60.21 [54.77, 65.65] | +1.17 [+0.94, +1.41] | +1.20 [-1.38, +3.78] | +1.17 |
| self-training vs CRF head | leave one paper out | fine | 4 | 50.42 [42.93, 57.91] | +2.91 [-0.73, +6.56] | +2.17 [-3.18, +7.53] | +2.91 |
| self-training vs CRF head | leave one paper out | coarse | 4 | 58.92 [53.72, 64.11] | -0.12 [-1.97, +1.73] | -0.21 [-2.37, +1.96] | -0.12 |
| multi-task (fine + coarse heads) vs CRF head | new random split per seed | fine | 5 | 79.16 [75.39, 82.94] | +1.87 [-2.25, +6.00] | +1.99 [-1.29, +5.27] | +1.87 |
| multi-task (fine + coarse heads) vs CRF head | new random split per seed | coarse | 5 | 78.40 [76.15, 80.64] | +0.75 [+0.10, +1.39] | -0.50 [-2.63, +1.64] | +0.75 |
| multi-task (fine + coarse heads) vs CRF head | leave one paper out | fine | 4 | 50.90 [42.78, 59.03] | +3.39 [+0.45, +6.34] | +2.26 [-4.24, +8.76] | +3.39 |
| multi-task (fine + coarse heads) vs CRF head | leave one paper out | coarse | 4 | 58.43 [52.05, 64.81] | -0.60 [-4.55, +3.34] | -2.30 [-9.74, +5.14] | -0.60 |
| in-domain pre-training vs CRF head | new random split per seed | fine | 5 | 77.40 [73.48, 81.33] | +0.11 [-2.74, +2.97] | -0.16 [-3.69, +3.37] | +0.11 |
| in-domain pre-training vs CRF head | new random split per seed | coarse | 5 | 77.50 [75.84, 79.17] | -0.15 [-1.06, +0.76] | -0.63 [-3.29, +2.04] | -0.15 |
| in-domain pre-training vs CRF head | leave one paper out | fine | 4 | 47.65 [38.46, 56.83] | +0.14 [-2.60, +2.88] | +0.15 [-2.20, +2.50] | +0.14 |
| in-domain pre-training vs CRF head | leave one paper out | coarse | 4 | 57.50 [51.87, 63.13] | -1.54 [-4.34, +1.26] | -3.30 [-12.75, +6.15] | -1.54 |

#### NER week-4 comparisons (3-seed means; CPU votes, GPU runs)

Δ = variant − reference on the same split, scored as in week 3: a model trained with 3 model seeds scores their mean, a vote over them scores once. Pre-registered rule (week-4 protocol): a change improves on its reference only if the 95% Student-t CI of the mean micro-F1 difference lies entirely above 0. The CPU rows are votes over the saved predictions of the week-3 runs, decided over the 4 held-out papers against the 3-model seed ensemble. The GPU rows are runs on Kaggle T4 GPUs, compared only with GPU runs: the multi-task vote against the GPU seed ensemble over the 5 random splits, label semantics on the multi-task model against the GPU CRF head over the 5 random splits and over the 4 held-out papers, and self-training with 3 teachers against the GPU CRF head over the 4 held-out papers. The other GPU rows are reported without a decision: the other protocol, comparisons with each change's parent and with the GPU seed ensemble, and repeats of the week-3 comparisons and of the CPU votes.

| Comparison | Runs | Protocol | Granularity | Pairs | Variant micro F1 [95% CI] | Δ micro [95% CI] | Δ macro [95% CI] | Role |
|---|---|---|---|---|---|---|---|---|
| label-semantics vote vs 3-model seed ensemble | CPU | new random split per seed | fine | 5 | 78.59 [75.14, 82.05] | -1.16 [-2.31, -0.01] | -0.51 [-2.72, +1.70] | reported |
| label-semantics vote vs 3-model seed ensemble | CPU | new random split per seed | coarse | 5 | 78.51 [76.45, 80.57] | -0.95 [-1.88, -0.02] | -3.75 [-10.02, +2.52] | reported |
| label-semantics vote vs 3-model seed ensemble | CPU | leave one paper out | fine | 4 | 49.91 [41.51, 58.30] | +0.36 [-2.53, +3.24] | -0.16 [-8.05, +7.74] | decision |
| label-semantics vote vs 3-model seed ensemble | CPU | leave one paper out | coarse | 4 | 63.05 [57.09, 69.02] | +1.63 [+0.11, +3.16] | +1.27 [-2.89, +5.42] | decision |
| self-training vote vs 3-model seed ensemble | CPU | leave one paper out | fine | 4 | 53.23 [45.05, 61.41] | +3.68 [-1.15, +8.52] | +3.33 [-5.60, +12.25] | decision |
| self-training vote vs 3-model seed ensemble | CPU | leave one paper out | coarse | 4 | 61.33 [55.29, 67.37] | -0.09 [-3.52, +3.35] | -0.38 [-4.88, +4.12] | decision |
| label-semantics vote vs CRF head | CPU | new random split per seed | fine | 5 | 78.59 [75.14, 82.05] | +1.30 [-1.08, +3.68] | +1.98 [-0.71, +4.68] | reported |
| label-semantics vote vs CRF head | CPU | new random split per seed | coarse | 5 | 78.51 [76.45, 80.57] | +0.86 [-0.25, +1.96] | -1.36 [-7.20, +4.49] | reported |
| label-semantics vote vs CRF head | CPU | leave one paper out | fine | 4 | 49.91 [41.51, 58.30] | +2.40 [+0.30, +4.50] | +1.66 [-3.44, +6.76] | reported |
| label-semantics vote vs CRF head | CPU | leave one paper out | coarse | 4 | 63.05 [57.09, 69.02] | +4.02 [+3.18, +4.86] | +4.36 [+2.20, +6.53] | reported |
| self-training vote vs CRF head | CPU | leave one paper out | fine | 4 | 53.23 [45.05, 61.41] | +5.72 [+2.33, +9.12] | +5.14 [-0.37, +10.66] | reported |
| self-training vote vs CRF head | CPU | leave one paper out | coarse | 4 | 61.33 [55.29, 67.37] | +2.30 [-0.62, +5.21] | +2.71 [+1.64, +3.79] | reported |
| multi-task vote vs 3-model seed ensemble | GPU | new random split per seed | fine | 5 | 81.16 [79.03, 83.29] | +1.26 [-1.08, +3.59] | +1.09 [-1.80, +3.98] | decision |
| multi-task vote vs 3-model seed ensemble | GPU | new random split per seed | coarse | 5 | 79.12 [77.37, 80.87] | -0.07 [-2.02, +1.87] | -2.20 [-9.67, +5.26] | decision |
| multi-task vote vs 3-model seed ensemble | GPU | leave one paper out | fine | 4 | 52.55 [46.03, 59.08] | +4.09 [-0.89, +9.07] | +4.85 [-9.15, +18.85] | reported |
| multi-task vote vs 3-model seed ensemble | GPU | leave one paper out | coarse | 4 | 61.20 [54.53, 67.86] | -0.10 [-5.04, +4.85] | -0.49 [-11.76, +10.78] | reported |
| label semantics on the multi-task model vs CRF head | GPU | new random split per seed | fine | 5 | 78.36 [75.89, 80.83] | +0.76 [-0.57, +2.10] | +2.09 [+1.00, +3.18] | decision |
| label semantics on the multi-task model vs CRF head | GPU | new random split per seed | coarse | 5 | 78.24 [75.78, 80.70] | +0.65 [-0.63, +1.93] | -0.73 [-3.29, +1.84] | decision |
| label semantics on the multi-task model vs CRF head | GPU | leave one paper out | fine | 4 | 50.45 [40.91, 60.00] | +4.08 [-1.99, +10.14] | +2.64 [-5.89, +11.17] | decision |
| label semantics on the multi-task model vs CRF head | GPU | leave one paper out | coarse | 4 | 59.27 [52.19, 66.34] | -0.01 [-2.64, +2.63] | -0.03 [-4.57, +4.51] | decision |
| self-training with 3 teachers vs CRF head | GPU | new random split per seed | fine | 5 | 78.82 [75.28, 82.36] | +1.22 [+0.60, +1.84] | +1.60 [-0.10, +3.29] | reported |
| self-training with 3 teachers vs CRF head | GPU | new random split per seed | coarse | 5 | 78.29 [75.40, 81.19] | +0.71 [-0.96, +2.38] | -0.99 [-5.58, +3.59] | reported |
| self-training with 3 teachers vs CRF head | GPU | leave one paper out | fine | 4 | 47.44 [40.17, 54.72] | +1.07 [+0.33, +1.80] | -0.54 [-1.96, +0.88] | decision |
| self-training with 3 teachers vs CRF head | GPU | leave one paper out | coarse | 4 | 60.12 [53.23, 67.02] | +0.85 [-0.58, +2.29] | +0.51 [-2.48, +3.50] | decision |
| label-semantics vote vs 3-model seed ensemble | GPU | new random split per seed | fine | 5 | 78.16 [74.24, 82.07] | -1.74 [-3.84, +0.35] | -0.99 [-4.71, +2.73] | reported |
| label-semantics vote vs 3-model seed ensemble | GPU | new random split per seed | coarse | 5 | 78.59 [75.69, 81.49] | -0.60 [-2.98, +1.78] | -0.75 [-3.08, +1.58] | reported |
| label-semantics vote vs 3-model seed ensemble | GPU | leave one paper out | fine | 4 | 48.97 [42.10, 55.84] | +0.51 [-1.56, +2.59] | -0.20 [-5.51, +5.11] | reported |
| label-semantics vote vs 3-model seed ensemble | GPU | leave one paper out | coarse | 4 | 61.70 [56.04, 67.35] | +0.40 [-1.36, +2.16] | +1.64 [-2.95, +6.24] | reported |
| self-training vote vs 3-model seed ensemble | GPU | leave one paper out | fine | 4 | 50.86 [44.43, 57.28] | +2.40 [-2.53, +7.33] | +2.79 [-2.02, +7.60] | reported |
| self-training vote vs 3-model seed ensemble | GPU | leave one paper out | coarse | 4 | 61.20 [55.67, 66.74] | -0.09 [-2.85, +2.67] | -0.50 [-8.21, +7.21] | reported |
| 3-model seed ensemble vs CRF head | GPU | new random split per seed | fine | 5 | 79.90 [76.68, 83.12] | +2.30 [+1.75, +2.85] | +2.86 [+1.80, +3.92] | reported |
| 3-model seed ensemble vs CRF head | GPU | new random split per seed | coarse | 5 | 79.19 [77.86, 80.53] | +1.61 [+0.65, +2.57] | +0.88 [-1.04, +2.81] | reported |
| 3-model seed ensemble vs CRF head | GPU | leave one paper out | fine | 4 | 48.46 [40.81, 56.10] | +2.08 [+0.97, +3.19] | +0.81 [-1.99, +3.61] | reported |
| 3-model seed ensemble vs CRF head | GPU | leave one paper out | coarse | 4 | 61.30 [55.74, 66.85] | +2.02 [-0.03, +4.08] | +1.79 [-1.32, +4.91] | reported |
| multi-task vote vs CRF head | GPU | new random split per seed | fine | 5 | 81.16 [79.03, 83.29] | +3.56 [+1.39, +5.72] | +3.95 [+1.11, +6.79] | reported |
| multi-task vote vs CRF head | GPU | new random split per seed | coarse | 5 | 79.12 [77.37, 80.87] | +1.54 [-0.57, +3.64] | -1.32 [-7.66, +5.02] | reported |
| multi-task vote vs CRF head | GPU | leave one paper out | fine | 4 | 52.55 [46.03, 59.08] | +6.18 [+1.51, +10.84] | +5.66 [-5.76, +17.07] | reported |
| multi-task vote vs CRF head | GPU | leave one paper out | coarse | 4 | 61.20 [54.53, 67.86] | +1.93 [-1.68, +5.53] | +1.30 [-7.02, +9.62] | reported |
| multi-task (fine + coarse heads) vs CRF head | GPU | new random split per seed | fine | 5 | 79.34 [76.62, 82.07] | +1.74 [-0.36, +3.84] | +2.17 [+0.28, +4.05] | reported |
| multi-task (fine + coarse heads) vs CRF head | GPU | new random split per seed | coarse | 5 | 77.50 [75.66, 79.35] | -0.08 [-1.71, +1.55] | -2.48 [-9.28, +4.32] | reported |
| multi-task (fine + coarse heads) vs CRF head | GPU | leave one paper out | fine | 4 | 50.24 [41.49, 59.00] | +3.87 [-2.25, +9.99] | +3.99 [-6.38, +14.36] | reported |
| multi-task (fine + coarse heads) vs CRF head | GPU | leave one paper out | coarse | 4 | 58.59 [51.42, 65.77] | -0.68 [-4.51, +3.15] | -0.84 [-7.87, +6.19] | reported |
| label semantics vs CRF head | GPU | new random split per seed | fine | 5 | 76.84 [73.57, 80.12] | -0.76 [-3.61, +2.09] | +0.24 [-3.93, +4.42] | reported |
| label semantics vs CRF head | GPU | new random split per seed | coarse | 5 | 77.44 [74.55, 80.32] | -0.15 [-1.68, +1.38] | -1.49 [-3.73, +0.74] | reported |
| label semantics vs CRF head | GPU | leave one paper out | fine | 4 | 47.70 [41.02, 54.37] | +1.32 [-0.43, +3.06] | +0.42 [-3.75, +4.60] | reported |
| label semantics vs CRF head | GPU | leave one paper out | coarse | 4 | 59.34 [54.28, 64.39] | +0.07 [-2.47, +2.61] | +0.65 [-4.24, +5.54] | reported |
| self-training vs CRF head | GPU | leave one paper out | fine | 4 | 48.51 [42.66, 54.36] | +2.14 [-1.02, +5.29] | +0.38 [-1.99, +2.75] | reported |
| self-training vs CRF head | GPU | leave one paper out | coarse | 4 | 58.18 [51.57, 64.80] | -1.09 [-4.31, +2.14] | -1.57 [-7.91, +4.78] | reported |
| label semantics on the multi-task model vs multi-task (fine + coarse heads) | GPU | new random split per seed | fine | 5 | 78.36 [75.89, 80.83] | -0.98 [-2.24, +0.28] | -0.08 [-1.68, +1.52] | reported |
| label semantics on the multi-task model vs multi-task (fine + coarse heads) | GPU | new random split per seed | coarse | 5 | 78.24 [75.78, 80.70] | +0.73 [-0.22, +1.68] | +1.76 [-2.78, +6.29] | reported |
| label semantics on the multi-task model vs multi-task (fine + coarse heads) | GPU | leave one paper out | fine | 4 | 50.45 [40.91, 60.00] | +0.21 [-5.83, +6.24] | -1.36 [-5.60, +2.89] | reported |
| label semantics on the multi-task model vs multi-task (fine + coarse heads) | GPU | leave one paper out | coarse | 4 | 59.27 [52.19, 66.34] | +0.67 [-0.60, +1.95] | +0.81 [-1.81, +3.43] | reported |
| label semantics on the multi-task model vs label semantics | GPU | new random split per seed | fine | 5 | 78.36 [75.89, 80.83] | +1.52 [-1.34, +4.38] | +1.85 [-1.55, +5.24] | reported |
| label semantics on the multi-task model vs label semantics | GPU | new random split per seed | coarse | 5 | 78.24 [75.78, 80.70] | +0.80 [-0.74, +2.34] | +0.76 [-1.39, +2.92] | reported |
| label semantics on the multi-task model vs label semantics | GPU | leave one paper out | fine | 4 | 50.45 [40.91, 60.00] | +2.76 [-1.63, +7.14] | +2.22 [-4.21, +8.65] | reported |
| label semantics on the multi-task model vs label semantics | GPU | leave one paper out | coarse | 4 | 59.27 [52.19, 66.34] | -0.07 [-3.50, +3.35] | -0.68 [-4.00, +2.64] | reported |
| self-training with 3 teachers vs self-training | GPU | leave one paper out | fine | 4 | 47.44 [40.17, 54.72] | -1.07 [-4.74, +2.60] | -0.92 [-4.26, +2.42] | reported |
| self-training with 3 teachers vs self-training | GPU | leave one paper out | coarse | 4 | 60.12 [53.23, 67.02] | +1.94 [-2.68, +6.56] | +2.08 [-3.27, +7.42] | reported |
| self-training with 3 teachers vs 3-model seed ensemble | GPU | new random split per seed | fine | 5 | 78.82 [75.28, 82.36] | -1.08 [-2.14, -0.03] | -1.26 [-3.22, +0.69] | reported |
| self-training with 3 teachers vs 3-model seed ensemble | GPU | new random split per seed | coarse | 5 | 78.29 [75.40, 81.19] | -0.90 [-3.20, +1.39] | -1.88 [-7.25, +3.49] | reported |
| self-training with 3 teachers vs 3-model seed ensemble | GPU | leave one paper out | fine | 4 | 47.44 [40.17, 54.72] | -1.01 [-1.63, -0.39] | -1.35 [-2.89, +0.19] | reported |
| self-training with 3 teachers vs 3-model seed ensemble | GPU | leave one paper out | coarse | 4 | 60.12 [53.23, 67.02] | -1.17 [-2.61, +0.27] | -1.29 [-6.21, +3.64] | reported |

#### NER week-5 comparisons (10-seed means, GPU runs)

Δ = variant − reference on the same split, GPU runs only, with 10 model seeds per split: a model scores the mean of its 10 runs, a vote the mean of its votes over 3 disjoint seed triples (seeds 0–200, 300–500 and 600–800). Pre-registered rule (week-5 protocol): a change improves on its reference only if the 95% Student-t CI of the mean micro-F1 difference lies entirely above 0. Decisions: the claims of weeks 3 and 4 re-tested (multi-task over the 5 random splits; label semantics and self-training with 3 teachers over the 4 held-out papers), and each vote against the 3-model seed ensemble on both protocols (the reported system). The other rows are reported without a decision.

| Comparison | Protocol | Granularity | Pairs | Variant micro F1 [95% CI] | Δ micro [95% CI] | Δ macro [95% CI] | Role |
|---|---|---|---|---|---|---|---|
| multi-task (fine + coarse heads) vs CRF head | new random split per seed | fine | 5 | 79.31 [76.58, 82.05] | +1.83 [+0.32, +3.34] | +1.92 [+0.50, +3.33] | decision |
| multi-task (fine + coarse heads) vs CRF head | new random split per seed | coarse | 5 | 77.67 [75.27, 80.07] | +0.71 [-0.15, +1.56] | +0.26 [-2.76, +3.27] | decision |
| multi-task (fine + coarse heads) vs CRF head | leave one paper out | fine | 4 | 50.17 [42.01, 58.34] | +3.72 [+0.57, +6.87] | +3.76 [-2.44, +9.96] | reported |
| multi-task (fine + coarse heads) vs CRF head | leave one paper out | coarse | 4 | 58.05 [51.96, 64.13] | -0.56 [-1.91, +0.79] | -1.71 [-4.41, +0.99] | reported |
| label semantics vs CRF head | new random split per seed | fine | 5 | 77.03 [74.23, 79.83] | -0.45 [-1.63, +0.73] | -0.07 [-2.26, +2.12] | reported |
| label semantics vs CRF head | new random split per seed | coarse | 5 | 77.39 [75.01, 79.76] | +0.42 [-1.19, +2.04] | +0.59 [-2.59, +3.77] | reported |
| label semantics vs CRF head | leave one paper out | fine | 4 | 47.68 [39.01, 56.36] | +1.23 [-0.96, +3.42] | +0.74 [-4.16, +5.64] | decision |
| label semantics vs CRF head | leave one paper out | coarse | 4 | 58.24 [52.00, 64.49] | -0.36 [-1.66, +0.94] | -0.53 [-1.36, +0.29] | decision |
| self-training with 3 teachers vs CRF head | new random split per seed | fine | 5 | 78.53 [75.51, 81.55] | +1.05 [+0.10, +2.00] | +0.94 [-0.52, +2.41] | reported |
| self-training with 3 teachers vs CRF head | new random split per seed | coarse | 5 | 78.00 [75.70, 80.30] | +1.04 [+0.16, +1.91] | +1.32 [-1.64, +4.28] | reported |
| self-training with 3 teachers vs CRF head | leave one paper out | fine | 4 | 48.20 [40.74, 55.66] | +1.74 [-1.16, +4.64] | +1.90 [-1.11, +4.90] | decision |
| self-training with 3 teachers vs CRF head | leave one paper out | coarse | 4 | 60.16 [53.33, 67.00] | +1.56 [-0.88, +4.00] | +1.55 [-0.79, +3.89] | decision |
| multi-task vote vs 3-model seed ensemble | new random split per seed | fine | 5 | 80.82 [78.16, 83.49] | +1.14 [-0.98, +3.27] | +0.76 [-1.90, +3.41] | decision |
| multi-task vote vs 3-model seed ensemble | new random split per seed | coarse | 5 | 79.43 [77.17, 81.69] | +0.79 [+0.17, +1.42] | +0.33 [-2.76, +3.43] | decision |
| multi-task vote vs 3-model seed ensemble | leave one paper out | fine | 4 | 52.47 [43.77, 61.17] | +3.65 [+0.49, +6.82] | +4.58 [-3.55, +12.71] | decision |
| multi-task vote vs 3-model seed ensemble | leave one paper out | coarse | 4 | 60.79 [55.16, 66.42] | -0.32 [-1.07, +0.44] | -1.20 [-5.24, +2.83] | decision |
| label-semantics vote vs 3-model seed ensemble | new random split per seed | fine | 5 | 78.89 [75.42, 82.36] | -0.79 [-2.02, +0.43] | -0.42 [-2.08, +1.23] | decision |
| label-semantics vote vs 3-model seed ensemble | new random split per seed | coarse | 5 | 78.94 [76.69, 81.19] | +0.31 [-1.27, +1.89] | +0.77 [-2.08, +3.61] | decision |
| label-semantics vote vs 3-model seed ensemble | leave one paper out | fine | 4 | 49.38 [40.91, 57.86] | +0.57 [-1.46, +2.60] | -0.11 [-5.36, +5.14] | decision |
| label-semantics vote vs 3-model seed ensemble | leave one paper out | coarse | 4 | 60.48 [54.39, 66.57] | -0.63 [-1.72, +0.46] | -0.87 [-3.11, +1.37] | decision |
| vote of self-training with 3 teachers vs 3-model seed ensemble | new random split per seed | fine | 5 | 80.66 [77.88, 83.44] | +0.98 [-0.15, +2.10] | +0.54 [-0.57, +1.64] | decision |
| vote of self-training with 3 teachers vs 3-model seed ensemble | new random split per seed | coarse | 5 | 79.75 [77.16, 82.35] | +1.12 [-0.36, +2.60] | +1.40 [-1.75, +4.55] | decision |
| vote of self-training with 3 teachers vs 3-model seed ensemble | leave one paper out | fine | 4 | 49.94 [40.93, 58.94] | +1.12 [-2.71, +4.96] | +1.82 [-1.83, +5.48] | decision |
| vote of self-training with 3 teachers vs 3-model seed ensemble | leave one paper out | coarse | 4 | 62.04 [54.86, 69.21] | +0.93 [-1.66, +3.53] | +0.91 [-2.82, +4.64] | decision |
| 3-model seed ensemble vs CRF head | new random split per seed | fine | 5 | 79.68 [76.72, 82.65] | +2.20 [+1.61, +2.78] | +2.47 [+1.58, +3.35] | reported |
| 3-model seed ensemble vs CRF head | new random split per seed | coarse | 5 | 78.63 [76.94, 80.33] | +1.67 [+1.02, +2.32] | +1.57 [+0.19, +2.94] | reported |
| 3-model seed ensemble vs CRF head | leave one paper out | fine | 4 | 48.81 [42.13, 55.50] | +2.36 [+1.84, +2.88] | +1.96 [-0.23, +4.16] | reported |
| 3-model seed ensemble vs CRF head | leave one paper out | coarse | 4 | 61.10 [55.73, 66.47] | +2.50 [+1.86, +3.13] | +2.49 [+1.63, +3.34] | reported |

#### NER week-6 screen (3-seed means, held-out papers, GPU runs)

Δ = method − reference on the same held-out paper, GPU runs only, with seeds 0, 100 and 200: a model scores the mean of its 3 runs, a vote over them scores once. Week-6 protocol, stage 1: a method goes to stage 2 (10 seeds) if its mean micro-F1 difference against the CRF head is above 0 at either granularity. The screen decides nothing.

| Comparison | Granularity | Pairs | Variant micro F1 [95% CI] | Δ micro [95% CI] | Δ macro [95% CI] | Role |
|---|---|---|---|---|---|---|
| DeBERTa-v3-large vs CRF head | fine | 4 | 41.99 [33.69, 50.30] | -4.38 [-7.32, -1.45] | -3.34 [-4.55, -2.13] | screen |
| DeBERTa-v3-large vs CRF head | coarse | 4 | 51.26 [47.18, 55.34] | -8.01 [-11.72, -4.30] | -6.76 [-14.33, +0.80] | screen |
| document context vs CRF head | fine | 4 | 49.03 [43.40, 54.67] | +2.66 [+0.91, +4.40] | +2.11 [-2.30, +6.52] | screen |
| document context vs CRF head | coarse | 4 | 58.77 [50.81, 66.73] | -0.50 [-3.82, +2.83] | -1.25 [-7.43, +4.94] | screen |
| multi-task self-training vs CRF head | fine | 4 | 52.43 [44.84, 60.03] | +6.06 [+1.05, +11.06] | +5.73 [-5.31, +16.77] | screen |
| multi-task self-training vs CRF head | coarse | 4 | 59.52 [53.85, 65.19] | +0.25 [-1.12, +1.62] | +0.24 [-3.04, +3.52] | screen |
| multi-task self-training vs multi-task (fine + coarse heads) | fine | 4 | 52.43 [44.84, 60.03] | +2.19 [-1.61, +5.98] | +1.74 [-1.54, +5.01] | reported |
| multi-task self-training vs multi-task (fine + coarse heads) | coarse | 4 | 59.52 [53.85, 65.19] | +0.93 [-2.82, +4.68] | +1.09 [-3.33, +5.50] | reported |
| DeBERTa-v3-large vote vs 3-model seed ensemble | fine | 4 | 46.29 [36.19, 56.39] | -2.17 [-5.12, +0.78] | +2.13 [-0.25, +4.50] | reported |
| DeBERTa-v3-large vote vs 3-model seed ensemble | coarse | 4 | 54.83 [49.19, 60.47] | -6.47 [-10.54, -2.39] | -5.22 [-16.95, +6.52] | reported |
| document-context vote vs 3-model seed ensemble | fine | 4 | 51.38 [45.60, 57.16] | +2.92 [+0.02, +5.83] | +4.08 [-1.05, +9.21] | reported |
| document-context vote vs 3-model seed ensemble | coarse | 4 | 61.59 [54.24, 68.94] | +0.29 [-4.64, +5.22] | -0.20 [-8.28, +7.87] | reported |
| multi-task self-training vote vs 3-model seed ensemble | fine | 4 | 54.19 [45.13, 63.25] | +5.73 [+0.70, +10.77] | +6.58 [-8.25, +21.41] | reported |
| multi-task self-training vote vs 3-model seed ensemble | coarse | 4 | 61.53 [55.40, 67.67] | +0.24 [-1.45, +1.93] | +0.04 [-5.70, +5.78] | reported |

#### NER week-6 comparisons (10-seed means, held-out papers, GPU runs)

Δ = method − reference on the same held-out paper, GPU runs only, with 10 model seeds per paper: a model scores the mean of its 10 runs, a vote the mean of its votes over 3 disjoint seed triples. Pre-registered rule (week-6 protocol): a change improves on its reference only if the 95% Student-t CI of the mean micro-F1 difference lies entirely above 0. Decisions: each method against the CRF head, and each method's vote against the strongest system on unseen papers (the multi-task vote fine-grained, the 3-model seed ensemble coarse-grained).

| Comparison | Granularity | Pairs | Variant micro F1 [95% CI] | Δ micro [95% CI] | Δ macro [95% CI] | Role |
|---|---|---|---|---|---|---|
| document context vs CRF head | fine | 4 | 47.59 [39.86, 55.33] | +1.14 [-0.94, +3.22] | +1.82 [-3.39, +7.04] | decision |
| document context vs CRF head | coarse | 4 | 58.72 [52.23, 65.22] | +0.12 [-2.03, +2.27] | -1.05 [-3.28, +1.17] | decision |
| document-context vote vs multi-task vote | fine | 4 | 51.02 [42.70, 59.35] | -1.44 [-3.80, +0.91] | -1.46 [-6.72, +3.80] | decision |
| document-context vote vs multi-task vote | coarse | 4 | 61.86 [54.85, 68.87] | +1.07 [-1.23, +3.38] | +0.44 [-2.53, +3.41] | reported |
| document-context vote vs 3-model seed ensemble | fine | 4 | 51.02 [42.70, 59.35] | +2.21 [+0.01, +4.41] | +3.13 [-2.08, +8.33] | reported |
| document-context vote vs 3-model seed ensemble | coarse | 4 | 61.86 [54.85, 68.87] | +0.76 [-1.79, +3.31] | -0.76 [-3.29, +1.77] | decision |

#### NER week-7 comparisons, part 1 (the consistency rule, 10-seed means, GPU runs)

Δ = with the rule − without it, on the same split, GPU runs only, 10 model seeds: a model scores the mean of its 10 runs, a vote the mean of its votes over 3 disjoint seed triples. Week-7 protocol, part 1: the consistency rule (`ner_consistency.py`, fixed before it was applied) on the saved test predictions. Pre-registered rule: a change improves on its reference only if the 95% Student-t CI of the mean micro-F1 difference lies entirely above 0. Decisions, on the 4 held-out papers: the rule on the CRF head, and on the strongest system on unseen papers (the multi-task vote fine-grained, the 3-model seed ensemble coarse-grained). The other rows are reported without a decision.

| Comparison | Protocol | Granularity | Pairs | Variant micro F1 [95% CI] | Δ micro [95% CI] | Δ macro [95% CI] | Role |
|---|---|---|---|---|---|---|---|
| CRF head + the consistency rule vs CRF head | leave one paper out | fine | 4 | 46.75 [39.67, 53.83] | +0.30 [-0.35, +0.95] | +0.13 [-0.14, +0.39] | decision |
| CRF head + the consistency rule vs CRF head | leave one paper out | coarse | 4 | 58.73 [53.37, 64.09] | +0.12 [+0.01, +0.24] | +0.11 [-0.18, +0.40] | decision |
| CRF head + the consistency rule vs CRF head | new random split per seed | fine | 5 | 77.37 [74.44, 80.30] | -0.12 [-0.40, +0.16] | -0.10 [-0.39, +0.18] | reported |
| CRF head + the consistency rule vs CRF head | new random split per seed | coarse | 5 | 76.94 [74.73, 79.15] | -0.02 [-0.06, +0.02] | -0.03 [-0.06, +0.01] | reported |
| multi-task vote + the consistency rule vs multi-task vote | leave one paper out | fine | 4 | 52.80 [43.25, 62.34] | +0.33 [-0.74, +1.40] | +0.03 [-0.36, +0.42] | decision |
| multi-task vote + the consistency rule vs multi-task vote | leave one paper out | coarse | 4 | 60.96 [55.72, 66.19] | +0.17 [-0.22, +0.57] | +0.10 [-0.22, +0.41] | reported |
| 3-model seed ensemble + the consistency rule vs 3-model seed ensemble | leave one paper out | fine | 4 | 49.44 [42.14, 56.73] | +0.62 [-0.64, +1.89] | +0.27 [-0.12, +0.66] | reported |
| 3-model seed ensemble + the consistency rule vs 3-model seed ensemble | leave one paper out | coarse | 4 | 61.30 [56.26, 66.34] | +0.20 [-0.15, +0.55] | +0.07 [-0.08, +0.23] | decision |
| CRF head + the consistency rule with retyping vs CRF head | leave one paper out | fine | 4 | 46.77 [39.47, 54.06] | +0.31 [-0.73, +1.35] | +0.08 [-1.00, +1.16] | reported |
| CRF head + the consistency rule with retyping vs CRF head | leave one paper out | coarse | 4 | 59.25 [53.75, 64.74] | +0.64 [+0.29, +1.00] | +0.46 [-0.02, +0.95] | reported |
| CRF head + the consistency rule with retyping vs CRF head | new random split per seed | fine | 5 | 77.32 [74.30, 80.34] | -0.16 [-0.32, -0.01] | -0.17 [-0.34, +0.00] | reported |
| CRF head + the consistency rule with retyping vs CRF head | new random split per seed | coarse | 5 | 76.85 [74.68, 79.02] | -0.11 [-0.26, +0.03] | -0.08 [-0.18, +0.02] | reported |
| multi-task vote + the consistency rule with retyping vs multi-task vote | leave one paper out | fine | 4 | 52.58 [42.67, 62.48] | +0.11 [-1.62, +1.84] | -0.27 [-1.82, +1.28] | reported |
| multi-task vote + the consistency rule with retyping vs multi-task vote | leave one paper out | coarse | 4 | 61.38 [56.18, 66.59] | +0.60 [-0.20, +1.39] | +0.42 [-0.36, +1.20] | reported |
| 3-model seed ensemble + the consistency rule with retyping vs 3-model seed ensemble | leave one paper out | fine | 4 | 49.38 [41.66, 57.09] | +0.56 [-2.03, +3.16] | -0.03 [-1.54, +1.48] | reported |
| 3-model seed ensemble + the consistency rule with retyping vs 3-model seed ensemble | leave one paper out | coarse | 4 | 61.93 [56.90, 66.96] | +0.83 [+0.32, +1.34] | +0.60 [-0.11, +1.31] | reported |
| document context + the consistency rule vs document context | leave one paper out | fine | 4 | 47.70 [39.45, 55.95] | +0.10 [-0.67, +0.88] | -0.03 [-0.56, +0.49] | reported |
| document context + the consistency rule vs document context | leave one paper out | coarse | 4 | 58.78 [52.16, 65.41] | +0.06 [-0.09, +0.21] | +0.05 [-0.06, +0.17] | reported |
| document-context vote + the consistency rule vs document-context vote | leave one paper out | fine | 4 | 51.47 [42.68, 60.26] | +0.45 [-0.46, +1.36] | +0.12 [-0.36, +0.59] | reported |
| document-context vote + the consistency rule vs document-context vote | leave one paper out | coarse | 4 | 61.91 [54.91, 68.90] | +0.04 [-0.19, +0.28] | -0.08 [-0.33, +0.18] | reported |

#### NER week-7 comparisons, part 2a (recall calibration, 10-seed means, GPU runs)

Δ = with the bias − without it, on the same split, GPU runs only, 10 model seeds: a model scores the mean of its 10 runs, a vote the mean of its votes over 3 disjoint seed triples. Week-7 protocol, part 2a: recall calibration, a bias on the CRF head's emission of every tag but O before decoding, chosen per held-out paper on inner folds that never score that paper (`ner_bias.py`). Pre-registered rule: a change improves on its reference only if the 95% Student-t CI of the mean micro-F1 difference lies entirely above 0. Decisions, on the 4 held-out papers: the calibrated CRF head against the CRF head, and its vote against the strongest system on unseen papers after week 5 (the multi-task vote fine-grained, the 3-model seed ensemble coarse-grained). The other rows are reported without a decision.

| Comparison | Protocol | Granularity | Pairs | Variant micro F1 [95% CI] | Δ micro [95% CI] | Δ macro [95% CI] | Role |
|---|---|---|---|---|---|---|---|
| CRF head + recall calibration vs CRF head | leave one paper out | fine | 4 | 45.87 [40.51, 51.23] | -0.59 [-1.75, +0.57] | -0.72 [-1.97, +0.53] | decision |
| CRF head + recall calibration vs CRF head | leave one paper out | coarse | 4 | 58.78 [53.63, 63.92] | +0.17 [-0.14, +0.48] | +0.04 [-0.11, +0.18] | decision |
| vote of the CRF head + recall calibration vs multi-task vote | leave one paper out | fine | 4 | 48.94 [44.31, 53.57] | -3.53 [-8.10, +1.05] | -5.00 [-12.68, +2.68] | decision |
| vote of the CRF head + recall calibration vs multi-task vote | leave one paper out | coarse | 4 | 61.02 [56.25, 65.78] | +0.23 [-1.03, +1.49] | +1.31 [-3.10, +5.72] | reported |
| vote of the CRF head + recall calibration vs 3-model seed ensemble | leave one paper out | fine | 4 | 48.94 [44.31, 53.57] | +0.13 [-2.20, +2.46] | -0.42 [-3.19, +2.36] | reported |
| vote of the CRF head + recall calibration vs 3-model seed ensemble | leave one paper out | coarse | 4 | 61.02 [56.25, 65.78] | -0.09 [-0.75, +0.58] | +0.11 [-0.58, +0.80] | decision |
| CRF head + recall calibration chosen on validation vs CRF head | leave one paper out | fine | 4 | 46.77 [39.44, 54.11] | +0.32 [-0.83, +1.46] | +0.28 [-0.80, +1.37] | reported |
| CRF head + recall calibration chosen on validation vs CRF head | leave one paper out | coarse | 4 | 58.70 [53.23, 64.17] | +0.09 [-0.24, +0.43] | +0.15 [-0.25, +0.55] | reported |

#### NER week-7 comparisons, part 2c (coarse-to-fine completion, 10-seed means, GPU runs)

Δ = with the rule − without it, fine-grained, on the same held-out paper, GPU runs only, 10 model seeds: a model scores the mean of its 10 runs, a vote the mean of its votes over 3 disjoint seed triples. Week-7 protocol, part 2c: coarse-to-fine completion (`ner_coarse_to_fine.py`, fixed before it was applied): a coarse-grained predicted entity of a type that maps to one fine-grained type in the fold's training papers becomes a fine-grained entity where the fine level predicts only O. Pre-registered rule: a change improves on its reference only if the 95% Student-t CI of the mean micro-F1 difference lies entirely above 0. Decisions: the rule on the CRF head and on the multi-task vote; the multi-task model is reported. Tested on MaterioMiner only (SOFC-Exp's slot level was not yet converted).

| Comparison | Protocol | Granularity | Pairs | Variant micro F1 [95% CI] | Δ micro [95% CI] | Δ macro [95% CI] | Role |
|---|---|---|---|---|---|---|---|
| CRF head + coarse-to-fine completion vs CRF head | leave one paper out | fine | 4 | 46.80 [39.87, 53.72] | +0.34 [-0.91, +1.59] | +0.45 [-0.44, +1.35] | decision |
| multi-task vote + coarse-to-fine completion vs multi-task vote | leave one paper out | fine | 4 | 52.59 [44.00, 61.19] | +0.12 [-1.76, +2.01] | -0.47 [-1.59, +0.65] | decision |
| multi-task (fine + coarse heads) + coarse-to-fine completion vs multi-task (fine + coarse heads) | leave one paper out | fine | 4 | 50.24 [41.82, 58.67] | +0.07 [-1.40, +1.54] | -0.26 [-0.87, +0.36] | reported |

**Reading the numbers**
* The old "Term F1 0.91" corresponds to *0.12* pairwise alignment F1. The
  v5 outputs score higher (0.27, and 0.57–0.74 typing accuracy) only because
  their top-1 rows were the gold classes injected by `type_match`.
* The pipeline's zero-shot typing (0.38; 0.43 with LLM re-ranking) beats plain
  label matching (0.36) but stays well below the supervised hybrid (0.50) and
  hybrid + LLM (0.65) baselines, which can also use labelled terms. Closing
  that gap is the first item of the roadmap.
* Taxonomy and relations are near zero at class level. The term-level
  taxonomy and relation extractors were never designed to recover MMO
  `subClassOf` edges or restrictions; see the roadmap.
* Ablations: no component changes typing significantly. The GNN adds 0.006 (0.384 vs 0.378, McNemar p = 0.69), and the whole pipeline beats the plain-SBERT baseline by 0.029 (p = 0.08). The GNN embeddings *hurt* clustering (ARI 0.035 vs 0.195 with plain SBERT). The oracle preset (gold types allowed) reaches 1.000, which is what the v5 evaluation was effectively reporting.
* NER, fixed split: plain MatSciBERT reaches micro F1 72.99 (fine) and 75.57 (coarse) over 5 seeds, against the published 69.92 / 72.32. Macro F1 is much lower (67.99 / 67.14), so the comparison favours this repo only if the paper's "averaging over ... entity types" means micro averaging, as in the authors' 2025 follow-up.
* NER across splits (`--vary-split`, a new random split per seed, micro F1; pre-registered rule: the gain holds if the 95% CI lies above the published number): fine-grained the gain is not significant: 73.91, 95% CI [69.36, 78.45] includes 69.92; coarse-grained the gain holds: 74.57, 95% CI [73.15, 75.98], above 72.32. Macro F1 across splits is 71.19 (fine) and 65.53 (coarse).
* NER out of distribution (`--leave-one-paper-out`: train on three papers, test on the fourth): micro F1 43.51 fine-grained (per paper 38.9–47.2) and 54.34 coarse-grained (per paper 50.4–60.0). These numbers are not comparable with 69.92 / 72.32.
* NER, class-weighted loss vs the baseline (paired micro-F1 difference): fine-grained no significant change (-0.24, 95% CI [-3.09, +2.61]; strict IOB2 -0.64 [-3.36, +2.08]); coarse-grained no significant change (+0.41, 95% CI [-2.31, +3.13]; strict IOB2 +0.39 [-1.76, +2.54]) across the 5 random splits. Leaving one paper out: fine-grained no significant change (+1.81, 95% CI [-4.28, +7.89]; strict IOB2 +1.75 [-3.92, +7.43]); coarse-grained no significant change (+0.91, 95% CI [-5.15, +6.98]; strict IOB2 +0.76 [-4.42, +5.94]).
* NER, CRF head vs the baseline (paired micro-F1 difference): fine-grained improves (+2.46, 95% CI [+0.39, +4.52]; strict IOB2 +0.10 [-2.08, +2.29]); coarse-grained improves (+3.58, 95% CI [+2.43, +4.74]; strict IOB2 +1.91 [+1.17, +2.65]) across the 5 random splits. Leaving one paper out: fine-grained improves (+5.94, 95% CI [+2.70, +9.18]; strict IOB2 +4.40 [+2.09, +6.72]); coarse-grained improves (+5.47, 95% CI [+0.43, +10.52]; strict IOB2 +3.03 [-1.00, +7.07]). Pre-registered SOTA check (a paired gain over the baseline and a 95% CI above the published number): met for fine-grained 76.36 [71.00, 81.73] vs 69.92; coarse-grained 78.15 [76.61, 79.69] vs 72.32, if the paper reports micro F1.
* Why the CRF helps (exploratory, not pre-registered): it never predicts an invalid I- tag, while seqeval's default (conlleval) scoring counts the baseline's stray I- tags as extra, mostly wrong entities; strict IOB2 scoring ignores them. Fine-grained: precision 71.0 → 76.7, recall 77.1 → 76.1; under strict IOB2 scoring the two are level (+0.10 [-2.08, +2.29]), so the gain is span validity alone. Dropping invalid tags from the baseline's output gives the same benefit for free: its strict F1 is 76.26 [72.52, 80.01] against the published 69.92. Coarse-grained: precision 71.4 → 77.5, recall 78.1 → 78.9; under strict IOB2 scoring the CRF still gains +1.91 [+1.17, +2.65], so part of the gain is better recognition.
* NER, multi-task (fine + coarse heads) vs the CRF head (paired micro-F1 difference): fine-grained no significant change (+2.46, 95% CI [-0.91, +5.84]; strict IOB2 +2.46 [-0.91, +5.84]); coarse-grained no significant change (+0.24, 95% CI [-2.54, +3.02]; strict IOB2 +0.24 [-2.54, +3.02]) across the 5 random splits. Leaving one paper out: fine-grained no significant change (+0.11, 95% CI [-8.56, +8.79]; strict IOB2 +0.11 [-8.56, +8.79]); coarse-grained no significant change (-1.30, 95% CI [-4.80, +2.20]; strict IOB2 -1.30 [-4.80, +2.20]).
* NER, domain-adaptive pre-training vs the CRF head (paired micro-F1 difference): fine-grained no significant change (+0.68, 95% CI [-2.55, +3.90]; strict IOB2 +0.68 [-2.55, +3.90]); coarse-grained no significant change (-0.07, 95% CI [-3.01, +2.86]; strict IOB2 -0.07 [-3.01, +2.86]) across the 5 random splits. Leaving one paper out: fine-grained no significant change (-0.91, 95% CI [-3.13, +1.31]; strict IOB2 -0.91 [-3.13, +1.31]); coarse-grained no significant change (-1.06, 95% CI [-3.25, +1.12]; strict IOB2 -1.06 [-3.25, +1.12]).
* NER, 3-model ensemble vs the CRF head (paired micro-F1 difference): fine-grained no significant change (+3.39, 95% CI [-1.35, +8.14]; strict IOB2 +3.39 [-1.35, +8.14]); coarse-grained no significant change (+1.30, 95% CI [-1.01, +3.62]; strict IOB2 +1.30 [-1.01, +3.62]) across the 5 random splits. Leaving one paper out: fine-grained no significant change (+0.11, 95% CI [-1.99, +2.20]; strict IOB2 +0.11 [-1.99, +2.20]); coarse-grained improves (+1.60, 95% CI [+0.52, +2.69]; strict IOB2 +1.60 [+0.52, +2.69]).
* NER, 3-model ensemble vs the mean of its own members on the same split (exploratory, not pre-registered; no single reference run enters): fine-grained +2.47 [+1.17, +3.77]; coarse-grained +1.80 [+0.91, +2.69] across the 5 random splits; fine-grained +2.04 [-0.23, +4.32]; coarse-grained +2.39 [+1.56, +3.21] leaving one paper out.
* NER week 3, mixed-backbone ensemble vs the 3-model seed ensemble (3-seed means, paired micro-F1 difference): fine-grained no significant change (+0.03, 95% CI [-1.80, +1.86]; strict IOB2 +0.03 [-1.80, +1.86]); coarse-grained no significant change (-0.87, 95% CI [-2.09, +0.35]; strict IOB2 -0.87 [-2.09, +0.35]) across the 5 random splits (the pre-registered decision). Leaving one paper out (reported, no decision): fine-grained no significant change (-1.34, 95% CI [-4.76, +2.09]; strict IOB2 -1.34 [-4.76, +2.09]); coarse-grained no significant change (-1.45, 95% CI [-5.74, +2.84]; strict IOB2 -1.45 [-5.74, +2.84]).
* NER week 3, mixed-backbone ensemble vs the CRF head (3-seed means, paired micro-F1 difference): fine-grained improves (+2.50, 95% CI [+0.07, +4.92]; strict IOB2 +2.50 [+0.07, +4.92]); coarse-grained no significant change (+0.93, 95% CI [-0.52, +2.38]; strict IOB2 +0.93 [-0.52, +2.38]) across the 5 random splits (reported, no decision). Leaving one paper out (reported, no decision): fine-grained no significant change (+0.71, 95% CI [-3.77, +5.18]; strict IOB2 +0.71 [-3.77, +5.18]); coarse-grained no significant change (+0.94, 95% CI [-3.20, +5.07]; strict IOB2 +0.94 [-3.20, +5.07]).
* NER week 3, 3-model seed ensemble vs the CRF head (3-seed means, paired micro-F1 difference): fine-grained improves (+2.47, 95% CI [+1.17, +3.77]; strict IOB2 +2.47 [+1.17, +3.77]); coarse-grained improves (+1.80, 95% CI [+0.91, +2.69]; strict IOB2 +1.80 [+0.91, +2.69]) across the 5 random splits (reported, no decision). Leaving one paper out (reported, no decision): fine-grained no significant change (+2.04, 95% CI [-0.23, +4.32]; strict IOB2 +2.04 [-0.23, +4.32]); coarse-grained improves (+2.39, 95% CI [+1.56, +3.21]; strict IOB2 +2.39 [+1.56, +3.21]).
* NER week 3, SciBERT backbone vs the CRF head (3-seed means, paired micro-F1 difference): fine-grained no significant change (-0.38, 95% CI [-2.63, +1.87]; strict IOB2 -0.38 [-2.63, +1.87]); coarse-grained hurts (-0.99, 95% CI [-1.71, -0.26]; strict IOB2 -0.99 [-1.71, -0.26]) across the 5 random splits (reported, no decision). Leaving one paper out (reported, no decision): fine-grained hurts (-4.83, 95% CI [-6.66, -3.00]; strict IOB2 -4.83 [-6.66, -3.00]); coarse-grained hurts (-4.52, 95% CI [-8.17, -0.87]; strict IOB2 -4.52 [-8.17, -0.87]).
* NER week 3, PubMedBERT backbone vs the CRF head (3-seed means, paired micro-F1 difference): fine-grained no significant change (-1.42, 95% CI [-5.21, +2.37]; strict IOB2 -1.42 [-5.21, +2.37]); coarse-grained hurts (-2.99, 95% CI [-4.67, -1.32]; strict IOB2 -2.99 [-4.67, -1.32]) across the 5 random splits (reported, no decision). Leaving one paper out (reported, no decision): fine-grained no significant change (-4.94, 95% CI [-13.22, +3.34]; strict IOB2 -4.94 [-13.22, +3.34]); coarse-grained no significant change (-3.75, 95% CI [-8.41, +0.90]; strict IOB2 -3.75 [-8.41, +0.90]).
* NER week 3, label semantics vs the CRF head (3-seed means, paired micro-F1 difference): fine-grained no significant change (-0.58, 95% CI [-3.41, +2.24]; strict IOB2 -0.58 [-3.41, +2.24]); coarse-grained no significant change (-0.77, 95% CI [-1.74, +0.21]; strict IOB2 -0.77 [-1.74, +0.21]) across the 5 random splits (reported, no decision). Leaving one paper out (the pre-registered decision): fine-grained no significant change (-0.16, 95% CI [-2.13, +1.80]; strict IOB2 -0.16 [-2.13, +1.80]); coarse-grained improves (+1.17, 95% CI [+0.94, +1.41]; strict IOB2 +1.17 [+0.94, +1.41]).
* NER week 3, self-training vs the CRF head (3-seed means, paired micro-F1 difference): leaving one paper out (the pre-registered decision): fine-grained no significant change (+2.91, 95% CI [-0.73, +6.56]; strict IOB2 +2.91 [-0.73, +6.56]); coarse-grained no significant change (-0.12, 95% CI [-1.97, +1.73]; strict IOB2 -0.12 [-1.97, +1.73]).
* NER week 3, multi-task (fine + coarse heads) vs the CRF head (3-seed means, paired micro-F1 difference): fine-grained no significant change (+1.87, 95% CI [-2.25, +6.00]; strict IOB2 +1.87 [-2.25, +6.00]); coarse-grained improves (+0.75, 95% CI [+0.10, +1.39]; strict IOB2 +0.75 [+0.10, +1.39]) across the 5 random splits (the pre-registered decision). Leaving one paper out (reported, no decision): fine-grained improves (+3.39, 95% CI [+0.45, +6.34]; strict IOB2 +3.39 [+0.45, +6.34]); coarse-grained no significant change (-0.60, 95% CI [-4.55, +3.34]; strict IOB2 -0.60 [-4.55, +3.34]).
* NER week 3, in-domain pre-training vs the CRF head (3-seed means, paired micro-F1 difference): fine-grained no significant change (+0.11, 95% CI [-2.74, +2.97]; strict IOB2 +0.11 [-2.74, +2.97]); coarse-grained no significant change (-0.15, 95% CI [-1.06, +0.76]; strict IOB2 -0.15 [-1.06, +0.76]) across the 5 random splits (the pre-registered decision). Leaving one paper out (reported, no decision): fine-grained no significant change (+0.14, 95% CI [-2.60, +2.88]; strict IOB2 +0.14 [-2.60, +2.88]); coarse-grained no significant change (-1.54, 95% CI [-4.34, +1.26]; strict IOB2 -1.54 [-4.34, +1.26]).
* NER week 4 (CPU runs), label-semantics vote vs the 3-model seed ensemble (paired micro-F1 difference): fine-grained hurts (-1.16, 95% CI [-2.31, -0.01]; strict IOB2 -1.16 [-2.31, -0.01]); coarse-grained hurts (-0.95, 95% CI [-1.88, -0.02]; strict IOB2 -0.95 [-1.88, -0.02]) across the 5 random splits (reported, no decision). Leaving one paper out (the pre-registered decision): fine-grained no significant change (+0.36, 95% CI [-2.53, +3.24]; strict IOB2 +0.36 [-2.53, +3.24]); coarse-grained improves (+1.63, 95% CI [+0.11, +3.16]; strict IOB2 +1.63 [+0.11, +3.16]).
* NER week 4 (CPU runs), self-training vote vs the 3-model seed ensemble (paired micro-F1 difference): leaving one paper out (the pre-registered decision): fine-grained no significant change (+3.68, 95% CI [-1.15, +8.52]; strict IOB2 +3.68 [-1.15, +8.52]); coarse-grained no significant change (-0.09, 95% CI [-3.52, +3.35]; strict IOB2 -0.09 [-3.52, +3.35]).
* NER week 4 (GPU runs), multi-task vote vs the 3-model seed ensemble (paired micro-F1 difference): fine-grained no significant change (+1.26, 95% CI [-1.08, +3.59]; strict IOB2 +1.26 [-1.08, +3.59]); coarse-grained no significant change (-0.07, 95% CI [-2.02, +1.87]; strict IOB2 -0.07 [-2.02, +1.87]) across the 5 random splits (the pre-registered decision). Leaving one paper out (reported, no decision): fine-grained no significant change (+4.09, 95% CI [-0.89, +9.07]; strict IOB2 +4.09 [-0.89, +9.07]); coarse-grained no significant change (-0.10, 95% CI [-5.04, +4.85]; strict IOB2 -0.10 [-5.04, +4.85]).
* NER week 4 (GPU runs), label semantics on the multi-task model vs the CRF head (paired micro-F1 difference): fine-grained no significant change (+0.76, 95% CI [-0.57, +2.10]; strict IOB2 +0.76 [-0.57, +2.10]); coarse-grained no significant change (+0.65, 95% CI [-0.63, +1.93]; strict IOB2 +0.65 [-0.63, +1.93]) across the 5 random splits (the pre-registered decision). Leaving one paper out (the pre-registered decision): fine-grained no significant change (+4.08, 95% CI [-1.99, +10.14]; strict IOB2 +4.08 [-1.99, +10.14]); coarse-grained no significant change (-0.01, 95% CI [-2.64, +2.63]; strict IOB2 -0.01 [-2.64, +2.63]).
* NER week 4 (GPU runs), self-training with 3 teachers vs the CRF head (paired micro-F1 difference): fine-grained improves (+1.22, 95% CI [+0.60, +1.84]; strict IOB2 +1.22 [+0.60, +1.84]); coarse-grained no significant change (+0.71, 95% CI [-0.96, +2.38]; strict IOB2 +0.71 [-0.96, +2.38]) across the 5 random splits (reported, no decision). Leaving one paper out (the pre-registered decision): fine-grained improves (+1.07, 95% CI [+0.33, +1.80]; strict IOB2 +1.07 [+0.33, +1.80]); coarse-grained no significant change (+0.85, 95% CI [-0.58, +2.29]; strict IOB2 +0.85 [-0.58, +2.29]).
* NER week 4 (GPU runs), label-semantics vote vs the 3-model seed ensemble (paired micro-F1 difference): fine-grained no significant change (-1.74, 95% CI [-3.84, +0.35]; strict IOB2 -1.74 [-3.84, +0.35]); coarse-grained no significant change (-0.60, 95% CI [-2.98, +1.78]; strict IOB2 -0.60 [-2.98, +1.78]) across the 5 random splits (reported, no decision). Leaving one paper out (reported, no decision): fine-grained no significant change (+0.51, 95% CI [-1.56, +2.59]; strict IOB2 +0.51 [-1.56, +2.59]); coarse-grained no significant change (+0.40, 95% CI [-1.36, +2.16]; strict IOB2 +0.40 [-1.36, +2.16]).
* NER week 4 (GPU runs), self-training vote vs the 3-model seed ensemble (paired micro-F1 difference): leaving one paper out (reported, no decision): fine-grained no significant change (+2.40, 95% CI [-2.53, +7.33]; strict IOB2 +2.40 [-2.53, +7.33]); coarse-grained no significant change (-0.09, 95% CI [-2.85, +2.67]; strict IOB2 -0.09 [-2.85, +2.67]).
* NER week 4 (GPU runs), multi-task (fine + coarse heads) vs the CRF head (paired micro-F1 difference): fine-grained no significant change (+1.74, 95% CI [-0.36, +3.84]; strict IOB2 +1.74 [-0.36, +3.84]); coarse-grained no significant change (-0.08, 95% CI [-1.71, +1.55]; strict IOB2 -0.08 [-1.71, +1.55]) across the 5 random splits (reported, no decision). Leaving one paper out (reported, no decision): fine-grained no significant change (+3.87, 95% CI [-2.25, +9.99]; strict IOB2 +3.87 [-2.25, +9.99]); coarse-grained no significant change (-0.68, 95% CI [-4.51, +3.15]; strict IOB2 -0.68 [-4.51, +3.15]).
* NER week 4 (GPU runs), label semantics vs the CRF head (paired micro-F1 difference): fine-grained no significant change (-0.76, 95% CI [-3.61, +2.09]; strict IOB2 -0.76 [-3.61, +2.09]); coarse-grained no significant change (-0.15, 95% CI [-1.68, +1.38]; strict IOB2 -0.15 [-1.68, +1.38]) across the 5 random splits (reported, no decision). Leaving one paper out (reported, no decision): fine-grained no significant change (+1.32, 95% CI [-0.43, +3.06]; strict IOB2 +1.32 [-0.43, +3.06]); coarse-grained no significant change (+0.07, 95% CI [-2.47, +2.61]; strict IOB2 +0.07 [-2.47, +2.61]).
* NER week 4 (GPU runs), self-training vs the CRF head (paired micro-F1 difference): leaving one paper out (reported, no decision): fine-grained no significant change (+2.14, 95% CI [-1.02, +5.29]; strict IOB2 +2.14 [-1.02, +5.29]); coarse-grained no significant change (-1.09, 95% CI [-4.31, +2.14]; strict IOB2 -1.09 [-4.31, +2.14]).
* NER, GPU minus CPU (descriptive, not a comparison of methods): the CRF head with the same seeds and splits, 3-seed means, fine-grained +0.31 [-2.38, +3.00]; coarse-grained -0.07 [-1.19, +1.05] over the 5 random splits; fine-grained -1.13 [-3.07, +0.81]; coarse-grained +0.24 [-1.52, +1.99] over the 4 held-out papers. This is why GPU runs are compared only with GPU runs.
* NER week 5 (GPU runs, 10 seeds), multi-task (fine + coarse heads) vs the CRF head (paired micro-F1 difference): fine-grained improves (+1.83, 95% CI [+0.32, +3.34]; strict IOB2 +1.83 [+0.32, +3.34]); coarse-grained no significant change (+0.71, 95% CI [-0.15, +1.56]; strict IOB2 +0.71 [-0.15, +1.56]) across the 5 random splits (the pre-registered decision). Leaving one paper out (reported, no decision): fine-grained improves (+3.72, 95% CI [+0.57, +6.87]; strict IOB2 +3.72 [+0.57, +6.87]); coarse-grained no significant change (-0.56, 95% CI [-1.91, +0.79]; strict IOB2 -0.56 [-1.91, +0.79]).
* NER week 5 (GPU runs, 10 seeds), label semantics vs the CRF head (paired micro-F1 difference): fine-grained no significant change (-0.45, 95% CI [-1.63, +0.73]; strict IOB2 -0.45 [-1.63, +0.73]); coarse-grained no significant change (+0.42, 95% CI [-1.19, +2.04]; strict IOB2 +0.42 [-1.19, +2.04]) across the 5 random splits (reported, no decision). Leaving one paper out (the pre-registered decision): fine-grained no significant change (+1.23, 95% CI [-0.96, +3.42]; strict IOB2 +1.23 [-0.96, +3.42]); coarse-grained no significant change (-0.36, 95% CI [-1.66, +0.94]; strict IOB2 -0.36 [-1.66, +0.94]).
* NER week 5 (GPU runs, 10 seeds), self-training with 3 teachers vs the CRF head (paired micro-F1 difference): fine-grained improves (+1.05, 95% CI [+0.10, +2.00]; strict IOB2 +1.05 [+0.10, +2.00]); coarse-grained improves (+1.04, 95% CI [+0.16, +1.91]; strict IOB2 +1.04 [+0.16, +1.91]) across the 5 random splits (reported, no decision). Leaving one paper out (the pre-registered decision): fine-grained no significant change (+1.74, 95% CI [-1.16, +4.64]; strict IOB2 +1.74 [-1.16, +4.64]); coarse-grained no significant change (+1.56, 95% CI [-0.88, +4.00]; strict IOB2 +1.56 [-0.88, +4.00]).
* NER week 5 (GPU runs, 10 seeds), multi-task vote vs the 3-model seed ensemble (paired micro-F1 difference): fine-grained no significant change (+1.14, 95% CI [-0.98, +3.27]; strict IOB2 +1.14 [-0.98, +3.27]); coarse-grained improves (+0.79, 95% CI [+0.17, +1.42]; strict IOB2 +0.79 [+0.17, +1.42]) across the 5 random splits (the pre-registered decision). Leaving one paper out (the pre-registered decision): fine-grained improves (+3.65, 95% CI [+0.49, +6.82]; strict IOB2 +3.65 [+0.49, +6.82]); coarse-grained no significant change (-0.32, 95% CI [-1.07, +0.44]; strict IOB2 -0.32 [-1.07, +0.44]).
* NER week 5 (GPU runs, 10 seeds), label-semantics vote vs the 3-model seed ensemble (paired micro-F1 difference): fine-grained no significant change (-0.79, 95% CI [-2.02, +0.43]; strict IOB2 -0.79 [-2.02, +0.43]); coarse-grained no significant change (+0.31, 95% CI [-1.27, +1.89]; strict IOB2 +0.31 [-1.27, +1.89]) across the 5 random splits (the pre-registered decision). Leaving one paper out (the pre-registered decision): fine-grained no significant change (+0.57, 95% CI [-1.46, +2.60]; strict IOB2 +0.57 [-1.46, +2.60]); coarse-grained no significant change (-0.63, 95% CI [-1.72, +0.46]; strict IOB2 -0.63 [-1.72, +0.46]).
* NER week 5 (GPU runs, 10 seeds), vote of self-training with 3 teachers vs the 3-model seed ensemble (paired micro-F1 difference): fine-grained no significant change (+0.98, 95% CI [-0.15, +2.10]; strict IOB2 +0.98 [-0.15, +2.10]); coarse-grained no significant change (+1.12, 95% CI [-0.36, +2.60]; strict IOB2 +1.12 [-0.36, +2.60]) across the 5 random splits (the pre-registered decision). Leaving one paper out (the pre-registered decision): fine-grained no significant change (+1.12, 95% CI [-2.71, +4.96]; strict IOB2 +1.12 [-2.71, +4.96]); coarse-grained no significant change (+0.93, 95% CI [-1.66, +3.53]; strict IOB2 +0.93 [-1.66, +3.53]).
* NER week 5 (GPU runs, 10 seeds), 3-model seed ensemble vs the CRF head (paired micro-F1 difference): fine-grained improves (+2.20, 95% CI [+1.61, +2.78]; strict IOB2 +2.20 [+1.61, +2.78]); coarse-grained improves (+1.67, 95% CI [+1.02, +2.32]; strict IOB2 +1.67 [+1.02, +2.32]) across the 5 random splits (reported, no decision). Leaving one paper out (reported, no decision): fine-grained improves (+2.36, 95% CI [+1.84, +2.88]; strict IOB2 +2.36 [+1.84, +2.88]); coarse-grained improves (+2.50, 95% CI [+1.86, +3.13]; strict IOB2 +2.50 [+1.86, +3.13]).
* NER, the published split (week-5 amendment: the release's partition and tags, micro F1 as its code scores it; 10 model seeds, GPU runs; the 95% CIs are over the seeds of one split): plain MatSciBERT, the paper's model, scores 70.19 [69.03, 71.35] fine-grained and 72.52 [71.88, 73.16] coarse-grained. Both published numbers lie within these intervals. The CRF head adds +3.12 [+1.62, +4.62] fine-grained and +2.96 [+1.30, +4.61] coarse-grained, paired by seed (strict IOB2 +1.70 [+0.21, +3.19] and +1.07 [-0.82, +2.97]), and its votes over the 3 seed triples score 74.57 and 77.00. Macro F1 is lower: 67.73 and 60.66 for plain MatSciBERT, 71.84 and 64.33 for the CRF head.
* NER, how far a 3-seed design misleads (week 5, exploratory): for each family against the CRF head on the protocol of its decision, the share of the 120 triples of the 10 seeds (the same triple on both sides) whose 3-seed comparison passes the rule, and whose mean difference has the opposite sign to the 10-seed one. Multi-task (fine + coarse heads) over the 5 random splits: fine-grained (+1.83 [+0.32, +3.34] with 10 seeds) 41% pass, 0% opposite sign; coarse-grained (+0.71 [-0.15, +1.56] with 10 seeds) 15% pass, 8% opposite sign. Label semantics over the 4 held-out papers: fine-grained (+1.23 [-0.96, +3.42] with 10 seeds) 18% pass, 8% opposite sign; coarse-grained (-0.36 [-1.66, +0.94] with 10 seeds) 2% pass, 37% opposite sign. Self-training with 3 teachers over the 4 held-out papers: fine-grained (+1.74 [-1.16, +4.64] with 10 seeds) 16% pass, 6% opposite sign; coarse-grained (+1.56 [-0.88, +4.00] with 10 seeds) 8% pass, 2% opposite sign. The SD of a split's micro F1 over the 10 seeds of the CRF head averages 1.69 fine-grained and 1.46 coarse-grained over the 5 random splits, and 2.48 fine and 2.23 coarse over the 4 held-out papers.
* NER, the coarse-grained types that keep macro F1 down (week 5, exploratory; the 3 votes of the reported system `crf_mt_ens3` over the 5 random splits, F1 and test entities per split): SDE 17.7 (3.0), MIP 17.7 (5.2), PRP 39.9 (8.0), MOD 41.7 (1.8), CPR 41.7 (2.0). Macro F1 counts each type alike: without SDE alone it would be 73.11 instead of 71.02 on the splits that have it.
* NER week 6, the screen (GPU runs, seeds 0, 100 and 200, leaving one paper out; mean paired micro-F1 difference [95% CI]; it decides only which methods get 10 seeds): DeBERTa-v3-large stops; against the CRF head fine-grained -4.38 [-7.32, -1.45], coarse-grained -8.01 [-11.72, -4.30]; its vote against the 3-model seed ensemble fine-grained -2.17 [-5.12, +0.78], coarse-grained -6.47 [-10.54, -2.39].
* NER week 6, the screen (GPU runs, seeds 0, 100 and 200, leaving one paper out; mean paired micro-F1 difference [95% CI]; it decides only which methods get 10 seeds): document context goes to stage 2; against the CRF head fine-grained +2.66 [+0.91, +4.40], coarse-grained -0.50 [-3.82, +2.83]; its vote against the 3-model seed ensemble fine-grained +2.92 [+0.02, +5.83], coarse-grained +0.29 [-4.64, +5.22].
* NER week 6, the screen (GPU runs, seeds 0, 100 and 200, leaving one paper out; mean paired micro-F1 difference [95% CI]; it decides only which methods get 10 seeds): multi-task self-training goes to stage 2; against the CRF head fine-grained +6.06 [+1.05, +11.06], coarse-grained +0.25 [-1.12, +1.62]; its vote against the 3-model seed ensemble fine-grained +5.73 [+0.70, +10.77], coarse-grained +0.24 [-1.45, +1.93]; against multi-task (fine + coarse heads) fine-grained +2.19 [-1.61, +5.98], coarse-grained +0.93 [-2.82, +4.68].
* NER week 6 (GPU runs, 10 seeds), document context vs the CRF head (paired micro-F1 difference, leaving one paper out): fine-grained no significant change (+1.14, 95% CI [-0.94, +3.22]; strict IOB2 +1.14 [-0.94, +3.22]; the pre-registered decision); coarse-grained no significant change (+0.12, 95% CI [-2.03, +2.27]; strict IOB2 +0.12 [-2.03, +2.27]; the pre-registered decision).
* NER week 6 (GPU runs, 10 seeds), document-context vote vs the multi-task vote (paired micro-F1 difference, leaving one paper out): fine-grained no significant change (-1.44, 95% CI [-3.80, +0.91]; strict IOB2 -1.44 [-3.80, +0.91]; the pre-registered decision); coarse-grained no significant change (+1.07, 95% CI [-1.23, +3.38]; strict IOB2 +1.07 [-1.23, +3.38]; reported, no decision).
* NER week 6 (GPU runs, 10 seeds), document-context vote vs the 3-model seed ensemble (paired micro-F1 difference, leaving one paper out): fine-grained improves (+2.21, 95% CI [+0.01, +4.41]; strict IOB2 +2.21 [+0.01, +4.41]; reported, no decision); coarse-grained no significant change (+0.76, 95% CI [-1.79, +3.31]; strict IOB2 +0.76 [-1.79, +3.31]; the pre-registered decision).
* NER week 7, part 1 (GPU runs; the consistency rule on the saved predictions of 10 seeds), CRF head + the consistency rule vs the CRF head (paired micro-F1 difference, leaving one paper out): fine-grained no significant change (+0.30, 95% CI [-0.35, +0.95]; strict IOB2 +0.30 [-0.35, +0.95]; the pre-registered decision); coarse-grained improves (+0.12, 95% CI [+0.01, +0.24]; strict IOB2 +0.12 [+0.01, +0.24]; the pre-registered decision).
* NER week 7, part 1 (GPU runs; the consistency rule on the saved predictions of 10 seeds), multi-task vote + the consistency rule vs the multi-task vote (paired micro-F1 difference, leaving one paper out): fine-grained no significant change (+0.33, 95% CI [-0.74, +1.40]; strict IOB2 +0.33 [-0.74, +1.40]; the pre-registered decision); coarse-grained no significant change (+0.17, 95% CI [-0.22, +0.57]; strict IOB2 +0.17 [-0.22, +0.57]; reported, no decision).
* NER week 7, part 1 (GPU runs; the consistency rule on the saved predictions of 10 seeds), 3-model seed ensemble + the consistency rule vs the 3-model seed ensemble (paired micro-F1 difference, leaving one paper out): fine-grained no significant change (+0.62, 95% CI [-0.64, +1.89]; strict IOB2 +0.62 [-0.64, +1.89]; reported, no decision); coarse-grained no significant change (+0.20, 95% CI [-0.15, +0.55]; strict IOB2 +0.20 [-0.15, +0.55]; the pre-registered decision).
* NER week 7, part 2a (GPU runs; recall calibration, its bias chosen on inner folds, 10 seeds), CRF head + recall calibration vs the CRF head (paired micro-F1 difference, leaving one paper out): fine-grained no significant change (-0.59, 95% CI [-1.75, +0.57]; strict IOB2 -0.59 [-1.75, +0.57]; the pre-registered decision); coarse-grained no significant change (+0.17, 95% CI [-0.14, +0.48]; strict IOB2 +0.17 [-0.14, +0.48]; the pre-registered decision).
* NER week 7, part 2a (GPU runs; recall calibration, its bias chosen on inner folds, 10 seeds), vote of the CRF head + recall calibration vs the multi-task vote (paired micro-F1 difference, leaving one paper out): fine-grained no significant change (-3.53, 95% CI [-8.10, +1.05]; strict IOB2 -3.53 [-8.10, +1.05]; the pre-registered decision); coarse-grained no significant change (+0.23, 95% CI [-1.03, +1.49]; strict IOB2 +0.23 [-1.03, +1.49]; reported, no decision).
* NER week 7, part 2a (GPU runs; recall calibration, its bias chosen on inner folds, 10 seeds), vote of the CRF head + recall calibration vs the 3-model seed ensemble (paired micro-F1 difference, leaving one paper out): fine-grained no significant change (+0.13, 95% CI [-2.20, +2.46]; strict IOB2 +0.13 [-2.20, +2.46]; reported, no decision); coarse-grained no significant change (-0.09, 95% CI [-0.75, +0.58]; strict IOB2 -0.09 [-0.75, +0.58]; the pre-registered decision).
* NER week 7, part 2c (GPU runs; coarse-to-fine completion on the saved predictions of 10 seeds), CRF head + coarse-to-fine completion vs the CRF head (paired micro-F1 difference, leaving one paper out): fine-grained no significant change (+0.34, 95% CI [-0.91, +1.59]; strict IOB2 +0.34 [-0.91, +1.59]; the pre-registered decision).
* NER week 7, part 2c (GPU runs; coarse-to-fine completion on the saved predictions of 10 seeds), multi-task vote + coarse-to-fine completion vs the multi-task vote (paired micro-F1 difference, leaving one paper out): fine-grained no significant change (+0.12, 95% CI [-1.76, +2.01]; strict IOB2 +0.12 [-1.76, +2.01]; the pre-registered decision).
* NER on SOFC-Exp (week 6; its official split, 26 training, 8 development and 11 test articles; 10 model seeds, GPU runs; the 95% CIs are over seeds): plain MatSciBERT 82.53 [81.84, 83.23], the CRF head 83.75 [82.85, 84.65] (macro 81.11), against the published MatSciBERT-CRF 84.77 micro and 82.39 macro. S1, the CRF head against plain MatSciBERT, paired by seed: improves (+1.22, 95% CI [+0.50, +1.93]; strict IOB2 +0.73 [-0.04, +1.51]); the CRF votes over the 3 seed triples score 85.13. Week 7, the consistency rule on the CRF head's predictions, paired by seed: 83.34 [82.56, 84.11], hurts (-0.42, 95% CI [-0.61, -0.22]; strict IOB2 -0.42 [-0.61, -0.22]). Week 7, the consistency rule with retyping on the CRF head's predictions, paired by seed: 83.38 [82.62, 84.13], hurts (-0.38, 95% CI [-0.60, -0.15]; strict IOB2 -0.38 [-0.60, -0.15]). Week 7, part 2a, recall calibration of the CRF head, its bias chosen on the validation articles, against the CRF head, paired by seed: 83.55 [82.66, 84.43], no significant change (-0.21, 95% CI [-0.58, +0.17]; strict IOB2 -0.21 [-0.58, +0.17]).
<!-- RESULTS:END -->

## Evaluation

The pipeline starts from the gold entity spans, so it is evaluated on **term
typing**. For each of the 315 graph terms, its predicted MMO class is
compared with the term's annotated class(es). `eval_f1.py` reports:

* **Typing (headline):**
  - accuracy (= P = R = F1 at full coverage) with a bootstrap 95% CI;
  - LLMs4OL Task-B set precision/recall/F1 (predicted and gold types as sets);
  - macro-F1 over majority classes and acc@1/3/5;
  - hierarchical P/R/F1 (credit for ancestors, without the single ontology root);
  - area under the risk–coverage curve, and a sweep over score thresholds.
* **Alignment rows:** pairwise P/R/F1 per threshold. A row counts only if its
  class is a gold class of the term. This replaces the old "Term F1".
* **Clustering:** ARI, V-measure, NMI and B-cubed against the majority class,
  with all-singletons and one-cluster baselines. This replaces the old
  "Concept F1".
* **Taxonomy:** term is-a edges lifted to class edges via the predicted types
  and compared with MMO `subClassOf`, together with the chance level.
* **Relations:** extracted triples lifted to class level and compared with the
  ontology restrictions, with sub-property matching (`causeOf` ⊑
  `prov:influenced`).

**NER** (`ner_baseline.py`) is evaluated end to end, with no gold spans given:
* **Metric:** entity-level seqeval F1 (exact span and type). Micro F1 is
  primary; macro, support-weighted and strict IOB2 F1 are reported alongside.
  The authors' 2025 follow-up uses micro averaging; the 2024 paper does not
  say.
* **Protocols:**
  - the fixed split with 5 model seeds;
  - a new random split per seed (5 splits);
  - leave one paper out (4 folds, out of distribution).
* **Intervals:** every result has a 95% Student-t CI over runs. Training
  variants are compared with the baseline run by run, on the same split and
  model seed.
* **Rules, fixed before the runs:**
  - a gain over the paper holds only if the CI over the 5 random splits lies
    above the published number;
  - a variant improves on the baseline only if the CI of the paired
    difference lies above 0.

**What was wrong before (v5):**
1. `align.py` auto-accepted each term's *gold* class at similarity ≥ 0.85.
2. `cluster.py` built the clusters from the gold types.
3. `rag_typing.py` overwrote the gold labels file with predictions.
4. Axiom pruning used the gold types.
5. The type level never read pipeline output: it measured that the dataset's
   labels are ontology class names.
6. Term precision only checked that the target was *some* ontology label.
7. Recall gave ancestor credit, and its "active" denominator came from the
   system's own output.

Gold labels now live in `gold_term_types.json`. A unit test checks that every
module reading that file is gated by `ORACLE_TYPES` (or is the evaluation).
Oracle runs are reported only as upper bounds.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# or: pip install https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl
```

LLM features use an OpenAI-compatible API. Set `OPENROUTER_API_KEY` (preferred,
default model `openai/gpt-4o-mini`) or `OPENAI_API_KEY`. `LLM_MODEL` and
`LLM_BASE_URL` override the defaults. Answers are cached in
`data/processed/llm_cache.json`, keyed by model.

## Running

```bash
python src/graph/run_pipeline.py                    # full pipeline (cleans stale outputs)
python src/graph/run_pipeline.py --from align.py    # resume from a stage
python src/graph/run_ablation.py                    # all ablations, LLM off, paired tests vs full
python src/graph/term_typing_baseline.py [--llm]    # cross-validated typing baselines
python src/graph/eval_f1.py --legacy-alignment-csv old/ontology_alignment.csv  # score old outputs

# NER (GPU recommended; on CPU one run takes 10-30 minutes)
python src/graph/ner_baseline.py --granularity both --seeds 0 1 2 3 4      # fixed split
python src/graph/ner_baseline.py --granularity both --vary-split           # new split per seed
python src/graph/ner_baseline.py --granularity both --leave-one-paper-out --seeds 0
python src/graph/ner_baseline.py --granularity fine --vary-split --crf     # or --class-weights
python src/graph/ner_baseline.py --compare results                         # paired variant vs baseline
```

NER runs write `data/processed/ner/ner_results_<granularity>[_<protocol>][_<variant>].json`
after every run and skip finished runs when restarted.

Configuration is read from environment variables (see `src/graph/config.py`):
`ORACLE_TYPES`, `USE_GNN_EMBEDDINGS`, `USE_BIDIRECTIONAL`, `USE_COMBINED_SCORING`,
`USE_HEARST_PATTERNS`, `USE_DEP_PARSING`, `CORPUS_AUGMENT`, `LLM_ALIGNMENT`,
`LLM_RELATIONS`, `RAG_TYPING`, `EMBEDDER_MODEL`, `SEED`, `PROCESSED_DIR`.
Runs are deterministic: seeded RNGs, single-threaded GNN training and a fixed
`PYTHONHASHSEED`.

## Development

```bash
pip install -r requirements-dev.txt   # no torch needed
ruff check src tests
pytest
```

CI (GitHub Actions) runs ruff and the test suite on every push and pull request.
The CRF tests (`tests/test_ner_crf.py`) need torch, so they are skipped where it
is missing, as in CI; run them locally after installing torch.

After adding or changing files in `results/`, regenerate the results sections of
this README and the ROADMAP with `python src/graph/report_results.py`. The test
suite fails while they are out of date; `--check` prints the difference. For NER
results, first run `python src/graph/ner_seedmeans.py`, which rewrites the
week-3 comparisons of 3-seed means (`results/ner_week3_comparisons.json`).

## Next steps

See [docs/ROADMAP.md](docs/ROADMAP.md) for the prioritised plan to raise the
metrics and to compete with the published state of the art.

For NER, the reported system is decided on 10 model seeds per split (week 5,
on Kaggle's free T4 GPUs; `src/graph/kaggle_jobs.py`). Fine-grained it is the
3-model seed ensemble of the CRF head (`crf_ens3`) on random splits and the
multi-task vote (`crf_mt_ens3`) on unseen papers; coarse-grained it is the
other way round. Over 5 random splits it beats the published fine-grained
number under micro, macro and weighted F1, and the coarse-grained number
under micro and weighted F1 but not macro F1.

On the split that the MaterioMiner release ships, scored like its code,
plain MatSciBERT reproduces the published 69.92 and 72.32 within its 10-seed
intervals, and the CRF head beats both by about 3 points.

Ten seeds per split also re-tested the gains of weeks 3 and 4, which used 3:
- fine-grained multi-task training improves on the CRF head over the random
  splits (+1.83 [+0.32, +3.34]);
- the coarse-grained gains of week 3 (multi-task in distribution, label
  semantics on unseen papers) and the fine-grained self-training gain of
  week 4 on unseen papers do not stand;
- on the same runs, a 3-seed comparison passes for the 1.8-point multi-task
  gain in only 41% of the 120 seed triples, and up to 37% of the triples
  give the opposite sign to the 10-seed difference.

New NER experiments should use the CRF head (`--crf`) and 10 seeds per split
before a claim. GPU runs are compared only with GPU runs.

The work aims at a methods paper, with an evaluation paper if every option
fails ([docs/PAPER.md](docs/PAPER.md); a draft of the sections that hold
either way is in [docs/PAPER_DRAFT.md](docs/PAPER_DRAFT.md)). Where it
stands (ROADMAP §4, weeks 6 and 7):
1. Methods for unseen papers, screened with 3 seeds and decided with 10.
   DeBERTa-v3-large stopped at the screen. Document context passed the
   screen but neither 10-seed decision. Multi-task self-training passed the
   screen (+6.06 fine-grained) and its 10-seed decisions run next; its repeat
   on SOFC-Exp is fixed before them (an SOFC corpus of 40,000 sentences, cut
   into words as SOFC-Exp's training articles are).
2. SOFC-Exp, the second dataset. The CRF head's gain repeats there (+1.22
   [+0.50, +1.93]). Its second level of types (slot types, converted in
   week 7) lets multi-task training be repeated there (part 2d, next). Its
   test articles turn out to be cut into words differently from its
   training articles (ChemDataExtractor for training, Java's BreakIterator
   for test); part 2e measures what that costs.
3. Week 7: document-level label consistency and coarse-to-fine completion
   give no claim (the first hurts on SOFC-Exp). Recall calibration (a bias
   on the CRF head's emissions, chosen on inner folds) gives no claim
   either: the entities the CRF head misses on unseen papers are not near
   its decision threshold, so a decoding change cannot recover them.
   Mention replacement is screened next.
4. Why unseen papers are hard (exploratory, no GPU; `ner_mentions.py`):
   81% of a held-out paper's fine-grained entities are names that its
   three training papers never annotate with that type, against 35% on
   random splits, and the CRF head finds 35% of them (59% on random
   splits). Every gain that holds on unseen papers, and the CRF head's own
   gain, is a gain on these unseen names, so a method for unseen papers has
   to improve on them. Scored more loosely (`ner_matching.py`, the criteria
   of SemEval-2013 Task 9.1), the model finds almost every coarse-grained
   entity of a new paper (overlap F1 90.7 against 93.4 on random splits);
   most of the strict gap lies in span boundaries and types that the
   papers' annotators set differently.
5. Then the combinations of the methods that pass, and the rare
   coarse-grained types that keep macro F1 down.

## License

Apache-2.0 (see `LICENSE`). The MaterioMiner dataset and the MMO are © their
authors (CC BY 4.0).
