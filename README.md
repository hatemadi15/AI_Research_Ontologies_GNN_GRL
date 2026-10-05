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
> checked whether a term received the right class. Details and the
> before/after comparison are in [Evaluation](#evaluation) and
> [docs/ROADMAP.md](docs/ROADMAP.md). All numbers below are leakage-free.

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

## Results

<!-- RESULTS:START -->
All numbers are leakage-free and come from the JSON files in [`results/`](results/)
(CPU, `SEED=0`; LLM = `openai/gpt-4o-mini` via OpenRouter). Gold term spans
are given to the pipeline and the typing baselines. "Gold types used" marks
outputs whose predictions had access to the gold labels: those rows are
reported only to show the size of the leak.

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

#### NER (MatSciBERT, MaterioMiner protocol, seqeval entity F1; mean ± population SD over seeds)

| Granularity | Classes | Published (5 seeds) | This repo, CPU | Strict IOB2 | Seeds |
|---|---|---|---|---|---|
| fine | 27 | 69.92 | 73.29 ± 1.57 | 75.23 | 4 |
| coarse | 27 | 72.32 | 75.39 ± 1.08 | 77.00 | 4 |

> **Still running when this snapshot was taken:** the last MatSciBERT NER seed per granularity (5 per granularity are planned). These rows will be added when the runs finish.

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
* NER: our MatSciBERT reproduction reaches 73.29 (fine, 4 seeds) and 75.39 (coarse, 4 seeds) test F1, against the published 69.92 / 72.32. All seeds share one random split (the paper's split is not published), so before calling this SOTA, run all 5 seeds with `--vary-split` and `--leave-one-paper-out` and test significance.
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
python src/graph/ner_baseline.py --granularity both --seeds 0 1 2 3 4   # GPU recommended
python src/graph/eval_f1.py --legacy-alignment-csv old/ontology_alignment.csv  # score old outputs
```

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

## Next steps

See [docs/ROADMAP.md](docs/ROADMAP.md) for the prioritised plan to raise the
metrics and to compete with the published state of the art.

## License

Apache-2.0 (see `LICENSE`). The MaterioMiner dataset and the MMO are © their
authors (CC BY 4.0).
