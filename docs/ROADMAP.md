# Roadmap: from honest baselines to state of the art

This document says where the project stands after the evaluation was made
leakage-free, what "state of the art" means for this data, and which
experiments are most likely to move each metric. Numbers marked *(est.)* are
expectations from the literature, not results.

## 1. Where we stand (measured, leakage-free)

<!-- STANDING:START -->
| Task | Best leakage-free result in this repo | Reference |
|---|---|---|
| NER, fine / coarse (MatSciBERT, MaterioMiner protocol) | fine: micro 72.99 [70.86, 75.12] (macro 67.99) on the fixed split, 73.91 [69.36, 78.45] over 5 new splits, 43.51 leaving one paper out; coarse: micro 75.57 [74.14, 77.01] (macro 67.14) on the fixed split, 74.57 [73.15, 75.98] over 5 new splits, 54.34 leaving one paper out | 69.92 / 72.32 (Kumar et al. 2024; averaging not stated) |
| Term typing, 315 terms, pipeline (zero-shot) | accuracy 0.384 (LLM on: 0.429) | majority class 0.083 |
| Term typing, 5-fold CV baselines | hybrid 0.496; hybrid + LLM 0.654 (set F1 0.598) | LLMs4OL 2025 MatOnto best F1 0.667 (different dataset) |
| Clustering vs gold classes | ARI 0.035, B-cubed F1 0.183 (all-singletons baseline 0.475) | – |
| Taxonomy, class level | edge F1 0.006; precision vs closure 0.027 (chance 0.013) | LLMs4OL 2025 MatOnto best F1 0.662 (different dataset) |
| Relations, class level vs 70 restrictions | F1 0.000 | no text-level gold exists |

> **Still running when this snapshot was taken:** the remaining NER variant runs (class weights: coarse leave-one-paper-out; CRF: all). These rows will be added when the runs finish.

Full tables: [README](../README.md#results).
<!-- STANDING:END -->

## 2. What "SOTA" means here

| Task | Published reference | Comparable today? |
|---|---|---|
| **NER on MaterioMiner** (span + class, seqeval entity F1) | MatSciBERT 69.92 fine-grained (top-27 classes) / 72.32 coarse-grained, random 65/15/20 split, 5 seeds, averaging not stated ([Kumar et al., Sci Data 2024](https://www.nature.com/articles/s41597-024-03926-5)). Out-of-distribution (different protocol): roughly 60 micro F1 for MatSciBERT (read from a figure), 36% above GPT-4 in-context learning ([Sci Rep 2025](https://www.nature.com/articles/s41598-025-03619-y)) | **Yes**, via `ner_baseline.py`. The paper's split is not published, so compare means with 95% CIs over several splits (§1). |
| **Term typing**, gold spans given (MaterioMiner → MMO) | No published number. Closest public task: LLMs4OL 2025 Task B on MatOnto, best F1 0.667 (IRIS) ([overview](https://www.tib-op.org/ojs/index.php/ocp/article/view/2913)) | Our baselines establish the first numbers. MatOnto is a different dataset. |
| **Taxonomy discovery** | LLMs4OL 2025 Task C on MatOnto, best F1 0.662 (SBU-NLP) | Not yet: the pipeline builds term-level is-a edges, which is a different task. |
| **Relation extraction** | No text-level gold in MaterioMiner | No. Only class-level checks against 70 ontology restrictions are possible. |

The previous claim of "Type F1 0.979 exceeds OAEI 2024 SOTA (~0.95)" compared
a property of the dataset (NER labels are ontology class names) with a
different benchmark (OAEI, which matches ontologies to ontologies). It should
not be reused.

## 3. Priorities

### P0: Keep the evaluation honest (partly done in this PR)
- Done: leakage-free typing, alignment, clustering, taxonomy and relation
  metrics; the gold file is isolated behind `ORACLE_TYPES`; a unit test
  enforces that; deterministic runs; CI; paired exact McNemar tests
  (`metrics.mcnemar_exact`) for every ablation preset.
- Next: report every headline number as mean ± std over seeds or folds plus a
  bootstrap CI (the test sets are tiny: 96 NER test sentences, 315 terms). Use
  paired tests (McNemar for typing, paired bootstrap over sentences for NER)
  whenever claiming an improvement.
- Next: write one results JSON per run (git SHA, config, seed, model ids) and
  never tune thresholds or prompts on test folds. `FIXED_THRESHOLD` and the
  baseline hyper-parameters are fixed in advance for this reason.

### P1: Beat the published NER numbers (69.92 / 72.32)
The pipeline currently consumes gold spans. A real end-to-end system needs NER,
and NER is the only task with a published SOTA on this data.
1. **Check that the reproduction holds across splits before claiming SOTA.**
   Result with a new random split per seed (`--vary-split`, 5 splits, micro
   F1). The rule, fixed before the runs: the gain holds only if the 95% CI
   lies above the published number.
   - Coarse-grained: 74.57, 95% CI [73.15, 75.98] vs 72.32. The gain holds.
   - Fine-grained: 73.91, 95% CI [69.36, 78.45] vs 69.92. Not significant;
     single splits range from 70.9 to 80.1.

   Caveats:
   - The 2024 paper does not say how its F1 is averaged. The authors' 2025
     follow-up uses micro F1, as we do. Our macro F1 is far below the
     published number for coarse (65.5 across splits) and close to it for
     fine (71.2).
   - Each test split has only about 96 sentences.

   Steps:
   - Done: 5 seeds × 2 granularities on the fixed split and on 5 new splits,
     with micro, macro and weighted F1 and 95% CIs (`summary` in the results
     JSON).
   - Done: `--leave-one-paper-out` (§1). Micro F1 averaged over the four
     held-out papers is 43.5 fine-grained and 54.3 coarse-grained, 20–30
     points below the random splits.
   - Next: compare every new system with this baseline on the same splits
     (paired bootstrap over test sentences). A fine-grained SOTA claim needs
     a significant gain over the baseline, not just a higher mean.

   A CPU run takes 9–25 minutes; on one GPU the whole protocol takes minutes.
2. **Hierarchy-aware multi-task training.** Coarse labels are the fine labels
   propagated up the MMO taxonomy, so a joint fine + coarse head (or a loss
   over the label hierarchy) shares signal with the 85 fine classes that have
   ≤ 3 examples. *(est. +1–3 F1 fine-grained)*
3. **Class-balanced or focal loss** (`--class-weights` is implemented) for the
   minority classes the paper names as the main weakness. *(est. +0.5–2)*
4. **CRF or span-based decoding** to cut span errors, which are about 10% of
   errors in the 2025 study. *(est. +0.5–1.5)*
5. **Domain-adaptive pre-training.** Continue MLM on the 19k PubMed sentences
   already in the repo, plus open-access fatigue papers, before fine-tuning.
   *(est. +1–3)*
6. **Label semantics.** Bi-encoder span–type matching (BINDER, GLiNER-style)
   with the 420 MMO `skos:definition`s as type descriptions. This helps rare
   and unseen classes, and it supports the full 177-class setting, not just
   the top 27.
7. **Semi-supervised data.** Self-train on PubMed with a confidence threshold.
   Generate LLM silver labels with ontology-definition prompts, filter them by
   agreement with the teacher model, and add them as weak supervision.
8. **Ensembles.** Average across seeds and backbones (MatSciBERT, SciBERT,
   PubMedBERT, DeBERTa-v3-large); this usually gives a reliable +1–2.
9. **Track out-of-distribution results** with `--leave-one-paper-out` for
   every change. It is cheap and closer to real use than random sentence
   splits. The baseline drops from about 74 to 44 (fine-grained) and 54
   (coarse-grained) on an unseen paper. That gap is larger than any
   in-distribution gain on this list, so items 5–7 should be judged on this
   protocol as well.

### P2: Term typing (LLMs4OL Task B style)
The cross-validated hybrid + LLM baseline is far ahead of the pipeline's
zero-shot typing (§1). Next steps:
1. **Make `align.py` use the baseline's components**: definition scoring, and
   kNN over labelled seed terms when any are available. Keep the zero-shot
   path for unseen ontologies.
2. **Context-aware term embeddings**: encode the term inside its sentence
   (MatSciBERT mean-pooled span) rather than the bare string. Many errors are
   short ambiguous strings ("data", "fatigue", "diameter").
3. **Contrastive bi-encoder, then cross-encoder.** Train on (term + context) ↔
   (label + definition) pairs, with hard negatives taken from sibling classes
   in the MMO tree. This was the IRIS recipe, which won MatOnto in 2025.
4. **Hierarchical decoding**: pick the coarse class first, then a child.
   Hierarchical F1 is already reported, which makes "near misses" visible.
5. **Better LLM prompting**:
   - retrieve demonstrations from labelled terms only;
   - ensemble several prompts and models (DREAM-LLMs style deliberation);
   - abstain on low agreement;
   - note the contamination caveat: 213 of the 420 MMO definitions were
     generated by GPT-4 (163 of them with manual adjustments).
6. **Abstention.** The risk–coverage curve shows precision > 0.8 at about 15%
   coverage. Calibrate an abstention threshold on validation folds for
   high-precision ontology population.

### P3: Make the graph learning matter
Diagnosis from this PR:
- The link-prediction GNN had collapsed to rank-1 embeddings. Its cosine
  similarities were meaningless; they drove clustering, taxonomy and 2,207
  "GNN relations" in v5.
- After the fix (residual projection, lr 1e-3, chosen on validation
  link-prediction loss), the GNN contributes only through term-side score
  smoothing, and the ablations show that this does not matter:
  - typing accuracy is 0.384 with it and 0.378 without (McNemar p = 0.69);
  - clustering on the GNN embeddings is far worse than on plain SBERT
    (ARI 0.035 vs 0.195).
- Link-prediction embeddings encode which terms co-occur, not which class
  they belong to. The steps below train the graph model for the typing
  objective instead.

Next:
1. **Heterogeneous term–class graph.**
   - Edges: term–term PMI edges, class–class `subClassOf` edges, and term–class
     edges for training-fold terms.
   - Train for node classification or link prediction to classes (R-GCN,
     GATv2 with edge weights) and evaluate inside the same cross-validation
     folds as the baselines. This is the natural "GNN for ontology learning"
     experiment.
2. **Use the PMI weights** in message passing (`SAGEConv` ignores them today).
   Compare co-occurrence windows: sentence, paragraph, document.
3. **Pre-train contrastively** (GraphCL/DGI), then fine-tune for typing.

### P4: Taxonomy discovery (class level)
The current term-level taxonomy (most similar more-general term plus Hearst
patterns) is not designed to recover MMO `subClassOf`, and it scores near
chance at class level. For a Task C-style result:
1. Predict hypernym pairs among the MMO classes that occur in the corpus.
   Inputs are label, definition and a few typed example terms; use an LLM or a
   fine-tuned cross-encoder, cross-validated over edges.
2. Decode a consistent hierarchy with a maximum spanning arborescence or a
   transitive-reduction constraint.
3. Report edge F1, ancestor F1 and the inverted-edge rate (already
   implemented in `metrics.taxonomy_edge_metrics`).

### P5: Relations
1. Annotate a relation layer for the four papers. The MaterioMiner authors
   plan this using the MMO object properties. Even 200 relation mentions would
   allow a real text-level relation-extraction benchmark.
2. Meanwhile, use distant supervision from the 70 class-level restrictions:
   - prompt an LLM with the property inventory and OWL domain/range;
   - score at class level (`eval_relations.py`).
3. Drop relation types without an ontology property, such as `relatedTo`, or
   map them to `associatedWith`.

### P6: Engineering
- YAML or Hydra configs instead of environment variables. Cache embeddings
  across stages and presets.
- Run the large-embedder presets (`domain_specter`, `domain_scibert`) and the
  NER script on GPU.
- Use the real `emmo.ttl` and `bfo.owl` files in `cross_domain_align.py`
  instead of hand-written upper-level class lists, with a gold mapping through
  PMDco where it exists.

## 4. Suggested order (4 weeks)

| Week | Work | Exit criterion |
|---|---|---|
| 1 | Class weights and CRF on the reproduced baseline (the baseline on all three split protocols is done, §1) | A paired, significant gain over the baseline on the varied splits, also measured leaving one paper out |
| 2 | Domain-adaptive pre-training + hierarchy-aware multi-task NER, ensembles | Beats the reproduced baseline on the same splits (paired bootstrap, p < 0.05) and 69.92 / 72.32 on average |
| 3 | Context-aware bi-encoder/cross-encoder typing + definition kNN in `align.py`; heterogeneous GNN | Typing accuracy above the hybrid + LLM baseline (§1) |
| 4 | Class-level taxonomy (LLM or cross-encoder + arborescence); relation annotation pilot | First class-level taxonomy F1 above chance; relation guidelines |
