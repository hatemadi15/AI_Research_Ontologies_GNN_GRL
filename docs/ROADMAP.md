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
| NER with the CRF head (`--crf`) | fine: micro 76.36 [71.00, 81.73] over 5 new splits, 49.44 leaving one paper out; coarse: micro 78.15 [76.61, 79.69] over 5 new splits, 59.82 leaving one paper out | same; the in-distribution fine-grained gain over the baseline disappears under strict IOB2 scoring |
| Term typing, 315 terms, pipeline (zero-shot) | accuracy 0.384 (LLM on: 0.429) | majority class 0.083 |
| Term typing, 5-fold CV baselines | hybrid 0.496; hybrid + LLM 0.654 (set F1 0.598) | LLMs4OL 2025 MatOnto best F1 0.667 (different dataset) |
| Clustering vs gold classes | ARI 0.035, B-cubed F1 0.183 (all-singletons baseline 0.475) | – |
| Taxonomy, class level | edge F1 0.006; precision vs closure 0.027 (chance 0.013) | LLMs4OL 2025 MatOnto best F1 0.662 (different dataset) |
| Relations, class level vs 70 restrictions | F1 0.000 | no text-level gold exists |

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
- Done for NER: mean ± SD and 95% CIs over seeds, splits and held-out
  papers, and paired differences by split and model seed
  (`ner_baseline.py --compare`).
- Next: report every other headline number as mean ± std over seeds or folds
  plus a CI (the test sets are tiny: 96 NER test sentences, 315 terms). Use
  paired tests (McNemar for typing, paired differences by split and seed for
  NER) whenever claiming an improvement.
- Next: write one results JSON per run (git SHA, config, seed, model ids) and
  never tune thresholds or prompts on test folds. `FIXED_THRESHOLD` and the
  baseline hyper-parameters are fixed in advance for this reason.

### P1: Beat the published NER numbers (69.92 / 72.32)
The pipeline currently consumes gold spans. A real end-to-end system needs NER,
and NER is the only task with a published SOTA on this data.

Status: with the CRF head (item 4), both granularities pass the
pre-registered check across 5 random splits, if the paper reports micro F1:
- fine-grained: 76.36 [71.00, 81.73] vs 69.92;
- coarse-grained: 78.15 [76.61, 79.69] vs 72.32.

Week 2 tested three changes against the CRF head: multi-task training
(item 2), domain-adaptive pre-training on the PubMed sentences (item 5) and
a 3-model ensemble (item 8). None passes the pre-registered rule. The
ensemble scores highest (79.76 fine-grained, 79.45 coarse-grained), beats
the mean of its own members on every split, and is the only change that
improves on unseen papers (coarse-grained, +1.60 [+0.52, +2.69]). But one
reference model varies by several F1 points from run to run, which hides
gains of 1–3 points in a comparison against a single run per split.

The largest remaining gap is out of distribution (item 9).

Next, after week 2, in this order (week 3 runs items 1–5; protocol in §4):
1. **A comparison that can detect 1–3 points.** Compare the mean of 3 model
   seeds per split for the variant against the same for the reference,
   instead of single runs. The ensemble members `crf_m0`, `crf_m100` and
   `crf_m200` already give such a CRF reference on the 5 random splits and
   the 4 papers. Fix the design (seeds, splits, rule) before the runs, as
   in weeks 1 and 2.
2. **Ensembles for the reported system.** A seed ensemble only costs
   training and inference time, and it gained on every split against its
   members. Next, mix backbones (SciBERT, PubMedBERT, DeBERTa-v3), whose
   errors should differ more than those of seeds.
3. **Out of distribution.** Leaving one paper out costs 18–27 points. Of
   the week-2 changes only the ensemble narrows it, by 1.6 points
   coarse-grained. Label semantics (item 6) and semi-supervised data
   (item 7) target it directly.
4. **Multi-task again, under the new design.** Its fine-grained difference
   was positive on all five splits (+2.46 on average) but not significant.
5. **Domain-adaptive pre-training only on in-domain text.** PubMed did not
   help; the open-access fatigue papers of item 5 are untested. Low
   priority for its cost.
6. **A GPU.** A run takes 13–27 minutes on 2 CPU threads; on one GPU it
   takes about a minute, which makes more seeds and larger backbones
   (DeBERTa-v3-large) practical.

The P1 items, with what has been measured so far:

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
   - Done: paired comparison with this baseline on the same splits and model
     seeds (`--compare`, `results/ner_comparisons.json`). A fine-grained SOTA
     claim needs a significant gain over the baseline, not just a higher
     mean; the CRF head has one (item 4).

   A CPU run takes 9–25 minutes; on one GPU the whole protocol takes minutes.
2. **Hierarchy-aware multi-task training.** Coarse labels are the fine labels
   propagated up the MMO taxonomy, so a joint fine + coarse head (or a loss
   over the label hierarchy) shares signal with the 85 fine classes that have
   ≤ 3 examples. *(est. +1–3 F1 fine-grained)*

   Measured with `--crf --multi-task`: one MatSciBERT encoder with a CRF head
   per granularity, against the CRF head on the same splits and model seeds.
   No significant change in any of the four paired comparisons, so it is not
   adopted.

   | Micro F1 Δ [95% CI] | Fine-grained | Coarse-grained |
   |---|---|---|
   | 5 random splits | +2.46 [−0.91, +5.84] | +0.24 [−2.54, +3.02] |
   | Leaving one paper out | +0.11 [−8.56, +8.79] | −1.30 [−4.80, +2.20] |

   All five fine-grained differences across random splits are positive
   (+0.25 to +7.17), but one split (+7.17) gives most of the mean and the
   interval includes 0. On unseen papers the fine-grained differences range
   from −6.21 to +4.89.
3. **Class-balanced or focal loss** for the minority classes the paper names
   as the main weakness. *(est. +0.5–2)*

   Measured with `--class-weights` (inverse square-root label frequency): no
   significant change in any of the four paired comparisons.

   | Micro F1 Δ [95% CI] | Fine-grained | Coarse-grained |
   |---|---|---|
   | 5 random splits | −0.24 [−3.09, +2.61] | +0.41 [−2.31, +3.13] |
   | Leaving one paper out | +1.81 [−4.28, +7.89] | +0.91 [−5.15, +6.98] |

   Macro F1, which gives rare classes more weight, does not change
   significantly either. Not adopted. A focal loss targets the same
   imbalance, so it now ranks below items 2 and 5.
4. **CRF or span-based decoding** to cut span errors, which are about 10% of
   errors in the 2025 study. *(est. +0.5–1.5)*

   Measured with `--crf`: a linear-chain CRF over first-subword emissions
   whose transitions forbid invalid IOB2 sequences. It improves all four
   paired comparisons, by more than the estimate.

   | Δ [95% CI] | Fine-grained | Coarse-grained |
   |---|---|---|
   | Micro F1, 5 random splits | +2.46 [+0.39, +4.52] | +3.58 [+2.43, +4.74] |
   | Strict IOB2 micro F1, 5 random splits | +0.10 [−2.08, +2.29] | +1.91 [+1.17, +2.65] |
   | Micro F1, leaving one paper out | +5.94 [+2.70, +9.18] | +5.47 [+0.43, +10.52] |
   | Strict IOB2 micro F1, leaving one paper out | +4.40 [+2.09, +6.72] | +3.03 [−1.00, +7.07] |

   Under strict IOB2 scoring, the in-distribution fine-grained gain vanishes.
   It comes from no longer emitting stray I- tags, which seqeval's default
   scoring counts as entities. The baseline's own strict F1 is already 76.26
   [72.52, 80.01]. The coarse-grained gain across random splits and the
   fine-grained gain on an unseen paper survive strict scoring.

   Next:
   - Use `--crf` for every new NER experiment, and judge items 2 and 5–8
     against the CRF results rather than the plain baseline.
   - Report strict micro F1 next to the default.
   - For models without a CRF, constrained decoding (repair or drop stray
     I- tags) gives the span-validity part of the gain for free.
   - Span-based decoding remains untested.
5. **Domain-adaptive pre-training.** Continue MLM on the 19k PubMed sentences
   already in the repo, plus open-access fatigue papers, before fine-tuning.
   *(est. +1–3)*

   Pre-training is done (`dapt_mlm.py`, PubMed sentences only): 3 epochs on
   18,353 sentences, after dropping 1 duplicate and 13 sentences that share a
   word 8-gram with MaterioMiner. Held-out MLM perplexity fell from 12.17 to
   5.51 (`results/dapt_mlm.json`).

   Fine-tuned with `--crf` from this checkpoint, against the CRF head on the
   same splits and model seeds: no significant change in any of the four
   paired comparisons, so it is not adopted.

   | Micro F1 Δ [95% CI] | Fine-grained | Coarse-grained |
   |---|---|---|
   | 5 random splits | +0.68 [−2.55, +3.90] | −0.07 [−3.01, +2.86] |
   | Leaving one paper out | −0.91 [−3.13, +1.31] | −1.06 [−3.25, +1.12] |

   The lower perplexity on PubMed text did not carry over to MaterioMiner;
   on unseen papers 6 of the 8 differences are negative. The second corpus
   named above, open-access fatigue papers, was not used.
6. **Label semantics.** Bi-encoder span–type matching (BINDER, GLiNER-style)
   with the 420 MMO `skos:definition`s as type descriptions. This helps rare
   and unseen classes, and it supports the full 177-class setting, not just
   the top 27.
7. **Semi-supervised data.** Self-train on PubMed with a confidence threshold.
   Generate LLM silver labels with ontology-definition prompts, filter them by
   agreement with the teacher model, and add them as weak supervision.
8. **Ensembles.** Average across seeds and backbones (MatSciBERT, SciBERT,
   PubMedBERT, DeBERTa-v3-large); this usually gives a reliable +1–2.

   Measured with `ner_ensemble.py`: an entity-level majority vote of three
   CRF models (model seeds K, K + 100, K + 200), against the CRF head on the
   same splits. No significant change across the 5 random splits, so under
   the pre-registered rule it does not improve on the CRF head.

   | Micro F1 Δ [95% CI] | Fine-grained | Coarse-grained |
   |---|---|---|
   | 5 random splits | +3.39 [−1.35, +8.14] | +1.30 [−1.01, +3.62] |
   | Leaving one paper out | +0.11 [−1.99, +2.20] | +1.60 [+0.52, +2.69] |

   The ensemble has the highest scores so far: 79.76 [76.14, 83.38]
   fine-grained and 79.45 [77.82, 81.09] coarse-grained. The CRF head it is
   compared with is one model per split, and one model is noisy: re-running
   it with 2 threads moves single fine-grained splits by −3.28 to +6.35.
   Against the mean of its own three members on the same split, which no
   single run dominates, the vote gains +2.47 [+1.17, +3.77] fine-grained
   and +1.80 [+0.91, +2.69] coarse-grained, on every split (exploratory,
   not pre-registered).

   On unseen papers the ensemble reaches 49.55 fine-grained and 61.42
   coarse-grained. Coarse-grained, it is the only week-2 change that
   improves on the CRF head there, and it beats its own members on all
   four papers (+2.39 [+1.56, +3.21]). Fine-grained it gains +2.04
   [−0.23, +4.32] over its members, also positive on every paper.
9. **Track out-of-distribution results** with `--leave-one-paper-out` for
   every change. It is cheap and closer to real use than random sentence
   splits. The baseline drops from about 74 to 44 (fine-grained) and 54
   (coarse-grained) on an unseen paper; the CRF head lifts these to 49 and
   60, and the 3-model ensemble to 50 and 61. That gap is larger than any in-distribution gain on this list, so
   items 5–7 should be judged on this protocol as well.

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

## 4. Suggested order (5 weeks)

| Week | Work | Exit criterion |
|---|---|---|
| 1 | Done: class weights (no significant change) and CRF (significant gain in all four comparisons), P1 items 3–4 | Met by the CRF head |
| 2 | Done: multi-task NER (fine + coarse heads), domain-adaptive pre-training and a 3-model ensemble, all with `--crf` (P1 items 2, 5 and 8; protocol below) | Not met: no 95% CI lies above 0 against the CRF head on the random splits; the ensemble comes closest (+3.39 fine-grained, +1.30 coarse-grained) and is the only change that improves on unseen papers (coarse-grained) |
| 3 | NER, P1 "Next, after week 2" items 1–5: comparisons of 3-seed means, a mixed-backbone ensemble, label semantics and self-training for unseen papers, multi-task again, in-domain pre-training (protocol below) | Per variant, as in the protocol below |
| 4 | Context-aware bi-encoder/cross-encoder typing + definition kNN in `align.py`; heterogeneous GNN | Typing accuracy above the hybrid + LLM baseline (§1) |
| 5 | Class-level taxonomy (LLM or cross-encoder + arborescence); relation annotation pilot | First class-level taxonomy F1 above chance; relation guidelines |

### Week 2 protocol (fixed before any run)

The reference is the CRF head (`--crf`) on the same splits and model seeds:
the 5 random splits (`--vary-split`) and the 4 held-out papers. Each variant
changes one thing; every other setting is that of the CRF runs.

1. **Multi-task training** (`--crf --multi-task`, P1 item 2).
   - One MatSciBERT encoder with a CRF head per granularity. The loss is the
     fine NLL plus the coarse NLL, weighted equally.
   - Each head trains on exactly the sentences and tags of its single-task
     run. The fine and coarse files tokenise 15 of the 476 sentences
     differently; those enter as two examples, each labelled for one head.
   - Each granularity's test score uses the epoch with its best validation
     F1, as in the single-task runs. One run gives both granularities.
2. **Domain-adaptive pre-training** (`dapt_mlm.py`, then
   `--crf --model <checkpoint>`, P1 item 5).
   - Continued masked-LM training of MatSciBERT on
     `data/raw/dataset/pubmed_sentences.txt`. Duplicates and every sentence
     that shares a word 8-gram with MaterioMiner text are dropped first, so
     no evaluation text is seen.
   - 15% masking (80/10/10), lr 5e-5, batch 32, at most 128 tokens,
     3 epochs; the last checkpoint is used. 5% of the sentences are held out
     to report MLM perplexity.
3. **Ensembles** (`ner_ensemble.py`, P1 item 8).
   - Three CRF models per split: model seeds K, K + 100 and K + 200 on
     split K, and seeds 0, 100 and 200 for each held-out paper.
   - Entity-level majority vote: an entity is kept if at least 2 of the 3
     models predict the same span and type.

Decision rule, as in week 1: a variant improves on the CRF head for a
granularity only if the 95% CI of the mean paired micro-F1 difference over
the 5 random splits lies above 0. The same comparison over the 4 held-out
papers is reported as out-of-distribution robustness, and strict IOB2 and
macro F1 are reported alongside. Results are reported in the order above;
runs may overlap when CPU is free.

Deviation, noted after the first runs: two jobs share the 4 CPUs, so the
week-2 runs use 2 threads where the CRF results used 3. The thread count
alone changes a run with the same seeds (fine-grained split 0: 78.76 with 3
threads, 76.37 with 2), which adds noise to every difference. The `crf_m0`
ensemble members re-run the CRF head with 2 threads and measure the effect.

### Week 3 protocol (fixed before any run)

Week 3 runs items 1–5 of P1 "Next, after week 2". Every run uses `--crf`
and 2 CPU threads, and the week-2 code paths stay unchanged: new options only
add code.

**Design (item 1).**
- Every model is trained three times per split: model seeds K, K + 100 and
  K + 200 on random split K, and seeds 0, 100 and 200 on each held-out
  paper. A single model scores the mean micro F1 of its three runs on that
  split; an ensemble scores once.
- The reference is the CRF head scored the same way: the mean of the week-2
  runs `crf_m0`, `crf_m100` and `crf_m200`.
- Decision rule: a variant improves on its comparison if the 95% Student-t
  CI of the mean paired difference over the 5 random splits lies above 0.
  The same statistic over the 4 held-out papers is reported as
  out-of-distribution robustness. For the out-of-distribution methods
  (variants 2 and 3 below) the decision is taken on the 4 held-out papers
  instead, and the random splits are reported alongside.
- Strict IOB2 and macro F1 are reported with every comparison.
- With the week-2 spread between seeds (about 1.6 F1 per run,
  fine-grained), this design should detect gains of about 2 F1 points; one
  run per split could not detect gains below about 5.

**Variants**, reported in this order:

1. **Mixed-backbone ensemble** (P1 item 8).
   - CRF models from SciBERT (`allenai/scibert_scivocab_uncased`, results
     `crf_scibert`) and PubMedBERT
     (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`,
     `crf_pubmedbert`), with the MatSciBERT settings and one model seed per
     split (K on split K, 0 on each paper).
   - `crf_mix3`: the entity-level majority vote of MatSciBERT (`crf_m0`),
     SciBERT and PubMedBERT.
   - Primary comparison: `crf_mix3` against the MatSciBERT seed ensemble
     `crf_ens3` on the same splits, i.e. does mixing backbones add to mixing
     seeds? Both are also reported against the CRF reference.
   - The reported NER system (README, §1) is `crf_ens3`, unless `crf_mix3`
     improves on it under the rule above.
2. **Label semantics** (P1 item 6), `crf_ls_m0`, `crf_ls_m100`,
   `crf_ls_m200`.
   - The emission score of tags B-X and I-X is the dot product of the
     token's encoder state with a learned linear map (one for B, one for I)
     of the description embedding of X; O has its own learned vector. The
     CRF and all settings are unchanged.
   - Descriptions: for fine-grained labels, the class name split into words
     and its MMO `skos:definition`; for coarse-grained types, which have no
     definitions, the names of the MMO classes annotated with that type in
     the fine-grained files.
   - Each description is embedded once before training, as the mean
     last-layer state of the pre-trained encoder over its tokens; the linear
     maps are trained.
   - 3 seeds per split and per paper; decision on the held-out papers.
3. **Self-training** (P1 item 7), `crf_st_m0`, `crf_st_m100`, `crf_st_m200`.
   - Teacher: the CRF head trained on the run's training data.
   - It tags a fixed random sample (seed 0) of 2,000 sentences from the
     in-domain corpus of variant 5. MaterioMiner text is never used.
   - Silver data: the tagged sentences that contain at least one entity and
     in which every word's tag has a CRF marginal probability of at least
     0.9.
   - Student: same initialisation and seed, trained on gold + silver with
     the silver loss weighted 0.5; its epoch is chosen on the gold
     validation set.
   - Leaving one paper out only (3 seeds × 4 papers × 2 granularities), to
     bound CPU time; decision on the held-out papers.
4. **Multi-task again** (P1 item 2): `--crf --multi-task` with seed offsets
   100 and 200 (`crf_mt_m100`, `crf_mt_m200`), added to the week-2 runs
   (`crf_mt`, offset 0); compared with the CRF reference.
5. **In-domain pre-training** (P1 item 5), `crf_dapt2_m0`, `crf_dapt2_m100`,
   `crf_dapt2_m200`.
   - Corpus: open-access full texts from Europe PMC whose title mentions
     fatigue and whose title or abstract names a metallic material; the four
     MaterioMiner papers are excluded by DOI. The query and filters are
     fixed in code before any download, and the corpus report records the
     article IDs and the corpus SHA-256.
   - Sentences are deduplicated, and every sentence sharing a word 8-gram
     with MaterioMiner text is dropped, as in week 2. At most 40,000
     sentences are kept (a fixed random sample, seed 0).
   - MLM settings as in week 2 (`dapt_mlm.py`), then the CRF head from the
     new checkpoint, 3 seeds per split and paper; compared with the CRF
     reference.

Execution: multi-task (4) and the SciBERT and PubMedBERT runs (1) start
first, since they need no new code; label semantics (2), the corpus and
pre-training (5), and self-training (3) follow as their code is ready. Each
item is reported when its runs finish. The whole plan takes about 75 CPU
hours, about 2 days on two job slots.
