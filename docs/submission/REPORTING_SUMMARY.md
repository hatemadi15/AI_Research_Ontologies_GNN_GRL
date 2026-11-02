# Reporting Summary: draft answers

Nature Portfolio journals ask for a Reporting Summary, a form in the
submission system (also available as a PDF from
https://www.nature.com/documents/nr-reporting-summary-flat.pdf). For
*npj Computational Materials* it is encouraged at submission and required
after review. The answers below come from the paper; copy each into the
form under the same heading.

## Statistics

The form asks you to confirm that each item is in the paper. All the
applicable items are:

| Item | Answer | Where |
|---|---|---|
| Exact sample size (n) for each group | Yes | The held-out units of each decision: 7 MuLMS subfields, 4 MaterioMiner papers, 5 SOFC-Exp folds, 11 pooled units. Each unit is the mean over its seeds (10, or MuLMS's 7 decision seeds). Table 2, Table S16; Methods, *Evaluation protocol* |
| Distinct samples or repeated measures | Yes | Every comparison is paired: both systems are scored on the same held-out documents with the same seeds. Methods, *Statistics and reproducibility* |
| Statistical tests, one- or two-sided | Yes | The one-sample t-test of the mean paired difference over units, with the decision rule one-sided at 2.5% (the 95% interval above 0) and two-sided P values reported. Also the exact sign-flip test, two-sided. Methods |
| Covariates | Not applicable | Paired design, no covariates |
| Assumptions and corrections | Yes | t-test normality, checked against the exact sign-flip test. The units' dependence through shared training documents, by a sensitivity analysis (corrected resampled t-test). No correction for the number of decisions; the count and a Bonferroni reading are reported. Methods, *Multiplicity* |
| Central tendency and variation | Yes | Mean paired differences in F1 points with 95% Student-t intervals; standard deviations in Table S16 |
| Test statistic, CI, effect size, df, exact P | Yes | Table 2 (P), Table S16 (t, df, one- and two-sided P, sign-flip P); a P value at each inference in the text |
| Bayesian analysis | Not applicable | |
| Hierarchical or complex designs | Yes | The unit of analysis is the held-out document unit, with seeds averaged within it. The pooled decisions over two corpora are described in the Methods |
| Effect sizes | Yes | Mean paired differences in F1 points with 95% intervals (no standardized effect sizes) |

## Software and code

**Data collection.** No software was used to collect primary data. The
unlabeled corpora were built from Europe PMC's open-access full texts by
the repository's scripts. These are `build_topic_corpus.py`,
`build_mixed_corpus.py`, `build_fatigue_corpus.py` and
`build_sofc_corpus.py`, and each corpus report records its queries,
article identifiers, licenses and SHA-256 hash.

**Data analysis.**
- Training on Kaggle T4 GPUs with deterministic CUDA: Python 3, PyTorch
  2.11.0 with CUDA 12.8, Hugging Face Transformers 5.18.0.
- Scoring with seqeval 1.2.2.
- Statistics with SciPy 1.17 and NumPy 1.26 (Python 3.11).
- All custom code is in github.com/hatemadi15/AI_Research_Ontologies_GNN_GRL
  (Apache 2.0). Among it are `ner_baseline.py` (training and scoring),
  `decision_tests.py` (the tests), and `names_route_record.py` and
  `paper_tables.py` (the tables).

## Data

Paste the paper's Data availability statement (its section of
`docs/PAPER_A.md`). In short:

> The labeled corpora are public: MaterioMiner, release v1.0.1 (Fordatis,
> https://doi.org/10.24406/fordatis/329.2; CC BY 4.0); SOFC-Exp
> (github.com/boschresearch/sofc-exp_textmining_resources; CC BY 4.0; at
> commit bbf75fd); and MuLMS (Hugging Face,
> huggingface.co/datasets/timo-pierre-schrader/MuLMS at revision bc1c97d;
> CC BY-SA 4.0). The repository,
> github.com/hatemadi15/AI_Research_Ontologies_GNN_GRL, holds every
> results file with its seeds, environment and saved test predictions,
> and the corpus reports. The unlabeled corpora, built from Europe PMC's
> open-access full texts, are not redistributed; they rebuild from their
> reports, with hash checks.

## Research involving human participants, their data, or biological material

Not applicable: no human participants, personal data or biological
material.

## Field-specific reporting

The form asks you to pick one field. For a computational study the usual
choice is **Life sciences**. Its five study-design questions are
answered below; if the editors prefer another choice, the same answers
apply.

## Study design

**Sample size.** No statistical method was used to predetermine sample
size. The held-out units are all that the three annotated corpora allow:
MuLMS's 7 subfields, MaterioMiner's 4 papers and SOFC-Exp's 5 article
folds. Each system was trained with 10 seeds per unit (3 in screens), a
number fixed in protocols committed before the runs. Supplementary Note 9
gives the difference each comparison could detect.

**Data exclusions.** No run or unit was excluded after its results. The
exclusions fixed in advance are:
- MuLMS's nested entities are reduced to the outermost, dropping 31.4% of
  its entities, and 276 damaged tokens are repaired.
- From SOFC-Exp only the 876 of its 9,466 sentences that carry entity
  annotation are used.
- The unlabeled corpora exclude the datasets' own articles and every
  sentence sharing a word 8-gram with a dataset sentence (for MuLMS also
  a 50-character run).

GPU sessions that failed or hung were rerun from the same commit.

**Replication.** Every run is deterministic. From 15 October every
completed GPU session but two began by repeating an earlier run, and
every repeat was identical (Supplementary Note 2). The results are
replicated across held-out units and seeds. A third corpus, SOFC-Exp,
shows no gain, and this is reported.

**Randomization.** Seeds fix each run's initialization, data order and
augmentation. The held-out units are whole documents (subfields, papers,
article folds), not random sentences.

**Blinding.** Not applicable: scoring is automatic. Instead, each
decision's analysis was fixed in a protocol committed before its results
(Supplementary Note 1 dates each one), and every deviation is disclosed.

## Reporting for specific materials, systems and methods

Tick **n/a** for every item: antibodies, eukaryotic cell lines,
palaeontology and archaeology, animals and other organisms, clinical
data, dual-use research of concern, plants, ChIP-seq, flow cytometry and
MRI-based neuroimaging.
