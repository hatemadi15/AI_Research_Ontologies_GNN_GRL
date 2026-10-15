# The published split of MaterioMiner

The train, validation and test sentences of the experiments in

> A. R. Durmaz, A. Thomas, L. Mishra, R. Niranjan Murthy and T. Straub, "An
> ontology-based text mining dataset for extraction of process-structure-property
> entities", *Scientific Data* 11, 1112 (2024),
> https://doi.org/10.1038/s41597-024-03926-5

as shipped with the dataset: release v1.0.1 on Fordatis,
https://doi.org/10.24406/fordatis/329.2 (`materio-miner-v1.0.1/examples/data/`),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The six
files are copied unchanged.

| File | Sentences | SHA-256 |
|---|---|---|
| `fine_grained_training/train.txt` | 323 | `6736fdb501a43317e4f2e6101bee5c911816f6dda813ae0807bd64c9f6cf8ae9` |
| `fine_grained_training/val.txt` | 57 | `e874dcb5232bde370f2f659767254266ec98d48dcab86ade5de4271b324727ee` |
| `fine_grained_training/test.txt` | 96 | `043962bb9b022c42a474d5bc2b1fe3f7df799b90da0c6356dc1b73ecafad858a` |
| `coarse_grained_training/train.txt` | 323 | `b0a3ba3146bebdbbc7c9937d474aded18cc2f81da950bc72879fa8ed9f22598b` |
| `coarse_grained_training/val.txt` | 57 | `7bf6965e2bfde17290f70dca5323d0f17674ede4c7fa17c05ab9d213495579db` |
| `coarse_grained_training/test.txt` | 96 | `3ecc82bc6e740a82866ae51b8f4271fead4e7b9bee9824692254fd5fbe6cb0a2` |

What we checked (ROADMAP §4, week-5 amendment):
- Every sentence of the six files is one of the 476 sentences of the per-paper
  CoNLL files in `../fine_grained_ner/` and `../coarse_grained_ner/`, each used
  once. The fine-grained and the coarse-grained files partition the corpus
  differently.
- The fine-grained files keep the 27 classes that `ner_baseline.py --top-k 27`
  keeps, with the same tie-break (TensileYieldStrength over Dislocation).
- Their tags equal those of the corpus files after that restriction, except 7
  tokens in 6 fine-grained sentences (5 training, 1 test), where these files
  start an entity with B- where the corpus files continue one with I-. The
  coarse-grained tags are equal.
- The release code scores with seqeval 1.2.2's `f1_score` and
  `classification_report` with their defaults: micro F1 in the default
  (conlleval) mode, which is `f1` in our results files.

`ner_baseline.py --published-split` trains on these partitions and tags.
