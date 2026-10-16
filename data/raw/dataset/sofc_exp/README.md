# SOFC-Exp as CoNLL files

The second materials NER dataset of ROADMAP P1 "Next, after week 4" (item 7), converted for
`ner_baseline.py`:

> A. Friedrich et al., "The SOFC-Exp Corpus and Neural Approaches to Information Extraction in the Materials
> Science Domain", ACL 2020, https://aclanthology.org/2020.acl-main.116

Source: https://github.com/boschresearch/sofc-exp_textmining_resources, commit
`bbf75fdda4ba68fe8041fdc2780acd2a134e9770` (21 July 2023), directory `sofc-exp-corpus`.
The annotations are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
the texts are those of 45 open-access articles published under CC-BY licences, listed with
their DOIs in the repository's `SOFC-Exp-Metadata.csv`.

`python src/graph/sofc_exp.py <clone>/sofc-exp-corpus` writes these files:
- `<PMC id>.conll`, one per article: its annotated sentences (those that describe
  experiments), in text order, one "token tag" line per token (BIO tags), with a blank line
  between sentences. The tokens are cut out of the texts at the corpus's offsets.
- `split.json`: the official split by article (26 train, 8 dev, 11 test).

The conversion checks the counts of the paper: 45 articles, 876 sentences and 4,590
entities. These are 1,492 MATERIAL, 1,475 VALUE, 1,090 EXPERIMENT and 533 DEVICE.
