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
- `../sofc_exp_slot/<PMC id>.conll`: the same sentences with the corpus's slot tags, a finer
  level of types on the same tokens (19 slot types, 4,263 mentions; every slot token lies
  inside an entity, and each slot type within one entity type). `ner_baseline.py
  --granularity sofc_slot` keeps the 17 that SOFC-Exp's evaluation scores (not
  `experiment_evoking_word`, the EXPERIMENT entity under another name, nor the single
  `interconnect_material`).

The conversion checks the counts of the paper: 45 articles, 876 sentences and 4,590
entities. These are 1,492 MATERIAL, 1,475 VALUE, 1,090 EXPERIMENT and 533 DEVICE.

Tokens. The release's README says Stanford CoreNLP cut the texts into tokens, and the paper
names ChemDataExtractor. Checked on the release's texts and offsets (ROADMAP §4, week-6
amendment, "multi-task self-training on SOFC-Exp"):
- the 26 training and 8 development articles: ChemDataExtractor 1.3.0's word tokenizer gives
  the tokens of all 7,630 sentences (`src/graph/chem_tokenize.py` vendors it);
- the 11 test articles: Java's word BreakIterator (US locale) gives 1,833 of their 1,836
  sentences.

So the test articles are cut into words differently from the training ones: `Fig .`,
`cm − 2` and `high-temperature` in the test articles, `Fig.`, `cm−2` and
`high - temperature` in training. The files here keep the release's tokens.
