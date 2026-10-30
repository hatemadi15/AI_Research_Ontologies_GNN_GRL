# Reproducibility appendix and code-release checklist (both papers)

*Drafted 30 October for paper A (`docs/PAPER_A.md`) and paper B
(`docs/PAPER_B.md`). The first part is the appendix each paper attaches,
written once and cut to the venue's checklist; the second is what has to
happen to the repository before either is submitted. Facts below are the
repository's as of commit `eff8c73` (30 October); paths are relative to
its root.*

## Part 1: the reproducibility appendix

### A.1 What is released

- **Code.** `src/graph/` holds every experiment: the tagger and its
  variants (`ner_baseline.py`), the votes (`ner_ensemble.py`), the
  document-level rule (`ner_consistency.py`), the corpus builders
  (`build_fatigue_corpus.py`, `build_sofc_corpus.py`,
  `build_topic_corpus.py`, `build_mixed_corpus.py`), the lexicon probe
  (`silver_lexicon.py`), the corpus conversions (`sofc_exp.py`,
  `mulms.py`, `mulms_articles.py`), the analyses (`ner_seedmeans.py`,
  `ner_seed_analysis.py`, `ner_mentions.py`, `mulms_coverage.py`) and the
  generators of the README's results sections, the tables and the figures
  (`report_results.py`, `paper_tables.py`, `paper_figures.py`). The GPU
  runner is `src/graph/kaggle_jobs.py` with the Kaggle-side script
  `src/kaggle/run_jobs.py`. Licence: Apache 2.0 (`LICENSE`).
- **Protocols.** Every protocol is in `docs/ROADMAP.md` §4 under its week,
  with the date it was committed, its decisions, the comparisons reported
  without a decision, the expected number of false passes, and every
  amendment with its reason and timing. The repository's history dates
  each protocol before its runs; the session notes under each protocol
  name the frozen commit of each batch.
- **Results.** `results/` and `results/gpu/` hold every results file the
  papers cite: per run, the seed, the scores (micro, strict, macro and
  weighted F1, per class), the validation curve, the test indices, the
  gold and predicted tags, the runtime, the environment (device, CUDA,
  torch and transformers versions, the deterministic flag) and, for a
  run with teachers, the teachers' seeds and scores, the silver counts and
  the lexicon's coverage of the unseen test names. No name of the lexicon
  is stored. The analyses' outputs (`ner_week*_comparisons.json`,
  `ner_week5_seed_analysis.json`, `ner_mentions.json`,
  `ner_week7_consistency.json` and the rest, as indexed in
  `results/README.md`) are committed beside the files they read.
- **Tables and figures.** `docs/tables/table1–6` and
  `docs/figures/fig1–7` are generated from `results/` by `paper_tables.py`
  and `paper_figures.py`; no number in either paper is typed by hand
  (`report_results.py --check` fails the test suite when the README or the
  ROADMAP disagrees with the results).
- **Data.** MaterioMiner's CoNLL files, its released split and the MMO
  ontology are in `data/raw/dataset/` (release v1.0.1, CC BY 4.0). SOFC-Exp
  is converted by `sofc_exp.py` from the authors' repository
  (`boschresearch/sofc-exp_textmining_resources`; annotations CC BY 4.0);
  the converted files are in `data/raw/dataset/sofc_exp*/`. MuLMS is read
  by `mulms.py` from its Hugging Face release
  (`timo-pierre-schrader/MuLMS`, CC BY-SA 4.0) at a pinned revision; our
  conversion keeps the outermost entities and repairs 276 damaged tokens
  (paper B, Section 2), and the results files carry MuLMS's tags.
- **Not released, and why.** The unlabeled corpora (40,000 sentences each
  of open-access full texts from Europe PMC) are not redistributed; each
  is rebuilt from its report (`results/fatigue_corpus.json`,
  `results/sofc_corpus.json`, `results/topic_corpora/<subfield>.json`,
  `results/topic_corpora/mixed_<subfield>.json`), which records the query,
  the filters, the article identifiers, the counts at every stage and the
  corpus's SHA-256; the runner refuses a corpus whose hash differs. The
  harvested names are not stored (they are recomputed by the teachers).
  Model checkpoints are not released: every run retrains from the pinned
  MatSciBERT snapshot in under an hour on a T4.

### A.2 Environment

- **Software.** Python 3.10 or later; `requirements.txt` (torch 2.1+,
  transformers 4.40+, seqeval 1.2.2+, pyarrow for MuLMS) and
  `requirements-dev.txt` for the tests and the lint. The GPU runs pin
  `transformers==5.18.0`, `tokenizers==0.23.2`, `huggingface_hub==1.33.0`,
  `safetensors==0.8.0` and `rdflib==7.6.0` on the Kaggle image whose
  digest is `DOCKER_IMAGE` in `kaggle_jobs.py` (torch is the image's);
  model snapshots are pinned by commit and loaded offline.
- **Hardware.** Kaggle "GPU T4 x2" sessions: two lanes, one Tesla T4 each,
  12-hour sessions, a weekly quota of 30 to 45 hours. Deterministic CUDA
  (`torch.use_deterministic_algorithms`, cuDNN deterministic,
  `CUBLAS_WORKSPACE_CONFIG`). Each session first repeats an earlier run
  (`smoke*` tags); every repeat over weeks 4 to 14 was identical to the
  stored run. Weeks 1 to 4's CPU runs (2 or 3 threads) are kept in
  `results/` and are never compared with GPU runs.
- **Compute.** The GPU work is accounted per batch in the ROADMAP's
  session notes (session hours per batch); a submission states the total
  for its runs. A single CRF run on MaterioMiner takes 3 to 8 minutes on a
  T4, a MuLMS held-out subfield 6 to 12, a teacher-student job (3 teachers
  tagging 40,000 sentences and 3 to 4 students) 45 to 70 minutes; the
  pre-registered decisions of weeks 10 to 14 cost about 25 to 45 session
  hours each week.

### A.3 From a clean checkout to the papers' tables

1. `pip install -r requirements.txt` (and torch for the machine), then
   `pytest` to check the installation (the CRF tests need torch).
2. Data: MaterioMiner is in the repository; `python src/graph/sofc_exp.py`
   converts SOFC-Exp from the authors' repository;
   `python src/graph/mulms.py` reads MuLMS at the pinned revision.
3. Corpora (only to rerun teachers or self-training): the four builders
   rebuild each corpus from its report and check the SHA-256;
   `build_mixed_corpus.py` builds the mixed corpora from the topic
   corpora.
4. Runs: a batch is `kaggle_jobs.py bundle --commit <sha>` (the code, the
   corpora and the results so far into a private Kaggle dataset) and
   `kaggle_jobs.py push --batch <name>` (the job list under `BATCHES`,
   one entry per week and part; `batch_spec` prints the exact
   `ner_baseline.py` command of every job); `pull` and `collect` bring the
   results files into `results/gpu/`. The same commands run on any
   machine with a GPU: `python src/graph/ner_baseline.py <job>` with
   `PROCESSED_DIR` set.
5. Analyses, in order: `ner_seedmeans.py --week <n> --gpu --results
   results/gpu` for each week's comparisons, `ner_seed_analysis.py`,
   `ner_mentions.py`, `ner_consistency.py` where the week used it; then
   `report_results.py` (and `--check`), `paper_tables.py`,
   `paper_figures.py`.
6. Every number in the papers traces to a claim row in `docs/PAPER.md`
   (C1–C26), which names the results file and the analysis that produced
   it.

### A.4 Statistical reporting

Every decision is the 95% Student-t interval of the mean paired
difference in micro F1 over the held-out units (papers, folds, subfields,
or splits; seeds on a single split), each side the mean of 10 model seeds;
a pooled decision adds the condition that no corpus's own interval lies
below 0. Intervals are reported for every comparison, decided or not.
Exploratory analyses (per kind of name, per type, looser match criteria,
the 3-seed triples) are labelled as such and never decided on. Post hoc
numbers are labelled post hoc.

### A.5 Checklist items the venues ask for

| Item (ARR responsible NLP checklist, Nature Portfolio reporting summary) | Where |
|---|---|
| Limitations section | paper A §6, paper B §10 |
| Data licences and terms | A.1 above; MaterioMiner CC BY 4.0, SOFC-Exp CC BY 4.0, MuLMS CC BY-SA 4.0, Europe PMC open-access full texts |
| Data preprocessing and splits | paper B §2–3; `sofc_exp.py`, `mulms.py`; `docs/ROADMAP.md` §4 |
| Model and hyperparameters | paper B §4 (MatSciBERT, AdamW 5e-5, batch 16, up to 20 epochs, patience 3); `ner_baseline.py` defaults |
| Number of runs and variance | 10 seeds per unit everywhere decided; SDs in Figure 2; intervals on every number |
| Compute | A.2; per batch in the ROADMAP's session notes |
| Pre-registration | `docs/ROADMAP.md` §4 with commit dates; amendments marked |
| Code and data availability | the repository (anonymised for review, see Part 2) |
| Use of AI assistants | the code and the drafts were written with an AI coding assistant under the authors' direction; stated in the acknowledgements |
| Human subjects, personal data | none |

## Part 2: the code-release checklist

Before either paper is submitted:

1. **Freeze a release commit** after `w14c`'s results and the last
   docs update; tag it (`paper-a-v1`, `paper-b-v1`, the same commit for
   both if submitted together). Run `pytest`, `report_results.py --check`,
   `paper_tables.py` and `paper_figures.py` on the tagged commit and
   confirm the tree is clean.
2. **Anonymise for double-blind review** (ACL Rolling Review requires it;
   the materials journals do not). Make an anonymous snapshot of the
   release commit without history: author names and emails occur in the
   git history, the LICENSE header if any, and the README; `docs/` must
   not name the authors or the institution. Host it on an anonymous
   service (e.g. anonymous.4open.science) and cite that URL in the
   submission; the public repository and its history are named at
   camera-ready.
3. **A paper-facing README.** The current README documents the whole
   project, including the knowledge-graph pipeline that neither paper
   reports. Add a short `docs/REPRODUCE.md` (or a section at the top of
   the README) that gives A.3 as commands, names the results file behind
   each table and figure, and points to the two papers' drafts; keep the
   project README as it is below it.
4. **Pin what is loose.** `requirements.txt` gives lower bounds; add a
   `requirements-lock.txt` with the exact versions of the environment that
   regenerated the tables (and the Kaggle pins for the GPU runs, already
   in `kaggle_jobs.py`).
5. **Data statements.** For each corpus: source, version or revision,
   licence, what the conversion changed (MuLMS: outermost entities, 276
   repaired tokens, 2 dropped entities; SOFC-Exp: the official split's
   tokenization shift and the matched-tokenization test set). Confirm the
   MaterioMiner release's redistribution terms before including its CoNLL
   files in the release (CC BY 4.0 allows it with attribution).
6. **The corpora.** State that the unlabeled corpora are rebuilt from
   their reports, give the rebuild command and the expected SHA-256 of
   each, and note that Europe PMC's content can change (a rebuild that
   differs is detected by the hash; the reports list the article
   identifiers so a differing rebuild can be diagnosed).
7. **Predictions and analyses.** Keep every run's test predictions in the
   results files (they are what the unseen-name and match-criterion
   analyses read); state their size. Confirm that each table's generator
   reads only committed files.
8. **The pre-registration record.** Export the dated protocol texts with
   their commit hashes (a `docs/PROTOCOLS.md` generated from `git log`
   over `docs/ROADMAP.md` §4, or the hashes listed in each protocol's
   first line) so a reviewer can verify the order of protocol and result
   without cloning the history.
9. **Compute statement.** Sum the session hours per week from the ROADMAP
   notes into one number per paper (A: weeks 10 to 14; B: weeks 4 to 14)
   and the number of GPU runs each paper's tables rest on.
10. **Authors' confirmation.** The email to MaterioMiner's authors on the
    released split and the scoring (the user's) is sent before submission;
    its answer goes into paper B §5.1 and the limitations.
11. **Acknowledgements and AI-assistance statement** per venue policy.
12. **Camera-ready.** Replace the anonymous URL by the public repository,
    add the Zenodo DOI of the release tag, and update `docs/PAPER.md`'s
    claims table with the final numbers.
