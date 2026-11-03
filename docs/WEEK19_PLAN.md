# Week 19 plan (draft for review, 3 November 2026): paper B, and graph representation learning where it can matter

*Status: a draft for an independent review before anything runs. Once
reviewed, the GPU study's protocol goes into `docs/ROADMAP.md` §4 as the
week-19 protocol, committed before any run, as every protocol has been.
Nothing here has been run.*

## 1. Where things stand

- **Paper A** (the names method) was submitted to npj Computational
  Materials on 2 November (`docs/PAPER_A.md`; `main` at `325019a`, which
  stays as submitted while it is under review). Its 12 decisions: D3, D4,
  D5 and D11 pass. Its bottleneck, in its own numbers: the harvested
  lexicon (names three CRF teachers tag alike, every tag's marginal at
  least 0.9, in 40,000 unlabeled in-topic sentences) holds only 16–32% of
  a held-out unit's unseen test names (SI Table S2), and over 6-copy
  mention replacement the names add +0.80 [−0.66, +2.26] (D7, 11 units,
  not established).
- **Paper B** (the evaluation paper, `docs/PAPER_B.md`) is a first draft
  of 30 October, before weeks 15, 17 and 18 and before paper A's 18
  review rounds. Its count (86 decisions, 18 pass) and its summary of
  paper A (8 decisions, 3 pass) are stale. Venue (`docs/VENUES.md`):
  ARR, 4 January 2027 (ACL 2027), recommended; TACL's 1 December date is
  possible.
- **Graph representation learning in this repository**: the original
  ontology pipeline trained GraphSAGE by link prediction on a term
  co-occurrence graph (`src/graph/gnn.py`, `builder.py`). After the
  leakage fix (PR #5) it did nothing for typing (0.384 vs 0.378 without
  it, McNemar p = 0.69) and hurt clustering (ARI 0.035 vs 0.195 with plain
  SBERT): link-prediction embeddings encode co-occurrence, not class
  (ROADMAP P3). Nothing in papers A or B uses a graph.
- **Compute**: Kaggle GPU quota 22.6 h left this week (45 h a week; reset
  7 November 00:00 UTC). A week-17 names job (3 multi-task teachers
  tagging 40,000 sentences with two heads, then 10 students) took about
  2.8 GPU hours, 80 minutes of it the harvest.

## 2. Track B: paper B to a submittable state (no GPU)

1. Sync with the record: the count (now 90 decisions, 19 passes, to be
   recounted from the protocols, not by hand-editing: weeks 17 and 18
   add D9–D12 of paper A), §7.6's summary of paper A (12 decisions,
   4 pass), "fourteen weeks" → eighteen, and every number paper A's
   review rounds corrected.
2. A number audit against the results files and a consistency check
   against the submitted paper A (the same procedure as paper A's rounds
   17–18), then adversarial review rounds until none blocks.
3. Venue form: ARR long paper (8 pages, unlimited appendices, mandatory
   Limitations section, the Responsible NLP checklist), double blind: the
   public repository names the author, so the submission links an
   anonymized mirror and cites paper A in the third person.
4. When B is submitted anywhere, npj's policy asks for a copy of related
   manuscripts under consideration elsewhere: send B to npj then.

## 3. Track G: graph-propagated names (the GPU study; protocol to be fixed before any run)

**Question.** Paper A admits a name only when three teachers agree with
high confidence. Can a graph neural network, trained on the training
names and run over a graph of every name the teachers tag at all, assign
types to the many names the teachers are unsure of, so that the lexicon
holds more of a held-out unit's unseen names at no loss of precision,
and does that lexicon then help mention replacement more than paper A's?

**Units and seeds** (identical to paper A's D7, so the stored controls
pair by unit and seed): MuLMS's 7 held-out subfields (week 10's fold:
400 training sentences, split seed 0; decision seeds 300–900, 7 seeds)
and MaterioMiner's 4 held-out papers, fine-grained (10 seeds, 0–900).
Unlabeled text: each MuLMS subfield's own topic corpus and the fatigue
corpus for MaterioMiner (40,000 sentences each, as in paper A; all on
disk, SHA-256 recorded in their reports).

**Phase 0, the probe (no decision; gates the GPU decisions).**
1. Harvest (Kaggle, one job per unit): paper A's three CRF teachers
   (seeds 0, 100, 200, trained on the unit's training sentences, as in
   paper A, deterministic CUDA) tag the corpus; new option
   `--save-candidates FILE` dumps every span any teacher tags as an
   entity, with each teacher's type and least marginal and the sentence
   index; paper A's own lexicon is rebuilt in the same job
   (`--save-names`, the week-15 mechanism) as the comparison.
   About 0.6 GPU h per unit, 7 h in all.
2. Graph (CPU, `src/graph/graph_names.py`, new). Nodes: the distinct
   candidate names (type-free surface forms) plus the training names.
   Edges: (a) sentence co-occurrence weighted by positive PMI, (b) the
   k = 10 nearest neighbours by character 3–5-gram TF-IDF cosine (names
   like "Ti-6Al-4V" and "Ti-6Al-7Nb" share form), (c) a shared head word.
   Node features: per type, the fraction of teacher votes and the mean
   least marginal; log frequency; the character n-gram vector reduced to
   128 dimensions. Model: a 2-layer GraphSAGE node classifier over the
   types (PyTorch Geometric, mean aggregation, dropout 0.3, early
   stopping on a validation split), trained on the training names that
   occur in the corpus, labelled with their gold type (a name with two
   gold types takes its majority type). Compared within the probe, no
   decision: label propagation on the same graph, and the teachers' vote
   alone.
3. Admission threshold, fixed without test data: 5-fold
   cross-validation over the training names in the corpus (hide a fold,
   train on the rest, predict the hidden fold), then the least
   confidence τ at which the hidden names' type accuracy is at least that
   of paper A's lexicon on the same hidden names (its precision on names
   whose gold type is known). The graph lexicon: every candidate whose
   predicted type has probability at least τ and that at least one
   teacher tags with that type somewhere, with its mention count.
4. Probe numbers per unit (counts and rates only; lexicons are never
   versioned, as in paper A, because they derive from corpora that are
   not redistributed): lexicon size; coverage of the unit's unseen test
   names (`name_coverage`, by type and words); the wrong-type rate (test
   names the lexicon holds with another type); the same for paper A's
   lexicon; the dictionary probe of `silver_lexicon.py` on the stored
   predictions as a secondary.
5. **Gate** (fixed now): the GPU decisions run only if, over the 11
   units, the graph lexicon's mean coverage of unseen test names exceeds
   paper A's by at least 5 points and its mean wrong-type rate is at most
   paper A's plus 5 points. Otherwise the study stops, the probe is
   reported as a negative screen and counted as one, and paper B says so.

**Phase 1, the decisions (only if the gate passes).** Students
`crf_mrg6`: paper A's recipe exactly (gold sentences plus 6 copies,
p = 0.5, the decision seeds) with the graph lexicon as the names, loaded
through a new `--names-file` (its SHA-256 recorded in each results
file). The controls are stored: `crf_mr6` (6 copies, no names) and
`crf_mrm6` (paper A's names).
- **G1**: `crf_mrg6` minus `crf_mr6`, pooled over the 11 units (each unit
  the mean of its seeds).
- **G2**: `crf_mrg6` minus `crf_mrm6`, pooled over the 11 units.
- Rule: as always, a 95% Student-t CI of the mean paired micro-F1
  difference above 0. Per corpus, the same differences reported, no
  verdict. Reported: recall and precision on the unseen test names
  (week 15's split by lexicon membership).
- A training-equality check before any decision: one stored `crf_mrm6`
  job re-run from the new bundle must reproduce its stored scores
  exactly.
- Budget: about 10 GPU hours of students plus 1 for the check; Phase 0's
  7 this week, Phase 1 after the 7 November reset.

**What each outcome means.** A pass of G1 and G2: graph learning
improves the harvested-names route; paper B reports it as an addition
that passes, and it is the core of a third paper. A pass of G1 only: the
graph lexicon works, but not better than paper A's. Neither, or a
failed gate: graph propagation does not rescue the names route, reported
as such in paper B's count. Either way the result is reported.

## 4. Track G0 (optional, CPU only): the original ontology GNN, trained for its task

ROADMAP P3 item 1: a heterogeneous term–class graph (term–term PMI
edges, class–class `subClassOf`, term–class edges for training-fold
terms), an R-GCN node classifier trained for typing, evaluated in the
same 5-fold grouped cross-validation as the supervised hybrid baseline
(0.50) on the 315 graph terms; one decision against that baseline
(paired by term, exact McNemar, α = 0.05). About a day of CPU work. It
answers whether the original pipeline's graph can matter when trained
for typing rather than link prediction; its result would go to the
repository's README, not to paper B.

## 5. Order

1. Now: track B's sync (item 1), the week-19 code with tests (the
   candidate dump, `graph_names.py`, `--names-file`), the protocol
   committed, then Phase 0's harvest batch on Kaggle.
2. While it runs: track B's audit and review rounds.
3. Phase 0's probe and gate; Phase 1 after the reset if the gate passes;
   decisions, then paper B's count and §7.
4. Track G0 only if time is left before paper B's venue work.

## 6. Risks

- The graph may merely re-find the teachers' confident names; the
  coverage gate measures exactly that, before any GPU decision.
- Type noise in admitted names could hurt precision; the wrong-type
  rate in the gate bounds it, and τ is matched to paper A's precision on
  hidden training names.
- Eleven units give a wide CI (paper A's D7 width was about 2.9 points);
  a small true gain may not pass. Reported either way.
- Kaggle sessions can hang; the runner's timeouts and resumable results
  handle it (weeks 7–18).
