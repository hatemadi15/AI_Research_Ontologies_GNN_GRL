# Week 19 plan (3 November 2026): paper B, and graph representation learning where it can matter

*Status: revised after an independent review (a Fable-model referee, 3
November; verdict "approve with required changes", items 1–2 blocking;
every claim it made about the code and data was checked against the
files before this revision). The GPU study's protocol goes into
`docs/ROADMAP.md` §4 as the week-19 protocol, committed before any run.
Nothing here has been run. The first draft is in this file's history
(`440beed`).*

## 1. Where things stand

- **Paper A** (the names method) was submitted to npj Computational
  Materials on 2 November (`docs/PAPER_A.md`; `main` at `325019a`, which
  stays as submitted while it is under review). Its 12 decisions: D3, D4,
  D5 and D11 pass; its headline is that the names are not shown to add to
  mention replacement. Its bottleneck: the harvested lexicon (spans three
  CRF teachers tag alike, every tag's marginal at least 0.9, in 40,000
  unlabeled in-topic sentences) holds 7–36% of a held-out unit's unseen
  test names (SI Table S2, per unit), while the corpora hold 55–81% of
  them; over 6-copy mention replacement the names add +0.80 [−0.66,
  +2.26] (D7, 11 units, not established). Its ablations (SI, the table of
  the five ablations) show that a lower marginal threshold (0.7) alone
  raises the coverage from 22.2% to 31.5% and 2-of-3 agreement to 28.8%,
  with F1 within 0.16 of the method: more names are cheap to get, and
  more names alone did not help.
- **Paper B** (the evaluation paper, `docs/PAPER_B.md`) is a first draft
  of 30 October, before weeks 15, 17 and 18 and before paper A's 18
  review rounds: its count, its summary of paper A and its framing of
  paper A ("passes a pooled decision on two corpora") are stale.
- **Graph representation learning in this repository**: the original
  ontology pipeline trained GraphSAGE by link prediction on a term
  co-occurrence graph (`src/graph/gnn.py`, `builder.py`). After the
  leakage fix (PR #5) it did nothing for typing (0.384 vs 0.378 without
  it, McNemar p = 0.69) and hurt clustering (ARI 0.035 vs 0.195 with plain
  SBERT): co-occurrence edges join a material to its property, value and
  unit, so they are heterophilous for type (ROADMAP P3). Nothing in
  papers A or B uses a graph.
- **Compute**: Kaggle GPU quota 22.6 h left this week (45 h a week; reset
  7 November 00:00 UTC). Week 15's harvest jobs (3 teachers tagging
  40,000 sentences, no students) took 16–18 minutes each (`runtime_sec`
  946–1220 in `results/gpu_names/`); week 18's 7-student MuLMS jobs 32–39
  minutes; a MaterioMiner 10-student names job 83–99 minutes.

## 2. Track G: graph-propagated names (the GPU study)

**Question.** At an equal number of admitted names, does a graph neural
network over every name the teachers tag at all admit more of a held-out
unit's unseen names, at no more type errors, than (a) the teachers' vote
with a lowered threshold, the cheap rival, and (b) the same model with
the graph removed? And if it does, does its lexicon help mention
replacement more than the size-matched vote lexicon?

**Units and seeds** (those of paper A's D7, so stored runs pair by unit
and seed): MuLMS's 7 held-out subfields (week 10's fold: 400 training
sentences, split seed 0; decision seeds 300–900) and MaterioMiner's 4
held-out papers, fine-grained (seeds 0–900). Unlabeled text: each MuLMS
subfield's own topic corpus and the fatigue corpus for MaterioMiner
(40,000 sentences each, as in paper A). Stored controls, verified to
exist for all 11 units at these seeds with predictions:
`crf_mr6` (6 copies, no names) and paper A's names,
`crf_mrm_topic6` (MuLMS) and `crf_mrm6` (MaterioMiner fine).

**Phase 0a, the harvest (Kaggle, batch `w19a`).** Per unit, a
`--students 0` job: paper A's three CRF teachers (seeds 0, 100, 200,
trained on the unit's training sentences, deterministic CUDA) tag the
corpus. `--save-names` rebuilds paper A's lexicon; new
`--save-candidates FILE` dumps, for every span any teacher tags as an
entity: its words and sentence index and, per teacher, the type it tags
there (or none) and, from the full CRF marginals (`WordCRF.marginals`;
`teacher_tags` keeps only the Viterbi tag's today), the span's score for
that teacher's two most probable types, the least over the span of the
B- marginal at its first word and the I- marginals after it. Checks: the
rebuilt teachers' test F1 equal the stored `crf_m0/m100/m200` runs, and
the rebuilt lexicon counts equal `results/gpu_names/` (week 15's checks).
The dumps hold corpus text and stay local (never versioned). Budget:
about 20 minutes a unit, under 4 GPU hours, on two lanes.

**Phase 0b, the probe (CPU, `src/graph/graph_names.py`, no decision).**
Per unit:
1. Nodes: the distinct surface forms of the candidate spans, features
   aggregated over all their spans (the teachers' boundaries differ).
   Labelled nodes: the training names among them, with their gold type
   (the majority type if several). Node features: per type, the share of
   the node's spans (over occurrences and teachers) tagged with it and
   the mean span score; log frequency; the character 3–5-gram TF-IDF
   vector reduced to 128 dimensions.
2. Edges, kept apart by type: sentence co-occurrence (positive PMI, the
   20 strongest per node); the 10 nearest neighbours by character n-gram
   cosine; a shared head word (heads held by at most 200 nodes). The
   homophily of each edge type on labelled pairs is reported.
3. Rankers of (name, type), over unlabelled nodes, each type restricted
   to names at least one teacher tags with it somewhere:
   - **V**, the vote: the lexicons of the (marginal threshold 0.5, 0.7,
     0.9) × (1, 2, 3 of 3 teachers agree) grid, the semantics of
     `silver_mention_names`;
   - **M**, graph removed: a 2-layer MLP on the node features, trained on
     the labelled nodes;
   - **G**, the graph: a 2-layer relational GNN (R-GCN, one weight matrix
     per edge type, mean aggregation per type), same features and
     labels;
   - **LP**, label propagation on the union graph (reported only).
   M and G use fixed hyperparameters (hidden 128, dropout 0.3, Adam lr
   0.01, weight decay 5e-4, 200 epochs, seed 0); their 5-fold accuracy
   on the labelled nodes is reported as a diagnostic labelled
   "teacher-seen" (the teachers trained on those names), and no
   threshold is set from it.
4. Size matching instead of a threshold: each ranker's lexicon is its
   top N new names (names that are no training name), one type per name
   (its most probable). N primary: the number of new names in V at 0.7,
   3 of 3; N secondary: at 0.9, 3 of 3 (paper A's setting). A name's
   count, in every lexicon alike: the corpus mentions at which at least
   one teacher tags it with the admitted type.
5. Per unit and ranker, at both sizes: coverage of the unit's unseen test
   mentions (type and words in the lexicon); the wrong-type rate (unseen
   test mentions whose words the lexicon holds under another type only,
   over those whose words it holds at all); the candidate set's coverage
   as the ceiling; pooled rates beside the means over units.
6. **Gate, a test-set screen** (it reads held-out test names, as SI
   Table S2's diagnostic did; it sets no threshold, only whether the GPU
   decisions run). At N primary, means over the 11 units:
   coverage(G) ≥ coverage(V 0.7, 3 of 3) + 5 points, coverage(G) ≥
   coverage(M) + 2 points, and wrong-type rate(G) ≤ that of V 0.7 + 5
   points. A failed gate stops the study: it is reported as a negative
   lexicon screen (not a decision) in paper B.

**Phase 1, the decisions (only if the gate passes; after the 7 November
reset).** Two new arms, paper A's recipe exactly (gold sentences plus 6
copies, p = 0.5, the decision seeds) with an external lexicon loaded by a
new `--names-file` (its SHA-256 in every results file):
`crf_mrg6` (G's lexicon at N primary) and `crf_mrv6` (V at 0.7, 3 of 3,
the same size).
- **G1**: `crf_mrg6` minus `crf_mr6`, pooled over the 11 units (each unit
  the mean of its seeds).
- **G2**: `crf_mrg6` minus `crf_mrv6`, pooled over the 11 units.
- Rule: a 95% Student-t CI of the mean paired micro-F1 difference above 0,
  as always. Reported, no verdict: per corpus; both arms against paper A's
  names; recall and precision on the unseen test names (week 15's split
  by lexicon membership); `n_augmented` and the names' share of draws per
  arm.
- Equality checks before any decision: `--names-file` replaying paper A's
  rebuilt lexicon, in `silver_mention_names`'s order (mention
  replacement draws by `rng.choice`, so order matters), reproduces one
  stored `crf_mrm_topic6` run byte-identically, predictions included.
- Power, stated now: D7's sd over the 11 units is 2.17, so a difference of
  about 2.9 points is detectable; G2's expected effect is under a point
  (the names' whole increment over MR6 is +0.80). A null reads "not shown
  within about 3 points".
- Budget: about 5.5 GPU hours per arm (MuLMS 7 jobs of about 35 minutes,
  MaterioMiner 4 papers at 10 students), two arms and the check about 12
  of the 45.

**What each outcome means.** G1 and G2 pass: graph learning improves the
names route beyond what more names give; paper B reports it, and it is the
core of a third paper. G1 only: graph-chosen names help, but not more
than as many voted names. Neither, or a failed gate: graph propagation
does not rescue the names route at matched size. Every outcome is
reported, in paper B's count.

## 3. Track B: paper B to a submittable state (no GPU)

1. Sync with the record, framing first: paper A's final headline (the
   names not shown to add to mention replacement; D11 passing on recall
   only through its three flagged subfields); §7.6's table generated from
   paper A's Table 2 and `results/gpu/ner_decision_tests.json` (D1–D12,
   with the Bonferroni note), not typed; the count recounted from the
   protocols (weeks 17 and 18 add D9–D12; the "10 seeds" group holds
   MuLMS's 7-seed decisions and must say so); "fourteen weeks" →
   eighteen; every number paper A's review rounds corrected; the figures
   regenerated after the recount.
2. The count closes on the day paper B's text is frozen for submission;
   every decision whose results are in by then is counted, week 19's
   included (a failed gate as one lexicon screen, not a decision).
3. A number audit against the results files and a consistency check
   against the submitted paper A (paper A's rounds 17–18 procedure), then
   adversarial review rounds until none blocks.
4. Venue (the author's choice): TACL (1 December) or ARR (4 January). For
   ARR: 8 pages with a compressed §2 (data and protocol stay in the main
   text), the mandatory Limitations section, the Responsible NLP
   checklist (the MuLMS and SOFC-Exp licences, the AI-assistance
   disclosure paper A carries, compute totals), and double-blind
   anonymity: an anonymized mirror with squashed history and scrubbed
   Kaggle kernel names, and paper A cited in the third person.
5. When B is submitted anywhere, npj's policy asks for a copy of related
   manuscripts under consideration elsewhere: send B to npj then.

## 4. Track G0 (deferred until paper B is submitted)

ROADMAP P3 item 1, the ontology graph trained for typing, compared with
the supervised hybrid baseline (0.496, 15-fold grouped CV). First a
30-minute screen: the PMI graph's type homophily on the 315 terms and
label propagation against the hybrid; if label propagation does not
beat the hybrid, G0 is dropped. Its result would go to the README, not
to paper B.

## 5. Order

1. Now: the week-19 protocol in ROADMAP §4, committed; the code with tests
   (`--save-candidates`, `graph_names.py`, `--names-file`), first run on
   the CPU teachers' caches of `silver_lexicon.py` and dry-run on stored
   files; then batch `w19a` on Kaggle.
2. While it runs: track B's sync and audit.
3. The probe and the gate; Phase 1 after the reset if the gate passes;
   decisions; paper B's count and §7.
4. Track G0 after paper B is submitted, if its screen passes.

## 6. Risks

- The graph re-finds the teachers' confident names: matched size and the
  graph-removed rival measure exactly that.
- Heterophilous co-occurrence edges dilute the signal: relation-specific
  aggregation and the reported homophily per edge type address it.
- Few labelled nodes (about 100–300 per unit over 13 MuLMS types): the
  GNN may not train; the teacher-seen diagnostic shows it.
- Eleven units give a wide CI (about 2.9 points): a small true gain may
  not pass; reported either way.
- Kaggle sessions can hang; the runner's timeouts and resumable results
  handle it (weeks 7–18).
