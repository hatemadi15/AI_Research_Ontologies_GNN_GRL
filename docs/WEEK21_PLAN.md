# Week 21 plan (3 November 2026): a generalist NER baseline, and paper B for TACL

*Status: checked before any run by an independent referee (a Fable-model
referee: "approve with required changes"). The binding protocol, with its
required changes (gliner_medium for every arm, the fixed recipe and labels, the
environment kept, the detectable sizes, the readings and the checks), is
`docs/ROADMAP.md` §4's week-21 protocol. This draft keeps the first reasons.
Run and decided on 4 November: none of 21.1–21.4 passes; fine-tuned GLiNER is not
shown to differ from the CRF head on MaterioMiner fine-grained, MuLMS or SOFC-Exp and
is below it at MaterioMiner's coarse level (ROADMAP §4, the week-21 outcome).*

## 1. Why

- Paper B's first referee (R11) asked for an LLM or generalist-NER
  baseline on the held-out units. Such a model need not rely on the
  training documents' names, which are the bottleneck paper B measures.
- The author chose TACL. TACL takes 7 to 10 pages, references uncounted,
  plus a capped appendix, and reviews double-blind.
- The author has no funding, so the baseline must be free: open weights
  on Kaggle's free GPUs, with no API.

## 2. The baseline: GLiNER

GLiNER (Zaratiana et al., NAACL 2024) is a span-based generalist NER model
that takes the entity types as natural-language labels.
- **Model:** `urchade/gliner_large-v2.1` (Apache-2.0), with its Hugging
  Face revision and the `gliner` package version pinned. A CPU smoke
  test with the small model comes before any GPU run.
- **Labels:** each corpus's types as words: MaterioMiner's fine-grained
  classes with their camel case split, and MuLMS's and SOFC-Exp's
  documented types written out. MaterioMiner's coarse labels are bare
  abbreviations with no documented expansion, so the zero-shot arm is
  fine-grained only. Writing expansions for them would be a labelling
  choice made after the data were seen.
- **Scoring:** spans are mapped to IOB2 tags on the same words, then
  seqeval is applied as for every system. Overlapping spans are resolved
  by GLiNER's flat-NER decoding.

**Arms, on the held-out units of D7 and paper B:**
- **GZ (zero-shot):** one deterministic run per unit, on MaterioMiner's 4
  papers (fine), MuLMS's 7 subfields and SOFC-Exp's 5 folds. The
  threshold is chosen on each unit's validation sentences over 0.3,
  0.4, 0.5 and 0.6, so no test label sets it.
- **GF (fine-tuned):** GLiNER fine-tuned on the same training sentences
  as the CRF head, with the epoch chosen on the same validation
  sentences. Seed offsets 0–900, as the CRF head has, on all three
  corpora (MaterioMiner at both granularities).

**Decisions (the rule as before):**
- **21.1–21.4:** GF minus the CRF head, paired over the units. MaterioMiner
  fine and coarse (4 papers), MuLMS (7 subfields, the decision seeds
  300–900) and SOFC-Exp (5 folds), each unit the mean of its seeds.
- **Reported, no decision:** GZ against the CRF head; GZ and GF recall on
  unseen and seen names (week 10's split); the detectable differences;
  the post hoc dependence allowance.

**What each outcome means:**
- GF passes on two corpora: a generalist pre-trained tagger transfers
  better than MatSciBERT, and paper B says so.
- GF fails everywhere: paper B's conclusions extend to a generalist
  tagger within the detectable size.
- If GZ is far below the CRF head, a generalist without training does
  not close the gap.

**Budget:** about 20 runs per seed. Small data (234–370 training
sentences), estimated at 1–2 minutes per GLiNER run on a T4. That is
about 6–8 GPU hours for GF, 0.3 for GZ, and a smoke and repeat check.
It runs after `w20a`, within this week's quota (about 16.7 hours left
after `w20a`) or after the 7 November reset.

**Checks:**
- A CPU smoke test that the conversion of spans to tags reproduces the
  gold on gold spans.
- The session's repeat of smoke4.
- GF's seed 0 run twice, which must be identical.

## 3. Paper B for TACL

- **Main text:**
  - one Setup section, merging Sections 2 to 4 into one table and
    prose;
  - the gap and its anatomy (5);
  - seeds (6);
  - the count table, with the post hoc columns (7.7);
  - data against method (8);
  - the generalist baseline (new, short);
  - lessons (9), related work, and limitations.
- **Appendix,** within TACL's caps: 5.1, 7.1–7.6, week 19's screen, and
  the protocol table.
- **Anonymized** (corrected 3 November: TACL allows no supplementary
  material and no links to it, so there is no mirror; ROADMAP §4, week
  21):
  - no repository paths, kernel names or commit hashes;
  - the release is stated in anonymous terms;
  - paper A is cited in the third person;
  - week numbers become stage numbers defined in Setup.
- **Target:** 1 December, or 1 January if weeks 20 and 21 and the review
  rounds need it.
