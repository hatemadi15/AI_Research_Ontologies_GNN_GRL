# Plan toward a methods paper (19 October 2026)

The standing goal: a methods paper, preferably; the evaluation paper if the
methods route is exhausted. This plan says what the methods paper would
claim, why one route is worth the remaining budget, how every kind of compute
we have (Kaggle T4 GPU, Kaggle TPU, this environment's CPU) is used, with
dates, decision points and stop rules. Every test in it is pre-registered in
`ROADMAP.md` §4 before its runs; the evaluation paper is written in parallel
whatever happens, since its findings (`PAPER.md`, C1–C23) are complete.

## 1. Where we stand

What the evidence base says, with the numbers that decide the route:

- **The gap is unseen names.** On a held-out MaterioMiner paper 81% of the
  fine-grained entities are names no training paper annotates with that
  type, and the CRF head finds 35% of them (seen names 88%). On a held-out
  MuLMS subfield 88% are unseen and it finds 39%. Every gain that holds on
  unseen papers is a gain on unseen names (C17). On SOFC-Exp, where the
  articles share one topic, 46% of the names are unseen, the gap is 3.5
  points, and no method helps: when the topic is shared, methods do not
  matter.
- **Which types.** On MuLMS the held-out loss is in MAT (21 points, 40% of
  the entities) and FORM (34), the name types; SAMPLE and DEV are barely
  found anywhere (4.0 and 14.7 F1). Coarse-grained on MaterioMiner, the
  model finds almost every entity (overlap F1 90.7) and most of the strict
  gap is boundaries and types that the papers' annotators set differently:
  not learnable.
- **What works.** Mention replacement (MR) is the one single-model method
  that improves on the CRF head on unseen papers at both granularities
  (+2.77 fine, +1.28 coarse; C20), on unseen names. With multi-task training
  it reaches +6.73 fine-grained (C21). On MuLMS's held-out subfields MR
  gains +2.10 [−0.31, +4.51], which does not pass, while its votes pass
  (C22). Seed votes are worth about +2 everywhere.
- **What does not.** Label semantics, in-domain MLM pre-training,
  self-training with 3 teachers (10 seeds), document context, the
  consistency rule, recall calibration, DeBERTa-v3-large (−4.4 and −8.0 on
  unseen papers), class weights, and names from the other subfields (C23:
  they held 4.4% of a held-out subfield's unseen names and added −0.06).
- **What week 12 tests.** A corpus of each subfield's own topic holds
  55–81% of the held-out subfield's unseen test names (77–90% of the unseen
  MAT names, 85–96% of FORM). Untested: whether the teachers' filter admits
  those names, and whether names in the pool turn into F1 on the held-out
  documents.
- **Power.** With 7 subfields the paired test passes for a mean gain of
  about 2.4–2.9 points. MR already gives +2.1, so a names variant needs
  about half a point more on top of MR to pass against the CRF head.

## 2. The methods bet and what it would claim

**Thesis.** The names of a new topic can be given to a tagger from the
topic's unlabeled text: teachers trained on the labeled data tag the text,
the mentions they agree on confidently become a lexicon, and the lexicon
enters training through mention replacement in gold contexts. Nothing silver
is trained on; no target annotation or external name list is needed.

**Why it can work where self-training did not.** Self-training trains on
silver sentences, whose errors are learned. Mention replacement keeps the
gold contexts and only swaps the names, so what is transferred is the
lexicon, which is exactly what unseen documents lack. The teachers' recall on
unseen names is low (39%), but a 40,000-sentence corpus shows most names in
many contexts, so a name the teachers miss in the test document can still be
harvested from an easy context elsewhere, which is why the topic corpus is a
better source than the test documents alone.

**Claim, if it passes.** "Mention replacement with a lexicon harvested from
unlabeled text of the target topic improves NER on unseen documents of a new
topic", with the pre-registered decisions on MuLMS (7 subfields) and the
MaterioMiner analogue (4 papers), the SOFC-Exp control where it is not
expected to help, and the ablations of §4C.

**Novelty, checked on 19 October.** Dai and Adel (2020) replace mentions
with the training set's own names. Amalvy, Labatut and Dufour (TALN 2022)
use mention replacement to adapt CoNLL-2003 to fantasy literature with
*predefined lists* of target-domain person names (from an annotated target
corpus or WikiGold). BOND (Liang et al., 2020) and PATNER use dictionaries
from knowledge bases as distant supervision and self-train on the silver
labels. EnTDA (Hu et al., 2022) and LLM-DA generate entity-to-text examples.
None harvests the lexicon from unlabeled target-topic text by teacher
agreement and injects it into gold contexts; none evaluates on held-out
topics of a materials corpus. Sources:
[Amalvy et al. 2022](https://aclanthology.org/2022.jeptalnrecital-taln.19.pdf),
[BOND](https://arxiv.org/pdf/2006.15509),
[PATNER](https://par.nsf.gov/biblio/10380151),
[EnTDA](https://arxiv.org/abs/2210.10343v2),
[LLM-DA](https://www.themoonlight.io/en/review/llm-da-data-augmentation-via-large-language-models-for-few-shot-named-entity-recognition),
[Chen et al. 2021](https://arxiv.org/pdf/2109.01758).

**Honest odds.** Decision 1 (against the CRF head) on MuLMS: about 0.45,
because MR is already close to the line and the corpus holds the names.
Decision 2 (names beyond MR): about 0.3. The MaterioMiner analogue, given a
MuLMS pass: about 0.6. A methods paper overall: about 0.35–0.4. The CPU probe
of §4B moves these within days, before any GPU hour is spent.

## 3. What a reviewer will ask, and what the plan answers

1. *Why not plain self-training on the in-topic corpus?* It is the obvious
   baseline, so it joins the screen (V6).
2. *Is it the names or just more augmentation?* The 6-copy control (V3) and
   its names version (V4) separate the two.
3. *Does the topic matter, or would any materials corpus do?* Phase 0 says
   45–65 points of the coverage are generic. The ablation "another topic's
   corpus" (§4C) answers it; if the topic does not matter, the claim is
   "a lexicon from any large unlabeled materials corpus", which is simpler
   and still a method.
4. *Where is the LLM baseline?* Not affordable through an API (no credits
   are spent); a local open model on the TPU is an option for the evaluation
   paper (§5), not a route to a method.
5. *Three attempts on the same corpus.* Reported with the count; the decision
   seeds are disjoint from the screen's.

## 4. The work, ranked

### A. The week-12 screen, amended before any run

The protocol of `ROADMAP.md` §4 stands; amendment 1 (19 October, committed
as 77c37be before any run) adds two variants to the screen, each with the
same 3-seed design and selection rule:

| Variant | What it is | Why |
|---|---|---|
| V1 `crf_mrs_topic` | names by the sentence filter | week 11's method on the right corpus |
| V2 `crf_mrm_topic` | names by the mention filter | admits names from sentences that are not wholly confident |
| V3 `crf_mr6` | 6 copies, no names | the control for more augmentation |
| V4 `crf_mrm_topic6` | names by the mention filter, 6 copies | more exposure per name: with thousands of silver names and about 1,500 replacement slots per fold, 3 copies show most names never |
| V6 `crf_st3_topic` | self-training on the topic corpus (silver sentences trained on, week 4's method) | the baseline a method must beat |

Selection: the largest mean gain over mention replacement, if above 0, goes
to the decision on seeds 300–900 (decisions 1 and 2 as fixed). If V6 wins,
the decision is V6's and the finding is "in-topic self-training", scoped
as such. If V3 wins, names are not the mechanism.

### B. The CPU probe (this week, before any GPU hour)

`silver_lexicon.py`, on this environment's CPU, per held-out subfield:
1. train week 10's 3 CRF teachers (seeds 0, 100, 200; CPU numbers differ
   from the GPU's, which does not matter for a probe);
2. tag the subfield's topic corpus (40,000 sentences) with each, keeping
   the per-word marginals;
3. harvest the lexicon under both filters (sentence, mention), at the
   protocol's threshold 0.9 and at 0.7 and 0.5;
4. measure what each filter admits: the lexicon's size and its coverage of
   the held-out subfield's unseen test names (phase 0 measured the corpus;
   this measures the pool the students would see);
5. the dictionary probe: on week 10's saved CRF predictions (10 seeds), tag
   every maximal verbatim match of a lexicon name where the model predicted
   O, and score it; the same with the gold unseen names as the leaky
   ceiling. This is the gazetteer baseline the method must beat, and it says
   whether the names would convert into F1 if they reached the model.

Readings: if the admitted coverage is far below the corpus coverage, the
protocol's threshold or filter is changed in the amendment (before the
runs). If the dictionary probe gains little even with the gold names, the
mechanism has no ceiling on MuLMS and the route ends there, with the GPU
hours saved for the evaluation paper's remaining items. Cost: about 2.5–4
hours per subfield on 2 lanes; all 7 by about 22 October.

### C. If names pass: the ablations that make the paper

On the TPU if the spike succeeds (its own references), else on the GPU in
the week of 31 October, 10 seeds each on MuLMS:
- another topic's corpus (the topic's role; phase 0's coverage table says
  which pairs are informative: sofc with electrolysis's corpus, poly with
  graphene's);
- corpus size (10,000, 20,000, 40,000 sentences);
- the filter's threshold and the number of teachers (1, 2 of 3, 3 of 3);
- the share of silver names in the draws (capped at 50%, uncapped);
- MaterioMiner: the winner on the multi-task model (the strongest
  MaterioMiner system is MT+MR) and SOFC-Exp with the SOFC corpus (the
  control where unseen names are few).

### D. If names fail and in-topic self-training passes

The claim is the baseline's: "in-topic unlabeled text helps through
self-training where cross-topic text does not". Scope it, run the
MaterioMiner analogue (week 4's `crf_st3` already exists there and did not
pass at 10 seeds, so the claim would be MuLMS-only, which is weak), and
weigh it against the evaluation paper.

### E. If both fail: one last mechanism test, then stop

The transductive variant: names from the held-out subfield's own documents
(their text, never their labels), the standard unsupervised
domain-adaptation setting, declared as such. It runs only if the probe's
dictionary ceiling shows that names which reach the model do help (else
there is nothing to reach for). It is one pre-registered decision, 7 GPU
hours, and the last attempt: after it the paper is the evaluation paper.

### F. Not pursued, and why

Bigger general-domain encoders (DeBERTa-v3-large loses 4–8 points on unseen
papers), LLM APIs (no credits), extra corpora with heads (MuLMS's authors
found no NER gain), label semantics, document context, the consistency rule,
recall calibration, votes across methods (cannot repeat on a second corpus):
all tested or ruled out, in `ROADMAP.md`.

## 5. Compute: what each is for

| Resource | Capacity | Used for | Rules |
|---|---|---|---|
| Kaggle T4 GPU | about 30 hours a week, reset Friday 00:00 UTC (next: 24 October); 12-hour sessions, 2 at a time | every pre-registered decision (comparable with the existing GPU references) | the existing bundle and batch rules |
| Kaggle TPU v5e-8 | separate quota (20 hours a week historically; the account's settings show the current figure); 9-hour sessions; 8 cores | a bounded feasibility spike, then the ablations of §4C and any new reference family | TPU numbers are compared only with TPU numbers (as CPU with CPU, GPU with GPU) |
| this environment's CPU | 4 cores, 15 GB; 2 lanes of 2 threads as in weeks 2–3: a 340-sentence CRF run in 15–20 minutes, tagging 40,000 sentences in about 75 minutes per teacher | the probe (§4B), corpus building, every analysis, CRF-only references if needed | results written per run and committed daily (the container can be reclaimed) |

**The TPU spike** (cap: 2 days of engineering, 1 session of at most 3 TPU
hours). Kaggle replaced the v3-8 with the v5e-8 in 2025
([announcement](https://www.kaggle.com/product-announcements/607202)); a
2025-10-07 environment update broke some setups, so the environment is
pinned ([report](https://www.kaggle.com/questions-and-answers/611001)).
PyTorch/XLA runs on Kaggle TPU VMs
([PyTorch/XLA](https://github.com/standardgalactic/pytorch-xla)); the loop is
spawned once per core, and all XLA calls must happen inside the spawned
function
([forum](https://discuss.pytorch.org/t/enable-multiprocessing-on-pytorch-xla-for-tpu-vm/177673)).
The port: `--device xla` in `ner_baseline.py` with fixed-length padding (one
compilation per shape), the CRF forward on the device, Viterbi on the CPU,
fp32 matmul precision, one independent job per core (8 runs in parallel,
different seeds or folds). It is developed and tested here with
`PJRT_DEVICE=CPU` (XLA on the CPU) before the session. The session runs one
MaterioMiner CRF job on one chip (`tpu-smoke1`), then the same job on all 8
chips at once (`tpu-smoke`): the wall time per run with every chip busy, and
whether the 8 files are identical. Go if a run on one chip, with all 8 busy,
takes no longer than the same run on a T4 (8 chips against a session's 2
GPUs: 4× a T4 session's throughput); otherwise the TPU is dropped and the
ablations wait for GPU weeks.

**GPU budget, week of 24 October** (hours): `w11b` 2.1; `w12a` 14.6 (lane 0:
V1 3.2, V4 3.2, one `crf_mr6` job 0.7; lane 1: V2 3.2, V6 3.0, two `crf_mr6`
jobs 1.4; about 7.5 hours of wall time); `w12b` 7; the MaterioMiner analogue
2.1 if decision 1 passes. Total about 26 of 30.

## 6. Schedule and decision points

| When | What | Decides |
|---|---|---|
| 19–23 Oct (no GPU) | the CPU probe on all 7 subfields; V4 and V6 in the code with tests; amendment 1 committed; the TPU port tested on XLA-CPU; related work for the names route | the amendment's filter/threshold; whether the route has a ceiling |
| 24 Oct (reset) | `w11b`; then `w12a`; the TPU spike session in parallel | week-11 decisions 3–4; the screen's winner; TPU go/no-go |
| 25–27 Oct | `w12b:<winner>`; the MaterioMiner analogue if decision 1 passes | the methods criterion |
| 31 Oct–7 Nov | ablations (§4C) on TPU or GPU; or route E; the evaluation paper's remaining items either way | the paper's claims |
| 7–14 Nov | the paper assembled as what the results say; title and abstract fixed by 14 Nov | — |

Stop rules: the methods route gets the two GPU weeks after the reset and
route E at most; no fourth protocol on MuLMS. A pass on MuLMS alone (the
MaterioMiner analogue failing) is reported as a scoped result inside the
evaluation paper, not as a methods paper.

## 7. The paper in each outcome

**Methods paper.** Title on the lines of "Giving a tagger the names of a new
topic: lexicons from unlabeled text for materials NER on unseen documents".
Sections: the unseen-name analysis (C17, C22) as the motivation; the method;
pre-registered decisions on MuLMS and MaterioMiner; the SOFC-Exp control;
ablations; the evaluation findings (random splits, seeds, votes) as the
protocol that makes the claim credible. The current `PAPER_DRAFT.md` keeps
its §2–5 and gains the method as §6's centre.

**Evaluation paper.** As drafted: the gap and its anatomy, the pre-registered
tests of every popular addition with 10 seeds and three corpora, the seed
and split lessons, the reproduction, and the methods attempts as negative
results with the attempt count. Its abstract and introduction get the
conditional sentence removed on 14 November.

## 8. Risks

- The filter admits few names (week 11 admitted 181–419 sentences of
  8,000): the probe measures it first, and the amendment can lower the
  threshold before the runs.
- Dilution: with thousands of silver names, 3 copies show each name rarely;
  V4 tests more copies, and the silver-share cap is an ablation.
- The container is reclaimed mid-probe: per-subfield outputs are written as
  they finish and committed daily; a lost subfield is rerun.
- The TPU port stalls: capped at 2 days and 1 session; the plan does not
  depend on it.
- Kaggle queues and 429s: the existing rules (one bundle, pull before push,
  no more than 2 sessions).
- Licences: the corpora's text and the names are never committed; reports
  with counts and SHA-256 are, and the corpora rebuild from the caches.
