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

**Novelty, checked on 19 October (a second search the same evening).**
The pieces are old; the combination and the setting are not. Kozareva
(EACL 2006 SRW) generated gazetteers from unlabeled Spanish news by
preposition patterns, used them as features and, when bootstrapping, added
an unlabeled example only when two classifiers agreed on it. Zafarian and
Asghari (NSURL 2019) decoded an unlabeled Farsi news corpus with a
supervised tagger, kept the entities whose word and sentence confidence
passed a threshold as a gazetteer, and retrained with the gazetteer as a
feature: the harvest by a tagger is theirs, with one model, no agreement
filter, and the lexicon entering as a feature rather than through the
training contexts. Song, Lawrie, Finin and Mayfield (FLAIRS 2020) found
Wikidata match features worth +0.52 F1 on OntoNotes, and that replacing
annotated entities with random gazetteer entries of the same type "did not
produce statistically significant improvements": mention replacement with
an external lexicon failed where the test names were not the bottleneck.
Dai and Adel (2020) replace with the training set's own names; Amalvy,
Labatut and Dufour (TALN 2022; arXiv 2023) adapt CoNLL-2003 to fantasy
novels with *predefined lists* of target names and gain recall at a
precision cost; COSINER (Bartolini et al., 2024) picks the training-set
replacement that fits the context best; Torres et al. (2024) find mention
replacement helps small training sets and hurts larger ones (BERT on
MaSciP, a materials synthesis corpus, gains at every size); Sharma et al.
(ACL 2023) find that most augmentations make the tagger memorise entities
and score unseen entities separately. On the teacher side, Wu et al. (ACL
2020) train a student on unlabeled target-language text from several
teachers, and Qu et al. (AAAI 2023) use two teachers' agreement and
disagreement in distantly supervised NER; BOND (Liang et al., 2020) and
PATNER self-train on dictionary-matched silver labels; EnTDA (Hu et al.,
2022) and LLM-DA generate entity-to-text examples; GEMNET (Meng et al.,
NAACL 2021) gates gazetteer features against the context. None harvests
the lexicon of an unseen topic from that topic's unlabeled text by the
agreement of several teachers and injects it into gold contexts by
mention replacement with the topic held out of the labeled data, and none
evaluates on held-out topics of a materials corpus. The claim is therefore
about the combination and the setting, never about a piece, and the paper
owes two baselines: the lexicon as a dictionary rule, which the probe
already prices (fill: +5.9 on sofc, −10.0 on steel), and, if names pass,
the lexicon as a match feature on the encoder output (Song et al.'s
one-hot match; one more variant, on the GPU, in §4C). Sources:
[Kozareva 2006](https://aclanthology.org/E06-3004/),
[Zafarian and Asghari 2019](https://aclanthology.org/2019.nsurl-1.16/),
[Song et al. 2020](https://arxiv.org/abs/2003.03072),
[Amalvy et al. 2022](https://aclanthology.org/2022.jeptalnrecital-taln.19.pdf),
[Amalvy et al. 2023](https://arxiv.org/abs/2302.04555),
[COSINER](https://ceur-ws.org/Vol-3741/paper20.pdf),
[Torres et al. 2024](https://arxiv.org/abs/2411.14551),
[Sharma et al. 2023](https://aclanthology.org/2023.acl-long.390/),
[Wu et al. 2020](https://aclanthology.org/2020.acl-main.581/),
[Qu et al. 2023](https://arxiv.org/abs/2212.06522),
[BOND](https://arxiv.org/pdf/2006.15509),
[PATNER](https://par.nsf.gov/biblio/10380151),
[EnTDA](https://arxiv.org/abs/2210.10343v2),
[LLM-DA](https://www.themoonlight.io/en/review/llm-da-data-augmentation-via-large-language-models-for-few-shot-named-entity-recognition),
[GEMNET](https://aclanthology.org/2021.naacl-main.118/),
[Chen et al. 2021](https://arxiv.org/pdf/2109.01758).

**After 14.2 (29 October, 22:30).** The topic's own corpus against a mixed corpus of the six other topics: −0.18 [−2.01, +1.65] paired by subfield, no; the mixed variant gains +3.38 [−0.89, +7.64] over the CRF head and +0.70 [−1.19, +2.58] over the copy control (reported), with 18.3% of the unseen test names among its silver names against 22.2%. Both of week 14's decisions fail. What the method is, as the paper can state it: mention replacement with 6 copies, its pool holding names that three CRF teachers find in a large unlabeled materials corpus of any topic; it improves on the CRF head on held-out documents of two corpora (13.2, pre-registered, +3.58 [+0.95, +6.21]); the names' share beyond the copies is +0.8 [−0.7, +2.3] and not established (14.1); the corpus's topic adds nothing (14.2). Two framings remain, and the choice is the user's: a methods paper whose main pass is that gain, with the two ablations stated as its limits, or the evaluation paper with these as its method findings ("what helps on unseen documents is more augmentation; harvested names add little, from any corpus"). §7 is written for both; the week-15 work is the write-up either way, after `w14c`'s reported ablations at the reset.

**After 14.1 (29 October, 17:30).** The names against 6 copies without names, pooled over the same 11 units and seeds as 13.2: +0.80 [−0.66, +2.26], no (MuLMS +0.51, MaterioMiner fine-grained +1.29); the control alone gains +2.56 [+0.94, +4.18] over the CRF head (reported). The claim's wording changes with it: V4 improves on the CRF head on held-out documents of two corpora (13.2, pre-registered), about +2.6 of its +3.58 being mention replacement with 6 copies and about +0.8 the harvested names, whose share is not established. 14.2 (the topic's role, after the reset) decides whether the source of the names matters; a fail there with the mixed variant as good leaves "mention replacement with more copies, and names from any large materials corpus" as the method, and the paper's title drops "the names of a new topic". Odds that 14.2 passes: about 0.3 (the mixed corpora hold 49–70% of the unseen names against the topic corpora's 55–81%, and V4's whole gain over the control is +0.5 on MuLMS).

**After 13.4 (25 October, 11:30).** The third corpus does not pass (−0.04 [−1.21, +1.14] against the CRF head on SOFC-Exp's folds; +0.45 [+0.04, +0.86] over plain mention replacement, reported). The paper is a methods paper with a two-corpus claim (C26: pooled +3.58 [+0.95, +6.21]; MaterioMiner +4.27 and +1.77; MuLMS +3.19 without a pass alone), a second, smaller two-corpus result (C25, the consistency rule), and one boundary condition shared by both: on SOFC-Exp, where half the test names are seen in training, neither helps. Week 13's GPU work is done (33.3 of 45 hours used this week).

**After 13.2 (25 October).** The MaterioMiner analogue passed at both granularities (+4.27 [+1.39, +7.15] fine-grained, +1.77 [+0.67, +2.88] coarse-grained) and the pooled decision 13.2 passed (+3.58 [+0.95, +6.21], no corpus losing): the revised methods criterion is met, so the paper is a methods paper on two corpora (C26), with the consistency rule (C25) as a second, smaller result and SOFC-Exp as the boundary case for mention replacement (13.3a −0.48). Its open questions: the third corpus (13.4, running; a pass there is about 0.3, as only 36% of SOFC-Exp's unseen names are in its corpus and only half its test names are unseen) and the names' share beyond plain mention replacement (+1.23 [−0.22, +2.68] pooled, post hoc; not established).

**Odds after the decision (24 October).** Decisions 1 and 2 failed (+3.19 [−1.13, +7.51] and +0.92 [−1.63, +3.48]); the screen's estimate held, the 7-subfield test lacked the power. 13.2, the pooled decision over 11 units, is the remaining route: about 0.5 (it passes if the MaterioMiner analogue gains about +2 or more on its 4 papers, where mention replacement gained +2.77 with a spread of 0.8); 13.4 runs only after a pass. A methods paper through the names method: about 0.35, through 13.2; the consistency rule's two-corpus claim (C25) stands either way.

**Honest odds, revised 20 October after the probe and the screen's first part.** The feature variants (V7, V8) and the copy control (V3) are not above mention replacement on the 3-seed screen, and the probe shows the lexicon's value swinging with the subfield's recall gap; decision 1 on MuLMS for one of V1, V2, V4 and V6 is now about 0.3, decision 2 about 0.15, a methods paper about 0.2. *Revised again the same day after the screen's second part:* V4 `crf_mrm_topic6` wins the screen, +1.45 [−1.35, +4.25] over mention replacement and +3.16 [−1.96, +8.28] over the CRF head (V2 +0.97, V1 +0.80, V6 +0.16; sofc +7.4 and graphene +3.1 over mention replacement, steel −1.8). Decision 1 on MuLMS: about 0.3 (V4's own between-subfield spread against the CRF head is 5.5 points on the screen, sofc +12.9 against poly −3.7, so 7 subfields detect a mean near 5, not the 2.4–2.9 of weeks 10–11; +3.16 passes only if the spread shrinks on 7 seeds per subfield); decision 2: about 0.15; the MaterioMiner analogue against the CRF head: about 0.5 (mention replacement passed there with a spread of 0.8); the pooled decision 13.2 over 11 units: about 0.4 (half-width near 2.9 for a mean near 3); a methods paper through this route: about 0.3. A third corpus is pre-registered for a pass of decision 1 or of 13.2 (ROADMAP §4, amendment 13.4: the analogue on SOFC-Exp's article folds with the SOFC corpus, which holds 36% of the folds' unseen names); its pass given that: about 0.3. *As written on 19 October:* Decision 1 (against the CRF head) on MuLMS: about 0.45,
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
6. *Why not use the lexicon as a feature, or as a dictionary?* Both are
   baselines the paper owes (§2): the dictionary rule is priced by the probe
   on every subfield; the match feature joins the ablations if names pass
   (§4C).

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
| V7 `crf_gzm_topic` | V2's names as a gazetteer match feature of the plain CRF head (amendment 2) | the injection baseline the related work demands; a learned term can weigh a lexicon whose value swings by subfield |
| V8 `crf_gzs_topic` | V1's names (the sentence filter) as the same match feature (amendment 3) | the probe finds the sentence filter's lexicon the safer dictionary and the mention filter's the richer; the feature is tried on both |

Selection: the largest mean gain over mention replacement, if above 0, goes
to the decision on seeds 300–900 (decisions 1 and 2 as fixed). If V6 wins,
the decision is V6's and the finding is "in-topic self-training", scoped
as such. If V3 wins, names are not the mechanism.

**Amendment 2 (19 October, adopted; committed before any run).** The
related work (§2) makes the lexicon as a *feature* the baseline a reviewer
will ask for: the same harvested names, injected not through mention
replacement but as a match feature of the plain CRF head
(`--names-feature`: a term learned from zero at learning rate 1e-2 on two
columns per type, first word and later words of every verbatim match;
variant `crf_gzm_topic`, batch `w12c` for its 3-seed screen, `w12b:gzm`
for its decision). It answers on the screen's seeds whether the injection
mechanism or the lexicon carries the gain. If V7 wins the screen, the
claim becomes "a lexicon harvested from the topic's unlabeled text,
injected as a feature, improves NER on unseen topics", and mention
replacement with the lexicon is its ablation; if an MR variant wins, V7 is
the feature baseline the paper reports. Budget: about 2 session
hours for `w12c` (§5), within the week's remaining quota.
The bundle is rebuilt from the commit that closes the probe.

**Amendment 3 (20 October, adopted; committed before any run).** V8
`crf_gzs_topic`, the match feature on the sentence filter's lexicon, joins
`w12c` (7 jobs per lane, about 3.5 session hours). The probe's six
subfields price the sentence filter's 0.9 lexicon as the safer dictionary
(mean fill +1.8, never below −0.3) and the mention filter's as the richer
and the riskier (mean −0.3, steel −10.0); the feature learns its own
trust, and the screen tries it on both. The threshold stays 0.9.

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

**Outcome of the probe (20 October).** Seven subfields in
`results/silver_lexicons/`, the reading in ROADMAP §4 ("The seven
subfields"): the mention filter at 0.9 admits 9–34% of the unseen test
names and the sentence filter 5–26%; as a dictionary the lexicon helps
where the held-out head's recall gap is 15 points or more (sofc +5.9,
pemfc +4.0, semi +2.3) and hurts elsewhere (steel −10.0, graphene −5.1,
poly −3.3), averaging −1.0 (mention) and +1.1 (sentence) over the seven;
0.9 is the right threshold for both filters; the agreement filter picks
context-free names (pemfc: silver +4.0 against gold −10.5). No amendment
of the threshold; amendments 2 and 3 (the feature on both lexicons) are
the probe's consequence. The dictionary rule per subfield is the baseline
reported beside the screen.

### C. If names pass: the ablations that make the paper

*Fixed on 29 October as the week-14 protocol (ROADMAP §4), after 13.2 passed: two decisions, 14.1 the names against the 6-copy control (pooled over MuLMS and MaterioMiner, 11 units) and 14.2 the topic's own corpus against a mixed corpus of the six other topics (7 subfields), and the ablations below at the screen's 3 seeds, reported: the corpus's size (10,000 and 20,000 sentences), the threshold (0.7), the agreement (2 of 3 teachers) and the silver share (capped at 0.5). Not run: the winner on the multi-task model (one corpus) and the match feature (screened in week 12, below mention replacement); the dictionary rule is Table 5.*

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
- the two lexicon baselines the related work demands (§2): the dictionary
  rule on the winner's predictions (no training; the probe's code) and the
  lexicon as a one-hot match feature on the encoder output (Song et al.
  2020), 10 seeds on MuLMS, decided against the CRF head and reported
  against the winner.

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
| Kaggle T4 GPU | 45 hours a week since 19 October (30 before), charged per session hour whatever the session's two GPUs do (`w11b`: 3.4 hours for a 3.4-hour session); reset Friday 00:00 UTC (next: 24 October); 12-hour sessions, 2 at a time; 12.7 hours left this week after `w11b` | every pre-registered decision (comparable with the existing GPU references) | the existing bundle and batch rules |
| Kaggle TPU v5e-8 | 20 hours a week on paper; none attached in six sessions (19 October) | nothing for now (the spike's outcome below); the port stays | TPU numbers are compared only with TPU numbers (as CPU with CPU, GPU with GPU) |
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

*Outcome, 19 October: no-go, for want of a TPU.* The port is done and
tested on XLA-CPU (repeats exactly; the CPU path byte-identical). Six
Kaggle sessions, five pushed through the API (script and notebook kernels
with `enable_tpu`) and one committed from the editor with "TPU VM v5e-8"
chosen, all ran on the standard CPU image without a TPU (no accelerator
devices or variables, 0.00 TPU hours charged); each notebook session lost
77 minutes in libtpu's metadata lookups before failing at the first XLA
call, and the editor's draft session with the same setting never started.
The spike's cap is spent. The TPU leaves the schedule; the ablations run
on the GPU, whose quota rose to 45 hours a week the same day. The XLA
port stays in the code for a TPU Kaggle may attach another day
(`tpu-smoke1`, `tpu-smoke`, the go/no-go rule above).

**GPU budget** (session hours, which is what Kaggle charges; the earlier
figures counted lane hours and doubled them). From `w11a`'s 7 MuLMS jobs of
3 teachers and 10 students (16–31 minutes each) a screen job of 3 students
takes about 16 minutes, a variant's 7 jobs about 1.9 hours of lane time:
`w12a` about 5 hours (lane 0: V1, V4 and one `crf_mr6` job; lane 1: V2, V6
and two `crf_mr6` jobs), `w12c` about 2 hours (V7 over two lanes with the
smoke repeat), `w12b` about 2.5 hours (7 jobs of 3 or 4 students per lane),
the MaterioMiner analogue about 1.1 hours. In all about 11 hours, within the
12.7 left this week after `w11b` (3.4): the whole of week 12 runs before the
reset, as soon as the probe's reading fixes the filter and the bundle is
rebuilt.

## 6. Schedule and decision points

| When | What | Decides |
|---|---|---|
| 19–23 Oct (no GPU) | the CPU probe on all 7 subfields; V4 and V6 in the code with tests; amendment 1 committed; the TPU port tested on XLA-CPU; related work for the names route | the amendment's filter/threshold; whether the route has a ceiling |
| 24 Oct (reset) | `w12a`'s remaining jobs (V6, `crf_mr6`), then `w12b:<winner>`; `w11b`, `w12c` and the names part of `w12a` already run this week on the raised quota | week-11 decisions 3–4; the screen's winner |
| 25–27 Oct | `w12b:<winner>`; the MaterioMiner analogue if decision 1 passes | the methods criterion |
| 31 Oct–7 Nov | ablations (§4C) on the GPU; or route E; the evaluation paper's remaining items either way | the paper's claims |
| 7–14 Nov | the paper assembled as what the results say; title and abstract fixed by 14 Nov | — |
| 1–7 Nov (decided 30 Oct: two papers, §7) | paper A's draft, the methods paper, after `w14c`'s reported ablations | — |
| 8–14 Nov | paper B's draft, the evaluation paper; both titles and abstracts fixed by 14 Nov | — |

**Revised on 20 October (the user's directive: push as hard as possible for
a methods paper).** The stop rule below is lifted: week 13 (ROADMAP §4) adds
the consistency rule on MuLMS (free), a pooled decision across corpora for
the week-12 winner, and mention replacement with and without multi-task
training on SOFC-Exp's article folds, with a revised methods criterion (two
corpora, or a pooled decision with no corpus losing); the attempt count is
reported in the paper. Post hoc, the pooled estimate for plain mention
replacement over MaterioMiner's papers and MuLMS's subfields is +2.34
[+0.94, +3.75] fine-grained, which is what a pooled decision for a variant
at least as good as mention replacement would be expected to show; the
narrow miss of decision 1 on MuLMS (+2.10 [−0.31, +4.51]) is a power
problem of 7 units, not a sign problem (6 of 7 subfields gain).

*13.1 done (20 October).* The consistency rule on the CRF head passes on
MuLMS (+0.82 [+0.18, +1.46], every subfield gaining) and, from week 7, on
coarse-grained MaterioMiner (+0.12), and loses on SOFC-Exp's article folds
(−0.44 [−0.72, −0.16]) as on its official split; it adds nothing to mention
replacement on either corpus. By the revised criterion that is a methods
claim of two corpora, scoped and under a point, with a boundary condition
the annotation convention explains. It is not the paper's main method;
13.2 and 13.3 (mention replacement pooled across corpora and on SOFC-Exp's
folds) decide whether mention replacement joins it, after the reset.

*Week 14 (29 October).* With 13.2 passed, the ablations of §4C are pre-registered as the week-14 protocol (ROADMAP §4): 14.1 names against matched copies and 14.2 the topic's role, each a decision, and four ablations reported; `w14a` (the copy control, about 5.5 session hours) runs on this week's remaining quota, `w14b` and `w14c` (about 4.5 and 9 hours) after the reset of 31 October. The paper's assembly moves to the week of 7 November.

Stop rules (as written on 19 October): the methods route gets the two GPU weeks after the reset and
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

**Decided on 30 October: both papers.** The user chose to write both; the
division below keeps every decision a result in one paper and a citation
in the other.

- *Paper A, the methods paper* (working title: "Mention replacement with
  more copies and harvested names improves materials NER on unseen
  documents"; it no longer names the topic). Its claims: C26 as stated
  after week 14 (the pre-registered, pooled gain over the CRF head on
  held-out documents of two corpora, +3.58 [+0.95, +6.21]; the names'
  share beyond matched copies +0.8 [−0.7, +2.3], not established; no
  difference detected between the topic's corpus and a mixed corpus of
  other topics, −0.18 [−2.01, +1.65]); the ablations of 14.3 (reported);
  the SOFC-Exp boundary condition (13.4); the consistency rule as a
  second, smaller result (C25); the unseen-name mechanism (C17, C22, the
  recall-for-precision trade) as motivation and explanation; the
  evaluation protocol in one section, as the reason the numbers can be
  trusted; the attempt count. Draft: `docs/PAPER_A.md`.
- *Paper B, the evaluation paper* (draft: `docs/PAPER_B.md`, begun 30
  October from `docs/PAPER_DRAFT.md`):
  the gap between random splits and unseen documents and its anatomy
  (C2, C17, C22), seed variance and votes (C3–C6), the reproduction and
  the split question (C1), data against method (C19), the pre-registered
  tests of the popular additions with their attempt counts (C7–C16,
  C20–C24), and the lessons for evaluating on unseen documents. It cites
  A for the method and keeps each of A's results to one summary row.
- *Shared material* (the corpora, the protocol, the CRF head, the seed
  design) is described in both, briefly in A with a pointer to B.

Schedule: `w14c`'s reported ablations at the reset (31 October); week 15
(1–7 November) paper A's draft (abstract, introduction, method,
experiments with Table 6 and the ablations, the mechanism figure,
limitations; related work cut from §8); week 16 (8–14 November) paper B's
draft from `PAPER_DRAFT.md` §2–5, 7 and 8 with the method sections cut to
summaries (a first complete draft was written on 30 October, `docs/PAPER_B.md`); the titles and abstracts of both fixed by 14 November; then
the submission checklists (the reproducibility appendix and the code
release for both, drafted 30 October in `docs/REPRODUCIBILITY.md`; the authors' confirmation email, the user's; the
venues, the user's: a materials-informatics venue for A, a
scholarly-document-processing or NLP-evaluation venue for B; the
shortlist with deadlines is `docs/VENUES.md`). No GPU
decision is planned after `w14c`; a reviewer's cheap reported item (the
winner on the multi-task model, one corpus) stays an option.

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
