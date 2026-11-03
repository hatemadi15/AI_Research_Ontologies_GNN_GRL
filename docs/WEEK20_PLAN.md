# Week 20 plan (3 November 2026): two controls paper B's first referee asked for

*Status: run and decided on 3 November: 20.1, 20.2 and 20.3 pass (`docs/ROADMAP.md` §4, the week-20 outcome). Checked before any run by an independent referee (a Fable-model
referee, 3 November): "approve with required changes", with nine required
items and seven suggestions. Every factual claim it made was checked
against the files, and its required items are applied below, as are six
of the seven suggestions (the third MuLMS arm is not). The binding
protocol is `docs/ROADMAP.md` §4's week-20 protocol, committed before any
run; this file keeps the reasons. The first draft is in this file's
history.*

## 1. Why

Paper B's first review round (3 November; a referee, a number audit and a
consistency check of `44f9677`, recorded in `docs/PAPER_B_NOTES.md`,
committed with this protocol) found two confounds in paper B's headline
claims that only new runs can resolve.

1. **MuLMS's held-out-subfield gap mixes topic shift with document
   diversity.**
   - A held-out subfield's training pool (`ner_baseline.subfield_folds`,
     `--train-pool 400`) takes the other subfields' articles in a
     shuffled order until they hold 400 sentences: 2–3 articles, the last
     cut to its first sentences, from 1–3 subfields
     (`decision_tests.MULMS_POOLS`).
   - The random-split pool (`pool_split`) draws 400 sentences from all
     sentences and touches 49–50 of the 50 articles.
   - So the 17-point gap of paper B's abstract (the CRF head, seeds 0–900,
     48.66 against 65.72) compares 2–3 training articles with about 50, at
     equal size. Paper B's own Section 8 shows that document diversity at
     equal size matters on MaterioMiner.
2. **The week-8 diversity decisions rest on 3 seeds.** "A third paper adds
   +1.18 and +3.59 at equal sentence counts" (week 8 part 2, seeds 0, 100
   and 200) heads one of the abstract's three lessons. Yet Section 6.2
   shows that 3-seed designs mislead, and the fine-grained interval does
   not survive the dependence allowance paper A applies: [−0.29, +2.65].

## 2. The designs, and what the referee changed

**20a.** A new `--pool-by-sentence` (with `--leave-one-subfield-out
--train-pool N`) shuffles the other subfields' sentences, not their
documents, before the first N are taken. The CRF head runs on the 7
held-out subfields with seed offsets 0–900 (`crf_spool`). Decision 20.1 is
the sentence pool against the stored article pool, paired over the 7
subfields. The referee required:
- the reading of each outcome, fixed now;
- the detectable difference stated now from a named proxy, rather than
  after the result;
- a repeat of the stored article-pool runs, on the current code path;
- a disclosure of everything else that differs between the two pools:
  - their subfield diversity (1–3 subfields against all six);
  - their validation sentences;
- the remaining gap described as what it is: topic shift and the novelty
  of unseen documents together, with test-set composition;
- the dependence rule for the post hoc reading, fixed now.

**20b.** Week 8 part 2's two sides at seed offsets 300–900: three papers at
the size of two (`crf_p3n2`), and the two papers alone (the `crf_bias`
inner folds read at D = 0). Decisions 20.2 and 20.3 are taken on those 7
seeds, which week 8 did not see, as week 12 decided MuLMS on seeds it had
not screened. The referee required:
- that the 10-seed reading cannot alter the verdicts;
- the repeat checks' tags, comparison and failure rule;
- the precedent cited correctly.

**Code hazards the referee found, both fixed with tests before any run:**
- `ner_seed_analysis.week12_variants` would read `crf_spool` as a week-12
  variant and change the stored analysis (as week 18's `crf_mr6_bias` once
  did).
- `ner_mentions.run_split` rebuilds a run's training sentences from its
  `config` and knows nothing of a sentence pool. The seen/unseen split of
  `crf_spool` would silently use the article pool's names.

## 3. Not in this plan, and why

- **An LLM or generalist-NER baseline** (the referee's R11) is a new method
  on new code. It would spend the author's API credits or pull a new
  model. It is the author's choice, to be made with the choice of venue;
  until then paper B is scoped to fine-tuned encoder taggers, and the item
  stays open in `docs/PAPER_B_NOTES.md`.
- **Adjudicating 200 held-out errors as convention or model error** (R13)
  is manual annotation of corpus text. Paper B's wording is softened
  instead.
- **Re-selecting the epoch on a held-out training paper** (R14) would
  re-run the whole CRF reference. It is a limitation in paper B.
- **The Fable referee's optional third arm**: MuLMS document folds with a
  sentence pool, tested on held-out documents of seen subfields. It would
  split the remaining gap into topic shift and document novelty. It is not
  run: it needs a new fold design on MuLMS, so it waits for the result of
  20.1.
