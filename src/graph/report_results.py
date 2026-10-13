"""
report_results.py - regenerate the results sections of the docs from results/*.json

Rewrites two generated blocks and leaves the rest of each file alone:

    README.md         between <!-- RESULTS:START --> and <!-- RESULTS:END -->
    docs/ROADMAP.md   between <!-- STANDING:START --> and <!-- STANDING:END -->

Every number comes from the JSON files in results/; only the wording around the
numbers lives here. Standard library only (CI has no torch, seqeval or scipy),
and all paths are derived from this file's location.

Usage:
  python src/graph/report_results.py                   # rewrite both files in place
  python src/graph/report_results.py --check           # exit 1 and print a diff if out of date
  python src/graph/report_results.py --pending "NER coarse CRF" "NER ensemble"
  python src/graph/report_results.py --results DIR --readme PATH --roadmap PATH   # for tests

Exit codes: 0 = written / up to date, 1 = `--check` found a difference, 2 = bad
input (missing or invalid file, missing marker, bad arguments).

NER comparisons (results/ner_comparisons.json, written by `ner_baseline.py --compare`)
---------------------------------------------------------------------------------------
An entry compares one variant with a reference run on the same split and model
seed. Its `reference` field says which run: `baseline` (plain MatSciBERT) or
`crf` (the CRF head); entries without the field are older baseline comparisons.

  * "NER training variants vs the baseline": variants `cw` and `crf`, reference
    `baseline`. Also feeds the "At a glance" table and the ROADMAP standing table.
  * "NER week-2 variants vs the CRF head": variants `crf_mt` (multi-task), `crf_dapt`
    (domain-adaptive pre-training) and `crf_ens3` (3-model ensemble), reference `crf`.
    Its intro notes that these runs use 2 CPU threads and the CRF head 3; once the
    `crf_m0` comparisons are complete it also gives their difference to the CRF head,
    the effect of the thread count alone.
    Nothing from this table goes into "At a glance" or the standing table yet.

Which entries get a row or a verdict is decided by an allow-list of variants plus the
reference, so anything else in the file never gets one: ensemble members (`crf_m0`,
`crf_m100`, `crf_m200`; `crf_m0` only appears in the thread-count note above),
bare-tag variants, comparisons against another reference.
A row or verdict needs a complete comparison: 5 pairs for random splits (5 splits or
5 model seeds), 4 for leave one paper out. Until the first complete week-2 comparison
exists the generated text is identical to what it was before they were supported.
"""
import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

GRANULARITIES = ('fine', 'coarse')

# Variants shown in the tables and verdicts, with their display names. Only variants
# listed here are ever displayed (see the module docstring).
BASELINE_VARIANTS = {'cw': 'class-weighted loss', 'crf': 'CRF head'}
WEEK2_VARIANTS = {'crf_mt': 'multi-task (fine + coarse heads)',
                  'crf_dapt': 'domain-adaptive pre-training',
                  'crf_ens3': '3-model ensemble'}

PROTOCOL_NAMES = {'random_split_vary': 'new random split per seed',
                  'leave_one_paper_out': 'leave one paper out',
                  'random_split_fixed': 'fixed split'}
# Pairs a comparison needs before it is shown: the 5 splits (or model seeds) and the
# 4 held-out papers.
REQUIRED_PAIRS = {'random_split_vary': 5, 'random_split_fixed': 5, 'leave_one_paper_out': 4}
PROTOCOL_FILE_PART = {'random_split_fixed': '', 'random_split_vary': '_random_split_vary',
                      'leave_one_paper_out': '_leave_one_paper_out'}

NER_PROTOCOLS = [('', 'fixed split (seed 0), 5 model seeds'),
                 ('_random_split_vary', 'new random split per seed'),
                 ('_leave_one_paper_out', 'leave one paper out')]

COMPARISON_COLUMNS = ("| Variant | Protocol | Granularity | Pairs | Variant micro F1 [95% CI] | "
                      "Δ micro [95% CI] | Δ macro [95% CI] | Δ strict micro |")


class ReportError(Exception):
    """An input problem the caller can fix: missing or invalid file, missing marker."""


class Results:
    """Cached reader for the JSON files of one results directory."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self._cache = {}

    def get(self, name):
        """Parsed content of <directory>/<name>, or None if the file does not exist."""
        if name not in self._cache:
            path = self.directory / name
            data = None
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding='utf-8'))
                except ValueError as exc:
                    raise ReportError(f"{path} is not valid JSON: {exc}") from exc
            self._cache[name] = data
        return self._cache[name]

    def require(self, name):
        data = self.get(name)
        if data is None:
            raise ReportError(f"missing results file {self.directory / name}")
        return data

    def comparisons(self):
        """results/ner_comparisons.json ({} when absent)."""
        comps = self.get('ner_comparisons.json') or {}
        if not isinstance(comps, dict):
            raise ReportError(f"{self.directory / 'ner_comparisons.json'} must hold an object "
                              "with one entry per results file")
        return {key: c for key, c in comps.items() if isinstance(c, dict)}


# --------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------
def fmt(x, digits=3):
    return '–' if x is None else f"{x:.{digits}f}"


def ci_text(c):
    return f"[{c[0]:.2f}, {c[1]:.2f}]"


def interval(m, signed=False):
    """'mean [lo, hi]' of a summary (mean, ci95) or of a paired difference (mean_diff, ci95)."""
    f = '{:+.2f}' if signed else '{:.2f}'
    return (f"{f.format(m['mean'] if 'mean' in m else m['mean_diff'])} "
            f"[{f.format(m['ci95'][0])}, {f.format(m['ci95'][1])}]")


def delta_cell(d):
    lo, hi = d['ci95']
    return f"{d['mean_diff']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def dig(d, *keys):
    """d[k1][k2]... or None as soon as a level is missing."""
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def ner_file(granularity, protocol, variant=None):
    """Name of a NER results file, e.g. ner_results_fine_random_split_vary_crf.json."""
    return (f"ner_results_{granularity}{PROTOCOL_FILE_PART[protocol]}"
            f"{'_' + variant if variant else ''}.json")


def has_runs(r, n):
    """True if the NER results file `r` summarises at least `n` runs."""
    return bool(r and r.get('summary') and r['summary']['n_runs'] >= n)


def baseline_ner(res):
    """Baseline results per granularity as (fixed split, varied splits, leave one paper out)."""
    return tuple({g: res.get(ner_file(g, protocol)) for g in GRANULARITIES}
                 for protocol in ('random_split_fixed', 'random_split_vary', 'leave_one_paper_out'))


# --------------------------------------------------------------------------------
# NER comparisons: which entries count, and what they say
# --------------------------------------------------------------------------------
def reference_of(c):
    """Run a comparison is measured against; entries without the field are baseline ones."""
    return c.get('reference') or 'baseline'


def is_complete(c):
    """True if a comparison has all its pairs (see REQUIRED_PAIRS)."""
    need = REQUIRED_PAIRS.get(c.get('protocol'))
    return need is not None and (c.get('n_pairs') or 0) >= need


def selected_comparisons(comps, variants, reference):
    """(key, entry) of the complete comparisons of `variants` against `reference`.

    Ordered as the tables are: by variant (in the order of `variants`), then
    protocol, then granularity. Everything else in `comps` is left out: other
    variants (ensemble members included), other references, incomplete comparisons.
    """
    wanted = [(key, c) for key, c in comps.items()
              if c.get('variant') in variants and reference_of(c) == reference
              and c.get('granularity') in GRANULARITIES and is_complete(c)]
    return sorted(wanted, key=lambda kc: (list(variants).index(kc[1]['variant']),
                                          list(PROTOCOL_NAMES).index(kc[1]['protocol']),
                                          GRANULARITIES.index(kc[1]['granularity'])))


def find_comparison(comps, granularity, protocol, variant, reference):
    """The complete comparison of one variant for one protocol and granularity, or None."""
    for c in comps.values():
        if (c.get('variant') == variant and c.get('granularity') == granularity
                and c.get('protocol') == protocol and reference_of(c) == reference
                and is_complete(c)):
            return c
    return None


def paired_verdict(c):
    """(word, numbers) of a paired micro-F1 comparison: improves, hurts or no significant change."""
    d, s = c['micro'], c['micro_strict']
    lo, hi = d['ci95']
    word = 'improves' if lo > 0 else ('hurts' if hi < 0 else 'no significant change')
    return word, (f"{d['mean_diff']:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}]; strict IOB2 "
                  f"{s['mean_diff']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]")


# --------------------------------------------------------------------------------
# Tables (the part of README.md between the glance table and "Reading the numbers")
# --------------------------------------------------------------------------------
def typing_row(name, gold, ty, note=''):
    return (f"| {name} | {gold} | {fmt(ty['accuracy'])} {ci_text(ty['accuracy_ci95'])}{note} | "
            f"{fmt(ty['set_f1'])} | {fmt(ty['macro_f1'])} | {fmt(ty['acc@5'])} | "
            f"{fmt(ty['h_f1'])} |")


def term_typing_tables(res):
    off = res.require('eval_pipeline_llm_off.json')
    on = res.get('eval_pipeline_llm_on.json')
    v5off, v5on = res.get('eval_v5_rescored_llm_off.json'), res.get('eval_v5_rescored_llm_on.json')
    abl = res.get('ablation_summary.json') or []
    oracle = next((a for a in abl if a['preset'] == 'oracle'), None)
    tt = res.get('term_typing_mindf2.json')

    out = ["#### Term typing: 315 graph terms, gold spans given\n",
           "| System | Gold types used? | Accuracy [95% CI] | Set F1 (LLMs4OL) | Macro-F1 | "
           "Acc@5 | Hier. F1 |",
           "|---|---|---|---|---|---|---|"]
    ref = off['reference_majority_class']['accuracy']
    out.append(f"| Majority-class reference | – | {fmt(ref)} | – | – | – | – |")
    out.append(typing_row('**Pipeline v6, LLM off**', 'no', off['typing']))
    if on:
        out.append(typing_row('**Pipeline v6, LLM on** (RAG re-ranking)', 'no', on['typing']))
    if oracle:
        out.append(f"| Pipeline v6, ORACLE (upper bound) | **yes** | "
                   f"{fmt(oracle['typing_accuracy'])} "
                   f"{ci_text(oracle['typing_accuracy_ci95'])} | {fmt(oracle['typing_set_f1'])} | "
                   f"{fmt(oracle['typing_macro_f1'])} | {fmt(oracle['typing_acc@5'])} | "
                   f"{fmt(oracle['typing_h_f1'])} |")
    if v5off:
        out.append(typing_row('v5 outputs re-scored, LLM off', '**yes** (leak)', v5off['typing'],
                              f" (coverage {v5off['typing']['coverage']:.2f})"))
    if v5on:
        out.append(typing_row('v5 outputs re-scored, LLM on', '**yes** (leak)', v5on['typing'],
                              f" (coverage {v5on['typing']['coverage']:.2f})"))

    if tt:
        out.append("\n#### Term-typing baselines (`term_typing_baseline.py`): 5-fold grouped CV, "
                   "mean ± sd over folds\n")
        out.append("| Method | Setting | Accuracy | Set F1 | Macro-F1 | Acc@5 | Hier. F1 | "
                   "Acc. on classes unseen in training |")
        out.append("|---|---|---|---|---|---|---|---|")
        for m, r in tt['methods'].items():
            mean, sd = r['mean_over_folds'], r['std_over_folds']
            runs = r['n_fold_runs']
            out.append(f"| {m} | {r['setting']} ({runs} folds) | {fmt(mean['accuracy'])} ± "
                       f"{fmt(sd['accuracy'])} | "
                       f"{fmt(mean['set_f1'])} | {fmt(mean['macro_f1'])} | {fmt(mean['acc@5'])} | "
                       f"{fmt(mean['h_f1'])} | {fmt(mean['acc_unseen_classes'])} |")
    return out


def other_levels_table(res):
    off, on = res.get('eval_pipeline_llm_off.json'), res.get('eval_pipeline_llm_on.json')
    v5off = res.get('eval_v5_rescored_llm_off.json')
    rel_off, rel_on = res.get('relation_eval_llm_off.json'), res.get('relation_eval_llm_on.json')
    return [
        "\n#### Other levels: old metric vs leakage-free metric\n",
        "| Level | Old v5 metric (invalid) | Leakage-free metric | v5 outputs | v6 LLM off | "
        "v6 LLM on |",
        "|---|---|---|---|---|---|",
        "| Type | Type F1 0.997 | – (dataset property: 177/177 labels are class names) "
        "| – | – | – |",
        f"| Term | Term F1@0.5 0.893 (0.935 LLM on) | alignment pairwise F1@0.5 | "
        f"{fmt(dig(v5off, 'alignment_rows', 'thresholds', '0.5', 'f1'))} | "
        f"{fmt(dig(off, 'alignment_rows', 'thresholds', '0.5', 'f1'))} | "
        f"{fmt(dig(on, 'alignment_rows', 'thresholds', '0.5', 'f1'))} |",
        f"| Concept | Concept F1 0.447 | clustering ARI / B-cubed F1 | "
        f"{fmt(dig(v5off, 'clustering', 'ari'))} / {fmt(dig(v5off, 'clustering', 'bcubed_f1'))} "
        f"(gold-type clusters) | "
        f"{fmt(dig(off, 'clustering', 'ari'))} / {fmt(dig(off, 'clustering', 'bcubed_f1'))} | "
        f"{fmt(dig(on, 'clustering', 'ari'))} / {fmt(dig(on, 'clustering', 'bcubed_f1'))} |",
        f"| Taxonomy | – | class-level edge F1 (precision vs closure; chance "
        f"{fmt(dig(off, 'taxonomy', 'random_precision_closure'))}) | "
        f"{fmt(dig(v5off, 'taxonomy', 'f1'))} "
        f"({fmt(dig(v5off, 'taxonomy', 'precision_closure'))}) | "
        f"{fmt(dig(off, 'taxonomy', 'f1'))} ({fmt(dig(off, 'taxonomy', 'precision_closure'))}) | "
        f"{fmt(dig(on, 'taxonomy', 'f1'))} ({fmt(dig(on, 'taxonomy', 'precision_closure'))}) |",
        f"| Relations | partial-entity recall 0.925 | class-level F1 vs 70 restrictions | – | "
        f"{fmt(dig(rel_off, 'predicted_typing', 'f1'))} | "
        f"{fmt(dig(rel_on, 'predicted_typing', 'f1'))} |",
    ]


def paired_delta(a):
    d, p = a.get('typing_diff_vs_base'), a.get('typing_mcnemar_p')
    if d is None or p is None:
        return '–'
    return f"{d:+.3f} ({'<0.001' if p < 0.001 else f'{p:.2f}'})"


def ablation_table(res):
    abl = res.get('ablation_summary.json') or []
    if not abl:
        return []
    out = ["\n#### Ablations (`run_ablation.py`, LLM off; oracle = upper bound)\n",
           "Δ is the paired typing-accuracy difference to `full_pipeline` on the same 315 terms; "
           "p is the exact McNemar test.\n",
           "| Preset | Typing acc. | Δ vs full (p) | Set F1 | Macro-F1 | Hier. F1 | Clust. ARI | "
           "Taxonomy P (closure) | Relation class-F1 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for a in abl:
        name = a['preset'] + (' *(oracle)*' if a['oracle'] else '')
        out.append(f"| {name} | {fmt(a['typing_accuracy'])} | {paired_delta(a)} | "
                   f"{fmt(a['typing_set_f1'])} | {fmt(a['typing_macro_f1'])} | "
                   f"{fmt(a['typing_h_f1'])} | {fmt(a['clustering_ari'])} | "
                   f"{fmt(a['taxonomy_precision_closure'])} | {fmt(a['relations_class_f1'])} |")
    return out


def ner_protocol_table(res):
    rows = []
    for suffix, label in NER_PROTOCOLS:
        for g in GRANULARITIES:
            r = res.get(f'ner_results_{g}{suffix}.json')
            if r and r.get('runs') and r.get('summary'):
                rows.append((suffix, label, g, r))
    if not rows:
        return []
    out = ["\n#### NER (MatSciBERT, MaterioMiner protocol, seqeval entity F1)\n",
           "Mean ± population SD over runs, and the 95% Student-t CI of the mean. Micro F1 is the "
           "primary metric: the authors' 2025 follow-up states micro averaging, while the 2024 "
           "paper only says \"averaging over the five random initializations and entity types\". "
           "The published numbers come from one unpublished random split and have no reported "
           "SD.\n",
           "| Protocol | Granularity | Runs | Micro F1 [95% CI] | Macro F1 | Weighted F1 | "
           "Strict micro | Published |",
           "|---|---|---|---|---|---|---|---|"]

    def cell(m):
        lo, hi = m['ci95']
        return f"{m['mean']:.2f} ± {m['pop_sd']:.2f} [{lo:.2f}, {hi:.2f}]"

    for suffix, label, g, r in rows:
        s = r['summary']
        n = s['n_runs']
        runs = f"{n} papers" if suffix == '_leave_one_paper_out' else f"{n} seeds"
        published = ('– (out of distribution)' if suffix == '_leave_one_paper_out'
                     else f"{r['paper_test_f1']:.2f}")
        out.append(f"| {label} | {g} | {runs} | {cell(s['micro'])} | {s['macro']['mean']:.2f} | "
                   f"{s['weighted']['mean']:.2f} | {s['micro_strict']['mean']:.2f} | "
                   f"{published} |")
    return out


def comparison_table(res, variants, reference, title, intro):
    """Paired-comparison table of `variants` against `reference` ([] if no row is complete)."""
    rows = selected_comparisons(res.comparisons(), variants, reference)
    if not rows:
        return []
    out = [title, intro, COMPARISON_COLUMNS, "|---|---|---|---|---|---|---|---|"]
    for key, c in rows:
        v = (res.get(f'{key}.json') or {}).get('summary', {}).get('micro')
        vcell = f"{v['mean']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]" if v else '–'
        out.append(f"| {variants[c['variant']]} | {PROTOCOL_NAMES[c['protocol']]} | "
                   f"{c['granularity']} | {c['n_pairs']} | {vcell} | {delta_cell(c['micro'])} | "
                   f"{delta_cell(c['macro'])} | {c['micro_strict']['mean_diff']:+.2f} |")
    return out


def ner_variants_table(res):
    """cw and crf against the baseline."""
    return comparison_table(
        res, BASELINE_VARIANTS, 'baseline',
        "\n#### NER training variants vs the baseline (paired by split and model seed)\n",
        "Δ = variant − baseline on the same split and model seed, with the 95% Student-t CI of "
        "the mean difference. Pre-registered rule: a variant improves on the baseline only if "
        "the micro-F1 interval over the 5 random splits lies entirely above 0.\n")


THREAD_NOTE = ("Week-2 runs use 2 CPU threads and the CRF head 3. With the same seeds, the "
               "thread count alone changes a run (floating-point summation order), which adds "
               "noise to each difference.")


def thread_note(res):
    """THREAD_NOTE, plus the measured effect once the `crf_m0` comparisons are complete.

    `crf_m0` is the CRF head re-run with 2 threads on the same splits and model seeds, so
    its difference to the CRF head is the effect of the thread count alone.
    """
    comps = res.comparisons()
    found = [(g, find_comparison(comps, g, 'random_split_vary', 'crf_m0', 'crf'))
             for g in GRANULARITIES]
    parts = [f"{g}-grained {delta_cell(c['micro'])}" for g, c in found if c]
    if not parts:
        return THREAD_NOTE
    return (f"{THREAD_NOTE} Re-running the CRF head with 2 threads (ensemble member `crf_m0`) "
            f"changes micro F1 by {' and '.join(parts)} over the 5 random splits.")


def ner_week2_table(res):
    """Multi-task, domain-adaptive pre-training and the 3-model ensemble against the CRF head."""
    return comparison_table(
        res, WEEK2_VARIANTS, 'crf',
        "\n#### NER week-2 variants vs the CRF head (paired by split and model seed)\n",
        "Δ = variant − CRF head on the same split and model seed, with the 95% Student-t CI of "
        "the mean difference. Pre-registered rule: a variant improves on the CRF head only if "
        f"the micro-F1 interval over the 5 random splits lies entirely above 0. "
        f"{thread_note(res)}\n")


def build_tables(res):
    """All tables between the "At a glance" table and "Reading the numbers"."""
    out = (term_typing_tables(res) + other_levels_table(res) + ablation_table(res)
           + ner_protocol_table(res) + ner_variants_table(res) + ner_week2_table(res))
    return '\n'.join(out).strip()


# --------------------------------------------------------------------------------
# "Reading the numbers" notes
# --------------------------------------------------------------------------------
def pending_note(pending):
    """Note for runs that were still going when the snapshot was taken ('' if none)."""
    if not pending:
        return ''
    return ('\n\n> **Still running when this snapshot was taken:** ' + ', '.join(pending) +
            '. These rows will be added when the runs finish.')


def ablation_note(res):
    a = {row['preset']: row for row in (res.get('ablation_summary.json') or [])}
    if not a:
        return ''
    return (
        f"* Ablations: no component changes typing significantly. The GNN adds "
        f"{a['full_pipeline']['typing_accuracy'] - a['no_gnn']['typing_accuracy']:.3f} "
        f"({a['full_pipeline']['typing_accuracy']:.3f} vs {a['no_gnn']['typing_accuracy']:.3f}, "
        f"McNemar p = {a['no_gnn']['typing_mcnemar_p']:.2f}), and the whole pipeline beats the "
        f"plain-SBERT baseline by {-a['baseline_sbert']['typing_diff_vs_base']:.3f} "
        f"(p = {a['baseline_sbert']['typing_mcnemar_p']:.2f}). The GNN embeddings *hurt* "
        f"clustering "
        f"(ARI {a['full_pipeline']['clustering_ari']:.3f} vs {a['no_gnn']['clustering_ari']:.3f} "
        f"with plain SBERT). The oracle preset (gold types allowed) reaches "
        f"{a['oracle']['typing_accuracy']:.3f}, which is what the v5 evaluation was effectively "
        f"reporting.\n")


def verdict_line(res, comps, variant, name, reference, versus, sota=False):
    """One "Reading the numbers" bullet for a variant ('' if no comparison is complete).

    Verdicts are the paired micro-F1 differences (improves / hurts / no significant
    change, with strict IOB2 numbers) over the 5 random splits and leaving one paper
    out. `versus` completes "vs ..." in the bullet. `sota` appends the pre-registered
    check against the published number, which only makes sense against the baseline.
    """
    split_parts, ood_parts, beaten = [], [], []
    for g in GRANULARITIES:
        c = find_comparison(comps, g, 'random_split_vary', variant, reference)
        if c:
            word, num = paired_verdict(c)
            split_parts.append(f"{g}-grained {word} ({num})")
            v = res.get(ner_file(g, 'random_split_vary', variant))
            if (sota and word == 'improves' and v
                    and v['summary']['micro']['ci95'][0] > v['paper_test_f1']):
                sm = v['summary']['micro']
                beaten.append(f"{g}-grained {sm['mean']:.2f} [{sm['ci95'][0]:.2f}, "
                              f"{sm['ci95'][1]:.2f}] vs {v['paper_test_f1']:.2f}")
        c = find_comparison(comps, g, 'leave_one_paper_out', variant, reference)
        if c:
            word, num = paired_verdict(c)
            ood_parts.append(f"{g}-grained {word} ({num})")
    if not (split_parts or ood_parts):
        return ''
    line = f"* NER, {name} vs {versus} (paired micro-F1 difference): "
    line += ('; '.join(split_parts) + ' across the 5 random splits' if split_parts else '')
    if ood_parts:
        line += ('. Leaving one paper out: ' if split_parts else 'leaving one paper out: ')
        line += '; '.join(ood_parts)
    if beaten:
        line += ('. Pre-registered SOTA check (a paired gain over the baseline and a 95% CI '
                 'above the published number): met for ' + '; '.join(beaten)
                 + ', if the paper reports micro F1')
    return line + '.\n'


def mean_precision_recall(result):
    n = len(result['runs'])
    return (sum(r['precision'] for r in result['runs']) / n,
            sum(r['recall'] for r in result['runs']) / n)


def crf_explanation(res, comps, vary):
    """Why the CRF helps (exploratory): default seqeval scoring vs strict IOB2."""
    parts = []
    for g in GRANULARITIES:
        c = find_comparison(comps, g, 'random_split_vary', 'crf', 'baseline')
        b, v = vary[g], res.get(ner_file(g, 'random_split_vary', 'crf'))
        if not (c and b and v):
            continue
        (bp, br), (vp, vr) = mean_precision_recall(b), mean_precision_recall(v)
        bs, st = b['summary']['micro_strict'], c['micro_strict']
        slo, shi = st['ci95']
        if slo > 0:
            reading = (f"under strict IOB2 scoring the CRF still gains {st['mean_diff']:+.2f} "
                       f"[{slo:+.2f}, {shi:+.2f}], so part of the gain is better recognition")
        else:
            reading = (f"under strict IOB2 scoring the two are level ({st['mean_diff']:+.2f} "
                       f"[{slo:+.2f}, {shi:+.2f}]), so the gain is span validity alone. Dropping "
                       "invalid tags from the baseline's output gives the same benefit for free: "
                       f"its strict F1 is {bs['mean']:.2f} [{bs['ci95'][0]:.2f}, "
                       f"{bs['ci95'][1]:.2f}] against the published {b['paper_test_f1']:.2f}")
        parts.append(f"{g.capitalize()}-grained: precision {bp:.1f} → {vp:.1f}, recall "
                     f"{br:.1f} → {vr:.1f}; {reading}")
    if not parts:
        return ''
    return ("* Why the CRF helps (exploratory, not pre-registered): it never predicts an invalid "
            "I- tag, while seqeval's default (conlleval) scoring counts the baseline's stray I- "
            "tags as extra, mostly wrong entities; strict IOB2 scoring ignores them. "
            + '. '.join(parts) + ".\n")


def ner_notes(res):
    """The NER bullets of "Reading the numbers"."""
    fixed, vary, lopo = baseline_ner(res)
    comps = res.comparisons()
    note = ''
    if all(has_runs(fixed[g], 1) for g in GRANULARITIES):
        f_, c_ = fixed['fine']['summary'], fixed['coarse']['summary']
        note += (
            f"* NER, fixed split: plain MatSciBERT reaches micro F1 "
            f"{f_['micro']['mean']:.2f} (fine) "
            f"and {c_['micro']['mean']:.2f} (coarse) over 5 seeds, against the published 69.92 / "
            f"72.32. Macro F1 is much lower ({f_['macro']['mean']:.2f} / "
            f"{c_['macro']['mean']:.2f}), "
            f"so the comparison favours this repo only if the paper's \"averaging over ... entity "
            f"types\" means micro averaging, as in the authors' 2025 follow-up.\n")
    if all(has_runs(vary[g], 5) for g in GRANULARITIES):
        verdicts = []
        for g in GRANULARITIES:
            s, paper = vary[g]['summary']['micro'], vary[g]['paper_test_f1']
            lo, hi = s['ci95']
            if lo > paper:
                v = f"holds: {s['mean']:.2f}, 95% CI [{lo:.2f}, {hi:.2f}], above {paper:.2f}"
            elif hi < paper:
                v = f"reverses: {s['mean']:.2f}, 95% CI [{lo:.2f}, {hi:.2f}], below {paper:.2f}"
            else:
                v = (f"is not significant: {s['mean']:.2f}, 95% CI [{lo:.2f}, {hi:.2f}] "
                     f"includes {paper:.2f}")
            verdicts.append(f"{g}-grained the gain {v}")
        note += ("* NER across splits (`--vary-split`, a new random split per seed, micro F1; "
                 "pre-registered rule: the gain holds if the 95% CI lies above the published "
                 "number): " + '; '.join(verdicts) + ". Macro F1 across splits is "
                 f"{vary['fine']['summary']['macro']['mean']:.2f} (fine) and "
                 f"{vary['coarse']['summary']['macro']['mean']:.2f} (coarse).\n")
    else:
        note += ("* NER across splits: the check with a new random split per seed "
                 "(`--vary-split`) and with one paper held out (`--leave-one-paper-out`) is "
                 "running. Until it finishes, the fixed-split gain is not a SOTA claim.\n")
    if all(has_runs(lopo[g], 4) for g in GRANULARITIES):
        parts = []
        for g in GRANULARITIES:
            s = lopo[g]['summary']['micro']
            parts.append(f"{s['mean']:.2f} {g}-grained (per paper "
                         f"{min(s['per_run']):.1f}–{max(s['per_run']):.1f})")
        note += ("* NER out of distribution (`--leave-one-paper-out`: train on three papers, "
                 "test on the fourth): micro F1 " + ' and '.join(parts) + ". These numbers are "
                 "not comparable with 69.92 / 72.32.\n")

    for var, name in BASELINE_VARIANTS.items():
        note += verdict_line(res, comps, var, name, 'baseline', 'the baseline', sota=True)
    note += crf_explanation(res, comps, vary)
    for var, name in WEEK2_VARIANTS.items():
        note += verdict_line(res, comps, var, name, 'crf', 'the CRF head')
    note += ensemble_member_note(res)
    return note


def ensemble_member_note(res):
    """Exploratory bullet: the 3-model vote against the mean of its own members.

    Read from `summary.vs_member_mean` of the crf_ens3 results files that have all
    their runs ('' if none has).
    """
    parts = []
    for protocol, phrase in (('random_split_vary', 'across the 5 random splits'),
                             ('leave_one_paper_out', 'leaving one paper out')):
        found = []
        for g in GRANULARITIES:
            r = res.get(ner_file(g, protocol, 'crf_ens3'))
            gain = dig(r, 'summary', 'vs_member_mean')
            if gain and has_runs(r, REQUIRED_PAIRS[protocol]):
                found.append(f"{g}-grained {delta_cell(gain)}")
        if found:
            parts.append(f"{'; '.join(found)} {phrase}")
    if not parts:
        return ''
    return ("* NER, 3-model ensemble vs the mean of its own members on the same split "
            "(exploratory, not pre-registered; no single reference run enters): "
            + '; '.join(parts) + ".\n")


# --------------------------------------------------------------------------------
# "At a glance" table and the generated blocks
# --------------------------------------------------------------------------------
def crf_gain_note(comps, g):
    """Short reading of a CRF gain for the at-a-glance table."""
    c = find_comparison(comps, g, 'random_split_vary', 'crf', 'baseline')
    if not c:
        return ''
    st = c['micro_strict']
    if st['ci95'][0] > 0:
        return f"; {st['mean_diff']:+.2f} of the gain survives strict IOB2 scoring"
    return "; the gain comes from valid spans only (see below)"


def glance_table(res):
    """The "At a glance" table with its trailing blank line ('' if it has no row).

    Only the baseline variants (cw, crf) appear here; the week-2 variants do not yet.
    """
    _, vary, lopo = baseline_ner(res)
    comps = res.comparisons()
    off, on = res.get('eval_pipeline_llm_off.json'), res.get('eval_pipeline_llm_on.json')
    tt = res.get('term_typing_mindf2.json')
    llm = tt['methods'].get('hybrid+llm', {}).get('mean_over_folds', {}) if tt else {}

    glance = ["**At a glance** (95% CIs in brackets; details below)", "",
              "| Task | Result in this repo | Reference |", "|---|---|---|"]
    if all(has_runs(vary[g], 5) for g in GRANULARITIES):
        for g in GRANULARITIES:
            s, paper = vary[g]['summary']['micro'], vary[g]['paper_test_f1']
            if s['ci95'][0] > paper:
                ref = f"published {paper:.2f}: **beaten** if the paper used micro F1"
            elif s['ci95'][1] < paper:
                ref = f"published {paper:.2f}: below"
            else:
                ref = f"published {paper:.2f}: not significantly better"
            glance.append(f"| NER, {g}-grained, 5 random splits (micro F1) | {interval(s)} | "
                          f"{ref} |")
    if all(has_runs(lopo[g], 4) for g in GRANULARITIES):
        glance.append(f"| NER on an unseen paper (leave one paper out) | "
                      f"{lopo['fine']['summary']['micro']['mean']:.2f} fine, "
                      f"{lopo['coarse']['summary']['micro']['mean']:.2f} coarse | "
                      "no published equivalent |")
    for var, vname in BASELINE_VARIANTS.items():
        found = {g: find_comparison(comps, g, 'random_split_vary', var, 'baseline')
                 for g in GRANULARITIES}
        cells = [f"{g} {interval(found[g]['micro'], signed=True)}"
                 for g in GRANULARITIES if found[g]]
        if cells:
            glance.append(f"| NER with a {vname}: paired gain over the baseline | "
                          f"{'; '.join(cells)} | counts only if the CI lies above 0 |")
        for g in GRANULARITIES:
            c = found[g]
            v = res.get(ner_file(g, 'random_split_vary', var))
            if (c and c['micro']['ci95'][0] > 0
                    and v and v['summary']['micro']['ci95'][0] > v['paper_test_f1']):
                glance.append(f"| NER, {g}-grained with the {vname}, 5 random splits (micro F1) | "
                              f"{interval(v['summary']['micro'])} | published "
                              f"{v['paper_test_f1']:.2f}: **beaten** if the paper used micro F1"
                              f"{crf_gain_note(comps, g) if var == 'crf' else ''} |")
    if off:
        glance.append(f"| Term typing, 315 terms, pipeline (zero-shot) | accuracy "
                      f"{off['typing']['accuracy']:.3f}"
                      + (f" ({on['typing']['accuracy']:.3f} with LLM re-ranking)" if on else '')
                      + f" | supervised + LLM baseline {llm.get('accuracy', 0):.3f}; "
                      "majority class 0.083 |")
        glance.append(f"| Clustering / taxonomy / relations, class level | ARI "
                      f"{off['clustering']['ari']:.3f} / edge F1 {off['taxonomy']['f1']:.3f} / "
                      "F1 0.000 | near chance; see the roadmap |")
    return '\n'.join(glance) + '\n\n' if len(glance) > 4 else ''


def readme_block(res, note=''):
    """The generated block of README.md, markers included."""
    return f"""<!-- RESULTS:START -->
All numbers are leakage-free and come from the JSON files in [`results/`](results/)
(CPU, `SEED=0`; LLM = `openai/gpt-4o-mini` via OpenRouter). Gold term spans
are given to the pipeline and the typing baselines. "Gold types used" marks
outputs whose predictions had access to the gold labels: those rows are
reported only to show the size of the leak.

{glance_table(res)}{build_tables(res)}{note}

**Reading the numbers**
* The old "Term F1 0.91" corresponds to *0.12* pairwise alignment F1. The
  v5 outputs score higher (0.27, and 0.57–0.74 typing accuracy) only because
  their top-1 rows were the gold classes injected by `type_match`.
* The pipeline's zero-shot typing (0.38; 0.43 with LLM re-ranking) beats plain
  label matching (0.36) but stays well below the supervised hybrid (0.50) and
  hybrid + LLM (0.65) baselines, which can also use labelled terms. Closing
  that gap is the first item of the roadmap.
* Taxonomy and relations are near zero at class level. The term-level
  taxonomy and relation extractors were never designed to recover MMO
  `subClassOf` edges or restrictions; see the roadmap.
{ablation_note(res)}{ner_notes(res)}<!-- RESULTS:END -->"""


def standing_block(res, note=''):
    """The generated block of docs/ROADMAP.md (the "where we stand" table), markers included."""
    off = res.require('eval_pipeline_llm_off.json')
    on = res.get('eval_pipeline_llm_on.json')
    tt = res.get('term_typing_mindf2.json')
    fixed, vary, lopo = baseline_ner(res)

    ner_cells = []
    for g in GRANULARITIES:
        if not has_runs(fixed[g], 1):
            continue
        s = fixed[g]['summary']
        cell = (f"{g}: micro {s['micro']['mean']:.2f} [{s['micro']['ci95'][0]:.2f}, "
                f"{s['micro']['ci95'][1]:.2f}] (macro {s['macro']['mean']:.2f}) on the fixed split")
        if has_runs(vary[g], 5):
            v = vary[g]['summary']['micro']
            cell += f", {v['mean']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] over 5 new splits"
        if has_runs(lopo[g], 4):
            cell += f", {lopo[g]['summary']['micro']['mean']:.2f} leaving one paper out"
        ner_cells.append(cell)

    crf_cells = []
    for g in GRANULARITIES:
        v = res.get(ner_file(g, 'random_split_vary', 'crf'))
        out_of_dist = res.get(ner_file(g, 'leave_one_paper_out', 'crf'))
        if v and v.get('summary', {}).get('n_runs', 0) >= 5:
            m = v['summary']['micro']
            cell = (f"{g}: micro {m['mean']:.2f} [{m['ci95'][0]:.2f}, {m['ci95'][1]:.2f}] "
                    "over 5 new splits")
            if out_of_dist and out_of_dist.get('summary'):
                cell += (f", {out_of_dist['summary']['micro']['mean']:.2f} "
                         "leaving one paper out")
            crf_cells.append(cell)

    hyb = tt['methods']['hybrid']['mean_over_folds'] if tt else {}
    llm = tt['methods'].get('hybrid+llm', {}).get('mean_over_folds', {}) if tt else {}
    lines = [
        "| Task | Best leakage-free result in this repo | Reference |",
        "|---|---|---|",
        "| NER, fine / coarse (MatSciBERT, MaterioMiner protocol) | "
        + ('; '.join(ner_cells) if ner_cells else 'run pending (`ner_baseline.py`)')
        + " | 69.92 / 72.32 (Kumar et al. 2024; averaging not stated) |",
    ]
    if crf_cells:
        lines.append(f"| NER with the CRF head (`--crf`) | {'; '.join(crf_cells)} | same; the "
                     "in-distribution fine-grained gain over the baseline disappears under strict "
                     "IOB2 scoring |")
    lines += [
        f"| Term typing, 315 terms, pipeline (zero-shot) | accuracy {off['typing']['accuracy']:.3f}"
        + (f" (LLM on: {on['typing']['accuracy']:.3f})" if on else '')
        + " | majority class 0.083 |",
        f"| Term typing, 5-fold CV baselines | hybrid {hyb.get('accuracy', 0):.3f}; hybrid + LLM "
        f"{llm.get('accuracy', 0):.3f} (set F1 {llm.get('set_f1', 0):.3f}) | LLMs4OL 2025 MatOnto "
        "best F1 0.667 (different dataset) |",
        f"| Clustering vs gold classes | ARI {off['clustering']['ari']:.3f}, B-cubed F1 "
        f"{off['clustering']['bcubed_f1']:.3f} (all-singletons baseline "
        f"{off['clustering']['baseline_singletons']['bcubed_f1']:.3f}) | – |",
        f"| Taxonomy, class level | edge F1 {off['taxonomy']['f1']:.3f}; precision vs closure "
        f"{off['taxonomy']['precision_closure']:.3f} (chance "
        f"{off['taxonomy']['random_precision_closure']:.3f}) "
        "| LLMs4OL 2025 MatOnto best F1 0.662 (different dataset) |",
        "| Relations, class level vs 70 restrictions | F1 0.000 | no text-level gold exists |",
    ]
    return ("<!-- STANDING:START -->\n" + '\n'.join(lines) + note
            + "\n\nFull tables: [README](../README.md#results).\n<!-- STANDING:END -->")


# --------------------------------------------------------------------------------
# Rewriting the documents
# --------------------------------------------------------------------------------
def replace_block(text, name, block):
    """Replace the generated block `name` ('RESULTS' or 'STANDING') of `text` with `block`.

    The block runs from the START marker to the END marker, each alone at the start of
    its line. A document that has only the bare placeholder `<!-- NAME -->` gets the
    block in its place (first use). Anything else is an error: silently doing nothing
    would let `--check` pass on a document that is no longer generated.
    """
    start, end = f'<!-- {name}:START -->', f'<!-- {name}:END -->'
    found = list(re.finditer(rf'^{re.escape(start)}.*?^{re.escape(end)}', text, flags=re.S | re.M))
    if len(found) > 1:
        raise ReportError(f"more than one {start} ... {end} block")
    if found:
        return text[:found[0].start()] + block + text[found[0].end():]
    placeholder = f'<!-- {name} -->'
    if placeholder in text:
        return text.replace(placeholder, block)
    raise ReportError(f"no {start} ... {end} block (each marker alone at the start of a line) "
                      f"and no {placeholder} placeholder")


def render(res, readme_text, roadmap_text, pending=()):
    """New contents of README.md and docs/ROADMAP.md, as (readme, roadmap)."""
    note = pending_note(pending)
    return (replace_block(readme_text, 'RESULTS', readme_block(res, note)),
            replace_block(roadmap_text, 'STANDING', standing_block(res, note)))


def read_document(path):
    try:
        return Path(path).read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise ReportError(f"cannot read {path}: {getattr(exc, 'strerror', None) or exc}") from exc


def unified_diff(path, old, new):
    return '\n'.join(difflib.unified_diff(old.split('\n'), new.split('\n'),
                                          fromfile=f'{path} (current)',
                                          tofile=f'{path} (generated)', lineterm=''))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog='report_results.py', allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Regenerate the results block of README.md and the standing table of "
                    "docs/ROADMAP.md from the JSON files in results/. By default both files "
                    "are rewritten in place.",
        epilog="exit codes: 0 written or up to date, 1 --check found a difference, "
               "2 bad input")
    p.add_argument('--check', action='store_true',
                   help="write nothing; print a unified diff and exit 1 if either file would "
                        "change")
    p.add_argument('--pending', nargs='+', metavar='TEXT', default=[],
                   help="names of runs still going; adds a 'Still running when this snapshot "
                        "was taken' note below the tables and the standing table")
    p.add_argument('--results', type=Path, default=REPO_ROOT / 'results', metavar='DIR',
                   help="directory with the result JSON files (default: results/ of the repo)")
    p.add_argument('--readme', type=Path, default=REPO_ROOT / 'README.md', metavar='PATH',
                   help="README to update (default: README.md of the repo)")
    p.add_argument('--roadmap', type=Path, default=REPO_ROOT / 'docs' / 'ROADMAP.md',
                   metavar='PATH', help="roadmap to update (default: docs/ROADMAP.md of the repo)")
    return p.parse_args(argv)


def main(argv=None):
    """Command-line entry point; returns the exit code (see the module docstring)."""
    args = parse_args(argv)
    try:
        if not args.results.is_dir():
            raise ReportError(f"results directory not found: {args.results}")
        paths = [args.readme, args.roadmap]
        old = [read_document(path) for path in paths]
        new = list(render(Results(args.results), old[0], old[1], args.pending))
        updates = list(zip(paths, old, new))
        if not args.check:
            for path, before, text in updates:
                if text == before:
                    print(f"unchanged {path}")
                    continue
                try:
                    path.write_text(text, encoding='utf-8')
                except OSError as exc:
                    raise ReportError(f"cannot write {path}: {exc.strerror or exc}") from exc
                print(f"updated {path}")
            return 0
    except ReportError as exc:
        print(f"report_results: {exc}", file=sys.stderr)
        return 2
    stale = [path for path, before, text in updates if text != before]
    for path, before, text in updates:
        if text != before:
            print(unified_diff(path, before, text))
    if stale:
        print("report_results: " + ', '.join(str(path) for path in stale)
              + " out of date; run `python src/graph/report_results.py` to rewrite",
              file=sys.stderr)
        return 1
    print("up to date: " + ', '.join(str(path) for path in paths))
    return 0


if __name__ == '__main__':
    for _stream in (sys.stdout, sys.stderr):
        # The diff holds characters such as Δ and −; do not crash on a legacy console encoding.
        if hasattr(_stream, 'reconfigure'):
            _stream.reconfigure(errors='replace')
    try:
        code = main()
        sys.stdout.flush()
    except BrokenPipeError:
        # The reader closed early (`--check | head`). Point stdout at devnull so that
        # Python does not report a second error while shutting down.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = 1
    sys.exit(code)
