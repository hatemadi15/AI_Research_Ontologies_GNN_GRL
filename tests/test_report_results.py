"""Tests for report_results.py, the generator of the README/ROADMAP results sections.

Standard library and pytest only. Synthetic tests build a small results directory and
two stub documents in tmp_path; the regression test at the end regenerates the real
blocks from the real results/ (into tmp_path) and compares them with the current docs.
"""
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import report_results as rr

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / 'src' / 'graph' / 'report_results.py'

README_STUB = "# Project\n\nintro\n\n<!-- RESULTS:START -->\nstale\n<!-- RESULTS:END -->\n\nafter\n"
ROADMAP_STUB = "# Roadmap\n\n<!-- STANDING:START -->\nstale\n<!-- STANDING:END -->\n\nafter\n"

# The smallest pipeline output the generator needs (the rest of results/ is optional).
EVAL_OFF = {
    'reference_majority_class': {'accuracy': 0.083},
    'typing': {'accuracy': 0.384, 'accuracy_ci95': [0.33, 0.43], 'set_f1': 0.351,
               'macro_f1': 0.457, 'acc@5': 0.502, 'h_f1': 0.522},
    'clustering': {'ari': 0.035, 'bcubed_f1': 0.183, 'baseline_singletons': {'bcubed_f1': 0.475}},
    'taxonomy': {'f1': 0.006, 'precision_closure': 0.027, 'random_precision_closure': 0.013},
}

FULL = {'random_split_vary': 5, 'leave_one_paper_out': 4, 'random_split_fixed': 5}
OMIT = object()   # reference=OMIT: write an entry without a `reference` field (old files)

BASELINE_HEADING = '#### NER training variants vs the baseline (paired by split and model seed)'
WEEK2_HEADING = '#### NER week-2 variants vs the CRF head (paired by split and model seed)'
WEEK3_HEADING = '#### NER week-3 comparisons (3-seed means)'
COLUMNS = ("| Variant | Protocol | Granularity | Pairs | Variant micro F1 [95% CI] | "
           "Δ micro [95% CI] | Δ macro [95% CI] | Δ strict micro |")
WEEK3_COLUMNS = ("| Comparison | Protocol | Granularity | Pairs | Variant micro F1 [95% CI] | "
                 "Δ micro [95% CI] | Δ macro [95% CI] | Δ strict micro |")


def stat(mean, lo, hi):
    return {'mean_diff': mean, 'ci95': [lo, hi], 'per_run': []}


class Workspace:
    """A results directory plus stub README/ROADMAP files in tmp_path."""

    def __init__(self, root):
        self.results = root / 'results'
        self.results.mkdir()
        (self.results / 'eval_pipeline_llm_off.json').write_text(json.dumps(EVAL_OFF))
        self.readme = root / 'README.md'
        self.roadmap = root / 'ROADMAP.md'
        self.readme.write_text(README_STUB, encoding='utf-8')
        self.roadmap.write_text(ROADMAP_STUB, encoding='utf-8')
        self.comps = {}
        self.week3_comps = {}

    def compare(self, variant, granularity, protocol, *, n_pairs=None, micro=(0.5, -0.5, 1.5),
                strict=(0.4, -0.6, 1.4), macro=(0.3, -0.7, 1.3), reference='crf', own=None,
                key=None):
        """Add one ner_comparisons.json entry; `own` = (mean, lo, hi) writes the variant's file."""
        key = key or f'ner_results_{granularity}_{protocol}_{variant}'
        n = FULL[protocol] if n_pairs is None else n_pairs
        entry = {'variant': variant, 'granularity': granularity, 'protocol': protocol,
                 'n_pairs': n, 'run_ids': [f'r{i}' for i in range(n)],
                 'micro': stat(*micro), 'micro_strict': stat(*strict),
                 'macro': stat(*macro), 'weighted': stat(*micro)}
        if reference is not OMIT:
            entry['reference'] = reference
        self.comps[key] = entry
        if own:
            mean, lo, hi = own
            (self.results / f'{key}.json').write_text(json.dumps({
                'paper_test_f1': 69.92,
                'summary': {'n_runs': n, 'micro': {'mean': mean, 'ci95': [lo, hi]}}}))
        self.save()
        return key

    def save(self):
        (self.results / 'ner_comparisons.json').write_text(json.dumps(self.comps))

    def week3(self, variant, reference, granularity, protocol, *, n_pairs=None, role='reported',
              micro=(0.5, -0.5, 1.5), strict=(0.4, -0.6, 1.4), macro=(0.3, -0.7, 1.3),
              score=(80.0, 78.0, 82.0)):
        """Add one ner_week3_comparisons.json entry; `score` = the variant's (mean, lo, hi)."""
        n = FULL[protocol] if n_pairs is None else n_pairs
        key = f'ner_results_{granularity}_{protocol}_{variant}'
        key += '' if reference == 'crf' else f'_vs_{reference}'
        self.week3_comps[key] = {
            'variant': variant, 'reference': reference, 'granularity': granularity,
            'protocol': protocol, 'role': role, 'n_pairs': n,
            'run_ids': [f'r{i}' for i in range(n)], 'complete': n >= FULL[protocol],
            'variant_scores': {'micro': {'mean': score[0], 'ci95': list(score[1:]), 'per_run': []}},
            'micro': stat(*micro), 'micro_strict': stat(*strict), 'macro': stat(*macro),
            'weighted': stat(*micro)}
        (self.results / rr.WEEK3_FILE).write_text(json.dumps(self.week3_comps))
        return key

    def week4(self, variant, reference, granularity, protocol, *, gpu=False, role='reported',
              micro=(0.5, -0.5, 1.5), score=(80.0, 78.0, 82.0), name=rr.WEEK4_FILE):
        """Add one week-4 comparison: results/ner_week4_comparisons.json (CPU) or
        results/gpu/ner_week4_comparisons.json (gpu=True); with name=rr.WEEK5_FILE and
        gpu=True, a week-5 comparison."""
        n = FULL[protocol]
        path = self.results / ('gpu' if gpu else '') / name
        path.parent.mkdir(exist_ok=True)
        comps = json.loads(path.read_text()) if path.exists() else {}
        key = f'ner_results_{granularity}_{protocol}_{variant}'
        key += '' if reference == 'crf' else f'_vs_{reference}'
        comps[key] = {
            'variant': variant, 'reference': reference, 'granularity': granularity,
            'protocol': protocol, 'role': role, 'n_pairs': n,
            'run_ids': [f'r{i}' for i in range(n)], 'complete': True,
            'variant_scores': {'micro': {'mean': score[0], 'ci95': list(score[1:]), 'per_run': []}},
            'micro': stat(*micro), 'micro_strict': stat(*micro), 'macro': stat(*micro),
            'weighted': stat(*micro)}
        path.write_text(json.dumps(comps))
        return key

    def summary(self, granularity, protocol, variant, n_runs, micro, paper=69.92, **averages):
        """Write a variant's results file with only its summary: micro = (mean, lo, hi), and
        the same for any other average given by name (macro=..., weighted=...)."""
        summary = {'n_runs': n_runs}
        for name, (mean, lo, hi) in {'micro': micro, **averages}.items():
            summary[name] = {'mean': mean, 'ci95': [lo, hi]}
        (self.results / rr.ner_file(granularity, protocol, variant)).write_text(json.dumps({
            'paper_test_f1': paper, 'summary': summary}))

    def run(self, *extra):
        return rr.main(['--results', str(self.results), '--readme', str(self.readme),
                        '--roadmap', str(self.roadmap), *extra])

    def generate(self, *extra):
        """Run the generator and return (README text, ROADMAP text)."""
        assert self.run(*extra) == 0
        return self.readme.read_text(encoding='utf-8'), self.roadmap.read_text(encoding='utf-8')


def section(text, heading):
    """Lines of the section whose '#### ' heading starts with `heading`."""
    lines = text.split('\n')
    start = next(i for i, line in enumerate(lines) if line.startswith(heading))
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith('#### ') or lines[i].startswith('**Reading the numbers**')),
               len(lines))
    return lines[start:end]


def table_rows(text, heading):
    return [line for line in section(text, heading)
            if line.startswith('| ') and not line.startswith(('| Variant |', '| Comparison |'))]


def headings(text):
    return [line for line in text.split('\n') if line.startswith('#### ')]


def bullets(text, start):
    return [line for line in text.split('\n') if line.startswith(start)]


def block(text, name):
    begin, end = f'<!-- {name}:START -->', f'<!-- {name}:END -->'
    return text[text.index(begin):text.index(end) + len(end)]


def glance(readme):
    """The "At a glance" table of a generated README."""
    return readme[readme.index('**At a glance**'):readme.index('#### ')]


def added_lines(before, after):
    """The lines that `after` adds to `before`; nothing else may differ."""
    old = before.split('\n')
    new = [line for line in after.split('\n') if line not in old]
    assert [line for line in after.split('\n') if line not in new] == old
    return new


# ---------------------------------------------------------------------------
# Table rows and verdict wording
# ---------------------------------------------------------------------------
def test_week2_table_follows_the_baseline_table_with_the_same_columns(tmp_path):
    ws = Workspace(tmp_path)
    ws.compare('cw', 'fine', 'random_split_vary', reference='baseline', micro=(-0.24, -3.09, 2.61),
               strict=(-0.64, -3.4, 2.1), macro=(0.02, -4.99, 5.02), own=(73.67, 69.77, 77.56))
    # written out of order on purpose: the table sorts by variant, protocol, granularity
    ws.compare('crf_ens3', 'coarse', 'leave_one_paper_out', micro=(2.4, 0.7, 4.1),
               strict=(1.9, 0.3, 3.5), macro=(2.2, 0.5, 3.9), own=(62.22, 55.0, 69.4))
    ws.compare('crf_dapt', 'fine', 'leave_one_paper_out', micro=(0.4, -1.1, 1.9),
               strict=(0.35, -1.0, 1.7), macro=(0.2, -2.0, 2.4), own=(47.5, 41.0, 54.0))
    ws.compare('crf_mt', 'coarse', 'random_split_vary', micro=(0.3, -0.8, 1.4),
               strict=(0.25, -0.6, 1.1), macro=(0.6, -1.2, 2.4))        # no own results file
    ws.compare('crf_mt', 'fine', 'random_split_vary', micro=(1.2, 0.35, 2.05),
               strict=(0.9, 0.1, 1.7), macro=(1.5, -0.4, 3.4), own=(77.56, 72.4, 82.7))
    readme, _ = ws.generate()

    assert headings(readme)[-2:] == [BASELINE_HEADING, WEEK2_HEADING]
    assert table_rows(readme, BASELINE_HEADING) == [
        "| class-weighted loss | new random split per seed | fine | 5 | 73.67 [69.77, 77.56] "
        "| -0.24 [-3.09, +2.61] | +0.02 [-4.99, +5.02] | -0.64 |"]
    assert table_rows(readme, WEEK2_HEADING) == [
        "| multi-task (fine + coarse heads) | new random split per seed | fine | 5 "
        "| 77.56 [72.40, 82.70] | +1.20 [+0.35, +2.05] | +1.50 [-0.40, +3.40] | +0.90 |",
        "| multi-task (fine + coarse heads) | new random split per seed | coarse | 5 "
        "| – | +0.30 [-0.80, +1.40] | +0.60 [-1.20, +2.40] | +0.25 |",
        "| domain-adaptive pre-training | leave one paper out | fine | 4 "
        "| 47.50 [41.00, 54.00] | +0.40 [-1.10, +1.90] | +0.20 [-2.00, +2.40] | +0.35 |",
        "| 3-model ensemble | leave one paper out | coarse | 4 "
        "| 62.22 [55.00, 69.40] | +2.40 [+0.70, +4.10] | +2.20 [+0.50, +3.90] | +1.90 |"]
    # same columns as the existing table
    for heading in (BASELINE_HEADING, WEEK2_HEADING):
        # heading, blank, intro, blank, then the column header and its separator
        assert section(readme, heading)[4:6] == [COLUMNS, '|---|---|---|---|---|---|---|---|']
    intro = section(readme, WEEK2_HEADING)[2]
    assert intro.startswith("Δ = variant − CRF head on the same split and model seed, with the "
                            "95% Student-t CI of the mean difference.")
    assert "a variant improves on the CRF head only if the micro-F1 interval over the 5 random " \
           "splits lies entirely above 0." in intro


def test_week2_intro_notes_the_thread_count_and_its_measured_effect(tmp_path):
    ws = Workspace(tmp_path)
    ws.compare('crf_mt', 'fine', 'random_split_vary', own=(77.56, 72.4, 82.7))
    ws.compare('crf_m0', 'fine', 'random_split_vary', n_pairs=3, micro=(-1.1, -2.9, 0.7))
    readme, _ = ws.generate()
    assert section(readme, WEEK2_HEADING)[2].endswith(f"lies entirely above 0. {rr.THREAD_NOTE}")

    # complete crf_m0 comparisons add the measured effect, but never a row
    ws.compare('crf_m0', 'fine', 'random_split_vary', micro=(-1.1, -2.9, 0.7))
    ws.compare('crf_m0', 'coarse', 'random_split_vary', micro=(0.4, -1.0, 1.8))
    readme, _ = ws.generate()
    assert section(readme, WEEK2_HEADING)[2].endswith(
        f"{rr.THREAD_NOTE} Re-running the CRF head with 2 threads (ensemble member `crf_m0`) "
        "changes micro F1 by fine-grained -1.10 [-2.90, +0.70] and coarse-grained "
        "+0.40 [-1.00, +1.80] over the 5 random splits.")
    assert len(table_rows(readme, WEEK2_HEADING)) == 1


def test_week2_verdict_lines(tmp_path):
    ws = Workspace(tmp_path)
    # multi-task: improves / no significant change over the splits, hurts leaving a paper out
    ws.compare('crf_mt', 'fine', 'random_split_vary', micro=(1.2, 0.35, 2.05),
               strict=(0.9, 0.1, 1.7))
    ws.compare('crf_mt', 'coarse', 'random_split_vary', micro=(0.3, -0.8, 1.4),
               strict=(0.25, -0.6, 1.1))
    ws.compare('crf_mt', 'fine', 'leave_one_paper_out', micro=(-2.1, -3.5, -0.7),
               strict=(-1.8, -3.1, -0.5))
    # domain-adaptive pre-training: only the out-of-distribution comparison is complete
    ws.compare('crf_dapt', 'coarse', 'leave_one_paper_out', micro=(2.0, 0.5, 3.5),
               strict=(1.5, -0.2, 3.2))
    readme, roadmap = ws.generate()

    assert bullets(readme, '* NER, multi-task') == [
        "* NER, multi-task (fine + coarse heads) vs the CRF head (paired micro-F1 difference): "
        "fine-grained improves (+1.20, 95% CI [+0.35, +2.05]; strict IOB2 +0.90 [+0.10, +1.70]); "
        "coarse-grained no significant change (+0.30, 95% CI [-0.80, +1.40]; strict IOB2 "
        "+0.25 [-0.60, +1.10]) across the 5 random splits. Leaving one paper out: fine-grained "
        "hurts (-2.10, 95% CI [-3.50, -0.70]; strict IOB2 -1.80 [-3.10, -0.50])."]
    assert bullets(readme, '* NER, domain-adaptive') == [
        "* NER, domain-adaptive pre-training vs the CRF head (paired micro-F1 difference): "
        "leaving one paper out: coarse-grained improves (+2.00, 95% CI [+0.50, +3.50]; strict "
        "IOB2 +1.50 [-0.20, +3.20])."]
    assert bullets(readme, '* NER, 3-model') == []          # no complete comparison, no line
    assert 'week-2' not in roadmap


def test_ensemble_vs_its_members_bullet(tmp_path):
    ws = Workspace(tmp_path)
    ws.compare('crf_ens3', 'fine', 'random_split_vary', micro=(3.39, -1.36, 8.14))
    start = '* NER, 3-model ensemble vs the mean'
    readme, _ = ws.generate()
    assert bullets(readme, start) == []                     # no ensemble results file yet

    def ens_file(granularity, protocol, n_runs, gain):
        (ws.results / rr.ner_file(granularity, protocol, 'crf_ens3')).write_text(json.dumps({
            'paper_test_f1': 69.92,
            'summary': {'n_runs': n_runs, 'micro': {'mean': 79.76, 'ci95': [76.14, 83.38]},
                        'vs_member_mean': stat(*gain)}}))
    ens_file('fine', 'random_split_vary', 5, (2.47, 1.19, 3.75))
    ens_file('coarse', 'random_split_vary', 3, (1.0, -1.0, 3.0))    # incomplete: left out
    ens_file('fine', 'leave_one_paper_out', 4, (1.5, -0.5, 3.5))
    readme, roadmap = ws.generate()
    assert bullets(readme, start) == [
        "* NER, 3-model ensemble vs the mean of its own members on the same split (exploratory, "
        "not pre-registered; no single reference run enters): fine-grained +2.47 [+1.19, +3.75] "
        "across the 5 random splits; fine-grained +1.50 [-0.50, +3.50] leaving one paper out."]
    assert start not in roadmap


@pytest.mark.parametrize('micro, word', [
    ((1.0, 0.01, 2.0), 'improves'),
    ((1.0, 0.0, 2.0), 'no significant change'),      # the interval must lie entirely above 0
    ((-1.0, -2.0, -0.01), 'hurts'),
    ((-1.0, -2.0, 0.0), 'no significant change'),
    ((0.0, -1.0, 1.0), 'no significant change'),
])
def test_paired_verdict_words(micro, word):
    got, numbers = rr.paired_verdict({'micro': stat(*micro), 'micro_strict': stat(0.5, -0.5, 1.5)})
    assert got == word
    assert numbers.endswith("; strict IOB2 +0.50 [-0.50, +1.50]")


def test_baseline_variants_keep_their_wording(tmp_path):
    ws = Workspace(tmp_path)
    ws.compare('crf', 'fine', 'random_split_vary', reference='baseline', micro=(2.46, 0.39, 4.52),
               strict=(0.1, -2.08, 2.29), own=(76.36, 71.0, 81.73))
    ws.compare('cw', 'fine', 'leave_one_paper_out', reference=OMIT, micro=(1.81, -4.28, 7.89),
               strict=(1.75, -3.92, 7.43))
    readme, _ = ws.generate()
    assert bullets(readme, '* NER, CRF head') == [
        "* NER, CRF head vs the baseline (paired micro-F1 difference): fine-grained improves "
        "(+2.46, 95% CI [+0.39, +4.52]; strict IOB2 +0.10 [-2.08, +2.29]) across the 5 random "
        "splits. Pre-registered SOTA check (a paired gain over the baseline and a 95% CI above "
        "the published number): met for fine-grained 76.36 [71.00, 81.73] vs 69.92, if the "
        "paper reports micro F1."]
    assert bullets(readme, '* NER, class-weighted') == [
        "* NER, class-weighted loss vs the baseline (paired micro-F1 difference): leaving one "
        "paper out: fine-grained no significant change (+1.81, 95% CI [-4.28, +7.89]; strict "
        "IOB2 +1.75 [-3.92, +7.43])."]


# ---------------------------------------------------------------------------
# Complete comparisons only, and never the ensemble members
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('protocol, pairs, shown', [
    ('random_split_vary', 4, False), ('random_split_vary', 5, True),
    ('leave_one_paper_out', 3, False), ('leave_one_paper_out', 4, True),
    ('random_split_fixed', 4, False), ('random_split_fixed', 5, True),
])
@pytest.mark.parametrize('variant, reference, heading', [
    ('crf_mt', 'crf', WEEK2_HEADING),
    ('cw', 'baseline', BASELINE_HEADING),          # the rule is the same for the old table
])
def test_only_complete_comparisons_are_shown(tmp_path, variant, reference, heading, protocol,
                                             pairs, shown):
    ws = Workspace(tmp_path)
    ws.compare(variant, 'fine', protocol, n_pairs=pairs, reference=reference)
    readme, _ = ws.generate()
    assert (heading in readme) is shown
    if shown:
        assert len(table_rows(readme, heading)) == 1
    if protocol != 'random_split_fixed':           # verdict lines cover the two main protocols
        assert bool(bullets(readme, '* NER, ')) is shown


def test_incomplete_and_member_entries_leave_the_output_unchanged(tmp_path):
    """Until a complete week-2 comparison exists, the generated text does not change."""
    ws = Workspace(tmp_path)
    ws.compare('cw', 'fine', 'random_split_vary', reference='baseline')
    before = ws.generate()

    ws.compare('crf_mt', 'fine', 'random_split_vary', n_pairs=3)
    ws.compare('crf_dapt', 'coarse', 'leave_one_paper_out', n_pairs=2)
    ws.compare('crf_ens3', 'fine', 'random_split_vary', n_pairs=4)
    for member in ('crf_m0', 'crf_m100', 'crf_m200'):
        for granularity in rr.GRANULARITIES:
            for protocol in ('random_split_vary', 'leave_one_paper_out'):
                ws.compare(member, granularity, protocol, own=(70.0, 65.0, 75.0))
    assert ws.generate() == before


def test_ensemble_members_get_no_row_or_verdict(tmp_path):
    ws = Workspace(tmp_path)
    ws.compare('crf_ens3', 'fine', 'random_split_vary', micro=(2.0, 0.6, 3.4),
               own=(78.36, 73.0, 83.7))
    for member in ('crf_m0', 'crf_m100', 'crf_m200'):
        for protocol in ('random_split_vary', 'leave_one_paper_out'):
            ws.compare(member, 'fine', protocol, micro=(5.0, 4.0, 6.0), own=(88.88, 80.0, 90.0))
            # even filed under the baseline (as a bare tag would be), they stay out
            ws.compare(member, 'fine', protocol, reference='baseline', micro=(5.0, 4.0, 6.0),
                       key=f'baseline_{member}_{protocol}')
    readme, roadmap = ws.generate()
    assert [row.split(' | ')[0] for row in table_rows(readme, WEEK2_HEADING)] == [
        '| 3-model ensemble']
    assert BASELINE_HEADING not in readme
    # the one place a member appears: crf_m0 vs the CRF head as the thread-count effect
    note = (f"{rr.THREAD_NOTE} Re-running the CRF head with 2 threads (ensemble member "
            "`crf_m0`) changes micro F1 by fine-grained +5.00 [+4.00, +6.00] over the 5 random "
            "splits.")
    assert readme.count(note) == 1
    for text in (readme.replace(note, ''), roadmap):
        assert 'crf_m' not in text and '88.88' not in text and '+5.00' not in text
    assert len(bullets(readme, '* NER, ')) == 1


def test_entries_with_missing_or_unknown_fields_are_skipped(tmp_path):
    ws = Workspace(tmp_path)
    ws.compare('cw', 'fine', 'random_split_vary', reference='baseline')
    before = ws.generate()
    ws.comps.update({
        'null_entry': None,
        'list_entry': [1, 2],
        'empty': {},
        'only_variant': {'variant': 'crf_mt'},
        'no_pairs': {'variant': 'crf_mt', 'granularity': 'fine', 'protocol': 'random_split_vary',
                     'reference': 'crf'},
        'null_pairs': {'variant': 'crf_mt', 'granularity': 'fine', 'n_pairs': None,
                       'protocol': 'random_split_vary', 'reference': 'crf'},
        'new_protocol': {'variant': 'crf_mt', 'granularity': 'fine', 'protocol': 'cross_domain',
                         'reference': 'crf', 'n_pairs': 99},
        'new_granularity': {'variant': 'crf_mt', 'granularity': 'medium', 'reference': 'crf',
                            'protocol': 'random_split_vary', 'n_pairs': 5},
        'new_variant': {'variant': 'crf_foo', 'granularity': 'fine', 'reference': 'crf',
                        'protocol': 'random_split_vary', 'n_pairs': 5},
    })
    ws.save()
    assert ws.generate() == before


# ---------------------------------------------------------------------------
# The `reference` field
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('reference', [OMIT, None, 'baseline'])
def test_no_reference_means_baseline(tmp_path, reference):
    ws = Workspace(tmp_path)
    ws.compare('cw', 'fine', 'random_split_vary', reference=reference)
    ws.compare('crf', 'coarse', 'leave_one_paper_out', reference=reference, micro=(5.0, 1.0, 9.0))
    readme, _ = ws.generate()
    assert [r.split(' | ')[0] for r in table_rows(readme, BASELINE_HEADING)] == [
        '| class-weighted loss', '| CRF head']
    assert WEEK2_HEADING not in readme
    assert len(bullets(readme, '* NER, class-weighted loss vs the baseline')) == 1
    assert len(bullets(readme, '* NER, CRF head vs the baseline')) == 1
    # an older entry (no field) and an explicit 'baseline' generate the same text
    (tmp_path / 'explicit').mkdir()
    other = Workspace(tmp_path / 'explicit')
    other.compare('cw', 'fine', 'random_split_vary', reference='baseline')
    other.compare('crf', 'coarse', 'leave_one_paper_out', reference='baseline',
                  micro=(5.0, 1.0, 9.0))
    assert other.generate()[0] == readme


def test_crf_reference_goes_to_the_week2_table_only(tmp_path):
    ws = Workspace(tmp_path)
    ws.compare('crf', 'fine', 'random_split_vary', reference='baseline')
    ws.compare('crf_mt', 'fine', 'random_split_vary', reference='crf')
    readme, roadmap = ws.generate()
    assert [r.split(' | ')[0] for r in table_rows(readme, BASELINE_HEADING)] == ['| CRF head']
    assert [r.split(' | ')[0] for r in table_rows(readme, WEEK2_HEADING)] == [
        '| multi-task (fine + coarse heads)']
    assert len(bullets(readme, '* NER, CRF head vs the baseline')) == 1
    assert len(bullets(readme, '* NER, multi-task (fine + coarse heads) vs the CRF head')) == 1


@pytest.mark.parametrize('variant, reference', [
    ('crf_mt', 'baseline'), ('crf_mt', OMIT), ('crf_mt', None),   # week-2 variant, wrong reference
    ('cw', 'crf'), ('crf', 'crf'),                                 # week-1 variant, wrong reference
    ('crf_mt', 'cw'), ('cw', 'cw'),                                # some other reference
])
def test_wrong_reference_is_shown_nowhere(tmp_path, variant, reference):
    ws = Workspace(tmp_path)
    ws.compare('cw', 'coarse', 'random_split_vary', reference='baseline')
    before = ws.generate()
    ws.compare(variant, 'fine', 'random_split_vary', reference=reference, key='extra_entry')
    assert ws.generate() == before


def test_standing_row_for_the_crf_head(tmp_path):
    ws = Workspace(tmp_path)

    def write(name, summary_micro=None, n_runs=5):
        data = {'runs': []} if summary_micro is None else {
            'paper_test_f1': 69.92,
            'summary': {'n_runs': n_runs, 'micro': dict(zip(('mean', 'ci95'), summary_micro))}}
        (ws.results / name).write_text(json.dumps(data))

    row = "| NER with the CRF head (`--crf`) | "
    tail = (" | same; the in-distribution fine-grained gain over the baseline disappears under "
            "strict IOB2 scoring |")

    def crf_rows():
        return [line for line in ws.generate()[1].split('\n') if line.startswith(row)]

    assert crf_rows() == []                          # no CRF file: no row
    write('ner_results_fine_random_split_vary_crf.json', (76.36, [71.0, 81.73]))
    write('ner_results_coarse_random_split_vary_crf.json', (78.15, [76.61, 79.69]), n_runs=4)
    write('ner_results_fine_leave_one_paper_out_crf.json')            # no summary yet
    assert crf_rows() == [row + "fine: micro 76.36 [71.00, 81.73] over 5 new splits" + tail]
    write('ner_results_fine_leave_one_paper_out_crf.json', (49.44, [41.0, 57.89]), n_runs=4)
    write('ner_results_coarse_random_split_vary_crf.json', (78.15, [76.61, 79.69]))
    assert crf_rows() == [
        row + "fine: micro 76.36 [71.00, 81.73] over 5 new splits, 49.44 leaving one paper out; "
        "coarse: micro 78.15 [76.61, 79.69] over 5 new splits" + tail]


def test_week2_comparisons_do_not_reach_the_glance_or_standing_tables(tmp_path):
    """Only the reported system does (week-3 protocol): the crf_ens3 results files."""
    ws = Workspace(tmp_path)
    ws.compare('crf', 'fine', 'random_split_vary', reference='baseline', own=(76.36, 71.0, 81.73),
               micro=(2.46, 0.39, 4.52))
    readme_before, roadmap_before = ws.generate()
    glance_before = glance(readme_before)
    assert 'paired gain over the baseline' in glance_before

    for variant in rr.WEEK2_VARIANTS:
        for granularity in rr.GRANULARITIES:
            ws.compare(variant, granularity, 'random_split_vary', micro=(3.0, 1.0, 5.0),
                       own=(99.0, 98.0, 99.5))
            ws.compare(variant, granularity, 'leave_one_paper_out', micro=(3.0, 1.0, 5.0))
    readme, roadmap = ws.generate()
    assert WEEK2_HEADING in readme and len(table_rows(readme, WEEK2_HEADING)) == 12
    # the crf_ens3 files written above (5 runs) are the reported system's; nothing else changes
    assert added_lines(glance_before, glance(readme)) == [
        f"| NER, {g}-grained, reported system (week-3 protocol): 3-model seed ensemble `crf_ens3` "
        "| micro F1 99.00 [98.00, 99.50] over 5 random splits | published 69.92 (none for an "
        "unseen paper): **beaten** under micro averaging |" for g in rr.GRANULARITIES]
    assert added_lines(roadmap_before, roadmap) == [
        "| NER, reported system (week-3 protocol) | fine: 3-model seed ensemble `crf_ens3`, micro "
        "99.00 [98.00, 99.50] over 5 new splits; coarse: 3-model seed ensemble `crf_ens3`, micro "
        "99.00 [98.00, 99.50] over 5 new splits | same; fine-grained beaten under micro "
        "averaging, coarse-grained beaten under micro averaging |"]


# ---------------------------------------------------------------------------
# Week 3: comparisons of 3-seed means and the reported system
# ---------------------------------------------------------------------------
def test_week3_table_lists_complete_comparisons_in_the_protocol_order(tmp_path):
    ws = Workspace(tmp_path)
    ws.compare('crf_mt', 'fine', 'random_split_vary')                # the week-2 table comes first
    # written out of order on purpose: the table follows the protocol, then protocol, granularity
    ws.week3('crf_dapt2', 'crf', 'coarse', 'random_split_vary', role='decision',
             micro=(1.0, -0.5, 2.5), strict=(0.9, -0.6, 2.4), macro=(0.5, -1.0, 2.0),
             score=(78.5, 76.0, 81.0))
    ws.week3('crf_ens3', 'crf', 'fine', 'leave_one_paper_out', micro=(2.04, -0.23, 4.32),
             strict=(2.04, -0.23, 4.32), macro=(1.82, -2.54, 6.18), score=(49.55, 40.11, 58.99))
    ws.week3('crf_ens3', 'crf', 'coarse', 'random_split_vary', micro=(1.8, 0.91, 2.69))
    ws.week3('crf_ens3', 'crf', 'fine', 'random_split_vary', micro=(2.47, 1.17, 3.77))
    ws.week3('crf_ls', 'crf', 'fine', 'leave_one_paper_out', role='decision')
    ws.week3('crf_mix3', 'crf_ens3', 'fine', 'random_split_vary', role='decision',
             micro=(-0.2, -1.1, 0.7), score=(79.56, 76.0, 83.1))
    ws.week3('crf_mt', 'crf', 'fine', 'random_split_vary', role='decision', n_pairs=3)  # no row
    readme, roadmap = ws.generate()

    assert headings(readme)[-2:] == [WEEK2_HEADING, WEEK3_HEADING]
    assert table_rows(readme, WEEK3_HEADING) == [
        "| mixed-backbone ensemble vs 3-model seed ensemble | new random split per seed | fine | 5 "
        "| 79.56 [76.00, 83.10] | -0.20 [-1.10, +0.70] | +0.30 [-0.70, +1.30] | +0.40 |",
        "| 3-model seed ensemble vs CRF head | new random split per seed | fine | 5 "
        "| 80.00 [78.00, 82.00] | +2.47 [+1.17, +3.77] | +0.30 [-0.70, +1.30] | +0.40 |",
        "| 3-model seed ensemble vs CRF head | new random split per seed | coarse | 5 "
        "| 80.00 [78.00, 82.00] | +1.80 [+0.91, +2.69] | +0.30 [-0.70, +1.30] | +0.40 |",
        "| 3-model seed ensemble vs CRF head | leave one paper out | fine | 4 "
        "| 49.55 [40.11, 58.99] | +2.04 [-0.23, +4.32] | +1.82 [-2.54, +6.18] | +2.04 |",
        "| label semantics vs CRF head | leave one paper out | fine | 4 "
        "| 80.00 [78.00, 82.00] | +0.50 [-0.50, +1.50] | +0.30 [-0.70, +1.30] | +0.40 |",
        "| in-domain pre-training vs CRF head | new random split per seed | coarse | 5 "
        "| 78.50 [76.00, 81.00] | +1.00 [-0.50, +2.50] | +0.50 [-1.00, +2.00] | +0.90 |"]
    lines = section(readme, WEEK3_HEADING)
    assert lines[2] == rr.WEEK3_INTRO.strip() and lines[4:6] == [
        WEEK3_COLUMNS, '|---|---|---|---|---|---|---|---|']
    for part in ("a model trained with 3 model seeds", "scores their mean",
                 "the ensembles and the one-seed SciBERT and PubMedBERT runs score once",
                 "the CRF head is the mean of `crf_m0`, `crf_m100` and `crf_m200`",
                 "over the 5 random splits for the mixed-backbone ensemble",
                 "over the 4 held-out papers for label semantics and self-training"):
        assert part in lines[2]
    assert WEEK3_HEADING not in roadmap


def test_week3_verdict_lines_mark_the_decision(tmp_path):
    ws = Workspace(tmp_path)
    ws.week3('crf_mt', 'crf', 'fine', 'random_split_vary', role='decision',
             micro=(1.2, 0.35, 2.05), strict=(0.9, 0.1, 1.7))
    ws.week3('crf_mt', 'crf', 'coarse', 'random_split_vary', role='decision',
             micro=(0.3, -0.8, 1.4), strict=(0.25, -0.6, 1.1))
    ws.week3('crf_mt', 'crf', 'fine', 'leave_one_paper_out', micro=(-2.1, -3.5, -0.7),
             strict=(-1.8, -3.1, -0.5))
    ws.week3('crf_ls', 'crf', 'coarse', 'leave_one_paper_out', role='decision',
             micro=(2.0, 0.5, 3.5), strict=(1.5, -0.2, 3.2))
    ws.week3('crf_mix3', 'crf_ens3', 'fine', 'random_split_vary', role='decision',
             micro=(0.4, -0.6, 1.4), strict=(0.4, -0.6, 1.4))
    readme, roadmap = ws.generate()
    # in the protocol's order, whatever the order of the file
    assert bullets(readme, '* NER week 3, ') == [
        "* NER week 3, mixed-backbone ensemble vs the 3-model seed ensemble (3-seed means, paired "
        "micro-F1 difference): fine-grained no significant change (+0.40, 95% CI [-0.60, +1.40]; "
        "strict IOB2 +0.40 [-0.60, +1.40]) across the 5 random splits (the pre-registered "
        "decision).",
        "* NER week 3, label semantics vs the CRF head (3-seed means, paired micro-F1 difference): "
        "leaving one paper out (the pre-registered decision): coarse-grained improves (+2.00, 95% "
        "CI [+0.50, +3.50]; strict IOB2 +1.50 [-0.20, +3.20]).",
        "* NER week 3, multi-task (fine + coarse heads) vs the CRF head (3-seed means, paired "
        "micro-F1 difference): fine-grained improves (+1.20, 95% CI [+0.35, +2.05]; strict IOB2 "
        "+0.90 [+0.10, +1.70]); coarse-grained no significant change (+0.30, 95% CI [-0.80, "
        "+1.40]; strict IOB2 +0.25 [-0.60, +1.10]) across the 5 random splits (the pre-registered "
        "decision). Leaving one paper out (reported, no decision): fine-grained hurts (-2.10, 95% "
        "CI [-3.50, -0.70]; strict IOB2 -1.80 [-3.10, -0.50])."]
    assert bullets(readme, '* NER, ') == [] and 'week 3' not in roadmap


def test_week3_entries_that_are_incomplete_or_not_listed_change_nothing(tmp_path):
    ws = Workspace(tmp_path)
    ws.compare('cw', 'fine', 'random_split_vary', reference='baseline')
    ws.summary('fine', 'random_split_vary', 'crf_ens3', 5, (79.76, 76.14, 83.38))
    ws.summary('fine', 'random_split_vary', 'crf_mix3', 5, (88.88, 80.0, 90.0))
    before = ws.generate()
    ws.week3('crf_mt', 'crf', 'fine', 'random_split_vary', n_pairs=4)
    ws.week3('crf_ls', 'crf', 'coarse', 'leave_one_paper_out', n_pairs=3)
    ws.week3('crf_foo', 'crf', 'fine', 'random_split_vary')               # not a week-3 variant
    ws.week3('crf_mt', 'crf_ens3', 'fine', 'random_split_vary')           # not its reference
    ws.week3('crf_dapt', 'crf', 'fine', 'random_split_vary')              # a week-2 variant
    ws.week3('crf_ens3', 'crf', 'fine', 'random_split_fixed')             # not a week-3 protocol
    # an incomplete primary comparison changes neither the table nor the reported system
    ws.week3('crf_mix3', 'crf_ens3', 'fine', 'random_split_vary', n_pairs=4, micro=(3.0, 1.0, 5.0))
    ws.week3_comps.update({'null_entry': None, 'list_entry': [1, 2]})
    (ws.results / rr.WEEK3_FILE).write_text(json.dumps(ws.week3_comps))
    assert ws.generate() == before


def test_reported_system_is_the_seed_ensemble_unless_the_mixed_one_improves_on_it(tmp_path):
    ws = Workspace(tmp_path)

    def reported():
        readme, roadmap = ws.generate()
        return ([line for line in glance(readme).split('\n') if 'reported system' in line],
                [line for line in roadmap.split('\n') if 'reported system' in line])

    assert reported() == ([], [])                                   # no crf_ens3 results yet
    ws.summary('fine', 'random_split_vary', 'crf_ens3', 5, (79.76, 76.14, 83.38))
    ws.summary('fine', 'leave_one_paper_out', 'crf_ens3', 4, (49.55, 40.11, 58.99))
    ws.summary('coarse', 'random_split_vary', 'crf_ens3', 4, (79.45, 77.82, 81.09))  # 4 of 5
    ws.summary('coarse', 'leave_one_paper_out', 'crf_ens3', 4, (61.42, 55.81, 67.03))
    assert reported() == (
        ["| NER, fine-grained, reported system (week-3 protocol): 3-model seed ensemble `crf_ens3` "
         "| micro F1 79.76 [76.14, 83.38] over 5 random splits, 49.55 on an unseen paper "
         "| published 69.92 (none for an unseen paper): **beaten** under micro averaging |"],
        ["| NER, reported system (week-3 protocol) | fine: 3-model seed ensemble `crf_ens3`, micro "
         "79.76 [76.14, 83.38] over 5 new splits, 49.55 leaving one paper out | same; "
         "fine-grained beaten under micro averaging |"])

    ws.summary('coarse', 'random_split_vary', 'crf_ens3', 5, (79.45, 77.82, 81.09), paper=72.32)
    for g, protocol, n, micro in (('fine', 'random_split_vary', 5, (81.2, 78.0, 84.4)),
                                  ('fine', 'leave_one_paper_out', 3, (51.0, 44.0, 58.0)),
                                  ('coarse', 'random_split_vary', 5, (80.1, 78.5, 81.7))):
        ws.summary(g, protocol, 'crf_mix3', n, micro)
    # crf_mix3 replaces crf_ens3 where its random-split CI vs crf_ens3 lies above 0, and only there
    ws.week3('crf_mix3', 'crf_ens3', 'fine', 'random_split_vary', role='decision',
             micro=(1.4, 0.2, 2.6))
    ws.week3('crf_mix3', 'crf_ens3', 'coarse', 'random_split_vary', role='decision',
             micro=(0.6, -0.4, 1.6))
    ws.week3('crf_mix3', 'crf_ens3', 'coarse', 'leave_one_paper_out', micro=(2.0, 1.0, 3.0))
    assert reported() == (
        ["| NER, fine-grained, reported system (week-3 protocol): mixed-backbone ensemble "
         "`crf_mix3` | micro F1 81.20 [78.00, 84.40] over 5 random splits | published 69.92 (none "
         "for an unseen paper): **beaten** under micro averaging |",  # 3 held-out papers: no mean
         "| NER, coarse-grained, reported system (week-3 protocol): 3-model seed ensemble "
         "`crf_ens3` | micro F1 79.45 [77.82, 81.09] over 5 random splits, 61.42 on an unseen "
         "paper | published 72.32 (none for an unseen paper): **beaten** under micro averaging |"],
        ["| NER, reported system (week-3 protocol) | fine: mixed-backbone ensemble `crf_mix3`, "
         "micro 81.20 [78.00, 84.40] over 5 new splits; coarse: 3-model seed ensemble `crf_ens3`, "
         "micro 79.45 [77.82, 81.09] over 5 new splits, 61.42 leaving one paper out | same; "
         "fine-grained beaten under micro averaging, coarse-grained beaten under micro "
         "averaging |"])
    ws.week3('crf_mix3', 'crf_ens3', 'fine', 'random_split_vary', role='decision', n_pairs=4,
             micro=(1.4, 0.2, 2.6))                                  # incomplete: back to crf_ens3
    glance_rows, standing_rows = reported()
    assert 'crf_mix3' not in ''.join(glance_rows + standing_rows)
    assert glance_rows[0].startswith("| NER, fine-grained, reported system (week-3 protocol): "
                                     "3-model seed ensemble `crf_ens3` | micro F1 79.76")


def test_reported_system_is_checked_under_each_averaging(tmp_path):
    """The paper does not state its averaging: micro, macro and weighted F1 are each checked."""
    ws = Workspace(tmp_path)

    def references():
        readme, roadmap = ws.generate()
        rows = [line for line in glance(readme).split('\n') if 'reported system' in line]
        standing = [line for line in roadmap.split('\n') if 'reported system' in line]
        return [row.rsplit(' | ', 1)[1] for row in rows], standing[0].rsplit(' | ', 1)[1]

    ws.summary('fine', 'random_split_vary', 'crf_ens3', 5, (79.76, 76.14, 83.38),
               macro=(77.15, 74.05, 80.25), weighted=(79.23, 76.32, 82.13))
    ws.summary('coarse', 'random_split_vary', 'crf_ens3', 5, (79.45, 77.82, 81.09), paper=72.32,
               macro=(73.03, 67.79, 78.27), weighted=(78.75, 76.78, 80.72))
    assert references() == (
        ["published 69.92 (none for an unseen paper): **beaten** under micro, macro and weighted "
         "averaging |",
         "published 72.32 (none for an unseen paper): **beaten** under micro and weighted "
         "averaging; macro 73.03 [67.79, 78.27] is not significantly above |"],
        "same; fine-grained beaten under micro, macro and weighted averaging, coarse-grained "
        "beaten under micro and weighted averaging |")

    ws.summary('fine', 'random_split_vary', 'crf_ens3', 5, (73.0, 70.5, 75.5),
               macro=(69.0, 65.0, 73.0), weighted=(71.0, 69.0, 73.0))
    ws.summary('coarse', 'random_split_vary', 'crf_ens3', 5, (73.91, 69.36, 78.45), paper=72.32,
               macro=(71.19, 65.0, 77.0))
    assert references() == (
        ["published 69.92 (none for an unseen paper): **beaten** under micro averaging; macro "
         "69.00 [65.00, 73.00], weighted 71.00 [69.00, 73.00] are not significantly above |",
         "published 72.32 (none for an unseen paper): not significantly better |"],
        "same; fine-grained beaten under micro averaging, coarse-grained not significantly "
        "better |")


def test_week4_lists_follow_ner_seedmeans():
    ns = pytest.importorskip('ner_seedmeans')
    assert rr.WEEK4_FILE == ns.OUT_NAMES[4]
    assert list(rr.WEEK4_CPU_COMPARISONS) == [(v, r) for v, r, _ in ns.comparison_list(4)]
    assert list(rr.WEEK4_GPU_COMPARISONS) == [(v, r) for v, r, _ in ns.comparison_list(4, True)]
    assert set(rr.WEEK4_NAMES) >= set(ns.SYSTEMS)
    # a verdict bullet for every decision, and for the GPU repeats of the CPU votes
    decisions = {(v, r) for v, r, d in ns.comparison_list(4) + ns.comparison_list(4, True) if d}
    assert decisions <= {(v, r) for _, v, r, _ in rr.WEEK4_VERDICTS}


def test_week4_unseen_paper_slot_takes_a_vote_that_passes(tmp_path):
    """A CPU vote that improves on crf_ens3 over the held-out papers holds the unseen-paper
    slot; the in-distribution system and its published check do not change."""
    ws = Workspace(tmp_path)
    for g, paper in (('fine', 69.92), ('coarse', 72.32)):
        ws.summary(g, 'random_split_vary', 'crf_ens3', 5, (79.5, 77.8, 81.1), paper=paper)
        ws.summary(g, 'leave_one_paper_out', 'crf_ens3', 4, (61.42, 55.8, 67.0), paper=paper)
        ws.summary(g, 'leave_one_paper_out', 'crf_ls_ens3', 4, (63.05, 57.1, 69.0), paper=paper)
        ws.summary(g, 'leave_one_paper_out', 'crf_st_ens3', 4, (62.0, 57.0, 67.0), paper=paper)
    # fine: neither vote passes; coarse: both pass, the label-semantics vote by more
    ws.week4('crf_ls_ens3', 'crf_ens3', 'fine', 'leave_one_paper_out', role='decision',
             micro=(0.36, -2.53, 3.24))
    ws.week4('crf_ls_ens3', 'crf_ens3', 'coarse', 'leave_one_paper_out', role='decision',
             micro=(1.63, 0.11, 3.16), score=(63.05, 57.1, 69.0))
    ws.week4('crf_st_ens3', 'crf_ens3', 'coarse', 'leave_one_paper_out', role='decision',
             micro=(0.6, 0.1, 1.1))
    readme, roadmap = ws.generate()
    rows = [line for line in glance(readme).split('\n') if 'reported system' in line]
    assert rows[0].startswith('| NER, fine-grained, reported system (week-4 protocol): 3-model '
                              'seed ensemble `crf_ens3` | micro F1 79.50 [77.80, 81.10] over 5 '
                              'random splits, 61.42 on an unseen paper |')
    assert rows[1].startswith('| NER, coarse-grained, reported system (week-4 protocol): 3-model '
                              'seed ensemble `crf_ens3`; on unseen papers the label-semantics vote '
                              '`crf_ls_ens3` | micro F1 79.50 [77.80, 81.10] over 5 random '
                              'splits, 63.05 on an unseen paper (`crf_ens3` 61.42) | published '
                              '72.32')
    standing = next(line for line in roadmap.split('\n') if 'reported system' in line)
    assert ('coarse: 3-model seed ensemble `crf_ens3`, micro 79.50 [77.80, 81.10] over 5 new '
            'splits; on unseen papers the label-semantics vote `crf_ls_ens3`, 63.05 leaving one '
            'paper out (`crf_ens3` 61.42)') in standing
    assert standing.startswith('| NER, reported system (week-4 protocol) |')
    # the week-4 table lists the CPU rows, and each decision gets a verdict bullet
    rows = table_rows(readme, '#### NER week-4 comparisons')
    assert [r.split(' | ')[:4] for r in rows] == [
        ['| label-semantics vote vs 3-model seed ensemble', 'CPU', 'leave one paper out', 'fine'],
        ['| label-semantics vote vs 3-model seed ensemble', 'CPU', 'leave one paper out', 'coarse'],
        ['| self-training vote vs 3-model seed ensemble', 'CPU', 'leave one paper out', 'coarse']]
    assert ('* NER week 4 (CPU runs), label-semantics vote vs the 3-model seed ensemble (paired '
            'micro-F1 difference): leaving one paper out (the pre-registered decision): '
            'fine-grained no significant change') in readme


def test_week4_gpu_multitask_vote_takes_the_in_distribution_slot(tmp_path):
    """The GPU multi-task vote that improves on the GPU crf_ens3 over the random splits is
    reported with its GPU results; descriptive GPU-minus-CPU entries stay out of the table."""
    ws = Workspace(tmp_path)
    (ws.results / 'gpu').mkdir()
    for g, paper in (('fine', 69.92), ('coarse', 72.32)):
        ws.summary(g, 'random_split_vary', 'crf_ens3', 5, (79.5, 77.8, 81.1), paper=paper)
        ws.summary(g, 'leave_one_paper_out', 'crf_ens3', 4, (61.42, 55.8, 67.0), paper=paper)
    (ws.results / 'gpu' / rr.ner_file('fine', 'random_split_vary', 'crf_mt_ens3')).write_text(
        json.dumps({'paper_test_f1': 69.92, 'summary': {'n_runs': 5, 'micro': {
            'mean': 81.0, 'ci95': [79.0, 83.0]}}}))
    ws.week4('crf_mt_ens3', 'crf_ens3', 'fine', 'random_split_vary', gpu=True, role='decision',
             micro=(1.2, 0.4, 2.0), score=(81.0, 79.0, 83.0))
    ws.week4('crf', 'crf', 'fine', 'random_split_vary', gpu=True, role='descriptive',
             micro=(0.9, -0.5, 2.3))
    readme, roadmap = ws.generate()
    rows = [line for line in glance(readme).split('\n') if 'reported system' in line]
    # the unseen-paper slot stays with the CPU crf_ens3: only the CPU votes may take it
    assert rows[0].startswith('| NER, fine-grained, reported system (week-4 protocol): '
                              'multi-task vote `crf_mt_ens3` (GPU runs); on unseen papers the '
                              '3-model seed ensemble `crf_ens3` | micro F1 81.00 [79.00, 83.00] '
                              'over 5 random splits, 61.42 on an unseen paper | published 69.92')
    assert '3-model seed ensemble `crf_ens3`' in rows[1]
    rows = table_rows(readme, '#### NER week-4 comparisons')
    assert [r.split(' | ')[:5] for r in rows] == [
        ['| multi-task vote vs 3-model seed ensemble', 'GPU', 'new random split per seed', 'fine',
         '5']]
    assert ('* NER, GPU minus CPU (descriptive, not a comparison of methods): the CRF head with '
            'the same seeds and splits, 3-seed means, fine-grained +0.90 [-0.50, +2.30] over the 5 '
            'random splits.') in readme


def test_week5_lists_follow_ner_seedmeans():
    ns = pytest.importorskip('ner_seedmeans')
    assert rr.WEEK5_FILE == ns.OUT_NAMES[5]
    assert list(rr.WEEK5_COMPARISONS) == [(v, r) for v, r, _ in ns.comparison_list(5, True)]
    assert set(rr.WEEK5_NAMES) >= set(ns.WEEK5_SYSTEMS)


def test_week5_table_and_verdicts(tmp_path):
    """The week-5 comparisons (10-seed means of GPU runs) get their own table, in the
    protocol's order, and a verdict bullet per comparison."""
    ws = Workspace(tmp_path)
    ws.week4('crf_st3_ens3', 'crf_ens3', 'fine', 'random_split_vary', gpu=True, role='decision',
             micro=(0.4, -0.2, 1.0), name=rr.WEEK5_FILE)
    for g, micro in (('fine', (1.1, 0.5, 1.7)), ('coarse', (0.8, -0.6, 2.2))):
        ws.week4('crf_st3', 'crf', g, 'leave_one_paper_out', gpu=True, role='decision',
                 micro=micro, score=(47.4, 40.2, 54.7), name=rr.WEEK5_FILE)
    readme, _ = ws.generate()
    rows = table_rows(readme, '#### NER week-5 comparisons')
    assert [r.split(' | ')[:4] for r in rows] == [
        ['| self-training with 3 teachers vs CRF head', 'leave one paper out', 'fine', '4'],
        ['| self-training with 3 teachers vs CRF head', 'leave one paper out', 'coarse', '4'],
        ['| vote of self-training with 3 teachers vs 3-model seed ensemble',
         'new random split per seed', 'fine', '5']]
    assert all(r.endswith('| decision |') for r in rows)
    assert ('* NER week 5 (GPU runs, 10 seeds), self-training with 3 teachers vs the CRF head '
            '(paired micro-F1 difference): leaving one paper out (the pre-registered decision): '
            'fine-grained improves') in readme
    assert '#### NER week-4 comparisons' not in readme


def test_week5_reported_system_replaces_the_earlier_rows(tmp_path):
    """With week-5 comparisons, the reported system is decided on them: crf_ens3 (its scores:
    the 10-seed vote family's) unless a vote passes. A vote that replaces crf_ens3 is named,
    with the score of crf_ens3 in parentheses, and so is the system on unseen papers if it
    differs."""
    ws = Workspace(tmp_path)
    for g, paper in (('fine', 69.92), ('coarse', 72.32)):
        ws.summary(g, 'random_split_vary', 'crf_ens3', 5, (79.5, 77.8, 81.1), paper=paper)
        ws.summary(g, 'leave_one_paper_out', 'crf_ens3', 4, (61.42, 55.8, 67.0), paper=paper)
    for g in ('fine', 'coarse'):
        ws.week4('crf_ens3', 'crf', g, 'random_split_vary', gpu=True, micro=(2.0, 1.0, 3.0),
                 score=(80.5, 78.0, 83.0), name=rr.WEEK5_FILE)
        ws.week4('crf_ens3', 'crf', g, 'leave_one_paper_out', gpu=True, micro=(2.0, 1.0, 3.0),
                 score=(50.0, 44.0, 56.0), name=rr.WEEK5_FILE)
    # coarse, unseen papers: the vote of self-training passes, the label-semantics vote too
    # but by less
    ws.week4('crf_st3_ens3', 'crf_ens3', 'coarse', 'leave_one_paper_out', gpu=True,
             role='decision', micro=(1.5, 0.5, 2.5), score=(62.0, 56.0, 68.0), name=rr.WEEK5_FILE)
    ws.week4('crf_ls_ens3', 'crf_ens3', 'coarse', 'leave_one_paper_out', gpu=True,
             role='decision', micro=(1.0, 0.2, 1.8), score=(61.5, 56.0, 67.0), name=rr.WEEK5_FILE)
    readme, roadmap = ws.generate()
    rows = [line for line in glance(readme).split('\n') if 'reported system' in line]
    assert len(rows) == 2
    assert rows[0].startswith('| NER, fine-grained, reported system (week-5 protocol; 10 seeds, '
                              'GPU runs): 3-model seed ensemble `crf_ens3` | micro F1 80.50 '
                              '[78.00, 83.00] over 5 random splits, 50.00 on an unseen paper | '
                              'published 69.92 (none for an unseen paper): **beaten**')
    assert rows[1].startswith('| NER, coarse-grained, reported system (week-5 protocol; 10 seeds, '
                              'GPU runs): 3-model seed ensemble `crf_ens3`; on unseen papers the '
                              'vote of self-training with 3 teachers `crf_st3_ens3` | micro F1 '
                              '80.50 [78.00, 83.00] over 5 random splits; 62.00 on an unseen paper '
                              '(`crf_ens3` 50.00) | published 72.32')
    standing = next(line for line in roadmap.split('\n') if 'reported system' in line)
    assert standing.startswith(
        '| NER, reported system (week-5 protocol; 10 seeds, GPU runs) | fine: 3-model seed '
        'ensemble `crf_ens3`, micro F1 80.50 [78.00, 83.00] over 5 new splits, 50.00 leaving one '
        'paper out; coarse: 3-model seed ensemble `crf_ens3`, micro F1 80.50 [78.00, 83.00] over '
        '5 new splits; on unseen papers the vote of self-training with 3 teachers '
        '`crf_st3_ens3`, 62.00 leaving one paper out (`crf_ens3` 50.00) | same; ')

    # fine, random splits: the multi-task vote passes; its scores decide against the paper
    ws.week4('crf_mt_ens3', 'crf_ens3', 'fine', 'random_split_vary', gpu=True,
             role='decision', micro=(1.0, 0.2, 1.8), score=(81.5, 79.0, 84.0), name=rr.WEEK5_FILE)
    readme, roadmap = ws.generate()
    row = next(line for line in glance(readme).split('\n') if 'fine-grained, reported' in line)
    assert row.startswith('| NER, fine-grained, reported system (week-5 protocol; 10 seeds, GPU '
                          'runs): multi-task vote `crf_mt_ens3`; on unseen papers the 3-model '
                          'seed ensemble `crf_ens3` | micro F1 81.50 [79.00, 84.00] over 5 random '
                          'splits (`crf_ens3` 80.50); 50.00 on an unseen paper | published 69.92')
    standing = next(line for line in roadmap.split('\n') if 'reported system' in line)
    assert ('fine: multi-task vote `crf_mt_ens3`, micro F1 81.50 [79.00, 84.00] over 5 new '
            'splits (`crf_ens3` 80.50); on unseen papers the 3-model seed ensemble `crf_ens3`, '
            '50.00 leaving one paper out; coarse:') in standing


def test_published_split_row(tmp_path):
    ws = Workspace(tmp_path)
    (ws.results / 'gpu').mkdir()
    for g, base, crf in (('fine', (70.5, 69.0, 72.0), (73.0, 72.0, 74.0)),
                         ('coarse', (75.0, 74.0, 76.0), (77.0, 76.0, 78.0))):
        for variant, micro in ((None, base), ('crf', crf)):
            (ws.results / 'gpu' / rr.ner_file(g, 'published_split', variant)).write_text(
                json.dumps({'summary': {'n_runs': 10, 'micro': {
                    'mean': micro[0], 'ci95': list(micro[1:])}}}))
    readme, _ = ws.generate()
    rows = [line for line in glance(readme).split('\n') if 'published split' in line]
    assert rows == [
        "| NER, fine-grained, the published split (the release's partition, micro F1), 10 seeds "
        "(GPU runs) | plain MatSciBERT 70.50 [69.00, 72.00]; CRF head 73.00 [72.00, 74.00] | "
        "published 69.92: within the 95% CI of plain MatSciBERT (reproduced), below (**beaten**) "
        "that of the CRF head; the CIs are over seeds only |",
        "| NER, coarse-grained, the published split (the release's partition, micro F1), 10 seeds "
        "(GPU runs) | plain MatSciBERT 75.00 [74.00, 76.00]; CRF head 77.00 [76.00, 78.00] | "
        "published 72.32: below the 95% CI of plain MatSciBERT, below (**beaten**) that of the "
        "CRF head; the CIs are over seeds only |"]


def _spread(mean, lo, hi, diff=False):
    return {'mean_diff' if diff else 'mean': mean, 'ci95': [lo, hi]}


def test_week5_analysis_bullets(tmp_path):
    """The exploratory week-5 analyses (ner_seed_analysis.py) give three bullets: the
    published split, how far a 3-seed design misleads, and the rare coarse-grained types of
    the reported system."""
    ws = Workspace(tmp_path)
    (ws.results / 'gpu').mkdir()
    readme, _ = ws.generate()
    assert 'published split (week-5' not in readme and 'how far a 3-seed' not in readme
    published = {}
    for g, paper, base, crf in (('fine', 69.92, 70.2, 73.3), ('coarse', 72.32, 72.5, 75.5)):
        published[g] = {
            'published': paper,
            'baseline': {'micro': _spread(base, base - 1, base + 1),
                         'macro': _spread(base - 3, base - 4, base - 2),
                         'published_lies': 'within'},
            'crf': {'micro': _spread(crf, crf - 1, crf + 1),
                    'macro': _spread(crf - 2, crf - 3, crf - 1), 'published_lies': 'below'},
            'crf_minus_baseline': {'micro': _spread(3.1, 1.6, 4.6, diff=True),
                                   'micro_strict': _spread(1.7, 0.2, 3.2, diff=True)},
            'crf_votes': {'micro': _spread(crf + 1, crf, crf + 2)}}
    triple = {'n_triples': 120, 'share_above_0': 0.41, 'share_below_0': 0.0,
              'share_opposite_sign': 0.083,
              'ten_seed': {'mean_diff': 1.83, 'ci95': [0.32, 3.34], 'passes': True}}
    types = {'SDE': {'f1': 17.7, 'support_per_split': 3.0, 'n': 12, 'macro_without': 73.11,
                     'macro_with': 71.02},
             'MIP': {'f1': 17.71, 'support_per_split': 5.2, 'n': 15, 'macro_without': 72.0,
                     'macro_with': 71.0},
             'MAT': {'f1': 92.2, 'support_per_split': 17.6, 'n': 15, 'macro_without': 70.6,
                     'macro_with': 71.0}}
    analysis = {'published_split': published,
                'triples': {'fine_random_split_vary': {'crf_mt': triple}},
                'spread': {f'{g}_{p}': {'crf': {'mean_sd': sd}} for g, p, sd in (
                    ('fine', 'random_split_vary', 1.69), ('coarse', 'random_split_vary', 1.46),
                    ('fine', 'leave_one_paper_out', 2.48),
                    ('coarse', 'leave_one_paper_out', 2.23))},
                'types': {'coarse_random_split_vary': {'crf_ens3': types}}}
    (ws.results / 'gpu' / rr.WEEK5_ANALYSIS).write_text(json.dumps(analysis))
    readme, _ = ws.generate()
    assert ("* NER, the published split (week-5 amendment: the release's partition and tags, "
            "micro F1 as its code scores it; 10 model seeds, GPU runs; the 95% CIs are over the "
            "seeds of one split): plain MatSciBERT, the paper's model, scores 70.20 [69.20, "
            "71.20] fine-grained and 72.50 [71.50, 73.50] coarse-grained. Both published "
            "numbers lie within these intervals. The CRF head adds +3.10 [+1.60, +4.60] "
            "fine-grained and +3.10 [+1.60, +4.60] coarse-grained, paired by seed (strict IOB2 "
            "+1.70 [+0.20, +3.20] and +1.70 [+0.20, +3.20]), and its votes over the 3 seed "
            "triples score 74.30 and 76.50. Macro F1 is lower: 67.20 and 69.50 for plain "
            "MatSciBERT, 71.30 and 73.50 for the CRF head.\n") in readme
    assert ("* NER, how far a 3-seed design misleads (week 5, exploratory): for each family "
            "against the CRF head on the protocol of its decision, the share of the 120 triples "
            "of the 10 seeds (the same triple on both sides) whose 3-seed comparison passes the "
            "rule, and whose mean difference has the opposite sign to the 10-seed one. "
            "Multi-task (fine + coarse heads) over the 5 random splits: fine-grained (+1.83 "
            "[+0.32, +3.34] with 10 seeds) 41% pass, 8% opposite sign. The SD of a split's "
            "micro F1 over the 10 seeds of the CRF head averages 1.69 fine-grained and 1.46 "
            "coarse-grained over the 5 random splits, and 2.48 fine and 2.23 coarse over the 4 "
            "held-out papers.\n") in readme
    # without week-5 comparisons the reported system is crf_ens3
    assert ("* NER, the coarse-grained types that keep macro F1 down (week 5, exploratory; the "
            "3 votes of the reported system `crf_ens3` over the 5 random splits, F1 and test "
            "entities per split): SDE 17.7 (3.0), MIP 17.7 (5.2), MAT 92.2 (17.6). Macro F1 "
            "counts each type alike: without SDE alone it would be 73.11 instead of 71.02 on "
            "the splits that have it.\n") in readme
    published['coarse']['baseline']['published_lies'] = 'below'
    (ws.results / 'gpu' / rr.WEEK5_ANALYSIS).write_text(json.dumps(analysis))
    readme, _ = ws.generate()
    assert ('The published 69.92 lies within the fine-grained interval, 72.32 below the '
            'coarse-grained one.') in readme


def test_week6_lists_follow_ner_seedmeans():
    ns = pytest.importorskip('ner_seedmeans')
    assert rr.WEEK6_FILE == ns.OUT_NAMES[6] and rr.WEEK6_SCREEN_FILE == ns.SCREEN_NAME
    assert rr.WEEK6_METHODS == ns.WEEK6_METHODS and rr.UNSEEN_BEST == ns.UNSEEN_BEST
    assert list(rr.WEEK6_SCREEN_COMPARISONS) == [
        (v, r) for v, r, _ in ns.comparison_list(6, True, screen=True)]
    assert list(rr.WEEK6_COMPARISONS) == [(v, r) for v, r, _ in ns.comparison_list(6, True)]
    assert set(rr.WEEK6_NAMES) >= set(ns.WEEK6_SYSTEMS) | set(ns.WEEK6_SCREEN_SYSTEMS)


def test_week6_tables_and_bullets(tmp_path):
    """The screen and the 10-seed comparisons get a table each, over the held-out papers, and
    a bullet per comparison with each granularity's role."""
    ws = Workspace(tmp_path)
    lopo = 'leave_one_paper_out'
    for g, micro in (('fine', (1.2, -0.5, 2.9)), ('coarse', (-0.4, -1.0, 0.2))):
        ws.week4('crf_deberta', 'crf', g, lopo, gpu=True, role='screen', micro=micro,
                 score=(50.0, 44.0, 56.0), name=rr.WEEK6_SCREEN_FILE)
    ws.week4('crf_deberta_ens3', 'crf_mt_ens3', 'fine', lopo, gpu=True, role='decision',
             micro=(2.0, 0.5, 3.5), score=(55.0, 48.0, 62.0), name=rr.WEEK6_FILE)
    ws.week4('crf_deberta_ens3', 'crf_mt_ens3', 'coarse', lopo, gpu=True, role='reported',
             micro=(1.0, -1.0, 3.0), score=(62.0, 56.0, 68.0), name=rr.WEEK6_FILE)
    readme, _ = ws.generate()
    rows = table_rows(readme, '#### NER week-6 screen')
    assert [r.split(' | ')[:3] for r in rows] == [
        ['| DeBERTa-v3-large vs CRF head', 'fine', '4'],
        ['| DeBERTa-v3-large vs CRF head', 'coarse', '4']]
    assert all(r.endswith('| screen |') for r in rows)
    rows = table_rows(readme, '#### NER week-6 comparisons')
    assert [r.split(' | ')[0] for r in rows] == ['| DeBERTa-v3-large vote vs multi-task vote'] * 2
    assert ('* NER week 6, the screen (GPU runs, seeds 0, 100 and 200, leaving one paper out; '
            'mean paired micro-F1 difference [95% CI]; it decides only which methods get 10 '
            'seeds): DeBERTa-v3-large goes to stage 2; against the CRF head fine-grained +1.20 '
            '[-0.50, +2.90], coarse-grained -0.40 [-1.00, +0.20].\n') in readme
    assert ('* NER week 6 (GPU runs, 10 seeds), DeBERTa-v3-large vote vs the multi-task vote '
            '(paired micro-F1 difference, leaving one paper out): fine-grained improves (+2.00, '
            '95% CI [+0.50, +3.50]') in readme
    assert 'the pre-registered decision); coarse-grained no significant change' in readme


def test_week7_table_and_bullets(tmp_path):
    """Week 7, part 1: the consistency rule gets a table over both protocols (the held-out
    papers first) and a bullet per decision, with each granularity's role."""
    ws = Workspace(tmp_path)
    lopo, rand = 'leave_one_paper_out', 'random_split_vary'
    for g, micro, role in (('fine', (0.30, -0.35, 0.95), 'decision'),
                           ('coarse', (0.12, 0.01, 0.24), 'decision')):
        ws.week4('crf_dc', 'crf', g, lopo, gpu=True, role=role, micro=micro,
                 score=(49.0, 45.0, 53.0), name=rr.WEEK7_FILE)
    ws.week4('crf_dc', 'crf', 'fine', rand, gpu=True, micro=(-0.12, -0.40, 0.16),
             score=(77.5, 76.0, 79.0), name=rr.WEEK7_FILE)
    ws.week4('crf_mt_ens3_dc', 'crf_mt_ens3', 'fine', lopo, gpu=True, role='decision',
             micro=(0.33, -0.74, 1.40), score=(53.0, 47.0, 59.0), name=rr.WEEK7_FILE)
    ws.week4('crf_dcr', 'crf', 'coarse', lopo, gpu=True, micro=(0.64, 0.29, 1.00),
             score=(59.0, 55.0, 63.0), name=rr.WEEK7_FILE)
    readme, _ = ws.generate()
    rows = table_rows(readme, '#### NER week-7 comparisons, part 1')
    assert [r.split(' | ')[:3] for r in rows] == [
        ['| CRF head + the consistency rule vs CRF head', 'leave one paper out', 'fine'],
        ['| CRF head + the consistency rule vs CRF head', 'leave one paper out', 'coarse'],
        ['| CRF head + the consistency rule vs CRF head', 'new random split per seed', 'fine'],
        ['| multi-task vote + the consistency rule vs multi-task vote', 'leave one paper out',
         'fine'],
        ['| CRF head + the consistency rule with retyping vs CRF head', 'leave one paper out',
         'coarse']]
    assert [r.rsplit(' | ', 1)[1] for r in rows] == ['decision |', 'decision |', 'reported |',
                                                     'decision |', 'reported |']
    assert ('* NER week 7, part 1 (GPU runs; the consistency rule on the saved predictions of 10 '
            'seeds), CRF head + the consistency rule vs the CRF head (paired micro-F1 difference, '
            'leaving one paper out): fine-grained no significant change (+0.30, 95% CI [-0.35, '
            '+0.95]') in readme
    assert ('the pre-registered decision); coarse-grained improves (+0.12, 95% CI [+0.01, '
            '+0.24]') in readme
    assert '* NER week 7, part 1 (GPU runs; the consistency rule on the saved predictions of 10 ' \
        'seeds), multi-task vote + the consistency rule vs the multi-task vote' in readme
    # retyping is reported in the table only
    assert 'retyping vs the CRF head' not in readme


def test_week7_part2a_table_and_bullets(tmp_path):
    """Week 7, part 2a: recall calibration gets a table of its own and a bullet per decision;
    the bias chosen on validation is reported in the table only."""
    ws = Workspace(tmp_path)
    lopo = 'leave_one_paper_out'
    for g, micro in (('fine', (1.50, 0.40, 2.60)), ('coarse', (0.20, -0.30, 0.70))):
        ws.week4('crf_eb', 'crf', g, lopo, gpu=True, role='decision', micro=micro,
                 score=(48.0, 44.0, 52.0), name=rr.WEEK7_FILE)
    ws.week4('crf_eb_ens3', 'crf_mt_ens3', 'fine', lopo, gpu=True, role='decision',
             micro=(-0.50, -2.00, 1.00), score=(51.0, 45.0, 57.0), name=rr.WEEK7_FILE)
    ws.week4('crf_ebv', 'crf', 'fine', lopo, gpu=True, micro=(0.05, -0.10, 0.20),
             score=(46.6, 42.0, 51.0), name=rr.WEEK7_FILE)
    readme, _ = ws.generate()
    rows = table_rows(readme, '#### NER week-7 comparisons, part 2a')
    assert [r.split(' | ')[:3] for r in rows] == [
        ['| CRF head + recall calibration vs CRF head', 'leave one paper out', 'fine'],
        ['| CRF head + recall calibration vs CRF head', 'leave one paper out', 'coarse'],
        ['| vote of the CRF head + recall calibration vs multi-task vote', 'leave one paper out',
         'fine'],
        ['| CRF head + recall calibration chosen on validation vs CRF head',
         'leave one paper out', 'fine']]
    assert [r.rsplit(' | ', 1)[1] for r in rows] == ['decision |'] * 3 + ['reported |']
    assert ('* NER week 7, part 2a (GPU runs; recall calibration, its bias chosen on inner folds, '
            '10 seeds), CRF head + recall calibration vs the CRF head (paired micro-F1 '
            'difference, leaving one paper out): fine-grained improves (+1.50, 95% CI [+0.40, '
            '+2.60]') in readme
    assert ('* NER week 7, part 2a (GPU runs; recall calibration, its bias chosen on inner folds, '
            '10 seeds), vote of the CRF head + recall calibration vs the multi-task vote') in readme
    assert 'chosen on validation vs the CRF head' not in readme
    assert '#### NER week-7 comparisons, part 1' not in readme


def test_week6_screen_bullets_follow_the_screen_rule(tmp_path):
    """A method stops unless a granularity's mean difference is above 0, whatever the CIs; the
    screen's vote comparison rides along; a method with one granularity gets no bullet."""
    ns = pytest.importorskip('ner_seedmeans')
    ws = Workspace(tmp_path)
    lopo = 'leave_one_paper_out'
    for g, micro in (('fine', (-0.2, -0.4, -0.1)), ('coarse', (-3.0, -5.0, -1.0))):
        ws.week4('crf_ctx', 'crf', g, lopo, gpu=True, role='screen', micro=micro,
                 score=(50.0, 44.0, 56.0), name=rr.WEEK6_SCREEN_FILE)
        ws.week4('crf_ctx_ens3', 'crf_ens3', g, lopo, gpu=True, role='reported',
                 micro=(0.5, -0.5, 1.5), score=(52.0, 46.0, 58.0), name=rr.WEEK6_SCREEN_FILE)
    ws.week4('crf_mt_st3', 'crf', 'fine', lopo, gpu=True, role='screen', micro=(4.0, 3.0, 5.0),
             score=(50.0, 44.0, 56.0), name=rr.WEEK6_SCREEN_FILE)
    readme, _ = ws.generate()
    assert ('document context stops; against the CRF head fine-grained -0.20 [-0.40, -0.10], '
            'coarse-grained -3.00 [-5.00, -1.00]; its vote against the 3-model seed ensemble '
            'fine-grained +0.50 [-0.50, +1.50], coarse-grained +0.50 [-0.50, +1.50].\n') in readme
    assert 'multi-task self-training goes' not in readme and 'improves' not in readme
    screen = json.loads((ws.results / 'gpu' / rr.WEEK6_SCREEN_FILE).read_text())
    assert ns.screen_verdicts(screen) == {'crf_ctx': 'stops (fine -0.20, coarse -3.00)',
                                          'crf_mt_st3': 'incomplete'}


def test_week7_mention_replacement_screen_and_decisions(tmp_path):
    """Part 2b: mention replacement's screen gets a week-7 bullet (from its own file), and its
    stage-2 decisions a bullet each, as the week-6 methods do."""
    ns = pytest.importorskip('ner_seedmeans')
    ws = Workspace(tmp_path)
    lopo = 'leave_one_paper_out'
    for g, micro in (('fine', (0.4, -1.0, 1.8)), ('coarse', (-0.3, -1.0, 0.4))):
        ws.week4('crf_mr', 'crf', g, lopo, gpu=True, role='screen', micro=micro,
                 score=(47.0, 40.0, 54.0), name=rr.WEEK7_SCREEN_FILE)
    readme, _ = ws.generate()
    assert ('* NER week 7, part 2b, the screen (GPU runs, seeds 0, 100 and 200, leaving one '
            'paper out; mean paired micro-F1 difference [95% CI]; it decides only which methods '
            'get 10 seeds): mention replacement goes to stage 2; against the CRF head '
            'fine-grained +0.40 [-1.00, +1.80], coarse-grained -0.30 [-1.00, +0.40].\n') in readme
    screen = json.loads((ws.results / 'gpu' / rr.WEEK7_SCREEN_FILE).read_text())
    assert ns.screen_verdicts(screen, ns.WEEK7_METHODS) == {
        'crf_mr': 'goes to stage 2 (fine +0.40, coarse -0.30)'}
    for g in ('fine', 'coarse'):
        ws.week4('crf_mr', 'crf', g, lopo, gpu=True, role='decision', micro=(0.2, -0.9, 1.3),
                 score=(47.0, 40.0, 54.0), name=rr.WEEK7_FILE)
    readme, _ = ws.generate()
    assert ('* NER week 7, part 2b (GPU runs; mention replacement, 10 seeds), mention '
            'replacement vs the CRF head (paired micro-F1 difference, leaving one paper out): '
            'fine-grained no significant change') in readme


def test_week9_table_and_bullets(tmp_path):
    """Week 9: mention replacement on the multi-task model, a table of its comparisons and a
    bullet per decision, from its own file."""
    ws = Workspace(tmp_path)
    lopo = 'leave_one_paper_out'
    for g in ('fine', 'coarse'):
        ws.week4('crf_mt_mr', 'crf_mt', g, lopo, gpu=True, role='decision',
                 micro=(1.5, 0.4, 2.6), score=(51.0, 44.0, 58.0), name=rr.WEEK9_FILE)
        ws.week4('crf_mt_mr', 'crf_mr', g, lopo, gpu=True, micro=(0.5, -0.4, 1.4),
                 score=(51.0, 44.0, 58.0), name=rr.WEEK9_FILE)
    ws.week4('crf_mt_mr_ens3', 'crf_mt_ens3', 'fine', lopo, gpu=True, role='decision',
             micro=(0.9, -1.0, 2.8), score=(53.0, 45.0, 61.0), name=rr.WEEK9_FILE)
    readme, _ = ws.generate()
    assert ('* NER week 9 (GPU runs; mention replacement on the multi-task model, 10 seeds), '
            'multi-task + mention replacement vs the multi-task (fine + coarse heads) (paired '
            'micro-F1 difference, leaving one paper out): fine-grained improves') in readme
    assert ('multi-task + mention-replacement vote vs the multi-task vote (paired micro-F1 '
            'difference, leaving one paper out): fine-grained no significant change') in readme
    assert '#### NER week-9 comparisons, mention replacement on the multi-task model' in readme
    assert '| multi-task + mention replacement vs mention replacement |' in readme


def test_week11_table_and_bullet(tmp_path):
    """Week 11: mention replacement with silver names, a table of its MaterioMiner
    comparisons and a bullet per granularity's decision, from its own file."""
    ws = Workspace(tmp_path)
    lopo = 'leave_one_paper_out'
    for g in ('fine', 'coarse'):
        ws.week4('crf_mrs', 'crf_mr', g, lopo, gpu=True, role='decision',
                 micro=(0.8, 0.1, 1.5), score=(52.0, 45.0, 59.0), name=rr.WEEK11_FILE)
        ws.week4('crf_mrs', 'crf', g, lopo, gpu=True, micro=(3.5, 2.0, 5.0),
                 score=(52.0, 45.0, 59.0), name=rr.WEEK11_FILE)
    readme, _ = ws.generate()
    assert ('* NER week 11 (GPU runs; mention replacement with silver names, 10 seeds), '
            'mention replacement with silver names vs the mention replacement (paired '
            'micro-F1 difference, leaving one paper out): fine-grained improves') in readme
    assert '#### NER week-11 comparisons, mention replacement with silver names' in readme
    assert '| mention replacement with silver names vs CRF head |' in readme


def test_week12_analogue_table_and_bullet(tmp_path):
    """Week 12: the MaterioMiner analogue of the screen's winner, a table of its comparisons
    and a bullet per decision against the CRF head, from its own file."""
    ws = Workspace(tmp_path)
    lopo = 'leave_one_paper_out'
    for g in ('fine', 'coarse'):
        ws.week4('crf_mrm6', 'crf', g, lopo, gpu=True, role='decision',
                 micro=(2.8, 1.1, 4.5), score=(52.0, 45.0, 59.0), name=rr.WEEK12_FILE)
        ws.week4('crf_mrm6', 'crf_mr', g, lopo, gpu=True, micro=(0.5, -1.0, 2.0),
                 score=(52.0, 45.0, 59.0), name=rr.WEEK12_FILE)
        ws.week4('crf_st3', 'crf', g, lopo, gpu=True, micro=(1.7, -1.2, 4.6),
                 score=(48.0, 41.0, 55.0), name=rr.WEEK12_FILE)
    readme, _ = ws.generate()
    assert ("* NER week 12 (GPU runs; the MaterioMiner analogue of the screen's winner, 10 "
            'seeds), mention replacement with silver mentions, 6 copies vs the CRF head (paired '
            'micro-F1 difference, leaving one paper out): fine-grained improves') in readme
    assert "#### NER week-12 comparisons, the MaterioMiner analogue of the screen's winner" in readme
    assert '| mention replacement with silver mentions, 6 copies vs mention replacement |' in readme
    assert 'self-training with 3 teachers vs the CRF head (paired' not in readme   # reported


def test_a_week6_vote_that_passes_takes_the_unseen_slot(tmp_path):
    ws = Workspace(tmp_path)
    lopo = 'leave_one_paper_out'
    for g in ('fine', 'coarse'):
        ws.week4('crf_ens3', 'crf', g, 'random_split_vary', gpu=True, micro=(2.0, 1.0, 3.0),
                 score=(80.5, 78.0, 83.0), name=rr.WEEK5_FILE)
        ws.week4('crf_ens3', 'crf', g, lopo, gpu=True, micro=(2.0, 1.0, 3.0),
                 score=(50.0, 44.0, 56.0), name=rr.WEEK5_FILE)
    ws.week4('crf_mt_ens3', 'crf_ens3', 'fine', lopo, gpu=True, role='decision',
             micro=(3.6, 0.5, 6.8), score=(52.5, 44.0, 61.0), name=rr.WEEK5_FILE)
    readme, roadmap = ws.generate()
    fine = next(r for r in glance(readme).split('\n') if 'fine-grained, reported' in r)
    assert 'on unseen papers the multi-task vote `crf_mt_ens3`' in fine
    assert '(week-5 protocol; 10 seeds, GPU runs)' in fine
    # fine-grained, the document-context vote beats the multi-task vote (decision 2)
    ws.week4('crf_ctx_ens3', 'crf_mt_ens3', 'fine', lopo, gpu=True, role='decision',
             micro=(1.5, 0.3, 2.7), score=(54.0, 46.0, 62.0), name=rr.WEEK6_FILE)
    # reported only: does not count
    ws.week4('crf_deberta_ens3', 'crf_mt_ens3', 'coarse', lopo, gpu=True, role='reported',
             micro=(5.0, 1.0, 9.0), score=(66.0, 60.0, 72.0), name=rr.WEEK6_FILE)
    readme, roadmap = ws.generate()
    rows = [r for r in glance(readme).split('\n') if 'reported system' in r]
    assert ('(week-5 and week-6 protocols; 10 seeds, GPU runs): 3-model seed ensemble '
            '`crf_ens3`; on unseen papers the document-context vote `crf_ctx_ens3` | micro F1 '
            '80.50 [78.00, 83.00] over 5 random splits; 54.00 on an unseen paper') in rows[0]
    assert 'on unseen papers' not in rows[1]                  # coarse keeps crf_ens3
    standing = next(line for line in roadmap.split('\n') if 'reported system' in line)
    assert standing.startswith('| NER, reported system (week-5 and week-6 protocols;')


def test_sofc_slot_row_and_bullet(tmp_path):
    """Week 7, part 2d: SOFC-Exp's slot level gets its own row and bullet (the CRF head
    against plain MatSciBERT reported, not decided), and multi-task training decides on each
    level, its votes reported against the CRF votes."""
    ws = Workspace(tmp_path)
    (ws.results / 'gpu').mkdir()
    for variant, micro in ((None, (71.0, 70.0, 72.0)), ('crf', (72.5, 71.9, 73.1))):
        (ws.results / 'gpu' / rr.ner_file('sofc_slot', 'published_split', variant)).write_text(
            json.dumps({'summary': {'n_runs': 10, 'micro': {'mean': micro[0],
                                                            'ci95': list(micro[1:])},
                                    'macro': {'mean': micro[0] - 8, 'ci95': [0, 0]}}}))
    readme, _ = ws.generate()
    row = next(r for r in glance(readme).split('\n') if 'slot types' in r)
    assert row == ("| NER on SOFC-Exp's slot types, the official split, 10 seeds (GPU runs) | "
                   'plain MatSciBERT 71.00 [70.00, 72.00]; CRF head 72.50 [71.90, 73.10] (macro '
                   '64.50) | published MatSciBERT-CRF of SOFC-Slot 72.85 micro, 65.35 macro '
                   "(Gupta et al. 2022): within the CRF head's 95% CI (over seeds only) |")

    def diff(mean, lo, hi, n=10):
        return {'mean_diff': mean, 'ci95': [lo, hi], 'per_run': [mean] * n}
    mt = {'micro': {'mean': 74.0, 'ci95': [73.4, 74.6]},
          'minus_crf': {'micro': diff(1.5, 0.9, 2.1), 'micro_strict': diff(1.2, 0.6, 1.8)},
          'n_pairs': 10, 'votes': {'micro': {'mean': 75.1, 'ci95': [74.0, 76.2]}},
          'votes_minus_crf_votes': {'micro': diff(1.0, -0.5, 2.5, 3),
                                    'micro_strict': diff(0.8, -0.7, 2.3, 3)}}
    analysis = {'published_split': {'sofc_slot': {
        'published': 72.85, 'published_macro': 65.35,
        'baseline': {'micro': {'mean': 71.0, 'ci95': [70.0, 72.0]}},
        'crf': {'micro': {'mean': 72.5, 'ci95': [71.9, 73.1]},
                'macro': {'mean': 64.5, 'ci95': [63.0, 66.0]}},
        'crf_minus_baseline': {'micro': diff(1.5, 0.5, 2.5), 'micro_strict': diff(1.0, 0.0, 2.0)},
        'crf_votes': {'micro': {'mean': 74.1, 'ci95': [73.0, 75.2]}},
        'methods': {'crf_mt': mt}}}}
    (ws.results / 'gpu' / rr.WEEK5_ANALYSIS).write_text(json.dumps(analysis))
    readme, _ = ws.generate()
    assert ("* NER on SOFC-Exp's slot types (week 7, part 2d; the 17 slot types that its "
            'evaluation scores, on the same sentences and official split; 10 model seeds, GPU '
            'runs; the 95% CIs are over seeds): plain MatSciBERT 71.00 [70.00, 72.00], the CRF '
            'head 72.50 [71.90, 73.10] (macro 64.50), against the published MatSciBERT-CRF of '
            'SOFC-Slot 72.85 micro and 65.35 macro. The CRF head against plain MatSciBERT, paired '
            'by seed: +1.50, 95% CI [+0.50, +2.50]; strict IOB2 +1.00 [+0.00, +2.00] (reported; '
            'no decision); the CRF votes over the 3 seed triples score 74.10. Week 7, part 2d, '
            'multi-task training on both levels against the CRF head, paired by seed: 74.00 '
            '[73.40, 74.60], improves (+1.50, 95% CI [+0.90, +2.10]; strict IOB2 +1.20 [+0.60, '
            '+1.80]); its votes over the 3 seed triples score 75.10, against the CRF votes, '
            'paired by triple: +1.00, 95% CI [-0.50, +2.50]; strict IOB2 +0.80 [-0.70, +2.30] '
            '(reported; no decision).\n') in readme
    assert '* NER on SOFC-Exp (week 6' not in readme           # no entity-level results


def test_sofc_row_and_bullet(tmp_path):
    ws = Workspace(tmp_path)
    (ws.results / 'gpu').mkdir()
    for variant, micro in ((None, (80.5, 79.9, 81.1)), ('crf', (84.8, 84.3, 85.3))):
        (ws.results / 'gpu' / rr.ner_file('sofc', 'published_split', variant)).write_text(
            json.dumps({'summary': {'n_runs': 10, 'micro': {'mean': micro[0],
                                                            'ci95': list(micro[1:])},
                                    'macro': {'mean': micro[0] - 2, 'ci95': [0, 0]}}}))
    readme, _ = ws.generate()
    row = next(r for r in glance(readme).split('\n') if 'SOFC-Exp' in r)
    assert row == ('| NER on SOFC-Exp, the official split, 10 seeds (GPU runs) | plain '
                   'MatSciBERT 80.50 [79.90, 81.10]; CRF head 84.80 [84.30, 85.30] (macro 82.80) '
                   '| published MatSciBERT-CRF 84.77 micro, 82.39 macro (Gupta et al. 2022): '
                   "within the CRF head's 95% CI (over seeds only) |")
    spread = {'mean': 84.8, 'ci95': [84.3, 85.3]}

    def diff(mean, lo, hi, n=10):
        return {'mean_diff': mean, 'ci95': [lo, hi], 'per_run': [mean] * n}
    analysis = {'published_split': {'sofc': {
        'published': 84.77, 'published_macro': 82.39,
        'baseline': {'micro': {'mean': 80.5, 'ci95': [79.9, 81.1]}},
        'crf': {'micro': spread, 'macro': {'mean': 82.8, 'ci95': [82.0, 83.6]}},
        'crf_minus_baseline': {'micro': diff(4.3, 3.5, 5.1), 'micro_strict': diff(2.0, 1.0, 3.0)},
        'crf_votes': {'micro': {'mean': 85.21, 'ci95': [84.0, 86.4]}},
        'methods': {'crf_deberta': {'micro': {'mean': 84.4, 'ci95': [83.8, 85.0]},
                                    'minus_crf': {'micro': diff(-0.4, -1.1, 0.3),
                                                  'micro_strict': diff(-0.2, -0.9, 0.5)},
                                    'n_pairs': 10},
                    # an incomplete method (a session that stopped): reported, no verdict
                    'crf_ctx': {'micro': {'mean': 86.0, 'ci95': [85.0, 87.0]},
                                'minus_crf': {'micro': diff(1.2, 0.2, 2.2, 4),
                                              'micro_strict': diff(1.0, 0.1, 1.9, 4)},
                                'n_pairs': 4},
                    # week 7: on SOFC-Exp both rules decide (the amendment to part 1)
                    'crf_dc': {'micro': {'mean': 85.3, 'ci95': [84.8, 85.8]},
                               'minus_crf': {'micro': diff(0.5, 0.2, 0.8),
                                             'micro_strict': diff(0.4, 0.1, 0.7)},
                               'n_pairs': 10},
                    'crf_dcr': {'micro': {'mean': 85.2, 'ci95': [84.7, 85.7]},
                                'minus_crf': {'micro': diff(0.4, 0.1, 0.7),
                                              'micro_strict': diff(0.3, 0.0, 0.6)},
                                'n_pairs': 10},
                    # week 7, part 2a: recall calibration, chosen on the validation articles
                    'crf_ebv': {'micro': {'mean': 85.0, 'ci95': [84.5, 85.5]},
                                'minus_crf': {'micro': diff(0.2, -0.1, 0.5),
                                              'micro_strict': diff(0.1, -0.2, 0.4)},
                                'n_pairs': 10},
                    # week 7, part 2e: the CRF head on the retokenized test articles
                    'crf_cdetest': {'micro': {'mean': 85.7, 'ci95': [85.2, 86.2]},
                                    'minus_crf': {'micro': diff(0.9, 0.4, 1.4),
                                                  'micro_strict': diff(0.9, 0.4, 1.4)},
                                    'n_pairs': 10}},
        # and plain MatSciBERT on them (part 2e, decision 2)
        'baseline_retokenized_test': {'micro': {'mean': 82.4, 'ci95': [81.9, 82.9]},
                                      'minus_baseline': {'micro': diff(1.9, 1.3, 2.5),
                                                         'micro_strict': diff(1.7, 1.2, 2.2)},
                                      'n_pairs': 10}}}}
    (ws.results / 'gpu' / rr.WEEK5_ANALYSIS).write_text(json.dumps(analysis))
    readme, _ = ws.generate()
    assert ('* NER on SOFC-Exp (week 6; its official split, 26 training, 8 development and 11 '
            'test articles; 10 model seeds, GPU runs; the 95% CIs are over seeds): plain '
            'MatSciBERT 80.50 [79.90, 81.10], the CRF head 84.80 [84.30, 85.30] (macro 82.80), '
            'against the published MatSciBERT-CRF 84.77 micro and 82.39 macro. S1, the CRF head '
            'against plain MatSciBERT, paired by seed: improves (+4.30, 95% CI [+3.50, +5.10]; '
            'strict IOB2 +2.00 [+1.00, +3.00]); the CRF votes over the 3 seed triples score '
            '85.21. S2, DeBERTa-v3-large against the CRF head, paired by seed: 84.40 [83.80, '
            '85.00], no significant change (-0.40, 95% CI [-1.10, +0.30]; strict IOB2 -0.20 '
            '[-0.90, +0.50]). S2, document context against the CRF head, paired by seed: 86.00 '
            '[85.00, 87.00], +1.20, 95% CI [+0.20, +2.20]; strict IOB2 +1.00 [+0.10, +1.90] over '
            '4 of the 10 seeds (reported; no decision). Week 7, the consistency rule on the CRF '
            "head's predictions, paired by seed: 85.30 [84.80, 85.80], improves (+0.50, 95% CI "
            '[+0.20, +0.80]; strict IOB2 +0.40 [+0.10, +0.70]). Week 7, the consistency rule with '
            "retyping on the CRF head's predictions, paired by seed: 85.20 [84.70, 85.70], improves "
            '(+0.40, 95% CI [+0.10, +0.70]; strict IOB2 +0.30 [+0.00, +0.60]). Week 7, part 2a, '
            'recall calibration of the CRF head, its bias chosen on the validation articles, '
            'against the CRF head, paired by seed: 85.00 [84.50, 85.50], no significant change '
            '(+0.20, 95% CI [-0.10, +0.50]; strict IOB2 +0.10 [-0.20, +0.40]). Week 7, part 2e, '
            'the CRF head on the test articles cut into words as the training articles are, '
            'against the CRF head on the released ones, paired by seed: 85.70 [85.20, 86.20], '
            'improves (+0.90, 95% CI [+0.40, +1.40]; strict IOB2 +0.90 [+0.40, +1.40]). Week 7, '
            'part 2e, plain MatSciBERT on the retokenized test articles, against plain MatSciBERT '
            'on the released ones, paired by seed: 82.40 [81.90, 82.90], improves (+1.90, 95% CI '
            '[+1.30, +2.50]; strict IOB2 +1.70 [+1.20, +2.20]).\n') in readme


def test_mulms_bullet(tmp_path):
    """Week 10: the 2 decisions over the held-out subfields and the random pool splits; no
    bullet until the held-out subfields are complete."""
    ws = Workspace(tmp_path)
    (ws.results / 'gpu').mkdir()

    def diff(mean, lo, hi, n=7):
        return {'mean_diff': mean, 'ci95': [lo, hi], 'per_run': [mean] * n}

    def score(mean, lo, hi):
        return {'micro': {'mean': mean, 'ci95': [lo, hi]}}
    analysis = {'mulms': {'random_split_pool': {**score(70.1, 69.0, 71.2), 'n_splits': 5}}}
    (ws.results / 'gpu' / rr.WEEK5_ANALYSIS).write_text(json.dumps(analysis))
    readme, _ = ws.generate()
    assert '* NER week 10' not in readme
    analysis['mulms'].update({
        'leave_one_subfield_out': {
            'crf': score(55.2, 48.0, 62.4), 'crf_mr': score(57.0, 50.1, 63.9), 'n_subfields': 7,
            'crf_mr_minus_crf': {'micro': diff(1.8, 0.6, 3.0), 'micro_strict': diff(1.7, 0.5, 2.9)},
            'votes': {'crf': score(56.0, 49.0, 63.0), 'crf_mr': score(57.5, 50.5, 64.5),
                      'crf_mr_minus_crf': {'micro': diff(1.5, -0.2, 3.2),
                                           'micro_strict': diff(1.4, -0.3, 3.1)}}},
        'gap': {'micro': 14.9}})
    (ws.results / 'gpu' / rr.WEEK5_ANALYSIS).write_text(json.dumps(analysis))
    readme, _ = ws.generate()
    assert ('* NER week 10 (MuLMS, a third corpus: its outermost entities of 13 types; GPU runs, '
            '10 model seeds; each of its 7 materials subfields held out in turn, 340 training and '
            '60 validation sentences from the others; the 95% CIs are over the subfields): the '
            'CRF head 55.20 [48.00, 62.40], mention replacement 57.00 [50.10, 63.90]. Decision 1, '
            'mention replacement against the CRF head, paired by subfield: improves (+1.80, 95% '
            'CI [+0.60, +3.00]; strict IOB2 +1.70 [+0.50, +2.90]). Decision 2, its votes over the '
            '3 seed triples against the CRF votes (57.50 [50.50, 64.50] against 56.00 [49.00, '
            '63.00]), paired by subfield: no significant change (+1.50, 95% CI [-0.20, +3.20]; '
            'strict IOB2 +1.40 [-0.30, +3.10]). The CRF head on 5 random splits of the same size '
            '(descriptive): 70.10 [69.00, 71.20], 14.90 points above the held-out '
            'subfields.\n') in readme
    analysis['mulms']['silver_names'] = {
        'crf_mrs': score(58.0, 51.0, 65.0),
        'minus_crf': {'micro': diff(2.8, 0.9, 4.7), 'micro_strict': diff(2.7, 0.8, 4.6)},
        'minus_crf_mr': {'micro': diff(1.0, -0.4, 2.4), 'micro_strict': diff(0.9, -0.5, 2.3)},
        'unseen_in_silver': 0.123}
    (ws.results / 'gpu' / rr.WEEK5_ANALYSIS).write_text(json.dumps(analysis))
    readme, _ = ws.generate()
    assert ('14.90 points above the held-out subfields. Week 11, mention replacement with the '
            "teachers' silver names in its pool (the fold's sentences outside its pool and test "
            'set): 58.00 [51.00, 65.00]. Week-11 decision 1, against the CRF head, paired by '
            'subfield: improves (+2.80, 95% CI [+0.90, +4.70]; strict IOB2 +2.70 [+0.80, '
            '+4.60]). Week-11 decision 2, against mention replacement, paired by subfield: no '
            'significant change (+1.00, 95% CI [-0.40, +2.40]; strict IOB2 +0.90 [-0.50, '
            '+2.30]). 12.3% of the unseen test names are silver names.\n') in readme
    # week 12: the screen of two variants (no verdict words), the decision for one of them
    analysis['mulms']['week12'] = {
        'crf_mr6': {'screen': {'seeds': [0, 100, 200], 'crf_mr6': score(57.1, 50.0, 64.2),
                               'minus_crf': {'micro': diff(1.9, 0.5, 3.3)},
                               'minus_crf_mr': {'micro': diff(0.1, -0.9, 1.1)}}},
        'crf_mrm_topic': {
            'screen': {'seeds': [0, 100, 200], 'crf_mrm_topic': score(59.0, 52.0, 66.0),
                       'minus_crf': {'micro': diff(3.8, 1.5, 6.1)},
                       'minus_crf_mr': {'micro': diff(2.0, 0.4, 3.6)}, 'unseen_in_silver': 0.41},
            'decision': {'seeds': list(range(300, 1000, 100)),
                         'crf_mrm_topic': score(59.2, 52.1, 66.3),
                         'minus_crf': {'micro': diff(4.0, 1.9, 6.1),
                                       'micro_strict': diff(3.9, 1.8, 6.0)},
                         'minus_crf_mr': {'micro': diff(2.2, -0.1, 4.5),
                                          'micro_strict': diff(2.1, -0.2, 4.4)},
                         'unseen_in_silver': 0.434, 'decision_1': True, 'decision_2': False}}}
    (ws.results / 'gpu' / rr.WEEK5_ANALYSIS).write_text(json.dumps(analysis))
    readme, _ = ws.generate()
    assert ('12.3% of the unseen test names are silver names. Week 12, names from the held-out '
            "subfield's topic corpus, 6 copies (crf_mr6), the 3-seed screen against mention "
            'replacement: +0.10 [-0.90, +1.10]. Week 12, names from the held-out subfield\'s '
            'topic corpus, silver mentions (crf_mrm_topic), the 3-seed screen against mention '
            'replacement: +2.00 [+0.40, +3.60]. Week-12 decision for silver mentions on seeds '
            '300 to 900: 59.20 [52.10, 66.30]; decision 1, against the CRF head, paired by '
            'subfield: improves (+4.00, 95% CI [+1.90, +6.10]; strict IOB2 +3.90 [+1.80, '
            '+6.00]); decision 2, against mention replacement, paired by subfield: no '
            'significant change (+2.20, 95% CI [-0.10, +4.50]; strict IOB2 +2.10 [-0.20, '
            '+4.40]). 43.4% of the unseen test names are its silver names.\n') in readme


def test_a_week3_file_that_is_not_an_object_is_reported(tmp_path, capsys):
    ws = Workspace(tmp_path)
    (ws.results / rr.WEEK3_FILE).write_text('[1, 2]')
    assert ws.run() == 2 and rr.WEEK3_FILE in capsys.readouterr().err
    (ws.results / rr.WEEK3_FILE).write_text('{not json')
    assert ws.run() == 2 and rr.WEEK3_FILE in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def test_check_exit_codes_and_diff(tmp_path, capsys):
    ws = Workspace(tmp_path)
    stale = (ws.readme.read_bytes(), ws.roadmap.read_bytes())

    assert ws.run('--check') == 1                   # the stub blocks are stale
    out = capsys.readouterr().out
    assert f'--- {ws.readme} (current)' in out and f'+++ {ws.readme} (generated)' in out
    assert f'--- {ws.roadmap} (current)' in out
    assert '\n-stale' in out and '\n+All numbers are leakage-free' in out
    assert (ws.readme.read_bytes(), ws.roadmap.read_bytes()) == stale    # --check never writes

    assert ws.run() == 0                            # rewrite in place
    assert ws.readme.read_bytes() != stale[0] and ws.roadmap.read_bytes() != stale[1]
    capsys.readouterr()
    assert ws.run('--check') == 0
    assert capsys.readouterr().out.startswith('up to date')

    # a complete week-2 comparison appears in results/: only the README block changes
    ws.compare('crf_mt', 'fine', 'random_split_vary')
    assert ws.run('--check') == 1
    out = capsys.readouterr().out
    assert f'+++ {ws.readme} (generated)' in out and f'--- {ws.roadmap}' not in out
    assert '\n+#### NER week-2 variants vs the CRF head' in out
    assert ws.run() == 0 and ws.run('--check') == 0


def test_check_notices_edits_inside_the_block_but_not_outside(tmp_path, capsys):
    ws = Workspace(tmp_path)
    assert ws.run() == 0
    generated = ws.readme.read_text(encoding='utf-8')

    ws.readme.write_text(generated + "\nsomething written by hand below the block\n",
                         encoding='utf-8')
    assert ws.run('--check') == 0                  # outside the markers: not our business

    edited = generated.replace('**Reading the numbers**', '**Reading the NUMBERS**')
    assert edited != generated
    ws.readme.write_text(edited, encoding='utf-8')
    assert ws.run('--check') == 1
    assert '-**Reading the NUMBERS**' in capsys.readouterr().out
    assert ws.run() == 0 and ws.readme.read_text(encoding='utf-8') == generated


def test_a_second_run_changes_nothing(tmp_path, capsys):
    ws = Workspace(tmp_path)
    assert ws.run() == 0
    assert capsys.readouterr().out.count('updated ') == 2
    first = (ws.readme.read_bytes(), ws.roadmap.read_bytes())
    assert ws.run() == 0
    assert capsys.readouterr().out.count('unchanged ') == 2
    assert (ws.readme.read_bytes(), ws.roadmap.read_bytes()) == first


def test_pending_note(tmp_path):
    ws = Workspace(tmp_path)
    note = ("> **Still running when this snapshot was taken:** NER coarse CRF, ensemble. "
            "These rows will be added when the runs finish.")
    readme, roadmap = ws.generate('--pending', 'NER coarse CRF', 'ensemble')
    assert readme.count(note) == 1 and roadmap.count(note) == 1
    assert readme.index(note) < readme.index('**Reading the numbers**')
    assert roadmap.index(note) < roadmap.index('Full tables:')
    readme, roadmap = ws.generate()
    assert 'Still running' not in readme and 'Still running' not in roadmap


def test_check_takes_pending_into_account(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.run('--pending', 'a') == 0
    assert ws.run('--check', '--pending', 'a') == 0
    assert ws.run('--check') == 1
    assert ws.run('--check', '--pending', 'b') == 1


def test_bad_input_exits_2_and_writes_nothing(tmp_path, capsys):
    ws = Workspace(tmp_path)
    original = (ws.readme.read_bytes(), ws.roadmap.read_bytes())

    ws.roadmap.write_text("# Roadmap\n\nno markers here\n", encoding='utf-8')   # nothing to replace
    broken = ws.roadmap.read_bytes()
    for extra in ((), ('--check',)):
        assert ws.run(*extra) == 2
        assert 'STANDING:START' in capsys.readouterr().err
    assert (ws.readme.read_bytes(), ws.roadmap.read_bytes()) == (original[0], broken)

    ws.roadmap.write_text(ROADMAP_STUB + ROADMAP_STUB, encoding='utf-8')         # two blocks
    assert ws.run() == 2 and 'more than one' in capsys.readouterr().err

    ws.roadmap.write_text(ROADMAP_STUB.replace('<!-- STANDING:END -->', ''), encoding='utf-8')
    assert ws.run() == 2                                                         # no END marker

    ws.roadmap.write_text(ROADMAP_STUB, encoding='utf-8')
    (ws.results / 'ner_comparisons.json').write_text('{not json')
    assert ws.run() == 2 and 'ner_comparisons.json' in capsys.readouterr().err

    (ws.results / 'ner_comparisons.json').unlink()
    (ws.results / 'eval_pipeline_llm_off.json').unlink()
    assert ws.run() == 2 and 'eval_pipeline_llm_off.json' in capsys.readouterr().err

    ws.readme.unlink()
    assert ws.run() == 2 and 'README.md' in capsys.readouterr().err


def test_unusable_inputs_are_reported_not_raised(tmp_path, capsys, monkeypatch):
    ws = Workspace(tmp_path)

    (ws.results / 'ner_comparisons.json').write_text('[1, 2]')        # not an object
    assert ws.run() == 2 and 'ner_comparisons.json' in capsys.readouterr().err
    (ws.results / 'ner_comparisons.json').unlink()

    ws.roadmap.write_bytes(b'\xff\xfe not utf-8 \x80')
    assert ws.run() == 2 and 'ROADMAP.md' in capsys.readouterr().err
    ws.roadmap.write_text(ROADMAP_STUB, encoding='utf-8')

    args = ['--results', str(tmp_path / 'no_such_dir'), '--readme', str(ws.readme),
            '--roadmap', str(ws.roadmap)]
    assert rr.main(args) == 2
    assert 'results directory not found' in capsys.readouterr().err

    def refuse(self, *args, **kwargs):
        raise PermissionError(13, 'Permission denied')

    monkeypatch.setattr(Path, 'write_text', refuse)
    assert ws.run() == 2 and 'cannot write' in capsys.readouterr().err
    assert ws.run('--check') == 1                   # checking needs no write access


def test_a_bare_placeholder_is_replaced_on_first_use(tmp_path):
    ws = Workspace(tmp_path)
    ws.readme.write_text("# P\n\n<!-- RESULTS -->\n\nend\n", encoding='utf-8')
    ws.roadmap.write_text("# R\n\n<!-- STANDING -->\n\nend\n", encoding='utf-8')
    readme, roadmap = ws.generate()
    assert block(readme, 'RESULTS').startswith('<!-- RESULTS:START -->\nAll numbers')
    assert block(roadmap, 'STANDING').endswith('<!-- STANDING:END -->')
    assert readme.startswith('# P\n\n') and readme.endswith('\n\nend\n')
    assert ws.run('--check') == 0


def test_script_runs_standalone_with_the_path_overrides(tmp_path):
    ws = Workspace(tmp_path)
    cmd = [sys.executable, str(MODULE), '--results', str(ws.results),
           '--readme', str(ws.readme), '--roadmap', str(ws.roadmap)]
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}

    def run(*extra):
        return subprocess.run(cmd + list(extra), capture_output=True, text=True,
                              encoding='utf-8', cwd=tmp_path, env=env)

    stale = run('--check')
    assert stale.returncode == 1
    assert stale.stdout.startswith('---') and '+++' in stale.stdout and '@@' in stale.stdout
    assert run().returncode == 0
    assert run('--check').returncode == 0
    ws.compare('crf_ens3', 'coarse', 'leave_one_paper_out')
    assert run('--check').returncode == 1
    assert subprocess.run([sys.executable, str(MODULE), '--bogus'], capture_output=True,
                          cwd=tmp_path).returncode == 2
    helped = subprocess.run([sys.executable, str(MODULE), '--help'], capture_output=True,
                            text=True, encoding='utf-8', cwd=tmp_path, env=env)
    assert helped.returncode == 0
    for option in ('--check', '--pending', '--results', '--readme', '--roadmap'):
        assert option in helped.stdout


# ---------------------------------------------------------------------------
# Module hygiene
# ---------------------------------------------------------------------------
def test_module_uses_only_the_standard_library_and_no_absolute_paths():
    tree = ast.parse(MODULE.read_text(encoding='utf-8'))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split('.')[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module
            imported.add(node.module.split('.')[0])
    assert imported and imported <= set(sys.stdlib_module_names), \
        imported - set(sys.stdlib_module_names)
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                and isinstance(n.value, str)]
    assert [s for s in literals if re.fullmatch(r'(/[\w.-]+)+/?|[A-Za-z]:\\\S*', s)] == []


def test_defaults_come_from_the_file_location():
    assert rr.REPO_ROOT == Path(rr.__file__).resolve().parents[2] == REPO
    args = rr.parse_args([])
    assert args.results == REPO / 'results'
    assert args.readme == REPO / 'README.md'
    assert args.roadmap == REPO / 'docs' / 'ROADMAP.md'
    assert args.pending == [] and args.check is False
    assert rr.parse_args(['--pending', 'a b', 'c']).pending == ['a b', 'c']


# ---------------------------------------------------------------------------
# Regression: the real results/ regenerate the blocks in the current docs
# ---------------------------------------------------------------------------
def test_real_results_regenerate_the_current_blocks(tmp_path):
    """README.md and docs/ROADMAP.md must be what the generator makes from results/.

    If this fails after changing results/ or the generator, run
    `python src/graph/report_results.py` and commit the two files.
    """
    current = {'RESULTS': (REPO / 'README.md').read_text(encoding='utf-8'),
               'STANDING': (REPO / 'docs' / 'ROADMAP.md').read_text(encoding='utf-8')}
    # start from copies whose generated blocks are wiped, so that a generator which does
    # nothing cannot pass
    copies = {}
    for name, text in current.items():
        stale = f'<!-- {name}:START -->\nSTALE\n<!-- {name}:END -->'
        copies[name] = tmp_path / f'{name}.md'
        copies[name].write_text(text.replace(block(text, name), stale), encoding='utf-8')
        assert 'STALE' in copies[name].read_text(encoding='utf-8')

    assert rr.main(['--results', str(REPO / 'results'), '--readme', str(copies['RESULTS']),
                    '--roadmap', str(copies['STANDING'])]) == 0

    for name, text in current.items():
        generated = copies[name].read_text(encoding='utf-8')
        want, got = block(text, name), block(generated, name)
        assert len(want) > 500 and 'STALE' not in generated
        if got != want:
            diff = rr.unified_diff(f'{name} block', want, got)
            pytest.fail(f"the {name} block of the docs is out of date; run "
                        f"`python src/graph/report_results.py`\n{diff[:3000]}")
        # everything outside the block is carried over untouched
        assert generated == text


def test_week13_bullet(tmp_path):
    """Week 13: the consistency rule's outputs against their inputs per corpus (a verdict
    for a decision, reported otherwise), the screen's winner and the pooled estimates; no
    bullet without the week13 section."""
    ws = Workspace(tmp_path)
    (ws.results / 'gpu').mkdir()

    def diff(mean, lo, hi, n=7):
        return {'mean_diff': mean, 'ci95': [lo, hi], 'per_run': [mean] * n}

    def score(mean, lo, hi):
        return {'micro': {'mean': mean, 'ci95': [lo, hi]}}

    def entry(out, inp, mean, lo, hi, added, decision=None):
        e = {'output': score(*out), 'input': score(*inp), 'n_units': 7,
             'minus_input': {'micro': diff(mean, lo, hi), 'micro_strict': diff(mean, lo, hi)},
             'n_added_per_seed': added, 'n_retyped_per_seed': 0.0}
        if decision:
            e.update({'decision': decision, 'passes': lo > 0})
        return e
    analysis = {'mulms': {}, 'week13': {'consistency': {}, 'week12_winner': None, 'pooled': {}}}
    (ws.results / 'gpu' / rr.WEEK5_ANALYSIS).write_text(json.dumps(analysis))
    readme, _ = ws.generate()
    assert '* NER week 13' not in readme
    analysis['week13'] = {
        'consistency': {
            'mulms': {'crf_dc': entry((49.47, 44.0, 55.0), (48.66, 43.0, 54.0), 0.82, 0.18, 1.46,
                                      1213.0, '13.1a'),
                      'crf_dcr': entry((49.7, 44.0, 55.0), (48.66, 43.0, 54.0), 1.01, 0.28, 1.74,
                                       1213.0)},
            'sofc_folds': {'crf_dc': entry((82.34, 78.0, 86.0), (82.79, 78.5, 87.0), -0.44,
                                           -0.72, -0.16, 169.0)}},
        'week12_winner': 'crf_mrm_topic',
        'pooled': {'crf_mr': {
            'corpora': {'mulms': {'variant': 'crf_mr', 'n_units': 7, 'loses': False,
                                  'minus_crf': diff(2.1, -0.31, 4.51)},
                        'fine': {'variant': 'crf_mr', 'n_units': 4, 'loses': False,
                                 'minus_crf': diff(2.77, 1.51, 4.03, 4)}},
            'n_units': 11, 'minus_crf': diff(2.34, 0.94, 3.75, 11), 'status': 'reported',
            'passes': True, 'no_corpus_loses': True}}}
    (ws.results / 'gpu' / rr.WEEK5_ANALYSIS).write_text(json.dumps(analysis))
    readme, _ = ws.generate()
    assert ("* NER week 13 (the consistency rule of week 7 on the saved 10-seed predictions of "
            "held-out documents, no training; the 95% CIs are over the subfields, papers or "
            "folds). MuLMS's 7 held-out subfields, on the CRF head: 49.47 [44.00, 55.00] against "
            "48.66 [43.00, 54.00], 1213 entities added per seed; decision 13.1a, improves (+0.82, "
            "95% CI [+0.18, +1.46]; strict IOB2 +0.82 [+0.18, +1.46]). MuLMS's 7 held-out "
            "subfields, on the CRF head with retyping: 49.70 [44.00, 55.00] against 48.66 [43.00, "
            "54.00], 1213 entities added per seed; +1.01, 95% CI [+0.28, +1.74]; strict IOB2 "
            "+1.01 [+0.28, +1.74] (reported; no decision). SOFC-Exp's 5 article folds, on the CRF "
            "head: 82.34 [78.00, 86.00] against 82.79 [78.50, 87.00], 169 entities added per "
            "seed; -0.44, 95% CI [-0.72, -0.16]; strict IOB2 -0.44 [-0.72, -0.16] (reported; no "
            "decision). The week-12 screen's winner: silver mentions (crf_mrm_topic). Pooled "
            "over 11 held-out units, mention replacement against the CRF head: +2.34 [+0.94, "
            "+3.75] (MuLMS's 7 held-out subfields +2.10 [-0.31, +4.51]; MaterioMiner's 4 "
            "held-out papers +2.77 [+1.51, +4.03]), passes (reported; post hoc for plain "
            "mention replacement until 13.3 runs).\n") in readme
    # 13.3: the folds' decisions, and the pooled estimate over three corpora is 13.3's
    def folds_entry(system, mean, lo, hi, decision, **more):
        return {'system': score(*system), 'crf': score(82.79, 78.5, 87.0), 'n_units': 5,
                'minus_crf': {'micro': diff(mean, lo, hi, 5), 'micro_strict': diff(mean, lo, hi, 5)},
                'units': {}, 'decision': decision, 'passes': lo > 0, **more}
    analysis['week13']['folds'] = {
        'crf_mr': folds_entry((83.1, 79.0, 87.0), 0.31, -0.2, 0.82, '13.3a'),
        'crf_mt_mr': folds_entry((83.5, 79.0, 88.0), 0.71, 0.1, 1.32, '13.3b',
                                 minus_crf_mr={'micro': diff(0.4, -0.1, 0.9, 5),
                                               'micro_strict': diff(0.4, -0.1, 0.9, 5)},
                                 slot={'micro': {'mean': 71.2, 'ci95': [68.0, 74.4]},
                                       'n_units': 5})}
    pooled = analysis['week13']['pooled']['crf_mr']
    pooled['corpora']['sofc_folds'] = {'variant': 'crf_mr', 'n_units': 5, 'loses': False,
                                       'minus_crf': diff(0.31, -0.2, 0.82, 5)}
    pooled.update({'n_units': 16, 'minus_crf': diff(1.7, 0.6, 2.8, 16)})
    (ws.results / 'gpu' / rr.WEEK5_ANALYSIS).write_text(json.dumps(analysis))
    readme, _ = ws.generate()
    assert ("(reported; no decision). SOFC-Exp's 5 article folds, decision 13.3a, mention "
            "replacement against the CRF head, paired by fold: 83.10 [79.00, 87.00] against "
            "82.79 [78.50, 87.00]; no significant change (+0.31, 95% CI [-0.20, +0.82]; strict "
            "IOB2 +0.31 [-0.20, +0.82]). SOFC-Exp's 5 article folds, decision 13.3b, mention "
            "replacement on the multi-task model against the CRF head, paired by fold: 83.50 "
            "[79.00, 88.00] against 82.79 [78.50, 87.00]; improves (+0.71, 95% CI [+0.10, "
            "+1.32]; strict IOB2 +0.71 [+0.10, +1.32]); against mention replacement +0.40 "
            "[-0.10, +0.90] (reported; no decision); its slot level 71.20 [68.00, 74.40] "
            "(reported). The week-12 screen's winner: silver mentions (crf_mrm_topic). Pooled "
            "over 16 held-out units, mention replacement against the CRF head: +1.70 [+0.60, "
            "+2.80] (MuLMS's 7 held-out subfields +2.10 [-0.31, +4.51]; MaterioMiner's 4 "
            "held-out papers +2.77 [+1.51, +4.03]; SOFC-Exp's 5 article folds +0.31 [-0.20, "
            "+0.82]), passes (reported; 13.3's pooled estimate, no decision).\n") in readme
    # 13.4: the winner's analogue on the folds, and the winner's three-corpus estimate
    analysis['week13']['folds'] = {
        'crf_mrm6': folds_entry((83.0, 79.0, 87.0), 0.21, -0.3, 0.72, '13.4',
                                minus_crf_mr={'micro': diff(-0.1, -0.6, 0.4, 5),
                                              'micro_strict': diff(-0.1, -0.6, 0.4, 5)})}
    analysis['week13']['pooled'] = {'crf_mrm_topic6': {
        'corpora': {'mulms': {'variant': 'crf_mrm_topic6', 'n_units': 7, 'loses': False,
                              'minus_crf': diff(3.1, 0.4, 5.8)},
                    'fine': {'variant': 'crf_mrm6', 'n_units': 4, 'loses': False,
                             'minus_crf': diff(2.0, 0.5, 3.5, 4)}},
        'n_units': 11, 'minus_crf': diff(2.7, 1.1, 4.3, 11), 'status': 'decision',
        'passes': True, 'no_corpus_loses': True,
        'with_sofc_folds': {
            'corpora': {'mulms': {'variant': 'crf_mrm_topic6', 'n_units': 7},
                        'fine': {'variant': 'crf_mrm6', 'n_units': 4},
                        'sofc_folds': {'variant': 'crf_mrm6', 'n_units': 5}},
            'sofc_folds': {'variant': 'crf_mrm6', 'n_units': 5, 'loses': False,
                           'minus_crf': diff(0.21, -0.3, 0.72, 5)},
            'n_units': 16, 'minus_crf': diff(1.9, 0.8, 3.0, 16), 'status': 'reported',
            'passes': True, 'no_corpus_loses': True}}}
    (ws.results / 'gpu' / rr.WEEK5_ANALYSIS).write_text(json.dumps(analysis))
    readme, _ = ws.generate()
    assert ("SOFC-Exp's 5 article folds, decision 13.4, mention replacement with the SOFC "
            "corpus's silver mentions and 6 copies (the week-12 winner's analogue) against the "
            "CRF head, paired by fold: 83.00 [79.00, 87.00] against 82.79 [78.50, 87.00]; no "
            "significant change (+0.21, 95% CI [-0.30, +0.72]; strict IOB2 +0.21 [-0.30, "
            "+0.72]); against mention replacement -0.10 [-0.60, +0.40] (reported; no "
            "decision). The week-12 screen's winner: silver mentions (crf_mrm_topic). Pooled "
            "over 11 held-out units, silver mentions with 6 copies against the CRF head: +2.70 "
            "[+1.10, +4.30] (MuLMS's 7 held-out subfields +3.10 [+0.40, +5.80]; MaterioMiner's "
            "4 held-out papers +2.00 [+0.50, +3.50]), passes (decision 13.2); with SOFC-Exp's "
            "5 article folds (crf_mrm6, +0.21 [-0.30, +0.72]): 16 units, +1.90 [+0.80, +3.00], "
            "passes (reported; the three-corpus estimate of 13.4).\n") in readme

def test_week14_bullet(tmp_path):
    """Week 14: 14.1's pooled decision with its corpora and the per-corpus comparisons with
    the control's own differences, 14.2's decision with the mixed variant's references, and
    the ablations; no bullet without the week14 section."""
    ws = Workspace(tmp_path)
    (ws.results / 'gpu').mkdir()

    def diff(mean, lo, hi, n=7):
        return {'mean_diff': mean, 'ci95': [lo, hi], 'per_run': [mean] * n}

    def both(mean, lo, hi, n=7):
        return {'micro': diff(mean, lo, hi, n), 'micro_strict': diff(mean, lo, hi, n)}

    def score(mean, lo, hi):
        return {'micro': {'mean': mean, 'ci95': [lo, hi]}}
    analysis = {'mulms': {}, 'week13': {'consistency': {}, 'week12_winner': None, 'pooled': {}},
                'week14': {'copies': {}, 'topic': {}, 'ablations': {}}}
    (ws.results / 'gpu' / rr.WEEK5_ANALYSIS).write_text(json.dumps(analysis))
    readme, _ = ws.generate()
    assert '* NER week 14' not in readme
    control = {'crf_mr6': score(56.1, 50.0, 62.0), 'minus_crf': both(2.5, 1.0, 4.0),
               'minus_crf_mr': both(0.5, -0.5, 1.5), 'n_units': 7}
    analysis['week14'] = {
        'copies': {
            'mulms': {'decision': {'crf_mrm_topic6': score(57.6, 51.0, 64.0),
                                   'minus_crf_mr6': both(1.5, 0.2, 2.8), 'control': control,
                                   'n_units': 7, 'seeds': list(range(300, 1000, 100))},
                      'ten_seeds': {'crf_mrm_topic6': score(57.0, 51.0, 63.0),
                                    'minus_crf_mr6': both(1.2, -0.1, 2.5), 'control': control,
                                    'n_units': 7}},
            'fine': {'crf_mrm6': score(49.0, 44.0, 54.0), 'minus_crf_mr6': both(1.5, 0.5, 2.5, 4),
                     'control': {'crf_mr6': score(47.5, 43.0, 52.0),
                                 'minus_crf': both(2.5, 1.5, 3.5, 4),
                                 'minus_crf_mr': both(0.2, -0.2, 0.6, 4), 'n_units': 4},
                     'n_units': 4},
            'pooled': {'decision': '14.1', 'n_units': 11, 'minus_control': diff(1.5, 0.6, 2.4, 11),
                       'passes': True, 'no_corpus_loses': True,
                       'corpora': {'mulms': {'n_units': 7, 'minus_control': diff(1.5, 0.2, 2.8),
                                             'loses': False},
                                   'fine': {'n_units': 4, 'minus_control': diff(1.5, 0.5, 2.5, 4),
                                            'loses': False}}}},
        'topic': {'decision': {
            'crf_mrm_topic6': score(57.6, 51.0, 64.0), 'minus_crf_mrm_mixed6': both(1.0, 0.3, 1.7),
            'n_units': 7, 'decision': '14.2', 'passes': True,
            'mixed': {'crf_mrm_mixed6': score(56.6, 50.0, 63.0), 'minus_crf': both(3.0, 1.0, 5.0),
                      'minus_crf_mr': both(1.0, -0.5, 2.5), 'minus_crf_mr6': both(0.5, -0.2, 1.2),
                      'unseen_in_silver': 0.18, 'n_units': 7}}},
        'ablations': {'crf_mrm_topic10k6': {'minus_crf_mrm_topic6': both(-0.5, -1.5, 0.5),
                                            'unseen_in_silver': 0.15, 'n_units': 7},
                      'crf_mrm_topiccap6': {'minus_crf_mrm_topic6': both(0.2, -0.8, 1.2),
                                            'n_units': 7}}}
    (ws.results / 'gpu' / rr.WEEK5_ANALYSIS).write_text(json.dumps(analysis))
    readme, _ = ws.generate()
    bullet = next(line for line in readme.splitlines() if line.startswith('* NER week 14'))
    assert ("14.1, the names against matched copies: pooled over 11 held-out units, the names "
            "method against 6 copies without names (`crf_mr6`) +1.50 [+0.60, +2.40] (MuLMS's 7 "
            "held-out subfields +1.50 [+0.20, +2.80]; MaterioMiner's 4 held-out papers +1.50 "
            "[+0.50, +2.50]), passes (decision 14.1)") in bullet
    assert ("MuLMS's 7 held-out subfields: the names method 57.60 [51.00, 64.00] against the "
            "control's 56.10 [50.00, 62.00], +1.50 [+0.20, +2.80]; the control against the CRF "
            "head +2.50 [+1.00, +4.00] and against mention replacement +0.50 [-0.50, +1.50] "
            "(reported)") in bullet
    assert ("MaterioMiner's 4 held-out papers, fine-grained: the names method 49.00 [44.00, "
            "54.00] against the control's 47.50") in bullet
    assert 'on MuLMS over all 10 seeds +1.20 [-0.10, +2.50] against the control (reported)' in bullet
    assert ("14.2, the topic's role: V4 with the subfield's own corpus against V4 with the mixed "
            "corpus of the six other topics, paired by subfield over seeds 300 to 900: 57.60 "
            "[51.00, 64.00] against 56.60 [50.00, 63.00]; improves (+1.00, 95% CI [+0.30, +1.70]; "
            "strict IOB2 +1.00 [+0.30, +1.70]) (decision 14.2); the mixed variant against the CRF "
            "head +3.00 [+1.00, +5.00] and against mention replacement +1.00 [-0.50, +2.50], "
            "against the copy control +0.50 [-0.20, +1.20] (reported); 18.0% of the unseen test "
            "names are its silver names") in bullet
    assert ("14.3, the ablations at the screen's seeds 0 to 200, against V4 on the same seeds "
            "(reported; no decision): the corpus's first 10,000 sentences -0.50 [-1.50, +0.50] "
            "(15.0% of the unseen names); the silver share capped at 0.5 +0.20 [-0.80, +1.20].")\
        in bullet
    assert bullet.endswith('.')
