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
        '| NER, fine-grained, the published split, 10 seeds (GPU runs) | plain MatSciBERT 70.50 '
        '[69.00, 72.00]; CRF head 73.00 [72.00, 74.00] | published 69.92, within its 95% CI for '
        'plain MatSciBERT (the CI is over seeds only) |',
        '| NER, coarse-grained, the published split, 10 seeds (GPU runs) | plain MatSciBERT 75.00 '
        '[74.00, 76.00]; CRF head 77.00 [76.00, 78.00] | published 72.32, below its 95% CI for '
        'plain MatSciBERT (the CI is over seeds only) |']


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
