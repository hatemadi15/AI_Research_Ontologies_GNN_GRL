"""Week-3 comparisons of 3-seed means (ner_seedmeans.py). Pure Python (no torch or seqeval);
the tests that compute a confidence interval need scipy and skip without it."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import ner_baseline as nb
import ner_seedmeans as ns
import report_results as rr

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'src' / 'graph' / 'ner_seedmeans.py'
SEEDS = [f'seed{k}' for k in range(5)]
PAPERS = ['a.conll', 'b.conll', 'c.conll', 'd.conll']
ENTRY_KEYS = ['variant', 'reference', 'granularity', 'protocol', 'role', 'variant_files',
              'reference_files', 'n_pairs', 'run_ids', 'complete', 'variant_scores',
              'reference_scores', 'micro', 'micro_strict', 'macro', 'weighted']


def _run(run_id, micro, strict=None, macro=None, weighted=None):
    """A run with the fields that F1_AVERAGES reads (macro and weighted as seqeval fractions)."""
    macro = micro - 10 if macro is None else macro
    weighted = micro - 1 if weighted is None else weighted
    return {'run_id': run_id, 'f1': micro, 'f1_strict': micro if strict is None else strict,
            'per_class': {'macro avg': {'f1-score': macro / 100},
                          'weighted avg': {'f1-score': weighted / 100}}}


def _scores(micro):
    return {'micro': micro, 'micro_strict': micro, 'macro': micro - 10, 'weighted': micro - 1}


def _write(directory, granularity, protocol, variant, micro):
    """A results file of `variant` with one run per value of `micro`, in run_id order."""
    ids = SEEDS if protocol == 'random_split_vary' else PAPERS
    name = f'ner_results_{granularity}{nb.result_suffix(protocol, variant)}.json'
    (directory / name).write_text(json.dumps({
        'granularity': granularity, 'protocol': protocol, 'variant': variant,
        'runs': [_run(i, m) for i, m in zip(ids, micro)]}))
    return name


# ---------------------------------------------------------------------------
# The design, fixed in code
# ---------------------------------------------------------------------------
def test_families_and_comparisons_follow_the_week3_protocol():
    assert ns.FAMILIES == {'crf': ('crf_m0', 'crf_m100', 'crf_m200'),
                           'crf_mt': ('crf_mt', 'crf_mt_m100', 'crf_mt_m200'),
                           'crf_ls': ('crf_ls_m0', 'crf_ls_m100', 'crf_ls_m200'),
                           'crf_st': ('crf_st_m0', 'crf_st_m100', 'crf_st_m200'),
                           'crf_dapt2': ('crf_dapt2_m0', 'crf_dapt2_m100', 'crf_dapt2_m200')}
    assert ns.SINGLE == ('crf_ens3', 'crf_mix3', 'crf_scibert', 'crf_pubmedbert')
    # the reporting order of the protocol
    assert [(v, r) for v, r, _ in ns.COMPARISONS] == [
        ('crf_mix3', 'crf_ens3'), ('crf_mix3', 'crf'), ('crf_ens3', 'crf'), ('crf_scibert', 'crf'),
        ('crf_pubmedbert', 'crf'), ('crf_ls', 'crf'), ('crf_st', 'crf'), ('crf_mt', 'crf'),
        ('crf_dapt2', 'crf')]
    assert {(v, r): d for v, r, d in ns.COMPARISONS if d} == {
        ('crf_mix3', 'crf_ens3'): 'random_split_vary', ('crf_mt', 'crf'): 'random_split_vary',
        ('crf_dapt2', 'crf'): 'random_split_vary', ('crf_ls', 'crf'): 'leave_one_paper_out',
        ('crf_st', 'crf'): 'leave_one_paper_out'}
    assert ns.PROTOCOLS == {'random_split_vary': 5, 'leave_one_paper_out': 4}
    # report_results (standard library only) keeps its own copy of the order and the file name
    assert list(rr.WEEK3_COMPARISONS) == [(v, r) for v, r, _ in ns.COMPARISONS]
    assert set(rr.WEEK3_NAMES) == set(ns.SYSTEMS) and rr.WEEK3_FILE == ns.OUT_NAME


def test_file_names_and_keys():
    assert ns.file_names('fine', 'random_split_vary', 'crf') == [
        f'ner_results_fine_random_split_vary_crf_m{n}.json' for n in (0, 100, 200)]
    assert ns.file_names('coarse', 'leave_one_paper_out', 'crf_mt') == [
        'ner_results_coarse_leave_one_paper_out_crf_mt.json',
        'ner_results_coarse_leave_one_paper_out_crf_mt_m100.json',
        'ner_results_coarse_leave_one_paper_out_crf_mt_m200.json']
    assert ns.file_names('fine', 'leave_one_paper_out', 'crf_mix3') == [
        'ner_results_fine_leave_one_paper_out_crf_mix3.json']
    # every file is named as ner_baseline.py and ner_ensemble.py write it
    for system, variants in ns.SYSTEMS.items():
        for protocol in ns.PROTOCOLS:
            assert [nb.parse_result_name(fn) for fn in ns.file_names('coarse', protocol, system)] \
                == [('coarse', protocol, v) for v in variants]
    # keyed like ner_comparisons.json; the one reference other than crf is named
    assert ns.comparison_key('fine', 'random_split_vary', 'crf_mt', 'crf') == \
        'ner_results_fine_random_split_vary_crf_mt'
    assert ns.comparison_key('coarse', 'leave_one_paper_out', 'crf_mix3', 'crf_ens3') == \
        'ner_results_coarse_leave_one_paper_out_crf_mix3_vs_crf_ens3'


# ---------------------------------------------------------------------------
# Scores and paired differences
# ---------------------------------------------------------------------------
def test_a_family_scores_the_mean_of_its_seeds_on_the_run_ids_they_share():
    files = [{'runs': [_run('seed0', 70.0), _run('seed1', 80.0), _run('seed2', 75.0)]},
             {'runs': [_run('seed1', 83.0), _run('seed0', 72.0)]},          # no seed2, other order
             {'runs': [_run('seed0', 74.0, strict=71.0, macro=60.0, weighted=73.0),
                       _run('seed1', 77.0), _run('seed3', 90.0)]}]
    scores = ns.system_scores(files)
    assert list(scores) == ['seed0', 'seed1']            # in all three members, first file's order
    assert scores['seed0'] == pytest.approx({'micro': 72.0, 'micro_strict': 71.0, 'macro': 182 / 3,
                                             'weighted': 71.0})
    assert scores['seed1'] == pytest.approx(_scores(80.0))
    single = ns.system_scores(files[2:])                 # one file: its own runs
    assert list(single) == ['seed0', 'seed1', 'seed3'] and single['seed3'] == pytest.approx(
        _scores(90.0))


def test_compare_pairs_the_scores_by_run_id():
    pytest.importorskip('scipy')
    variant = {'seed1': _scores(80.0), 'seed0': _scores(75.0), 'seed2': _scores(70.0),
               'seed9': _scores(99.0)}                   # no partner: left out
    reference = {'seed0': _scores(74.0), 'seed1': _scores(77.5), 'seed2': _scores(70.5)}
    comp = ns.compare(variant, reference)
    assert list(comp) == ['run_ids', 'variant_scores', 'reference_scores', 'micro', 'micro_strict',
                          'macro', 'weighted']
    assert comp['run_ids'] == ['seed1', 'seed0', 'seed2']            # the variant's order
    # differences 2.5, 1.0, -0.5: sample SD 1.5, t(0.975, 2) = 4.303, half-width 3.73
    for name in ('micro', 'micro_strict', 'macro', 'weighted'):
        assert comp[name] == {'mean_diff': 1.0, 'ci95': [-2.73, 4.73], 'per_run': [2.5, 1.0, -0.5]}
    assert comp['variant_scores']['micro'] == {'mean': 75.0, 'ci95': [62.58, 87.42],
                                               'per_run': [80.0, 75.0, 70.0]}
    assert comp['reference_scores']['macro']['per_run'] == [67.5, 64.0, 60.5]


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_week3_comparisons_from_results_files(tmp_path, capsys):
    pytest.importorskip('scipy')
    vary, lopo = 'random_split_vary', 'leave_one_paper_out'
    for n, offset in ((0, 0), (100, 1), (200, 2)):           # the CRF reference: means 71 ... 79
        _write(tmp_path, 'fine', vary, f'crf_m{n}', [70 + offset + 2 * k for k in range(5)])
        _write(tmp_path, 'fine', lopo, f'crf_m{n}', [50 + offset + 2 * k for k in range(4)])
        _write(tmp_path, 'fine', lopo, f'crf_ls_m{n}', [53 + 2 * k for k in range(4)])
    _write(tmp_path, 'fine', vary, 'crf_ens3', [73, 75, 77, 79, 81])
    _write(tmp_path, 'fine', vary, 'crf_mix3', [74, 75, 78, 79, 82])
    _write(tmp_path, 'fine', vary, 'crf_scibert', [69, 70, 71, 72, 73])
    _write(tmp_path, 'fine', vary, 'crf_mt', [72, 74, 76, 78, 80])
    _write(tmp_path, 'fine', vary, 'crf_mt_m100', [72, 74, 76, 78, 80])
    _write(tmp_path, 'fine', vary, 'crf_mt_m200', [75, 77, 79])               # seeds 0-2 so far
    _write(tmp_path, 'fine', vary, 'crf_dapt2_m0', [80] * 5)                  # m200 missing
    _write(tmp_path, 'fine', vary, 'crf_dapt2_m100', [80] * 5)

    comps = ns.main(['--results', str(tmp_path)])

    assert json.loads((tmp_path / 'ner_week3_comparisons.json').read_text()) == comps
    key = 'ner_results_fine_random_split_vary_'
    assert list(comps) == [key + 'crf_mix3_vs_crf_ens3', key + 'crf_mix3', key + 'crf_ens3',
                           key + 'crf_scibert', 'ner_results_fine_leave_one_paper_out_crf_ls',
                           key + 'crf_mt']
    assert all(list(c) == ENTRY_KEYS for c in comps.values())
    summary = {k[len('ner_results_fine_'):]: (c['variant'], c['reference'], c['role'],
                                             c['n_pairs'], c['complete'])
               for k, c in comps.items()}
    assert summary == {
        'random_split_vary_crf_mix3_vs_crf_ens3': ('crf_mix3', 'crf_ens3', 'decision', 5, True),
        'random_split_vary_crf_mix3': ('crf_mix3', 'crf', 'reported', 5, True),
        'random_split_vary_crf_ens3': ('crf_ens3', 'crf', 'reported', 5, True),
        'random_split_vary_crf_scibert': ('crf_scibert', 'crf', 'reported', 5, True),
        'leave_one_paper_out_crf_ls': ('crf_ls', 'crf', 'decision', 4, True),
        'random_split_vary_crf_mt': ('crf_mt', 'crf', 'decision', 3, False)}
    ens = comps[key + 'crf_ens3']
    assert ens['variant_files'] == [key + 'crf_ens3.json']
    assert ens['reference_files'] == [f'{key}crf_m{n}.json' for n in (0, 100, 200)]
    assert ens['reference_scores']['micro']['per_run'] == [71.0, 73.0, 75.0, 77.0, 79.0]
    assert ens['micro'] == {'mean_diff': 2.0, 'ci95': [2.0, 2.0], 'per_run': [2.0] * 5}
    assert comps[key + 'crf_mix3_vs_crf_ens3']['micro']['per_run'] == [1.0, 0.0, 1.0, 0.0, 1.0]
    assert comps[key + 'crf_mix3_vs_crf_ens3']['reference_files'] == [key + 'crf_ens3.json']
    assert comps['ner_results_fine_leave_one_paper_out_crf_ls']['micro']['per_run'] == [2.0] * 4
    mt = comps[key + 'crf_mt']
    assert mt['run_ids'] == ['seed0', 'seed1', 'seed2']       # a run_id needs all three seeds
    assert mt['variant_scores']['micro']['per_run'] == [73.0, 75.0, 77.0]
    printed = capsys.readouterr().out
    assert f'{key}crf_dapt2: no {key}crf_dapt2_m200.json, skipped' in printed
    assert f'{key}crf_pubmedbert: no {key}crf_pubmedbert.json, skipped' in printed
    assert f'{key}crf_mt: micro F1 difference +2.00 vs crf' in printed and 'incomplete' in printed
    assert 'Saved' in printed and '6 comparisons, 5 complete' in printed


def test_out_option_and_errors(tmp_path, capsys):
    pytest.importorskip('scipy')
    for n in (0, 100, 200):
        _write(tmp_path, 'coarse', 'leave_one_paper_out', f'crf_m{n}', [60, 61, 62, 63])
    _write(tmp_path, 'coarse', 'leave_one_paper_out', 'crf_ens3', [62, 64, 63, 65])
    out = tmp_path / 'other' / 'w3.json'
    out.parent.mkdir()
    comps = ns.main(['--results', str(tmp_path), '--out', str(out)])
    assert list(comps) == ['ner_results_coarse_leave_one_paper_out_crf_ens3']
    assert json.loads(out.read_text()) == comps
    assert not (tmp_path / ns.OUT_NAME).exists()
    capsys.readouterr()
    with pytest.raises(SystemExit) as excinfo:
        ns.main(['--results', str(tmp_path / 'missing')])
    assert excinfo.value.code == 2 and 'results directory not found' in capsys.readouterr().err


def test_no_shared_run_id_skips_the_comparison(tmp_path, capsys):
    for n in (0, 100, 200):
        _write(tmp_path, 'fine', 'random_split_vary', f'crf_m{n}', [70] * 5)
    (tmp_path / 'ner_results_fine_random_split_vary_crf_ens3.json').write_text(json.dumps(
        {'runs': [_run('seed7', 80.0)]}))
    assert ns.week3_comparisons(str(tmp_path)) == {}
    assert 'crf_ens3: no run_id with scores on both sides, skipped' in capsys.readouterr().out


def test_script_runs_standalone(tmp_path):
    proc = subprocess.run([sys.executable, str(SCRIPT), '--results', str(tmp_path)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert json.loads((tmp_path / ns.OUT_NAME).read_text()) == {}


# ---------------------------------------------------------------------------
# The committed results
# ---------------------------------------------------------------------------
def test_seed_ensemble_vs_crf_is_the_vote_against_its_own_members():
    """The 3-seed CRF reference is the ensemble's own members, so crf_ens3 vs crf equals
    summary.vs_member_mean of the crf_ens3 files (up to rounding), and the variant's scores
    are the ensemble's summary."""
    pytest.importorskip('scipy')
    comps = ns.week3_comparisons(str(REPO / 'results'))
    found = {k: c for k, c in comps.items()
             if (c['variant'], c['reference']) == ('crf_ens3', 'crf')}
    assert sorted((c['granularity'], c['protocol'], c['complete']) for c in found.values()) == [
        ('coarse', 'leave_one_paper_out', True), ('coarse', 'random_split_vary', True),
        ('fine', 'leave_one_paper_out', True), ('fine', 'random_split_vary', True)]
    for key, c in found.items():
        ensemble = json.loads((REPO / 'results' / f'{key}.json').read_text())['summary']
        gain = ensemble['vs_member_mean']
        assert c['micro']['mean_diff'] == pytest.approx(gain['mean_diff'], abs=0.01)
        assert c['micro']['ci95'] == pytest.approx(gain['ci95'], abs=0.01)
        assert c['micro']['per_run'] == pytest.approx(gain['per_run'], abs=0.01)
        assert c['variant_scores']['micro'] == {k: ensemble['micro'][k]
                                                for k in ('mean', 'ci95', 'per_run')}
        assert c['role'] == 'reported' and c['run_ids'] == ensemble['run_ids']


def test_committed_week3_comparisons_are_up_to_date(tmp_path):
    """results/ner_week3_comparisons.json is what the script makes from results/.

    If this fails after new runs, run `python src/graph/ner_seedmeans.py`, then
    `python src/graph/report_results.py`, and commit the results.
    """
    pytest.importorskip('scipy')
    out = tmp_path / ns.OUT_NAME
    ns.main(['--results', str(REPO / 'results'), '--out', str(out)])
    assert out.read_text() == (REPO / 'results' / ns.OUT_NAME).read_text()
