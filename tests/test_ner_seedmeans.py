"""Week-3 and week-4 comparisons of 3-seed means (ner_seedmeans.py). Pure Python (no torch or
seqeval); the tests that compute a confidence interval need scipy and skip without it."""
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
    week3 = {**ns.FAMILIES, **{name: (name,) for name in ns.SINGLE}}
    assert set(rr.WEEK3_NAMES) == set(week3) and rr.WEEK3_FILE == ns.OUT_NAME
    assert ns.comparison_list(3) is ns.COMPARISONS


def test_week4_systems_and_comparisons_follow_the_week4_protocol():
    assert ns.WEEK4_FAMILIES == {'crf_mt_ls': ('crf_mt_ls_m0', 'crf_mt_ls_m100', 'crf_mt_ls_m200'),
                                 'crf_st3': ('crf_st3_m0', 'crf_st3_m100', 'crf_st3_m200')}
    assert ns.WEEK4_SINGLE == ('crf_ls_ens3', 'crf_st_ens3', 'crf_mt_ens3')
    assert set(ns.SYSTEMS) == {*ns.FAMILIES, *ns.WEEK4_FAMILIES, *ns.SINGLE, *ns.WEEK4_SINGLE}
    # CPU: the label-semantics and self-training votes, decided on the held-out papers
    assert ns.comparison_list(4) == (
        ('crf_ls_ens3', 'crf_ens3', 'leave_one_paper_out'),
        ('crf_st_ens3', 'crf_ens3', 'leave_one_paper_out'),
        ('crf_ls_ens3', 'crf', None), ('crf_st_ens3', 'crf', None))
    # GPU: the decisions of items 1-3 first, then the reported comparisons
    assert ns.comparison_list(4, gpu=True) == (
        ('crf_mt_ens3', 'crf_ens3', 'random_split_vary'),
        ('crf_mt_ls', 'crf', ('random_split_vary', 'leave_one_paper_out')),
        ('crf_st3', 'crf', 'leave_one_paper_out'),
        ('crf_ls_ens3', 'crf_ens3', None), ('crf_st_ens3', 'crf_ens3', None),
        ('crf_ens3', 'crf', None), ('crf_mt_ens3', 'crf', None), ('crf_mt', 'crf', None),
        ('crf_ls', 'crf', None), ('crf_st', 'crf', None), ('crf_mt_ls', 'crf_mt', None),
        ('crf_mt_ls', 'crf_ls', None), ('crf_st3', 'crf_st', None), ('crf_st3', 'crf_ens3', None))
    with pytest.raises(ValueError):
        ns.comparison_list(3, gpu=True)
    assert ns.OUT_NAMES == {3: 'ner_week3_comparisons.json', 4: 'ner_week4_comparisons.json',
                            5: 'ner_week5_comparisons.json', 6: 'ner_week6_comparisons.json',
                            7: 'ner_week7_comparisons.json', 9: 'ner_week9_comparisons.json'}


def test_week5_systems_and_comparisons_follow_the_week5_protocol():
    """10 seeds per split (offsets 0 to 900), votes over 3 disjoint seed triples."""
    offsets = range(0, 1000, 100)
    assert ns.WEEK5_SYSTEMS['crf'] == tuple(f'crf_m{n}' for n in offsets)
    assert ns.WEEK5_SYSTEMS['crf_mt'] == ('crf_mt',) + tuple(f'crf_mt_m{n}' for n in offsets[1:])
    assert ns.WEEK5_SYSTEMS['crf_ls'] == tuple(f'crf_ls_m{n}' for n in offsets)
    assert ns.WEEK5_SYSTEMS['crf_st3'] == tuple(f'crf_st3_m{n}' for n in offsets)
    for stem in ('crf', 'crf_mt', 'crf_ls', 'crf_st3'):
        assert ns.WEEK5_SYSTEMS[f'{stem}_ens3'] == (
            f'{stem}_ens3', f'{stem}_ens3_m300', f'{stem}_ens3_m600')
    assert len(ns.WEEK5_SYSTEMS) == 8
    both = ('random_split_vary', 'leave_one_paper_out')
    assert ns.comparison_list(5, gpu=True) == ns.WEEK5_COMPARISONS == (
        ('crf_mt', 'crf', 'random_split_vary'),
        ('crf_ls', 'crf', 'leave_one_paper_out'),
        ('crf_st3', 'crf', 'leave_one_paper_out'),
        ('crf_mt_ens3', 'crf_ens3', both), ('crf_ls_ens3', 'crf_ens3', both),
        ('crf_st3_ens3', 'crf_ens3', both),
        ('crf_ens3', 'crf', None))
    with pytest.raises(ValueError):
        ns.comparison_list(5)
    assert ns.system_list(5) is ns.WEEK5_SYSTEMS and ns.system_list(4) is ns.SYSTEMS
    # every file is named as ner_baseline.py and ner_ensemble.py write it
    for system, variants in ns.WEEK5_SYSTEMS.items():
        for protocol in ns.PROTOCOLS:
            names = ns.file_names('fine', protocol, system, ns.WEEK5_SYSTEMS)
            assert [nb.parse_result_name(fn) for fn in names] == \
                [('fine', protocol, v) for v in variants]


def test_week5_comparisons_from_results_files(tmp_path, capsys):
    """--week 5 --gpu: 10-seed means and 3-vote means, written to the week-5 file."""
    pytest.importorskip('scipy')
    base = {'random_split_vary': [70, 71, 72, 73, 74], 'leave_one_paper_out': [50, 51, 52, 53]}
    for protocol, scores in base.items():
        for n in range(0, 1000, 100):
            _write(tmp_path, 'fine', protocol, f'crf_m{n}', scores)
            # self-training: +1 on average, the seeds spread around it
            _write(tmp_path, 'fine', protocol, f'crf_st3_m{n}',
                   [s + 1 + (n - 450) / 1000 for s in scores])
        for stem, gain in (('crf', 0.0), ('crf_st3', 0.5)):
            for name in ns.WEEK5_SYSTEMS[f'{stem}_ens3']:
                _write(tmp_path, 'fine', protocol, name, [s + 2 + gain for s in scores])
    comps = ns.main(['--results', str(tmp_path), '--week', '5', '--gpu'])
    key = 'ner_results_fine_{}_{}'
    assert {k: c['role'] for k, c in comps.items()} == {
        key.format('random_split_vary', 'crf_st3'): 'reported',
        key.format('leave_one_paper_out', 'crf_st3'): 'decision',
        key.format('random_split_vary', 'crf_st3_ens3_vs_crf_ens3'): 'decision',
        key.format('leave_one_paper_out', 'crf_st3_ens3_vs_crf_ens3'): 'decision',
        key.format('random_split_vary', 'crf_ens3'): 'reported',
        key.format('leave_one_paper_out', 'crf_ens3'): 'reported'}
    st3 = comps[key.format('leave_one_paper_out', 'crf_st3')]
    assert len(st3['variant_files']) == len(st3['reference_files']) == 10
    assert st3['micro']['mean_diff'] == 1.0 and st3['complete'] and st3['n_pairs'] == 4
    vote = comps[key.format('random_split_vary', 'crf_st3_ens3_vs_crf_ens3')]
    assert vote['variant_files'] == [
        f'ner_results_fine_random_split_vary_crf_st3_ens3{t}.json' for t in ('', '_m300', '_m600')]
    assert vote['micro']['mean_diff'] == 0.5 and vote['n_pairs'] == 5
    assert comps[key.format('random_split_vary', 'crf_ens3')]['micro']['mean_diff'] == 2.0
    assert json.loads((tmp_path / 'ner_week5_comparisons.json').read_text()) == comps
    # a family is compared only with all 10 seeds: without one, it is skipped
    (tmp_path / 'ner_results_fine_leave_one_paper_out_crf_st3_m900.json').unlink()
    comps = ns.main(['--results', str(tmp_path), '--week', '5', '--gpu'])
    assert key.format('leave_one_paper_out', 'crf_st3') not in comps
    assert 'crf_st3_m900.json, skipped' in capsys.readouterr().out
    with pytest.raises(SystemExit) as excinfo:
        ns.main(['--results', str(tmp_path), '--week', '5'])
    assert excinfo.value.code == 2 and '--week 5 needs --gpu' in capsys.readouterr().err


def test_week6_systems_and_comparisons_follow_the_week6_protocol():
    """Stage 1: 3-seed families and first votes; stage 2: 10 seeds, and each method's vote
    decided against the strongest system on unseen papers of its granularity."""
    lopo = 'leave_one_paper_out'
    assert ns.WEEK6_METHODS == ('crf_deberta', 'crf_ctx', 'crf_mt_st3')
    screen = ns.system_list(6, screen=True)
    assert screen['crf_deberta'] == ('crf_deberta_m0', 'crf_deberta_m100', 'crf_deberta_m200')
    assert screen['crf_mt'] == ('crf_mt', 'crf_mt_m100', 'crf_mt_m200')
    assert screen['crf_ctx_ens3'] == ('crf_ctx_ens3',)
    assert ns.comparison_list(6, gpu=True, screen=True) == (
        ('crf_deberta', 'crf', lopo), ('crf_ctx', 'crf', lopo), ('crf_mt_st3', 'crf', lopo),
        ('crf_mt_st3', 'crf_mt', None), ('crf_deberta_ens3', 'crf_ens3', None),
        ('crf_ctx_ens3', 'crf_ens3', None), ('crf_mt_st3_ens3', 'crf_ens3', None))
    stage2 = ns.system_list(6)
    assert stage2['crf_mt_st3'] == tuple(f'crf_mt_st3_m{n}' for n in range(0, 1000, 100))
    assert stage2['crf_ctx_ens3'] == ('crf_ctx_ens3', 'crf_ctx_ens3_m300', 'crf_ctx_ens3_m600')
    assert stage2['crf_mt_ens3'] == ns.WEEK5_SYSTEMS['crf_mt_ens3']
    assert ns.UNSEEN_BEST == {'fine': 'crf_mt_ens3', 'coarse': 'crf_ens3'}
    assert ns.comparison_list(6, gpu=True)[3:5] == (
        ('crf_deberta_ens3', 'crf_mt_ens3', {'fine': lopo}),
        ('crf_deberta_ens3', 'crf_ens3', {'coarse': lopo}))
    with pytest.raises(ValueError):
        ns.comparison_list(6)
    assert ns.decides_on({'fine': lopo}, lopo, 'fine')
    assert not ns.decides_on({'fine': lopo}, lopo, 'coarse')
    assert not ns.decides_on({'fine': lopo}, 'random_split_vary', 'fine')


def test_week6_screen_from_results_files(tmp_path, capsys):
    """--week 6 --gpu --screen: the 3-seed screen on the held-out papers, its verdicts."""
    pytest.importorskip('scipy')
    lopo = 'leave_one_paper_out'
    for g, deberta, ctx in (('fine', +1.0, -0.5), ('coarse', -0.2, -0.3)):
        for n in (0, 100, 200):
            _write(tmp_path, g, lopo, f'crf_m{n}', [50, 51, 52, 53])
            _write(tmp_path, g, lopo, f'crf_deberta_m{n}', [x + deberta for x in (50, 51, 52, 53)])
            _write(tmp_path, g, lopo, f'crf_ctx_m{n}', [x + ctx for x in (50, 51, 52, 53)])
        for name in ('crf_ens3', 'crf_deberta_ens3'):
            _write(tmp_path, g, lopo, name, [52, 53, 54, 55])
    comps = ns.main(['--results', str(tmp_path), '--week', '6', '--gpu', '--screen'])
    key = 'ner_results_fine_leave_one_paper_out_{}'
    assert comps[key.format('crf_deberta')]['role'] == 'screen'
    assert comps[key.format('crf_deberta')]['micro']['mean_diff'] == 1.0
    assert comps[key.format('crf_deberta_ens3_vs_crf_ens3')]['role'] == 'reported'
    out = capsys.readouterr().out
    assert 'screen: crf_deberta goes to stage 2 (fine +1.00, coarse -0.20)' in out
    assert 'screen: crf_ctx stops (fine -0.50, coarse -0.30)' in out
    assert 'crf_mt_st3' not in ns.screen_verdicts(comps)              # no files at all
    assert (tmp_path / ns.SCREEN_NAME).exists()
    assert not (tmp_path / 'ner_week6_comparisons.json').exists()
    with pytest.raises(SystemExit):
        ns.main(['--results', str(tmp_path), '--week', '5', '--gpu', '--screen'])


def test_week6_decisions_from_results_files(tmp_path):
    """--week 6 --gpu: a method's vote is decided against crf_mt_ens3 fine-grained and
    crf_ens3 coarse-grained; the other pairing is reported."""
    pytest.importorskip('scipy')
    lopo = 'leave_one_paper_out'
    for g in ('fine', 'coarse'):
        for n in range(0, 1000, 100):
            _write(tmp_path, g, lopo, f'crf_m{n}', [50, 51, 52, 53])
            _write(tmp_path, g, lopo, f'crf_ctx_m{n}', [51, 52, 53, 54])
        for stem, plus in (('crf', 2), ('crf_mt', 3), ('crf_ctx', 4)):
            for name in ns.vote_family(stem):
                _write(tmp_path, g, lopo, name, [x + plus for x in (50, 51, 52, 53)])
    comps = ns.main(['--results', str(tmp_path), '--week', '6', '--gpu'])
    key = 'ner_results_{}_leave_one_paper_out_{}'
    roles = {k: c['role'] for k, c in comps.items()}
    assert roles == {
        key.format('fine', 'crf_ctx'): 'decision', key.format('coarse', 'crf_ctx'): 'decision',
        key.format('fine', 'crf_ctx_ens3_vs_crf_mt_ens3'): 'decision',
        key.format('coarse', 'crf_ctx_ens3_vs_crf_mt_ens3'): 'reported',
        key.format('fine', 'crf_ctx_ens3_vs_crf_ens3'): 'reported',
        key.format('coarse', 'crf_ctx_ens3_vs_crf_ens3'): 'decision'}
    assert comps[key.format('fine', 'crf_ctx_ens3_vs_crf_mt_ens3')]['micro']['mean_diff'] == 1.0
    assert len(comps[key.format('fine', 'crf_ctx')]['variant_files']) == 10
    assert (tmp_path / 'ner_week6_comparisons.json').exists()


def test_week7_systems_and_comparisons_follow_the_week7_protocol():
    """Part 1: the consistency rule on the CRF head and on the strongest vote of each
    granularity (decisions on the held-out papers); retyping and context reported."""
    lopo = 'leave_one_paper_out'
    systems = ns.system_list(7)
    assert systems['crf_dc'] == tuple(f'crf_dc_m{n}' for n in range(0, 1000, 100))
    assert systems['crf_mt_ens3_dc'] == ('crf_mt_ens3_dc', 'crf_mt_ens3_dc_m300',
                                         'crf_mt_ens3_dc_m600')
    assert systems['crf_ens3_dcr'] == ('crf_ens3_dcr', 'crf_ens3_dcr_m300', 'crf_ens3_dcr_m600')
    assert systems['crf_ctx_dc'] == tuple(f'crf_ctx_dc_m{n}' for n in range(0, 1000, 100))
    assert systems['crf_mt_ens3'] == ns.WEEK5_SYSTEMS['crf_mt_ens3']
    comps = ns.comparison_list(7, gpu=True)
    assert comps[:3] == (
        ('crf_dc', 'crf', lopo), ('crf_mt_ens3_dc', 'crf_mt_ens3', {'fine': lopo}),
        ('crf_ens3_dc', 'crf_ens3', {'coarse': lopo}))
    # part 2a: recall calibration, the bias chosen on inner folds; on validation reported
    assert systems['crf_eb'] == tuple(f'crf_eb_m{n}' for n in range(0, 1000, 100))
    assert systems['crf_eb_ens3'] == ('crf_eb_ens3', 'crf_eb_ens3_m300', 'crf_eb_ens3_m600')
    part2a = comps.index(('crf_eb', 'crf', lopo))
    assert comps[part2a:part2a + 4] == (
        ('crf_eb', 'crf', lopo), ('crf_eb_ens3', 'crf_mt_ens3', {'fine': lopo}),
        ('crf_eb_ens3', 'crf_ens3', {'coarse': lopo}), ('crf_ebv', 'crf', None))
    assert all(decides is None for _, _, decides in comps[3:part2a])
    # part 2c: coarse-to-fine completion, fine-grained decisions on the held-out papers
    assert systems['crf_cf'] == tuple(f'crf_cf_m{n}' for n in range(0, 1000, 100))
    assert systems['crf_mt_cf'][:2] == ('crf_mt_cf', 'crf_mt_cf_m100')
    assert systems['crf_mt_ens3_cf'] == ('crf_mt_ens3_cf', 'crf_mt_ens3_cf_m300',
                                         'crf_mt_ens3_cf_m600')
    assert comps[part2a + 4:part2a + 7] == (
        ('crf_cf', 'crf', {'fine': lopo}), ('crf_mt_ens3_cf', 'crf_mt_ens3', {'fine': lopo}),
        ('crf_mt_cf', 'crf_mt', None))
    # part 2b, stage 2: mention replacement decided as the week-6 methods were
    assert systems['crf_mr'] == tuple(f'crf_mr_m{n}' for n in range(0, 1000, 100))
    assert systems['crf_mr_ens3'] == ('crf_mr_ens3', 'crf_mr_ens3_m300', 'crf_mr_ens3_m600')
    assert comps[part2a + 7:] == (
        ('crf_mr', 'crf', lopo), ('crf_mr_ens3', 'crf_mt_ens3', {'fine': lopo}),
        ('crf_mr_ens3', 'crf_ens3', {'coarse': lopo}))
    # its screen: seeds 0-200 against the CRF head; the first votes reported
    screen = ns.system_list(7, screen=True)
    assert screen['crf_mr'] == ('crf_mr_m0', 'crf_mr_m100', 'crf_mr_m200')
    assert screen['crf_mr_ens3'] == ('crf_mr_ens3',) and screen['crf_ens3'] == ('crf_ens3',)
    assert ns.comparison_list(7, gpu=True, screen=True) == (
        ('crf_mr', 'crf', lopo), ('crf_mr_ens3', 'crf_ens3', None))
    with pytest.raises(ValueError):
        ns.comparison_list(7)


def test_week7_screen_from_results_files(tmp_path, capsys):
    """--week 7 --gpu --screen: mention replacement's 3-seed screen, its own file."""
    pytest.importorskip('scipy')
    lopo = 'leave_one_paper_out'
    for g, gain in (('fine', -0.5), ('coarse', 0.25)):
        for n in (0, 100, 200):
            _write(tmp_path, g, lopo, f'crf_m{n}', [50, 51, 52, 53])
            _write(tmp_path, g, lopo, f'crf_mr_m{n}', [50 + gain, 51 + gain, 52 + gain, 53 + gain])
    comps = ns.main(['--results', str(tmp_path), '--week', '7', '--gpu', '--screen'])
    assert comps[ns.comparison_key('fine', lopo, 'crf_mr', 'crf')]['role'] == 'screen'
    assert (tmp_path / 'ner_week7_screen.json').exists()
    assert not (tmp_path / 'ner_week6_screen.json').exists()
    assert 'screen: crf_mr goes to stage 2 (fine -0.50, coarse +0.25)' in capsys.readouterr().out


def test_week7_decisions_from_results_files(tmp_path):
    """--week 7 --gpu: the rule's decisions, and the random splits reported."""
    pytest.importorskip('scipy')
    lopo, rand = 'leave_one_paper_out', 'random_split_vary'
    for g in ('fine', 'coarse'):
        for n in range(0, 1000, 100):
            _write(tmp_path, g, lopo, f'crf_m{n}', [50, 51, 52, 53])
            _write(tmp_path, g, lopo, f'crf_dc_m{n}', [51, 52, 53, 55])
            _write(tmp_path, g, rand, f'crf_m{n}', [70, 71, 72, 73, 74])
            _write(tmp_path, g, rand, f'crf_dc_m{n}', [70, 71, 72, 73, 75])
        for stem, plus in (('crf_ens3', 2), ('crf_mt_ens3', 3), ('crf_ens3_dc', 3),
                           ('crf_mt_ens3_dc', 3.5)):
            for name in ns.system_list(7)[stem]:
                _write(tmp_path, g, lopo, name, [x + plus for x in (50, 51, 52, 53)])
    comps = ns.main(['--results', str(tmp_path), '--week', '7', '--gpu'])
    key = 'ner_results_{}_{}_{}'
    roles = {k: c['role'] for k, c in comps.items()}
    assert roles == {
        key.format('fine', lopo, 'crf_dc'): 'decision',
        key.format('coarse', lopo, 'crf_dc'): 'decision',
        key.format('fine', rand, 'crf_dc'): 'reported',
        key.format('coarse', rand, 'crf_dc'): 'reported',
        key.format('fine', lopo, 'crf_mt_ens3_dc_vs_crf_mt_ens3'): 'decision',
        key.format('coarse', lopo, 'crf_mt_ens3_dc_vs_crf_mt_ens3'): 'reported',
        key.format('fine', lopo, 'crf_ens3_dc_vs_crf_ens3'): 'reported',
        key.format('coarse', lopo, 'crf_ens3_dc_vs_crf_ens3'): 'decision'}
    assert comps[key.format('fine', lopo, 'crf_dc')]['micro']['mean_diff'] == 1.25
    assert comps[key.format('fine', lopo, 'crf_mt_ens3_dc_vs_crf_mt_ens3')]['micro'][
        'mean_diff'] == 0.5
    assert (tmp_path / 'ner_week7_comparisons.json').exists()


def test_week9_mention_replacement_on_the_multi_task_model(tmp_path):
    """Week 9: crf_mt_mr against the multi-task model and its vote against the strongest
    system on unseen papers decide; against the CRF head and mention replacement alone
    reported. Its own file."""
    pytest.importorskip('scipy')
    lopo = 'leave_one_paper_out'
    systems = ns.system_list(9)
    assert systems['crf_mt_mr'] == tuple(f'crf_mt_mr_m{n}' for n in range(0, 1000, 100))
    assert systems['crf_mt_mr_ens3'] == ('crf_mt_mr_ens3', 'crf_mt_mr_ens3_m300',
                                         'crf_mt_mr_ens3_m600')
    assert systems['crf_mt'] == ns.WEEK5_SYSTEMS['crf_mt']
    assert ns.comparison_list(9, gpu=True) == (
        ('crf_mt_mr', 'crf_mt', lopo), ('crf_mt_mr_ens3', 'crf_mt_ens3', {'fine': lopo}),
        ('crf_mt_mr_ens3', 'crf_ens3', {'coarse': lopo}), ('crf_mt_mr', 'crf', None),
        ('crf_mt_mr', 'crf_mr', None))
    with pytest.raises(ValueError):
        ns.comparison_list(9)
    for g in ('fine', 'coarse'):
        for stem, plus in (('crf', 0), ('crf_mt', 1), ('crf_mr', 1.5), ('crf_mt_mr', 2.5)):
            for name in systems[stem]:
                _write(tmp_path, g, lopo, name, [x + plus for x in (50, 51, 52, 53)])
        for stem, plus in (('crf_ens3', 2), ('crf_mt_ens3', 3), ('crf_mt_mr_ens3', 4)):
            for name in systems[stem]:
                _write(tmp_path, g, lopo, name, [x + plus for x in (50, 51, 52, 53)])
    comps = ns.main(['--results', str(tmp_path), '--week', '9', '--gpu'])
    key = 'ner_results_{}_{}_{}'
    roles = {k: c['role'] for k, c in comps.items()}
    assert roles == {
        key.format('fine', lopo, 'crf_mt_mr_vs_crf_mt'): 'decision',
        key.format('coarse', lopo, 'crf_mt_mr_vs_crf_mt'): 'decision',
        key.format('fine', lopo, 'crf_mt_mr_ens3_vs_crf_mt_ens3'): 'decision',
        key.format('coarse', lopo, 'crf_mt_mr_ens3_vs_crf_mt_ens3'): 'reported',
        key.format('fine', lopo, 'crf_mt_mr_ens3_vs_crf_ens3'): 'reported',
        key.format('coarse', lopo, 'crf_mt_mr_ens3_vs_crf_ens3'): 'decision',
        key.format('fine', lopo, 'crf_mt_mr'): 'reported',
        key.format('coarse', lopo, 'crf_mt_mr'): 'reported',
        key.format('fine', lopo, 'crf_mt_mr_vs_crf_mr'): 'reported',
        key.format('coarse', lopo, 'crf_mt_mr_vs_crf_mr'): 'reported'}
    assert comps[key.format('fine', lopo, 'crf_mt_mr_vs_crf_mt')]['micro']['mean_diff'] == 1.5
    assert comps[key.format('fine', lopo, 'crf_mt_mr_ens3_vs_crf_mt_ens3')]['micro'][
        'mean_diff'] == 1.0
    assert (tmp_path / 'ner_week9_comparisons.json').exists()


def test_decides_on_one_protocol_or_several():
    assert ns.decides_on('random_split_vary', 'random_split_vary')
    assert not ns.decides_on('random_split_vary', 'leave_one_paper_out')
    both = ('random_split_vary', 'leave_one_paper_out')
    assert ns.decides_on(both, 'random_split_vary') and ns.decides_on(both, 'leave_one_paper_out')
    assert not ns.decides_on(None, 'random_split_vary')


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


def test_week4_gpu_comparisons_decide_on_both_protocols(tmp_path, capsys):
    """--week 4 --gpu: crf_mt_ls is decided on both protocols, written to the week-4 file."""
    pytest.importorskip('scipy')
    for protocol, base in (('random_split_vary', [70, 71, 72, 73, 74]),
                           ('leave_one_paper_out', [50, 51, 52, 53])):
        for n in (0, 100, 200):
            _write(tmp_path, 'coarse', protocol, f'crf_m{n}', base)
            _write(tmp_path, 'coarse', protocol, f'crf_mt_ls_m{n}', [b + 1 + n / 100 for b in base])
    comps = ns.main(['--results', str(tmp_path), '--week', '4', '--gpu'])
    assert {k: c['role'] for k, c in comps.items()} == {
        'ner_results_coarse_random_split_vary_crf_mt_ls': 'decision',
        'ner_results_coarse_leave_one_paper_out_crf_mt_ls': 'decision'}
    for c in comps.values():
        assert c['micro']['mean_diff'] == 2.0 and c['complete']
    assert json.loads((tmp_path / 'ner_week4_comparisons.json').read_text()) == comps
    assert not (tmp_path / ns.OUT_NAME).exists()
    capsys.readouterr()
    with pytest.raises(SystemExit) as excinfo:
        ns.main(['--results', str(tmp_path), '--gpu'])
    assert excinfo.value.code == 2 and '--gpu needs --week 4' in capsys.readouterr().err


def test_gpu_minus_cpu_is_descriptive(tmp_path):
    """--gpu adds, for systems in both directories, the GPU minus the CPU scores per split."""
    pytest.importorskip('scipy')
    gpu = tmp_path / 'gpu'
    gpu.mkdir()
    for n in (0, 100, 200):
        _write(tmp_path, 'fine', 'leave_one_paper_out', f'crf_m{n}', [50, 51, 52, 53])
        _write(gpu, 'fine', 'leave_one_paper_out', f'crf_m{n}', [51, 52, 53, 54])
    comps = ns.main(['--results', str(gpu), '--week', '4', '--gpu'])
    assert list(comps) == ['ner_results_fine_leave_one_paper_out_crf_gpu_vs_cpu']
    c = comps['ner_results_fine_leave_one_paper_out_crf_gpu_vs_cpu']
    assert (c['role'], c['hardware'], c['complete']) == (
        'descriptive', {'variant': 'gpu', 'reference': 'cpu'}, True)
    assert c['micro']['mean_diff'] == 1.0 and c['variant_files'] == c['reference_files']
    # another CPU directory, or none with these files: no descriptive entry
    assert ns.main(['--results', str(gpu), '--week', '4', '--gpu', '--cpu', str(gpu / 'x')]) == {}


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


def test_committed_week4_comparisons_are_up_to_date(tmp_path):
    """results/ner_week4_comparisons.json is what `--week 4` makes from results/ (the CPU
    votes); results/gpu/ner_week4_comparisons.json, once there, from results/gpu/."""
    pytest.importorskip('scipy')
    for directory, extra in ((REPO / 'results', []), (REPO / 'results' / 'gpu', ['--gpu'])):
        committed = directory / ns.OUT_NAMES[4]
        if not committed.exists():
            continue
        out = tmp_path / f"{directory.name}_{ns.OUT_NAMES[4]}"
        ns.main(['--results', str(directory), '--week', '4', *extra, '--out', str(out)])
        assert out.read_text() == committed.read_text()
