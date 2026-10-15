"""Exploratory analyses of the week-5 10-seed runs (ner_seed_analysis.py). Pure Python; the
tests that compute a confidence interval need scipy and skip without it."""
import json

import pytest

import ner_seed_analysis as sa
import ner_seedmeans as ns

SEEDS = [f'seed{k}' for k in range(5)]


def _run(run_id, micro, classes=None):
    classes = classes or {'MAT': (1.0, 10), 'PRO': (0.5, 2)}
    per_class = {t: {'precision': f, 'recall': f, 'f1-score': f, 'support': s}
                 for t, (f, s) in classes.items()}
    macro = sum(f for f, _ in classes.values()) / len(classes)
    per_class.update({'micro avg': {'f1-score': micro / 100}, 'macro avg': {'f1-score': macro},
                      'weighted avg': {'f1-score': micro / 100}})
    return {'run_id': run_id, 'f1': micro, 'f1_strict': micro, 'per_class': per_class}


def _family(directory, system, score, protocol='random_split_vary', granularity='fine'):
    """Write the files of a week-5 system; score(member index, split index) gives micro F1."""
    for j, name in enumerate(ns.file_names(granularity, protocol, system, ns.WEEK5_SYSTEMS)):
        runs = [_run(i, score(j, k)) for k, i in enumerate(SEEDS)]
        (directory / name).write_text(json.dumps({'runs': runs}))


def test_seed_spread_is_the_sd_over_the_seeds_per_split(tmp_path):
    _family(tmp_path, 'crf', lambda j, k: 70 + k + (j - 4.5) / 10)
    files = sa.load_files(tmp_path, ns.file_names('fine', 'random_split_vary', 'crf',
                                                  ns.WEEK5_SYSTEMS))
    spread = sa.seed_spread(files)
    assert spread['per_split'] == {i: 0.29 for i in SEEDS} and spread['mean_sd'] == 0.29
    assert sa.load_files(tmp_path, ['nope.json']) is None


def test_every_triple_passes_a_constant_gain():
    pytest.importorskip('scipy')
    crf = [{'runs': [_run(i, 70 + k) for k, i in enumerate(SEEDS)]} for _ in range(10)]
    mt = [{'runs': [_run(i, 71 + k) for k, i in enumerate(SEEDS)]} for _ in range(10)]
    out = sa.triple_comparisons(mt, crf)
    assert out == {'n_triples': 120,
                   'ten_seed': {'mean_diff': 1.0, 'ci95': [1.0, 1.0], 'passes': True},
                   'share_above_0': 1.0, 'share_below_0': 0.0, 'share_opposite_sign': 0.0}


def test_triples_of_unlucky_seeds_reverse_the_ten_seed_result():
    """+2 for seeds 0-4, -1 for seeds 5-9: the 10-seed gain is +0.5. Triples with 3 or 2
    lucky seeds pass (10 + 50 of 120); 1 lucky seed gives exactly 0 (50); none, -1 (10)."""
    pytest.importorskip('scipy')
    crf = [{'runs': [_run(i, 70 + k) for k, i in enumerate(SEEDS)]} for _ in range(10)]
    st = [{'runs': [_run(i, 70 + k + (2 if j < 5 else -1)) for k, i in enumerate(SEEDS)]}
          for j in range(10)]
    out = sa.triple_comparisons(st, crf)
    assert out['ten_seed']['mean_diff'] == 0.5 and out['ten_seed']['passes']
    assert (out['share_above_0'], out['share_below_0'], out['share_opposite_sign']) == (
        0.5, 0.083, 0.083)


def test_type_scores_rank_the_types_that_pull_macro_down():
    votes = [{'runs': [_run(i, 80) for i in SEEDS[:2]]} for _ in range(3)]
    votes[0]['runs'].append(_run('seed2', 80, {'MAT': (1.0, 10), 'PRO': (0.5, 2),
                                               'DEF': (0.0, 1)}))
    out = sa.type_scores(votes)
    assert list(out) == ['DEF', 'PRO', 'MAT']               # worst first
    assert out['DEF'] == {'f1': 0.0, 'support_per_split': 1.0, 'n': 1, 'macro_without': 75.0,
                          'macro_with': 50.0}
    assert out['PRO']['f1'] == 50.0 and out['PRO']['n'] == 7
    assert out['PRO']['macro_without'] == pytest.approx((6 * 100 + 50) / 7, abs=0.01)
    assert out['MAT']['support_per_split'] == 10.0


def test_analyse_uses_the_systems_with_all_their_files(tmp_path, capsys):
    pytest.importorskip('scipy')
    _family(tmp_path, 'crf', lambda j, k: 70 + k)
    _family(tmp_path, 'crf_mt', lambda j, k: 71 + k)
    _family(tmp_path, 'crf_ls', lambda j, k: 71 + k)
    (tmp_path / ns.file_names('fine', 'random_split_vary', 'crf_ls', ns.WEEK5_SYSTEMS)[9]).unlink()
    _family(tmp_path, 'crf_ens3', lambda j, k: 80 + k)
    out = sa.main(['--results', str(tmp_path)])
    key = 'fine_random_split_vary'
    assert set(out['spread'][key]) == {'crf', 'crf_mt'}       # crf_ls lacks a seed
    assert set(out['triples'][key]) == {'crf_mt'}
    assert out['triples'][key]['crf_mt']['share_above_0'] == 1.0
    assert list(out['types'][key]) == ['PRO', 'MAT'] and out['system'] == 'crf_ens3'
    assert set(out['spread']) == set(out['triples']) == set(out['types']) == {key}
    assert json.loads((tmp_path / sa.OUT_NAME).read_text()) == json.loads(json.dumps(out))
    assert 'crf_mt vs crf: 10 seeds +1.00' in capsys.readouterr().out


def test_published_split_analysis(tmp_path):
    """Plain MatSciBERT and the CRF head over 10 seeds on the published split: the CI against
    the published number, the paired difference, and the CRF votes over 3 seed triples."""
    pytest.importorskip('scipy')
    pytest.importorskip('seqeval')
    gold = ['B-X O', 'O B-X']

    def runs(shift):
        out = []
        for j in range(10):
            run = _run(f'seed{100 * j}', 70 + shift + j / 10)
            run.update({'test_index': [0, 1], 'test_gold': gold, 'test_pred': gold})
            out.append(run)
        return {'runs': out}
    (tmp_path / 'ner_results_fine_published_split.json').write_text(json.dumps(runs(0)))
    (tmp_path / 'ner_results_fine_published_split_crf.json').write_text(json.dumps(runs(2)))
    out = sa.published_split(tmp_path)
    assert list(out) == ['fine']                         # coarse has no files
    entry = out['fine']
    assert entry['published'] == 69.92
    assert entry['baseline']['micro']['mean'] == 70.45
    assert entry['baseline']['published_lies'] == 'below'     # 69.92 below the CI over seeds
    assert entry['crf']['micro']['mean'] == 72.45 and entry['crf']['published_lies'] == 'below'
    assert entry['crf_minus_baseline']['micro']['mean_diff'] == 2.0
    assert entry['crf_votes']['micro']['mean'] == 100.0       # every member predicts the gold
    assert sa.analyse(str(tmp_path))['published_split'] == out
