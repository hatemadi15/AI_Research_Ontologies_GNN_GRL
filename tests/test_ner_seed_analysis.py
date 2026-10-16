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
    _family(tmp_path, 'crf_mt_ens3', lambda j, k: 81 + k)
    out = sa.main(['--results', str(tmp_path)])
    key = 'fine_random_split_vary'
    assert set(out['spread'][key]) == {'crf', 'crf_mt'}       # crf_ls lacks a seed
    assert set(out['triples'][key]) == {'crf_mt'}
    assert out['triples'][key]['crf_mt']['share_above_0'] == 1.0
    assert list(out['types'][key]) == ['crf_ens3', 'crf_mt_ens3']   # one per complete vote
    assert all(list(t) == ['PRO', 'MAT'] for t in out['types'][key].values())
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


def test_published_split_analysis_of_sofc_exp_and_a_method(tmp_path):
    """SOFC-Exp's official split (week 6): plain MatSciBERT and the CRF head against the
    published MatSciBERT-CRF scores, and DeBERTa-v3-large against the CRF head by seed."""
    pytest.importorskip('scipy')
    pytest.importorskip('seqeval')
    gold = ['B-X O', 'O B-X']

    def runs(shift, ids=range(10)):
        out = []
        for j in ids:
            run = _run(f'seed{100 * j}', 80 + shift + j / 10)
            run.update({'test_index': [0, 1], 'test_gold': gold, 'test_pred': gold})
            out.append(run)
        return {'runs': out}
    for variant, shift in (('', 0), ('_crf', 4.4), ('_crf_deberta', 5.4)):
        (tmp_path / f'ner_results_sofc_published_split{variant}.json').write_text(
            json.dumps(runs(shift)))
    out = sa.published_split(tmp_path)
    assert list(out) == ['sofc']
    sofc = out['sofc']
    assert (sofc['published'], sofc['published_macro']) == (84.77, 82.39)
    assert sofc['crf']['micro']['mean'] == 84.85 and sofc['crf']['published_lies'] == 'within'
    assert sofc['baseline']['published_lies'] == 'above'           # 84.77 above 80.45's CI
    deberta = sofc['methods']['crf_deberta']
    assert deberta['micro']['mean'] == 85.85 and deberta['n_pairs'] == 10
    assert deberta['minus_crf']['micro']['mean_diff'] == 1.0
    assert 'crf_ctx' not in sofc['methods']
    # a method with fewer seeds is paired on the seeds both have
    (tmp_path / 'ner_results_sofc_published_split_crf_ctx.json').write_text(
        json.dumps(runs(3.4, ids=range(3))))
    ctx = sa.published_split(tmp_path)['sofc']['methods']['crf_ctx']
    assert ctx['n_pairs'] == 3 and ctx['minus_crf']['micro']['mean_diff'] == -1.0
    # week 7: the consistency rule on the CRF head's predictions, paired in the same way
    (tmp_path / 'ner_results_sofc_published_split_crf_dc.json').write_text(
        json.dumps(runs(4.9)))
    methods = sa.published_split(tmp_path)['sofc']['methods']
    assert list(methods) == ['crf_deberta', 'crf_ctx', 'crf_dc']
    assert methods['crf_dc']['minus_crf']['micro']['mean_diff'] == 0.5
    # week 7, part 2a: recall calibration, its bias chosen on the validation articles
    (tmp_path / 'ner_results_sofc_published_split_crf_ebv.json').write_text(
        json.dumps(runs(5.4)))
    methods = sa.published_split(tmp_path)['sofc']['methods']
    assert list(methods)[-1] == 'crf_ebv' and methods['crf_ebv']['minus_crf']['micro'][
        'mean_diff'] == 1.0


def test_published_split_of_sofc_exp_slots_and_multi_task(tmp_path):
    """Week 7, part 2d: SOFC-Exp's slot level against the published SOFC-Slot scores, and
    multi-task training against the CRF head, paired by seed and, for its votes over the seed
    triples, by triple."""
    pytest.importorskip('scipy')
    pytest.importorskip('seqeval')
    gold = ['B-anode_material O', 'O B-working_temperature']
    miss = ['B-anode_material O', 'O O']

    def runs(shift, pred):
        out = []
        for j in range(10):
            run = _run(f'seed{100 * j}', 70 + shift + j / 10)
            run.update({'test_index': [0, 1], 'test_gold': gold, 'test_pred': pred})
            out.append(run)
        return {'runs': out}
    for variant, shift, pred in (('', 0, gold), ('_crf', 2, gold), ('_crf_mt', 3, miss)):
        (tmp_path / f'ner_results_sofc_slot_published_split{variant}.json').write_text(
            json.dumps(runs(shift, pred)))
    slot = sa.published_split(tmp_path)['sofc_slot']
    assert (slot['published'], slot['published_macro']) == (72.85, 65.35)
    assert slot['crf']['published_lies'] == 'above'           # 72.85 above 72.45's CI
    mt = slot['methods']['crf_mt']
    assert mt['n_pairs'] == 10 and mt['minus_crf']['micro']['mean_diff'] == 1.0
    # every CRF vote finds both entities; every multi-task vote one of the two
    assert slot['crf_votes']['micro']['mean'] == 100.0
    assert mt['votes']['micro']['mean'] == pytest.approx(66.67, abs=0.01)
    assert mt['votes_minus_crf_votes']['micro']['mean_diff'] == pytest.approx(-33.33, abs=0.01)
    assert len(mt['votes_minus_crf_votes']['micro']['per_run']) == 3
    # the other methods are not voted
    (tmp_path / 'ner_results_sofc_slot_published_split_crf_ebv.json').write_text(
        json.dumps(runs(2.5, gold)))
    assert 'votes' not in sa.published_split(tmp_path)['sofc_slot']['methods']['crf_ebv']


def test_retokenized_test_of_sofc_exp(tmp_path):
    """Week 7, part 2e: the CRF head and plain MatSciBERT scored on the retokenized test
    articles, each against the same model on the released test articles, paired by seed."""
    pytest.importorskip('scipy')

    pytest.importorskip('seqeval')
    gold = ['B-MATERIAL O', 'O B-VALUE']

    def runs(shift):
        out = []
        for j in range(10):
            run = _run(f'seed{100 * j}', 80 + shift + j / 10)
            run.update({'test_index': [0, 1], 'test_gold': gold, 'test_pred': gold})
            out.append(run)
        return {'runs': out}
    for variant, shift in (('', 0), ('_crf', 1), ('_crf_cdetest', 1.5), ('_cdetest', 0.25)):
        (tmp_path / f'ner_results_sofc_published_split{variant}.json').write_text(
            json.dumps(runs(shift)))
    sofc = sa.published_split(tmp_path)['sofc']
    crf = sofc['methods']['crf_cdetest']
    assert crf['n_pairs'] == 10 and crf['minus_crf']['micro']['mean_diff'] == 0.5
    plain = sofc['baseline_retokenized_test']
    assert plain['n_pairs'] == 10 and plain['minus_baseline']['micro']['mean_diff'] == 0.25


def test_week6_type_differences_pair_seeds_and_papers(tmp_path, capsys):
    """Per entity type, a week-6 method against the CRF head on the held-out papers, paired by
    seed offset and paper over the pairs whose test set has the type (reported alongside the
    week-6 decisions, no decision); a method missing a seed is left out."""
    pytest.importorskip('scipy')

    def write(system, classes):
        for j, name in enumerate(ns.file_names('fine', 'leave_one_paper_out', system,
                                               {system: ns.seed_family(system)})):
            runs = [_run(p, 70, classes(j, p)) for p in ('p1', 'p2')]
            (tmp_path / name).write_text(json.dumps({'runs': runs}))
    # PRO is predicted on p2 but absent from its test set: no pair there
    write('crf', lambda j, p: ({'MAT': (0.8 - j / 100, 10), 'PRO': (0.5, 2)} if p == 'p1' else
                               {'MAT': (0.6, 20), 'PRO': (0.0, 0)}))
    write('crf_ctx', lambda j, p: ({'MAT': (0.9 - j / 100, 10), 'PRO': (0.75, 2)} if p == 'p1'
                                   else {'MAT': (0.6, 20), 'PRO': (0.5, 0)}))
    write('crf_deberta', lambda j, p: {'MAT': (0.5, 10)})
    (tmp_path / ns.file_names('fine', 'leave_one_paper_out', 'crf_deberta',
                              {'crf_deberta': ns.seed_family('crf_deberta')})[9]).unlink()
    out = sa.week6_type_differences(tmp_path)
    assert list(out) == ['fine'] and list(out['fine']) == ['crf_ctx']
    mat, pro = out['fine']['crf_ctx'].values()
    assert list(out['fine']['crf_ctx']) == ['MAT', 'PRO']          # most frequent first
    assert mat == {'crf': 67.75, 'method': 72.75, 'diff': 5.0, 'n_splits': 2,
                   'support_per_split': 15.0}
    assert pro == {'crf': 50.0, 'method': 75.0, 'diff': 25.0, 'n_splits': 1,
                   'support_per_split': 2.0}
    assert sa.main(['--results', str(tmp_path)])['week6_types'] == out
    assert 'crf_ctx, fine-grained, per type' in capsys.readouterr().out


def test_error_kinds_of_the_crf_head():
    """Each gold entity is correct, missed (no overlapping prediction), of the right span and
    the wrong type, or a boundary error (of the right or of another type); each prediction
    without any overlapping gold entity is spurious."""
    gold = ['B-A I-A O B-B O B-A O B-B I-B O B-A', 'O O']
    pred = ['B-A I-A O B-A O O O B-B O O O', 'B-A O']
    out = sa.error_kinds([{'runs': [{'test_gold': gold, 'test_pred': pred}]}])
    # gold: A(0,2) correct, B(3,4) wrong type, A(5,6) missed, B(7,9) boundary, A(10,11) missed
    assert out == {'n_gold': 5, 'n_pred': 4, 'correct': 0.2, 'missed': 0.4, 'wrong_type': 0.2,
                   'boundary': 0.2, 'boundary_and_type': 0.0, 'spurious': 0.25}
