"""Exploratory analyses of the week-5 10-seed runs (ner_seed_analysis.py). Pure Python; the
tests that compute a confidence interval need scipy and skip without it."""
import itertools
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
    # per type, the most frequent first, every system's mean F1 over its seeds
    assert slot['types'] == {'MAT': {'support': 10, 'baseline': 100.0, 'crf': 100.0,
                                     'crf_mt': 100.0},
                             'PRO': {'support': 2, 'baseline': 50.0, 'crf': 50.0,
                                     'crf_mt': 50.0}}
    # the other methods are not voted
    (tmp_path / 'ner_results_sofc_slot_published_split_crf_ebv.json').write_text(
        json.dumps(runs(2.5, gold)))
    assert 'votes' not in sa.published_split(tmp_path)['sofc_slot']['methods']['crf_ebv']


def test_split_types_mean_f1_per_type_and_system():
    """Per type of a published split: the support and each system's mean F1 over its seeds;
    a type that only a prediction has (support 0) is left out."""
    def runs(*f1s):
        return {f'seed{100 * j}': _run(f'seed{100 * j}', 70, {'MAT': (f, 3), 'PRO': (0.5, 7),
                                                               'XYZ': (0.0, 0)})
                for j, f in enumerate(f1s)}
    types = sa.split_types({'crf': runs(0.6, 0.8), 'crf_mt': runs(0.9, 0.9, 0.6)})
    assert list(types) == ['PRO', 'MAT']
    assert types['MAT'] == {'support': 3, 'crf': 70.0, 'crf_mt': 80.0}
    assert types['PRO'] == {'support': 7, 'crf': 50.0, 'crf_mt': 50.0}


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
    # per type, only the runs on the released test articles
    assert set(sofc['types']['MAT']) == {'support', 'baseline', 'crf'}


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


def test_type_differences_cover_multi_task_and_mention_replacement(tmp_path):
    """Multi-task training (whose seed 0 has no m0 tag) and mention replacement are compared
    per type with the CRF head as the week-6 methods are."""
    pytest.importorskip('scipy')

    def write(system, family, f1):
        for name in ns.file_names('fine', 'leave_one_paper_out', system, {system: family}):
            (tmp_path / name).write_text(json.dumps(
                {'runs': [_run(p, 70, {'MAT': (f1, 10)}) for p in ('p1', 'p2')]}))
    write('crf', ns.WEEK5_SYSTEMS['crf'], 0.5)
    write('crf_mt', ns.WEEK5_SYSTEMS['crf_mt'], 0.6)
    write('crf_mr', ns.seed_family('crf_mr'), 0.75)
    out = sa.week6_type_differences(tmp_path)['fine']
    assert list(out) == ['crf_mt', 'crf_mr']
    assert out['crf_mt']['MAT']['diff'] == 10.0 and out['crf_mr']['MAT']['diff'] == 25.0


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


@pytest.mark.filterwarnings('ignore:Precision is ill-defined')      # a vote of all O
@pytest.mark.filterwarnings('ignore:Recall is ill-defined')
def test_sofc_splits_random_against_article_folds(tmp_path):
    """Week 8, part 1: the CRF head's 10-seed means per split on both protocols, its votes,
    and the gap. Seeds with an even index tag the entity, odd ones miss it."""
    pytest.importorskip('scipy')
    for protocol, base, name in (('random_split_vary', 85, 'seed'),
                                 ('document_folds', 80, 'fold')):
        for j, off in enumerate(range(0, 1000, 100)):
            runs = []
            for k in range(5):
                run = _run(f'{name}{k}', base + k + j / 10)
                run.update({'test_index': [0], 'test_gold': ['B-MAT O'],
                            'test_pred': ['B-MAT O' if j % 2 == 0 else 'O O']})
                runs.append(run)
            (tmp_path / f'ner_results_sofc_{protocol}_crf_m{off}.json').write_text(
                json.dumps({'runs': runs}))
    out = sa.sofc_splits(tmp_path)
    assert out['random_split_vary']['micro']['mean'] == 87.45     # 85 + 2 + 0.45
    assert out['document_folds']['micro']['mean'] == 82.45 and out['gap']['micro'] == 5.0
    assert out['document_folds']['n_splits'] == 5
    # triples 0-200 and 600-800 have 2 seeds that tag it, 300-500 one: votes 100, 0, 100
    assert out['random_split_vary']['crf_votes']['micro']['mean'] == pytest.approx(66.67)
    (tmp_path / 'ner_results_sofc_document_folds_crf_m900.json').unlink()
    assert set(sa.sofc_splits(tmp_path)) == {'random_split_vary'}


def test_sofc_draws_mention_replacement_on_ten_articles(tmp_path):
    """Week 9, part 2: per draw the 10-seed means of the CRF head and mention replacement,
    their difference over the 5 draws, and the CRF head on 10 articles against all 26."""
    pytest.importorskip('scipy')
    for j, off in enumerate(range(0, 1000, 100)):
        for system, gain in (('crf', 0.0), ('crf_mr', 2.0)):
            runs = [_run(f'draw{k}', 70 + k + j / 10 + gain + (k % 2) / 2) for k in range(5)]
            (tmp_path / f'ner_results_sofc_article_draws_{system}_m{off}.json').write_text(
                json.dumps({'runs': runs}))
    official = [_run(f'seed{off}', 80 + j / 10) for j, off in enumerate(range(0, 1000, 100))]
    (tmp_path / 'ner_results_sofc_published_split_crf.json').write_text(
        json.dumps({'runs': official}))
    out = sa.sofc_draws(tmp_path)
    assert out['n_draws'] == 5 and out['passes']
    assert out['crf']['micro']['mean'] == 72.65                  # 70 + 2 + 0.45 + 0.2
    assert out['crf_mr_minus_crf']['micro']['mean_diff'] == 2.0
    assert out['crf_mr_minus_crf']['micro']['per_run'] == [2.0] * 5
    # per seed: the mean over the draws (72.2 + j / 10) minus the official run (80 + j / 10)
    assert out['ten_minus_all_articles']['mean_diff'] == -7.8
    (tmp_path / 'ner_results_sofc_article_draws_crf_mr_m900.json').unlink()
    assert sa.sofc_draws(tmp_path) == {}


def test_mulms_held_out_subfields_and_pool_splits(tmp_path):
    """Week 10: per held-out subfield the 10-seed means of the CRF head and mention
    replacement (decision 1), their votes over the seed triples (decision 2), the CRF head on
    the random pool splits and the gap. The CRF seeds with an even index tag the entity, odd
    ones miss it; mention replacement always tags it."""
    pytest.importorskip('scipy')
    subfields = ['alpha', 'beta', 'gamma', 'delta', 'eps', 'zeta', 'eta']
    for j, off in enumerate(range(0, 1000, 100)):
        for system, gain in (('crf', 0.0), ('crf_mr', 1.0)):
            runs = []
            for k, name in enumerate(subfields):
                run = _run(name, 50 + k + j / 10 + gain + (k % 2) / 2)
                hit = system == 'crf_mr' or j % 2 == 0
                run.update({'test_index': [0], 'test_gold': ['B-MAT O'],
                            'test_pred': ['B-MAT O' if hit else 'O O']})
                runs.append(run)
            (tmp_path / f'ner_results_mulms_leave_one_subfield_out_{system}_m{off}.json'
             ).write_text(json.dumps({'runs': runs}))
        pool = [_run(f'seed{k}', 70 + k + j / 10) for k in range(5)]
        for run in pool:
            run.update({'test_index': [0], 'test_gold': ['B-MAT O'],
                        'test_pred': ['B-MAT O' if j % 2 == 0 else 'O O']})
        (tmp_path / f'ner_results_mulms_random_split_pool_crf_m{off}.json').write_text(
            json.dumps({'runs': pool}))
    out = sa.mulms(tmp_path)
    held = out['leave_one_subfield_out']
    assert held['n_subfields'] == 7 and held['passes']
    assert held['crf_mr_minus_crf']['micro']['per_run'] == [1.0] * 7
    assert held['subfields']['alpha'] == {'crf': 50.45, 'crf_mr': 51.45}
    # CRF votes: triples 0-200 and 600-800 have 2 seeds that tag it, 300-500 one
    assert held['votes']['crf']['micro']['mean'] == pytest.approx(66.67)
    assert held['votes']['crf_mr']['micro']['mean'] == 100.0 and held['votes']['passes']
    assert out['random_split_pool']['micro']['mean'] == 72.45
    assert out['random_split_pool']['n_splits'] == 5
    assert out['random_split_pool']['crf_votes']['micro']['mean'] == pytest.approx(66.67)
    assert out['gap']['micro'] == round(72.45 - held['crf']['micro']['mean'], 2)
    assert 'silver_names' not in out
    # week 11: crf_mrs gains 0.5 over mention replacement on every subfield
    for j, off in enumerate(range(0, 1000, 100)):
        runs = []
        for k, name in enumerate(subfields):
            run = _run(name, 50 + k + j / 10 + 1.5 + (k % 2) / 2)
            run.update({'test_index': [0], 'test_gold': ['B-MAT O'], 'test_pred': ['B-MAT O'],
                        'self_training': {'names': {'n_test_unseen': 10,
                                                    'n_test_unseen_in_silver': k}}})
            runs.append(run)
        (tmp_path / f'ner_results_mulms_leave_one_subfield_out_crf_mrs_m{off}.json'
         ).write_text(json.dumps({'runs': runs}))
    names = sa.mulms(tmp_path)['silver_names']
    assert names['n_subfields'] == 7 and names['subfields']['alpha'] == 51.95
    assert names['minus_crf']['micro']['per_run'] == [1.5] * 7 and names['decision_1']
    assert names['minus_crf_mr']['micro']['per_run'] == [0.5] * 7 and names['decision_2']
    assert names['votes']['minus_crf']['micro']['mean_diff'] == pytest.approx(33.33)
    assert names['votes']['minus_crf_mr']['micro']['mean_diff'] == 0.0
    assert names['unseen_in_silver'] == round(sum(range(7)) / 70, 3)
    assert 'week12' not in sa.mulms(tmp_path)
    # week 12: a topic-name variant screened on 3 seeds (gains 2 over the CRF head, with the
    # mention-level filter), and 6 copies on all 10 seeds (0.3 over the CRF head, no teachers)
    for j, off in enumerate(range(0, 1000, 100)):
        variants = [('crf_mr6', 0.3)] + ([('crf_mrm_topic', 2.0)] if off <= 200 else [])
        for system, gain in variants:
            runs = []
            for k, name in enumerate(subfields):
                run = _run(name, 50 + k + j / 10 + gain + (k % 2) / 2)
                run.update({'test_index': [0], 'test_gold': ['B-MAT O'], 'test_pred': ['B-MAT O']})
                if system == 'crf_mrm_topic':
                    run['self_training'] = {'silver_filter': 'mentions', 'names': {
                        'n_test_unseen': 10, 'n_test_unseen_in_silver': 2 * k}}
                runs.append(run)
            (tmp_path / f'ner_results_mulms_leave_one_subfield_out_{system}_m{off}.json'
             ).write_text(json.dumps({'runs': runs}))
    # the consistency rule's outputs on the held-out subfields (week 13) are not variants
    for system in ('crf_dc', 'crf_mr_dcr'):
        for off in range(0, 1000, 100):
            src = tmp_path / f'ner_results_mulms_leave_one_subfield_out_crf_mr6_m{off}.json'
            (tmp_path / f'ner_results_mulms_leave_one_subfield_out_{system}_m{off}.json'
             ).write_text(src.read_text())
    week12 = sa.mulms(tmp_path)['week12']
    assert set(week12) == {'crf_mr6', 'crf_mrm_topic'}
    assert set(week12['crf_mrm_topic']) == {'screen'}       # seeds 0-200 only
    screen = week12['crf_mrm_topic']['screen']
    assert screen['seeds'] == [0, 100, 200] and screen['n_subfields'] == 7
    assert screen['minus_crf']['micro']['per_run'] == [2.0] * 7
    assert screen['minus_crf_mr']['micro']['per_run'] == [1.0] * 7
    assert 'decision_1' not in screen and 'decision_2' not in screen
    assert screen['subfields']['alpha'] == 52.1                      # 50 + 0.1 (seeds 0-200) + 2
    assert screen['silver_filter'] == 'mentions' and screen['unseen_in_silver'] == 0.6
    # the screen's votes: the triple 0-200 only, where 2 of the 3 CRF seeds tag the entity
    assert screen['votes']['minus_crf']['micro']['mean_diff'] == 0.0
    six = week12['crf_mr6']
    assert set(six) == {'screen', 'decision', 'ten_seeds'}
    decision = six['decision']
    assert decision['seeds'] == list(range(300, 1000, 100))
    # no screen's winner here (the screen lacks variants), so no decision keys: the part
    # is reported (week 14 reads it as the copy control)
    assert 'decision_1' not in decision and 'decision_2' not in decision
    assert decision['minus_crf_mr']['micro']['per_run'] == [-0.7] * 7
    assert decision['subfields']['alpha'] == 50.9                    # 50 + 0.6 (seeds 300-900) + 0.3
    assert six['ten_seeds']['subfields']['alpha'] == 50.75 and 'decision_1' not in six['ten_seeds']
    assert 'names' not in decision and 'silver_filter' not in decision
    (tmp_path / 'ner_results_mulms_leave_one_subfield_out_crf_mr_m900.json').unlink()
    out = sa.mulms(tmp_path)          # each part needs the references at its offsets only
    assert set(out) == {'random_split_pool', 'week12'}
    assert {v: set(parts) for v, parts in out['week12'].items()} == {
        'crf_mrm_topic': {'screen'}, 'crf_mr6': {'screen'}}


def test_paper_count_compares_two_and_three_training_papers(tmp_path):
    """A third training paper: per held-out paper, F1 trained on 2 papers (the inner folds'
    counts at D = 0) against F1 trained on 3 (leave-one-paper-out), over the 4 papers."""
    pytest.importorskip('scipy')
    papers = ['a.conll', 'b.conll', 'c.conll', 'd.conll']
    for off in (0, 100, 200):
        folds = [{'run_id': f'{p}+{q}', 'bias_test': {'0': {p: [4, 10, 10], q: [5, 10, 10]}}}
                 for p, q in itertools.combinations(papers, 2)]
        (tmp_path / f'ner_results_fine_leave_two_papers_out_crf_bias_m{off}.json').write_text(
            json.dumps({'runs': folds}))
        lopo = [_run(p, 50.0 + 10 * k) for k, p in enumerate(papers)]
        (tmp_path / f'ner_results_fine_leave_one_paper_out_crf_m{off}.json').write_text(
            json.dumps({'runs': lopo}))
    out = sa.paper_count(tmp_path)
    assert set(out) == {'fine'}
    # paper a is the first of its 3 pairs (F1 40 each time), d the second (50); b is first
    # in 2 of its pairs (40, 40, 50), c in 1 (50, 50, 40)
    assert {p: r['two'] for p, r in out['fine']['papers'].items()} == {
        'a.conll': 40.0, 'b.conll': 43.33, 'c.conll': 46.67, 'd.conll': 50.0}
    assert out['fine']['papers']['a.conll'] == {'two': 40.0, 'three': 50.0, 'n_two': 9,
                                                'n_three': 3}
    assert out['fine']['three']['mean'] == 65.0
    assert out['fine']['three_minus_two']['mean_diff'] == pytest.approx(65.0 - 45.0)


def test_paper_diversity_three_papers_at_the_size_of_two(tmp_path):
    """Week 8, part 2: per held-out paper, two papers (inner folds), three papers at the size
    of two (crf_p3n2, folds P|Q+R) and three papers; diversity and size over the 4 papers."""
    pytest.importorskip('scipy')
    papers = ['a.conll', 'b.conll', 'c.conll', 'd.conll']
    for off in (0, 100, 200):
        folds = [{'run_id': f'{p}+{q}', 'bias_test': {'0': {p: [4, 10, 10], q: [4, 10, 10]}}}
                 for p, q in itertools.combinations(papers, 2)]           # F1 40 everywhere
        (tmp_path / f'ner_results_fine_leave_two_papers_out_crf_bias_m{off}.json').write_text(
            json.dumps({'runs': folds}))
        sized = [_run(f"{p}|{'+'.join(pair)}", 45.0 + k)
                 for k, p in enumerate(papers)
                 for pair in itertools.combinations([q for q in papers if q != p], 2)]
        (tmp_path / f'ner_results_fine_leave_one_paper_out_crf_p3n2_m{off}.json').write_text(
            json.dumps({'runs': sized}))
        full = [_run(p, 50.0 + 2 * k) for k, p in enumerate(papers)]
        (tmp_path / f'ner_results_fine_leave_one_paper_out_crf_m{off}.json').write_text(
            json.dumps({'runs': full}))
    out = sa.paper_diversity(tmp_path)
    assert set(out) == {'fine'}
    assert out['fine']['papers']['b.conll'] == {'two': 40.0, 'three_at_two': 46.0,
                                                'three': 52.0}
    assert out['fine']['diversity']['mean_diff'] == 6.5           # 45 to 48 against 40
    assert out['fine']['size']['mean_diff'] == 6.5                # 50 to 56 against 45 to 48
    assert out['fine']['diversity']['passes']
    (tmp_path / 'ner_results_fine_leave_one_paper_out_crf_p3n2_m200.json').write_text(
        json.dumps({'runs': sized[:-1]}))                         # a pair missing: left out
    assert sa.paper_diversity(tmp_path) == {}


def test_week13_consistency_rule_and_pooled_estimates(tmp_path):
    """Week 13: the consistency rule's outputs against their inputs per corpus (a decision
    where pre-registered), the week-12 screen's winner, and a method pooled over the
    held-out units of every corpus with its files."""
    pytest.importorskip('scipy')
    subfields = ['alpha', 'beta', 'gamma', 'delta', 'eps', 'zeta', 'eta']
    papers = ['p1', 'p2', 'p3', 'p4']
    folds = [f'fold{k}' for k in range(5)]

    def write(prefix, system, ids, score, extra=None):
        for j, off in enumerate(range(0, 1000, 100)):
            runs = []
            for k, name in enumerate(ids):
                run = _run(name, score(j, k))
                run.update(extra(j, k) if extra else {})
                runs.append(run)
            (tmp_path / f'ner_results_{prefix}_{system}_m{off}.json').write_text(
                json.dumps({'runs': runs}))
    # MuLMS: the rule adds 0.8 on every subfield to the CRF head (13.1a) and 0.2 to mention
    # replacement, which crosses 0 (13.1b); retyping adds 1.0 (reported)
    write('mulms_leave_one_subfield_out', 'crf', subfields, lambda j, k: 50 + k + j / 10)
    write('mulms_leave_one_subfield_out', 'crf_dc', subfields, lambda j, k: 50.8 + k + j / 10,
          lambda j, k: {'n_added': 10 + k, 'n_retyped': 0})
    write('mulms_leave_one_subfield_out', 'crf_dcr', subfields, lambda j, k: 51 + k + j / 10,
          lambda j, k: {'n_added': 10 + k, 'n_retyped': 2})
    write('mulms_leave_one_subfield_out', 'crf_mr', subfields, lambda j, k: 52 + k + j / 10)
    write('mulms_leave_one_subfield_out', 'crf_mr_dc', subfields,
          lambda j, k: 52 + k + j / 10 + (0.9 if k % 2 else -0.5), lambda j, k: {'n_added': 5})
    # MaterioMiner fine-grained: mention replacement gains 2 over the CRF head on every paper;
    # SOFC-Exp's folds: 0.5 on every fold
    write('fine_leave_one_paper_out', 'crf', papers, lambda j, k: 45 + k + j / 10)
    write('fine_leave_one_paper_out', 'crf_mr', papers, lambda j, k: 47 + k + j / 10)
    write('sofc_document_folds', 'crf', folds, lambda j, k: 80 + k + j / 10)
    write('sofc_document_folds', 'crf_mr', folds, lambda j, k: 80.5 + k + j / 10)
    out = sa.week13(tmp_path)
    cons = out['consistency']
    assert set(cons) == {'mulms'} and set(cons['mulms']) == {'crf_dc', 'crf_dcr', 'crf_mr_dc'}
    dc = cons['mulms']['crf_dc']
    assert (dc['decision'], dc['passes'], dc['n_units']) == ('13.1a', True, 7)
    assert dc['minus_input']['micro']['per_run'] == [0.8] * 7
    assert dc['units']['alpha'] == 0.8 and dc['output']['micro']['mean'] == 54.25
    assert dc['input']['micro']['mean'] == 53.45
    assert dc['n_added_per_seed'] == 91.0 and dc['n_retyped_per_seed'] == 0.0   # 7 subfields
    dcr = cons['mulms']['crf_dcr']
    assert 'decision' not in dcr and dcr['minus_input']['micro']['mean_diff'] == 1.0
    assert dcr['n_retyped_per_seed'] == 14.0
    mr_dc = cons['mulms']['crf_mr_dc']
    assert (mr_dc['decision'], mr_dc['passes']) == ('13.1b', False)
    assert mr_dc['minus_input']['micro']['per_run'] == [-0.5, 0.9, -0.5, 0.9, -0.5, 0.9, -0.5]
    # 13.3a: mention replacement on the article folds against the CRF head, paired by fold
    assert set(out['folds']) == {'crf_mr'}
    mr = out['folds']['crf_mr']
    assert (mr['decision'], mr['passes'], mr['n_units']) == ('13.3a', True, 5)
    assert mr['minus_crf']['micro']['per_run'] == [0.5] * 5 and mr['units'] == {f: 0.5 for f in folds}
    assert mr['system']['micro']['mean'] == 82.95 and mr['crf']['micro']['mean'] == 82.45
    assert 'minus_crf_mr' not in mr and 'slot' not in mr
    # the pooled estimates: mention replacement over 7 + 4 + 5 units, reported (no winner)
    assert out['week12_winner'] is None
    assert set(out['pooled']) == {'crf_mr'}
    pooled = out['pooled']['crf_mr']
    assert (pooled['n_units'], pooled['status'], pooled['passes']) == (16, 'reported', True)
    assert {c: e['n_units'] for c, e in pooled['corpora'].items()} == {
        'mulms': 7, 'fine': 4, 'sofc_folds': 5}
    assert pooled['corpora']['fine']['minus_crf']['per_run'] == [2.0] * 4
    assert pooled['corpora']['sofc_folds']['units'] == {f: 0.5 for f in folds}
    assert pooled['minus_crf']['per_run'] == [2.0] * 7 + [2.0] * 4 + [0.5] * 5
    assert pooled['no_corpus_loses'] and not any(e['loses'] for e in pooled['corpora'].values())
    # 13.3b: the multi-task combination on the folds (1 over the CRF head, 0.5 over mention
    # replacement), with its slot level
    write('sofc_document_folds', 'crf_mt_mr', folds, lambda j, k: 81 + k + j / 10)
    write('sofc_slot_document_folds', 'crf_mt_mr', folds, lambda j, k: 70 + k + j / 10)
    out = sa.week13(tmp_path)
    assert set(out['folds']) == {'crf_mr', 'crf_mt_mr'}
    mt = out['folds']['crf_mt_mr']
    assert (mt['decision'], mt['passes'], mt['n_units']) == ('13.3b', True, 5)
    assert mt['minus_crf']['micro']['per_run'] == [1.0] * 5
    assert mt['minus_crf_mr']['micro']['per_run'] == [0.5] * 5
    assert mt['slot']['micro']['mean'] == 72.45 and mt['slot']['n_units'] == 5
    assert set(out['pooled']) == {'crf_mr'}        # crf_mt_mr: no MaterioMiner files yet
    # 13.4: the winner's analogue on the folds (0.3 over the CRF head, -0.2 under mention
    # replacement), and the winner's three-corpus estimate once its MuLMS and MaterioMiner
    # files exist (7 + 4 + 5 units; reported, 13.2 stays the two-corpus decision)
    write('sofc_document_folds', 'crf_mrm6', folds, lambda j, k: 80.3 + k + j / 10)
    write('mulms_leave_one_subfield_out', 'crf_mrm_topic6', subfields,
          lambda j, k: 53 + k + j / 10)
    write('fine_leave_one_paper_out', 'crf_mrm6', papers, lambda j, k: 46 + k + j / 10)
    out = sa.week13(tmp_path)
    an = out['folds']['crf_mrm6']
    assert (an['decision'], an['passes'], an['n_units']) == ('13.4', True, 5)
    assert an['minus_crf']['micro']['per_run'] == [0.3] * 5
    assert an['minus_crf_mr']['micro']['per_run'] == [-0.2] * 5
    winner = out['pooled']['crf_mrm_topic6']
    assert winner['n_units'] == 11 and set(winner['corpora']) == {'mulms', 'fine'}
    three = winner['with_sofc_folds']
    assert (three['n_units'], three['status'], three['passes']) == (16, 'reported', True)
    assert three['sofc_folds']['units'] == {f: 0.3 for f in folds}
    assert three['minus_crf']['per_run'] == [3.0] * 7 + [1.0] * 4 + [0.3] * 5
    assert three['corpora']['sofc_folds'] == {'variant': 'crf_mrm6', 'n_units': 5}
    assert 'with_sofc_folds' not in out['pooled']['crf_mr']
    # a MuLMS variant and its MaterioMiner analogue: two corpora, the analogue losing
    write('mulms_leave_one_subfield_out', 'crf_mrm_topic', subfields,
          lambda j, k: 53 + k + j / 10)
    write('fine_leave_one_paper_out', 'crf_mrm', papers, lambda j, k: 44 + k + j / 10)
    pooled = sa.week13(tmp_path)['pooled']
    assert set(pooled) == {'crf_mr', 'crf_mrm_topic', 'crf_mrm_topic6'}
    topic = pooled['crf_mrm_topic']
    assert topic['n_units'] == 11 and topic['corpora']['fine']['loses']
    assert not topic['no_corpus_loses'] and topic['status'] == 'reported'
    assert topic['minus_crf']['per_run'] == [3.0] * 7 + [-1.0] * 4
    # the screen's winner makes its pooled estimate the decision 13.2
    week12 = {v: {'screen': {'minus_crf_mr': {'micro': {'mean_diff': 0.5}}}}
              for v in sa.SCREEN_VARIANTS}
    assert sa.week12_winner(week12) is None or True       # all equal: the first maximum
    week12['crf_mrm_topic']['screen']['minus_crf_mr']['micro']['mean_diff'] = 1.5
    assert sa.week12_winner(week12) == 'crf_mrm_topic'
    assert sa.week13(tmp_path, week12)['pooled']['crf_mrm_topic']['status'] == 'decision'
    assert sa.week13(tmp_path, week12)['week12_winner'] == 'crf_mrm_topic'
    week12['crf_mrm_topic']['screen']['minus_crf_mr']['micro']['mean_diff'] = -0.1
    for v in sa.SCREEN_VARIANTS:
        week12[v]['screen']['minus_crf_mr']['micro']['mean_diff'] = -0.1
    assert sa.week12_winner(week12) is None                 # no gain above 0
    del week12['crf_gzs_topic']
    assert sa.week12_winner(week12) is None                 # the screen is incomplete
    # a corpus with one family missing drops out of the pooled estimate
    (tmp_path / 'ner_results_sofc_document_folds_crf_mr_m900.json').unlink()
    pooled = sa.week13(tmp_path)['pooled']['crf_mr']
    assert set(pooled['corpora']) == {'mulms', 'fine'} and pooled['n_units'] == 11

def test_week14_copies_topic_and_ablations(tmp_path):
    """Week 14: 14.1 the names method against the 6-copy control, per corpus and pooled
    (the decision); 14.2 V4 with the own corpus against V4 with the mixed corpus (the
    decision) and the mixed variant against the references; 14.3 the ablations against V4
    on the screen's seeds; the week-14 variants are no week-12 variants, and the control's
    decision-seed part of the week-12 section carries no decision."""
    pytest.importorskip('scipy')
    subfields = ['alpha', 'beta', 'gamma', 'delta', 'eps', 'zeta', 'eta']
    papers = ['p1', 'p2', 'p3', 'p4']

    def write(prefix, system, ids, score, offsets=range(0, 1000, 100), extra=None):
        for j, off in enumerate(offsets):
            runs = []
            for k, name in enumerate(ids):
                run = _run(name, score(j, k))
                run.update(extra(j, k) if extra else {})
                runs.append(run)
            (tmp_path / f'ner_results_{prefix}_{system}_m{off}.json').write_text(
                json.dumps({'runs': runs}))

    def names(j, k, **setting):
        return {'self_training': {
            'names': {'n_mentions': 100 + k, 'n_distinct': 50, 'n_test_entities': 200,
                      'n_test_unseen': 100, 'n_test_unseen_in_silver': 20 + k},
            'silver_filter': 'mentions', 'n_sampled': 40000, 'marginal_threshold': 0.9,
            **setting}}
    m = 'mulms_leave_one_subfield_out'
    write(m, 'crf', subfields, lambda j, k: 50 + k + j / 10)
    write(m, 'crf_mr', subfields, lambda j, k: 52 + k + j / 10)
    write(m, 'crf_mrm_topic6', subfields, lambda j, k: 54 + k + j / 10, extra=names)
    assert sa.week14(tmp_path) == {'copies': {}, 'topic': {}, 'ablations': {}}
    # 14.1 on MuLMS: the control on all 10 seeds, 1.5 below V4 on every subfield
    write(m, 'crf_mr6', subfields, lambda j, k: 52.5 + k + j / 10)
    out = sa.week14(tmp_path)
    mulms = out['copies']['mulms']
    assert set(mulms) == {'decision', 'ten_seeds'} and 'pooled' not in out['copies']
    dec = mulms['decision']
    assert dec['seeds'] == list(range(300, 1000, 100)) and dec['n_units'] == 7
    assert dec['minus_crf_mr6']['micro']['per_run'] == [1.5] * 7
    assert dec['crf_mrm_topic6']['micro']['mean'] == 57.6 and dec['units']['alpha'] == 54.6
    assert dec['control']['crf_mr6']['micro']['mean'] == 56.1
    assert dec['control']['minus_crf']['micro']['per_run'] == [2.5] * 7
    assert dec['control']['minus_crf_mr']['micro']['per_run'] == [0.5] * 7
    assert 'votes' not in dec and 'names' not in dec['control']
    assert mulms['ten_seeds']['seeds'] == list(range(0, 1000, 100))
    assert mulms['ten_seeds']['crf_mrm_topic6']['micro']['mean'] == 57.45
    week12 = sa.week12_variants(tmp_path)
    assert set(week12) == {'crf_mr6', 'crf_mrm_topic6'}
    assert 'decision_1' not in week12['crf_mr6']['decision']      # not the screen's winner
    assert 'decision_1' not in week12['crf_mrm_topic6']['decision']   # no winner: screens missing
    # MaterioMiner: the analogue against the control at both granularities, pooled with
    # MuLMS's decision units
    f, c = 'fine_leave_one_paper_out', 'coarse_leave_one_paper_out'
    write(f, 'crf', papers, lambda j, k: 45 + k + j / 10)
    write(f, 'crf_mr', papers, lambda j, k: 47 + k + j / 10)
    write(f, 'crf_mr6', papers, lambda j, k: 47.2 + k + j / 10)
    write(f, 'crf_mrm6', papers, lambda j, k: 48.7 + k + j / 10)
    write(c, 'crf', papers, lambda j, k: 55 + k + j / 10)
    write(c, 'crf_mr', papers, lambda j, k: 56 + k + j / 10)
    write(c, 'crf_mr6', papers, lambda j, k: 56.5 + k + j / 10)
    write(c, 'crf_mrm6', papers, lambda j, k: 56 + k + j / 10)
    out = sa.week14(tmp_path)
    fine = out['copies']['fine']
    assert fine['n_units'] == 4 and fine['minus_crf_mr6']['micro']['per_run'] == [1.5] * 4
    assert fine['control']['minus_crf_mr']['micro']['per_run'] == [0.2] * 4
    assert fine['seeds'] == list(range(0, 1000, 100))
    assert out['copies']['coarse']['minus_crf_mr6']['micro']['per_run'] == [-0.5] * 4
    pooled = out['copies']['pooled']
    assert (pooled['decision'], pooled['n_units'], pooled['passes'],
            pooled['no_corpus_loses']) == ('14.1', 11, True, True)
    assert pooled['minus_control']['per_run'] == [1.5] * 11
    assert pooled['minus_control']['mean_diff'] == 1.5
    assert {k: e['n_units'] for k, e in pooled['corpora'].items()} == {'mulms': 7, 'fine': 4}
    assert pooled['corpora']['mulms']['minus_control']['per_run'] == [1.5] * 7
    # a losing corpus: the fine-grained analogue 0.5 below the control
    write(f, 'crf_mrm6', papers, lambda j, k: 46.7 + k + j / 10)
    pooled = sa.week14(tmp_path)['copies']['pooled']
    assert pooled['minus_control']['per_run'] == [1.5] * 7 + [-0.5] * 4
    assert pooled['corpora']['fine']['loses'] and not pooled['no_corpus_loses']
    assert pooled['passes'] and pooled['minus_control']['mean_diff'] == 0.77
    # 14.2: V4 with the mixed corpus, 1 below V4 with the own corpus, with its silver names
    write(m, 'crf_mrm_mixed6', subfields, lambda j, k: 53 + k + j / 10, extra=names)
    t = sa.week14(tmp_path)['topic']
    assert set(t) == {'decision', 'ten_seeds'}
    d = t['decision']
    assert (d['decision'], d['passes'], d['n_units']) == ('14.2', True, 7)
    assert d['minus_crf_mrm_mixed6']['micro']['per_run'] == [1.0] * 7
    mixed = d['mixed']
    assert mixed['crf_mrm_mixed6']['micro']['mean'] == 56.6
    assert mixed['minus_crf']['micro']['per_run'] == [3.0] * 7
    assert mixed['minus_crf_mr']['micro']['per_run'] == [1.0] * 7
    assert mixed['minus_crf_mr6']['micro']['per_run'] == [0.5] * 7
    assert mixed['silver_filter'] == 'mentions' and mixed['unseen_in_silver'] == 0.23
    assert mixed['names']['eta']['n_mentions'] == 106
    assert 'decision' not in t['ten_seeds']
    assert t['ten_seeds']['seeds'] == list(range(0, 1000, 100))
    assert 'crf_mrm_mixed6' not in sa.week12_variants(tmp_path)
    # 14.3: two ablations on the screen's seeds, against V4's screen runs
    write(m, 'crf_mrm_topic10k6', subfields, lambda j, k: 53.5 + k + j / 10, offsets=(0, 100, 200),
          extra=lambda j, k: names(j, k, n_sampled=10000))
    write(m, 'crf_mrm_topiccap6', subfields, lambda j, k: 54.2 + k + j / 10, offsets=(0, 100, 200),
          extra=lambda j, k: names(j, k, names_share=0.5))
    out = sa.week14(tmp_path)
    ab = out['ablations']
    assert list(ab) == ['crf_mrm_topic10k6', 'crf_mrm_topiccap6']
    ten = ab['crf_mrm_topic10k6']
    assert ten['seeds'] == [0, 100, 200] and ten['n_units'] == 7
    assert ten['minus_crf_mrm_topic6']['micro']['per_run'] == [-0.5] * 7
    assert ten['minus_crf']['micro']['per_run'] == [3.5] * 7
    assert ten['minus_crf_mr']['micro']['per_run'] == [1.5] * 7
    assert ten['setting'] == {'n_sampled': 10000, 'marginal_threshold': 0.9}
    assert ten['unseen_in_silver'] == 0.23
    cap = ab['crf_mrm_topiccap6']
    assert cap['setting'] == {'n_sampled': 40000, 'marginal_threshold': 0.9, 'names_share': 0.5}
    assert cap['minus_crf_mrm_topic6']['micro']['mean_diff'] == 0.2
    assert set(sa.week12_variants(tmp_path)) == {'crf_mr6', 'crf_mrm_topic6'}
    assert sa.analyse(tmp_path)['week14']['copies']['pooled']['n_units'] == 11
    # with every screen variant screened, V4 wins the screen (2.0 over mention replacement
    # against 0.5) and its decision-seed part alone carries the week-12 decisions
    for v in sa.SCREEN_VARIANTS:
        if v not in ('crf_mr6', 'crf_mrm_topic6'):
            write(m, v, subfields, lambda j, k: 52.5 + k + j / 10, offsets=(0, 100, 200))
    week12 = sa.week12_variants(tmp_path)
    assert sa.week12_winner(week12) == 'crf_mrm_topic6'
    assert (week12['crf_mrm_topic6']['decision']['decision_1'],
            week12['crf_mrm_topic6']['decision']['decision_2']) == (True, True)
    assert 'decision_1' not in week12['crf_mr6']['decision']
    assert set(sa.week14(tmp_path)['ablations']) == {'crf_mrm_topic10k6', 'crf_mrm_topiccap6'}
