"""The tables of the paper plan (paper_tables.py) from toy results files."""
import json

import paper_tables as pt


def _spread(mean, lo, hi, key='mean'):
    return {key: mean, 'ci95': [lo, hi]}


def _averages(mean, key='mean'):
    return {a: _spread(mean, mean - 1, mean + 1, key) for a, _ in pt.AVERAGES}


def _published():
    return {'fine': {'published': 69.92, 'baseline': _averages(70.0), 'crf': _averages(73.0),
                     'crf_minus_baseline': _averages(3.0, 'mean_diff')},
            'sofc': {'published': 84.77, 'published_macro': 82.39,
                     'baseline': _averages(82.5), 'crf': _averages(83.75),
                     'crf_minus_baseline': _averages(1.22, 'mean_diff'),
                     'methods': {'crf_dc': {'minus_crf': _averages(-0.42, 'mean_diff')}}}}


def test_numbers_and_intervals():
    assert pt.number(-0.5) == '−0.50' and pt.number(0.5, signed=True) == '+0.50'
    assert pt.interval(_spread(1.0, -0.5, 2.5, 'mean_diff'), 'mean_diff', signed=True) == \
        '+1.00 [−0.50, +2.50]'


def test_table1_puts_the_published_score_on_its_system():
    lines = pt.table1(_published()).splitlines()
    rows = [line.split(' | ') for line in lines if line.startswith('| MaterioMiner') or
            line.startswith('| SOFC')]
    plain_fine, crf_fine, diff_fine, plain_sofc, crf_sofc, diff_sofc, dc = rows
    assert plain_fine[-1] == '69.92 |' and crf_fine[-1] == ' |'
    assert plain_sofc[-1] == ' |' and crf_sofc[-1] == '84.77 (macro 82.39) |'
    assert diff_fine[1] == 'CRF head − plain' and diff_fine[2] == '+3.00 [+2.00, +4.00]'
    assert dc[1] == '`crf_dc` − CRF head' and dc[2] == '−0.42 [−1.42, +0.58]'


def _comparison(variant, role, lo, protocol='leave_one_paper_out'):
    return {'variant': variant, 'reference': 'crf', 'granularity': 'fine',
            'protocol': protocol, 'role': role, 'n_pairs': 4,
            'micro': {'mean_diff': lo + 1, 'ci95': [lo, lo + 2]}}


def test_table2_marks_the_decisions_that_pass():
    files = [('5', 'GPU', '10', {'a': _comparison('crf_mt', 'decision', 0.3),
                                 'b': _comparison('crf_ls', 'decision', -1.0),
                                 'c': _comparison('crf_st3', 'reported', 0.5,
                                                  'random_split_vary')})]
    decided = pt.comparison_rows(files, pt.DECIDED)
    assert [e['variant'] for *_, e in decided] == ['crf_mt', 'crf_ls']
    table = pt.table2(decided)
    assert '| 5 | GPU | 10 | fine | unseen papers | `crf_mt` vs `crf` | decision | 4 | ' \
           '+1.30 [+0.30, +2.30] | passes |' in table
    assert '| no |' in table
    every = pt.table2(pt.comparison_rows(files), 'Table A1')
    assert every.startswith('Table A1:') and 'random splits' in every
    assert every.splitlines()[-1].endswith('| reported | 4 | +1.50 [+0.50, +2.50] |  |')


def test_table3_and_table4():
    t3 = pt.table3(_published())
    assert '| MaterioMiner, fine-grained | CRF head − plain | +3.00 [+2.00, +4.00] | ' \
           '+3.00 [+2.00, +4.00] |' in t3
    kinds = {k: {'share': s, 'recall': r} for k, s, r in (('seen', 0.2, 71.4),
                                                          ('unseen', 0.81, 35.0))}
    mentions = {'systems': {key: {'crf': kinds} for key in ('fine_leave_one_paper_out',
                                                             'mulms_leave_one_subfield_out')}}
    matching = {'systems': {key: {'crf': {
        c: {'mean': m} for c, m in zip(pt.CRITERIA, (46.46, 56.1, 56.36, 70.58))}}
        for key in ('fine_leave_one_paper_out', 'mulms_leave_one_subfield_out')}}
    t4 = pt.table4(mentions, matching)
    assert '| fine, unseen papers | 81.0 | 71.4 | 35.0 | 46.46 | 56.10 | 56.36 | 70.58 |' in t4
    assert t4.splitlines()[-1].startswith('| MuLMS, held-out subfields | 81.0 |')
    assert 'random splits' not in t4.split('\n\n', 1)[1]     # settings without results left out


def _probe(subfield, fill_s, fill_m, precision=60.0, recall=40.0):
    seed = {'before': {'precision': precision, 'recall': recall, 'f1': 48.0}}
    return {'subfield': subfield,
            'lexicons': {'sentences_0.9': {'unseen_share_in_lexicon': 0.133},
                         'mentions_0.9': {'unseen_share_in_lexicon': 0.25}},
            'probe': {'sentences_0.9': {'fill': {'delta_f1': fill_s, 'per_seed': {'m0': seed}}},
                      'mentions_0.9': {'fill': {'delta_f1': fill_m, 'per_seed': {'m0': seed}}},
                      'ceiling_unseen_gold': {'fill': {'delta_f1': -1.82},
                                              'override': {'delta_f1': 4.13}}}}


def test_table5_lists_the_probe_per_subfield_with_a_mean():
    coverage = {'subfields': {'steel': {'corpora': {'topic_steel': {'exact': 0.55}}}}}
    t5 = pt.table5([_probe('steel', 0.01, -10.03), _probe('sofc', 5.79, 5.91)], coverage)
    lines = t5.splitlines()
    assert '| steel | 55.0 | 13.3 | 25.0 | 60.0 | 40.0 | +0.01 | −10.03 | −1.82 | +4.13 |' in lines
    assert '| sofc |  | 13.3 | 25.0 | 60.0 | 40.0 | +5.79 | +5.91 | −1.82 | +4.13 |' in lines
    assert lines[-1] == '| mean | 55.0 | 13.3 | 25.0 | 60.0 | 40.0 | +2.90 | −2.06 | −1.82 | +4.13 |'
    assert lines[2].startswith('| subfield | corpus holds (%) |')


def test_table6_lists_the_held_out_decisions():
    """Table 6: every pre-registered decision on held-out documents, in week order, the
    reported comparisons and the screen's losers left out; the pooled decision needs every
    corpus not to lose."""
    def d(mean, lo, hi):
        return {'micro': _spread(mean, lo, hi, 'mean_diff')}
    held = {'n_subfields': 7, 'passes': False, 'crf_mr_minus_crf': d(2.1, -0.31, 4.51),
            'votes': {'passes': True, 'crf_mr_minus_crf': d(2.71, 0.22, 5.2)}}
    names = {'n_subfields': 7, 'minus_crf': d(2.04, -0.85, 4.92), 'decision_1': False,
             'minus_crf_mr': d(-0.06, -1.09, 0.96), 'decision_2': False}
    part = {'n_subfields': 7, 'minus_crf': d(3.0, 1.0, 5.0), 'decision_1': True,
            'minus_crf_mr': d(1.0, -0.5, 2.5), 'decision_2': False}
    week12 = {'crf_mrm_topic': {'decision': part}, 'crf_mr6': {'decision': part}}
    consistency = {
        'mulms': {'crf_dc': {'decision': '13.1a', 'passes': True, 'n_units': 7,
                             'minus_input': d(0.82, 0.18, 1.46)},
                  'crf_dcr': {'n_units': 7, 'minus_input': d(1.01, 0.28, 1.74)}},
        'coarse': {'crf_dc': {'decision': 'week 7', 'passes': True, 'n_units': 4,
                              'minus_input': d(0.12, 0.01, 0.24)}}}
    folds = {'crf_mr': {'decision': '13.3a', 'passes': False, 'n_units': 5,
                        'minus_crf': d(0.3, -0.2, 0.8)}}
    pooled = {'crf_mrm_topic': {'status': 'decision', 'passes': True, 'no_corpus_loses': False,
                                'n_units': 11, 'minus_crf': _spread(2.0, 0.5, 3.5, 'mean_diff'),
                                'corpora': {'mulms': {'n_units': 7}, 'fine': {'n_units': 4}}},
              'crf_mr': {'status': 'reported', 'passes': True, 'no_corpus_loses': True,
                         'n_units': 11, 'minus_crf': _spread(2.3, 0.9, 3.7, 'mean_diff'),
                         'corpora': {'mulms': {'n_units': 7}, 'fine': {'n_units': 4}}}}
    analysis = {'mulms': {'leave_one_subfield_out': held, 'silver_names': names,
                          'week12': week12},
                'week13': {'consistency': consistency, 'week12_winner': 'crf_mrm_topic',
                           'folds': folds, 'pooled': pooled}}
    week12 = {'a': {**_comparison('crf_mrm6', 'decision', 4.27), 'granularity': 'fine'},
              'b': {**_comparison('crf_mrm6', 'reported', 1.5)}}
    rows = pt.held_out_rows(analysis, week12)
    assert [r[0] for r in rows] == [7, 10, 10, 11, 11, 12, 12, 12, 13, 13, 13]
    assert ("| 12 | MaterioMiner fine-grained, unseen papers | `crf_mrm6` vs `crf` | week 12, "
            "the winner's analogue | 4 | +5.27 [+4.27, +6.27] | passes |") in pt.table6(rows)
    rows = pt.held_out_rows(analysis)
    assert [r[0] for r in rows] == [7, 10, 10, 11, 11, 12, 12, 13, 13, 13]
    table = pt.table6(rows)
    assert table.startswith('Table 6. Pre-registered decisions on held-out documents')
    assert ('| 7 | MaterioMiner coarse-grained, unseen papers | `crf_dc` vs `crf` | week 7 '
            '(recomputed) | 4 | +0.12 [+0.01, +0.24] | passes |') in table
    assert ('| 10 | MuLMS, held-out subfields | `crf_mr` votes vs `crf` votes | week 10, '
            'decision 2 | 7 | +2.71 [+0.22, +5.20] | passes |') in table
    assert ('| 11 | MuLMS, held-out subfields | `crf_mrs` vs `crf_mr` | week 11, decision 2 '
            '| 7 | \u22120.06 [\u22121.09, +0.96] | no |') in table
    assert ('| 12 | MuLMS, held-out subfields | `crf_mrm_topic` vs `crf_mr` | week 12, '
            'decision 2 | 7 | +1.00 [\u22120.50, +2.50] | no |') in table
    assert ('| 13 | MuLMS, held-out subfields | `crf_dc` vs `crf` | week 13, 13.1a | 7 | '
            '+0.82 [+0.18, +1.46] | passes |') in table
    assert ('| 13 | SOFC-Exp, article folds | `crf_mr` vs `crf` | week 13, 13.3a | 5 | '
            '+0.30 [\u22120.20, +0.80] | no |') in table
    assert ('| 13 | pooled: MuLMS, held-out subfields (7) + MaterioMiner fine-grained, unseen '
            'papers (4) | `crf_mrm_topic` vs `crf` | week 13, 13.2, a corpus loses | 11 | '
            '+2.00 [+0.50, +3.50] | no |') in table
    assert 'crf_dcr' not in table and 'crf_mr6' not in table and table.count('`crf_mr` vs') == 2
    assert pt.held_out_rows({}) == [] and pt.held_out_rows({'week13': {}}) == []


def test_main_writes_what_it_has(tmp_path, capsys):
    results = tmp_path / 'results'
    (results / 'gpu').mkdir(parents=True)
    (results / 'gpu' / 'ner_week5_seed_analysis.json').write_text(
        json.dumps({'published_split': _published()}))
    (results / 'gpu' / 'ner_week5_comparisons.json').write_text(
        json.dumps({'a': _comparison('crf_mt', 'decision', 0.3)}))
    written = pt.main(['--results', str(results), '--out', str(tmp_path / 'tables')])
    assert written == ['table1_published_split.md', 'table3_strict.md', 'table2_decisions.md',
                       'tableA1_all_comparisons.md']
    out = capsys.readouterr().out
    assert 'table 4: no mention or matching analysis' in out
    assert 'table 5: no probe reports' in out
    assert 'table 6: no held-out decisions' in out
    assert (tmp_path / 'tables' / 'table2_decisions.md').read_text().count('| 5 | GPU |') == 1
