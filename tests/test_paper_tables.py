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
    # week 14: the pooled decision 14.1 and the mixed-corpus decision 14.2
    analysis['week14'] = {
        'copies': {'pooled': {'decision': '14.1', 'n_units': 11, 'passes': True,
                              'no_corpus_loses': True,
                              'minus_control': _spread(1.5, 0.6, 2.4, 'mean_diff'),
                              'corpora': {'mulms': {'n_units': 7}, 'fine': {'n_units': 4}}}},
        'topic': {'decision': {'n_units': 7, 'passes': False,
                               'minus_crf_mrm_mixed6': d(0.4, -0.3, 1.1)}}}
    rows = pt.held_out_rows(analysis)
    assert rows[-2:] == [
        [14, 'pooled: MuLMS, held-out subfields (7) + MaterioMiner fine-grained, unseen papers (4)',
         '`crf_mrm_topic6` / `crf_mrm6` vs `crf_mr6`', 'week 14, 14.1', 11,
         '+1.50 [+0.60, +2.40]', 'passes'],
        [14, 'MuLMS, held-out subfields', '`crf_mrm_topic6` vs `crf_mrm_mixed6`', 'week 14, 14.2',
         7, '+0.40 [\u22120.30, +1.10]', 'no']]
    analysis['week14']['copies']['pooled']['no_corpus_loses'] = False
    assert pt.held_out_rows(analysis)[-2][3:] == ['week 14, 14.1, a corpus loses', 11,
                                                  '+1.50 [+0.60, +2.40]', 'no']
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


def test_table7_lists_the_ablations_against_the_method():
    """Table 7: the names method on the screen's seeds first, then each ablation of
    week14.ablations in ABLATION_LABELS order, with its distinct silver names, the share of
    the unseen test names among them and its differences against the method, mention
    replacement and the CRF head; ablations without files are left out."""
    def d(mean, lo, hi):
        return {'micro': _spread(mean, lo, hi, 'mean_diff')}

    def names(distinct, unseen, in_silver):
        return {'n_mentions': 3 * distinct, 'n_distinct': distinct, 'n_test_entities': unseen + 10,
                'n_test_unseen': unseen, 'n_test_unseen_in_silver': in_silver}
    screen = {'n_subfields': 2, 'crf_mrm_topic6': {'micro': _spread(52.09, 45.72, 58.46)},
              'minus_crf': d(3.16, -1.96, 8.28), 'minus_crf_mr': d(1.45, -1.35, 4.25),
              'names': {'a': names(100, 50, 10), 'b': names(200, 50, 12)},
              'unseen_in_silver': 0.22}
    low = {'n_units': 2, 'crf_mrm_topiclow6': {'micro': _spread(51.5, 44.0, 59.0)},
           'minus_crf_mrm_topic6': d(-0.59, -2.0, 0.82), 'minus_crf_mr': d(0.86, -1.0, 2.72),
           'minus_crf': d(2.57, -2.5, 7.64), 'seeds': [0, 100, 200],
           'names': {'a': names(300, 50, 20), 'b': names(400, 50, 12)}, 'unseen_in_silver': 0.32,
           'setting': {'n_sampled': 40000, 'marginal_threshold': 0.7}}
    ten = {**low, 'crf_mrm_topic10k6': low['crf_mrm_topiclow6'], 'unseen_in_silver': 0.164,
           'names': {'a': names(10, 50, 8), 'b': names(20, 50, 8)}}
    del ten['crf_mrm_topiclow6']
    analysis = {'week13': {'week12_winner': 'crf_mrm_topic6'},
                'mulms': {'week12': {'crf_mrm_topic6': {'screen': screen}}},
                'week14': {'ablations': {'crf_mrm_topiclow6': low, 'crf_mrm_topic10k6': ten}}}
    rows = pt.ablation_rows(analysis)
    assert [r[0] for r in rows] == [pt.METHOD_LABEL, pt.ABLATION_LABELS['crf_mrm_topic10k6'],
                                    pt.ABLATION_LABELS['crf_mrm_topiclow6']]
    assert rows[0][1:5] == [2, 300, 0.22, 52.09] and rows[0][5] is None
    assert rows[2][1:5] == [2, 700, 0.32, 51.5]
    table = pt.table7(rows)
    assert table.startswith('Table 7. The ablations of the names method')
    assert ('| marginal threshold 0.7 (0.9) | 2 | 700 | 32.0 | 51.50 | −0.59 [−2.00, +0.82] | '
            '+0.86 [−1.00, +2.72] | +2.57 [−2.50, +7.64] |') in table
    assert ("| the method: 40,000 sentences, threshold 0.9, 3 of 3 teachers, a uniform pool | 2 "
            "| 300 | 22.0 | 52.09 |  | +1.45 [−1.35, +4.25] | +3.16 [−1.96, +8.28] |") in table
    assert pt.ablation_rows({}) == []


def test_table8_per_type_and_table9_robustness(tmp_path):
    """Table 8: per type, the CRF head, MR6 and the names method with the differences, per
    part; Table 9: the pooled comparisons per basis with positives, the sign-flip p and the
    leave-one-out minimum, then the MaterioMiner grids."""
    types = {'mulms': {'crf_mrm_topic6': {'MAT': {'crf': 50.0, 'method': 53.0, 'diff': 3.0,
                                                  'n_splits': 7, 'support_per_split': 1800.0}},
                       'crf_mr6': {'MAT': {'crf': 50.0, 'method': 52.0, 'diff': 2.0,
                                           'n_splits': 7, 'support_per_split': 1800.0}}}}
    rows = pt.type_rows({'week14': {'types': types}}, 'mulms')
    assert rows == [['MAT', 1800.0, 7, 50.0, 52.0, 53.0, 3.0, 1.0]]
    table = pt.table8({'mulms': rows})
    assert '| MAT | 1800.0 | 7 | 50.00 | 52.00 | 53.00 | +3.00 | +1.00 |' in table
    assert pt.type_rows({}, 'mulms') == []
    entry = {'mean_diff': 3.58, 'ci95': [0.95, 6.21], 'per_run': [1] * 11, 'n_positive': 9,
             'n_units': 11, 'sign_flip_p': 0.0107,
             'leave_one_out': {'min_lower': 0.44, 'all_above_zero': True, 'rows': []}}
    robust = {'pooled': {'ten': {'names_vs_crf': entry, 'units': ['a'] * 11}},
              'grids': {'fine': {'names_vs_crf': {'mean_diff': 4.27, 'ci95': [1.39, 7.15]}}}}
    rows = pt.robustness_rows(robust)
    assert rows[0][:2] == ['MuLMS seeds 0–900 (as decided)', 'MR6 + names − CRF head']
    table = pt.table9(robust)
    assert ('| MuLMS seeds 0–900 (as decided) | MR6 + names − CRF head | +3.58 [+0.95, +6.21] | 9/11 '
            '| 0.0107 | +0.44 |') in table
    assert '| MR6 + names − CRF head | +4.27 [+1.39, +7.15] |' in table
    assert pt.robustness_rows({}) == []


def test_table10_lexicon_membership():
    """Table 10 lists per setting and system the recall in and out of the lexicon and the paired
    gains, and names the units used and the excluded ones."""
    unit = {'recall_in': 50.0, 'recall_out': 40.0, 'precision_in': 60.0, 'precision_out': None,
            'gold_share_in': 25.0, 'pred_share_in': 30.0, 'gold': {'in': 1, 'out': 3},
            'correct': {'in': 1, 'out': 1}, 'pred': {'in': 1, 'out': 0}, 'pred_correct': {'in': 1, 'out': 0}}
    comp = {'per_unit': {}, 'gain_in': {'mean_diff': 5.0, 'ci95': [1.0, 9.0], 'n_units': 7, 'n_positive': 6},
            'gain_out': {'mean_diff': 2.0, 'ci95': [-1.0, 5.0], 'n_units': 7, 'n_positive': 4},
            'in_minus_out': {'mean_diff': 3.0, 'ci95': [0.5, 5.5], 'n_units': 7, 'n_positive': 6}}
    data = {'mulms': {'systems': {'crf': 'CRF head', 'crf_mrm_topic6': 'MR6 + names'},
                      'units': ['semi', 'sofc'], 'excluded': ['poly'],
                      'pooled': {'crf': unit, 'crf_mrm_topic6': unit},
                      'comparisons': {'names_vs_mr6': comp}}}
    scores, comparisons = pt.lexicon_rows(data)
    assert [s[1] for s in scores] == ['CRF head', 'MR6 + names']
    assert comparisons[0][1] == 'MR6 + names − MR6'
    text = pt.table10(data)
    assert '| MuLMS, 7 held-out subfields (seeds 300–900) | CRF head | 25.0 | 50.0 | 40.0 | 30.0 | 60.0 |  |' in text
    assert '+3.00 [+0.50, +5.50] | 6/7 |' in text and 'semi, sofc' in text and "['poly']" in text


def test_table11_week17():
    """Table 11: per granularity the names' and the control's rows with the decision's verdict
    and the reported differences; empty rows without the week17 section."""
    def spread(mean, lo, hi):
        return {'micro': {'mean': mean, 'ci95': [lo, hi]}}

    def diff(mean, lo, hi):
        return {'micro': {'mean_diff': mean, 'ci95': [lo, hi]}}
    e = {'decision': '17.1', 'passes': False, 'crf_mt_mrm6': spread(53.0, 48.0, 58.0),
         'minus_crf_mt_mr6': diff(0.5, -1.0, 2.0),
         'names': {'minus_crf_mt_mr': diff(1.0, -0.5, 2.5), 'minus_crf_mt': diff(4.0, 1.0, 7.0),
                   'minus_crf': diff(7.7, 4.0, 11.4)},
         'control': {'crf_mt_mr6': spread(52.5, 47.5, 57.5),
                     'minus_crf_mt_mr': diff(0.5, -0.8, 1.8), 'minus_crf_mt': diff(3.5, 0.5, 6.5),
                     'minus_crf': diff(7.2, 3.5, 10.9)}}
    assert pt.week17_rows({}) == []
    rows = pt.week17_rows({'week17': {'fine': e}})
    text = pt.table11(rows)
    assert text.startswith('Table 11.') and 'decision 17.1, no' in text
    lines = [l for l in text.split('\n') if l.startswith('| fine')]
    assert len(lines) == 2 and 'multi-task + MR6 + names' in lines[0] and 'the control' in lines[1]
    assert '+0.50 [−1.00, +2.00]' in lines[0] and '+7.20 [+3.50, +10.90]' in lines[1]


def _matched():
    def paired(per_unit, mean, lo, hi, verdict=None):
        out = {'per_unit': per_unit, 'mean_diff': mean, 'ci95': [lo, hi], 'n_units': 2,
               'n_positive': 2, 'above_0': lo > 0}
        return out | ({'verdict': verdict} if verdict else {})
    keys = ('precision', 'unseen_recall', 'unseen_precision', 'unseen_f1', 'recall_out',
            'in_minus_out', 'bias_f1_change')
    return {'decisions': {'18.1': paired({'semi': 5.0, 'sofc': 3.0}, 4.0, 1.0, 7.0, 'passes'),
                          '18.2': paired({'semi': 1.0, 'sofc': -0.4}, 0.3, -2.0, 2.6, 'no')},
            'prespecified': {k: paired({'semi': 0.5, 'sofc': 0.1}, 0.3, -0.2, 0.8) for k in keys},
            'per_unit': {'semi': {'target_recall': 31.8, 'control_recall_at_0': 26.8,
                                  'd_star': 1.25, 'flagged': False},
                         'sofc': {'target_recall': 50.4, 'control_recall_at_0': 46.5,
                                  'd_star': 5.0, 'flagged': True}},
            'checks': {'n_runs_checked': 49, 'd0_differences': {}}}


def test_table12_week18():
    """Table 12: per subfield the matching and the names minus the control at D*, then the two
    decisions with their verdicts and the pre-specified comparisons."""
    text = pt.table12(_matched())
    assert text.startswith('Table 12.') and 'in minus out' in text
    rows = [line for line in text.split('\n') if line.startswith('| ')]
    assert rows[1].startswith('| semi | 31.80 | 26.80 | 1.25 | +5.00 | +1.00 | +0.50 |')
    assert '5.00 (flagged)' in rows[2]
    assert rows[3].startswith('| mean [95% CI], 2 subfields |')
    assert '+4.00 [+1.00, +7.00] (18.1, passes)' in rows[3] and '(18.2, no)' in rows[3]
    assert 'without the flagged' not in text and 'post hoc)' not in text    # no extra readings
    m = _matched()
    one = {'per_unit': {'semi': 5.0}, 'mean_diff': 5.0, 'ci95': [-1.0, 11.0], 'n_units': 1,
           'n_positive': 1, 'above_0': False}
    two = {'per_unit': {'semi': 0.5, 'sofc': 0.2}, 'mean_diff': 0.35, 'ci95': [-1.0, 1.7],
           'n_units': 2, 'n_positive': 2, 'above_0': False}
    m['without_flagged'] = {'flagged': ['sofc'], 'decisions': {'18.1': one, '18.2': one},
                            'prespecified': {'precision': one}}
    m['post_hoc_nearest'] = {'matched': {'sofc': {'d': 2.5, 'control_recall': 49.93,
                                                  'target_recall': 50.36}},
                             'decisions': {'18.1': two, '18.2': two},
                             'prespecified': {'precision': two}}
    for u, r0, r1 in (('semi', 26.8, 31.84), ('sofc', 46.5, 49.93)):
        m['per_unit'][u]['curve'] = {'0': {'recall': r0, 'precision': 50.0, 'f1_pooled': 40.0},
                                     '2.5': {'recall': r1, 'precision': 45.0, 'f1_pooled': 40.0}}
        m['per_unit'][u]['names'] = {'recall': m['per_unit'][u]['target_recall'],
                                     'precision': 50.0}
        m['per_unit'][u]['between'] = [0.0, 2.5]
        m['per_unit'][u]['w'] = 0.5
    rows = [line for line in pt.table12(m).split('\n') if line.startswith('| ')]
    assert '| 5.00 (flagged; 2.5 post hoc) | +3.00 (+0.20) |' in rows[2]
    assert rows[4].startswith('| without the flagged subfields, 1 (pre-specified) |')
    assert '+5.00 [−1.00, +11.00], 1/1 above 0' in rows[4]
    assert rows[5].startswith('| the flagged subfields at the nearest recall, 2 (post hoc) |')
    assert rows[6].startswith('| subfield | rate | names |')
    assert rows[7] == '| semi | recall | 31.80 | 26.80 | 31.84 |'

def test_table12_parts_from_the_curves():
    """Table 12's second to fourth parts: the control's recall, precision and F1 at every bias
    beside the names' (F1 from their pooled recall and precision), the bracket of D* and the
    pre-specified comparisons per subfield, and the post hoc curves on the unseen names with
    the names minus the control at its best bias."""
    def paired(per_unit, mean=1.0):
        return {'per_unit': per_unit, 'mean_diff': mean, 'ci95': [-1.0, 3.0], 'n_units': 2,
                'n_positive': 2}
    keys = ('unseen_recall', 'unseen_precision', 'unseen_f1', 'recall_out', 'in_minus_out',
            'bias_f1_change', 'precision')
    curve = {'0': {'recall': 40.0, 'precision': 60.0, 'f1_pooled': 48.0},
             '5': {'recall': 45.0, 'precision': 50.0, 'f1_pooled': 47.37}}
    m = {'per_unit': {'semi': {'names': {'recall': 50.0, 'precision': 50.0}, 'curve': curve,
                               'between': [0.0, 5.0], 'w': 0.5, 'flagged': False},
                      'sofc': {'names': {'recall': 60.0, 'precision': 40.0}, 'curve': curve,
                               'between': [5.0, 5.0], 'w': 0.0, 'flagged': True}},
         'prespecified': {k: paired({'semi': 2.0, 'sofc': -1.0}) for k in keys},
         'post_hoc_nearest': {'prespecified': {k: paired({'semi': 2.0, 'sofc': 0.5}) for k in keys}},
         'post_hoc_curves': {'units': {u: {'names': {'recall_in': 80.0, 'recall_out': 30.0},
                                           'control': {'0': {'recall_in': 70.0, 'recall_out': 32.0},
                                                       '5': {'recall_in': 60.0, 'recall_out': 35.0}}}
                                       for u in ('semi', 'sofc')},
                             'names_minus_best': {'recall_in': paired({'semi': 10.0, 'sofc': 10.0}, 10.0),
                                                  'recall_out': paired({'semi': -5.0, 'sofc': -5.0}, -5.0)}}}
    text = pt.table12_curves(m)
    rows = [line for line in text.split('\n') if line.startswith('| ')]
    assert '| sofc | F1 | 48.00 | 48.00 | 47.37 |' in rows      # 2 * 60 * 40 / 100 = 48
    assert '| semi | 0.00 to 5.00, w = 0.50 | +2.00 |' in text
    assert '| sofc | 5.00 | −1.00 (+0.50) |' in text
    assert '| semi | held | 80.00 | 70.00 | 60.00 | +10.00 |' in rows
    assert any(r.startswith('| mean [95% CI] | not held |') and r.endswith('2/2 above 0 |') for r in rows)
    assert pt.table12_curves({'per_unit': {'x': {}}}) == ''



def test_table12_post_hoc_parts():
    """Table 12's post hoc parts: the decomposition rows, the typed split with its unit counts,
    the sign-flip p-values and the lexicons' type counts; '' without them."""
    def e(mean, n=7, pos=7):
        return {'mean_diff': mean, 'ci95': [mean - 1, mean + 1], 'n_units': n, 'n_positive': pos}
    m = {'post_hoc_decomposition': {'names_minus_mr6_at_0': e(7.29),
                                    'mr6_at_0_minus_control_at_d_star': e(3.96),
                                    'matched_only': {'names_minus_mr6_at_0': e(5.92, 4, 4),
                                                     'mr6_at_0_minus_control_at_d_star': e(0.14, 4, 1)}},
         'post_hoc_types': {'minus_control_at_d_star': {'MAT_in': e(12.02)},
                            'minus_control_nearest': {'MAT_in': e(9.87)},
                            'minus_mr6_at_0': {'MAT_in': None},
                            'gold_per_run': {'a': {'MAT_in': 184.0}, 'b': {'MAT_in': 1022.0}}},
         'post_hoc_sign_flip': {'18.1': 0.0156, '18.2': 0.1406},
         'post_hoc_lexicon_types': {'a': {'MAT': {'distinct': 2392, 'entries': 29748}}}}
    text = pt.table12_post_hoc(m)
    assert '| the 4 matched subfields | +5.92 [+4.92, +6.92] | +0.14 [−0.86, +1.14] |' in text
    assert '| MAT | held | +12.02 [+11.02, +13.02], 7/7 | +9.87 [+8.87, +10.87], 7/7 |  | 184–1022 |' in text
    assert '18.1 0.0156, 18.2 0.1406' in text and '| a | 2,392 (29,748) |' in text
    assert pt.table12_post_hoc({}) == ''


def test_table14_chain_and_table15_lexicon_units():
    """Table 14: each system against the one it builds on, on the unseen and seen names and all
    entities, with MuLMS's decision seeds too; Table 15: the lexicon split per unit."""
    def q(m, r=1.0, p=-1.0):
        return {'mean_diff': m, 'ci95': [m - 1, m + 1],
                'recall': {'mean_diff': r, 'ci95': [r - 1, r + 1]},
                'precision': {'mean_diff': p, 'ci95': [p - 1, p + 1]}}
    mentions = {'chain': {'mulms_leave_one_subfield_out': {
        'crf_mrm_topic6': {'reference': 'crf_mr6', 'unseen': q(0.93, 3.24, -2.23), 'seen': q(0.77),
                           'all': q(0.85), 'decision': {'unseen': q(0.56, 2.56, -2.22)}}},
        'fine_leave_one_paper_out': {'crf_mr': {'reference': 'crf', 'unseen': q(3.19),
                                                'decision': {'unseen': q(9.9)}}}}}
    text = pt.table14(mentions)
    assert '| MuLMS, 7 subfields | MR6 + names − MR6 | all ten | unseen names | +0.93 [−0.07, +1.93] | +3.24 [+2.24, +4.24] | −2.23 [−3.23, −1.23] |' in text
    assert '| MuLMS, 7 subfields | MR6 + names − MR6 | decision seeds (300–900) | unseen names | +0.56' in text
    assert 'MR3 − CRF head | all ten | unseen names' in text and '+9.90' not in text   # no decision seeds off MuLMS
    data = {'mulms': {'comparisons': {'names_vs_mr6': {'per_unit': {
        'sofc': {'gain_in': 14.09, 'gain_out': 1.51, 'in_minus_out': 12.58}}}}},
            'fine': {'comparisons': {'names_vs_mr6': {'per_unit': {
                '10.1038_ncomms7164.conll': {'gain_in': 1.0, 'gain_out': -0.5, 'in_minus_out': 1.5}}}}}}
    text = pt.table15(data)
    assert '| MR6 + names − MR6 | sofc | +14.09 | +1.51 | +12.58 |' in text
    assert '| 10.1038_ncomms7164 | +1.00 | −0.50 | +1.50 |' in text
