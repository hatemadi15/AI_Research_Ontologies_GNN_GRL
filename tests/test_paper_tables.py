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
    mentions = {'systems': {'fine_leave_one_paper_out': {'crf': kinds}}}
    matching = {'systems': {'fine_leave_one_paper_out': {'crf': {
        c: {'mean': m} for c, m in zip(pt.CRITERIA, (46.46, 56.1, 56.36, 70.58))}}}}
    t4 = pt.table4(mentions, matching)
    assert '| fine, unseen papers | 81.0 | 71.4 | 35.0 | 46.46 | 56.10 | 56.36 | 70.58 |' in t4
    assert 'random splits' not in t4.split('\n\n', 1)[1]     # settings without results left out


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
    assert 'table 4: no mention or matching analysis' in capsys.readouterr().out
    assert (tmp_path / 'tables' / 'table2_decisions.md').read_text().count('| 5 | GPU |') == 1
