"""MuLMS's documents on Europe PMC (mulms_articles.py): title candidates, queries, scores and
the best match, with a fake search."""
import json

import mulms_articles as ma


def write_doc(folder, name, segments):
    lines = []
    for segment in segments:
        lines += [f'{w} O' for w in segment.split()] + ['']
    (folder / f'{name}.conll').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def test_title_candidates_skip_boilerplate(tmp_path):
    write_doc(tmp_path, 'steel_03', ['Article | The Influence of Magnetic Field on Fatigue of a '
                                     '35CrMo Steel', 'Abstract :', 'Fatigue tests were run .',
                                     'Specimens were machined from rolled bars .'])
    assert ma.title_candidates(str(tmp_path / 'steel_03.conll')) == [
        'The Influence of Magnetic Field on Fatigue of a 35CrMo Steel',
        'Specimens were machined from rolled bars']       # 'Fatigue tests were run': 3 words
    write_doc(tmp_path, 'pemfc_02', ['Journal of the Electrochemical Society OPEN ACCESS | '
                                     'Ink Solvent Dependence of the Ionomer Distribution'])
    assert ma.title_candidates(str(tmp_path / 'pemfc_02.conll')) == [
        'Ink Solvent Dependence of the Ionomer Distribution']


def test_queries_and_score():
    title = 'Study of 13Cr - 4Ni - ( Mo ) Steel Grade Heat Treatment for Maximum Hardness'
    first, longest = ma.queries(title)
    assert first == ('TITLE:("study" AND "13cr" AND "steel" AND "grade" AND "heat" AND '
                     '"treatment" AND "maximum" AND "hardness")')
    assert longest == ('TITLE:("treatment" AND "hardness" AND "maximum" AND "grade" AND '
                       '"steel")')
    assert ma.score(title, 'Study of 13Cr-4Ni-(Mo) Steel Grade Heat Treatment for Maximum '
                           'Hardness Control') == 0.89
    assert ma.score(title, 'Something else entirely') == 0.0


def test_run_with_a_fake_search(tmp_path):
    write_doc(tmp_path, 'sofc_02', ['Micro solid oxide fuel cell fabricated on porous '
                                    'stainless steel', 'A new micro - SOFC is designed .'])
    write_doc(tmp_path, 'steel_07', ['On the Influence of Control Type and Strain Rate on the '
                                     'Lifetime of 50CrMo4'])
    (tmp_path / 'subfields.json').write_text(json.dumps({'sofc_02': 'sofc', 'steel_07': 'steel'}))
    calls = []

    def search(q):
        calls.append(q)
        if '"micro"' in q and '"porous"' in q:
            return [{'pmcid': 'PMC4772004', 'pmid': '1', 'doi': '10.1/x',
                     'title': 'Micro solid oxide fuel cell fabricated on porous stainless '
                              'steel: a new strategy'}]
        return []
    found = ma.run(str(tmp_path), str(tmp_path / 'out.json'), search)
    assert found['sofc_02']['pmcid'] == 'PMC4772004' and found['sofc_02']['subfield'] == 'sofc'
    assert found['sofc_02']['score'] >= ma.MATCH_SCORE and found['steel_07'] is None
    assert json.load(open(tmp_path / 'out.json')) == found
    assert len([q for q in calls if '"50crmo4"' in q]) == 2       # both queries were tried
