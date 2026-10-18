"""Unseen-name coverage of held-out MuLMS subfields by unlabeled corpora (mulms_coverage.py), on
a toy corpus and toy folds."""
import json

import pytest

import mulms_coverage as mc

TOY = [(['steel', 'is', 'hard'], ['B-MAT', 'O', 'O'], 'a'),
       (['the', '42CrMo4', 'bar'], ['O', 'B-MAT', 'O'], 'a'),
       (['YSZ', 'anode', 'at', '750', '°C'], ['B-MAT', 'B-DEV', 'O', 'B-VALUE', 'I-VALUE'], 'b'),
       (['steel', 'and', 'La0.6Sr0.4CoO3', 'cathode'], ['B-MAT', 'O', 'B-MAT', 'B-DEV'], 'b'),
       (['YSZ', 'again'], ['B-MAT', 'O'], 'b'),
       (['the', 'La0.6Sr0.4Co0.2Fe0.8O3', 'cathode', 'was', 'screen', 'printed', 'onto', 'the',
         'electrolyte'], ['O', 'B-MAT', 'B-DEV', 'O', 'O', 'O', 'O', 'O', 'B-DEV'], 'b')]
LONG = 'the La0.6Sr0.4Co0.2Fe0.8O3 cathode was screen printed onto the electrolyte'


def test_parse_corpus():
    assert mc.parse_corpus('sofc=/x/sofc.txt') == ('sofc', ('/x/sofc.txt', False))
    assert mc.parse_corpus('fat=/x/f.txt:raw') == ('fat', ('/x/f.txt', True))
    with pytest.raises(Exception, match='NAME=PATH'):
        mc.parse_corpus('nonsense')


def test_occurrences_and_unseen():
    sentences = [['the', 'YSZ', 'anode'], ['YSZ', 'and', 'steel'], ['nothing']]
    found = mc.occurrences(sentences, {('YSZ',), ('steel',), ('YSZ', 'anode'), ('gone',)})
    assert found == {('YSZ',): {0, 1}, ('steel',): {1}, ('YSZ', 'anode'): {0}}
    mentions, unseen = mc.unseen_mentions(TOY, train=[0, 1], test=[2, 3, 4, 5])
    assert len(mentions) == 10
    assert unseen == [('MAT', ('YSZ',)), ('DEV', ('anode',)), ('VALUE', ('750', '°C')),
                      ('MAT', ('La0.6Sr0.4CoO3',)), ('DEV', ('cathode',)), ('MAT', ('YSZ',)),
                      ('MAT', ('La0.6Sr0.4Co0.2Fe0.8O3',)), ('DEV', ('cathode',)),
                      ('DEV', ('electrolyte',))]


def test_coverage_with_leakage_and_types(tmp_path, monkeypatch):
    corpus = tmp_path / 'topic.txt'
    corpus.write_text('The YSZ anode at 750 °C in a cell .\n'          # exact: YSZ, anode, 750 °C
                      'A La0.6Sr0.4CoO3 - δ cathode layer .\n'       # the MAT with a dash
                      'steel and La0.6Sr0.4CoO3 cathode were tested\n'  # too short to leak
                      'Unrelated words here .\n', encoding='utf-8')
    folds = {'b': ([0, 1], [2, 3, 4, 5])}
    monkeypatch.setattr(mc, 'mulms_folds', lambda results_dir: (TOY, folds))
    report = mc.analyse('unused', {'topic': (str(corpus), False)})
    b = report['subfields']['b']
    assert (b['n_test'], b['n_unseen'], b['unseen_share']) == (10, 9, 0.9)
    c = b['corpora']['topic']
    # no corpus sentence shares 50 characters with the toy documents, so nothing leaks
    assert c['leaked'] == 0
    # exact: YSZ (twice), anode, 750 °C, La0.6Sr0.4CoO3 and cathode (twice) = 7 of 9; the long
    # formula and 'electrolyte' are nowhere in the corpus
    assert c['exact'] == round(7 / 9, 3) and c['loose'] == round(7 / 9, 3)
    assert c['per_type'] == {'DEV': {'n_unseen': 4, 'exact': 0.75, 'loose': 0.75},
                             'MAT': {'n_unseen': 4, 'exact': 0.75, 'loose': 0.75},
                             'VALUE': {'n_unseen': 1, 'exact': 1.0, 'loose': 1.0}}
    assert report['corpora']['topic']['sentences'] == 4
    # a corpus that holds a test sentence leaks it and counts nothing from it
    leaking = tmp_path / 'leak.txt'
    leaking.write_text(f'{LONG}\nThe YSZ anode alone .\n', encoding='utf-8')
    report = mc.analyse('unused', {'leak': (str(leaking), False)})
    c = report['subfields']['b']['corpora']['leak']
    assert c['leaked'] == 1
    assert c['per_type']['MAT'] == {'n_unseen': 4, 'exact': 0.5, 'loose': 0.5}  # YSZ twice
    assert c['per_type']['DEV'] == {'n_unseen': 4, 'exact': 0.25, 'loose': 0.25}  # anode
    assert c['exact'] == round(3 / 9, 3)


def test_main_writes_the_report(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / 'c.txt'
    corpus.write_text('YSZ anode .\n', encoding='utf-8')
    monkeypatch.setattr(mc, 'mulms_folds', lambda results_dir: (TOY, {'b': ([0, 1], [2, 3, 4, 5])}))
    out = tmp_path / 'out.json'
    mc.main(['--corpus', f'c={corpus}', '--out', str(out)])
    saved = json.load(open(out))
    assert saved['subfields']['b']['corpora']['c']['exact'] == round(3 / 9, 3)
    assert 'b              90.0%' in capsys.readouterr().out
