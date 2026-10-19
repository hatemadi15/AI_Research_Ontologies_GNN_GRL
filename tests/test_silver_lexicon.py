"""The CPU probe's pure parts (silver_lexicon.py): the lexicons of both filters at several
thresholds, the dictionary rule in its two modes, the strict scorer and the probe over stored
runs."""
import silver_lexicon as sl


def test_lexicons_of_both_filters_at_thresholds():
    sentences = [['YSZ', 'anode', '.'], ['the', 'LSCF', 'cathode'], ['nothing', 'here']]
    # teacher votes: (tags, least marginal, marginals); sentence 0 is confident everywhere,
    # sentence 1 only on its mention, sentence 2 has no entity
    votes = [[(('B-MAT', 'O', 'O'), 0.95, (0.95, 0.99, 0.99))] * 3,
             [(('O', 'B-MAT', 'O'), 0.6, (0.6, 0.92, 0.75))] * 3,
             [(('O', 'O'), 0.99, (0.99, 0.99))] * 3]
    out = sl.lexicons_from_votes(sentences, votes, thresholds=(0.9, 0.5))
    assert out['sentences_0.9'] == [('MAT', ('YSZ',))]
    assert out['mentions_0.9'] == [('MAT', ('YSZ',)), ('MAT', ('LSCF',))]
    assert out['sentences_0.5'] == [('MAT', ('YSZ',)), ('MAT', ('LSCF',))]
    # a disagreeing teacher removes the sentence and the span
    votes[0][2] = (('O', 'O', 'O'), 0.99, (0.99, 0.99, 0.99))
    out = sl.lexicons_from_votes(sentences, votes, thresholds=(0.5,))
    assert out['sentences_0.5'] == [('MAT', ('LSCF',))] and out['mentions_0.5'] == [('MAT', ('LSCF',))]


def test_dictionary_rule_fill_and_override():
    index = sl.lexicon_index([('MAT', ('YSZ',)), ('MAT', ('La', 'Sr', 'Co')), ('DEV', ('Co',)),
                              ('MAT', ('La', 'Sr', 'Co')), ('DEV', ('cell',))])
    assert index[('La', 'Sr', 'Co')] == 'MAT' and index[('Co',)] == 'DEV'
    words = ['the', 'La', 'Sr', 'Co', 'cell', 'and', 'YSZ', 'again']
    tags = ['O', 'O', 'O', 'B-DEV', 'I-DEV', 'O', 'O', 'O']
    # fill: the longest match wins where every word is O; 'La Sr Co' overlaps the DEV
    # prediction at 'Co', so only 'Co'... no: 'Co' is not O either; nothing from it; YSZ fills
    assert sl.apply_lexicon(words, tags, index, 'fill') == [
        'O', 'O', 'O', 'B-DEV', 'I-DEV', 'O', 'B-MAT', 'O']
    # override: 'La Sr Co' is tagged MAT and the cut DEV entity reopens at 'cell'
    assert sl.apply_lexicon(words, tags, index, 'override') == [
        'O', 'B-MAT', 'I-MAT', 'I-MAT', 'B-DEV', 'O', 'B-MAT', 'O']
    assert sl.apply_lexicon(words, tags, {}, 'fill') == tags


def test_strict_scores_and_probe():
    gold = [['B-MAT', 'O', 'B-DEV'], ['B-MAT', 'I-MAT', 'O']]
    pred = [['B-MAT', 'O', 'O'], ['B-MAT', 'O', 'O']]
    assert sl.strict_prf(gold, pred) == {'precision': 50.0, 'recall': 33.33, 'f1': 40.0}
    data = [(['YSZ', 'is', 'cell'], gold[0], 'a'), (['La', 'Sr', 'x'], gold[1], 'a')]
    run = {'run_id': 's', 'test_index': [0, 1], 'test_gold': [' '.join(g) for g in gold],
           'test_pred': [' '.join(p) for p in pred]}
    index = sl.lexicon_index([('DEV', ('cell',)), ('MAT', ('La', 'Sr'))])
    out = sl.probe([(0, run), (100, run)], data, index)
    assert out['fill']['n_seeds'] == 2 and out['fill']['f1_before'] == 40.0
    # 'cell' fills the DEV; 'La Sr' overlaps the predicted 'La' entity, so fill leaves it
    assert out['fill']['per_seed']['m0']['after'] == {'precision': 66.67, 'recall': 66.67,
                                                      'f1': 66.67}
    assert out['override']['per_seed']['m100']['after']['f1'] == 100.0
    assert out['override']['delta_f1'] == 60.0
