"""NER majority-vote ensemble. Voting and validation are pure Python (no torch, seqeval or
scipy); the tests that score or summarise skip when seqeval or scipy is missing."""
import argparse
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

import ner_baseline as nb
import ner_ensemble as ne

SCRIPT = Path(__file__).resolve().parents[1] / 'src' / 'graph' / 'ner_ensemble.py'
ALPHABET = ['O', 'B-A', 'I-A', 'B-B', 'I-B']


def _tags(text):
    return text.split()


def _vote(*members):
    """(voted tags as text, invalid I- tags) for members given as tag text."""
    tags, invalid = ne.vote_tags([_tags(m) for m in members])
    return ' '.join(tags), invalid


# ---------------------------------------------------------------------------
# Spans <-> tags
# ---------------------------------------------------------------------------
def test_tags_to_spans():
    assert ne.tags_to_spans(_tags('O B-A I-A O B-B')) == ([(1, 3, 'A'), (4, 5, 'B')], 0)
    assert ne.tags_to_spans([]) == ([], 0)
    assert ne.tags_to_spans(_tags('O O')) == ([], 0)
    # B- always opens a new entity, also right after an entity of the same type
    assert ne.tags_to_spans(_tags('B-A I-A B-A B-B I-B')) == (
        [(0, 2, 'A'), (2, 3, 'A'), (3, 5, 'B')], 0)
    # an entity may end the sentence; type names may contain hyphens
    assert ne.tags_to_spans(_tags('O B-Foo-Bar I-Foo-Bar')) == ([(1, 3, 'Foo-Bar')], 0)


def test_tags_to_spans_rejects_malformed_tags():
    for bad in ('E-A', 'B-', 'A', 'I', 'BA'):
        with pytest.raises(ValueError, match='IOB2'):
            ne.tags_to_spans(['O', bad])


def test_spans_to_tags():
    assert ne.spans_to_tags([], 3) == ['O', 'O', 'O']
    assert ne.spans_to_tags([(3, 4, 'B'), (1, 3, 'A')], 5) == _tags('O B-A I-A B-B O')   # any order
    assert ne.spans_to_tags([(0, 2, 'A'), (2, 3, 'A')], 3) == _tags('B-A I-A B-A')
    for bad in ([(0, 0, 'A')], [(2, 4, 'A')], [(-1, 1, 'A')], [(0, 2, 'A'), (1, 3, 'B')],
                [(0, 2, 'A'), (0, 2, 'A')]):
        with pytest.raises(ValueError, match='span'):
            ne.spans_to_tags(bad, 3)


def test_spans_tags_round_trip_and_structure():
    """All tag sequences up to length 5: spans cover the entity tokens exactly once."""
    for n in range(6):
        for seq in itertools.product(ALPHABET, repeat=n):
            spans, invalid = ne.tags_to_spans(seq)
            assert [i for s, e, _ in spans for i in range(s, e)] == [
                i for i, t in enumerate(seq) if t != 'O']
            for s, e, kind in spans:   # a span is a maximal run: opener, then I-type tags
                assert seq[s] in (f'B-{kind}', f'I-{kind}')
                assert all(t == f'I-{kind}' for t in seq[s + 1:e])
                assert e == n or seq[e] != f'I-{kind}'
            assert len(spans) == sum(t.startswith('B-') for t in seq) + invalid
            # spans -> tags -> spans is the identity, and tags -> spans -> tags too if valid
            assert ne.tags_to_spans(ne.spans_to_tags(spans, n)) == (spans, 0)
            if not invalid:
                assert ne.spans_to_tags(spans, n) == list(seq)


def test_spans_match_seqeval_default_mode():
    sequence_labeling = pytest.importorskip('seqeval.metrics.sequence_labeling')
    for n in range(7):
        for seq in itertools.product(ALPHABET, repeat=n):
            expected = [(s, e + 1, kind)
                        for kind, s, e in sequence_labeling.get_entities(list(seq))]
            assert ne.tags_to_spans(seq)[0] == expected


# ---------------------------------------------------------------------------
# Invalid I- tags
# ---------------------------------------------------------------------------
def test_invalid_i_tags_start_a_new_entity_and_are_counted():
    # an I-X that does not continue an entity of type X opens one (conlleval)
    assert ne.tags_to_spans(_tags('I-A I-A O')) == ([(0, 2, 'A')], 1)          # sentence start
    assert ne.tags_to_spans(_tags('O I-A I-A')) == ([(1, 3, 'A')], 1)          # after O
    assert ne.tags_to_spans(_tags('B-A I-B I-B')) == ([(0, 1, 'A'), (1, 3, 'B')], 1)   # type change
    assert ne.tags_to_spans(_tags('B-A I-A O I-A B-B I-B I-A')) == (
        [(0, 2, 'A'), (3, 4, 'A'), (4, 6, 'B'), (6, 7, 'A')], 2)
    assert ne.tags_to_spans(_tags('B-A I-A B-B I-B')) == ([(0, 2, 'A'), (2, 4, 'B')], 0)


def test_vote_counts_invalid_i_tags_and_lets_their_entities_vote():
    # the leading I-A of the first member is the entity (0, 2, A), as for the second member
    assert _vote('I-A I-A O', 'B-A I-A O', 'O O O') == ('B-A I-A O', 1)
    # counted in every member; the voted tags are valid IOB2 again
    tags, invalid = _vote('O I-A I-A', 'O I-A I-A', 'O B-A I-A')
    assert (tags, invalid) == ('O B-A I-A', 2)
    assert ne.tags_to_spans(_tags(tags))[1] == 0
    assert _vote('B-A I-A I-A', 'B-A I-A I-A', 'B-A I-A I-A')[1] == 0


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------
K3 = {
    'all agree': (['B-A I-A O B-B I-B'] * 3, 'B-A I-A O B-B I-B'),
    '2 of 3 kept': (['B-A I-A O O O', 'B-A I-A O O O', 'O O O O O'], 'B-A I-A O O O'),
    '1 of 3 dropped': (['B-A I-A O O O', 'O O O O O', 'O O O O O'], 'O O O O O'),
    'type mismatch outvoted': (['B-A I-A O O O', 'B-B I-B O O O', 'B-A I-A O O O'],
                               'B-A I-A O O O'),
    'type mismatch, no majority': (['B-A I-A O O O', 'B-B I-B O O O', 'O O O O O'],
                                   'O O O O O'),
    'boundary mismatch outvoted': (['B-A I-A O O O', 'B-A I-A I-A O O', 'B-A I-A O O O'],
                                   'B-A I-A O O O'),
    'boundary mismatch, no majority': (['B-A I-A O O O', 'B-A I-A I-A O O', 'O B-A I-A O O'],
                                       'O O O O O'),
    'split entity outvoted': (['B-A I-A I-A O O', 'B-A I-A B-A O O', 'B-A I-A I-A O O'],
                              'B-A I-A I-A O O'),
    'two entities, each with 2 votes': (['B-A I-A O B-B I-B', 'B-A I-A O O O',
                                         'O O O B-B I-B'], 'B-A I-A O B-B I-B'),
    'adjacent entities of one type': (['B-A I-A B-A I-A O', 'B-A I-A B-A I-A O',
                                       'B-A I-A I-A I-A O'], 'B-A I-A B-A I-A O'),
}
K5 = {
    'all agree': (['O B-A I-A B-B O'] * 5, 'O B-A I-A B-B O'),
    '3 of 5 kept': (['B-A I-A O O O'] * 3 + ['O O O O O'] * 2, 'B-A I-A O O O'),
    '2 of 5 dropped': (['B-A I-A O O O'] * 2 + ['O O O O O'] * 3, 'O O O O O'),
    'type mismatch outvoted': (['B-A I-A O O O'] * 3 + ['B-B I-B O O O'] * 2, 'B-A I-A O O O'),
    'type mismatch, no majority': (['B-A I-A O O O'] * 2 + ['B-B I-B O O O'] * 2
                                   + ['O O O O O'], 'O O O O O'),
    'boundary mismatch outvoted': (['B-A I-A O O O'] * 3 + ['B-A I-A I-A O O'] * 2,
                                   'B-A I-A O O O'),
    'boundary mismatch, no majority': (['B-A I-A O O O'] * 2 + ['B-A I-A I-A O O'] * 2
                                       + ['O B-A I-A O O'], 'O O O O O'),
    'one entity reaches 3, the other 2': (['B-A I-A O B-B I-B'] * 2 + ['B-A I-A O O O']
                                          + ['O O O O O'] * 2, 'B-A I-A O O O'),
    'four supporters, one dissent': (['B-A I-A O O O'] * 4 + ['O O B-A I-A O'],
                                     'B-A I-A O O O'),
}


@pytest.mark.parametrize('members, expected', list(K3.values()), ids=list(K3))
def test_majority_vote_of_three(members, expected):
    assert _vote(*members) == (expected, 0)


@pytest.mark.parametrize('members, expected', list(K5.values()), ids=list(K5))
def test_majority_vote_of_five(members, expected):
    assert _vote(*members) == (expected, 0)


def test_majority_spans():
    a, b, c = (0, 2, 'A'), (0, 3, 'A'), (0, 2, 'B')
    assert ne.majority_spans([[a], [a], []]) == [a]
    assert ne.majority_spans([[a], [], []]) == []
    assert ne.majority_spans([[a], [b], [c]]) == []
    assert ne.majority_spans([[a, (3, 4, 'B')], [a], [(3, 4, 'B')]]) == [a, (3, 4, 'B')]
    # a member counts once, even if its list repeats a span
    assert ne.majority_spans([[a, a], [], []]) == []
    # k = 5: 3 supporters are enough, 2 are not
    assert ne.majority_spans([[a], [a], [a], [b], [b]]) == [a]
    assert ne.majority_spans([[a], [a], [b], [b], []]) == []


def test_even_number_of_members_needs_a_strict_majority():
    assert _vote('B-A I-A', 'B-A I-A') == ('B-A I-A', 0)            # k = 2: both must agree
    assert _vote('B-A I-A', 'O O') == ('O O', 0)
    four = ['B-A I-A O'] * 2 + ['O O O'] * 2
    assert _vote(*four) == ('O O O', 0)                             # k = 4: 2 of 4 is a tie
    assert _vote(*(['B-A I-A O'] * 3 + ['O O O'])) == ('B-A I-A O', 0)   # 3 of 4


def test_vote_is_symmetric_in_the_members():
    members = ['B-A I-A O B-B', 'B-A O O B-B', 'O B-A I-A B-B', 'B-A I-A O O', 'I-A I-A O B-B']
    expected = _vote(*members)
    for perm in itertools.permutations(members):
        assert _vote(*perm) == expected


def test_vote_on_random_members_equals_a_direct_count():
    """Noisy copies of a base sequence (invalid I- tags included): kept spans are exactly the
    ones with > k/2 supporters, never overlap, and the written tags read back to them."""
    rng = random.Random(0)
    kept_something = 0
    for _ in range(400):
        k = rng.randint(2, 7)
        base = [rng.choice(ALPHABET) for _ in range(rng.randint(0, 9))]
        members = [[t if rng.random() > 0.3 else rng.choice(ALPHABET) for t in base]
                   for _ in range(k)]
        parsed = [ne.tags_to_spans(m) for m in members]
        candidates = {span for spans, _ in parsed for span in spans}
        expected = sorted(s for s in candidates if 2 * sum(s in spans for spans, _ in parsed) > k)
        covered = [i for s, e, _ in expected for i in range(s, e)]
        assert len(covered) == len(set(covered))   # no token in two kept entities
        tags, invalid = ne.vote_tags(members)
        assert ne.tags_to_spans(tags) == (expected, 0)
        assert invalid == sum(bad for _, bad in parsed)
        kept_something += bool(expected)
    assert kept_something > 100   # the test is not vacuous


def test_vote_tags_edge_cases():
    assert _vote('', '', '') == ('', 0)   # a sentence without tokens
    with pytest.raises(ValueError, match='different numbers'):
        ne.vote_tags([['O', 'O'], ['O']])


# ---------------------------------------------------------------------------
# Variant and file names
# ---------------------------------------------------------------------------
def test_variant_naming():
    assert ne.ensemble_variant(['crf_m0', 'crf_m100', 'crf_m200']) == 'crf_ens3'
    assert ne.ensemble_variant([f'crf_m{n}' for n in range(5)]) == 'crf_ens5'
    assert ne.ensemble_variant(['cw_m0', 'cw_m100']) == 'cw_ens2'
    assert ne.ensemble_variant(['crf', 'crf_m100']) == 'crf_ens2'      # untagged member
    assert ne.ensemble_variant(['m0', 'm100', 'm200']) == 'ens3'       # tagged baseline runs
    # only the trailing tag is a seed; members with later seeds name the first one
    assert ne.ensemble_variant(['crf_m0_m0', 'crf_m0_m6']) == 'crf_m0_ens2'
    assert ne.ensemble_variant(['crf_m0_m5', 'crf_m0_m6']) == 'crf_m0_ens2_m5'
    for bad in (['crf_m0', 'cw_m100'], ['crf_m0', 'm100'], ['crf_m0', 'crf_x1']):
        with pytest.raises(ne.EnsembleError, match='variant'):
            ne.ensemble_variant(bad)


def test_votes_over_later_seed_triples_get_files_of_their_own():
    """Week 5: the votes over m300-m500 and m600-m800 are <family>_ens3_m300 and _m600."""
    assert ne.ensemble_variant(['crf_m300', 'crf_m400', 'crf_m500']) == 'crf_ens3_m300'
    assert ne.ensemble_variant(['crf_m600', 'crf_m700', 'crf_m800']) == 'crf_ens3_m600'
    assert ne.ensemble_variant(['crf_mt_m300', 'crf_mt_m400', 'crf_mt_m500']) == \
        'crf_mt_ens3_m300'
    assert ne.ensemble_variant(['crf_mt', 'crf_mt_m100', 'crf_mt_m200']) == 'crf_mt_ens3'
    assert ne.ensemble_variant(['crf_st3_m300', 'crf_st3_m400', 'crf_st3_m500']) == \
        'crf_st3_ens3_m300'
    assert ne.ensemble_variant(['m300', 'm400', 'm500']) == 'ens3_m300'
    assert [ne.member_offset(v) for v in ('crf_mt', 'crf_m0', 'crf_ls_m900', 'm300', 'crf')] == \
        [0, 0, 900, 300, 0]
    # ner_seedmeans names the vote families with the same files
    import ner_seedmeans as ns
    for stem in ('crf', 'crf_mt', 'crf_ls', 'crf_st3'):
        members = ns.WEEK5_SYSTEMS[stem]
        votes = [ne.ensemble_variant(members[k:k + 3]) for k in (0, 3, 6)]
        assert tuple(votes) == ns.WEEK5_SYSTEMS[f'{stem}_ens3']


def test_variant_option_names():
    for name in ('crf_mix3', 'mix3', 'crf_mix3_v2', 'crf_ens3'):
        assert ne.variant_option(name) == name
    # not a plain variant of ner_baseline.py, not a seed's name, and usable in a file name
    for bad in ('crf', 'crf_mt', 'cw', 'baseline', 'crf_m100', 'm0', 'crf_mix3_m5', 'CRF_mix3',
                'crf-mix3', '_mix3', 'mix3_', 'crf__mix3', 'crf mix3', ''):
        with pytest.raises(argparse.ArgumentTypeError, match='invalid --variant'):
            ne.variant_option(bad)


def test_results_file_names_follow_ner_baseline():
    assert ne.results_name('fine', 'random_split_vary', 'crf_ens3') == \
        'ner_results_fine_random_split_vary_crf_ens3.json'
    assert ne.results_name('coarse', 'leave_one_paper_out', 'crf_ens3') == \
        'ner_results_coarse_leave_one_paper_out_crf_ens3.json'
    # the fixed split has no protocol part
    assert ne.results_name('fine', 'random_split_fixed', 'crf_ens3') == \
        'ner_results_fine_crf_ens3.json'


# ---------------------------------------------------------------------------
# Synthetic member files (the contract of ner_baseline.py --save-predictions)
# ---------------------------------------------------------------------------
GOLD = {'seed0': ['B-A I-A O B-B', 'O O B-A', 'B-B I-B O O'],
        'seed1': ['O B-A I-A O', 'B-B O O', 'B-A I-A I-A']}
INDEX = {'seed0': [4, 0, 7], 'seed1': [2, 5, 3]}
PREDS = {   # seed0: m1 has an invalid I-A; the vote recovers the gold; seed1: the vote errs once
    'm0': {'seed0': ['B-A I-A O B-B', 'O O O', 'B-B I-B O O'],
           'seed1': ['O B-A I-A O', 'B-B B-A O', 'B-A I-A O']},
    'm1': {'seed0': ['B-A I-A O O', 'O O I-A', 'B-A O O O'],
           'seed1': ['O B-A I-A B-B', 'B-A O O', 'B-A I-A I-A']},
    'm2': {'seed0': ['O O O B-B', 'O O B-A', 'B-B I-B O O'],
           'seed1': ['O O B-A I-A', 'B-B O O', 'B-A I-A O']},
}
NAMES = [f'ner_results_fine_random_split_vary_crf_m{n}.json' for n in (0, 100, 200)]


def _member(variant, preds, scored=False):
    """A results file with predictions; `scored` fills in real seqeval metrics."""
    runs = []
    for run_id in GOLD:
        run = {'run_id': run_id, 'seed': 0, 'f1': 50.0, 'f1_strict': 50.0,
               'test_index': list(INDEX[run_id]), 'test_gold': list(GOLD[run_id]),
               'test_pred': list(preds[run_id])}
        if scored:
            run.update(nb.score([g.split() for g in GOLD[run_id]],
                                [p.split() for p in preds[run_id]]))
        runs.append(run)
    return {'granularity': 'fine', 'protocol': 'random_split_vary', 'variant': variant,
            'model': 'toy/model', 'labels': ['A', 'B'], 'label_tie': {}, 'paper_test_f1': 69.92,
            'config': {}, 'runs': runs}


def _trio(scored=False):
    return [_member(f'crf_m{n}', PREDS[m], scored) for n, m in zip((0, 100, 200), PREDS)]


def _write(tmp_path, members, names=NAMES):
    paths = []
    for member, name in zip(members, names):
        (tmp_path / name).write_text(json.dumps(member))
        paths.append(str(tmp_path / name))
    return paths


def _fails(members, match):
    with pytest.raises(ne.EnsembleError, match=match) as excinfo:
        ne.combine(members, NAMES[:len(members)])
    return str(excinfo.value)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_different_run_ids_are_rejected():
    members = _trio()
    members[1]['runs'][1]['run_id'] = 'seed7'
    msg = _fails(members, 'run_ids')
    assert 'seed1' in msg and 'seed7' in msg and NAMES[1] in msg
    members = _trio()
    del members[2]['runs'][1]
    _fails(members, 'run_ids')


def test_different_test_sentences_are_rejected():
    members = _trio()
    members[2]['runs'][0]['test_index'] = [4, 0, 8]
    msg = _fails(members, 'test_index')
    assert 'seed0' in msg and NAMES[2] in msg and 'same split' in msg


def test_different_gold_tags_are_rejected():
    members = _trio()
    members[1]['runs'][1]['test_gold'][0] = 'O O O O'
    _fails(members, 'test_gold')


def test_missing_predictions_are_rejected():
    members = _trio()
    del members[1]['runs'][0]['test_pred']
    msg = _fails(members, 'test_pred')
    assert NAMES[1] in msg and 'seed0' in msg and '--save-predictions' in msg
    members = _trio()
    members[0]['runs'][1]['test_pred'] = None
    _fails(members, 'test_pred')
    members = _trio()
    del members[2]['runs'][0]['f1']
    _fails(members, 'no f1')


@pytest.mark.parametrize('key, value', [
    ('granularity', 'coarse'), ('protocol', 'leave_one_paper_out'), ('model', 'other/model'),
    ('labels', ['A', 'C'])])
def test_members_must_share_granularity_protocol_model_and_labels(key, value):
    members = _trio()
    members[2][key] = value
    msg = _fails(members, f'differ in {key}')
    assert NAMES[2] in msg


def test_label_mismatch_names_the_classes():
    members = _trio()
    members[1]['labels'] = ['A', 'C']
    msg = _fails(members, 'labels')
    assert "['B']" in msg and "['C']" in msg
    members[1]['labels'] = ['B', 'A']
    _fails(members, 'different order')


def test_variants_must_agree_once_the_tag_is_removed():
    members = _trio()
    members[1]['variant'] = 'cw_m100'
    _fails(members, 'variant')


def test_an_ensemble_needs_two_members():
    _fails(_trio()[:1], 'at least 2')


def test_member_files_must_look_like_results_files():
    members = _trio()
    del members[1]['labels']
    assert "'labels'" in _fails(members, 'missing')
    members = _trio()
    members[0]['runs'] = []
    _fails(members, 'run_ids')
    members = _trio()
    members[1]['runs'][1]['run_id'] = 'seed0'   # repeated run_id
    _fails(members, 'unique run_ids')
    members = _trio()
    del members[2]['runs'][0]['run_id']
    _fails(members, 'run_ids')


def test_prediction_shape_and_tags_are_checked():
    members = _trio()
    members[1]['runs'][0]['test_pred'] = members[1]['runs'][0]['test_pred'][:2]
    _fails(members, 'differ in length')
    members = _trio()
    members[1]['runs'][1]['test_pred'][2] = 'B-A I-A'    # 2 tags for 3 gold tags
    msg = _fails(members, 'sentence 2')
    assert NAMES[1] in msg and 'seed1' in msg
    members = _trio()
    members[2]['runs'][0]['test_pred'][1] = 'O O E-A'
    msg = _fails(members, 'IOB2')
    assert NAMES[2] in msg and 'E-A' in msg
    members = _trio()
    members[0]['runs'][0]['test_pred'] = [None, None, None]
    _fails(members, 'strings')
    members = _trio()
    for m in members:
        m['runs'][0].update(test_index=[], test_gold=[], test_pred=[])
    _fails(members, 'no test sentences')


def test_run_order_may_differ_between_members():
    members = _trio()
    members[1]['runs'].reverse()
    by_id = [ne.index_runs(m, n) for m, n in zip(members, NAMES)]
    ne.check_runs(by_id, NAMES)   # no error


MIXED = [('crf_m0', 'm3rg-iitd/matscibert'), ('crf_scibert', 'allenai/scibert_scivocab_uncased'),
         ('crf_pubmedbert', 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')]
MIXED_NAMES = [f'ner_results_fine_random_split_vary_{variant}.json' for variant, _ in MIXED]


def _mixed(scored=False):
    """The trio's predictions as three backbones: the variants and models differ."""
    members = _trio(scored)
    for member, (variant, model) in zip(members, MIXED):
        member['variant'], member['model'] = variant, model
    return members


def test_mixed_members_need_a_variant_name():
    with pytest.raises(ne.EnsembleError, match='differ in model'):
        ne.combine(_mixed(), MIXED_NAMES)
    members = _mixed()
    for member in members:
        member['model'] = 'toy/model'
    with pytest.raises(ne.EnsembleError, match='variant'):     # not the seeds of one variant
        ne.combine(members, MIXED_NAMES)


def test_a_named_ensemble_keeps_every_other_check():
    """--variant lifts the model and variant checks only."""
    def fails(members, match):
        with pytest.raises(ne.EnsembleError, match=match):
            ne.combine(members, MIXED_NAMES, 'crf_mix3')

    for key, value in (('granularity', 'coarse'), ('protocol', 'leave_one_paper_out'),
                       ('labels', ['A', 'C'])):
        members = _mixed()
        members[2][key] = value
        fails(members, f'differ in {key}')
    members = _mixed()
    members[1]['runs'][1]['run_id'] = 'seed7'
    fails(members, 'run_ids')
    members = _mixed()
    members[2]['runs'][0]['test_index'] = [4, 0, 8]
    fails(members, 'test_index')
    members = _mixed()
    members[1]['runs'][1]['test_gold'][0] = 'O O O O'
    fails(members, 'test_gold')
    members = _mixed()
    del members[0]['runs'][0]['test_pred']
    fails(members, 'test_pred')
    fails(_mixed()[:1], 'at least 2')


def test_identical_members_are_reported():
    trio = _trio()
    trio[1]['runs'] = [dict(r) for r in trio[0]['runs']]
    by_id = [ne.index_runs(m, n) for m, n in zip(trio, NAMES)]
    assert ne.identical_members(by_id, NAMES) == [(NAMES[0], NAMES[1])]
    trio = _trio()
    by_id = [ne.index_runs(m, n) for m, n in zip(trio, NAMES)]
    assert ne.identical_members(by_id, NAMES) == []


# ---------------------------------------------------------------------------
# End to end: scoring, results schema, command line
# ---------------------------------------------------------------------------
def test_end_to_end_results_schema(tmp_path, capsys):
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    members = _trio(scored=True)
    paths = _write(tmp_path, members)

    ne.main(['--members', *paths])

    out = tmp_path / 'ner_results_fine_random_split_vary_crf_ens3.json'   # default location
    assert out.exists() and sorted(p.name for p in tmp_path.iterdir()) == sorted(
        NAMES + [out.name])
    res = json.loads(out.read_text())
    # fields copied from the members, in the order of ner_baseline.py
    assert list(res)[:9] == ['granularity', 'protocol', 'variant', 'model', 'labels',
                             'label_tie', 'paper_test_f1', 'config', 'runs']
    assert (res['granularity'], res['protocol'], res['variant']) == (
        'fine', 'random_split_vary', 'crf_ens3')
    assert (res['model'], res['labels'], res['label_tie'], res['paper_test_f1']) == (
        'toy/model', ['A', 'B'], {}, 69.92)
    assert res['config'] == {'members': NAMES, 'voting': 'entity_strict_majority', 'k': 3}
    # one entry per run_id: seqeval metrics plus the ensemble fields
    assert [r['run_id'] for r in res['runs']] == ['seed0', 'seed1']
    for run, i in zip(res['runs'], (0, 1)):
        assert {'f1', 'precision', 'recall', 'f1_strict', 'per_class', 'n_members', 'member_f1',
                'member_f1_strict', 'invalid_i_tags'} <= set(run)
        assert run['n_members'] == 3
        assert run['member_f1'] == [m['runs'][i]['f1'] for m in members]
        assert run['member_f1_strict'] == [m['runs'][i]['f1_strict'] for m in members]
        assert {'macro avg', 'weighted avg', 'micro avg'} <= set(run['per_class'])
    seed0, seed1 = res['runs']
    assert seed0['member_f1'] == [85.71, 57.14, 85.71] and seed0['member_f1_strict'][1] == 33.33
    assert (seed0['f1'], seed0['precision'], seed0['recall'], seed0['f1_strict']) == (
        100.0, 100.0, 100.0, 100.0)
    assert seed0['per_class']['A']['support'] == 2 and seed0['invalid_i_tags'] == 1
    assert seed1['member_f1'] == [57.14, 57.14, 33.33]
    assert (seed1['f1'], seed1['precision'], seed1['recall']) == (66.67, 66.67, 66.67)
    assert seed1['invalid_i_tags'] == 0
    # summary fields of ner_baseline.summarize
    s = res['summary']
    assert s['n_runs'] == 2 and s['run_ids'] == ['seed0', 'seed1']
    assert s['micro']['per_run'] == [100.0, 66.67]
    assert s['micro']['mean'] == pytest.approx(83.335, abs=0.01)
    assert res['test_f1_mean'] == pytest.approx(83.335, abs=0.01)
    assert {'micro', 'micro_strict', 'macro', 'weighted'} <= set(s)
    assert s['micro']['ci95'][0] < s['micro']['mean'] < s['micro']['ci95'][1]
    # the vote minus the mean of its members, run by run: 100 - 76.19, 66.67 - 49.20
    gain = s['vs_member_mean']
    assert gain['per_run'] == [23.81, 17.47] and gain['mean_diff'] == 20.64
    assert gain['ci95'][0] < gain['mean_diff'] < gain['ci95'][1]
    # the printed report: a line per run and the micro F1 with its CI
    printed = capsys.readouterr().out
    assert 'seed0: ensemble F1 100.00' in printed and 'members 85.71 57.14 85.71' in printed
    assert 'seed1: ensemble F1 66.67' in printed and '1 invalid I- tags' in printed
    assert 'micro F1 83.3' in printed and '95% CI' in printed and 'paper 69.92' in printed
    # the paired comparison of ner_baseline works against a single member
    comp = nb.compare_to_baseline(members[0], res)
    assert comp['n_pairs'] == 2 and comp['run_ids'] == ['seed0', 'seed1']
    assert comp['micro']['per_run'] == pytest.approx([14.29, 9.53], abs=0.01)
    assert set(comp) >= {'micro', 'micro_strict', 'macro', 'weighted'}


def test_ensemble_of_identical_members_reproduces_the_member(tmp_path, capsys):
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    members = [_member(f'crf_m{n}', PREDS['m0'], scored=True) for n in (0, 100, 200)]
    ne.main(['--members', *_write(tmp_path, members)])
    res = json.loads((tmp_path / 'ner_results_fine_random_split_vary_crf_ens3.json').read_text())
    for run, own in zip(res['runs'], members[0]['runs']):
        for key in ('f1', 'precision', 'recall', 'f1_strict', 'per_class'):
            assert run[key] == own[key]
        assert run['member_f1'] == [own['f1']] * 3
    assert res['summary']['vs_member_mean'] == {'mean_diff': 0.0, 'ci95': [0.0, 0.0],
                                                'per_run': [0.0, 0.0]}
    warnings = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith('warning:')]
    assert len(warnings) == 3 and NAMES[0] in warnings[0]   # all three pairs are identical


def test_out_option_and_member_order(tmp_path):
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    paths = _write(tmp_path, _trio(scored=True))
    out = tmp_path / 'ensembles' / 'new' / 'vote.json'
    ne.main(['--members', *reversed(paths), '--out', str(out)])
    res = json.loads(out.read_text())
    assert not (tmp_path / 'ner_results_fine_random_split_vary_crf_ens3.json').exists()
    assert res['config']['members'] == NAMES[::-1]
    assert res['runs'][0]['member_f1'] == [85.71, 57.14, 85.71]   # member order is kept
    assert res['runs'][1]['member_f1'] == [33.33, 57.14, 57.14]
    assert [r['f1'] for r in res['runs']] == [100.0, 66.67]       # the vote is order-free


def test_other_protocols_and_two_members(tmp_path):
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    members = _trio(scored=True)[:2]
    for m in members:
        m['protocol'] = 'random_split_fixed'
        m['granularity'] = 'coarse'
    ne.main(['--members', *_write(tmp_path, members)])
    res = json.loads((tmp_path / 'ner_results_coarse_crf_ens2.json').read_text())
    assert res['config']['k'] == 2 and res['runs'][0]['n_members'] == 2
    # with two members both must agree: seed0 keeps only A(0,2), which m0 and m1 share
    assert res['runs'][0]['recall'] == 25.0


def test_mixed_backbone_ensemble(tmp_path, capsys):
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    paths = _write(tmp_path, _mixed(scored=True), MIXED_NAMES)
    ne.main(['--members', *paths, '--variant', 'crf_mix3'])
    out = tmp_path / 'ner_results_fine_random_split_vary_crf_mix3.json'   # named by --variant
    assert nb.parse_result_name(out.name) == ('fine', 'random_split_vary', 'crf_mix3')
    res = json.loads(out.read_text())
    assert (res['variant'], res['model']) == ('crf_mix3', 'mixed')
    assert res['config'] == {'members': MIXED_NAMES, 'voting': 'entity_strict_majority', 'k': 3,
                             'models': [model for _, model in MIXED]}
    # the vote and its scores do not depend on the names: those of the seed ensemble
    seeds = ne.combine(_trio(scored=True), NAMES)
    assert res['runs'] == seeds['runs'] and res['summary'] == seeds['summary']
    assert 'models: m3rg-iitd/matscibert, allenai/scibert_scivocab_uncased, microsoft/' in \
        capsys.readouterr().out


def test_a_named_ensemble_of_one_model_keeps_its_name():
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    res = ne.combine(_trio(scored=True), NAMES, 'crf_seeds3')
    assert (res['variant'], res['model']) == ('crf_seeds3', 'toy/model')
    assert res['config']['models'] == ['toy/model'] * 3
    assert 'models' not in ne.combine(_trio(scored=True), NAMES)['config']   # without --variant


def test_command_line_errors(tmp_path, capsys):
    members = _trio()
    members[1]['runs'][0]['test_index'] = [9, 9, 9]
    paths = _write(tmp_path, members)
    with pytest.raises(SystemExit) as excinfo:
        ne.main(['--members', *paths])
    assert str(excinfo.value).startswith('error: ') and 'test_index' in str(excinfo.value)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(NAMES)   # nothing written

    with pytest.raises(SystemExit) as excinfo:   # k >= 2
        ne.main(['--members', paths[0]])
    assert excinfo.value.code == 2 and 'at least 2' in capsys.readouterr().err

    with pytest.raises(SystemExit) as excinfo:   # an ensemble must not take a single model's name
        ne.main(['--members', *paths, '--variant', 'crf'])
    assert excinfo.value.code == 2 and 'invalid --variant' in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(NAMES)

    with pytest.raises(SystemExit, match='cannot read'):
        ne.main(['--members', paths[0], str(tmp_path / 'missing.json')])
    (tmp_path / 'broken.json').write_text('{not json')
    with pytest.raises(SystemExit, match='broken.json'):
        ne.main(['--members', paths[0], str(tmp_path / 'broken.json')])
    (tmp_path / 'list.json').write_text('[]')
    with pytest.raises(SystemExit, match='not a ner_baseline'):
        ne.main(['--members', paths[0], str(tmp_path / 'list.json')])


def test_out_must_not_overwrite_a_member(tmp_path):
    pytest.importorskip('seqeval')
    pytest.importorskip('scipy')
    paths = _write(tmp_path, _trio(scored=True))
    before = Path(paths[0]).read_text()
    with pytest.raises(SystemExit, match='one of the members'):
        ne.main(['--members', *paths, '--out', paths[0]])
    assert Path(paths[0]).read_text() == before


def test_script_exits_with_a_message_not_a_traceback(tmp_path):
    members = _trio()
    members[2]['runs'][1]['run_id'] = 'seed9'
    proc = subprocess.run([sys.executable, str(SCRIPT), '--members', *_write(tmp_path, members),
                           '--out', str(tmp_path / 'out.json')],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 1
    assert proc.stderr.startswith('error: ') and 'run_ids' in proc.stderr
    assert 'Traceback' not in proc.stderr and not (tmp_path / 'out.json').exists()
