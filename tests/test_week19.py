"""Week 19 (ROADMAP §4, the week-19 protocol): the candidate dump (--save-candidates), the
external names (--names-file), the probe (graph_names.py) and the harvest batch (w19a)."""
import gzip
import hashlib
import json

import numpy as np
import pytest

import graph_names as gn
import kaggle_jobs as kj
import ner_baseline as nb
from test_ner_self_training import RESULT_KEYS, _corpus, _load
from test_ner_self_training import tiny_model, tiny_run  # noqa: F401  (fixtures)


# --- the dump --------------------------------------------------------------------------------

LABELS = ['O', 'B-MAT', 'I-MAT', 'B-PROP', 'I-PROP']
ID2LABEL = dict(enumerate(LABELS))


def _top(rows):
    """teacher_tags' topk part from per-word [(label, prob), ...]."""
    ids = np.array([[LABELS.index(lab) for lab, _ in row] for row in rows], dtype='int16')
    probs = np.array([[p for _, p in row] for row in rows], dtype='float32')
    return ids, probs


def test_span_type_scores_take_the_least_marginal_over_the_span():
    top = _top([[('B-MAT', 0.6), ('B-PROP', 0.3), ('O', 0.1)],
                [('I-MAT', 0.5), ('I-PROP', 0.4), ('O', 0.1)],
                [('O', 0.9), ('I-MAT', 0.05), ('B-MAT', 0.05)]])
    scores = nb.span_type_scores(top, 0, 2, ID2LABEL)
    assert scores == pytest.approx({'MAT': 0.5, 'PROP': 0.3})
    # a label outside a word's kept labels counts 0: I-PROP is not among word 2's
    assert nb.span_type_scores(top, 0, 3, ID2LABEL) == pytest.approx({'MAT': 0.05, 'PROP': 0.0})


def test_candidate_spans_hold_the_union_with_each_teachers_view():
    words = ['steel', 'alloy', 'strength']
    sure = _top([[('B-MAT', 0.95), ('B-PROP', 0.04)], [('I-MAT', 0.9), ('O', 0.1)],
                 [('B-PROP', 0.8), ('O', 0.2)]])
    teacher_a = (['B-MAT', 'I-MAT', 'B-PROP'], [0.95, 0.9, 0.8], sure)
    teacher_b = (['B-MAT', 'O', 'O'], [0.7, 0.6, 0.55], _top(
        [[('B-MAT', 0.7), ('B-PROP', 0.2)], [('O', 0.6), ('I-MAT', 0.3)],
         [('O', 0.55), ('B-PROP', 0.45)]]))
    spans = nb.candidate_spans([words], [[teacher_a], [teacher_b]], ID2LABEL)
    # the union of the Viterbi spans, sorted by start and end
    assert [(s, e, w) for _, s, e, w, _ in spans] == [(0, 1, ['steel']), (0, 2, ['steel', 'alloy']),
                                                      (2, 3, ['strength'])]
    (_, _, _, _, one), (_, _, _, _, two), (_, _, _, _, three) = spans
    # (0, 1): only b tags it, MAT at 0.7; a's best other types there are scored
    assert one[1][:2] == ['MAT', 0.7] and one[0][:2] == [None, None]
    assert one[0][2] == [['MAT', 0.95], ['PROP', 0.04]]
    # (0, 2): a's MAT with its exact least marginal; the tagged type is not among "others"
    assert two[0][:2] == ['MAT', 0.9] and all(t != 'MAT' for t, _ in two[0][2])
    assert three[0][:2] == ['PROP', 0.8] and three[1][:2] == [None, None]


def _random_votes(rng, sentences, teachers):
    """Random teachers' teacher_tags(..., topk) over toy sentences."""
    out = []
    for _ in range(teachers):
        votes = []
        for words in sentences:
            tags, last = [], 'O'
            for _ in words:
                choice = rng.choice(['O', 'B-MAT', 'B-PROP', 'I'])
                tag = (f'I-{last[2:]}' if choice == 'I' and last != 'O' else
                       ('O' if choice == 'I' else choice))
                tags.append(tag)
                last = tag
            marginals = [float(rng.choice([0.4, 0.6, 0.75, 0.95])) for _ in words]
            votes.append((tags, marginals, _top([[(t, m), ('O', 1 - m)] if t != 'O' else
                                                 [(t, m), ('B-MAT', 1 - m)]
                                                 for t, m in zip(tags, marginals)])))
        out.append(votes)
    return out


@pytest.mark.parametrize('teachers', [2, 3])
def test_the_vote_rebuilt_from_the_dump_is_silver_mention_names(teachers):
    """V at any threshold and agreement, rebuilt from the dump, equals silver_mention_names
    on the same votes, order included (the replay check depends on it)."""
    rng = np.random.default_rng(0)
    sentences = [['w%d' % rng.integers(5) for _ in range(rng.integers(1, 7))] for _ in range(60)]
    votes = _random_votes(rng, sentences, teachers)
    spans = nb.candidate_spans(sentences, votes, ID2LABEL)
    per_sentence = [[(tuple(t), min(m), tuple(m)) for t, m, _ in (v[i] for v in votes)]
                    for i in range(len(sentences))]
    for threshold in (0.5, 0.7, 0.9):
        assert gn.vote_names(spans, threshold, teachers) == \
            nb.silver_mention_names(sentences, per_sentence, threshold)
        for agree in range(1, teachers + 1):
            assert gn.vote_names(spans, threshold, agree) == \
                nb.silver_mention_names(sentences, per_sentence, threshold, agree)


def test_teacher_tags_keep_each_words_most_probable_labels(tiny_model):  # noqa: F811
    torch = pytest.importorskip('torch')
    import transformers
    from types import SimpleNamespace
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_model)
    vocab = tokenizer.get_vocab()
    table = torch.zeros(len(vocab), len(LABELS))
    for word, tag, value in (('steel', 'B-MAT', 10), ('high', 'O', 10), ('room', 'O', 5),
                             ('room', 'B-MAT', 5)):
        table[vocab[word], LABELS.index(tag)] = value

    class TableModel(torch.nn.Module):
        config = SimpleNamespace(max_position_embeddings=64)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=table[input_ids])

    tagger = SimpleNamespace(model=TableModel(), crf=nb.make_crf(LABELS), tokenizer=tokenizer,
                             label2id={t: i for i, t in enumerate(LABELS)}, id2label=ID2LABEL)
    sentences = [['steel', 'high'], ['steel', 'room']]
    plain = nb.teacher_tags(tagger, sentences, 2, 'cpu')
    full = nb.teacher_tags(tagger, sentences, 2, 'cpu', topk=4)
    assert [(t, m) for t, m, _ in full] == plain           # the tags and marginals unchanged
    ids, probs = full[1][2]
    assert ids.shape == probs.shape == (2, 4) and ids.dtype == np.int16
    assert {LABELS[j] for j in ids[1][:2]} == {'O', 'B-MAT'}  # room: a toss-up, both kept
    assert probs[0][0] == pytest.approx(full[1][1][0], abs=1e-6)   # steel: its tag first


# --- --names-file ------------------------------------------------------------------------------

def test_names_file_options(tmp_path, capsys):
    path = tmp_path / 'names.json'
    path.write_text(json.dumps({'runs': {}}))
    base = ['--granularity', 'fine', '--leave-one-paper-out', '--crf', '--augment-mentions', '6']
    args = nb.parse_args([*base, '--names-file', str(path), '--names-arm', 'g'])
    assert nb.variant_name(args) == 'crf_mrg6'
    assert nb.variant_name(nb.parse_args([*base, '--names-file', str(path), '--names-arm', 'v',
                                          '--tag', 'm300'])) == 'crf_mrv6_m300'
    for extra, error in ((['--names-arm', 'g'], '--names-arm needs --names-file'),
                         (['--names-file', str(path)], 'needs --names-arm'),
                         (['--names-file', str(path), '--names-arm', 'm'], 'other than m and s'),
                         (['--names-file', str(path), '--names-arm', 'gg'], 'one lowercase'),
                         (['--names-file', str(tmp_path / 'none.json'), '--names-arm', 'g'],
                          'no such file')):
        with pytest.raises(SystemExit):
            nb.parse_args([*base, *extra])
        assert error in capsys.readouterr().err
    with pytest.raises(SystemExit):
        nb.parse_args(['--granularity', 'fine', '--crf', '--names-file', str(path),
                       '--names-arm', 'g'])
    assert 'needs --crf and --augment-mentions' in capsys.readouterr().err


def test_names_of_run_keeps_the_files_order():
    store = {'runs': {'a': {'names': [['MAT', 'Ti 6Al'], ['PROP', 'yield'], ['MAT', 'Ti 6Al']]}}}
    assert nb.names_of_run(store, 'a') == [('MAT', ('Ti', '6Al')), ('PROP', ('yield',)),
                                           ('MAT', ('Ti', '6Al'))]
    with pytest.raises(SystemExit, match='no names for run b'):
        nb.names_of_run(store, 'b')


def test_save_candidates_needs_save_names(tmp_path, capsys):
    corpus = _corpus(tmp_path)
    with pytest.raises(SystemExit):
        nb.parse_args(['--granularity', 'fine', '--crf', '--self-train', str(corpus),
                       '--teachers', '2', '--augment-mentions', '2', '--augment-names',
                       '--save-candidates', 'c.json.gz'])
    assert '--save-candidates needs --save-names' in capsys.readouterr().err


def test_the_replay_of_a_harvested_lexicon_reproduces_the_names_run(tiny_run,  # noqa: F811
                                                                      tmp_path):
    """The Phase-1 replay check, locally: a names run with --save-names and --save-candidates;
    its names rebuilt from the dump (V at its threshold, every teacher agreeing) and given
    back through --names-file train a run equal to the names run's student, its predictions
    included."""
    corpus = _corpus(tmp_path)
    flags = ('--vary-split', '--seeds', '0')
    tiny_run('--crf', '--self-train', str(corpus), '--teachers', '2', '--self-train-sample', '0',
             '--augment-mentions', '2', '--augment-names', '--silver-mentions',
             '--silver-threshold', '0.01', *flags, '--save-predictions',
             '--save-names', 'names.json', '--save-candidates', 'cands.json.gz')
    with gzip.open(tmp_path / 'ner' / 'cands.json.gz', 'rt', encoding='utf-8') as f:
        dump = json.load(f)
    record = dump['runs']['seed0']
    assert 'O' in record['labels'] and record['n_sentences'] == 5
    assert [t['seed'] for t in record['teachers']] == [0, 100] and record['spans']
    names = gn.vote_names(record['spans'], 0.01, 2)
    stored = _load(tmp_path, 'names.json')['runs']['seed0']
    assert names and gn.names_check({gn.SECONDARY: names}, stored)['equal']
    store = gn.names_file({'seed0': names}, 'f', 'test')
    path = tmp_path / 'replay.json'
    path.write_text(json.dumps(store))
    names_run = _load(tmp_path, 'ner_results_fine_random_split_vary_crf_mrm2_m0.json')['runs'][0]
    tiny_run('--crf', '--augment-mentions', '2', *flags, '--tag', 'm0', '--save-predictions',
             '--names-file', str(path), '--names-arm', 'f')
    res = _load(tmp_path, 'ner_results_fine_random_split_vary_crf_mrf2_m0.json')
    replay = res['runs'][0]
    keys = (*RESULT_KEYS, 'test_pred', 'n_augmented')
    assert {k: replay[k] for k in keys} == {k: names_run[k] for k in keys}
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert replay['names_file']['sha256'] == res['config']['names_file_sha256'] == sha
    assert replay['names_file']['coverage'] == names_run['self_training']['names']


# --- graph_names --------------------------------------------------------------------------------

def _span(i, s, e, words, *teachers):
    return [i, s, e, list(words), [list(t) for t in teachers]]


def _toy_dump():
    """Three teachers over four sentences; 'Ti6Al4V' and 'Ti6Al7Nb' alike in form."""
    sure, unsure = ['MAT', 0.95, [['PROP', 0.01]]], ['MAT', 0.6, [['PROP', 0.3]]]
    none = [None, None, [['MAT', 0.2]]]
    return [
        _span(0, 0, 1, ['steel'], sure, sure, sure),
        _span(0, 2, 3, ['Ti6Al4V'], sure, sure, unsure),
        _span(1, 0, 1, ['Ti6Al7Nb'], unsure, none, unsure),
        _span(1, 3, 4, ['hardness'], ['PROP', 0.9, []], ['PROP', 0.8, []], none),
        _span(2, 0, 1, ['steel'], sure, sure, sure),
        _span(2, 1, 3, ['yield', 'strength'], ['PROP', 0.95, []], ['PROP', 0.95, []],
              ['PROP', 0.95, []]),
        _span(3, 0, 1, ['Ti6Al7Nb'], unsure, unsure, ['PROP', 0.5, []]),
    ]


TRAIN = [(['steel', 'and', 'yield', 'strength'], ['B-MAT', 'O', 'B-PROP', 'I-PROP']),
         (['steel'], ['B-PROP'])]           # steel: MAT once, PROP once: ties by type name


def test_build_graph_nodes_labels_features_and_edges():
    graph = gn.build_graph(_toy_dump(), LABELS, TRAIN)
    forms = graph['forms']
    assert forms == [('steel',), ('Ti6Al4V',), ('Ti6Al7Nb',), ('hardness',),
                     ('yield', 'strength')]
    assert graph['types'] == ['MAT', 'PROP']
    # labelled: the training names, majority type, ties by type name (steel: MAT < PROP)
    assert graph['labelled'] == {0: 0, 4: 1}
    assert graph['new'] == [1, 2, 3]
    assert graph['masks'][2] == ['MAT', 'PROP']     # some teacher tags Ti6Al7Nb as PROP once
    vote = graph['features'][:, graph['vote_cols']]
    # Ti6Al7Nb: 6 span-teacher pairs, 4 tagged MAT, 1 PROP; mean MAT score
    # (0.6 * 4 + 0.2) / 6 and PROP (0.3 * 4 + 0.5) / 6
    assert vote[2] == pytest.approx([4 / 6, 1 / 6, (0.6 * 4 + 0.2) / 6, (0.3 * 4 + 0.5) / 6])
    assert graph['edges']['cooccurrence'] == gn.symmetric(graph['edges']['cooccurrence'])
    assert all(i < j for i, j in graph['edges']['knn'])
    hom = gn.homophily(graph)
    assert set(hom) == {'cooccurrence', 'knn', 'head'}


def test_ppmi_keeps_the_strongest_positive_pairs():
    sentences = [{0, 1}, {0, 1}, {2, 3}, {0, 2}]
    pairs = gn.ppmi_edges(sentences, 4, top=1)
    assert (0, 1) in pairs and (1, 0) in pairs and (2, 3) in pairs
    # 0 and 2 co-occur once but each is frequent: PMI log(1*4/(3*2)) < 0, no edge
    assert (0, 2) not in pairs


def test_knn_and_symmetric():
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    pairs = gn.knn_edges(vectors, k=1)
    assert sorted(pairs) == [(0, 1), (1, 0), (2, 3), (3, 2)]
    assert gn.symmetric(pairs + [(2, 2)]) == [(0, 1), (2, 3)]


def test_head_groups_and_their_mean_aggregation():
    torch = pytest.importorskip('torch')
    forms = [('tensile', 'strength'), ('yield', 'strength'), ('strength',), ('steel',)]
    groups = gn.head_groups(forms, limit=3)
    assert groups == [[0, 1, 2]]
    assert gn.head_groups(forms, limit=2) == []
    graph = {'edges': {'cooccurrence': [], 'knn': []}, 'heads': groups}
    _, _, head = gn.aggregators(graph, 4)
    h = torch.tensor([[1.0], [2.0], [4.0], [8.0]])
    assert head(h).tolist() == [[3.0], [2.5], [1.5], [0.0]]   # the mean over the others


def test_ranked_lexicon_masks_types_breaks_ties_and_takes_new_names_only():
    graph = gn.build_graph(_toy_dump(), LABELS, TRAIN)
    probs = np.zeros((5, 2))
    probs[1] = [0.7, 0.3]      # Ti6Al4V: MAT (only MAT is in its mask anyway)
    probs[2] = [0.4, 0.6]      # Ti6Al7Nb: PROP is in its mask, and more probable
    probs[3] = [0.9, 0.1]      # hardness: MAT more probable, but no teacher tags it MAT
    lex = gn.ranked_lexicon(graph, probs, 3)
    assert lex == {('Ti6Al4V',): 'MAT', ('Ti6Al7Nb',): 'PROP', ('hardness',): 'PROP'}
    # ranked by the admitted type's probability: Ti6Al4V (0.7) first
    assert gn.ranked_lexicon(graph, probs, 1) == {('Ti6Al4V',): 'MAT'}
    probs[2] = [0.7, 0.3]      # a tie at 0.7 (Ti6Al7Nb now MAT): broken by the words
    assert gn.ranked_lexicon(graph, probs, 1) == {('Ti6Al4V',): 'MAT'}


def test_mention_entries_one_per_supported_mention_in_corpus_order():
    lexicon = {('Ti6Al7Nb',): 'PROP', ('Ti6Al4V',): 'MAT', ('nothing',): 'MAT'}
    entries = gn.mention_entries(_toy_dump(), lexicon)
    # Ti6Al4V once (sentence 0); Ti6Al7Nb as PROP only where some teacher tags PROP (3)
    assert entries == [('MAT', ('Ti6Al4V',)), ('PROP', ('Ti6Al7Nb',))]


def test_coverage_counts_rates_and_the_ceiling():
    unseen = [('MAT', ('Ti6Al7Nb',)), ('PROP', ('Ti6Al7Nb',)), ('MAT', ('x',)),
              ('PROP', ('y',))]
    lexicon = {('Ti6Al7Nb',): 'MAT', ('y',): 'MAT'}
    counts = gn.coverage_counts(lexicon, unseen, {('Ti6Al7Nb',): {'MAT', 'PROP'},
                                                  ('y',): {'PROP'}})
    assert counts == {'unseen': 4, 'covered': 1, 'held': 3, 'wrong': 2, 'ceiling': 3}
    assert gn.rates(counts) == {'coverage': 25.0, 'wrong_type': 66.67}
    assert gn.rates({'unseen': 0, 'covered': 0, 'held': 0, 'wrong': 0}) == \
        {'coverage': None, 'wrong_type': None}


def test_the_gate_needs_all_three_parts():
    def unit(g, v, m, gw, vw):
        return {'primary': {'G': {'coverage': g, 'wrong_type': gw},
                            'V': {'coverage': v, 'wrong_type': vw},
                            'M': {'coverage': m, 'wrong_type': 0.0}}}
    passing = {'a': unit(40, 30, 35, 20, 18), 'b': unit(30, 26, 27, 10, 9)}
    g = gn.gate(passing)
    assert g['passes'] is True and g['means']['coverage_G'] == 35.0
    assert gn.gate({'a': unit(40, 36, 35, 20, 18)})['parts']['coverage_vs_vote'] is False
    assert gn.gate({'a': unit(40, 30, 39, 20, 18)})['parts']['coverage_vs_mlp'] is False
    assert gn.gate({'a': unit(40, 30, 35, 30, 18)})['parts']['wrong_type_vs_vote'] is False
    assert gn.gate({'a': unit(None, 30, 35, None, 18)})['passes'] is None


def test_rankers_are_deterministic_and_train_with_and_without_the_vote():
    pytest.importorskip('torch')
    graph = gn.build_graph(_toy_dump(), LABELS, TRAIN)
    nodes = sorted(graph['labelled'])
    for use_graph in (False, True):
        for use_vote in (True, False):
            a = gn.train_ranker(graph, use_graph, use_vote, nodes)
            b = gn.train_ranker(graph, use_graph, use_vote, nodes)
            assert a.shape == (5, 2) and np.array_equal(a, b)
            assert np.allclose(a.sum(axis=1), 1.0, atol=1e-5)
    lp = gn.label_propagation(graph)
    assert lp[0].tolist() == [1.0, 0.0] and lp[4].tolist() == [0.0, 1.0]   # clamped


def test_probe_unit_reports_every_ranker_at_both_sizes():
    pytest.importorskip('torch')
    unseen = [('MAT', ('Ti6Al7Nb',)), ('PROP', ('hardness',))]
    report, lexicons, votes = gn.probe_unit(_toy_dump(), LABELS, TRAIN, unseen)
    # V at 0.7, 3 of 3 admits steel (a training name, so not new) and yield strength
    # (training too) and Ti6Al4V? no: its third teacher is at 0.6. So N primary is 0 here
    assert report['sizes'] == {'primary': 0, 'secondary': 0}
    assert set(report['primary']) == {'V', 'M', 'G', 'M_novote', 'G_novote', 'LP'}
    assert all(r['size'] == 0 for r in report['primary'].values())
    assert report['ceiling'] == 2 and len(report['vote_grid']) == 9
    assert votes[(0.5, 2)] and report['vote_grid']['0.5_1']['size'] >= 1
    assert ('G', 'primary') in lexicons


def test_names_file_store_and_the_names_check():
    names = [('MAT', ('Ti', '6Al')), ('PROP', ('yield',)), ('MAT', ('Ti', '6Al'))]
    store = gn.names_file({'u': names}, 'v', 'test')
    assert store['runs']['u']['names'] == [['MAT', 'Ti 6Al'], ['PROP', 'yield'],
                                           ['MAT', 'Ti 6Al']]
    assert nb.names_of_run(store, 'u') == names              # the round trip keeps the order
    record = {'names': [['MAT', 'Ti 6Al', 2], ['PROP', 'yield', 1]]}
    assert gn.names_check({gn.SECONDARY: names}, record)['equal'] is True
    record['names'][0][2] = 3
    assert gn.names_check({gn.SECONDARY: names}, record)['equal'] is False


# --- the harvest batch and its collection ------------------------------------------------------

def test_week19_harvest_batch(monkeypatch):
    lane0, lane1 = kj.batch_lanes('w19a')
    assert lane0[0][-2:] == ['--tag', 'smokebf'] and len(lane0) == 8 and len(lane1) == 4
    with monkeypatch.context() as m:
        m.setattr(nb.os.path, 'isfile', lambda path: True)    # the corpus is not here
        parsed = [nb.parse_args(job) for job in lane0[1:] + lane1]
    for job, subfield in zip(parsed[:7], kj.MULMS_SUBFIELD_NAMES):
        assert job.save_candidates == f'ner_candidates_mulms_{subfield}.json.gz'
        assert job.save_names == ('ner_names_mulms_leave_one_subfield_out_crf_mrm_topic6_'
                                  f'{subfield}_w19.json')
        assert (job.students, job.teachers, job.silver_mentions) == (0, 3, True)
    for job, paper in zip(parsed[7:], kj.PAPERS):
        assert job.granularity == 'fine' and job.papers == [paper]
        assert job.save_candidates == 'ner_candidates_fine_leave_one_paper_out.json.gz'
        assert job.save_names == 'ner_names_fine_leave_one_paper_out_crf_mrm6_w19.json'


def test_collect_copies_the_dumps_apart_from_the_names(tmp_path):
    pulled = tmp_path / 'pulled' / 'ner'
    pulled.mkdir(parents=True)
    (pulled / 'ner_names_x_w19.json').write_text('{}')
    (pulled / 'ner_candidates_mulms_sofc.json.gz').write_bytes(gzip.compress(b'{}'))
    names, dumps = tmp_path / 'names', tmp_path / 'dumps'
    copied = kj.collect_names(tmp_path / 'pulled', names, dumps)
    assert sorted(copied) == ['ner_candidates_mulms_sofc.json.gz', 'ner_names_x_w19.json']
    assert (names / 'ner_names_x_w19.json').exists()
    assert (dumps / 'ner_candidates_mulms_sofc.json.gz').exists()


def test_teachers_check_against_the_stored_names_runs(tmp_path):
    def results(offset, f1s):
        run = {'run_id': 'sofc', 'test_pred': [],      # stored runs keep their predictions
               'self_training': {'teachers': [{'test_f1': f} for f in f1s]}}
        return {'runs': [run]}
    for offset in range(300, 1000, 100):
        (tmp_path / f'ner_results_mulms_leave_one_subfield_out_crf_mrm_topic6_m{offset}.json'
         ).write_text(json.dumps(results(offset, [50.1, 49.8, 51.0])))
    teachers = [{'test_f1': 50.1}, {'test_f1': 49.8}, {'test_f1': 51.0}]
    check = gn.teachers_check(teachers, 'mulms', 'sofc', tmp_path)
    assert check == {'equal': True, 'runs_checked': 7, 'rebuilt_teachers_f1': [50.1, 49.8, 51.0]}
    teachers[2]['test_f1'] = 51.1
    assert gn.teachers_check(teachers, 'mulms', 'sofc', tmp_path)['equal'] is False
    assert gn.teachers_check(teachers, 'mulms', 'steel', tmp_path)['equal'] is False
