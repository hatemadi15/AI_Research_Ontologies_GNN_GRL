"""Document folds in ner_baseline.py (--document-folds K; ROADMAP §4, week 8): groups of whole
documents held out, each the test set of one fold. The folds and the parser need no torch."""
import itertools
from types import SimpleNamespace

import pytest

import ner_baseline as nb

# five documents of 2 to 4 sentences
DATA = [([f'{doc}{k}w{j}' for j in range(3)], ['O'] * 3, f'{doc}.conll')
        for doc, n in (('a', 4), ('b', 3), ('c', 2), ('d', 3), ('e', 2)) for k in range(n)]


def _args(k, split_seed=0):
    return SimpleNamespace(leave_one_paper_out=False, document_folds=k, split_seed=split_seed)


def test_document_groups_deal_every_document_once():
    docs = [f'doc{i}' for i in range(45)]
    groups = nb.document_groups(set(docs), 5, 0)
    assert len(groups) == 5 and all(len(g) == 9 for g in groups)
    assert sorted(itertools.chain(*groups)) == sorted(docs)
    assert all(list(g) == sorted(g) for g in groups)
    assert nb.document_groups(docs, 5, 0) == groups               # the input order is ignored
    assert nb.document_groups(docs, 5, 1) != groups               # the split seed shuffles
    assert [len(g) for g in nb.document_groups(docs[:7], 3, 0)] == [3, 2, 2]
    for k in (1, 46):
        with pytest.raises(SystemExit, match='between 2 and the 45 documents'):
            nb.document_groups(docs, k, 0)


def test_document_folds():
    data = [(w, t, d) for k in range(5) for w, t, d in DATA]       # 70 sentences
    folds = nb.make_folds(data, _args(2), indices=True)
    groups = nb.document_groups({d for _, _, d in data}, 2, 0)
    assert [name for name, *_ in folds] == ['fold0', 'fold1']
    for (name, train, val, test), group in zip(folds, groups):
        assert {data[i][2] for i in test} == set(group) and test == sorted(test)
        assert {data[i][2] for i in train + val} == {d for _, _, d in data} - set(group)
        assert len(val) == round(0.15 * len(train + val)) and not set(train) & set(val)
    # a group of one document is that document's leave-one-paper-out fold
    one = nb.make_folds(data, _args(5), indices=True)
    lopo = {name: rest for name, *rest in
            nb.make_folds(data, SimpleNamespace(leave_one_paper_out=True, split_seed=0), True)}
    for name, *rest in one:
        test_doc = data[rest[2][0]][2]
        assert rest == lopo[test_doc], name


def test_parser_and_names():
    args = nb.parse_args(['--granularity', 'sofc', '--document-folds', '5', '--crf', '--tag',
                          'm100', '--seed-offset', '100'])
    assert nb.protocol_name(args) == 'document_folds' and nb.held_out_protocol(args)
    assert nb.stored_config(args)['document_folds'] == 5
    assert 'document_folds' not in nb.stored_config(nb.parse_args(['--crf']))   # unchanged
    assert not nb.held_out_protocol(nb.parse_args(['--crf', '--vary-split']))
    assert nb.parse_result_name('ner_results_sofc_document_folds_crf_m0.json') == \
        ('sofc', 'document_folds', 'crf_m0')
    for argv in (['--document-folds', '1'], ['--document-folds', '5', '--vary-split'],
                 ['--document-folds', '5', '--leave-one-paper-out'],
                 ['--document-folds', '5', '--leave-two-papers-out'],
                 ['--document-folds', '5', '--published-split']):
        with pytest.raises(SystemExit):
            nb.parse_args(['--crf', *argv])


def test_sofc_exps_article_folds():
    """SOFC-Exp's 45 articles in 5 groups of 9: every annotated sentence is tested once."""
    data, _, _ = nb.granularity_data('sofc', SimpleNamespace(top_k=27, published_split=False,
                                                             retokenized_test=False))
    folds = nb.make_folds(data, _args(5), indices=True)
    assert len(data) == 876 and len({d for _, _, d in data}) == 45
    tested = sorted(itertools.chain(*(test for *_, test in folds)))
    assert tested == list(range(876))
    assert [len({data[i][2] for i in test}) for *_, test in folds] == [9] * 5


def test_two_paper_size_folds():
    """--two-paper-size (week 8, part 2): per held-out paper and pair of the other papers, the
    paper's fold with as many training sentences as the pair's --leave-two-papers-out fold."""
    data = [(w, t, d) for k in range(5) for w, t, d in DATA if d != 'e.conll']   # 4 papers
    args = nb.parse_args(['--leave-one-paper-out', '--crf', '--two-paper-size', '--tag', 'm0'])
    folds = nb.make_folds(data, args, indices=True)
    lopo = {name: rest for name, *rest in nb.make_folds(
        data, SimpleNamespace(leave_one_paper_out=True, split_seed=0), True)}
    two = {name: rest for name, *rest in nb.make_folds(
        data, SimpleNamespace(leave_one_paper_out=False, leave_two_papers_out=True,
                              split_seed=0), True)}
    assert len(folds) == 4 * 3
    for name, train, val, test in folds:
        paper, pair = name.split('|')
        q, r = pair.split('+')
        third = ({'a.conll', 'b.conll', 'c.conll', 'd.conll'} - {paper, q, r}).pop()
        full_train, full_val, full_test = lopo[paper]
        assert (val, test) == (full_val, full_test)
        assert train == full_train[:len(two['+'.join(sorted((paper, third)))][0])]
    assert nb.variant_name(args) == 'crf_p3n2_m0'
    one = nb.parse_args(['--leave-one-paper-out', '--crf', '--two-paper-size', '--papers', 'a'])
    assert {name.split('|')[0] for name, *_ in nb.make_folds(data, one, True)} == {'a.conll'}
    for argv in (['--crf', '--two-paper-size'], ['--leave-one-paper-out', '--two-paper-size'],
                 ['--leave-one-paper-out', '--crf', '--two-paper-size', '--multi-task'],
                 ['--leave-one-paper-out', '--crf', '--two-paper-size', '--context', '8'],
                 ['--leave-one-paper-out', '--crf', '--tag', 'p3n2']):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
