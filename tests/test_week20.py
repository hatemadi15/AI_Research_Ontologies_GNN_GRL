"""Week 20 (ROADMAP §4, the week-20 protocol): the sentence pool of MuLMS's held-out subfields
(--pool-by-sentence, crf_spool), its rebuild and its seen and unseen names in ner_mentions.py,
its structural numbers on the real corpus, batch w20a, and the week-20 section of
ner_seed_analysis.py dry-run on the stored files with a fake family. The parser and the folds
need no torch; the runs with a tiny random model need torch, transformers, seqeval and scipy,
and the analysis scipy (skipped without them)."""
import copy
import json
import os
import random
import statistics
from pathlib import Path

import pytest

import decision_tests as dt
import kaggle_jobs as kj
import metrics
import ner_baseline as nb
import ner_mentions as nm
import ner_seed_analysis as sa
from test_ner_mulms_folds import toy_corpus
from test_ner_self_training import tiny_model  # noqa: F401  (fixture)

GPU = Path(__file__).resolve().parents[1] / 'results' / 'gpu'
HELD = ['--granularity', 'mulms', '--leave-one-subfield-out', '--train-pool', '400']
SPOOL = ['--pool-by-sentence', '--crf']
# MuLMS's 7 held-out subfields: the sentence pools' documents (counted with the rule of the
# protocol before any run; each from the 6 other subfields)
SPOOL_DOCUMENTS = {'electrolysis': 46, 'graphene': 48, 'pemfc': 42, 'poly': 39, 'semi': 39,
                   'sofc': 46, 'steel': 40}


# --- --pool-by-sentence -------------------------------------------------------------------------

def test_pool_by_sentence_options(capsys):
    args = nb.parse_args([*HELD, *SPOOL, '--tag', 'm300', '--seed-offset', '300',
                          '--save-predictions'])
    assert (nb.protocol_name(args), nb.variant_name(args)) == ('leave_one_subfield_out',
                                                               'crf_spool_m300')
    assert nb.variant_name(nb.parse_args([*HELD, *SPOOL])) == 'crf_spool'
    assert nb.results_path('out', 'mulms', args) == os.path.join(
        'out', 'ner_results_mulms_leave_one_subfield_out_crf_spool_m300.json')
    config = nb.stored_config(args)
    assert (config['pool_by_sentence'], config['train_pool'], config['seed_offset']) == (
        True, 400, 300)
    assert 'crf_spool' in nb.PLAIN_VARIANTS
    # without the flag no option stores it
    for argv in ([*HELD, '--crf'], [*HELD, '--crf', '--augment-mentions', '6', '--entity-bias', '1'],
                 ['--granularity', 'mulms', '--vary-split', '--train-pool', '400', '--crf'],
                 ['--crf'], ['--leave-one-paper-out', '--crf', '--two-paper-size'],
                 ['--leave-two-papers-out', '--crf', '--entity-bias', '1']):
        assert 'pool_by_sentence' not in nb.stored_config(nb.parse_args(argv))
    needs = 'needs --leave-one-subfield-out and --train-pool'
    for argv, error in (
            (['--granularity', 'mulms', '--leave-one-subfield-out', *SPOOL], needs),
            (['--granularity', 'mulms', '--vary-split', '--train-pool', '400', *SPOOL], needs),
            (['--granularity', 'fine', '--leave-one-paper-out', *SPOOL], needs),
            (['--granularity', 'fine', '--vary-split', *SPOOL], needs),
            (SPOOL, needs),
            ([*HELD, '--pool-by-sentence'], 'needs --crf'),
            ([*HELD, '--pool-by-sentence', '--class-weights'], 'needs --crf'),
            ([*HELD, *SPOOL, '--augment-mentions', '3'], 'not available with --augment-mentions'),
            ([*HELD, *SPOOL, '--entity-bias', '1'], 'not available with --entity-bias'),
            ([*HELD, *SPOOL, '--teachers', '3', '--self-train', 'unused'],
             'not available with --self-train'),
            # a tag must not take the family's name: crf_spool is a plain variant
            ([*HELD, '--crf', '--tag', 'spool'], 'would overwrite'),
            ([*HELD, '--crf', '--tag', 'spool_m100'], 'would overwrite')):
        with pytest.raises(SystemExit):
            nb.parse_args(argv)
        assert error in capsys.readouterr().err


def test_sentence_pools_take_the_other_subfields_sentences_in_a_shuffled_order(monkeypatch):
    """Fold i: the other subfields' sentences in corpus order, shuffled with random.Random(split
    seed + i), the first N; their split into training and validation (15%, split seed) and the
    test set as with the article pool."""
    data, subfields = toy_corpus()
    folds = nb.subfield_folds(data, subfields, split_seed=0, pool=5, indices=True,
                              by_sentence=True)
    articles = nb.subfield_folds(data, subfields, split_seed=0, pool=5, indices=True)
    assert [name for name, *_ in folds] == ['alpha', 'beta', 'gamma']
    for i, ((name, train, val, test), article) in enumerate(zip(folds, articles)):
        assert test == article[3] == [j for j, x in enumerate(data) if subfields[x[2]] == name]
        rest = [j for j, x in enumerate(data) if subfields[x[2]] != name]
        random.Random(i).shuffle(rest)
        pool = rest[:5]
        shuffled = list(range(5))
        random.Random(0).shuffle(shuffled)
        assert val == [pool[k] for k in shuffled[:1]] and train == [pool[k] for k in shuffled[1:]]
        assert nb.pool_record(data, train + val, subfields) == {
            'n_documents': len({data[j][2] for j in pool}),
            'n_subfields': len({subfields[data[j][2]] for j in pool})}
    seeded = nb.subfield_folds(data, subfields, 3, 5, indices=True, by_sentence=True)
    rest = [j for j, x in enumerate(data) if subfields[x[2]] != 'beta']
    random.Random(3 + 1).shuffle(rest)
    assert sorted(seeded[1][1] + seeded[1][2]) == sorted(rest[:5])
    as_data = nb.subfield_folds(data, subfields, 0, 5, by_sentence=True)
    assert as_data[0][1] == [data[j] for j in folds[0][1]]
    with pytest.raises(ValueError, match='needs its size'):
        nb.subfield_folds(data, subfields, 0, by_sentence=True)
    with pytest.raises(SystemExit, match='hold only'):
        nb.subfield_folds(data, subfields, 0, pool=100, by_sentence=True)
    monkeypatch.setattr(nb, 'mulms_subfields', lambda folder=None: subfields)
    args = nb.parse_args(['--granularity', 'mulms', '--leave-one-subfield-out', '--train-pool',
                          '5', *SPOOL])
    assert nb.make_folds(data, args, indices=True) == folds
    plain = nb.parse_args(['--granularity', 'mulms', '--leave-one-subfield-out', '--train-pool',
                           '5', '--crf'])
    assert nb.make_folds(data, plain, indices=True) == articles


def test_the_sentence_pools_of_the_real_corpus():
    """The structural numbers registered before any run: each sentence pool draws on 46, 48,
    42, 39, 39, 46 and 40 documents (electrolysis to steel), each of all 6 other subfields;
    the article pools on 2-3 documents of 1-3 subfields (decision_tests.MULMS_POOLS); the
    test sets are the same. Only the training side's document IDs are read."""
    if not os.path.isfile(os.path.join(nb.MULMS_DIR, 'subfields.json')):
        pytest.skip('MuLMS not converted here (python src/graph/mulms.py)')
    data = nb.load_split_corpus('mulms')
    subfields = nb.mulms_subfields()
    folds = nb.subfield_folds(data, subfields, 0, 400, indices=True, by_sentence=True)
    articles = nb.subfield_folds(data, subfields, 0, 400, indices=True)
    pools = {name: nb.pool_record(data, tr + va, subfields) for name, tr, va, _ in folds}
    assert {name: p['n_documents'] for name, p in pools.items()} == SPOOL_DOCUMENTS
    assert all(p['n_subfields'] == 6 for p in pools.values())
    assert all((len(tr), len(va)) == (340, 60) for _, tr, va, _ in folds)
    for (name, tr, va, te), article in zip(folds, articles):
        record = nb.pool_record(data, article[1] + article[2], subfields)
        assert 2 <= record['n_documents'] <= 3 and 1 <= record['n_subfields'] <= 3
        assert record['n_documents'] == len(dt.MULMS_POOLS[name])
        assert te == article[3]


# --- runs with a tiny random model ---------------------------------------------------------------

def _toy():
    """6 documents of 3 subfields, 3 sentences each (words of the tiny model's vocabulary)."""
    data = [([w, 'alloy', 'high'] + ['high'] * (k % 3), ['B-MAT', 'I-MAT', 'O'] + ['O'] * (k % 3),
             f'{"aabbcc"[k // 3]}{k // 3}.conll') for k, w in enumerate(['steel', 'yield'] * 9)]
    subfields = {f'{"aabbcc"[i]}{i}.conll': {'a': 'alpha', 'b': 'beta', 'c': 'gamma'}["aabbcc"[i]]
                 for i in range(6)}
    return data, subfields


@pytest.fixture
def mulms_run(tiny_model, tmp_path, monkeypatch):  # noqa: F811
    """run(*flags) on the toy MuLMS corpus in tmp_path/ner; returns the results file."""
    data, subfields = _toy()
    monkeypatch.setattr(nb, 'PROCESSED_DIR', str(tmp_path))
    monkeypatch.setattr(nb, 'load_split_corpus', lambda granularity: data)
    monkeypatch.setattr(nb, 'mulms_subfields', lambda folder=None: subfields)
    monkeypatch.setattr(nm, 'mulms_subfields', lambda: subfields)

    def run(*flags):
        args = nb.parse_args(['--granularity', 'mulms', '--model', tiny_model, '--epochs', '1',
                              '--threads', '1', '--device', 'cpu', *flags])
        nb.run(args)
        return json.loads(Path(nb.results_path(str(tmp_path / 'ner'), 'mulms', args)).read_text())
    return run


def test_a_run_on_the_sentence_pools_and_its_rebuild(mulms_run):
    """The results file records pool_by_sentence and every run its pool's documents and
    subfields; ner_mentions.run_split rebuilds the sentence pool from the config and checks the
    recorded counts and the test positions. Without the flag neither record is written."""
    data, subfields = _toy()
    flags = ('--leave-one-subfield-out', '--train-pool', '6', '--crf', '--tag', 'm100',
             '--seed-offset', '100', '--save-predictions')
    res = mulms_run(*flags, '--pool-by-sentence')
    assert res['variant'] == 'crf_spool_m100' and res['config']['pool_by_sentence'] is True
    folds = nb.subfield_folds(data, subfields, 0, 6, indices=True, by_sentence=True)
    assert [r['run_id'] for r in res['runs']] == ['alpha', 'beta', 'gamma']
    for run, (name, tr, va, te) in zip(res['runs'], folds):
        assert (run['seed'], run['n_train'], run['n_val'], run['test_index']) == (100, 5, 1, te)
        assert run['pool'] == nb.pool_record(data, tr + va, subfields)
        assert nm.run_split(res, run, data, 'mulms') == (data, tr, te)
    counts = nm.run_counts(res, data, 'mulms')
    assert sum(c['gold', 'seen'] + c['gold', 'unseen'] for c in counts.values()) == len(data)
    for change, error in ((lambda r: r['pool'].update(n_documents=r['pool']['n_documents'] + 1),
                           'not the rebuilt sentence pool'),
                          (lambda r: r.pop('pool'), 'not recorded'),
                          (lambda r: r.update(test_index=r['test_index'][1:]), 'test sentences')):
        bad = copy.deepcopy(res)
        change(bad['runs'][0])
        with pytest.raises(nm.MentionError, match=error):
            nm.run_split(bad, bad['runs'][0], data, 'mulms')
    article = mulms_run(*flags)
    assert article['variant'] == 'crf_m100' and 'pool_by_sentence' not in article['config']
    articles = nb.subfield_folds(data, subfields, 0, 6, indices=True)
    for run, (name, tr, va, te) in zip(article['runs'], articles):
        assert 'pool' not in run and run['test_index'] == te
        assert nm.run_split(article, run, data, 'mulms') == (data, tr, te)
    assert set(article['runs'][0]) == set(res['runs'][0]) - {'pool'}


# --- ner_mentions: seen and unseen names of the two pools ----------------------------------------

# a name per document, shared across subfields: a0 p, a1 q, b0 p, b1 r, g0 q, g1 r
NAMES = {'a0': 'p', 'a1': 'q', 'b0': 'p', 'b1': 'r', 'g0': 'q', 'g1': 'r'}


def _named():
    data = [(['the', NAMES[doc], 'x'], ['O', 'B-MAT', 'O'], f'{doc}.conll')
            for doc in NAMES for _ in range(4)]
    subfields = {f'{doc}.conll': {'a': 'alpha', 'b': 'beta', 'g': 'gamma'}[doc[0]] for doc in NAMES}
    return data, subfields


def _family(directory, variant, data, subfields, by_sentence, hit_unseen):
    """The 10 seed files of a family on the toy: gold everywhere except, unless hit_unseen,
    the unseen names (left out); the folds of an article or a sentence pool of 4."""
    folds = nb.subfield_folds(data, subfields, 0, 4, indices=True, by_sentence=by_sentence)
    runs, unseen = [], {}
    for name, tr, va, te in folds:
        seen = {data[j][0][1] for j in tr}
        unseen[name] = sum(data[j][0][1] not in seen for j in te)
        pred = [' '.join(data[j][1]) if hit_unseen or data[j][0][1] in seen else 'O O O'
                for j in te]
        run = {'run_id': name, 'test_index': te, 'test_gold': [' '.join(data[j][1]) for j in te],
               'test_pred': pred, 'f1': 50.0, 'f1_strict': 50.0}
        if by_sentence:
            run['pool'] = nb.pool_record(data, tr + va, subfields)
        runs.append(run)
    config = {'split_seed': 0, 'train_pool': 4, **({'pool_by_sentence': True} if by_sentence else {})}
    for off in range(0, 1000, 100):
        (directory / f'ner_results_mulms_leave_one_subfield_out_{variant}_m{off}.json').write_text(
            json.dumps({'protocol': 'leave_one_subfield_out', 'config': config, 'runs': runs}))
    return {name: (unseen[name], len(te)) for name, _, _, te in folds}


def test_seen_and_unseen_recall_of_the_two_pools(tmp_path, monkeypatch):
    """ner_mentions' `spool`: each pool's unseen share (its training sentences' names against
    the test names) and recall per kind, per subfield and paired; absent without the files."""
    pytest.importorskip('scipy')
    data, subfields = _named()
    monkeypatch.setattr(nm, 'corpus', lambda granularity, retokenized_test=False: data)
    monkeypatch.setattr(nm, 'mulms_subfields', lambda: subfields)
    article = _family(tmp_path, 'crf', data, subfields, False, hit_unseen=False)
    assert nm.sentence_pool(str(tmp_path)) == {}
    sentence = _family(tmp_path, 'crf_spool', data, subfields, True, hit_unseen=True)
    out = nm.sentence_pool(str(tmp_path))
    assert out['seeds'] == list(range(0, 1000, 100)) and out['reference'] == 'crf'
    for system, kinds in (('crf', article), ('crf_spool', sentence)):
        n_unseen, n = sum(u for u, _ in kinds.values()), sum(t for _, t in kinds.values())
        assert out['unseen_share'][system] == round(n_unseen / n, 3)
        assert out['systems'][system]['n_runs'] == 30
        assert out['systems'][system]['all']['n_gold'] == 10 * n
        for unit, (u, t) in kinds.items():
            assert out['subfields'][unit][system]['unseen_share'] == round(u / t, 3)
    # the article pool's runs find only seen names, the sentence pool's every name
    assert out['systems']['crf']['unseen']['recall'] == 0.0
    assert out['systems']['crf_spool']['unseen']['recall'] == (
        100.0 if any(u for u, _ in sentence.values()) else 0.0)
    assert out['systems']['crf']['seen']['recall'] == out['systems']['crf_spool']['seen'][
        'recall'] == 100.0
    assert out['minus_crf']['all']['n_pairs'] == 3
    gain = [100.0 * u / t for u, t in (article[k] for k in sorted(article))]
    assert out['minus_crf']['all']['recall']['mean_diff'] == pytest.approx(
        statistics.fmean(gain), abs=0.01)
    result = nm.analyse(str(tmp_path), corpus_path=tmp_path / 'no_corpus.txt')
    assert result['spool'] == out and list(result)[-1] == 'spool'
    (tmp_path / 'ner_results_mulms_leave_one_subfield_out_crf_spool_m900.json').unlink()
    assert 'spool' not in nm.analyse(str(tmp_path), corpus_path=tmp_path / 'no_corpus.txt')


# --- decision_tests: every pair at 1/4 ----------------------------------------------------------

def test_every_pair_correlated_at_a_quarter():
    pytest.importorskip('scipy')
    import math
    values = [1.0, 2.5, 0.5, 3.0, 2.0, 1.5, 4.0]
    units = dt.layout(7)
    assert dt.pair_rho(units[0], units[1]) == 0.0                     # no shared article
    assert dt.pair_rho(units[0], units[1], {'every_pair': 0.25}) == 0.25
    e = dt.dependent_t(values, units, {'every_pair': 0.25})
    # all 21 pairs at rho: Var = S^2 / (1 - rho) * (1 + (n - 1) rho) / n
    assert e['se_ratio'] == round(math.sqrt((1 + 6 * 0.25) / 0.75), 3)
    assert e['correlated_pairs'] == 21
    assert dt.dependent_t(values, units)['correlated_pairs'] == len(dt.shared_pool_pairs()) == 5


# --- batch w20a --------------------------------------------------------------------------------

def _stored_config(name):
    return json.loads((GPU / name).read_text())['config']


def _same_but(config, stored, keys):
    return {k: v for k, v in config.items() if k not in keys} == \
        {k: v for k, v in stored.items() if k not in keys}


def test_week20_batch(monkeypatch):
    """w20a: the four checks first (smokebh, smokebi in lane 0; smokebj, smokebk in lane 1),
    each the job that made its stored run with a smoke tag; then 20a's 10 crf_spool jobs and
    20b's 28 jobs, each writing its own files, balanced over the lanes."""
    monkeypatch.setattr(kj, 'corpus_sha256', lambda report=None: 'x' * 64)
    monkeypatch.setattr(kj, 'mulms_files', lambda: {'a_01.conll': 'a' * 64})
    lane0, lane1 = kj.batch_lanes('w20a')
    jobs = lane0 + lane1
    assert len(jobs) == 4 + 10 + 28 and abs(len(lane0) - len(lane1)) <= 1
    # smokebh: w19a's smokebf, the CRF run of smoke4, but for the tag
    smokebf = kj.batch_lanes('w19a')[0][0]
    assert lane0[0] == smokebf[:-1] + ['smokebh']
    # smokebi, smokebj, smokebk: the jobs of w10a (crf_m0), w8b (coarse crf_p3n2_m0) and w7a
    # (coarse inner folds crf_bias_m0) with a smoke tag
    w10a = [j for lane in kj.batch_lanes('w10a') for j in lane]
    w8b, w7a = kj.batch_lanes('w8b')[1], [j for lane in kj.batch_lanes('w7a') for j in lane]
    retag = (lambda j, tag: [tag if a == 'm0' else a for a in j])
    for check, made, tag in ((lane0[1], w10a, 'smokebi'), (lane1[0], w8b, 'smokebj'),
                             (lane1[1], w7a, 'smokebk')):
        assert check in [retag(j, tag) for j in made
                         if '--tag' in j and j[j.index('--tag') + 1] == 'm0']
    stored = {'smokebi': 'ner_results_mulms_leave_one_subfield_out_crf_m0.json',
              'smokebj': 'ner_results_coarse_leave_one_paper_out_crf_p3n2_m0.json',
              'smokebk': 'ner_results_coarse_leave_two_papers_out_crf_bias_m0.json'}
    checks = [nb.parse_args(j) for j in (lane0[1], lane1[0], lane1[1])]
    for args, (tag, name) in zip(checks, stored.items()):
        assert args.tag == tag and 'smoke' in nb.results_path('', args.granularity, args)
        assert _same_but(nb.stored_config(args), _stored_config(name), ('tag',))
        # the analysis compares the file the repeat writes with the stored one
        written = os.path.basename(nb.results_path('', args.granularity, args))
        assert sa.WEEK20_REPEATS[tag] == (written, name)
    assert nb.parse_args(lane0[0]).tag == 'smokebh'
    rest = [nb.parse_args(j) for j in lane0[2:] + lane1[2:]]
    spool = [a for a in rest if getattr(a, 'pool_by_sentence', False)]
    assert sorted(nb.variant_name(a) for a in spool) == sorted(
        f'crf_spool_m{off}' for off in range(0, 1000, 100))
    crf_m0 = _stored_config('ner_results_mulms_leave_one_subfield_out_crf_m0.json')
    for a in spool:
        assert (a.granularity, nb.protocol_name(a), a.train_pool, a.save_predictions) == (
            'mulms', 'leave_one_subfield_out', 400, True)
        assert getattr(a, 'seed_offset', 0) == int(a.tag[1:])
        # the stored CRF head's flags, the pool drawn by sentence
        assert _same_but(nb.stored_config(a), crf_m0, ('tag', 'seed_offset', 'pool_by_sentence'))
    week20b = [a for a in rest if not getattr(a, 'pool_by_sentence', False)]
    assert len(week20b) == 28
    for a in week20b:
        protocol, variant = nb.protocol_name(a), nb.variant_name(a)
        family = variant.rpartition('_m')[0]
        assert (protocol, family) in (('leave_one_paper_out', 'crf_p3n2'),
                                      ('leave_two_papers_out', 'crf_bias'))
        assert 300 <= a.seed_offset == int(variant.rpartition('_m')[2]) <= 900
        assert a.save_predictions == (family == 'crf_p3n2')
        # the flags of week 8's and week 7's stored files (their seed 200)
        stored200 = _stored_config(f'ner_results_{a.granularity}_{protocol}_{family}_m200.json')
        assert _same_but(nb.stored_config(a), stored200, ('tag', 'seed_offset'))
    assert sorted((a.granularity, nb.variant_name(a)) for a in week20b) == sorted(
        (g, f'{family}_m{off}') for g in ('fine', 'coarse') for family in ('crf_p3n2', 'crf_bias')
        for off in range(300, 1000, 100))
    files = [(a.granularity, nb.protocol_name(a), nb.variant_name(a))
             for a in (nb.parse_args(j) for j in jobs)]
    assert len(files) == len(set(files))                     # no two jobs write one file
    assert 'w20a' in kj.BATCHES and kj.batch_name('w20a') == 'w20a'
    spec = kj.batch_spec('w20a')
    assert spec['lanes'] == [lane0, lane1] and spec['mulms_files'] == {'a_01.conll': 'a' * 64}
    assert 'topic_corpora' not in spec and 'week19_names' not in spec


# --- the week-20 section of ner_seed_analysis.py, dry-run on the stored files --------------------

MULMS = 'ner_results_mulms_leave_one_subfield_out'
POOL = 'ner_results_mulms_random_split_pool'
SHIFT = {'electrolysis': 1.0, 'graphene': 2.0, 'pemfc': 3.0, 'poly': 1.5, 'semi': 2.5,
         'sofc': 0.5, 'steel': 4.0}
SIZED_SHIFT = 0.7


def _link(directory, name):
    (directory / name).symlink_to(GPU / name)


def _write(directory, name, data):
    (directory / name).write_text(json.dumps(data))


@pytest.fixture(scope='module')
def dry(tmp_path_factory):
    """A results directory with the stored files that week 20 reads (linked) and a fake
    family: crf_spool, the stored crf files with micro F1 shifted by SHIFT per subfield and the
    sentence pools' records; 20b's seeds 300 to 900, copies of the stored seed-0 files (crf_p3n2
    with micro F1 shifted by SIZED_SHIFT, the inner folds unchanged); and a pulled output with
    the three repeats, copies of their stored runs (another runtime_sec)."""
    if not (GPU / f'{MULMS}_crf_m0.json').exists():
        pytest.skip('the stored results are not here')
    root = tmp_path_factory.mktemp('w20_dry')
    results, pulled = root / 'results', root / 'pulled' / 'processed' / 'ner'
    results.mkdir()
    pulled.mkdir(parents=True)
    for off in range(0, 1000, 100):
        for name in (f'{MULMS}_crf_m{off}.json', f'{MULMS}_crf_mr_m{off}.json',
                     f'{POOL}_crf_m{off}.json',
                     *(f'ner_results_{g}_leave_one_paper_out_crf_m{off}.json'
                       for g in ('fine', 'coarse'))):
            _link(results, name)
        spool = json.loads((GPU / f'{MULMS}_crf_m{off}.json').read_text())
        spool['variant'], spool['config']['pool_by_sentence'] = f'crf_spool_m{off}', True
        for run in spool['runs']:
            run['f1'] += SHIFT[run['run_id']]
            run['pool'] = {'n_documents': SPOOL_DOCUMENTS[run['run_id']], 'n_subfields': 6}
        _write(results, f'{MULMS}_crf_spool_m{off}.json', spool)
    for g in ('fine', 'coarse'):
        for protocol, family, shift in (('leave_one_paper_out', 'crf_p3n2', SIZED_SHIFT),
                                        ('leave_two_papers_out', 'crf_bias', 0.0)):
            for off in (0, 100, 200):
                _link(results, f'ner_results_{g}_{protocol}_{family}_m{off}.json')
            fake = json.loads((GPU / f'ner_results_{g}_{protocol}_{family}_m0.json').read_text())
            for run in fake['runs']:
                run['f1'] += shift
            for off in range(300, 1000, 100):
                _write(results, f'ner_results_{g}_{protocol}_{family}_m{off}.json', fake)
    for tag, (name, stored) in sa.WEEK20_REPEATS.items():
        repeat = json.loads((GPU / stored).read_text())
        for run in repeat['runs']:
            run['runtime_sec'] = 1.0
        _write(pulled, name, repeat)
    return results, root / 'pulled'


def _scores(name):
    return {r['run_id']: r for r in json.loads((GPU / name).read_text())['runs']}


def _family_means(names, get):
    runs = [_scores(n) for n in names]
    return {i: statistics.fmean(get(r[i]) for r in runs) for i in runs[0]}


def test_the_spool_section_on_the_stored_files(dry):
    """20.1 on the fake family: the per-subfield differences are the shifts; the remaining
    part and the gap are Welch differences of the random-split pool's 5 split means and the
    held-out subfields' 7 means; the dependence readings are decision_tests.dependent_t's."""
    pytest.importorskip('scipy')
    from scipy import stats
    results, pulled = dry
    w20 = sa.week20(results, pulled)
    assert set(w20) == {'repeats', 'spool', 'diversity'}
    s = w20['spool']
    assert (s['decision'], s['n_subfields'], s['seeds']) == ('20.1', 7, list(range(0, 1000, 100)))
    d = s['minus_crf']['micro']
    assert d['per_run'] == [SHIFT[u] for u in dt.MULMS_UNITS]
    lo, hi = metrics.mean_ci([SHIFT[u] for u in dt.MULMS_UNITS])['ci']
    assert d['ci95'] == [round(lo, 2), round(hi, 2)] and s['passes'] is (lo > 0)
    assert s['decided'] is True and 'flag' not in s
    crf = _family_means([f'{MULMS}_crf_m{off}.json' for off in range(0, 1000, 100)],
                        lambda r: r['f1'])
    assert s['crf']['micro']['mean'] == round(statistics.fmean(crf.values()), 2) == 48.66
    for u, both in s['subfields'].items():
        assert both['crf']['micro'] == round(crf[u], 2)
        assert both['crf_spool']['micro'] == round(crf[u] + SHIFT[u], 2)
        assert both['crf_spool']['macro'] == both['crf']['macro']           # micro F1 shifted only
    assert s['pools'] == {u: {'n_documents': n, 'n_subfields': 6}
                          for u, n in SPOOL_DOCUMENTS.items()}
    epochs = _family_means([f'{MULMS}_crf_m{off}.json' for off in range(0, 1000, 100)],
                           lambda r: r['best_epoch'])
    assert s['validation']['crf'] == s['validation']['crf_spool']
    assert s['validation']['crf']['best_epoch'] == round(statistics.fmean(epochs.values()), 2)
    pool = list(_family_means([f'{POOL}_crf_m{off}.json' for off in range(0, 1000, 100)],
                              lambda r: r['f1']).values())
    for key, side in (('remaining', [crf[u] + SHIFT[u] for u in crf]), ('gap', list(crf.values()))):
        welch = stats.ttest_ind(pool, side, equal_var=False)       # scipy's Welch test
        ci = welch.confidence_interval(0.95)
        got = s[key]['micro']
        assert got['n'] == [5, 7] and got['df'] == pytest.approx(welch.df, abs=0.051)
        assert got['mean_diff'] == pytest.approx(statistics.fmean(pool) - statistics.fmean(side),
                                                 abs=0.006)
        assert got['ci95'] == pytest.approx([ci.low, ci.high], abs=0.006)
    analysis = json.loads((GPU / sa.OUT_NAME).read_text())
    assert s['gap']['micro']['mean_diff'] == pytest.approx(analysis['mulms']['gap']['micro'],
                                                           abs=0.011)   # the 17 points
    values = [SHIFT[u] for u in dt.MULMS_UNITS]
    shared = dt.dependent_t(values, dt.layout(7))
    every = dt.dependent_t(values, dt.layout(7), {'every_pair': 0.25})
    assert s['dependence']['shared_articles'] == dict(shared, above_0=shared['ci95'][0] > 0)
    assert s['dependence']['every_pair'] == dict(every, above_0=every['ci95'][0] > 0)
    assert s['dependence']['every_pair']['correlated_pairs'] == 21


def _diversity(results, g, offsets):
    """Per held-out paper, three papers at the size of two minus two papers alone, from the
    files: the mean over (pair, seed) of crf_p3n2's F1 minus that of the inner folds' F1 at D
    = 0 (their counts), and the three papers' mean F1 (crf)."""
    two, sized, full = {}, {}, {}
    for off in offsets:
        prefix = f'ner_results_{g}_'
        for run in json.loads((results / f'{prefix}leave_two_papers_out_crf_bias_m{off}.json'
                               ).read_text())['runs']:
            for paper, (c, p, gold) in run['bias_test']['0'].items():
                two.setdefault(paper, []).append(200 * c / (p + gold) if p + gold else 0.0)
        for run in json.loads((results / f'{prefix}leave_one_paper_out_crf_p3n2_m{off}.json'
                               ).read_text())['runs']:
            sized.setdefault(run['run_id'].split('|')[0], []).append(run['f1'])
        for run in json.loads((results / f'{prefix}leave_one_paper_out_crf_m{off}.json'
                               ).read_text())['runs']:
            full.setdefault(run['run_id'], []).append(run['f1'])
    papers = sorted(full)
    return ([statistics.fmean(sized[p]) - statistics.fmean(two[p]) for p in papers],
            [statistics.fmean(full[p]) - statistics.fmean(sized[p]) for p in papers])


def test_the_diversity_section_on_the_stored_files(dry):
    """20.2 and 20.3 on the fake seeds 300 to 900, with size and the dependence reading; the
    10-seed reading; the seed set leaves the stored paper_diversity (seeds 0 to 200) as it
    was, and no existing section reads the fake files."""
    pytest.importorskip('scipy')
    results, pulled = dry
    div = sa.week20(results, pulled)['diversity']
    for g, decision in (('fine', '20.2'), ('coarse', '20.3')):
        e = div['decision'][g]
        assert (e['decision'], e['seeds']) == (decision, list(range(300, 1000, 100)))
        diversity, size = _diversity(results, g, range(300, 1000, 100))
        lo, hi = metrics.mean_ci(diversity)['ci']
        assert e['diversity']['per_run'] == pytest.approx(diversity, abs=0.006)
        assert e['diversity']['ci95'] == pytest.approx([lo, hi], abs=0.006)
        assert e['passes'] is e['diversity']['passes'] is (e['diversity']['ci95'][0] > 0)
        assert e['size']['per_run'] == pytest.approx(size, abs=0.006)
        dep = dt.dependent_t(e['diversity']['per_run'], dt.layout(4))
        assert e['dependence'] == dict(dep, above_0=dep['ci95'][0] > 0)
        # the fake seeds are seed 0's files with crf_p3n2 shifted: diversity moves by the shift
        seed0, _ = _diversity(results, g, (0,))
        assert e['diversity']['mean_diff'] == pytest.approx(statistics.fmean(seed0) + SIZED_SHIFT,
                                                            abs=0.011)
        ten = div['ten_seeds'][g]
        diversity, size = _diversity(results, g, range(0, 1000, 100))
        assert ten['seeds'] == list(range(0, 1000, 100)) and 'passes' not in ten['diversity']
        assert ten['diversity']['per_run'] == pytest.approx(diversity, abs=0.006)
        assert ten['diversity']['above_0'] is (ten['diversity']['ci95'][0] > 0)
        assert ten['size']['per_run'] == pytest.approx(size, abs=0.006)
    analysis = json.loads((GPU / sa.OUT_NAME).read_text())
    assert json.loads(json.dumps(sa.paper_diversity(results))) == analysis['paper_diversity']
    assert json.loads(json.dumps(sa.paper_count(results))) == analysis['paper_count']
    assert sa.week12_variants(results) == {}         # crf_spool is not a week-12 variant
    assert 'crf_spool' in sa.WEEK20_VARIANTS


def test_the_repeats_and_their_consequences(dry, tmp_path):
    """The repeats equal their stored runs in every field but runtime_sec; a repeat that
    differs leaves 20.1 undecided and the 10-seed readings unreported; a differing environment
    stops the comparisons with stored runs; without the repeats 20.1 carries a flag."""
    pytest.importorskip('scipy')
    results, pulled = dry
    w20 = sa.week20(results, pulled)
    reps = w20['repeats']
    assert reps['status'] == {'checked': True, 'unchecked': [], 'environment_differs': False,
                              'differs': False}
    assert {t: (r['equal'], r['environment_equal'], r['n_runs'], r['differ'])
            for t, r in reps.items() if t != 'status'} == {
        'smokebi': (True, True, 7, []), 'smokebj': (True, True, 12, []),
        'smokebk': (True, True, 6, [])}
    # without the repeats: no repeats part, 20.1 flagged, the 10-seed readings reported
    alone = sa.week20(results)
    assert 'repeats' not in alone and alone['spool']['decided'] is None
    assert alone['spool']['flag'] == ('the repeats smokebi, smokebj, smokebk are not checked '
                                      'here (ner_seed_analysis.py --repeats)')
    assert 'fine' in alone['diversity']['ten_seeds']
    with pytest.raises(SystemExit, match='no directory'):
        sa.week20(results, tmp_path / 'nowhere')
    # a repeat that differs in a field
    differing = tmp_path / 'differing'
    differing.mkdir()
    for tag, (name, _) in sa.WEEK20_REPEATS.items():
        repeat = json.loads((pulled / 'processed' / 'ner' / name).read_text())
        if tag == 'smokebj':
            repeat['runs'][3]['best_epoch'] += 1
            repeat['runs'][3]['test_pred'][0] = 'O'
        (differing / name).write_text(json.dumps(repeat))
    w20 = sa.week20(results, differing)
    r = w20['repeats']['smokebj']
    assert (r['equal'], r['environment_equal'], r['differ']) == (False, True,
                                                                 ['best_epoch', 'test_pred'])
    assert r['runs_differ'] == [json.loads((GPU / sa.WEEK20_REPEATS['smokebj'][1]).read_text())[
        'runs'][3]['run_id']]
    assert w20['spool']['decided'] is False and 'not decided' in w20['spool']['flag']
    assert w20['diversity']['ten_seeds'] == {
        'not_reported': 'a repeat differs (ROADMAP §4, week 20, the checks)'}
    assert set(w20['diversity']['decision']) == {'fine', 'coarse'}
    # a differing environment: no comparison with stored runs
    name, stored = sa.WEEK20_REPEATS['smokebi']
    repeat = json.loads((GPU / stored).read_text())
    for run in repeat['runs']:
        run['environment'] = dict(run['environment'], torch='2.12.0')
    (differing / name).write_text(json.dumps(repeat))
    w20 = sa.week20(results, differing)
    assert w20['repeats']['smokebi']['environment_equal'] is False
    assert 'environment' in w20['repeats']['smokebi']['differ']
    assert w20['repeats']['status']['environment_differs'] is True
    assert set(w20['spool']) == {'stopped'}
    for e in w20['diversity']['decision'].values():
        assert 'size' not in e and 'three' not in e and 'three' not in e['papers'][
            sorted(e['papers'])[0]]
    # a missing run, and a repeat twice under the directory
    del repeat['runs'][0]
    for run in repeat['runs']:
        run['environment'] = json.loads((GPU / stored).read_text())['runs'][0]['environment']
    (differing / name).write_text(json.dumps(repeat))
    r = sa.week20(results, differing)['repeats']['smokebi']
    assert (r['equal'], r['n_runs'], r['missing']) == (False, 6, ['electrolysis'])
    (differing / 'again').mkdir()
    (differing / 'again' / name).write_text(json.dumps(repeat))
    with pytest.raises(SystemExit, match='2 times'):
        sa.week20(results, differing)
    (tmp_path / 'empty').mkdir()
    assert sa.week20(results, tmp_path / 'empty') == alone
    # one repeat there, two not: those two are named
    only = tmp_path / 'only'
    only.mkdir()
    (only / sa.WEEK20_REPEATS['smokebk'][0]).write_text(
        (pulled / 'processed' / 'ner' / sa.WEEK20_REPEATS['smokebk'][0]).read_text())
    part = sa.week20(results, only)
    assert part['repeats']['status']['unchecked'] == ['smokebi', 'smokebj']
    assert part['spool']['decided'] is None and 'smokebi, smokebj' in part['spool']['flag']


def test_main_prints_the_week20_section(dry, tmp_path, capsys):
    pytest.importorskip('scipy')
    results, pulled = dry
    sa.print_week20(sa.week20(results, pulled))
    out = capsys.readouterr().out
    assert 'the repeat smokebi against' in out and 'identical (7 runs' in out
    assert 'week 20, 20.1: the sentence pool' in out and '(passes)' in out
    assert 'the gap, the random-split pool minus the article pool (Welch' in out
    assert 'week 20, 20.2 (fine-grained, seeds 300-900)' in out and '20.3' in out
    assert 'fine-grained, 10 seeds (reported)' in out
