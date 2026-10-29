"""build_mixed_corpus.py: the mixed corpora of week 14 (ROADMAP §4, 14.2) on tiny topic
corpora."""
import json

import pytest

import build_mixed_corpus as bm


def _topics(tmp_path, lines):
    """Write tiny topic corpora and their reports; lines: {topic: [sentence]}."""
    corpus_dir, report_dir = tmp_path / 'corpus', tmp_path / 'reports'
    corpus_dir.mkdir()
    report_dir.mkdir()
    for topic, sentences in lines.items():
        path = bm.topic_path(topic, corpus_dir)
        path.write_text(''.join(s + '\n' for s in sentences), encoding='utf-8')
        bm.report_path(topic, report_dir).write_text(json.dumps(
            {'corpus': {'sha256': bm.file_sha256(path)}}))
    return corpus_dir, report_dir


def test_quotas_share_the_remainder_among_the_first_sources():
    assert bm.quotas(40000, 6) == [6667, 6667, 6667, 6667, 6666, 6666]
    assert bm.quotas(6, 3) == [2, 2, 2] and bm.quotas(7, 3) == [3, 2, 2]


def test_mixed_sentences_skip_the_own_and_duplicate_sentences_deterministically():
    own = ['shared with a', 'own only']
    sources = {'a': [f'a{i}' for i in range(6)] + ['shared with a', 'dup'],
               'b': [f'b{i}' for i in range(6)] + ['dup']}
    out, counts = bm.mixed_sentences(own, sources, size=6, seed=0)
    assert len(out) == 6 and len(set(out)) == 6 and 'shared with a' not in out
    assert [s[0] for s in out] == ['a'] * 3 + ['b'] * 3        # source after source
    assert out == bm.mixed_sentences(own, sources, size=6, seed=0)[0]      # seeded
    assert out != bm.mixed_sentences(own, sources, size=6, seed=1)[0]
    assert counts['a']['quota'] == 3 and counts['a']['taken'] == 3
    assert counts['a']['sentences'] == 8 and counts['b']['sentences'] == 7
    assert counts['a']['visited'] == 3 + counts['a']['skipped_own'] + counts['a']['skipped_duplicate']
    # the chosen sentences of a source keep the corpus order
    a_rows = [int(s[1:]) for s in out[:3] if s[1:].isdigit()]
    assert a_rows == sorted(a_rows)
    # too few sentences after the exclusions
    with pytest.raises(SystemExit, match='only'):
        bm.mixed_sentences(own, {'a': ['shared with a', 'x'], 'b': ['y', 'z']}, size=4)
    # a duplicate of a sentence already taken from another source is skipped, once taken
    out, counts = bm.mixed_sentences([], {'a': ['dup', 'a1'], 'b': ['dup', 'b1', 'b2']},
                                     size=4, seed=0)
    assert sorted(out) == ['a1', 'b1', 'b2', 'dup'] and counts['b']['skipped_duplicate'] == 1


def test_build_writes_the_corpus_and_its_report(tmp_path):
    lines = {t: [f'{t} sentence {i} words' for i in range(10)] for t in ('alpha', 'beta', 'gamma')}
    lines['beta'][0] = lines['alpha'][0]                       # shared with alpha's corpus
    corpus_dir, report_dir = _topics(tmp_path, lines)
    report = bm.build('alpha', corpus_dir, report_dir, topics=('alpha', 'beta', 'gamma'),
                      size=8, seed=0)
    out = bm.topic_path('mixed_alpha', corpus_dir)
    sentences = bm.read_lines(out)
    assert len(sentences) == 8 and lines['alpha'][0] not in sentences
    assert all(s in lines['beta'] or s in lines['gamma'] for s in sentences)
    assert report['topic'] == 'mixed_alpha' and report['held_out'] == 'alpha'
    assert set(report['sources']) == {'beta', 'gamma'}
    assert report['sources']['beta']['quota'] == 4 and report['sources']['gamma']['taken'] == 4
    assert report['sources']['beta']['sha256'] == bm.file_sha256(bm.topic_path('beta', corpus_dir))
    assert report['corpus']['sentences'] == 8 and report['corpus']['words'] == 8 * 4
    assert report['corpus']['sha256'] == bm.file_sha256(out)
    saved = json.loads(bm.report_path('mixed_alpha', report_dir).read_text())
    assert saved['corpus']['sha256'] == report['corpus']['sha256']
    assert saved['settings'] == {'seed': 0, 'sentences': 8, 'rule': bm.RULE}
    # the same input gives the same corpus
    again = bm.build('alpha', corpus_dir, report_dir, topics=('alpha', 'beta', 'gamma'),
                     size=8, seed=0)
    assert again['corpus']['sha256'] == report['corpus']['sha256']
    # a source that is not the one its report records
    bm.topic_path('gamma', corpus_dir).write_text('changed\n' * 10, encoding='utf-8')
    with pytest.raises(SystemExit, match='SHA-256'):
        bm.build('alpha', corpus_dir, report_dir, topics=('alpha', 'beta', 'gamma'), size=8)
    with pytest.raises(SystemExit, match='missing'):
        bm.build('delta', corpus_dir, report_dir, topics=('alpha', 'delta'), size=2)


def test_main_builds_the_asked_subfields(tmp_path, capsys):
    lines = {t: [f'{t} sentence {i}' for i in range(12)] for t in bm.TOPICS}
    corpus_dir, report_dir = _topics(tmp_path, lines)
    bm.main(['--subfields', 'sofc', '--corpus-dir', str(corpus_dir), '--report-dir',
             str(report_dir), '--sentences', '12'])
    out = capsys.readouterr().out
    assert out.startswith('sofc') and '12 sentences' in out
    assert bm.report_path('mixed_sofc', report_dir).is_file()
    assert not bm.report_path('mixed_steel', report_dir).exists()
    report = json.loads(bm.report_path('mixed_sofc', report_dir).read_text())
    assert [c['quota'] for c in report['sources'].values()] == [2] * 6
