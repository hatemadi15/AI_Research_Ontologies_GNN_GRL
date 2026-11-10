"""Tests of the SOTA plan's P3 (ROADMAP §4, the P3 protocol): its jobs, checks and report."""
import json

import kaggle_jobs as kj
import ner_baseline as nb
import sota_p2
import sota_p3 as p3


def test_jobs_pair_with_the_stored_crf_head():
    jobs = [j for m, j in p3.jobs()]
    assert jobs[0] == p3.SMOKE_JOB and len(jobs) == 1 + 2 * len(p3.OFFSETS)
    for j in jobs[1:]:
        args = nb.parse_args(j)
        assert args.seeds == [0] and args.seed_offset in p3.OFFSETS
        assert args.tag.startswith('p3') and args.tag.endswith(f'm{args.seed_offset}')
    names = [sota_p2.tagger_file(j, p3.RUN[k]['granularity'])
             for k in p3.RUN for j in (p3.job(k, o) for o in p3.OFFSETS)]
    assert len(set(names)) == len(names)
    assert 'p3a' in kj.BATCHES and kj.batch_extras('p3a') == kj.P1_PINS


def test_checks_need_the_smoke_repeat(tmp_path):
    assert p3.checks(tmp_path) == {'smoke': ['smokebu is missing']}


def test_paired_values_check_test_sentences():
    ref = {('p', 0): {'f1': 50.0, 'test_index': [1, 2], 'test_gold': ['O', 'O']}}
    good = {('p', 0): {'f1': 52.0, 'test_index': [1, 2], 'test_gold': ['O', 'O']}}
    assert p3.paired_values(good, ref, 'x') == {('p', 0): (52.0, 50.0)}
    bad = {('p', 0): {'f1': 52.0, 'test_index': [1, 3], 'test_gold': ['O', 'O']}}
    try:
        p3.paired_values(bad, ref, 'x')
        raise AssertionError('accepted other test sentences')
    except ValueError:
        pass


def test_protocol_record():
    pairs = {(f'f{i % 2}', o): (50.0 + i % 3, 48.0) for i, o in enumerate(range(0, 1000, 100))}
    rec = p3.protocol_record(pairs)
    assert rec['paired_difference']['n'] == 10 and rec['crf_head']['mean'] == 48.0
    assert set(rec['per_fold']) == {'f0', 'f1'}
