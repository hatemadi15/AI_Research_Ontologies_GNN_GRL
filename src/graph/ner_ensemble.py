"""
ner_ensemble.py - entity-level majority-vote ensemble of NER models

Combines the test predictions of k models trained on the same splits (k >= 2)
and scores the vote with the metrics and the results-file schema of
ner_baseline.py, so that `ner_baseline.py --compare` can set the ensemble
against a single model, run by run (same run_id, same test sentences).

  - Members: the results files of `ner_baseline.py --crf --save-predictions
    --seed-offset <N> --tag m<N>` (variants crf_m0, crf_m100, crf_m200). They
    must agree on granularity, protocol, model, labels and run_ids and, per
    run_id, on the test sentences (test_index) and their gold tags, and each
    run needs its test_pred. Otherwise the script stops with an error.
  - Mixed ensembles: `--variant NAME` (e.g. crf_mix3) names the ensemble, which
    lets the members differ in model and variant (crf_m0, crf_scibert and
    crf_pubmedbert: MatSciBERT, SciBERT and PubMedBERT); every other check
    stands. config['models'] lists the members' models, and `model` is 'mixed'
    if they differ.
  - Voting (fixed in advance): the entities (start, end, type) of every member
    are read from its predicted IOB2 tags; an entity is kept if strictly more
    than k/2 members predict exactly the same span and type (2 of 3, 3 of 5;
    2 of 4 is a tie and is dropped). The kept entities are written back as IOB2
    tags and scored with ner_baseline.score.
  - An I-X that does not continue an entity of type X starts a new entity, as
    in seqeval's default (conlleval) mode. A CRF with IOB2 constraints never
    produces one; the number of them in the members' predictions is saved per
    run as `invalid_i_tags`.
  - Output: ner_results_<granularity>[_<protocol>]_<variant>.json in the
    directory of the first member (--out overrides it). variant is NAME with
    --variant, else the members' variant without its m<N> tag plus ens<k>:
    crf_m0, crf_m100, crf_m200 give crf_ens3. Every run adds n_members, member_f1,
    member_f1_strict and invalid_i_tags to the seqeval metrics; the summary is
    that of ner_baseline.summarize plus `vs_member_mean`: the vote's micro F1
    minus the mean of its members', per run, with its mean and 95% CI
    (exploratory: what the vote adds over one model of the same kind).

Usage:
  python ner_ensemble.py --members FILE FILE FILE [--out PATH] [--variant NAME]
  python ner_ensemble.py --members ner_results_fine_random_split_vary_crf_m{0,100,200}.json
  python ner_ensemble.py --variant crf_mix3 --members CRF_M0 CRF_SCIBERT CRF_PUBMEDBERT
"""

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter

import metrics
from ner_baseline import PLAIN_VARIANTS, score, summarize

VOTING = 'entity_strict_majority'
# `ner_baseline.py --tag m<N>` appends m<N> to the variant (crf_m100; just m100 for the baseline)
MEMBER_TAG = re.compile(r'(?:^|_)m\d+$')
VARIANT_NAME = re.compile(r'[a-z0-9]+(?:_[a-z0-9]+)*')   # --variant: crf_mix3
IOB2_TAG = re.compile(r'O|[BI]-.+')   # O, B-<type> or I-<type>
# top-level keys of a member file that are copied to the ensemble file
COPIED_KEYS = ('granularity', 'protocol', 'model', 'labels', 'label_tie', 'paper_test_f1')
# keys every run of a member needs
RUN_KEYS = ('test_index', 'test_gold', 'test_pred', 'f1', 'f1_strict')


class EnsembleError(ValueError):
    """The results files cannot be combined (the command line reports it without a traceback)."""


# ---------------------------------------------------------------------------
# Voting (pure Python)
# ---------------------------------------------------------------------------
def tags_to_spans(tags):
    """(spans, n_invalid) of one IOB2 tag sequence.

    spans are the entities (start, end_exclusive, type) in sentence order. An
    I-X that does not continue an entity of type X (after O, after a tag of
    another type, or at the start) starts a new entity, as in seqeval's
    default (conlleval) mode; n_invalid counts these tags.
    """
    spans, n_invalid = [], 0
    start, kind = 0, None   # the open entity, if kind is set
    for i, tag in enumerate(list(tags) + ['O']):   # the sentinel closes the last entity
        if not IOB2_TAG.fullmatch(tag):
            raise ValueError(f'not an IOB2 tag: {tag!r} (position {i})')
        if kind is not None and (tag[0] != 'I' or tag[2:] != kind):
            spans.append((start, i, kind))
            kind = None
        if tag != 'O' and kind is None:
            if tag[0] == 'I':
                n_invalid += 1
            start, kind = i, tag[2:]
    return spans, n_invalid


def spans_to_tags(spans, n):
    """IOB2 tags for a sentence of n tokens from non-overlapping (start, end, type) spans."""
    tags = ['O'] * n
    for start, end, kind in sorted(spans):
        if not 0 <= start < end <= n or any(t != 'O' for t in tags[start:end]):
            raise ValueError(f'span {(start, end, kind)} is empty, outside the {n} tokens '
                             f'or overlaps another span')
        tags[start:end] = [f'I-{kind}'] * (end - start)
        tags[start] = f'B-{kind}'
    return tags


def majority_spans(member_spans):
    """Spans (same start, end and type) that strictly more than half of the members predict."""
    # The kept spans cannot overlap: two spans that each have a strict majority (> k/2) share
    # a supporting member, and the entities of one member never overlap.
    votes = Counter(span for spans in member_spans for span in set(spans))   # once per member
    return sorted(span for span, n in votes.items() if 2 * n > len(member_spans))


def vote_tags(member_tags):
    """(IOB2 tags of the majority vote, invalid I- tags of the members) for one sentence.

    member_tags: one tag list per member, all for the same tokens.
    """
    n = len(member_tags[0])
    if any(len(tags) != n for tags in member_tags):
        raise ValueError('the members tag different numbers of tokens')
    parsed = [tags_to_spans(tags) for tags in member_tags]
    spans = majority_spans([spans for spans, _ in parsed])
    return spans_to_tags(spans, n), sum(bad for _, bad in parsed)


# ---------------------------------------------------------------------------
# Members: reading and validation
# ---------------------------------------------------------------------------
def load_member(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:   # includes JSONDecodeError
        raise EnsembleError(f'{path}: cannot read ({e})') from None
    if not isinstance(data, dict):
        raise EnsembleError(f'{path}: not a ner_baseline.py results file')
    return data


def ensemble_variant(variants):
    """Variant of the ensemble: the members' variant without its m<N> tag, plus ens<k>.

    crf_m0, crf_m100, crf_m200 give crf_ens3 (a bare tag, m0, m100, gives ens3).
    What is left of the members' variants must be the same.
    """
    stems = [MEMBER_TAG.sub('', v) for v in variants]
    if len(set(stems)) != 1:
        raise EnsembleError(f'the members differ in variant once the m<N> tag is removed: '
                            f'{list(variants)}')
    ens = f'ens{len(variants)}'
    return f'{stems[0]}_{ens}' if stems[0] else ens


def variant_option(value):
    """argparse type of --variant: a variant name that no single-model run writes.

    Lowercase letters and digits joined by single underscores; neither a plain variant of
    ner_baseline.py (crf, ...) nor a name that ends in an m<N> tag (a seed's results).
    """
    if not VARIANT_NAME.fullmatch(value) or value in PLAIN_VARIANTS or MEMBER_TAG.search(value):
        raise argparse.ArgumentTypeError(
            f"invalid --variant {value!r}: use lowercase letters, digits and underscores, not "
            f"one of {', '.join(PLAIN_VARIANTS)} and no trailing m<N> tag (crf_mix3)")
    return value


def results_name(granularity, protocol, variant):
    """File name used by ner_baseline.py: the fixed split has no protocol part."""
    part = '' if protocol == 'random_split_fixed' else f'_{protocol}'
    return f'ner_results_{granularity}{part}_{variant}.json'


def index_runs(member, name):
    """{run_id: run} of one member; the run_ids must exist and be unique."""
    runs = member['runs']
    if not isinstance(runs, list) or not all(isinstance(r, dict) for r in runs):
        raise EnsembleError(f'{name}: runs must be a list of runs')
    ids = [r.get('run_id') for r in runs]
    if not ids or None in ids or len(set(ids)) != len(ids):
        raise EnsembleError(f'{name}: runs must be a non-empty list with unique run_ids, '
                            f'got {ids}')
    return dict(zip(ids, runs))


def check_members(members, names, same_model=True):
    """Raise EnsembleError unless the top-level fields of the members agree.

    The model is checked only with `same_model` (not for a mixed ensemble, --variant).
    """
    if len(members) < 2:
        raise EnsembleError(f'an ensemble needs at least 2 members, got {len(members)}')
    for m, name in zip(members, names):
        missing = [k for k in COPIED_KEYS + ('variant', 'runs') if k not in m]
        if missing:
            raise EnsembleError(f'{name}: missing {missing} (not a ner_baseline.py results file?)')
    first = members[0]
    keys = ('granularity', 'protocol', 'model', 'labels') if same_model else (
        'granularity', 'protocol', 'labels')
    for key in keys:
        for m, name in zip(members[1:], names[1:]):
            if m[key] == first[key]:
                continue
            if key == 'labels':
                a, b = set(first[key]), set(m[key])
                detail = (f'only in {names[0]}: {sorted(a - b)}, only in {name}: {sorted(b - a)}'
                          if a != b else 'same classes in a different order')
            else:
                detail = f'{first[key]!r} in {names[0]}, {m[key]!r} in {name}'
            raise EnsembleError(f'the members differ in {key}: {detail}')


def check_prediction(run, where):
    """Raise EnsembleError unless test_pred has valid IOB2 tags for the tokens of test_gold."""
    gold, pred = run['test_gold'], run['test_pred']
    for key in ('test_index', 'test_gold', 'test_pred'):
        if not isinstance(run[key], list):
            raise EnsembleError(f'{where}: {key} must be a list')
    if not all(isinstance(s, str) for s in gold + pred):
        raise EnsembleError(f'{where}: test_gold and test_pred must be lists of strings')
    if not len(run['test_index']) == len(gold) == len(pred):
        raise EnsembleError(f'{where}: test_index, test_gold and test_pred differ in length')
    if not gold:
        raise EnsembleError(f'{where}: no test sentences')
    for j, (g, p) in enumerate(zip(gold, pred)):
        tags = p.split()
        if len(tags) != len(g.split()):
            raise EnsembleError(f'{where}, sentence {j}: {len(tags)} predicted tags for '
                                f'{len(g.split())} gold tags')
        try:
            tags_to_spans(tags)
        except ValueError as e:
            raise EnsembleError(f'{where}, sentence {j}: {e}') from None


def check_runs(by_id, names):
    """Raise EnsembleError unless the runs agree on run_ids, test sentences and gold tags.

    Every member run also needs its test_pred (and the F1 values copied into member_f1).
    """
    ids = set(by_id[0])
    for runs, name in zip(by_id[1:], names[1:]):
        if set(runs) != ids:
            raise EnsembleError(f'the members have different run_ids: only in {names[0]}: '
                                f'{sorted(ids - set(runs))}, only in {name}: '
                                f'{sorted(set(runs) - ids)}')
    for run_id in by_id[0]:
        runs = [b[run_id] for b in by_id]
        for r, name in zip(runs, names):
            for key in RUN_KEYS:
                if r.get(key) is None:
                    hint = ' (run ner_baseline.py with --save-predictions)' \
                        if key.startswith('test_') else ''
                    raise EnsembleError(f'{name}, run {run_id}: no {key}{hint}')
        for key in ('test_index', 'test_gold'):
            for r, name in zip(runs[1:], names[1:]):
                if r[key] != runs[0][key]:
                    raise EnsembleError(f'run {run_id}: {name} and {names[0]} differ in {key} '
                                        f'(the members need the same split)')
        for r, name in zip(runs, names):
            check_prediction(r, f'{name}, run {run_id}')


def identical_members(by_id, names):
    """Pairs of members with the same predictions in every run (a forgotten --seed-offset?)."""
    return [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))
            if all(by_id[i][r]['test_pred'] == by_id[j][r]['test_pred'] for r in by_id[0])]


# ---------------------------------------------------------------------------
# Ensemble results
# ---------------------------------------------------------------------------
def ensemble_run(run_id, runs):
    """Metrics of the majority vote for one run_id; runs: the members' (checked) runs."""
    gold = [g.split() for g in runs[0]['test_gold']]
    preds = [[p.split() for p in r['test_pred']] for r in runs]   # member x sentence
    votes = [vote_tags(list(sentence)) for sentence in zip(*preds)]
    res = score(gold, [tags for tags, _ in votes])
    res.update({'run_id': run_id, 'n_members': len(runs),
                'member_f1': [r['f1'] for r in runs],
                'member_f1_strict': [r['f1_strict'] for r in runs],
                'invalid_i_tags': sum(bad for _, bad in votes)})
    return res


def member_gain(runs):
    """Micro F1 of the vote minus the mean of its members' micro F1, run by run.

    `ner_baseline.py --compare` pairs the ensemble with a single reference model;
    this pairs it with its own members on the same test sentences, so a lucky or
    unlucky reference run does not enter. Exploratory, not part of the decision rule.
    """
    diffs = [r['f1'] - statistics.fmean(r['member_f1']) for r in runs]
    return {'mean_diff': round(statistics.fmean(diffs), 2),
            'ci95': [round(x, 2) for x in metrics.mean_ci(diffs)['ci']],
            'per_run': [round(d, 2) for d in diffs]}


def combine(members, names, variant=None):
    """Results (ner_baseline schema) of the majority-vote ensemble of the parsed member files.

    `variant` (--variant) names a mixed ensemble: the members may differ in model and
    variant, config['models'] lists their models, and `model` is 'mixed' if they differ.
    """
    named = variant is not None
    check_members(members, names, same_model=not named)
    if not named:
        variant = ensemble_variant([m['variant'] for m in members])
    by_id = [index_runs(m, name) for m, name in zip(members, names)]
    check_runs(by_id, names)
    first = members[0]
    models = [m['model'] for m in members]
    config = {'members': list(names), 'voting': VOTING, 'k': len(members)}
    if named:
        config['models'] = models
    results = {'granularity': first['granularity'], 'protocol': first['protocol'],
               'variant': variant, 'model': models[0] if len(set(models)) == 1 else 'mixed',
               'labels': first['labels'], 'label_tie': first['label_tie'],
               'paper_test_f1': first['paper_test_f1'], 'config': config,
               'runs': [ensemble_run(run_id, [b[run_id] for b in by_id]) for run_id in by_id[0]]}
    summarize(results)
    if results['runs']:
        results['summary']['vs_member_mean'] = member_gain(results['runs'])
    return results


def run(paths, out=None, variant=None):
    """Read the member files, write the ensemble results file and print a summary."""
    members = [load_member(p) for p in paths]
    names = [os.path.basename(p) for p in paths]
    results = combine(members, names, variant)
    out = out or os.path.join(os.path.dirname(paths[0]), results_name(
        results['granularity'], results['protocol'], results['variant']))
    if os.path.realpath(out) in {os.path.realpath(p) for p in paths}:
        raise EnsembleError(f'{out} is one of the members; choose another --out')

    by_id = [index_runs(m, name) for m, name in zip(members, names)]
    for a, b in identical_members(by_id, names):
        print(f'warning: {a} and {b} predict the same tags in every run '
              f'(same seed? --seed-offset forgotten?)', file=sys.stderr)

    cfg = results['config']
    print(f"\n=== NER ensemble {results['granularity']}: {cfg['k']} members, protocol "
          f"{results['protocol']}, variant {results['variant']} ===")
    if 'models' in cfg:
        print(f"  models: {', '.join(cfg['models'])}")
    for r in results['runs']:
        invalid = f", {r['invalid_i_tags']} invalid I- tags" if r['invalid_i_tags'] else ''
        print(f"  {r['run_id']}: ensemble F1 {r['f1']:.2f} (strict {r['f1_strict']:.2f}); "
              f"members {' '.join(f'{x:.2f}' for x in r['member_f1'])}{invalid}")
    s = results['summary']
    member_mean = statistics.mean(x for r in results['runs'] for x in r['member_f1'])
    # the published numbers are in-distribution; no reference for held-out papers
    ref = ('' if results['protocol'] == 'leave_one_paper_out'
           else f"; paper {results['paper_test_f1']}")
    print(f"  micro F1 {s['micro']['mean']:.2f} 95% CI {s['micro']['ci95']} over "
          f"{s['n_runs']} runs (members' mean {member_mean:.2f}{ref})")

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Entity-level majority-vote ensemble of NER models trained on the same splits')
    parser.add_argument('--members', nargs='+', required=True, metavar='FILE',
                        help='k >= 2 results files written by ner_baseline.py --save-predictions '
                             '(--seed-offset N --tag mN)')
    parser.add_argument('--out', default=None, metavar='PATH',
                        help='output file (default: ner_results_<granularity>[_<protocol>]_'
                             '<variant>.json next to the first member)')
    parser.add_argument('--variant', type=variant_option, default=None, metavar='NAME',
                        help='variant of the ensemble and its results file (e.g. crf_mix3); the '
                             'members may then differ in model and variant (mixed backbones)')
    args = parser.parse_args(argv)
    if len(args.members) < 2:
        parser.error('--members needs at least 2 files')
    try:
        run(args.members, args.out, args.variant)
    except EnsembleError as e:
        sys.exit(f'error: {e}')


if __name__ == '__main__':
    main()
