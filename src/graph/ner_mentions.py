"""
ner_mentions.py - seen and unseen test mentions of the saved NER runs (exploratory)

Augenstein et al. (2017) split a test set's entities by whether their surface form is an entity
of the training data, and Babl et al. (2025) show that random splits of NER corpora put many of
a test set's names into its training set. This module splits the gold test entities of every
stored run with predictions (ner_baseline.py --save-predictions) in that way:
  - seen: its words are, exactly, an entity of the same type in the run's training sentences;
  - seen_other: they are an entity there, but only of other types;
  - unseen: they are no entity there.
The training sentences are the ones the run's model learned from, not its validation
sentences. The run's split is rebuilt as ner_baseline.py builds it (a random split from the
run's split seed, a held-out paper's fold, a fold of --document-folds, the published split),
and its test sentences and
gold tags must be the stored ones. A predicted entity takes a kind in the same way, by its words
and its predicted type, so that recall, precision and F1 are given per kind; a correct
prediction has the kind of its gold entity. Votes (<family>_ens3 over the seed triples 0-200,
300-500 and 600-800) are made again from their members' predictions (ner_ensemble.ensemble_run).

No training and no decision. Output, results/gpu/ner_mentions.json:
  - systems: per granularity and protocol (MaterioMiner's random splits and held-out papers,
    SOFC-Exp's random splits and article folds, the published splits), for each system whose
    files are all there, the
    counts of gold, predicted and correct entities of each kind over all its runs (seeds x
    splits or papers), the recall, precision and F1 they give, and each kind's share of the
    gold entities. A family has its 10 seeds, or the screen's 3 (0, 100, 200) if only those
    are there;
  - papers: per granularity, each held-out paper's number of gold entities and shares of the
    kinds;
  - types: for the CRF head on held-out papers and on the published splits, per entity type
    (the most frequent first), its gold entities per run, the share of them that are unseen,
    and the recall on its seen and on its unseen ones;
  - gap: per granularity, the CRF head's recall on the random splits and on held-out papers
    (SOFC-Exp: article folds),
    and two counterfactual recalls: held-out papers' recall per kind with the random splits'
    shares of the kinds, and the random splits' recall per kind with held-out papers' shares;
  - methods: on held-out papers and published splits, each method against the CRF head with
    the same seeds (and each vote against the CRF head's vote of the same triple), for the
    seen and unseen kinds and all entities: F1 per pair (held-out paper, 10-seed mean; or
    model seed on a published split) and the mean paired difference with its 95% Student-t CI,
    the rule of the decisions, here without a decision.

Usage:
  python src/graph/ner_mentions.py [--results results/gpu] [--out PATH]
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import metrics
from ner_baseline import (DOCUMENT_FOLDS, document_groups, granularity_data, paper_fold,
                          published_split, random_split)
from ner_ensemble import ensemble_run, tags_to_spans
from ner_seedmeans import LOPO, SCREEN_OFFSETS, WEEK5_OFFSETS, WEEK5_TRIPLES, seed_family

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'gpu'
OUT_NAME = 'ner_mentions.json'
KINDS = ('seen', 'seen_other', 'unseen')
COMPARED = ('seen', 'unseen', 'all')
TOP_K = 27                  # the fine-grained types kept, as in every stored run
RANDOM = 'random_split_vary'
PUBLISHED = 'published_split'
REFERENCE = 'crf'
# per granularity: the random splits and the protocol of unseen documents (MaterioMiner's
# held-out papers; SOFC-Exp's article folds, week 8)
HELD_OUT = {'fine': (RANDOM, LOPO), 'coarse': (RANDOM, LOPO), 'sofc': (RANDOM, DOCUMENT_FOLDS)}
# families with 10 seeds per split (or the screen's 3) on the random splits and held-out papers
FAMILIES = ('crf', 'crf_mt', 'crf_ls', 'crf_st3', 'crf_ctx', 'crf_mt_st3', 'crf_mr')
VOTED = ('crf', 'crf_mt', 'crf_ctx', 'crf_mt_st3', 'crf_mr')
# a published split's systems: plain MatSciBERT, the CRF head and the methods run on it
PUBLISHED_SPLITS = ('fine', 'coarse', 'sofc', 'sofc_slot')
PUBLISHED_METHODS = ('crf_ctx', 'crf_mt', 'crf_mr', 'crf_cdetest')
BASELINE = 'baseline'
RETOKENIZED = ('crf_cdetest', 'cdetest')     # scored on SOFC-Exp's retokenized test articles
# what a published split's system is set against (the methods: the CRF head)
PUBLISHED_REFERENCE = {REFERENCE: BASELINE, 'cdetest': BASELINE}


class MentionError(ValueError):
    """A stored run that does not match the split rebuilt for it."""


_CORPORA = {}


def corpus(granularity, retokenized_test=False):
    """The data of a granularity as ner_baseline.py trains on it (top-27 fine-grained types),
    cached; retokenized_test: SOFC-Exp's test articles cut as its training articles are."""
    key = granularity, retokenized_test
    if key not in _CORPORA:
        args = argparse.Namespace(top_k=TOP_K, published_split=False,
                                  retokenized_test=retokenized_test)
        _CORPORA[key] = granularity_data(granularity, args)[0]
    return _CORPORA[key]


def run_split(results, run, data, granularity):
    """(data, training positions, test positions) of a stored run, its split rebuilt as
    ner_baseline.py builds it; the data has the release's tags on a published split."""
    protocol, run_id = results['protocol'], run['run_id']
    config = results.get('config') or {}
    split_seed = config.get('split_seed') or 0
    if protocol == RANDOM:
        train, _, test = random_split(len(data), int(run_id.removeprefix('seed')))
    elif protocol == 'random_split_fixed':
        train, _, test = random_split(len(data), split_seed)
    elif protocol == LOPO:
        train, _, test = paper_fold(data, (run_id,), split_seed, indices=True)
    elif protocol == DOCUMENT_FOLDS:
        groups = document_groups({d for _, _, d in data}, config['document_folds'], split_seed)
        train, _, test = paper_fold(data, groups[int(run_id.removeprefix('fold'))], split_seed,
                                    indices=True)
    elif protocol == PUBLISHED:
        (train, _, test), data = published_split(data, granularity)
    else:
        raise MentionError(f'run {run_id}: no split for protocol {protocol}')
    if config.get('max_train'):
        train = train[:config['max_train']]
    if test != run['test_index']:
        raise MentionError(f'run {run_id}: its test sentences are not those of the rebuilt '
                           f'{protocol} split')
    return data, train, test


def entity_forms(data, positions):
    """{words: {types}} of the gold entities of the sentences of `data` at `positions`."""
    forms = {}
    for i in positions:
        words, tags, _ = data[i]
        for start, end, kind in tags_to_spans(tags)[0]:
            forms.setdefault(tuple(words[start:end]), set()).add(kind)
    return forms


def kind_of(words, entity_type, forms):
    """seen, seen_other or unseen: the kind of an entity with these words and this type."""
    types = forms.get(tuple(words))
    if types is None:
        return 'unseen'
    return 'seen' if entity_type in types else 'seen_other'


def run_counts(results, data, granularity):
    """{run_id: Counter} of a parsed results file: ('gold', kind), ('pred', kind) and
    ('correct', kind) over each run's test sentences, and ('gold', kind, type) and
    ('correct', kind, type) per entity type."""
    out = {}
    for run in results['runs']:
        missing = [k for k in ('test_index', 'test_gold', 'test_pred') if k not in run]
        if missing:
            raise MentionError(f"run {run['run_id']}: no {', '.join(missing)}")
        run_data, train, test = run_split(results, run, data, granularity)
        forms = entity_forms(run_data, train)
        counts = Counter()
        for i, gold_tags, pred_tags in zip(test, run['test_gold'], run['test_pred'],
                                           strict=True):
            words, tags, _ = run_data[i]
            if gold_tags.split() != list(tags):
                raise MentionError(f"run {run['run_id']}: the gold tags of test sentence {i} "
                                   f"are not the corpus's")
            gold = set(tags_to_spans(gold_tags.split())[0])
            pred = set(tags_to_spans(pred_tags.split())[0])
            for start, end, kind in gold:
                k = kind_of(words[start:end], kind, forms)
                found = (start, end, kind) in pred
                counts['gold', k] += 1
                counts['correct', k] += found
                counts['gold', k, kind] += 1
                counts['correct', k, kind] += found
            for start, end, kind in pred:
                counts['pred', kind_of(words[start:end], kind, forms)] += 1
        out[run['run_id']] = counts
    return out


def kind_scores(counts, kinds):
    """Recall, precision and F1 (in %) of the entities of `kinds` in a Counter, with counts."""
    gold, pred, correct = (sum(counts[part, k] for k in kinds)
                           for part in ('gold', 'pred', 'correct'))
    recall = 100 * correct / gold if gold else 0.0
    precision = 100 * correct / pred if pred else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {'n_gold': gold, 'n_pred': pred, 'n_correct': correct, 'recall': round(recall, 2),
            'precision': round(precision, 2), 'f1': round(f1, 2)}


def scores(counts):
    """Per kind and for all entities: kind_scores and the kind's share of the gold entities."""
    n = sum(counts['gold', k] for k in KINDS)
    out = {}
    for k in (*KINDS, 'all'):
        out[k] = kind_scores(counts, KINDS if k == 'all' else (k,))
        out[k]['share'] = round(out[k]['n_gold'] / n, 3) if n else 0.0
    return out


def type_kinds(counts, n_runs):
    """Per entity type, the most frequent first: its gold entities per run, the share of them
    that are unseen, and the recall on its seen and unseen ones (None without any)."""
    types = {key[2] for key in counts if len(key) == 3}
    rows = {}
    for t in types:
        gold = {k: counts['gold', k, t] for k in KINDS}
        n = sum(gold.values())
        rows[t] = {'n_gold_per_run': round(n / n_runs, 1),
                   'unseen_share': round(gold['unseen'] / n, 3),
                   **{f'recall_{k}': (round(100 * counts['correct', k, t] / gold[k], 2)
                                      if gold[k] else None) for k in ('seen', 'unseen')}}
    return dict(sorted(rows.items(), key=lambda kv: (-kv[1]['n_gold_per_run'], kv[0])))


def votes(member_files):
    """Results of the votes over the seed triples 0-200, 300-500 and 600-800 of a family's 10
    member files (in seed order), with their predictions, run by run_id."""
    out = []
    for first in WEEK5_TRIPLES:
        trio = member_files[first // 100:first // 100 + 3]
        by_id = [{r['run_id']: r for r in f['runs']} for f in trio]
        out.append({'protocol': trio[0]['protocol'], 'config': trio[0].get('config'),
                    'runs': [ensemble_run(i, [b[i] for b in by_id], save_predictions=True)
                             for i in by_id[0] if all(i in b for b in by_id)]})
    return out


def load(directory, names):
    """The parsed results files, or None unless all exist and keep their predictions."""
    paths = [os.path.join(directory, n) for n in names]
    if not all(os.path.exists(p) for p in paths):
        return None
    files = []
    for p in paths:
        with open(p) as f:
            files.append(json.load(f))
    if not all('test_pred' in r for f in files for r in f['runs']):
        return None
    return files


def family_files(directory, granularity, protocol, family, offsets=WEEK5_OFFSETS):
    """A family's member files with the given seed offsets, or None."""
    variants = seed_family(family, untagged_zero=family == 'crf_mt', offsets=offsets)
    return load(directory, [f'ner_results_{granularity}_{protocol}_{v}.json' for v in variants])


def family_counts(files, data, granularity):
    """[{run_id: Counter}] of a family's member files (or votes)."""
    return [run_counts(f, data, granularity) for f in files]


def pooled(members):
    """The Counter of every run of every member."""
    total = Counter()
    for runs in members:
        for counts in runs.values():
            total.update(counts)
    return total


def summary(members):
    """scores of the pooled counts, with the number of runs."""
    return {'n_runs': sum(len(runs) for runs in members), **scores(pooled(members))}


def paired(method, reference):
    """For each compared kind: the method's and the reference's F1 per pair and their mean
    paired difference with its 95% CI. method, reference: [{run_id: Counter}] with members
    matched by position; a pair is a run_id, its F1 the mean over the members (a held-out
    paper over the seeds or votes; on a published split, one model seed)."""
    out = {}
    for kind in COMPARED:
        kinds = KINDS if kind == 'all' else (kind,)
        values = {}
        for ours, ref in zip(method, reference, strict=True):
            for run_id, counts in ref.items():
                if run_id not in ours:
                    continue
                pair = values.setdefault(run_id, ([], []))
                pair[0].append(kind_scores(ours[run_id], kinds)['f1'])
                pair[1].append(kind_scores(counts, kinds)['f1'])
        pairs = sorted(values)
        ours = [sum(values[p][0]) / len(values[p][0]) for p in pairs]
        ref = [sum(values[p][1]) / len(values[p][1]) for p in pairs]
        diffs = [a - b for a, b in zip(ours, ref)]
        lo, hi = metrics.mean_ci(diffs)['ci']
        out[kind] = {'method': round(sum(ours) / len(ours), 2),
                     'reference': round(sum(ref) / len(ref), 2),
                     'mean_diff': round(sum(diffs) / len(diffs), 2),
                     'ci95': [round(lo, 2), round(hi, 2)], 'n_pairs': len(pairs)}
    return out


def held_out_systems(directory, granularity, protocol):
    """{system: [{run_id: Counter}]} of the families and votes of one protocol, each family with
    its 10 seeds or, failing that, the screen's 3; and {system: offsets}."""
    data = corpus(granularity)
    systems, offsets = {}, {}
    for family in FAMILIES:
        for seeds in (WEEK5_OFFSETS, SCREEN_OFFSETS):
            files = family_files(directory, granularity, protocol, family, seeds)
            if files:
                systems[family] = family_counts(files, data, granularity)
                offsets[family] = seeds
                if family in VOTED and seeds == WEEK5_OFFSETS:
                    systems[f'{family}_ens3'] = family_counts(votes(files), data, granularity)
                    offsets[f'{family}_ens3'] = WEEK5_TRIPLES
                break
    return systems, offsets


def reference_counts(directory, granularity, protocol, offsets):
    """The CRF head's counts with the given seed offsets."""
    files = family_files(directory, granularity, protocol, REFERENCE, offsets)
    return family_counts(files, corpus(granularity), granularity) if files else None


def gap(random, held_out):
    """The CRF head's recall under both protocols and the two counterfactual mixes."""
    def mixed(recalls, shares):
        return round(sum(shares[k]['share'] * recalls[k]['recall'] for k in KINDS), 2)
    return {'random_splits_recall': random['all']['recall'],
            'held_out_recall': held_out['all']['recall'],
            'held_out_recall_with_random_mix': mixed(held_out, random),
            'random_recall_with_held_out_mix': mixed(random, held_out),
            'unseen_share': {'random_splits': random['unseen']['share'],
                             'held_out': held_out['unseen']['share']}}


def published_systems(directory, granularity):
    """{system: [{run_id: Counter}]} on a published split: one member per file."""
    names = {BASELINE: f'ner_results_{granularity}_{PUBLISHED}.json',
             REFERENCE: f'ner_results_{granularity}_{PUBLISHED}_crf.json',
             **{m: f'ner_results_{granularity}_{PUBLISHED}_{m}.json'
                for m in PUBLISHED_METHODS},
             'cdetest': f'ner_results_{granularity}_{PUBLISHED}_cdetest.json'}
    systems = {}
    for system, name in names.items():
        files = load(directory, [name])
        if files:
            data = corpus(granularity, retokenized_test=system in RETOKENIZED)
            systems[system] = family_counts(files, data, granularity)
    return systems


def held_out_comparisons(directory, granularity, protocol, systems, offsets):
    """{system: paired} of each method against the CRF head with the same seeds, and of each
    vote against the CRF head's vote, on one protocol."""
    comparisons = {}
    for system, members in systems.items():
        if system in (REFERENCE, f'{REFERENCE}_ens3'):
            continue
        if system.endswith('_ens3'):
            reference, ref = f'{REFERENCE}_ens3', systems.get(f'{REFERENCE}_ens3')
        elif offsets[system] == WEEK5_OFFSETS:
            reference, ref = REFERENCE, systems.get(REFERENCE)
        else:
            reference = REFERENCE
            ref = reference_counts(directory, granularity, protocol, offsets[system])
        if ref:
            comparisons[system] = {'reference': reference, 'seeds': list(offsets[system]),
                                   **paired(members, ref)}
    return comparisons


def analyse(directory):
    """The analyses of every granularity and protocol that has the files (module docstring)."""
    out = {'kinds': {'seen': 'its words are an entity of the same type in the training '
                             'sentences',
                     'seen_other': 'they are an entity there, only of other types',
                     'unseen': 'they are no entity there'},
           'systems': {}, 'papers': {}, 'types': {}, 'gap': {}, 'methods': {}}
    for g, protocols in HELD_OUT.items():
        reference = {}
        for protocol in protocols:
            systems, offsets = held_out_systems(directory, g, protocol)
            if not systems:
                continue
            key = f'{g}_{protocol}'
            out['systems'][key] = {s: {'seeds': list(offsets[s]), **summary(m)}
                                   for s, m in systems.items()}
            out['methods'][key] = held_out_comparisons(directory, g, protocol, systems, offsets)
            if REFERENCE not in systems:
                continue
            reference[protocol] = scores(pooled(systems[REFERENCE]))
            if protocol == protocols[1]:
                # a document's shares of the kinds do not depend on the model: the first seed's
                out['papers'][g] = {run_id: {'n_gold': s['all']['n_gold'],
                                             **{k: s[k]['share'] for k in KINDS}}
                                    for run_id, s in ((i, scores(c)) for i, c in
                                                      systems[REFERENCE][0].items())}
                out['types'][key] = type_kinds(pooled(systems[REFERENCE]),
                                               sum(map(len, systems[REFERENCE])))
        if len(reference) == len(protocols):
            out['gap'][g] = gap(reference[RANDOM], reference[protocols[1]])
    for g in PUBLISHED_SPLITS:
        systems = published_systems(directory, g)
        if not systems:
            continue
        key = f'{g}_{PUBLISHED}'
        out['systems'][key] = {s: summary(m) for s, m in systems.items()}
        if REFERENCE in systems:
            out['types'][key] = type_kinds(pooled(systems[REFERENCE]),
                                           sum(map(len, systems[REFERENCE])))
        comparisons = {}
        for system, members in systems.items():
            reference = PUBLISHED_REFERENCE.get(system, REFERENCE)
            if system != BASELINE and reference in systems:
                comparisons[system] = {'reference': reference,
                                       **paired(members, systems[reference])}
        out['methods'][key] = comparisons
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='Seen and unseen test mentions of the saved '
                                                 'NER runs (exploratory; no training)')
    parser.add_argument('--results', default=str(RESULTS_DIR), metavar='DIR')
    parser.add_argument('--out', default=None, metavar='PATH')
    args = parser.parse_args(argv)
    result = analyse(args.results)
    for key, systems in result['systems'].items():
        for system, s in systems.items():
            print(f"  {key} {system}: unseen {100 * s['unseen']['share']:.1f}% of "
                  f"{s['all']['n_gold']} entities; recall seen {s['seen']['recall']:.1f}, "
                  f"unseen {s['unseen']['recall']:.1f}; F1 seen {s['seen']['f1']:.1f}, unseen "
                  f"{s['unseen']['f1']:.1f}, all {s['all']['f1']:.1f}")
    for g, entry in result['gap'].items():
        print(f"  {g}-grained recall of the CRF head: random splits "
              f"{entry['random_splits_recall']:.1f}, held-out papers {entry['held_out_recall']:.1f}"
              f"; held-out recall per kind with the random mix "
              f"{entry['held_out_recall_with_random_mix']:.1f}, random recall per kind with the "
              f"held-out mix {entry['random_recall_with_held_out_mix']:.1f}")
    for key, methods in result['methods'].items():
        for system, m in methods.items():
            print(f"  {key} {system} minus the reference, F1: "
                  + ', '.join(f"{k} {m[k]['mean_diff']:+.2f} {m[k]['ci95']}" for k in COMPARED))
    out = args.out or os.path.join(args.results, OUT_NAME)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out}")
    return result


if __name__ == '__main__':
    main()
