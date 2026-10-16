"""
ner_matching.py - the saved NER predictions scored under looser matching criteria (exploratory)

seqeval's micro F1 counts a prediction only if its span and its type are exactly those of a gold
entity. SemEval-2013 Task 9.1 (Segura-Bedmar et al., 2013) also scored entity recognition with
looser criteria. This module scores every stored run with predictions under four of them:
  - strict: the same span and type (seqeval's micro F1, default mode);
  - exact: the same span, any type;
  - type: an overlapping span of the same type;
  - overlap: an overlapping span, any type.
In a sentence each gold entity pairs with at most one prediction and each prediction with at
most one gold entity: pairs with the same span first, then the other overlapping pairs, the
largest overlap first (under type, only pairs of the same type). A paired overlap counts as a
full match, so F1 is 2 x pairs / (gold entities + predictions). No training and no decision.

Output, results/gpu/ner_matching.json:
  - systems: per granularity and protocol, for each system whose files are all there (the CRF
    head and the multi-task model with their 10 seeds, and their votes over the seed triples,
    made again from the members' predictions; plain MatSciBERT and the CRF head on the published
    splits), each criterion's F1: the mean over the random splits or held-out papers, each the
    mean over the seeds or votes (on a published split, the mean over the seeds), with its 95%
    Student-t CI;
  - gap: per granularity and system, each criterion's mean F1 on the random splits minus that
    on held-out papers;
  - near_misses: per granularity and protocol, the CRF head's predictions over all its runs:
    the shares that are right, of the right span and another type (fine-grained, also the
    share whose type is an MMO ancestor or descendant of the gold one), inside a longer gold
    entity, around a shorter one, crossing one, or overlapping none (spurious).

Usage:
  python src/graph/ner_matching.py [--results results/gpu] [--out PATH]
"""

import argparse
import json
import os
from pathlib import Path

from ner_ensemble import tags_to_spans
from ner_mentions import family_files, load, votes
from ner_seedmeans import GRANULARITIES, LOPO, PROTOCOLS, spread

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results' / 'gpu'
OUT_NAME = 'ner_matching.json'
CRITERIA = ('strict', 'exact', 'type', 'overlap')
RANDOM = 'random_split_vary'
PUBLISHED = 'published_split'
FAMILIES = ('crf', 'crf_mt')                 # 10 seeds on both protocols; and their votes
PUBLISHED_SPLITS = ('fine', 'coarse', 'sofc', 'sofc_slot')
PUBLISHED_SYSTEMS = {'baseline': '', 'crf': '_crf'}


def paired(gold, pred, same_type):
    """The number of one-to-one pairs of overlapping gold and predicted spans (of the same type
    if same_type): pairs with the same span first, then the largest overlap first."""
    pairs = []
    for g in gold:
        for p in pred:
            overlap = min(g[1], p[1]) - max(g[0], p[0])
            if overlap > 0 and (not same_type or g[2] == p[2]):
                pairs.append((g[:2] != p[:2], -overlap, g, p))
    used_gold, used_pred = set(), set()
    for _, _, g, p in sorted(pairs):
        if g not in used_gold and p not in used_pred:
            used_gold.add(g)
            used_pred.add(p)
    return len(used_gold)


def sentence_counts(gold_tags, pred_tags):
    """{'gold', 'pred', criterion: matches} of one sentence's tag sequences."""
    gold = set(tags_to_spans(gold_tags)[0])
    pred = set(tags_to_spans(pred_tags)[0])
    return {'gold': len(gold), 'pred': len(pred),
            'strict': len(gold & pred),
            'exact': len({g[:2] for g in gold} & {p[:2] for p in pred}),
            'type': paired(gold, pred, same_type=True),
            'overlap': paired(gold, pred, same_type=False)}


def run_f1(run):
    """{criterion: F1 in %} of a stored run's test predictions."""
    total = dict.fromkeys(('gold', 'pred', *CRITERIA), 0)
    for gold, pred in zip(run['test_gold'], run['test_pred'], strict=True):
        for key, n in sentence_counts(gold.split(), pred.split()).items():
            total[key] += n
    n = total['gold'] + total['pred']
    return {c: round(200 * total[c] / n, 2) if n else 0.0 for c in CRITERIA}


def member_means(files):
    """{run_id: {criterion: F1}}: the mean over the files (members) of each run_id's F1."""
    by_id = {}
    for f in files:
        for run in f['runs']:
            by_id.setdefault(run['run_id'], []).append(run_f1(run))
    return {i: {c: sum(r[c] for r in rows) / len(rows) for c in CRITERIA}
            for i, rows in by_id.items()}


def summary(per_pair):
    """{criterion: spread over the pairs (splits, papers or seeds)}, with the pairs' count."""
    out = {c: spread([v[c] for v in per_pair.values()]) for c in CRITERIA}
    out['n_pairs'] = len(per_pair)
    return out


def mmo_related(a, b):
    """Whether the MMO classes a and b are the same or one is a subclass of the other."""
    from ontology_utils import MMO_NS, load_ontology
    ontology = load_ontology()
    return ontology.is_subclass(MMO_NS + a, MMO_NS + b) or ontology.is_subclass(MMO_NS + b,
                                                                             MMO_NS + a)


NEAR_MISSES = ('right', 'wrong_type', 'wrong_type_related', 'inside_longer_gold',
               'around_shorter_gold', 'crossing', 'spurious')


def near_misses(files, related=None):
    """Shares of the predictions of every run of the files (see the module docstring); related(a,
    b) tells whether two types are related (wrong_type_related is left out without it)."""
    counts = dict.fromkeys(NEAR_MISSES, 0)
    n = 0
    for f in files:
        for run in f['runs']:
            for gold_tags, pred_tags in zip(run['test_gold'], run['test_pred'], strict=True):
                gold = set(tags_to_spans(gold_tags.split())[0])
                for p in tags_to_spans(pred_tags.split())[0]:
                    n += 1
                    same = [g for g in gold if g[:2] == p[:2]]
                    over = [g for g in gold if g[0] < p[1] and p[0] < g[1]]
                    if p in gold:
                        counts['right'] += 1
                    elif same:
                        counts['wrong_type'] += 1
                        counts['wrong_type_related'] += bool(related and related(p[2],
                                                                                 same[0][2]))
                    elif over:
                        g = max(over, key=lambda g: (min(g[1], p[1]) - max(g[0], p[0]), g))
                        counts['inside_longer_gold' if g[0] <= p[0] and p[1] <= g[1] else
                               'around_shorter_gold' if p[0] <= g[0] and g[1] <= p[1] else
                               'crossing'] += 1
                    else:
                        counts['spurious'] += 1
    out = {'n_pred': n, **{k: round(v / n, 3) if n else 0.0 for k, v in counts.items()}}
    if related is None:
        del out['wrong_type_related']
    return out


def held_out_systems(directory, granularity, protocol):
    """{system: per-pair F1} of the families with all 10 seeds and of their votes."""
    out = {}
    for family in FAMILIES:
        files = family_files(directory, granularity, protocol, family)
        if files:
            out[family] = member_means(files)
            out[f'{family}_ens3'] = member_means(votes(files))
    return out


def analyse(directory):
    """The criteria for every granularity, protocol and published split with the files."""
    out = {'criteria': {'strict': 'same span and type', 'exact': 'same span, any type',
                        'type': 'overlapping span, same type',
                        'overlap': 'overlapping span, any type'},
           'systems': {}, 'gap': {}, 'near_misses': {}}
    for g in GRANULARITIES:
        found = {}
        for protocol in PROTOCOLS:
            systems = held_out_systems(directory, g, protocol)
            if systems:
                found[protocol] = systems
                out['systems'][f'{g}_{protocol}'] = {s: summary(v) for s, v in systems.items()}
            crf = family_files(directory, g, protocol, 'crf')
            if crf:
                out['near_misses'][f'{g}_{protocol}'] = near_misses(
                    crf, mmo_related if g == 'fine' else None)
        if len(found) == len(PROTOCOLS):
            out['gap'][g] = {
                s: {c: round(out['systems'][f'{g}_{RANDOM}'][s][c]['mean']
                             - out['systems'][f'{g}_{LOPO}'][s][c]['mean'], 2)
                    for c in CRITERIA}
                for s in found[RANDOM] if s in found[LOPO]}
    for g in PUBLISHED_SPLITS:
        systems = {}
        for system, suffix in PUBLISHED_SYSTEMS.items():
            files = load(directory, [f'ner_results_{g}_{PUBLISHED}{suffix}.json'])
            if files:
                systems[system] = summary({r['run_id']: run_f1(r) for r in files[0]['runs']})
        if systems:
            out['systems'][f'{g}_{PUBLISHED}'] = systems
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='Saved NER predictions under looser matching '
                                                 'criteria (exploratory; no training)')
    parser.add_argument('--results', default=str(RESULTS_DIR), metavar='DIR')
    parser.add_argument('--out', default=None, metavar='PATH')
    args = parser.parse_args(argv)
    result = analyse(args.results)
    for key, systems in result['systems'].items():
        for system, s in systems.items():
            print(f'  {key} {system}: ' + ', '.join(f"{c} {s[c]['mean']:.2f} {s[c]['ci95']}"
                                                   for c in CRITERIA))
    for key, shares in result['near_misses'].items():
        print(f'  {key}, the CRF head\'s predictions: ' + ', '.join(
            f'{k} {100 * v:.1f}%' for k, v in shares.items() if k != 'n_pred'))
    for g, systems in result['gap'].items():
        for system, gaps in systems.items():
            print(f'  {g}-grained {system}, random splits minus held-out papers: '
                  + ', '.join(f'{c} {d:.2f}' for c, d in gaps.items()))
    out = args.out or os.path.join(args.results, OUT_NAME)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out}")
    return result


if __name__ == '__main__':
    main()
