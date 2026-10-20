"""
silver_lexicon.py - what the silver filter admits of a topic corpus, and the dictionary ceiling

The CPU probe of week 12's route (docs/METHODS_PLAN.md §4B; no decision). For one held-out
MuLMS subfield (week 10's fold: --train-pool 400, split seed 0):
  1. the fold's 3 CRF teachers (model seeds 0, 100, 200) are trained here, on the CPU unless
     --device says otherwise (their numbers differ from the GPU runs', which does not matter
     for a probe);
  2. each tags the subfield's topic corpus (build_topic_corpus.py), with its CRF marginals;
  3. the lexicon of the sentence filter (ner_baseline.unanimous_silver: every teacher gives
     the same tags, every tag's marginal is at least t under every teacher, the sentence has an
     entity) and of the mention filter (ner_baseline.silver_mention_names: a span every
     teacher tags alike, each of its tags at least t under every teacher), at THRESHOLDS;
  4. per lexicon: its size and its coverage of the held-out subfield's unseen test names
     (ner_baseline.name_coverage): the pool a student would see;
  5. the dictionary probe on week 10's saved CRF predictions (10 seeds; RESULTS_DIR): every
     maximal verbatim match of a lexicon name in a test sentence becomes that entity, either
     only where the model tagged every word of it O ('fill') or whatever it tagged
     ('override'); strict micro P, R and F1 before and after, per seed and their mean; the
     same with the subfield's gold unseen test names and with all its gold test names as the
     leaky ceilings of the rule.
Outputs: LEXICON_DIR/<subfield>.json (the names with their counts; local, never versioned: the
corpus is not redistributed) and results/silver_lexicons/<subfield>.json (counts and scores).

Usage:
  python src/graph/silver_lexicon.py --subfield sofc [--corpus PATH] [--max-sentences N]
      [--threads 2] [--device cpu] [--epochs 20] [--max-train 0]
"""

import argparse
import json
import os
import pickle
import time
from collections import Counter

import dapt_mlm
import ner_baseline as nb
from config import PROCESSED_DIR, PROJECT_ROOT

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'gpu')
OUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'silver_lexicons')
LEXICON_DIR = os.path.join(PROCESSED_DIR, 'silver')
CORPUS = os.path.join(PROCESSED_DIR, 'corpus', 'topic_{subfield}_sentences.txt')
CRF_FILE = 'ner_results_mulms_leave_one_subfield_out_crf_m{offset}.json'
THRESHOLDS = (0.9, 0.7, 0.5)
TEACHER_OFFSETS = (0, 100, 200)
SEED_OFFSETS = tuple(range(0, 1000, 100))
POOL, SPLIT_SEED = 400, 0
MAX_NAME = nb.LEXICON_MAX_WORDS   # words of a name the dictionary rule matches


def read_corpus(path, limit=0):
    """The corpus's sentences as word lists (one per line, words separated by spaces)."""
    with open(path, encoding='utf-8') as f:
        sentences = [line.split() for line in f if line.strip()]
    return sentences[:limit] if limit else sentences


def fold_of(subfield, data):
    """(training, validation, test) positions of week 10's fold of the subfield."""
    folds = {name: (tr, va, te) for name, tr, va, te in nb.subfield_folds(
        data, nb.mulms_subfields(), SPLIT_SEED, POOL, indices=True)}
    if subfield not in folds:
        raise SystemExit(f'no subfield {subfield!r}; known: {sorted(folds)}')
    return folds[subfield]


def lexicons_from_votes(sentences, votes, thresholds=THRESHOLDS):
    """{'sentences_<t>' | 'mentions_<t>': [(type, words)]}: the silver names of each filter at
    each threshold; votes: per sentence, per teacher (tags, least marginal, marginals)."""
    out = {}
    for t in thresholds:
        silver = []
        for words, teachers in zip(sentences, votes):
            tags = teachers[0][0]
            if (all(other[0] == tags for other in teachers)
                    and any(tag.startswith('B-') for tag in tags)
                    and all(least >= t for _, least, _ in teachers)):
                silver.append((words, list(tags)))
        out[f'sentences_{t}'] = nb.silver_mentions(silver)
        out[f'mentions_{t}'] = nb.silver_mention_names(sentences, votes, threshold=t)
    return out


lexicon_index = nb.lexicon_index        # {words: type}, the most frequent type per name


def apply_lexicon(words, tags, index, mode='fill'):
    """The tags with every maximal verbatim match of an index name (longest first, left to
    right) tagged as its entity: in mode 'fill' only where every word of the match is tagged
    O, in mode 'override' wherever it is (an entity cut by the match is closed before it and
    reopened after it)."""
    out = list(tags)
    longest = min(MAX_NAME, max((len(w) for w in index), default=0))
    i = 0
    while i < len(words):
        hit = None
        for n in range(min(longest, len(words) - i), 0, -1):
            kind = index.get(tuple(words[i:i + n]))
            if kind is not None and (mode == 'override' or all(t == 'O' for t in out[i:i + n])):
                hit = n, kind
                break
        if hit is None:
            i += 1
            continue
        n, kind = hit
        out[i:i + n] = [f'B-{kind}'] + [f'I-{kind}'] * (n - 1)
        if i + n < len(out) and out[i + n].startswith('I-'):
            out[i + n] = 'B-' + out[i + n][2:]
        i += n
    return out


def strict_prf(gold, pred):
    """Strict micro precision, recall and F1 (span and type) over tag sequences."""
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        gs, ps = set(nb.entity_spans(g)), set(nb.entity_spans(p))
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': round(100 * precision, 2), 'recall': round(100 * recall, 2),
            'f1': round(100 * f1, 2)}


def crf_runs(subfield, results_dir=RESULTS_DIR, offsets=SEED_OFFSETS):
    """[(offset, run)] of week 10's CRF runs of the subfield that exist."""
    runs = []
    for offset in offsets:
        path = os.path.join(results_dir, CRF_FILE.format(offset=offset))
        if not os.path.isfile(path):
            continue
        with open(path, encoding='utf-8') as f:
            results = json.load(f)
        for run in results['runs']:
            if run['run_id'] == subfield and 'test_pred' in run:
                runs.append((offset, run))
    return runs


def probe(runs, data, index):
    """The dictionary probe of an index over the runs: per mode, the strict scores before and
    after, per seed offset and their means (f1 and the F1 change)."""
    out = {}
    for mode in ('fill', 'override'):
        per_seed, before_all, after_all = {}, [], []
        for offset, run in runs:
            words = [data[i][0] for i in run['test_index']]
            gold = [g.split() for g in run['test_gold']]
            pred = [p.split() for p in run['test_pred']]
            if any(len(w) != len(g) or len(w) != len(p) for w, g, p in zip(words, gold, pred)):
                raise SystemExit(f'run m{offset}: its test sentences do not match the corpus')
            before = strict_prf(gold, pred)
            after = strict_prf(gold, [apply_lexicon(w, p, index, mode) for w, p in zip(words, pred)])
            per_seed[f'm{offset}'] = {'before': before, 'after': after,
                                      'delta_f1': round(after['f1'] - before['f1'], 2)}
            before_all.append(before['f1'])
            after_all.append(after['f1'])
        n = len(per_seed)
        out[mode] = {'n_seeds': n, 'per_seed': per_seed,
                     'f1_before': round(sum(before_all) / n, 2) if n else None,
                     'f1_after': round(sum(after_all) / n, 2) if n else None,
                     'delta_f1': round(sum(a - b for a, b in zip(after_all, before_all)) / n, 2)
                     if n else None}
    return out


def gold_names(rows):
    """(type, words) of every entity of the rows (words, tags, doc)."""
    return [(kind, tuple(words[s:e])) for words, tags, *_ in rows
            for s, e, kind in nb.entity_spans(tags)]


def run(args):
    import torch
    torch.set_num_threads(args.threads)
    device = torch.device(args.device or 'cpu')
    started = time.perf_counter()
    nb_args = nb.parse_args(['--granularity', 'mulms', '--leave-one-subfield-out', '--train-pool',
                             str(POOL), '--crf', '--epochs', str(args.epochs), '--max-train',
                             str(args.max_train), '--threads', str(args.threads)])
    data, labels, _ = nb.granularity_data('mulms', nb_args)
    tr, va, te = fold_of(args.subfield, data)
    train, val, test = [data[i] for i in tr], [data[i] for i in va], [data[i] for i in te]
    corpus_path = args.corpus or CORPUS.format(subfield=args.subfield)
    corpus = read_corpus(corpus_path, args.max_sentences)
    print(f'{args.subfield}: {len(train)} training, {len(val)} validation, {len(test)} test '
          f'sentences; corpus {corpus_path}: {len(corpus):,} sentences', flush=True)
    teachers, tagged, sentences, skipped = [], [], None, 0
    os.makedirs(args.lexicon_dir, exist_ok=True)
    for offset in args.offsets:
        # a teacher's votes are kept locally (never committed) so that a killed probe resumes
        # from its last finished teacher
        cache = os.path.join(args.lexicon_dir, f'{args.subfield}_votes_m{offset}.pkl')
        if os.path.isfile(cache):
            with open(cache, 'rb') as f:
                saved = pickle.load(f)
            if saved['corpus_sha256'] == dapt_mlm.file_sha256(corpus_path):
                sentences, skipped = saved['sentences'], saved['skipped']
                tagged.append(saved['votes'])
                teachers.append(saved['teacher'])
                print(f"  teacher seed {offset}: votes restored from {cache}", flush=True)
                continue
        t0 = time.perf_counter()
        record, tagger = nb.train_one(offset, train, val, test, labels, nb_args, device,
                                      keep_tagger=True)
        if sentences is None:
            sentences, skipped = nb.fit_sample(tagger.tokenizer, corpus,
                                               tagger.model.config.max_position_embeddings)
        t1 = time.perf_counter()
        out = [(tuple(tags), min(marginals), tuple(marginals))
               for tags, marginals in nb.teacher_tags(tagger, sentences, nb_args.batch_size,
                                                      device)]
        del tagger
        tagged.append(out)
        teachers.append({'seed': offset, 'val_f1': record['val_f1'], 'test_f1': record['f1'],
                         'best_epoch': record['best_epoch'],
                         'train_minutes': round((t1 - t0) / 60, 1),
                         'tag_minutes': round((time.perf_counter() - t1) / 60, 1)})
        print(f"  teacher seed {offset}: val F1 {record['val_f1']:.2f}, test F1 "
              f"{record['f1']:.2f}; trained in {teachers[-1]['train_minutes']} min, tagged "
              f"{len(sentences):,} sentences in {teachers[-1]['tag_minutes']} min", flush=True)
        with open(cache, 'wb') as f:
            pickle.dump({'corpus_sha256': dapt_mlm.file_sha256(corpus_path),
                         'sentences': sentences, 'skipped': skipped, 'votes': out,
                         'teacher': teachers[-1]}, f)
    votes = list(zip(*tagged))
    lexicons = lexicons_from_votes(sentences, votes)
    unseen = [m for m in gold_names(test) if m not in set(gold_names(train))]
    runs = crf_runs(args.subfield, args.results)
    report = {'subfield': args.subfield, 'protocol': 'docs/METHODS_PLAN.md §4B: the CPU probe, '
              'no decision', 'fold': {'train_pool': POOL, 'split_seed': SPLIT_SEED,
                                      'n_train': len(train), 'n_val': len(val),
                                      'n_test': len(test)},
              'corpus': {'path': dapt_mlm.display_path(corpus_path), 'sentences': len(corpus),
                         'tagged': len(sentences), 'skipped': skipped,
                         'sha256': dapt_mlm.file_sha256(corpus_path)},
              'teachers': teachers, 'device': str(device), 'threads': args.threads,
              'thresholds': list(THRESHOLDS), 'lexicons': {}, 'probe': {},
              'crf_runs': [f'm{off}' for off, _ in runs]}
    local = {}
    for key, names in lexicons.items():
        coverage = nb.name_coverage(names, train, test)
        counts = Counter(kind for kind, _ in names)
        report['lexicons'][key] = {
            **coverage, 'per_type': dict(sorted(counts.items())),
            'unseen_share_in_lexicon': (round(coverage['n_test_unseen_in_silver']
                                              / coverage['n_test_unseen'], 3)
                                        if coverage['n_test_unseen'] else None)}
        local[key] = {kind: dict(Counter(' '.join(w) for k, w in names if k == kind))
                      for kind in counts}
        if runs:
            report['probe'][key] = probe(runs, data, lexicon_index(names))
        print(f"  {key}: {coverage['n_mentions']:,} mentions, {coverage['n_distinct']:,} names; "
              f"{100 * (report['lexicons'][key]['unseen_share_in_lexicon'] or 0):.1f}% of the "
              f"{coverage['n_test_unseen']} unseen test names"
              + (f"; fill {report['probe'][key]['fill']['delta_f1']:+.2f}, override "
                 f"{report['probe'][key]['override']['delta_f1']:+.2f} F1" if runs else ''),
              flush=True)
    if runs:
        report['probe']['ceiling_unseen_gold'] = probe(runs, data, lexicon_index(unseen))
        report['probe']['ceiling_all_gold'] = probe(runs, data, lexicon_index(gold_names(test)))
        for key in ('ceiling_unseen_gold', 'ceiling_all_gold'):
            print(f"  {key}: fill {report['probe'][key]['fill']['delta_f1']:+.2f}, override "
                  f"{report['probe'][key]['override']['delta_f1']:+.2f} F1", flush=True)
    report['runtime_minutes'] = round((time.perf_counter() - started) / 60, 1)
    os.makedirs(args.lexicon_dir, exist_ok=True)
    with open(os.path.join(args.lexicon_dir, f'{args.subfield}.json'), 'w', encoding='utf-8') as f:
        json.dump(local, f, indent=1)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'{args.subfield}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    print(f"Saved {path} ({report['runtime_minutes']} min)")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='The CPU probe of the silver lexicons of a '
                                                 "held-out MuLMS subfield's topic corpus")
    parser.add_argument('--subfield', required=True)
    parser.add_argument('--corpus', default=None, metavar='PATH')
    parser.add_argument('--max-sentences', type=int, default=0, metavar='N',
                        help='the first N corpus sentences (0: all)')
    parser.add_argument('--offsets', type=int, nargs='+', default=list(TEACHER_OFFSETS))
    parser.add_argument('--results', default=RESULTS_DIR, metavar='DIR')
    parser.add_argument('--out-dir', default=OUT_DIR, metavar='DIR')
    parser.add_argument('--lexicon-dir', default=LEXICON_DIR, metavar='DIR')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--max-train', type=int, default=0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
