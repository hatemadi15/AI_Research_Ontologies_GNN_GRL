"""
mulms_coverage.py - does an unlabeled corpus hold a held-out MuLMS subfield's unseen names?

Week 12, phase 0 (docs/ROADMAP.md §4): the gate of the topic-corpus route. No training and no
decision. For each held-out subfield (the folds of week 10's CRF runs, rebuilt as
ner_mentions.py rebuilds them), the gold test entities whose words are no entity of the fold's
training sentences are *unseen* (ner_mentions.py). For each corpus given:
  - leaked: the corpus sentences sharing a run of build_topic_corpus.SHINGLE characters (50)
    of their text without spaces, in lower case, with a sentence of the subfield's documents (a
    MuLMS article in the corpus, build_topic_corpus.shingles); they are left out of the counts;
  - exact: the share of the unseen test entities, and of their distinct names, whose words
    occur in a row in a remaining sentence. A corpus's words are its space-separated words
    (--corpus NAME=PATH, or NAME=PATH:raw for a corpus of raw sentences, cut like the
    MaterioMiner files by ner_baseline.corpus_words), with MuLMS's hyphen rule applied
    (build_topic_corpus.hyphen_split); names of more than MAX_WORDS words count as absent;
  - loose: the same, ignoring word boundaries and case;
  - per type: the unseen entities and the exact and loose shares.

Output (results/mulms_coverage.json): corpora {name: {path, sentences, sha256}}, subfields
{name: {n_test, n_unseen, unseen_share, corpora: {corpus: {leaked, exact, exact_distinct,
loose, loose_distinct, per_type: {type: {n_unseen, exact, loose}}}}}}.

Usage:
  python src/graph/mulms_coverage.py [--results results/gpu] [--corpus NAME=PATH[:raw] ...]
      [--out results/mulms_coverage.json]
The default corpora: every data/processed/corpus/topic_<topic>_sentences.txt that exists
(build_topic_corpus.py), sofc_sentences.txt (build_sofc_corpus.py) and fatigue_sentences.txt
(build_fatigue_corpus.py, raw).
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import dapt_mlm
import ner_mentions as nm
from build_topic_corpus import SHINGLE, TOPICS, hyphen_split, shingles
from ner_baseline import corpus_words, entity_spans
from config import PROCESSED_DIR, PROJECT_ROOT

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'gpu')
OUT = os.path.join(PROJECT_ROOT, 'results', 'mulms_coverage.json')
CORPUS_DIR = os.path.join(PROCESSED_DIR, 'corpus')
MAX_WORDS = 8
FOLD_FILE = 'ner_results_mulms_leave_one_subfield_out_crf_m0.json'


def default_corpora():
    """{name: (path, raw)} of the corpora on disk."""
    out = {}
    for topic in sorted(TOPICS):
        path = os.path.join(CORPUS_DIR, f'topic_{topic}_sentences.txt')
        if os.path.exists(path):
            out[f'topic_{topic}'] = (path, False)
    for name, raw in (('sofc', False), ('fatigue', True)):
        path = os.path.join(CORPUS_DIR, f'{name}_sentences.txt')
        if os.path.exists(path):
            out[name] = (path, raw)
    return out


def parse_corpus(spec):
    """NAME=PATH[:raw] -> (name, (path, raw))."""
    name, _, rest = spec.partition('=')
    path, _, flag = rest.rpartition(':') if rest.endswith(':raw') else (rest, '', '')
    if not name or not path or flag not in ('', 'raw'):
        raise argparse.ArgumentTypeError(f'--corpus takes NAME=PATH or NAME=PATH:raw, not {spec!r}')
    return name, (path, flag == 'raw')


def read_corpus(path, raw):
    """The corpus as word lists, with MuLMS's hyphen rule."""
    with open(path, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return [hyphen_split(corpus_words(line) if raw else line.split()) for line in lines]


def mulms_folds(results_dir):
    """(data, {subfield: (training positions, test positions)}) from week 10's CRF file."""
    data = nm.corpus('mulms')
    with open(os.path.join(results_dir, FOLD_FILE), encoding='utf-8') as f:
        results = json.load(f)
    folds = {}
    for run in results['runs']:
        _, train, test = nm.run_split(results, run, data, 'mulms')
        folds[run['run_id']] = (train, test)
    return data, folds


def unseen_mentions(data, train, test):
    """(all test mentions, the unseen ones): (type, words) in order."""
    known = {tuple(data[i][0][s:e]) for i in train for s, e, _ in entity_spans(data[i][1])}
    mentions = [(k, tuple(data[i][0][s:e])) for i in test for s, e, k in entity_spans(data[i][1])]
    return mentions, [m for m in mentions if m[1] not in known]


def occurrences(sentences, names):
    """{name words: set of sentence indices} of the names (word tuples of at most MAX_WORDS)
    that occur in a row in a sentence."""
    wanted = {w for w in names if len(w) <= MAX_WORDS}
    found = defaultdict(set)
    lengths = sorted({len(w) for w in wanted})
    for j, words in enumerate(sentences):
        for n in lengths:
            for i in range(len(words) - n + 1):
                gram = tuple(words[i:i + n])
                if gram in wanted:
                    found[gram].add(j)
    return found


def coverage(data, folds, corpora):
    """The subfields section: per subfield and corpus, the counts of the module docstring."""
    out = {}
    unseen = {name: unseen_mentions(data, train, test) for name, (train, test) in folds.items()}
    all_names = {w for _, (_, ms) in unseen.items() for _, w in ms}
    for cname, sentences in corpora.items():
        found = occurrences(sentences, all_names)
        flat = [''.join(words).lower() for words in sentences]
        for sname, (train, test) in folds.items():
            mentions, ms = unseen[sname]
            entry = out.setdefault(sname, {
                'n_test': len(mentions), 'n_unseen': len(ms),
                'unseen_share': round(len(ms) / len(mentions), 3) if mentions else 0.0,
                'corpora': {}})
            test_shingles = set()
            for i in test:
                test_shingles |= shingles(' '.join(data[i][0]))
            leaked = {j for j, text in enumerate(flat)
                      if not test_shingles.isdisjoint(shingles(text))}
            blob = '\n'.join(text for j, text in enumerate(flat) if j not in leaked)
            weights, distinct = Counter(ms), sorted(set(ms))
            exact = {m for m in distinct if found.get(m[1], set()) - leaked}
            loose = {m for m in distinct if ''.join(m[1]).lower() in blob}

            def share(hit, kind=None):
                pool = [m for m in distinct if kind is None or m[0] == kind]
                total = sum(weights[m] for m in pool)
                return round(sum(weights[m] for m in pool if m in hit) / total, 3) if total else 0.0
            per_type = {k: {'n_unseen': sum(weights[m] for m in distinct if m[0] == k),
                            'exact': share(exact, k), 'loose': share(loose, k)}
                        for k in sorted({m[0] for m in distinct})}
            entry['corpora'][cname] = {
                'leaked': len(leaked), 'exact': share(exact),
                'exact_distinct': round(len(exact) / len(distinct), 3) if distinct else 0.0,
                'loose': share(loose),
                'loose_distinct': round(len(loose) / len(distinct), 3) if distinct else 0.0,
                'per_type': per_type}
    return out


def analyse(results_dir, corpus_specs):
    """The whole report for {name: (path, raw)}."""
    corpora = {name: read_corpus(path, raw) for name, (path, raw) in corpus_specs.items()}
    data, folds = mulms_folds(results_dir)
    return {'rule': {'unseen': 'a test entity whose words are no entity of the training '
                               'sentences (ner_mentions.py)',
                     'leaked': f'a corpus sentence sharing a run of {SHINGLE} characters '
                               'without spaces, in lower case, with a sentence of the '
                               "held-out subfield's documents; left out",
                     'exact': "the name's words in a row in a remaining sentence, the corpus "
                              "cut with MuLMS's hyphen rule; names of more than "
                              f'{MAX_WORDS} words count as absent',
                     'loose': 'the name without spaces, in lower case, inside a remaining '
                              'sentence without spaces'},
            'corpora': {name: {'path': dapt_mlm.display_path(path), 'raw_sentences': raw,
                               'sentences': len(corpora[name]),
                               'sha256': dapt_mlm.file_sha256(path)}
                        for name, (path, raw) in corpus_specs.items()},
            'subfields': coverage(data, folds, corpora)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unseen-name coverage of MuLMS's held-out "
                                                 'subfields by unlabeled corpora')
    parser.add_argument('--results', default=RESULTS_DIR, metavar='DIR')
    parser.add_argument('--corpus', type=parse_corpus, action='append', metavar='NAME=PATH[:raw]',
                        help='a corpus (default: the topic, SOFC and fatigue corpora on disk)')
    parser.add_argument('--out', default=OUT, metavar='PATH')
    args = parser.parse_args(argv)
    specs = dict(args.corpus) if args.corpus else default_corpora()
    if not specs:
        raise SystemExit('no corpus found; build one or pass --corpus')
    report = analyse(args.results, specs)
    names = list(specs)
    print('subfield       unseen ' + ' '.join(f'{n[:14]:>14s}' for n in names))
    for sname, entry in report['subfields'].items():
        cells = [f"{100 * entry['corpora'][n]['exact']:5.1f}/{100 * entry['corpora'][n]['loose']:4.1f}"
                 for n in names]
        print(f"{sname:13s} {100 * entry['unseen_share']:5.1f}% " + ' '.join(f'{c:>14s}' for c in cells))
    print('(exact/loose, % of the unseen test entities; leaked sentences left out)')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    print(f'Saved {args.out}')
    return report


if __name__ == '__main__':
    main()
