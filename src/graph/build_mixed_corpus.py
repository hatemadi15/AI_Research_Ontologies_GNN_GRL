"""Mixed corpora for week 14's decision 14.2 (ROADMAP §4, week 14: the topic's role).

For a MuLMS subfield, the mixed corpus holds SENTENCES sentences drawn from the topic corpora
of the six other subfields (build_topic_corpus.py), the same number from each (the first
sources get one more when SENTENCES does not divide): the topic corpus's size, source and
pipeline without its topic, which is what a large unlabeled materials corpus of other topics
gives a tagger. A sentence that is also in the held-out subfield's own topic corpus is
skipped, and so is one already taken from another source (the corpora share at most a few
hundred sentences pairwise). Every sentence comes from a topic corpus, so phase 0's
exclusions (MuLMS's articles; every sentence sharing an 8-word n-gram or a 50-character run
with a MuLMS sentence) hold. One random.Random(SEED) shuffles each source's sentence numbers
in TOPICS order; the chosen sentences of a source are written in corpus order, source after
source. The corpus goes to data/processed/corpus/topic_mixed_<subfield>_sentences.txt (never
committed) and its report, with the sources' SHA-256, the counts and the corpus's SHA-256,
to results/topic_corpora/mixed_<subfield>.json, which kaggle_jobs.py's bundle reads as it
reads the topic corpora's reports.

Usage:
  python src/graph/build_mixed_corpus.py                 # every subfield
  python src/graph/build_mixed_corpus.py --subfields sofc poly
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = PROJECT_ROOT / 'data' / 'processed' / 'corpus'
REPORT_DIR = PROJECT_ROOT / 'results' / 'topic_corpora'
TOPICS = ('electrolysis', 'graphene', 'pemfc', 'poly', 'semi', 'sofc', 'steel')
SENTENCES = 40000
SEED = 0
PROTOCOL = "docs/ROADMAP.md §4, week 14: 14.2, the topic's role"
RULE = ('from each of the other topic corpora, in TOPICS order, the first sentences of a '
        'seeded shuffle that are neither in the held-out subfield\'s own corpus nor already '
        'taken, as many as its quota; written in corpus order, source after source')


def topic_path(topic, corpus_dir=CORPUS_DIR):
    """The corpus file of a topic (or of a mixed corpus, topic 'mixed_<subfield>')."""
    return Path(corpus_dir) / f'topic_{topic}_sentences.txt'


def report_path(topic, report_dir=REPORT_DIR):
    return Path(report_dir) / f'{topic}.json'


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_lines(path):
    """The sentences of a corpus file, one per line, without a trailing empty line."""
    lines = Path(path).read_text(encoding='utf-8').split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    return lines


def quotas(size, sources):
    """How many sentences each of `sources` sources gives, the first ones one more when
    `size` does not divide: quotas(40000, 6) == [6667, 6667, 6667, 6667, 6666, 6666]."""
    base, extra = divmod(size, sources)
    return [base + (1 if i < extra else 0) for i in range(sources)]


def source_corpus(topic, corpus_dir=CORPUS_DIR, report_dir=REPORT_DIR):
    """The sentences and SHA-256 of a topic corpus, which must be the one its report records
    (SystemExit otherwise)."""
    path = topic_path(topic, corpus_dir)
    report = report_path(topic, report_dir)
    if not path.is_file() or not report.is_file():
        raise SystemExit(f'{path} or {report} is missing (python src/graph/build_topic_corpus.py '
                         f'--topic {topic} rebuilds the corpus, with --offline from its cache)')
    recorded = json.loads(report.read_text(encoding='utf-8'))['corpus']['sha256']
    digest = file_sha256(path)
    if digest != recorded:
        raise SystemExit(f'{path}: SHA-256 {digest}, not the recorded {recorded}')
    return read_lines(path), digest


def mixed_sentences(own, sources, size=SENTENCES, seed=SEED):
    """The mixed corpus of a subfield: (sentences, {source: counts}). own: the subfield's
    own corpus sentences; sources: {topic: sentences} of the other topics, in order."""
    own = set(own)
    taken, out, counts = set(), [], {}
    rng = random.Random(seed)
    for (topic, lines), quota in zip(sources.items(), quotas(size, len(sources))):
        order = list(range(len(lines)))
        rng.shuffle(order)
        chosen, skipped_own, skipped_duplicate, visited = [], 0, 0, 0
        for i in order:
            visited += 1
            sentence = lines[i]
            if sentence in own:
                skipped_own += 1
            elif sentence in taken:
                skipped_duplicate += 1
            else:
                chosen.append(i)
                taken.add(sentence)
                if len(chosen) == quota:
                    break
        if len(chosen) < quota:
            raise SystemExit(f'{topic}: only {len(chosen)} of the {quota} sentences wanted '
                             f'after the exclusions ({len(lines)} in the corpus)')
        chosen.sort()
        out.extend(lines[i] for i in chosen)
        counts[topic] = {'sentences': len(lines), 'quota': quota, 'visited': visited,
                         'taken': len(chosen), 'skipped_own': skipped_own,
                         'skipped_duplicate': skipped_duplicate}
    return out, counts


def build(subfield, corpus_dir=CORPUS_DIR, report_dir=REPORT_DIR, topics=TOPICS,
          size=SENTENCES, seed=SEED):
    """Write the mixed corpus of `subfield` and its report; returns the report."""
    started = time.time()
    own, own_digest = source_corpus(subfield, corpus_dir, report_dir)
    sources, digests = {}, {}
    for topic in topics:
        if topic != subfield:
            sources[topic], digests[topic] = source_corpus(topic, corpus_dir, report_dir)
    sentences, counts = mixed_sentences(own, sources, size, seed)
    name = f'mixed_{subfield}'
    out = topic_path(name, corpus_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(s + '\n' for s in sentences), encoding='utf-8')
    report = {
        'script': 'src/graph/build_mixed_corpus.py', 'protocol': PROTOCOL, 'topic': name,
        'held_out': subfield,
        'settings': {'seed': seed, 'sentences': size, 'rule': RULE},
        'own_corpus': {'path': str(topic_path(subfield, corpus_dir).relative_to(PROJECT_ROOT))
                       if topic_path(subfield, corpus_dir).is_relative_to(PROJECT_ROOT)
                       else str(topic_path(subfield, corpus_dir)),
                       'sentences': len(own), 'sha256': own_digest},
        'sources': {t: {'sha256': digests[t], **counts[t]} for t in sources},
        'corpus': {'path': str(out.relative_to(PROJECT_ROOT)) if out.is_relative_to(PROJECT_ROOT)
                   else str(out),
                   'sentences': len(sentences), 'words': sum(len(s.split()) for s in sentences),
                   'sha256': file_sha256(out)},
        'run': {'finished': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'seconds': round(time.time() - started, 1)},
    }
    path = report_path(name, report_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--subfields', nargs='+', default=list(TOPICS), choices=TOPICS,
                        metavar='SUBFIELD')
    parser.add_argument('--corpus-dir', default=CORPUS_DIR, type=Path, metavar='DIR')
    parser.add_argument('--report-dir', default=REPORT_DIR, type=Path, metavar='DIR')
    parser.add_argument('--sentences', type=int, default=SENTENCES, metavar='N')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args(argv)
    for subfield in args.subfields:
        report = build(subfield, args.corpus_dir, args.report_dir, size=args.sentences,
                       seed=args.seed)
        skipped = sum(c['skipped_own'] + c['skipped_duplicate'] for c in report['sources'].values())
        print(f"{subfield:13s} {report['corpus']['sentences']} sentences, "
              f"{report['corpus']['words']} words, {skipped} skipped; "
              f"{report['corpus']['sha256'][:12]}  {report['corpus']['path']}")


if __name__ == '__main__':
    main()
