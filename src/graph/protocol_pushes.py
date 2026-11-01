"""When each protocol commit reached GitHub: an independent timestamp for paper A's protocol
record (Supplementary Table S1), beside the commit times the author's machine sets.

GitHub's public event feed of the repository (the REST events API: the latest 300 events, at most
90 days old) lists each push with the branch's commit before it and after it (its head). A commit
first reached GitHub at the earliest push whose range before..head holds it. The feed can miss a
push; then the earliest push whose `before` already descends from the commit bounds the time from
above ("on GitHub by"). The archive keeps every push event ever fetched (results/
github_push_events.tsv: time, before, head, ref), since the feed drops the oldest.

  python src/graph/protocol_pushes.py --fetch                    # add the feed to the archive
  python src/graph/protocol_pushes.py a08356f 40631ce 2886267    # when these commits were pushed
"""
import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'results' / 'github_push_events.tsv'
REPO = 'hatemadi15/AI_Research_Ontologies_GNN_GRL'
HEADER = 'created_at\tbefore\thead\tref'


def fetch_events(repo=REPO, pages=3):
    """[(created_at, before, head, ref)] of the PushEvents in the repository's public feed."""
    rows = []
    for page in range(1, pages + 1):
        url = f'https://api.github.com/repos/{repo}/events?per_page=100&page={page}'
        request = urllib.request.Request(url, headers={'Accept': 'application/vnd.github+json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            events = json.load(response)
        for e in events:
            if e.get('type') == 'PushEvent':
                p = e.get('payload', {})
                rows.append((e['created_at'], p.get('before') or '', p.get('head') or '',
                             p.get('ref') or ''))
    return rows


def read_archive(path=ARCHIVE):
    """The archived push events, [] without the file."""
    if not Path(path).exists():
        return []
    lines = Path(path).read_text().splitlines()
    return [tuple(line.split('\t')) for line in lines[1:] if line]


def write_archive(rows, path=ARCHIVE):
    """The union of the archive and `rows`, sorted by time, written back; returns it."""
    merged = sorted(set(read_archive(path)) | {tuple(r) for r in rows})
    Path(path).write_text('\n'.join([HEADER] + ['\t'.join(r) for r in merged]) + '\n')
    return merged


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)


def rev_list(before, head):
    """The commits a push carried: before..head (head's whole history for a new branch);
    None if git does not know them."""
    new = not before or set(before) == {'0'}
    out = git('rev-list', head if new else f'{before}..{head}')
    return out.stdout.split() if out.returncode == 0 else None


def is_ancestor(commit, other):
    """Whether `commit` is `other` or one of its ancestors."""
    return bool(other) and set(other) != {'0'} and \
        git('merge-base', '--is-ancestor', commit, other).returncode == 0


def push_times(rows, commits, rev_list=rev_list, is_ancestor=is_ancestor, resolve=None):
    """{commit: {'pushed': the time of the earliest push that carried it, or None (also when
    that push is later than the bound: the commit's own push is missing from the feed),
    'on_github_by': the time of the earliest push whose `before` already held it, or None}}."""
    resolve = resolve or (lambda c: git('rev-parse', c).stdout.strip() or c)
    carried = {}
    for created, before, head, _ in sorted(rows):
        for c in rev_list(before, head) or []:
            carried.setdefault(c, created)
    out = {}
    for commit in commits:
        full = resolve(commit)
        bound = next((created for created, before, _, _ in sorted(rows)
                      if is_ancestor(full, before)), None)
        pushed = carried.get(full)
        if pushed and bound and pushed > bound:
            pushed = None          # a later push (say, a merge into main) carried it again
        out[commit] = {'pushed': pushed, 'on_github_by': bound}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('commits', nargs='*')
    parser.add_argument('--fetch', action='store_true', help='add the public feed to the archive')
    parser.add_argument('--archive', default=str(ARCHIVE))
    args = parser.parse_args(argv)
    if args.fetch:
        rows = write_archive(fetch_events(), args.archive)
        print(f'{len(rows)} push events archived in {args.archive} ({rows[0][0]} to {rows[-1][0]})')
    rows = read_archive(args.archive)
    for commit, t in push_times(rows, args.commits).items():
        print(f"{commit}  pushed {t['pushed'] or '-'}  on GitHub by {t['on_github_by'] or '-'}")


if __name__ == '__main__':
    main()
