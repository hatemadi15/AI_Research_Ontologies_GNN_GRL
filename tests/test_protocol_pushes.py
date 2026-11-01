"""The push times of protocol commits from GitHub's event feed and activity log (protocol_pushes.py): the earliest
push that carried a commit, the bound from a later push's `before`, and the archive's union."""
import json

import protocol_pushes as pp


def test_push_times_from_ranges_and_bounds():
    # a - b - c - d on one branch; the feed has the push of a..c and of c..d, not the one of b
    history = ['a', 'b', 'c', 'd']
    rows = [('2026-10-20T10:00:00Z', 'b', 'c', 'refs/heads/x'),
            ('2026-10-20T09:00:00Z', '0000', 'a', 'refs/heads/x'),
            ('2026-10-20T11:00:00Z', 'c', 'd', 'refs/heads/x')]

    def rev_list(before, head):
        stop = history.index(before) + 1 if before in history else 0
        return history[stop:history.index(head) + 1][::-1]

    def is_ancestor(commit, other):
        return other in history and history.index(commit) <= history.index(other)

    out = pp.push_times(rows, ['a', 'b', 'c', 'd'], rev_list=rev_list, is_ancestor=is_ancestor,
                        resolve=lambda c: c)
    assert out['a'] == {'pushed': '2026-10-20T09:00:00Z', 'on_github_by': '2026-10-20T10:00:00Z'}
    assert out['b'] == {'pushed': None, 'on_github_by': '2026-10-20T10:00:00Z'}   # its push missing
    assert out['c'] == {'pushed': '2026-10-20T10:00:00Z', 'on_github_by': '2026-10-20T11:00:00Z'}
    assert out['d'] == {'pushed': '2026-10-20T11:00:00Z', 'on_github_by': None}
    # a merge into another branch later carries b again: not b's first arrival
    merged = rows + [('2026-10-30T11:00:00Z', 'z', 'd', 'refs/heads/main')]

    def rev_list_main(before, head):
        return history[::-1] if before == 'z' else rev_list(before, head)
    again = pp.push_times(merged, ['b'], rev_list=rev_list_main, is_ancestor=is_ancestor,
                          resolve=lambda c: c)
    assert again['b'] == {'pushed': None, 'on_github_by': '2026-10-20T10:00:00Z'}


def test_archive_keeps_the_union(tmp_path):
    path = tmp_path / 'pushes.tsv'
    pp.write_archive([('2026-10-20T10:00:00Z', 'b', 'c', 'r')], path)
    rows = pp.write_archive([('2026-10-20T09:00:00Z', '0', 'a', 'r'),
                             ('2026-10-20T10:00:00Z', 'b', 'c', 'r')], path)
    assert rows == [('2026-10-20T09:00:00Z', '0', 'a', 'r'), ('2026-10-20T10:00:00Z', 'b', 'c', 'r')]
    assert pp.read_archive(path) == rows and path.read_text().startswith(pp.HEADER)


class _Response:
    def __init__(self, items, link=None):
        self.items, self.headers = items, {'Link': link} if link else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.items).encode()


def test_fetch_activity_follows_repository_scoped_pages():
    """The activity log's pushes, page after page, with GitHub's repositories/{id} next links
    rewritten to the repos/{owner}/{repo} path."""
    first = [{'timestamp': '2026-11-01T10:51:13Z', 'before': 'a' * 40, 'after': 'b' * 40,
              'ref': 'refs/heads/x', 'activity_type': 'push'},
             {'timestamp': '2026-11-01T10:50:00Z', 'activity_type': 'branch_deletion'}]
    second = [{'timestamp': '2026-10-20T00:51:49Z', 'before': 'c' * 40, 'after': 'a' * 40,
               'ref': 'refs/heads/x', 'activity_type': 'force_push'}]
    link = ('<https://api.github.com/repositories/42/activity?per_page=100&after=CUR>; '
            'rel="next", <https://api.github.com/repositories/42/activity?per_page=100>; rel="first"')
    urls = []

    def opener(request, timeout):
        urls.append(request.full_url)
        return _Response(first, link) if len(urls) == 1 else _Response(second)
    rows = pp.fetch_activity('o/r', opener=opener)
    assert urls == ['https://api.github.com/repos/o/r/activity?per_page=100',
                    'https://api.github.com/repos/o/r/activity?per_page=100&after=CUR']
    assert rows == [('2026-11-01T10:51:13Z', 'a' * 40, 'b' * 40, 'refs/heads/x'),
                    ('2026-10-20T00:51:49Z', 'c' * 40, 'a' * 40, 'refs/heads/x')]
    assert pp.next_link(None) is None and pp.next_link('<u?x=1>; rel="last"') is None
