"""The push times of protocol commits from GitHub's event feed (protocol_pushes.py): the earliest
push that carried a commit, the bound from a later push's `before`, and the archive's union."""
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
