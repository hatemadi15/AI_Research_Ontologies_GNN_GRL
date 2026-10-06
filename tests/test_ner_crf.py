"""CRF head of ner_baseline.py against brute force (needs torch; skipped in CI)."""
import itertools

import pytest

import ner_baseline as nb

torch = pytest.importorskip('torch')

LABELS = ['O', 'B-A', 'I-A', 'B-B', 'I-B']


def _crf(seed=0):
    torch.manual_seed(seed)
    crf = nb.make_crf(LABELS)
    with torch.no_grad():
        for p in crf.parameters():
            p.normal_()
    return crf


def _path_score(crf, emissions, path):
    trans, start, end = crf.scores()
    s = start[path[0]] + emissions[0, path[0]]
    for t in range(1, len(path)):
        s = s + trans[path[t - 1], path[t]] + emissions[t, path[t]]
    return s + end[path[-1]]


def _valid(path):
    allowed, start = nb.bio_allowed(LABELS)
    return start[path[0]] and all(allowed[a][b] for a, b in zip(path, path[1:]))


def test_nll_and_viterbi_match_brute_force():
    crf = _crf()
    emissions = torch.randn(4, len(LABELS))
    paths = list(itertools.product(range(len(LABELS)), repeat=4))
    scores = torch.stack([_path_score(crf, emissions, p) for p in paths])
    gold = (1, 2, 0, 3)
    expected_nll = torch.logsumexp(scores, 0) - _path_score(crf, emissions, gold)
    mask = torch.ones(1, 4, dtype=torch.bool)
    nll = crf.nll(emissions.unsqueeze(0), torch.tensor([gold]), mask)
    assert nll.item() == pytest.approx(expected_nll.item(), abs=1e-4)
    best = list(paths[int(scores.argmax())])
    assert crf.decode(emissions.unsqueeze(0), mask) == [best]
    assert _valid(best)


def test_padding_is_ignored():
    crf = _crf(1)
    long_em, short_em = torch.randn(5, len(LABELS)), torch.randn(3, len(LABELS))
    long_tags, short_tags = torch.tensor([0, 1, 2, 0, 3]), torch.tensor([3, 4, 0])
    em = torch.stack([long_em, torch.cat([short_em, 50 * torch.randn(2, len(LABELS))])])
    tags = torch.stack([long_tags, torch.cat([short_tags, torch.tensor([2, 2])])])
    mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    one = torch.ones(1, 5, dtype=torch.bool)
    expected = (crf.nll(long_em[None], long_tags[None], one)
                + crf.nll(short_em[None], short_tags[None], one[:, :3])) / 2
    assert crf.nll(em, tags, mask).item() == pytest.approx(expected.item(), abs=1e-4)
    assert crf.decode(em, mask) == (crf.decode(long_em[None], one)
                                    + crf.decode(short_em[None], one[:, :3]))


def test_decoding_respects_iob2_even_against_emissions():
    crf = nb.make_crf(LABELS)       # zero transitions: only the constraints act
    emissions = torch.full((2, 6, len(LABELS)), -5.0)
    emissions[:, :, LABELS.index('I-A')] = 5.0        # I-A everywhere is invalid at t=0
    emissions[1, 3, LABELS.index('I-B')] = 20.0       # and I-B cannot follow I-A
    paths = crf.decode(emissions, torch.ones(2, 6, dtype=torch.bool))
    assert all(_valid(p) for p in paths)
    assert paths[0][0] == LABELS.index('B-A')


def test_word_view_gathers_first_subwords():
    logits = torch.arange(2 * 6 * 3, dtype=torch.float).view(2, 6, 3)
    labels = torch.tensor([[-100, 1, -100, 0, 2, -100],
                           [-100, 2, 1, -100, -100, -100]])
    emissions, tags, mask = nb.word_view(logits, labels)
    assert emissions.shape == (2, 3, 3)
    assert torch.equal(emissions[0], logits[0, [1, 3, 4]])
    assert torch.equal(emissions[1, :2], logits[1, [1, 2]])
    assert tags.tolist() == [[1, 0, 2], [2, 1, 0]]
    assert mask.tolist() == [[True, True, True], [True, True, False]]
