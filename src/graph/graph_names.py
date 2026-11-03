"""graph_names.py - week 19's probe: graph-propagated names at matched lexicon size

ROADMAP §4, the week-19 protocol (fixed 3 November, 00:32 UTC); docs/WEEK19_PLAN.md. No
decision is taken here; the probe's gate decides whether the GPU decisions G1 and G2 run.

Inputs, per held-out unit (MuLMS's 7 subfields; MaterioMiner's 4 papers, fine-grained): the
candidate dump of batch w19a (ner_baseline.py --students 0 --save-names --save-candidates,
collected to data/processed/week19/, never versioned: it holds corpus text), and the unit's
training and test sentences, rebuilt from a stored crf_mr6 results file (ner_mentions.run_split
checks the test positions). Per unit:
  1. the vote lexicons V of the grid (THRESHOLDS x AGREEMENTS), rebuilt exactly from the dump
     with silver_mention_names' semantics and order (vote_names);
  2. the graph (build_graph): nodes, the distinct surface forms of the dump's spans; labelled
     nodes, the training names with their majority gold type; features, the vote block (per
     type, the share of the node's span-teacher pairs tagged with it and the mean span score),
     log frequency and the character 3-5-gram TF-IDF vector reduced to 128 dimensions;
     relations, positive-PMI sentence co-occurrence (the 20 strongest per node), the 10
     nearest neighbours by character cosine (both selections symmetrised) and a shared head
     word (heads of at most 200 nodes, a clique aggregated by its group mean);
  3. the rankers M (a 2-layer MLP on the features), G (a 2-layer R-GCN over the relations, mean
     aggregation per relation), both without the vote block too, and LP (label propagation,
     reported only); during training the vote block of each node is dropped with probability
     0.5; their 5-fold accuracy on labelled nodes is reported as "teacher-seen";
  4. lexicons of N new names (no training name), N primary from V at 0.7, 3 of 3, N secondary
     from V at 0.9, 3 of 3; a name's type, the ranker's most probable among those some teacher
     tags it with (any marginal), ties by type name;
  5. coverage and wrong-type rate on the unit's unseen test mentions, and the ceiling.
Then the gate over the 11 units (GATE). Outputs: results/week19/graph_names_probe.json (counts
and rates only) and, if asked, the lexicon files for --names-file (names_files; local only).

Usage:
  python src/graph/graph_names.py                  # the probe over the units with a dump
  python src/graph/graph_names.py --names-files    # also write the Phase-1 lexicon files
"""
import argparse
import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DUMP_DIR = ROOT / 'data' / 'processed' / 'week19'      # dumps and lexicon files, git-ignored
OUTPUT = ROOT / 'results' / 'week19' / 'graph_names_probe.json'
GPU_RESULTS = ROOT / 'results' / 'gpu'

THRESHOLDS = (0.5, 0.7, 0.9)
AGREEMENTS = (1, 2, 3)
PRIMARY, SECONDARY = (0.7, 3), (0.9, 3)
PPMI_TOP, KNN, HEAD_MAX, SVD_DIM = 20, 10, 200, 128
HIDDEN, DROPOUT, VOTE_DROPOUT, LR, WEIGHT_DECAY, EPOCHS, SEED, FOLDS = (
    128, 0.3, 0.5, 0.01, 5e-4, 100, 0, 5)
LP_ITERATIONS = 30
GATE = {'coverage_vs_vote': 5.0, 'coverage_vs_mlp': 2.0, 'wrong_type_vs_vote': 5.0}

NAMES_DIR = ROOT / 'results' / 'gpu_names'          # the --save-names files (git-ignored)
# (corpus, dump file, the stored results file that rebuilds the units' splits, week 15's
# names files, w19a's names files)
SETTINGS = {
    'mulms': ('mulms', 'ner_candidates_mulms_{unit}.json.gz',
              'ner_results_mulms_leave_one_subfield_out_crf_mr6_m300.json',
              'ner_names_mulms_leave_one_subfield_out_crf_mrm_topic6_{unit}.json',
              'ner_names_mulms_leave_one_subfield_out_crf_mrm_topic6_{unit}_w19.json'),
    'fine': ('fine', 'ner_candidates_fine_leave_one_paper_out.json.gz',
             'ner_results_fine_leave_one_paper_out_crf_mr6_m0.json',
             'ner_names_fine_leave_one_paper_out_crf_mrm6.json',
             'ner_names_fine_leave_one_paper_out_crf_mrm6_w19.json'),
}


# 1. the vote ------------------------------------------------------------------------------

def vote_names(spans, threshold, agree):
    """(type, words) of the silver mentions at a marginal threshold and an agreement, in
    silver_mention_names' order (sentence index, then sorted (start, end, type)), from the
    dump's spans: a span's type counts when that teacher's Viterbi span there has the type
    and its least marginal is at least the threshold; a (start, end, type) with at least
    `agree` such teachers is a name (agree = the number of teachers: every teacher agrees,
    silver_mention_names' default)."""
    names = []
    for _, _, _, words, teachers in spans:
        counts = Counter(kind for kind, least, _ in teachers
                         if kind is not None and least >= threshold)
        for kind in sorted(counts):
            if counts[kind] >= agree:
                names.append((kind, tuple(words)))
    return names


def one_type(names):
    """{words: type}: each name's most frequent type among its entries, ties by type name."""
    counts = defaultdict(Counter)
    for kind, words in names:
        counts[words][kind] += 1
    return {w: min(c, key=lambda k: (-c[k], k)) for w, c in counts.items()}


# 2. the graph -----------------------------------------------------------------------------

def training_types(sentences):
    """{words: majority gold type} of the gold entities of (words, tags) sentences, ties by
    type name, and the set of (type, words) mentions."""
    from ner_baseline import entity_spans
    counts, mentions = defaultdict(Counter), set()
    for words, tags in sentences:
        for start, end, kind in entity_spans(tags):
            counts[tuple(words[start:end])][kind] += 1
            mentions.add((kind, tuple(words[start:end])))
    return {w: min(c, key=lambda k: (-c[k], k)) for w, c in counts.items()}, mentions


def char_vectors(forms, dim=SVD_DIM, seed=SEED):
    """L2-normalised truncated-SVD reduction of the character 3-5-gram TF-IDF vectors of the
    forms (space-joined words), at most `dim` dimensions (fewer for a tiny graph)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), lowercase=False)
    matrix = tfidf.fit_transform([' '.join(f) for f in forms])
    k = max(1, min(dim, matrix.shape[1] - 1, len(forms) - 1))
    reduced = TruncatedSVD(n_components=k, random_state=seed).fit_transform(matrix)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    return reduced / np.where(norms > 0, norms, 1.0)


def knn_edges(vectors, k=KNN, chunk=1024):
    """Directed (i, j) pairs: each node's k nearest other nodes by cosine (rows of `vectors`
    are unit length), ties by index: a partial sort keeps the k + 8 best of each row, which
    are then ordered by (similarity descending, index)."""
    n = len(vectors)
    k = min(k, n - 1)
    if k <= 0:
        return []
    keep = min(k + 8, n - 1)
    pairs = []
    for start in range(0, n, chunk):
        rows = range(start, min(start + chunk, n))
        sims = vectors[start:start + chunk] @ vectors.T
        sims[np.arange(len(rows)), np.array(rows)] = -np.inf
        best = np.argpartition(-sims, keep - 1, axis=1)[:, :keep]
        for row, i in enumerate(rows):
            cand = sorted(best[row].tolist(), key=lambda j: (-sims[row, j], j))[:k]
            pairs += [(i, int(j)) for j in cand]
    return pairs


def ppmi_edges(sentence_nodes, n_nodes, top=PPMI_TOP):
    """Directed (i, j) pairs: each node's `top` co-occurring nodes by positive PMI over the
    sentences (node sets), ties by index."""
    occurrences, pair_counts = Counter(), Counter()
    for nodes in sentence_nodes:
        nodes = sorted(nodes)
        occurrences.update(nodes)
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                pair_counts[nodes[a], nodes[b]] += 1
    total = len(sentence_nodes)
    best = defaultdict(list)
    for (a, b), c in pair_counts.items():
        pmi = math.log(c * total / (occurrences[a] * occurrences[b]))
        if pmi > 0:
            best[a].append((-pmi, b))
            best[b].append((-pmi, a))
    pairs = []
    for i in range(n_nodes):
        pairs += [(i, j) for _, j in sorted(best[i])[:top]]
    return pairs


def symmetric(pairs):
    """The undirected edges of directed selections (an edge kept if either end selects it),
    as sorted (i, j) with i < j."""
    return sorted({(min(i, j), max(i, j)) for i, j in pairs if i != j})


def head_groups(forms, limit=HEAD_MAX):
    """Groups (lists of node indices, at least 2) of the forms sharing a head word (the last
    word), heads of at most `limit` nodes."""
    groups = defaultdict(list)
    for i, form in enumerate(forms):
        groups[form[-1]].append(i)
    return [g for _, g in sorted(groups.items()) if 2 <= len(g) <= limit]


def build_graph(spans, labels, train_sentences):
    """The unit's graph: {'forms', 'features', 'vote_cols', 'types', 'labelled' {node: type
    index}, 'masks' (per node, the types some teacher tags it with), 'new' (nodes that are no
    training name), 'edges' {'cooccurrence', 'knn'} (undirected), 'heads' (groups)}."""
    types = sorted({lab[2:] if lab[:2] in ('B-', 'I-') else lab for lab in labels
                    if lab != 'O'})          # the dump's BIO labels, or bare types
    t_index = {t: k for k, t in enumerate(types)}
    index, forms = {}, []
    n_pairs, tagged, scores, masks = Counter(), defaultdict(Counter), defaultdict(Counter), {}
    sentence_nodes = defaultdict(set)
    for sentence, _, _, words, teachers in spans:
        form = tuple(words)
        if form not in index:
            index[form] = len(forms)
            forms.append(form)
        i = index[form]
        sentence_nodes[sentence].add(i)
        for kind, least, others in teachers:
            n_pairs[i] += 1
            if kind is not None:
                tagged[i][kind] += 1
                scores[i][kind] += least
                masks.setdefault(i, set()).add(kind)
            for other, score in others:
                scores[i][other] += score
    n, n_types = len(forms), len(types)
    vote = np.zeros((n, 2 * n_types), dtype=np.float32)
    for i in range(n):
        for kind, c in tagged[i].items():
            vote[i, t_index[kind]] = c / n_pairs[i]
        for kind, s in scores[i].items():
            if kind in t_index:
                vote[i, n_types + t_index[kind]] = s / n_pairs[i]
    span_counts = Counter(index[tuple(words)] for _, _, _, words, _ in spans)
    freq = np.log1p(np.array([span_counts[i] for i in range(n)], dtype=np.float32))[:, None]
    chars = char_vectors(forms).astype(np.float32)
    features = np.concatenate([vote, freq, chars], axis=1)
    train_types, _ = training_types(train_sentences)
    labelled = {index[w]: t_index[t] for w, t in train_types.items()
                if w in index and t in t_index}
    new = [i for i, f in enumerate(forms) if f not in train_types]
    edges = {'cooccurrence': symmetric(ppmi_edges(list(sentence_nodes.values()), n)),
             'knn': symmetric(knn_edges(chars))}
    return {'forms': forms, 'features': features, 'vote_cols': list(range(2 * n_types)),
            'types': types, 'labelled': labelled,
            'masks': [sorted(masks.get(i, ())) for i in range(n)], 'new': new,
            'edges': edges, 'heads': head_groups(forms)}


def homophily(graph):
    """Per relation, the share of edges between labelled nodes that join nodes of the same
    gold type, with the number of such edges (head cliques counted pair by pair)."""
    lab = graph['labelled']
    out = {}
    for name, edges in graph['edges'].items():
        pairs = [(lab[i], lab[j]) for i, j in edges if i in lab and j in lab]
        out[name] = {'labelled_edges': len(pairs),
                     'same_type': round(sum(a == b for a, b in pairs) / len(pairs), 4)
                     if pairs else None}
    same = total = 0
    for group in graph['heads']:
        kinds = Counter(lab[i] for i in group if i in lab)
        k = sum(kinds.values())
        total += k * (k - 1) // 2
        same += sum(c * (c - 1) // 2 for c in kinds.values())
    out['head'] = {'labelled_edges': total,
                   'same_type': round(same / total, 4) if total else None}
    return out


# 3. the rankers ---------------------------------------------------------------------------

def aggregators(graph, n):
    """Per relation a function h -> the mean of h over each node's neighbours (0 without
    any): sparse row-normalised adjacency for co-occurrence and kNN, group means for heads."""
    import torch
    out = []
    for name in ('cooccurrence', 'knn'):
        edges = graph['edges'][name]
        if edges:
            rows = [i for i, j in edges] + [j for i, j in edges]
            cols = [j for i, j in edges] + [i for i, j in edges]
            degree = Counter(rows)
            values = [1.0 / degree[r] for r in rows]
            adj = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(values),
                                          (n, n), check_invariants=False).coalesce()
            out.append(lambda h, adj=adj: torch.sparse.mm(adj, h))
        else:
            out.append(lambda h: torch.zeros_like(h))
    member, size = [-1] * n, []
    for g, group in enumerate(graph['heads']):
        size.append(len(group))
        for i in group:
            member[i] = g
    member_t = torch.tensor(member)
    inside = member_t >= 0
    size_t = torch.tensor(size, dtype=torch.float32) if size else torch.zeros(0)

    def head_mean(h):
        res = torch.zeros_like(h)
        if not size:
            return res
        sums = torch.zeros((len(size), h.size(1)), dtype=h.dtype)
        sums.index_add_(0, member_t[inside], h[inside])
        idx = member_t[inside]
        res[inside] = (sums[idx] - h[inside]) / (size_t[idx] - 1).unsqueeze(1)
        return res
    out.append(head_mean)
    return out


def make_model(in_dim, n_types, relations):
    """A 2-layer R-GCN with `relations` relations (mean aggregation per relation, a weight
    per relation and one for the node itself); with 0 relations, the 2-layer MLP M."""
    import torch

    class Ranker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self1 = torch.nn.Linear(in_dim, HIDDEN)
            self.rel1 = torch.nn.ModuleList(torch.nn.Linear(in_dim, HIDDEN, bias=False)
                                            for _ in range(relations))
            self.self2 = torch.nn.Linear(HIDDEN, n_types)
            self.rel2 = torch.nn.ModuleList(torch.nn.Linear(HIDDEN, n_types, bias=False)
                                            for _ in range(relations))

        def forward(self, x, aggs):
            h = self.self1(x) + sum(lin(agg(x)) for lin, agg in zip(self.rel1, aggs))
            h = torch.nn.functional.dropout(torch.relu(h), DROPOUT, self.training)
            return self.self2(h) + sum(lin(agg(h)) for lin, agg in zip(self.rel2, aggs))
    return Ranker()


def train_ranker(graph, use_graph, use_vote, train_nodes, seed=SEED):
    """Train M (use_graph False) or G on `train_nodes` (labelled), full batch, EPOCHS epochs;
    the vote block is dropped per node with probability VOTE_DROPOUT during training (or
    removed with use_vote False). Returns the softmax probabilities of every node."""
    import torch
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x = torch.tensor(graph['features'])
    vote_cols = torch.tensor(graph['vote_cols'])
    if not use_vote:
        keep = [c for c in range(x.size(1)) if c not in set(graph['vote_cols'])]
        x, vote_cols = x[:, keep], torch.tensor([], dtype=torch.long)
    n = x.size(0)
    aggs = aggregators(graph, n) if use_graph else []
    model = make_model(x.size(1), len(graph['types']), len(aggs))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    nodes = torch.tensor(train_nodes)
    targets = torch.tensor([graph['labelled'][i] for i in train_nodes])
    for _ in range(EPOCHS):
        model.train()
        xin = x
        if len(vote_cols):
            dropped = torch.rand(n, generator=gen) < VOTE_DROPOUT
            xin = x.clone()
            xin[dropped.nonzero(as_tuple=True)[0].unsqueeze(1), vote_cols] = 0.0
        loss = torch.nn.functional.cross_entropy(model(xin, aggs)[nodes], targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x, aggs), dim=1).numpy()


def label_propagation(graph):
    """LP over the union of the relations: each node's distribution the mean of its
    relations' neighbour means, labelled nodes clamped to their type, LP_ITERATIONS steps."""
    import torch
    n, k = len(graph['forms']), len(graph['types'])
    aggs = aggregators(graph, n)
    y = torch.zeros((n, k))
    lab = graph['labelled']
    idx = torch.tensor(sorted(lab))
    if len(idx):
        y[idx, torch.tensor([lab[i] for i in sorted(lab)])] = 1.0
    f = y.clone()
    for _ in range(LP_ITERATIONS):
        f = sum(agg(f) for agg in aggs) / len(aggs)
        if len(idx):
            f[idx] = y[idx]
    return f.numpy()


def teacher_seen_accuracy(graph, use_graph, use_vote, folds=FOLDS, seed=SEED):
    """The 5-fold accuracy (unmasked argmax) on the labelled nodes: a diagnostic only, the
    teachers having trained on these names ("teacher-seen")."""
    nodes = sorted(graph['labelled'])
    if len(nodes) < folds:
        return None
    random.Random(seed).shuffle(nodes)
    right = 0
    for f in range(folds):
        held = nodes[f::folds]
        train = [i for i in nodes if i not in set(held)]
        probs = train_ranker(graph, use_graph, use_vote, sorted(train), seed)
        right += sum(int(np.argmax(probs[i])) == graph['labelled'][i] for i in held)
    return round(right / len(nodes), 4)


# 4. lexicons ------------------------------------------------------------------------------

def ranked_lexicon(graph, probs, size):
    """{words: type} of the `size` best-ranked new names: each name's type the most probable
    among those some teacher tags it with (ties by type name), ranked by that probability
    (ties by the words)."""
    types = graph['types']
    t_index = {t: k for k, t in enumerate(types)}
    scored = []
    for i in graph['new']:
        mask = [t for t in graph['masks'][i] if t in t_index]
        if not mask:
            continue
        kind = min(mask, key=lambda t: (-probs[i][t_index[t]], t))
        scored.append((-float(probs[i][t_index[kind]]), ' '.join(graph['forms'][i]),
                       graph['forms'][i], kind))
    scored.sort(key=lambda s: (s[0], s[1]))
    return {form: kind for _, _, form, kind in scored[:size]}


def new_only(lexicon, train_types):
    return {w: t for w, t in lexicon.items() if w not in train_types}


def mention_entries(spans, lexicon):
    """The lexicon file's entries, the same rule for every arm: one (type, words) per span of
    the dump whose words are an admitted name and at which some teacher's span has the
    admitted type (any marginal), in corpus order (sentence index, then start and end)."""
    out = []
    for _, _, _, words, teachers in spans:
        form = tuple(words)
        kind = lexicon.get(form)
        if kind is not None and any(k == kind for k, _, _ in teachers):
            out.append((kind, form))
    return out


# 5. the numbers ---------------------------------------------------------------------------

def coverage_counts(lexicon, unseen, masks_by_form=None):
    """Counts on the unseen test mentions [(type, words)]: covered (type and words in the
    lexicon), held (words in it at all), wrong (held under another type only) and, with
    masks_by_form, the candidates' ceiling (words a candidate the teachers tag with that
    type somewhere)."""
    covered = sum(lexicon.get(w) == t for t, w in unseen)
    held = sum(w in lexicon for t, w in unseen)
    out = {'unseen': len(unseen), 'covered': covered, 'held': held, 'wrong': held - covered}
    if masks_by_form is not None:
        out['ceiling'] = sum(t in masks_by_form.get(w, ()) for t, w in unseen)
    return out


def rates(counts):
    return {'coverage': round(100 * counts['covered'] / counts['unseen'], 2)
            if counts['unseen'] else None,
            'wrong_type': round(100 * counts['wrong'] / counts['held'], 2)
            if counts['held'] else None}


def gate(units):
    """The protocol's gate over the units' rates at N primary."""
    def mean(key, ranker):
        vals = [u['primary'][ranker][key] for u in units.values()
                if u['primary'][ranker][key] is not None]
        return round(sum(vals) / len(vals), 2) if vals else None
    g, v, m = (mean('coverage', r) for r in ('G', 'V', 'M'))
    gw, vw = mean('wrong_type', 'G'), mean('wrong_type', 'V')
    parts = {
        'coverage_vs_vote': None if None in (g, v) else g - v >= GATE['coverage_vs_vote'],
        'coverage_vs_mlp': None if None in (g, m) else g - m >= GATE['coverage_vs_mlp'],
        'wrong_type_vs_vote': (None if None in (gw, vw)
                               else gw <= vw + GATE['wrong_type_vs_vote'])}
    return {'n_units': len(units), 'means': {'coverage_G': g, 'coverage_V': v,
                                             'coverage_M': m, 'wrong_type_G': gw,
                                             'wrong_type_V': vw},
            'margins': GATE, 'parts': parts,
            'passes': all(parts.values()) if None not in parts.values() else None}


def probe_unit(spans, labels, train_sentences, unseen):
    """Everything the probe reports for one unit, and its lexicons {(ranker, size): lexicon}."""
    train_types, _ = training_types(train_sentences)
    votes = {(t, a): vote_names(spans, t, a) for t in THRESHOLDS for a in AGREEMENTS}
    vote_lex = {key: new_only(one_type(names), train_types) for key, names in votes.items()}
    sizes = {'primary': len(vote_lex[PRIMARY]), 'secondary': len(vote_lex[SECONDARY])}
    graph = build_graph(spans, labels, train_sentences)
    masks_by_form = {f: set(m) for f, m in zip(graph['forms'], graph['masks'])}
    labelled = sorted(graph['labelled'])
    probs = {'M': train_ranker(graph, False, True, labelled),
             'G': train_ranker(graph, True, True, labelled),
             'M_novote': train_ranker(graph, False, False, labelled),
             'G_novote': train_ranker(graph, True, False, labelled),
             'LP': label_propagation(graph)}
    lexicons = {}
    report = {'nodes': len(graph['forms']), 'labelled': len(labelled), 'new': len(graph['new']),
              'types': graph['types'], 'homophily': homophily(graph),
              'edges': {k: len(v) for k, v in graph['edges'].items()},
              'head_groups': len(graph['heads']), 'sizes': sizes,
              'teacher_seen_accuracy': {
                  'M': teacher_seen_accuracy(graph, False, True),
                  'G': teacher_seen_accuracy(graph, True, True)},
              'vote_grid': {f'{t}_{a}': {'size': len(vote_lex[(t, a)]),
                                         **coverage_counts(vote_lex[(t, a)], unseen)}
                            for t in THRESHOLDS for a in AGREEMENTS},
              'ceiling': coverage_counts({}, unseen, masks_by_form)['ceiling']}
    for which, key in (('primary', PRIMARY), ('secondary', SECONDARY)):
        size = sizes[which]
        lexicons[('V', which)] = vote_lex[key]
        for name, p in probs.items():
            lexicons[(name, which)] = ranked_lexicon(graph, p, size)
        report[which] = {}
        for name in ('V', *probs):
            counts = coverage_counts(lexicons[(name, which)], unseen)
            report[which][name] = {**counts, **rates(counts),
                                   'size': len(lexicons[(name, which)])}
    return report, lexicons, votes


# inputs, checks and the main loop ---------------------------------------------------------

def load_dump(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def unit_parts(setting):
    """{unit: (train sentences [(words, tags)], unseen test mentions [(type, words)])}, the
    splits rebuilt from the stored crf_mr6 file (test positions checked)."""
    import ner_mentions as nm
    from ner_baseline import entity_spans
    corpus, _, results_name, *_ = SETTINGS[setting]
    results = json.loads((GPU_RESULTS / results_name).read_text())
    data = nm.corpus(corpus)
    out = {}
    for run in results['runs']:
        rows, train, test = nm.run_split(results, run, data, corpus)
        train_s = [(rows[i][0], rows[i][1]) for i in train]
        _, seen = training_types(train_s)
        mentions = [(kind, tuple(rows[i][0][s:e])) for i in test
                    for s, e, kind in entity_spans(rows[i][1])]
        out[run['run_id']] = (train_s, [m for m in mentions if m not in seen])
    return out


def dumps_of(setting, dump_dir=DUMP_DIR):
    """{unit: dump run record} of a setting's dump files present in dump_dir."""
    pattern = SETTINGS[setting][1]
    runs = {}
    for path in sorted(Path(dump_dir).glob(pattern.replace('{unit}', '*'))):
        runs.update(load_dump(path)['runs'])
    return runs


def names_records(setting, which, names_dir=NAMES_DIR):
    """{unit: --save-names record} of week 15's files (which = 3) or w19a's (which = 4); a
    week-15 MuLMS pattern does not take the _w19 files."""
    pattern = SETTINGS[setting][which]
    runs = {}
    for path in sorted(Path(names_dir).glob(pattern.replace('{unit}', '*'))):
        if which == 3 and path.stem.endswith('_w19'):
            continue
        runs.update(json.loads(path.read_text()).get('runs', {}))
    return runs


# the stored names runs whose teachers the harvest rebuilds: (granularity, protocol, family,
# seed offsets)
NAMES_RUNS = {'mulms': ('mulms', 'leave_one_subfield_out', 'crf_mrm_topic6', range(300, 1000, 100)),
              'fine': ('fine', 'leave_one_paper_out', 'crf_mrm6', range(0, 1000, 100))}


def teachers_check(teachers, setting, unit, directory=GPU_RESULTS):
    """Whether the rebuilt teachers' test F1 equal those stored with every names run of the
    unit (week 15's check: deterministic CUDA gives the same teachers)."""
    import ner_mentions as nm
    granularity, protocol, family, offsets = NAMES_RUNS[setting]
    files = nm.family_files(directory, granularity, protocol, family, tuple(offsets)) or []
    rebuilt = [t['test_f1'] for t in teachers or []]
    same = [[t['test_f1'] for t in run['self_training']['teachers']] == rebuilt
            for f in files for run in f['runs'] if run['run_id'] == unit]
    return {'equal': bool(same) and all(same), 'runs_checked': len(same),
            'rebuilt_teachers_f1': rebuilt}


def names_check(votes, record):
    """V at 0.9, 3 of 3 rebuilt from the dump against the --save-names record of the same job
    (week 15's format: [type, words, count] rows): equal counts or the differences."""
    rebuilt = Counter((k, ' '.join(w)) for k, w in votes[SECONDARY])
    stored = Counter({(k, w): c for k, w, c in record['names']})
    return {'equal': rebuilt == stored, 'n_rebuilt': sum(rebuilt.values()),
            'n_stored': sum(stored.values())}


def names_file(entries_by_unit, arm, source):
    """A --names-file store: {'runs': {unit: {'names': [[type, words], ...]}}, ...}."""
    return {'arm': arm, 'source': source, 'created_by': 'graph_names.py (week 19)',
            'runs': {u: {'names': [[k, ' '.join(w)] for k, w in entries]}
                     for u, entries in sorted(entries_by_unit.items())}}


def run_probe(dump_dir=DUMP_DIR, names_dir=NAMES_DIR, write_names=False):
    units, files = {}, defaultdict(dict)
    for setting in SETTINGS:
        dumps = dumps_of(setting, dump_dir)
        if not dumps:
            continue
        parts = unit_parts(setting)
        week15, w19a = names_records(setting, 3, names_dir), names_records(setting, 4, names_dir)
        for unit in sorted(dumps):
            record = dumps[unit]
            train_s, unseen = parts[unit]
            report, lexicons, votes = probe_unit(record['spans'], record['labels'], train_s,
                                                 unseen)
            report['teachers'] = record.get('teachers')
            # the checks: V at 0.9, 3 of 3 from the dump against the same job's names and
            # against week 15's rebuild (deterministic teachers give the same names)
            report['checks'] = {
                'teachers': teachers_check(record.get('teachers'), setting, unit),
                'dump_vs_w19a_names': (names_check(votes, w19a[unit]) if unit in w19a
                                       else None),
                'dump_vs_week15_names': (names_check(votes, week15[unit]) if unit in week15
                                         else None)}
            units[f'{setting}:{unit}'] = report
            if write_names:
                spans = record['spans']
                files[(setting, 'g')][unit] = mention_entries(spans, lexicons[('G', 'primary')])
                files[(setting, 'v')][unit] = mention_entries(spans, lexicons[('V', 'primary')])
                files[(setting, 'f')][unit] = votes[SECONDARY]
    out = {'protocol': 'ROADMAP §4, week 19 (fixed 3 November 2026, 00:32 UTC)',
           'units': units, 'gate': gate(units) if units else None}
    written = {}
    if write_names:
        for (setting, arm), entries in sorted(files.items()):
            store = names_file(entries, arm, f'{setting} candidate dumps')
            path = Path(dump_dir) / f'names_{setting}_{arm}.json'
            raw = json.dumps(store).encode('utf-8')
            path.write_bytes(raw)
            written[path.name] = hashlib.sha256(raw).hexdigest()
        out['names_files_sha256'] = written
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--dumps', default=str(DUMP_DIR), help='the candidate dumps')
    parser.add_argument('--names-files', action='store_true',
                        help="also write the Phase-1 lexicon files (local; never versioned)")
    parser.add_argument('--output', default=str(OUTPUT))
    args = parser.parse_args(argv)
    out = run_probe(Path(args.dumps), write_names=args.names_files)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1) + '\n')
    g = out['gate']
    print(f"{len(out['units'])} units; gate: {g['passes'] if g else 'no units'}")
    if g:
        print(json.dumps(g['means']))


if __name__ == '__main__':
    main()
