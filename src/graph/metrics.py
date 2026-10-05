"""
metrics.py - Leakage-free evaluation metrics (pure functions)

Every metric here compares a *prediction* with the *gold annotation of the
same item*. Nothing is credited just for producing a valid ontology label, and
no denominator is derived from the system's own output (both were problems of
the previous evaluation).

Conventions
  - Universe U: the evaluated terms (keys of `gold`).
  - gold[t]: set of gold class URIs of term t (a surface form can carry several
    annotated types; 47 of the 315 graph terms do).
  - majority[t]: the majority gold class of t (single-label view).
  - pred[t]: predicted class URI, or None when the system abstains. Terms that
    are missing from `pred` count as abstentions, i.e. errors for recall.

Only the standard library is needed, except clustering_metrics (scikit-learn)
and mean_ci (SciPy, a scikit-learn dependency).
"""

import math
import random
import statistics
from collections import Counter, defaultdict


def safe_div(num, den):
    return num / den if den else 0.0


def f1_score(precision, recall):
    return safe_div(2 * precision * recall, precision + recall)


def _r(x):
    return round(float(x), 4)


# ---------------------------------------------------------------------------
# Term typing
# ---------------------------------------------------------------------------
def typing_metrics(pred, gold, majority):
    """Core term-typing metrics.

    Returns:
      accuracy: correct / |U| (abstentions are wrong). With full coverage this
        equals precision = recall = F1 of the single-label view.
      precision / recall / f1: correct / predicted, correct / |U|.
      set_precision / set_recall / set_f1: LLMs4OL Task-B style, predicted and
        gold types as sets per term (P = sum|p&g| / sum|p|, R = sum|p&g| / sum|g|).
      macro_f1: per-class F1 against the majority label, averaged over the
        classes that occur as majority labels (abstentions are false negatives).
      class_precision / class_recall: fraction of predicted classes that are
        correct for at least one term / fraction of active gold classes
        (union of gold sets) assigned correctly to at least one term.
    """
    terms = sorted(gold)
    n = len(terms)
    correct = 0
    n_pred = 0
    set_inter = set_pred = set_gold = 0
    tp, fp, fn = Counter(), Counter(), Counter()
    predicted_classes = set()
    correct_classes = set()
    active_classes = set()

    for t in terms:
        g = gold[t]
        p = pred.get(t)
        active_classes |= g
        set_gold += len(g)
        maj = majority[t]
        if p is None:
            fn[maj] += 1
            continue
        n_pred += 1
        set_pred += 1
        predicted_classes.add(p)
        if p in g:
            correct += 1
            set_inter += 1
            correct_classes.add(p)
        if p == maj:
            tp[maj] += 1
        else:
            fp[p] += 1
            fn[maj] += 1

    labels = sorted(set(majority[t] for t in terms))
    per_class_f1 = []
    for c in labels:
        p_c = safe_div(tp[c], tp[c] + fp[c])
        r_c = safe_div(tp[c], tp[c] + fn[c])
        per_class_f1.append(f1_score(p_c, r_c))

    precision = safe_div(correct, n_pred)
    recall = safe_div(correct, n)
    set_p = safe_div(set_inter, set_pred)
    set_r = safe_div(set_inter, set_gold)
    class_p = safe_div(len(correct_classes), len(predicted_classes))
    class_r = safe_div(len(correct_classes & active_classes), len(active_classes))
    return {
        'n_terms': n,
        'n_predicted': n_pred,
        'coverage': _r(safe_div(n_pred, n)),
        'n_correct': correct,
        'accuracy': _r(safe_div(correct, n)),
        'precision': _r(precision),
        'recall': _r(recall),
        'f1': _r(f1_score(precision, recall)),
        'set_precision': _r(set_p),
        'set_recall': _r(set_r),
        'set_f1': _r(f1_score(set_p, set_r)),
        'macro_f1': _r(safe_div(sum(per_class_f1), len(per_class_f1))),
        'n_majority_classes': len(labels),
        'class_precision': _r(class_p),
        'class_recall': _r(class_r),
        'class_f1': _r(f1_score(class_p, class_r)),
        'n_active_classes': len(active_classes),
    }


def topk_accuracy(candidates, gold, ks=(1, 3, 5)):
    """Fraction of terms whose gold set intersects the top-k candidates.

    candidates[t] is a ranked list of class URIs (missing -> no candidates).
    """
    terms = sorted(gold)
    out = {}
    for k in ks:
        hits = sum(1 for t in terms if set(candidates.get(t, [])[:k]) & gold[t])
        out[f'acc@{k}'] = _r(safe_div(hits, len(terms)))
    return out


def hierarchical_prf(pred, gold, majority, ancestors_fn):
    """Hierarchical precision/recall/F1 (Kiritchenko et al., 2005).

    ancestors_fn(c) must return the set {c} plus its ancestors, *excluding*
    the ontology root (which every class shares and would give free credit).
    For a multi-label gold set the gold class that maximises the per-term
    hierarchical F1 is used (ties go to the majority class). Abstentions add
    |A(majority)| to the recall denominator only.
    """
    num = den_p = den_r = 0
    for t in sorted(gold):
        p = pred.get(t)
        maj = majority[t]
        if p is None:
            den_r += len(ancestors_fn(maj))
            continue
        ap = ancestors_fn(p)
        best = None
        for g in sorted(gold[t]):
            ag = ancestors_fn(g)
            inter = len(ap & ag)
            hf = f1_score(safe_div(inter, len(ap)), safe_div(inter, len(ag)))
            key = (hf, g == maj)
            if best is None or key > best[0]:
                best = (key, inter, len(ag))
        num += best[1]
        den_p += len(ap)
        den_r += best[2]
    hp = safe_div(num, den_p)
    hr = safe_div(num, den_r)
    return {'h_precision': _r(hp), 'h_recall': _r(hr), 'h_f1': _r(f1_score(hp, hr))}


def threshold_sweep(pred, scores, gold, thresholds):
    """P/R/F1 when predictions with score < threshold are turned into abstentions.

    Scores are only comparable within one run/scorer; report a threshold fixed
    in advance (or the full-coverage accuracy) when comparing systems.
    """
    terms = sorted(gold)
    n = len(terms)
    out = {}
    for th in thresholds:
        n_pred = correct = 0
        for t in terms:
            p = pred.get(t)
            if p is None or scores.get(t, float('-inf')) < th:
                continue
            n_pred += 1
            correct += int(p in gold[t])
        prec = safe_div(correct, n_pred)
        rec = safe_div(correct, n)
        out[str(th)] = {'precision': _r(prec), 'recall': _r(rec),
                        'f1': _r(f1_score(prec, rec)), 'n_predicted': n_pred}
    return out


def risk_coverage(pred, scores, gold):
    """Area under the risk-coverage curve (lower is better) and its curve.

    Terms are ranked by score (abstentions last, counted as errors); risk at
    coverage k/N is the error rate among the k most confident terms. This is
    threshold-free, so it compares systems whose scores live on different
    scales.
    """
    terms = sorted(gold)
    ranked = sorted(
        terms,
        key=lambda t: (pred.get(t) is None, -scores.get(t, float('-inf')), t))
    errors = 0
    risks = []
    for k, t in enumerate(ranked, 1):
        p = pred.get(t)
        errors += int(p is None or p not in gold[t])
        risks.append(errors / k)
    aurc = safe_div(sum(risks), len(risks))
    return {'aurc': _r(aurc)}


def bootstrap_ci(items, stat_fn, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of stat_fn over resampled items (seeded)."""
    items = list(items)
    if not items:
        return [0.0, 0.0]
    rng = random.Random(seed)
    stats = sorted(stat_fn([items[rng.randrange(len(items))] for _ in items])
                   for _ in range(n_boot))
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return [_r(lo), _r(hi)]


def accuracy_ci(pred, gold, n_boot=1000, seed=0):
    """Bootstrap 95% CI of accuracy over terms."""
    hits = [int(pred.get(t) is not None and pred.get(t) in gold[t])
            for t in sorted(gold)]
    return bootstrap_ci(hits, lambda xs: safe_div(sum(xs), len(xs)),
                        n_boot=n_boot, seed=seed)


def mcnemar_exact(pred_a, pred_b, gold):
    """Exact two-sided McNemar test of two systems' accuracy on the same terms.

    Only discordant terms count: only_a = terms system A gets right and B gets
    wrong, only_b the reverse. p is the two-sided binomial(only_a + only_b,
    0.5) tail. Use it before claiming that one configuration beats another.
    """
    only_a = only_b = 0
    for t in gold:
        a_ok = pred_a.get(t) is not None and pred_a.get(t) in gold[t]
        b_ok = pred_b.get(t) is not None and pred_b.get(t) in gold[t]
        only_a += int(a_ok and not b_ok)
        only_b += int(b_ok and not a_ok)
    n = only_a + only_b
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1))
    p = min(1.0, 2 * tail / 2 ** n) if n else 1.0
    return {'only_a': only_a, 'only_b': only_b,
            'accuracy_diff': _r(safe_div(only_a - only_b, len(gold))),
            'p_value': _r(p)}


def mean_ci(values, confidence=0.95):
    """Mean of repeated runs (seeds, splits) with a Student-t confidence interval.

    The interval uses the sample SD (n - 1); with a single value it collapses
    to that value. Use it to compare a mean over a few runs with a published
    number: with 5 runs the half-width is 2.78 sample SDs / sqrt(5).
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if not n:
        return {'n': 0, 'mean': 0.0, 'sd': 0.0, 'ci': [0.0, 0.0]}
    mean = statistics.fmean(vals)
    if n == 1:
        return {'n': 1, 'mean': _r(mean), 'sd': 0.0, 'ci': [_r(mean), _r(mean)]}
    from scipy import stats
    sd = statistics.stdev(vals)
    half = float(stats.t.ppf((1 + confidence) / 2, n - 1)) * sd / math.sqrt(n)
    return {'n': n, 'mean': _r(mean), 'sd': _r(sd), 'ci': [_r(mean - half), _r(mean + half)]}


# ---------------------------------------------------------------------------
# Alignment rows (term, class, score) - honest counterpart of the old Term F1
# ---------------------------------------------------------------------------
def pairwise_alignment_metrics(rows, gold, thresholds):
    """P/R/F1 of (term, class) alignment rows at each score threshold.

    rows: iterable of (term, class_uri, score). A row is correct only if the
    class is one of the term's gold classes. Recall is over all gold
    (term, class) pairs of the universe. Rows for terms outside the universe
    are ignored.
    """
    rows = [(t, c, s) for t, c, s in rows if t in gold]
    total_gold = sum(len(g) for g in gold.values())
    out = {}
    for th in thresholds:
        sel = [(t, c) for t, c, s in rows if s >= th]
        correct_rows = [(t, c) for t, c in sel if c in gold[t]]
        prec = safe_div(len(correct_rows), len(sel))
        rec = safe_div(len(set(correct_rows)), total_gold)
        out[str(th)] = {'precision': _r(prec), 'recall': _r(rec),
                        'f1': _r(f1_score(prec, rec)), 'n_rows': len(sel)}
    return out


# ---------------------------------------------------------------------------
# Clustering (concept level)
# ---------------------------------------------------------------------------
def bcubed(cluster_of, label_of):
    """B-cubed precision/recall/F1 of a clustering against single labels."""
    items = sorted(label_of)
    by_cluster = defaultdict(set)
    by_label = defaultdict(set)
    for i in items:
        by_cluster[cluster_of[i]].add(i)
        by_label[label_of[i]].add(i)
    p_sum = r_sum = 0.0
    for i in items:
        c = by_cluster[cluster_of[i]]
        lab = by_label[label_of[i]]
        inter = len(c & lab)
        p_sum += inter / len(c)
        r_sum += inter / len(lab)
    p = safe_div(p_sum, len(items))
    r = safe_div(r_sum, len(items))
    return {'bcubed_precision': _r(p), 'bcubed_recall': _r(r),
            'bcubed_f1': _r(f1_score(p, r))}


def clustering_metrics(cluster_of, label_of):
    """ARI, V-measure, NMI and B-cubed of a clustering vs. majority labels.

    Items of `label_of` missing from `cluster_of` become singleton clusters.
    Also reports the all-singletons and one-cluster baselines, because B-cubed
    rewards singletons when many classes have a single term.
    """
    from sklearn.metrics import (adjusted_rand_score,
                                 normalized_mutual_info_score, v_measure_score)

    items = sorted(label_of)
    assign = {}
    for i in items:
        assign[i] = ('c', cluster_of[i]) if i in cluster_of else ('s', i)
    ids = {c: k for k, c in enumerate(sorted(set(assign.values()), key=str))}
    y_clu = [ids[assign[i]] for i in items]
    y_lab = [label_of[i] for i in items]

    def score(y_c):
        res = {'ari': _r(adjusted_rand_score(y_lab, y_c)),
               'v_measure': _r(v_measure_score(y_lab, y_c)),
               'nmi': _r(normalized_mutual_info_score(y_lab, y_c)),
               'n_clusters': len(set(y_c))}
        res.update(bcubed(dict(zip(items, y_c)), dict(zip(items, y_lab))))
        return res

    result = score(y_clu)
    result['baseline_singletons'] = score(list(range(len(items))))
    result['baseline_one_cluster'] = score([0] * len(items))
    return result


# ---------------------------------------------------------------------------
# Class-level taxonomy edges
# ---------------------------------------------------------------------------
def taxonomy_edge_metrics(pred_edges, ontology, active_classes):
    """Evaluate (child_class, parent_class) edges against the ontology.

    precision_closure: share of edges whose parent is an ancestor of the child.
    inverted: share of edges pointing the wrong way (child is an ancestor).
    precision / recall / f1: exact match against the transitive reduction of
    subClassOf restricted to the active classes (classes in the corpus).
    random_precision_closure: expected precision_closure of random ordered
    pairs of distinct active classes (chance level).
    """
    edges = {(c, p) for c, p in pred_edges if c != p}
    gold = ontology.reduced_edges(active_classes)
    n = len(edges)
    in_closure = sum(1 for c, p in edges if ontology.is_subclass(c, p))
    inverted = sum(1 for c, p in edges if ontology.is_subclass(p, c))
    exact = len(edges & gold)
    prec = safe_div(exact, n)
    rec = safe_div(exact, len(gold))
    active = sorted(set(active_classes) - {ontology.root})
    n_pairs = len(active) * (len(active) - 1)
    closure_pairs = sum(1 for c in active for p in active
                        if c != p and ontology.is_subclass(c, p))
    return {
        'n_edges': n,
        'precision_closure': _r(safe_div(in_closure, n)),
        'inverted': _r(safe_div(inverted, n)),
        'precision': _r(prec),
        'recall': _r(rec),
        'f1': _r(f1_score(prec, rec)),
        'n_gold_edges': len(gold),
        'random_precision_closure': _r(safe_div(closure_pairs, n_pairs)),
    }


# ---------------------------------------------------------------------------
# Class-level relation triples
# ---------------------------------------------------------------------------
def triple_metrics(pred_triples, gold_triples, property_ancestors_fn=None):
    """P/R/F1 of (subject, property, object) class triples.

    A predicted triple matches a gold triple with the same subject and object
    when the gold property equals the predicted property or is one of its
    super-properties (e.g. a predicted causeOf matches a gold prov:influenced,
    since causeOf is declared a sub-property of it).
    """
    pred_triples = set(pred_triples)
    gold_triples = set(gold_triples)
    anc = property_ancestors_fn or (lambda p: {p})
    gold_by_pair = defaultdict(set)
    for s, p, o in gold_triples:
        gold_by_pair[(s, o)].add(p)
    matched_pred = 0
    matched_gold = set()
    for s, p, o in pred_triples:
        hits = gold_by_pair.get((s, o), set()) & anc(p)
        if hits:
            matched_pred += 1
            matched_gold |= {(s, gp, o) for gp in hits}
    prec = safe_div(matched_pred, len(pred_triples))
    rec = safe_div(len(matched_gold), len(gold_triples))
    return {'precision': _r(prec), 'recall': _r(rec), 'f1': _r(f1_score(prec, rec)),
            'n_predicted': len(pred_triples), 'n_gold': len(gold_triples),
            'n_matched_gold': len(matched_gold)}
