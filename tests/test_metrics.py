"""Tests for the leakage-free metrics (metrics.py)."""

import pytest

import metrics

GOLD = {'crack': {'Crack'}, 'alloy': {'Alloy'}, 'data': {'DataSet', 'TestData'},
        'fatigue': {'Mechanism'}}
MAJ = {'crack': 'Crack', 'alloy': 'Alloy', 'data': 'DataSet', 'fatigue': 'Mechanism'}


def test_perfect_predictions():
    pred = {t: MAJ[t] for t in GOLD}
    res = metrics.typing_metrics(pred, GOLD, MAJ)
    assert res['accuracy'] == res['precision'] == res['recall'] == res['f1'] == 1.0
    assert res['macro_f1'] == 1.0
    # set recall < 1: 'data' has two gold types but only one is predicted
    assert res['set_precision'] == 1.0
    assert res['set_recall'] == pytest.approx(4 / 5)


def test_valid_but_wrong_class_scores_zero():
    """The old evaluator gave precision 1.0 to any prediction that was *some*
    ontology class. A wrong class must score zero here."""
    pred = {'crack': 'Alloy', 'alloy': 'Crack', 'data': 'Mechanism', 'fatigue': 'DataSet'}
    res = metrics.typing_metrics(pred, GOLD, MAJ)
    assert res['accuracy'] == 0.0
    assert res['precision'] == 0.0
    assert res['set_f1'] == 0.0
    assert res['macro_f1'] == 0.0
    assert res['class_precision'] == 0.0


def test_abstentions_count_against_recall_only():
    pred = {'crack': 'Crack', 'alloy': None}
    res = metrics.typing_metrics(pred, GOLD, MAJ)
    assert res['coverage'] == 0.25
    assert res['precision'] == 1.0
    assert res['recall'] == 0.25
    assert res['accuracy'] == 0.25


def test_any_gold_type_counts_as_correct():
    pred = {'data': 'TestData'}
    res = metrics.typing_metrics(pred, {'data': GOLD['data']}, {'data': 'DataSet'})
    assert res['accuracy'] == 1.0
    # macro-F1 is computed against the majority label
    assert res['macro_f1'] == 0.0


def test_topk_accuracy():
    cand = {'crack': ['Alloy', 'Crack'], 'alloy': ['Alloy'], 'data': [], 'fatigue': ['X', 'Y', 'Z', 'Mechanism']}
    res = metrics.topk_accuracy(cand, GOLD, ks=(1, 2, 5))
    assert res == {'acc@1': 0.25, 'acc@2': 0.5, 'acc@5': 0.75}


def test_hierarchical_credit_for_parent_class():
    tree = {'Crack': {'Crack', 'Defect'}, 'Defect': {'Defect'}, 'Alloy': {'Alloy'}}
    gold = {'crack': {'Crack'}}
    maj = {'crack': 'Crack'}
    exact = metrics.hierarchical_prf({'crack': 'Crack'}, gold, maj, tree.get)
    parent = metrics.hierarchical_prf({'crack': 'Defect'}, gold, maj, tree.get)
    wrong = metrics.hierarchical_prf({'crack': 'Alloy'}, gold, maj, tree.get)
    none = metrics.hierarchical_prf({}, gold, maj, tree.get)
    assert exact['h_f1'] == 1.0
    assert parent['h_precision'] == 1.0 and parent['h_recall'] == 0.5
    assert wrong['h_f1'] == 0.0
    assert none['h_recall'] == 0.0


def test_threshold_sweep_and_risk_coverage():
    pred = {'crack': 'Crack', 'alloy': 'Crack', 'data': 'DataSet', 'fatigue': None}
    scores = {'crack': 0.9, 'alloy': 0.2, 'data': 0.6}
    sweep = metrics.threshold_sweep(pred, scores, GOLD, [0.5])
    assert sweep['0.5'] == {'precision': 1.0, 'recall': 0.5, 'f1': pytest.approx(0.6667),
                            'n_predicted': 2}
    # ranked: crack(ok) data(ok) alloy(err) fatigue(abstain, err)
    expected = (0 + 0 + 1 / 3 + 2 / 4) / 4
    assert metrics.risk_coverage(pred, scores, GOLD)['aurc'] == pytest.approx(expected, abs=1e-4)


def test_bootstrap_is_deterministic():
    pred = {t: MAJ[t] for t in list(GOLD)[:2]}
    assert metrics.accuracy_ci(pred, GOLD, seed=3) == metrics.accuracy_ci(pred, GOLD, seed=3)


def test_mcnemar_exact():
    gold = {f't{i}': {'A'} for i in range(10)}
    good = {t: 'A' for t in gold}
    worse = {t: ('A' if i < 4 else 'B') for i, t in enumerate(sorted(gold))}
    res = metrics.mcnemar_exact(good, worse, gold)
    assert (res['only_a'], res['only_b']) == (6, 0)
    assert res['accuracy_diff'] == 0.6
    assert res['p_value'] == pytest.approx(2 / 2 ** 6, abs=1e-4)
    assert metrics.mcnemar_exact(worse, good, gold)['p_value'] == res['p_value']
    # 4 vs 2 discordant terms: p = 2 * (1 + 6 + 15) / 64, far from significant
    mixed = dict(good, t4='B', t5='B')
    other = dict(good, t0='B', t1='B', t2='B', t3='B')
    assert metrics.mcnemar_exact(mixed, other, gold)['p_value'] == pytest.approx(0.6875)
    assert metrics.mcnemar_exact(good, good, gold)['p_value'] == 1.0


def test_pairwise_alignment_rows():
    rows = [('crack', 'Crack', 0.9), ('crack', 'Alloy', 0.8), ('alloy', 'Alloy', 0.4),
            ('unknown', 'Crack', 0.99)]
    res = metrics.pairwise_alignment_metrics(rows, GOLD, [0.5])
    assert res['0.5']['precision'] == 0.5          # one of two rows is right
    assert res['0.5']['recall'] == pytest.approx(1 / 5)  # 5 gold (term, class) pairs


def test_clustering_metrics_and_baselines():
    labels = {'a': 'X', 'b': 'X', 'c': 'Y', 'd': 'Y'}
    perfect = metrics.clustering_metrics({'a': 0, 'b': 0, 'c': 1, 'd': 1}, labels)
    assert perfect['ari'] == 1.0 and perfect['bcubed_f1'] == 1.0
    assert perfect['baseline_one_cluster']['ari'] == 0.0
    # missing items become singletons
    partial = metrics.clustering_metrics({'a': 0, 'b': 0}, labels)
    assert partial['n_clusters'] == 3


class _ToyOntology:
    root = 'Root'
    _anc = {'Crack': {'Defect'}, 'Defect': set(), 'Alloy': set()}

    def is_subclass(self, child, parent):
        return parent in self._anc.get(child, set())

    def reduced_edges(self, active):
        return {(c, p) for c in active for p in self._anc.get(c, set()) if p in active}


def test_taxonomy_edge_metrics():
    onto = _ToyOntology()
    res = metrics.taxonomy_edge_metrics({('Crack', 'Defect'), ('Defect', 'Crack'),
                                         ('Alloy', 'Defect')}, onto,
                                        {'Crack', 'Defect', 'Alloy'})
    assert res['precision'] == pytest.approx(1 / 3, abs=1e-4)
    assert res['recall'] == 1.0
    assert res['inverted'] == pytest.approx(1 / 3, abs=1e-4)
    assert res['random_precision_closure'] == pytest.approx(1 / 6, abs=1e-4)


def test_triple_metrics_with_super_properties():
    gold = {('C1', 'influenced', 'C2'), ('C3', 'composedOf', 'C4')}
    pred = {('C1', 'causeOf', 'C2'), ('C1', 'causeOf', 'C4')}
    anc = {'causeOf': {'causeOf', 'associatedWith', 'influenced'}}.get
    res = metrics.triple_metrics(pred, gold, anc)
    assert res['precision'] == 0.5 and res['recall'] == 0.5
    # without the property hierarchy nothing matches
    assert metrics.triple_metrics(pred, gold)['f1'] == 0.0
