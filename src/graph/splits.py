"""
splits.py - Leakage-safe cross-validation folds over terms

Term-typing baselines that learn from labelled terms (kNN, LLM few-shot
demonstrations) must never see the gold type of a test term. Surface variants
such as 'crack' / 'cracks' would leak the label across folds, so terms are
grouped by a crude lemma key (text_match.lemma_key) and every group stays in
one fold. Folds are stratified by the majority class where possible
(StratifiedGroupKFold); many classes have a single term, so scikit-learn's
"least populated class" warning is expected and silenced.
"""

import warnings

from text_match import lemma_key


def term_groups(terms):
    """Map each term to an integer group id shared by its surface variants."""
    keys = {}
    groups = []
    for t in terms:
        k = lemma_key(t)
        if k not in keys:
            keys[k] = len(keys)
        groups.append(keys[k])
    return groups


def group_kfold_splits(terms, labels, n_splits=5, seed=0):
    """Return [(train_terms, test_terms)] for stratified group k-fold.

    terms: list of term strings; labels: majority class per term (same order).
    Every term appears in exactly one test fold; groups never straddle folds.
    When no class has n_splits members, stratification is impossible and a
    seeded GroupKFold over shuffled group ids is used instead.
    """
    import random
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    terms = list(terms)
    groups = term_groups(terms)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                            random_state=seed)
            index_folds = list(splitter.split(terms, labels, groups))
        except ValueError:
            ids = sorted(set(groups))
            random.Random(seed).shuffle(ids)
            relabel = {g: i for i, g in enumerate(ids)}
            index_folds = list(GroupKFold(n_splits=n_splits).split(
                terms, labels, [relabel[g] for g in groups]))
    return [([terms[i] for i in train_idx], [terms[i] for i in test_idx])
            for train_idx, test_idx in index_folds]
