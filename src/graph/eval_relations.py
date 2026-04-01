"""
eval_relations.py - Evaluate Extracted Relations Against Gold Standard

Compares predicted relations (from relations.py) against gold standard
relations (from extract_gold_relations.py) using:
  - Strict match: exact (subject, relation, object) match
  - Relaxed match: correct relation type + at least one entity matches
  - Type-only match: correct relation type regardless of entities

Outputs metrics: Precision, Recall, F1 for each match mode.
"""

import os
import re
import json

import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def normalize(term):
    """Normalize a term for comparison."""
    if not isinstance(term, str):
        return ""
    t = term.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_relation(rel):
    """Normalize relation name for comparison."""
    if not isinstance(rel, str):
        return ""
    r = rel.lower().strip()
    r = re.sub(r"[\s_-]+", "", r)
    return r


def load_predicted(csv_path=None):
    """Load predicted relations from relations.csv."""
    if csv_path is None:
        csv_path = os.path.join(PROCESSED_DIR, "relations.csv")
    if not os.path.exists(csv_path):
        print(f"WARNING: Predicted relations not found at {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # Standardize column names
    col_map = {"subj": "subject", "rel": "relation_type", "obj": "object"}
    df = df.rename(columns=col_map)
    return df


def load_gold(csv_path=None):
    """Load gold standard relations."""
    if csv_path is None:
        csv_path = os.path.join(PROCESSED_DIR, "gold_relations.csv")
    if not os.path.exists(csv_path):
        print(f"WARNING: Gold relations not found at {csv_path}")
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def compute_prf(tp, fp, fn):
    """Compute precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def evaluate_relations(predicted_csv=None, gold_csv=None):
    """Evaluate extracted relations against gold standard."""
    pred_df = load_predicted(predicted_csv)
    gold_df = load_gold(gold_csv)

    if pred_df.empty:
        print("No predicted relations to evaluate.")
        return {}
    if gold_df.empty:
        print("No gold relations to evaluate against.")
        return {}

    print(f"Predicted relations: {len(pred_df)}")
    print(f"Gold relations: {len(gold_df)}")

    # Build normalized sets
    pred_set = set()
    for _, r in pred_df.iterrows():
        subj = normalize(str(r.get("subject", "")))
        rel = normalize_relation(str(r.get("relation_type", "")))
        obj = normalize(str(r.get("object", "")))
        if subj and rel and obj:
            pred_set.add((subj, rel, obj))

    gold_set = set()
    for _, r in gold_df.iterrows():
        subj = normalize(str(r.get("subject", "")))
        rel = normalize_relation(str(r.get("relation", "")))
        obj = normalize(str(r.get("object", "")))
        if subj and rel and obj:
            gold_set.add((subj, rel, obj))

    print(f"Unique predicted (normalized): {len(pred_set)}")
    print(f"Unique gold (normalized): {len(gold_set)}")

    # ---- Strict match: exact triple match ----
    strict_tp = len(pred_set & gold_set)
    strict_fp = len(pred_set - gold_set)
    strict_fn = len(gold_set - pred_set)
    strict = compute_prf(strict_tp, strict_fp, strict_fn)

    # ---- Relaxed match: correct relation type + at least one entity ----
    relaxed_tp = 0
    relaxed_matched_gold = set()
    for p in pred_set:
        for g_item in gold_set:
            if p[1] == g_item[1] and (p[0] == g_item[0] or p[2] == g_item[2]):
                relaxed_tp += 1
                relaxed_matched_gold.add(g_item)
                break
    relaxed_fp = len(pred_set) - relaxed_tp
    relaxed_fn = len(gold_set) - len(relaxed_matched_gold)
    relaxed = compute_prf(relaxed_tp, relaxed_fp, relaxed_fn)

    # ---- Partial entity match: either entity matches (any relation) ----
    partial_tp = 0
    partial_matched_gold = set()
    for p in pred_set:
        for g_item in gold_set:
            if p[0] == g_item[0] or p[2] == g_item[2] or \
               p[0] == g_item[2] or p[2] == g_item[0]:
                partial_tp += 1
                partial_matched_gold.add(g_item)
                break
    partial_fp = len(pred_set) - partial_tp
    partial_fn = len(gold_set) - len(partial_matched_gold)
    partial = compute_prf(partial_tp, partial_fp, partial_fn)

    # ---- Fuzzy entity match: substring match on entities ----
    fuzzy_tp = 0
    fuzzy_matched_gold = set()
    for p in pred_set:
        for g_item in gold_set:
            if p[1] != g_item[1]:
                continue
            subj_match = (p[0] in g_item[0] or g_item[0] in p[0]) if (
                len(p[0]) >= 3 and len(g_item[0]) >= 3) else p[0] == g_item[0]
            obj_match = (p[2] in g_item[2] or g_item[2] in p[2]) if (
                len(p[2]) >= 3 and len(g_item[2]) >= 3) else p[2] == g_item[2]
            if subj_match and obj_match:
                fuzzy_tp += 1
                fuzzy_matched_gold.add(g_item)
                break
    fuzzy_fp = len(pred_set) - fuzzy_tp
    fuzzy_fn = len(gold_set) - len(fuzzy_matched_gold)
    fuzzy = compute_prf(fuzzy_tp, fuzzy_fp, fuzzy_fn)

    # ---- Relation type coverage ----
    pred_rel_types = {p[1] for p in pred_set}
    gold_rel_types = {g[1] for g in gold_set}
    rel_overlap = pred_rel_types & gold_rel_types
    type_coverage = {
        "predicted_types": sorted(pred_rel_types),
        "gold_types": sorted(gold_rel_types),
        "overlap_types": sorted(rel_overlap),
        "coverage_ratio": round(
            len(rel_overlap) / len(gold_rel_types), 4
        ) if gold_rel_types else 0.0,
    }

    # ---- Per-relation-type breakdown ----
    per_type = {}
    for rel_type in gold_rel_types | pred_rel_types:
        p_subset = {t for t in pred_set if t[1] == rel_type}
        g_subset = {t for t in gold_set if t[1] == rel_type}
        tp = len(p_subset & g_subset)
        fp = len(p_subset - g_subset)
        fn = len(g_subset - p_subset)
        per_type[rel_type] = {
            **compute_prf(tp, fp, fn),
            "n_predicted": len(p_subset),
            "n_gold": len(g_subset),
        }

    results = {
        "strict_match": strict,
        "relaxed_match": relaxed,
        "partial_entity_match": partial,
        "fuzzy_match": fuzzy,
        "type_coverage": type_coverage,
        "per_relation_type": per_type,
        "counts": {
            "predicted": len(pred_set),
            "gold": len(gold_set),
        },
    }

    # Print summary
    print("\n" + "=" * 60)
    print("RELATION EVALUATION RESULTS")
    print("=" * 60)

    for mode_name, mode_results in [
        ("Strict Match", strict),
        ("Relaxed Match (rel + 1 entity)", relaxed),
        ("Partial Entity Match (any entity)", partial),
        ("Fuzzy Match (rel + substring entities)", fuzzy),
    ]:
        print(f"\n{mode_name}:")
        print(f"  Precision: {mode_results['precision']:.4f}")
        print(f"  Recall:    {mode_results['recall']:.4f}")
        print(f"  F1:        {mode_results['f1']:.4f}")
        print(f"  TP={mode_results['tp']}  FP={mode_results['fp']}  "
              f"FN={mode_results['fn']}")

    print(f"\nRelation Type Coverage: "
          f"{len(rel_overlap)}/{len(gold_rel_types)} "
          f"({type_coverage['coverage_ratio']:.1%})")

    # Save results
    results_path = os.path.join(PROCESSED_DIR, "relation_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to {results_path}")

    return results


if __name__ == "__main__":
    evaluate_relations()
