import os
import pandas as pd
import numpy as np

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Try LLM file first, then base
llm_path = os.path.join(PROCESSED_DIR, 'ontology_alignment_llm.csv')
base_path = os.path.join(PROCESSED_DIR, 'ontology_alignment.csv')

if os.path.exists(llm_path):
    print(f"Loading {llm_path}...")
    df = pd.read_csv(llm_path)
elif os.path.exists(base_path):
    print(f"Loading {base_path} (no LLM data)...")
    df = pd.read_csv(base_path)
else:
    raise FileNotFoundError("No alignment CSVs found in data/processed/")

gold = ['Microstructure', 'Crack', 'Dislocation', 'GrainBoundary', 'FatigueStrength', 
        'StrainRate', 'Alloy', 'Stress', 'Damage']  # Unique PMD heads [file:4]

def calculate_metrics(df_subset, gold_list):
    """Calculate Precision, Recall, F1 with correct unique-gold-term recall."""
    if len(df_subset) == 0:
        return 0.0, 0.0, 0.0

    # Recall: Fraction of Gold terms that are covered by at least one retrieved item
    refs = df_subset['reference'].astype(str).unique()
    covered_gold = set()
    for g in gold_list:
        # Check if gold term is contained in any retrieved reference
        # Case insensitive match
        g_lower = g.lower()
        for r in refs:
            if g_lower in r.lower():
                covered_gold.add(g)
                break
    rec = len(covered_gold) / len(gold_list)

    # Precision: Fraction of retrieved items that are relevant (match any gold term)
    # Note: Using simple string containment
    hits_mask = df_subset['reference'].str.contains('|'.join(gold_list), case=False, na=False)
    prec = hits_mask.mean()

    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1

# Variable thresholds evaluation
thresholds = [0.85, 0.75, 0.65]
results = []
for thresh in thresholds:
    subset = df[df.similarity >= thresh]
    prec, rec, f1 = calculate_metrics(subset, gold)
    results.append({'thresh': thresh, 'prec': prec, 'rec': rec, 'f1': f1})

# LLM Boost: Confirmed ambiguous → count as hit
if 'llm_confirmed' in df.columns:
    # Union of Strict (>0.85) AND LLM-Confirmed
    combined_mask = (df.similarity >= 0.85) | (df.llm_confirmed == 1)
    subset_llm = df[combined_mask]
    
    prec_llm, rec_llm, f1_llm = calculate_metrics(subset_llm, gold)
    print(f"+LLM Confirmed: P={prec_llm:.2f} R={rec_llm:.2f} F1={f1_llm:.2f}")

df_results = pd.DataFrame(results)
print("\nVariable Threshold Results:")
print(df_results)
df_results.to_csv(os.path.join(PROCESSED_DIR, 'f1_ablation.csv'), index=False)
