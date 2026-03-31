import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
df = pd.read_csv(os.path.join(PROCESSED_DIR, 'ontology_alignment_llm.csv'))

gold_old = ['Microstructure', 'Crack', 'Dislocation', 'GrainBoundary', 'FatigueStrength', 'Strain', 'Alloy']
gold_new = ['Microstructure', 'Crack', 'Dislocation', 'GrainBoundary', 'FatigueStrength', 'StrainRate', 'Alloy', 'Stress', 'Damage']

def check(name, gold):
    hits = df[df.similarity >= 0.85]['reference'].str.contains('|'.join(gold), case=False, na=False)
    hit_terms = df[df.similarity >= 0.85][hits]['reference'].unique()
    prec = hits.mean()
    rec = hits.sum() / len(gold)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    print(f"[{name}] F1: {f1:.2f} (P={prec:.2f} R={rec:.2f})")
    print(f"  Hits: {sorted(list(hit_terms))}")
    print(f"  Misses: {sorted(list(set(gold) - set(hit_terms)))}")

print("--- Comparing Gold Standards (Strict 0.85) ---")
check("OLD Gold", gold_old)
check("NEW Gold", gold_new)
