"""
run_ablation.py - Automated Ablation Experiment Runner

Runs the alignment + evaluation pipeline under different ablation
configurations and collects comparative results.

Usage:
    python run_ablation.py                          # Run all presets
    python run_ablation.py --preset full_pipeline   # Run one preset
    python run_ablation.py --list                   # List presets
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ABLATION_PRESETS, get_preset, list_presets

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ABLATION_DIR = os.path.join(PROCESSED_DIR, "ablation_results")


def run_ablation(preset_name):
    """Run alignment and evaluation with a specific ablation preset."""
    preset = get_preset(preset_name)
    print(f"\n{'=' * 60}")
    print(f"ABLATION: {preset_name}")
    print(f"  {preset['description']}")
    print(f"{'=' * 60}")

    # Set ablation flags by modifying align.py module globals
    import align
    align.USE_GNN_EMBEDDINGS = preset['USE_GNN_EMBEDDINGS']
    align.USE_BIDIRECTIONAL = preset['USE_BIDIRECTIONAL']
    align.USE_COMBINED_SCORING = preset['USE_COMBINED_SCORING']

    # Update embedder model if specified in preset
    from config import DEFAULT_EMBEDDER_MODEL
    embedder_model = preset.get('embedder_model', DEFAULT_EMBEDDER_MODEL)
    if embedder_model != align.EMBEDDER_MODEL:
        from sentence_transformers import SentenceTransformer
        print(f"  Switching embedder to: {embedder_model}")
        align.EMBEDDER_MODEL = embedder_model
        align.EMBEDDER = SentenceTransformer(embedder_model)

    # Run alignment
    start_time = time.time()
    try:
        align.run_alignment()
    except Exception as e:
        print(f"  Alignment failed: {e}")
        return None

    # Run evaluation
    import eval_f1
    if embedder_model != eval_f1.EMBEDDER_MODEL:
        from sentence_transformers import SentenceTransformer
        eval_f1.EMBEDDER_MODEL = embedder_model
        eval_f1.EMBEDDER = SentenceTransformer(embedder_model)
    try:
        eval_f1.run_evaluation()
    except Exception as e:
        print(f"  Evaluation failed: {e}")
        return None

    elapsed = time.time() - start_time

    # Collect results
    results_path = os.path.join(PROCESSED_DIR, 'eval_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {}

    results['preset'] = preset_name
    results['config'] = {k: v for k, v in preset.items()
                         if k.startswith('USE_')}
    results['elapsed_seconds'] = round(elapsed, 2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument('--preset', type=str, default=None,
                        help='Run a specific preset')
    parser.add_argument('--list', action='store_true',
                        help='List available presets')
    args = parser.parse_args()

    if args.list:
        print("Available ablation presets:")
        list_presets()
        return

    os.makedirs(ABLATION_DIR, exist_ok=True)

    if args.preset:
        presets_to_run = [args.preset]
    else:
        presets_to_run = list(ABLATION_PRESETS.keys())

    all_results = []
    for preset_name in presets_to_run:
        result = run_ablation(preset_name)
        if result:
            # Save individual result
            result_path = os.path.join(
                ABLATION_DIR, f'{preset_name}_results.json'
            )
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            all_results.append(result)
            print(f"  Saved: {result_path}")

    # Summary comparison
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("ABLATION SUMMARY")
        print(f"{'=' * 60}")

        rows = []
        for r in all_results:
            preset = r.get('preset', '?')
            for level in ['type_level', 'term_level', 'concept_level']:
                level_data = r.get(level, {})
                if 'thresholds' in level_data:
                    # Use threshold 0.65 as representative
                    metrics = level_data['thresholds'].get(
                        0.65, level_data['thresholds'].get('0.65', {})
                    )
                    if metrics:
                        rows.append({
                            'preset': preset,
                            'level': level.replace('_level', ''),
                            'P@0.65': metrics.get('precision', 0),
                            'R@0.65': metrics.get('recall', 0),
                            'F1@0.65': metrics.get('f1', 0),
                        })

        if rows:
            summary_df = pd.DataFrame(rows)
            print(summary_df.to_string(index=False))
            summary_path = os.path.join(ABLATION_DIR, 'ablation_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSaved: {summary_path}")


if __name__ == '__main__':
    main()
