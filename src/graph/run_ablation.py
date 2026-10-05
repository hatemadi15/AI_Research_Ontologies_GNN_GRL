"""
run_ablation.py - Environment-driven Ablation Runner

Every preset in config.ABLATION_PRESETS is a set of environment overrides
plus the first stage that has to be re-run. The runner

  1. runs the base configuration ('full_pipeline') into
     data/processed/ablations/full_pipeline,
  2. for every other preset copies the base outputs into
     data/processed/ablations/<preset>/ and re-runs the pipeline from the
     preset's first affected stage with PROCESSED_DIR pointing there,
  3. collects the leakage-free metrics (eval_results.json,
     relation_eval_results.json) into ablation_summary.csv/.json.

All presets run with the LLM features off unless --llm is given, so the
comparison is deterministic and free. Presets marked oracle use gold types
and are upper bounds, not model results.

Usage:
    python run_ablation.py                          # all presets except the
                                                    # embedder presets
    python run_ablation.py --preset no_gnn          # one preset (+ base)
    python run_ablation.py --list                   # list presets
"""

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ABLATION_PRESETS, DEFAULT_PROCESSED_DIR, get_preset, list_presets  # noqa: E402
from run_pipeline import SCRIPTS, run_stages, stage_env  # noqa: E402

ABLATION_DIR = os.path.join(DEFAULT_PROCESSED_DIR, "ablations")
BASE_PRESET = 'full_pipeline'
# Downloading/encoding with large embedders is slow; run them explicitly
DEFAULT_SKIP = {'domain_specter', 'domain_scibert'}
LLM_OFF = {'LLM_ALIGNMENT': 'false', 'LLM_RELATIONS': 'false',
           'RAG_TYPING': 'false', 'LLM_MODE': 'false'}


def preset_dir(name):
    return os.path.join(ABLATION_DIR, name)


def copy_base_outputs(src, dst):
    """Copy the base run's files (not its evaluation results) to dst."""
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        path = os.path.join(src, fn)
        if os.path.isfile(path) and not fn.startswith(('eval_', 'relation_eval')):
            shutil.copy2(path, os.path.join(dst, fn))


def run_preset(name, llm=False):
    """Run one preset; returns its summary row (or None on failure)."""
    preset = get_preset(name)
    out_dir = preset_dir(name)
    env_over = dict(preset['env'])
    if not llm:
        env_over.update(LLM_OFF)
    env_over['PROCESSED_DIR'] = out_dir

    from_stage = preset['from_stage']
    if name != BASE_PRESET and from_stage != SCRIPTS[0]:
        if not os.path.exists(os.path.join(preset_dir(BASE_PRESET), 'eval_results.json')):
            print(f"Base run missing; running '{BASE_PRESET}' first")
            if run_preset(BASE_PRESET, llm) is None:
                return None
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        copy_base_outputs(preset_dir(BASE_PRESET), out_dir)
    else:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    print(f"\n{'#' * 70}\nABLATION: {name} - {preset['description']}\n"
          f"  overrides: {preset['env'] or '(defaults)'}; from {from_stage}\n{'#' * 70}")
    start = time.time()
    stages = SCRIPTS[SCRIPTS.index(from_stage):]
    if not run_stages(stages, stage_env(env_over)):
        return None
    return summarize(name, out_dir, time.time() - start)


def summarize(name, out_dir, elapsed):
    with open(os.path.join(out_dir, 'eval_results.json')) as f:
        ev = json.load(f)
    rel = {}
    rel_path = os.path.join(out_dir, 'relation_eval_results.json')
    if os.path.exists(rel_path):
        with open(rel_path) as f:
            rel = json.load(f)
    ty = ev['typing']
    cl = ev.get('clustering', {})
    tx = ev.get('taxonomy', {})
    al = ev.get('alignment_rows', {}).get('thresholds', {}).get('0.5', {})
    return {
        'preset': name,
        'oracle': ev['oracle'],
        'typing_accuracy': ty['accuracy'],
        'typing_accuracy_ci95': ty['accuracy_ci95'],
        'typing_set_f1': ty['set_f1'],
        'typing_macro_f1': ty['macro_f1'],
        'typing_acc@5': ty['acc@5'],
        'typing_h_f1': ty['h_f1'],
        'typing_aurc': ty['aurc'],
        'alignment_pairwise_f1@0.5': al.get('f1'),
        'clustering_ari': cl.get('ari'),
        'clustering_bcubed_f1': cl.get('bcubed_f1'),
        'n_clusters': cl.get('n_clusters'),
        'taxonomy_f1': tx.get('f1'),
        'taxonomy_precision_closure': tx.get('precision_closure'),
        'relations': rel.get('n_relations'),
        'relations_class_f1': rel.get('predicted_typing', {}).get('f1'),
        'runtime_sec': round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument('--preset', action='append', default=None,
                        help='preset(s) to run (repeatable); default: all')
    parser.add_argument('--llm', action='store_true',
                        help='keep the LLM features on (costs API calls)')
    parser.add_argument('--list', action='store_true', help='list presets')
    args = parser.parse_args()

    if args.list:
        print("Available ablation presets:")
        list_presets()
        return

    names = args.preset or [n for n in ABLATION_PRESETS if n not in DEFAULT_SKIP]
    if BASE_PRESET not in names:
        names = [BASE_PRESET] + names
    names = [BASE_PRESET] + [n for n in names if n != BASE_PRESET]

    rows = []
    for name in names:
        row = run_preset(name, llm=args.llm)
        if row is None:
            print(f"  Preset {name} failed")
            continue
        rows.append(row)

    if not rows:
        return
    import pandas as pd
    os.makedirs(ABLATION_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ABLATION_DIR, 'ablation_summary.csv'), index=False)
    with open(os.path.join(ABLATION_DIR, 'ablation_summary.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    cols = ['preset', 'oracle', 'typing_accuracy', 'typing_set_f1', 'typing_macro_f1',
            'typing_h_f1', 'clustering_ari', 'taxonomy_f1', 'relations_class_f1']
    print("\n" + "=" * 100)
    print("ABLATION SUMMARY (leakage-free metrics; oracle rows are upper bounds)")
    print("=" * 100)
    print(df[cols].to_string(index=False))
    print(f"\nSaved: {os.path.join(ABLATION_DIR, 'ablation_summary.csv')}")


if __name__ == '__main__':
    main()
