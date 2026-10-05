"""
run_pipeline.py - Ontology Learning Pipeline Runner

Execution order:
  1. builder.py            - Parse CoNLL BIO tags, build co-occurrence graph
  2. gnn.py                - Train GraphSAGE, produce GNN embeddings
  3. cluster.py            - Cluster nodes (uses GNN embeddings)
  4. taxonomy.py           - Build taxonomy from clusters
  5. align.py              - Type terms + align to the reference ontology
  6. rag_typing.py         - LLM re-ranking of low-confidence types (optional)
  7. relations.py          - Extract relations (patterns + dep parse + GNN + LLM)
  8. relations_axioms.py   - OWL axiom pruning of inconsistent relations
  9. eval_f1.py            - Leakage-free typing / alignment / clustering /
                             taxonomy evaluation
 10. eval_relations.py     - Class-level relation evaluation

Every stage runs as its own process with the current environment (see
config.py for the switches) and PYTHONHASHSEED fixed, so runs are
reproducible. Stale outputs in PROCESSED_DIR are removed before a full run
(LLM response caches are kept).

Usage:
  python run_pipeline.py                     # full run
  python run_pipeline.py --from align.py     # re-run from a stage
"""

import argparse
import os
import subprocess
import sys

SCRIPTS = [
    "builder.py",           # 1. Build graph from CoNLL BIO tags
    "gnn.py",               # 2. Train GraphSAGE embeddings
    "cluster.py",           # 3. Cluster nodes
    "taxonomy.py",          # 4. Build taxonomy from clusters
    "align.py",             # 5. Type terms / align to reference ontology
    "rag_typing.py",        # 6. LLM re-ranking of low-confidence types
    "relations.py",         # 7. Extract typed relations (+ LLM classification)
    "relations_axioms.py",  # 8. OWL axiom pruning
    "eval_f1.py",           # 9. Leakage-free evaluation
    "eval_relations.py",    # 10. Class-level relation evaluation
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEEP_PREFIXES = ('llm_',)   # LLM response caches survive a clean run
KEEP_DIRS = ('ablations',)


def stage_env(extra=None):
    """Environment for a stage: current env + fixed hash seed + overrides."""
    env = dict(os.environ)
    env.setdefault('PYTHONHASHSEED', '0')
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def clean_outputs(processed_dir):
    """Remove previous outputs so no stale file can be read by a later stage."""
    if not os.path.isdir(processed_dir):
        return
    removed = 0
    for name in os.listdir(processed_dir):
        path = os.path.join(processed_dir, name)
        if os.path.isfile(path) and not name.startswith(KEEP_PREFIXES):
            os.remove(path)
            removed += 1
        elif os.path.isdir(path) and name not in KEEP_DIRS:
            pass  # leave other directories alone
    if removed:
        print(f"Removed {removed} stale output files from {processed_dir}")


def run_script(script_name, env=None):
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {script_name}")
    print(f"{'=' * 60}", flush=True)
    try:
        subprocess.run(
            [sys.executable, script_name],
            check=True,
            cwd=SCRIPT_DIR,
            env=env or stage_env(),
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running {script_name}: exit code {e.returncode}")
        return False


def run_stages(scripts, env=None):
    """Run stages in order; returns False at the first failure."""
    for script in scripts:
        if not run_script(script, env):
            print(f"\nPipeline stopped due to error in {script}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Run the ontology learning pipeline')
    parser.add_argument('--from', dest='from_stage', default=SCRIPTS[0],
                        choices=SCRIPTS, help='first stage to run')
    args = parser.parse_args()

    sys.path.insert(0, SCRIPT_DIR)
    from config import PROCESSED_DIR

    os.chdir(SCRIPT_DIR)
    stages = SCRIPTS[SCRIPTS.index(args.from_stage):]
    print(f"Working Directory: {os.getcwd()}")
    print(f"Output directory: {PROCESSED_DIR}")
    print(f"Pipeline stages: {' -> '.join(s.replace('.py', '') for s in stages)}")
    if args.from_stage == SCRIPTS[0]:
        clean_outputs(PROCESSED_DIR)

    if not run_stages(stages):
        sys.exit(1)
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
