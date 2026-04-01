"""
run_pipeline.py - Ontology Learning Pipeline Runner

Correct execution order:
  1. builder.py            - Parse CoNLL BIO tags, build co-occurrence graph
  2. gnn.py                - Train GraphSAGE, produce GNN embeddings
  3. cluster.py            - Cluster nodes (uses GNN embeddings)
  4. taxonomy.py           - Build taxonomy from clusters
  5. align.py              - Align to reference ontology (+ LLM augmentation & expansion)
  6. rag_typing.py         - RAG-based term typing (optional, LLM-powered)
  7. eval_f1.py            - Multi-level evaluation (type/term/concept)
  8. relations.py          - Extract relations using patterns + dep parse + GNN + LLM
  9. relations_axioms.py   - OWL axiom pruning of inconsistent relations
"""

import subprocess
import sys
import os

SCRIPTS = [
    "builder.py",           # 1. Build graph from CoNLL BIO tags
    "gnn.py",               # 2. Train GraphSAGE embeddings
    "cluster.py",           # 3. Cluster nodes (uses GNN embeddings)
    "taxonomy.py",          # 4. Build taxonomy from clusters
    "align.py",             # 5. Align to reference ontology (+ LLM augmentation & expansion)
    "rag_typing.py",        # 6. RAG-based term typing (optional, LLM-powered)
    "eval_f1.py",           # 7. Evaluate alignment quality
    "relations.py",         # 8. Extract typed relations (+ LLM classification)
    "relations_axioms.py",  # 9. OWL axiom pruning
]


def run_script(script_name):
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {script_name}")
    print(f"{'=' * 60}")
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\nError running {script_name}: exit code {e.returncode}")
        return False


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working Directory: {os.getcwd()}")
    print(f"Pipeline stages: {' -> '.join(s.replace('.py', '') for s in SCRIPTS)}")

    for script in SCRIPTS:
        success = run_script(script)
        if not success:
            print(f"\nPipeline stopped due to error in {script}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'=' * 60}")
