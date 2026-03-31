import subprocess
import sys
import os

SCRIPTS = [
    "builder.py",
    "cluster.py",
    "gnn.py",  # Added GNN training step
    "taxonomy.py",
    "align.py",
    "relations.py",
    "eval_f1.py"
]

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    try:
        # Run using the same python interpreter
        result = subprocess.run([sys.executable, script_name], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}: {e}")
        return False

if __name__ == "__main__":
    # Ensure we are in the correct directory (src/graph)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working Directory: {os.getcwd()}")
    
    for script in SCRIPTS:
        success = run_script(script)
        if not success:
            print(f"\n⛔ Pipeline stopped due to error in {script}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
