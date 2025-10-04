import subprocess
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path("dcrm_ai_diagnostics")
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FEATURES_FILE = DATA_PROCESSED / "features_extracted.csv"

def run_step(script, desc):
    print(f"\n🚀 Running: {desc}")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error in {desc}")
        print(result.stderr)
        sys.exit(1)
    else:
        print(result.stdout)

if __name__ == "__main__":
    # Step 0: Optionally generate multiple raw samples for a richer dataset
    run_step(Path("create_structure.py"), "Generate multiple raw CSV samples")

    # Step 1: Generate a single synthetic raw DCRM sample (kept for compatibility)
    run_step(PROJECT_ROOT / "src" / "utils" / "generate_synthetic.py", "Synthetic Data Generation")

    # Step 2: Preprocess raw data into processed (smoothed) signals
    run_step(PROJECT_ROOT / "src" / "preprocessing" / "preprocess.py", "Preprocessing")

    # Step 3: Extract features from processed signals
    run_step(PROJECT_ROOT / "src" / "features" / "feature_extractor.py", "Feature Extraction")

    print("\nPipeline completed successfully!")
    print(f"Features saved at: {FEATURES_FILE}")
