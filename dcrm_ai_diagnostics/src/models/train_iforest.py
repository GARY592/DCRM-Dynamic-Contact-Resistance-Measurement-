from pathlib import Path
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib


PROJECT_ROOT = Path("dcrm_ai_diagnostics")
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features_extracted.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "iforest.pkl"


if not FEATURES_CSV.exists():
    raise FileNotFoundError(f"Features file not found at {FEATURES_CSV}. Run the pipeline first.")

df = pd.read_csv(FEATURES_CSV)
if df.empty:
    raise ValueError("Features CSV is empty. Ensure feature extraction produced rows.")

# Use all numeric features (drop non-numeric like file)
X = df.drop(columns=[c for c in df.columns if c in ("file", "label")])

# Train IsolationForest (unsupervised)
clf = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
clf.fit(X)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print("IsolationForest model saved to", MODEL_PATH)


