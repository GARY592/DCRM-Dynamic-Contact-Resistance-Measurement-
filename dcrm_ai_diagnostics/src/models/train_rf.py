import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np

# Paths
PROJECT_ROOT = Path("dcrm_ai_diagnostics")
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features_extracted.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.pkl"

# Load features
if not FEATURES_CSV.exists():
    raise FileNotFoundError(f"Features file not found at {FEATURES_CSV}. Run the pipeline first.")

df = pd.read_csv(FEATURES_CSV)
if df.empty:
    raise ValueError("Features CSV is empty. Ensure feature extraction produced rows.")

# For demo, simulate labels: 0 = healthy, 1 = worn contact, 2 = misaligned mech
np.random.seed(42)
df["label"] = np.random.choice([0,1,2], size=len(df))

# Split features and labels
X = df.drop(columns=["file", "label"]) if "file" in df.columns else df.drop(columns=["label"]) 
y = df["label"]

n_samples = len(X)
if n_samples < 2:
    # Only one sample — train on all and skip evaluation
    X_train, X_test, y_train, y_test = X, None, y, None
elif n_samples < 4:
    # Very small dataset — ensure at least one test sample using integer test_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
if X_test is not None:
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
else:
    print("Warning: Too few samples to create a test split; trained on all data.")

# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print("Random Forest model saved to", MODEL_PATH)
