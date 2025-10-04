import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from scipy.integrate import simpson

# Paths
PROJECT_ROOT = Path("dcrm_ai_diagnostics")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_CSV = PROCESSED_DIR / "features_extracted.csv"

# Function to extract features from a single DataFrame
def extract_from_df(df):
    # Safely get the signal
    y = df.get("resistance_smooth")
    if y is None:
        y = df.get("resistance")
    if y is None:
        raise ValueError(f"No 'resistance' or 'resistance_smooth' column found in {df.columns}")
    y = y.values

    # Time vector
    time = df["time"].values if "time" in df.columns else np.arange(len(y))

    # Feature dictionary
    feats = {}
    feats["mean"] = float(np.mean(y))
    feats["std"] = float(np.std(y))
    feats["max"] = float(np.max(y))
    feats["min"] = float(np.min(y))
    feats["range"] = feats["max"] - feats["min"]
    feats["area"] = float(simpson(y, time))

    # Linear trend (slope)
    coef = np.polyfit(time, y, 1)
    feats["slope"] = float(coef[0])

    # Peaks count
    peaks, _ = find_peaks(y, height=np.mean(y) + np.std(y)*0.5)
    feats["peaks_count"] = int(len(peaks))

    return pd.DataFrame([feats])

# Process all processed CSV files
if __name__ == "__main__":
    rows = []
    for f in PROCESSED_DIR.glob("*.csv"):
        if f.name == "features_extracted.csv":   # <-- skip output file
            continue
        df = pd.read_csv(f)
        feats = extract_from_df(df)
        feats["file"] = f.name
        rows.append(feats)
    
    if rows:
        df_all = pd.concat(rows, ignore_index=True)
        df_all.to_csv(FEATURES_CSV, index=False)
        print("Wrote features to:", FEATURES_CSV)
    else:
        print("No processed files found.")


