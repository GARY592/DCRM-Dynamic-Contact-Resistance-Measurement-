from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Paths
RAW_DIR = Path("dcrm_ai_diagnostics/data/raw")
PROCESSED_DIR = Path("dcrm_ai_diagnostics/data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Function to smooth a signal
def smooth_signal(y, window=31, poly=3):
    if len(y) < 5:
        return y
    if window >= len(y):
        window = len(y)-1 if (len(y)-1)%2==1 else len(y)-2
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window, poly)

# Process a single file
def process_file(pth):
    df = pd.read_csv(pth)
    df["resistance_smooth"] = smooth_signal(df["resistance"].values)
    out = PROCESSED_DIR / pth.name
    df.to_csv(out, index=False)
    print("Processed ->", out)

# Process all CSVs in raw folder
if __name__ == "__main__":
    for f in RAW_DIR.glob("*.csv"):
        process_file(f)
