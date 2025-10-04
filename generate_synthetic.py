import os
import numpy as np
import pandas as pd

OUTPUT_DIR = "dcrm_ai_diagnostics/data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_synthetic_dcrm(file_name, n_points=200, fault=False):
    time = np.linspace(0, 1, n_points)
    # baseline sinusoidal resistance
    resistance = 100 + 5 * np.sin(2 * np.pi * 5 * time)
    
    # add noise
    resistance += np.random.normal(0, 1, size=n_points)
    
    # faulty signal has bigger drops/spikes
    if fault:
        resistance[50:70] -= np.linspace(0, 20, 20)
        resistance[120:140] += np.linspace(0, 15, 20)
    
    df = pd.DataFrame({
        "time": time,
        "resistance": resistance,
        "resistance_smooth": pd.Series(resistance).rolling(5, min_periods=1).mean()
    })
    df.to_csv(os.path.join(OUTPUT_DIR, file_name), index=False)

# Generate multiple files
for i in range(10):
    generate_synthetic_dcrm(f"healthy_{i}.csv", fault=False)
for i in range(10):
    generate_synthetic_dcrm(f"faulty_{i}.csv", fault=True)

print("✅ Synthetic DCRM data generated in", OUTPUT_DIR)



