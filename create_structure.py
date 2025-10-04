import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("dcrm_ai_diagnostics/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

for i in range(10):
    OUT = RAW_DIR / f"sample_dcrm_{i}.csv"
    t = np.linspace(0, 1, 500)
    resistance = 100 + 5*np.sin(2*np.pi*3*t) - 20*np.exp(-((t-0.4)/0.02)**2)
    resistance += np.random.normal(0, 0.5, size=t.shape)
    df = pd.DataFrame({"time": t, "resistance": resistance})
    df.to_csv(OUT, index=False)
    print("Wrote sample to", OUT)

