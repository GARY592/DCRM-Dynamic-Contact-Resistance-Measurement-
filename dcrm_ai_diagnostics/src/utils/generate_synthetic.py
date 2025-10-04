import numpy as np
import pandas as pd
from pathlib import Path

# Output CSV path
OUT = Path("dcrm_ai_diagnostics/data/raw/sample_dcrm.csv")

# Simulate a DCRM waveform
t = np.linspace(0, 1, 500)
resistance = 100 + 5*np.sin(2*np.pi*3*t) - 20*np.exp(-((t-0.4)/0.02)**2)
resistance += np.random.normal(0, 0.5, size=t.shape)

# Save as CSV
df = pd.DataFrame({"time": t, "resistance": resistance})
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print("Wrote sample to", OUT)
