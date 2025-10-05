DCRM AI Diagnostics
====================

AI-based Dynamic Contact Resistance Measurement (DCRM) analysis for EHV circuit breakers.

Features
--------
- FastAPI inference API with file and batch ZIP upload
- Streamlit dashboard: waveform plot, predictions, anomaly, SHAP, component insights, health score, RUL, history & trends, PDF export
- Models: RandomForest (supervised), IsolationForest (unsupervised), Autoencoder/LSTM/CNN training scripts
- SQLite logging with history and trends

Setup (Windows)
---------------
```powershell
.\\venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python .\\generate_synthetic.py
python .\\dcrm_ai_diagnostics\\src\\preprocessing\\preprocess.py
python .\\dcrm_ai_diagnostics\\src\\features\\feature_extractor.py
python .\\dcrm_ai_diagnostics\\src\\models\\train_rf.py
python .\\dcrm_ai_diagnostics\\src\\models\\train_iforest.py
```

Run
---
- API:
```powershell
uvicorn dcrm_ai_diagnostics.src.api.main:app --reload
```
- Dashboard:
```powershell
streamlit run .\\dcrm_ai_diagnostics\\src\\dashboard\\app.py
```

Env flags
---------
- `USE_IFOREST=true|false`
- `USE_AUTOENCODER=true|false`
- `AE_ANOMALY_THRESHOLD=0.01` (tune per data)

Batch API
--------
- `POST /batch-predict` with `multipart/form-data` file field containing a `.zip` of CSVs.
- `GET /history?limit=50` returns recent analysis entries.

Docker (local)
--------------
Build images:
```bash
docker build -t dcrm-api -f Dockerfile.api .
docker build -t dcrm-dashboard -f Dockerfile.dashboard .
```
Run:
```bash
docker run -p 8000:8000 -e USE_IFOREST=true -e USE_AUTOENCODER=false dcrm-api
```
```bash
docker run -p 8501:8501 dcrm-dashboard
```

Free deployment options
-----------------------
- API: Render free tier, Railway free tier, or Fly.io. Use `Dockerfile.api`.
- Dashboard: Streamlit Community Cloud (free) using `requirements.txt`.
- Database: SQLite file checked out with the app (ephemeral) or use free Postgres on Render/Railway (requires code change).

Roadmap
-------
- Replace synthetic labels with expert labels; calibrate thresholds
- Integrate deep models in live API behind env toggles (AE already integrated for anomaly)
- Add CI (GitHub Actions) and container registry


