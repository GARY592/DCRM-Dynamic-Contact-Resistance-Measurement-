from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from dcrm_ai_diagnostics.src.models.infer import predict_from_csv_bytes, load_model, load_iforest, predict_with_anomaly_from_df
import pandas as pd
import io


app = FastAPI(title="DCRM Inference API", version="0.1.0")
_model = None


@app.on_event("startup")
def _load():
    global _model
    try:
        _model = load_model()
    except FileNotFoundError:
        _model = None
    # Load IsolationForest if available
    global _iforest
    try:
        _iforest = load_iforest()
    except Exception:
        _iforest = None


@app.get("/health")
def health():
    status = "ok" if _model is not None else "model_not_loaded"
    return {"status": status}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        result = predict_with_anomaly_from_df(df, model=_model, iforest=_iforest)
        res = {"prediction": int(result["prediction"]) }
        if result["probabilities"] is not None:
            res["probabilities"] = [float(x) for x in result["probabilities"][0]]
        if result["anomaly"] is not None:
            res["anomaly"] = bool(result["anomaly"])
            res["anomaly_score"] = float(result["anomaly_score"]) if result["anomaly_score"] is not None else None
        if result.get("top_features") is not None:
            res["top_features"] = result["top_features"]
        if result.get("component_insights") is not None:
            res["component_insights"] = result["component_insights"]
        if result.get("maintenance_recommendations") is not None:
            res["maintenance_recommendations"] = result["maintenance_recommendations"]
        if result.get("health_score") is not None:
            res["health_score"] = int(result["health_score"])
        return JSONResponse(res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


