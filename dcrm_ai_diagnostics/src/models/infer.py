import io
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import os

import joblib
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.signal import find_peaks, savgol_filter
import shap
from dcrm_ai_diagnostics.src.utils.component_mapping import get_component_insights, get_maintenance_recommendations, get_health_score
from dcrm_ai_diagnostics.src.utils.database import log_analysis
from dcrm_ai_diagnostics.src.utils.rul_estimator import estimate_rul_from_analysis
import torch
from dcrm_ai_diagnostics.src.models.train_autoencoder import DCRMAutoencoder


PROJECT_ROOT = Path("dcrm_ai_diagnostics")
MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.pkl"
IFOREST_PATH = PROJECT_ROOT / "models" / "iforest.pkl"
AE_MODEL_PATH = PROJECT_ROOT / "models" / "autoencoder_model.pkl"
AE_SCALER_PATH = PROJECT_ROOT / "models" / "autoencoder_scaler.pkl"


def smooth_signal(values: np.ndarray, window: int = 31, poly: int = 3) -> np.ndarray:
    if len(values) < 5:
        return values
    if window >= len(values):
        window = len(values) - 1 if (len(values) - 1) % 2 == 1 else len(values) - 2
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    return savgol_filter(values, window, poly)


def extract_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    y = df.get("resistance_smooth")
    if y is None:
        y = df.get("resistance")
    if y is None:
        raise ValueError(f"Input must have 'resistance' or 'resistance_smooth' column. Got: {list(df.columns)}")
    y = y.values

    time = df["time"].values if "time" in df.columns else np.arange(len(y))

    feats: Dict[str, Any] = {}
    feats["mean"] = float(np.mean(y))
    feats["std"] = float(np.std(y))
    feats["max"] = float(np.max(y))
    feats["min"] = float(np.min(y))
    feats["range"] = feats["max"] - feats["min"]
    feats["area"] = float(simpson(y, time))

    coef = np.polyfit(time, y, 1)
    feats["slope"] = float(coef[0])

    peaks, _ = find_peaks(y, height=np.mean(y) + np.std(y) * 0.5)
    feats["peaks_count"] = int(len(peaks))

    return pd.DataFrame([feats])


def ensure_smoothed(df: pd.DataFrame) -> pd.DataFrame:
    if "resistance_smooth" not in df.columns and "resistance" in df.columns:
        df = df.copy()
        df["resistance_smooth"] = smooth_signal(df["resistance"].values)
    return df


def load_model(model_path: Optional[Path] = None):
    path = model_path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Train the model first.")
    return joblib.load(path)


def load_iforest() -> Optional[any]:
    if IFOREST_PATH.exists():
        return joblib.load(IFOREST_PATH)
    return None


def load_autoencoder() -> Tuple[Optional[Any], Optional[Any]]:
    """Load autoencoder model and scaler if available."""
    try:
        if AE_MODEL_PATH.exists() and AE_SCALER_PATH.exists():
            scaler = joblib.load(AE_SCALER_PATH)
            # Infer input size from scaler/feats dimension at runtime
            # We'll instantiate with a safe default and adjust at use time if needed
            return ("placeholder", scaler)
    except Exception:
        pass
    return (None, None)


def predict_from_df(df: pd.DataFrame, model=None) -> Tuple[int, Optional[np.ndarray]]:
    df = ensure_smoothed(df)
    feats = extract_features_from_df(df)
    model = model or load_model()
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feats.values)
    pred = int(model.predict(feats.values)[0])
    return pred, proba


def predict_with_anomaly_from_df(df: pd.DataFrame, model=None, iforest=None):
    df = ensure_smoothed(df)
    feats = extract_features_from_df(df)
    model = model or load_model()
    iforest = iforest if iforest is not None else load_iforest()
    use_autoencoder = os.environ.get("USE_AUTOENCODER", "false").lower() == "true"
    use_iforest = os.environ.get("USE_IFOREST", "true").lower() == "true"

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feats.values)
    pred = int(model.predict(feats.values)[0])

    anomaly = None
    anomaly_score = None
    # IsolationForest anomaly
    if use_iforest and iforest is not None:
        anomaly_score = float(iforest.score_samples(feats.values)[0])
        is_inlier = int(iforest.predict(feats.values)[0])
        anomaly = (is_inlier == -1)

    # Autoencoder anomaly (reconstruction error on engineered features)
    ae_score = None
    if use_autoencoder and AE_MODEL_PATH.exists() and AE_SCALER_PATH.exists():
        try:
            scaler = joblib.load(AE_SCALER_PATH)
            X = scaler.transform(feats.values)
            input_size = X.shape[1]
            model_ae = DCRMAutoencoder(input_size=input_size, encoding_dim=min(32, input_size // 2))
            state_dict = torch.load(AE_MODEL_PATH, map_location="cpu")
            model_ae.load_state_dict(state_dict)
            model_ae.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(X)
                recon = model_ae(x_tensor)
                mse = torch.mean((x_tensor - recon) ** 2, dim=1)
                ae_score = float(mse.numpy()[0])
            # Simple threshold; can be tuned via env
            ae_thresh = float(os.environ.get("AE_ANOMALY_THRESHOLD", "0.01"))
            ae_is_anom = ae_score > ae_thresh
            anomaly = bool(anomaly or ae_is_anom) if anomaly is not None else ae_is_anom
        except Exception:
            ae_score = None

    # SHAP explanation (best effort)
    top_features = None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feats)
        feature_names = list(feats.columns)
        # Handle binary/multiclass: pick the predicted class vector
        if isinstance(shap_values, list):
            shap_vec = shap_values[pred][0]
        else:
            shap_vec = shap_values[0]
        pairs = list(zip(feature_names, shap_vec))
        # Sort by absolute contribution and take top 5
        pairs.sort(key=lambda x: abs(float(x[1])), reverse=True)
        top5 = pairs[:5]
        top_features = [{"name": n, "shap_value": float(v)} for n, v in top5]
    except Exception:
        top_features = None

    # Component-level analysis
    features_dict = {col: float(val) for col, val in zip(feats.columns, feats.values[0])}
    component_insights = get_component_insights(features_dict, pred)
    maintenance_recommendations = get_maintenance_recommendations(component_insights, pred, anomaly)
    health_score = get_health_score(component_insights)

    # RUL estimation
    analysis_result = {
        "prediction": pred,
        "probabilities": proba,
        "anomaly": anomaly,
        "anomaly_score": anomaly_score,
        "ae_score": ae_score,
        "top_features": top_features,
        "component_insights": component_insights,
        "maintenance_recommendations": maintenance_recommendations,
        "health_score": health_score,
    }
    
    rul_estimation = estimate_rul_from_analysis(analysis_result)

    # Log analysis to database
    label_map = {0: "Healthy", 1: "Worn Arcing Contact", 2: "Misaligned Mechanism"}
    try:
        analysis_id = log_analysis(
            file_name="uploaded_file.csv",  # Will be updated by caller
            file_source="upload",
            prediction=pred,
            prediction_label=label_map.get(pred, str(pred)),
            anomaly=anomaly,
            anomaly_score=anomaly_score,
            health_score=health_score,
            component_insights=component_insights,
            top_features=top_features,
            maintenance_recommendations=maintenance_recommendations
        )
    except Exception:
        analysis_id = None

    return {
        "prediction": pred,
        "probabilities": proba,
        "anomaly": anomaly,
        "anomaly_score": anomaly_score,
        "top_features": top_features,
        "component_insights": component_insights,
        "maintenance_recommendations": maintenance_recommendations,
        "health_score": health_score,
        "rul_estimation": rul_estimation,
        "analysis_id": analysis_id,
    }


def predict_from_csv_bytes(content: bytes, model=None) -> Tuple[int, Optional[np.ndarray]]:
    df = pd.read_csv(io.BytesIO(content))
    return predict_from_df(df, model=model)


if __name__ == "__main__":
    sample = PROJECT_ROOT / "data" / "processed" / "sample_dcrm.csv"
    if not sample.exists():
        raise SystemExit(f"No sample CSV found at {sample}. Run the pipeline first.")
    df_in = pd.read_csv(sample)
    y, proba = predict_from_df(df_in)
    print("Prediction:", y)
    if proba is not None:
        print("Probabilities:", proba.tolist())


