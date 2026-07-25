"""FastAPI service exposing the trained rent price prediction model."""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "latest.pkl"
METRICS_PATH = BASE_DIR / "models" / "latest_metrics.json"

app = FastAPI(title="Morocco Rental Price Prediction API", version="1.0")

_artifact = None
_metrics = None
_explainer = None


def get_artifact():
    global _artifact, _metrics, _explainer
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not trained yet. Run train.py first.")
        _artifact = joblib.load(MODEL_PATH)
        if METRICS_PATH.exists():
            with open(METRICS_PATH, encoding="utf-8") as f:
                _metrics = json.load(f)
        if _artifact["model_name"] == "xgboost":
            _explainer = shap.TreeExplainer(_artifact["pipeline"].named_steps["model"])
    return _artifact


class PredictRequest(BaseModel):
    city: str = Field(..., examples=["Casablanca"])
    neighborhood: Optional[str] = Field(None, examples=["Maarif"])
    property_type: str = Field("appartement", examples=["appartement"])
    source_site: str = Field("agenz", examples=["agenz"])
    surface_m2: Optional[float] = Field(None, examples=[90])
    bedrooms: Optional[int] = Field(None, examples=[2])
    bathrooms: Optional[int] = Field(None, examples=[1])
    furnished: Optional[bool] = Field(None, examples=[False])
    amenities: Optional[list[str]] = Field(None, examples=[["ascenseur", "parking"]])


class PredictResponse(BaseModel):
    predicted_rent_mad: float
    confidence_interval_80pct: list[float]
    model_used: str
    top_drivers: list[dict]
    warning: Optional[str] = None


@app.get("/health")
def health():
    artifact = get_artifact()
    return {
        "status": "ok",
        "model": artifact["model_name"],
        "trained_at": artifact["trained_at"],
        "n_train_samples": artifact["n_train_samples"],
    }


@app.get("/model-info")
def model_info():
    get_artifact()
    if _metrics is None:
        raise HTTPException(status_code=404, detail="No metrics file found.")
    return _metrics


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    artifact = get_artifact()
    pipeline = artifact["pipeline"]

    row = {
        "city": req.city,
        "neighborhood": req.neighborhood or "Unknown",
        "property_type": req.property_type,
        "source_site": req.source_site,
        "surface_m2": req.surface_m2,
        "bedrooms": req.bedrooms,
        "bathrooms": req.bathrooms,
        "amenity_count": len(req.amenities) if req.amenities else 0,
        "furnished_num": (1 if req.furnished else 0) if req.furnished is not None else None,
    }
    X = pd.DataFrame([row])[artifact["feature_columns"]]

    log_pred = pipeline.predict(X)[0]
    predicted = float(np.expm1(log_pred))

    residual_std = artifact["residual_std"]
    low = max(0.0, predicted - 1.28 * residual_std)
    high = predicted + 1.28 * residual_std

    top_drivers = []
    warning = None
    if _explainer is not None:
        X_transformed = pipeline.named_steps["prep"].transform(X)
        shap_values = _explainer.shap_values(X_transformed)[0]
        feature_names = pipeline.named_steps["prep"].get_feature_names_out()
        pairs = sorted(zip(feature_names, shap_values), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        top_drivers = [
            {"feature": name, "impact_on_log_price": round(float(val), 4)}
            for name, val in pairs
        ]
    else:
        warning = "Model is linear regression fallback; per-prediction SHAP not computed."

    unknown_city = req.city not in _known_cities()
    if unknown_city:
        warning = (warning + " " if warning else "") + (
            f"City '{req.city}' was rare or absent in training data; prediction may be unreliable."
        )

    return PredictResponse(
        predicted_rent_mad=round(predicted, -1),
        confidence_interval_80pct=[round(low, -1), round(high, -1)],
        model_used=artifact["model_name"],
        top_drivers=top_drivers,
        warning=warning,
    )


def _known_cities() -> set:
    artifact = get_artifact()
    try:
        ohe = artifact["pipeline"].named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        city_idx = artifact["pipeline"].named_steps["prep"].transformers[0][2].index("city")
        return set(ohe.categories_[city_idx])
    except Exception:
        return set()
