"""FastAPI service exposing the trained rent price prediction model."""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "latest.pkl"
METRICS_PATH = BASE_DIR / "models" / "latest_metrics.json"

app = FastAPI(title="Morocco Rental Price Prediction API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


SNAPSHOT_PATH = BASE_DIR / "data" / "listings_snapshot.json"


@app.get("/export")
def export_listings():
    """Read-only export of the current listings snapshot, for downstream local
    services (e.g. rag_service) that need listing data without talking to n8n directly."""
    if not SNAPSHOT_PATH.exists():
        raise HTTPException(status_code=404, detail="No listings snapshot found.")
    with open(SNAPSHOT_PATH, encoding="utf-8") as f:
        return json.load(f)


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


MIN_PLAUSIBLE_RENT_MAD = 1000  # below this is almost always bad scraped data (deposit, daily rate, typo), not a real deal


@app.get("/deals")
def deals(threshold_pct: float = 15.0, limit: int = 50, min_actual_price: float = MIN_PLAUSIBLE_RENT_MAD):
    """Compare each listing's actual price against the model's predicted price
    for a comparable property; flag listings priced >= threshold_pct% below the
    prediction as potential deals. Batches every listing through the pipeline in
    one call rather than one HTTP round trip per listing.

    Listings priced below min_actual_price are excluded up front — a handful of
    scraped listings have implausibly low prices (250-400 MAD/month) that are
    clearly data errors (daily rate or deposit picked up instead of monthly
    rent), not genuine deals, and would otherwise dominate the results."""
    artifact = get_artifact()
    pipeline = artifact["pipeline"]
    if not SNAPSHOT_PATH.exists():
        raise HTTPException(status_code=404, detail="No listings snapshot found.")
    with open(SNAPSHOT_PATH, encoding="utf-8") as f:
        listings = json.load(f)

    priced = [
        l for l in listings
        if isinstance(l.get("rent_price"), (int, float)) and l["rent_price"] >= min_actual_price
    ]
    if not priced:
        return {"count": 0, "threshold_pct": threshold_pct, "deals": []}

    rows = [{
        "city": l.get("city"),
        "neighborhood": l.get("neighborhood") or "Unknown",
        "property_type": l.get("property_type") or "appartement",
        "source_site": l.get("source_site") or "agenz",
        "surface_m2": l.get("surface_m2"),
        "bedrooms": l.get("bedrooms"),
        "bathrooms": l.get("bathrooms"),
        "amenity_count": len(l["amenities"].split(",")) if l.get("amenities") else 0,
        "furnished_num": (1 if l.get("furnished") else 0) if l.get("furnished") is not None else None,
    } for l in priced]

    X = pd.DataFrame(rows)[artifact["feature_columns"]]
    predicted = np.expm1(pipeline.predict(X))
    residual_std = artifact["residual_std"]

    threshold = threshold_pct / 100.0
    flagged = []
    for listing, pred in zip(priced, predicted):
        if pred <= 0:
            continue
        actual = listing["rent_price"]
        discount = (pred - actual) / pred
        # Same 80% band /predict already shows as confidence_interval_80pct: require the
        # actual price to fall genuinely outside the model's own uncertainty band, not just
        # below the point estimate — a flat % threshold alone flags far too much noise given
        # this model's modest R² (~0.47, see ml/price_prediction/README).
        low_bound_80pct = max(0.0, pred - 1.28 * residual_std)
        if discount >= threshold and actual < low_bound_80pct:
            flagged.append({
                "source_site": listing.get("source_site"),
                "city": listing.get("city"),
                "neighborhood": listing.get("neighborhood"),
                "rent_price": actual,
                "predicted_rent_mad": round(float(pred), -1),
                "confidence_low_bound_mad": round(float(low_bound_80pct), -1),
                "discount_pct": round(discount * 100, 1),
                "surface_m2": listing.get("surface_m2"),
                "bedrooms": listing.get("bedrooms"),
                "bathrooms": listing.get("bathrooms"),
                "url": listing.get("source_url"),
            })

    flagged.sort(key=lambda d: d["discount_pct"], reverse=True)
    return {"count": len(flagged), "threshold_pct": threshold_pct, "deals": flagged[:limit]}


def _label_tiers(cluster_stats: pd.DataFrame) -> dict:
    """Label clusters by centroid characteristics: highest price/m² -> Premium,
    lowest -> Value. Among any remaining middle clusters, the one with the most
    average bedrooms -> Family-oriented, the rest -> Mid-range."""
    by_price = cluster_stats.sort_values("avg_price_per_m2")
    ids = by_price["cluster"].tolist()
    tier: dict = {}
    if len(ids) == 1:
        tier[ids[0]] = "Balanced"
        return tier
    tier[ids[0]] = "Value"
    tier[ids[-1]] = "Premium"
    middle_ids = ids[1:-1]
    if middle_ids:
        middle = cluster_stats[cluster_stats["cluster"].isin(middle_ids)]
        family_id = middle.sort_values("avg_bedrooms", ascending=False)["cluster"].iloc[0]
        for cid in middle_ids:
            tier[cid] = "Family-oriented" if cid == family_id else "Mid-range"
    return tier


@app.get("/neighborhood-tiers")
def neighborhood_tiers(n_clusters: int = 4, min_listings: int = 3):
    """Cluster neighborhoods into market tiers via KMeans on median price/m²,
    average bedroom count, and average amenity count. Tiers are labeled from
    cluster centroid characteristics, not arbitrary cluster indices."""
    if not SNAPSHOT_PATH.exists():
        raise HTTPException(status_code=404, detail="No listings snapshot found.")
    with open(SNAPSHOT_PATH, encoding="utf-8") as f:
        listings = json.load(f)

    df = pd.DataFrame(listings)
    df = df[df["rent_price"].notna() & df["city"].notna()].copy()
    df["neighborhood"] = df["neighborhood"].fillna("Unknown")
    df["surface_m2"] = pd.to_numeric(df["surface_m2"], errors="coerce")
    df["price_per_m2"] = df["rent_price"] / df["surface_m2"]
    df["amenity_count"] = df["amenities"].apply(lambda a: len(a.split(",")) if isinstance(a, str) and a else 0)

    grouped = df.groupby(["city", "neighborhood"]).agg(
        listing_count=("rent_price", "count"),
        median_price_per_m2=("price_per_m2", "median"),
        avg_bedrooms=("bedrooms", "mean"),
        avg_amenity_count=("amenity_count", "mean"),
    ).reset_index()
    grouped = grouped[grouped["listing_count"] >= min_listings].dropna(subset=["median_price_per_m2"])

    if len(grouped) < n_clusters:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough neighborhoods with >= {min_listings} listings to form {n_clusters} clusters (have {len(grouped)}).",
        )

    features = grouped[["median_price_per_m2", "avg_bedrooms", "avg_amenity_count"]].fillna(0)
    X_scaled = StandardScaler().fit_transform(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    grouped["cluster"] = km.fit_predict(X_scaled)

    cluster_stats = grouped.groupby("cluster").agg(
        avg_price_per_m2=("median_price_per_m2", "mean"),
        avg_bedrooms=("avg_bedrooms", "mean"),
    ).reset_index()
    cluster_to_tier = _label_tiers(cluster_stats)
    grouped["tier"] = grouped["cluster"].map(cluster_to_tier)
    grouped = grouped.sort_values("median_price_per_m2", ascending=False)

    result = grouped[[
        "city", "neighborhood", "tier", "listing_count",
        "median_price_per_m2", "avg_bedrooms", "avg_amenity_count",
    ]].round({"median_price_per_m2": 1, "avg_bedrooms": 1, "avg_amenity_count": 1}).to_dict(orient="records")

    return {"n_clusters": n_clusters, "neighborhoods": result}


def _known_cities() -> set:
    artifact = get_artifact()
    try:
        ohe = artifact["pipeline"].named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        city_idx = artifact["pipeline"].named_steps["prep"].transformers[0][2].index("city")
        return set(ohe.categories_[city_idx])
    except Exception:
        return set()
