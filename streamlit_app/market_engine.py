"""Pure-pandas reimplementation of the ML service endpoints (/predict, /deals,
/neighborhood-tiers) so the Streamlit app is self-contained — no FastAPI service
needs to be running. Logic mirrors ml/price_prediction/api/main.py exactly so
numbers agree with the existing HTML dashboard.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from data_loader import load_model, load_metrics, load_snapshot

FEATURE_COLUMNS = [
    "city", "neighborhood", "property_type", "source_site",
    "surface_m2", "bedrooms", "bathrooms", "amenity_count", "furnished_num",
]

MIN_PLAUSIBLE_RENT_MAD = 1000  # below this is almost always bad scraped data (deposit, daily rate, typo)


def model_meta() -> dict:
    artifact = load_model()
    return {
        "model_name": artifact["model_name"],
        "trained_at": artifact["trained_at"],
        "n_train_samples": artifact["n_train_samples"],
        "residual_std": float(artifact["residual_std"]),
        "metrics": load_metrics(),
    }


def _known_cities() -> set[str]:
    artifact = load_model()
    try:
        ohe = artifact["pipeline"].named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        city_idx = artifact["pipeline"].named_steps["prep"].transformers[0][2].index("city")
        return set(ohe.categories_[city_idx])
    except Exception:
        return set()


def _row_to_frame(
    city: str,
    neighborhood: str | None = None,
    property_type: str = "appartement",
    source_site: str = "agenz",
    surface_m2: float | None = None,
    bedrooms: int | None = None,
    bathrooms: int | None = None,
    furnished: bool | None = None,
    amenities: list[str] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "city": city,
        "neighborhood": neighborhood or "Unknown",
        "property_type": property_type,
        "source_site": source_site,
        "surface_m2": surface_m2,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "amenity_count": len(amenities) if amenities else 0,
        "furnished_num": (1 if furnished else 0) if furnished is not None else None,
    }])[FEATURE_COLUMNS]


@lru_cache(maxsize=1)
def _shap_explainer():
    """Cached so repeated predictions don't rebuild the TreeExplainer each time."""
    artifact = load_model()
    if artifact["model_name"] != "xgboost":
        return None
    try:
        import shap
    except ImportError:
        return None
    try:
        return shap.TreeExplainer(artifact["pipeline"].named_steps["model"])
    except Exception:
        return None


def _shap_drivers(X: pd.DataFrame) -> list[dict]:
    explainer = _shap_explainer()
    if explainer is None:
        return []
    try:
        artifact = load_model()
        prep = artifact["pipeline"].named_steps["prep"]
        shap_values = explainer.shap_values(prep.transform(X))[0]
        feature_names = prep.get_feature_names_out()
        pairs = sorted(zip(feature_names, shap_values), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        return [{"feature": name, "impact_on_log_price": round(float(val), 4)} for name, val in pairs]
    except Exception:
        return []


def predict_rent(
    city: str,
    neighborhood: str | None = None,
    surface_m2: float | None = None,
    bedrooms: int | None = None,
    bathrooms: int | None = None,
    furnished: bool | None = None,
    amenities: list[str] | None = None,
    property_type: str = "appartement",
    source_site: str = "agenz",
) -> dict:
    """Returns the same shape as POST /predict on the FastAPI service."""
    artifact = load_model()
    X = _row_to_frame(
        city=city, neighborhood=neighborhood, surface_m2=surface_m2,
        bedrooms=bedrooms, bathrooms=bathrooms, furnished=furnished,
        amenities=amenities, property_type=property_type, source_site=source_site,
    )
    log_pred = artifact["pipeline"].predict(X)[0]
    predicted = float(np.expm1(log_pred))
    residual_std = artifact["residual_std"]

    low = max(0.0, predicted - 1.28 * residual_std)
    high = predicted + 1.28 * residual_std

    warning = None
    if city not in _known_cities():
        warning = (
            f"City '{city}' was rare or absent in training data; prediction may be unreliable."
        )

    return {
        "predicted_rent_mad": round(predicted, -1),
        "confidence_interval_80pct": [round(low, -1), round(high, -1)],
        "model_used": artifact["model_name"],
        "top_drivers": _shap_drivers(X),
        "warning": warning,
    }


def find_deals(
    threshold_pct: float = 15.0,
    limit: int = 50,
    min_actual_price: float = MIN_PLAUSIBLE_RENT_MAD,
) -> dict:
    """Same as GET /deals — flags listings priced meaningfully below what the
    model predicts for a comparable property (below the 80% confidence band, not
    just the point estimate)."""
    artifact = load_model()
    listings = load_snapshot()

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

    X = pd.DataFrame(rows)[FEATURE_COLUMNS]
    predicted = np.expm1(artifact["pipeline"].predict(X))
    residual_std = artifact["residual_std"]

    threshold = threshold_pct / 100.0
    flagged = []
    for listing, pred in zip(priced, predicted):
        pred = float(pred)
        if pred <= 0:
            continue
        actual = listing["rent_price"]
        discount = (pred - actual) / pred
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
    lowest -> Value. Among middle clusters, the one with most avg bedrooms ->
    Family-oriented, the rest -> Mid-range."""
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


def neighborhood_tiers(n_clusters: int = 4, min_listings: int = 3) -> dict:
    """Same as GET /neighborhood-tiers — KMeans clustering of neighborhoods into
    Value / Mid-range / Family-oriented / Premium tiers."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    df = pd.DataFrame(load_snapshot())
    df = df[df["rent_price"].notna() & df["city"].notna()].copy()
    df["neighborhood"] = df["neighborhood"].fillna("Unknown")
    df["surface_m2"] = pd.to_numeric(df["surface_m2"], errors="coerce")
    df["price_per_m2"] = df["rent_price"] / df["surface_m2"]
    df["amenity_count"] = df["amenities"].apply(
        lambda a: len(a.split(",")) if isinstance(a, str) and a else 0
    )

    grouped = df.groupby(["city", "neighborhood"]).agg(
        listing_count=("rent_price", "count"),
        median_price_per_m2=("price_per_m2", "median"),
        avg_bedrooms=("bedrooms", "mean"),
        avg_amenity_count=("amenity_count", "mean"),
    ).reset_index()
    grouped = grouped[grouped["listing_count"] >= min_listings].dropna(subset=["median_price_per_m2"])

    if len(grouped) < n_clusters:
        return {
            "n_clusters": n_clusters,
            "error": (
                f"Not enough neighborhoods with >= {min_listings} listings to form "
                f"{n_clusters} clusters (have {len(grouped)})."
            ),
            "neighborhoods": [],
        }

    features = grouped[["median_price_per_m2", "avg_bedrooms", "avg_amenity_count"]].fillna(0)
    X_scaled = StandardScaler().fit_transform(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    grouped["cluster"] = km.fit_predict(X_scaled)

    cluster_stats = grouped.groupby("cluster").agg(
        avg_price_per_m2=("median_price_per_m2", "mean"),
        avg_bedrooms=("avg_bedrooms", "mean"),
    ).reset_index()
    grouped["tier"] = grouped["cluster"].map(_label_tiers(cluster_stats))
    grouped = grouped.sort_values("median_price_per_m2", ascending=False)

    result = grouped[[
        "city", "neighborhood", "tier", "listing_count",
        "median_price_per_m2", "avg_bedrooms", "avg_amenity_count",
    ]].round({
        "median_price_per_m2": 1, "avg_bedrooms": 1, "avg_amenity_count": 1,
    }).to_dict(orient="records")

    return {"n_clusters": n_clusters, "neighborhoods": result}


def market_stats(city: str | None = None) -> dict:
    """Market stats for the chat assistant / NL filter — from frontend/data.json."""
    from data_loader import load_dashboard_data

    data = load_dashboard_data()
    summary = data["summary"]
    stats = {
        "total_listings": summary["totalListings"],
        "with_price_count": summary["withPriceCount"],
        "avg_rent_mad": summary["avgRent"],
        "median_rent_mad": summary["medianRent"],
        "min_rent_mad": summary["minRent"],
        "max_rent_mad": summary["maxRent"],
        "by_site": summary["bySite"],
    }
    if city:
        match = next((c for c in summary["cityStats"] if c["city"].lower() == city.lower()), None)
        if match:
            stats["city"] = match["city"]
            stats["city_count"] = match["count"]
            stats["city_avg_rent_mad"] = match["avgRent"]
            stats["city_median_rent_mad"] = match["medianRent"]
            stats["city_avg_surface_m2"] = match["avgSurface"]
    return stats


def search_listings(
    city: str | None = None,
    max_price: float | None = None,
    min_bedrooms: int | None = None,
    keyword: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Lightweight listing search for the chat assistant, over frontend/data.json."""
    from data_loader import load_dashboard_data

    rows = load_dashboard_data()["listings"]
    if city:
        rows = [r for r in rows if r.get("city") and r["city"].lower() == city.lower()]
    if max_price is not None:
        rows = [r for r in rows if isinstance(r.get("rent_price"), (int, float)) and r["rent_price"] <= max_price]
    if min_bedrooms is not None:
        rows = [r for r in rows if isinstance(r.get("bedrooms"), (int, float)) and r["bedrooms"] >= min_bedrooms]
    if keyword:
        kw = keyword.lower()
        rows = [
            r for r in rows
            if kw in str(r.get("city") or "").lower()
            or kw in str(r.get("neighborhood") or "").lower()
        ]
    return rows[:limit]


def known_city_names() -> list[str]:
    from data_loader import load_dashboard_data
    return [c["city"] for c in load_dashboard_data()["summary"]["cityStats"]]
