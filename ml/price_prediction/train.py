"""
Train a rent price prediction model on scraped Moroccan rental listings.

Data source: a JSON snapshot pulled from the n8n `rental_listings` Data Table
(see ../data/listings_snapshot.json). Only one day of scraping has happened so
far, so this script does a random train/test split rather than a time-based
split. Once the daily scraper has accumulated multiple weeks of history,
switch to splitting by `scraped_at` (train on older data, test on newer) to
avoid leakage and get a realistic read on forecasting-style performance.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "listings_snapshot.json"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

CATEGORICAL_FEATURES = ["city", "neighborhood", "property_type", "source_site"]
NUMERIC_FEATURES = ["surface_m2", "bedrooms", "bathrooms", "amenity_count", "furnished_num"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "rent_price"


def load_data() -> pd.DataFrame:
    with open(DATA_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} raw rows from {DATA_PATH.name}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["neighborhood"] = df["neighborhood"].fillna("Unknown")
    df["amenity_count"] = df["amenities"].fillna("").apply(
        lambda s: len([a for a in s.split(",") if a.strip()])
    )
    # furnished is bool/None in the source data; encode as 0/1/NaN so the
    # numeric imputer (most_frequent) can fill unknowns rather than silently
    # treating "unknown" as "not furnished".
    df["furnished_num"] = df["furnished"].map({True: 1, False: 0})
    return df


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    lo, hi = df[TARGET].quantile([0.01, 0.99])
    df = df[(df[TARGET] >= lo) & (df[TARGET] <= hi)].copy()
    print(f"Dropped {before - len(df)} rent_price outliers outside [{lo:.0f}, {hi:.0f}] MAD")
    return df


def build_preprocessor() -> ColumnTransformer:
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    return ColumnTransformer([
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ("num", numeric_pipeline, NUMERIC_FEATURES),
    ])


def evaluate(name: str, y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"[{name}] RMSE={rmse:,.0f} MAD  MAE={mae:,.0f} MAD  R2={r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    df = load_data()
    df = engineer_features(df)
    df = clean_outliers(df)

    X = df[FEATURE_COLUMNS]
    y_log = np.log1p(df[TARGET])

    X_train, X_test, y_train_log, y_test_log, y_train, y_test = train_test_split(
        X, y_log, df[TARGET], test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")

    preprocessor = build_preprocessor()

    baseline = Pipeline([("prep", preprocessor), ("model", LinearRegression())])
    baseline.fit(X_train, y_train_log)
    baseline_pred = np.expm1(baseline.predict(X_test))
    baseline_metrics = evaluate("LinearRegression baseline", y_test, baseline_pred)

    xgb_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
        )),
    ])
    xgb_pipeline.fit(X_train, y_train_log)
    xgb_pred = np.expm1(xgb_pipeline.predict(X_test))
    xgb_metrics = evaluate("XGBoost", y_test, xgb_pred)

    chosen_name, chosen_pipeline, chosen_metrics, chosen_pred = (
        ("xgboost", xgb_pipeline, xgb_metrics, xgb_pred)
        if xgb_metrics["rmse"] <= baseline_metrics["rmse"]
        else ("linear_regression", baseline, baseline_metrics, baseline_pred)
    )
    print(f"Chosen model: {chosen_name}")

    residuals = y_test.to_numpy() - chosen_pred
    residual_std = float(np.std(residuals))

    feature_names = chosen_pipeline.named_steps["prep"].get_feature_names_out()
    importance = {}
    if chosen_name == "xgboost":
        explainer = shap.TreeExplainer(chosen_pipeline.named_steps["model"])
        X_test_transformed = chosen_pipeline.named_steps["prep"].transform(X_test)
        sample_size = min(200, X_test_transformed.shape[0])
        shap_values = explainer.shap_values(X_test_transformed[:sample_size])
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = dict(sorted(
            zip(feature_names, mean_abs_shap.tolist()),
            key=lambda kv: kv[1], reverse=True,
        )[:15])
    else:
        coefs = np.abs(chosen_pipeline.named_steps["model"].coef_)
        importance = dict(sorted(
            zip(feature_names, coefs.tolist()),
            key=lambda kv: kv[1], reverse=True,
        )[:15])

    print("Top features:")
    for k, v in list(importance.items())[:8]:
        print(f"  {k}: {v:.4f}")

    artifact = {
        "pipeline": chosen_pipeline,
        "model_name": chosen_name,
        "feature_columns": FEATURE_COLUMNS,
        "log_target": True,
        "residual_std": residual_std,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    versioned_path = MODELS_DIR / f"price_model_{chosen_name}_{stamp}.pkl"
    latest_path = MODELS_DIR / "latest.pkl"
    joblib.dump(artifact, versioned_path)
    joblib.dump(artifact, latest_path)
    print(f"Saved model to {versioned_path} and {latest_path}")

    metrics_log = {
        "trained_at": artifact["trained_at"],
        "chosen_model": chosen_name,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "baseline_linear_regression": baseline_metrics,
        "xgboost": xgb_metrics,
        "residual_std": residual_std,
        "top_features": importance,
    }
    metrics_path = MODELS_DIR / f"metrics_{stamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2, ensure_ascii=False)
    with open(MODELS_DIR / "latest_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    sys.exit(main())
