"""Data and model loading for the Streamlit app.

All paths resolve against the repository root via __file__, so the app works
regardless of the working directory Streamlit Community Cloud uses. Each path
can be overridden with an environment variable (useful if you deploy the
streamlit_app/ folder as a standalone repo and copy the files in).
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]

def _env_path(var: str, default: Path) -> Path:
    override = os.getenv(var)
    return Path(override) if override else default

DATA_JSON_PATH = _env_path("STREAMLIT_DATA_JSON", REPO_ROOT / "frontend" / "data.json")
SNAPSHOT_JSON_PATH = _env_path("STREAMLIT_SNAPSHOT_JSON", REPO_ROOT / "ml" / "price_prediction" / "data" / "listings_snapshot.json")
MODEL_PKL_PATH = _env_path("STREAMLIT_MODEL_PKL", REPO_ROOT / "ml" / "price_prediction" / "models" / "latest.pkl")
METRICS_JSON_PATH = _env_path("STREAMLIT_METRICS_JSON", REPO_ROOT / "ml" / "price_prediction" / "models" / "latest_metrics.json")


@lru_cache(maxsize=1)
def load_dashboard_data() -> dict:
    """frontend/data.json — the {summary, listings} snapshot the HTML dashboard reads."""
    with open(DATA_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_snapshot() -> list[dict]:
    """ml/price_prediction/data/listings_snapshot.json — full listings used by the model."""
    with open(SNAPSHOT_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_metrics() -> dict | None:
    if not METRICS_JSON_PATH.exists():
        return None
    with open(METRICS_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_model() -> dict:
    """The joblib artifact saved by train.py (pipeline, model_name, residual_std, ...)."""
    return joblib.load(MODEL_PKL_PATH)


def file_status() -> dict[str, bool]:
    """Which backing files were found — shown in the app sidebar."""
    return {
        "Dashboard data (data.json)": DATA_JSON_PATH.exists(),
        "Listings snapshot": SNAPSHOT_JSON_PATH.exists(),
        "Trained model (latest.pkl)": MODEL_PKL_PATH.exists(),
        "Model metrics": METRICS_JSON_PATH.exists(),
    }
