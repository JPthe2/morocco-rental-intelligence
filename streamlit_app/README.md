# Streamlit app

A self-contained Streamlit port of the platform's dashboard. It reads the static
snapshot (`frontend/data.json`) and runs the trained XGBoost model in-process —
**no FastAPI services need to be running**, so it deploys with a single command.

## Features (mirrors `frontend/index.html`)

- **Overview** — KPI tiles, median rent by city, rent distribution, bedroom stats,
  data quality, and neighborhood tiers (KMeans).
- **Listings** — plain filters (city / max rent / min bedrooms / keyword) plus a
  natural-language filter box (e.g. *"3-bedroom furnished apartments in Rabat
  under 7000 MAD"*).
- **Predict Price** — runs `ml/price_prediction/models/latest.pkl` in-process and
  returns the predicted rent, 80% confidence interval, and top SHAP drivers.
- **Deals** — replicas of the FastAPI `/deals` logic (priced below the model's own
  80% confidence band).
- **Chat Assistant** — local rule-based Q&A by default; switches to an
  OpenAI-compatible tool-calling LLM automatically if an API key is configured.

## Run locally

```bash
cd streamlit_app
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt   # Windows
.\.venv\Scripts\streamlit run app.py
```

Data/model are read from the repo (paths resolved from this folder): `../frontend/data.json`,
`../ml/price_prediction/data/listings_snapshot.json`, `../ml/price_prediction/models/latest.pkl`.
Each can be overridden with `STREAMLIT_DATA_JSON`, `STREAMLIT_SNAPSHOT_JSON`,
`STREAMLIT_MODEL_PKL`, `STREAMLIT_METRICS_JSON` if you move them.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo/branch.
3. Set **Main file path** to `streamlit_app/app.py`. The app's requirements live in
   `streamlit_app/requirements.txt` (same folder as the main file, so the cloud
   picks it up automatically).
4. Deploy. The repo is cloned in full, so the relative data/model paths work as-is.

### Optional: enable the LLM-backed chat

Add these as Streamlit **secrets** (Advanced settings → Secrets) to upgrade the
chat from local rules to a tool-calling LLM:

```toml
OPENROUTER_API_KEY = "sk-or-..."          # or use Gemini instead:
# GEMINI_API_KEY = "..."
# LLM_PROVIDER = "gemini"
# OPENROUTER_MODEL = "openai/gpt-oss-20b:free"   # optional, defaults shown
```

Without a key the app still works — chat just uses the built-in local answerer.

## Notes

- `requirements.txt` pins `scikit-learn==1.9.0` (and compatible `xgboost`/`joblib`)
  because the saved pipeline in `../ml/price_prediction/models/latest.pkl` was
  trained with that version. Rebuild the model with a different sklearn and update
  the pin accordingly.
- The static snapshots are as of the last export; regenerate `frontend/data.json`
  and `ml/price_prediction/data/listings_snapshot.json` to refresh the numbers.
