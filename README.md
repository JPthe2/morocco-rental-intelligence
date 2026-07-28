# Morocco Rental Market Intelligence Platform

An AI-powered pipeline that scrapes Moroccan rental listings, cleans and centralizes them, predicts missing/hypothetical prices with a trained ML model, and exposes it all through a conversational chatbot.

Built as an n8n-first system for scraping, storage, and orchestration (n8n cloud). The chat brain, retrieval, voice, and market-intelligence features on top of that data now live in standalone local Python/FastAPI services — see [Beyond the original n8n scope](#beyond-the-original-n8n-scope) and `CHANGELOG.md` for the full phase-by-phase build history.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              n8n (cloud)                  │
                    │                                           │
  agenz.ma  ───┐    │  ┌─────────────────────────────────┐      │
                ├──▶│  │ Rental Listings Scraper - Morocco │      │
  avito.ma  ───┘    │  │ (daily 06:00, both sites)         │      │
                    │  └───────────────┬───────────────────┘      │
                    │                  │ upsert                   │
                    │                  ▼                          │
                    │     ┌─────────────────────────┐             │
                    │     │ Data Table:              │             │
                    │     │ rental_listings           │◀────────┐  │
                    │     │ scrape_errors             │         │  │
                    │     └─────────────────────────┘          │  │
                    │                  ▲                        │  │
                    │                  │ query                  │  │
                    │     ┌─────────────────────────┐           │  │
                    │     │ Morocco Rental Market     │──────────┘  │
                    │     │ Chatbot (AI Agent)         │             │
                    │     │  - search_listings (tool)  │             │
                    │     │  - get_market_stats (tool) │             │
                    │     │  - predict_rent (tool) ────┼──HTTP──┐    │
                    │     └─────────────────────────┘         │    │
                    └───────────────────────────────────────────┼────┘
                                                                 │
                                                                 ▼
                                              ┌──────────────────────────────┐
                                              │ FastAPI ML service            │
                                              │ (deployed on Render)          │
                                              │  - /predict (XGBoost model)   │
                                              │  - /health, /model-info       │
                                              └──────────────────────────────┘
```

## What's live

| Component | Where | Status |
|---|---|---|
| Scraper (agenz.ma + avito.ma) | n8n workflow "Rental Listings Scraper - Morocco" | Active, daily 06:00, ~700 listings/day |
| Storage | n8n Data Tables: `rental_listings`, `scrape_errors` | Live |
| Chatbot | n8n workflow "Morocco Rental Market Chatbot" | Built, LLM = `openai/gpt-oss-20b:free` via OpenRouter |
| Price prediction API | `ml/price_prediction/` (this repo), deployed on Render | See deployment section below |

## Data pipeline

1. **Scrape** — daily, paginated HTTP fetch of agenz.ma and avito.ma listing pages (browser headers, 1s delay between requests).
2. **Extract** — regex-based field extraction per listing card (price, surface, bedrooms/bathrooms, city/neighborhood, amenities, image). Falls back to parsing price out of the listing title/description text when the structured price block is missing.
3. **Dedupe** — upsert into `rental_listings` keyed on `source_url`, so re-running the daily job updates existing listings instead of duplicating them.
4. **Errors** — failed fetches are logged to `scrape_errors` instead of crashing the run.

**Known fragility**: both sites use CSS-module/styled-components hashed class names tied to their frontend build. If either site redeploys its frontend, the extraction may silently drop fields until selectors are re-verified. `data-*` attributes and structural anchors (e.g. `title=` attributes, `data-testid`) were preferred over class names where possible for exactly this reason.

## Price prediction (`ml/price_prediction/`)

- `train.py` — loads a snapshot of `rental_listings`, engineers features (amenity count, furnished flag, log-transformed target), trains a baseline `LinearRegression` and an `XGBRegressor`, picks the better one, computes SHAP feature importances, and saves the model + metrics under `models/`.
- `api/main.py` — FastAPI service exposing:
  - `POST /predict` — takes property features, returns predicted rent, an 80% confidence interval (based on residual std), and the top SHAP-driven price factors.
  - `GET /health` — model metadata.
  - `GET /model-info` — full metrics (train/test size, RMSE/MAE/R² for both models, feature importances).

**Current model performance** (trained on 706 listings, one day of scraped history): XGBoost R² = 0.47 vs. 0.33 for the linear baseline. Modest by design — one day of data and ~46% missing neighborhood values limit signal. Retrain periodically as the daily scraper accumulates more history (`python train.py` after refreshing `data/listings_snapshot.json`).

### Local development

```bash
cd ml/price_prediction
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt   # Windows
python train.py
./.venv/Scripts/python -m uvicorn api.main:app --reload
```

### Deployment (Render)

`render.yaml` at the repo root defines the service. On Render: New → Blueprint → connect this repo → Render reads `render.yaml` and provisions the free web service automatically. Root directory, build command, and start command are all pre-configured.

Once deployed, update the `predict_rent` tool's URL in the n8n chatbot workflow to the Render service's public URL + `/predict`.

## Chatbot (original n8n version)

n8n AI Agent (workflow: "Morocco Rental Market Chatbot") with tools querying `rental_listings` directly and calling the FastAPI `/predict` endpoint for hypothetical properties. LLM: `openai/gpt-oss-20b:free` via OpenRouter. **Superseded as the active chat brain by `agent_service/`** (see below) — the n8n workflow itself is untouched and still works standalone, but the frontend and voice interface now talk to `agent_service` instead, which adds memory, guardrails, more tools, and voice on top of the same underlying data.

## Beyond the original n8n scope

Four standalone local services were added on top of the n8n pipeline (each has its own README with setup/run instructions):

| Service | Port | Adds |
|---|---|---|
| `ml/price_prediction/` | 8000 | Beyond `/predict`: `GET /export` (listings for other local services), `GET /deals` (flags listings priced outside the model's own 80% confidence band), `GET /neighborhood-tiers` (KMeans clustering into Value/Mid-range/Family-oriented/Premium tiers) |
| `rag_service/` | 8001 | Semantic search over listing descriptions (`sentence-transformers` + `chromadb`, fully local) |
| `agent_service/` | 8002 | The actual chat brain now: a hand-rolled tool-calling agent loop (OpenRouter or Gemini) with per-session memory, code-enforced guardrails, full turn logging, voice (local Whisper STT + Piper TTS), and a one-shot natural-language-to-filter endpoint for the Listings tab |
| `frontend/` | — | Plain HTML/JS dashboard: Overview (incl. neighborhood tiers), Listings (incl. NL filters), Predict, Deals, Chat (incl. voice) |

An n8n workflow "Deal Finder" (schedule-triggered, calls `ml_service` `/deals` and upserts into a new `deal_alerts` Data Table) was also built as a portfolio artifact — left **inactive** since it needs `ml_service`'s local URL made public (ngrok or real deployment) before its schedule can actually fire; see `CHANGELOG.md` Phase 4.

Full build history, verification notes, and known limitations for all of the above: `CHANGELOG.md`.

## Deliberately deferred

- **Forecasting / trend analysis** — needs weeks of scraped history to mean anything; only one day of data exists so far. Revisit once the daily scraper has accumulated enough history.
- **Multi-agent orchestration, weekly auto-generated reports** — considered for Phase 4, not built; see `CHANGELOG.md` Phase 4 for what was prioritized instead and why.
