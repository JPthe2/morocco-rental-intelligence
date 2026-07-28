# rag_service

Semantic search over rental listing descriptions. Lets the agent answer fuzzy
queries like *"something quiet near a school in Agadir under 5000 MAD"* instead
of only structured filters (city, price range, bedrooms).

Fully local: embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`), vector
storage via `chromadb` in persistent local mode. No API keys, no external calls.

## How it fits in

```
ml_service (127.0.0.1:8000)  --/export-->  rag_service ingest/sync  -->  chroma_db/
                                                    |
                                            POST /rag/search
```

`rag_service` pulls listing data from the ML service's `GET /export` endpoint
(which just serves the current `listings_snapshot.json`). If that service isn't
running, it falls back to reading the snapshot file directly from
`../ml/price_prediction/data/listings_snapshot.json`.

**Known limitation**: the `description` field in the scraped data is short and
title-like (e.g. `"Appartement à louer 8 000 dh 150 m², 3 chambres - Maarif"`),
not a full free-text description. Semantic search still works well for
matching on neighborhood/amenities/property type/price, but qualitative
attributes that were never scraped (e.g. "quiet", "near a school") won't be
found unless a listing happens to mention them.

## Setup

```bash
cd rag_service
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt   # Windows
```

## Ingest listings (first run)

Make sure the ML service is running first (`cd ml/price_prediction && ./.venv/Scripts/python -m uvicorn api.main:app --reload`), or the local snapshot file must exist.

```bash
./.venv/Scripts/python ingest.py
```

This embeds every listing and upserts it into `./chroma_db`. Re-running it is
safe and cheap — a listing is only re-embedded if its derived text actually
changed (a content hash is tracked in `state.db`). Use `--force` to re-embed
everything regardless.

## Run the service

```bash
./.venv/Scripts/python -m uvicorn main:app --reload --port 8001
```

## Endpoints

- `GET /health` — indexed listing count, last sync time, embedding model name.
- `POST /rag/search` — `{"query": "...", "top_k": 5, "city_filter": "Agadir"}` → ranked listings with similarity scores.
- `POST /rag/sync` — `{"force": false}` → re-runs ingestion (new/changed listings only, unless forced). Same logic as `ingest.py`, exposed over HTTP so the agent service or a scheduler can trigger it.

### Example

```bash
curl -X POST http://127.0.0.1:8001/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "quiet apartment near a school in Agadir under 5000 MAD", "top_k": 5}'
```

## Tests

```bash
cd rag_service
./.venv/Scripts/pytest
```

Tests exercise the core logic against an isolated temp Chroma DB (never touches
`./chroma_db`): doc-text building, sync idempotency (unchanged listings are
skipped, changed ones are re-embedded), city filtering, and semantic relevance
of search results.

## Notes / tradeoffs

- One vector per listing (no per-listing fragmentation): `chunk_text()` exists
  for long descriptions but is a no-op on this dataset's short descriptions.
- `city_filter` is an exact (case-sensitive) match against the stored `city`
  metadata field, since that's how the source data is cased consistently.
- The embedding model (~80MB) downloads once from Hugging Face on first use
  and is cached locally afterward — no ongoing network dependency or cost.
