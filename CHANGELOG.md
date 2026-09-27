# Changelog

All notable changes to the Morocco Rental Market Intelligence Platform, by phase.

## Phase 1 — RAG layer over listing descriptions

- New standalone service `rag_service/` (FastAPI, `127.0.0.1:8001`) providing
  semantic search over rental listing descriptions using local, free tooling:
  `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings and `chromadb`
  (persistent local mode) for vector storage. No API keys, no external calls.
- `rag_service/core.py` — shared logic for fetching listings, building the
  per-listing embedding text, chunking (no-op on this dataset's short
  descriptions, but present for longer text), and idempotent sync (a content
  hash per listing in `state.db` skips re-embedding unchanged listings).
- `rag_service/ingest.py` — one-shot CLI for the initial embed of all listings.
- `POST /rag/search` — natural-language query → ranked listings with
  similarity scores, optional `city_filter`.
- `POST /rag/sync` — incremental re-embed of new/changed listings only
  (same idempotent logic as `ingest.py`, exposed over HTTP).
- `GET /health` — indexed listing count, last sync timestamp, embedding model.
- Added `GET /export` to the existing ML service
  (`ml/price_prediction/api/main.py`) — a small read-only endpoint serving the
  current listings snapshot, so `rag_service` (and future services) have a
  clean local source of listing data without talking to n8n directly.
  `/predict`, `/health`, and `/model-info` are unchanged.
- pytest coverage (`rag_service/tests/test_core.py`): doc-text building, sync
  idempotency (unchanged listings skipped, changed listings re-embedded),
  city filtering, and semantic relevance of search results.

**Known limitation**: the scraped `description` field is short/title-like, not
a full free-text description, so qualitative attributes never present in the
source text (e.g. "quiet") won't surface — semantic search adds the most value
on neighborhood/amenity/property-type/price phrasing rather than subjective
qualities the scraper never captured.

## Phase 2 — Agent service (the chatbot's new brain)

- New standalone service `agent_service/` (FastAPI, `127.0.0.1:8002`) — a
  hand-rolled tool-calling agent loop against OpenRouter, replacing n8n's
  built-in AI Agent node as the chat brain. n8n's scraping/storage workflows
  are untouched; the n8n webhook/frontend now call this instead.
- `agent.py` — the manual loop: call the model, execute any tool calls it
  requests, feed results back, repeat until a final answer or the per-turn
  tool budget is hit. No agent framework — deliberately explainable.
- `tools.py` — 7 tools: `search_listings` / `get_market_stats` (mirror the
  n8n Code Tool nodes, read listings via `ml_service` `/export` with a local
  snapshot-file fallback), `predict_rent` (guardrail-checked against
  `ml_service` `/predict`), `explain_prediction` (new — turns SHAP
  `top_drivers` into a plain-language sentence), `rag_search` (calls Phase
  1's `/rag/search`), `live_web_lookup` (new — a Python port of n8n's "Live
  Market Lookup" sub-workflow's regex extraction for avito.ma/agenz.ma, so
  the standalone service doesn't depend on n8n at request time), and
  `remember_preference` (writes into session memory).
- `memory.py` — per-session SQLite (`sessions.db`): message history +
  extracted preferences (via the `remember_preference` tool call), injected
  back into the system prompt on every subsequent turn. Sessions inactive
  longer than `SESSION_TTL_HOURS` (default 6) are expired automatically.
- `guardrails.py` — enforced in code, not just prompted: price-prediction
  sanity bounds per city (median × [0.15, 6], global fallback for thin
  data), a tool-call budget per turn (`MAX_TOOL_CALLS_PER_TURN`, default 5),
  and forced "unverified" labeling of `live_web_lookup` results even if the
  model's final answer doesn't mention it.
- `observability.py` — every turn logged to `logs/turns.db`: tool calls
  (name, args, latency, result summary), total latency, prompt/completion
  token counts, success/failure.
- Graceful degradation: missing API key, network failure, HTTP 429 (quota
  exhausted), and HTTP 402 (insufficient credits) all return a clear
  `degraded: true` response instead of a crash.
- pytest coverage (20 tests): guardrail price bounds and tool-call budget,
  session lifecycle/preference merging/expiry, and tool filtering logic
  (network calls monkeypatched out).

- Added a provider abstraction (`LLM_PROVIDER=openrouter|gemini` in `.env`) —
  OpenRouter and Google's Gemini API both expose an OpenAI-compatible chat
  completions endpoint with identical request/response shape (including tool
  calls), so `agent.py` doesn't need to know which one it's talking to.
  Added after hitting OpenRouter's real free-tier daily cap (50
  requests/day account-wide across *all* free models — confirmed by testing
  a second free model directly, which hit the same `free-models-per-day`
  error) mid-verification; switched to `gemini-flash-latest` to keep testing
  unblocked. Switching back is a one-line `.env` change.

**Verified live** (session `verify-gemini-1..3`, via `gemini-flash-latest`):
- Turn 1 ("I want a 2-bedroom apartment in Rabat under 6000 MAD, remember
  that") → `remember_preference` ×3 (`preferred_city=Rabat`,
  `min_bedrooms=2`, `max_budget_mad=6000`) then `search_listings` with those
  filters, returning one real matching listing.
- Turn 2 ("What about in Casablanca instead?") → recalled the stored
  budget/bedroom preferences *without restating them*, called only
  `remember_preference` (city update) + `search_listings` +
  `get_market_stats` — correct, and no redundant re-saving of preferences
  already known.
- Turn 3, a fuzzy query ("something quiet near a school in Agadir") →
  correctly chose `rag_search` (not `search_listings`) with a `city_filter`.
- Turn 4, a hypothetical property ("furnished 3-bed 120m² in Marrakech with
  a pool") → correctly chose `explain_prediction` + `get_market_stats`,
  returned a predicted rent with a plain-language SHAP-driven explanation.
- Every turn logged to `logs/turns.db` with tool calls, arguments, latency,
  and token counts, confirmed by direct query.
- The earlier OpenRouter 429 was also confirmed logged correctly
  (`success=0`, exact error reason) before the provider switch — i.e. the
  graceful-degradation guardrail was observed firing under real conditions,
  not just in tests.

**Known limitations**:
- Free-tier LLM quotas are real and will interrupt heavy testing —
  OpenRouter's free models share one 50-requests/day account-wide cap
  (1000/day with $10 credit); Gemini's free tier has its own per-model
  limits. A single conversational turn can burn several requests when it
  makes multiple tool calls.
- `live_web_lookup`'s regex selectors are tied to avito.ma/agenz.ma's
  current frontend builds, same fragility n8n's own scraper already has.
  Not yet exercised in a live conversation turn (only via code review + the
  n8n workflow it was ported from) since the stored dataset covered every
  city tested.
- Preference "extraction" is the model choosing to call a tool, not a
  separate NLP step — only as reliable as the model's judgment.

## Phase 3 — Voice mode ("Jarvis mode")

- **STT**: `faster-whisper` (`voice.py`, `WHISPER_MODEL_SIZE`, default
  `base`), CPU/int8. Model auto-downloads from Hugging Face on first use,
  then runs fully offline — no API key, no per-request cost.
- **TTS**: Piper (`PIPER_VOICE`, default `en_US-lessac-medium`). Voice
  `.onnx` files auto-download to `agent_service/voice_models/` on first use
  via `piper.download_voices`, then run fully offline.
- New endpoints on `agent_service` (no separate voice service — it shares
  the agent's tools/memory/guardrails directly): `POST /voice/transcribe`,
  `POST /voice/speak`, `POST /voice/turn` (the combined
  record→transcribe→agent→synthesize round trip the frontend uses,
  returning `{transcript, response_text, audio_url, degraded}` together),
  `GET /voice/audio/{filename}`.
- `voice.py`'s WAV-encoding logic is factored out as `chunks_to_wav_bytes()`
  so it's unit-testable without loading a real Piper voice model — 5 pytest
  cases (valid WAV header, sample rate/frame count preserved, multi-chunk
  concatenation, out-of-range sample clipping, empty input).
- New "Voice" tab in `frontend/index.html`: push-to-talk mic button
  (MediaRecorder API), a "listening → transcribing → thinking → speaking"
  status indicator, transcript + response text displayed together,
  autoplaying response audio, and a pulsing ring during playback.
- Also fixed a Phase 2 loose end: the frontend's Chat tab was still calling
  the n8n webhook directly instead of the new `agent_service` — repointed it
  to `POST http://127.0.0.1:8002/chat`.

**Honesty note on the status indicator**: `/voice/turn` is one synchronous
request/response round trip, not a streaming API — the backend doesn't emit
incremental progress events. The "transcribing → thinking" status
progression in the frontend is a client-side timed display against that
single in-flight request, not literal server-sent state. Documented as such
in `agent_service/README.md` rather than implying real-time streaming the
stack doesn't do.

**Verified**: a synthesize → transcribe round trip (Piper generates a WAV
from a rent-related sentence, faster-whisper transcribes it back) confirmed
both directions work end-to-end without a live microphone.

**Known limitations**:
- First-run model downloads (Whisper `base` ~140MB, Piper voice ~60MB) can
  be slow on a poor connection — one-time cost, cached afterward.
- Push-to-talk only (click to start, click to stop) — no voice-activity
  detection or continuous listening, per the spec's own guidance to avoid
  half-working streaming.
- Single default voice/language pair (English). The scraped data and
  frontend copy are English/French-mixed; `PIPER_VOICE` can be swapped for
  a French voice via `.env` if needed.

### Post-Phase-3 fixes (same day)

- **Bug**: the first live test produced garbled, unintelligible audio.
  Root cause — agent replies are markdown (`**bold**`, bullet lists,
  tables, links) meant for on-screen text, and that markdown was being fed
  straight to Piper, which phonemizes whatever it's given literally
  (trying to pronounce `**`, `|`, `#`, etc.). Fixed with
  `voice.strip_markdown_for_speech()`, called inside `synthesize()` before
  any text reaches Piper. Verified by transcribing the synthesized output
  back through Whisper — before the fix this produced nonsense, after it
  produces a clean, accurate transcript of the original response (one
  follow-up fix: `~13,400` was being read as "till the 13,400"; added a
  `~` → "about" substitution). 4 new pytest cases cover the stripping logic.
- **UX change**: merged the standalone "Voice" tab into "Chat Assistant" —
  one mic button in the chat input row instead of a separate tab, sharing
  the same `sessionId` and message thread so a spoken question and a typed
  follow-up are part of the same conversation (both to the agent's memory
  and visually in the chat log). Spoken turns appear as chat bubbles
  (🎤-prefixed user message + bot reply) with the response audio
  autoplaying alongside.

## Phase 4 — Deal Finder, neighborhood clustering, natural-language filters

Picked 3 of the 5 proposed additions, prioritizing demoable + explainable
over breadth: **Deal Finder**, **neighborhood clustering**, and
**natural-language Listings filters**. Skipped multi-agent orchestration
(would mean redesigning `agent_service` into Scout/Analyst/Concierge —
high effort, and the single-agent version already demonstrates correct
tool routing) and the weekly auto-report (lower value-per-effort than the
other three given time available).

### Deal Finder

- `GET /deals` on `ml_service` — batches every priced listing through the
  trained pipeline in one call (not one HTTP round trip per listing) and
  flags listings priced below the model's own 80% confidence band (not
  just below the point estimate — see below for why that mattered) as
  potential deals.
- **Data-quality guardrail**: listings priced below `MIN_PLAUSIBLE_RENT_MAD`
  (1000 MAD) are excluded before scoring — a handful of scraped listings
  have prices like 250-400 MAD/month that are clearly data errors (a daily
  rate or deposit picked up instead of monthly rent), not real deals, and
  dominated the results before this was added.
- **Calibration fix during build**: a flat "≥15% below predicted" threshold
  alone flagged 180 of ~600 priced listings (30%) — mostly noise, given
  this model's modest R² (~0.47). Requiring the actual price to also fall
  outside the model's own residual-based 80% confidence interval (the same
  band `/predict` already surfaces as `confidence_interval_80pct`) brought
  that down to 14 — a defensible, statistically grounded list instead of
  threshold noise.
- New n8n workflow **"Deal Finder"** (`IDAZmpjwQtXXV5Wp`, built via the n8n
  MCP tools): Schedule Trigger (daily 07:00, after the 06:00 scraper) +
  manual-trigger test path → `GET {ml_service}/deals` → Split Out → insert
  into a new `deal_alerts` Data Table (created via `create_data_table`).
  Left **inactive** — same placeholder-URL pattern already established for
  the chatbot's `predict_rent` tool (`ml_service` isn't publicly reachable
  yet), and activating a *scheduled* workflow against an unresolvable
  placeholder would just spam failed executions, unlike the chatbot's
  on-demand tool call which only fails when actually used.
- New "Deals" tab in the frontend, reading `GET /deals` directly, with an
  explicit caveat about the model's modest R² so results read as leads to
  investigate, not guarantees.
- 7 new pytest cases in `ml/price_prediction/tests/test_main.py`: tier
  labeling logic, deal-flagging on a synthetic underpriced listing (using
  the real trained model against a real training city), and the
  min-price exclusion.

### Neighborhood clustering

- `GET /neighborhood-tiers` on `ml_service` — KMeans (`scikit-learn`,
  already a dependency) on each neighborhood's median price/m², average
  bedroom count, and average amenity count. Tiers are labeled from cluster
  *centroid characteristics*, not arbitrary cluster indices: highest
  price/m² → "Premium", lowest → "Value"; among any remaining middle
  clusters, the one with the most average bedrooms → "Family-oriented",
  the rest → "Mid-range" (falls back to "Balanced" for a single cluster).
- Surfaced as a new panel on the Overview tab: a color-coded tier badge per
  neighborhood alongside price/m², avg bedrooms, and listing count.

### Natural-language Listings filters

- `POST /nl-filter` on `agent_service` — a *one-shot* LLM call (not the
  full agent loop: no memory, no multi-tool round trips) with a forced
  function call (`apply_filters`) that translates free text like "3-bedroom
  furnished apartments in Rabat with parking under 7000 MAD" into exactly
  the filters the Listings tab's existing dropdowns already support (city,
  max_price, min_bedrooms) — cheap, since it reuses infrastructure Phase 2
  already built (`agent._call_llm` generalized to accept custom
  tool schemas instead of only the full toolset).
- Listings tab gets a free-text input above the existing filter dropdowns;
  submitting it calls `/nl-filter` and fills in / re-applies the same
  filter controls, rather than replacing them.
- 4 new pytest cases (`agent_service/tests/test_agent_filters.py`), LLM
  call monkeypatched out so they run free/offline.

### Provider note (mid-build)

Hit **two** separate free-tier walls while testing this phase:
OpenRouter's 50/day account-wide cap (already known from Phase 2) hadn't
reset yet, and `gemini-flash-latest` (the Phase 2 switch-to) turned out to
have its *own* daily cap that got exhausted from cumulative testing across
phases. Unlike OpenRouter, Gemini's cap is genuinely per-model (confirmed
by testing several other Gemini models directly) — switched
`GEMINI_MODEL` to `gemini-flash-lite-latest`, verified tool-calling still
works, no code changes needed.

**Known limitations**:
- `/deals` and `/neighborhood-tiers` compute on-demand from the local
  snapshot file each request — fine at this data scale (~1k listings),
  would need caching or a background job at larger scale.
- The Deal Finder n8n workflow is built and structurally verified
  (`get_workflow_details`) but not execution-tested end-to-end, since doing
  so requires the placeholder URL to be replaced with a real public one
  first — consistent with the existing, already-accepted state of
  `predict_rent`.
- Clustering re-runs from scratch on every `/neighborhood-tiers` call
  (KMeans with `random_state=42` for reproducibility, but no caching) —
  fine for ~60 neighborhoods, would need caching at a much larger scale.

## Phase 4 (continued) — Multi-agent orchestration

The last of the five original Phase 4 options, picked up after a pause.
Restructured the single Concierge agent into an orchestrator/workers
pattern: one supervisor agent, two specialist sub-agents exposed to it as
callable tools — not a framework, the same manual loop from `agent.py`
reused at a smaller scope per sub-agent.

- **`llm_client.py`** (new) — the OpenAI-compatible chat-completions client
  extracted out of `agent.py` so both Concierge and the sub-agents can share
  it without a circular import (`agent.py` → `subagents.py` →  needs the LLM
  client → would need `agent.py` back, if it had stayed there).
  `agent._call_llm` / `agent.DegradedResponseError` are kept as aliases so
  existing code and tests didn't need to change.
- **`subagents.py`** (new) — **Scout** (finds/monitors listings:
  `search_listings`, `rag_search`, `live_web_lookup`, and a new
  `list_recent_deals` wrapping Phase 4's `/deals`) and **Analyst** (market
  stats/predictions/trends: `get_market_stats`, `predict_rent`,
  `explain_prediction`, and a new `get_neighborhood_tiers` wrapping Phase
  4's `/neighborhood-tiers`). Each runs its own bounded tool loop
  (`SUB_AGENT_MAX_TOOL_CALLS = 3`, separate from Concierge's own budget) over
  only its own tools — Scout can't call `predict_rent`, Analyst can't call
  `live_web_lookup`.
- **Concierge** (`agent.py`, rewritten) — now holds only 3 tools:
  `remember_preference`, `ask_scout`, `ask_analyst`. No longer touches
  `search_listings`/`predict_rent`/etc. directly; it delegates and
  synthesizes. System prompt rewritten accordingly.
- **Observability carries through the delegation tree**: each sub-agent
  accepts the Concierge's `observability.TurnLogger` and logs its own tool
  calls into the *same* turn record, name-prefixed (`scout:search_listings`,
  `analyst:get_market_stats`) — one turn's log shows the whole tree, not
  just the top-level `ask_scout`/`ask_analyst` calls.
- **Live-lookup guardrail still holds across the delegation boundary**: if
  Scout uses `live_web_lookup`, that fact propagates back up through
  `ask_scout`'s return value so Concierge's final-answer disclosure check
  (`guardrails.ensure_live_lookup_disclosed`) still fires — this was the
  trickiest part of the refactor to get right (a flag needs to survive
  crossing from the sub-agent's local scope into the Concierge's turn-level
  state), so it's verified with a dedicated test at the `agent.run_turn`
  level, not just inside the sub-agent itself.
- `POST /agents/scout` / `POST /agents/analyst` (new, `main.py`) — talk to
  either specialist directly, bypassing Concierge/memory. Useful for
  debugging and for demoing each specialist in isolation.
- 10 new pytest cases: `tests/test_subagents.py` (6 — tool-schema scoping is
  disjoint between Scout/Analyst, tool dispatch → final answer, live-lookup
  flag propagation, per-sub-agent budget enforcement (the 4th of 4
  consecutive tool-call attempts is budget-blocked and never actually
  executes), degraded-LLM handling, the two new tool functions) and
  `tests/test_agent_orchestration.py` (4 — Concierge-level live-lookup
  propagation with and without the flag set, `remember_preference`
  dispatch still works post-rewrite, Concierge's tool schemas are exactly
  the 3 delegation-only tools). 43/43 tests passing service-wide
  (`test_agent_filters.py` needed zero changes, confirming the `_call_llm`
  alias worked).

**Verified live**: `/agents/scout` and `/agents/analyst` each answered
correctly in isolation using only their scoped tools. Then a full `/chat`
turn ("I want a 2-bedroom in Rabat under 6000 MAD, remember that. Also,
what's a good area price-wise?") correctly triggered `remember_preference`
×3, `ask_analyst` (which internally called `get_market_stats` twice and
attempted `get_neighborhood_tiers`), and `ask_scout` (which internally
called `search_listings`) — synthesized into one coherent answer citing
both the specific listing Scout found and the market averages Analyst
computed. The observability log confirmed the full nested tree was
captured correctly, including a graceful in-place recovery when
`get_neighborhood_tiers` hit the still-stale `ml_service` process's 404
(the pre-existing, unrelated restart-needed issue) — Analyst just
proceeded without that data instead of failing the turn.

**Known limitations**: see the new "Known limitations" additions in
`agent_service/README.md` — orchestration costs real latency/tokens (3+ LLM
round trips minimum for any query touching both specialists, vs. 1 for the
single-agent version), and sub-agents can't share context with each other
mid-delegation (each `ask_scout`/`ask_analyst` call is independent; Concierge
concatenates their answers, but Scout and Analyst can't inform each other's
reasoning within one delegation).

## Post-Phase-4 fix — Streamlit app dead in the browser (Starlette gzip mismatch)

- **Symptom**: the Streamlit app started fine and the script itself ran
  cleanly (headless `AppTest` reported zero exceptions, every tab rendered),
  but any real browser session got nothing usable out of the server. The app
  looked broken while every offline test passed.
- **Root cause**: `streamlit==1.61.0` vendors a subclass of Starlette's
  `GZipResponder` and instantiates it without the `thread_minimum_size`
  keyword argument. `starlette==1.4.0` had made that argument required
  (keyword-only, no default), so every HTTP response that negotiated gzip —
  i.e. every browser request, since browsers send `Accept-Encoding: gzip` —
  raised `TypeError: GZipResponder.__init__() missing 1 required keyword-only
  argument: 'thread_minimum_size'` inside the ASGI middleware, returning a 500
  and leaving the page as an empty shell. `curl` requests without the header
  still returned 200, which is why it looked healthy from the terminal.
- **Why the test suite missed it**: `AppTest` imports and runs `app.py`
  in-process, bypassing the ASGI/HTTP layer entirely, so the middleware bug
  never executed. Verified via a real headless Chrome session over the
  DevTools protocol instead (DOM text confirmed the dashboard hydrates).
- **Fix**: pin `starlette>=1.4.1,<2` in `streamlit_app/requirements.txt`
  (1.4.1 restored the default value for `thread_minimum_size`; 1.6.0 was
  installed and verified). No application code changed.
- **Verified after fix**: HTTP requests with `Accept-Encoding: gzip` return
  200 for `/`, `/healthz`, `/_stcore/health` and static assets; a headless
  Chrome session renders the sidebar, all five tabs, KPI tiles
  (1,799 listings, 7,500 MAD median), charts, and the neighborhood-tier
  panel. Zero ASGI exceptions in the server log.
