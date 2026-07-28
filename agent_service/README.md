# agent_service

The chatbot's new brain: a standalone FastAPI service (`127.0.0.1:8002`) with a
hand-rolled tool-calling agent loop, replacing n8n's built-in AI Agent node for
the parts that needed things it doesn't support well — persistent memory,
code-enforced guardrails, and voice. The n8n scraping/storage workflows are
untouched; this only replaces the chat brain. The frontend's Chat and Voice
tabs both call this service directly (`http://127.0.0.1:8002`).

## Why a manual loop instead of a framework

A framework (LangGraph, etc.) buys abstraction we don't need at this scale. A
plain `while` loop — call the model, run any tool calls it asks for, feed
results back, repeat — is ~150 lines, has no hidden control flow, and is
easier to reason about and explain than a framework's internals. See `agent.py`.

## Architecture

```
frontend / n8n webhook
        |
        v
   POST /chat  (main.py)
        |
        v
  agent.run_turn()  (agent.py)
        |  loop: call OpenRouter -> execute requested tool calls -> feed back
        v
     tools.py  ---->  ml_service  (127.0.0.1:8000)   /predict, /export
                  ---->  rag_service (127.0.0.1:8001)  /rag/search
                  ---->  avito.ma / agenz.ma            (live_web_lookup only)
        |
        v
  memory.py (sessions.db)      guardrails.py           observability.py
  conversation history +       price sanity checks,     every turn logged
  extracted preferences        tool-call budget,         to logs/turns.db
                                live-lookup labeling
```

## Setup

```bash
cd agent_service
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt   # Windows
copy .env.example .env
# edit .env: set LLM_PROVIDER to "openrouter" or "gemini", and the matching API key
#   OpenRouter: https://openrouter.ai/keys (free tier: 50 req/day account-wide, 1000/day with $10 credit)
#   Gemini:     https://aistudio.google.com/apikey (free tier: per-model limits)
```

The ML service (`127.0.0.1:8000`) and RAG service (`127.0.0.1:8001`) should be
running first — the agent's tools call them directly.

## Run

```bash
./.venv/Scripts/python -m uvicorn main:app --reload --port 8002
```

## Endpoints

- `GET /health` — whether an OpenRouter key is configured, and which model.
- `POST /chat` — `{"session_id": "demo-1", "message": "..."}` → `{"response_text": "...", "degraded": false}`.
  `degraded: true` means the LLM backend couldn't be reached (missing key,
  OpenRouter free-tier quota exhausted, etc.) — `response_text` explains why,
  in plain language, instead of the request failing with a 500.

### Example: multi-turn conversation with memory

```bash
curl -X POST http://127.0.0.1:8002/chat -H "Content-Type: application/json" \
  -d "{\"session_id\": \"demo-1\", \"message\": \"I want a 2-bedroom in Rabat under 6000 MAD\"}"

curl -X POST http://127.0.0.1:8002/chat -H "Content-Type: application/json" \
  -d "{\"session_id\": \"demo-1\", \"message\": \"What about Casablanca instead?\"}"
```
The second call should apply the budget/bedroom preference stated in the first
without you repeating it — the agent calls `remember_preference` internally
and re-injects known preferences into every subsequent turn's system prompt.

## Voice ("Jarvis mode")

Fully local speech I/O — no cloud STT/TTS API, no extra cost:

- **STT**: `faster-whisper` (`WHISPER_MODEL_SIZE`, default `base`), CPU/int8.
  Downloads its model from Hugging Face on first use, then runs offline.
- **TTS**: Piper (`PIPER_VOICE`, default `en_US-lessac-medium`). The `.onnx`
  voice model auto-downloads to `voice_models/` on first use via
  `piper.download_voices`, then runs offline.

| Endpoint | Purpose |
|---|---|
| `POST /voice/transcribe` | multipart audio file → `{"transcript": "..."}` |
| `POST /voice/speak` | `{"text": "..."}` → WAV audio bytes |
| `POST /voice/turn?session_id=...` | multipart audio file → transcribe, run the Phase 2 agent, synthesize the reply, **in one request** → `{"transcript", "response_text", "audio_url", "degraded"}` |
| `GET /voice/audio/{filename}` | serves a synthesized reply written to `voice_output/` |

`/voice/turn` is the one the frontend's Voice tab uses — a single
record → send → {transcript + text + audio} round trip, not a streaming API.
The frontend's "listening → transcribing → thinking → speaking" status
indicator is a client-side timed display against that one request, not
literal server-sent progress events — kept simple and honest rather than
faking real-time streaming the stack doesn't actually do.

```bash
curl -X POST http://127.0.0.1:8002/voice/turn \
  -F "file=@question.wav"
```

## Tools available to the agent

| Tool | Backing | Notes |
|---|---|---|
| `search_listings` | `ml_service` `/export` | mirrors the n8n Code Tool node of the same name |
| `get_market_stats` | `ml_service` `/export` | ditto |
| `predict_rent` | `ml_service` `/predict` | guardrail-checked against a plausible per-city range before being returned |
| `explain_prediction` | `ml_service` `/predict` | turns SHAP `top_drivers` into a plain-language sentence |
| `rag_search` | `rag_service` `/rag/search` | Phase 1's semantic search |
| `live_web_lookup` | avito.ma / agenz.ma directly | Python port of n8n's "Live Market Lookup" sub-workflow's regex extraction; always labeled unverified |
| `remember_preference` | local | writes into the session's stored preferences, not an external call |

## Guardrails (enforced in code, not just prompted)

- **Price sanity**: `predict_rent` computes a plausible `[low, high]` MAD range
  per city from the local listings snapshot (median × [0.15, 6], falling back
  to a generous global range for cities with too little local data) and
  rejects predictions outside it rather than returning them as fact.
- **Tool-call budget**: `guardrails.ToolCallBudget` caps tool calls per turn
  (`MAX_TOOL_CALLS_PER_TURN`, default 5) — if the model tries to keep calling
  tools past that, it's told to give its best final answer instead.
- **Live-lookup labeling**: `live_web_lookup` results are stamped
  `unverified: true` with a disclaimer at the code level
  (`guardrails.label_live_lookup_result`), and if the model's final answer
  doesn't mention that a live lookup was used, the disclaimer is appended
  automatically (`guardrails.ensure_live_lookup_disclosed`) — this doesn't
  depend on the model choosing to comply.
- **Graceful LLM degradation**: missing API key, network failure, HTTP 429
  (quota exhausted), and HTTP 402 (insufficient credits) are all caught and
  turned into a clear `degraded: true` response instead of a stack trace.

## Memory

`sessions.db` (SQLite) stores per-`session_id` message history and a
`preferences` JSON blob. Sessions inactive for longer than
`SESSION_TTL_HOURS` (default 6) are deleted on the next `/chat` call
(`memory.expire_stale_sessions`).

## Observability

Every turn is logged to `logs/turns.db`: the user message, every tool call
made (name, arguments, latency, a truncated result summary), total turn
latency, prompt/completion token counts (from OpenRouter's `usage` field), and
whether the turn succeeded. This is meant as the foundation for a future eval
harness, not a harness itself.

## Tests

```bash
cd agent_service
./.venv/Scripts/pytest
```

Covers guardrails (price bounds, tool-call budget, live-lookup labeling),
memory (session lifecycle, preference merging, expiry), and the pure tool
logic (`search_listings`/`get_market_stats` filtering, prediction rejection,
SHAP humanization) with the network calls monkeypatched out.

## Known limitations / tradeoffs

- **Free-tier LLM quotas are real.** OpenRouter's free models share a single
  50-requests/day cap *account-wide* (confirmed by testing — a second free
  model hit the identical `free-models-per-day` error, so switching models
  within OpenRouter does not help; only adding $10 credit, for 1000/day, or
  waiting for the daily reset does). Gemini's free tier has its own
  per-model limits. `LLM_PROVIDER` in `.env` switches between them with no
  code changes, since both expose an OpenAI-compatible endpoint.
- **Tool-calling support depends on the model.** Defaults to
  `openai/gpt-oss-20b:free` (OpenRouter, same model the n8n chatbot already
  uses) or `gemini-flash-latest` (Gemini) — both confirmed to support
  OpenAI-format tool calling through their respective endpoints.
- **`live_web_lookup` is fragile by nature** — it's a direct port of the regex
  selectors already in n8n's "Live Market Lookup" workflow, which the main
  README already flags as tied to avito.ma/agenz.ma's frontend builds. If
  either site redeploys, this (and the n8n version) may silently drop fields
  until re-verified.
- **Preference extraction is tool-call-based, not a separate NLP step** — the
  model is instructed to call `remember_preference` when the user states
  something worth remembering. Simple and explainable, but only as reliable
  as the model's judgment about what counts as "lasting."
