"""The agent loop: a manual (non-framework) tool-calling loop against OpenRouter.

Deliberately hand-rolled rather than built on LangGraph or similar — call the
model, execute any tool calls it asks for, feed results back, repeat until it
gives a final answer or the per-turn tool-call budget is exhausted.
"""
import json

import requests

import config
import guardrails
import memory
import observability
import tools

SYSTEM_PROMPT = """You are a rental market assistant for Morocco, backed by real listings scraped from agenz.ma and avito.ma. You have tools to search stored listings, compute market stats, predict rent for hypothetical properties (with an explanation of price drivers), semantically search listing descriptions for fuzzy requests, and — as a last resort — fetch live listings directly from the web.

Rules:
- Always use a tool to ground any numeric answer (rent averages, comparable listings, price predictions). Never state a rent figure that did not come from a tool result.
- Prefer search_listings / get_market_stats first. Use rag_search for fuzzy, qualitative requests structured filters can't express (e.g. "quiet", "near a school"). Use predict_rent / explain_prediction only for hypothetical properties not in the dataset. Only use live_web_lookup if the stored dataset returned zero matches for the city/area asked about.
- If the user states a lasting preference (budget, minimum bedrooms, preferred city, etc.), call remember_preference so it isn't lost between turns.
- Live web lookups are an unverified snapshot — always say so when you use one.
- Keep answers concise and mention how many listings a statistic is based on."""


class DegradedResponseError(Exception):
    """Raised when the LLM backend can't be used (missing key, quota exhausted,
    network error). Caught by run_turn to return an honest message instead of
    a crash."""


def _call_llm(messages: list[dict], tool_schemas: list[dict] | None = None, tool_choice="auto") -> dict:
    if not config.LLM_API_KEY:
        raise DegradedResponseError(
            f"No API key is configured for LLM_PROVIDER={config.LLM_PROVIDER}. "
            "Set it in agent_service/.env to enable chat."
        )
    try:
        resp = requests.post(
            f"{config.LLM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.LLM_MODEL,
                "messages": messages,
                "tools": tool_schemas if tool_schemas is not None else tools.TOOL_SCHEMAS,
                "tool_choice": tool_choice,
            },
            timeout=60,
        )
    except requests.RequestException as exc:
        raise DegradedResponseError(f"Could not reach {config.LLM_PROVIDER}: {exc}") from exc

    if resp.status_code == 429:
        raise DegradedResponseError(
            f"{config.LLM_PROVIDER}'s free-tier quota looks exhausted right now (HTTP 429). "
            "Try again shortly, or switch LLM_PROVIDER / add credits in agent_service/.env."
        )
    if resp.status_code == 402:
        raise DegradedResponseError(f"{config.LLM_PROVIDER} reports insufficient credits for this model (HTTP 402).")
    if not resp.ok:
        raise DegradedResponseError(f"{config.LLM_PROVIDER} request failed: HTTP {resp.status_code} — {resp.text[:300]}")

    return resp.json()


def _dispatch_tool_call(name: str, args: dict, session_id: str) -> dict:
    if name == "remember_preference":
        key, value = args.get("key"), args.get("value")
        if key:
            memory.update_preferences(session_id, {key: value})
        return tools.remember_preference(args)

    fn = tools.TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        return fn(args)
    except requests.RequestException as exc:
        return {"error": f"Tool '{name}' failed to reach its backing service: {exc}"}
    except Exception as exc:  # noqa: BLE001 - tool failures must degrade, not crash the turn
        return {"error": f"Tool '{name}' raised an unexpected error: {exc}"}


def run_turn(session_id: str, user_message: str) -> dict:
    """Run one full agent turn (possibly several tool round-trips).
    Returns {"response_text": str, "degraded": bool}."""
    memory.touch_session(session_id)
    memory.append_message(session_id, "user", user_message)

    preferences = memory.get_preferences(session_id)
    history = memory.get_history(session_id, limit=20)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if preferences:
        messages.append({"role": "system", "content": f"Known user preferences: {json.dumps(preferences)}"})
    messages.extend(history)

    logger = observability.TurnLogger(session_id, user_message)
    budget = guardrails.ToolCallBudget()
    used_live_lookup = False

    try:
        while True:
            completion = _call_llm(messages)
            usage = completion.get("usage", {})
            logger.add_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

            choice = completion["choices"][0]
            message = choice["message"]
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                response_text = message.get("content") or ""
                response_text = guardrails.ensure_live_lookup_disclosed(response_text, used_live_lookup)
                memory.append_message(session_id, "assistant", response_text)
                logger.finish(success=True)
                return {"response_text": response_text, "degraded": False}

            messages.append(message)

            if budget.exhausted:
                messages.append({
                    "role": "system",
                    "content": "You have used the maximum number of tool calls for this turn. "
                               "Give your best final answer now using what you already have, "
                               "and say clearly if something couldn't be verified.",
                })

            for call in tool_calls:
                fn_name = call["function"]["name"]
                try:
                    fn_args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    fn_args = {}

                if fn_name == "live_web_lookup":
                    used_live_lookup = True

                if not budget.try_consume():
                    result = {"error": "Tool call budget exhausted for this turn."}
                else:
                    with logger.record_tool_call(fn_name, fn_args) as record:
                        result = _dispatch_tool_call(fn_name, fn_args, session_id)
                        record["result_summary"] = str(result)[:300]

                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, default=str),
                })
    except DegradedResponseError as exc:
        logger.finish(success=False, error=str(exc))
        return {"response_text": str(exc), "degraded": True}
    except Exception as exc:  # noqa: BLE001 - a turn must never 500, always report degraded
        logger.finish(success=False, error=str(exc))
        return {"response_text": f"Something went wrong handling that: {exc}", "degraded": True}


FILTER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "apply_filters",
        "description": "Extract structured rental search filters from a natural-language request.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name if mentioned, exact casing e.g. Casablanca, Rabat, Marrakech"},
                "max_price": {"type": "number", "description": "Maximum monthly rent in MAD if mentioned"},
                "min_bedrooms": {"type": "number", "description": "Minimum number of bedrooms if mentioned"},
            },
        },
    },
}


def extract_filters(query: str) -> dict:
    """One-shot (non-agentic) LLM call that translates a natural-language search
    request into the structured filters the Listings tab already supports
    (city / max_price / min_bedrooms) — no tool loop, no memory, just a single
    forced function call. Returns {"filters": {...}, "degraded": bool}."""
    messages = [
        {
            "role": "system",
            "content": "Extract rental search filters from the user's request by calling apply_filters. "
                       "Only include fields the user actually specified — omit anything not mentioned.",
        },
        {"role": "user", "content": query},
    ]
    try:
        completion = _call_llm(
            messages,
            tool_schemas=[FILTER_TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "apply_filters"}},
        )
    except DegradedResponseError as exc:
        return {"filters": {}, "degraded": True, "error": str(exc)}

    message = completion["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return {"filters": {}, "degraded": False}
    try:
        filters = json.loads(tool_calls[0]["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        filters = {}
    return {"filters": filters, "degraded": False}
