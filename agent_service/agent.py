"""Concierge: the agent the user actually talks to. Deliberately holds almost
no tools itself — it delegates to two specialist sub-agents (subagents.py:
Scout for finding/monitoring listings, Analyst for stats/predictions/trends)
and synthesizes their answers. A plain orchestrator/workers pattern: one
supervisor agent, two specialists exposed to it as callable tools, each
running the same kind of manual tool-calling loop at a smaller scope — no
framework, still explainable end to end.
"""
import json

import guardrails
import llm_client
import memory
import observability
import subagents
import tools

# Re-exported so existing callers/tests (`agent._call_llm`, `agent.DegradedResponseError`)
# keep working — llm_client.py is the actual implementation, shared with subagents.py.
_call_llm = llm_client.call_llm
DegradedResponseError = llm_client.DegradedResponseError

SYSTEM_PROMPT = """You are Concierge, the conversational front door for a Morocco rental market intelligence system. You don't search listings or compute statistics yourself — you delegate to two specialist sub-agents and synthesize their answers for the user:
- ask_scout: for finding/monitoring listings — structured search, semantic/fuzzy search (e.g. "quiet", "near a school"), live web lookups for areas not in the stored dataset, and recently flagged underpriced deals.
- ask_analyst: for market statistics, ML rent predictions (with explanations of price drivers), and neighborhood market-tier analysis.

Rules:
- Always delegate to Scout or Analyst before answering any numeric question (rent averages, listings, price predictions) — never state a rent figure that didn't come from a sub-agent's answer.
- If the user states a lasting preference (budget, minimum bedrooms, preferred city, etc.), call remember_preference so it isn't lost between turns.
- If a sub-agent's answer mentions a live/unverified lookup, relay that caveat to the user — don't drop it.
- Keep answers concise and mention how many listings a statistic is based on when the sub-agent provides that."""

ASK_SCOUT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_scout",
        "description": "Delegate to Scout, a specialist sub-agent for finding/monitoring rental listings (structured search, semantic search, live web lookups, recently flagged deals). Ask it one specific question.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "The specific question to ask Scout"}},
            "required": ["question"],
        },
    },
}

ASK_ANALYST_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_analyst",
        "description": "Delegate to Analyst, a specialist sub-agent for market statistics, ML price predictions with explanations, and neighborhood tier analysis. Ask it one specific question.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "The specific question to ask Analyst"}},
            "required": ["question"],
        },
    },
}

_REMEMBER_PREFERENCE_SCHEMA = next(s for s in tools.TOOL_SCHEMAS if s["function"]["name"] == "remember_preference")
CONCIERGE_TOOL_SCHEMAS = [_REMEMBER_PREFERENCE_SCHEMA, ASK_SCOUT_SCHEMA, ASK_ANALYST_SCHEMA]


def run_turn(session_id: str, user_message: str) -> dict:
    """Run one full Concierge turn (possibly several delegations to Scout/Analyst,
    each of which may itself make several tool calls). Returns
    {"response_text": str, "degraded": bool}."""
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
            completion = _call_llm(messages, tool_schemas=CONCIERGE_TOOL_SCHEMAS)
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
                    "content": "You have used the maximum number of delegations for this turn. "
                               "Give your best final answer now using what you already have, "
                               "and say clearly if something couldn't be verified.",
                })

            for call in tool_calls:
                fn_name = call["function"]["name"]
                try:
                    fn_args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    fn_args = {}

                if not budget.try_consume():
                    result = {"error": "Delegation budget exhausted for this turn."}
                elif fn_name == "remember_preference":
                    with logger.record_tool_call(fn_name, fn_args) as record:
                        key, value = fn_args.get("key"), fn_args.get("value")
                        if key:
                            memory.update_preferences(session_id, {key: value})
                        result = tools.remember_preference(fn_args)
                        record["result_summary"] = str(result)[:300]
                elif fn_name == "ask_scout":
                    with logger.record_tool_call(fn_name, fn_args) as record:
                        sub_result = subagents.run_scout(fn_args.get("question", ""), parent_logger=logger)
                        used_live_lookup = used_live_lookup or sub_result.get("used_live_lookup", False)
                        result = {"answer": sub_result["answer"]}
                        record["result_summary"] = str(result)[:300]
                elif fn_name == "ask_analyst":
                    with logger.record_tool_call(fn_name, fn_args) as record:
                        sub_result = subagents.run_analyst(fn_args.get("question", ""), parent_logger=logger)
                        result = {"answer": sub_result["answer"]}
                        record["result_summary"] = str(result)[:300]
                else:
                    result = {"error": f"Unknown tool '{fn_name}'."}

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
