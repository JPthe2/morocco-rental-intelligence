"""Multi-agent orchestration: the Concierge (agent.py, the agent users talk to)
delegates to two specialist sub-agents instead of holding every tool itself —
Scout (finds/monitors listings) and Analyst (market stats, predictions,
trends). Each runs its own small, bounded tool-calling loop over a
*restricted* toolset. This is a plain orchestrator/workers pattern: one
supervisor agent, two specialists exposed to it as callable tools — not a
framework, just the same manual loop from agent.py reused at a smaller scope.
"""
import json

import requests

import config
import guardrails
import llm_client
import tools

SUB_AGENT_MAX_TOOL_CALLS = 3

SCOUT_SYSTEM_PROMPT = (
    "You are Scout, a specialist sub-agent that finds and monitors rental listings "
    "for a Concierge agent. You have search_listings (structured search of the stored "
    "dataset), rag_search (semantic search for fuzzy/qualitative requests), "
    "live_web_lookup (last resort — only if the stored dataset has nothing for the "
    "city/area asked about), and list_recent_deals (listings flagged as priced well "
    "below the ML model's prediction). Answer the question you were asked, concisely "
    "and factually, grounded only in tool results. Never state a rent figure that "
    "didn't come from a tool call. If you used live_web_lookup, say so explicitly."
)

ANALYST_SYSTEM_PROMPT = (
    "You are Analyst, a specialist sub-agent that interprets market trends and "
    "answers quantitative questions for a Concierge agent. You have get_market_stats "
    "(averages/medians for a city/area), predict_rent and explain_prediction (ML price "
    "prediction, with SHAP-based reasoning, for hypothetical properties), and "
    "get_neighborhood_tiers (KMeans-clustered market tiers — Value/Mid-range/"
    "Family-oriented/Premium). Answer the question you were asked, concisely and "
    "factually, grounded only in tool results."
)


def list_recent_deals(args: dict) -> dict:
    params = {k: v for k, v in args.items() if v is not None}
    resp = requests.get(f"{config.ML_SERVICE_URL}/deals", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_neighborhood_tiers(args: dict) -> dict:
    resp = requests.get(f"{config.ML_SERVICE_URL}/neighborhood-tiers", timeout=15)
    resp.raise_for_status()
    return resp.json()


LIST_RECENT_DEALS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_recent_deals",
        "description": "List rental listings currently flagged as priced well below the ML model's predicted price for a comparable property (potential deals).",
        "parameters": {
            "type": "object",
            "properties": {
                "threshold_pct": {"type": "number", "description": "Minimum discount %, default 15"},
                "limit": {"type": "number", "description": "Max results, default 50"},
            },
        },
    },
}

GET_NEIGHBORHOOD_TIERS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_neighborhood_tiers",
        "description": "Get neighborhoods grouped into market tiers (Value/Mid-range/Family-oriented/Premium) via KMeans clustering on price/m2, bedrooms, and amenities.",
        "parameters": {"type": "object", "properties": {}},
    },
}

_SCOUT_TOOL_NAMES = {"search_listings", "rag_search", "live_web_lookup"}
_ANALYST_TOOL_NAMES = {"get_market_stats", "predict_rent", "explain_prediction"}

SCOUT_TOOL_SCHEMAS = [s for s in tools.TOOL_SCHEMAS if s["function"]["name"] in _SCOUT_TOOL_NAMES] + [LIST_RECENT_DEALS_SCHEMA]
ANALYST_TOOL_SCHEMAS = [s for s in tools.TOOL_SCHEMAS if s["function"]["name"] in _ANALYST_TOOL_NAMES] + [GET_NEIGHBORHOOD_TIERS_SCHEMA]

SUB_AGENT_TOOL_FUNCTIONS = {
    **tools.TOOL_FUNCTIONS,
    "list_recent_deals": list_recent_deals,
    "get_neighborhood_tiers": get_neighborhood_tiers,
}


def _safe_call(fn_name: str, fn_args: dict) -> dict:
    fn = SUB_AGENT_TOOL_FUNCTIONS.get(fn_name)
    if fn is None:
        return {"error": f"Unknown tool '{fn_name}'."}
    try:
        return fn(fn_args)
    except requests.RequestException as exc:
        return {"error": f"Tool '{fn_name}' failed to reach its backing service: {exc}"}
    except Exception as exc:  # noqa: BLE001 - tool failures must degrade, not crash the sub-agent
        return {"error": f"Tool '{fn_name}' raised an unexpected error: {exc}"}


def run_subagent(name: str, system_prompt: str, tool_schemas: list[dict], question: str, parent_logger=None) -> dict:
    """Run a small, bounded tool-calling loop for one specialist sub-agent.
    Returns {"answer": str, "used_live_lookup": bool, "degraded": bool}.
    If parent_logger is given (an observability.TurnLogger from the calling
    Concierge turn), the sub-agent's own tool calls are logged into it too,
    name-prefixed, so one turn's log shows the full delegation tree."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    budget = guardrails.ToolCallBudget(max_calls=SUB_AGENT_MAX_TOOL_CALLS)
    used_live_lookup = False

    try:
        while True:
            completion = llm_client.call_llm(messages, tool_schemas=tool_schemas, tool_choice="auto")
            if parent_logger is not None:
                usage = completion.get("usage", {})
                parent_logger.add_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

            message = completion["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                return {"answer": message.get("content") or "", "used_live_lookup": used_live_lookup, "degraded": False}

            messages.append(message)

            if budget.exhausted:
                messages.append({
                    "role": "system",
                    "content": f"{name} has used its maximum tool calls for this delegation. Give your best final answer now.",
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
                    result = {"error": f"{name}'s tool call budget exhausted for this delegation."}
                elif parent_logger is not None:
                    with parent_logger.record_tool_call(f"{name.lower()}:{fn_name}", fn_args) as record:
                        result = _safe_call(fn_name, fn_args)
                        record["result_summary"] = str(result)[:300]
                else:
                    result = _safe_call(fn_name, fn_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, default=str),
                })
    except llm_client.DegradedResponseError as exc:
        return {"answer": f"{name} could not respond: {exc}", "used_live_lookup": used_live_lookup, "degraded": True}


def run_scout(question: str, parent_logger=None) -> dict:
    return run_subagent("Scout", SCOUT_SYSTEM_PROMPT, SCOUT_TOOL_SCHEMAS, question, parent_logger=parent_logger)


def run_analyst(question: str, parent_logger=None) -> dict:
    return run_subagent("Analyst", ANALYST_SYSTEM_PROMPT, ANALYST_TOOL_SCHEMAS, question, parent_logger=parent_logger)
