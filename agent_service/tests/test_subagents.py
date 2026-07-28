"""Tests for the Scout/Analyst sub-agent loops. The LLM call is monkeypatched
out (subagents.llm_client.call_llm) so these run offline / for free."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import llm_client
import subagents


def _completion_with_tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}],
            }
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _completion_with_text(text: str):
    return {"choices": [{"message": {"role": "assistant", "content": text}}], "usage": {}}


def test_scout_and_analyst_tool_schemas_are_disjoint_and_scoped():
    scout_names = {s["function"]["name"] for s in subagents.SCOUT_TOOL_SCHEMAS}
    analyst_names = {s["function"]["name"] for s in subagents.ANALYST_TOOL_SCHEMAS}
    assert scout_names == {"search_listings", "rag_search", "live_web_lookup", "list_recent_deals"}
    assert analyst_names == {"get_market_stats", "predict_rent", "explain_prediction", "get_neighborhood_tiers"}
    assert scout_names.isdisjoint(analyst_names)


def test_run_subagent_dispatches_tool_then_returns_final_answer(monkeypatch):
    calls = iter([
        _completion_with_tool_call("search_listings", {"city": "Rabat"}),
        _completion_with_text("Found 3 listings in Rabat."),
    ])
    monkeypatch.setattr(llm_client, "call_llm", lambda *a, **k: next(calls))
    monkeypatch.setitem(subagents.SUB_AGENT_TOOL_FUNCTIONS, "search_listings", lambda args: {"matchCount": 3, "results": []})

    result = subagents.run_scout("Any listings in Rabat?")
    assert result["answer"] == "Found 3 listings in Rabat."
    assert result["degraded"] is False


def test_run_subagent_flags_live_web_lookup_usage(monkeypatch):
    calls = iter([
        _completion_with_tool_call("live_web_lookup", {"city": "Essaouira"}),
        _completion_with_text("Found some live listings in Essaouira."),
    ])
    monkeypatch.setattr(llm_client, "call_llm", lambda *a, **k: next(calls))
    monkeypatch.setitem(subagents.SUB_AGENT_TOOL_FUNCTIONS, "live_web_lookup", lambda args: {"found": True, "results": []})

    result = subagents.run_scout("Anything in Essaouira?")
    assert result["used_live_lookup"] is True


def test_run_subagent_respects_its_own_tool_call_budget(monkeypatch):
    # 4 tool-call responses in a row (one more than SUB_AGENT_MAX_TOOL_CALLS), then a
    # final answer — the 4th tool call should be budget-blocked, never executed.
    call_count = {"n": 0}

    def fake_search(args):
        call_count["n"] += 1
        return {"matchCount": 0, "results": []}

    monkeypatch.setitem(subagents.SUB_AGENT_TOOL_FUNCTIONS, "search_listings", fake_search)

    responses = iter([
        _completion_with_tool_call("search_listings", {"city": "Rabat"}),
        _completion_with_tool_call("search_listings", {"city": "Rabat"}),
        _completion_with_tool_call("search_listings", {"city": "Rabat"}),
        _completion_with_tool_call("search_listings", {"city": "Rabat"}),  # 4th call: over budget
        _completion_with_text("Done."),
    ])
    monkeypatch.setattr(llm_client, "call_llm", lambda *a, **k: next(responses))

    result = subagents.run_scout("Search repeatedly")
    assert result["answer"] == "Done."
    assert call_count["n"] == subagents.SUB_AGENT_MAX_TOOL_CALLS  # 4th call was budget-blocked, never executed


def test_run_subagent_reports_degraded_on_llm_failure(monkeypatch):
    def raise_degraded(*a, **k):
        raise llm_client.DegradedResponseError("quota exhausted")
    monkeypatch.setattr(llm_client, "call_llm", raise_degraded)

    result = subagents.run_analyst("What's the average rent in Fes?")
    assert result["degraded"] is True
    assert "quota exhausted" in result["answer"]


def test_list_recent_deals_and_get_neighborhood_tiers_call_ml_service(monkeypatch):
    import requests as requests_module

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"count": 1, "deals": [{"city": "Casablanca"}]}

    monkeypatch.setattr(requests_module, "get", lambda *a, **k: FakeResponse())
    result = subagents.list_recent_deals({"threshold_pct": 20})
    assert result["count"] == 1
