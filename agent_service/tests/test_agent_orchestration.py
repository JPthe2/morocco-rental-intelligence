"""Tests for agent.run_turn (the Concierge) after the multi-agent rewrite:
delegation dispatch, and — the trickiest part of the refactor — that the
live-lookup guardrail still fires correctly when the *sub-agent*, not
Concierge itself, is the one that used live_web_lookup. LLM calls and
sub-agent runs are monkeypatched out so these run offline / for free.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent
import config
import subagents


def _completion_with_tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}],
            }
        }],
        "usage": {},
    }


def _completion_with_text(text: str):
    return {"choices": [{"message": {"role": "assistant", "content": text}}], "usage": {}}


def _isolate_dbs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS_DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(config, "LOGS_DB_PATH", tmp_path / "turns.db")


def test_run_turn_propagates_live_lookup_flag_from_scout(tmp_path, monkeypatch):
    _isolate_dbs(tmp_path, monkeypatch)
    responses = iter([
        _completion_with_tool_call("ask_scout", {"question": "anything in Essaouira?"}),
        _completion_with_text("Here's what I found in Essaouira."),  # doesn't mention "live"/"unverified" itself
    ])
    monkeypatch.setattr(agent, "_call_llm", lambda *a, **k: next(responses))
    monkeypatch.setattr(
        subagents, "run_scout",
        lambda question, parent_logger=None: {"answer": "One live listing found.", "used_live_lookup": True, "degraded": False},
    )

    result = agent.run_turn("test-session-live", "Anything available in Essaouira?")
    assert result["degraded"] is False
    assert "unverified" in result["response_text"].lower()


def test_run_turn_does_not_add_disclaimer_when_scout_used_no_live_lookup(tmp_path, monkeypatch):
    _isolate_dbs(tmp_path, monkeypatch)
    responses = iter([
        _completion_with_tool_call("ask_scout", {"question": "anything in Rabat?"}),
        _completion_with_text("Found 3 listings in Rabat."),
    ])
    monkeypatch.setattr(agent, "_call_llm", lambda *a, **k: next(responses))
    monkeypatch.setattr(
        subagents, "run_scout",
        lambda question, parent_logger=None: {"answer": "3 listings.", "used_live_lookup": False, "degraded": False},
    )

    result = agent.run_turn("test-session-no-live", "Anything in Rabat?")
    assert result["response_text"] == "Found 3 listings in Rabat."


def test_run_turn_dispatches_remember_preference_directly(tmp_path, monkeypatch):
    _isolate_dbs(tmp_path, monkeypatch)
    responses = iter([
        _completion_with_tool_call("remember_preference", {"key": "preferred_city", "value": "Marrakech"}),
        _completion_with_text("Got it, I'll remember that."),
    ])
    monkeypatch.setattr(agent, "_call_llm", lambda *a, **k: next(responses))

    result = agent.run_turn("test-session-pref", "I like Marrakech")
    assert result["response_text"] == "Got it, I'll remember that."

    import memory
    assert memory.get_preferences("test-session-pref") == {"preferred_city": "Marrakech"}


def test_concierge_tool_schemas_are_delegation_only():
    names = {s["function"]["name"] for s in agent.CONCIERGE_TOOL_SCHEMAS}
    assert names == {"remember_preference", "ask_scout", "ask_analyst"}
