"""Tests for agent.extract_filters (the one-shot NL-to-filter translator).
The LLM call itself is monkeypatched out so these run offline / for free."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent


def _fake_completion_with_tool_call(arguments: dict):
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "apply_filters", "arguments": json.dumps(arguments)},
                }],
            }
        }]
    }


def test_extract_filters_parses_tool_call_arguments(monkeypatch):
    monkeypatch.setattr(
        agent, "_call_llm",
        lambda *a, **k: _fake_completion_with_tool_call({"city": "Rabat", "max_price": 7000, "min_bedrooms": 3}),
    )
    result = agent.extract_filters("3-bedroom apartments in Rabat under 7000 MAD")
    assert result["filters"] == {"city": "Rabat", "max_price": 7000, "min_bedrooms": 3}
    assert result["degraded"] is False


def test_extract_filters_omits_unmentioned_fields(monkeypatch):
    monkeypatch.setattr(agent, "_call_llm", lambda *a, **k: _fake_completion_with_tool_call({"city": "Marrakech"}))
    result = agent.extract_filters("something in Marrakech")
    assert result["filters"] == {"city": "Marrakech"}


def test_extract_filters_handles_missing_tool_call(monkeypatch):
    fake_completion = {"choices": [{"message": {"role": "assistant", "content": "unsure"}}]}
    monkeypatch.setattr(agent, "_call_llm", lambda *a, **k: fake_completion)
    result = agent.extract_filters("asdkfjh")
    assert result["filters"] == {}
    assert result["degraded"] is False


def test_extract_filters_reports_degraded_on_llm_failure(monkeypatch):
    def raise_degraded(*a, **k):
        raise agent.DegradedResponseError("quota exhausted")
    monkeypatch.setattr(agent, "_call_llm", raise_degraded)
    result = agent.extract_filters("2-bedroom in Fes")
    assert result["degraded"] is True
    assert result["filters"] == {}
