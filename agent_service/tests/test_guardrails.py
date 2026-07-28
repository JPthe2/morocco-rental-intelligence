import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import guardrails

LISTINGS = [{"city": "Casablanca", "rent_price": p} for p in [4000, 4500, 5000, 5500, 6000, 6500, 7000]]


def test_validate_price_within_range_is_accepted():
    ok, reason = guardrails.validate_price(5200, "Casablanca", LISTINGS)
    assert ok
    assert reason == ""


def test_validate_price_rejects_absurdly_high_value():
    ok, reason = guardrails.validate_price(500_000, "Casablanca", LISTINGS)
    assert not ok
    assert "outside the plausible range" in reason


def test_validate_price_rejects_non_positive():
    ok, _ = guardrails.validate_price(-100, "Casablanca", LISTINGS)
    assert not ok


def test_validate_price_falls_back_to_global_bounds_for_unknown_city():
    ok, _ = guardrails.validate_price(2000, "Nowhereville", [])
    assert ok  # within the generous global default range


def test_tool_call_budget_stops_after_max():
    budget = guardrails.ToolCallBudget(max_calls=2)
    assert budget.try_consume() is True
    assert budget.try_consume() is True
    assert budget.try_consume() is False
    assert budget.exhausted


def test_live_lookup_result_is_labeled_unverified():
    labeled = guardrails.label_live_lookup_result({"found": True, "results": []})
    assert labeled["unverified"] is True
    assert "disclaimer" in labeled


def test_ensure_live_lookup_disclosed_appends_when_missing():
    text = guardrails.ensure_live_lookup_disclosed("Here are some listings.", used_live_lookup=True)
    assert "unverified" in text.lower()


def test_ensure_live_lookup_disclosed_noop_when_not_used():
    text = guardrails.ensure_live_lookup_disclosed("Here are some listings.", used_live_lookup=False)
    assert text == "Here are some listings."
