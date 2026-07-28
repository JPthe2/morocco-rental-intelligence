"""Guardrails for the agent loop: price sanity checks, a per-turn tool-call
budget, and enforced labeling of live/unverified data. These are checked in
code, not left to prompt instructions, since an LLM can't be trusted to
self-enforce safety constraints.
"""
from typing import Optional

import config

LIVE_LOOKUP_DISCLAIMER = (
    "This data was fetched live just now from avito.ma/agenz.ma and is an "
    "unverified snapshot, not the full scraped dataset."
)


def compute_city_bounds(listings: list[dict], city: str) -> tuple[float, float]:
    """Derive a plausible [low, high] MAD rent range for a city from local listings.
    Falls back to a generous global range when there isn't enough local data
    (fewer than 5 listings) to trust a city-specific median."""
    prices = sorted(
        l["rent_price"] for l in listings
        if isinstance(l.get("rent_price"), (int, float)) and (l.get("city") or "").lower() == city.lower()
    )
    if len(prices) < 5:
        return float(config.GLOBAL_MIN_RENT_MAD), float(config.GLOBAL_MAX_RENT_MAD)
    median = prices[len(prices) // 2]
    low = max(config.GLOBAL_MIN_RENT_MAD, median * 0.15)
    high = min(config.GLOBAL_MAX_RENT_MAD, median * 6)
    return float(low), float(high)


def validate_price(price: Optional[float], city: str, listings: list[dict]) -> tuple[bool, str]:
    """Return (is_valid, reason). reason is empty when valid."""
    if price is None or not isinstance(price, (int, float)) or price <= 0:
        return False, "Price must be a positive number."
    low, high = compute_city_bounds(listings, city)
    if price < low or price > high:
        return False, (
            f"Predicted rent {price:.0f} MAD is outside the plausible range for "
            f"{city} ({low:.0f}-{high:.0f} MAD) and was rejected rather than shown as fact."
        )
    return True, ""


def label_live_lookup_result(result: dict) -> dict:
    """Stamp a live_web_lookup tool result with an unverified-source marker
    that survives regardless of what the model does with it."""
    return {**result, "source": "live_web_lookup", "unverified": True, "disclaimer": LIVE_LOOKUP_DISCLAIMER}


def ensure_live_lookup_disclosed(response_text: str, used_live_lookup: bool) -> str:
    """If live_web_lookup was used this turn but the model's final answer doesn't
    mention it, append the disclaimer in code so it's never silently dropped."""
    if used_live_lookup and "unverified" not in response_text.lower() and "live" not in response_text.lower():
        return response_text + f"\n\n⚠️ Note: {LIVE_LOOKUP_DISCLAIMER}"
    return response_text


class ToolCallBudget:
    """Caps tool calls per agent turn so a confused model can't loop forever."""

    def __init__(self, max_calls: Optional[int] = None):
        self.max_calls = max_calls if max_calls is not None else config.MAX_TOOL_CALLS_PER_TURN
        self.used = 0

    def try_consume(self) -> bool:
        if self.used >= self.max_calls:
            return False
        self.used += 1
        return True

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_calls
