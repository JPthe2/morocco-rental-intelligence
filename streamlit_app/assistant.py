"""Chat assistant for the Streamlit app.

Two modes:
  1. Local (default, zero-config) — a rule-based Q&A that answers the common
     market questions from frontend/data.json and can run the price model.
  2. LLM-backed (optional) — if OPENROUTER_API_KEY or GEMINI_API_KEY is present
     in the environment (or Streamlit secrets), runs an OpenAI-compatible
     tool-calling loop over the same data functions, mirroring agent_service.
     Falls back to the local answerer if the LLM call fails (quota, network...).

The same parser powers the natural-language filter box on the Listings tab.
"""

import json
import os
import re
from functools import lru_cache

import requests

import market_engine

GLOBAL_MIN_RENT_MAD = 500
GLOBAL_MAX_RENT_MAD = 100_000

_CITY_ALIASES = {
    "casa": "Casablanca", "casablanca": "Casablanca", "marrakech": "Marrakech",
    "marrakesh": "Marrakech", "rabat": "Rabat", "tanger": "Tanger",
    "tangier": "Tanger", "tangiers": "Tanger", "mohammedia": "Mohammedia",
    "agadir": "Agadir", "temara": "Temara", "dar bouazza": "Dar Bouazza",
    "bouskoura": "Bouskoura", "kenitra": "Kénitra", "kénitra": "Kénitra",
    "sale": "Salé", "salé": "Salé", "fes": "Fès", "fès": "Fès",
    "cabo negro": "Cabo Negro", "benslimane": "Benslimane",
    "el jadida": "El Jadida", "berrechid": "Berrechid", "martil": "Martil",
    "nouaceur": "Nouaceur", "bouznika": "Bouznika", "el mansouria": "El Mansouria",
}


@lru_cache(maxsize=1)
def _known_cities() -> list[str]:
    names = market_engine.known_city_names()
    for alias in _CITY_ALIASES.values():
        if alias not in names:
            names.append(alias)
    return names


def _fmt_mad(n) -> str:
    if n is None:
        return "—"
    return f"{round(n):,} MAD"


def extract_city(text: str) -> str | None:
    low = text.lower()
    for alias, canonical in sorted(_CITY_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True):
        if alias in low:
            return canonical
    for name in sorted(_known_cities(), key=len, reverse=True):
        if name.lower() in low:
            return name
    return None


def _extract_price(text: str) -> float | None:
    low = text.lower()
    numbers = [float(m) for m in re.findall(r"\d[\d\s]*", text.replace(",", ""))]
    numbers = [n for n in numbers if GLOBAL_MIN_RENT_MAD <= n <= GLOBAL_MAX_RENT_MAD]
    if not numbers:
        return None
    low_markers = ["under", "below", "max", "less than", "<", "cheaper than", "at most"]
    for marker in low_markers:
        idx = low.find(marker)
        if idx != -1:
            for n in numbers:
                if n >= 1000:
                    return n
    return min(numbers)


def _extract_bedrooms(text: str) -> int | None:
    m = re.search(r"(\d)\s*-?\s*(?:bed(?:room)?s?|chambres?|br)\b", text.lower())
    if m:
        return int(m.group(1))
    return None


def _extract_furnished(text: str) -> bool | None:
    low = text.lower()
    if "unfurnished" in low or "non furnished" in low or "meublé non" in low:
        return False
    if "furnished" in low or "meubl" in low:
        return True
    return None


def parse_nl_filter(query: str) -> dict:
    """Translate a free-text listing request into the Listings-tab filters.
    Returns {city, max_price, min_bedrooms, furnished} with None for unset."""
    return {
        "city": extract_city(query),
        "max_price": _extract_price(query),
        "min_bedrooms": _extract_bedrooms(query),
        "furnished": _extract_furnished(query),
    }


# ---------------------------------------------------------------------------
# Local rule-based answerer
# ---------------------------------------------------------------------------

def _city_stats_text(city: str | None) -> str:
    stats = market_engine.market_stats(city)
    if city and "city" in stats:
        return (
            f"{stats['city']}: {stats['city_count']:,} priced listings, "
            f"median rent {_fmt_mad(stats['city_median_rent_mad'])}, "
            f"average {_fmt_mad(stats['city_avg_rent_mad'])} "
            f"(avg surface {stats['city_avg_surface_m2']:.0f} m²)."
        )
    if city:
        return f"I don't have any scraped listings for '{city}' yet — the model may still guess a price, but the data behind it is thin."
    return (
        f"Overall: {stats['total_listings']:,} listings ({stats['by_site'].get('agenz', 0):,} agenz · "
        f"{stats['by_site'].get('avito', 0):,} avito), median rent {_fmt_mad(stats['median_rent_mad'])}, "
        f"average {_fmt_mad(stats['avg_rent_mad'])}."
    )


def _deals_text() -> str:
    result = market_engine.find_deals()
    if not result["deals"]:
        return "No deals flagged right now."
    lines = [f"{result['count']} potential deal(s) found (priced below the model's 80% confidence band). Top ones:"]
    for d in result["deals"][:5]:
        city = d["city"] or "?"
        lines.append(f"- {city} / {d['neighborhood'] or '—'}: {_fmt_mad(d['rent_price'])} vs predicted {_fmt_mad(d['predicted_rent_mad'])} (-{d['discount_pct']}%)")
    return "\n".join(lines)


def _predict_text(city: str, query: str) -> str:
    try:
        result = market_engine.predict_rent(
            city=city,
            bedrooms=_extract_bedrooms(query),
            furnished=_extract_furnished(query),
        )
    except Exception as exc:
        return f"Sorry, the price model errored: {exc}"
    low, high = result["confidence_interval_80pct"]
    warning = f"\n(Note: {result['warning']})" if result.get("warning") else ""
    return (
        f"My model predicts about {_fmt_mad(result['predicted_rent_mad'])}/month for that "
        f"{city} property (80% confidence range {_fmt_mad(low)} – {_fmt_mad(high)}).{warning}"
    )


def _cheapest_expensive() -> str:
    from data_loader import load_dashboard_data
    cities = [c for c in load_dashboard_data()["summary"]["cityStats"] if c["count"] >= 5]
    if not cities:
        return "Not enough per-city data."
    cheapest = min(cities, key=lambda c: c["medianRent"])
    priciest = max(cities, key=lambda c: c["medianRent"])
    return (
        f"Most affordable city by median rent: {cheapest['city']} ({_fmt_mad(cheapest['medianRent'])}, {cheapest['count']} listings). "
        f"Priciest: {priciest['city']} ({_fmt_mad(priciest['medianRent'])}, {priciest['count']} listings)."
    )


_CAPABILITIES = (
    "I can answer market questions from the scraped dataset — try things like:\n"
    "- \"average rent in Casablanca\"\n"
    "- \"cheapest city to rent in\"\n"
    "- \"how many listings are there\"\n"
    "- \"show me deals\"\n"
    "- \"predict rent for a 2-bedroom flat in Rabat\""
)

_WELCOME = (
    "Hi! I'm the market assistant for the Morocco rental dataset. " + _CAPABILITIES
    + "\n\nType your question in the box below."
)


def local_answer(query: str) -> str:
    low = query.lower()

    if not query.strip():
        return _WELCOME
    if low.strip() in {"hi", "hello", "hey", "salam", "bonjour", "مرحبا"}:
        return _WELCOME

    city = extract_city(query)

    if "deal" in low:
        return _deals_text()

    if "predict" in low or "estimate" in low or "how much would" in low:
        if city:
            return _predict_text(city, query)
        return "I can predict a rent if you tell me the city — e.g. \"predict rent for a 2-bedroom flat in Rabat\". Or use the Predict Price tab."

    if "cheap" in low and "city" in low:
        return _cheapest_expensive()
    if "expensive" in low:
        return _cheapest_expensive()

    if ("average" in low or "median" in low or "mean" in low or "rent in" in low
            or "price in" in low or "how much" in low):
        return _city_stats_text(city)

    if "how many" in low or "total" in low or "count" in low or "listings" in low and "many" in low:
        stats = market_engine.market_stats()
        return (
            f"I have {stats['total_listings']:,} scraped listings in total, "
            f"{stats['with_price_count']:,} with a confirmed price "
            f"({stats['by_site'].get('agenz', 0):,} from agenz.ma, {stats['by_site'].get('avito', 0):,} from avito.ma)."
        )

    return "I'm not sure about that one. " + _CAPABILITIES


# ---------------------------------------------------------------------------
# Optional LLM-backed tool-calling loop (OpenRouter or Gemini)
# ---------------------------------------------------------------------------

def llm_config() -> dict | None:
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            return None
        return {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": key,
            "model": os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
            "provider": "gemini",
        }
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return None
    return {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": key,
        "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free"),
        "provider": "openrouter",
    }


_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "market_stats",
            "description": "Overall or per-city market statistics (listings count, average/median rent).",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": ["string", "null"], "description": "Optional city name."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_listings",
            "description": "Search scraped listings with optional filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": ["string", "null"]},
                    "max_price": {"type": ["number", "null"]},
                    "min_bedrooms": {"type": ["integer", "null"]},
                    "keyword": {"type": ["string", "null"]},
                    "limit": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_rent",
            "description": "Predict monthly rent for a property using the trained ML model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "neighborhood": {"type": ["string", "null"]},
                    "surface_m2": {"type": ["number", "null"]},
                    "bedrooms": {"type": ["integer", "null"]},
                    "bathrooms": {"type": ["integer", "null"]},
                    "furnished": {"type": ["boolean", "null"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_deals",
            "description": "Listings priced well below what the model predicts (potential deals).",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}, "threshold_pct": {"type": "number"}},
            },
        },
    },
]


def _run_tool(name: str, args: dict) -> str:
    try:
        if name == "market_stats":
            return json.dumps(market_engine.market_stats(args.get("city")), ensure_ascii=False)
        if name == "search_listings":
            rows = market_engine.search_listings(
                city=args.get("city"),
                max_price=args.get("max_price"),
                min_bedrooms=args.get("min_bedrooms"),
                keyword=args.get("keyword"),
                limit=int(args.get("limit") or 20),
            )
            return json.dumps(rows, ensure_ascii=False)
        if name == "predict_rent":
            return json.dumps(
                market_engine.predict_rent(
                    city=args["city"],
                    neighborhood=args.get("neighborhood"),
                    surface_m2=args.get("surface_m2"),
                    bedrooms=args.get("bedrooms"),
                    bathrooms=args.get("bathrooms"),
                    furnished=args.get("furnished"),
                ),
                ensure_ascii=False,
            )
        if name == "top_deals":
            return json.dumps(
                market_engine.find_deals(
                    threshold_pct=float(args.get("threshold_pct") or 15),
                    limit=int(args.get("limit") or 5),
                )["deals"],
                ensure_ascii=False,
            )
        return f"Unknown tool: {name}"
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def llm_chat(message: str, history: list[dict], max_tool_calls: int = 4) -> dict:
    """One OpenAI-compatible tool-calling round trip. history is a list of
    {role: 'user'|'assistant', content} from the current session (excl. message)."""
    cfg = llm_config()
    if cfg is None:
        raise RuntimeError("No LLM API key configured.")

    system = (
        "You are the chat assistant for a Morocco rental market intelligence platform. "
        "You answer questions about scraped rental listings (agenz.ma + avito.ma) using the "
        "provided tools. Be concise, use MAD for prices, and note when the model's "
        "prediction may be unreliable."
    )
    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": message}]

    for _ in range(max_tool_calls):
        resp = requests.post(
            f"{cfg['base_url']}/chat/completions",
            headers={
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
            },
            json={"model": cfg["model"], "messages": messages, "tools": _TOOL_SCHEMAS, "tool_choice": "auto"},
            timeout=60,
        )
        if resp.status_code == 429:
            raise RuntimeError(f"{cfg['provider']}'s free-tier quota looks exhausted (HTTP 429).")
        if resp.status_code == 402:
            raise RuntimeError(f"{cfg['provider']} reports insufficient credits (HTTP 402).")
        if not resp.ok:
            raise RuntimeError(f"{cfg['provider']} request failed: HTTP {resp.status_code} — {resp.text[:300]}")

        payload = resp.json()
        choice = payload["choices"][0]["message"]
        if choice.get("tool_calls"):
            messages.append(choice)
            for tc in choice["tool_calls"]:
                fn = tc["function"]
                args = json.loads(fn.get("arguments") or "{}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": _run_tool(fn["name"], args),
                })
            continue
        return {"text": (choice.get("content") or "").strip(), "degraded": False}

    return {"text": "I couldn't finish that in the allowed number of steps.", "degraded": True}


def chat(message: str, history: list[dict]) -> dict:
    """Top-level entry point used by app.py. Returns {text, degraded}."""
    try:
        if llm_config() is not None:
            result = llm_chat(message, history)
            return result
    except Exception as exc:
        return {
            "text": local_answer(message) + f"\n\n[LLM mode unavailable ({exc}) — showing local answer.]",
            "degraded": True,
        }
    return {"text": local_answer(message), "degraded": False}
