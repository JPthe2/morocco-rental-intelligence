"""Tool implementations available to the agent, plus their OpenAI-format schemas.

search_listings / get_market_stats mirror the logic of the n8n Code Tool nodes
of the same name (they read the same listings data). predict_rent and
explain_prediction call the existing ML service. rag_search calls the Phase 1
RAG service. live_web_lookup is a Python port of the n8n "Live Market Lookup"
sub-workflow's regex-based extraction for avito.ma and agenz.ma.
"""
import json
import re
import time
from pathlib import Path
from typing import Optional

import requests

import config
import guardrails

_listings_cache: dict = {"data": None, "fetched_at": 0.0}
_LISTINGS_CACHE_TTL_SECONDS = 60

# Same fallback path rag_service uses: if the ML service's /export endpoint isn't
# reachable (e.g. an older running instance predating that route), read the
# snapshot file it would have served directly instead of hard-failing every tool.
_FALLBACK_SNAPSHOT_PATH = Path(__file__).parent.parent / "ml" / "price_prediction" / "data" / "listings_snapshot.json"


def get_listings(force_refresh: bool = False) -> list[dict]:
    now = time.time()
    if force_refresh or _listings_cache["data"] is None or now - _listings_cache["fetched_at"] > _LISTINGS_CACHE_TTL_SECONDS:
        try:
            resp = requests.get(f"{config.ML_SERVICE_URL}/export", timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            if not _FALLBACK_SNAPSHOT_PATH.exists():
                raise
            with open(_FALLBACK_SNAPSHOT_PATH, encoding="utf-8") as f:
                data = json.load(f)
        _listings_cache["data"] = data
        _listings_cache["fetched_at"] = now
    return _listings_cache["data"]


# ---------------------------------------------------------------------------
# search_listings
# ---------------------------------------------------------------------------

def search_listings(args: dict) -> dict:
    rows = get_listings()
    filtered = rows
    city = args.get("city")
    neighborhood = args.get("neighborhood")
    max_price = args.get("max_price")
    min_bedrooms = args.get("min_bedrooms")
    limit = int(args.get("limit") or 5)

    if city:
        filtered = [r for r in filtered if (r.get("city") or "").lower() == str(city).lower()]
    if neighborhood:
        filtered = [r for r in filtered if neighborhood.lower() in (r.get("neighborhood") or "").lower()]
    if max_price is not None:
        filtered = [r for r in filtered if r.get("rent_price") is not None and r["rent_price"] <= float(max_price)]
    if min_bedrooms is not None:
        filtered = [r for r in filtered if r.get("bedrooms") is not None and r["bedrooms"] >= float(min_bedrooms)]

    filtered = sorted(filtered, key=lambda r: r.get("rent_price") or 0)
    results = [
        {
            "site": r.get("source_site"),
            "city": r.get("city"),
            "neighborhood": r.get("neighborhood"),
            "rent_price": r.get("rent_price"),
            "surface_m2": r.get("surface_m2"),
            "bedrooms": r.get("bedrooms"),
            "bathrooms": r.get("bathrooms"),
            "url": r.get("source_url"),
        }
        for r in filtered[:limit]
    ]
    return {"matchCount": len(filtered), "results": results}


# ---------------------------------------------------------------------------
# get_market_stats
# ---------------------------------------------------------------------------

def get_market_stats(args: dict) -> dict:
    rows = get_listings()
    filtered = rows
    city = args.get("city")
    neighborhood = args.get("neighborhood")
    bedrooms = args.get("bedrooms")
    property_type = args.get("property_type")

    if city:
        filtered = [r for r in filtered if (r.get("city") or "").lower() == str(city).lower()]
    if neighborhood:
        filtered = [r for r in filtered if neighborhood.lower() in (r.get("neighborhood") or "").lower()]
    if bedrooms is not None:
        filtered = [r for r in filtered if r.get("bedrooms") == int(bedrooms)]
    if property_type:
        filtered = [r for r in filtered if r.get("property_type") == property_type]

    prices = sorted(r["rent_price"] for r in filtered if isinstance(r.get("rent_price"), (int, float)))
    surfaces = [r["surface_m2"] for r in filtered if isinstance(r.get("surface_m2"), (int, float))]

    def median(values):
        if not values:
            return None
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    return {
        "matchCount": len(filtered),
        "avgRentMAD": round(sum(prices) / len(prices)) if prices else None,
        "medianRentMAD": round(median(prices)) if prices else None,
        "minRentMAD": prices[0] if prices else None,
        "maxRentMAD": prices[-1] if prices else None,
        "avgSurfaceM2": round(sum(surfaces) / len(surfaces)) if surfaces else None,
    }


# ---------------------------------------------------------------------------
# predict_rent / explain_prediction
# ---------------------------------------------------------------------------

def _call_predict(args: dict) -> dict:
    payload = {
        "city": args.get("city"),
        "neighborhood": args.get("neighborhood"),
        "property_type": args.get("property_type") or "appartement",
        "surface_m2": args.get("surface_m2"),
        "bedrooms": args.get("bedrooms"),
        "bathrooms": args.get("bathrooms"),
        "furnished": args.get("furnished"),
        "amenities": args.get("amenities"),
    }
    resp = requests.post(f"{config.ML_SERVICE_URL}/predict", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def predict_rent(args: dict) -> dict:
    city = args.get("city")
    if not city:
        return {"error": "city is required to predict a rent."}
    prediction = _call_predict(args)
    listings = get_listings()
    is_valid, reason = guardrails.validate_price(prediction.get("predicted_rent_mad"), city, listings)
    if not is_valid:
        return {"error": f"Prediction rejected by guardrail: {reason}", "raw_prediction": prediction}
    return prediction


def explain_prediction(args: dict) -> dict:
    prediction = _call_predict(args)
    drivers = prediction.get("top_drivers") or []
    if not drivers:
        return {
            "predicted_rent_mad": prediction.get("predicted_rent_mad"),
            "explanation": prediction.get("warning") or "No per-prediction feature breakdown is available for this model.",
        }

    def humanize(feature_name: str) -> str:
        name = feature_name.split("__", 1)[-1]  # drop sklearn ColumnTransformer prefix
        if name.startswith("city_"):
            return f"being located in {name[len('city_'):]}"
        if name.startswith("neighborhood_"):
            return f"the {name[len('neighborhood_'):]} neighborhood"
        if name.startswith("amenities_"):
            return f"having {name[len('amenities_'):]}"
        if name.startswith("property_type_"):
            return f"being a {name[len('property_type_'):]}"
        return name.replace("_", " ")

    positive = [d for d in drivers if d.get("impact_on_log_price", 0) > 0]
    negative = [d for d in drivers if d.get("impact_on_log_price", 0) < 0]

    sentence = "This listing's predicted price is "
    if positive:
        sentence += f"pushed up mainly by {humanize(positive[0]['feature'])}"
        if len(positive) > 1:
            sentence += f" and {humanize(positive[1]['feature'])}"
    else:
        sentence += "close to the model's baseline for comparable properties"
    if negative:
        sentence += f", offset by {humanize(negative[0]['feature'])}"
    sentence += "."

    return {
        "predicted_rent_mad": prediction.get("predicted_rent_mad"),
        "confidence_interval_80pct": prediction.get("confidence_interval_80pct"),
        "explanation": sentence,
        "raw_drivers": drivers,
    }


# ---------------------------------------------------------------------------
# rag_search
# ---------------------------------------------------------------------------

def rag_search(args: dict) -> dict:
    payload = {
        "query": args.get("query", ""),
        "top_k": int(args.get("top_k") or 5),
    }
    if args.get("city_filter"):
        payload["city_filter"] = args["city_filter"]
    resp = requests.post(f"{config.RAG_SERVICE_URL}/rag/search", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# live_web_lookup — Python port of the n8n "Live Market Lookup" sub-workflow
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9",
}


def _slugify(value: str) -> str:
    import unicodedata
    normalized = unicodedata.normalize("NFD", value)
    stripped = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    slug = re.sub(r"[^a-z0-9]+", "_", stripped.lower()).strip("_")
    return slug


def _extract_avito(html: str) -> list[dict]:
    cards = html.split('<div class="sc-efda8edc-2')[1:]
    results = []
    for chunk in cards:
        head = chunk[:600]
        href_match = re.search(r'<a href="([^"]+)"[^>]*data-testid="ad-card-v2-(\d+)"', head)
        if not href_match:
            continue
        title_match = re.search(r'<h3[^>]*title="([^"]+)"', chunk)
        bedrooms_match = re.search(r'title="Chambres">(\d+)', chunk)
        surface_match = re.search(r'title="Surface totale">(\d+)', chunk)
        price_match = re.search(r'<span[^>]*>([\d\s]+)</span><span[^>]*>DH</span>', chunk)
        loc_match = re.search(r'<span class="sc-j5d10c-23[^"]*">([^<]+)</span>', chunk)
        city = loc_match.group(1).split(",")[0].strip() if loc_match else None
        results.append({
            "source_site": "avito",
            "source_url": href_match.group(1),
            "title": title_match.group(1) if title_match else None,
            "city": city,
            "rent_price": int(price_match.group(1).replace(" ", "")) if price_match else None,
            "surface_m2": int(surface_match.group(1)) if surface_match else None,
            "bedrooms": int(bedrooms_match.group(1)) if bedrooms_match else None,
        })
    return results


def _extract_agenz(html: str) -> list[dict]:
    cards = html.split('<div class="_listingCard_zunkh_1"')[1:]
    results = []
    for chunk in cards:
        head = chunk[:350]
        rel_url_match = re.search(r'data-url="([^"]+)"', head)
        if not rel_url_match:
            continue
        price_idx = chunk.find("_priceContainer_1n17q_1")
        price_slice = chunk[price_idx:price_idx + 700] if price_idx > -1 else ""
        price_matches = [int(m.replace(" ", "")) for m in re.findall(r'_nouveau_1n17q_54"[^>]*>([\d\s]+)<', price_slice)]
        surface_match = re.search(r'data-highlight="surface"[\s\S]{0,400}?_highlightValue_1xhcb_115">(\d+)<', chunk)
        bedrooms_match = re.search(r'data-highlight="typologie"[\s\S]{0,400}?_highlightValue_1xhcb_115">(\d+)<', chunk)
        title_match = re.search(r'_locationAdress_1xhcb_205"[^>]*\btitle="([^"]+)"', chunk)
        city_neighborhood_match = re.search(r'_locationAdress_1xhcb_205"[\s\S]{0,400}?<span>([^<]+)</span>', chunk)
        city = city_neighborhood_match.group(1).split("-")[0].strip() if city_neighborhood_match else None
        results.append({
            "source_site": "agenz",
            "source_url": "https://agenz.ma" + rel_url_match.group(1),
            "title": title_match.group(1) if title_match else None,
            "city": city,
            "rent_price": price_matches[0] if price_matches else None,
            "surface_m2": int(surface_match.group(1)) if surface_match else None,
            "bedrooms": int(bedrooms_match.group(1)) if bedrooms_match else None,
        })
    return results


def live_web_lookup(args: dict) -> dict:
    city = (args.get("city") or "").strip()
    max_results = int(args.get("max_results") or 5)

    avito_url = (
        f"https://www.avito.ma/fr/{_slugify(city)}/appartements-à_louer"
        if city else "https://www.avito.ma/fr/maroc/appartements-à_louer"
    )
    agenz_url = "https://agenz.ma/fr/louer/location-appartements"

    all_results: list[dict] = []
    for url, extractor in ((avito_url, _extract_avito), (agenz_url, _extract_agenz)):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
            all_results.extend(extractor(resp.text))
        except requests.RequestException as exc:
            # A single site being unreachable shouldn't fail the whole lookup.
            all_results.append({"error": f"Could not fetch {url}: {exc}"})

    matches = [r for r in all_results if r.get("rent_price") is not None]
    if city:
        city_lower = city.lower()
        city_matches = [
            r for r in matches
            if city_lower in (r.get("city") or "").lower() or city_lower in (r.get("title") or "").lower()
        ]
        if not city_matches:
            return guardrails.label_live_lookup_result({
                "found": False,
                "message": f'No live listings found for "{city}" on avito.ma or agenz.ma right now.',
                "results": [],
            })
        matches = city_matches

    matches = matches[:max_results]
    return guardrails.label_live_lookup_result({"found": len(matches) > 0, "count": len(matches), "results": matches})


# ---------------------------------------------------------------------------
# remember_preference — memory writes handled specially in agent.py, this is
# just the schema-facing name the model calls.
# ---------------------------------------------------------------------------

def remember_preference(args: dict) -> dict:
    # Actual persistence happens in agent.py, which intercepts this tool name
    # before dispatch so it can write to the session's preferences.
    return {"acknowledged": True, "key": args.get("key"), "value": args.get("value")}


TOOL_FUNCTIONS = {
    "search_listings": search_listings,
    "get_market_stats": get_market_stats,
    "predict_rent": predict_rent,
    "explain_prediction": explain_prediction,
    "rag_search": rag_search,
    "live_web_lookup": live_web_lookup,
    "remember_preference": remember_preference,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_listings",
            "description": "Search real, currently scraped rental listings by city, neighborhood, max price, or minimum bedrooms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City to filter by, e.g. Casablanca, Rabat, Marrakech"},
                    "neighborhood": {"type": "string", "description": "Neighborhood to filter by (optional, partial match)"},
                    "max_price": {"type": "number", "description": "Maximum monthly rent in MAD (optional)"},
                    "min_bedrooms": {"type": "number", "description": "Minimum number of bedrooms (optional)"},
                    "limit": {"type": "number", "description": "Max number of results to return, default 5"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_stats",
            "description": "Compute average/median/min/max rent and average surface for real scraped listings matching the given filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City to compute stats for"},
                    "neighborhood": {"type": "string", "description": "Neighborhood to narrow down to (optional)"},
                    "bedrooms": {"type": "number", "description": "Filter to exactly this many bedrooms (optional)"},
                    "property_type": {"type": "string", "description": "Filter by property type, e.g. appartement (optional)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_rent",
            "description": "Predict the monthly rent (MAD) for a hypothetical property (not an existing listing) using the trained ML model. Requires city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "neighborhood": {"type": "string"},
                    "property_type": {"type": "string", "description": "e.g. appartement, villa, studio. Default appartement"},
                    "surface_m2": {"type": "number"},
                    "bedrooms": {"type": "number"},
                    "bathrooms": {"type": "number"},
                    "furnished": {"type": "boolean"},
                    "amenities": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_prediction",
            "description": "Predict rent for a hypothetical property AND explain, in plain language, which features pushed the price up or down (based on SHAP values).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "neighborhood": {"type": "string"},
                    "property_type": {"type": "string"},
                    "surface_m2": {"type": "number"},
                    "bedrooms": {"type": "number"},
                    "bathrooms": {"type": "number"},
                    "furnished": {"type": "boolean"},
                    "amenities": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Semantic search over listing descriptions for fuzzy, natural-language queries (e.g. 'quiet apartment near a school') that structured filters can't express.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language description of what the user wants"},
                    "top_k": {"type": "number", "description": "Number of results, default 5"},
                    "city_filter": {"type": "string", "description": "Optional exact city name to restrict results to"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "live_web_lookup",
            "description": "Live-fetch listings from avito.ma and agenz.ma right now, for a city/area NOT covered by search_listings or get_market_stats (i.e. those returned zero matches). Results are an unverified live snapshot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "max_results": {"type": "number", "description": "Default 5"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_preference",
            "description": "Call this when the user states a lasting preference (budget, min bedrooms, preferred city, etc.) that should be remembered for the rest of the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short preference name, e.g. 'max_budget_mad', 'min_bedrooms', 'preferred_city'"},
                    "value": {"type": "string", "description": "The preference value as a string"},
                },
                "required": ["key", "value"],
            },
        },
    },
]
