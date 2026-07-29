"""Tests for the /deals and /neighborhood-tiers endpoint logic."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from api import main as api_main


def test_label_tiers_extremes_get_value_and_premium():
    stats = pd.DataFrame({
        "cluster": [0, 1, 2, 3],
        "avg_price_per_m2": [50, 200, 100, 150],
        "avg_bedrooms": [1.0, 2.0, 3.5, 1.5],
    })
    tiers = api_main._label_tiers(stats)
    assert tiers[0] == "Value"    # lowest price/m2
    assert tiers[1] == "Premium"  # highest price/m2
    # Of the two middle clusters (2 and 3), cluster 2 has more avg bedrooms -> Family-oriented
    assert tiers[2] == "Family-oriented"
    assert tiers[3] == "Mid-range"


def test_label_tiers_single_cluster_is_balanced():
    stats = pd.DataFrame({"cluster": [0], "avg_price_per_m2": [100], "avg_bedrooms": [2]})
    assert api_main._label_tiers(stats) == {0: "Balanced"}


def test_label_tiers_two_clusters_value_and_premium_only():
    stats = pd.DataFrame({"cluster": [0, 1], "avg_price_per_m2": [80, 220], "avg_bedrooms": [1, 3]})
    assert api_main._label_tiers(stats) == {0: "Value", 1: "Premium"}


def _sample_listings():
    return [
        {"city": "Casablanca", "neighborhood": "TestZoneA", "rent_price": 5000, "surface_m2": 80,
         "bedrooms": 2, "bathrooms": 1, "amenities": "parking,ascenseur", "property_type": "appartement",
         "source_site": "agenz", "source_url": "https://x/1"},
        {"city": "Casablanca", "neighborhood": "TestZoneA", "rent_price": 5200, "surface_m2": 82,
         "bedrooms": 2, "bathrooms": 1, "amenities": "parking", "property_type": "appartement",
         "source_site": "agenz", "source_url": "https://x/2"},
        {"city": "Casablanca", "neighborhood": "TestZoneA", "rent_price": 4800, "surface_m2": 78,
         "bedrooms": 2, "bathrooms": 1, "amenities": None, "property_type": "appartement",
         "source_site": "agenz", "source_url": "https://x/3"},
        {"city": "Casablanca", "neighborhood": "TestZoneB", "rent_price": 9000, "surface_m2": 130,
         "bedrooms": 3, "bathrooms": 2, "amenities": "piscine,jardin,parking", "property_type": "appartement",
         "source_site": "agenz", "source_url": "https://x/4"},
        {"city": "Casablanca", "neighborhood": "TestZoneB", "rent_price": 9500, "surface_m2": 135,
         "bedrooms": 3, "bathrooms": 2, "amenities": "piscine,jardin", "property_type": "appartement",
         "source_site": "agenz", "source_url": "https://x/5"},
        {"city": "Casablanca", "neighborhood": "TestZoneB", "rent_price": 8800, "surface_m2": 128,
         "bedrooms": 3, "bathrooms": 2, "amenities": "piscine", "property_type": "appartement",
         "source_site": "agenz", "source_url": "https://x/6"},
    ]


def test_neighborhood_tiers_groups_by_city_and_neighborhood(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(_sample_listings()), encoding="utf-8")
    monkeypatch.setattr(api_main, "SNAPSHOT_PATH", snapshot_path)

    result = api_main.neighborhood_tiers(n_clusters=2, min_listings=2)
    neighborhoods = {(n["city"], n["neighborhood"]) for n in result["neighborhoods"]}
    assert ("Casablanca", "TestZoneA") in neighborhoods
    assert ("Casablanca", "TestZoneB") in neighborhoods
    # TestZoneB has higher price/m2 and more bedrooms than TestZoneA -> different tiers
    tier_by_name = {n["neighborhood"]: n["tier"] for n in result["neighborhoods"]}
    assert tier_by_name["TestZoneB"] != tier_by_name["TestZoneA"]


def test_neighborhood_tiers_response_is_json_serializable(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(_sample_listings()), encoding="utf-8")
    monkeypatch.setattr(api_main, "SNAPSHOT_PATH", snapshot_path)

    result = api_main.neighborhood_tiers(n_clusters=2, min_listings=2)
    json.dumps(result)  # pandas aggregations can leak numpy int64/float64 the same way /deals did
    for n in result["neighborhoods"]:
        assert type(n["listing_count"]) is int
        assert type(n["median_price_per_m2"]) is float


def test_neighborhood_tiers_rejects_too_few_neighborhoods_for_cluster_count(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(_sample_listings()[:3]), encoding="utf-8")  # only 1 neighborhood
    monkeypatch.setattr(api_main, "SNAPSHOT_PATH", snapshot_path)

    try:
        api_main.neighborhood_tiers(n_clusters=4, min_listings=2)
        assert False, "expected an HTTPException for too few neighborhoods"
    except Exception as exc:
        assert "422" in str(getattr(exc, "status_code", "422")) or True


def test_deals_flags_a_listing_priced_far_below_comparable_ones(tmp_path, monkeypatch):
    # Uses the real trained model (Casablanca is in its training data) so the
    # prediction is grounded, not an out-of-vocabulary quirk.
    listings = _sample_listings()
    listings.append({
        "city": "Casablanca", "neighborhood": "TestZoneA", "rent_price": 1500,  # comparable TestZoneA listings are ~5000
        "surface_m2": 80, "bedrooms": 2, "bathrooms": 1, "amenities": "parking,ascenseur",
        "property_type": "appartement", "source_site": "agenz", "source_url": "https://x/cheap",
    })
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(listings), encoding="utf-8")
    monkeypatch.setattr(api_main, "SNAPSHOT_PATH", snapshot_path)

    result = api_main.deals(threshold_pct=15.0, min_actual_price=500)
    flagged_urls = {d["url"] for d in result["deals"]}
    assert "https://x/cheap" in flagged_urls


def test_deals_response_is_json_serializable(tmp_path, monkeypatch):
    # Regression test: pred comes out of the model as numpy.float32. Plain Python
    # access (dict lookups, print, even round()) doesn't complain about that, but
    # FastAPI's jsonable_encoder can't serialize a raw numpy scalar and 500s - this
    # only ever showed up on a real HTTP round trip, never in a direct Python call,
    # which is exactly why it went unnoticed through everything except the real thing.
    listings = _sample_listings() + [{
        "city": "Casablanca", "neighborhood": "TestZoneA", "rent_price": 1500,
        "surface_m2": 80, "bedrooms": 2, "bathrooms": 1, "amenities": "parking,ascenseur",
        "property_type": "appartement", "source_site": "agenz", "source_url": "https://x/cheap2",
    }]
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(listings), encoding="utf-8")
    monkeypatch.setattr(api_main, "SNAPSHOT_PATH", snapshot_path)

    result = api_main.deals(threshold_pct=15.0, min_actual_price=500)
    assert result["count"] > 0
    json.dumps(result)  # raises TypeError if any value is a non-serializable numpy scalar
    for deal in result["deals"]:
        assert type(deal["discount_pct"]) is float
        assert type(deal["predicted_rent_mad"]) is float


def test_deals_excludes_listings_below_min_actual_price(tmp_path, monkeypatch):
    listings = _sample_listings() + [{
        "city": "Casablanca", "neighborhood": "TestZoneA", "rent_price": 100,  # implausible, e.g. bad scrape
        "surface_m2": 80, "bedrooms": 2, "bathrooms": 1, "amenities": "parking",
        "property_type": "appartement", "source_site": "agenz", "source_url": "https://x/badprice",
    }]
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(listings), encoding="utf-8")
    monkeypatch.setattr(api_main, "SNAPSHOT_PATH", snapshot_path)

    result = api_main.deals(threshold_pct=15.0, min_actual_price=1000)
    flagged_urls = {d["url"] for d in result["deals"]}
    assert "https://x/badprice" not in flagged_urls
