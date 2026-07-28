import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tools

FIXTURE = [
    {"city": "Rabat", "neighborhood": "Agdal", "rent_price": 5000, "surface_m2": 80,
     "bedrooms": 2, "bathrooms": 1, "source_site": "agenz", "source_url": "https://x/1",
     "property_type": "appartement"},
    {"city": "Rabat", "neighborhood": "Hay Riad", "rent_price": 9000, "surface_m2": 120,
     "bedrooms": 3, "bathrooms": 2, "source_site": "avito", "source_url": "https://x/2",
     "property_type": "appartement"},
    {"city": "Casablanca", "neighborhood": "Maarif", "rent_price": 7000, "surface_m2": 90,
     "bedrooms": 2, "bathrooms": 1, "source_site": "agenz", "source_url": "https://x/3",
     "property_type": "appartement"},
]


def test_search_listings_filters_by_city_and_price(monkeypatch):
    monkeypatch.setattr(tools, "get_listings", lambda *a, **k: FIXTURE)
    result = tools.search_listings({"city": "Rabat", "max_price": 6000})
    assert result["matchCount"] == 1
    assert result["results"][0]["neighborhood"] == "Agdal"


def test_search_listings_min_bedrooms_filter(monkeypatch):
    monkeypatch.setattr(tools, "get_listings", lambda *a, **k: FIXTURE)
    result = tools.search_listings({"min_bedrooms": 3})
    assert result["matchCount"] == 1
    assert result["results"][0]["neighborhood"] == "Hay Riad"


def test_get_market_stats_computes_expected_values(monkeypatch):
    monkeypatch.setattr(tools, "get_listings", lambda *a, **k: FIXTURE)
    stats = tools.get_market_stats({"city": "Rabat"})
    assert stats["matchCount"] == 2
    assert stats["avgRentMAD"] == 7000
    assert stats["minRentMAD"] == 5000
    assert stats["maxRentMAD"] == 9000


def test_get_market_stats_empty_match_returns_nones(monkeypatch):
    monkeypatch.setattr(tools, "get_listings", lambda *a, **k: FIXTURE)
    stats = tools.get_market_stats({"city": "Nowhere"})
    assert stats["matchCount"] == 0
    assert stats["avgRentMAD"] is None


def test_remember_preference_returns_acknowledgement():
    result = tools.remember_preference({"key": "max_budget_mad", "value": "6000"})
    assert result["acknowledged"] is True
    assert result["key"] == "max_budget_mad"


def test_explain_prediction_humanizes_features(monkeypatch):
    def fake_call_predict(args):
        return {
            "predicted_rent_mad": 6500,
            "confidence_interval_80pct": [6000, 7000],
            "top_drivers": [
                {"feature": "num__surface_m2", "impact_on_log_price": 0.12},
                {"feature": "cat__city_Casablanca", "impact_on_log_price": -0.05},
            ],
        }
    monkeypatch.setattr(tools, "_call_predict", fake_call_predict)
    result = tools.explain_prediction({"city": "Casablanca"})
    assert result["predicted_rent_mad"] == 6500
    assert "surface" in result["explanation"].lower()


def test_predict_rent_rejects_out_of_range_prediction(monkeypatch):
    monkeypatch.setattr(tools, "_call_predict", lambda args: {"predicted_rent_mad": 999_999})
    monkeypatch.setattr(tools, "get_listings", lambda *a, **k: FIXTURE)
    result = tools.predict_rent({"city": "Rabat"})
    assert "error" in result


def test_predict_rent_requires_city():
    result = tools.predict_rent({})
    assert "error" in result
