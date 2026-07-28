"""Tests for rag_service core logic: doc building, hashing/idempotency, sync, and search."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import core as rag_core

SAMPLE_LISTINGS = [
    {
        "id": 1,
        "city": "Agadir",
        "neighborhood": "Talborjt",
        "property_type": "appartement",
        "rent_price": 4500,
        "surface_m2": 70,
        "bedrooms": 2,
        "bathrooms": 1,
        "furnished": False,
        "amenities": "parking,ascenseur",
        "description": "Quiet apartment near a school in Agadir, close to shops.",
        "source_url": "https://example.com/1",
    },
    {
        "id": 2,
        "city": "Casablanca",
        "neighborhood": "Maarif",
        "property_type": "appartement",
        "rent_price": 9000,
        "surface_m2": 120,
        "bedrooms": 3,
        "bathrooms": 2,
        "furnished": True,
        "amenities": "piscine,jardin",
        "description": "Bright furnished apartment in a lively central neighborhood.",
        "source_url": "https://example.com/2",
    },
]


@pytest.fixture
def isolated_core(tmp_path, monkeypatch):
    """Point Chroma + the SQLite state DB at a temp dir so tests never touch real data."""
    monkeypatch.setattr(rag_core, "CHROMA_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(rag_core, "STATE_DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(rag_core, "_client", None)
    monkeypatch.setattr(rag_core, "_collection", None)
    return rag_core


def test_build_doc_text_includes_key_fields():
    text = rag_core.build_doc_text(SAMPLE_LISTINGS[0])
    assert "Agadir" in text
    assert "Talborjt" in text
    assert "school" in text.lower()


def test_sync_then_search_returns_relevant_listing(isolated_core):
    summary = isolated_core.sync_listings(listings=SAMPLE_LISTINGS)
    assert summary["embedded"] == 2
    assert summary["skipped_unchanged"] == 0

    results = isolated_core.search("quiet place near a school in Agadir", top_k=2)
    assert len(results) > 0
    assert results[0]["listing_id"] == "1"
    assert results[0]["city"] == "Agadir"


def test_sync_is_idempotent(isolated_core):
    first = isolated_core.sync_listings(listings=SAMPLE_LISTINGS)
    assert first["embedded"] == 2

    second = isolated_core.sync_listings(listings=SAMPLE_LISTINGS)
    assert second["embedded"] == 0
    assert second["skipped_unchanged"] == 2


def test_sync_reembeds_changed_listing(isolated_core):
    isolated_core.sync_listings(listings=SAMPLE_LISTINGS)
    changed = [dict(SAMPLE_LISTINGS[0], description="Completely different description now.")] + SAMPLE_LISTINGS[1:]
    summary = isolated_core.sync_listings(listings=changed)
    assert summary["embedded"] == 1
    assert summary["skipped_unchanged"] == 1


def test_city_filter_excludes_other_cities(isolated_core):
    isolated_core.sync_listings(listings=SAMPLE_LISTINGS)
    results = isolated_core.search("apartment", top_k=5, city_filter="Casablanca")
    assert all(r["city"] == "Casablanca" for r in results)


def test_force_reembeds_unchanged_listings(isolated_core):
    isolated_core.sync_listings(listings=SAMPLE_LISTINGS)
    summary = isolated_core.sync_listings(listings=SAMPLE_LISTINGS, force=True)
    assert summary["embedded"] == 2
    assert summary["skipped_unchanged"] == 0
