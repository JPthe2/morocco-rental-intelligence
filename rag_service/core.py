"""Shared RAG logic: fetching listings, embedding, syncing to Chroma, and search.

Both main.py (the FastAPI service) and ingest.py (the standalone CLI) call into
this module so there is exactly one implementation of "embed + upsert" and it is
idempotent either way: a listing is only re-embedded when its derived document
text actually changed (tracked via a content hash in state.db).
"""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
import requests
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"
STATE_DB_PATH = BASE_DIR / "state.db"
COLLECTION_NAME = "listing_descriptions"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

ML_SERVICE_EXPORT_URL = "http://127.0.0.1:8000/export"
FALLBACK_SNAPSHOT_PATH = BASE_DIR.parent / "ml" / "price_prediction" / "data" / "listings_snapshot.json"

_model: Optional[SentenceTransformer] = None
_client = None
_collection = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def get_state_db() -> sqlite3.Connection:
    STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(STATE_DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sync_meta (key TEXT PRIMARY KEY, value TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS listing_hashes ("
        "listing_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, synced_at TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def fetch_listings() -> list[dict]:
    """Pull the current listings snapshot from the ML service's /export endpoint,
    falling back to reading the local snapshot file directly if that service is down."""
    try:
        resp = requests.get(ML_SERVICE_EXPORT_URL, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        if FALLBACK_SNAPSHOT_PATH.exists():
            with open(FALLBACK_SNAPSHOT_PATH, encoding="utf-8") as f:
                return json.load(f)
        raise RuntimeError(
            "Could not reach ML service at "
            f"{ML_SERVICE_EXPORT_URL} and no local snapshot file was found at "
            f"{FALLBACK_SNAPSHOT_PATH}."
        )


def build_doc_text(listing: dict) -> str:
    """Compose the text that gets embedded for a listing, from its structured
    fields plus its (usually short, title-like) raw description."""
    parts = [
        f"{listing.get('property_type', 'property')} in "
        f"{listing.get('neighborhood') or 'an unspecified neighborhood'}, "
        f"{listing.get('city', 'Morocco')}.",
        listing.get("description") or "",
    ]
    amenities = listing.get("amenities")
    if amenities:
        parts.append(f"Amenities: {amenities}.")
    details = []
    if listing.get("bedrooms") is not None:
        details.append(f"{listing['bedrooms']} bedrooms")
    if listing.get("bathrooms") is not None:
        details.append(f"{listing['bathrooms']} bathrooms")
    if listing.get("surface_m2") is not None:
        details.append(f"{listing['surface_m2']} m2")
    if details:
        parts.append(", ".join(details) + ".")
    if listing.get("furnished") is not None:
        parts.append("Furnished." if listing["furnished"] else "Unfurnished.")
    return " ".join(p for p in parts if p).strip()


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """Split long text on sentence boundaries. A no-op for typical (short) listing
    text; only kicks in for unusually long descriptions."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current = ""
    for sentence in text.replace("\n", " ").split(". "):
        candidate = f"{current}. {sentence}" if current else sentence
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return chunks


def _listing_id(listing: dict) -> Optional[str]:
    value = listing.get("id")
    if value is None:
        value = listing.get("source_url") or listing.get("listing_external_id")
    return str(value) if value is not None else None


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sync_listings(listings: Optional[list[dict]] = None, force: bool = False) -> dict:
    """Embed and upsert new/changed listings into Chroma.

    Idempotent: a listing whose derived document text hash is unchanged since the
    last sync is skipped, so re-running (or calling /rag/sync repeatedly) does not
    duplicate vectors or waste embedding calls. Pass force=True to re-embed everything.
    """
    if listings is None:
        listings = fetch_listings()

    db = get_state_db()
    collection = get_collection()
    model = get_model()

    existing = dict(db.execute("SELECT listing_id, content_hash FROM listing_hashes").fetchall())

    to_embed_ids: list[str] = []
    to_embed_docs: list[str] = []
    to_embed_meta: list[dict] = []
    skipped = 0

    for listing in listings:
        listing_id = _listing_id(listing)
        if not listing_id:
            continue
        doc_text = build_doc_text(listing)
        content_hash = _content_hash(doc_text)
        if not force and existing.get(listing_id) == content_hash:
            skipped += 1
            continue
        chunks = chunk_text(doc_text)
        # One vector per listing: embed the primary chunk. Listings' descriptions are
        # short (title-like) in this dataset, so splitting rarely triggers in practice.
        to_embed_ids.append(listing_id)
        to_embed_docs.append(chunks[0])
        to_embed_meta.append({
            "listing_id": listing_id,
            "city": listing.get("city") or "",
            "neighborhood": listing.get("neighborhood") or "",
            "price": listing.get("rent_price") if listing.get("rent_price") is not None else -1,
            "url": listing.get("source_url") or "",
        })

    if to_embed_docs:
        embeddings = model.encode(to_embed_docs, show_progress_bar=False).tolist()
        collection.upsert(
            ids=to_embed_ids, embeddings=embeddings, documents=to_embed_docs, metadatas=to_embed_meta
        )
        now = datetime.now(timezone.utc).isoformat()
        db.executemany(
            "INSERT INTO listing_hashes (listing_id, content_hash, synced_at) VALUES (?, ?, ?) "
            "ON CONFLICT(listing_id) DO UPDATE SET content_hash=excluded.content_hash, synced_at=excluded.synced_at",
            [(lid, _content_hash(doc), now) for lid, doc in zip(to_embed_ids, to_embed_docs)],
        )
        db.execute(
            "INSERT INTO sync_meta (key, value) VALUES ('last_synced_at', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (now,),
        )
        db.commit()

    db.close()
    return {
        "total_listings_seen": len(listings),
        "embedded": len(to_embed_ids),
        "skipped_unchanged": skipped,
        "last_synced_at": get_last_synced_at(),
    }


def get_last_synced_at() -> Optional[str]:
    db = get_state_db()
    row = db.execute("SELECT value FROM sync_meta WHERE key='last_synced_at'").fetchone()
    db.close()
    return row[0] if row else None


def search(query: str, top_k: int = 5, city_filter: Optional[str] = None) -> list[dict]:
    collection = get_collection()
    model = get_model()
    query_embedding = model.encode([query]).tolist()
    where = {"city": city_filter} if city_filter else None
    results = collection.query(query_embeddings=query_embedding, n_results=top_k, where=where)

    output = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for listing_id, doc, meta, dist in zip(ids, docs, metas, dists):
        # Cosine distance lives in [0, 2]; map to a [0, 1] similarity score for display.
        similarity = max(0.0, 1 - dist / 2)
        output.append({
            "listing_id": listing_id,
            "similarity_score": round(similarity, 4),
            "description": doc,
            **meta,
        })
    return output
