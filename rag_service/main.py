"""FastAPI service: semantic search (RAG) over rental listing descriptions."""
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import core

app = FastAPI(title="Morocco Rental RAG Service", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., examples=["quiet apartment near a school in Agadir under 5000 MAD"])
    top_k: int = Field(5, ge=1, le=50)
    city_filter: Optional[str] = Field(None, examples=["Agadir"])


class SyncRequest(BaseModel):
    force: bool = Field(False, description="Re-embed every listing regardless of content hash")


@app.get("/health")
def health():
    collection = core.get_collection()
    return {
        "status": "ok",
        "indexed_listings": collection.count(),
        "last_synced_at": core.get_last_synced_at(),
        "embedding_model": core.EMBEDDING_MODEL_NAME,
    }


@app.post("/rag/search")
def rag_search(req: SearchRequest):
    if core.get_collection().count() == 0:
        raise HTTPException(
            status_code=503,
            detail="No listings indexed yet. Run ingest.py or POST /rag/sync first.",
        )
    try:
        results = core.search(req.query, top_k=req.top_k, city_filter=req.city_filter)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc
    return {"query": req.query, "count": len(results), "results": results}


@app.post("/rag/sync")
def rag_sync(req: SyncRequest = SyncRequest()):
    try:
        return core.sync_listings(force=req.force)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
