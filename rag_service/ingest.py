"""One-shot CLI ingestion: fetch current listings and (idempotently) embed/upsert into Chroma.

Usage:
    python ingest.py            # embed new/changed listings only
    python ingest.py --force    # re-embed everything
"""
import argparse
import sys

import core


def main():
    parser = argparse.ArgumentParser(description="Ingest rental listings into the RAG vector store.")
    parser.add_argument("--force", action="store_true", help="Re-embed all listings, ignoring content hashes")
    args = parser.parse_args()

    print("Fetching listings...")
    try:
        listings = core.fetch_listings()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Fetched {len(listings)} listings.")

    print("Embedding and upserting into Chroma (downloads the embedding model on first run)...")
    summary = core.sync_listings(listings=listings, force=args.force)
    print(
        f"Done. Embedded {summary['embedded']} listing(s), "
        f"skipped {summary['skipped_unchanged']} unchanged, "
        f"{summary['total_listings_seen']} total seen."
    )


if __name__ == "__main__":
    main()
