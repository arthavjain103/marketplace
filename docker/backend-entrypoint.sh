#!/bin/sh
set -e

# Ingest policy PDFs into Pinecone + BM25 when credentials are present and no
# BM25 model exists yet. Idempotent: re-running overwrites Pinecone vectors.
# Ignore ingestion failures so the API still starts in deterministic-only mode.
if [ -n "$PINECONE_API_KEY" ] && [ -n "$PINECONE_INDEX" ] && [ ! -f data/bm25.pkl ]; then
  echo "Ingesting policy PDFs (Pinecone + BM25)..."
  python ingest.py || echo "Ingestion failed or skipped; starting API in degraded mode."
elif [ ! -f data/bm25.pkl ]; then
  echo "No Pinecone credentials provided; skipping ingestion. AI review unavailable."
fi

exec uvicorn backend.app:app --host 0.0.0.0 --port 8000
