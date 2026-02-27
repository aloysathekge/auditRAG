"""Ingest API: trigger document ingestion (FinanceBench or single doc). To be wired when POST /ingest is implemented."""
from fastapi import APIRouter

router = APIRouter(tags=["ingest"])

# POST /ingest will call ingestion pipeline (loader -> parser -> chunker -> embedder -> upsert)
