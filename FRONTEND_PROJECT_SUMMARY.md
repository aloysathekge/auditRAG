## Project Summary (Frontend-Friendly) — `auditRAG`

### What this project is
`auditRAG` is a backend RAG (Retrieval-Augmented Generation) service designed to answer questions about **financial PDF documents** (e.g., SEC 10-K / 10-Q filings). It provides:

- A **REST API** for:
  - **Ingesting** documents (download PDF → extract text → chunk → embed → store in vector DB)
  - **Querying** (retrieve relevant chunks → optionally generate an LLM answer with citations)
  - Viewing **health** and **metrics**
- A minimal **telemetry/metrics** layer stored in **PostgreSQL**
- A **vector database** (Qdrant) where chunk embeddings + metadata live

Your frontend will be a client that:
1. Triggers ingestion of documents (one-by-one or FinanceBench dataset items)
2. Sends questions to `/query`
3. Renders:
   - Answer (if generated)
   - Retrieved context chunks (always available)
   - Citations / sources (doc + page)
   - Latency + optional usage/cost
4. Optionally shows “system metrics” from `/metrics`

---

## Core Architecture (mental model)

### Components
- **FastAPI app**: `auditrag/main.py`
- **Ingestion pipeline**: `auditrag/ingestion/*`
- **Retrieval pipeline**: `auditrag/retrieval/*`
- **Answer generation**: `auditrag/generation/*`
- **Telemetry & metrics**: `auditrag/observability/*`, `auditrag/db/*`
- **External services**
  - **Qdrant**: stores chunk vectors + payload (doc/page/text)
  - **Postgres**: stores query logs (latency/cost/model/etc.)
  - **LLM provider**: OpenAI or Anthropic (optional)
  - **Embedding provider**: OpenAI or local sentence-transformers

### Data stored where
- **Qdrant** holds the “knowledge base”:
  - each chunk is a point with:
    - `vector`: embedding
    - `payload`: `doc_name`, `page`, `text`, `chunk_id`
- **Postgres** holds observability only:
  - query text, latency, optional cost, model used, timestamp
  - not used for retrieval

---

## Main User Flows (what UI needs to support)

### 1) Ingestion (build the knowledge base)
Triggered via API; backend does:
- Download PDF (cached locally under `data/financebench/pdfs/`)
- Extract text page-by-page using `pdfplumber`
- Chunk text into overlapping word windows
- Generate embeddings (OpenAI or local model)
- Upsert into Qdrant collection `auditrag_chunks`
- Invalidate BM25 sparse cache (so sparse/hybrid searches see new data)

**Frontend implication**: ingestion is a “job-like” operation:
- can take time (download + embedding + upload)
- should show loading/progress state (even if backend only returns final summary)
- should show success/failure per document
- should probably provide “re-ingest” toggle (`skip_if_exists=false`)

### 2) Querying (retrieve + optionally generate answer)
Triggered via `/query`:
- Retrieval returns `top_k` chunks with `doc_name`, `page`, `text`, `score`
- If enabled, generation calls LLM with strict prompt to use only provided context and cite sources
- Telemetry writes to Postgres (best-effort; failures don’t break responses)

**Frontend implication**: show in results view:
- The user’s question
- Answer (optional)
- Sources list (doc + page)
- Retrieved chunks list with scores
- Latency breakdown (retrieve vs generate)
- Optional token usage and cost if present

---

## API Endpoints (what the frontend calls)

### Base URL
Typically: `http://localhost:8000`

CORS is enabled for:
- `http://localhost:3000`
- `http://localhost:5173`
- `https://auditrag.aloysathekge.com`

So a React/Vite frontend on 5173 is expected to work without extra proxying.

---

### `GET /health`
**Purpose**: basic liveness + likely checks (router code not shown here, but README references it)

**Frontend use**:
- Show backend online/offline badge
- Possibly show Postgres/Qdrant connectivity if the endpoint returns it

---

### `POST /query`
**Purpose**: main user-facing endpoint

**Request JSON**
```json
{
  "question": "string",
  "top_k": 5,
  "generate_answer": true
}
```

**Response (typical)**
```json
{
  "question": "string",
  "chunks": [
    { "text": "…", "doc_name": "…", "page": 12, "score": 0.73 }
  ],
  "latency_ms": { "retrieve_ms": 123, "generate_ms": 456, "total_ms": 579 },
  "answer": "…",
  "sources": [
    { "doc_name": "…", "page": 12 }
  ],
  "usage": { "input_tokens": 123, "output_tokens": 45 },
  "cost_usd": 0.000123
}
```

**Error behavior**
- If `question` is missing/blank you may get `400`
- If retrieval fails (Qdrant down, embedding error), you get `502` with `Retrieval failed: ...`
- If generation is requested but keys are missing:
  - backend returns chunks, but may omit `answer`/`sources` (generation returns `None`)
- If no relevant chunks:
  - generation returns `No relevant context was retrieved.` and empty sources

---

### `GET /query`
Same as POST but via query params:
- `/query?question=...&top_k=5&generate_answer=true`

**Frontend**: You’ll almost always prefer `POST /query`.

---

### `POST /ingest`
**Purpose**: ingest a single document by URL

**Request JSON**
```json
{
  "doc_name": "string",
  "doc_link": "https://..."
}
```

**Query param**
- `skip_if_exists=true|false` (default true)

**Response**
```json
{
  "status": "ok",
  "doc_name": "…",
  "chunks_created": 120,
  "chunks_upserted": 120
}
```

**Error behavior**
- `400` if PDF yields no extracted text, etc.
- `502` for unexpected failures (download errors, Qdrant issues, embedding issues)

---

### `POST /ingest/financebench`
**Purpose**: ingest a FinanceBench dataset item by index (backend fetches dataset row and downloads its doc_link)

**Params**
- `index=0` (default 0)
- `skip_if_exists=true|false`

**Response**: same shape as ingestion result.

**Frontend use**:
- Provide a “quick demo ingest” UI: index field + ingest button

---

### `POST /ingest/financebench/bulk`
**Purpose**: bulk ingest a range; deduplicates by `doc_name`

**Params**
- `limit=10`
- `start=0`
- `skip_if_exists=true|false`

**Response**
```json
{
  "status": "ok",
  "range_indices": 10,
  "unique_docs": 8,
  "results": [
    { "doc_name": "…", "chunks_created": 120, "chunks_upserted": 120 }
  ]
}
```

**Frontend use**:
- Admin/maintenance screen with bulk ingestion controls

---

### `GET /metrics`
**Purpose**: aggregate stats over last N query logs (Postgres)

**Response (success)**
```json
{
  "count": 123,
  "latency_p50_ms": 400.12,
  "latency_p95_ms": 1200.55,
  "avg_cost_usd": 0.000321
}
```

**Response (when unavailable / no data)**
```json
{
  "message": "…",
  "count": 0,
  "latency_p50_ms": null,
  "latency_p95_ms": null,
  "avg_cost_usd": null
}
```

---

## Retrieval Modes (important for UI messaging)
Configured via `.env` (`RETRIEVAL_MODE`):
- **dense**: semantic vector search in Qdrant (query embedding required)
- **sparse**: BM25 keyword search
  - implemented by loading *all* Qdrant chunk payloads and building a BM25 index in-memory (cached)
- **hybrid**: combines dense+sparse using Reciprocal Rank Fusion (RRF)

**Frontend**: you typically don’t control this per request; it’s server config. But you can show it in a Settings/About page if you add an endpoint later.

---

## LLM + Embeddings Providers (what affects UX)

### Embeddings (`EMBEDDING_PROVIDER`)
- `local` (default): uses sentence-transformers `BAAI/bge-small-en-v1.5`
  - heavier first-run load but no API cost
- `openai`: uses `text-embedding-3-small`
  - faster and consistent but requires `OPENAI_API_KEY`

### Generation (`GENERATION_PROVIDER`)
- `openai`: uses Chat Completions (model is hardcoded in code as `gpt-4o-mini`)
- `anthropic`: uses `ANTHROPIC_MODEL` (default `claude-sonnet-4-5-20250929`)

**Frontend**:
- If no generation key is configured, the UX should still work:
  - show retrieved chunks
  - show a banner like “Answer generation not configured; showing retrieved context only.”

---

## Suggested Frontend Screens (so an agent can implement quickly)

### 1) “Query” (main screen)
- Inputs:
  - Question textbox
  - `top_k` selector (1–20 recommended)
  - “Generate answer” toggle
- Outputs:
  - Answer panel (only if present)
  - Sources list (doc/page)
  - Latency + optional cost
  - Retrieved chunks list:
    - doc/page
    - score
    - collapsible chunk text preview

### 2) “Ingest” (admin screen)
- Single URL ingestion:
  - doc_name, doc_link, skip_if_exists toggle
- FinanceBench ingestion:
  - index input + button
- Bulk ingestion:
  - start, limit, skip_if_exists
  - results table (doc_name, chunks_upserted, errors)

### 3) “System / Metrics”
- Health indicator
- Metrics cards (count, p50/p95, avg cost)

---

## Runtime Assumptions (frontend integration)
- Backend is at `:8000`
- Frontend dev servers supported by CORS:
  - Vite (`5173`)
  - CRA (`3000`)
- Backend expects environment variables configured in `.env`
- Qdrant + Postgres may be:
  - local via Docker Compose
  - or external (cloud) via overrides
