# auditRAG Implementation Plan

This plan turns your idea into a production-style RAG system for SEC filings, while keeping it beginner-friendly so you can learn by implementing each layer yourself.

## 1) Goal and Learning Outcomes

### Product Goal
Build `auditRAG`, a production-grade RAG API that answers financial questions from 10-K/10-Q filings with source citations, plus observability for latency, cost, and hallucination risk.

### Learning Goals
By the end, you should be comfortable with:
- Building a modular FastAPI backend.
- Designing an ingestion pipeline (PDF -> chunks -> embeddings -> vector DB).
- Implementing hybrid retrieval (dense + sparse + fusion + rerank).
- Prompt engineering for grounded answers with citations.
- Instrumenting latency and token-based cost tracking.
- Running objective evaluation with FinanceBench + RAGAS.
- Shipping with Docker and cloud deployment.

## 2) Final System Architecture (auditRAG)

Core flow:
1. `POST /ingest` receives docs (or FinanceBench references).
2. Pipeline extracts text, chunks it, embeds chunks, and stores in Qdrant + metadata.
3. `POST /query` performs hybrid retrieval:
   - Dense search (Qdrant vectors)
   - Sparse search (BM25)
   - RRF fusion
   - Optional Cohere reranking
4. Top chunks are injected into a strict prompt.
5. GPT-4o generates answer + explicit citations.
6. Observability logs latency, cost, and faithfulness signals to PostgreSQL.
7. `GET /metrics` and evaluation reports expose system quality over time.

## 3) Suggested Project Structure

Use this structure as your implementation target:

```text
auditRAG/
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── routers/
│   │   ├── ingest.py
│   │   ├── query.py
│   │   ├── metrics.py
│   │   └── health.py
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── retrieval/
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   ├── generation/
│   │   ├── prompt.py
│   │   └── llm.py
│   ├── observability/
│   │   ├── latency.py
│   │   ├── cost.py
│   │   └── hallucination.py
│   ├── evaluation/
│   │   ├── harness.py
│   │   └── metrics.py
│   ├── db/
│   │   ├── models.py
│   │   └── session.py
│   └── schemas/
│       ├── ingest.py
│       ├── query.py
│       └── metrics.py
├── data/
│   └── financebench/
├── scripts/
│   ├── bootstrap.py
│   └── ingest_financebench.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_api.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## 4) Phase-by-Phase Implementation (14 Days)

## Phase 1 (Days 1-2): Foundation and Environment

### What you build
- Python project scaffold.
- FastAPI app with `/health`.
- Local services in Docker Compose:
  - `qdrant`
  - `postgres`
  - `api`
- Environment config loading (`.env`).

### Why this matters
This gives you a stable base and prevents debugging retrieval logic while infra is broken.

### Deliverables
- `docker-compose up` starts all services.
- `/health` returns healthy status and dependency checks.
- DB connection test and Qdrant ping test pass.

### Learn while building
- FastAPI dependency injection basics.
- Docker networking (`service_name:port`).
- 12-factor config management.

---

## Phase 2 (Days 3-4): Ingestion Pipeline

### What you build
- FinanceBench loader (download metadata and filing links).
- PDF extraction (start with `pdfplumber`; fall back logic if parsing fails).
- Chunking strategy:
  - target size around 512 tokens
  - overlap around 50 tokens
- Embedding generation via `text-embedding-3-small`.
- Qdrant upsert with metadata fields:
  - `company`
  - `doc_name`
  - `fiscal_year`
  - `page`
  - `chunk_id`
  - `source_url`

### Why this matters
Retrieval quality is mostly decided by chunking quality and metadata discipline.

### Deliverables
- One command/script to ingest a subset of FinanceBench docs.
- Qdrant collection created and populated.
- You can inspect random payloads and confirm metadata correctness.

### Learn while building
- Tradeoff between chunk size and recall precision.
- Idempotent ingestion patterns (safe re-runs).
- API rate limits and retry handling for embeddings.

---

## Phase 3 (Days 5-6): Retrieval Engine

### What you build
1. Dense retrieval from Qdrant.
2. Sparse retrieval using BM25 (`rank_bm25`) over same chunk corpus.
3. Reciprocal Rank Fusion (RRF) to combine dense + sparse results.
4. Optional Cohere rerank pass on fused candidates.

### Why this matters
Hybrid retrieval reduces failure modes where either pure semantic or pure keyword search misses critical financial context.

### Deliverables
- Retrieval debug mode that returns:
  - dense hits
  - sparse hits
  - fused ranking
  - reranked final top-k
- Configurable `top_k` and weights.

### Learn while building
- Why BM25 still matters in finance (exact terms, line items, ratios).
- Rank fusion intuition (agreement across retrievers).
- Practical reranking cost/latency tradeoff.

---

## Phase 4 (Days 7-8): Generation Layer

### What you build
- Prompt template with strict grounding rules:
  - answer only from provided context
  - cite chunk sources
  - explicitly say insufficient evidence when needed
- LLM wrapper for GPT-4o.
- Token accounting from request/response usage fields.

### Why this matters
Good retrieval can still fail if prompting allows model improvisation.

### Deliverables
- `POST /query` returns:
  - answer text
  - structured citations
  - retrieval metadata
- Negative test cases where model correctly refuses unsupported answers.

### Learn while building
- Grounding instructions vs verbosity instructions.
- Designing response schemas that are API-consumer friendly.
- Prompt versioning for reproducibility.

---

## Phase 5 (Day 9): Observability Layer

### What you build
- Stage-level timers:
  - embed
  - retrieve
  - rerank
  - generate
  - total
- Cost estimation from token usage and model pricing config.
- Hallucination risk scoring with RAGAS faithfulness.
- PostgreSQL logging table(s) for query telemetry.

### Why this matters
Observability is the difference between "works once" and "can be trusted in production."

### Deliverables
- Every query produces a log record.
- `GET /metrics` can show:
  - p50/p95 latency
  - average cost per query
  - recent faithfulness trend

### Learn while building
- Basic SRE thinking for AI APIs.
- How to pick meaningful SLIs for RAG systems.

---

## Phase 6 (Day 10): API Surface and Error Handling

### What you build
- Endpoints:
  - `POST /ingest`
  - `POST /query`
  - `GET /metrics`
  - `GET /health`
- Pydantic request/response schemas.
- Uniform error shape for clients.
- Sensible timeouts and retry boundaries.

### Why this matters
A clean API contract makes your system usable by frontends and external consumers.

### Deliverables
- OpenAPI docs are clear and accurate.
- Input validation catches malformed requests early.
- 4xx/5xx errors include actionable details.

### Learn while building
- API ergonomics and schema-first design.
- Exception middleware patterns in FastAPI.

---

## Phase 7 (Days 11-12): Evaluation Harness (FinanceBench)

### What you build
- Harness to run benchmark questions through full pipeline.
- RAGAS metrics:
  - context precision
  - context recall
  - faithfulness
  - answer relevance
- Result persistence in PostgreSQL.
- Summary report generation.

### Why this matters
You need objective evidence your changes improve system quality.

### Deliverables
- Reproducible eval run command.
- Baseline score snapshot saved.
- Breakdown by failure mode (retrieval miss vs generation miss).

### Learn while building
- Evaluation-first iteration loops.
- Interpreting metric deltas without overfitting prompts.

---

## Phase 8 (Days 13-14): Containerization and Deployment

### What you build
- Production-ready `Dockerfile`.
- Final `docker-compose.yml` for local full stack.
- Deployment target (Railway or Render).
- README with:
  - architecture
  - setup
  - API usage
  - eval steps
  - known limitations

### Why this matters
Deployability and documentation are critical for real-world and portfolio value.

### Deliverables
- Clean `docker compose up` from fresh clone.
- Live deployment URL.
- End-to-end smoke test script.

### Learn while building
- Runtime config in hosted environments.
- Minimal production hardening checklist.

## 5) Quality Gates (Do Not Skip)

Add these gates before moving to next phase:
- Unit tests for each new core module.
- Basic integration tests for `/ingest` and `/query`.
- Retrieval sanity tests (known question -> expected supporting chunk appears in top-k).
- Lint + type check pass.
- Small load test (concurrent queries) to observe bottlenecks.

## 6) Recommended Milestone Metrics

Track these over time:
- Retrieval:
  - Recall@k (where possible)
  - MRR / NDCG (if labels available)
- Generation:
  - RAGAS faithfulness
  - answer relevance
- System:
  - p50/p95 latency
  - avg cost/query
  - error rate

Set initial targets (example, adjust later):
- p95 total latency < 8s
- avg cost/query < $0.03
- faithfulness >= 0.75 baseline

## 7) Suggested Weekly Build Rhythm

Use this cycle continuously:
1. Implement one small feature.
2. Add/adjust tests.
3. Run quick evaluation slice.
4. Inspect metrics/logs.
5. Improve retrieval/prompt based on evidence.

This keeps learning high and avoids "big bang" debugging.

## 8) Common Pitfalls and How to Avoid Them

- Overly large chunks -> noisy retrieval  
  - Fix: tighten chunk size and preserve section headers in metadata.
- Missing source attribution in answers  
  - Fix: enforce citation schema in response model.
- Hallucination despite good retrieval  
  - Fix: stronger refusal instruction + lower generation temperature.
- Cost blowups during evaluation  
  - Fix: run eval in sampled batches first; cache embeddings aggressively.
- Slow queries due to serial calls  
  - Fix: parallelize independent steps when safe.

## 9) Immediate Next Steps (Today)

1. Initialize scaffold with the directory layout above.
2. Create `.env.example` and config loader.
3. Write `docker-compose.yml` for FastAPI + Qdrant + Postgres.
4. Implement `/health` with dependency checks.
5. Verify local stack works end-to-end before writing RAG logic.

Once these are done, start Phase 2 ingestion on a small FinanceBench subset (5-10 docs), not the full dataset.
