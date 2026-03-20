# Session-Based General PDF RAG — Implementation Plan

## Goal
Convert the current FinanceBench-wired RAG backend into a **general-purpose, session-based PDF RAG system** where a user:

1. Creates a **session**
2. Uploads one or more **PDFs** to that session
3. Chats with the PDFs in that session (multi-turn)
4. Runs an **evaluation** of retrieval + generation quality (per session and/or offline harness)

**Session-based** means all indexing and retrieval is scoped by `session_id`, and no PDFs/chunks leak across sessions.

---

## Non-Goals (Phase 1)
- Full authentication / user accounts
- Team sharing / collaboration
- Full OCR for scanned PDFs (can be Phase 3)
- Large-scale sparse retrieval over all sessions (hybrid can come later)

---

## Current State (Baseline)
- Ingestion is coupled to FinanceBench dataset and downloads PDFs by URL.
- Chunks are embedded and stored in Qdrant collection `auditrag_chunks`.
- Retrieval modes:
  - Dense: Qdrant vector search
  - Sparse: in-memory BM25 built by scrolling *all* payloads from Qdrant
  - Hybrid: RRF fusion of dense+sparse
- Generation: OpenAI or Anthropic with citations.
- Telemetry: writes query logs to Postgres, `/metrics` aggregates.

---

## High-Level Design
### Core Entities
- **Session**: temporary workspace identified by a UUID `session_id`.
- **Document**: a PDF uploaded into a session, identified by a UUID `doc_id`.
- **Chunk**: text segment derived from a document; stored in Qdrant as a point with payload metadata.
- **Conversation Message**: chat messages per session (user/assistant), stored in Postgres.

### Storage
- **PDF file storage** (Phase 1): local disk
  - Path: `data/sessions/{session_id}/{doc_id}.pdf`
- **Vector storage**: Qdrant collection `auditrag_chunks`
- **Relational**: Postgres for:
  - chat message history
  - ingestion records (optional but recommended)
  - evaluation runs + results

---

## Qdrant Payload + Filtering (Critical)
### Payload schema (add these fields)
For each Qdrant point:
- `session_id: str`  
- `doc_id: str`  
- `doc_name: str` (original filename or user-provided name)
- `page: int`
- `text: str`
- `chunk_id: str` (unique)

### Chunk ID format
Use `doc_id` to avoid collisions:
- `chunk_id = "{doc_id}_chunk_{n}"`

### Retrieval filtering
All retrieval must filter by session:
- `session_id == {session_id}`

Optionally filter by selected documents:
- `doc_id in {doc_ids[]}`

**Result:** one shared Qdrant collection can safely support multiple sessions.

---

## API Design (Backend)

### 1) Session endpoints
#### `POST /sessions`
- Creates a new session.
- Returns:
  - `{ "session_id": "uuid" }`

Optional additions:
- session TTL (expires_at) for cleanup

#### `GET /sessions/{session_id}` (optional)
- Returns session metadata and document list.

---

### 2) Document upload + ingestion
#### `POST /sessions/{session_id}/documents`
- Content-Type: `multipart/form-data`
- Accepts: `files[]` (one or many)
- Behavior:
  1. Save file(s)
  2. Extract text
  3. Chunk
  4. Embed
  5. Upsert to Qdrant with payload including `session_id` and `doc_id`
- Returns:
  - list of `{ doc_id, filename, chunks_created, chunks_upserted }`

#### `GET /sessions/{session_id}/documents`
- Returns list of documents for UI.

#### `DELETE /sessions/{session_id}/documents/{doc_id}` (Phase 2/3)
- Deletes Qdrant points filtered by `(session_id, doc_id)` and removes stored PDF.

---

### 3) Chat endpoints
#### `POST /sessions/{session_id}/chat`
Request JSON:
- `message: string`
- `top_k: int` (optional)
- `doc_ids: string[] | null` (optional)

Behavior:
1. Load recent chat history from Postgres (last N turns)
2. Retrieve chunks from Qdrant filtered by `session_id` (+ doc_ids if provided)
3. Generate answer with citations using:
   - system prompt
   - short chat history
   - retrieved context
4. Store user and assistant messages

Response JSON:
- `answer: string`
- `sources: [{doc_name, page, doc_id}]`
- `chunks: [{text, doc_name, page, score, doc_id}]` (optional but recommended for UI/debug)
- `latency_ms: {retrieve_ms, generate_ms, total_ms}`

#### `GET /sessions/{session_id}/chat`
- Returns message history for UI.

---

### 4) Evaluation endpoints (Phase 2+)
Two useful layers:

#### A) Run evaluation with a question set
`POST /sessions/{session_id}/eval`
- Provide a list of questions (and optionally expected citations/answers)
- Run retrieval + generation for each
- Store results
- Return summary metrics

#### B) Fetch evaluation results
`GET /sessions/{session_id}/eval/{eval_id}`
- Returns details, per-question results, aggregate metrics

---

## Backend Code Changes (Where to Edit)

### A) Ingestion pipeline
Files today:
- `auditrag/ingestion/loader.py` (FinanceBench + URL download)
- `auditrag/ingestion/parser.py` (pdfplumber)
- `auditrag/ingestion/chunker.py` (word-window chunks)
- `auditrag/ingestion/embedder.py` (embeddings + Qdrant upsert)

Required changes:
1. Add a new ingestion function that accepts a **local PDF path** (not URL) and produces chunks.
2. Update chunk metadata to include `session_id`, `doc_id`, and `doc_name`.
3. Update `upsert_chunks_to_qdrant` to store new payload fields.

### B) Retrieval pipeline
Files today:
- `auditrag/retrieval/dense.py`
- `auditrag/retrieval/search.py`
- `auditrag/retrieval/sparse.py`
- `auditrag/retrieval/hybrid.py`

Required changes:
1. Update `search(...)` to accept `session_id` and optional `doc_ids`.
2. Dense search: use Qdrant `Filter` on `session_id` (+ doc_ids).
3. Phase 1 recommendation: set config to `RETRIEVAL_MODE=dense` for session-based MVP.

Sparse/hybrid note:
- Current BM25 implementation is global-corpus and not session-friendly.
- Reintroduce sparse later by either:
  - building BM25 per session
  - or switching to Qdrant sparse vectors / OpenSearch.

### C) Generation
Files today:
- `auditrag/generation/llm.py`
- `auditrag/generation/prompt.py`

Required changes:
1. Extend prompt builder to include a short chat history window.
2. Ensure citations remain doc/page based.

### D) Persistence (Postgres)
Existing:
- `auditrag/db/models.py` has `QueryLog`

Add tables (Phase 2):
- `Session` (optional)
- `Document` (recommended)
- `ChatMessage`
- `EvalRun`, `EvalItemResult`

---

## Data Model (Suggested SQLAlchemy Models)
### Document
- `doc_id (uuid str)`
- `session_id (uuid str)`
- `filename`
- `stored_path`
- `created_at`
- `chunks_created`, `chunks_upserted`

### ChatMessage
- `id`
- `session_id`
- `role` = `user | assistant`
- `content`
- `created_at`
- optional: `model_used`, `cost_usd`, `latency_ms`

### EvalRun / EvalItemResult
- EvalRun: `eval_id`, `session_id`, `created_at`, config snapshot
- EvalItemResult: question, retrieved sources, answer, judge scores, etc.

---

## Evaluation Strategy (Practical)

### Phase 1: lightweight regression
- Store every chat turn’s retrieved chunk metadata.
- Add a manual feedback endpoint:
  - thumbs up/down
  - “answer supported by citations?”

### Phase 2: structured eval runs
- Let a user define an eval set:
  - list of questions
  - optionally expected doc/page citations or expected answer snippets
- Compute:
  - **hit@k**: did any retrieved chunk match expected doc/page?
  - **recall@k** (if multiple expected sources)
  - **faithfulness** via LLM judge: is answer supported by context?

### Phase 3: offline harness
- A CLI tool that:
  1. creates a session
  2. uploads PDFs from a folder
  3. runs eval JSON/YAML
  4. outputs a report to `eval_results/`

---

## Cleanup / TTL (Phase 3)
Because sessions are temporary, implement cleanup:
- Session TTL (e.g., 24h or configurable)
- Cleanup job:
  - delete stored PDFs under `data/sessions/{session_id}`
  - delete Qdrant points by filter `session_id == ...`
  - delete DB records

---

## Implementation Phases (Step-by-Step)

### Phase 1 — MVP (General PDFs, session-scoped, single-turn)
1. Add `POST /sessions`
2. Add `POST /sessions/{session_id}/documents` (upload + ingest)
3. Update Qdrant payload to include `session_id` and `doc_id`
4. Update dense retrieval to filter by `session_id`
5. Add `POST /sessions/{session_id}/query` (or reuse `/query` but require session_id)
6. Frontend: create session, upload PDFs, ask questions

Deliverable: user can upload PDFs and ask questions grounded in their session.

### Phase 2 — Chat (multi-turn)
1. Add `ChatMessage` table
2. Add `POST /sessions/{session_id}/chat`
3. Prompt: include last N turns
4. Add `GET /sessions/{session_id}/chat`

Deliverable: full chat experience per session.

### Phase 3 — Evaluation
1. Add eval tables
2. Add `POST /sessions/{session_id}/eval`
3. Add report export (JSON)
4. Add human feedback capture

Deliverable: repeatable evaluation runs with stored metrics.

### Phase 4 — Scale + Hybrid
1. Replace global BM25 cache with a scalable sparse approach
2. Background jobs for ingestion
3. OCR for scanned PDFs

---

## Frontend Notes (for the UI agent)
Minimum UI screens:
- **Session start**: creates session + stores session_id in localStorage
- **Upload screen**: multi-file upload, show per-file ingest result
- **Chat screen**: chat transcript, streaming optional, show sources
- **Eval screen** (Phase 3): upload questions set, run eval, show metrics

---

## Risks / Gotchas
- **Scanned PDFs** produce no text with pdfplumber → plan OCR fallback.
- Current **sparse retrieval** design is not session-friendly; start dense-only.
- Qdrant filtering must be applied everywhere; otherwise sessions leak.
- Large PDFs can create many chunks → watch timeouts and add background jobs later.

