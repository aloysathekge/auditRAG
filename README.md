# auditRAG

Production-style RAG API for financial document intelligence (10-K/10-Q filings): ingest PDFs, hybrid retrieval (dense + sparse + optional rerank), LLM answers with citations, and observability (latency, cost, query logs).

## Stack

- **API:** FastAPI
- **Vector DB:** Qdrant (local or [Qdrant Cloud](https://cloud.qdrant.io))
- **Relational DB:** PostgreSQL (query telemetry, metrics)
- **Embeddings:** OpenAI `text-embedding-3-small` or local `BAAI/bge-small-en-v1.5`
- **Generation:** OpenAI  or Anthropic Claude

## Quick start (local)

```bash
# Dependencies (uv)
uv sync

# Optional: run Postgres + Qdrant via Docker
docker compose up -d postgres qdrant

# Or use Qdrant Cloud + any Postgres; set in .env:
# QDRANT_URL_OVERRIDE, QDRANT_API_KEY, POSTGRES_HOST, etc.

cp .env.example .env   # edit with your keys and URLs
uv run uvicorn auditrag.main:app --host 0.0.0.0 --port 8000
```

- **Health:** `GET http://localhost:8000/health`
- **Query:** `POST http://localhost:8000/query` with `{"question": "...", "top_k": 5, "generate_answer": true}`
- **Ingest:** `POST http://localhost:8000/ingest/financebench?index=0` or bulk `POST .../ingest/financebench/bulk?limit=10`
- **Metrics:** `GET http://localhost:8000/metrics`

## CLI

```bash
# Interactive Q&A
uv run python -m auditrag

# One-off retrieval + answer
uv run python -m auditrag.retrieval "What was 3M capital expenditure?" -a -k 5

# Bulk ingest FinanceBench (unique docs in range)
uv run python scripts/ingest_financebench_bulk.py --limit 20

# Evaluation
uv run python -m auditrag.evaluation --limit 10
# or: uv run python scripts/run_eval.py --limit 10
```

## Tests

```bash
uv run pytest
```

## Deploying on EC2

### What you need before deploy

1. **PostgreSQL** – RDS or a Postgres instance reachable from EC2 (create DB and user; set `POSTGRES_HOST`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` in `.env`).
2. **Qdrant** – Qdrant Cloud (recommended) or self-hosted. Set `QDRANT_URL_OVERRIDE` and `QDRANT_API_KEY` (or `QDRANT_HOST`/`QDRANT_PORT` for local).
3. **API keys** – At least one of `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`; set `EMBEDDING_PROVIDER` and `GENERATION_PROVIDER` in `.env`.
4. **Python 3.11+** on the EC2 instance (or use the Docker image).

### Option A: Run with Docker on EC2

1. Launch an EC2 instance (e.g. Ubuntu 22.04, t3.small or larger). Open security group: 22 (SSH), 8000 (or 80 if you put a reverse proxy in front).
2. Install Docker (and Docker Compose v2):

   ```bash
   sudo apt-get update && sudo apt-get install -y ca-certificates curl
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   sudo chmod a+r /etc/apt/keyrings/docker.gpg
   echo "deb [arch=$(dpkg --print-architecture)] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli docker-compose-plugin
   ```

3. Clone the repo and configure env:

   ```bash
   git clone <your-repo-url> auditRAG && cd auditRAG
   cp .env.example .env
   # Edit .env: POSTGRES_HOST=<rds-or-ip>, QDRANT_URL_OVERRIDE, QDRANT_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY
   ```

4. Run **only the API** container (Postgres and Qdrant are external):

   ```bash
   docker compose run -d --name auditrag-api -p 8000:8000 --env-file .env api
   ```

   Or build and run the image yourself:

   ```bash
   docker build -t auditrag .
   docker run -d --name auditrag-api -p 8000:8000 --env-file .env auditrag
   ```

5. Check: `curl http://localhost:8000/health` (and from outside: `http://<EC2-public-ip>:8000/health` if port 8000 is open).

### Option B: Run without Docker on EC2

1. Install Python (3.11+; Ubuntu 24.04 has 3.12), clone or upload repo, create venv:

   ```bash
   sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-dev git
   git clone <your-repo-url> auditRAG && cd auditRAG
   # Or upload from laptop: scp -i your-key.pem -r ./auditRAG ubuntu@<ec2-ip>:~/
   python3 -m venv .venv && source .venv/bin/activate
   pip install uv && uv sync
   ```

2. Create `.env` from `.env.example` and set `POSTGRES_HOST`, `QDRANT_URL_OVERRIDE`, `QDRANT_API_KEY`, and LLM keys.

3. Run the app (production, no reload):

   ```bash
   uv run uvicorn auditrag.main:app --host 0.0.0.0 --port 8000
   ```

   Or run under systemd (example unit `/etc/systemd/system/auditrag.service`):

   ```ini
   [Unit]
   Description=auditRAG API
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/auditRAG
   EnvironmentFile=/home/ubuntu/auditRAG/.env
   ExecStart=/home/ubuntu/auditRAG/.venv/bin/uv run uvicorn auditrag.main:app --host 0.0.0.0 --port 8000
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

   Then: `sudo systemctl daemon-reload && sudo systemctl enable --now auditrag`.

### After deploy

- **HTTPS / domain:** Put Nginx (or Caddy) in front of the app, bind to 80/443, proxy to `http://127.0.0.1:8000`.
- **Firewall:** Restrict 8000 to localhost if you use a reverse proxy; only expose 80/443 and 22.
- **Ingestion:** Run bulk ingest from your machine or a one-off job on EC2:  
  `uv run python scripts/ingest_financebench_bulk.py --limit 20` (ensure `.env` has the same Qdrant/Postgres as the API).

## Configuration (.env)

| Variable | Description |
|----------|-------------|
| `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | PostgreSQL (defaults assume Docker Compose names) |
| `QDRANT_HOST`, `QDRANT_PORT` | Local Qdrant |
| `QDRANT_URL_OVERRIDE`, `QDRANT_API_KEY` | Qdrant Cloud (overrides host/port) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | LLM and optional embedding keys |
| `EMBEDDING_PROVIDER` | `openai` or `local` |
| `GENERATION_PROVIDER` | `openai` or `anthropic` |
| `RETRIEVAL_MODE` | `dense`, `sparse`, or `hybrid` |
| `USE_RERANKER`, `COHERE_API_KEY` | Optional Cohere rerank |

## License

MIT.
