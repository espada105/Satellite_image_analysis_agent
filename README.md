# Satellite Image Analysis Agent (MVP Skeleton)

## Architecture

```text
orchestrator_api/
  main.py
  schemas.py
  config.py
  router.py
  security.py
  mcp_client.py
  services/
    chat_service.py
  rag/
    ingest.py
    retrieve.py
    parser.py
    chunker.py
    embedder.py
    store.py
mcp_satellite_server/
  server.py
  schemas.py
  opencv_ops.py
  utils.py
data/
  docs/
  imagery/
tests/
  test_auth.py
  test_chat.py
  test_ingest.py
  test_router.py
  test_mcp_tool.py
pyproject.toml
```

## Environment setup with uv

```bash
# 1) Python version pin (already set by .python-version)
uv python install 3.11

# 2) Create venv
uv venv

# 3) Install dependencies (+dev)
uv sync --extra dev
```

## Environment variables

Use `.env.example` as reference. The app loads `.env` automatically.

- `VERIFIED_USER_IDS`: comma-separated allowed user IDs
- `MCP_BASE_URL`: MCP server base URL
- `RAG_STORE_DB_PATH`: SQLite path for persistent vector store
- `RAG_MIN_SCORE`: minimum retrieval score threshold
- `RAG_SPARSE_MODEL`: sparse retriever model id (default: `telepix/PIXIE-Splade-v1.0`)
- `RAG_SPARSE_MIN_WEIGHT`: SPLADE token weight cutoff
- `RAG_HYBRID_ALPHA`: hybrid score weight (semantic vs BM25)
- `RAG_BM25_K1`, `RAG_BM25_B`: BM25 parameters
- `RAG_RERANK_HEADING_BOOST`: heading-coverage rerank boost
- `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`: optional LLM synthesis
- `USE_LANGCHAIN_PIPELINE`: chat orchestration path toggle (`true` default, set `false` for legacy path)
- `LOG_LEVEL`, `LOG_JSON`: logging controls
- `ENABLE_METRICS`: expose `/metrics` Prometheus endpoint
- `RATE_LIMIT_ENABLED`, `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`: API rate limiting

## LangChain migration mode

- Default runtime path is LangChain pipeline (`USE_LANGCHAIN_PIPELINE=true`).
- Current retrieval model/scoring remains custom (`telepix/PIXIE-Splade-v1.0` + hybrid rerank), wrapped by a LangChain retriever adapter.
- MCP image analysis is invoked through a LangChain `StructuredTool` wrapper.

To force the previous legacy orchestration path:

```bash
export USE_LANGCHAIN_PIPELINE=false
```

## Access control (verified IDs)

When `VERIFIED_USER_IDS` is configured, `/ingest` and `/chat` require `x-user-id` header.

```bash
export VERIFIED_USER_IDS="alice,bob"
```

Request example:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "x-user-id: alice" \
  -d '{"question":"구름 영향 알려줘"}'
```

## Run commands

```bash
# Orchestrator API
uv run uvicorn orchestrator_api.main:app \
  --host 0.0.0.0 --port 8000 --reload \
  --reload-dir orchestrator_api --reload-dir frontend \
  --reload-exclude ".venv/*" --reload-exclude "data/*"

# MCP server (separate terminal)
uv run satellite-mcp-server

# or (dev reload)
uv run uvicorn mcp_satellite_server.server:app \
  --host 0.0.0.0 --port 8100 --reload \
  --reload-dir mcp_satellite_server \
  --reload-exclude ".venv/*" --reload-exclude "data/*"

# Start both (split processes) in one command
./scripts/dev_split_run.sh
```

Web UI:
- Login page: `http://127.0.0.1:8000/`
- Chatbot page: `http://127.0.0.1:8000/chatbot`

## Operational quality

- App startup initialization uses FastAPI `lifespan` (startup event warning removed).
- Structured access logs are emitted with `request_id`, method/path, status, latency.
- In-memory rate limiting is enabled by default (per `x-user-id`, fallback IP).
- Prometheus text metrics endpoint: `GET /metrics`.

Metrics examples:

```bash
curl http://127.0.0.1:8000/metrics
```

Key metrics:
- `http_requests_total`
- `http_requests_by_status_total{status="..."}`
- `http_request_latency_ms_sum`, `http_request_latency_ms_count`
- `http_rate_limited_total`

## Search quality upgrades

- Header-aware chunking for markdown-style docs (`#`, `##`, ...).
- Hybrid retrieval score: semantic sparse retrieval + BM25 lexical retrieval.
- Lightweight reranking boost for heading/query term coverage.

Reindex docs into vector store:

```bash
curl -X POST http://127.0.0.1:8000/reindex-docs \
  -H "x-user-id: alice"
```

If you get `curl: (7) Failed to connect`:
1. Start orchestrator server on port `8000`.
2. Start MCP server on port `8100`.
3. Retry the command.

Test isolation:
- Tests use a separate SQLite vector DB at `/tmp/satellite_agent_test_rag_store.sqlite3`.
- Development/runtime DB path comes from `RAG_STORE_DB_PATH`.

## Image upload from frontend

- Chatbot form supports local image file upload.
- Uploaded files are saved under `data/imagery/uploads`.
- The backend returns `image_uri` and uses it for MCP analysis.
- MCP analysis artifacts (mask/edge results) are saved under `data/imagery/artifacts`.
- Artifact preview URLs are returned as `/imagery/artifacts/<file>.png`.
- MCP server exposes standard MCP streamable HTTP endpoint at `/mcp`.

## Test commands

```bash
uv run pytest -q
uv run ruff check .
```

## Deployment packaging

Included templates:
- `Dockerfile.orchestrator`
- `Dockerfile.mcp`
- `docker-compose.yml` (orchestrator + mcp + caddy reverse proxy)
- `deploy/Caddyfile` (HTTPS via Caddy + domain)
- `deploy/grafana-dashboard.json` (Grafana import template)
- `.github/workflows/ci.yml` (lint/test + docker build)

Example:

```bash
export DOMAIN=your-domain.example.com
export VERIFIED_USER_IDS=alice,bob
export LLM_API_KEY=...
docker compose up -d --build
```
