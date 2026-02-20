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
- `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`: optional LLM synthesis

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
uv run uvicorn orchestrator_api.main:app --host 0.0.0.0 --port 8000 --reload

# MCP server (separate terminal)
uv run uvicorn mcp_satellite_server.server:app --host 0.0.0.0 --port 8100 --reload

# Start both (split processes) in one command
./scripts/dev_split_run.sh
```

Web UI:
- Login page: `http://127.0.0.1:8000/`
- Chatbot page: `http://127.0.0.1:8000/chatbot`

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

## Test commands

```bash
uv run pytest -q
uv run ruff check .
```
