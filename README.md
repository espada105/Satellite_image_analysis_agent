# Satellite Image Analysis Agent (MVP Skeleton)

## Architecture

```text
orchestrator_api/
  main.py
  schemas.py
  config.py
  router.py
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

## Run commands

```bash
# Orchestrator API
uv run uvicorn orchestrator_api.main:app --host 0.0.0.0 --port 8000 --reload

# MCP server (separate terminal)
uv run uvicorn mcp_satellite_server.server:app --host 0.0.0.0 --port 8100 --reload
```

## Test commands

```bash
uv run pytest -q
uv run ruff check .
```
