#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cleanup() {
  if [[ -n "${MCP_PID:-}" ]]; then
    kill "$MCP_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${API_PID:-}" ]]; then
    kill "$API_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

echo "[dev] starting MCP server on :8100"
uv run uvicorn mcp_satellite_server.server:app \
  --host 0.0.0.0 \
  --port 8100 \
  --reload \
  --reload-dir mcp_satellite_server \
  --reload-exclude ".venv/*" \
  --reload-exclude "data/*" &
MCP_PID=$!

echo "[dev] starting Orchestrator API on :8000"
uv run uvicorn orchestrator_api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --reload-dir orchestrator_api \
  --reload-dir frontend \
  --reload-exclude ".venv/*" \
  --reload-exclude "data/*" &
API_PID=$!

echo "[dev] servers are running. press Ctrl+C to stop."
wait "$MCP_PID" "$API_PID"
