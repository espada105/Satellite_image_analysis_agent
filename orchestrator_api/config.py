import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mcp_base_url: str = os.getenv("MCP_BASE_URL", "http://127.0.0.1:8100")
    rag_index_name: str = os.getenv("RAG_INDEX_NAME", "default")


settings = Settings()
