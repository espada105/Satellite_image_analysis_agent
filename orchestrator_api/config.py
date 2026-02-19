import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mcp_base_url: str = os.getenv("MCP_BASE_URL", "http://127.0.0.1:8100")
    rag_index_name: str = os.getenv("RAG_INDEX_NAME", "default")


settings = Settings()


def get_verified_user_ids() -> set[str]:
    raw = os.getenv("VERIFIED_USER_IDS", "")
    return {user_id.strip() for user_id in raw.split(",") if user_id.strip()}
