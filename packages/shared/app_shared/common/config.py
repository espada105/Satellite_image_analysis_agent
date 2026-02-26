import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SharedSettings:
    verified_user_ids: str = os.getenv("VERIFIED_USER_IDS", "")
    file_service_base_url: str = os.getenv("FILE_SERVICE_BASE_URL", "http://file-service:8011")
    retrieve_service_base_url: str = os.getenv("RETRIEVE_SERVICE_BASE_URL", "http://retrieve-service:8012")
    embedding_service_base_url: str = os.getenv("EMBEDDING_SERVICE_BASE_URL", "http://embedding-service:8013")
    health_service_base_url: str = os.getenv("HEALTH_SERVICE_BASE_URL", "http://health-service:8014")
    mcp_base_url: str = os.getenv("MCP_BASE_URL", "http://mcp:8100")


settings = SharedSettings()


def get_verified_user_ids() -> set[str]:
    return {item.strip() for item in settings.verified_user_ids.split(",") if item.strip()}
