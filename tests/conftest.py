import os
import tempfile
from pathlib import Path

# Ensure tests never use the development vector DB.
TEST_DB_PATH = Path(tempfile.gettempdir()) / "satellite_agent_test_rag_store.sqlite3"
os.environ["RAG_STORE_DB_PATH"] = str(TEST_DB_PATH)


def pytest_sessionstart(session) -> None:  # noqa: ARG001
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ARG001
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
