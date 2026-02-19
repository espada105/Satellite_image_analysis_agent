from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader


def parse_document(path_or_text: str) -> tuple[str, str]:
    path = Path(path_or_text)
    if path.exists() and path.is_file():
        return str(path), _parse_file(path)
    return "inline", path_or_text


def _parse_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return BeautifulSoup(raw, "html.parser").get_text(" ")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return path.read_text(encoding="utf-8", errors="ignore")
