import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    text: str
    line_start: int
    line_end: int


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[TextChunk]:
    text = text.strip("\n")
    if not text:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    lines = text.splitlines()
    sections = _split_sections(lines)
    chunks: list[TextChunk] = []
    for section in sections:
        chunks.extend(
            _chunk_section_lines(
                lines=section["lines"],
                line_offset=section["start_line"],
                chunk_size=chunk_size,
                overlap=overlap,
                header=section["header"],
            )
        )
    return chunks


def _split_sections(lines: list[str]) -> list[dict]:
    if not lines:
        return []

    sections: list[dict] = []
    current = {"header": "", "start_line": 1, "lines": []}
    heading_pattern = re.compile(r"^\s{0,3}#{1,6}\s+(.*)")

    for idx, line in enumerate(lines, start=1):
        heading = heading_pattern.match(line)
        if heading:
            if current["lines"]:
                sections.append(current)
            current = {"header": heading.group(1).strip(), "start_line": idx, "lines": [line]}
            continue
        current["lines"].append(line)

    if current["lines"]:
        sections.append(current)
    return sections


def _chunk_section_lines(
    lines: list[str],
    line_offset: int,
    chunk_size: int,
    overlap: int,
    header: str,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    i = 0
    while i < len(lines):
        acc: list[str] = []
        start_idx = i
        total_len = 0

        while i < len(lines):
            next_line = lines[i]
            projected = total_len + len(next_line) + (1 if acc else 0)
            if acc and projected > chunk_size:
                break
            acc.append(next_line)
            total_len = projected
            i += 1

        chunk_text_value = "\n".join(acc).strip()
        if chunk_text_value:
            if header and not chunk_text_value.lower().startswith(f"section: {header}".lower()):
                chunk_text_value = f"Section: {header}\n{chunk_text_value}"
            chunks.append(
                TextChunk(
                    text=chunk_text_value,
                    line_start=line_offset + start_idx,
                    line_end=line_offset + i - 1,
                )
            )

        if i >= len(lines):
            break

        overlap_len = 0
        back = i - 1
        while back >= start_idx and overlap_len < overlap:
            overlap_len += len(lines[back]) + 1
            back -= 1
        i = max(start_idx + 1, back + 1)

    return chunks
