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
            chunks.append(
                TextChunk(
                    text=chunk_text_value,
                    line_start=start_idx + 1,
                    line_end=i,
                )
            )

        if i >= len(lines):
            break

        # approximate overlap by lines using character budget
        overlap_len = 0
        back = i - 1
        while back >= start_idx and overlap_len < overlap:
            overlap_len += len(lines[back]) + 1
            back -= 1
        i = max(start_idx + 1, back + 1)

    return chunks
