import hashlib


def embed_text(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [int.from_bytes(digest[i : i + 4], "big") for i in range(0, dim * 4, 4)]
    norm = max(sum(values), 1)
    return [round(v / norm, 8) for v in values]
