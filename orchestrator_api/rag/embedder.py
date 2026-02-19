import math
import re
from collections import Counter

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9가-힣_]+")


def embed_text(text: str) -> Counter[str]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    return Counter(tokens)


def cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0

    common_terms = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common_terms)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
