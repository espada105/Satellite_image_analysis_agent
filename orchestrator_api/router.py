MCP_KEYWORDS = (
    "구름",
    "경계",
    "변화",
    "면적",
    "탐지",
    "분할",
    "비교",
    "달라졌",
    "cloud",
    "edge",
    "change",
    "detect",
    "segment",
)


def should_invoke_mcp(question: str, image_uri: str | None) -> bool:
    _ = question  # reserved for future rule tuning
    # If image exists, run analysis by default.
    if image_uri:
        return True
    return False
