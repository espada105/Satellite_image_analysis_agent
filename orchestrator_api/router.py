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
    if not image_uri:
        return False

    normalized = question.lower()
    if any(keyword in question for keyword in MCP_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in MCP_KEYWORDS):
        return True
    return False
