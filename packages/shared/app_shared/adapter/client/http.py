import httpx


async def post_json(url: str, payload: dict, headers: dict[str, str] | None = None) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30.0) as client:
        return await client.post(url, json=payload, headers=headers)


async def get_json(url: str, headers: dict[str, str] | None = None) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30.0) as client:
        return await client.get(url, headers=headers)
