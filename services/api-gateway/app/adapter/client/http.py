import httpx


async def post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


async def get_json(url: str, headers: dict[str, str] | None = None) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
