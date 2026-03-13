import asyncio

import httpx


async def main() -> None:
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        body = {
            "q": ["Hello world", "How are you?"],
            "source": "en",
            "target": "fr",
        }
        resp = await client.post("/language/translate/v2", json=body)
        print(resp.status_code)
        print(resp.json())


if __name__ == "__main__":
    asyncio.run(main())

