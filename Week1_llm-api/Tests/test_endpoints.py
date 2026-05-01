from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        r = await c.get("/health")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_chat():
    mock = AsyncMock()
    mock.choices[0].message.content = "Hi!"
    with patch("app.llm.call_llm", return_value=mock):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            r = await c.post(
                "/chat", json={"messages": [{"role": "user", "content": "Hey"}]}
            )
    assert r.json()["content"] == "Hi!"
