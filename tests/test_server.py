from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from inferr.config import Config
import inferr.server as server_module


def _reset_server_state() -> None:
    server_module.session_id = None
    server_module.context_assembler = None
    server_module.conversation_history = []
    server_module.last_activity = None


def test_websocket_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERR_NO_BROWSER", "1")
    _reset_server_state()

    with TestClient(server_module.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "ping", "payload": ""})
            data = ws.receive_json()
            assert data["type"] == "pong"


@pytest.mark.asyncio
async def test_health_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERR_NO_BROWSER", "1")
    _reset_server_state()

    transport = ASGITransport(app=server_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_active"] is False


@pytest.mark.asyncio
async def test_session_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERR_NO_BROWSER", "1")
    _reset_server_state()

    class DummyAssembler:
        def __init__(self, config: Config) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.started = False

    monkeypatch.setattr(server_module, "ContextAssembler", DummyAssembler)

    transport = ASGITransport(app=server_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        start_response = await client.post("/session/start")
        stop_response = await client.post("/session/stop")

    assert start_response.status_code == 200
    assert start_response.json()["status"] == "started"
    assert stop_response.status_code == 200
    assert stop_response.json()["status"] == "stopped"
