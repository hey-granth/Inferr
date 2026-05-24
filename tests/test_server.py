from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from inferr.config import Config
import inferr.server as server_module
from inferr.models import ContextObject, ConversationTurn, FlaggedError


def _reset_server_state() -> None:
    server_module.session_id = None
    server_module.context_assembler = None
    server_module.conversation_history = []
    server_module.tts_backend = None
    server_module.last_activity = None
    server_module._shell_command_buffer.clear()


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
async def test_context_endpoint_no_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERR_NO_BROWSER", "1")
    _reset_server_state()

    transport = ASGITransport(app=server_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/context")

    assert response.status_code == 404
    assert response.json()["error"] == "No active session"


@pytest.mark.asyncio
async def test_health_includes_tts_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERR_NO_BROWSER", "1")
    _reset_server_state()

    transport = ASGITransport(app=server_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    payload = response.json()
    assert "tts_backend" in payload


@pytest.mark.asyncio
async def test_browser_config_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERR_NO_BROWSER", "1")
    _reset_server_state()

    transport = ASGITransport(app=server_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/config/browser")

    assert response.status_code == 200
    payload = response.json()
    assert "tts_backend" in payload
    assert "deepgram_enabled" in payload


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


def test_websocket_no_session_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERR_NO_BROWSER", "1")
    _reset_server_state()

    with TestClient(server_module.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "transcript", "payload": "hello"})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "No active session" in data["message"]


def test_resolve_tone_urgent_with_errors() -> None:
    context = ContextObject(
        terminal_buffer=[],
        shell_history=[],
        active_file=None,
        flagged_errors=[
            FlaggedError(type="marker", summary="[!]", raw=""),
            FlaggedError(type="python_traceback", summary="KeyError", raw="..."),
        ],
        conversation_history=[],
        session_id="test",
        timestamp=datetime.now(timezone.utc),
    )

    assert server_module.resolve_tone(context) == "urgent"


def test_resolve_tone_neutral_on_first_query() -> None:
    context = ContextObject(
        terminal_buffer=[],
        shell_history=[],
        active_file=None,
        flagged_errors=[],
        conversation_history=[],
        session_id="test",
        timestamp=datetime.now(timezone.utc),
    )

    assert server_module.resolve_tone(context) == "neutral"


def test_resolve_tone_neutral_otherwise() -> None:
    context = ContextObject(
        terminal_buffer=[],
        shell_history=[],
        active_file=None,
        flagged_errors=[],
        conversation_history=[ConversationTurn(role="user", content="hi")],
        session_id="test",
        timestamp=datetime.now(timezone.utc),
    )

    assert server_module.resolve_tone(context) == "neutral"


@pytest.mark.asyncio
async def test_capture_command_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERR_NO_BROWSER", "1")
    _reset_server_state()
    transport = ASGITransport(app=server_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/capture/command",
            json={"command": "python3 app.py", "exit_code": 0},
        )
    assert response.status_code == 200
    assert "python3 app.py" in server_module._shell_command_buffer


@pytest.mark.asyncio
async def test_capture_command_exit_code_annotated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERR_NO_BROWSER", "1")
    _reset_server_state()
    transport = ASGITransport(app=server_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/capture/command",
            json={"command": "bad_cmd", "exit_code": 127},
        )
    assert any("exit 127" in entry for entry in server_module._shell_command_buffer)
