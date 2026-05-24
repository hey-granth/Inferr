from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from inferr import server as server_module
from inferr.config import Config
from inferr.models import (
    ContextObject,
    DeepgramConfig,
    GeminiConfig,
    GroqConfig,
    SilkConfig,
)


def _test_config() -> Config:
    return Config(
        terminal_buffer_lines=50,
        history_depth=20,
        file_lines=150,
        language="english",
        ignored_dirs=[".git"],
        host="127.0.0.1",
        port=7331,
        silk=SilkConfig(),
        gemini=GeminiConfig(),
        groq=GroqConfig(api_key="test-key", model="llama-3.3-70b-versatile"),
        deepgram=DeepgramConfig(),
    )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("INFERR_NO_BROWSER", "1")
    server_module.CONFIG_OVERRIDE = _test_config()
    server_module.session_id = "test-session"
    server_module.context_assembler = None
    server_module.conversation_history = []
    server_module.tts_backend = None
    return TestClient(server_module.app)


def test_debug_pipeline_status(client: TestClient) -> None:
    response = client.get("/debug/pipeline/status")
    assert response.status_code == 200
    data = response.json()
    assert "stages" in data
    assert "LLM_OK" in data["stages"]


@pytest.mark.asyncio
async def test_debug_pipeline_llm(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    async def fake_query(*_args: object, **_kwargs: object) -> str:
        return "debug response"

    monkeypatch.setattr(server_module, "query_llm", fake_query)

    response = client.post("/debug/pipeline/llm", json={"transcript": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "debug response"
    assert data["pipeline_stage"]["stage"] == "LLM_OK"


def test_debug_pipeline_tts_no_backend(client: TestClient) -> None:
    response = client.post(
        "/debug/pipeline/tts",
        json={"text": "hello from inferr"},
    )
    assert response.status_code == 503
