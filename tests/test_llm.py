from __future__ import annotations

from datetime import datetime, timezone

import pytest

from inferr.config import Config
from inferr.llm import build_system_prompt, query_llm
from inferr.models import ContextObject, QueryRequest


def test_build_system_prompt_hinglish() -> None:
    prompt = build_system_prompt("hinglish")
    assert "hinglish" in prompt.lower()
    assert "bhai" in prompt.lower()


def test_build_system_prompt_english() -> None:
    prompt = build_system_prompt("english")
    assert "bhai" not in prompt.lower()
    assert "yaar" not in prompt.lower()


@pytest.mark.asyncio
async def test_query_llm_uses_model(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, object] = {}

    class FakeBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeResponse:
        def __init__(self) -> None:
            self.content = [FakeBlock("ok")]

    class FakeMessages:
        async def create(self, **kwargs: object) -> FakeResponse:
            recorded.update(kwargs)
            return FakeResponse()

    class FakeClient:
        def __init__(self) -> None:
            self.messages = FakeMessages()

    def _fake_client() -> FakeClient:
        return FakeClient()

    monkeypatch.setattr("inferr.llm.AsyncAnthropic", _fake_client)

    context = ContextObject(
        terminal_buffer=["line"],
        shell_history=["ls"],
        active_file=None,
        flagged_errors=[],
        conversation_history=[],
        session_id="session",
        timestamp=datetime.now(timezone.utc),
    )
    request = QueryRequest(transcript="hello", context=context)
    config = Config(
        terminal_buffer_lines=50,
        history_depth=20,
        file_lines=150,
        language="english",
        ignored_dirs=[".git"],
        host="127.0.0.1",
        port=7331,
    )

    result = await query_llm(request, config)

    assert result == "ok"
    assert recorded.get("model") == "claude-sonnet-4-6"
    messages = recorded.get("messages")
    assert isinstance(messages, list)
    assert "<context>" in messages[-1]["content"]
    assert "</context>" in messages[-1]["content"]
