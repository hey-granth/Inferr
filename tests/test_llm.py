from __future__ import annotations

from datetime import datetime, timezone

import pytest

from inferr.config import Config
from inferr.llm import build_system_prompt, query_llm
from inferr.models import (
    ContextObject,
    DeepgramConfig,
    GeminiConfig,
    GroqConfig,
    QueryRequest,
    SilkConfig,
)


def test_build_system_prompt_hinglish() -> None:
    prompt = build_system_prompt("hinglish")
    assert "hinglish" in prompt.lower()


def test_build_system_prompt_english() -> None:
    prompt = build_system_prompt("english")
    assert "bhai" not in prompt.lower()
    assert "yaar" not in prompt.lower()


def _base_context(history: list[tuple[str, str]] | None = None) -> ContextObject:
    turns = []
    if history:
        for role, content in history:
            turns.append({"role": role, "content": content})
    return ContextObject(
        terminal_buffer=["line"],
        shell_history=["ls"],
        active_file=None,
        flagged_errors=[],
        conversation_history=turns,
        session_id="session",
        timestamp=datetime.now(timezone.utc),
    )


def _base_config() -> Config:
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


def _install_fake_groq(
    monkeypatch: pytest.MonkeyPatch,
    recorded: dict[str, object],
    *,
    text: str = "ok",
    choices: list[object] | None = None,
    create_raises: Exception | None = None,
) -> None:
    class FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class FakeChoice:
        def __init__(self, content: str, finish_reason: str = "stop") -> None:
            self.message = FakeMessage(content)
            self.finish_reason = finish_reason

    class FakeResponse:
        def __init__(self, response_choices: list[object]) -> None:
            self.choices = response_choices

    class FakeCompletions:
        def create(self, **kwargs: object) -> FakeResponse:
            if create_raises is not None:
                raise create_raises
            recorded.update(kwargs)
            if choices is not None:
                return FakeResponse(choices)
            return FakeResponse([FakeChoice(text)])

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.chat = FakeChat()

    monkeypatch.setattr("inferr.llm.Groq", FakeClient)


@pytest.mark.asyncio
async def test_query_llm_uses_correct_model(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, object] = {}
    _install_fake_groq(monkeypatch, recorded)

    request = QueryRequest(transcript="hello", context=_base_context())
    result = await query_llm(request, _base_config())

    assert result == "ok"
    assert recorded.get("model") == "llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_query_llm_injects_context_json(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, object] = {}
    _install_fake_groq(monkeypatch, recorded)

    request = QueryRequest(transcript="hello", context=_base_context())
    await query_llm(request, _base_config())

    messages = recorded.get("messages")
    assert isinstance(messages, list)
    assert messages
    last_message = messages[-1]
    assert isinstance(last_message, dict)
    assert "<context>" in str(last_message.get("content", ""))
    assert "</context>" in str(last_message.get("content", ""))


@pytest.mark.asyncio
async def test_query_llm_raises_runtime_error_on_api_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}
    _install_fake_groq(
        monkeypatch,
        recorded,
        create_raises=Exception("quota exceeded"),
    )

    request = QueryRequest(transcript="hello", context=_base_context())

    with pytest.raises(RuntimeError, match="Groq API error"):
        await query_llm(request, _base_config())


@pytest.mark.asyncio
async def test_query_llm_returns_empty_on_no_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}
    _install_fake_groq(monkeypatch, recorded, choices=[])

    request = QueryRequest(transcript="hello", context=_base_context())
    result = await query_llm(request, _base_config())

    assert result == ""


@pytest.mark.asyncio
async def test_query_llm_history_uses_assistant_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}
    _install_fake_groq(monkeypatch, recorded)

    history = [("user", "u1"), ("assistant", "a1"), ("user", "u2")]
    request = QueryRequest(transcript="hello", context=_base_context(history=history))
    await query_llm(request, _base_config())

    messages = recorded.get("messages")
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "user"
