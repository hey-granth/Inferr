from __future__ import annotations

from datetime import datetime, timezone

import pytest

from inferr.config import Config
from inferr.llm import build_system_prompt, query_llm
from inferr.models import (
    ContextObject,
    DeepgramConfig,
    ElevenLabsConfig,
    GeminiConfig,
    QueryRequest,
    SilkConfig,
)


def test_build_system_prompt_hinglish() -> None:
    prompt = build_system_prompt("hinglish")
    assert "hinglish" in prompt.lower()
    assert "bhai" in prompt.lower()


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
        elevenlabs=ElevenLabsConfig(),
        gemini=GeminiConfig(api_key="test-key", model="gemini-flash-latest"),
        deepgram=DeepgramConfig(),
    )


@pytest.mark.asyncio
async def test_query_llm_uses_correct_model(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, object] = {}

    class FakePart:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeContent:
        def __init__(self, text: str) -> None:
            self.parts = [FakePart(text)]

    class FakeCandidate:
        def __init__(self, text: str) -> None:
            self.content = FakeContent(text)

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.candidates = [FakeCandidate(text)]

    class FakeModels:
        def generate_content(self, **kwargs: object) -> FakeResponse:
            recorded.update(kwargs)
            return FakeResponse("ok")

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.models = FakeModels()

    monkeypatch.setattr("inferr.llm.genai.Client", FakeClient)

    request = QueryRequest(transcript="hello", context=_base_context())
    result = await query_llm(request, _base_config())

    assert result == "ok"
    assert recorded.get("model") == "gemini-flash-latest"


@pytest.mark.asyncio
async def test_query_llm_injects_context_json(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, object] = {}

    class FakePart:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeContent:
        def __init__(self, text: str) -> None:
            self.parts = [FakePart(text)]

    class FakeCandidate:
        def __init__(self, text: str) -> None:
            self.content = FakeContent(text)

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.candidates = [FakeCandidate(text)]

    class FakeModels:
        def generate_content(self, **kwargs: object) -> FakeResponse:
            recorded.update(kwargs)
            return FakeResponse("ok")

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    monkeypatch.setattr("inferr.llm.genai.Client", FakeClient)

    request = QueryRequest(transcript="hello", context=_base_context())
    await query_llm(request, _base_config())

    contents = recorded.get("contents")
    assert isinstance(contents, list)
    assert contents
    last_text = contents[-1].parts[0].text
    assert "<context>" in last_text
    assert "</context>" in last_text


@pytest.mark.asyncio
async def test_query_llm_raises_runtime_error_on_api_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModels:
        def generate_content(self, **kwargs: object) -> object:
            raise Exception("quota exceeded")

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    monkeypatch.setattr("inferr.llm.genai.Client", FakeClient)

    request = QueryRequest(transcript="hello", context=_base_context())

    with pytest.raises(RuntimeError, match="Gemini API error"):
        await query_llm(request, _base_config())


@pytest.mark.asyncio
async def test_query_llm_returns_empty_on_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __init__(self) -> None:
            self.candidates: list[object] = []

    class FakeModels:
        def generate_content(self, **kwargs: object) -> FakeResponse:
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    monkeypatch.setattr("inferr.llm.genai.Client", FakeClient)

    request = QueryRequest(transcript="hello", context=_base_context())
    result = await query_llm(request, _base_config())

    assert result == ""


@pytest.mark.asyncio
async def test_query_llm_history_uses_model_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    class FakePart:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeContent:
        def __init__(self, text: str) -> None:
            self.parts = [FakePart(text)]

    class FakeCandidate:
        def __init__(self, text: str) -> None:
            self.content = FakeContent(text)

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.candidates = [FakeCandidate(text)]

    class FakeModels:
        def generate_content(self, **kwargs: object) -> FakeResponse:
            recorded.update(kwargs)
            return FakeResponse("ok")

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    monkeypatch.setattr("inferr.llm.genai.Client", FakeClient)

    history = [("user", "u1"), ("assistant", "a1"), ("user", "u2")]
    request = QueryRequest(transcript="hello", context=_base_context(history=history))
    await query_llm(request, _base_config())

    contents = recorded.get("contents")
    assert isinstance(contents, list)
    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert contents[2].role == "user"
