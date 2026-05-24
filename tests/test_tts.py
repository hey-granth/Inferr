from __future__ import annotations

import io
import pytest

from inferr.config import Config
from inferr.models import ElevenLabsConfig, GeminiConfig, SilkConfig
from inferr.tts import (
    BrowserTTSBackend,
    ElevenLabsTTSBackend,
    Pyttsx3TTSBackend,
    SilkTTSBackend,
    _preprocess_tts_text,
    get_tts_backend,
    speak_pyttsx3,
    tts_stub_available,
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
        gemini=GeminiConfig(),
    )


def test_get_tts_backend_returns_silk_when_configured() -> None:
    config = _base_config()
    config.silk.api_key = "key"
    config.silk.api_url = "https://api.silk.ai"

    backend = get_tts_backend(config)

    assert backend.name() == "silk"


def test_get_tts_backend_falls_back_to_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config()

    monkeypatch.setattr(SilkTTSBackend, "is_available", lambda self: False)
    monkeypatch.setattr(ElevenLabsTTSBackend, "is_available", lambda self: False)
    monkeypatch.setattr(Pyttsx3TTSBackend, "is_available", lambda self: False)

    backend = get_tts_backend(config)

    assert backend.name() == "browser"


def test_get_tts_backend_returns_elevenlabs(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config()
    config.elevenlabs.api_key = "key"

    monkeypatch.setattr(SilkTTSBackend, "is_available", lambda self: False)
    monkeypatch.setattr(Pyttsx3TTSBackend, "is_available", lambda self: False)

    backend = get_tts_backend(config)

    assert backend.name() == "elevenlabs"


def test_silk_backend_raises_not_implemented() -> None:
    backend = SilkTTSBackend(SilkConfig(api_key="key", api_url="https://api.silk.ai"))

    with pytest.raises(NotImplementedError):
        backend.speak("test")


def test_preprocess_strips_markdown() -> None:
    output = _preprocess_tts_text("**bhai** ye `error` hai")

    assert output == "bhai ye error hai"


def test_preprocess_bullet_to_spoken_hinglish() -> None:
    output = _preprocess_tts_text("bhai\n- pehla\n- doosra\n- teesra")

    assert output.startswith("bhai Teen cheezein: ")


def test_preprocess_bullet_to_spoken_english() -> None:
    output = _preprocess_tts_text("- first\n- second")

    assert output.startswith("Two things: ")


def test_preprocess_truncates_at_600() -> None:
    output = _preprocess_tts_text("a" * 700)

    assert len(output) <= 603


def test_preprocess_tts_text_is_module_level() -> None:
    assert callable(_preprocess_tts_text)


@pytest.mark.parametrize(
    "backend",
    [
        SilkTTSBackend(SilkConfig()),
        ElevenLabsTTSBackend(ElevenLabsConfig(api_key="test-key")),
        BrowserTTSBackend(),
        Pyttsx3TTSBackend(),
    ],
)
def test_invalid_tone_raises_value_error(backend: object) -> None:
    with pytest.raises(ValueError):
        # type: ignore[attr-defined]
        backend.speak("text", tone="invalid")


def test_elevenlabs_backend_name() -> None:
    backend = ElevenLabsTTSBackend(ElevenLabsConfig(api_key="test-key"))
    assert backend.name() == "elevenlabs"


def test_elevenlabs_not_available_without_key() -> None:
    backend = ElevenLabsTTSBackend(ElevenLabsConfig(api_key=""))
    assert backend.is_available() is False


def test_elevenlabs_available_with_key() -> None:
    backend = ElevenLabsTTSBackend(ElevenLabsConfig(api_key="test-key"))
    assert backend.is_available() is True


def test_elevenlabs_invalid_tone_raises() -> None:
    backend = ElevenLabsTTSBackend(ElevenLabsConfig(api_key="test-key"))
    with pytest.raises(ValueError):
        backend.speak("text", tone="robot")


def test_elevenlabs_no_ws_connection_logs_and_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ElevenLabsTTSBackend(ElevenLabsConfig(api_key="test-key"))
    err = io.StringIO()
    monkeypatch.setattr("sys.stderr", err)

    backend._synthesize_and_send("text", "neutral")

    assert "no WebSocket connection" in err.getvalue()


def test_browser_backend_speak_is_noop() -> None:
    result = BrowserTTSBackend().speak("anything")
    assert result is None


def test_pyttsx3_backwards_compat() -> None:
    assert callable(speak_pyttsx3)
    assert callable(tts_stub_available)
