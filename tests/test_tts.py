from __future__ import annotations

import io
import pytest

from inferr.config import Config
from inferr.models import DeepgramConfig, ElevenLabsConfig, GeminiConfig, SilkConfig
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
        deepgram=DeepgramConfig(),
    )


def test_get_tts_backend_returns_silk_when_configured() -> None:
    config = Config(
        terminal_buffer_lines=50,
        history_depth=20,
        file_lines=150,
        language="english",
        ignored_dirs=[".git"],
        host="127.0.0.1",
        port=7331,
        silk=SilkConfig(api_key="key", api_url="https://silk-api.rumik.ai"),
        elevenlabs=ElevenLabsConfig(),
        gemini=GeminiConfig(),
        deepgram=DeepgramConfig(),
    )

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
    config = Config(
        terminal_buffer_lines=50,
        history_depth=20,
        file_lines=150,
        language="english",
        ignored_dirs=[".git"],
        host="127.0.0.1",
        port=7331,
        silk=SilkConfig(),
        elevenlabs=ElevenLabsConfig(api_key="key"),
        gemini=GeminiConfig(),
        deepgram=DeepgramConfig(),
    )

    monkeypatch.setattr(SilkTTSBackend, "is_available", lambda self: False)
    monkeypatch.setattr(Pyttsx3TTSBackend, "is_available", lambda self: False)

    backend = get_tts_backend(config)

    assert backend.name() == "elevenlabs"


def test_silk_backend_raises_not_implemented() -> None:
    backend = SilkTTSBackend(SilkConfig(api_key="key", api_url="https://api.silk.ai"))

    with pytest.raises(NotImplementedError):
        backend.speak("test")


def test_preprocess_strips_markdown() -> None:
    output = _preprocess_tts_text("**bhai** ye `error` hai", tone="neutral")

    assert output == "bhai ye error hai"


def test_preprocess_bullet_to_spoken_hinglish() -> None:
    output = _preprocess_tts_text("bhai\n- pehla\n- doosra\n- teesra", tone="neutral")

    assert output.startswith("bhai Teen cheezein: ")


def test_preprocess_bullet_to_spoken_english() -> None:
    output = _preprocess_tts_text("- first\n- second", tone="neutral")

    assert output.startswith("Two things: ")


def test_preprocess_truncates_at_400() -> None:
    output = _preprocess_tts_text("a" * 500, tone="neutral")

    assert len(output) <= 410


def test_preprocess_tts_text_is_module_level() -> None:
    assert callable(_preprocess_tts_text)


def test_preprocess_tone_marker_prepended() -> None:
    output = _preprocess_tts_text("hello world", tone="urgent")

    assert str(output).startswith("[angry]")


def test_preprocess_warm_tone_has_chuckle() -> None:
    output = _preprocess_tts_text("hello world", tone="warm")

    assert "[happy]" in str(output)
    assert "<chuckle>" in str(output)


def test_preprocess_neutral_tone_marker() -> None:
    output = _preprocess_tts_text("hello world", tone="neutral")

    assert str(output).startswith("[neutral]")


def test_preprocess_latin_text_preserved() -> None:
    text = "bhai KeyError aa raha hai line 23 pe"
    output = _preprocess_tts_text(text, tone="urgent")

    assert "KeyError" in str(output)
    assert "line 23" in str(output)


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


def test_silk_backend_is_available_with_key() -> None:
    backend = SilkTTSBackend(
        SilkConfig(api_key="key", api_url="https://silk-api.rumik.ai")
    )

    assert backend.is_available() is True


def test_silk_backend_not_available_without_key() -> None:
    backend = SilkTTSBackend(SilkConfig())

    assert backend.is_available() is False


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
