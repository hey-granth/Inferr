from __future__ import annotations

import pytest

from inferr.config import Config
from inferr.models import SilkConfig
from inferr.tts import (
    BrowserTTSBackend,
    Pyttsx3TTSBackend,
    SilkTTSBackend,
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
    monkeypatch.setattr(Pyttsx3TTSBackend, "is_available", lambda self: False)

    backend = get_tts_backend(config)

    assert backend.name() == "browser"


def test_silk_backend_raises_not_implemented() -> None:
    backend = SilkTTSBackend(
        SilkConfig(api_key="key", api_url="https://api.silk.ai")
    )

    with pytest.raises(NotImplementedError):
        backend.speak("test")


def test_preprocess_strips_markdown() -> None:
    backend = SilkTTSBackend(SilkConfig())

    output = backend._preprocess_text("**bhai** ye `error` hai")

    assert output == "bhai ye error hai"


def test_preprocess_bullet_to_spoken_hinglish() -> None:
    backend = SilkTTSBackend(SilkConfig())

    output = backend._preprocess_text("bhai\n- pehla\n- doosra\n- teesra")

    assert output.startswith("bhai Teen cheezein: ")


def test_preprocess_bullet_to_spoken_english() -> None:
    backend = SilkTTSBackend(SilkConfig())

    output = backend._preprocess_text("- first\n- second")

    assert output.startswith("Two things: ")


def test_preprocess_truncates_at_600() -> None:
    backend = SilkTTSBackend(SilkConfig())

    output = backend._preprocess_text("a" * 700)

    assert len(output) <= 603


@pytest.mark.parametrize(
    "backend",
    [SilkTTSBackend(SilkConfig()), BrowserTTSBackend(), Pyttsx3TTSBackend()],
)
def test_invalid_tone_raises_value_error(backend: object) -> None:
    with pytest.raises(ValueError):
        # type: ignore[attr-defined]
        backend.speak("text", tone="invalid")


def test_browser_backend_speak_is_noop() -> None:
    result = BrowserTTSBackend().speak("anything")
    assert result is None


def test_pyttsx3_backwards_compat() -> None:
    assert callable(speak_pyttsx3)
    assert callable(tts_stub_available)
