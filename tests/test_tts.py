from __future__ import annotations

import io
import pytest

from inferr.config import Config
from inferr.models import DeepgramConfig, GeminiConfig, SilkConfig
from inferr.tts import (
    BrowserTTSBackend,
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
        gemini=GeminiConfig(),
        deepgram=DeepgramConfig(),
    )

    backend = get_tts_backend(config)

    assert backend.name() == "silk"


def test_get_tts_backend_falls_back_to_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config()

    monkeypatch.setattr(SilkTTSBackend, "is_available", lambda self: False)

    monkeypatch.setattr(Pyttsx3TTSBackend, "is_available", lambda self: False)

    backend = get_tts_backend(config)

    assert backend.name() == "browser"


async def test_silk_backend_raises_not_implemented() -> None:
    backend = SilkTTSBackend(SilkConfig(api_key="key", api_url="https://api.silk.ai"))

    with pytest.raises(NotImplementedError):
        await backend.speak("test")


def test_preprocess_strips_markdown() -> None:
    output = _preprocess_tts_text("**bhai** ye `error` hai", tone="neutral")

    assert output == "bhai ye error hai"


def test_preprocess_bullet_to_spoken_hinglish() -> None:
    output = _preprocess_tts_text("bhai\n- pehla\n- doosra\n- teesra", tone="neutral")

    assert output.startswith("bhai Teen cheezein: ")


def test_preprocess_bullet_to_spoken_english() -> None:
    output = _preprocess_tts_text("- first\n- second", tone="neutral")

    assert output.startswith("Two things: ")


def test_preprocess_truncates_at_800() -> None:
    output = _preprocess_tts_text("a" * 900, tone="neutral")

    assert len(output) <= 810


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend",
    [
        SilkTTSBackend(SilkConfig()),
        BrowserTTSBackend(),
        Pyttsx3TTSBackend(),
    ],
)
async def test_invalid_tone_raises_value_error(backend: object) -> None:
    with pytest.raises(ValueError):
        # type: ignore[attr-defined]
        await backend.speak("text", tone="invalid")


def test_silk_backend_is_available_with_key() -> None:
    backend = SilkTTSBackend(
        SilkConfig(api_key="key", api_url="https://silk-api.rumik.ai")
    )

    assert backend.is_available() is True


def test_silk_backend_not_available_without_key() -> None:
    backend = SilkTTSBackend(SilkConfig())

    assert backend.is_available() is False


async def test_browser_backend_speak_is_noop() -> None:
    result = await BrowserTTSBackend().speak("anything")
    assert result is None


def test_pyttsx3_backwards_compat() -> None:
    assert callable(speak_pyttsx3)
    assert callable(tts_stub_available)


@pytest.mark.asyncio
async def test_silk_backend_stream_flow() -> None:
    from unittest.mock import AsyncMock, patch, MagicMock
    import json
    from inferr.tts import SilkAPIError

    config = SilkConfig(
        api_key="key", api_url="https://custom-api.silk.ai", stream=True
    )
    backend = SilkTTSBackend(config)

    mock_ws = AsyncMock()
    backend.set_ws_connection(mock_ws)

    # Mock response from httpx.AsyncClient.post
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "ws_url": "wss://custom-api.silk.ai/ws",
        "token": "test-session-token",
    }

    # Mock the AsyncClient post method
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        # Mock websockets.connect context manager
        mock_silk_ws = AsyncMock()

        async def mock_async_iter(*args, **kwargs):
            yield b"chunk1"
            yield b"chunk2"
            yield json.dumps({"type": "done"})

        mock_silk_ws.__aiter__ = mock_async_iter

        # Use an AsyncMock context manager for websockets.connect
        mock_connect_cm = AsyncMock()
        mock_connect_cm.__aenter__.return_value = mock_silk_ws

        with patch("websockets.connect", return_value=mock_connect_cm) as mock_connect:
            await backend.speak("hello world", tone="neutral")

            # Assertions
            mock_post.assert_called_once()
            mock_connect.assert_called_once_with(
                "wss://custom-api.silk.ai/ws?token=test-session-token"
            )
            mock_silk_ws.send.assert_called_once()
            sent_payload = json.loads(mock_silk_ws.send.call_args[0][0])
            assert sent_payload["text"] == "[neutral] hello world"

            # Assert browser websocket sends
            assert mock_ws.send_bytes.call_count == 2
            mock_ws.send_bytes.assert_any_call(b"chunk1")
            mock_ws.send_bytes.assert_any_call(b"chunk2")
            assert mock_ws.send_text.call_count == 2
            mock_ws.send_text.assert_any_call(json.dumps({"type": "silk_end"}))


@pytest.mark.asyncio
async def test_silk_backend_non_stream_flow() -> None:
    from unittest.mock import AsyncMock, patch, MagicMock
    import json

    config = SilkConfig(
        api_key="key", api_url="https://custom-api.silk.ai", stream=False
    )
    backend = SilkTTSBackend(config)

    mock_ws = AsyncMock()
    backend.set_ws_connection(mock_ws)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"full_audio_bytes"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        await backend.speak("hello non-stream", tone="urgent")

        mock_post.assert_called_once()
        mock_ws.send_bytes.assert_called_once_with(b"full_audio_bytes")
        assert mock_ws.send_text.call_count == 2
        mock_ws.send_text.assert_any_call(json.dumps({"type": "silk_end"}))


@pytest.mark.asyncio
async def test_silk_backend_http_error_handling() -> None:
    from unittest.mock import AsyncMock, patch, MagicMock
    from inferr.tts import SilkAPIError

    config = SilkConfig(
        api_key="key", api_url="https://custom-api.silk.ai", stream=False
    )
    backend = SilkTTSBackend(config)

    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        with pytest.raises(SilkAPIError) as exc_info:
            await backend.speak("trigger error")

        assert "Silk API server error: 500" in str(exc_info.value)
