"""Tests for offline LLM fallback (ollama) and config loading."""
from __future__ import annotations

import pytest
import httpx

from inferr.models import OllamaConfig, WakeWordConfig
from inferr.llm import _query_ollama, _try_ollama_fallback
from inferr.config import load_config


class TestOllamaConfig:
    def test_defaults(self):
        cfg = OllamaConfig()
        assert cfg.enabled is True
        assert cfg.model == "llama3.2:3b"
        assert cfg.url == "http://localhost:11434"
        assert cfg.timeout_seconds == 20.0

    def test_custom_values(self):
        cfg = OllamaConfig(
            enabled=False,
            model="mistral:7b",
            url="http://192.168.1.10:11434",
            timeout_seconds=30.0,
        )
        assert not cfg.enabled
        assert cfg.model == "mistral:7b"


class TestWakeWordConfig:
    def test_defaults(self):
        cfg = WakeWordConfig()
        assert cfg.enabled is False
        assert cfg.model_name == "alexa"
        assert cfg.model_path == ""
        assert cfg.threshold == 0.5
        assert cfg.cooldown_seconds == 3.0

    def test_custom_model_path(self):
        cfg = WakeWordConfig(model_path="/home/user/inferr.onnx")
        assert cfg.model_path == "/home/user/inferr.onnx"


class TestQueryOllama:
    @pytest.mark.asyncio
    async def test_raises_on_connect_error(self):
        """When ollama isn't running, raise a RuntimeError with helpful message."""
        cfg = OllamaConfig(url="http://127.0.0.1:19999")  # Port nothing listens on
        with pytest.raises(RuntimeError, match="Ollama not reachable"):
            await _query_ollama(
                system_prompt="You are a helpful assistant.",
                user_content="Hello",
                config_ollama=cfg,
                max_tokens=50,
            )

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self, respx_mock=None):
        """4xx/5xx from ollama raises RuntimeError."""
        import respx
        from httpx import Response

        cfg = OllamaConfig(url="http://mockollama:11434")

        async def _run():
            with respx.mock:
                respx.post("http://mockollama:11434/api/chat").mock(
                    return_value=Response(503, json={"error": "service unavailable"})
                )
                with pytest.raises(RuntimeError, match="Ollama HTTP error: 503"):
                    await _query_ollama(
                        system_prompt="You are helpful.",
                        user_content="test",
                        config_ollama=cfg,
                        max_tokens=50,
                    )

        await _run()

    @pytest.mark.asyncio
    async def test_parses_valid_response(self):
        """Successful ollama response is parsed and returned as string."""
        import respx
        from httpx import Response

        cfg = OllamaConfig(url="http://mockollama:11434")
        mock_response = {
            "model": "llama3.2:3b",
            "message": {
                "role": "assistant",
                "content": "Stack trace dekh bhai, line 42 mein issue hai.",
            },
            "done": True,
        }

        async def _run():
            with respx.mock:
                respx.post("http://mockollama:11434/api/chat").mock(
                    return_value=Response(200, json=mock_response)
                )
                result = await _query_ollama(
                    system_prompt="You are Inferr.",
                    user_content="kya ho raha hai?",
                    config_ollama=cfg,
                    max_tokens=100,
                )
            assert "line 42" in result
            assert isinstance(result, str)

        await _run()

    @pytest.mark.asyncio
    async def test_raises_on_malformed_response(self):
        """Missing 'message' key raises RuntimeError."""
        import respx
        from httpx import Response

        cfg = OllamaConfig(url="http://mockollama:11434")

        async def _run():
            with respx.mock:
                respx.post("http://mockollama:11434/api/chat").mock(
                    return_value=Response(200, json={"wrong_key": "oops"})
                )
                with pytest.raises(RuntimeError, match="unexpected"):
                    await _query_ollama(
                        system_prompt="sys",
                        user_content="user",
                        config_ollama=cfg,
                        max_tokens=50,
                    )

        await _run()


class TestOllamaUrlBuilding:
    @pytest.mark.asyncio
    async def test_trailing_slash_stripped(self):
        """URL with trailing slash is normalised correctly."""
        import respx
        from httpx import Response

        cfg = OllamaConfig(url="http://mockollama:11434/")
        mock_response = {
            "message": {"role": "assistant", "content": "ok"},
        }

        async def _run():
            with respx.mock:
                route = respx.post("http://mockollama:11434/api/chat").mock(
                    return_value=Response(200, json=mock_response)
                )
                await _query_ollama(
                    system_prompt="s", user_content="u",
                    config_ollama=cfg, max_tokens=50
                )
                assert route.called

        await _run()


class TestConfigOllamaWakeWord:
    def test_load_config_has_ollama(self, tmp_path, monkeypatch):
        """load_config() returns a Config with OllamaConfig attached."""
        import os
        # Point config to a fresh temp dir
        monkeypatch.setenv("INFERR_HOME", str(tmp_path))
        monkeypatch.setattr("inferr.config.Path.home", lambda: tmp_path)
        # Suppress Gemini key warning
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        cfg = load_config()
        assert hasattr(cfg, "ollama")
        assert isinstance(cfg.ollama, OllamaConfig)
        assert hasattr(cfg, "wakeword")
        assert isinstance(cfg.wakeword, WakeWordConfig)

    def test_env_override_ollama_url(self, tmp_path, monkeypatch):
        monkeypatch.setattr("inferr.config.Path.home", lambda: tmp_path)
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        monkeypatch.setenv("OLLAMA_URL", "http://myserver:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "phi3:mini")
        cfg = load_config()
        assert cfg.ollama.url == "http://myserver:11434"
        assert cfg.ollama.model == "phi3:mini"

    def test_env_disable_offline(self, tmp_path, monkeypatch):
        monkeypatch.setattr("inferr.config.Path.home", lambda: tmp_path)
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        monkeypatch.setenv("INFERR_OFFLINE", "false")
        cfg = load_config()
        assert cfg.ollama.enabled is False

    def test_env_enable_wakeword(self, tmp_path, monkeypatch):
        monkeypatch.setattr("inferr.config.Path.home", lambda: tmp_path)
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        monkeypatch.setenv("INFERR_WAKEWORD_ENABLED", "true")
        monkeypatch.setenv("INFERR_WAKEWORD_MODEL", "hey_mycroft")
        cfg = load_config()
        assert cfg.wakeword.enabled is True
        assert cfg.wakeword.model_name == "hey_mycroft"
