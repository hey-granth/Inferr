from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Mapping, cast
import tomllib

from inferr.models import DeepgramConfig, GeminiConfig, GroqConfig, OllamaConfig, SilkConfig, WakeWordConfig


@dataclass(frozen=True)
class Config:
    terminal_buffer_lines: int
    history_depth: int
    file_lines: int
    language: str
    ignored_dirs: list[str]
    host: str
    port: int
    silk: SilkConfig = field(default_factory=SilkConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    deepgram: DeepgramConfig = field(default_factory=DeepgramConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    wakeword: WakeWordConfig = field(default_factory=WakeWordConfig)


_DEFAULT_TERMINAL_BUFFER_LINES = 50
_DEFAULT_HISTORY_DEPTH = 20
_DEFAULT_FILE_LINES = 150
_DEFAULT_LANGUAGE = "hinglish"
_DEFAULT_IGNORED_DIRS = ["node_modules", ".git", "__pycache__", ".venv"]
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 7331


def _default_toml() -> str:
    return (
        "[inferr]\n"
        "terminal_buffer_lines = 50\n"
        "history_depth = 20\n"
        "file_lines = 150\n"
        'language = "hinglish"\n'
        'ignored_dirs = ["node_modules", ".git", "__pycache__", ".venv"]\n'
        'host = "127.0.0.1"\n'
        "port = 7331\n"
        "\n"
        "[silk]\n"
        'api_url = "https://silk-api.rumik.ai"\n'
        'api_key = ""\n'
        'voice_id = "muga"\n'
        "stream = true\n"
        "\n"
        "[gemini]\n"
        'api_key = ""\n'
        'model = "gemini-flash-latest"\n'
        "\n"
        "[groq]\n"
        'api_key = ""\n'
        'model = "llama-3.3-70b-versatile"\n'
        "\n"
        "[deepgram]\n"
        'api_key = ""\n'
        'model = "nova-2"\n'
        'language = "en-IN"\n'
        "\n"
        "# Offline LLM fallback — used when Gemini is unreachable\n"
        "[ollama]\n"
        "enabled = true\n"
        'model = "llama3.2:3b"\n'
        'url = "http://localhost:11434"\n'
        "timeout_seconds = 20.0\n"
        "\n"
        "# Wake word detector (requires openwakeword + sounddevice)\n"
        "[wakeword]\n"
        "enabled = false\n"
        '# model_name = "alexa"  # pre-trained model name\n'
        '# model_path = ""      # full path to custom .onnx; overrides model_name\n'
        "# threshold = 0.5\n"
        "# cooldown_seconds = 3.0\n"
    )


def _get_table(data: Mapping[str, object]) -> dict[str, object]:
    table_obj = data.get("inferr")
    if not isinstance(table_obj, dict):
        return {}

    raw_table = cast(dict[object, object], table_obj)
    table: dict[str, object] = {}
    for key, value in raw_table.items():
        key_obj: object = key
        value_obj: object = value
        key_str = str(key_obj)
        table[key_str] = value_obj
    return table


def _coerce_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _coerce_str(value: object, default: str) -> str:
    if isinstance(value, str) and value:
        return value
    return default


def _coerce_str_list(value: object, default: list[str]) -> list[str]:
    if isinstance(value, list):
        items = cast(list[object], value)
        return [str(item) for item in items]
    return list(default)


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _ensure_config_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_default_toml(), encoding="utf-8")


def load_config() -> Config:
    config_path = Path.home() / ".inferr" / "config.toml"
    _ensure_config_file(config_path)

    raw_text = config_path.read_text(encoding="utf-8")
    try:
        data: dict[str, object] = tomllib.loads(raw_text)
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"Invalid config TOML: {exc}") from exc

    table = _get_table(data)

    terminal_buffer_lines = _coerce_int(
        table.get("terminal_buffer_lines"), _DEFAULT_TERMINAL_BUFFER_LINES
    )
    history_depth = _coerce_int(table.get("history_depth"), _DEFAULT_HISTORY_DEPTH)
    file_lines = _coerce_int(table.get("file_lines"), _DEFAULT_FILE_LINES)
    language = _coerce_str(table.get("language"), _DEFAULT_LANGUAGE)
    ignored_dirs = _coerce_str_list(table.get("ignored_dirs"), _DEFAULT_IGNORED_DIRS)
    host = _coerce_str(table.get("host"), _DEFAULT_HOST)
    port = _coerce_int(table.get("port"), _DEFAULT_PORT)
    silk_table_obj = data.get("silk")
    silk_table = (
        cast(dict[object, object], silk_table_obj)
        if isinstance(silk_table_obj, dict)
        else {}
    )
    silk_api_url = _coerce_str(silk_table.get("api_url"), "https://silk-api.rumik.ai")
    silk_api_key = _coerce_str(silk_table.get("api_key"), "")
    silk_voice_id = _coerce_str(silk_table.get("voice_id"), "muga")
    silk_stream = _coerce_bool(silk_table.get("stream"), True)

    env_silk_api_key = os.environ.get("SILK_API_KEY", "")
    env_silk_api_url = os.environ.get("SILK_API_URL", "")
    if env_silk_api_key:
        silk_api_key = env_silk_api_key
        silk_api_url = "https://silk-api.rumik.ai"
    elif env_silk_api_url:
        silk_api_url = env_silk_api_url

    silk_config = SilkConfig(
        api_url=silk_api_url,
        api_key=silk_api_key,
        voice_id=silk_voice_id,
        stream=silk_stream,
    )
    raw_gemini_obj = data.get("gemini")
    raw_gemini = (
        cast(dict[object, object], raw_gemini_obj)
        if isinstance(raw_gemini_obj, dict)
        else {}
    )
    raw_gemini_keys = os.environ.get(
        "GEMINI_API_KEY",
        _coerce_str(raw_gemini.get("api_key"), ""),
    )
    gemini_keys = [key.strip() for key in raw_gemini_keys.split(",") if key.strip()]
    gemini_api_key = gemini_keys[0] if gemini_keys else ""
    gemini_model = os.environ.get(
        "GEMINI_MODEL",
        _coerce_str(raw_gemini.get("model"), "gemini-flash-latest"),
    )
    gemini_config = GeminiConfig(
        api_key=gemini_api_key,
        api_keys=gemini_keys,
        model=gemini_model,
    )

    raw_groq_obj = data.get("groq")
    raw_groq = (
        cast(dict[object, object], raw_groq_obj)
        if isinstance(raw_groq_obj, dict)
        else {}
    )
    raw_groq_keys = os.environ.get(
        "GROQ_API_KEY",
        _coerce_str(raw_groq.get("api_key"), ""),
    )
    groq_keys = [key.strip() for key in raw_groq_keys.split(",") if key.strip()]
    groq_api_key = groq_keys[0] if groq_keys else ""
    groq_model = os.environ.get(
        "GROQ_MODEL",
        _coerce_str(raw_groq.get("model"), "llama-3.3-70b-versatile"),
    )
    groq_config = GroqConfig(
        api_key=groq_api_key,
        api_keys=groq_keys,
        model=groq_model,
    )

    if not gemini_api_key and not groq_api_key:
        import sys

        print(
            "[inferr config] WARNING: Neither GEMINI_API_KEY nor GROQ_API_KEY is set. "
            "Add one to your .env file.",
            file=sys.stderr,
        )

    raw_deepgram_obj = data.get("deepgram")
    raw_deepgram = (
        cast(dict[object, object], raw_deepgram_obj)
        if isinstance(raw_deepgram_obj, dict)
        else {}
    )
    deepgram_api_key = os.environ.get(
        "DEEPGRAM_API_KEY",
        _coerce_str(raw_deepgram.get("api_key"), ""),
    )
    deepgram_model = _coerce_str(raw_deepgram.get("model"), "nova-2")
    deepgram_language = _coerce_str(raw_deepgram.get("language"), "en-IN")
    deepgram_config = DeepgramConfig(
        api_key=deepgram_api_key,
        model=deepgram_model,
        language=deepgram_language,
    )

    # ---- Ollama (offline fallback) -----------------------------------------------
    raw_ollama_obj = data.get("ollama")
    raw_ollama = (
        cast(dict[object, object], raw_ollama_obj)
        if isinstance(raw_ollama_obj, dict)
        else {}
    )
    ollama_enabled = _coerce_bool(
        os.environ.get("INFERR_OFFLINE", raw_ollama.get("enabled", True)), True
    )
    ollama_model = os.environ.get(
        "OLLAMA_MODEL", _coerce_str(raw_ollama.get("model"), "llama3.2:3b")
    )
    ollama_url = os.environ.get(
        "OLLAMA_URL", _coerce_str(raw_ollama.get("url"), "http://localhost:11434")
    )
    ollama_timeout = float(
        _coerce_str(raw_ollama.get("timeout_seconds", "20.0"), "20.0") or "20.0"
    )
    ollama_config = OllamaConfig(
        enabled=ollama_enabled,
        model=ollama_model,
        url=ollama_url,
        timeout_seconds=ollama_timeout,
    )

    # ---- Wake word ---------------------------------------------------------------
    raw_ww_obj = data.get("wakeword")
    raw_ww = (
        cast(dict[object, object], raw_ww_obj)
        if isinstance(raw_ww_obj, dict)
        else {}
    )
    ww_enabled = _coerce_bool(
        os.environ.get("INFERR_WAKEWORD_ENABLED", raw_ww.get("enabled", False)),
        False,
    )
    ww_model_name = os.environ.get(
        "INFERR_WAKEWORD_MODEL",
        _coerce_str(raw_ww.get("model_name"), "alexa"),
    )
    ww_model_path = _coerce_str(raw_ww.get("model_path"), "")
    ww_threshold = float(
        _coerce_str(raw_ww.get("threshold", "0.5"), "0.5") or "0.5"
    )
    ww_cooldown = float(
        _coerce_str(raw_ww.get("cooldown_seconds", "3.0"), "3.0") or "3.0"
    )
    wakeword_config = WakeWordConfig(
        enabled=ww_enabled,
        model_name=ww_model_name,
        model_path=ww_model_path,
        threshold=ww_threshold,
        cooldown_seconds=ww_cooldown,
    )

    return Config(
        terminal_buffer_lines=terminal_buffer_lines,
        history_depth=history_depth,
        file_lines=file_lines,
        language=language,
        ignored_dirs=ignored_dirs,
        host=host,
        port=port,
        silk=silk_config,
        gemini=gemini_config,
        groq=groq_config,
        deepgram=deepgram_config,
        ollama=ollama_config,
        wakeword=wakeword_config,
    )
