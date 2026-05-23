from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class Config:
    terminal_buffer_lines: int
    history_depth: int
    file_lines: int
    language: str
    ignored_dirs: list[str]
    host: str
    port: int


_DEFAULTS: dict[str, Any] = {
    "terminal_buffer_lines": 50,
    "history_depth": 20,
    "file_lines": 150,
    "language": "hinglish",
    "ignored_dirs": ["node_modules", ".git", "__pycache__", ".venv"],
    "host": "127.0.0.1",
    "port": 7331,
}


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
    )


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
        data = tomllib.loads(raw_text)
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"Invalid config TOML: {exc}") from exc

    table = data.get("inferr", {})
    if not isinstance(table, dict):
        table = {}

    terminal_buffer_lines = int(
        table.get("terminal_buffer_lines", _DEFAULTS["terminal_buffer_lines"])
    )
    history_depth = int(table.get("history_depth", _DEFAULTS["history_depth"]))
    file_lines = int(table.get("file_lines", _DEFAULTS["file_lines"]))
    language = str(table.get("language", _DEFAULTS["language"]))
    ignored_dirs_raw = table.get("ignored_dirs", _DEFAULTS["ignored_dirs"])
    if isinstance(ignored_dirs_raw, list):
        ignored_dirs = [str(item) for item in ignored_dirs_raw]
    else:
        ignored_dirs = list(_DEFAULTS["ignored_dirs"])
    host = str(table.get("host", _DEFAULTS["host"]))
    port = int(table.get("port", _DEFAULTS["port"]))

    return Config(
        terminal_buffer_lines=terminal_buffer_lines,
        history_depth=history_depth,
        file_lines=file_lines,
        language=language,
        ignored_dirs=ignored_dirs,
        host=host,
        port=port,
    )
