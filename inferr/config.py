from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast
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
    ignored_dirs = _coerce_str_list(
        table.get("ignored_dirs"), _DEFAULT_IGNORED_DIRS
    )
    host = _coerce_str(table.get("host"), _DEFAULT_HOST)
    port = _coerce_int(table.get("port"), _DEFAULT_PORT)

    return Config(
        terminal_buffer_lines=terminal_buffer_lines,
        history_depth=history_depth,
        file_lines=file_lines,
        language=language,
        ignored_dirs=ignored_dirs,
        host=host,
        port=port,
    )
