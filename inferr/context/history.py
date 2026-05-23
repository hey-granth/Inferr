from __future__ import annotations

import os
import re
from pathlib import Path


_ZSH_PREFIX = re.compile(r"^: \d+:\d+;")


def _strip_zsh_prefix(line: str) -> str:
    return _ZSH_PREFIX.sub("", line).strip()


def read_shell_history(depth: int) -> list[str]:
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        history_path = Path.home() / ".zsh_history"
        is_zsh = True
    else:
        history_path = Path.home() / ".bash_history"
        is_zsh = False

    if not history_path.exists():
        return []

    try:
        raw_lines = history_path.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()
    except OSError:
        return []

    cleaned: list[str] = []
    prev: str | None = None
    for raw in raw_lines:
        line = _strip_zsh_prefix(raw) if is_zsh else raw.strip()
        if not line:
            continue
        if line == prev:
            continue
        cleaned.append(line)
        prev = line

    if depth <= 0:
        return []
    return cleaned[-depth:]
