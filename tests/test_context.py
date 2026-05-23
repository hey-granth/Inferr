from __future__ import annotations

from datetime import datetime, timezone

import pytest

from inferr.config import Config
from inferr.context import ContextAssembler
from inferr.context.errors import extract_errors
from inferr.context.history import read_shell_history
from inferr.models import ConversationTurn


def test_read_shell_history_dedup(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/bash")
    history_path = tmp_path / ".bash_history"
    history_path.write_text("ls\nls\npwd\npwd\necho hi\n", encoding="utf-8")

    result = read_shell_history(10)

    assert result == ["ls", "pwd", "echo hi"]


def test_read_shell_history_zsh(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    history_path = tmp_path / ".zsh_history"
    history_path.write_text(
        ": 1710000000:0;git status\n: 1710000001:0;git status\n: 1710000002:0;ls\n",
        encoding="utf-8",
    )

    result = read_shell_history(10)

    assert result == ["git status", "ls"]


def test_read_shell_history_missing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/bash")

    result = read_shell_history(10)

    assert result == []


def test_extract_errors_python_traceback() -> None:
    lines = [
        "random line",
        "Traceback (most recent call last)",
        '  File "demo.py", line 1, in <module>',
        "KeyError: missing",
    ]

    errors = extract_errors(lines)

    assert len(errors) == 1
    assert errors[0].type == "python_traceback"
    assert "KeyError" in errors[0].summary


def test_extract_errors_none() -> None:
    lines = ["all good", "no errors here"]
    assert extract_errors(lines) == []


def test_context_assembler(monkeypatch: pytest.MonkeyPatch) -> None:
    from inferr import context as context_module

    class DummyTerminal:
        def __init__(self, max_lines: int) -> None:
            self.max_lines = max_lines

        def get_buffer(self) -> list[str]:
            return ["line 1", "line 2"]

        def stop(self) -> None:
            return

    class DummyWatcher:
        def __init__(self, config: Config) -> None:
            self.config = config

        def start(self) -> None:
            return

        def stop(self) -> None:
            return

        def get_active_file(self):
            return None

    monkeypatch.setattr(context_module, "TerminalCapture", DummyTerminal)
    monkeypatch.setattr(context_module, "FileWatcher", DummyWatcher)
    monkeypatch.setattr(context_module, "read_shell_history", lambda depth: ["ls"])
    monkeypatch.setattr(context_module, "extract_errors", lambda lines: [])

    config = Config(
        terminal_buffer_lines=50,
        history_depth=20,
        file_lines=150,
        language="hinglish",
        ignored_dirs=[".git"],
        host="127.0.0.1",
        port=7331,
    )

    assembler = ContextAssembler(config)
    assembler.start()
    context = assembler.assemble(
        session_id="session",
        conversation_history=[
            ConversationTurn(role="user", content="hi"),
            ConversationTurn(role="assistant", content="hello"),
        ],
    )

    assert isinstance(context.timestamp, datetime)
    assert context.session_id == "session"
    assert context.terminal_buffer == ["line 1", "line 2"]
    assert context.shell_history == ["ls"]
    assert context.active_file is None
    assert context.flagged_errors == []
    assert context.conversation_history
    assert context.timestamp.tzinfo == timezone.utc
