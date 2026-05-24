from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from collections import deque
import threading

import pytest

from inferr.config import Config
from inferr.context import ContextAssembler
from inferr.context.files import FileWatcher
from inferr.context.terminal import TerminalCapture
from inferr.context.errors import extract_errors
from inferr.context.history import read_shell_history
from inferr.models import ActiveFile, ConversationTurn, FlaggedError
from inferr.models import DeepgramConfig, GeminiConfig, SilkConfig


def _base_config() -> Config:
    return Config(
        terminal_buffer_lines=50,
        history_depth=20,
        file_lines=150,
        language="hinglish",
        ignored_dirs=[".git"],
        host="127.0.0.1",
        port=7331,
        silk=SilkConfig(),
        gemini=GeminiConfig(),
        deepgram=DeepgramConfig(),
    )


def test_read_shell_history_dedup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/bash")
    history_path = tmp_path / ".bash_history"
    history_path.write_text("ls\nls\npwd\npwd\necho hi\n", encoding="utf-8")

    result = read_shell_history(10)

    assert result == ["ls", "pwd", "echo hi"]


def test_read_shell_history_zsh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    history_path = tmp_path / ".zsh_history"
    history_path.write_text(
        ": 1710000000:0;git status\n: 1710000001:0;git status\n: 1710000002:0;ls\n",
        encoding="utf-8",
    )

    result = read_shell_history(10)

    assert result == ["git status", "ls"]


def test_read_shell_history_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/bash")

    result = read_shell_history(10)

    assert result == []


def test_read_shell_history_depth_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/bash")
    history_path = tmp_path / ".bash_history"
    history_path.write_text("\n".join(f"cmd{i}" for i in range(50)), encoding="utf-8")

    result = read_shell_history(5)

    assert len(result) == 5
    assert result[-1] == "cmd49"


def test_read_shell_history_zero_depth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/bash")
    history_path = tmp_path / ".bash_history"
    history_path.write_text("ls\npwd\n", encoding="utf-8")

    result = read_shell_history(0)

    assert result == []


def test_extract_errors_python_traceback() -> None:
    lines = [
        "random line",
        "Traceback (most recent call last)",
        '  File "demo.py", line 1, in <module>',
        "KeyError: missing",
    ]

    errors = extract_errors(lines)

    assert len(errors) == 2
    assert errors[0].type == "marker"
    assert errors[1].type == "python_traceback"
    assert "KeyError" in errors[1].summary


def test_extract_errors_none() -> None:
    lines = ["all good", "no errors here"]
    assert extract_errors(lines) == []


def test_terminal_capture_filters_short_lines() -> None:
    cap = TerminalCapture.__new__(TerminalCapture)
    cap._buffer = deque(maxlen=50)
    cap._lock = threading.Lock()
    cap._append_line("ab")
    cap._append_line("")
    cap._append_line("x")
    assert cap.get_buffer() == []


def test_terminal_capture_filters_prompts() -> None:
    cap = TerminalCapture.__new__(TerminalCapture)
    cap._buffer = deque(maxlen=50)
    cap._lock = threading.Lock()
    cap._append_line("user@host:~$")
    cap._append_line("root@host:#")
    cap._append_line("user%")
    assert cap.get_buffer() == []


def test_terminal_capture_keeps_valid_lines() -> None:
    cap = TerminalCapture.__new__(TerminalCapture)
    cap._buffer = deque(maxlen=50)
    cap._lock = threading.Lock()
    cap._append_line("uvicorn running on port 8000")
    cap._append_line("KeyError: user_id")
    assert cap.get_buffer() == [
        "uvicorn running on port 8000",
        "KeyError: user_id",
    ]


def test_terminal_capture_inject_line() -> None:
    cap = TerminalCapture.__new__(TerminalCapture)
    cap._buffer = deque(maxlen=50)
    cap._lock = threading.Lock()
    cap.inject_line("git push origin main")
    assert cap.get_buffer() == ["git push origin main"]


def test_file_watcher_ignores_ignored_dirs(tmp_path: Path) -> None:
    config = Config(
        terminal_buffer_lines=50,
        history_depth=20,
        file_lines=150,
        language="hinglish",
        ignored_dirs=["node_modules", ".git"],
        host="127.0.0.1",
        port=7331,
    )
    watcher = FileWatcher(config)
    watcher._root = tmp_path

    ignored = tmp_path / "node_modules" / "some_lib.js"
    ignored.parent.mkdir()
    ignored.write_text("console.log('hi')", encoding="utf-8")

    watcher.process_path(ignored)

    assert watcher.get_active_file() is None


def test_sanitize_terminal_line_strips_carriage_return() -> None:
    from inferr.context.terminal import sanitize_terminal_line

    assert sanitize_terminal_line("foo\r") == "foo"
    assert sanitize_terminal_line("\x1b[31merror\x1b[0m") == "error"


def test_file_watcher_hint_from_shell_command(tmp_path: Path) -> None:
    config = Config(
        terminal_buffer_lines=50,
        history_depth=20,
        file_lines=150,
        language="hinglish",
        ignored_dirs=[".git"],
        host="127.0.0.1",
        port=7331,
    )
    watcher = FileWatcher(config)
    watcher._root = tmp_path

    py_file = tmp_path / "main.py"
    py_file.write_text("x = 1\n", encoding="utf-8")

    watcher.hint_from_shell_command("vim main.py")

    active = watcher.get_active_file()
    assert active is not None
    assert active.path.endswith("main.py")


def test_file_watcher_seed_recent_files(tmp_path: Path) -> None:
    config = Config(
        terminal_buffer_lines=50,
        history_depth=20,
        file_lines=150,
        language="hinglish",
        ignored_dirs=[".git"],
        host="127.0.0.1",
        port=7331,
    )
    watcher = FileWatcher(config)
    watcher._root = tmp_path

    older = tmp_path / "old.py"
    newer = tmp_path / "new.py"
    older.write_text("old\n", encoding="utf-8")
    newer.write_text("new\n", encoding="utf-8")
    import os
    import time

    os.utime(older, (time.time() - 10, time.time() - 10))
    os.utime(newer, (time.time(), time.time()))

    watcher.seed_recent_files()
    active = watcher.get_active_file()
    assert active is not None
    assert "new" in active.content


def test_file_watcher_detects_python_file(tmp_path: Path) -> None:
    config = Config(
        terminal_buffer_lines=50,
        history_depth=20,
        file_lines=150,
        language="hinglish",
        ignored_dirs=[".git"],
        host="127.0.0.1",
        port=7331,
    )
    watcher = FileWatcher(config)
    watcher._root = tmp_path

    py_file = tmp_path / "app.py"
    py_file.write_text("print('hello')\n", encoding="utf-8")

    watcher.process_path(py_file)

    active = watcher.get_active_file()
    assert active is not None
    assert active.language == "python"
    assert "print" in active.content


def test_file_watcher_skips_binary_files(tmp_path: Path) -> None:
    config = Config(
        terminal_buffer_lines=50,
        history_depth=20,
        file_lines=5,
        language="hinglish",
        ignored_dirs=[".git"],
        host="127.0.0.1",
        port=7331,
    )
    watcher = FileWatcher(config)
    watcher._root = tmp_path

    bin_file = tmp_path / "image.py"
    bin_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")

    watcher.process_path(bin_file)


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

        def get_active_file(self) -> ActiveFile | None:
            return None

        def hint_from_shell_command(self, command: str) -> None:
            return

    def _fake_history(depth: int) -> list[str]:
        return ["ls"]

    def _fake_errors(lines: list[str]) -> list[FlaggedError]:
        return []

    monkeypatch.setattr(context_module, "TerminalCapture", DummyTerminal)
    monkeypatch.setattr(context_module, "FileWatcher", DummyWatcher)
    monkeypatch.setattr(context_module, "read_shell_history", _fake_history)
    monkeypatch.setattr(context_module, "extract_errors", _fake_errors)

    config = _base_config()

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
