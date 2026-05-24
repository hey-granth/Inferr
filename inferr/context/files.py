from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import threading
from typing import Iterator, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from inferr.config import Config
from inferr.models import ActiveFile


_SHELL_FILE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:vim|nvim|nano|emacs|code|cursor|code-insiders)\s+([^\s|;&>]+)"
    ),
    re.compile(r"\bpython3?\s+(?:-m\s+\S+\s+)?([^\s|;&>]+\.py)\b"),
    re.compile(r"\b(?:pytest|ruff|mypy|black|uv)\s+run\s+([^\s|;&>]+)"),
    re.compile(r"\b(?:cat|head|tail|less|more)\s+([^\s|;&>]+)"),
)

_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".go": "go",
    ".js": "javascript",
    ".ts": "typescript",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".rb": "ruby",
    ".sh": "shell",
}


def _normalize_event_path(value: str | bytes) -> Path | None:
    if isinstance(value, bytes):
        try:
            return Path(value.decode())
        except UnicodeDecodeError:
            return None
    return Path(value)


class _FileEventHandler(FileSystemEventHandler):
    def __init__(self, watcher: "FileWatcher") -> None:
        self._watcher = watcher

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = _normalize_event_path(event.src_path)
        if path is None:
            return
        self._watcher.process_path(path)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = _normalize_event_path(event.src_path)
        if path is None:
            return
        self._watcher.process_path(path)


class FileWatcher:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._observer = Observer()
        self._handler = _FileEventHandler(self)
        self._active_file: Optional[ActiveFile] = None
        self._lock = threading.Lock()
        self._root = Path.cwd()
        self._started = False

    def _resolve_path(self, raw: str) -> Path | None:
        token = raw.strip().strip("\"'")
        if not token or token.startswith("-") or token in {"|", "&&", ";"}:
            return None
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        try:
            candidate = candidate.resolve()
        except OSError:
            return None
        if not candidate.is_file():
            return None
        try:
            candidate.relative_to(self._root.resolve())
        except ValueError:
            return None
        return candidate

    def hint_from_shell_command(self, command: str) -> None:
        """Set active file when a shell command references a workspace source file."""
        for pattern in _SHELL_FILE_PATTERNS:
            match = pattern.search(command)
            if not match:
                continue
            resolved = self._resolve_path(match.group(1))
            if resolved is not None:
                self.process_path(resolved)
                return

    def _iter_source_files(self, max_depth: int = 4) -> Iterator[Path]:
        def walk(directory: Path, depth: int) -> Iterator[Path]:
            if depth > max_depth:
                return
            try:
                entries = list(directory.iterdir())
            except OSError:
                return
            for entry in entries:
                if entry.is_dir():
                    try:
                        rel = entry.relative_to(self._root)
                    except ValueError:
                        continue
                    if any(part in self._config.ignored_dirs for part in rel.parts):
                        continue
                    yield from walk(entry, depth + 1)
                elif entry.suffix.lower() in _LANGUAGE_MAP:
                    try:
                        rel = entry.relative_to(self._root)
                    except ValueError:
                        continue
                    if len(rel.parts) <= 8:
                        yield entry

        yield from walk(self._root, 0)

    def seed_recent_files(self) -> None:
        """Bootstrap active file from the newest source file in the workspace."""
        newest: Path | None = None
        newest_mtime = 0.0
        for path in self._iter_source_files():
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= newest_mtime:
                newest_mtime = mtime
                newest = path
        if newest is not None:
            self.process_path(newest)

    def process_path(self, path: Path) -> None:
        try:
            relative = path.relative_to(self._root)
        except ValueError:
            return

        if len(relative.parts) > 8:
            return

        if any(part in self._config.ignored_dirs for part in relative.parts):
            return

        if path.suffix:
            language = _LANGUAGE_MAP.get(path.suffix.lower(), "unknown")
        else:
            language = "unknown"

        try:
            with path.open("r", encoding="utf-8") as handle:
                lines: list[str] = []
                for _ in range(self._config.file_lines):
                    line = handle.readline()
                    if not line:
                        break
                    lines.append(line.rstrip("\n"))
        except (UnicodeDecodeError, FileNotFoundError, OSError):
            return

        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            modified = datetime.now(timezone.utc)

        active_file = ActiveFile(
            path=str(path),
            language=language,
            content="\n".join(lines),
            last_modified=modified,
        )

        with self._lock:
            self._active_file = active_file

    def get_active_file(self) -> ActiveFile | None:
        with self._lock:
            return self._active_file

    def start(self) -> None:
        if self._started:
            return
        self.seed_recent_files()
        self._observer.schedule(self._handler, str(self._root), recursive=True)
        self._observer.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._observer.stop()
        self._observer.join(timeout=2.0)
        self._started = False
