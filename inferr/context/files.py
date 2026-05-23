from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from inferr.config import Config
from inferr.models import ActiveFile


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

    def process_path(self, path: Path) -> None:
        try:
            relative = path.relative_to(self._root)
        except ValueError:
            return

        if len(relative.parts) > 2:
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
        self._observer.schedule(self._handler, str(self._root), recursive=True)
        self._observer.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._observer.stop()
        self._observer.join(timeout=2.0)
        self._started = False
