from __future__ import annotations

import os
import threading
from collections import deque
from typing import Deque

from ptyprocess import PtyProcess


class TerminalCapture:
    def __init__(self, max_lines: int) -> None:
        self._buffer: Deque[str] = deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        shell_path = os.environ.get("SHELL") or "/bin/bash"
        self._process = PtyProcess.spawn([shell_path])

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        pending = ""
        while not self._stop_event.is_set():
            try:
                chunk = self._process.read(1024)
            except OSError:
                break

            if not chunk:
                continue

            text = chunk.decode("utf-8", errors="ignore")

            pending += text
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                self._append_line(line)

    def _append_line(self, line: str) -> None:
        cleaned = line.strip()
        if len(cleaned) < 3:
            return
        if cleaned.endswith("$") or cleaned.endswith("#") or cleaned.endswith("%"):
            return
        with self._lock:
            self._buffer.append(cleaned)

    def get_buffer(self) -> list[str]:
        with self._lock:
            return list(self._buffer)

    def write(self, data: str) -> None:
        try:
            self._process.write(data.encode("utf-8"))
        except OSError:
            return

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._process.terminate(force=True)
        except OSError:
            return
