from __future__ import annotations

from datetime import datetime, timezone

from inferr.config import Config
from inferr.models import ContextObject, ConversationTurn
from inferr.context.errors import extract_errors
from inferr.context.files import FileWatcher
from inferr.context.history import read_shell_history
from inferr.context.terminal import TerminalCapture


class ContextAssembler:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._terminal = TerminalCapture(config.terminal_buffer_lines)
        self._file_watcher = FileWatcher(config)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._file_watcher.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            self._terminal.stop()
            return
        self._file_watcher.stop()
        self._terminal.stop()
        self._started = False

    def inject_command(self, line: str) -> None:
        """Forward a shell-plugin captured line into the terminal buffer."""
        self._terminal.inject_line(line)

    def assemble(
        self, session_id: str, conversation_history: list[ConversationTurn]
    ) -> ContextObject:
        from inferr.server import get_shell_command_buffer

        terminal_buffer = self._terminal.get_buffer()

        # Merge shell plugin commands (real terminal) with file history
        plugin_commands = list(get_shell_command_buffer()[-self._config.history_depth:])
        file_history = read_shell_history(self._config.history_depth)

        # Plugin commands take priority — they are real-time. Use file history as fallback.
        shell_history = plugin_commands if plugin_commands else file_history

        active_file = self._file_watcher.get_active_file()

        # Run error detection on both terminal buffer AND recent shell commands
        all_lines = terminal_buffer + plugin_commands
        flagged_errors = extract_errors(all_lines)

        limited_history = conversation_history[-6:]
        timestamp = datetime.now(timezone.utc)

        return ContextObject(
            terminal_buffer=terminal_buffer,
            shell_history=shell_history,
            active_file=active_file,
            flagged_errors=flagged_errors,
            conversation_history=limited_history,
            session_id=session_id,
            timestamp=timestamp,
        )
