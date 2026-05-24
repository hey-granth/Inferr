from __future__ import annotations

from datetime import datetime, timezone

from inferr.llm import _summarize_context
from inferr.models import ActiveFile, ContextObject, QueryRequest


def _request_with_context(context: ContextObject) -> QueryRequest:
    return QueryRequest(transcript="what is wrong?", context=context)


def test_context_summary_recent_commands_not_corrupted() -> None:
    context = ContextObject(
        terminal_buffer=["INFO: ok\r"],
        shell_history=["git status"],
        active_file=None,
        flagged_errors=[],
        conversation_history=[],
        session_id="s1",
        timestamp=datetime.now(timezone.utc),
    )
    summary = _summarize_context(_request_with_context(context))
    assert "Recent commands:" in summary
    assert "\r" not in summary
    assert not any(line.startswith("ecent commands:") for line in summary.splitlines())


def test_context_summary_includes_active_file_snippet() -> None:
    context = ContextObject(
        terminal_buffer=[],
        shell_history=[],
        active_file=ActiveFile(
            path="/proj/app.py",
            language="python",
            content="def main():\n    pass\n",
            last_modified=datetime.now(timezone.utc),
        ),
        flagged_errors=[],
        conversation_history=[],
        session_id="s1",
        timestamp=datetime.now(timezone.utc),
    )
    summary = _summarize_context(_request_with_context(context))
    assert "Active file: app.py" in summary
    assert "def main():" in summary
