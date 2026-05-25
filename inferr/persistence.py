"""Lightweight SQLite-backed session persistence for Inferr.

Design principles:
  - No ORM, no migration framework — just raw sqlite3
  - Single file database in ~/.inferr/memory.db
  - Three tables: conversation_turns, error_history, command_history
  - All writes are fire-and-forget (async thread pool)
  - Reads return plain lists/dicts — no model layer coupling

This gives the assistant longitudinal memory without heavyweight frameworks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger("inferr.persistence")

_DB_PATH = Path.home() / ".inferr" / "memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL,  -- 'user' | 'assistant'
    content     TEXT    NOT NULL,
    ts          REAL    NOT NULL   -- unix timestamp
);

CREATE TABLE IF NOT EXISTS error_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    error_type  TEXT    NOT NULL,
    summary     TEXT    NOT NULL,
    raw         TEXT    NOT NULL,
    resolved    INTEGER NOT NULL DEFAULT 0,  -- 0=unresolved, 1=resolved
    ts          REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS command_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    command     TEXT    NOT NULL,
    exit_code   INTEGER NOT NULL DEFAULT 0,
    output      TEXT    NOT NULL DEFAULT '',
    ts          REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conv_session  ON conversation_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_error_session ON error_history(session_id);
CREATE INDEX IF NOT EXISTS idx_error_summary ON error_history(summary);
CREATE INDEX IF NOT EXISTS idx_cmd_session   ON command_history(session_id);
"""


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(_DB_PATH),
        timeout=5.0,
        check_same_thread=False,  # Safe: we use a write lock below
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# Process-level connection shared across threads (WAL + check_same_thread=False)
_conn: sqlite3.Connection | None = None
_write_lock = threading.Lock()  # Serialise writes to prevent corruption


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = _connect()
        logger.info("PERSISTENCE_DB_OPENED path=%s", _DB_PATH)
    return _conn


def close_conn() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


# ---------------------------------------------------------------------------
# Async write helpers (fire-and-forget via thread pool)
# ---------------------------------------------------------------------------


async def save_turn(session_id: str, role: str, content: str) -> None:
    """Persist a conversation turn asynchronously."""
    await asyncio.to_thread(_save_turn_sync, session_id, role, content)


def _save_turn_sync(session_id: str, role: str, content: str) -> None:
    try:
        with _write_lock:
            conn = get_conn()
            conn.execute(
                "INSERT INTO conversation_turns (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time()),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("PERSISTENCE_SAVE_TURN_FAILED error=%s", exc)


async def save_error(
    session_id: str,
    error_type: str,
    summary: str,
    raw: str,
) -> None:
    """Persist a detected error asynchronously."""
    await asyncio.to_thread(_save_error_sync, session_id, error_type, summary, raw)


def _save_error_sync(
    session_id: str, error_type: str, summary: str, raw: str
) -> None:
    try:
        with _write_lock:
            conn = get_conn()
            conn.execute(
                "INSERT INTO error_history (session_id, error_type, summary, raw, ts) VALUES (?, ?, ?, ?, ?)",
                (session_id, error_type, summary, raw, time.time()),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("PERSISTENCE_SAVE_ERROR_FAILED error=%s", exc)


async def save_command(
    session_id: str, command: str, exit_code: int, output: str = ""
) -> None:
    """Persist a shell command asynchronously."""
    await asyncio.to_thread(_save_command_sync, session_id, command, exit_code, output)


def _save_command_sync(
    session_id: str, command: str, exit_code: int, output: str
) -> None:
    try:
        with _write_lock:
            conn = get_conn()
            conn.execute(
                "INSERT INTO command_history (session_id, command, exit_code, output, ts) VALUES (?, ?, ?, ?, ?)",
                (session_id, command, exit_code, output, time.time()),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("PERSISTENCE_SAVE_COMMAND_FAILED error=%s", exc)


# ---------------------------------------------------------------------------
# Sync read helpers (called from async context via to_thread if needed)
# ---------------------------------------------------------------------------


def load_recent_turns(session_id: str, limit: int = 20) -> list[dict]:
    """Return the most recent N conversation turns for a session."""
    try:
        conn = get_conn()
        rows = conn.execute(
            "SELECT role, content FROM conversation_turns "
            "WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        # Return in chronological order
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
    except Exception as exc:
        logger.warning("PERSISTENCE_LOAD_TURNS_FAILED error=%s", exc)
        return []


def find_similar_errors(summary: str, limit: int = 3) -> list[dict]:
    """Find past errors with similar summaries (simple substring match).

    Used for lightweight error recall: 'yeh pehle bhi hua tha' memory.
    Returns most recent N matching errors sorted by recency.
    """
    try:
        conn = get_conn()
        # Extract the error type keyword from summary for matching
        keywords = [w for w in summary.lower().split() if len(w) > 4][:3]
        if not keywords:
            return []

        # Simple keyword OR match across summaries
        placeholders = " OR ".join(["LOWER(summary) LIKE ?" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords]
        rows = conn.execute(
            f"SELECT error_type, summary, raw, ts FROM error_history "
            f"WHERE {placeholders} ORDER BY ts DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [
            {
                "error_type": row["error_type"],
                "summary": row["summary"],
                "raw": row["raw"],
                "ts": row["ts"],
            }
            for row in rows
        ]
    except Exception as exc:
        logger.warning("PERSISTENCE_FIND_ERRORS_FAILED error=%s", exc)
        return []


def get_session_stats(session_id: str) -> dict:
    """Return basic stats for a session (for debug/health endpoints)."""
    try:
        conn = get_conn()
        turns = conn.execute(
            "SELECT COUNT(*) as n FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()["n"]
        errors = conn.execute(
            "SELECT COUNT(*) as n FROM error_history WHERE session_id = ?",
            (session_id,),
        ).fetchone()["n"]
        commands = conn.execute(
            "SELECT COUNT(*) as n FROM command_history WHERE session_id = ?",
            (session_id,),
        ).fetchone()["n"]
        return {"turns": turns, "errors": errors, "commands": commands}
    except Exception as exc:
        logger.warning("PERSISTENCE_STATS_FAILED error=%s", exc)
        return {}
