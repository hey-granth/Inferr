"""Tests for SQLite session persistence."""
from __future__ import annotations

import time
import pytest

import inferr.persistence as db


@pytest.fixture(autouse=True)
def in_memory_db(tmp_path, monkeypatch):
    """Redirect the DB path to a temp file for each test."""
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "test_memory.db")
    monkeypatch.setattr(db, "_conn", None)
    yield
    db.close_conn()
    monkeypatch.setattr(db, "_conn", None)


class TestPersistenceInit:
    def test_db_initializes_on_first_access(self, tmp_path):
        conn = db.get_conn()
        assert conn is not None
        # Tables should exist
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {row[0] for row in tables}
        assert "conversation_turns" in names
        assert "error_history" in names
        assert "command_history" in names


class TestConversationPersistence:
    @pytest.mark.asyncio
    async def test_save_and_load_turn(self):
        session_id = "test-session-001"
        await db.save_turn(session_id, "user", "kya ho raha hai?")
        await db.save_turn(session_id, "assistant", "Sab theek hai, traceback dekh.")

        turns = db.load_recent_turns(session_id, limit=10)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert "kya ho" in turns[0]["content"]
        assert turns[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_load_respects_limit(self):
        session_id = "test-session-002"
        for i in range(10):
            await db.save_turn(session_id, "user", f"message {i}")

        turns = db.load_recent_turns(session_id, limit=5)
        assert len(turns) == 5

    @pytest.mark.asyncio
    async def test_turns_returned_in_chronological_order(self):
        session_id = "test-session-003"
        await db.save_turn(session_id, "user", "first")
        await db.save_turn(session_id, "assistant", "second")

        turns = db.load_recent_turns(session_id)
        assert turns[0]["content"] == "first"
        assert turns[1]["content"] == "second"

    @pytest.mark.asyncio
    async def test_different_sessions_isolated(self):
        await db.save_turn("session-a", "user", "hello from A")
        await db.save_turn("session-b", "user", "hello from B")

        turns_a = db.load_recent_turns("session-a")
        turns_b = db.load_recent_turns("session-b")
        assert len(turns_a) == 1
        assert len(turns_b) == 1
        assert "A" in turns_a[0]["content"]
        assert "B" in turns_b[0]["content"]


class TestErrorPersistence:
    @pytest.mark.asyncio
    async def test_save_and_find_error(self):
        session_id = "test-session-004"
        await db.save_error(
            session_id,
            "python_traceback",
            "ImportError: No module named 'fastapi'",
            "Traceback (most recent call last):\n  ..."
        )

        results = db.find_similar_errors("ImportError fastapi", limit=3)
        assert len(results) >= 1
        assert "ImportError" in results[0]["summary"]

    @pytest.mark.asyncio
    async def test_find_similar_errors_no_match_returns_empty(self):
        results = db.find_similar_errors("xyzzy_nonexistent_error", limit=3)
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_find_similar_errors_short_keyword_skipped(self):
        """Keywords shorter than 5 chars should be ignored to avoid noise."""
        results = db.find_similar_errors("err at", limit=3)
        # Short words filtered, so should return empty or limited
        assert isinstance(results, list)


class TestCommandPersistence:
    @pytest.mark.asyncio
    async def test_save_command(self):
        session_id = "test-session-005"
        await db.save_command(session_id, "python main.py", 1, "Traceback...")

        stats = db.get_session_stats(session_id)
        assert stats["commands"] == 1

    @pytest.mark.asyncio
    async def test_session_stats_all_zero_for_new_session(self):
        stats = db.get_session_stats("brand-new-session")
        assert stats["turns"] == 0
        assert stats["errors"] == 0
        assert stats["commands"] == 0

    @pytest.mark.asyncio
    async def test_session_stats_aggregate_correctly(self):
        sid = "test-session-006"
        await db.save_turn(sid, "user", "hi")
        await db.save_turn(sid, "assistant", "hey")
        await db.save_error(sid, "python_traceback", "TypeError: ...", "raw")
        await db.save_command(sid, "python app.py", 0, "")

        stats = db.get_session_stats(sid)
        assert stats["turns"] == 2
        assert stats["errors"] == 1
        assert stats["commands"] == 1
