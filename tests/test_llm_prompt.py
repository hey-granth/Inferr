from __future__ import annotations

from inferr.llm import _is_high_signal, _rank_terminal_lines, build_system_prompt


def test_hinglish_prompt_contains_latin_script_constraint() -> None:
    prompt = build_system_prompt("hinglish")
    assert "latin script" in prompt.lower() or "non-latin" in prompt.lower()


def test_english_prompt_contains_no_filler_instruction() -> None:
    prompt = build_system_prompt("english")
    assert "filler" in prompt.lower() or "no filler" in prompt.lower()


def test_urgent_tone_mentions_errors() -> None:
    prompt = build_system_prompt("hinglish", tone="urgent")
    assert "error" in prompt.lower()


def test_neutral_tone_is_operational() -> None:
    # Neutral prompt should enforce direct/concise behavior, not a persona
    prompt = build_system_prompt("hinglish", tone="neutral")
    assert "direct" in prompt.lower() or "concise" in prompt.lower() or "short" in prompt.lower()


def test_no_warm_welcome_in_prompt() -> None:
    # warm tone is removed — no first-session greeting behavior
    prompt_neutral = build_system_prompt("hinglish", tone="neutral")
    prompt_warm = build_system_prompt("hinglish", tone="warm")
    assert "welcom" not in prompt_neutral.lower()
    assert "welcom" not in prompt_warm.lower()


def test_no_roleplay_allowed() -> None:
    prompt = build_system_prompt("english", tone="neutral")
    assert "roleplay" in prompt.lower()


def test_no_brainstorm_without_request() -> None:
    prompt = build_system_prompt("english", tone="neutral")
    assert "brainstorm" in prompt.lower()


def test_english_prompt_no_forced_hindi_slang() -> None:
    # The hardened prompt must NOT hard-code 'bhai' or 'yaar'
    prompt = build_system_prompt("english")
    assert "bhai" not in prompt.lower()
    assert "yaar" not in prompt.lower()


def test_hinglish_prompt_no_forced_slang() -> None:
    # The hardened prompt removes slang pressure — 'bhai' should not appear
    prompt = build_system_prompt("hinglish")
    assert "bhai" not in prompt.lower()


# ── Terminal ranking tests ───────────────────────────────────────────────────


def test_high_signal_detects_traceback() -> None:
    assert _is_high_signal("Traceback (most recent call last):")


def test_high_signal_detects_error() -> None:
    assert _is_high_signal("RuntimeError: cannot run event loop")


def test_high_signal_detects_5xx() -> None:
    assert _is_high_signal("INFO:     127.0.0.1 - POST /query HTTP/1.1  500 Internal Server Error")


def test_low_signal_does_not_match_generic_log() -> None:
    assert not _is_high_signal("INFO:     Application startup complete.")
    assert not _is_high_signal("DEBUG: context assembled in 12ms")


def test_rank_puts_high_signal_before_low() -> None:
    lines = [
        "INFO: startup complete",
        "INFO: context assembled",
        "RuntimeError: asyncio loop closed",
        "DEBUG: healthy tick",
    ]
    ranked = _rank_terminal_lines(lines, limit=4)
    # High-signal lines should appear first (index 0 or 1)
    high_positions = [i for i, l in enumerate(ranked) if _is_high_signal(l)]
    low_positions = [i for i, l in enumerate(ranked) if not _is_high_signal(l)]
    assert all(h < l for h in high_positions for l in low_positions)


def test_rank_respects_limit() -> None:
    lines = [f"line {i}" for i in range(20)]
    ranked = _rank_terminal_lines(lines, limit=5)
    assert len(ranked) <= 5


def test_rank_skips_empty_lines() -> None:
    lines = ["", "  ", "RuntimeError: oops", ""]
    ranked = _rank_terminal_lines(lines, limit=5)
    assert all(l.strip() for l in ranked)
