from __future__ import annotations

from inferr.llm import build_system_prompt


def test_hinglish_prompt_contains_latin_script_constraint() -> None:
    prompt = build_system_prompt("hinglish")
    assert "latin script" in prompt.lower()
    assert "devanagari" in prompt.lower()


def test_english_prompt_contains_latin_script_constraint() -> None:
    prompt = build_system_prompt("english")
    assert "latin script" in prompt.lower()


def test_urgent_tone_mentions_errors() -> None:
    prompt = build_system_prompt("hinglish", tone="urgent")
    assert "error" in prompt.lower()


def test_warm_tone_mentions_welcoming() -> None:
    prompt = build_system_prompt("hinglish", tone="warm")
    assert "welcom" in prompt.lower()


def test_neutral_tone_mentions_concise() -> None:
    prompt = build_system_prompt("hinglish", tone="neutral")
    assert "concise" in prompt.lower() or "direct" in prompt.lower()


def test_unknown_tone_falls_back_to_neutral() -> None:
    prompt_unknown = build_system_prompt("hinglish", tone="robot")
    prompt_neutral = build_system_prompt("hinglish", tone="neutral")
    assert "direct" in prompt_unknown.lower() or "concise" in prompt_unknown.lower()
    assert "direct" in prompt_neutral.lower() or "concise" in prompt_neutral.lower()


def test_hinglish_prompt_contains_bhai() -> None:
    prompt = build_system_prompt("hinglish")
    assert "bhai" in prompt.lower()


def test_english_prompt_no_hindi_words() -> None:
    prompt = build_system_prompt("english")
    assert "bhai" not in prompt.lower()
    assert "yaar" not in prompt.lower()

