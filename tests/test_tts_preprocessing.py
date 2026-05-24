from __future__ import annotations
from inferr.tts import _preprocess_tts_text


def test_tone_marker_neutral() -> None:
    result = _preprocess_tts_text("hello world", tone="neutral")
    assert str(result).startswith("[neutral]")


def test_tone_marker_urgent() -> None:
    result = _preprocess_tts_text("error found", tone="urgent")
    assert str(result).startswith("[angry]")


def test_tone_marker_warm_has_chuckle() -> None:
    result = _preprocess_tts_text("hello", tone="warm")
    assert "[happy]" in result
    assert "<chuckle>" in result


def test_latin_text_preserved_through_preprocessing() -> None:
    text = "bhai KeyError aa raha hai line 23 pe"
    result = _preprocess_tts_text(text, tone="urgent")
    assert "KeyError" in result
    assert "line 23" in result


def test_markdown_stripped() -> None:
    result = _preprocess_tts_text("**bhai** ye `error` hai", tone="neutral")
    assert "**" not in result
    assert "`" not in result


def test_bullet_conversion_hinglish() -> None:
    result = _preprocess_tts_text("bhai\n- pehla\n- doosra\n- teesra", tone="neutral")
    assert "Teen cheezein" in result


def test_bullet_conversion_english() -> None:
    result = _preprocess_tts_text("- first\n- second", tone="neutral")
    assert "Two things" in result


def test_truncation_at_400() -> None:
    result = _preprocess_tts_text("a" * 500, tone="neutral")
    assert len(result) <= 410


def test_no_devanagari_in_output() -> None:
    # Preprocessing must not introduce Devanagari
    result = _preprocess_tts_text("yaar kya chal raha hai", tone="neutral")
    for char in result:
        assert not ('\u0900' <= char <= '\u097F'), f"Devanagari char found: {char}"
