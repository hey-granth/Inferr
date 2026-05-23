from __future__ import annotations

import pyttsx3


def speak_pyttsx3(text: str) -> None:
    """# Phase 0 stub. Replace with Silk API in Phase 1."""
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)
    engine.say(text)
    engine.runAndWait()


def tts_stub_available() -> bool:
    try:
        _ = pyttsx3.init()
        return True
    except Exception:
        return False
