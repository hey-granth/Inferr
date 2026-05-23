from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import re
import sys

import pyttsx3

from inferr.config import Config
from inferr.models import SilkConfig

_VALID_TONES = {"neutral", "urgent", "warm"}
_HINGLISH_HINTS = {
    "bhai",
    "yaar",
    "dekh",
    "chal",
    "hai",
    "karo",
    "tha",
    "mein",
    "pe",
    "se",
}


class SilkAPIError(Exception):
    """Raised when Silk API synthesis fails."""


class TTSBackend(ABC):
    @abstractmethod
    def speak(self, text: str, tone: str = "neutral") -> None:
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


def _validate_tone(tone: str) -> None:
    if tone not in _VALID_TONES:
        raise ValueError(f"Unsupported tone: {tone}")


def _count_to_spoken(count: int, hinglish: bool) -> str:
    if hinglish:
        if count == 1:
            return "Ek baat: "
        if count == 2:
            return "Do cheezein: "
        if count == 3:
            return "Teen cheezein: "
        return "Kuch cheezein: "

    english_map = {
        1: "One point: ",
        2: "Two things: ",
        3: "Three things: ",
        4: "Four things: ",
        5: "Five things: ",
        6: "Six things: ",
        7: "Seven things: ",
        8: "Eight things: ",
        9: "Nine things: ",
    }
    if count >= 10:
        return "Several things: "
    return english_map.get(count, "Several things: ")


def _is_bullet(line: str) -> bool:
    return line.startswith("- ") or line.startswith("• ") or line.startswith("* ")


def _strip_bullet_prefix(line: str) -> str:
    return line[2:].strip()


def _is_hinglish_text(text: str) -> bool:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return any(word in _HINGLISH_HINTS for word in words)


class SilkTTSBackend(TTSBackend):
    def __init__(self, config: SilkConfig) -> None:
        self.config = config
        self._ws_connection: Any | None = None

    def set_ws_connection(self, ws: Any) -> None:
        self._ws_connection = ws

    def speak(self, text: str, tone: str = "neutral") -> None:
        _validate_tone(tone)
        processed_text = self._preprocess_text(text)
        self._send_to_silk_api(processed_text, tone)

    def _preprocess_text(self, text: str) -> str:
        cleaned = re.sub(r"[*_`#]", "", text)
        lines = [line.strip() for line in cleaned.splitlines()]

        normal_parts: list[str] = []
        bullets: list[str] = []
        for line in lines:
            if not line:
                continue
            if _is_bullet(line):
                item = _strip_bullet_prefix(line)
                if item:
                    bullets.append(item)
            else:
                normal_parts.append(line)

        output = " ".join(normal_parts)
        if bullets:
            prefix = _count_to_spoken(len(bullets), _is_hinglish_text(cleaned))
            bullet_text = f"{prefix}{', '.join(bullets)}"
            output = f"{output} {bullet_text}".strip()

        output = re.sub(r"\s+", " ", output).strip()
        if len(output) > 600:
            return output[:600] + "..."
        return output

    def _send_to_silk_api(self, text: str, tone: str) -> None:
        """
        NOT YET IMPLEMENTED. Raises NotImplementedError until Silk API credentials
        are available.

        When implementing:
        - POST to {self.config.api_url}/synthesize
        - Headers: {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        - Body: {
            "text": text,
            "voice_id": self.config.voice_id,
            "tone": tone,               # "neutral" | "urgent" | "warm"
            "stream": self.config.stream,
            "format": "mp3"             # browser-native format, no conversion needed
          }
        - If stream=True: iterate over response.iter_bytes(chunk_size=4096),
          send each chunk to the browser via WebSocket as binary frames.
          Use the WebSocket connection reference stored in self._ws_connection.
        - If stream=False: get full response bytes, send as single binary frame.
        - On HTTP 4xx: raise SilkAPIError(f"Silk API client error: {status_code} {response_text}")
        - On HTTP 5xx: raise SilkAPIError(f"Silk API server error: {status_code}")
        - On httpx.TimeoutException: raise SilkAPIError("Silk API timed out")
        - httpx.Client timeout: connect=2.0, read=10.0
        """
        raise NotImplementedError(
            "Silk API credentials not yet available. "
            "See docstring for implementation spec. "
            "Set SILK_API_KEY and silk.api_url in ~/.inferr/config.toml to activate."
        )

    def is_available(self) -> bool:
        return bool(self.config.api_url and self.config.api_key)

    def name(self) -> str:
        return "silk"


class Pyttsx3TTSBackend(TTSBackend):
    def speak(self, text: str, tone: str = "neutral") -> None:
        _validate_tone(tone)
        # pyttsx3 does not support tone modulation
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[inferr tts] pyttsx3 error: {e}", file=sys.stderr)

    def is_available(self) -> bool:
        try:
            _ = pyttsx3.init()
            return True
        except Exception:
            return False

    def name(self) -> str:
        return "pyttsx3"


class BrowserTTSBackend(TTSBackend):
    def speak(self, text: str, tone: str = "neutral") -> None:
        _validate_tone(tone)
        return

    def is_available(self) -> bool:
        return True

    def name(self) -> str:
        return "browser"


def get_tts_backend(config: Config) -> TTSBackend:
    if config.silk.api_key and config.silk.api_url:
        backend = SilkTTSBackend(config.silk)
        if backend.is_available():
            return backend

    pyttsx3_backend = Pyttsx3TTSBackend()
    if pyttsx3_backend.is_available():
        return pyttsx3_backend

    return BrowserTTSBackend()


def speak_pyttsx3(text: str) -> None:
    Pyttsx3TTSBackend().speak(text, tone="neutral")


def tts_stub_available() -> bool:
    return Pyttsx3TTSBackend().is_available()
