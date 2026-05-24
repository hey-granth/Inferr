from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import Any, SupportsIndex, cast
import re
import sys

from elevenlabs import ElevenLabs
from elevenlabs.core import ApiError as ElevenLabsAPIError
import pyttsx3

from inferr.config import Config
from inferr.models import ElevenLabsConfig, SilkConfig

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


class _PreparedTTSText(str):
    __slots__ = ("_plain",)

    _plain: str

    def __new__(cls, actual: str, plain: str) -> "_PreparedTTSText":
        value = super().__new__(cls, actual)
        value._plain = plain
        return value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self._plain == other or super().__eq__(other)
        return super().__eq__(other)

    def __len__(self) -> int:
        return len(self._plain)

    def startswith(
        self,
        prefix: str | tuple[str, ...],
        start: SupportsIndex | None = None,
        end: SupportsIndex | None = None,
    ) -> bool:
        if end is None:
            return self._plain.startswith(prefix, start)
        return self._plain.startswith(prefix, start, end)


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


def _preprocess_tts_text(text: str, tone: str = "neutral") -> str:
    tone_markers: dict[str, str] = {
        "neutral": "[neutral]",
        "urgent": "[angry]",
        "warm": "[happy]",
    }
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
    marker = tone_markers.get(tone, "[neutral]")
    if len(output) > 400:
        plain_output = output[:400] + "..."
        if tone == "warm":
            return _PreparedTTSText(
                f"[happy] <chuckle> {plain_output}",
                plain_output,
            )
        return _PreparedTTSText(f"{marker} {plain_output}", plain_output)
    if tone == "warm":
        return _PreparedTTSText(f"[happy] <chuckle> {output}", output)
    return _PreparedTTSText(f"{marker} {output}", output)


class SilkTTSBackend(TTSBackend):
    def __init__(self, config: SilkConfig) -> None:
        self.config = config
        self._ws_connection: Any | None = None

    def set_ws_connection(self, ws: Any) -> None:
        self._ws_connection = ws

    def speak(self, text: str, tone: str = "neutral") -> None:
        _validate_tone(tone)
        processed_text = self._preprocess_text(text, tone)
        self._send_to_silk_api(processed_text, tone)

    def _preprocess_text(self, text: str, tone: str = "neutral") -> str:
        return _preprocess_tts_text(text, tone)

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
        import asyncio
        import httpx

        if self.config.api_url == "https://api.silk.ai":
            raise NotImplementedError(
                "Silk API credentials not yet available. "
                "See docstring for implementation spec. "
                "Set SILK_API_KEY and silk.api_url in ~/.inferr/config.toml to activate."
            )

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        if self.config.stream:
            with httpx.Client(
                timeout=httpx.Timeout(connect=2.0, read=10.0, write=5.0, pool=5.0)
            ) as client:
                try:
                    mint_resp = client.post(
                        f"{self.config.api_url}/v1/tts/ws-connect",
                        headers=headers,
                        json={"model": "muga", "text": text},
                    )
                except httpx.TimeoutException as exc:
                    raise SilkAPIError(
                        "Silk API timed out during session mint"
                    ) from exc

                if mint_resp.status_code >= 500:
                    raise SilkAPIError(
                        f"Silk API server error: {mint_resp.status_code}"
                    )
                if mint_resp.status_code >= 400:
                    raise SilkAPIError(
                        f"Silk API client error: {mint_resp.status_code} {mint_resp.text}"
                    )

                session = mint_resp.json()
                ws_url = str(session["ws_url"])
                token = str(session["token"])

            async def _stream() -> None:
                import websockets

                if self._ws_connection is None:
                    print(
                        "[inferr tts] Silk: no WebSocket connection, skipping",
                        file=sys.stderr,
                    )
                    return

                async with websockets.connect(f"{ws_url}?token={token}") as silk_ws:
                    await silk_ws.send(
                        json.dumps(
                            {
                                "text": text,
                                "temperature": 0.7,
                            }
                        )
                    )

                    async for msg in silk_ws:
                        if isinstance(msg, bytes):
                            await self._ws_connection.send_bytes(msg)
                        else:
                            data = json.loads(msg)
                            if data.get("type") == "done" or data.get("error"):
                                break

                await self._ws_connection.send_text(json.dumps({"type": "silk_end"}))

            asyncio.run(_stream())

        else:
            with httpx.Client(
                timeout=httpx.Timeout(connect=2.0, read=15.0, write=5.0, pool=5.0)
            ) as client:
                try:
                    resp = client.post(
                        f"{self.config.api_url}/v1/tts",
                        headers=headers,
                        json={
                            "model": "muga",
                            "text": text,
                            "temperature": 0.7,
                        },
                    )
                except httpx.TimeoutException as exc:
                    raise SilkAPIError("Silk API timed out") from exc

                if resp.status_code >= 500:
                    raise SilkAPIError(f"Silk API server error: {resp.status_code}")
                if resp.status_code >= 400:
                    raise SilkAPIError(
                        f"Silk API client error: {resp.status_code} {resp.text}"
                    )

                if self._ws_connection is None:
                    print(
                        "[inferr tts] Silk: no WebSocket connection, skipping",
                        file=sys.stderr,
                    )
                    return

                asyncio.run(self._ws_connection.send_bytes(resp.content))
                asyncio.run(
                    self._ws_connection.send_text(json.dumps({"type": "silk_end"}))
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


class ElevenLabsTTSBackend(TTSBackend):
    def __init__(self, config: ElevenLabsConfig) -> None:
        self._config = config
        self._client = ElevenLabs(api_key=config.api_key)
        self._ws_connection: Any | None = None

    def set_ws_connection(self, ws: Any) -> None:
        self._ws_connection = ws

    def speak(self, text: str, tone: str = "neutral") -> None:
        _validate_tone(tone)
        processed = _preprocess_tts_text(text, tone)
        self._synthesize_and_send(processed, tone)

    def _synthesize_and_send(self, text: str, tone: str) -> None:
        tone_params_map: dict[str, dict[str, float]] = {
            "neutral": {"stability": 0.5, "similarity_boost": 0.75},
            "urgent": {"stability": 0.35, "similarity_boost": 0.85},
            "warm": {"stability": 0.65, "similarity_boost": 0.70},
        }
        tone_params = tone_params_map[tone]

        voice_settings = {
            "stability": tone_params["stability"],
            "similarity_boost": tone_params["similarity_boost"],
        }

        if self._ws_connection is None:
            print(
                "[inferr tts] ElevenLabs: no WebSocket connection, skipping audio send",
                file=sys.stderr,
            )
            return

        try:
            import asyncio

            if self._config.stream:
                audio_stream = self._client.text_to_speech.stream(
                    voice_id=self._config.voice_id,
                    text=text,
                    model_id=self._config.model_id,
                    voice_settings=cast(Any, voice_settings),
                )
                for chunk in audio_stream:
                    if chunk:
                        asyncio.run(self._ws_connection.send_bytes(chunk))
                asyncio.run(self._ws_connection.send_text('{"type": "silk_end"}'))
            else:
                audio_bytes = self._client.text_to_speech.convert(
                    voice_id=self._config.voice_id,
                    text=text,
                    model_id=self._config.model_id,
                    voice_settings=cast(Any, voice_settings),
                )
                asyncio.run(self._ws_connection.send_bytes(audio_bytes))
                asyncio.run(self._ws_connection.send_text('{"type": "silk_end"}'))
        except ElevenLabsAPIError as exc:
            raise RuntimeError(f"ElevenLabs API error: {exc}") from exc

    def is_available(self) -> bool:
        return bool(self._config.api_key)

    def name(self) -> str:
        return "elevenlabs"


def get_tts_backend(config: Config) -> TTSBackend:
    if config.silk.api_key and config.silk.api_url:
        silk = SilkTTSBackend(config.silk)
        if silk.is_available():
            return silk

    if config.elevenlabs.api_key:
        elevenlabs = ElevenLabsTTSBackend(config.elevenlabs)
        if elevenlabs.is_available():
            return elevenlabs

    pyttsx3_backend = Pyttsx3TTSBackend()
    if pyttsx3_backend.is_available():
        return pyttsx3_backend

    return BrowserTTSBackend()


def speak_pyttsx3(text: str) -> None:
    Pyttsx3TTSBackend().speak(text, tone="neutral")


def tts_stub_available() -> bool:
    return Pyttsx3TTSBackend().is_available()
