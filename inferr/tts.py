from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import Any, SupportsIndex
import re
import sys

import asyncio
import pyttsx3

from inferr.config import Config
from inferr.models import SilkConfig
from inferr.debug import debug_logger

_VALID_TONES = {"neutral", "urgent", "warm"}
_TTS_TRUNCATE_LIMIT = 800
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

_TTS_MIN_TIMEOUT = 12.0
_TTS_MAX_TIMEOUT = 45.0

def tts_timeout_seconds(text: str) -> float:
    # Estimate: ~18 chars/sec, plus base buffer for startup latency.
    est = 8.0 + (max(len(text), 1) / 18.0)
    return max(_TTS_MIN_TIMEOUT, min(_TTS_MAX_TIMEOUT, est))

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
    async def speak(self, text: str, tone: str = "neutral") -> None:
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
    if len(output) > _TTS_TRUNCATE_LIMIT:
        plain_output = output[:_TTS_TRUNCATE_LIMIT] + "..."
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

    async def speak(self, text: str, tone: str = "neutral") -> None:
        _validate_tone(tone)
        
        debug_logger.log_stage("tts_start", {
            "backend": "silk",
            "tone": tone,
            "text_length": len(text)
        })
        
        processed_text = self._preprocess_text(text, tone)
        await self._send_to_silk_api(processed_text, tone)

    def _preprocess_text(self, text: str, tone: str = "neutral") -> str:
        return _preprocess_tts_text(text, tone)

    async def _send_to_silk_api(self, text: str, tone: str) -> None:
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
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=2.0, read=10.0, write=5.0, pool=5.0)
            ) as client:
                try:
                    mint_resp = await client.post(
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

            if self._ws_connection is None:
                print(
                    "[inferr tts] Silk: no WebSocket connection, skipping",
                    file=sys.stderr,
                )
                return

            import websockets

            chunk_count = 0
            total_bytes = 0
            start_time = asyncio.get_running_loop().time()
            async with websockets.connect(f"{ws_url}?token={token}") as silk_ws:
                try:
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
                            chunk_count += 1
                            total_bytes += len(msg)
                            await self._ws_connection.send_bytes(msg)
                        else:
                            data = json.loads(msg)
                            if data.get("type") == "done" or data.get("error"):
                                break
                finally:
                    elapsed_ms = int((asyncio.get_running_loop().time() - start_time) * 1000)
                    debug_logger.log_stage("tts_silk_stream_end", {
                        "chunk_count": chunk_count,
                        "total_bytes": total_bytes,
                        "elapsed_ms": elapsed_ms
                    })
                    
                    await self._ws_connection.send_text(
                        json.dumps(
                            {
                                "type": "diagnostic",
                                "stage": "silk_stream",
                                "chunk_count": chunk_count,
                                "total_bytes": total_bytes,
                                "elapsed_ms": elapsed_ms,
                            }
                        )
                    )
                    # Always emit completion marker so browser finalizes playback reliably.
                    await self._ws_connection.send_text(
                        json.dumps({"type": "silk_end"})
                    )

        else:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=2.0, read=15.0, write=5.0, pool=5.0)
            ) as client:
                try:
                    resp = await client.post(
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

                debug_logger.log_stage("tts_silk_full_end", {
                    "total_bytes": len(resp.content),
                })

                await self._ws_connection.send_bytes(resp.content)
                await self._ws_connection.send_text(
                    json.dumps(
                        {
                            "type": "diagnostic",
                            "stage": "silk_full",
                            "chunk_count": 1,
                            "total_bytes": len(resp.content),
                        }
                    )
                )
                await self._ws_connection.send_text(json.dumps({"type": "silk_end"}))

    def is_available(self) -> bool:
        return bool(self.config.api_url and self.config.api_key)

    def name(self) -> str:
        return "silk"


class Pyttsx3TTSBackend(TTSBackend):
    async def speak(self, text: str, tone: str = "neutral") -> None:
        _validate_tone(tone)
        # pyttsx3 does not support tone modulation, run in separate thread
        await asyncio.to_thread(self._speak_sync, text)

    def _speak_sync(self, text: str) -> None:
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
    async def speak(self, text: str, tone: str = "neutral") -> None:
        _validate_tone(tone)
        return

    def is_available(self) -> bool:
        return True

    def name(self) -> str:
        return "browser"


def get_tts_backend(config: Config) -> TTSBackend:
    if config.silk.api_key and config.silk.api_url:
        silk = SilkTTSBackend(config.silk)
        if silk.is_available():
            return silk

    pyttsx3_backend = Pyttsx3TTSBackend()
    if pyttsx3_backend.is_available():
        return pyttsx3_backend

    return BrowserTTSBackend()


async def speak_pyttsx3(text: str) -> None:
    await Pyttsx3TTSBackend().speak(text, tone="neutral")


def tts_stub_available() -> bool:
    return Pyttsx3TTSBackend().is_available()
