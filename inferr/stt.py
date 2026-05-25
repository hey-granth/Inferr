"""Local STT backend using faster-whisper.

Replaces Deepgram for Hinglish/code-switching speech that cloud STT
handles poorly.  Runs entirely on-device — no API key required.

Architecture:
  - Receives raw PCM16-LE audio bytes from the browser (16kHz, mono)
  - Uses lightweight energy-based endpointing (no external VAD library)
  - Runs faster-whisper on accumulated segments
  - Emits Deepgram-compatible JSON so the browser needs zero changes

Model selection (via config):
  tiny   → fastest, lowest accuracy   (~39 MB)
  base   → good balance               (~74 MB)
  small  → recommended default        (~244 MB)
  medium → high quality, slower       (~769 MB)
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import time
from typing import AsyncIterator

import numpy as np

logger = logging.getLogger("inferr.stt")

# Lazy import — faster_whisper may not be installed
_faster_whisper_available: bool | None = None


def is_faster_whisper_available() -> bool:
    global _faster_whisper_available
    if _faster_whisper_available is not None:
        return _faster_whisper_available
    try:
        import faster_whisper  # noqa: F401
        _faster_whisper_available = True
    except ImportError:
        _faster_whisper_available = False
    return _faster_whisper_available


# ---------------------------------------------------------------------------
# VAD / endpointing constants
# ---------------------------------------------------------------------------
_SAMPLE_RATE = 16_000          # Hz — must match browser capture
_FRAME_DURATION_MS = 30        # ms per analysis frame
_FRAME_SAMPLES = _SAMPLE_RATE * _FRAME_DURATION_MS // 1000   # 480 samples

# Energy threshold for speech detection (tuned for developer microphone)
_SPEECH_ENERGY_THRESHOLD = 0.01

# Consecutive silence frames required to trigger endpointing
_SILENCE_FRAMES_ENDPOINT = 25  # ~750 ms

# Minimum speech frames before we bother transcribing
_MIN_SPEECH_FRAMES = 8         # ~240 ms — avoids transcribing clicks

# Maximum segment length before forced flush (prevents runaway buffering)
_MAX_SEGMENT_SECONDS = 15


class WhisperSTTSession:
    """Single-client STT session backed by faster-whisper.

    Usage (in a WebSocket handler):
        session = WhisperSTTSession(model_size="small", language="hi")
        await session.load()
        async for result in session.process_audio_stream(audio_iter):
            await ws.send_text(json.dumps(result))
    """

    def __init__(
        self,
        model_size: str = "small",
        language: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        # None → auto-detect language (best for Hinglish)
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._sample_buffer: list[np.ndarray] = []
        self._silence_frames = 0
        self._speech_frames = 0
        self._total_samples = 0
        self._segment_start_samples = 0

    async def load(self) -> None:
        """Load the model in a thread pool (blocks ~0.5-2s first time)."""
        if self._model is not None:
            return
        if not is_faster_whisper_available():
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Run: pip install faster-whisper"
            )
        logger.info(
            "STT_MODEL_LOADING model=%s device=%s compute=%s",
            self.model_size,
            self.device,
            self.compute_type,
        )
        t0 = time.monotonic()
        self._model = await asyncio.to_thread(self._load_model_sync)
        elapsed = time.monotonic() - t0
        logger.info("STT_MODEL_LOADED elapsed_s=%.2f", elapsed)

    def _load_model_sync(self):
        from faster_whisper import WhisperModel  # type: ignore[import]
        return WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    # ------------------------------------------------------------------
    # Public streaming interface
    # ------------------------------------------------------------------

    async def process_audio_stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[dict]:
        """Yield Deepgram-compatible transcript results as they become ready.

        The iterator never yields on silence — only on detected speech
        segments after endpointing.
        """
        pcm_remainder: bytes = b""

        async for chunk in audio_chunks:
            # Accumulate partial frames
            combined = pcm_remainder + chunk
            n_full_frames = len(combined) // (2 * _FRAME_SAMPLES)
            n_bytes_full = n_full_frames * 2 * _FRAME_SAMPLES
            pcm_remainder = combined[n_bytes_full:]

            for frame_idx in range(n_full_frames):
                start = frame_idx * 2 * _FRAME_SAMPLES
                end = start + 2 * _FRAME_SAMPLES
                frame_bytes = combined[start:end]

                samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                energy = float(np.sqrt(np.mean(samples ** 2)))

                if energy >= _SPEECH_ENERGY_THRESHOLD:
                    self._speech_frames += 1
                    self._silence_frames = 0
                    self._sample_buffer.append(samples)
                else:
                    if self._speech_frames > 0:
                        # Padding: include some silence so whisper gets word endings
                        self._sample_buffer.append(samples)
                        self._silence_frames += 1

                        if self._silence_frames >= _SILENCE_FRAMES_ENDPOINT:
                            result = await self._flush_segment()
                            if result is not None:
                                yield result
                    # else: pure silence before any speech — discard

                # Force flush if segment runs too long
                buffered_seconds = (
                    sum(len(b) for b in self._sample_buffer) / _SAMPLE_RATE
                )
                if buffered_seconds >= _MAX_SEGMENT_SECONDS:
                    result = await self._flush_segment()
                    if result is not None:
                        yield result

    async def _flush_segment(self) -> dict | None:
        """Transcribe accumulated audio and return a Deepgram-like result dict."""
        if self._speech_frames < _MIN_SPEECH_FRAMES:
            self._reset_segment()
            return None
        if not self._sample_buffer:
            self._reset_segment()
            return None

        audio = np.concatenate(self._sample_buffer)
        speech_frames = self._speech_frames
        self._reset_segment()

        logger.info(
            "STT_TRANSCRIBE speech_frames=%d audio_len=%.2fs",
            speech_frames,
            len(audio) / _SAMPLE_RATE,
        )

        t0 = time.monotonic()
        try:
            transcript = await asyncio.to_thread(
                self._transcribe_sync, audio
            )
        except Exception as exc:
            logger.exception("STT_TRANSCRIBE_FAILED error=%s", exc)
            return None

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        transcript = transcript.strip()

        if not transcript:
            return None

        logger.info(
            "STT_RESULT elapsed_ms=%d transcript=%r",
            elapsed_ms,
            transcript[:100],
        )

        # Deepgram-compatible result format the browser already handles
        return {
            "type": "Results",
            "is_final": True,
            "speech_final": True,
            "channel": {
                "alternatives": [
                    {"transcript": transcript, "confidence": 0.95}
                ]
            },
        }

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        if self._model is None:
            return ""
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            beam_size=3,
            best_of=3,
            temperature=0.0,
            condition_on_previous_text=True,
            vad_filter=False,   # We handle VAD ourselves
            word_timestamps=False,
            # Suppress non-speech tokens that appear in technical audio
            suppress_tokens=[-1],
            initial_prompt=(
                "Developer speaking. Technical terms, code names, "
                "Python, JavaScript, terminal commands. Hinglish OK."
            ),
        )
        parts = [seg.text for seg in segments]
        return " ".join(parts)

    def _reset_segment(self) -> None:
        self._sample_buffer = []
        self._silence_frames = 0
        self._speech_frames = 0


# ---------------------------------------------------------------------------
# Model singleton — one model shared across all WebSocket connections
# ---------------------------------------------------------------------------

_shared_session: WhisperSTTSession | None = None
_shared_session_lock = asyncio.Lock()


async def get_shared_session(
    model_size: str = "small",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
) -> WhisperSTTSession:
    """Return the process-wide WhisperSTTSession, loading it if needed."""
    global _shared_session
    async with _shared_session_lock:
        if _shared_session is None:
            _shared_session = WhisperSTTSession(
                model_size=model_size,
                language=language,
                device=device,
                compute_type=compute_type,
            )
            await _shared_session.load()
        return _shared_session
