"""Wake word detection for Inferr using openwakeword.

Listens on the local microphone in a background thread. When the
configured wake phrase is detected, fires a callback (which the server
uses to push a "wake_word" WebSocket event to the browser, which then
auto-starts the mic for ambient hands-free interaction).

Default model: "alexa" (pre-trained, downloads automatically)
Custom "hey inferr" model: set wakeword.model_path in config.toml
  pointing to a trained .onnx file.

Training a custom model:
  pip install openwakeword
  python -m openwakeword.train --help

Architecture:
  background thread → sounddevice InputStream → audio queue
  → openwakeword.Model.predict() → callback → asyncio bridge → WS broadcast
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Callable

logger = logging.getLogger("inferr.wakeword")

# ---- Audio constants (must match openwakeword's expected format) ----
_SAMPLE_RATE = 16_000
# 80ms frames at 16kHz — openwakeword's recommended chunk size for low latency
_CHUNK_SIZE = 1_280

# How long to suppress re-detection after a wake word fires (prevents double-trigger)
_COOLDOWN_SECONDS = 3.0


def is_wakeword_available() -> bool:
    """Return True if openwakeword and sounddevice are importable."""
    try:
        import openwakeword  # noqa: F401
        import sounddevice  # noqa: F401
        return True
    except ImportError:
        return False


class WakeWordDetector:
    """Continuous microphone listener that fires on a configured wake phrase.

    Thread model:
        - Audio capture runs in a sounddevice InputStream callback (C thread)
        - Detection runs in a dedicated Python daemon thread
        - on_detected() is called from the detection thread — callers must
          use asyncio.get_event_loop().call_soon_threadsafe() to safely
          bridge back to the event loop (handled by server.py).
    """

    def __init__(
        self,
        model_name: str = "alexa",
        model_path: str = "",
        threshold: float = 0.5,
        cooldown_seconds: float = _COOLDOWN_SECONDS,
    ) -> None:
        # model_path (full .onnx path) takes priority over model_name
        self.model_spec: str = model_path if model_path else model_name
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds

        self._running = False
        self._thread: threading.Thread | None = None
        self._last_detected_at: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return is_wakeword_available()

    def start(self, on_detected: Callable[[str, float], None]) -> bool:
        """Start background detection. Returns False if deps unavailable."""
        if not self.is_available():
            logger.warning(
                "WAKEWORD_UNAVAILABLE install openwakeword and sounddevice: "
                "uv add openwakeword sounddevice"
            )
            return False
        if self._running:
            logger.debug("WAKEWORD_ALREADY_RUNNING")
            return True

        self._running = True
        self._thread = threading.Thread(
            target=self._detection_loop,
            args=(on_detected,),
            daemon=True,
            name="inferr-wakeword",
        )
        self._thread.start()
        logger.info(
            "WAKEWORD_STARTED model=%s threshold=%.2f cooldown_s=%.1f",
            self.model_spec,
            self.threshold,
            self.cooldown_seconds,
        )
        return True

    def stop(self) -> None:
        """Signal the detection thread to stop and wait for it."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=4.0)
            self._thread = None
        logger.info("WAKEWORD_STOPPED")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detection_loop(self, on_detected: Callable[[str, float], None]) -> None:
        """Top-level entry for the daemon thread — wraps _run for logging."""
        try:
            self._run(on_detected)
        except Exception as exc:
            logger.exception("WAKEWORD_LOOP_CRASHED error=%s", exc)
        finally:
            self._running = False

    def _run(self, on_detected: Callable[[str, float], None]) -> None:
        import numpy as np
        import sounddevice as sd

        try:
            from openwakeword.model import Model  # type: ignore[import]
        except ImportError as exc:
            logger.error("WAKEWORD_IMPORT_FAILED error=%s", exc)
            return

        # ---------- Load model (first run downloads ~5MB automatically) ----------
        logger.info("WAKEWORD_MODEL_LOADING spec=%r", self.model_spec)
        try:
            model = Model(
                wakeword_models=[self.model_spec],
                inference_framework="onnx",
            )
        except Exception as exc:
            logger.error(
                "WAKEWORD_MODEL_LOAD_FAILED spec=%r error=%s", self.model_spec, exc
            )
            return
        logger.info("WAKEWORD_MODEL_LOADED")

        audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)

        def _audio_cb(
            indata: np.ndarray, frames: int, t: object, status: object
        ) -> None:
            if status:
                logger.debug("WAKEWORD_AUDIO_STATUS %s", status)
            if not self._running:
                raise sd.CallbackStop()
            try:
                # float32 → int16; openwakeword expects raw 16-bit PCM
                pcm16 = (indata[:, 0] * 32767.0).astype(np.int16)
                audio_q.put_nowait(pcm16)
            except queue.Full:
                pass  # Drop frame — never block the audio callback

        try:
            stream = sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=_CHUNK_SIZE,
                callback=_audio_cb,
            )
        except Exception as exc:
            logger.error("WAKEWORD_STREAM_OPEN_FAILED error=%s", exc)
            return

        logger.info("WAKEWORD_LISTENING sample_rate=%d chunk=%d", _SAMPLE_RATE, _CHUNK_SIZE)
        with stream:
            while self._running:
                try:
                    chunk = audio_q.get(timeout=0.3)
                except queue.Empty:
                    continue

                try:
                    prediction: dict[str, float] = model.predict(chunk)
                except Exception as exc:
                    logger.debug("WAKEWORD_PREDICT_ERR error=%s", exc)
                    continue

                for phrase, score in prediction.items():
                    if score < self.threshold:
                        continue
                    now = time.monotonic()
                    if (now - self._last_detected_at) < self.cooldown_seconds:
                        continue  # Still in cooldown — ignore
                    self._last_detected_at = now
                    logger.info(
                        "WAKEWORD_DETECTED phrase=%s score=%.3f", phrase, score
                    )
                    try:
                        on_detected(phrase, float(score))
                    except Exception as exc:
                        logger.warning("WAKEWORD_CALLBACK_ERR error=%s", exc)
