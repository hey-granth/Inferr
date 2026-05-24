from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("inferr.pipeline")

# Structured stage markers — grep logs for PIPELINE_LLM_OK etc.
STT_OK = "STT_OK"
STT_FAILED = "STT_FAILED"
TRANSCRIPT_FINALIZED = "TRANSCRIPT_FINALIZED"
TRANSCRIPT_EMPTY = "TRANSCRIPT_EMPTY"
LLM_OK = "LLM_OK"
LLM_FAILED = "LLM_FAILED"
TTS_OK = "TTS_OK"
TTS_FAILED = "TTS_FAILED"
PLAYBACK_STARTED = "PLAYBACK_STARTED"
PLAYBACK_COMPLETED = "PLAYBACK_COMPLETED"
ASSISTANT_OK = "ASSISTANT_OK"
ASSISTANT_FAILED = "ASSISTANT_FAILED"


def log_pipeline_stage(
    stage: str,
    *,
    ok: bool = True,
    detail: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    """Log and return a structured pipeline stage record."""
    payload: dict[str, object] = {
        "type": "pipeline_stage",
        "stage": stage,
        "ok": ok,
        "detail": detail or {},
    }
    if error:
        payload["error"] = error

    status = "OK" if ok else "FAILED"
    line = f"PIPELINE_{stage}: {status}"
    if error:
        line = f"{line} — {error}"
    if detail:
        line = f"{line} | {detail}"

    if ok:
        logger.info(line)
    else:
        logger.error(line)

    return payload


def timed_detail(start: float, **extra: object) -> dict[str, object]:
    out: dict[str, object] = {"latency_ms": int((time.monotonic() - start) * 1000)}
    out.update(extra)
    return out
