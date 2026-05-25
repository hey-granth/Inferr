from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import subprocess
import time
import uuid
import json

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from inferr.config import Config, load_config
from inferr.context import ContextAssembler
from inferr.llm import query_llm
from inferr.models import (
    ContextObject,
    ConversationTurn,
    DebugAssistantRequest,
    DebugLlmRequest,
    DebugTtsRequest,
    QueryRequest,
    QueryResponse,
    ShellCommandCapture,
    WebSocketMessage,
)
from inferr import pipeline as pipeline_stages
from inferr.tts import SilkTTSBackend, TTSBackend, get_tts_backend, tts_timeout_seconds
from inferr.debug import debug_logger
from inferr.wakeword import WakeWordDetector
import inferr.persistence as db

logger = logging.getLogger("inferr.server")

session_id: str | None = None
context_assembler: ContextAssembler | None = None
conversation_history: list[ConversationTurn] = []
tts_backend: TTSBackend | None = None
last_activity: datetime | None = None

# Wake word detector (optional — requires openwakeword + sounddevice)
_wake_detector: WakeWordDetector | None = None
# All live /ws WebSocket connections (for wake-word broadcast)
_ws_clients: set[WebSocket] = set()
# Reference to the running event loop — set once on first WS connect
_event_loop: asyncio.AbstractEventLoop | None = None

_shell_command_buffer: list[str] = []
_SHELL_BUFFER_MAX = 200

# Active /ws client — used by Silk TTS debug path when no transcript flow is running.
_main_websocket: WebSocket | None = None


def get_shell_command_buffer() -> list[str]:
    """Return the current shell command buffer (public accessor for cross-module use)."""
    return _shell_command_buffer


CONFIG_OVERRIDE: Config | None = None


def _apply_env_overrides(config: Config) -> Config:
    port_env = os.environ.get("INFERR_PORT")
    if port_env is None:
        return config
    try:
        port = int(port_env)
    except ValueError:
        return config
    if port == config.port:
        return config
    return replace(config, port=port)


def get_config() -> Config:
    if CONFIG_OVERRIDE is not None:
        return CONFIG_OVERRIDE
    return _apply_env_overrides(load_config())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager."""
    browser_dir = Path(__file__).resolve().parent.parent / "browser"
    if browser_dir.exists():
        app.mount(
            "/", StaticFiles(directory=str(browser_dir), html=True), name="browser"
        )
    try:
        yield
    finally:
        if _wake_detector is not None:
            _wake_detector.stop()
        if context_assembler is not None:
            context_assembler.stop()
        db.close_conn()


app = FastAPI(lifespan=lifespan)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Return 204 No Content for favicon to prevent log pollution."""
    return Response(status_code=204)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return structured JSON for unexpected server errors."""
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


@app.post("/session/start")
async def start_session() -> dict[str, str]:
    """Initialize a new Inferr session and context capture."""
    global session_id, context_assembler, conversation_history, tts_backend
    global _wake_detector

    config = get_config()
    if context_assembler is not None:
        context_assembler.stop()

    context_assembler = ContextAssembler(config)
    context_assembler.start()
    tts_backend = get_tts_backend(config)
    session_id = str(uuid.uuid4())
    conversation_history = []

    # Start wake word detector if configured
    if config.wakeword.enabled:
        if _wake_detector is not None:
            _wake_detector.stop()
        _wake_detector = WakeWordDetector(
            model_name=config.wakeword.model_name,
            model_path=config.wakeword.model_path,
            threshold=config.wakeword.threshold,
            cooldown_seconds=config.wakeword.cooldown_seconds,
        )
        started = _wake_detector.start(on_detected=_on_wake_word_detected)
        if not started:
            logger.warning(
                "WAKEWORD_START_FAILED wakeword.enabled=true but deps unavailable — "
                "run: uv add openwakeword sounddevice"
            )

    if os.environ.get("INFERR_NO_BROWSER") != "1":
        try:
            subprocess.Popen(
                ["google-chrome", f"http://{config.host}:{config.port}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, FileNotFoundError):
            logger.info("Unable to open browser companion UI.")

    return {"session_id": session_id, "status": "started"}


@app.post("/session/stop")
async def stop_session() -> dict[str, str]:
    """Stop the current Inferr session."""
    global session_id, context_assembler, conversation_history, tts_backend
    global _wake_detector

    if _wake_detector is not None:
        _wake_detector.stop()
        _wake_detector = None
    if context_assembler is not None:
        context_assembler.stop()
    context_assembler = None
    tts_backend = None
    session_id = None
    conversation_history = []

    return {"status": "stopped"}


@app.get("/context")
async def get_context() -> JSONResponse:
    """Return the current context object."""
    if session_id is None or context_assembler is None:
        return JSONResponse(status_code=404, content={"error": "No active session"})

    context = context_assembler.assemble(session_id, conversation_history)
    return JSONResponse(status_code=200, content=context.model_dump(mode="json"))


@app.get("/health")
async def health() -> dict[str, object | None]:
    """Return server health status."""
    stats = {}
    if session_id:
        stats = db.get_session_stats(session_id)
    return {
        "status": "ok",
        "session_active": session_id is not None,
        "last_activity": last_activity.isoformat() if last_activity else None,
        "tts_backend": tts_backend.name() if tts_backend is not None else None,
        "session_stats": stats,
    }


@app.get("/config/browser")
async def browser_config() -> dict[str, object]:
    """Return config the browser needs at runtime. Only non-secret values."""
    cfg = get_config()
    from inferr.stt import is_faster_whisper_available
    use_local_stt = is_faster_whisper_available()
    return {
        "deepgram_enabled": bool(cfg.deepgram.api_key) and not use_local_stt,
        "local_stt_enabled": use_local_stt,
        "tts_backend": tts_backend.name() if tts_backend else "browser",
        "stt_sample_rate": 16000,
        "stt_encoding": "linear16",
        "stt_channels": 1,
        "wake_word_enabled": cfg.wakeword.enabled,
        "wake_word_model": cfg.wakeword.model_name,
    }


def _empty_debug_context(session_label: str = "debug") -> ContextObject:
    return ContextObject(
        terminal_buffer=[],
        shell_history=[],
        active_file=None,
        flagged_errors=[],
        conversation_history=[],
        session_id=session_label,
        timestamp=datetime.now(timezone.utc),
    )


async def _emit_stage(websocket: WebSocket | None, stage: str, **kwargs: object) -> None:
    payload = pipeline_stages.log_pipeline_stage(stage, **kwargs)  # type: ignore[arg-type]
    if websocket is not None:
        try:
            await websocket.send_json(payload)
        except RuntimeError:
            pass


async def _run_tts(
    websocket: WebSocket | None,
    text: str,
    tone: str,
) -> bool:
    """Run TTS; emit TTS_OK / TTS_FAILED. Returns True on success."""
    if tts_backend is None:
        await _emit_stage(
            websocket,
            pipeline_stages.TTS_FAILED,
            ok=False,
            error="no TTS backend configured",
        )
        return False

    if isinstance(tts_backend, SilkTTSBackend):
        if websocket is None:
            await _emit_stage(
                websocket,
                pipeline_stages.TTS_FAILED,
                ok=False,
                error="Silk TTS requires an open /ws connection",
            )
            return False
        tts_backend.set_ws_connection(websocket)

    tts_start = time.monotonic()
    try:
        tts_timeout = tts_timeout_seconds(text)
        await asyncio.wait_for(
            tts_backend.speak(text, tone=tone),
            timeout=tts_timeout,
        )
        await _emit_stage(
            websocket,
            pipeline_stages.TTS_OK,
            detail=pipeline_stages.timed_detail(
                tts_start,
                backend=tts_backend.name(),
                text_len=len(text),
            ),
        )
        return True
    except TimeoutError:
        await _emit_stage(
            websocket,
            pipeline_stages.TTS_FAILED,
            ok=False,
            error=f"{tts_backend.name()} timed out",
            detail={"timeout_s": tts_timeout},
        )
        return False
    except Exception as exc:
        await _emit_stage(
            websocket,
            pipeline_stages.TTS_FAILED,
            ok=False,
            error=str(exc),
            detail={"backend": tts_backend.name()},
        )
        return False


async def _handle_transcript(
    websocket: WebSocket,
    transcript: str,
    *,
    source: str = "websocket",
) -> None:
    """LLM → response JSON → TTS. Shared by live STT and debug transcript injection."""
    global last_activity

    trimmed = transcript.strip()
    if not trimmed:
        await _emit_stage(
            websocket,
            pipeline_stages.TRANSCRIPT_EMPTY,
            ok=False,
            error="empty transcript",
        )
        return

    await _emit_stage(
        websocket,
        pipeline_stages.TRANSCRIPT_FINALIZED,
        detail={"transcript": trimmed, "source": source},
    )

    if session_id is None or context_assembler is None:
        await _emit_stage(
            websocket,
            pipeline_stages.ASSISTANT_FAILED,
            ok=False,
            error="no active session",
        )
        await websocket.send_json(
            {"type": "error", "message": "No active session"}
        )
        return

    context = context_assembler.assemble(session_id, conversation_history)
    tone = resolve_tone(context)
    request = QueryRequest(transcript=trimmed, context=context)

    llm_start = time.monotonic()
    try:
        response_text = await query_llm(request, get_config(), tone=tone)
    except Exception as exc:
        await _emit_stage(
            websocket,
            pipeline_stages.LLM_FAILED,
            ok=False,
            error=str(exc),
            detail=pipeline_stages.timed_detail(llm_start),
        )
        await _emit_stage(
            websocket,
            pipeline_stages.ASSISTANT_FAILED,
            ok=False,
            error="LLM failed",
        )
        await websocket.send_json({"type": "error", "message": str(exc)})
        return

    await _emit_stage(
        websocket,
        pipeline_stages.LLM_OK,
        detail=pipeline_stages.timed_detail(
            llm_start,
            response_len=len(response_text),
            model=get_config().groq.model,
        ),
    )

    conversation_history.append(
        ConversationTurn(role="user", content=trimmed)
    )
    conversation_history.append(
        ConversationTurn(role="assistant", content=response_text)
    )
    if len(conversation_history) > 6:
        conversation_history[:] = conversation_history[-6:]

    last_activity = datetime.now(timezone.utc)

    has_errors = any(
        flagged_error.type != "marker"
        for flagged_error in context.flagged_errors
    )
    backend_name = tts_backend.name() if tts_backend is not None else "browser"

    active_file_name = None
    if context.active_file is not None:
        active_file_name = Path(context.active_file.path).name

    context_stats = {
        "terminal_lines": len(context.terminal_buffer),
        "active_file": active_file_name,
        "error_count": sum(
            1 for e in context.flagged_errors if e.type != "marker"
        ),
    }

    response_payload = {
        "type": "response",
        "text": response_text,
        "has_errors": has_errors,
        "tone": tone,
        "tts_backend": backend_name,
        "context_stats": context_stats,
    }
    await websocket.send_json(response_payload)

    tts_ok = await _run_tts(websocket, response_text, tone)
    if tts_ok:
        await _emit_stage(websocket, pipeline_stages.ASSISTANT_OK, detail={"source": source})
    else:
        await _emit_stage(
            websocket,
            pipeline_stages.ASSISTANT_FAILED,
            ok=False,
            error="TTS failed after LLM succeeded",
            detail={"llm_response_len": len(response_text)},
        )


def resolve_tone(context: ContextObject) -> str:
    """
    Returns 'urgent' if real (non-marker) errors are flagged.
    Returns 'neutral' otherwise.
    """
    if any(error.type != "marker" for error in context.flagged_errors):
        return "urgent"
    return "neutral"


# ---------------------------------------------------------------------------
# Wake word broadcast bridge (thread → asyncio event loop)
# ---------------------------------------------------------------------------

def _on_wake_word_detected(phrase: str, score: float) -> None:
    """Called from the WakeWordDetector background thread.

    Bridges to the asyncio event loop via call_soon_threadsafe so we can
    safely push a WebSocket message to all connected browser clients.
    """
    if _event_loop is None:
        return
    _event_loop.call_soon_threadsafe(
        lambda: asyncio.create_task(_broadcast_wake_word(phrase, score))
    )


async def _broadcast_wake_word(phrase: str, score: float) -> None:
    """Fan out a wake_word event to all live /ws clients."""
    if not _ws_clients:
        return
    payload = {"type": "wake_word", "phrase": phrase, "score": round(score, 3)}
    dead: set[WebSocket] = set()
    for ws in list(_ws_clients):
        try:
            await ws.send_json(payload)
            logger.info("WAKEWORD_BROADCAST phrase=%s score=%.3f", phrase, score)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


# ---------------------------------------------------------------------------
# Local STT WebSocket — replaces Deepgram proxy when faster-whisper available
# ---------------------------------------------------------------------------

@app.websocket("/ws/stt")
async def speech_to_text_endpoint(websocket: WebSocket) -> None:
    """Handle STT: local faster-whisper if available, else Deepgram proxy."""
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info("STT_CLIENT_CONNECTED client=%s", client_host)

    try:
        await websocket.accept()
    except Exception as e:
        logger.exception("STT_CLIENT_ACCEPT_FAILED client=%s error=%s", client_host, e)
        return

    from inferr.stt import is_faster_whisper_available

    if is_faster_whisper_available():
        await _handle_local_stt(websocket, client_host)
    else:
        await _handle_deepgram_stt(websocket, client_host)


async def _handle_local_stt(websocket: WebSocket, client_host: str) -> None:
    """Run faster-whisper on incoming PCM16 audio and stream results back."""
    from inferr.stt import get_shared_session
    cfg = get_config()

    try:
        model_size = getattr(cfg, "whisper_model", "small")
    except AttributeError:
        model_size = "small"

    logger.info("STT_LOCAL_WHISPER client=%s model=%s", client_host, model_size)

    try:
        session = await get_shared_session(model_size=model_size)
    except Exception as exc:
        logger.exception("STT_WHISPER_LOAD_FAILED error=%s", exc)
        await websocket.send_json({"type": "error", "message": f"STT init failed: {exc}"})
        await websocket.close(code=1011)
        return

    # Queue to bridge WebSocket receive → async generator
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)

    async def receive_audio() -> None:
        try:
            while True:
                message = await websocket.receive()
                msg_type = message.get("type")
                if msg_type == "websocket.disconnect":
                    logger.info("STT_CLIENT_DISCONNECTED client=%s", client_host)
                    await audio_queue.put(None)  # sentinel
                    return
                payload_bytes = message.get("bytes")
                if isinstance(payload_bytes, bytes):
                    try:
                        audio_queue.put_nowait(payload_bytes)
                    except asyncio.QueueFull:
                        logger.warning("STT_QUEUE_FULL dropping chunk client=%s", client_host)
        except WebSocketDisconnect:
            await audio_queue.put(None)
        except Exception as exc:
            logger.exception("STT_RECEIVE_FAILED client=%s error=%s", client_host, exc)
            await audio_queue.put(None)

    async def audio_chunks():
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                return
            yield chunk

    async def transcribe_and_send() -> None:
        try:
            async for result in session.process_audio_stream(audio_chunks()):
                await websocket.send_text(json.dumps(result))
                transcript = (
                    result.get("channel", {})
                    .get("alternatives", [{}])[0]
                    .get("transcript", "")
                )
                logger.info("STT_LOCAL_TRANSCRIPT client=%s text=%r", client_host, transcript[:80])
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.exception("STT_TRANSCRIBE_SEND_FAILED client=%s error=%s", client_host, exc)

    receive_task = asyncio.create_task(receive_audio())
    transcribe_task = asyncio.create_task(transcribe_and_send())

    try:
        done, pending = await asyncio.wait(
            {receive_task, transcribe_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except asyncio.CancelledError:
        pass
    except WebSocketDisconnect:
        pass


async def _handle_deepgram_stt(websocket: WebSocket, client_host: str) -> None:
    """Proxy browser PCM16 audio to Deepgram when local STT unavailable."""
    cfg = get_config()
    if not cfg.deepgram.api_key:
        logger.error("STT_DEEPGRAM_NOT_CONFIGURED client=%s", client_host)
        await websocket.send_json(
            {"type": "error", "message": "Deepgram is not configured and faster-whisper is not installed"}
        )
        await websocket.close(code=1011)
        return

    import websockets

    stt_sample_rate = 16000
    deepgram_url = (
        "wss://api.deepgram.com/v1/listen"
        f"?model={cfg.deepgram.model}"
        f"&language={cfg.deepgram.language}"
        "&punctuate=true&interim_results=false&endpointing=700"
        f"&encoding=linear16&sample_rate={stt_sample_rate}&channels=1"
    )
    logger.info("DEEPGRAM_CONNECTING url=%s", deepgram_url)

    try:
        deepgram_ws = await websockets.connect(
            deepgram_url,
            additional_headers={"Authorization": f"Token {cfg.deepgram.api_key}"},
        )
        logger.info("DEEPGRAM_CONNECTED client=%s", client_host)

        async def browser_to_deepgram() -> None:
            while True:
                message = await websocket.receive()
                message_type = message.get("type")
                if message_type == "websocket.disconnect":
                    logger.info("STT_CLIENT_DISCONNECTED client=%s", client_host)
                    try:
                        await deepgram_ws.send('{"type":"CloseStream"}')
                    except Exception as e:
                        logger.warning("STT_CLOSE_STREAM_FAILED client=%s error=%s", client_host, e)
                    return
                payload_bytes = message.get("bytes")
                if isinstance(payload_bytes, bytes):
                    await deepgram_ws.send(payload_bytes)
                    continue
                payload_text = message.get("text")
                if isinstance(payload_text, str):
                    await deepgram_ws.send(payload_text)

        async def deepgram_to_browser() -> None:
            try:
                async for message in deepgram_ws:
                    if isinstance(message, str):
                        await websocket.send_text(message)
            except websockets.exceptions.ConnectionClosed as e:
                logger.info("DEEPGRAM_CONNECTION_CLOSED client=%s code=%s", client_host, e.code)
            except Exception as e:
                logger.exception("DEEPGRAM_TO_BROWSER_FAILED client=%s error=%s", client_host, e)

        browser_task = asyncio.create_task(browser_to_deepgram())
        deepgram_task = asyncio.create_task(deepgram_to_browser())

        try:
            done, pending = await asyncio.wait(
                {browser_task, deepgram_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc
        except asyncio.CancelledError:
            pass

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("STT_WEBSOCKET_FAILURE client=%s error=%s", client_host, exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except RuntimeError:
            pass
        await websocket.close(code=1011)


@app.post("/debug/pipeline/llm")
async def debug_pipeline_llm(body: DebugLlmRequest) -> JSONResponse:
    """Isolate LLM: no STT, websocket transcript, or TTS."""
    llm_start = time.monotonic()
    context = _empty_debug_context()
    if session_id and context_assembler:
        context = context_assembler.assemble(session_id, conversation_history)
    tone = resolve_tone(context)
    request = QueryRequest(transcript=body.transcript, context=context)

    try:
        text = await query_llm(request, get_config(), tone=tone)
        stage = pipeline_stages.log_pipeline_stage(
            pipeline_stages.LLM_OK,
            detail=pipeline_stages.timed_detail(
                llm_start,
                transcript=body.transcript,
                response_len=len(text),
                model=get_config().groq.model,
            ),
        )
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "response": text,
                "pipeline_stage": stage,
            },
        )
    except Exception as exc:
        stage = pipeline_stages.log_pipeline_stage(
            pipeline_stages.LLM_FAILED,
            ok=False,
            error=str(exc),
            detail=pipeline_stages.timed_detail(llm_start),
        )
        return JSONResponse(
            status_code=502,
            content={"status": "error", "error": str(exc), "pipeline_stage": stage},
        )


@app.post("/debug/pipeline/tts")
async def debug_pipeline_tts(body: DebugTtsRequest) -> JSONResponse:
    """Isolate TTS: no STT or LLM. Silk requires open /ws in browser."""
    global _main_websocket

    if tts_backend is None:
        stage = pipeline_stages.log_pipeline_stage(
            pipeline_stages.TTS_FAILED,
            ok=False,
            error="no TTS backend",
        )
        return JSONResponse(
            status_code=503,
            content={"status": "error", "pipeline_stage": stage},
        )

    ws = _main_websocket
    tts_start = time.monotonic()
    ok = await _run_tts(ws, body.text, body.tone)
    detail = pipeline_stages.timed_detail(
        tts_start,
        backend=tts_backend.name(),
        ws_connected=ws is not None,
    )
    if ok:
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "backend": tts_backend.name(),
                "detail": detail,
            },
        )
    return JSONResponse(
        status_code=502,
        content={
            "status": "error",
            "backend": tts_backend.name(),
            "detail": detail,
            "hint": "Open the Inferr UI so /ws is connected before testing Silk TTS.",
        },
    )


@app.post("/debug/pipeline/assistant")
async def debug_pipeline_assistant(body: DebugAssistantRequest) -> JSONResponse:
    """Isolate assistant chain (LLM + optional TTS) without STT."""
    global _main_websocket

    ws = _main_websocket
    if ws is None:
        # LLM-only path still works over HTTP
        if body.skip_tts:
            llm_start = time.monotonic()
            context = (
                context_assembler.assemble(session_id, conversation_history)
                if session_id and context_assembler
                else _empty_debug_context()
            )
            tone = resolve_tone(context)
            try:
                text = await query_llm(
                    QueryRequest(transcript=body.transcript, context=context),
                    get_config(),
                    tone=tone,
                )
                return JSONResponse(
                    content={
                        "status": "ok",
                        "response": text,
                        "pipeline_stage": pipeline_stages.log_pipeline_stage(
                            pipeline_stages.LLM_OK,
                            detail=pipeline_stages.timed_detail(llm_start),
                        ),
                    }
                )
            except Exception as exc:
                return JSONResponse(
                    status_code=502,
                    content={"status": "error", "error": str(exc)},
                )

        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": "No /ws client connected. Open the UI for full assistant+TTS test.",
                "hint": "Use window.inferrDebug.invokeAssistant(text) in the browser console.",
            },
        )

    await _handle_transcript(ws, body.transcript, source="debug_http")
    return JSONResponse(
        content={
            "status": "ok",
            "message": "Assistant pipeline invoked on active websocket client.",
        },
    )


@app.get("/debug/pipeline/status")
async def debug_pipeline_status() -> dict[str, object]:
    """Which pipeline boundaries are currently reachable."""
    cfg = get_config()
    return {
        "session_active": session_id is not None,
        "ws_connected": _main_websocket is not None,
        "tts_backend": tts_backend.name() if tts_backend else None,
        "groq_configured": bool(cfg.groq.api_key),
        "deepgram_configured": bool(cfg.deepgram.api_key),
        "stages": [
            pipeline_stages.STT_OK,
            pipeline_stages.TRANSCRIPT_FINALIZED,
            pipeline_stages.LLM_OK,
            pipeline_stages.TTS_OK,
            pipeline_stages.PLAYBACK_STARTED,
            pipeline_stages.PLAYBACK_COMPLETED,
        ],
    }


# ---------------------------------------------------------------------------
# Main conversational WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket transcript streaming and LLM responses."""
    global _main_websocket, last_activity, _event_loop
    await websocket.accept()
    _main_websocket = websocket

    # Capture event loop reference for wake word thread bridge
    _event_loop = asyncio.get_running_loop()
    _ws_clients.add(websocket)

    debug_logger.log_stage("websocket_connected", {"client": websocket.client.host if websocket.client else "unknown"})

    try:
        if isinstance(tts_backend, SilkTTSBackend):
            tts_backend.set_ws_connection(websocket)
        while True:
            data = await websocket.receive_text()
            try:
                message = WebSocketMessage.model_validate_json(data)
            except ValidationError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid message payload"}
                )
                continue

            if message.type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if message.type == "transcript":
                await _handle_transcript(
                    websocket, message.payload, source="websocket"
                )


    except WebSocketDisconnect:
        debug_logger.log_stage("websocket_disconnected", {"reason": "client disconnected"})
        logger.info("WebSocket disconnected.")
    except Exception as exc:
        debug_logger.log_stage("websocket_error", {"error": str(exc)})
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        if _main_websocket is websocket:
            _main_websocket = None
        _ws_clients.discard(websocket)


# ---------------------------------------------------------------------------
# Shell command + output capture
# ---------------------------------------------------------------------------

class ShellOutputCapture(BaseModel):
    command: str
    exit_code: int = 0
    output: str = ""       # Combined stdout + stderr (may be truncated)
    output_lines: int = 0  # Total lines before truncation


@app.post("/capture/command")
async def capture_command(payload: ShellCommandCapture) -> dict[str, str]:
    """Receive shell commands from the shell integration plugin."""
    global _shell_command_buffer
    entry = payload.command
    if payload.exit_code != 0:
        entry = f"{payload.command}  [exit {payload.exit_code}]"
    _shell_command_buffer.append(entry)
    if len(_shell_command_buffer) > _SHELL_BUFFER_MAX:
        _shell_command_buffer = _shell_command_buffer[-_SHELL_BUFFER_MAX:]

    if context_assembler is not None:
        context_assembler.inject_command(entry)
        context_assembler.hint_active_file_from_command(payload.command)

    # Persist to SQLite
    if session_id:
        asyncio.create_task(
            db.save_command(session_id, payload.command, payload.exit_code)
        )

    return {"status": "ok"}


@app.post("/capture/output")
async def capture_output(payload: ShellOutputCapture) -> dict[str, str]:
    """Receive real stdout/stderr from the shell integration plugin.

    This gives the assistant visibility into actual command output:
    stack traces, compiler errors, test failures, etc.
    """
    global _shell_command_buffer

    # Build a command entry with the real output
    entry = payload.command
    if payload.exit_code != 0:
        entry = f"{payload.command}  [exit {payload.exit_code}]"

    # Inject command first (so context ordering is preserved)
    _shell_command_buffer.append(entry)
    if len(_shell_command_buffer) > _SHELL_BUFFER_MAX:
        _shell_command_buffer = _shell_command_buffer[-_SHELL_BUFFER_MAX:]

    if context_assembler is not None:
        context_assembler.inject_command(entry)
        context_assembler.hint_active_file_from_command(payload.command)

        # Inject real output lines into terminal buffer for error detection
        if payload.output:
            output_lines = payload.output.splitlines()
            # Limit to 100 lines max to prevent buffer flooding
            for line in output_lines[-100:]:
                if line.strip():
                    context_assembler.inject_command(line)

    # Persist command + output
    if session_id:
        asyncio.create_task(
            db.save_command(session_id, payload.command, payload.exit_code, payload.output[:4000])
        )

    logger.info(
        "SHELL_OUTPUT_CAPTURED cmd=%r exit=%d lines=%d",
        payload.command[:60],
        payload.exit_code,
        payload.output_lines,
    )

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Active file context push (for editor plugins)
# ---------------------------------------------------------------------------

class ActiveFilePayload(BaseModel):
    path: str
    content: str = ""
    language: str = ""


@app.post("/context/file")
async def push_active_file(payload: ActiveFilePayload) -> dict[str, str]:
    """Editor plugins push the active buffer here for ambient awareness.

    Accepts unsaved state — the developer's current focus.
    """
    if context_assembler is None:
        return {"status": "no_session"}

    context_assembler.push_active_file(
        path=payload.path,
        content=payload.content,
        language=payload.language,
    )
    logger.info(
        "ACTIVE_FILE_PUSHED path=%s language=%s content_len=%d",
        payload.path,
        payload.language,
        len(payload.content),
    )
    return {"status": "ok"}


@app.get("/context/file")
async def get_active_file_context(path: str = "") -> JSONResponse:
    """Read current active file context (for debugging/extension polling)."""
    if context_assembler is None:
        return JSONResponse(status_code=404, content={"error": "No active session"})
    active = context_assembler._file_watcher.get_active_file()
    if active is None:
        return JSONResponse(status_code=204, content={})
    return JSONResponse(
        status_code=200,
        content={
            "path": active.path,
            "language": active.language,
            "content_preview": active.content[:500],
        },
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """HTTP fallback endpoint for transcript queries."""
    global last_activity

    tone = resolve_tone(request.context)
    response_text = await query_llm(request, get_config(), tone=tone)
    last_activity = datetime.now(timezone.utc)
    return QueryResponse(text=response_text, session_id=request.context.session_id)
