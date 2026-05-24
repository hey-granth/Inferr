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

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

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

logger = logging.getLogger("inferr.server")

session_id: str | None = None
context_assembler: ContextAssembler | None = None
conversation_history: list[ConversationTurn] = []
tts_backend: TTSBackend | None = None
last_activity: datetime | None = None

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
        if context_assembler is not None:
            context_assembler.stop()


app = FastAPI(lifespan=lifespan)


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

    config = get_config()
    if context_assembler is not None:
        context_assembler.stop()

    context_assembler = ContextAssembler(config)
    context_assembler.start()
    tts_backend = get_tts_backend(config)
    session_id = str(uuid.uuid4())
    conversation_history = []

    if os.environ.get("INFERR_NO_BROWSER") != "1":
        try:
            subprocess.Popen(["google-chrome", f"http://{config.host}:{config.port}"])
        except (OSError, FileNotFoundError):
            logger.info("Unable to open browser companion UI.")

    return {"session_id": session_id, "status": "started"}


@app.post("/session/stop")
async def stop_session() -> dict[str, str]:
    """Stop the current Inferr session."""
    global session_id, context_assembler, conversation_history, tts_backend

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
    return {
        "status": "ok",
        "session_active": session_id is not None,
        "last_activity": last_activity.isoformat() if last_activity else None,
        "tts_backend": tts_backend.name() if tts_backend is not None else None,
    }


@app.get("/config/browser")
async def browser_config() -> dict[str, object]:
    """Return config the browser needs at runtime. Only non-secret values."""
    cfg = get_config()
    return {
        "deepgram_enabled": bool(cfg.deepgram.api_key),
        "tts_backend": tts_backend.name() if tts_backend else "browser",
        "stt_sample_rate": 16000,
        "stt_encoding": "linear16",
        "stt_channels": 1,
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


@app.websocket("/ws/stt")
async def speech_to_text_endpoint(websocket: WebSocket) -> None:
    """Proxy browser microphone audio to Deepgram without exposing API keys."""
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info("STT_CLIENT_CONNECTED client=%s", client_host)

    try:
        await websocket.accept()
    except Exception as e:
        logger.exception("STT_CLIENT_ACCEPT_FAILED client=%s error=%s", client_host, e)
        return

    cfg = get_config()
    if not cfg.deepgram.api_key:
        logger.error("STT_DEEPGRAM_NOT_CONFIGURED client=%s", client_host)
        await websocket.send_json(
            {"type": "error", "message": "Deepgram is not configured"}
        )
        await websocket.close(code=1011)
        return

    import websockets

    # Browser streams raw PCM16 (linear16) from Web Audio — not MediaRecorder WebM.
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
                logger.debug("STT_CLIENT_MESSAGE client=%s type=%s", client_host, message_type)

                if message_type == "websocket.disconnect":
                    logger.info("STT_CLIENT_DISCONNECTED client=%s", client_host)
                    try:
                        await deepgram_ws.send('{"type":"CloseStream"}')
                        logger.debug("STT_CLOSE_STREAM_SENT client=%s", client_host)
                    except Exception as e:
                        logger.warning("STT_CLOSE_STREAM_FAILED client=%s error=%s", client_host, e)
                    return

                payload_bytes = message.get("bytes")
                if isinstance(payload_bytes, bytes):
                    chunk_size = len(payload_bytes)
                    first_bytes_hex = payload_bytes[:16].hex() if payload_bytes else ""
                    logger.info(
                        "STT_CHUNK_META client=%s size=%d first_bytes=%s",
                        client_host, chunk_size, first_bytes_hex
                    )
                    await deepgram_ws.send(payload_bytes)
                    logger.debug("STT_CHUNK_SENT_TO_DEEPGRAM client=%s size=%d", client_host, chunk_size)
                    continue

                payload_text = message.get("text")
                if isinstance(payload_text, str):
                    logger.info("STT_TEXT_MESSAGE client=%s text=%s", client_host, payload_text[:100])
                    if payload_text.strip() == '{"type":"KeepAlive"}':
                        logger.debug("STT_KEEPALIVE_RECEIVED client=%s", client_host)
                    await deepgram_ws.send(payload_text)

        async def deepgram_to_browser() -> None:
            try:
                async for message in deepgram_ws:
                    if isinstance(message, str):
                        logger.debug("DEEPGRAM_RAW_MESSAGE client=%s len=%d", client_host, len(message))
                        try:
                            import json
                            msg_data = json.loads(message)
                            msg_type = msg_data.get("type", "unknown")
                            logger.info("DEEPGRAM_MESSAGE client=%s type=%s", client_host, msg_type)

                            if msg_type == "Results":
                                is_final = msg_data.get("is_final", False)
                                transcript = msg_data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                                if transcript:
                                    logger.info(
                                        "DEEPGRAM_TRANSCRIPT client=%s is_final=%s transcript=%s",
                                        client_host, is_final, transcript[:100]
                                    )
                            elif msg_type == "error":
                                error_msg = msg_data.get("error", "unknown")
                                logger.error("DEEPGRAM_ERROR client=%s error=%s", client_host, error_msg)
                        except json.JSONDecodeError:
                            logger.warning("DEEPGRAM_INVALID_JSON client=%s len=%d", client_host, len(message))

                        await websocket.send_text(message)
            except websockets.exceptions.ConnectionClosed as e:
                logger.info("DEEPGRAM_CONNECTION_CLOSED client=%s code=%s reason=%s",
                           client_host, e.code, e.reason)
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
                logger.debug("STT_TASK_CANCELLED client=%s task=%s", client_host, task)
            for task in done:
                exc = task.exception()
                if exc is not None:
                    logger.exception("STT_TASK_FAILED client=%s error=%s", client_host, exc)
                    raise exc
        except asyncio.CancelledError:
            logger.info("STT_TASKS_CANCELLED client=%s", client_host)

    except WebSocketDisconnect:
        logger.info("STT_WEBSOCKET_DISCONNECTED client=%s", client_host)
    except Exception as exc:
        logger.exception("STT_WEBSOCKET_FAILURE client=%s error=%s", client_host, exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except RuntimeError:
            logger.warning("STT_ERROR_SEND_FAILED client=%s", client_host)
            return
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket transcript streaming and LLM responses."""
    global _main_websocket
    await websocket.accept()
    _main_websocket = websocket

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
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1011)
    finally:
        if _main_websocket is websocket:
            _main_websocket = None


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

    # Also inject into active session's terminal buffer for error detection
    if context_assembler is not None:
        context_assembler.inject_command(entry)
        context_assembler.hint_active_file_from_command(payload.command)

    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """HTTP fallback endpoint for transcript queries."""
    global last_activity

    tone = resolve_tone(request.context)
    response_text = await query_llm(request, get_config(), tone=tone)
    last_activity = datetime.now(timezone.utc)
    return QueryResponse(text=response_text, session_id=request.context.session_id)
