from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import subprocess
import threading
import uuid

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
    QueryRequest,
    QueryResponse,
    WebSocketMessage,
)
from inferr.tts import SilkTTSBackend, TTSBackend, get_tts_backend

logger = logging.getLogger("inferr.server")

session_id: str | None = None
context_assembler: ContextAssembler | None = None
conversation_history: list[ConversationTurn] = []
tts_backend: TTSBackend | None = None
last_activity: datetime | None = None

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
    }


def resolve_tone(context: ContextObject) -> str:
    """
    Returns 'urgent' if any flagged errors are present.
    Returns 'warm' if conversation_history is empty (first query of the session).
    Returns 'neutral' otherwise.
    """
    if any(error.type != "marker" for error in context.flagged_errors):
        return "urgent"
    if not context.conversation_history:
        return "warm"
    return "neutral"


@app.websocket("/ws/stt")
async def speech_to_text_endpoint(websocket: WebSocket) -> None:
    """Proxy browser microphone audio to Deepgram without exposing API keys."""
    await websocket.accept()

    cfg = get_config()
    if not cfg.deepgram.api_key:
        await websocket.send_json(
            {"type": "error", "message": "Deepgram is not configured"}
        )
        await websocket.close(code=1011)
        return

    import websockets

    deepgram_url = (
        "wss://api.deepgram.com/v1/listen"
        f"?model={cfg.deepgram.model}"
        f"&language={cfg.deepgram.language}"
        "&punctuate=true&interim_results=false&endpointing=500"
    )

    try:
        async with websockets.connect(
            deepgram_url,
            additional_headers={"Authorization": f"Token {cfg.deepgram.api_key}"},
        ) as deepgram_ws:

            async def browser_to_deepgram() -> None:
                while True:
                    message = await websocket.receive()
                    message_type = message.get("type")
                    if message_type == "websocket.disconnect":
                        try:
                            await deepgram_ws.send('{"type":"CloseStream"}')
                        except Exception:
                            return
                        return

                    payload_bytes = message.get("bytes")
                    if isinstance(payload_bytes, bytes):
                        await deepgram_ws.send(payload_bytes)
                        continue

                    payload_text = message.get("text")
                    if isinstance(payload_text, str):
                        await deepgram_ws.send(payload_text)

            async def deepgram_to_browser() -> None:
                async for message in deepgram_ws:
                    if isinstance(message, str):
                        await websocket.send_text(message)

            browser_task = asyncio.create_task(browser_to_deepgram())
            deepgram_task = asyncio.create_task(deepgram_to_browser())
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
    except WebSocketDisconnect:
        logger.info("STT WebSocket disconnected.")
    except Exception as exc:
        logger.exception("Deepgram STT proxy failed: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except RuntimeError:
            return
        await websocket.close(code=1011)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket transcript streaming and LLM responses."""
    global last_activity
    await websocket.accept()

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
                if session_id is None or context_assembler is None:
                    await websocket.send_json(
                        {"type": "error", "message": "No active session"}
                    )
                    continue

                context = context_assembler.assemble(session_id, conversation_history)
                tone = resolve_tone(context)
                request = QueryRequest(transcript=message.payload, context=context)
                response_text = await query_llm(request, get_config(), tone=tone)

                conversation_history.append(
                    ConversationTurn(role="user", content=message.payload)
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
                backend_name = (
                    tts_backend.name() if tts_backend is not None else "browser"
                )
                await websocket.send_json(
                    {
                        "type": "response",
                        "text": response_text,
                        "has_errors": has_errors,
                        "tone": tone,
                        "tts_backend": backend_name,
                    }
                )

                if tts_backend is not None and tts_backend.name() == "silk":
                    try:
                        tts_backend.speak(response_text, tone=tone)
                    except NotImplementedError as exc:
                        logger.info("Silk backend not active yet: %s", exc)
                    except Exception as exc:
                        logger.exception("Silk TTS failed: %s", exc)
                elif tts_backend is not None:
                    try:
                        thread = threading.Thread(
                            target=tts_backend.speak,
                            args=(response_text, tone),
                            daemon=True,
                        )
                        thread.start()
                    except RuntimeError:
                        logger.info("TTS thread failed to start.")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1011)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """HTTP fallback endpoint for transcript queries."""
    global last_activity

    tone = resolve_tone(request.context)
    response_text = await query_llm(request, get_config(), tone=tone)
    last_activity = datetime.now(timezone.utc)
    return QueryResponse(text=response_text, session_id=request.context.session_id)
