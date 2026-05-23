from __future__ import annotations

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
    ConversationTurn,
    QueryRequest,
    QueryResponse,
    WebSocketMessage,
)
from inferr.tts import speak_pyttsx3

logger = logging.getLogger("inferr.server")

session_id: str | None = None
context_assembler: ContextAssembler | None = None
conversation_history: list[ConversationTurn] = []
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
    global session_id, context_assembler, conversation_history

    config = get_config()
    if context_assembler is not None:
        context_assembler.stop()

    context_assembler = ContextAssembler(config)
    context_assembler.start()
    session_id = str(uuid.uuid4())
    conversation_history = []

    if os.environ.get("INFERR_NO_BROWSER") != "1":
        try:
            subprocess.Popen(
                ["xdg-open", f"http://{config.host}:{config.port}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, FileNotFoundError):
            logger.info("Unable to open browser companion UI.")

    return {"session_id": session_id, "status": "started"}


@app.post("/session/stop")
async def stop_session() -> dict[str, str]:
    """Stop the current Inferr session."""
    global session_id, context_assembler, conversation_history

    if context_assembler is not None:
        context_assembler.stop()
    context_assembler = None
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
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket transcript streaming and LLM responses."""
    global last_activity
    await websocket.accept()

    try:
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
                request = QueryRequest(transcript=message.payload, context=context)
                response_text = await query_llm(request, get_config())

                conversation_history.append(
                    ConversationTurn(role="user", content=message.payload)
                )
                conversation_history.append(
                    ConversationTurn(role="assistant", content=response_text)
                )
                if len(conversation_history) > 6:
                    conversation_history[:] = conversation_history[-6:]

                last_activity = datetime.now(timezone.utc)

                await websocket.send_json({"type": "response", "text": response_text})

                try:
                    thread = threading.Thread(
                        target=speak_pyttsx3, args=(response_text,), daemon=True
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

    response_text = await query_llm(request, get_config())
    last_activity = datetime.now(timezone.utc)
    return QueryResponse(text=response_text, session_id=request.context.session_id)
