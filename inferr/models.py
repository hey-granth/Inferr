from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ActiveFile(BaseModel):
    path: str
    language: str
    content: str
    last_modified: datetime


class FlaggedError(BaseModel):
    type: str
    summary: str
    raw: str


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ContextObject(BaseModel):
    terminal_buffer: list[str]
    shell_history: list[str]
    active_file: ActiveFile | None
    flagged_errors: list[FlaggedError]
    conversation_history: list[ConversationTurn]
    session_id: str
    timestamp: datetime
    git_repo: str | None = None


class SilkConfig(BaseModel):
    api_url: str = "https://silk-api.rumik.ai"
    api_key: str = ""
    voice_id: str = "muga"
    stream: bool = True


class GeminiConfig(BaseModel):
    api_key: str = ""
    api_keys: list[str] = Field(default_factory=list)
    model: str = "gemini-2.0-flash"


class GroqConfig(BaseModel):
    api_key: str = ""
    api_keys: list[str] = Field(default_factory=list)
    model: str = "llama-3.3-70b-versatile"


class DeepgramConfig(BaseModel):
    api_key: str = ""
    model: str = "nova-2"
    language: str = "en-IN"


class OllamaConfig(BaseModel):
    """Offline LLM fallback via local ollama server."""

    enabled: bool = True  # Attempt ollama when Gemini fails
    model: str = "llama3.2:3b"
    url: str = "http://localhost:11434"
    timeout_seconds: float = 20.0


class WakeWordConfig(BaseModel):
    """openwakeword-based always-on wake phrase detector."""

    enabled: bool = False  # Opt-in (requires mic on server + openwakeword)
    model_name: str = "alexa"  # Pre-trained model; swap for custom .onnx path
    model_path: str = ""  # Full path to custom .onnx; overrides model_name
    threshold: float = 0.5
    cooldown_seconds: float = 3.0


class QueryRequest(BaseModel):
    transcript: str
    context: ContextObject


class QueryResponse(BaseModel):
    text: str
    session_id: str


class WebSocketMessage(BaseModel):
    type: Literal["transcript", "ping", "wake_word"]
    payload: str = ""


class ShellCommandCapture(BaseModel):
    command: str
    exit_code: int = 0


class DebugLlmRequest(BaseModel):
    transcript: str = "hello"


class DebugTtsRequest(BaseModel):
    text: str = "hello from inferr"
    tone: str = "neutral"


class DebugAssistantRequest(BaseModel):
    transcript: str = "hello"
    skip_tts: bool = False
