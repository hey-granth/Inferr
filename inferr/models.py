from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


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


class QueryRequest(BaseModel):
    transcript: str
    context: ContextObject


class QueryResponse(BaseModel):
    text: str
    session_id: str


class WebSocketMessage(BaseModel):
    type: Literal["transcript", "ping"]
    payload: str
