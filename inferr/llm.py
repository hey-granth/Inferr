from __future__ import annotations

from typing import Literal, cast

from anthropic import APIError, AsyncAnthropic
from anthropic.types import MessageParam

from inferr.config import Config
from inferr.models import QueryRequest


def build_system_prompt(language: str) -> str:
    if language == "hinglish":
        return (
            "You are a senior developer pair programming partner. Respond in hinglish "
            "(Hindi-English code-switching). Use developer-native phrasing like bhai, "
            "dekh, chal, and yaar. For routine questions, answer in 3-4 sentences; "
            "for complex errors, use 5-6 sentences. Be direct and specific about errors. "
            "Always use the injected context (terminal buffer, shell history, active file, "
            "flagged errors) and prioritize flagged errors when present. If you need to list "
            "items, convert bullets into spoken form (e.g., 'teen cheezein hain...')."
        )

    return (
        "You are a senior developer pair programming partner. Respond in plain English. "
        "For routine questions, answer in 3-4 sentences; for complex errors, use 5-6 sentences. "
        "Be direct and specific about errors. Always use the injected context (terminal buffer, "
        "shell history, active file, flagged errors) and prioritize flagged errors when present. "
        "If you need to list items, convert bullets into spoken form (e.g., 'three things...')."
    )


async def query_llm(request: QueryRequest, config: Config) -> str:
    client = AsyncAnthropic()
    system_prompt = build_system_prompt(config.language)

    context_json = request.context.model_dump_json()
    user_content = f"<context>\n{context_json}\n</context>\n\n{request.transcript}"

    def _as_message(role: Literal["user", "assistant"], content: str) -> MessageParam:
        return cast(MessageParam, {"role": role, "content": content})

    history_messages: list[MessageParam] = [
        _as_message(turn.role, turn.content)
        for turn in request.context.conversation_history[-3:]
    ]
    messages: list[MessageParam] = history_messages + [
        _as_message("user", user_content)
    ]

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system_prompt,
            messages=messages,
        )
    except APIError as exc:
        raise RuntimeError(f"Anthropic API error: {exc}") from exc

    content_blocks = getattr(response, "content", [])
    if not content_blocks:
        return ""
    first_block = content_blocks[0]
    text = getattr(first_block, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(first_block, str):
        return first_block
    return ""
