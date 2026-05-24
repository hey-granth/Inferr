from __future__ import annotations

import asyncio
from typing import Any
from pathlib import Path

from google import genai
from google.genai import types as genai_types

from inferr.config import Config
from inferr.models import QueryRequest


# Keywords indicating high-signal terminal lines that should be prioritized
_HIGH_SIGNAL_PATTERNS = (
    "traceback",
    "error",
    "exception",
    "failed",
    "warning",
    "panic",
    "conflict",
    " 4",   # catches 4xx status codes
    " 5",   # catches 5xx status codes
    "uvicorn",
    "assert",
    "syntaxerror",
    "typeerror",
    "valueerror",
    "keyerror",
    "importerror",
    "attributeerror",
    "runtimeerror",
    "oserror",
)


def _is_high_signal(line: str) -> bool:
    lower = line.lower()
    return any(kw in lower for kw in _HIGH_SIGNAL_PATTERNS)


def _rank_terminal_lines(lines: list[str], limit: int) -> list[str]:
    """Return up to `limit` lines, prioritising high-signal diagnostics."""
    high = [l for l in lines if l.strip() and _is_high_signal(l)]
    low = [l for l in lines if l.strip() and not _is_high_signal(l)]
    # High-signal lines first, most recent last, capped at limit
    combined = high[-limit:] + low[-(max(0, limit - len(high[-limit:]))):]  # noqa: E501
    # Preserve approximate temporal order within each bucket, re-merge
    merged = sorted(
        set(combined),
        key=lambda l: (0 if _is_high_signal(l) else 1, lines.index(l) if l in lines else 0)
    )
    return merged[:limit]


def build_system_prompt(language: str, tone: str = "neutral") -> str:
    # Operational constraints apply regardless of tone/language.
    # The assistant is a sharp, silent senior engineer — not a chatbot.
    operational_rules = (
        "Rules (non-negotiable):\n"
        "- Do not roleplay. Do not greet. Do not introduce yourself.\n"
        "- Do not brainstorm unless the developer explicitly asks.\n"
        "- Do not speculate beyond the observed runtime context.\n"
        "- Do not give educational explanations unless explicitly requested.\n"
        "- Do not use performative enthusiasm, filler phrases, or padding.\n"
        "- If evidence is weak or context is insufficient, say so in one short sentence.\n"
        "- Prioritize debugging and unblocking the developer's current coding task.\n"
        "- Answer in 2–4 short sentences. Diagnosis first. Concrete next step second.\n"
        "- Complete your response fully. Do not stop mid-sentence.\n"
        "- Prefer one complete concise thought over multiple fragmented ones.\n"
        "- Ground every answer in the observed terminal output, file, and error context.\n"
        "- If no relevant context is observed, answer directly from the question only."
    )

    if language == "hinglish":
        style = (
            "Style: natural technical Hinglish. "
            "Code-switch naturally — do not force Hindi or slang. "
            "If an answer works better in plain English, use plain English. "
            "Never use Devanagari or non-Latin script. Romanise Hindi words."
        )
    else:
        style = "Style: plain, direct English. No filler."

    urgency_note = ""
    if tone == "urgent":
        urgency_note = "\nContext: errors are flagged. Lead with the specific error and line if visible."

    return f"{operational_rules}\n\n{style}{urgency_note}"


def _shorten_line(value: str, max_len: int = 120) -> str:
    text = value.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _summarize_context(request: QueryRequest) -> str:
    context = request.context

    # Rank terminal lines: high-signal diagnostics first, then recency
    terminal_events = [
        _shorten_line(line)
        for line in _rank_terminal_lines(context.terminal_buffer, limit=8)
    ]
    shell_commands = [
        _shorten_line(line) for line in context.shell_history if line.strip()
    ][-4:]
    active_file = (
        Path(context.active_file.path).name
        if context.active_file is not None
        else None
    )

    real_errors = [e for e in context.flagged_errors if e.type != "marker"]
    error_summaries = [_shorten_line(e.summary) for e in real_errors[:3]]

    repeated_error_note = ""
    if error_summaries:
        all_terminal = "\n".join(context.terminal_buffer).lower()
        first_error = error_summaries[0].lower()
        if first_error and all_terminal.count(first_error) >= 2:
            repeated_error_note = "Note: same error appears repeatedly."

    sections: list[str] = []

    if error_summaries:
        sections.append("Flagged errors:")
        sections.extend([f"- {line}" for line in error_summaries])
        if repeated_error_note:
            sections.append(f"- {repeated_error_note}")
        sections.append("")

    sections.append("Terminal (high-signal first):")
    if terminal_events:
        sections.extend([f"- {line}" for line in terminal_events])
    else:
        sections.append("- none")

    if shell_commands:
        sections.append("\nRecent commands:")
        sections.extend([f"- {line}" for line in shell_commands])

    if active_file:
        sections.append(f"\nActive file: {active_file}")

    sections.append(f'\nUser said: "{request.transcript}"')
    return "\n".join(sections)


async def query_llm(
    request: QueryRequest, config: Config, tone: str = "neutral"
) -> str:
    system_prompt = build_system_prompt(config.language, tone=tone)
    user_content = _summarize_context(request)

    history: list[genai_types.Content] = []
    for turn in request.context.conversation_history[-3:]:
        role = "model" if turn.role == "assistant" else "user"
        history.append(
            genai_types.Content(
                role=role,
                parts=[genai_types.Part(text=turn.content)],
            )
        )

    history.append(
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=f"<context>\n{user_content}\n</context>")],
        )
    )

    generate_config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=220,
        temperature=0.2,
    )

    keys_to_try = (
        config.gemini.api_keys if config.gemini.api_keys else [config.gemini.api_key]
    )
    last_exc: Exception | None = None

    for api_key in keys_to_try:
        if not api_key:
            continue
        client = genai.Client(api_key=api_key)
        try:
            client_any: Any = client
            if hasattr(client_any, "models"):
                response = await asyncio.to_thread(
                    client_any.models.generate_content,
                    model=config.gemini.model,
                    contents=history,
                    config=generate_config,
                )
            else:
                response = await client_any.aio.models.generate_content(
                    model=config.gemini.model,
                    contents=history,
                    config=generate_config,
                )
            if not response.candidates:
                return ""
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                return ""
            return candidate.content.parts[0].text or ""
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if (
                "429" in exc_str
                or "resource_exhausted" in exc_str
                or "quota" in exc_str
            ):
                continue
            raise RuntimeError(f"Gemini API error: {exc}") from exc

    if len(keys_to_try) <= 1 and last_exc is not None:
        raise RuntimeError(f"Gemini API error: {last_exc}") from last_exc

    raise RuntimeError(
        f"All Gemini API keys exhausted or rate limited. Last error: {last_exc}"
    )
