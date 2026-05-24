from __future__ import annotations

import asyncio
from pathlib import Path

from groq import Groq

from inferr.config import Config
from inferr.context.terminal import sanitize_terminal_line
from inferr.models import QueryRequest
from inferr.debug import debug_logger


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
        "- Default to about 25 spoken words (1–3 short sentences) unless more detail is required.\n"
        "- Diagnosis first. Concrete next step second. Complete every sentence naturally.\n"
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

    prompt = f"{operational_rules}\n\n{style}{urgency_note}"
    
    debug_logger.log_stage("system_prompt_assembly", {
        "language": language,
        "tone": tone,
        "final_prompt": prompt
    })
    
    return prompt


def _sanitize_context_line(value: str) -> str:
    return sanitize_terminal_line(value)


def _shorten_line(value: str, max_len: int = 120) -> str:
    text = _sanitize_context_line(value)
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
    
    # ADDED logic to include repository details if available
    repo_name = context.git_repo if hasattr(context, 'git_repo') else None

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
        sections.append("Recent commands:")
        sections.extend([f"- {line}" for line in shell_commands])

    if context.active_file is not None:
        af = context.active_file
        sections.append(f"Active file: {active_file} ({af.language})")
        preview = [_sanitize_context_line(line) for line in af.content.splitlines()[:12]]
        preview = [line for line in preview if line]
        if preview:
            sections.append("Active file snippet:")
            sections.extend([f"  {line}" for line in preview])

    if repo_name:
        sections.append(f"Repository: {repo_name}")

    sections.append(f'User said: "{request.transcript}"')
    
    final_summary = "\n".join(sections)
    
    debug_logger.log_stage("context_summarization", {
        "active_file_tracked": bool(active_file),
        "terminal_lines_included": len(terminal_events),
        "error_count": len(error_summaries),
        "final_summary": final_summary
    })
    
    return final_summary


async def query_llm(
    request: QueryRequest, config: Config, tone: str = "neutral"
) -> str:
    system_prompt = build_system_prompt(config.language, tone=tone)
    user_content = _summarize_context(request)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    for turn in request.context.conversation_history[-3:]:
        role = "assistant" if turn.role == "assistant" else "user"
        messages.append({"role": role, "content": turn.content})

    messages.append({
        "role": "user",
        "content": f"<context>\n{user_content}\n</context>",
    })

    keys_to_try = (
        config.groq.api_keys if config.groq.api_keys else [config.groq.api_key]
    )
    last_exc: Exception | None = None

    debug_logger.log_stage("llm_request_start", {
        "model": config.groq.model,
        "temperature": 0.2,
        "max_output_tokens": 200,
        "user_content": user_content,
    })

    for api_key in keys_to_try:
        if not api_key:
            continue
        client = Groq(api_key=api_key)
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=config.groq.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=200,
                temperature=0.2,
            )

            raw_response_text = ""
            finish_reason = None
            if response.choices:
                choice = response.choices[0]
                finish_reason = choice.finish_reason
                raw_response_text = choice.message.content or ""

            clean_end = finish_reason == "stop"

            debug_logger.log_stage("llm_raw_response", {
                "model": config.groq.model,
                "response_length": len(raw_response_text),
                "finish_reason": str(finish_reason),
                "clean_end": clean_end,
                "raw_response": raw_response_text,
            })

            return raw_response_text

        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if (
                "429" in exc_str
                or "rate_limit" in exc_str
                or "quota" in exc_str
                or "rate limit" in exc_str
            ):
                continue

            debug_logger.log_stage("llm_request_failed", {
                "error": str(exc),
                "model": config.groq.model,
            })
            raise RuntimeError(f"Groq API error: {exc}") from exc

    if len(keys_to_try) <= 1 and last_exc is not None:
        raise RuntimeError(f"Groq API error: {last_exc}") from last_exc

    raise RuntimeError(
        f"All Groq API keys exhausted or rate limited. Last error: {last_exc}"
    )
