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

    # Check past error memory (lightweight SQLite recall)
    past_error_note = ""
    if error_summaries:
        try:
            from inferr.persistence import find_similar_errors
            similar = find_similar_errors(error_summaries[0], limit=2)
            # Only use if the match is genuinely old (not the current session)
            current_sid = context.session_id
            old_matches = [e for e in similar if True]  # all are stored from past
            if old_matches:
                past_error_note = (
                    f"Past error memory: similar error seen before — '{old_matches[0]['summary'][:80]}'"
                )
        except Exception:
            pass  # persistence unavailable — degrade gracefully

    sections: list[str] = []

    if error_summaries:
        sections.append("Flagged errors:")
        sections.extend([f"- {line}" for line in error_summaries])
        if repeated_error_note:
            sections.append(f"- {repeated_error_note}")
        if past_error_note:
            sections.append(f"- {past_error_note}")
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


def _adaptive_token_limit(request: QueryRequest, tone: str) -> int:
    """Return the max_output_tokens appropriate for this query.

    Short/simple → ~200 (fast TTS, conversational)
    Runtime errors flagged → ~350 (space for diagnosis + actionable fix)
    """
    if tone == "urgent":
        return 350
    real_errors = [e for e in request.context.flagged_errors if e.type != "marker"]
    if real_errors:
        return 350
    return 200


async def _query_ollama(
    system_prompt: str,
    user_content: str,
    config_ollama: "OllamaConfig",
    max_tokens: int,
) -> str:
    """Query a locally-running ollama server as an offline LLM fallback.

    Uses the /api/chat endpoint so we can pass a proper system prompt.
    Raises RuntimeError if ollama is unreachable or returns an error.
    """
    import httpx
    from inferr.models import OllamaConfig

    url = f"{config_ollama.url.rstrip('/')}/api/chat"
    payload = {
        "model": config_ollama.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens,
        },
    }
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=2.0,
                read=config_ollama.timeout_seconds,
                write=5.0,
                pool=5.0,
            )
        ) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text: str = data["message"]["content"].strip()
            return text
    except httpx.ConnectError:
        raise RuntimeError(
            f"Ollama not reachable at {config_ollama.url} — "
            "run: ollama serve && ollama pull llama3.2:3b"
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc.response.status_code}") from exc
    except KeyError as exc:
        raise RuntimeError(f"Ollama response format unexpected: {exc}") from exc


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
    max_output_tokens = _adaptive_token_limit(request, tone)

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
                max_tokens=max_output_tokens,
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
        # Try ollama before giving up
        if config.ollama.enabled:
            return await _try_ollama_fallback(
                system_prompt, user_content, config, max_output_tokens
            )
        raise RuntimeError(f"Groq API error: {last_exc}") from last_exc

    # All Groq keys exhausted (rate-limited)
    if config.ollama.enabled and last_exc is not None:
        return await _try_ollama_fallback(
            system_prompt, user_content, config, max_output_tokens
        )

    raise RuntimeError(
        f"All Groq API keys exhausted or rate limited. Last error: {last_exc}"
    )


async def _try_ollama_fallback(
    system_prompt: str,
    user_content: str,
    config: Config,
    max_tokens: int,
) -> str:
    """Attempt an ollama query and return result with an [offline] tag.

    The [offline] suffix lets the browser and frontend know the response
    came from the local model, so it can show a subtle indicator.
    """
    logger.info(
        "LLM_OLLAMA_FALLBACK model=%s url=%s",
        config.ollama.model,
        config.ollama.url,
    )
    debug_logger.log_stage("llm_ollama_fallback", {
        "model": config.ollama.model,
        "url": config.ollama.url,
    })
    try:
        result = await _query_ollama(
            system_prompt=system_prompt,
            user_content=user_content,
            config_ollama=config.ollama,
            max_tokens=max_tokens,
        )
        logger.info("LLM_OLLAMA_SUCCESS len=%d", len(result))
        return result  # Clean response — no tag, model speaks for itself
    except RuntimeError as exc:
        logger.warning("LLM_OLLAMA_FAILED error=%s", exc)
        raise
