from __future__ import annotations

import asyncio
from typing import Any

from google import genai
from google.genai import types as genai_types

from inferr.config import Config
from inferr.models import QueryRequest


def build_system_prompt(language: str, tone: str = "neutral") -> str:
    if language == "hinglish":
        base_prompt = (
            "You are a senior developer pair programming partner. Respond in hinglish "
            "(Hindi-English code-switching). Use developer-native phrasing like bhai, "
            "dekh, chal, and yaar. For routine questions, answer in 3-4 sentences; "
            "for complex errors, use 5-6 sentences. Be direct and specific about errors. "
            "Always use the injected context (terminal buffer, shell history, active file, "
            "flagged errors) and prioritize flagged errors when present. If you need to list "
            "items, convert bullets into spoken form (e.g., 'teen cheezein hain...'). "
            "Write all responses in Latin script only. Never use Devanagari or any other "
            "non-Latin script. All Hindi words must be romanised: yaar not यार, "
            "nahi not नहीं, karo not करो, bhai not भाई."
        )
    else:
        base_prompt = (
            "You are a senior developer pair programming partner. Respond in plain English. "
            "For routine questions, answer in 3-4 sentences; for complex errors, use 5-6 sentences. "
            "Be direct and specific about errors. Always use the injected context (terminal buffer, "
            "shell history, active file, flagged errors) and prioritize flagged errors when present. "
            "If you need to list items, convert bullets into spoken form (e.g., 'three things...'). "
            "Write all responses in Latin script only. Never use Devanagari or any other "
            "non-Latin script. Any Hindi words must be romanised: nahi not नहीं, "
            "karo not करो."
        )

    tone_instructions = {
        "urgent": (
            "Errors are flagged. Lead with the error directly. Be specific about "
            "line numbers and fix steps. Tone: direct, slightly urgent, no preamble."
        ),
        "warm": (
            "This is the developer's first query this session. Be welcoming. "
            "One sentence of orientation before the answer."
        ),
        "neutral": "Routine query. Be direct and concise.",
    }
    selected_tone = tone_instructions.get(tone, tone_instructions["neutral"])
    tone_context = (
        f"Current tone context: {tone}. Calibrate your response energy accordingly."
    )
    return f"{base_prompt}\n\n{selected_tone}\n\n{tone_context}"


async def query_llm(
    request: QueryRequest, config: Config, tone: str = "neutral"
) -> str:
    system_prompt = build_system_prompt(config.language, tone=tone)
    context_json = request.context.model_dump_json()
    user_content = f"<context>\n{context_json}\n</context>\n\n{request.transcript}"

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
            parts=[genai_types.Part(text=user_content)],
        )
    )

    generate_config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=512,
        temperature=0.7,
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
