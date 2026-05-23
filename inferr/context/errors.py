from __future__ import annotations

import re
import time
from typing import Iterable

from inferr.models import FlaggedError


_ERROR_DEDUP: dict[str, float] = {}

_PY_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*:( .+)?$")
_HTTP_ERROR = re.compile(r"< HTTP/\d+ [45]\d\d")


def _first_meaningful(lines: Iterable[str]) -> str:
    for line in lines:
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def _last_meaningful(lines: list[str]) -> str:
    for line in reversed(lines):
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def _should_emit(error_type: str, summary: str) -> bool:
    now = time.time()
    key = f"{error_type}:{summary}"
    last_seen = _ERROR_DEDUP.get(key)
    if last_seen is not None and (now - last_seen) < 10.0:
        return False
    _ERROR_DEDUP[key] = now
    return True


def _make_error(
    error_type: str, summary: str, raw_lines: list[str]
) -> FlaggedError | None:
    trimmed_summary = summary.strip()[:120]
    if not trimmed_summary:
        return None
    if not _should_emit(error_type, trimmed_summary):
        return None
    raw_block = "\n".join(raw_lines[:20])
    return FlaggedError(type=error_type, summary=trimmed_summary, raw=raw_block)


def extract_errors(lines: list[str]) -> list[FlaggedError]:
    results: list[FlaggedError] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if "Traceback (most recent call last)" in line:
            block = [line]
            j = i + 1
            while j < len(lines) and len(block) < 20:
                block.append(lines[j])
                if _PY_EXCEPTION_LINE.match(lines[j].strip()):
                    j += 1
                    break
                j += 1
            summary = _last_meaningful(block)
            error = _make_error("python_traceback", summary, block)
            if error is not None:
                results.append(error)
            i = j
            continue

        if "ERROR:" in line:
            block = [line]
            summary = _first_meaningful(block)
            error = _make_error("uvicorn_error", summary, block)
            if error is not None:
                results.append(error)
            i += 1
            continue

        if "panic:" in line:
            block = [line]
            summary = _first_meaningful(block)
            error = _make_error("go_panic", summary, block)
            if error is not None:
                results.append(error)
            i += 1
            continue

        if "UnhandledPromiseRejection" in line or (
            "Error:" in line and "node" in line.lower()
        ):
            block = [line]
            summary = _first_meaningful(block)
            error = _make_error("node_error", summary, block)
            if error is not None:
                results.append(error)
            i += 1
            continue

        if _HTTP_ERROR.search(line):
            block = [line]
            summary = _first_meaningful(block)
            error = _make_error("http_error", summary, block)
            if error is not None:
                results.append(error)
            i += 1
            continue

        if "CONFLICT (" in line:
            block = [line]
            summary = _first_meaningful(block)
            error = _make_error("git_conflict", summary, block)
            if error is not None:
                results.append(error)
            i += 1
            continue

        i += 1

    return results
