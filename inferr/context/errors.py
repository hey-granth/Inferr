from __future__ import annotations

import re
import time
from typing import Iterable

from inferr.models import FlaggedError


_ERROR_DEDUP: dict[str, float] = {}

_PY_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*:( .+)?$")
_PY_FRAME_LINE = re.compile(r'File "([^"]+)", line (\d+),')
_HTTP_ERROR = re.compile(r"< HTTP/\d+(?:\.\d+)? ([45]\d\d) (.+)")
_GIT_CONFLICT = re.compile(r"CONFLICT \([^)]+\): Merge conflict in (.+)")


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
            last_frame_path = ""
            last_frame_line = ""
            while j < len(lines) and len(block) < 20:
                block.append(lines[j])
                frame_match = _PY_FRAME_LINE.search(lines[j])
                if frame_match is not None:
                    full_path = frame_match.group(1)
                    path_parts = [part for part in full_path.split("/") if part]
                    short_path = (
                        "/".join(path_parts[-2:]) if len(path_parts) >= 2 else full_path
                    )
                    last_frame_path = short_path
                    last_frame_line = frame_match.group(2)
                if _PY_EXCEPTION_LINE.match(lines[j].strip()):
                    j += 1
                    break
                j += 1
            exception_line = _last_meaningful(block)
            if last_frame_path and last_frame_line and exception_line:
                summary = (
                    f"{exception_line} in {last_frame_path} line {last_frame_line}"
                )
            else:
                summary = exception_line
            error = _make_error("python_traceback", summary, block)
            if error is not None:
                results.append(error)
            i = j
            continue

        if "ERROR:    Exception in ASGI application" in line:
            block = [line]
            summary = "Exception in ASGI application"
            j = i + 1
            if j < len(lines) and "Traceback (most recent call last)" in lines[j]:
                block.append(lines[j])
                j += 1
                while j < len(lines) and len(block) < 20:
                    block.append(lines[j])
                    stripped = lines[j].strip()
                    if _PY_EXCEPTION_LINE.match(stripped):
                        summary = stripped
                        j += 1
                        break
                    j += 1
            error = _make_error("uvicorn_error", summary, block)
            if error is not None:
                results.append(error)
            i = j
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
            if "UnhandledPromiseRejectionWarning:" in summary:
                summary = summary.split("UnhandledPromiseRejectionWarning:", 1)[
                    1
                ].strip()
            error = _make_error("node_error", summary, block)
            if error is not None:
                results.append(error)
            i += 1
            continue

        http_match = _HTTP_ERROR.search(line)
        if http_match is not None:
            block = [line]
            status = http_match.group(1).strip()
            reason = http_match.group(2).strip()
            summary = f"HTTP {status} {reason}"
            error = _make_error("http_error", summary, block)
            if error is not None:
                results.append(error)
            i += 1
            continue

        conflict_match = _GIT_CONFLICT.search(line)
        if conflict_match is not None:
            block = [line]
            path = conflict_match.group(1).strip()
            summary = f"Merge conflict in {path}"
            error = _make_error("git_conflict", summary, block)
            if error is not None:
                results.append(error)
            i += 1
            continue

        i += 1

    if results:
        marker = FlaggedError(
            type="marker", summary="[!] errors detected", raw="marker"
        )
        return [marker, *results]
    return []
