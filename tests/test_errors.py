from __future__ import annotations

import pytest

from inferr.context import errors as errors_module
from inferr.context.errors import extract_errors


@pytest.fixture(autouse=True)
def _clear_error_dedup() -> None:
    errors_module._ERROR_DEDUP.clear()


def _get_real_errors(lines: list[str]) -> list[object]:
    return [error for error in extract_errors(lines) if error.type != "marker"]


def test_python_keyerror_traceback() -> None:
    lines = [
        "Traceback (most recent call last):",
        '  File "/app/routes/user.py", line 23, in get_user',
        '    user_id = request.json()["user_id"]',
        "KeyError: 'user_id'",
    ]

    errors = _get_real_errors(lines)

    assert errors
    assert errors[0].type == "python_traceback"
    assert "routes/user.py line 23" in errors[0].summary
    assert errors[0].raw


def test_uvicorn_error_traceback() -> None:
    lines = [
        "ERROR:    Exception in ASGI application",
        "Traceback (most recent call last):",
        '  File "/app/main.py", line 45, in __call__',
        "    await self.app(scope, receive, send)",
        "Exception: database connection failed",
    ]

    errors = _get_real_errors(lines)

    assert errors
    assert errors[0].type == "uvicorn_error"
    assert "Exception: database connection failed" in errors[0].summary
    assert errors[0].raw


def test_go_panic() -> None:
    lines = [
        "goroutine 1 [running]:",
        "panic: runtime error: index out of range [3] with length 3",
    ]

    errors = _get_real_errors(lines)

    assert errors
    assert errors[0].type == "go_panic"
    assert (
        "panic: runtime error: index out of range [3] with length 3"
        in errors[0].summary
    )
    assert errors[0].raw


def test_http_error_curl() -> None:
    lines = ["< HTTP/1.1 422 Unprocessable Entity"]

    errors = _get_real_errors(lines)

    assert errors
    assert errors[0].type == "http_error"
    assert "HTTP 422 Unprocessable Entity" in errors[0].summary
    assert errors[0].raw


def test_git_conflict() -> None:
    lines = ["CONFLICT (content): Merge conflict in inferr/server.py"]

    errors = _get_real_errors(lines)

    assert errors
    assert errors[0].type == "git_conflict"
    assert "inferr/server.py" in errors[0].summary
    assert errors[0].raw


def test_node_unhandled_promise_rejection() -> None:
    lines = [
        "UnhandledPromiseRejectionWarning: TypeError: Cannot read property 'id' of undefined"
    ]

    errors = _get_real_errors(lines)

    assert errors
    assert errors[0].type == "node_error"
    assert "TypeError: Cannot read property 'id' of undefined" in errors[0].summary
    assert errors[0].raw


def test_marker_present_when_errors_found() -> None:
    lines = ["panic: runtime error: index out of range [3] with length 3"]

    errors = extract_errors(lines)

    assert errors
    assert errors[0].type == "marker"


def test_marker_absent_when_no_errors() -> None:
    assert extract_errors(["all good", "nothing suspicious"]) == []


def test_deduplication_within_10_seconds() -> None:
    lines = ["panic: runtime error: index out of range [3] with length 3"]

    first = extract_errors(lines)
    second = extract_errors(lines)

    assert first
    assert second == []
