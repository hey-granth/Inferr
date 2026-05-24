## ARCHITECTURE

A software project composed of the following subsystems:

- **inferr/**: Primary subsystem containing 15 files
- **tests/**: Primary subsystem containing 7 files
- **browser/**: Primary subsystem containing 3 files
- **typings/**: Primary subsystem containing 2 files
- **Root**: Contains scripts and execution points

## ENTRY_POINTS

### `inferr/cli.py`

```python
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import threading
import time
from typing import Optional

import click
import httpx
import uvicorn

from inferr.config import Config, load_config
from inferr import server as server_module


@click.group()
def cli() -> None:
    """Inferr command line interface."""


def _resolve_port(config: Config, port_override: Optional[int]) -> int:
    if port_override is not None:
        return port_override
    env_port = os.environ.get("INFERR_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            return config.port
    return config.port


def _resolve_host(config: Config, host_override: Optional[str]) -> str:
    if host_override is not None:
        return host_override
    return config.host


def _start_session_when_ready(base_url: str) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=1.0) as client:
                response = client.get(f"{base_url}/health")
                if response.status_code == 200:
                    break
        except httpx.ConnectError:
            time.sleep(0.2)
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        return

    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(f"{base_url}/session/start")
    except httpx.HTTPError:
        return


@cli.command()
@click.option("--host", type=str, default=None)
@click.option("--port", type=int, default=None)
@click.option("--no-browser", is_flag=True, default=False)
@click.option("--lang", type=str, default=None)
def start(
    host: Optional[str], port: Optional[int], no_browser: bool, lang: Optional[str]
) -> None:
    """Start the Inferr server and initialize a session."""
    config = load_config()
    if not config.gemini.api_key:
        click.echo(
            "[inferr] ERROR: GEMINI_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable.",
            err=True,
        )
        raise SystemExit(1)
    resolved_port = _resolve_port(config, port)
    resolved_host = _resolve_host(config, host)
    resolved_lang = lang if lang is not None else config.language

    config = replace(
        config, host=resolved_host, port=resolved_port, language=resolved_lang
    )

    server_module.CONFIG_OVERRIDE = config
    os.environ["INFERR_PORT"] = str(resolved_port)
    if no_browser:
        os.environ["INFERR_NO_BROWSER"] = "1"

    base_url = f"http://{resolved_host}:{resolved_port}"
    thread = threading.Thread(
        target=_start_session_when_ready, args=(base_url,), daemon=True
    )
    thread.start()

    uvicorn.run(
        "inferr.server:app", host=resolved_host, port=resolved_port, log_level="info"
    )


@cli.command()
def stop() -> None:
    """Stop the current Inferr session."""
    config = load_config()
    port_value = _resolve_port(config, None)
    base_url = f"http://{config.host}:{port_value}"

    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(f"{base_url}/session/stop")
        click.echo("Inferr session stopped.")
    except httpx.ConnectError:
        click.echo("Inferr is not running. Start it with `inferr start`.")
    except httpx.HTTPError:
        click.echo("Failed to stop session.")


@cli.command()
def status() -> None:
    """Show the current Inferr status."""
    config = load_config()
    port_value = _resolve_port(config, None)
    base_url = f"http://{config.host}:{port_value}"

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/health")
        data = response.json()
        click.echo(f"Session active: {data.get('session_active')}")
        click.echo(f"Last activity: {data.get('last_activity')}")
    except httpx.ConnectError:
        click.echo("Inferr is not running. Start it with `inferr start`.")
    except httpx.HTTPError:
        click.echo("Failed to fetch status.")


@cli.command()
@click.option("--n", type=int, default=None)
def logs(n: Optional[int]) -> None:
    """Show recent terminal buffer lines and conversation history."""
    config = load_config()
    port_value = _resolve_port(config, None)
    base_url = f"http://{config.host}:{port_value}"
    limit = n if n is not None else config.terminal_buffer_lines

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/context")
        if response.status_code == 404:
            click.echo("No active session.")
            return
        data = response.json()
        terminal_lines = data.get("terminal_buffer", [])[-limit:]
        history = data.get("conversation_history", [])

        click.echo("--- Terminal Buffer ---")
        for line in terminal_lines:
            click.echo(line)

        click.echo("--- Conversation History ---")
        for turn in history:
            role = turn.get("role", "")
            content = turn.get("content", "")
            click.echo(f"{role}: {content}")
    except httpx.ConnectError:
        click.echo("Inferr is not running. Start it with `inferr start`.")
    except httpx.HTTPError:
        click.echo("Failed to fetch logs.")


@cli.command("install-shell")
@click.option("--shell", "shell_name", type=click.Choice(["zsh", "bash"]), default=None)
def install_shell(shell_name: Optional[str]) -> None:
    """Install the Inferr shell integration plugin."""
    import shutil

    detected = os.environ.get("SHELL", "")
    if shell_name is None:
        if "zsh" in detected:
            shell_name = "zsh"
        elif "bash" in detected:
            shell_name = "bash"
        else:
            click.echo("Could not detect shell. Use --shell zsh or --shell bash.")
            raise SystemExit(1)

    plugin_src = Path(__file__).resolve().parent / "shell" / f"inferr.{shell_name}"
    if not plugin_src.exists():
        click.echo(f"Plugin file not found: {plugin_src}")
        raise SystemExit(1)

    inferr_dir = Path.home() / ".inferr"
    inferr_dir.mkdir(parents=True, exist_ok=True)
    dest = inferr_dir / f"inferr.{shell_name}"
    shutil.copy(plugin_src, dest)

    rc_file = Path.home() / (f".{shell_name}rc")
    source_line = f'\nsource "{dest}"  # inferr shell integration\n'

    if rc_file.exists():
        content = rc_file.read_text(encoding="utf-8")
        if str(dest) in content:
            click.echo(f"Already installed in {rc_file}.")
            return
        rc_file.write_text(content + source_line, encoding="utf-8")
    else:
        rc_file.write_text(source_line, encoding="utf-8")

    click.echo(f"Installed inferr shell plugin to {rc_file}.")
    click.echo(f"Run: source {rc_file}")

```

## SYMBOL_INDEX

**`inferr/models.py`**
- class `ActiveFile`
- class `FlaggedError`
- class `ConversationTurn`
- class `ContextObject`
- class `SilkConfig`
- class `GeminiConfig`
- class `DeepgramConfig`
- class `QueryRequest`
- class `QueryResponse`
- class `WebSocketMessage`
- class `ShellCommandCapture`

**`inferr/config.py`**
- class `Config`
- `_default_toml()`
- `_get_table()`
- `_coerce_int()`
- `_coerce_str()`
- `_coerce_str_list()`
- `_coerce_bool()`
- `_ensure_config_file()`
- `load_config()`

**`inferr/cli.py`**
- `cli()`
- `_resolve_port()`
- `_resolve_host()`
- `_start_session_when_ready()`
- `start()`
- `stop()`
- `status()`
- `logs()`
- `install_shell()`

**`inferr/tts.py`**
- `tts_timeout_seconds()`
- class `_PreparedTTSText`
  - `__new__()`
  - `__eq__()`
  - `__len__()`
  - `startswith()`
- class `SilkAPIError`
- class `TTSBackend`
- `_validate_tone()`
- `_count_to_spoken()`
- `_is_bullet()`
- `_strip_bullet_prefix()`
- `_is_hinglish_text()`
- `_preprocess_tts_text()`
- class `SilkTTSBackend`
  - `__init__()`
  - `set_ws_connection()`
  - `speak()`
  - `_preprocess_text()`
  - `_send_to_silk_api()`
  - `is_available()`
  - `name()`
- class `Pyttsx3TTSBackend`
  - `speak()`
  - `_speak_sync()`
  - `is_available()`
  - `name()`
- class `BrowserTTSBackend`
  - `speak()`
  - `is_available()`
  - `name()`
- `get_tts_backend()`
- `speak_pyttsx3()`
- `tts_stub_available()`

**`inferr/llm.py`**
- `_is_high_signal()`
- `_rank_terminal_lines()`
- `build_system_prompt()`
- `_shorten_line()`
- `_summarize_context()`
- `query_llm()`

**`inferr/server.py`**
- `get_shell_command_buffer()`
- `_apply_env_overrides()`
- `get_config()`
- `lifespan()`
- `unhandled_exception_handler()`
- `start_session()`
- `stop_session()`
- `get_context()`
- `health()`
- `browser_config()`
- `resolve_tone()`
- `speech_to_text_endpoint()`
- `websocket_endpoint()`
- `capture_command()`
- `query_endpoint()`

**`inferr/context/errors.py`**
- `_first_meaningful()`
- `_last_meaningful()`
- `_should_emit()`
- `_make_error()`
- `extract_errors()`

**`inferr/context/terminal.py`**
- class `TerminalCapture`
  - `__init__()`
  - `_reader()`
  - `_append_line()`
  - `get_buffer()`
  - `inject_line()`
  - `write()`
  - `stop()`

**`browser/app.js`**
- `drawOscilloscope()`
- `escapeHtml()`
- `addExchange()`
- `setSpeaking()`
- `setWsStatus()`
- `setTtsBackend()`
- `setContextStats()`
- `triggerErrorBadge()`
- `loadBrowserConfig()`
- `speakInBrowser()`
- `showReconnectFailure()`
- `scheduleReconnect()`
- `handleTextMessage()`
- `connectWebSocket()`
- `sendTranscript()`
- `startMic()`
- `stopMic()`

## IMPORTANT_CALL_PATHS

cli.cli()
  → __init__()
## CORE_MODULES

### `inferr/models.py`

**Purpose:** Implements models.

**Types:**
- `ActiveFile` (bases: `BaseModel`)
- `ContextObject` (bases: `BaseModel`)
- `ConversationTurn` (bases: `BaseModel`)
- `DeepgramConfig` (bases: `BaseModel`)
- `FlaggedError` (bases: `BaseModel`)
- `GeminiConfig` (bases: `BaseModel`)

### `inferr/config.py`

**Purpose:** Implements config.
**Depends on:** `models`

**Types:**
- `Config`

**Functions:**
- `def _coerce_bool(value: object, default: bool) -> bool`
- `def _coerce_int(value: object, default: int) -> int`
- `def _coerce_str(value: object, default: str) -> str`
- `def _coerce_str_list(value: object, default: list[str]) -> list[str]`
- `def _default_toml() -> str`
- `def _ensure_config_file(path: Path) -> None`
- `def _get_table(data: Mapping[str, object]) -> dict[str, object]`
- `def load_config() -> Config`

### `inferr/tts.py`

**Purpose:** Implements tts.
**Depends on:** `config`, `debug`, `models`

**Types:**
- `BrowserTTSBackend` (bases: `TTSBackend`) methods: `is_available`, `name`, `speak`
- `Pyttsx3TTSBackend` (bases: `TTSBackend`) methods: `is_available`, `name`, `speak`

**Functions:**
- `def _count_to_spoken(count: int, hinglish: bool) -> str`
- `def _is_bullet(line: str) -> bool`
- `def _is_hinglish_text(text: str) -> bool`
- `def _preprocess_tts_text(text: str, tone: str = "neutral") -> str`
- `def _strip_bullet_prefix(line: str) -> str`
- `def _validate_tone(tone: str) -> None`

## SUPPORTING_MODULES

### `inferr/llm.py`

```python
def _is_high_signal(line: str) -> bool

def _rank_terminal_lines(lines: list[str], limit: int) -> list[str]
    """Return up to `limit` lines, prioritising high-signal diagnostics."""

def build_system_prompt(language: str, tone: str = "neutral") -> str

def _shorten_line(value: str, max_len: int = 120) -> str

def _summarize_context(request: QueryRequest) -> str

def query_llm(
    request: QueryRequest, config: Config, tone: str = "neutral"
) -> str

```

### `inferr/server.py`

```python
def get_shell_command_buffer() -> list[str]
    """Return the current shell command buffer (public accessor for cross-module use)."""

def _apply_env_overrides(config: Config) -> Config

def get_config() -> Config

def lifespan(app: FastAPI)
    """FastAPI lifespan manager."""

def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse
    """Return structured JSON for unexpected server errors."""

def start_session() -> dict[str, str]
    """Initialize a new Inferr session and context capture."""

def stop_session() -> dict[str, str]
    """Stop the current Inferr session."""

def get_context() -> JSONResponse
    """Return the current context object."""

def health() -> dict[str, object | None]
    """Return server health status."""

def browser_config() -> dict[str, object]
    """Return config the browser needs at runtime. Only non-secret values."""

def resolve_tone(context: ContextObject) -> str
    """Returns 'urgent' if real (non-marker) errors are flagged.
    Returns 'neutral' otherwise."""

def speech_to_text_endpoint(websocket: WebSocket) -> None
    """Proxy browser microphone audio to Deepgram without exposing API keys."""

def websocket_endpoint(websocket: WebSocket) -> None
    """Handle WebSocket transcript streaming and LLM responses."""

def capture_command(payload: ShellCommandCapture) -> dict[str, str]
    """Receive shell commands from the shell integration plugin."""

def query_endpoint(request: QueryRequest) -> QueryResponse
    """HTTP fallback endpoint for transcript queries."""

```

### `inferr/context/errors.py`

```python
def _first_meaningful(lines: Iterable[str]) -> str

def _last_meaningful(lines: list[str]) -> str

def _should_emit(error_type: str, summary: str) -> bool

def _make_error(
    error_type: str, summary: str, raw_lines: list[str]
) -> FlaggedError | None

def extract_errors(lines: list[str]) -> list[FlaggedError]

```

### `inferr/context/terminal.py`

```python
class TerminalCapture

```

### `browser/app.js`

```javascript
function drawOscilloscope()

function escapeHtml(text)

function addExchange(query, response, isError = false)

function setSpeaking(active)

function setWsStatus(connected)

function setTtsBackend(name)

function setContextStats(payload)

function triggerErrorBadge()

async function loadBrowserConfig()

function speakInBrowser(text)

function showReconnectFailure()

function scheduleReconnect()

function handleTextMessage(payload)

function connectWebSocket()

function sendTranscript(text)

async function startMic()

function stopMic()

```

### `browser/style.css`

*426 lines, 0 imports*

## DEPENDENCY_GRAPH

```mermaid
graph LR
    f0["inferr/models.py"]
    f1["inferr/config.py"]
    f2["inferr/cli.py"]
    f3["inferr/tts.py"]
    f4["inferr/llm.py"]
    f5["inferr/server.py"]
    f6["inferr/context/errors.py"]
    f7["inferr/context/terminal.py"]
    f8["browser/app.js"]
    f9["browser/style.css"]
    f10["pyproject.toml"]
    f11["inferr/context/__init__.py"]
    f12["inferr/__init__.py"]
    f13["browser/index.html"]
    f14["typings/ptyprocess.pyi"]
    f15["requirements.txt"]
    f16["inferr/context/files.py"]
    f17["inferr/debug.py"]
    f18["typings/pyttsx3.pyi"]
    f19["inferr/context/history.py"]
    f20["inferr/shell/inferr.bash"]
    f21["inferr/shell/inferr.zsh"]
    f22[".gitignore"]
    f1 --> f0
    f2 --> f5
    f2 --> f12
    f2 --> f1
    f3 --> f2
    f3 --> f17
    f3 --> f0
    f3 --> f1
    f4 --> f17
    f4 --> f0
    f4 --> f1
    f5 --> f17
    f5 --> f3
    f5 --> f0
    f5 --> f4
    f5 --> f11
    f5 --> f1
    f6 --> f2
    f6 --> f0
    f7 --> f2
    f7 --> f14
    f8 --> f2
    f9 --> f2
    f10 --> f2
    f11 --> f5
    f11 --> f7
    f11 --> f19
    f11 --> f16
    f11 --> f6
    f11 --> f0
    f11 --> f1
    f13 --> f2
    f16 --> f0
    f16 --> f1
```

### Cyclic Dependencies

> [!WARNING]
> The following circular import chains were detected:

1. `inferr/server.py` -> `inferr/context/__init__.py`

## RANKED_FILES

| File | Score | Tier | Tokens |
|------|-------|------|--------|
| `inferr/models.py` | 0.808 | structured summary | 92 |
| `inferr/config.py` | 0.792 | structured summary | 164 |
| `inferr/cli.py` | 0.784 | full source | 1611 |
| `inferr/tts.py` | 0.626 | structured summary | 197 |
| `inferr/llm.py` | 0.552 | signatures | 139 |
| `tests/test_llm.py` | 0.533 | one-liner | 21 |
| `inferr/server.py` | 0.526 | signatures | 331 |
| `tests/test_server.py` | 0.492 | one-liner | 20 |
| `tests/test_context.py` | 0.492 | one-liner | 20 |
| `tests/test_tts.py` | 0.476 | one-liner | 21 |
| `inferr/context/errors.py` | 0.436 | signatures | 101 |
| `inferr/context/terminal.py` | 0.409 | signatures | 18 |
| `browser/app.js` | 0.378 | signatures | 112 |
| `browser/style.css` | 0.378 | signatures | 14 |
| `pyproject.toml` | 0.368 | one-liner | 12 |
| `inferr/context/__init__.py` | 0.361 | one-liner | 23 |
| `inferr/__init__.py` | 0.312 | one-liner | 18 |
| `tests/test_tts_preprocessing.py` | 0.304 | one-liner | 23 |
| `browser/index.html` | 0.303 | one-liner | 11 |
| `tests/test_llm_prompt.py` | 0.296 | one-liner | 22 |
| `typings/ptyprocess.pyi` | 0.295 | one-liner | 22 |
| `tests/test_errors.py` | 0.292 | one-liner | 20 |
| `requirements.txt` | 0.261 | one-liner | 10 |
| `PROJECT_CONTEXT.md` | 0.253 | one-liner | 11 |
| `README.md` | 0.252 | one-liner | 10 |
| `inferr/context/files.py` | 0.248 | one-liner | 26 |
| `inferr/debug.py` | 0.233 | one-liner | 20 |
| `typings/pyttsx3.pyi` | 0.195 | one-liner | 23 |
| `inferr/context/history.py` | 0.190 | one-liner | 21 |
| `inferr/shell/inferr.bash` | 0.145 | one-liner | 17 |
| `inferr/shell/inferr.zsh` | 0.145 | one-liner | 17 |
| `.gitignore` | 0.115 | one-liner | 10 |

## PERIPHERY

- `tests/test_llm.py` — 9 functions, 5 imports, 227 lines
- `tests/test_server.py` — 13 functions, 7 imports, 203 lines
- `tests/test_context.py` — 16 functions, 14 imports, 285 lines
- `tests/test_tts.py` — 21 functions, 12 imports, 274 lines
- `pyproject.toml` — 54 lines
- `inferr/context/__init__.py` — 1 class, 9 imports, 76 lines
- `inferr/__init__.py` — 1 imports, 10 lines
- `tests/test_tts_preprocessing.py` — 9 functions, 1 imports, 54 lines
- `browser/index.html` — 95 lines
- `tests/test_llm_prompt.py` — 16 functions, 1 imports, 102 lines
- `typings/ptyprocess.pyi` — 1 class, 1 imports, 20 lines
- `tests/test_errors.py` — 15 functions, 3 imports, 177 lines
- `requirements.txt` — 71 lines
- `PROJECT_CONTEXT.md` — 266 lines
- `README.md` — 50 lines
- `inferr/context/files.py` — 2 classs, 1 function, 8 imports, 129 lines
- `inferr/debug.py` — 1 class, 4 imports, 41 lines
- `typings/pyttsx3.pyi` — 1 class, 1 function, 9 lines
- `inferr/context/history.py` — 2 functions, 3 imports, 48 lines
- `inferr/shell/inferr.bash` — 27 lines
- `inferr/shell/inferr.zsh` — 36 lines
- `.gitignore` — 30 lines

