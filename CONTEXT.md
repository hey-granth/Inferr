## ARCHITECTURE

A software project composed of the following subsystems:

- **inferr/**: Primary subsystem containing 14 files
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

**`inferr/models.py`**
- class `ActiveFile`
- class `FlaggedError`
- class `ConversationTurn`
- class `ContextObject`
- class `SilkConfig`
- class `ElevenLabsConfig`
- class `GeminiConfig`
- class `DeepgramConfig`
- class `QueryRequest`
- class `QueryResponse`
- class `WebSocketMessage`
- class `ShellCommandCapture`

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

**`inferr/llm.py`**
- `build_system_prompt()`
- `query_llm()`

**`inferr/context/errors.py`**
- `_first_meaningful()`
- `_last_meaningful()`
- `_should_emit()`
- `_make_error()`
- `extract_errors()`

**`inferr/tts.py`**
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
  - `is_available()`
  - `name()`
- class `BrowserTTSBackend`
  - `speak()`
  - `is_available()`
  - `name()`
- class `ElevenLabsTTSBackend`
  - `__init__()`
  - `set_ws_connection()`
  - `speak()`
  - `_synthesize_and_send()`
  - `is_available()`
  - `name()`
- `get_tts_backend()`
- `speak_pyttsx3()`
- `tts_stub_available()`

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
- `concatChunks()`
- `playSilkAudio()`
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

### `inferr/models.py`

**Purpose:** Implements models.

**Types:**
- `ActiveFile` (bases: `BaseModel`)
- `ContextObject` (bases: `BaseModel`)
- `ConversationTurn` (bases: `BaseModel`)
- `DeepgramConfig` (bases: `BaseModel`)
- `ElevenLabsConfig` (bases: `BaseModel`)
- `FlaggedError` (bases: `BaseModel`)

### `inferr/llm.py`

**Purpose:** Implements llm.
**Depends on:** `config`, `models`

**Functions:**
- `def build_system_prompt(language: str, tone: str = "neutral") -> str`
- `def query_llm(     request: QueryRequest, config: Config, tone: str = "neutral" ) -> str`

## SUPPORTING_MODULES

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

### `inferr/tts.py`

```python
class _PreparedTTSText(str)

class SilkAPIError(Exception)
    """Raised when Silk API synthesis fails."""

class TTSBackend(ABC)

def _validate_tone(tone: str) -> None

def _count_to_spoken(count: int, hinglish: bool) -> str

def _is_bullet(line: str) -> bool

def _strip_bullet_prefix(line: str) -> str

def _is_hinglish_text(text: str) -> bool

def _preprocess_tts_text(text: str, tone: str = "neutral") -> str

class SilkTTSBackend(TTSBackend)

class Pyttsx3TTSBackend(TTSBackend)

class BrowserTTSBackend(TTSBackend)

class ElevenLabsTTSBackend(TTSBackend)

def get_tts_backend(config: Config) -> TTSBackend

def speak_pyttsx3(text: str) -> None

def tts_stub_available() -> bool

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

function concatChunks(chunks)

function playSilkAudio(chunks)

function speakInBrowser(text)

function showReconnectFailure()

function scheduleReconnect()

function handleTextMessage(payload)

function connectWebSocket()

function sendTranscript(text)

async function startMic()

function stopMic()

```

### `browser/index.html`

*95 lines, 0 imports*

### `browser/style.css`

*433 lines, 0 imports*

### `inferr/__init__.py`

*10 lines, 1 imports*

## DEPENDENCY_GRAPH

```mermaid
graph LR
    f0["inferr/config.py"]
    f1["inferr/models.py"]
    f2["inferr/cli.py"]
    f3["inferr/llm.py"]
    f4["inferr/context/errors.py"]
    f5["inferr/tts.py"]
    f6["browser/app.js"]
    f7["browser/index.html"]
    f8["browser/style.css"]
    f9["pyproject.toml"]
    f10["inferr/__init__.py"]
    f11["inferr/server.py"]
    f12["typings/pyttsx3.pyi"]
    f13["inferr/context/files.py"]
    f14["inferr/context/terminal.py"]
    f15["typings/ptyprocess.pyi"]
    f16[".gitignore"]
    f17["requirements.txt"]
    f18["inferr/context/history.py"]
    f19["inferr/context/__init__.py"]
    f20["inferr/shell/inferr.bash"]
    f21["inferr/shell/inferr.zsh"]
    f0 --> f1
    f2 --> f10
    f2 --> f0
    f3 --> f1
    f3 --> f0
    f4 --> f2
    f4 --> f1
    f5 --> f2
    f5 --> f12
    f5 --> f1
    f5 --> f0
    f6 --> f2
    f7 --> f2
    f8 --> f2
    f9 --> f2
    f11 --> f5
    f11 --> f1
    f11 --> f3
    f11 --> f19
    f11 --> f0
    f13 --> f2
    f13 --> f1
    f13 --> f0
    f19 --> f11
    f19 --> f14
    f19 --> f18
    f19 --> f13
    f19 --> f4
    f19 --> f1
    f19 --> f0
```

### Cyclic Dependencies

> [!WARNING]
> The following circular import chains were detected:

1. `inferr/context/__init__.py` -> `inferr/server.py`

## RANKED_FILES

| File | Score | Tier | Tokens |
|------|-------|------|--------|
| `inferr/config.py` | 0.910 | structured summary | 164 |
| `inferr/models.py` | 0.830 | structured summary | 94 |
| `inferr/cli.py` | 0.687 | full source | 1612 |
| `inferr/llm.py` | 0.639 | structured summary | 79 |
| `inferr/context/errors.py` | 0.556 | signatures | 101 |
| `inferr/tts.py` | 0.556 | signatures | 210 |
| `tests/test_llm.py` | 0.530 | one-liner | 21 |
| `browser/app.js` | 0.447 | signatures | 126 |
| `browser/index.html` | 0.447 | signatures | 14 |
| `browser/style.css` | 0.447 | signatures | 14 |
| `pyproject.toml` | 0.447 | one-liner | 12 |
| `inferr/__init__.py` | 0.419 | signatures | 17 |
| `tests/test_context.py` | 0.408 | one-liner | 20 |
| `inferr/server.py` | 0.403 | one-liner | 20 |
| `typings/pyttsx3.pyi` | 0.366 | one-liner | 23 |
| `tests/test_server.py` | 0.336 | one-liner | 20 |
| `tests/test_errors.py` | 0.330 | one-liner | 20 |
| `tests/test_tts.py` | 0.330 | one-liner | 21 |
| `inferr/context/files.py` | 0.323 | one-liner | 26 |
| `inferr/context/terminal.py` | 0.298 | one-liner | 22 |
| `typings/ptyprocess.pyi` | 0.297 | one-liner | 22 |
| `.gitignore` | 0.201 | one-liner | 10 |
| `requirements.txt` | 0.201 | one-liner | 10 |
| `inferr/context/history.py` | 0.198 | one-liner | 21 |
| `README.md` | 0.150 | one-liner | 10 |
| `inferr/context/__init__.py` | 0.142 | one-liner | 23 |
| `tests/test_tts_preprocessing.py` | 0.133 | one-liner | 23 |
| `tests/test_llm_prompt.py` | 0.125 | one-liner | 22 |
| `inferr/shell/inferr.bash` | 0.099 | one-liner | 17 |
| `inferr/shell/inferr.zsh` | 0.099 | one-liner | 17 |

## PERIPHERY

- `tests/test_llm.py` — 9 functions, 5 imports, 230 lines
- `pyproject.toml` — 55 lines
- `tests/test_context.py` — 16 functions, 14 imports, 287 lines
- `inferr/server.py` — 15 functions, 20 imports, 386 lines
- `typings/pyttsx3.pyi` — 1 class, 1 function, 9 lines
- `tests/test_server.py` — 13 functions, 7 imports, 204 lines
- `tests/test_errors.py` — 15 functions, 3 imports, 177 lines
- `tests/test_tts.py` — 24 functions, 5 imports, 223 lines
- `inferr/context/files.py` — 2 classs, 1 function, 8 imports, 129 lines
- `inferr/context/terminal.py` — 1 class, 5 imports, 70 lines
- `typings/ptyprocess.pyi` — 1 class, 1 imports, 20 lines
- `.gitignore` — 30 lines
- `requirements.txt` — 68 lines
- `inferr/context/history.py` — 2 functions, 3 imports, 48 lines
- `README.md` — 50 lines
- `inferr/context/__init__.py` — 1 class, 8 imports, 70 lines
- `tests/test_tts_preprocessing.py` — 9 functions, 1 imports, 54 lines
- `tests/test_llm_prompt.py` — 8 functions, 1 imports, 49 lines
- `inferr/shell/inferr.bash` — 27 lines
- `inferr/shell/inferr.zsh` — 36 lines

