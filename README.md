# Inferr

Inferr is a voice-native ambient pair programming companion. Phase 0 provides an end-to-end loop: speech in the browser, context assembly in the backend, and a response from Claude that is spoken back via browser SpeechSynthesis (and a server-side pyttsx3 stub).

## Requirements

- Python 3.11+
- `uv` for dependency management
- A valid `ANTHROPIC_API_KEY` environment variable

## Install

```bash
uv pip install -e .
```

## Quick Start

```bash
inferr start
```

This launches the FastAPI server (default: `http://127.0.0.1:7331`) and opens the companion UI in your browser.

## CLI

```bash
inferr start [--host TEXT] [--port INTEGER] [--no-browser] [--lang TEXT]
inferr stop
inferr status
inferr logs [--n INTEGER]
```

## Configuration

Configuration is read from `~/.inferr/config.toml`. If the file does not exist, Inferr creates it with defaults.

```toml
[inferr]
terminal_buffer_lines = 50
history_depth = 20
file_lines = 150
language = "hinglish"
ignored_dirs = ["node_modules", ".git", "__pycache__", ".venv"]
host = "127.0.0.1"
port = 7331
```

## Environment Variables

- `ANTHROPIC_API_KEY`: required for Anthropic API access.
- `INFERR_NO_BROWSER`: if set to "1", skip `xdg-open` when starting a session.
- `INFERR_PORT`: optional override for the server port.

## Development

```bash
mypy --strict inferr/
pytest tests/
```
