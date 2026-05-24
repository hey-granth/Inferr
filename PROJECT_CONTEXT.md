# Inferr — Project Context (complete)

This document is a machine- and chatbot-friendly, comprehensive summary of the Inferr project. It is intended to be copy-pasted as the "context" for a chatbot so the bot can quickly understand the repository, architecture, configuration, runtime behavior, and where to look for important logic.

Checklist
- [x] High-level overview and goals
- [x] How to install, run, and test locally
- [x] Configuration and environment variables
- [x] Module-by-module explanation (types, responsibilities, important functions)
- [x] Server endpoints and WebSocket message formats
- [x] Data flow: how context is assembled and how LLM/tts are used
- [x] How shell integration works
- [x] TTS backends, limitations and how to enable/disable them
- [x] Important developer notes, caveats, and tests

Tip for chatbots: if asked a question about "where does X happen?" — check these files first: `inferr/server.py`, `inferr/context/__init__.py`, `inferr/context/errors.py`, `inferr/llm.py`, `inferr/tts.py`, `inferr/config.py`, `inferr/models.py`, `inferr/cli.py`, and `browser/app.js` (frontend behavior).


---

Project summary

- Name: Inferr
- Purpose: A voice-native ambient pair programming companion. It captures terminal context, shell history, open file contents, detects errors in terminal output, forwards user voice/text queries to an LLM (Gemini), and returns spoken or browser-synthesized answers. Target audience: developers who want a live, voice-first assistant that already "knows what's on your screen".
- Main capabilities:
  - Capture terminal output and shell commands (terminal capture + optional shell plugin)
  - Watch active files in the workspace and send a snippet to the LLM as part of context
  - Detect errors (Python tracebacks, uvicorn/ASGI errors, HTTP 4xx/5xx, git conflicts, Go panics, Node errors)
  - Expose a FastAPI server and a browser UI that connects via WebSocket to stream transcripts and receive LLM responses
  - Provide multiple TTS backends (Silk (Rumik), pyttsx3, or browser speech synthesis fallback)
  - Provide a CLI (`inferr` entry point) to start/stop the server and install shell integration

Repository layout (relevant files)
- inferr/: core python package
  - __init__.py
  - cli.py            — command-line interface and entry point
  - config.py         — configuration loader and default values
  - models.py         — pydantic models used across the system
  - llm.py            — LLM prompt composition + Gemini API client wrapper
  - server.py         — FastAPI server, REST endpoints, WebSocket endpoints, session lifecycle
  - tts.py            — TTS backends and preprocessing for spoken output
  - context/          — context assembly helpers
    - __init__.py     — ContextAssembler: integrates terminal, files, and history
    - errors.py       — error detection / extraction logic from terminal text
    - files.py        — FileWatcher: detects currently-active file & content
    - history.py      — read shell history (~ .bash_history / .zsh_history)
    - terminal.py     — TerminalCapture: captures live terminal output (pty)
  - shell/            — shell plugin templates to capture commands for real-time injection
- browser/: static UI assets
  - index.html
  - app.js            — browser-side logic: WebSocket, STT proxy, audio playback, UI
  - style.css
- tests/: pytest tests covering core functionality
- pyproject.toml and requirements.txt (package metadata and pins)
- .env.example — example env file


How to install & run (developer)

1) Create virtual environment and install dependencies (preferred: use pyproject with hatch or pip).

Example (pip + virtualenv):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Or install the package in editable mode (setup: pyproject declares `inferr` script)
pip install -e .
cp .env.example .env
# edit .env to add API keys
inferr start
```

2) CLI commands (via `inferr` script / `inferr.cli:cli`):
- inferr start [--host HOST] [--port PORT] [--no-browser] — launches server and creates session
- inferr stop — stops active session
- inferr status — check session health
- inferr logs [-n N] — fetch recent terminal and conversation buffer
- inferr install-shell [--shell zsh|bash] — installs capture plugin in home dir and sources it in shell rc

Notes: `inferr start` requires a Gemini API key (Gemini is used as the LLM). If GEMINI_API_KEY is missing the CLI will abort with an error. The server also uses Deepgram (optional) for browser STT; if Deepgram is not configured the UI hides the mic and only accepts typed queries.

Configuration & environment variables

- Primary runtime config location: `~/.inferr/config.toml`. If missing it is created with sane defaults by `inferr.config._ensure_config_file()`.
- Important defaults (see `inferr/config.py`):
  - terminal_buffer_lines: 50
  - history_depth: 20
  - file_lines: 150
  - language: "hinglish" (default -- LLM responds in a Hindi-English mixed style unless overridden)
  - ignored_dirs: ["node_modules", ".git", "__pycache__", ".venv"]
  - host: 127.0.0.1
  - port: 7331

- Secrets / environment variable overrides (checked by `load_config()`):
  - GEMINI_API_KEY (or set in config gemini.api_key) — required LLM API key(s). Multiple keys may be comma-separated.
  - GEMINI_MODEL — choose Gemini model override (defaults exist in config)
  - SILK_API_KEY / SILK_API_URL — Silk (Rumik) TTS API key and endpoint
  - DEEPGRAM_API_KEY — for browser STT (Deepgram). If missing, the browser mic is disabled.
  - INFERR_PORT — overrides port, used by cli/server glue logic
  - INFERR_NO_BROWSER — if set to "1" prevents the server from attempting to open a browser window

Example .env variables (copy `.env.example` to `.env`):
- GEMINI_API_KEY=YOUR_KEY
- DEEPGRAM_API_KEY=YOUR_KEY
- SILK_API_KEY=YOUR_KEY


Data model (key Pydantic classes, see `inferr/models.py`)
- ActiveFile: path, language, content (snippet), last_modified datetime
- FlaggedError: { type, summary, raw } — single detected error block
- ConversationTurn: role (`user` or `assistant`), content
- ContextObject: assembled context returned to the LLM: terminal_buffer, shell_history, active_file, flagged_errors, conversation_history, session_id, timestamp
- QueryRequest: { transcript: str, context: ContextObject }
- QueryResponse: { text: str, session_id: str }
- WebSocketMessage: messages received via the `/ws` WebSocket (type: `transcript` or `ping`)


High-level runtime flow

1) Start server: `inferr start` launches a uvicorn server for `inferr.server:app`.
2) The CLI also kicks off a background thread to call `/session/start` so the server will create a session and start background captures.
3) Server `/session/start`: sets up a ContextAssembler and a TTS backend (via `get_tts_backend`), optionally opens the browser UI.
4) ContextAssembler runs:
   - TerminalCapture: spawns a pty (via ptyprocess) and continuously reads terminal output into an in-memory deque.
   - FileWatcher: watchdog-based file system watch; reads content of small/nearby files and tracks the active file (within repository root, respects ignored_dirs and depth limits).
   - Shell history reader: read_shell_history reads `~/.bash_history` or `~/.zsh_history` for fallback shell history.
5) Browser client connects to `/ws` WebSocket. The browser either sends the transcript of the user's voice or typed text as: { type: "transcript", payload: "..." }.
6) Server `/ws` handler receives transcripts and:
   - Assembles a ContextObject using current terminal buffer, shell history, active file, and flagged errors (via `extract_errors`).
   - Resolves a tone: `urgent` if flagged errors present, `warm` if first query in session (no conversation history), `neutral` otherwise.
   - Sends transcript + context to `inferr.llm.query_llm()` which builds system prompt (language-aware) and calls Gemini API.
   - Appends assistant/user messages to the in-memory conversation history.
   - Returns a `response` object on the WebSocket: { type: "response", text, has_errors, tone, tts_backend, context_stats }
   - Optionally invokes the configured TTS backend to speak the response (Silk may stream bytes back to the browser over WebSocket; pyttsx3 runs locally in a thread; browser fallback uses Web Speech API on the client).


LLM integration (`inferr/llm.py`)
- build_system_prompt(language, tone): constructs a system instruction tailored to `hinglish` (code-switched Hindi-English) or plain English. It contains style guides (romanised Hindi only; avoid Devanagari), expected length, and to prefer flagged errors.
- query_llm(request: QueryRequest, config: Config, tone: str):
  - Prepares a textual user message that embeds a JSON dump of ContextObject inside <context>..</context> delimiters followed by the user transcript.
  - Takes up to last 3 conversation turns and maps assistant role -> `model` role (Gemini expects `model` role for previous assistant replies).
  - Calls Google GenAI client (`google.genai`, `genai.Client`) to generate content using provided Gemini API key(s).
  - Key rotation: query_llm attempts `config.gemini.api_keys` (a list) then `config.gemini.api_key`. On errors indicating rate limits/429 it will try the next key. On other exceptions it raises RuntimeError.

Important prompt behavior:
- When language is `hinglish` the system prompt instructs the model to respond in romanised Hindi-English (Hinglish) with conversational developer phrases like "bhai", "yaar" etc. This sets the assistant's personality.
- Tone instructions change based on `urgent`/`warm`/`neutral`.


Error detection (`inferr/context/errors.py`)
- extract_errors(lines: list[str]) -> list[FlaggedError]
  - Scans the supplied list of lines for known patterns:
    - Python tracebacks: finds "Traceback (most recent call last)" and captures file / line info and exception line -> returns a FlaggedError type `python_traceback`.
    - Uvicorn/ASGI errors: lines containing "Exception in ASGI application" -> `uvicorn_error`.
    - Go panics: lines containing "panic:" -> `go_panic`.
    - Node errors: "UnhandledPromiseRejection", or lines with "Error:" and "node" -> `node_error`.
    - HTTP error lines like "< HTTP/1.1 404 Not Found" -> `http_error`.
    - Git conflicts `CONFLICT (..): Merge conflict in path` -> `git_conflict`.
  - Deduplication: to avoid repeated alerts, `_ERROR_DEDUP` stores the last-seen timestamp for (type:summary) and suppresses duplicates for 10 seconds.
  - If any real errors were detected, the function prepends a synthetic `marker` FlaggedError with summary "[!] errors detected" so downstream can recognise that errors exist.


Context assembly (`inferr/context/__init__.py`, `files.py`, `terminal.py`, `history.py`)
- TerminalCapture
  - Spawns a pty process for the user's shell (env $SHELL or /bin/bash) using `ptyprocess.PtyProcess.spawn()` and continuously reads output.
  - It filters very short lines (<3 chars) and prompt lines that end with typical prompt characters ($, #, %).
  - Exposes: get_buffer(), inject_line(), write(), stop().
- FileWatcher
  - Uses watchdog Observer to watch the current working directory recursively.
  - When files change (created/modified), `process_path` reads up to `config.file_lines` lines from the file and sets `ActiveFile` with path, language (deduced from suffix), content, timestamp.
  - It ignores events outside repository root, files nested deeper than 2 parts, and directories in `ignored_dirs`.
- read_shell_history(depth)
  - Reads either `~/.zsh_history` (strips zsh timestamp prefix) or `~/.bash_history` and returns the last `depth` deduplicated commands.
- ContextAssembler.assemble(session_id, conversation_history)
  - Merges terminal buffer + recent plugin-captured shell commands (shell plugin writes to `inferr.server` via `/capture/command`) or falls back to file-based history.
  - Runs `extract_errors` on combined lines to surface flagged errors.
  - Builds and returns a `ContextObject` pydantic model.


Server API & WebSocket protocols (`inferr/server.py`)
- HTTP endpoints:
  - POST /session/start — starts context capture and creates a new session_id; starts FileWatcher, TerminalCapture and sets tts backend. Optionally opens browser UI (unless INFERR_NO_BROWSER=1).
  - POST /session/stop — stops capture and clears session state.
  - GET /context — returns the assembled ContextObject JSON for the active session.
  - GET /health — returns a simple health map with session_active, last_activity, tts_backend.
  - GET /config/browser — returns runtime config needed by browser UI (e.g., whether Deepgram is enabled, which TTS backend is active)
  - POST /capture/command — receives commands from the installed shell plugin and injects them into the server's shell command buffer and the ContextAssembler terminal buffer.
  - POST /query — fallback HTTP query endpoint that accepts a `QueryRequest` body and returns `QueryResponse` after LLM call.

- WebSocket endpoints:
  - /ws/stt — used by the browser to proxy microphone audio to Deepgram without exposing API keys. The server opens a WebSocket connection to Deepgram and multiplexes streaming audio & results between browser and Deepgram.
  - /ws — main transcript & response WebSocket. Browser sends messages as JSON strings (msg shape defined by `WebSocketMessage` in models):
    - { type: "ping", payload: "" } => server replies { type: "pong" }
    - { type: "transcript", payload: "...user utterance..." } => server validates session and passes transcript+context to LLM, then sends response object:
       { type: "response", text: <string>, has_errors: bool, tone: <urgent|warm|neutral>, tts_backend: <name>, context_stats: { terminal_lines, active_file, error_count } }
  - When TTS backends stream bytes (Silk), the server forwards binary frames to the browser WebSocket and also sends a JSON { type: "silk_end" } marker when streaming completes. The browser collects binary chunks and plays them with WebAudio.


TTS backends (`inferr/tts.py`)
- Preprocessing
  - `_preprocess_tts_text(text, tone)` strips Markdown characters, collapses lines, converts bullets into spoken enumerations (e.g., "Teen cheezein: a, b, c" for Hinglish or "Two things: a, b" for English), truncates at ~400 chars, and prepends tone markers.
  - Supported tones: `neutral`, `urgent`, `warm` — used to alter marker/timbre or voice settings.
- Provided backends (get_tts_backend picks the highest-priority available):
  1) SilkTTSBackend (Rumik) — supports streaming to browser via intermediate Silk WS; implementation has a docstring describing how to integrate. NOTE: If silk.api_url is defaulted to `https://api.silk.ai` and no credentials exist, a NotImplementedError is raised. The code contains an implementation that posts to silk endpoints when configured.
  3) Pyttsx3TTSBackend — local offline TTS (pyttsx3) if available. Run in a background thread in server to avoid blocking.
  4) BrowserTTSBackend — fallback that does nothing on server-side; the browser uses Web Speech API to speak responses via `speechSynthesis`.

Notes & caveats about TTS:
- Silk streaming path is implemented to stream binary frames to the browser WebSocket. The browser collects Int16 audio chunks and uses WebAudio to play them (24kHz assumed).
- The Silk path contains a guard: if config.api_url equals `https://api.silk.ai` (placeholder), it raises NotImplementedError telling developer to set SILK_API_KEY and silk.api_url. The tests expect this behavior.


Shell integration plugin

- `inferr/shell/inferr.zsh` and `inferr/shell/inferr.bash` are templates installed to `~/.inferr` and sourced from user RC file (CLI `install-shell` does this). The plugin calls the server's `POST /capture/command` endpoint to forward each executed shell command and exit code, so that the server has real-time knowledge of commands the developer runs across terminals. The server maintains `_shell_command_buffer` to hold last N plugin-captured commands.


Testing

- Tests are in `tests/` and use pytest + pytest-asyncio. They cover:
  - context assembly and file/terminal/history behavior (test_context.py)
  - error extraction logic and dedup behavior (test_errors.py)
  - llm prompt building and that the request payload contains the embedded <context> JSON (test_llm.py). For Gemini calls tests monkeypatch the genai client.
  - tts preprocessing and backend selection behavior (test_tts.py)
- Typical dev commands:
  - Run unit tests: `pytest -q` (after installing dev extras)


Developer notes / known issues / caveats

- Gemini (Google GenAI) is used in `inferr/llm.py`. The code supports multiple API keys by reading `GEMINI_API_KEY` as a comma-separated string or `gemini.api_keys` in config. The code handles 429 / quota errors by trying the next key.
- The server uses a spawned pty to capture terminal output. This may behave differently on different platforms (Linux/macOS). `ptyprocess` is used via a typed stub in `typings/ptyprocess.pyi`.
- The file watcher restricts active-file detection to shalow paths (at most 2 parts below repository root) to avoid scanning entire tree and to focus on files the developer likely edits.
- There is a circular import warning in `CONTEXT.md`: `inferr/context/__init__.py` -> `inferr/server.py`. In practice this occurs because `ContextAssembler.assemble()` imports `inferr.server.get_shell_command_buffer` at call-time to avoid a top-level import cycle; be careful when refactoring.
- Silk integration: the code contains an implementation skeleton and a client flow, but the default silk.api_url placeholder and tests expect the NotImplementedError when credentials are not present. When implementing for production, follow the docstring in `SilkTTSBackend._send_to_silk_api`.
- Deepgram STT proxy: `/ws/stt` proxies microphone audio from the browser to Deepgram and returns transcript messages. The browser will hide mic UI if deepgram is not configured. Deepgram streaming requires a valid API key.
- The browser UI assumes the server is served from the same origin (it mounts `browser/` static dir at `/`). The WebSocket endpoints are relative (`/ws`, `/ws/stt`).


How to present this context to a chatbot

- Use this file as-is as a single context payload when asking a chatbot about the codebase.
- If you need the bot to answer code-specific questions (e.g., "where is error detection implemented?"), point it to `inferr/context/errors.py`.
- For runtime troubleshooting, provide the output of `inferr status` / `GET /health` and recent lines from `GET /context`.


Appendix — quick reference (short)

- Start locally: `inferr start`
- Config path: `~/.inferr/config.toml`
- Required API keys: `GEMINI_API_KEY` (LLM). Optional: `DEEPGRAM_API_KEY` (STT), `SILK_API_KEY` (TTS)
- Main server file: `inferr/server.py` (WebSocket endpoints: `/ws`, `/ws/stt`)
- LLM wrapper: `inferr/llm.py`
- Error detection: `inferr/context/errors.py`
- TTS: `inferr/tts.py` (Silk/pyttsx3/browser)
- Context assembly: `inferr/context/__init__.py` (uses `terminal.py`, `files.py`, `history.py`)


If you want, I can:
- produce an even more compact 1-page summary for quick copy into chat windows,
- or produce a JSON version keyed by module that a programmatic agent can ingest.

End of PROJECT_CONTEXT.md

