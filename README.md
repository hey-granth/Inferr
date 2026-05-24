# Inferr

Voice-native ambient pair programming companion. Ask questions out loud, in Hinglish.
Inferr already knows what's on your screen.

## Setup

```bash
pip install inferr
cp .env.example .env
# Add your API keys to .env
inferr start
```

## Shell integration (recommended)

```bash
inferr install-shell
source ~/.zshrc   # or ~/.bashrc
```

Once installed, Inferr captures every command you run across all terminal windows.

## API keys required

| Key | Purpose | Get it |
|-----|---------|--------|
| `GEMINI_API_KEY` | LLM | aistudio.google.com |
| `SILK_API_KEY` | TTS (Rumik) | playground.rumik.ai |
| `DEEPGRAM_API_KEY` | STT | console.deepgram.com |

## Development

Install dev dependencies and run tests from the project virtualenv (not global/pipx `pytest`):

```bash
uv sync --extra dev
uv run pytest
# or: .venv/bin/pytest
```

## Commands

```bash
inferr start              # Start server + open browser UI
inferr stop               # Stop session
inferr status             # Check if running
inferr logs               # Show recent context
inferr install-shell      # Install shell integration
```

## Demo scenario

1. `inferr start`
2. `inferr install-shell && source ~/.zshrc`
3. Run a FastAPI app with a KeyError bug
4. Hit the endpoint: `curl -X POST http://localhost:8000/user -d '{"userId":"123"}'`
5. Ask Inferr: "yaar ye error kyu aa raha hai"
6. Inferr responds in Hinglish with the file, line number, and fix
