# Inferr VSCode Companion

Lightweight VSCode extension that gives Inferr ambient awareness of what you're editing.

## What it does

Pushes your **active editor buffer** (including unsaved changes) to Inferr's context endpoint every 2 seconds.

Inferr uses this to understand:
- **What file you're actively working in** — not just what was last saved
- **What code you're currently editing** — the full unsaved buffer
- **What language you're using** — for accurate context framing

## Install (dev mode)

```bash
# From the Inferr repo root:
cd extensions/vscode
code --install-extension . 2>/dev/null || code --extensionDevelopmentPath=$(pwd)
```

Or press `F5` in VSCode with this folder open to launch an Extension Development Host.

## Configuration

| Setting | Default | Description |
|---|---|---|
| `inferr.serverUrl` | `http://127.0.0.1:7331` | Inferr server URL |
| `inferr.pushIntervalMs` | `2000` | Push frequency in ms |

## Design constraints

- **No npm install required** — uses only Node built-ins + VS Code API
- **~100 LOC** — simple, auditable, no framework
- **Silent failures** — never disrupts the editor if Inferr is offline
- **Unsaved-first** — `doc.getText()` captures buffer state, not saved file
