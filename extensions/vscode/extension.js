// Inferr VSCode Companion Extension
// Pushes the active editor buffer to /context/file so Inferr
// knows what the developer is editing in real-time.
// ~60 LOC, no dependencies beyond the VS Code API.

const vscode = require("vscode");
const http = require("http");
const https = require("https");

let pushInterval = null;
let lastPushedUri = null;
let lastPushedVersion = -1;

/**
 * Post data to the Inferr server (fire-and-forget, no retries).
 * @param {string} serverUrl
 * @param {string} path
 * @param {object} body
 */
function post(serverUrl, path, body) {
  const url = new URL(path, serverUrl);
  const data = JSON.stringify(body);
  const lib = url.protocol === "https:" ? https : http;
  const req = lib.request(
    {
      hostname: url.hostname,
      port: url.port || (url.protocol === "https:" ? 443 : 80),
      path: url.pathname,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(data),
        "User-Agent": "inferr-vscode/0.1.0",
      },
    },
    () => {} // ignore response
  );
  req.on("error", () => {}); // silent — Inferr may not be running
  req.write(data);
  req.end();
}

/**
 * Determine a simple language string from a VS Code language ID.
 * @param {string} langId
 * @returns {string}
 */
function mapLanguage(langId) {
  const MAP = {
    python: "python",
    javascript: "javascript",
    typescript: "typescript",
    go: "go",
    rust: "rust",
    cpp: "cpp",
    c: "c",
    java: "java",
    ruby: "ruby",
    shellscript: "shell",
  };
  return MAP[langId] || langId;
}

/**
 * Push the active editor's buffer to Inferr.
 */
function pushActiveBuffer() {
  const cfg = vscode.workspace.getConfiguration("inferr");
  const serverUrl = cfg.get("serverUrl", "http://127.0.0.1:7331");

  const editor = vscode.window.activeTextEditor;
  if (!editor) return;

  const doc = editor.document;
  // Skip if nothing changed since last push
  if (
    doc.uri.toString() === lastPushedUri &&
    doc.version === lastPushedVersion
  ) {
    return;
  }
  lastPushedUri = doc.uri.toString();
  lastPushedVersion = doc.version;

  post(serverUrl, "/context/file", {
    path: doc.fileName,
    content: doc.getText(),
    language: mapLanguage(doc.languageId),
  });
}

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
  const cfg = vscode.workspace.getConfiguration("inferr");
  const intervalMs = cfg.get("pushIntervalMs", 2000);

  pushInterval = setInterval(pushActiveBuffer, intervalMs);

  // Also push immediately on editor focus change
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(() => pushActiveBuffer()),
    vscode.workspace.onDidChangeTextDocument(() => {
      // Debounce: let the interval handle it — just reset version cache
    })
  );

  context.subscriptions.push({
    dispose: () => {
      if (pushInterval) clearInterval(pushInterval);
    },
  });
}

function deactivate() {
  if (pushInterval) clearInterval(pushInterval);
}

module.exports = { activate, deactivate };
