const wsDot = document.getElementById("ws-dot");
const waveform = document.getElementById("waveform");
const micBtn = document.getElementById("mic-btn");
const sendBtn = document.getElementById("send-btn");
const typedInput = document.getElementById("typed-input");
const lastExchange = document.getElementById("last-exchange");
const voiceWarning = document.getElementById("voice-warning");
const errorDot = document.getElementById("error-dot");

let ws = null;
let recognition = null;
let listening = false;
let lastTranscript = "";
let reconnectAttempts = 0;
let shouldReconnect = true;
let silkChunks = [];

const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY_MS = 2000;

function ensureWaveformBars() {
    if (!waveform) {
        return;
    }
    while (waveform.children.length < 7) {
        const bar = document.createElement("span");
        bar.className = "bar";
        waveform.appendChild(bar);
    }
}

function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;");
}

function setWsStatus(connected) {
    if (connected) {
        wsDot.classList.add("connected");
        wsDot.classList.remove("disconnected");
        wsDot.title = "Connected";
    } else {
        wsDot.classList.remove("connected");
        wsDot.classList.add("disconnected");
    }
}

function triggerErrorDot() {
    if (!errorDot) {
        return;
    }
    errorDot.classList.add("error-active");
    window.setTimeout(() => {
        errorDot.classList.remove("error-active");
    }, 10000);
}

function concatChunks(chunks) {
    let totalLength = 0;
    for (const chunk of chunks) {
        totalLength += chunk.length;
    }
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
    }
    return combined;
}

function playSilkAudio(chunks) {
    if (!chunks.length) {
        document.dispatchEvent(new CustomEvent("silkPlaybackEnd"));
        return;
    }
    const bytes = concatChunks(chunks);
    const blob = new Blob([bytes], { type: "audio/mpeg" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => {
        URL.revokeObjectURL(url);
        document.dispatchEvent(new CustomEvent("silkPlaybackEnd"));
    };
    audio.onerror = () => {
        URL.revokeObjectURL(url);
        document.dispatchEvent(new CustomEvent("silkPlaybackEnd"));
    };
    void audio.play().catch(() => {
        URL.revokeObjectURL(url);
        document.dispatchEvent(new CustomEvent("silkPlaybackEnd"));
    });
}

function speakInBrowser(text) {
    try {
        const utterance = new SpeechSynthesisUtterance(text);
        waveform.classList.add("speaking");
        utterance.onend = () => {
            waveform.classList.remove("speaking");
        };
        utterance.onerror = () => {
            waveform.classList.remove("speaking");
        };
        window.speechSynthesis.speak(utterance);
    } catch (err) {
        waveform.classList.remove("speaking");
    }
}

function showReconnectFailure() {
    lastExchange.innerHTML = '<div class="error">Connection lost. Restart inferr.</div>';
}

function scheduleReconnect() {
    if (!shouldReconnect) {
        return;
    }
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        wsDot.title = "Reconnect failed";
        showReconnectFailure();
        return;
    }
    reconnectAttempts += 1;
    wsDot.title = `Reconnecting (attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`;
    window.setTimeout(() => {
        connectWebSocket();
    }, RECONNECT_DELAY_MS);
}

function handleTextMessage(payload) {
    if (payload.type === "response") {
        const text = String(payload.text || "");
        const hasErrors = payload.has_errors === true;
        const backend = String(payload.tts_backend || "browser");
        lastExchange.innerHTML =
            `<div><strong>You:</strong> ${escapeHtml(lastTranscript)}</div>` +
            `<div><strong>Inferr:</strong> ${escapeHtml(text)}</div>`;

        if (hasErrors) {
            triggerErrorDot();
        }

        if (backend === "browser") {
            speakInBrowser(text);
        }
    }

    if (payload.type === "silk_end") {
        playSilkAudio(silkChunks);
        silkChunks = [];
    }

    if (payload.type === "error") {
        lastExchange.innerHTML = `<div class="error">${escapeHtml(
            String(payload.message || "Unknown error")
        )}</div>`;
    }
}

function connectWebSocket() {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${scheme}://${window.location.host}/ws`;
    ws = new WebSocket(wsUrl);

    ws.addEventListener("open", () => {
        reconnectAttempts = 0;
        setWsStatus(true);
    });

    ws.addEventListener("close", () => {
        setWsStatus(false);
        scheduleReconnect();
    });

    ws.addEventListener("error", () => {
        setWsStatus(false);
    });

    ws.addEventListener("message", async (event) => {
        if (typeof event.data !== "string") {
            const arrayBuffer = await event.data.arrayBuffer();
            if (silkChunks.length === 0) {
                waveform.classList.add("speaking");
            }
            silkChunks.push(new Uint8Array(arrayBuffer));
            return;
        }

        let payload = null;
        try {
            payload = JSON.parse(event.data);
        } catch (err) {
            return;
        }
        handleTextMessage(payload);
    });
}

function sendTranscript(text) {
    const trimmed = text.trim();
    if (!trimmed) {
        return;
    }
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        lastExchange.innerHTML = '<div class="error">WebSocket is not connected.</div>';
        return;
    }
    lastTranscript = trimmed;
    ws.send(JSON.stringify({ type: "transcript", payload: trimmed }));
}

function setupSpeechRecognition() {
    const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        micBtn.classList.add("hidden");
        voiceWarning.classList.remove("hidden");
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
        listening = true;
        micBtn.classList.add("listening");
    };

    recognition.onend = () => {
        listening = false;
        micBtn.classList.remove("listening");
    };

    recognition.onerror = () => {
        listening = false;
        micBtn.classList.remove("listening");
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        sendTranscript(transcript);
    };

    micBtn.addEventListener("click", () => {
        if (!recognition) {
            return;
        }
        if (listening) {
            recognition.stop();
        } else {
            recognition.start();
        }
    });
}

sendBtn.addEventListener("click", () => {
    sendTranscript(typedInput.value);
    typedInput.value = "";
});

typedInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendTranscript(typedInput.value);
        typedInput.value = "";
    }
});

document.addEventListener("silkPlaybackEnd", () => {
    waveform.classList.remove("speaking");
});

ensureWaveformBars();
connectWebSocket();
setupSpeechRecognition();

setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping", payload: "" }));
    }
}, 30000);
