const wsDot = document.getElementById("ws-dot");
const micBtn = document.getElementById("mic-btn");
const sendBtn = document.getElementById("send-btn");
const typedInput = document.getElementById("typed-input");
const voiceWarning = document.getElementById("voice-warning");
const conversation = document.getElementById("conversation");

const canvas = document.getElementById("oscilloscope");
const ctx = canvas.getContext("2d");

let speaking = false;
let speakPhase = 0;
let idlePhase = 0;

let ws = null;
let lastTranscript = "";
let reconnectAttempts = 0;
let shouldReconnect = true;
let silkChunks = [];

let deepgramEnabled = false;
let deepgramSocket = null;
let mediaRecorder = null;
let keepAliveInterval = null;
let micActive = false;
let micStream = null;

const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY_MS = 2000;

function drawOscilloscope() {
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    ctx.fillStyle = "#080c10";
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "rgba(0, 212, 255, 0.06)";
    ctx.lineWidth = 1;
    for (let y = 0; y <= H; y += H / 4) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
    }

    ctx.beginPath();
    ctx.strokeStyle = speaking ? "#00d4ff" : "rgba(0, 212, 255, 0.35)";
    ctx.lineWidth = speaking ? 1.5 : 1;
    ctx.shadowBlur = speaking ? 8 : 0;
    ctx.shadowColor = "#00d4ff";

    const points = 64;
    for (let i = 0; i <= points; i++) {
        const x = (i / points) * W;
        let y = H / 2;

        if (speaking) {
            const t = speakPhase + i * 0.18;
            y = H / 2
                + Math.sin(t) * (H * 0.28)
                + Math.sin(t * 2.3 + 1.2) * (H * 0.12)
                + Math.sin(t * 0.7 + 2.4) * (H * 0.08);
        } else {
            const t = idlePhase + i * 0.25;
            y = H / 2 + Math.sin(t) * (H * 0.04);
        }

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    if (speaking) speakPhase += 0.09;
    else idlePhase += 0.008;

    requestAnimationFrame(drawOscilloscope);
}

function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;");
}

function addExchange(query, response, isError = false) {
    const div = document.createElement("div");
    div.className = "exchange" + (isError ? " exchange-error" : "");

    if (query) {
        const q = document.createElement("div");
        q.className = "exchange-query";
        q.innerHTML = `<span class="exchange-prefix">YOU</span><span class="exchange-text">${escapeHtml(query)}</span>`;
        div.appendChild(q);
    }

    const r = document.createElement("div");
    r.className = "exchange-response";
    r.innerHTML = `<span class="exchange-prefix">INF</span><span class="exchange-text">${escapeHtml(response)}</span>`;
    div.appendChild(r);

    conversation.appendChild(div);
    conversation.scrollTop = conversation.scrollHeight;
}

function setSpeaking(active) {
    speaking = active;
}

function setWsStatus(connected) {
    const dot = document.getElementById("ws-dot");
    const label = document.getElementById("ws-label");
    if (connected) {
        dot.classList.add("connected");
        label.textContent = "ONLINE";
    } else {
        dot.classList.remove("connected");
        label.textContent = "OFFLINE";
    }
}

function setTtsBackend(name) {
    document.getElementById("tts-label").textContent = `TTS: ${String(name).toUpperCase()}`;
}

function setContextStats(payload) {
    if (payload.terminal_lines != null)
        document.getElementById("stat-terminal").textContent = payload.terminal_lines + " lines";
    if (payload.active_file != null)
        document.getElementById("stat-file").textContent = payload.active_file || "—";
    if (payload.error_count != null)
        document.getElementById("stat-errors").textContent = payload.error_count;
}

function triggerErrorBadge() {
    const badge = document.getElementById("error-badge");
    badge.classList.remove("hidden");
    window.setTimeout(() => badge.classList.add("hidden"), 10000);
}

async function loadBrowserConfig() {
    try {
        const res = await fetch("/config/browser");
        const cfg = await res.json();
        deepgramEnabled = cfg.deepgram_enabled === true;
        if (cfg.tts_backend) {
            setTtsBackend(cfg.tts_backend);
        }
        if (!deepgramEnabled) {
            voiceWarning.textContent = "DEEPGRAM_API_KEY not set. Use text input.";
            voiceWarning.classList.remove("hidden");
            micBtn.classList.add("hidden");
        }
    } catch (error) {
        console.error("Failed to load browser config:", error);
        voiceWarning.textContent = "Runtime config unavailable. Use text input.";
        voiceWarning.classList.remove("hidden");
        micBtn.classList.add("hidden");
    }
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

    let totalSamples = 0;
    for (const chunk of chunks) totalSamples += chunk.length / 2;

    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) {
        document.dispatchEvent(new CustomEvent("silkPlaybackEnd"));
        return;
    }

    const audioCtx = new AudioContextCtor({ sampleRate: 24000 });
    const buffer = audioCtx.createBuffer(1, totalSamples, 24000);
    const channelData = buffer.getChannelData(0);

    let offset = 0;
    for (const chunk of chunks) {
        const int16 = new Int16Array(
            chunk.buffer,
            chunk.byteOffset,
            chunk.byteLength / 2
        );
        for (let i = 0; i < int16.length; i++) {
            channelData[offset++] = int16[i] / 32768;
        }
    }

    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);
    source.onended = () => {
        document.dispatchEvent(new CustomEvent("silkPlaybackEnd"));
        void audioCtx.close();
    };
    source.start();
}

function speakInBrowser(text) {
    try {
        const utterance = new SpeechSynthesisUtterance(text);
        setSpeaking(true);
        utterance.onend = () => {
            setSpeaking(false);
        };
        utterance.onerror = () => {
            setSpeaking(false);
        };
        window.speechSynthesis.speak(utterance);
    } catch (err) {
        setSpeaking(false);
    }
}

function showReconnectFailure() {
    addExchange("", "Connection lost. Restart inferr.", true);
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
        addExchange(lastTranscript, text);
        setTtsBackend(backend);

        if (hasErrors) {
            triggerErrorBadge();
        }

        if (payload.context_stats) {
            setContextStats(payload.context_stats);
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
        addExchange("", String(payload.message || "Unknown error"), true);
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
                setSpeaking(true);
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
        addExchange("", "WebSocket is not connected.", true);
        return;
    }
    lastTranscript = trimmed;
    ws.send(JSON.stringify({ type: "transcript", payload: trimmed }));
}

async function startMic() {
    if (!deepgramEnabled) {
        return;
    }

    let stream;
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
        voiceWarning.textContent = "Microphone access denied.";
        voiceWarning.classList.remove("hidden");
        return;
    }
    micStream = stream;

    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const sttUrl = `${scheme}://${window.location.host}/ws/stt`;
    deepgramSocket = new WebSocket(sttUrl);
    deepgramSocket.binaryType = "arraybuffer";

    deepgramSocket.onopen = () => {
        micActive = true;
        micBtn.classList.add("listening");

        try {
            mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
        } catch (error) {
            mediaRecorder = new MediaRecorder(stream);
        }

        mediaRecorder.addEventListener("dataavailable", (event) => {
            if (event.data.size > 0 && deepgramSocket.readyState === WebSocket.OPEN) {
                deepgramSocket.send(event.data);
            }
        });
        mediaRecorder.start(250);

        keepAliveInterval = window.setInterval(() => {
            if (deepgramSocket && deepgramSocket.readyState === WebSocket.OPEN) {
                deepgramSocket.send(JSON.stringify({ type: "KeepAlive" }));
            }
        }, 8000);
    };

    deepgramSocket.onmessage = (event) => {
        if (typeof event.data !== "string") {
            return;
        }
        let data;
        try {
            data = JSON.parse(event.data);
        } catch {
            return;
        }

        if (
            data.type === "Results" &&
            data.is_final === true &&
            data.channel?.alternatives?.[0]?.transcript
        ) {
            const transcript = data.channel.alternatives[0].transcript.trim();
            if (transcript) {
                sendTranscript(transcript);
                stopMic();
            }
            return;
        }

        if (data.type === "error" && data.message) {
            addExchange("", String(data.message), true);
            stopMic();
        }
    };

    deepgramSocket.onerror = () => stopMic();
    deepgramSocket.onclose = () => {
        micActive = false;
        micBtn.classList.remove("listening");
    };
}

function stopMic() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach((track) => track.stop());
    } else if (micStream) {
        micStream.getTracks().forEach((track) => track.stop());
    }
    if (keepAliveInterval !== null) {
        clearInterval(keepAliveInterval);
        keepAliveInterval = null;
    }
    if (deepgramSocket) {
        if (deepgramSocket.readyState === WebSocket.OPEN) {
            deepgramSocket.send(JSON.stringify({ type: "CloseStream" }));
        }
        if (deepgramSocket.readyState === WebSocket.OPEN || deepgramSocket.readyState === WebSocket.CONNECTING) {
            deepgramSocket.close();
        }
    }
    micActive = false;
    micBtn.classList.remove("listening");
    mediaRecorder = null;
    deepgramSocket = null;
    micStream = null;
}

micBtn.addEventListener("click", () => {
    if (micActive) {
        stopMic();
    } else {
        void startMic();
    }
});

sendBtn.addEventListener("click", () => {
    sendTranscript(typedInput.value);
    typedInput.value = "";
    typedInput.style.height = "auto";
});

typedInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendTranscript(typedInput.value);
        typedInput.value = "";
        typedInput.style.height = "auto";
    }
});

typedInput.addEventListener("input", () => {
    typedInput.style.height = "auto";
    typedInput.style.height = Math.min(typedInput.scrollHeight, 120) + "px";
});

document.addEventListener("silkPlaybackEnd", () => {
    setSpeaking(false);
});

drawOscilloscope();
connectWebSocket();
void loadBrowserConfig();

setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping", payload: "" }));
    }
}, 30000);
