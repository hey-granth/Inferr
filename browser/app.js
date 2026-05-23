const wsDot = document.getElementById("ws-dot");
const waveform = document.getElementById("waveform");
const micBtn = document.getElementById("mic-btn");
const sendBtn = document.getElementById("send-btn");
const typedInput = document.getElementById("typed-input");
const lastExchange = document.getElementById("last-exchange");
const voiceWarning = document.getElementById("voice-warning");

let ws = null;
let recognition = null;
let listening = false;
let lastTranscript = "";

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
    } else {
        wsDot.classList.remove("connected");
        wsDot.classList.add("disconnected");
    }
}

function connectWebSocket() {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${scheme}://${window.location.host}/ws`;
    ws = new WebSocket(wsUrl);

    ws.addEventListener("open", () => setWsStatus(true));
    ws.addEventListener("close", () => setWsStatus(false));
    ws.addEventListener("error", () => setWsStatus(false));

    ws.addEventListener("message", (event) => {
        let payload = null;
        try {
            payload = JSON.parse(event.data);
        } catch (err) {
            return;
        }

        if (payload.type === "response") {
            const text = String(payload.text || "");
            lastExchange.innerHTML =
                `<div><strong>You:</strong> ${escapeHtml(lastTranscript)}</div>` +
                `<div><strong>Inferr:</strong> ${escapeHtml(text)}</div>`;

            try {
                const utter = new SpeechSynthesisUtterance(text);
                window.speechSynthesis.speak(utter);
            } catch (err) {
                // Ignore TTS failures in the browser stub.
            }

            waveform.classList.add("speaking");
            setTimeout(() => waveform.classList.remove("speaking"), 3000);
        }

        if (payload.type === "error") {
            lastExchange.innerHTML = `<div class="error">${escapeHtml(
                String(payload.message || "Unknown error")
            )}</div>`;
        }
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

connectWebSocket();
setupSpeechRecognition();

setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping", payload: "" }));
    }
}, 30000);
