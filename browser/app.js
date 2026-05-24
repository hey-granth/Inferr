const wsDot = document.getElementById("ws-dot");
const micBtn = document.getElementById("mic-btn");
const sendBtn = document.getElementById("send-btn");
const typedInput = document.getElementById("typed-input");
const voiceWarning = document.getElementById("voice-warning");
const conversation = document.getElementById("conversation");

const canvas = document.getElementById("oscilloscope");
const ctx = canvas.getContext("2d");

let speaking = false;
let idlePhase = 0;
let audioLevel = 0;

let ws = null;
let lastTranscript = "";
let reconnectAttempts = 0;
let shouldReconnect = true;

let deepgramEnabled = false;
let deepgramSocket = null;
let keepAliveInterval = null;
let sttSampleRate = 16000;
let micActive = false;
let _micStarting = false;  // Guard: prevents duplicate startMic() calls
let micStream = null;
let silkAudioContext = null;
let silkAnalyser = null;
let silkLevelRaf = null;
let silkChunkCount = 0;
let silkChunkBytes = 0;
let silkFirstChunkAt = null;
let silkResponseStartedAt = null;

// Self-transcription guard: while the assistant is speaking, suppress incoming
// Deepgram transcripts so the assistant doesn't hear and respond to its own voice.
let assistantSpeaking = false;

// Progressive Silk audio player — plays each PCM chunk as it arrives
// instead of waiting for the full response before playback begins.
const silkPlayer = (() => {
    let _ctx = null;
    let _nextStartTime = 0;
    let _analyser = null;
    let _levelRaf = null;
    let _activeSources = 0;
    let _finalizePending = false;
    const SAMPLE_RATE = 24000;
    // Tail padding so the last PCM frame clears the analyser/output node.
    // Increased to 0.7s to prevent final-word cutoff on slower systems.
    const OUTPUT_TAIL_SEC = 0.7;

    function _ensureContext() {
        if (!_ctx || _ctx.state === "closed") {
            const Ctor = window.AudioContext || window.webkitAudioContext;
            if (!Ctor) return false;
            _ctx = new Ctor({ sampleRate: SAMPLE_RATE });
            _analyser = _ctx.createAnalyser();
            _analyser.fftSize = 1024;
            _analyser.connect(_ctx.destination);
            _nextStartTime = 0;
        }
        return true;
    }

    function _startLevelMonitor() {
        if (_levelRaf !== null) return;
        const buf = new Uint8Array(_analyser.fftSize);
        const tick = () => {
            if (!_analyser) return;
            _analyser.getByteTimeDomainData(buf);
            let sum = 0;
            for (let i = 0; i < buf.length; i++) {
                const n = (buf[i] - 128) / 128;
                sum += n * n;
            }
            audioLevel = Math.min(0.35, Math.sqrt(sum / buf.length) * 2.0);
            _levelRaf = window.requestAnimationFrame(tick);
        };
        _levelRaf = window.requestAnimationFrame(tick);
    }

    function _stopLevelMonitor() {
        if (_levelRaf !== null) {
            window.cancelAnimationFrame(_levelRaf);
            _levelRaf = null;
        }
    }

    function _maybePlaybackIdle() {
        if (!_ctx || _activeSources > 0) {
            return;
        }
        if (_ctx.currentTime + 0.02 < _nextStartTime) {
            return;
        }
        setSpeaking(false);
        _stopLevelMonitor();
        document.dispatchEvent(new CustomEvent("silkPlaybackEnd"));
    }

    return {
        // Feed a raw PCM-16LE Uint8Array chunk — schedules it for gapless playback
        pushChunk(chunk) {
            if (!_ensureContext()) return;
            const samples = chunk.byteLength / 2;
            if (samples < 1) return;

            const audioBuffer = _ctx.createBuffer(1, samples, SAMPLE_RATE);
            const channelData = audioBuffer.getChannelData(0);
            const int16 = new Int16Array(chunk.buffer, chunk.byteOffset, samples);
            for (let i = 0; i < samples; i++) {
                channelData[i] = int16[i] / 32768;
            }

            const source = _ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(_analyser);

            // Schedule gaplessly: start immediately if behind currentTime
            const startAt = Math.max(_ctx.currentTime + 0.01, _nextStartTime);
            source.start(startAt);
            _nextStartTime = startAt + audioBuffer.duration;
            _activeSources += 1;

            setSpeaking(true);
            _startLevelMonitor();

            source.onended = () => {
                _activeSources = Math.max(0, _activeSources - 1);
                if (!_finalizePending) {
                    _maybePlaybackIdle();
                }
            };
        },

        // Call when silk_end received: close the AudioContext after playback drains
        finalize() {
            if (!_ctx) return;
            _finalizePending = true;
            const ctx = _ctx;
            const scheduledEnd = _nextStartTime;
            const drainDelay =
                Math.max(0, scheduledEnd - ctx.currentTime) + OUTPUT_TAIL_SEC;

            const finish = () => {
                if (_activeSources > 0) {
                    window.setTimeout(finish, 50);
                    return;
                }
                _stopLevelMonitor();
                if (ctx.state !== "closed") {
                    void ctx.close();
                }
                if (_ctx === ctx) {
                    _ctx = null;
                    _analyser = null;
                    _nextStartTime = 0;
                }
                _finalizePending = false;
                setSpeaking(false);
                document.dispatchEvent(new CustomEvent("silkPlaybackEnd"));
            };

            window.setTimeout(finish, drainDelay * 1000);
        },

        // Hard reset on new utterance
        reset() {
            _finalizePending = false;
            _activeSources = 0;
            _stopLevelMonitor();
            if (_ctx && _ctx.state !== "closed") {
                void _ctx.close();
            }
            _ctx = null;
            _analyser = null;
            _nextStartTime = 0;
        }
    };
})();

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
    ctx.lineWidth = speaking ? 1.2 : 1;
    ctx.shadowBlur = speaking ? 2 : 0;
    ctx.shadowColor = "#00d4ff";

    const points = 64;
    const effectiveLevel = speaking ? Math.max(0.05, audioLevel) : 0.02;
    for (let i = 0; i <= points; i++) {
        const x = (i / points) * W;
        const t = idlePhase + i * 0.18;
        const y = H / 2 + Math.sin(t) * (H * effectiveLevel * 0.35);

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    idlePhase += speaking ? 0.06 : 0.01;
    audioLevel = Math.max(0, audioLevel * 0.92);

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
    if (!active) {
        audioLevel = 0;
    }
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
        if (cfg.stt_sample_rate) {
            sttSampleRate = Number(cfg.stt_sample_rate) || 16000;
        }
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



function speakInBrowser(text) {
    try {
        const utterance = new SpeechSynthesisUtterance(text);
        setSpeaking(true);
        audioLevel = 0.05;
        utterance.onend = () => {
            setSpeaking(false);
            // Browser TTS finished — re-enable transcript processing
            if (assistantSpeaking) {
                assistantSpeaking = false;
                console.log("[STT] assistantSpeaking cleared — browser TTS ended");
            }
        };
        utterance.onerror = () => {
            setSpeaking(false);
            if (assistantSpeaking) {
                assistantSpeaking = false;
                console.log("[STT] assistantSpeaking cleared — browser TTS error");
            }
        };
        window.speechSynthesis.speak(utterance);
    } catch (err) {
        setSpeaking(false);
        assistantSpeaking = false;
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

function logPipelineStage(payload) {
    const ok = payload.ok !== false;
    const tag = ok ? "OK" : "FAILED";
    const detail = payload.detail ? JSON.stringify(payload.detail) : "";
    const err = payload.error ? ` — ${payload.error}` : "";
    console.log(`[PIPELINE] ${payload.stage}: ${tag}${err}`, detail);
    document.dispatchEvent(new CustomEvent("inferrPipelineStage", { detail: payload }));
}

function handleTextMessage(payload) {
    if (payload.type === "pipeline_stage") {
        logPipelineStage(payload);
        return;
    }

    if (payload.type === "response") {
        logPipelineStage({
            stage: "PLAYBACK_STARTED",
            ok: true,
            detail: { backend: payload.tts_backend || "browser" },
        });
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

        // Reset player for new utterance before chunks begin arriving
        if (backend === "silk") {
            // Set guard BEFORE playback begins so Deepgram transcripts are suppressed
            assistantSpeaking = true;
            console.log("[STT] assistantSpeaking = true (silk TTS starting)");
            silkPlayer.reset();
            silkChunkCount = 0;
            silkChunkBytes = 0;
            silkFirstChunkAt = null;
            silkResponseStartedAt = Date.now();
        } else if (backend === "browser") {
            assistantSpeaking = true;
            console.log("[STT] assistantSpeaking = true (browser TTS starting)");
            speakInBrowser(text);
        }
    }

    if (payload.type === "silk_end") {
        // Drain and close context after all queued chunks have played out
        silkPlayer.finalize();
        if (silkResponseStartedAt !== null) {
            const totalMs = Date.now() - silkResponseStartedAt;
            console.info("[inferr] silk_end", {
                chunkCount: silkChunkCount,
                totalBytes: silkChunkBytes,
                totalMs
            });
        }
    }

    if (payload.type === "diagnostic") {
        console.info("[inferr] diagnostic", payload);
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
            // Feed chunk directly to the progressive player — no buffering
            const chunk = new Uint8Array(arrayBuffer);
            silkChunkCount += 1;
            silkChunkBytes += chunk.byteLength;
            if (silkFirstChunkAt === null) {
                silkFirstChunkAt = Date.now();
            }
            silkPlayer.pushChunk(chunk);
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

function sendTranscript(text, source = "inject") {
    const trimmed = text.trim();
    if (!trimmed) {
        logPipelineStage({
            stage: "TRANSCRIPT_EMPTY",
            ok: false,
            error: "empty transcript",
        });
        return;
    }
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        addExchange("", "WebSocket is not connected.", true);
        return;
    }
    lastTranscript = trimmed;
    logPipelineStage({
        stage: "TRANSCRIPT_FINALIZED",
        ok: true,
        detail: { transcript: trimmed, source },
    });
    ws.send(JSON.stringify({ type: "transcript", payload: trimmed }));
}

const STT_BUFFER_SIZE = 4096;
let sttAudioContext = null;
let sttSourceNode = null;
let sttProcessorNode = null;
let sttSilentGain = null;
let sttCaptureRate = 16000;
let sttFramesSent = 0;
let sttLastPcmLogAt = 0;

// Pending transcript buffer: accumulates is_final text until speech_final arrives.
// Deepgram separates these across two different packets:
//   Packet A: { is_final: true, speech_final: false, transcript: "Hello?" }
//   Packet B: { is_final: true, speech_final: true,  transcript: "" }
// We must buffer A and flush on B.
let pendingTranscript = "";

function float32ToPcm16(float32Array) {
    const pcm16 = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
        const clamped = Math.max(-1, Math.min(1, float32Array[i]));
        pcm16[i] = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
    }
    return pcm16.buffer;
}

function downsampleToSttRate(input, inputRate, outputRate) {
    if (inputRate === outputRate) {
        return input;
    }
    const ratio = inputRate / outputRate;
    const outputLength = Math.floor(input.length / ratio);
    const output = new Float32Array(outputLength);
    for (let i = 0; i < outputLength; i++) {
        output[i] = input[Math.floor(i * ratio)];
    }
    return output;
}

function handleDeepgramMessage(data) {
    if (data.type && data.type !== "Results") {
        console.log("[STT] Deepgram event:", data.type);
    }

    if (data.type !== "Results") {
        if (data.type === "error" && data.message) {
            console.error("[STT] Deepgram error:", data.message);
            addExchange("", String(data.message), true);
            stopMic();
        }
        return;
    }

    const transcript = data.channel?.alternatives?.[0]?.transcript?.trim() || "";
    const isFinal = data.is_final === true;
    const speechFinal = data.speech_final === true;

    console.log(
        "[STT] Results — is_final:", isFinal,
        "| speech_final:", speechFinal,
        "| transcript:", transcript || "(empty)"
    );

    // Step 1: Accumulate transcript text on every is_final packet.
    // Deepgram sends the actual words here, before speech_final fires.
    if (isFinal && transcript) {
        pendingTranscript = transcript;
        console.log("[STT] pendingTranscript updated:", pendingTranscript);
    }

    // Step 2: On speech_final, flush the buffer and invoke the assistant.
    // The speech_final packet itself typically carries an EMPTY transcript —
    // that is normal Deepgram behavior. Use the buffered text instead.
    if (speechFinal) {
        const toSend = pendingTranscript.trim();
        pendingTranscript = "";  // Always clear, even if we don't send

        if (!toSend) {
            console.log("[STT] speech_final received but pendingTranscript is empty — skipping");
            return;
        }

        if (assistantSpeaking) {
            console.log("[STT] Transcript SUPPRESSED (assistantSpeaking):", toSend);
            return;
        }

        logPipelineStage({
            stage: "STT_OK",
            ok: true,
            detail: { transcript: toSend, endpoint: "speech_final" },
        });
        console.log("[STT] speech_final → invoking assistant with buffered:", toSend);
        sendTranscript(toSend, "stt");
    }
}

async function startMic() {
    if (!deepgramEnabled) {
        return;
    }
    if (micActive || _micStarting) {
        console.warn("[STT] startMic() ignored — session already active or starting");
        return;
    }
    _micStarting = true;

    console.log("[STT] Requesting microphone access (PCM16 pipeline)");

    let stream;
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });
        console.log("[STT] Microphone access granted");
    } catch (error) {
        console.error("[STT] Microphone access denied:", error.name, error.message);
        voiceWarning.textContent = "Microphone access denied: " + (error.message || error.name);
        voiceWarning.classList.remove("hidden");
        _micStarting = false;
        return;
    }
    micStream = stream;

    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const sttUrl = `${scheme}://${window.location.host}/ws/stt`;
    console.log("[STT] Connecting to", sttUrl, "| target PCM rate:", sttSampleRate);
    deepgramSocket = new WebSocket(sttUrl);
    deepgramSocket.binaryType = "arraybuffer";

    deepgramSocket.onopen = async () => {
        console.log("[STT] WebSocket /ws/stt opened — starting PCM16 capture");
        micActive = true;
        _micStarting = false;
        micBtn.classList.add("listening");
        sttFramesSent = 0;

        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) {
            console.error("[STT] Web Audio API not available");
            stopMic();
            return;
        }

        try {
            sttAudioContext = new AudioContextClass({ sampleRate: sttSampleRate });
        } catch (error) {
            console.warn("[STT] Could not create AudioContext at", sttSampleRate, "— using default:", error);
            sttAudioContext = new AudioContextClass();
        }

        sttCaptureRate = sttAudioContext.sampleRate;
        if (sttCaptureRate !== sttSampleRate) {
            console.warn(
                "[STT] AudioContext rate", sttCaptureRate,
                "≠ Deepgram rate", sttSampleRate, "— downsampling in capture"
            );
        }

        try {
            await sttAudioContext.resume();
        } catch (error) {
            console.warn("[STT] AudioContext.resume failed:", error);
        }

        sttSourceNode = sttAudioContext.createMediaStreamSource(stream);
        sttProcessorNode = sttAudioContext.createScriptProcessor(STT_BUFFER_SIZE, 1, 1);
        sttSilentGain = sttAudioContext.createGain();
        sttSilentGain.gain.value = 0;

        sttProcessorNode.onaudioprocess = (audioEvent) => {
            if (!micActive) {
                return;
            }
            if (assistantSpeaking) {
                return;
            }

            let samples = audioEvent.inputBuffer.getChannelData(0);
            if (sttCaptureRate !== sttSampleRate) {
                samples = downsampleToSttRate(samples, sttCaptureRate, sttSampleRate);
            }

            let sumSq = 0;
            for (let i = 0; i < samples.length; i++) {
                sumSq += samples[i] * samples[i];
            }
            const rms = Math.sqrt(sumSq / Math.max(samples.length, 1));
            audioLevel = Math.min(0.35, rms * 4);

            const pcmBuffer = float32ToPcm16(samples);
            if (!deepgramSocket || deepgramSocket.readyState !== WebSocket.OPEN) {
                return;
            }

            deepgramSocket.send(pcmBuffer);
            sttFramesSent += 1;

            const now = Date.now();
            if (now - sttLastPcmLogAt >= 2000) {
                sttLastPcmLogAt = now;
                console.log("[STT] PCM streaming", {
                    frameBytes: pcmBuffer.byteLength,
                    framesSent: sttFramesSent,
                    captureRate: sttCaptureRate,
                    deepgramRate: sttSampleRate,
                    rms: rms.toFixed(4),
                });
            }
        };

        sttSourceNode.connect(sttProcessorNode);
        sttProcessorNode.connect(sttSilentGain);
        sttSilentGain.connect(sttAudioContext.destination);

        keepAliveInterval = window.setInterval(() => {
            if (deepgramSocket && deepgramSocket.readyState === WebSocket.OPEN) {
                deepgramSocket.send(JSON.stringify({ type: "KeepAlive" }));
                console.log("[STT] KeepAlive sent");
            }
        }, 8000);

        logPipelineStage({
            stage: "STT_OK",
            ok: true,
            detail: {
                mode: "pcm16_capture",
                encoding: "linear16",
                deepgramSampleRate: sttSampleRate,
                captureSampleRate: sttCaptureRate,
            },
        });
        console.log("[STT] PCM16 pipeline active", {
            encoding: "linear16",
            deepgramSampleRate: sttSampleRate,
            captureSampleRate: sttCaptureRate,
            bufferSamples: STT_BUFFER_SIZE,
        });
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
        handleDeepgramMessage(data);
    };

    deepgramSocket.onerror = (event) => {
        console.error("[STT] WebSocket error:", event);
        _micStarting = false;
        addExchange("", "Microphone connection error. Check console.", true);
        stopMic();
    };

    deepgramSocket.onclose = (event) => {
        console.log("[STT] WebSocket closed — code:", event.code, "reason:", event.reason || "(none)");
        _micStarting = false;
        micActive = false;
        micBtn.classList.remove("listening");
    };
}

function stopMic() {
    console.log("[STT] stopMic() called");
    micActive = false;
    _micStarting = false;
    micBtn.classList.remove("listening");
    pendingTranscript = "";  // Discard any buffered partial utterance

    if (sttProcessorNode) {
        try {
            sttProcessorNode.disconnect();
        } catch (error) {
            console.warn("[STT] processor disconnect:", error);
        }
        sttProcessorNode.onaudioprocess = null;
        sttProcessorNode = null;
    }
    if (sttSourceNode) {
        try {
            sttSourceNode.disconnect();
        } catch (error) {
            console.warn("[STT] source disconnect:", error);
        }
        sttSourceNode = null;
    }
    if (sttSilentGain) {
        try {
            sttSilentGain.disconnect();
        } catch (error) {
            console.warn("[STT] gain disconnect:", error);
        }
        sttSilentGain = null;
    }
    if (sttAudioContext && sttAudioContext.state !== "closed") {
        void sttAudioContext.close().catch((error) => {
            console.warn("[STT] AudioContext close:", error);
        });
        sttAudioContext = null;
    }

    if (micStream) {
        micStream.getTracks().forEach((track) => track.stop());
        micStream = null;
    }

    if (keepAliveInterval !== null) {
        clearInterval(keepAliveInterval);
        keepAliveInterval = null;
    }

    if (deepgramSocket) {
        if (deepgramSocket.readyState === WebSocket.OPEN) {
            try {
                deepgramSocket.send(JSON.stringify({ type: "CloseStream" }));
            } catch (error) {
                console.warn("[STT] CloseStream send failed:", error);
            }
        }
        if (
            deepgramSocket.readyState === WebSocket.OPEN ||
            deepgramSocket.readyState === WebSocket.CONNECTING
        ) {
            try {
                deepgramSocket.close();
            } catch (error) {
                console.warn("[STT] socket close:", error);
            }
        }
        deepgramSocket = null;
    }

    sttFramesSent = 0;
    console.log("[STT] stopMic() complete");
}

micBtn.addEventListener("click", () => {
    if (micActive || _micStarting) {
        // User explicitly disabling the conversation session
        console.log("[STT] User disabled mic — ending conversational session");
        stopMic();
    } else {
        // User enabling ambient conversational mode
        console.log("[STT] User enabled mic — starting conversational session");
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
    logPipelineStage({
        stage: "PLAYBACK_COMPLETED",
        ok: true,
        detail: {
            chunkCount: silkChunkCount,
            totalBytes: silkChunkBytes,
        },
    });
    if (assistantSpeaking) {
        assistantSpeaking = false;
        console.log("[STT] assistantSpeaking cleared — mic listening resumed after TTS");
    }
    if (silkFirstChunkAt !== null) {
        const playbackMs = Date.now() - silkFirstChunkAt;
        console.info("[inferr] silkPlaybackEnd", {
            chunkCount: silkChunkCount,
            totalBytes: silkChunkBytes,
            playbackMs
        });
    }
});

/**
 * Pipeline isolation helpers — use from the browser console during demo debugging.
 * Each method tests one boundary without running the full stack (unless noted).
 */
window.inferrDebug = {
    /** Bypass STT: LLM → TTS → playback (requires /ws + session). */
    invokeAssistant(text) {
        console.log("[inferrDebug] invokeAssistant — bypassing Deepgram STT");
        sendTranscript(String(text || "hello"), "debug_invoke");
    },
    /** Bypass STT + LLM: inject assistant response + run TTS/playback path. */
    injectResponse(text, backend = "silk") {
        console.log("[inferrDebug] injectResponse — bypassing STT and LLM");
        lastTranscript = "[debug inject]";
        handleTextMessage({
            type: "response",
            text: String(text || "hello from inferr debug playback"),
            has_errors: false,
            tts_backend: backend,
            context_stats: {},
        });
    },
    /** Bypass STT, LLM, TTS server: local PCM beep through silkPlayer only. */
    testPlayback() {
        console.log("[inferrDebug] testPlayback — local PCM only");
        assistantSpeaking = true;
        silkPlayer.reset();
        const sampleRate = 24000;
        const durationSec = 0.45;
        const freq = 440;
        const samples = Math.floor(sampleRate * durationSec);
        const pcm = new Int16Array(samples);
        for (let i = 0; i < samples; i++) {
            const t = i / sampleRate;
            pcm[i] = Math.round(Math.sin(2 * Math.PI * freq * t) * 12000);
        }
        const bytes = new Uint8Array(pcm.buffer);
        const chunkBytes = 4096;
        for (let off = 0; off < bytes.length; off += chunkBytes) {
            silkPlayer.pushChunk(bytes.subarray(off, off + chunkBytes));
        }
        window.setTimeout(() => silkPlayer.finalize(), 80);
        logPipelineStage({
            stage: "PLAYBACK_STARTED",
            ok: true,
            detail: { mode: "local_beep", sampleRate },
        });
    },
    async testLlm(transcript = "hello") {
        const res = await fetch("/debug/pipeline/llm", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ transcript }),
        });
        const data = await res.json();
        if (data.pipeline_stage) {
            logPipelineStage(data.pipeline_stage);
        }
        return data;
    },
    async testTts(text = "hello from inferr") {
        const res = await fetch("/debug/pipeline/tts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, tone: "neutral" }),
        });
        return res.json();
    },
    async pipelineStatus() {
        const res = await fetch("/debug/pipeline/status");
        return res.json();
    },
};

drawOscilloscope();
connectWebSocket();
void loadBrowserConfig();

setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping", payload: "" }));
    }
}, 30000);
