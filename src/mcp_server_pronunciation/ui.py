"""HTML resources for MCP Apps-compatible clients."""

VOICE_PANEL_URI = "ui://pronunciation/voice-panel"

VOICE_PANEL_RESOURCE_META = {
    "io.modelcontextprotocol/ui": {
        "type": "app",
        "permissions": ["microphone"],
    },
    "openai/widgetDescription": "Local pronunciation voice capture panel.",
    "openai/widgetPrefersBorder": True,
}

VOICE_PANEL_TOOL_META = {
    "openai/outputTemplate": VOICE_PANEL_URI,
    "io.modelcontextprotocol/ui": {
        "resourceUri": VOICE_PANEL_URI,
        "permissions": ["microphone"],
    },
}

VOICE_PANEL_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #141821;
      --muted: #5b6472;
      --line: #d8dee9;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --danger: #b42318;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #101318;
        --panel: #171b22;
        --text: #eef2f7;
        --muted: #a7b0bf;
        --line: #303846;
        --accent: #2dd4bf;
        --accent-strong: #5eead4;
        --danger: #fb7185;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      display: grid;
      gap: 12px;
      padding: 14px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    button {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--text);
      font: inherit;
      cursor: pointer;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #ffffff;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    .status,
    .result {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }
    .value {
      margin-top: 4px;
      font-size: 16px;
      font-weight: 650;
    }
    .meter {
      height: 12px;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: color-mix(in srgb, var(--line) 30%, transparent);
    }
    .bar {
      width: 0%;
      height: 100%;
      background: var(--accent);
      transition: width 120ms linear;
    }
    .transcript {
      min-height: 44px;
      margin-top: 8px;
      color: var(--text);
      white-space: pre-wrap;
    }
    .feedback {
      margin-top: 8px;
      color: var(--muted);
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <main>
    <section class="toolbar">
      <button class="primary" id="record" type="button">Record</button>
      <button id="stop" type="button" disabled>Stop</button>
      <button id="clear" type="button">Clear</button>
    </section>
    <section class="status">
      <div class="label">Status</div>
      <div class="value" id="status">Ready</div>
    </section>
    <section class="status">
      <div class="label">Input</div>
      <div class="meter"><div class="bar" id="level"></div></div>
    </section>
    <section class="result">
      <div class="label">Transcript</div>
      <div class="transcript" id="transcript"></div>
      <div class="label">Feedback</div>
      <div class="feedback" id="feedback"></div>
    </section>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const levelEl = document.getElementById("level");
    const transcriptEl = document.getElementById("transcript");
    const feedbackEl = document.getElementById("feedback");
    const recordButton = document.getElementById("record");
    const stopButton = document.getElementById("stop");
    let audioContext = null;
    let processor = null;
    let source = null;
    let stream = null;
    let chunks = [];
    let inputSampleRate = 48000;

    function setStatus(value) {
      statusEl.textContent = value;
    }

    function flatten(parts) {
      const length = parts.reduce((total, part) => total + part.length, 0);
      const out = new Float32Array(length);
      let offset = 0;
      for (const part of parts) {
        out.set(part, offset);
        offset += part.length;
      }
      return out;
    }

    function resample(samples, fromRate, toRate) {
      if (fromRate === toRate) return samples;
      const ratio = fromRate / toRate;
      const length = Math.round(samples.length / ratio);
      const out = new Float32Array(length);
      for (let i = 0; i < length; i += 1) {
        const sourceIndex = i * ratio;
        const left = Math.floor(sourceIndex);
        const right = Math.min(samples.length - 1, left + 1);
        const weight = sourceIndex - left;
        out[i] = samples[left] * (1 - weight) + samples[right] * weight;
      }
      return out;
    }

    function encodeWav(floatSamples, sampleRate) {
      const bytesPerSample = 2;
      const buffer = new ArrayBuffer(44 + floatSamples.length * bytesPerSample);
      const view = new DataView(buffer);
      const writeString = (offset, text) => {
        for (let i = 0; i < text.length; i += 1) {
          view.setUint8(offset + i, text.charCodeAt(i));
        }
      };
      writeString(0, "RIFF");
      view.setUint32(4, 36 + floatSamples.length * bytesPerSample, true);
      writeString(8, "WAVE");
      writeString(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * bytesPerSample, true);
      view.setUint16(32, bytesPerSample, true);
      view.setUint16(34, 16, true);
      writeString(36, "data");
      view.setUint32(40, floatSamples.length * bytesPerSample, true);
      let offset = 44;
      for (const sample of floatSamples) {
        const clamped = Math.max(-1, Math.min(1, sample));
        view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
        offset += 2;
      }
      return new Uint8Array(buffer);
    }

    function bytesToBase64(bytes) {
      let binary = "";
      const chunkSize = 0x8000;
      for (let i = 0; i < bytes.length; i += chunkSize) {
        binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
      }
      return btoa(binary);
    }

    async function callTool(name, args) {
      if (window.openai && typeof window.openai.callTool === "function") {
        return window.openai.callTool(name, args);
      }
      if (window.mcp && typeof window.mcp.callTool === "function") {
        return window.mcp.callTool(name, args);
      }
      throw new Error("MCP Apps tool bridge is unavailable.");
    }

    function unwrapResult(result) {
      if (result && result.structuredContent) return result.structuredContent;
      if (result && result.result && result.result.structuredContent) {
        return result.result.structuredContent;
      }
      return result || {};
    }

    async function startRecording() {
      chunks = [];
      transcriptEl.textContent = "";
      feedbackEl.textContent = "";
      levelEl.style.width = "0%";
      setStatus("Recording");
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      audioContext = new AudioContext();
      inputSampleRate = audioContext.sampleRate;
      source = audioContext.createMediaStreamSource(stream);
      processor = audioContext.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = event => {
        const input = event.inputBuffer.getChannelData(0);
        const copy = new Float32Array(input.length);
        copy.set(input);
        chunks.push(copy);
        let sum = 0;
        for (const value of input) sum += value * value;
        const rms = Math.sqrt(sum / input.length);
        levelEl.style.width = `${Math.min(100, Math.round(rms * 600))}%`;
      };
      source.connect(processor);
      processor.connect(audioContext.destination);
      recordButton.disabled = true;
      stopButton.disabled = false;
    }

    async function stopRecording() {
      stopButton.disabled = true;
      recordButton.disabled = false;
      if (processor) processor.disconnect();
      if (source) source.disconnect();
      if (stream) stream.getTracks().forEach(track => track.stop());
      if (audioContext) await audioContext.close();
      processor = null;
      source = null;
      stream = null;
      audioContext = null;
      setStatus("Analyzing");
      const samples = resample(flatten(chunks), inputSampleRate, 16000);
      const wav = encodeWav(samples, 16000);
      const result = unwrapResult(await callTool("analyze_uploaded_audio", {
        wav_base64: bytesToBase64(wav),
        mode: "conversation",
        reference_text: null
      }));
      setStatus(result.status || "Done");
      transcriptEl.textContent = result.transcript || "";
      feedbackEl.textContent = result.report_markdown || result.error || "";
    }

    recordButton.addEventListener("click", async () => {
      try {
        await startRecording();
      } catch (error) {
        setStatus("Error");
        feedbackEl.textContent = error.message || String(error);
        recordButton.disabled = false;
        stopButton.disabled = true;
      }
    });
    stopButton.addEventListener("click", async () => {
      try {
        await stopRecording();
      } catch (error) {
        setStatus("Error");
        feedbackEl.textContent = error.message || String(error);
      }
    });
    document.getElementById("clear").addEventListener("click", () => {
      statusEl.textContent = "Ready";
      transcriptEl.textContent = "";
      feedbackEl.textContent = "";
      levelEl.style.width = "0%";
    });
  </script>
</body>
</html>
"""
