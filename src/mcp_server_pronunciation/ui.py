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
    document.getElementById("record").addEventListener("click", () => {
      statusEl.textContent = "Panel loaded";
      feedbackEl.textContent = "Browser recording support will be enabled by the MCP host integration.";
      levelEl.style.width = "0%";
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
