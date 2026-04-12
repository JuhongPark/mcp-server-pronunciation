# mcp-server-pronunciation

<!-- mcp-name: io.github.juhongpark/pronunciation -->

An MCP (Model Context Protocol) server that lets you **talk to Claude by voice while getting English pronunciation, grammar, and fluency feedback** in the same turn. Use it for casual voice chat with light coaching, or switch to drill mode when you want to practice a specific sentence.

Built for Claude Desktop, Claude Code, and any other MCP client. Everything runs locally — audio is captured with your mic, transcribed by [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) on-device, and never leaves your machine.

`mcp-name: io.github.juhongpark/pronunciation`

## Why

Voice MCP servers today treat speech as a typing replacement. English tutor MCP servers are text-only. This one combines the two: you speak freely, Claude replies, and feedback on what you just said (pronunciation, grammar, fluency) surfaces inside the same tool call so Claude can weave it into a natural reply — or stay out of the way when you're just chatting.

## Features

- **Voice conversation** with Claude. Speak, auto-stop on silence, Claude reads your transcript and responds.
- **Inline English feedback**: pronunciation (word-level clarity + Korean-speaker tips), grammar (common irregular-verb errors), and fluency (pace + long pauses).
- **Drill mode** (`practice`, `quick_practice`, `retry`) for focused sentence practice.
- **Local-only**: Whisper model runs on your machine, audio never leaves it.
- **Cross-platform**: macOS, Linux, Windows, and WSL2 (recording auto-routes through Windows).
- **Fast startup**: lazy imports + background model pre-load keep the MCP handshake under a second.

## Requirements

- Python 3.11+
- A working microphone
- ~150 MB disk space for the default Whisper model (`base.en`)
- MCP spec: targets `2025-06-18` via the official Python SDK (`mcp>=1.2`)

## Installation

```bash
# Recommended: uvx (no global install, cached between runs)
uvx mcp-server-pronunciation

# Or install as a uv tool
uv tool install mcp-server-pronunciation

# Or pip
pip install mcp-server-pronunciation
```

### Linux: install PortAudio first

`sounddevice` ships PortAudio inside the wheel on macOS and Windows, but on Linux you need the system library:

```bash
# Debian / Ubuntu
sudo apt-get install libportaudio2

# Fedora / RHEL
sudo dnf install portaudio

# Arch
sudo pacman -S portaudio

# PipeWire-only systems may also need
sudo apt-get install pipewire-alsa
```

### First-time check

Before wiring the server into Claude, run the preflight:

```bash
uvx mcp-server-pronunciation doctor
```

Optional — pre-download the Whisper model (~150 MB) so the first call is instant:

```bash
uvx mcp-server-pronunciation pull-model base.en
```

## Add to your MCP client

### Claude Code

```bash
claude mcp add pronunciation -- uvx mcp-server-pronunciation
```

### Claude Desktop

Edit `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pronunciation": {
      "command": "uvx",
      "args": ["mcp-server-pronunciation"]
    }
  }
}
```

On macOS, if Claude Desktop can't find `uvx` (`spawn uvx ENOENT`), use an absolute path. Find it with `which uvx` in your terminal.

### Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "pronunciation": {
      "command": "uvx",
      "args": ["mcp-server-pronunciation"]
    }
  }
}
```

### VS Code (with MCP support)

Add to `.vscode/mcp.json` or your user settings:

```json
{
  "servers": {
    "pronunciation": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-pronunciation"]
    }
  }
}
```

## Usage Examples

### 1. Voice chat with feedback

> **You**: "Let's have a voice chat. I'll ask you about the weekend. Use the converse tool."
>
> **Claude** (calls `converse`): *records your speech, transcribes it, notes that you said "buyed" instead of "bought"*
>
> **Claude**: "Oh nice — what kind of apples did you **buy**? And by the way, the past tense of 'buy' is 'bought' — small thing, but I noticed it."

### 2. Drill a specific sentence

> **You**: "Give me a sentence to practice with 'th' sounds."
>
> **Claude** (calls `suggest_sentence` with `focus=th`): "Try this: *The three brothers thought thoroughly about their future.*"
>
> **You**: "Record me reading it."
>
> **Claude** (calls `practice` with that reference): *returns word-level assessment with Korean-speaker tips*

### 3. Retry after feedback

> **You**: "Let me try again."
>
> **Claude** (calls `retry`): *re-records the same target sentence and compares*

## Tools

| Tool | Purpose |
|---|---|
| **`converse`** | **Primary**. Record + transcribe + quick feedback + "For Claude" guidance for natural voice-chat-with-coaching. |
| `practice` | Drill mode: record user reading a specific reference sentence, return detailed assessment. |
| `quick_practice` | Pick a random sentence (by phoneme focus + difficulty) and drill it. |
| `retry` | Re-record the last sentence the user was practicing. |
| `suggest_sentence` | Return a practice sentence without recording. |
| `record` | Record audio and save a WAV file (raw, no analysis). |
| `assess` | Assess the last recording (or a specified WAV) without re-recording. |
| `check_mic` | List available audio input devices. |

## Configuration

### Whisper model

Set `MCP_PRONUNCIATION_MODEL` to pick a different model size:

```bash
# Default — fast, English-only (~150 MB)
export MCP_PRONUNCIATION_MODEL=base.en

# Smaller / faster (~75 MB)
export MCP_PRONUNCIATION_MODEL=tiny.en

# More accurate (~470 MB)
export MCP_PRONUNCIATION_MODEL=small.en

# Multilingual options (larger)
export MCP_PRONUNCIATION_MODEL=small
export MCP_PRONUNCIATION_MODEL=medium
```

Available: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v3`, `large-v3-turbo`. For English-only use, the `.en` variants are faster and more accurate at a given size.

GPU (CUDA 12 + cuDNN 9) is auto-detected when available; otherwise runs on CPU with int8 quantization.

### Cache location

By default Whisper weights are cached in `~/.cache/huggingface/hub/`. Override with `HF_HUB_CACHE`:

```bash
export HF_HUB_CACHE=/path/to/cache
```

### Model override in Claude Code

```bash
claude mcp add pronunciation -e MCP_PRONUNCIATION_MODEL=small.en -- uvx mcp-server-pronunciation
```

## Platform Support

| Platform | Recording method | Status |
|----------|------------------|--------|
| macOS | sounddevice (bundled PortAudio) | Supported |
| Linux | sounddevice (needs `libportaudio2`) | Supported |
| Windows | sounddevice (bundled PortAudio) | Supported |
| WSL2 | PowerShell MCI (winmm.dll) | Supported |

**WSL2 note**: WSLg's PulseAudio does not forward microphone audio from the Windows host. This server detects WSL2 automatically and records through PowerShell on the Windows side instead.

## Troubleshooting

### `uvx mcp-server-pronunciation doctor` is your first stop

It reports on PortAudio, input devices, Whisper model cache, free disk space, and Python version. Run it whenever something feels off.

### `sounddevice` import fails on Linux

You're missing `libportaudio2`. See the install section above. After installing:

```bash
uvx mcp-server-pronunciation doctor
```

### No audio captured / empty recording

- **macOS**: System Settings → Privacy & Security → Microphone. Grant access to the app that launched Claude Desktop / Claude Code.
- **Linux**: Check `pavucontrol` (PulseAudio) or `pw-cli list-objects` (PipeWire) for input levels. On PipeWire-only systems, install `pipewire-alsa`.
- **WSL2**: Test your mic in Windows Settings → Sound → Input. The server records through Windows, not through WSLg.

### First run is slow

The Whisper model downloads on first use (~150 MB for `base.en`). Pre-download it once:

```bash
uvx mcp-server-pronunciation pull-model base.en
```

Subsequent runs reuse the cached weights. If startup still feels slow, try `MCP_PRONUNCIATION_MODEL=tiny.en`.

### Claude Desktop on macOS: `spawn uvx ENOENT`

Claude Desktop launches MCP servers from a GUI-only environment without `~/.local/bin` on PATH. Use the absolute path to `uvx` in your config (`/Users/YOU/.local/bin/uvx` or wherever `which uvx` reports).

## Privacy

- All audio processing happens **locally** on your machine.
- Recordings are temporary `.wav` files under your system temp directory (`$TMPDIR`) and are deleted when the OS cleans them up.
- The Whisper model runs locally — no audio data is sent to any external service.
- No telemetry. No analytics. No network calls except the one-time Whisper weight download from Hugging Face.

## Development

```bash
git clone https://github.com/JuhongPark/mcp-server-pronunciation.git
cd mcp-server-pronunciation
uv sync --extra dev
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

## Support

Issues: https://github.com/JuhongPark/mcp-server-pronunciation/issues

## License

MIT. See [LICENSE](LICENSE).

Third-party components (all MIT / permissive):
- [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) — MIT
- [OpenAI Whisper models](https://github.com/openai/whisper) — MIT
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) — MIT
- [`sounddevice`](https://python-sounddevice.readthedocs.io/) — MIT
- [PortAudio](http://www.portaudio.com/) — MIT
