# mcp-server-pronunciation

An MCP (Model Context Protocol) server for English pronunciation practice. Record your voice, get it transcribed by Whisper, and receive word-level pronunciation feedback — all from within Claude Code or any MCP client.

## Features

- **Record** audio from your microphone
- **Transcribe** with OpenAI Whisper (via faster-whisper) with word-level timestamps
- **Assess** pronunciation with confidence scores, fluency metrics, and mismatch detection
- **Practice** mode: read a sentence aloud and get instant comparison feedback
- **Suggest sentences**: built-in practice sentences organized by phoneme focus and difficulty
- **Language-specific tips**: pronunciation feedback tailored to your native language (currently supports Korean; more languages welcome via PR)

## Requirements

- Python 3.10+
- A working microphone
- Disk space for Whisper model (~140 MB for `base`, ~1.5 GB for `large-v3-turbo`)

## Installation

```bash
# Using uv (recommended)
uv tool install mcp-server-pronunciation

# Using pip
pip install mcp-server-pronunciation
```

### Claude Code

```bash
claude mcp add pronunciation mcp-server-pronunciation
```

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pronunciation": {
      "command": "mcp-server-pronunciation"
    }
  }
}
```

## Tools

### `check_mic`
List available audio input devices and verify microphone access.

### `record`
Record audio from the microphone.
- `duration` (float): Recording duration in seconds (default: 10, max: 120)

### `assess`
Assess pronunciation of a recording.
- `reference_text` (string, optional): Expected text for comparison
- `audio_path` (string, optional): Path to WAV file (uses last recording if omitted)

### `practice`
Record and assess in one step — the main tool for pronunciation practice.
- `reference_text` (string): The sentence to practice reading aloud
- `duration` (float): Recording duration in seconds (default: 15, max: 120)

### `suggest_sentence`
Get a practice sentence with phoneme focus.
- `focus` (string, optional): `"th"`, `"f_v"`, `"r_l"`, `"vowels"`, or `"general"`
- `difficulty` (string, optional): `"beginner"`, `"intermediate"`, or `"advanced"`

## Configuration

### Whisper Model

Set the `MCP_PRONUNCIATION_MODEL` environment variable to choose the Whisper model:

```bash
# Fast, lightweight (~140 MB) — default
export MCP_PRONUNCIATION_MODEL=base

# Best accuracy (~1.5 GB, slower on CPU)
export MCP_PRONUNCIATION_MODEL=large-v3-turbo
```

Available models: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`

GPU (CUDA) is auto-detected and used when available.

### Claude Code with model override

```bash
claude mcp add pronunciation -e MCP_PRONUNCIATION_MODEL=small -- mcp-server-pronunciation
```

## Platform Support

| Platform | Recording method | Status |
|----------|-----------------|--------|
| macOS | sounddevice | Supported |
| Linux | sounddevice | Supported |
| Windows | sounddevice | Supported |
| WSL2 | PowerShell MCI (winmm.dll) | Supported |

> **WSL2 note:** WSLg's PulseAudio does not forward microphone audio from the Windows host. This server automatically detects WSL2 and records via PowerShell on the Windows side instead.

## Troubleshooting

### No audio captured / empty recording
- **macOS**: Check System Settings > Privacy & Security > Microphone. Grant access to your terminal app.
- **Linux**: Install PortAudio (`sudo apt install libportaudio2`) and check `pavucontrol` for input levels.
- **WSL2**: Ensure your Windows microphone works (test in Windows Settings > Sound > Input). The server records via Windows, not through WSLg.

### First run is slow
The Whisper model downloads on first use (~140 MB for `base`). Subsequent runs are fast. If it feels too slow, try `MCP_PRONUNCIATION_MODEL=tiny` for the fastest response.

### `sounddevice` import error on Linux
```bash
sudo apt install libportaudio2
```

## Development

```bash
git clone https://github.com/JuhongPark/mcp-server-pronunciation.git
cd mcp-server-pronunciation
uv sync --extra dev
uv run pytest -v
uv run ruff check .
```

## License

MIT
