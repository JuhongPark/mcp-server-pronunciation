# mcp-server-pronunciation

An MCP (Model Context Protocol) server for English pronunciation practice. Record your voice, get it transcribed by Whisper, and receive word-level pronunciation feedback — all from within Claude Code or any MCP client.

## Features

- **Record** audio from your microphone
- **Transcribe** with OpenAI Whisper (via faster-whisper) with word-level timestamps
- **Assess** pronunciation with confidence scores, fluency metrics, and mismatch detection
- **Practice** mode: read a sentence aloud and get instant comparison feedback
- **Korean-speaker tips**: targeted feedback for common pronunciation challenges (/θ/, /f/, /v/, /r/, /l/)

## Requirements

- Python 3.10+
- A working microphone
- ~1.5 GB disk space (for the Whisper model, downloaded on first use)

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

## Platform Support

| Platform | Recording method | Status |
|----------|-----------------|--------|
| macOS | sounddevice | Supported |
| Linux | sounddevice | Supported |
| Windows | sounddevice | Supported |
| WSL2 | PowerShell MCI (winmm.dll) | Supported |

> **WSL2 note:** WSLg's PulseAudio does not forward microphone audio from the Windows host. This server automatically detects WSL2 and records via PowerShell on the Windows side instead.

## Example Usage

In Claude Code:

```
> Use the practice tool with "The three brothers thought thoroughly about their future."
```

The server will:
1. Record your voice reading the sentence
2. Transcribe it with Whisper
3. Compare against the reference text
4. Return a detailed report with scores, mismatches, and tips

## License

MIT
