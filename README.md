# mcp-server-pronunciation

<!-- mcp-name: io.github.juhongpark/pronunciation -->

An MCP (Model Context Protocol) server that lets you **talk to Claude by voice while getting English pronunciation, grammar, and fluency feedback** in the same turn. Use it for casual voice chat with light coaching, or switch to drill mode when you want to practice a specific sentence.

Built for Claude Desktop, Claude Code, and any other MCP client. Everything runs locally — audio is captured with your mic, transcribed by [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) on-device, and never leaves your machine.

`mcp-name: io.github.juhongpark/pronunciation`

## Why

Voice MCP servers today treat speech as a typing replacement. English tutor MCP servers are text-only. This one combines the two: you speak freely, Claude replies, and feedback on what you just said (pronunciation, grammar, fluency) surfaces inside the same tool call so Claude can weave it into a natural reply — or stay out of the way when you're just chatting.

## Features

- **Voice conversation** with Claude. Speak, auto-stop on silence, Claude reads your transcript and responds.
- **Phoneme-level drill feedback** (when a reference sentence is given): Needleman-Wunsch word alignment, per-word expected vs produced IPA, Korean-L1 pattern detection (r/l, th→s, final cluster deletion, intrusive onset vowel, …) with Korean-language tips and minimal-pair drills, and prosody checks (word stress, final-rise intonation, intra-clause pauses).
- **Whisper-bias mitigation** via optional `[phoneme]` extra: wav2vec2 CTC forced alignment verifies whether the user actually produced each reference word, so rare proper nouns and domain-specific terms that Whisper rewrites toward more common alternatives no longer surface as mispronunciations.
- **Inline English feedback in conversation**: pronunciation, grammar (common irregular-verb errors), and fluency (pace + long pauses).
- **Drill mode** (`practice`, `quick_practice`, `retry`) for focused sentence practice.
- **Local-only**: Whisper model runs on your machine, audio never leaves it.
- **Cross-platform**: macOS, Linux, Windows, and WSL2 (recording auto-routes through Windows).
- **Fast startup**: lazy imports + background model pre-load keep the MCP handshake under a second.

## Requirements

- Python 3.11+
- A working microphone
- ~150 MB disk space for the default Whisper model (`base.en`)
- Additional ~360 MB if you install the optional `[phoneme]` extra (wav2vec2 weights for forced alignment)
- MCP spec: targets `2025-06-18` via the official Python SDK (`mcp>=1.2`)

## Installation

```bash
# Recommended: uvx (no global install, cached between runs)
uvx mcp-server-pronunciation

# Or install as a uv tool
uv tool install mcp-server-pronunciation

# Or pip
pip install mcp-server-pronunciation

# Optional: forced-alignment upgrade for Whisper-bias mitigation + tighter
# phoneme-level feedback. Adds ~200 MB of torch CPU wheels.
pip install 'mcp-server-pronunciation[phoneme]'
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
> **Claude** (calls `practice` with that reference): *returns an alignment table (match / sub / ins / del) with per-word acoustic confidence when the `[phoneme]` extra is installed, phoneme-level issues with expected vs produced IPA, detected Korean-L1 patterns (e.g. /θ/→/s/) with Korean-language tips and minimal-pair drills, and prosody notes (word stress, final-rise intonation, intra-clause pauses).*

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
| `assess` | Assess the last recording (or a specified WAV) without re-recording. When given a reference, runs the full drill pipeline (alignment, phoneme diff, Korean-L1 patterns, prosody). |
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

### Temporary recordings

Recordings are written as temporary WAV files so `assess` can inspect the last
recording. By default they are removed when the server process exits:

```bash
export MCP_PRONUNCIATION_AUDIO_RETENTION=session
```

Set `MCP_PRONUNCIATION_AUDIO_RETENTION=keep` if you want temporary recordings
to remain on disk for manual inspection.

### Model override in Claude Code

```bash
claude mcp add pronunciation -e MCP_PRONUNCIATION_MODEL=small.en -- uvx mcp-server-pronunciation
```

### Phoneme analysis extras

Installing `mcp-server-pronunciation[phoneme]` enables wav2vec2-based CTC forced alignment. It verifies which reference words the user acoustically produced, regardless of how Whisper's language-model-weighted decoder rewrote them — so rare proper nouns and domain terms no longer surface as false mispronunciations. On first run the extra downloads ~360 MB of weights into `~/.cache/torch/hub/` (override via `TORCH_HOME`). Inference is CPU-only by default and runtime-quantized to int8 (~95 MB RAM).

Without the extra, `assess` / `practice` still run the full pipeline except for the forced-alignment step: you get Needleman-Wunsch word alignment against the Whisper hypothesis, CMUdict phoneme-sequence diff, Korean-L1 pattern detection, and prosody.

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

It reports on PortAudio, input devices, Whisper model cache, pronunciation
resources, optional forced-alignment dependencies, free disk space, and Python
version. Run it whenever something feels off.

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

## Known Limitations

- Pronunciation scores are coaching signals, not standardized-test, clinical, or native-speaker-equivalence judgments.
- Whisper can still mishear rare names, domain terms, short clips, quiet audio, or heavily accented speech. The optional `[phoneme]` extra reduces some reference-sentence false positives but does not eliminate them.
- Prosody feedback is heuristic. Pitch tracking can be unreliable with noisy audio, very short utterances, vocal fry, overlapping speech, or clipped recordings.
- Korean-L1 pattern detection is intentionally rule-based. It can miss errors, over-trigger on ASR mistakes, and should be treated as a targeted practice aid.
- First-time setup may download model or pronunciation resources. Run `doctor` and `pull-model` before relying on the server in a live session.
- Temporary WAV recordings are written under the system temp directory so that the last recording can be assessed. By default they are removed when the server exits. Set `MCP_PRONUNCIATION_AUDIO_RETENTION=keep` if you want to inspect them later.

## Benchmark Status

This project is moving toward benchmark-backed scoring. Planned public benchmark work is tracked in [ROADMAP.md](ROADMAP.md), and the current benchmark helper docs live in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). The primary candidate is Speechocean762 because it has a permissive CC BY 4.0 license and multi-level expert pronunciation scores. L2-ARCTIC is useful for Korean-L1 and phone-error research checks, but its non-commercial license means it should remain optional and separate from default release claims.

## Privacy

- All audio processing happens **locally** on your machine.
- Recordings are temporary `.wav` files under your system temp directory (`$TMPDIR`) and are removed when the server exits unless `MCP_PRONUNCIATION_AUDIO_RETENTION=keep` is set.
- The Whisper model runs locally — no audio data is sent to any external service.
- When the optional `[phoneme]` extra is installed, the wav2vec2 forced aligner also runs locally. Weights are downloaded once from the PyTorch Hub.
- No telemetry. No analytics. No network calls except the one-time model weight downloads (Whisper from Hugging Face, wav2vec2 from PyTorch Hub).

## Development

```bash
git clone https://github.com/JuhongPark/mcp-server-pronunciation.git
cd mcp-server-pronunciation
uv sync --extra dev
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

To work on the optional wav2vec2 forced-alignment path, install the phoneme
extra as well:

```bash
uv sync --extra dev --extra phoneme
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
- [`cmudict`](https://pypi.org/project/cmudict/) — BSD
- [`g2p-en`](https://github.com/Kyubyong/g2p) — Apache 2.0
- [`librosa`](https://librosa.org/) — ISC
- Optional (`[phoneme]` extra): [PyTorch](https://pytorch.org/) — BSD, [torchaudio](https://pytorch.org/audio/) — BSD, [wav2vec2 weights](https://github.com/facebookresearch/fairseq) — MIT
