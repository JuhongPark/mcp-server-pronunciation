# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — Unreleased

### Changed

- **Repositioned**: the server is now primarily a voice-conversation-with-English-feedback
  tool, not a pure pronunciation drill app. Drill mode is still available through
  `practice`, `quick_practice`, and `retry`, but `converse` is the main entry point.
- Default Whisper model switched from `base` (multilingual) to `base.en` (English-only,
  same size, better pronunciation accuracy on English clips).
- Tightened dependency bounds: `mcp[cli]>=1.2,<2`, `faster-whisper>=1.0,<2`,
  `sounddevice>=0.4,<0.6`, `numpy>=1.24,<3`.
- Server module no longer imports `faster-whisper`, `sounddevice`, or `numpy` at
  module top. Imports are deferred into tool handlers and the background pre-load
  thread, so the MCP `initialize` handshake completes in milliseconds.
- PortAudio / `sounddevice` import failures now raise a `RuntimeError` with
  platform-specific install instructions instead of a cryptic `OSError`.

### Added

- **`converse` tool** — primary voice-conversation entry point. Records, transcribes,
  and returns a structured report with transcript, quick feedback bullets
  (pronunciation + grammar + fluency), and a "For Claude" section guiding the
  model on how to weave feedback into a natural reply.
- **Rule-based grammar feedback** for common ESL mistakes, covering over-regularized
  irregular verb past tenses (`buyed` → `bought`, `goed` → `went`, etc.).
- **`doctor` CLI subcommand** — preflight check for PortAudio, input devices,
  Whisper model cache, Python version, and free disk space. Run with
  `mcp-server-pronunciation doctor`.
- **`pull-model` CLI subcommand** — pre-downloads the Whisper model so the first
  MCP call is instant. Run with `mcp-server-pronunciation pull-model [size]`.
- `format_converse_report()` method on `AssessmentResult`, distinct from the
  existing `format_report()` used by drill tools.
- `server.json` at the repo root for submission to the official MCP Registry
  (namespace `io.github.juhongpark/pronunciation`).
- `mcp-name:` line in the README for MCP Registry PyPI ownership verification.
- Release workflow (`.github/workflows/release.yml`) using PyPI Trusted Publishers
  (OIDC) — no long-lived API tokens required.

### Fixed

- Recording errors on Linux now point users at the correct `libportaudio2`
  package name instead of suggesting `pip install sounddevice`.

## [0.1.0] — Initial release

- Whisper-based pronunciation assessment via `faster-whisper` word-level timestamps
- Tools: `record`, `assess`, `practice`, `retry`, `quick_practice`, `suggest_sentence`, `check_mic`
- Korean-speaker substitution hints for common phonemes (`/θ/`, `/ð/`, `/f/`, `/v/`, `/r/`, `/l/`)
- VAD auto-stop recording
- WSL2 support via PowerShell MCI
- Background model pre-load
