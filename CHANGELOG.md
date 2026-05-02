# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — Unreleased

### Changed

- **`assess` / `practice` rewritten around a new analysis pipeline** when a
  reference text is provided:
  1. **Needleman-Wunsch word alignment** replaces `difflib.SequenceMatcher`.
     A single dropped or added word no longer cascades into a chain of
     phantom substitutions.
  2. **wav2vec2 CTC forced alignment** (optional `[phoneme]` extra) runs in
     parallel with Whisper and verifies which reference words the user
     actually produced. Fixes Whisper-bias false positives — rare proper
     nouns and domain-specific terms that Whisper rewrites toward more
     common training-set n-grams. Entries where Whisper misheard but
     acoustic evidence matches the reference are flipped from `sub`/`del`
     to `match` with an explanatory note.
  3. **Phoneme-level feedback** via CMUdict lookup + ARPAbet→IPA rendering.
     Each mismatched word reports its expected and produced IPA plus which
     phonemes were weak.
  4. **Korean-L1 pattern detection**: `r_l_swap`, `f_to_p`, `v_to_b`,
     `th_to_s`, `th_to_t`, `dh_to_d`, `z_to_j`, `final_cluster_deletion`,
     `intrusive_onset_vowel`, `final_stop_unrelease`, `schwa_to_full_vowel`,
     `dark_l_confusion`, `article_omission`. Each carries a Korean-language
     tip and a minimal-pair drill list.
  5. **Prosody** via `librosa`: word-stress placement (CMUdict stress vs
     measured pitch+intensity peak), rising intonation on declaratives,
     intra-clause hesitation pauses.
  6. **Drill suggestions** derived from detected patterns + weak phonemes.
- **WPM caveat** when computed over <10 s of speech, explicitly labelled in
  the report ("computed over 4.8 s of speech"). Speech duration now excludes
  inter-word silence so long pauses no longer inflate the denominator.
- `AssessmentResult.to_dict()` emits the full structured JSON shape
  (alignment, phoneme_issues, korean_l1_patterns, prosody, drills).
  `format_report()` renders it to markdown.

### Added

- `alignment.py`: Needleman-Wunsch word aligner with soft-equality tie-break.
- `phonemes.py`: CMUdict lookup, ARPAbet→IPA, phoneme-sequence diff, Korean-L1
  pattern rules, minimal-pair drill catalog.
- `prosody.py`: librosa-based f0 + RMS feature extraction for prosody checks.
- `forced_align.py`: optional wav2vec2-base-960h CTC forced alignment with
  Viterbi trellis. Dynamically int8-quantized (~95 MB RAM, ~1.5 s for a
  6-s clip on CPU). Degrades gracefully when `torch`/`torchaudio` absent.
- `[phoneme]` optional-dependency extra for the forced-aligner stack.

### Dependencies

- New mandatory: `cmudict>=1.0`, `g2p-en>=2.1`, `librosa>=0.10,<1`.
- New optional (`[phoneme]`): `torch>=2.2`, `torchaudio>=2.2`.

## [0.2.0] — 2026-04-12

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

## [0.1.0] — 2026-04-11

- Whisper-based pronunciation assessment via `faster-whisper` word-level timestamps
- Tools: `record`, `assess`, `practice`, `retry`, `quick_practice`, `suggest_sentence`, `check_mic`
- Korean-speaker substitution hints for common phonemes (`/θ/`, `/ð/`, `/f/`, `/v/`, `/r/`, `/l/`)
- VAD auto-stop recording
- WSL2 support via PowerShell MCI
- Background model pre-load
