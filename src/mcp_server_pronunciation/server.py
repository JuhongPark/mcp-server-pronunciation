"""MCP server for pronunciation assessment."""

from __future__ import annotations

import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .assessor import PronunciationAssessor
from .recorder import check_audio_devices, record_audio

mcp = FastMCP("pronunciation")

# Lazy-loaded assessor (loads Whisper model on first use)
_assessor: PronunciationAssessor | None = None
# Store the last recorded file path for assess to use
_last_recording: Path | None = None


def _get_assessor() -> PronunciationAssessor:
    global _assessor
    if _assessor is None:
        _assessor = PronunciationAssessor()
    return _assessor


@mcp.tool()
def record(duration: float = 10.0) -> str:
    """
    Record audio from the microphone.

    Args:
        duration: Recording duration in seconds (default 10, max 120).

    Returns:
        Path to the recorded WAV file and recording info.
    """
    global _last_recording

    duration = min(max(duration, 1.0), 120.0)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="pronun_")
    output_path = Path(tmp.name)
    tmp.close()

    record_audio(duration, output_path)
    _last_recording = output_path

    return f"Recorded {duration:.0f}s of audio to {output_path}"


@mcp.tool()
def assess(reference_text: str | None = None, audio_path: str | None = None) -> str:
    """
    Assess pronunciation of the last recording (or a specific audio file).

    Transcribes the audio using Whisper and provides word-level pronunciation
    feedback including clarity scores, fluency metrics, and Korean-speaker tips.

    Args:
        reference_text: Expected text the speaker was trying to say (optional).
            If provided, enables comparison-based feedback.
        audio_path: Path to a WAV file. Uses the last recording if not specified.

    Returns:
        Detailed pronunciation assessment report.
    """
    if audio_path:
        path = Path(audio_path)
    elif _last_recording:
        path = _last_recording
    else:
        return "Error: No recording found. Use the 'record' tool first."

    if not path.exists():
        return f"Error: Audio file not found: {path}"

    assessor = _get_assessor()
    result = assessor.assess(path, reference_text=reference_text)
    return result.format_report()


@mcp.tool()
def practice(
    reference_text: str,
    duration: float = 15.0,
) -> str:
    """
    Full pronunciation practice: record and assess in one step.

    Shows a sentence, records the speaker reading it, then provides
    detailed pronunciation feedback with comparison to the reference.

    Args:
        reference_text: The sentence to practice reading aloud.
        duration: Recording duration in seconds (default 15, max 120).

    Returns:
        Detailed pronunciation assessment report.
    """
    global _last_recording

    duration = min(max(duration, 1.0), 120.0)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="pronun_")
    output_path = Path(tmp.name)
    tmp.close()

    record_audio(duration, output_path)
    _last_recording = output_path

    assessor = _get_assessor()
    result = assessor.assess(output_path, reference_text=reference_text)
    return result.format_report()


@mcp.tool()
def check_mic() -> str:
    """
    Check available audio input devices.

    Returns:
        List of available microphone devices.
    """
    return check_audio_devices()


def run() -> None:
    """Run the MCP server."""
    mcp.run()
