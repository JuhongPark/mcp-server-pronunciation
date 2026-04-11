"""MCP server for pronunciation assessment."""

from __future__ import annotations

import logging
import random
import tempfile
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .assessor import PronunciationAssessor
from .recorder import check_audio_devices, record_audio
from .sentences import SENTENCES

logger = logging.getLogger(__name__)

mcp = FastMCP("pronunciation")

_assessor: PronunciationAssessor | None = None
_last_recording: Path | None = None
_last_reference: str | None = None


def _get_assessor() -> PronunciationAssessor:
    global _assessor
    if _assessor is None:
        _assessor = PronunciationAssessor()
    return _assessor


def _preload_model() -> None:
    """Pre-load Whisper model in background so first practice call is fast."""

    def _load():
        try:
            assessor = _get_assessor()
            assessor._get_model()
            logger.info("Whisper model pre-loaded")
        except Exception as e:
            logger.warning("Failed to pre-load Whisper model: %s", e)

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()


# Start pre-loading as soon as the module is imported
_preload_model()


def _new_recording_path() -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="pronun_")
    path = Path(tmp.name)
    tmp.close()
    return path


@mcp.tool()
def record(duration: float = 10.0) -> str:
    """
    Record audio from the microphone.

    Recording auto-stops when you finish speaking (after detecting silence).
    The duration is the maximum time — you don't have to wait the full duration.

    Args:
        duration: Maximum recording duration in seconds (default 10, max 120).

    Returns:
        Path to the recorded WAV file.
    """
    global _last_recording
    duration = min(max(duration, 1.0), 120.0)
    output_path = _new_recording_path()

    record_audio(duration, output_path)
    _last_recording = output_path

    size_kb = output_path.stat().st_size / 1024
    actual_sec = size_kb / (16000 * 2 / 1024)  # rough estimate from file size
    return f"Recorded {actual_sec:.1f}s of audio to {output_path}"


@mcp.tool()
def assess(reference_text: str | None = None, audio_path: str | None = None) -> str:
    """
    Assess pronunciation of the last recording (or a specific audio file).

    Transcribes the audio using Whisper and provides word-level pronunciation
    feedback including clarity scores, fluency metrics, and language-specific tips.

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

    Recording auto-stops when you finish speaking. Read the sentence aloud,
    then wait briefly — the recording will stop automatically.

    Args:
        reference_text: The sentence to practice reading aloud.
        duration: Maximum recording duration in seconds (default 15, max 120).

    Returns:
        Detailed pronunciation assessment report.
    """
    global _last_recording, _last_reference
    duration = min(max(duration, 1.0), 120.0)
    output_path = _new_recording_path()

    record_audio(duration, output_path)
    _last_recording = output_path
    _last_reference = reference_text

    assessor = _get_assessor()
    result = assessor.assess(output_path, reference_text=reference_text)
    return result.format_report()


@mcp.tool()
def retry(duration: float = 15.0) -> str:
    """
    Retry the last practiced sentence.

    Re-records and re-assesses using the same reference text from the
    previous practice call. Use this to quickly try again after getting feedback.

    Args:
        duration: Maximum recording duration in seconds (default 15, max 120).

    Returns:
        Detailed pronunciation assessment report.
    """
    if not _last_reference:
        return "Error: No previous practice session. Use 'practice' first."
    return practice(_last_reference, duration)


@mcp.tool()
def quick_practice(
    focus: str | None = None,
    difficulty: str | None = None,
    duration: float = 15.0,
) -> str:
    """
    Pick a random sentence and start practicing immediately.

    Combines suggest_sentence + practice into one step: picks a sentence
    matching your criteria, then records and assesses your pronunciation.

    Args:
        focus: Phoneme focus area. Options: "th", "f_v", "r_l", "vowels", "general".
            If not specified, picks randomly.
        difficulty: Difficulty level. Options: "beginner", "intermediate", "advanced".
            If not specified, picks randomly.
        duration: Maximum recording duration in seconds (default 15, max 120).

    Returns:
        The sentence to read, followed by the pronunciation assessment.
    """
    pool = SENTENCES
    if focus:
        pool = [s for s in pool if s["focus"] == focus]
    if difficulty:
        pool = [s for s in pool if s["difficulty"] == difficulty]

    if not pool:
        return "No sentences match that filter. Try: focus=th/f_v/r_l/vowels/general, difficulty=beginner/intermediate/advanced"

    sentence = random.choice(pool)
    text = sentence["text"]

    header = (
        f"**Read aloud:** {text}\n"
        f"**Focus:** {sentence['focus']} | **Difficulty:** {sentence['difficulty']}\n\n"
        f"---\n\n"
    )

    result = practice(text, duration)
    return header + result


@mcp.tool()
def suggest_sentence(
    focus: str | None = None,
    difficulty: str | None = None,
) -> str:
    """
    Suggest a practice sentence for pronunciation practice.

    Args:
        focus: Phoneme focus area. Options: "th", "f_v", "r_l", "vowels", "general".
            If not specified, picks randomly.
        difficulty: Difficulty level. Options: "beginner", "intermediate", "advanced".
            If not specified, picks randomly.

    Returns:
        A practice sentence with its focus area and difficulty.
    """
    pool = SENTENCES
    if focus:
        pool = [s for s in pool if s["focus"] == focus]
    if difficulty:
        pool = [s for s in pool if s["difficulty"] == difficulty]

    if not pool:
        return "No sentences match that filter. Try: focus=th/f_v/r_l/vowels/general, difficulty=beginner/intermediate/advanced"

    sentence = random.choice(pool)
    return (
        f"**Practice this:**\n\n"
        f"> {sentence['text']}\n\n"
        f"**Focus:** {sentence['focus']} | **Difficulty:** {sentence['difficulty']}\n\n"
        f"When ready, use the `practice` tool with this sentence."
    )


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
