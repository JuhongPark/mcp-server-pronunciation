"""MCP server for voice conversation with Claude + English language feedback."""

from __future__ import annotations

import logging
import random
import tempfile
import threading
import atexit
from pathlib import Path
from typing import Annotated, Literal, TYPE_CHECKING

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .config import audio_retention_value, preload_enabled
from .sentences import SENTENCES

if TYPE_CHECKING:
    from .assessor import PronunciationAssessor

logger = logging.getLogger(__name__)

mcp = FastMCP("pronunciation")

_assessor: PronunciationAssessor | None = None
_assessor_lock = threading.Lock()
_last_recording: Path | None = None
_last_reference: str | None = None
_recordings_to_cleanup: set[Path] = set()

Focus = Literal["th", "f_v", "r_l", "vowels", "general"]
Difficulty = Literal["beginner", "intermediate", "advanced"]

TargetHint = Annotated[
    str | None,
    Field(
        description=(
            "Optional target sentence or phrase the user is trying to say. "
            "Leave null for open-ended voice conversation."
        )
    ),
]
ReferenceText = Annotated[
    str,
    Field(description="Exact sentence or phrase the user should read aloud for drill mode."),
]
OptionalReferenceText = Annotated[
    str | None,
    Field(
        description=(
            "Optional expected sentence for the recorded audio. "
            "When omitted, only transcript and general prosody feedback are returned."
        )
    ),
]
AudioPath = Annotated[
    str | None,
    Field(
        description=(
            "Optional local path to a WAV file to assess. "
            "When omitted, the most recent recording from this server session is used."
        )
    ),
]
DurationSeconds = Annotated[
    float,
    Field(
        description=(
            "Maximum recording duration in seconds. "
            "The server accepts 1 to 120 seconds and auto-stops earlier on silence."
        )
    ),
]
FocusFilter = Annotated[
    Focus | None,
    Field(
        description=(
            "Optional pronunciation focus filter. "
            "Use th, f_v, r_l, vowels, or general. Leave null to choose randomly."
        )
    ),
]
DifficultyFilter = Annotated[
    Difficulty | None,
    Field(
        description=(
            "Optional practice difficulty filter. "
            "Use beginner, intermediate, or advanced. Leave null to choose randomly."
        )
    ),
]


def _audio_retention() -> str:
    return audio_retention_value()


def _cleanup_recordings() -> None:
    if _audio_retention() == "keep":
        return
    for path in list(_recordings_to_cleanup):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not delete temporary recording: %s", path)
        finally:
            _recordings_to_cleanup.discard(path)


atexit.register(_cleanup_recordings)


def _get_assessor() -> PronunciationAssessor:
    """Return the assessor singleton, creating it lazily on first call."""
    global _assessor
    if _assessor is None:
        with _assessor_lock:
            if _assessor is None:
                from .assessor import PronunciationAssessor

                _assessor = PronunciationAssessor()
    return _assessor


def _preload_model() -> None:
    """Load Whisper weights in a daemon thread so the first tool call is fast.

    The main thread finishes module import immediately, so the MCP initialize
    handshake is never blocked by model loading.
    """

    def _load():
        try:
            assessor = _get_assessor()
            assessor._get_model()
            logger.info("Whisper model pre-loaded")
        except Exception as e:
            logger.warning("Failed to pre-load Whisper model: %s", e)

    thread = threading.Thread(target=_load, daemon=True, name="whisper-preload")
    thread.start()


def _preload_enabled() -> bool:
    return preload_enabled()


if _preload_enabled():
    _preload_model()


def _new_recording_path() -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="pronun_")
    path = Path(tmp.name)
    tmp.close()
    if _audio_retention() == "session":
        _recordings_to_cleanup.add(path)
    return path


# ---------------------------------------------------------------------------
# Primary tool — voice conversation with English feedback
# ---------------------------------------------------------------------------


@mcp.tool(title="Voice conversation with English feedback")
def converse(target_hint: TargetHint = None, duration: DurationSeconds = 30.0) -> str:
    """
    Record the user speaking, transcribe it, and return the transcript plus quick
    English feedback. This is the primary tool for voice conversations: call it,
    read the transcript + feedback, then respond conversationally in your own
    words — weaving the feedback in naturally or mentioning it only if it matters.

    Recording auto-stops when the user finishes speaking (silence detection).

    Use this tool when:
    - The user wants to chat with you by voice instead of typing
    - The user wants casual English feedback while talking with you
    - You want to hear what the user said rather than read a typed message

    For a focused drill where the user reads a specific sentence, use `practice`
    instead.

    Args:
        target_hint: Optional. Only set this if the user is explicitly trying
            to say a specific sentence (e.g. they asked "how do I say X?" and
            you told them X). Leave blank for free-form conversation.
        duration: Maximum recording duration in seconds (default 30, max 120).
            Auto-stops earlier on silence.

    Returns:
        Markdown report containing the user's transcript, brief English feedback
        (pronunciation + grammar + fluency), and a 'For Claude' section with
        guidance on how to respond.
    """
    global _last_recording, _last_reference
    from .recorder import record_audio

    duration = min(max(duration, 1.0), 120.0)
    output_path = _new_recording_path()
    record_audio(duration, output_path)
    _last_recording = output_path
    _last_reference = target_hint

    assessor = _get_assessor()
    result = assessor.assess(output_path, reference_text=target_hint)
    return result.format_converse_report(has_target=target_hint is not None)


# ---------------------------------------------------------------------------
# Practice mode — focused pronunciation drills
# ---------------------------------------------------------------------------


@mcp.tool(title="Focused pronunciation drill")
def practice(
    reference_text: ReferenceText,
    duration: DurationSeconds = 15.0,
) -> str:
    """
    Drill mode: the user reads a specific sentence aloud and gets a detailed
    pronunciation assessment. Use this when the user explicitly wants to
    practice reading a particular sentence, not for free-form chat.

    For voice conversation with casual feedback, use `converse` instead.

    Recording auto-stops when the user finishes speaking.

    Args:
        reference_text: The sentence the user will read aloud.
        duration: Maximum recording duration in seconds (default 15, max 120).

    Returns:
        Detailed pronunciation assessment report.
    """
    global _last_recording, _last_reference
    from .recorder import record_audio

    duration = min(max(duration, 1.0), 120.0)
    output_path = _new_recording_path()

    record_audio(duration, output_path)
    _last_recording = output_path
    _last_reference = reference_text

    assessor = _get_assessor()
    result = assessor.assess(output_path, reference_text=reference_text)
    return result.format_report()


@mcp.tool(title="Retry the last pronunciation drill")
def retry(duration: DurationSeconds = 15.0) -> str:
    """
    Retry the last sentence the user was practicing.

    Re-records and re-assesses using the same reference text from the previous
    `practice` or `converse` call. Use this to let the user try again after
    getting feedback.

    Args:
        duration: Maximum recording duration in seconds (default 15, max 120).

    Returns:
        Pronunciation assessment report for the new attempt.
    """
    if not _last_reference:
        return "Error: No previous practice session. Use 'practice' or 'converse' first."
    return practice(_last_reference, duration)


@mcp.tool(title="Random pronunciation drill")
def quick_practice(
    focus: FocusFilter = None,
    difficulty: DifficultyFilter = None,
    duration: DurationSeconds = 15.0,
) -> str:
    """
    Pick a random practice sentence and drill it immediately.

    Combines `suggest_sentence` + `practice` into one step: picks a sentence
    matching the criteria, then records and assesses.

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
        return (
            "No sentences match that filter. "
            "Try: focus=th/f_v/r_l/vowels/general, "
            "difficulty=beginner/intermediate/advanced"
        )

    sentence = random.choice(pool)
    text = sentence["text"]

    header = (
        f"**Read aloud:** {text}\n"
        f"**Focus:** {sentence['focus']} | **Difficulty:** {sentence['difficulty']}\n\n"
        f"---\n\n"
    )

    result = practice(text, duration)
    return header + result


@mcp.tool(title="Suggest a practice sentence")
def suggest_sentence(
    focus: FocusFilter = None,
    difficulty: DifficultyFilter = None,
) -> str:
    """
    Suggest a practice sentence the user can read aloud.

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
        return (
            "No sentences match that filter. "
            "Try: focus=th/f_v/r_l/vowels/general, "
            "difficulty=beginner/intermediate/advanced"
        )

    sentence = random.choice(pool)
    return (
        f"**Practice this:**\n\n"
        f"> {sentence['text']}\n\n"
        f"**Focus:** {sentence['focus']} | **Difficulty:** {sentence['difficulty']}\n\n"
        f"When ready, use the `practice` tool with this sentence."
    )


# ---------------------------------------------------------------------------
# Utility tools
# ---------------------------------------------------------------------------


@mcp.tool(title="Record microphone audio")
def record(duration: DurationSeconds = 10.0) -> str:
    """
    Record audio from the microphone without assessing it.

    Recording auto-stops when the user finishes speaking (silence detection).
    The duration is the maximum time — you don't have to wait the full duration.

    Most of the time prefer `converse` or `practice`, which record AND analyze
    in one step. Only use `record` alone if you want the raw WAV file.

    Args:
        duration: Maximum recording duration in seconds (default 10, max 120).

    Returns:
        Path to the recorded WAV file.
    """
    global _last_recording
    from .recorder import record_audio

    duration = min(max(duration, 1.0), 120.0)
    output_path = _new_recording_path()

    record_audio(duration, output_path)
    _last_recording = output_path

    size_kb = output_path.stat().st_size / 1024
    actual_sec = size_kb / (16000 * 2 / 1024)
    return f"Recorded {actual_sec:.1f}s of audio to {output_path}"


@mcp.tool(title="Assess a recorded pronunciation attempt")
def assess(reference_text: OptionalReferenceText = None, audio_path: AudioPath = None) -> str:
    """
    Assess the last recording (or a specific audio file) without re-recording.

    When `reference_text` is provided, the assessor:
      - Aligns the user's speech to the reference word-by-word (Needleman-Wunsch;
        single deletions/insertions no longer cascade into phantom substitutions).
      - Runs wav2vec2 CTC forced alignment to verify which reference words the
        user actually produced — mitigates Whisper-bias mistranscriptions on
        rare proper nouns and domain terms by checking acoustic evidence
        against the reference directly.
      - Surfaces per-word phoneme-level feedback (expected vs produced IPA,
        weak phonemes) from CMUdict.
      - Surfaces optional learner-profile pronunciation hints and drills when
        a rule pack matches. The bundled profile includes Korean-L1 patterns
        such as r/l, th→s, final cluster deletion, and intrusive onset vowel.
      - Adds prosody notes: word-stress placement, sentence-final rising
        intonation on declaratives, intra-clause hesitation pauses.

    Without a reference, only the transcript and prosody run.

    Args:
        reference_text: Expected text the user was trying to say (optional).
        audio_path: Path to a WAV file. Uses the last recording if not specified.

    Returns:
        Detailed pronunciation assessment report (markdown).
    """
    if audio_path:
        path = Path(audio_path)
    elif _last_recording:
        path = _last_recording
    else:
        return "Error: No recording found. Use the 'record' or 'converse' tool first."

    if not path.exists():
        return f"Error: Audio file not found: {path}"

    assessor = _get_assessor()
    result = assessor.assess(path, reference_text=reference_text)
    return result.format_report()


@mcp.tool(title="Check microphone devices")
def check_mic() -> str:
    """
    List available audio input devices and verify microphone access.

    Use this if the user reports recording problems — it shows which devices
    are available and which one is the default.

    Returns:
        List of available microphone devices.
    """
    from .recorder import check_audio_devices

    return check_audio_devices()


def run() -> None:
    """Run the MCP server."""
    mcp.run()
