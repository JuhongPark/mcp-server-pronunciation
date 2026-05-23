"""MCP server for voice conversation with Claude + English language feedback."""

from __future__ import annotations

import logging
import random
import tempfile
import threading
import atexit
from pathlib import Path
from typing import Annotated, Any, Literal, TYPE_CHECKING

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, Field

from .config import audio_retention_value, preload_enabled
from .sentences import SENTENCES

if TYPE_CHECKING:
    from .assessor import AssessmentResult, PronunciationAssessor

logger = logging.getLogger(__name__)

mcp = FastMCP("pronunciation")

_assessor: PronunciationAssessor | None = None
_assessor_lock = threading.Lock()
_last_recording: Path | None = None
_last_reference: str | None = None
_last_assessment: AssessmentResult | None = None
_recordings_to_cleanup: set[Path] = set()

Focus = Literal["th", "f_v", "r_l", "vowels", "general"]
Difficulty = Literal["beginner", "intermediate", "advanced"]
AssessmentMode = Literal["conversation", "practice", "assessment", "retry"]

READ_ONLY_TOOL = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)
LOCAL_RECORDING_TOOL = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)

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


class IssueSummary(BaseModel):
    """One high-signal coaching issue for clients to act on."""

    kind: str
    label: str
    detail: str
    severity: Literal["info", "practice", "important"] = "practice"


class NextAction(BaseModel):
    """Suggested follow-up tool call or user-facing practice action."""

    tool: str
    instruction: str
    reference_text: str | None = None
    focus: Focus | None = None
    difficulty: Difficulty | None = None


class RetryComparison(BaseModel):
    """Comparison between two attempts at the same target sentence."""

    previous_clarity_pct: int
    current_clarity_pct: int
    clarity_delta: int
    fixed_issue_count: int
    remaining_issue_count: int
    summary: str


class AssessmentToolResponse(BaseModel):
    """Structured tool response paired with a markdown report."""

    mode: AssessmentMode
    report_markdown: str
    transcript: str
    reference_text: str | None = None
    clarity_pct: int
    speaking_rate_wpm: int
    top_issue: IssueSummary | None = None
    next_action: NextAction | None = None
    retry_comparison: RetryComparison | None = None
    assessment: dict[str, Any]
    error: str | None = None


AssessmentCallResult = Annotated[CallToolResult, AssessmentToolResponse]


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


def _record_and_assess(reference_text: str | None, duration: float) -> AssessmentResult:
    global _last_recording, _last_reference, _last_assessment
    from .recorder import record_audio

    duration = min(max(duration, 1.0), 120.0)
    output_path = _new_recording_path()
    record_audio(duration, output_path)
    _last_recording = output_path
    _last_reference = reference_text

    assessor = _get_assessor()
    result = assessor.assess(output_path, reference_text=reference_text)
    _last_assessment = result
    return result


def _issue_keys(result: AssessmentResult) -> set[str]:
    keys: set[str] = set()
    if not result.transcript:
        keys.add("audio:silent")
    for wrong, correct, _explain in result.grammar_notes():
        keys.add(f"grammar:{wrong}->{correct}")
    for item in result.aligned:
        if item.op != "match" and item.ref:
            keys.add(f"alignment:{item.ref}:{item.op}")
        elif item.note and item.ref:
            keys.add(f"alignment:{item.ref}:note")
    for diff in result.phoneme_diffs:
        if diff.weak_phonemes and diff.produced_arpa:
            keys.add(f"phoneme:{diff.word}:{','.join(diff.weak_phonemes)}")
    for pattern in result.korean_l1_patterns:
        keys.add(f"profile:{pattern.pattern}")
    if result.prosody.final_rise_on_declarative:
        keys.add("prosody:final_rise")
    for stress in result.prosody.wrong_word_stress:
        keys.add(f"prosody:stress:{stress.word}")
    for pause in result.prosody.intra_clause_pauses:
        keys.add(f"prosody:pause:{pause.before}:{pause.after}")
    return keys


def _top_issue(result: AssessmentResult) -> IssueSummary | None:
    if not result.transcript:
        return IssueSummary(
            kind="audio",
            label="No clear speech detected",
            detail="The recording was silent, too quiet, or the microphone input was unavailable.",
            severity="important",
        )

    grammar = result.grammar_notes()
    if grammar:
        wrong, correct, explanation = grammar[0]
        return IssueSummary(
            kind="grammar",
            label=f"{wrong} -> {correct}",
            detail=explanation,
            severity="practice",
        )

    for item in result.aligned:
        if item.op == "sub" and item.ref and item.hyp:
            return IssueSummary(
                kind="pronunciation",
                label=f"{item.ref} heard as {item.hyp}",
                detail="The spoken word did not align cleanly with the target sentence.",
                severity="practice",
            )
        if item.op == "del" and item.ref:
            return IssueSummary(
                kind="pronunciation",
                label=f"{item.ref} skipped",
                detail="A target word was not detected in the recording.",
                severity="practice",
            )
        if item.note and item.ref:
            return IssueSummary(
                kind="confidence",
                label=item.ref,
                detail=item.note,
                severity="info",
            )

    phoneme_hits = [d for d in result.phoneme_diffs if d.weak_phonemes and d.produced_arpa]
    if phoneme_hits:
        diff = phoneme_hits[0]
        return IssueSummary(
            kind="phoneme",
            label=f"{diff.word}: weak {diff.weak_phonemes_ipa}",
            detail=f"Expected {diff.expected_ipa}, produced {diff.produced_ipa}.",
            severity="practice",
        )

    if result.korean_l1_patterns:
        pattern = result.korean_l1_patterns[0]
        return IssueSummary(
            kind="learner_profile",
            label=pattern.label,
            detail=pattern.tip_ko,
            severity="practice",
        )

    if result.prosody.final_rise_on_declarative:
        return IssueSummary(
            kind="prosody",
            label="Rising intonation",
            detail="The sentence ended with rising pitch, which can make a statement sound like a question.",
            severity="practice",
        )

    if result.prosody.wrong_word_stress:
        stress = result.prosody.wrong_word_stress[0]
        return IssueSummary(
            kind="prosody",
            label=f"Word stress: {stress.word}",
            detail=(
                f"Observed stress on syllable {stress.observed_stress_syllable + 1}; "
                f"expected syllable {stress.expected_stress_syllable + 1}."
            ),
            severity="practice",
        )

    if result.prosody.intra_clause_pauses:
        pause = result.prosody.intra_clause_pauses[0]
        return IssueSummary(
            kind="fluency",
            label="Mid-clause pause",
            detail=f"{pause.duration:.2f}s pause between {pause.before!r} and {pause.after!r}.",
            severity="info",
        )

    return None


def _next_action(result: AssessmentResult, mode: AssessmentMode) -> NextAction | None:
    if not result.transcript:
        return NextAction(
            tool="check_mic",
            instruction="Check microphone access, then record again.",
            reference_text=result.reference_text,
        )

    if result.reference_text:
        if result.drills:
            drill = result.drills[0]
            pairs = ", ".join(drill.minimal_pairs[:3])
            return NextAction(
                tool="retry",
                instruction=f"Try the same sentence again, focusing on {drill.reason}: {pairs}.",
                reference_text=result.reference_text,
            )
        return NextAction(
            tool="retry",
            instruction="Try the same sentence again and compare the new attempt.",
            reference_text=result.reference_text,
        )

    if mode == "conversation":
        return NextAction(
            tool="converse",
            instruction="Continue the voice conversation with brief feedback only when useful.",
        )

    return None


def _compare_attempts(
    previous: AssessmentResult | None,
    current: AssessmentResult,
) -> RetryComparison | None:
    if previous is None or previous.reference_text != current.reference_text:
        return None

    previous_keys = _issue_keys(previous)
    current_keys = _issue_keys(current)
    fixed = previous_keys - current_keys
    remaining = previous_keys & current_keys
    clarity_delta = current.clarity_pct - previous.clarity_pct

    if clarity_delta > 3:
        summary = f"Clarity improved by {clarity_delta} points."
    elif clarity_delta < -3:
        summary = f"Clarity dropped by {abs(clarity_delta)} points."
    elif fixed:
        summary = f"Clarity was similar, but {len(fixed)} issue(s) improved."
    else:
        summary = "Clarity was about the same."

    return RetryComparison(
        previous_clarity_pct=previous.clarity_pct,
        current_clarity_pct=current.clarity_pct,
        clarity_delta=clarity_delta,
        fixed_issue_count=len(fixed),
        remaining_issue_count=len(remaining),
        summary=summary,
    )


def _assessment_response(
    result: AssessmentResult,
    mode: AssessmentMode,
    report_markdown: str,
    retry_comparison: RetryComparison | None = None,
) -> AssessmentCallResult:
    payload = AssessmentToolResponse(
        mode=mode,
        report_markdown=report_markdown,
        transcript=result.transcript,
        reference_text=result.reference_text,
        clarity_pct=result.clarity_pct,
        speaking_rate_wpm=int(round(result.words_per_minute)),
        top_issue=_top_issue(result),
        next_action=_next_action(result, mode),
        retry_comparison=retry_comparison,
        assessment=result.to_dict(),
    )
    return CallToolResult(
        content=[TextContent(type="text", text=report_markdown)],
        structuredContent=payload.model_dump(mode="json"),
    )


def _error_response(
    mode: AssessmentMode,
    message: str,
    next_action: NextAction | None = None,
) -> AssessmentCallResult:
    payload = AssessmentToolResponse(
        mode=mode,
        report_markdown=message,
        transcript="",
        clarity_pct=0,
        speaking_rate_wpm=0,
        top_issue=IssueSummary(
            kind="error",
            label="Tool could not run",
            detail=message,
            severity="important",
        ),
        next_action=next_action,
        assessment={},
        error=message,
    )
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        structuredContent=payload.model_dump(mode="json"),
        isError=True,
    )


# ---------------------------------------------------------------------------
# Primary tool — voice conversation with English feedback
# ---------------------------------------------------------------------------


@mcp.tool(
    title="Voice conversation with English feedback",
    annotations=LOCAL_RECORDING_TOOL,
    structured_output=True,
)
def converse(
    target_hint: TargetHint = None, duration: DurationSeconds = 30.0
) -> AssessmentCallResult:
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
    result = _record_and_assess(target_hint, duration)
    report = result.format_converse_report(has_target=target_hint is not None)
    return _assessment_response(result, "conversation", report)


# ---------------------------------------------------------------------------
# Practice mode — focused pronunciation drills
# ---------------------------------------------------------------------------


@mcp.tool(
    title="Focused pronunciation drill",
    annotations=LOCAL_RECORDING_TOOL,
    structured_output=True,
)
def practice(
    reference_text: ReferenceText,
    duration: DurationSeconds = 15.0,
) -> AssessmentCallResult:
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
    result = _record_and_assess(reference_text, duration)
    return _assessment_response(result, "practice", result.format_report())


@mcp.tool(
    title="Retry the last pronunciation drill",
    annotations=LOCAL_RECORDING_TOOL,
    structured_output=True,
)
def retry(duration: DurationSeconds = 15.0) -> AssessmentCallResult:
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
    previous = _last_assessment
    if not _last_reference:
        return _error_response(
            "retry",
            "Error: No previous practice session. Use 'practice' or 'converse' first.",
            NextAction(
                tool="practice",
                instruction="Start a practice drill with a target sentence first.",
            ),
        )

    result = _record_and_assess(_last_reference, duration)
    comparison = _compare_attempts(previous, result)
    report = result.format_report()
    if comparison is not None:
        report = (
            f"## Retry Comparison\n\n"
            f"**Previous clarity:** {comparison.previous_clarity_pct}% | "
            f"**Current clarity:** {comparison.current_clarity_pct}% | "
            f"**Delta:** {comparison.clarity_delta:+d}\n\n"
            f"{comparison.summary}\n\n"
            f"---\n\n"
            f"{report}"
        )
    return _assessment_response(result, "retry", report, comparison)


@mcp.tool(
    title="Random pronunciation drill",
    annotations=LOCAL_RECORDING_TOOL,
    structured_output=True,
)
def quick_practice(
    focus: FocusFilter = None,
    difficulty: DifficultyFilter = None,
    duration: DurationSeconds = 15.0,
) -> AssessmentCallResult:
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
        return _error_response(
            "practice",
            "No sentences match that filter. "
            "Try: focus=th/f_v/r_l/vowels/general, "
            "difficulty=beginner/intermediate/advanced",
        )

    sentence = random.choice(pool)
    text = sentence["text"]

    header = (
        f"**Read aloud:** {text}\n"
        f"**Focus:** {sentence['focus']} | **Difficulty:** {sentence['difficulty']}\n\n"
        f"---\n\n"
    )

    result = _record_and_assess(text, duration)
    report = header + result.format_report()
    return _assessment_response(result, "practice", report)


@mcp.tool(title="Suggest a practice sentence", annotations=READ_ONLY_TOOL)
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
# Prompt shortcuts — make the main workflows discoverable in MCP clients
# ---------------------------------------------------------------------------


@mcp.prompt(
    title="Start voice chat",
    description="Start a local voice conversation with light English feedback.",
)
def start_voice_chat(
    topic: Annotated[
        str | None,
        Field(description="Optional conversation topic, such as weekend plans or work updates."),
    ] = None,
) -> str:
    topic_line = f" about {topic}" if topic else ""
    return (
        f"Let's have a voice chat{topic_line}. Use the `converse` tool to record me, "
        "then reply naturally to what I said. Keep pronunciation feedback brief unless I ask for details."
    )


@mcp.prompt(
    title="Daily pronunciation practice",
    description="Run a short pronunciation practice loop with a suggested sentence.",
)
def daily_practice(
    focus: FocusFilter = None,
    difficulty: DifficultyFilter = None,
) -> str:
    focus_text = f" with focus={focus}" if focus else ""
    difficulty_text = f" at {difficulty} difficulty" if difficulty else ""
    return (
        f"Start a short pronunciation practice session{focus_text}{difficulty_text}. "
        "Use `suggest_sentence` first, ask me to read the sentence aloud, then use `practice`. "
        "After feedback, offer one retry using `retry`."
    )


@mcp.prompt(
    title="Practice a focus area",
    description="Start a focused drill for a specific pronunciation area.",
)
def practice_focus(
    focus: FocusFilter = None,
    difficulty: DifficultyFilter = None,
) -> str:
    chosen_focus = focus or "general"
    chosen_difficulty = difficulty or "intermediate"
    return (
        f"Give me a {chosen_difficulty} pronunciation drill for focus={chosen_focus}. "
        "Use `quick_practice` if I am ready to record immediately; otherwise use `suggest_sentence`."
    )


@mcp.prompt(
    title="Troubleshoot microphone",
    description="Diagnose microphone or recording problems before practicing.",
)
def troubleshoot_mic() -> str:
    return (
        "Help me troubleshoot microphone recording for this pronunciation server. "
        "Call `check_mic`, explain the default input and configured VAD settings, "
        "then suggest the smallest next fix."
    )


# ---------------------------------------------------------------------------
# Utility tools
# ---------------------------------------------------------------------------


@mcp.tool(title="Record microphone audio", annotations=LOCAL_RECORDING_TOOL)
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


@mcp.tool(
    title="Assess a recorded pronunciation attempt",
    annotations=READ_ONLY_TOOL,
    structured_output=True,
)
def assess(
    reference_text: OptionalReferenceText = None,
    audio_path: AudioPath = None,
) -> AssessmentCallResult:
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
    global _last_reference, _last_assessment

    if audio_path:
        path = Path(audio_path)
    elif _last_recording:
        path = _last_recording
    else:
        return _error_response(
            "assessment",
            "Error: No recording found. Use the 'record' or 'converse' tool first.",
            NextAction(tool="record", instruction="Record audio before assessing it."),
        )

    if not path.exists():
        return _error_response("assessment", f"Error: Audio file not found: {path}")

    assessor = _get_assessor()
    result = assessor.assess(path, reference_text=reference_text)
    if reference_text:
        _last_reference = reference_text
        _last_assessment = result
    return _assessment_response(result, "assessment", result.format_report())


@mcp.tool(title="Check microphone devices", annotations=READ_ONLY_TOOL)
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
