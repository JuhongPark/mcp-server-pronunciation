"""Pronunciation assessment using faster-whisper word-level analysis."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Korean-speaker pronunciation tips
# ---------------------------------------------------------------------------

SUBSTITUTION_HINTS: dict[tuple[str, str], str] = {
    # th-sounds: Korean has no /θ/ or /ð/
    ("three", "tree"): "/θr/ cluster — tongue between teeth before the /r/",
    ("think", "sink"): "/θ/ at word start — tongue between teeth",
    ("think", "tink"): "/θ/ at word start — tongue between teeth, not /t/",
    ("this", "dis"): "/ð/ at word start — voiced, tongue between teeth",
    ("the", "de"): "/ð/ — tongue between teeth with voice",
    ("that", "dat"): "/ð/ — tongue between teeth with voice",
    ("with", "wis"): "Final /θ/ — end with tongue between teeth",
    ("with", "wit"): "Final /θ/ — tongue between teeth, not /t/",
    ("bath", "bas"): "Final /θ/ — tongue between teeth at the end",
    ("both", "bos"): "Final /θ/ — tongue between teeth at the end",
    ("thought", "taught"): "/θ/ — tongue between teeth, not /t/",
    ("through", "true"): "/θr/ — tongue between teeth before /r/",
    ("thoroughly", "truly"): "/θ/ at word start — tongue between teeth",
    # f/v confusion: Korean has no /f/ or /v/
    ("five", "pive"): "Initial /f/ — lower lip touches upper teeth",
    ("for", "por"): "Initial /f/ — lower lip touches upper teeth, not both lips",
    ("future", "puture"): "Initial /f/ — lower lip touches upper teeth",
    ("very", "berry"): "/v/ — lower lip touches upper teeth, voiced",
    ("have", "hab"): "Final /v/ — lower lip touches upper teeth",
    # r/l confusion
    ("rice", "lice"): "/r/ at word start — tongue curled back, not touching ridge",
    ("right", "light"): "/r/ at word start — tongue curled back",
    ("light", "right"): "/l/ at word start — tongue tip firmly on alveolar ridge",
    ("read", "lead"): "/r/ — tongue curled back, doesn't touch anything",
    ("world", "word"): "/rl/ cluster — maintain the /l/ sound",
    ("girl", "gir"): "Final /rl/ — tongue touches ridge for /l/ after /r/",
}

# Words that reveal common Korean pronunciation patterns
_TH_WORDS = {
    "the",
    "this",
    "that",
    "think",
    "three",
    "through",
    "with",
    "math",
    "both",
    "bath",
    "thought",
    "thoroughly",
    "brother",
}
_F_WORDS = {
    "five",
    "four",
    "first",
    "feel",
    "fine",
    "for",
    "from",
    "free",
    "off",
    "if",
    "future",
    "before",
}
_V_WORDS = {"very", "have", "over", "every", "never", "give", "live", "value"}
_RL_WORDS = {
    "right",
    "light",
    "read",
    "lead",
    "rice",
    "lice",
    "really",
    "world",
    "girl",
    "long",
    "wrong",
}


# Common ESL grammar mistakes — rule-based, no external deps.
# Covers irregular verb past tenses most Korean learners over-regularize.
_IRREGULAR_PAST_ERRORS: dict[str, tuple[str, str]] = {
    "buyed": ("bought", "Past tense of 'buy' is 'bought' (irregular)"),
    "goed": ("went", "Past tense of 'go' is 'went' (irregular)"),
    "runned": ("ran", "Past tense of 'run' is 'ran' (irregular)"),
    "seed": ("saw", "Past tense of 'see' is 'saw' (irregular)"),
    "eated": ("ate", "Past tense of 'eat' is 'ate' (irregular)"),
    "drinked": ("drank", "Past tense of 'drink' is 'drank' (irregular)"),
    "taked": ("took", "Past tense of 'take' is 'took' (irregular)"),
    "writed": ("wrote", "Past tense of 'write' is 'wrote' (irregular)"),
    "sleeped": ("slept", "Past tense of 'sleep' is 'slept' (irregular)"),
    "teached": ("taught", "Past tense of 'teach' is 'taught' (irregular)"),
    "catched": ("caught", "Past tense of 'catch' is 'caught' (irregular)"),
    "bringed": ("brought", "Past tense of 'bring' is 'brought' (irregular)"),
    "breaked": ("broke", "Past tense of 'break' is 'broke' (irregular)"),
    "thinked": ("thought", "Past tense of 'think' is 'thought' (irregular)"),
    "feeled": ("felt", "Past tense of 'feel' is 'felt' (irregular)"),
    "maked": ("made", "Past tense of 'make' is 'made' (irregular)"),
    "gived": ("gave", "Past tense of 'give' is 'gave' (irregular)"),
    "knowed": ("knew", "Past tense of 'know' is 'knew' (irregular)"),
    "finded": ("found", "Past tense of 'find' is 'found' (irregular)"),
    "standed": ("stood", "Past tense of 'stand' is 'stood' (irregular)"),
    "holded": ("held", "Past tense of 'hold' is 'held' (irregular)"),
    "leaved": ("left", "Past tense of 'leave' is 'left' (irregular)"),
    "readed": ("read", "Past tense of 'read' is 'read' (spelled the same, pronounced /red/)"),
    "swimmed": ("swam", "Past tense of 'swim' is 'swam' (irregular)"),
    "speaked": ("spoke", "Past tense of 'speak' is 'spoke' (irregular)"),
    "beginned": ("began", "Past tense of 'begin' is 'began' (irregular)"),
    "choosed": ("chose", "Past tense of 'choose' is 'chose' (irregular)"),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WordResult:
    """Assessment result for a single word."""

    word: str
    start: float
    end: float
    probability: float
    issue: str | None = None


@dataclass
class AssessmentResult:
    """Full pronunciation assessment result."""

    transcript: str
    reference_text: str | None
    words: list[WordResult] = field(default_factory=list)
    duration_sec: float = 0.0
    speech_duration_sec: float = 0.0
    language: str = "en"
    language_prob: float = 0.0

    @property
    def words_per_minute(self) -> float:
        if self.speech_duration_sec <= 0:
            return 0.0
        return len(self.words) / self.speech_duration_sec * 60

    @property
    def avg_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.probability for w in self.words) / len(self.words)

    @property
    def low_confidence_words(self) -> list[WordResult]:
        return [w for w in self.words if w.probability < 0.7]

    @property
    def flagged_words(self) -> list[WordResult]:
        return [w for w in self.words if w.issue]

    def get_pauses(self, threshold: float = 0.8) -> list[tuple[float, float, float]]:
        """Find pauses longer than threshold seconds between words."""
        pauses = []
        for i in range(1, len(self.words)):
            gap = self.words[i].start - self.words[i - 1].end
            if gap >= threshold:
                pauses.append((self.words[i - 1].end, self.words[i].start, gap))
        return pauses

    def grammar_notes(self) -> list[tuple[str, str, str]]:
        """Return (wrong_word, correction, explanation) for grammar errors in transcript."""
        notes = []
        seen: set[str] = set()
        for token in _normalize_words(self.transcript):
            if token in _IRREGULAR_PAST_ERRORS and token not in seen:
                seen.add(token)
                correct, explanation = _IRREGULAR_PAST_ERRORS[token]
                notes.append((token, correct, explanation))
        return notes

    def format_report(self) -> str:
        """Format a concise, actionable pronunciation report."""
        lines = []
        lines.append("## Pronunciation Assessment\n")

        # What you said vs reference
        lines.append(f"**You said:** {self.transcript or '(nothing detected)'}")
        if self.reference_text:
            lines.append(f"**Target:** {self.reference_text}")
        lines.append("")

        # Compact summary line
        confidence_pct = self.avg_confidence * 100
        wpm = self.words_per_minute
        lines.append(
            f"**Clarity:** {confidence_pct:.0f}% | **Speed:** {wpm:.0f} WPM | ",
        )

        if wpm < 100:
            lines[-1] += "Slow — try speaking a bit faster"
        elif wpm < 130:
            lines[-1] += "Careful pace — good for practice"
        elif wpm <= 170:
            lines[-1] += "Natural pace"
        else:
            lines[-1] += "Fast — watch clarity"
        lines.append("")

        # Mismatches — the most important section
        mismatches = self._find_mismatches() if self.reference_text else []
        if mismatches:
            lines.append("### What to fix")
            for ref_word, heard_word, hint in mismatches:
                if heard_word == "(skipped)":
                    lines.append(f'- **"{ref_word}"** — skipped or too quiet')
                elif heard_word == "(extra)":
                    pass  # Don't show extra words as errors
                else:
                    lines.append(f'- **"{ref_word}"** → heard **"{heard_word}"**')
                    if hint:
                        lines.append(f"  - {hint}")
            lines.append("")

        # Low confidence words (only those not already in mismatches)
        mismatch_words = {m[0] for m in mismatches}
        low_conf = [w for w in self.low_confidence_words if w.word.lower() not in mismatch_words]
        if low_conf:
            lines.append("### Unclear words")
            for w in low_conf:
                lines.append(f'- **"{w.word}"** ({w.probability:.0%} confidence)')
            lines.append("")

        # Pauses
        pauses = self.get_pauses(0.8)
        if pauses:
            lines.append("### Pauses")
            for start, end, dur in pauses[:3]:
                lines.append(f"- {dur:.1f}s pause at {start:.1f}s")
            lines.append("")

        # Korean-speaker tips
        tips = self._get_korean_tips()
        if tips:
            lines.append("### Tips for Korean speakers")
            for tip in tips:
                lines.append(f"- {tip}")
            lines.append("")

        if not mismatches and not low_conf and not tips:
            lines.append("### Great job! No major issues detected.\n")

        return "\n".join(lines)

    def format_converse_report(self, has_target: bool = False) -> str:
        """Format a conversation-oriented report.

        Unlike `format_report`, this is optimized for Claude to read and decide how
        to respond: user's transcript up top, compact feedback bullets, and a
        'For Claude' section telling the model how to weave the feedback into
        a natural conversational reply.
        """
        lines = []
        lines.append("## User said\n")
        if self.transcript:
            lines.append(f"> {self.transcript}\n")
        else:
            lines.append("> *(nothing clearly transcribed — may have been silent or too quiet)*\n")

        confidence_pct = self.avg_confidence * 100
        wpm = self.words_per_minute
        speed_label = _speed_label(wpm)
        lines.append(
            f"**Clarity:** {confidence_pct:.0f}% &nbsp;|&nbsp; "
            f"**Pace:** {wpm:.0f} WPM ({speed_label})\n"
        )

        feedback_bullets: list[str] = []

        # Grammar errors first — they're actionable and easy for Claude to weave in.
        grammar = self.grammar_notes()
        for wrong, correct, explanation in grammar:
            feedback_bullets.append(f'**Grammar:** *"{wrong}"* → *"{correct}"* — {explanation}')

        # Target comparison (only when user is explicitly practicing a sentence).
        if has_target and self.reference_text:
            mismatches = self._find_mismatches()
            shown = 0
            for ref_word, heard_word, hint in mismatches:
                if heard_word == "(extra)" or shown >= 3:
                    continue
                if heard_word == "(skipped)":
                    feedback_bullets.append(
                        f'**Pronunciation:** *"{ref_word}"* was skipped or too quiet'
                    )
                else:
                    bullet = f'**Pronunciation:** *"{ref_word}"* → heard *"{heard_word}"*'
                    if hint:
                        bullet += f" — {hint}"
                    feedback_bullets.append(bullet)
                shown += 1

        # Unclear words (independent of target comparison).
        low_conf = self.low_confidence_words[:3]
        already_flagged = {b.lower() for b in feedback_bullets}
        for w in low_conf:
            wlabel = w.word.strip().lower()
            if any(wlabel in b for b in already_flagged):
                continue
            feedback_bullets.append(
                f'**Pronunciation:** *"{w.word}"* was unclear ({w.probability:.0%} confidence)'
            )

        # Long pauses / fluency
        pauses = self.get_pauses(1.2)
        if len(pauses) >= 2:
            feedback_bullets.append(
                f"**Fluency:** {len(pauses)} long pauses — try to keep the flow"
            )
        elif wpm and wpm < 90 and len(self.words) >= 5:
            feedback_bullets.append(
                "**Fluency:** speaking quite slowly — natural pace is 120–150 WPM"
            )

        if feedback_bullets:
            lines.append("## Quick feedback\n")
            for b in feedback_bullets[:5]:
                lines.append(f"- {b}")
            lines.append("")
        else:
            lines.append("## Quick feedback\n")
            lines.append("- No obvious issues — clear and natural.\n")

        # Guidance for Claude on how to use this.
        lines.append("## For Claude\n")
        if not self.transcript:
            lines.append(
                "The user's recording was silent or very quiet. Ask them to repeat, "
                "and consider suggesting they run `check_mic` if this happens twice."
            )
        elif feedback_bullets:
            lines.append(
                "Respond conversationally to what the user actually said (above). "
                "You MAY weave the feedback in naturally — for example, subtly using "
                "the corrected form in your own reply, or mentioning a fix explicitly "
                "if the user asked for corrections. If the user is just chatting "
                "casually, prioritize the conversation over the feedback; only call "
                "out the most important item if any. Do not recite the whole feedback "
                "list back at them."
            )
        else:
            lines.append(
                "Respond conversationally to what the user said. No feedback issues "
                "to surface — just continue the chat naturally."
            )
        lines.append("")

        return "\n".join(lines)

    def _find_mismatches(self) -> list[tuple[str, str, str]]:
        """Compare transcript to reference using sequence alignment."""
        if not self.reference_text:
            return []

        ref_words = _normalize_words(self.reference_text)
        heard_words = _normalize_words(self.transcript)

        if not ref_words or not heard_words:
            return [(w, "(skipped)", "") for w in ref_words]

        matcher = SequenceMatcher(None, ref_words, heard_words)
        mismatches = []

        for op, ref_start, ref_end, heard_start, heard_end in matcher.get_opcodes():
            if op == "equal":
                continue
            elif op == "replace":
                # Words that were substituted
                for i, ref in enumerate(ref_words[ref_start:ref_end]):
                    if heard_start + i < heard_end:
                        heard = heard_words[heard_start + i]
                        hint = SUBSTITUTION_HINTS.get((ref, heard), "")
                        mismatches.append((ref, heard, hint))
                    else:
                        mismatches.append((ref, "(skipped)", ""))
            elif op == "delete":
                # Words in reference that were skipped
                for ref in ref_words[ref_start:ref_end]:
                    mismatches.append((ref, "(skipped)", ""))
            elif op == "insert":
                # Extra words the speaker added — mark but don't penalize
                for heard in heard_words[heard_start:heard_end]:
                    mismatches.append(("", "(extra)", ""))

        return mismatches

    def _get_korean_tips(self) -> list[str]:
        """Generate tips based on detected Korean-speaker patterns."""
        tips = []
        spoken_words = set(self.transcript.lower().split())

        # Check each phoneme group
        th_spoken = spoken_words & _TH_WORDS
        if th_spoken:
            low = [w for w in self.low_confidence_words if w.word.lower() in _TH_WORDS]
            if low:
                words_str = ", ".join(w.word for w in low)
                tips.append(f"/θ/ and /ð/: Place tongue between teeth. Check: {words_str}")

        f_spoken = spoken_words & _F_WORDS
        if f_spoken:
            low = [w for w in self.low_confidence_words if w.word.lower() in _F_WORDS]
            if low:
                tips.append("/f/: Bite lower lip gently and blow. Don't use both lips (/p/).")

        v_spoken = spoken_words & _V_WORDS
        if v_spoken:
            low = [w for w in self.low_confidence_words if w.word.lower() in _V_WORDS]
            if low:
                tips.append("/v/: Bite lower lip and vibrate vocal cords. Not /b/.")

        rl_spoken = spoken_words & _RL_WORDS
        if rl_spoken:
            low = [w for w in self.low_confidence_words if w.word.lower() in _RL_WORDS]
            if low:
                tips.append("/r/ vs /l/: For /r/ curl tongue back. For /l/ touch tongue to ridge.")

        return tips


def _normalize_words(text: str) -> list[str]:
    """Normalize text to lowercase word list for comparison."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()


def _speed_label(wpm: float) -> str:
    if wpm <= 0:
        return "n/a"
    if wpm < 90:
        return "slow"
    if wpm < 120:
        return "careful"
    if wpm <= 160:
        return "natural"
    return "fast"


# ---------------------------------------------------------------------------
# Whisper engine
# ---------------------------------------------------------------------------

# Default model — English-only variant for better pronunciation accuracy at
# small size. Override with MCP_PRONUNCIATION_MODEL env var.
DEFAULT_MODEL = os.environ.get("MCP_PRONUNCIATION_MODEL", "base.en")


def _detect_device() -> tuple[str, str]:
    """Auto-detect the best device and compute type."""
    try:
        import ctranslate2

        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            logger.info("CUDA detected, using GPU")
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


class PronunciationAssessor:
    """Pronunciation assessment engine using faster-whisper."""

    def __init__(self, model_size: str | None = None):
        self._model_size = model_size or DEFAULT_MODEL
        self._device, self._compute_type = _detect_device()
        self._model: WhisperModel | None = None

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading Whisper model '%s' on %s (%s)...",
                self._model_size,
                self._device,
                self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=os.cpu_count() or 4,
            )
            logger.info("Whisper model loaded")
        return self._model

    def assess(
        self,
        audio_path: Path,
        reference_text: str | None = None,
    ) -> AssessmentResult:
        """Assess pronunciation of an audio file."""
        model = self._get_model()

        segments, info = model.transcribe(
            str(audio_path),
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        words: list[WordResult] = []
        full_text_parts: list[str] = []

        for segment in segments:
            full_text_parts.append(segment.text.strip())
            if segment.words:
                for w in segment.words:
                    wr = WordResult(
                        word=w.word.strip(),
                        start=w.start,
                        end=w.end,
                        probability=w.probability,
                    )
                    if w.probability < 0.5:
                        wr.issue = f"Very unclear ({w.probability:.0%})"
                    elif w.probability < 0.7:
                        wr.issue = f"Unclear ({w.probability:.0%})"
                    words.append(wr)

        transcript = " ".join(full_text_parts)

        speech_dur = 0.0
        if words:
            speech_dur = words[-1].end - words[0].start

        return AssessmentResult(
            transcript=transcript,
            reference_text=reference_text,
            words=words,
            duration_sec=info.duration,
            speech_duration_sec=speech_dur,
            language=info.language,
            language_prob=info.language_probability,
        )
