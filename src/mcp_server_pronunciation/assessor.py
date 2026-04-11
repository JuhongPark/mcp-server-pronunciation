"""Pronunciation assessment using faster-whisper word-level analysis."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

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
_TH_WORDS = {"the", "this", "that", "think", "three", "through", "with",
             "math", "both", "bath", "thought", "thoroughly", "brother"}
_F_WORDS = {"five", "four", "first", "feel", "fine", "for", "from",
            "free", "off", "if", "future", "before"}
_V_WORDS = {"very", "have", "over", "every", "never", "give", "live", "value"}
_RL_WORDS = {"right", "light", "read", "lead", "rice", "lice", "really",
             "world", "girl", "long", "wrong"}


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
        lines.append(f"**Clarity:** {confidence_pct:.0f}% | **Speed:** {wpm:.0f} WPM | ", )

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
                    lines.append(f"- **\"{ref_word}\"** — skipped or too quiet")
                elif heard_word == "(extra)":
                    pass  # Don't show extra words as errors
                else:
                    lines.append(f"- **\"{ref_word}\"** → heard **\"{heard_word}\"**")
                    if hint:
                        lines.append(f"  - {hint}")
            lines.append("")

        # Low confidence words (only those not already in mismatches)
        mismatch_words = {m[0] for m in mismatches}
        low_conf = [w for w in self.low_confidence_words
                    if w.word.lower() not in mismatch_words]
        if low_conf:
            lines.append("### Unclear words")
            for w in low_conf:
                lines.append(f"- **\"{w.word}\"** ({w.probability:.0%} confidence)")
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
            low = [w for w in self.low_confidence_words
                   if w.word.lower() in _TH_WORDS]
            if low:
                words_str = ", ".join(w.word for w in low)
                tips.append(f"/θ/ and /ð/: Place tongue between teeth. Check: {words_str}")

        f_spoken = spoken_words & _F_WORDS
        if f_spoken:
            low = [w for w in self.low_confidence_words
                   if w.word.lower() in _F_WORDS]
            if low:
                tips.append("/f/: Bite lower lip gently and blow. Don't use both lips (/p/).")

        v_spoken = spoken_words & _V_WORDS
        if v_spoken:
            low = [w for w in self.low_confidence_words
                   if w.word.lower() in _V_WORDS]
            if low:
                tips.append("/v/: Bite lower lip and vibrate vocal cords. Not /b/.")

        rl_spoken = spoken_words & _RL_WORDS
        if rl_spoken:
            low = [w for w in self.low_confidence_words
                   if w.word.lower() in _RL_WORDS]
            if low:
                tips.append("/r/ vs /l/: For /r/ curl tongue back. For /l/ touch tongue to ridge.")

        return tips


def _normalize_words(text: str) -> list[str]:
    """Normalize text to lowercase word list for comparison."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()


# ---------------------------------------------------------------------------
# Whisper engine
# ---------------------------------------------------------------------------

# Default model — can be overridden via MCP_PRONUNCIATION_MODEL env var
DEFAULT_MODEL = os.environ.get("MCP_PRONUNCIATION_MODEL", "base")


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
            logger.info("Loading Whisper model '%s' on %s (%s)...",
                        self._model_size, self._device, self._compute_type)
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
