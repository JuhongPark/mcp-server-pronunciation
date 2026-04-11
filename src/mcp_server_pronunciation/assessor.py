"""Pronunciation assessment using faster-whisper word-level analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from faster_whisper import WhisperModel

# Korean-speaker common confusion patterns (source phoneme -> target phoneme)
KOREAN_SPEAKER_PATTERNS = {
    # th-sounds: Korean has no /θ/ or /ð/
    ("s", "th"): "Korean speakers often replace /θ/ (th) with /s/. Try placing tongue between teeth.",
    ("d", "th"): "Korean speakers often replace /ð/ (voiced th) with /d/. Tongue between teeth, vibrate.",
    ("z", "th"): "/θ/ replaced with /z/ — tongue tip should touch upper teeth lightly.",
    # f/p confusion: Korean has no /f/
    ("p", "f"): "Korean has no /f/ sound. Bite lower lip gently and blow air.",
    ("b", "v"): "Korean has no /v/ sound. Bite lower lip and vibrate vocal cords.",
    # r/l confusion
    ("r", "l"): "/r/ and /l/ are distinct in English. For /r/: curl tongue back. For /l/: touch tongue to ridge.",
    ("l", "r"): "/l/ and /r/ are distinct in English. For /l/: tongue tip touches the alveolar ridge firmly.",
    # vowel length
    ("ship", "sheep"): "Short /ɪ/ vs long /iː/ — 'ship' vs 'sheep'. Length and tension differ.",
    ("full", "fool"): "Short /ʊ/ vs long /uː/ — 'full' vs 'fool'. Watch vowel length.",
}

# Common word-level substitutions that indicate pronunciation issues
SUBSTITUTION_HINTS: dict[tuple[str, str], str] = {
    ("three", "tree"): "/θr/ cluster — tongue between teeth before the /r/",
    ("think", "sink"): "/θ/ at word start — tongue between teeth",
    ("this", "dis"): "/ð/ at word start — voiced, tongue between teeth",
    ("the", "de"): "/ð/ — tongue between teeth with voice",
    ("with", "wis"): "Final /θ/ — end with tongue between teeth",
    ("bath", "bas"): "Final /θ/ — tongue between teeth at the end",
    ("five", "pive"): "Initial /f/ — lower lip touches upper teeth",
    ("very", "berry"): "/v/ — lower lip touches upper teeth, voiced",
    ("rice", "lice"): "/r/ at word start — tongue curled back, not touching ridge",
    ("light", "right"): "/l/ at word start — tongue tip firmly on alveolar ridge",
    ("world", "word"): "/rl/ cluster — maintain the /l/ sound",
    ("girl", "gir"): "Final /rl/ — tongue touches ridge for /l/ after /r/",
}


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
        """Format a human-readable pronunciation report."""
        lines = []

        # Header
        lines.append("## Pronunciation Assessment Report\n")

        # Transcription
        lines.append(f"**What you said:** {self.transcript}")
        if self.reference_text:
            lines.append(f"**Reference text:** {self.reference_text}")
        lines.append("")

        # Overall scores
        lines.append("### Overall Scores")
        confidence_pct = self.avg_confidence * 100
        if confidence_pct >= 90:
            clarity = "Excellent"
        elif confidence_pct >= 75:
            clarity = "Good"
        elif confidence_pct >= 60:
            clarity = "Fair — some words unclear"
        else:
            clarity = "Needs work — many words unclear"
        lines.append(f"- **Clarity:** {confidence_pct:.0f}% ({clarity})")
        lines.append(f"- **Speaking rate:** {self.words_per_minute:.0f} WPM")

        # Fluency note
        wpm = self.words_per_minute
        if wpm < 100:
            lines.append(f"  - Slow pace. Native conversational is ~130-170 WPM. Try speaking a bit faster.")
        elif wpm < 130:
            lines.append(f"  - Slightly below native pace. Good for careful speech.")
        elif wpm <= 170:
            lines.append(f"  - Natural conversational pace. Good!")
        else:
            lines.append(f"  - Fast pace. Make sure clarity isn't sacrificed for speed.")

        # Pauses
        pauses = self.get_pauses(0.8)
        if pauses:
            lines.append(f"- **Long pauses:** {len(pauses)} (>{0.8}s)")
            for start, end, dur in pauses[:3]:
                lines.append(f"  - {dur:.1f}s pause at {start:.1f}s")
        lines.append("")

        # Word-level issues
        if self.reference_text:
            mismatches = self._find_mismatches()
            if mismatches:
                lines.append("### Pronunciation Issues (reference vs. heard)")
                for ref_word, heard_word, hint in mismatches:
                    lines.append(f"- **\"{ref_word}\"** heard as **\"{heard_word}\"**")
                    if hint:
                        lines.append(f"  - Tip: {hint}")
                lines.append("")

        # Low confidence words
        low_conf = self.low_confidence_words
        if low_conf:
            lines.append("### Unclear Words (low recognition confidence)")
            for w in low_conf:
                lines.append(f"- **\"{w.word}\"** (confidence: {w.probability:.0%}) at {w.start:.1f}s")
            lines.append("")

        # Flagged words with specific issues
        flagged = [w for w in self.flagged_words if w not in low_conf]
        if flagged:
            lines.append("### Specific Feedback")
            for w in flagged:
                lines.append(f"- **\"{w.word}\"**: {w.issue}")
            lines.append("")

        # Korean-speaker tips
        tips = self._get_korean_tips()
        if tips:
            lines.append("### Korean Speaker Tips")
            for tip in tips:
                lines.append(f"- {tip}")
            lines.append("")

        if not low_conf and not self.flagged_words and not (self.reference_text and self._find_mismatches()):
            lines.append("### Great job! No major issues detected.\n")

        return "\n".join(lines)

    def _find_mismatches(self) -> list[tuple[str, str, str]]:
        """Compare transcript to reference text and find substitutions."""
        if not self.reference_text:
            return []

        ref_words = _normalize_words(self.reference_text)
        heard_words = _normalize_words(self.transcript)

        mismatches = []
        # Simple alignment: match by position
        for i, ref in enumerate(ref_words):
            if i >= len(heard_words):
                mismatches.append((ref, "(missing)", ""))
                continue
            heard = heard_words[i]
            if ref != heard:
                hint = SUBSTITUTION_HINTS.get((ref, heard), "")
                mismatches.append((ref, heard, hint))

        return mismatches

    def _get_korean_tips(self) -> list[str]:
        """Generate tips based on detected Korean-speaker patterns."""
        tips = []
        transcript_lower = self.transcript.lower()

        # Check for words that commonly reveal Korean pronunciation patterns
        th_words = ["the", "this", "that", "think", "three", "through", "with", "math", "both", "bath"]
        has_th = any(w in transcript_lower.split() for w in th_words)
        if has_th:
            low_th = [w for w in self.low_confidence_words if any(th in w.word.lower() for th in th_words)]
            if low_th:
                tips.append("/θ/ and /ð/ sounds: Place tongue between teeth. Practice: 'the, this, think, three'")

        f_words = ["five", "four", "first", "feel", "fine", "for", "from", "free", "off", "if"]
        has_f = any(w in transcript_lower.split() for w in f_words)
        if has_f:
            low_f = [w for w in self.low_confidence_words if any(f in w.word.lower() for f in f_words)]
            if low_f:
                tips.append("/f/ sound: Gently bite lower lip and blow. Don't use both lips (that makes /p/).")

        return tips


def _normalize_words(text: str) -> list[str]:
    """Normalize text to lowercase word list for comparison."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()


class PronunciationAssessor:
    """Pronunciation assessment engine using faster-whisper."""

    def __init__(self, model_size: str = "large-v3-turbo", device: str = "cpu"):
        self._model_size = model_size
        self._device = device
        self._model: WhisperModel | None = None

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type="int8",
                cpu_threads=8,
            )
        return self._model

    def assess(
        self,
        audio_path: Path,
        reference_text: str | None = None,
    ) -> AssessmentResult:
        """Assess pronunciation of an audio file.

        Args:
            audio_path: Path to WAV file to analyze.
            reference_text: Optional expected text for comparison.

        Returns:
            AssessmentResult with detailed pronunciation feedback.
        """
        model = self._get_model()

        segments, info = model.transcribe(
            str(audio_path),
            language="en",
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4],
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
                    # Flag low-confidence words
                    if w.probability < 0.5:
                        wr.issue = f"Very low confidence ({w.probability:.0%}) — pronunciation may be unclear"
                    elif w.probability < 0.7:
                        wr.issue = f"Below average confidence ({w.probability:.0%}) — check pronunciation"
                    words.append(wr)

        transcript = " ".join(full_text_parts)

        # Calculate speech duration (first word start to last word end)
        speech_dur = 0.0
        if words:
            speech_dur = words[-1].end - words[0].start

        result = AssessmentResult(
            transcript=transcript,
            reference_text=reference_text,
            words=words,
            duration_sec=info.duration,
            speech_duration_sec=speech_dur,
            language=info.language,
            language_prob=info.language_probability,
        )

        return result
