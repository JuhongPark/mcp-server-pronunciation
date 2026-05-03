"""Pronunciation assessment: Whisper ASR + word alignment + phoneme diff + prosody.

Pipeline:
  audio + optional reference_text
    -> Whisper (biased by reference via `initial_prompt` when provided)
    -> word-level hypothesis tokens + per-word Whisper probability
    -> Needleman-Wunsch alignment vs reference tokens
    -> per-mismatched-word phoneme-sequence diff (CMUdict + g2p_en)
    -> learner-profile pattern scan over alignment + phoneme diffs
    -> librosa-based prosody (word stress / final-rise / intra-clause pauses)
    -> Drill suggestions
    -> AssessmentResult (dict JSON + markdown renderer)

When no reference_text is provided, only the transcript and prosody run — the
phoneme + alignment + pattern stages require a reference. This keeps the
`converse` voice-chat flow working.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .alignment import AlignedWord, align_words, tokenize
from .phonemes import (
    Drill,
    KoreanL1Pattern,
    PhonemeDiff,
    detect_patterns,
    diff_word,
    suggest_drills,
)
from .prosody import ProsodyResult, TimedWord, analyze as prosody_analyze
from . import forced_align
from .config import whisper_model_name

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grammar rule table, unchanged from v0.2.
# irregular past tenses. Still useful in the converse flow.
# ---------------------------------------------------------------------------

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
    """Per-word Whisper output. Kept so prosody + UI can look up timestamps."""

    word: str
    start: float
    end: float
    probability: float


@dataclass
class AssessmentResult:
    """Full assessment — serializable to the JSON shape documented in README."""

    transcript: str
    reference_text: str | None
    words: list[WordResult] = field(default_factory=list)
    duration_sec: float = 0.0  # total audio duration (incl. silence)
    speech_duration_sec: float = 0.0  # sum of word spans (speech only)
    language: str = "en"
    language_prob: float = 0.0

    aligned: list[AlignedWord] = field(default_factory=list)
    phoneme_diffs: list[PhonemeDiff] = field(default_factory=list)
    korean_l1_patterns: list[KoreanL1Pattern] = field(default_factory=list)
    prosody: ProsodyResult = field(default_factory=ProsodyResult)
    drills: list[Drill] = field(default_factory=list)

    # True when wav2vec2 forced alignment was available and used to verify
    # which reference words the user actually produced (fixes Whisper bias).
    forced_alignment_used: bool = False

    # ---- derived scalars ------------------------------------------

    @property
    def words_per_minute(self) -> float:
        if self.speech_duration_sec <= 0:
            return 0.0
        return len(self.words) / self.speech_duration_sec * 60

    @property
    def wpm_caveat(self) -> str | None:
        """Return a short caveat string when WPM is computed over too little speech."""
        if self.speech_duration_sec <= 0:
            return None
        if self.speech_duration_sec < 10.0:
            return f"computed over {self.speech_duration_sec:.1f}s of speech"
        return None

    @property
    def avg_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.probability for w in self.words) / len(self.words)

    @property
    def clarity_pct(self) -> int:
        """Clarity score on 0-100 scale.

        Combines Whisper's average per-word confidence (proxy for how
        identifiable each word was) with a penalty for mismatches against
        reference (if a reference was provided). Mismatches are weighted
        equally to confidence so a well-pronounced wrong word doesn't score
        higher than a nominally-recognized word that matches.
        """
        whisper_score = self.avg_confidence
        if not self.reference_text:
            return int(round(whisper_score * 100))

        total = sum(1 for a in self.aligned if a.ref is not None)
        if total == 0:
            return int(round(whisper_score * 100))
        matches = sum(1 for a in self.aligned if a.op == "match")
        align_score = matches / total
        return int(round(((whisper_score + align_score) / 2) * 100))

    # ---- grammar (ported unchanged) ----------------------------------

    def grammar_notes(self) -> list[tuple[str, str, str]]:
        """Return (wrong_word, correction, explanation) for grammar errors."""
        notes = []
        seen: set[str] = set()
        for token in tokenize(self.transcript):
            if token in _IRREGULAR_PAST_ERRORS and token not in seen:
                seen.add(token)
                correct, explanation = _IRREGULAR_PAST_ERRORS[token]
                notes.append((token, correct, explanation))
        return notes

    def get_pauses(self, threshold: float = 0.8) -> list[tuple[float, float, float]]:
        """Raw pauses longer than threshold seconds (used by converse report)."""
        pauses = []
        for i in range(1, len(self.words)):
            gap = self.words[i].start - self.words[i - 1].end
            if gap >= threshold:
                pauses.append((self.words[i - 1].end, self.words[i].start, gap))
        return pauses

    # ---- serialization ---------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serializable dict matching the documented output shape."""
        return {
            "clarity_pct": self.clarity_pct,
            "speaking_rate_wpm": int(round(self.words_per_minute)),
            "wpm_caveat": self.wpm_caveat,
            "transcript": self.transcript,
            "reference_text": self.reference_text,
            "alignment": [
                {
                    "ref": a.ref,
                    "hyp": a.hyp,
                    "op": a.op,
                    **(
                        {"forced_confidence": round(a.forced_confidence, 2)}
                        if a.forced_confidence is not None
                        else {}
                    ),
                    **({"note": a.note} if a.note else {}),
                }
                for a in self.aligned
            ],
            "forced_alignment_used": self.forced_alignment_used,
            "phoneme_issues": [
                {
                    "word": d.word,
                    "expected": d.expected_ipa,
                    "produced": d.produced_ipa,
                    "weak_phoneme": d.weak_phonemes_ipa,
                    "confidence": round(d.confidence, 2),
                }
                for d in self.phoneme_diffs
                # Skip fully-absent productions — those are already represented
                # as "del" in the alignment; listing them here as weak_phoneme
                # equal to the full word IPA is noise.
                if d.weak_phonemes and d.produced_arpa
            ],
            "korean_l1_patterns": [
                {
                    "pattern": p.pattern,
                    "label": p.label,
                    "examples": p.examples,
                    "count": p.count,
                    "tip_ko": p.tip_ko,
                    "drill": p.drill,
                }
                for p in self.korean_l1_patterns
            ],
            "prosody": {
                "wrong_word_stress": [
                    {
                        "word": e.word,
                        "expected_syllable": e.expected_stress_syllable,
                        "observed_syllable": e.observed_stress_syllable,
                    }
                    for e in self.prosody.wrong_word_stress
                ],
                "final_rise_on_declarative": self.prosody.final_rise_on_declarative,
                "intra_clause_pauses": [
                    {
                        "before": p.before,
                        "after": p.after,
                        "duration_sec": round(p.duration, 2),
                        "at_sec": round(p.start, 2),
                    }
                    for p in self.prosody.intra_clause_pauses
                ],
                "unavailable": self.prosody.unavailable,
            },
            "drills": [{"reason": d.reason, "minimal_pairs": d.minimal_pairs} for d in self.drills],
        }

    # ---- markdown rendering ----------------------------------------

    def format_report(self) -> str:
        """Markdown report for `practice` / `assess`.

        Replaces the old "X heard as Y" list with:
          - alignment table (match/sub/ins/del)
          - phoneme issues (expected vs produced IPA, weak phoneme)
          - learner-profile hints with tips + drills
          - prosody findings
        """
        lines: list[str] = []
        lines.append("## Pronunciation Assessment\n")

        lines.append(f"**You said:** {self.transcript or '(nothing detected)'}")
        if self.reference_text:
            lines.append(f"**Target:** {self.reference_text}")
        lines.append("")

        wpm = self.words_per_minute
        lines.append(
            f"**Clarity:** {self.clarity_pct}% | **Speed:** {wpm:.0f} WPM ({_speed_label(wpm)})"
        )
        if self.wpm_caveat:
            lines.append(f"*Note: WPM {self.wpm_caveat}.*")
        lines.append("")

        # Alignment — only show when there are real mismatches or FA notes.
        if self.aligned and self.reference_text:
            non_match = [a for a in self.aligned if a.op != "match" or a.note]
            if non_match:
                lines.append("### Alignment")
                if self.forced_alignment_used:
                    lines.append("| Reference | You said |  | Conf |")
                    lines.append("|---|---|---|---|")
                else:
                    lines.append("| Reference | You said |  |")
                    lines.append("|---|---|---|")
                for a in self.aligned:
                    ref = a.ref or "—"
                    hyp = a.hyp or "—"
                    marker = {
                        "match": "✓",
                        "sub": "≠",
                        "ins": "+",
                        "del": "−",
                    }[a.op]
                    conf_cell = ""
                    if self.forced_alignment_used:
                        c = a.forced_confidence
                        conf_cell = f" {c:.0%} |" if c is not None else " — |"
                    row = f"| {ref} | {hyp} | {marker} {a.op} |{conf_cell}"
                    lines.append(row)
                    if a.note:
                        lines.append(
                            f"|  |  | *{a.note}* |" + (" |" if self.forced_alignment_used else "")
                        )
                lines.append("")

        # Phoneme issues (skip fully-deleted words — already shown in alignment).
        phoneme_hits = [d for d in self.phoneme_diffs if d.weak_phonemes and d.produced_arpa]
        if phoneme_hits:
            lines.append("### Phoneme issues")
            for d in phoneme_hits:
                lines.append(
                    f"- **{d.word}** — expected {d.expected_ipa}, "
                    f"produced {d.produced_ipa} — weak: **{d.weak_phonemes_ipa}** "
                    f"({int(d.confidence * 100)}% phoneme match)"
                )
            lines.append("")

        # Learner-profile hints.
        if self.korean_l1_patterns:
            lines.append("### Learner-profile hints")
            for p in self.korean_l1_patterns:
                examples = ", ".join(p.examples[:3])
                lines.append(f"- **{p.label}** ({p.count}× — {examples})")
                lines.append(f"  - Tip: {p.tip_ko}")
                lines.append(f"  - Drill: {' · '.join(p.drill[:4])}")
            lines.append("")

        # Prosody.
        prosody_lines: list[str] = []
        if self.prosody.final_rise_on_declarative:
            prosody_lines.append(
                "- Sentence ended with rising intonation. Declaratives should fall."
            )
        if self.prosody.wrong_word_stress:
            words_fmt = ", ".join(
                f"{e.word} (syl {e.observed_stress_syllable + 1} instead of {e.expected_stress_syllable + 1})"
                for e in self.prosody.wrong_word_stress[:3]
            )
            prosody_lines.append(f"- Misplaced word stress: {words_fmt}")
        if self.prosody.intra_clause_pauses:
            pfmt = ", ".join(
                f"{p.duration:.2f}s between '{p.before}' and '{p.after}'"
                for p in self.prosody.intra_clause_pauses[:3]
            )
            prosody_lines.append(f"- Hesitation mid-clause: {pfmt}")
        if prosody_lines:
            lines.append("### Prosody")
            lines.extend(prosody_lines)
            lines.append("")

        # Drills.
        if self.drills:
            lines.append("### Drill these")
            for d in self.drills:
                pairs = ", ".join(d.minimal_pairs[:4])
                lines.append(f"- **{d.reason}**: {pairs}")
            lines.append("")

        # Grammar (still useful for any path).
        grammar = self.grammar_notes()
        if grammar:
            lines.append("### Grammar")
            for wrong, correct, _explain in grammar:
                lines.append(f'- *"{wrong}"* → *"{correct}"*')
            lines.append("")

        # Clean report if nothing surfaced.
        if (
            not phoneme_hits
            and not self.korean_l1_patterns
            and not prosody_lines
            and not grammar
            and (not self.aligned or all(a.op == "match" for a in self.aligned))
        ):
            lines.append("### Great job! No major issues detected.\n")

        return "\n".join(lines)

    def format_converse_report(self, has_target: bool = False) -> str:
        """Conversational flavor of the report for the `converse` tool.

        Drops the alignment table and drill list; keeps a couple of
        high-signal bullets and the "For Claude" guidance block.
        """
        lines: list[str] = []
        lines.append("## User said\n")
        if self.transcript:
            lines.append(f"> {self.transcript}\n")
        else:
            lines.append("> *(nothing clearly transcribed — may have been silent or too quiet)*\n")

        wpm = self.words_per_minute
        lines.append(
            f"**Clarity:** {self.clarity_pct}% &nbsp;|&nbsp; "
            f"**Pace:** {wpm:.0f} WPM ({_speed_label(wpm)})\n"
        )

        feedback: list[str] = []

        for wrong, correct, explanation in self.grammar_notes():
            feedback.append(f'**Grammar:** *"{wrong}"* → *"{correct}"* — {explanation}')

        if has_target and self.reference_text and self.aligned:
            subs = [a for a in self.aligned if a.op == "sub"][:2]
            dels = [a for a in self.aligned if a.op == "del"][:1]
            for a in subs:
                feedback.append(f'**Pronunciation:** *"{a.ref}"* → heard *"{a.hyp}"*')
            for a in dels:
                feedback.append(f'**Pronunciation:** *"{a.ref}"* was skipped')

        # Surface top phoneme issue (single bullet, keep the report short).
        phoneme_hits = [d for d in self.phoneme_diffs if d.weak_phonemes]
        if phoneme_hits and len(feedback) < 4:
            d = phoneme_hits[0]
            feedback.append(
                f'**Phoneme:** *"{d.word}"* — weak {d.weak_phonemes_ipa} '
                f"(expected {d.expected_ipa})"
            )

        # Top learner-profile hint (single bullet).
        if self.korean_l1_patterns and len(feedback) < 5:
            p = self.korean_l1_patterns[0]
            feedback.append(f"**Learner profile:** {p.label}: {p.tip_ko}")

        if self.prosody.final_rise_on_declarative:
            feedback.append(
                "**Intonation:** statement ended with rising pitch (sounded like a question)"
            )

        if self.get_pauses(1.2):
            count = len(self.get_pauses(1.2))
            if count >= 2:
                feedback.append(f"**Fluency:** {count} long pauses — try to keep the flow")

        if feedback:
            lines.append("## Quick feedback\n")
            for b in feedback[:5]:
                lines.append(f"- {b}")
            lines.append("")
        else:
            lines.append("## Quick feedback\n")
            lines.append("- No obvious issues — clear and natural.\n")

        lines.append("## For Claude\n")
        if not self.transcript:
            lines.append(
                "The user's recording was silent or very quiet. Ask them to repeat, "
                "and consider suggesting they run `check_mic` if this happens twice."
            )
        elif feedback:
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


DEFAULT_MODEL = whisper_model_name()


def _detect_device() -> tuple[str, str]:
    """Auto-detect the best device and compute type for faster-whisper."""
    try:
        import ctranslate2

        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            logger.info("CUDA detected, using GPU")
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


class PronunciationAssessor:
    """Orchestrates Whisper + alignment + phonemes + prosody."""

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
        """Assess pronunciation of an audio file.

        Whisper decoding is NOT biased by the reference text — biasing
        causes Whisper to fill in words the user actually skipped. Instead,
        Whisper-bias mitigation happens below via wav2vec2 CTC forced
        alignment, which checks acoustic evidence against each reference
        word independently of Whisper's language-model-weighted decoder.
        """
        model = self._get_model()

        # We intentionally do NOT bias Whisper with `initial_prompt=reference`:
        # that causes Whisper to fill in words the user actually skipped,
        # masking real errors. Whisper's own bias toward common n-grams is
        # instead mitigated by the wav2vec2 forced-alignment step below,
        # which checks whether the user acoustically produced each reference
        # word regardless of what Whisper's decoder output.
        segments_iter, info = model.transcribe(
            str(audio_path),
            language="en",
            beam_size=5,
            best_of=1,
            temperature=0.0,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        words: list[WordResult] = []
        full_text_parts: list[str] = []
        for segment in segments_iter:
            full_text_parts.append(segment.text.strip())
            if segment.words:
                for w in segment.words:
                    words.append(
                        WordResult(
                            word=w.word.strip(),
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                    )
        transcript = " ".join(full_text_parts).strip()
        transcript = re.sub(r"\s+", " ", transcript)

        # Speech duration: sum of per-word spans. Correctly excludes silence
        # between words and fixes the v0.2 WPM bug where long pauses inflated
        # the denominator.
        speech_dur = sum(max(0.0, w.end - w.start) for w in words)

        result = AssessmentResult(
            transcript=transcript,
            reference_text=reference_text,
            words=words,
            duration_sec=info.duration,
            speech_duration_sec=speech_dur,
            language=info.language,
            language_prob=info.language_probability,
        )

        # Alignment + phoneme diffs + learner-profile hints run only with a reference.
        if reference_text:
            self._run_reference_analysis(result, audio_path)

        # Prosody runs in all cases (needs audio + word timings only). When
        # forced alignment is available, prefer its timestamps over Whisper's —
        # they cover the full audio even when Whisper truncates the transcript.
        prosody_words = self._prosody_words(result, reference_text)
        result.prosody = prosody_analyze(audio_path, prosody_words, reference_text)

        return result

    # -----------------------------------------------------------------

    # Forced-alignment confidence thresholds. Calibrated on the five
    # regression clips; tune if they miscategorize many cases.
    _FA_MATCH_THRESHOLD = 0.50  # >= this -> user produced the word
    _FA_WEAK_THRESHOLD = 0.25  # between WEAK and MATCH -> unclear production

    def _run_reference_analysis(
        self,
        r: AssessmentResult,
        audio_path: Path,
    ) -> None:
        """Populate alignment / phoneme_diffs / korean_l1_patterns / drills."""
        ref_tokens = tokenize(r.reference_text or "")
        hyp_tokens = tokenize(r.transcript)
        aligned = align_words(ref_tokens, hyp_tokens)

        # --- forced alignment overlay -------------------------------
        # Run wav2vec2 CTC forced alignment when torch+torchaudio are
        # available. This verifies whether each reference word was actually
        # produced, even if Whisper misheard it.
        fa = forced_align.align(audio_path, r.reference_text or "")
        fa_by_ref_idx: dict[int, float] = {}
        fa_spans: dict[int, tuple[float, float]] = {}
        if fa is not None:
            r.forced_alignment_used = True
            for w in fa.words:
                fa_by_ref_idx[w.ref_index] = w.confidence
                fa_spans[w.ref_index] = (w.start, w.end)

        for a in aligned:
            if a.ref_index is not None and a.ref_index in fa_by_ref_idx:
                a.forced_confidence = fa_by_ref_idx[a.ref_index]

        # Adjust ops based on forced alignment. Two cases:
        #   (a) op=="sub" but forced confidence high -> user said it correctly,
        #       Whisper biased to a similar-sounding common word.
        #   (b) op=="del" but forced confidence high -> user said it, Whisper
        #       truncated or dropped the word.
        # In both cases flip to "match" and attach a note. The hyp field is
        # kept for debug visibility into Whisper's mistake.
        if r.forced_alignment_used:
            for a in aligned:
                if a.forced_confidence is None:
                    continue
                if a.op == "sub" and a.forced_confidence >= self._FA_MATCH_THRESHOLD:
                    a.op = "match"
                    a.note = f"Whisper misheard as '{a.hyp}'; acoustic evidence matched"
                elif a.op == "del" and a.forced_confidence >= self._FA_MATCH_THRESHOLD:
                    a.op = "match"
                    a.note = "Whisper dropped this word but acoustic evidence matched"
                elif a.op == "match" and a.forced_confidence < self._FA_WEAK_THRESHOLD:
                    # Whisper accepted it but acoustic evidence is weak — likely
                    # an unclear production that Whisper guessed from context.
                    a.note = f"Low acoustic confidence ({a.forced_confidence:.0%})"

        r.aligned = aligned

        # --- timestamps for learner-profile hint examples -----------
        ref_timestamps: dict[int, float] = {}
        # Prefer forced-alignment timestamps (cover full audio).
        for idx, (start, _end) in fa_spans.items():
            if start > 0:
                ref_timestamps[idx] = start
        # Back-fill from Whisper word timing when forced timing was zero
        # (word was missing per FA).
        hyp_token_to_word: dict[int, WordResult] = {}
        flat_hyp_idx = 0
        for wr in r.words:
            for _tok in tokenize(wr.word):
                hyp_token_to_word[flat_hyp_idx] = wr
                flat_hyp_idx += 1
        for a in aligned:
            if a.ref_index is None or a.ref_index in ref_timestamps:
                continue
            if a.hyp_index is not None and a.hyp_index in hyp_token_to_word:
                ref_timestamps[a.ref_index] = hyp_token_to_word[a.hyp_index].start

        # --- phoneme diffs (skip ops flipped to match by FA) ----------
        diffs: list[PhonemeDiff] = []
        for a in aligned:
            if a.op == "match" or a.ref is None:
                continue
            d = diff_word(a.ref, a.hyp)
            if d is not None:
                diffs.append(d)
        r.phoneme_diffs = diffs

        # Learner-profile rules still run over the full alignment: some patterns
        # (e.g. article_omission) fire on `del` ops, and cluster-deletion fires
        # on phoneme diffs.
        r.korean_l1_patterns = detect_patterns(aligned, diffs, ref_timestamps)
        r.drills = suggest_drills(r.korean_l1_patterns, diffs)

    def _prosody_words(
        self,
        r: AssessmentResult,
        reference_text: str | None,
    ) -> list[TimedWord]:
        """Build the TimedWord list prosody operates on.

        Prefers forced-alignment timestamps (reference-word-keyed, spans the
        full audio) and falls back to Whisper's word timings when FA is
        unavailable or when a specific reference word wasn't produced.
        """
        # FA timestamps would require a separate span channel on AlignedWord;
        # for now we use Whisper's word timings, which are adequate for the
        # coarse pause + pitch checks we run.
        from_whisper = [
            TimedWord(word=w.word.strip(" ,.?!"), start=w.start, end=w.end) for w in r.words
        ]

        # Annotate clause boundaries from the reference text when available.
        if reference_text:
            from .prosody import mark_clause_boundaries

            boundaries = mark_clause_boundaries(reference_text, r.aligned)
            # Whisper tokens don't map 1:1 to reference tokens, so this is
            # best-effort: mark the whisper word as boundary if its token
            # index matches a boundary in the reference.
            for i, tw in enumerate(from_whisper):
                if i in boundaries:
                    tw.is_clause_boundary = boundaries[i]

        return from_whisper
