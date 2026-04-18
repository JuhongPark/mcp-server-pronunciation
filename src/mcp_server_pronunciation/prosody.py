"""Prosody checks: word stress, final-rise intonation, intra-clause pauses.

Uses librosa's `pyin` for f0 tracking and RMS for intensity. All checks operate
on a single audio file plus per-word timestamps (from Whisper or forced
alignment). When a word can't be timed, that check is skipped — prosody is
always best-effort.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .phonemes import is_vowel, phonemes_for, stress_of

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StressError:
    """A word whose measured stress peak doesn't match the dictionary primary-stress syllable."""

    word: str
    expected_stress_syllable: int      # 0-based index of the primary-stress syllable
    observed_stress_syllable: int      # 0-based index of the syllable with highest pitch+intensity
    start: float
    end: float


@dataclass
class IntraClausePause:
    """A pause ≥ threshold between two words that don't sit on a clause boundary."""

    before: str       # word before the pause
    after: str        # word after the pause
    start: float
    end: float
    duration: float


@dataclass
class ProsodyResult:
    """All prosody findings for one assessment."""

    wrong_word_stress: list[StressError] = field(default_factory=list)
    final_rise_on_declarative: bool = False
    final_pitch_slope: float = 0.0       # semitones/sec over last ~500ms; >2 counts as rise
    intra_clause_pauses: list[IntraClausePause] = field(default_factory=list)
    unavailable: bool = False            # True if librosa not available / audio couldn't be read


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# Pause threshold tuned for coaching feedback: 250ms inside a clause is
# perceptible "hesitation", while cross-clause pauses can be longer without
# sounding unnatural. Users can revisit via empirical feedback.
_INTRA_CLAUSE_PAUSE_SEC = 0.25

# Final-rise threshold: semitones/sec of f0 slope over the last ~500ms of
# voiced audio. Rising ≥2 st/s on a declarative counts as question-like.
_FINAL_RISE_ST_PER_SEC = 2.0


@dataclass
class TimedWord:
    """Minimum timing info prosody needs about each spoken word."""

    word: str
    start: float
    end: float
    # Optional: whether the word is followed by clause-boundary punctuation in
    # the reference text (comma, period, question mark, etc.). When unknown,
    # leave False and we'll treat every gap as potentially intra-clause.
    is_clause_boundary: bool = False


def analyze(
    audio_path: Path,
    words: list[TimedWord],
    reference_text: str | None,
) -> ProsodyResult:
    """Run all prosody checks. Returns empty result with `unavailable=True` on error."""
    try:
        import librosa
        import numpy as np
    except Exception as e:
        logger.warning("librosa unavailable, skipping prosody: %s", e)
        return ProsodyResult(unavailable=True)

    try:
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    except Exception as e:
        logger.warning("failed to load audio for prosody: %s", e)
        return ProsodyResult(unavailable=True)

    result = ProsodyResult()

    # --- final rise on declarative ---------------------------------
    if reference_text and not reference_text.rstrip().endswith("?"):
        result.final_pitch_slope = _final_pitch_slope(y, sr, librosa, np)
        if result.final_pitch_slope >= _FINAL_RISE_ST_PER_SEC:
            result.final_rise_on_declarative = True

    # --- intra-clause pauses --------------------------------------
    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        if gap < _INTRA_CLAUSE_PAUSE_SEC:
            continue
        if words[i - 1].is_clause_boundary:
            continue
        result.intra_clause_pauses.append(
            IntraClausePause(
                before=words[i - 1].word,
                after=words[i].word,
                start=words[i - 1].end,
                end=words[i].start,
                duration=gap,
            )
        )

    # --- word stress --------------------------------------------
    result.wrong_word_stress = _find_stress_errors(y, sr, words, librosa, np)

    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _final_pitch_slope(y, sr, librosa, np) -> float:
    """Fit a linear slope (in semitones/second) to f0 over the last ~500ms."""
    window = int(0.5 * sr)
    tail = y[-window:] if len(y) > window else y
    if len(tail) < sr * 0.2:  # less than 200ms of audio — skip
        return 0.0
    try:
        f0, voiced_flag, _ = librosa.pyin(
            tail,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
        )
    except Exception:
        return 0.0

    import numpy as _np

    mask = _np.isfinite(f0) & voiced_flag
    if mask.sum() < 3:
        return 0.0
    times = _np.arange(len(f0)) * (len(tail) / sr) / len(f0)
    # Convert Hz to semitones relative to the median voiced pitch.
    median_hz = float(_np.nanmedian(f0[mask]))
    if median_hz <= 0:
        return 0.0
    st = 12.0 * _np.log2(f0[mask] / median_hz)
    t = times[mask]
    # Simple least-squares slope.
    if t[-1] - t[0] < 0.1:
        return 0.0
    slope, _intercept = _np.polyfit(t, st, 1)
    return float(slope)


def _find_stress_errors(
    y,
    sr,
    words: list[TimedWord],
    librosa,
    np,
) -> list[StressError]:
    """Per multi-syllable word: compare expected primary-stress syllable to
    the syllable with the highest combined pitch+RMS in the word's audio span."""
    import numpy as _np

    errors: list[StressError] = []

    for tw in words:
        phones = phonemes_for(tw.word)
        if not phones:
            continue
        # Build syllables by splitting on vowels. Each syllable is (index,
        # stress). We only check words with 2+ syllables.
        syllables: list[tuple[int, int]] = []
        for idx, p in enumerate(phones):
            if is_vowel(p):
                syllables.append((idx, stress_of(p)))
        if len(syllables) < 2:
            continue

        # Identify the primary-stress syllable (stress=1). Fall back to the
        # secondary-stress syllable (2) if no primary — rare, defensive.
        expected = next(
            (i for i, (_, s) in enumerate(syllables) if s == 1),
            None,
        )
        if expected is None:
            continue

        # Measure pitch+RMS across the word's audio span.
        i0 = max(0, int(tw.start * sr))
        i1 = min(len(y), int(tw.end * sr))
        if i1 - i0 < sr * 0.08:  # <80ms of audio — too short
            continue
        segment = y[i0:i1]
        try:
            f0, voiced, _ = librosa.pyin(
                segment,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C6"),
                sr=sr,
                fill_na=_np.nan,
            )
            rms = librosa.feature.rms(y=segment, frame_length=512, hop_length=128)[0]
        except Exception:
            continue

        # Normalize both signals to [0, 1] and combine.
        f0_clean = _np.where(voiced & _np.isfinite(f0), f0, _np.nan)
        if _np.all(_np.isnan(f0_clean)):
            continue
        f0_norm = (f0_clean - _np.nanmin(f0_clean)) / max(
            1e-9, (_np.nanmax(f0_clean) - _np.nanmin(f0_clean))
        )
        # pyin and rms have different frame rates; resample rms to f0 length.
        rms_resamp = _np.interp(
            _np.linspace(0, 1, len(f0_norm)),
            _np.linspace(0, 1, len(rms)),
            rms,
        )
        rms_norm = (rms_resamp - rms_resamp.min()) / max(1e-9, (rms_resamp.max() - rms_resamp.min()))
        # Combined energy per frame; nan-safe.
        combined = _np.where(_np.isnan(f0_norm), rms_norm * 0.5, f0_norm * 0.7 + rms_norm * 0.3)

        # Bucket combined energy into each syllable by proportional duration.
        # Syllables are approximately equally spaced in time; this is a coarse
        # heuristic but sufficient for "which syllable is loudest + highest".
        per_syllable_energy: list[float] = []
        n_frames = len(combined)
        for k in range(len(syllables)):
            lo = int(n_frames * k / len(syllables))
            hi = int(n_frames * (k + 1) / len(syllables))
            if hi <= lo:
                per_syllable_energy.append(0.0)
                continue
            per_syllable_energy.append(float(_np.nanmean(combined[lo:hi])))
        observed = int(_np.argmax(per_syllable_energy))
        if observed != expected:
            errors.append(
                StressError(
                    word=tw.word,
                    expected_stress_syllable=expected,
                    observed_stress_syllable=observed,
                    start=tw.start,
                    end=tw.end,
                )
            )
    return errors


# ---------------------------------------------------------------------------
# Reference-text clause-boundary helper
# ---------------------------------------------------------------------------


def mark_clause_boundaries(
    reference_text: str,
    aligned_words: list,
) -> dict[int, bool]:
    """Return ref_index -> True if that reference word is followed by a clause
    boundary (, . ; : ! ?) in the original reference text. Used by the caller
    to populate TimedWord.is_clause_boundary.
    """
    import re as _re

    if not reference_text:
        return {}
    # Find raw tokens together with their following punctuation.
    tokens = _re.findall(r"([A-Za-z0-9']+)([^A-Za-z0-9']*)", reference_text)
    out: dict[int, bool] = {}
    for idx, (_tok, trailing) in enumerate(tokens):
        out[idx] = any(c in trailing for c in ",.;:!?")
    return out
