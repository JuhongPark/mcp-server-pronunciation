"""CTC forced alignment of reference text against audio.

This module answers "did the user actually produce each reference word,
regardless of what Whisper transcribed?". It runs a wav2vec2 CTC model over
the audio and Viterbi-aligns the reference text's characters to frames.

Why this matters: Whisper biases decoding toward common n-grams in its
training data. Rare proper nouns and domain-specific terms get rewritten
toward more frequent alternatives, which the reference alignment then
reports as false mispronunciations on words the user actually said
correctly. Forced alignment against the reference side-steps Whisper's
lexical decoder entirely — it checks whether the acoustic evidence matches
the expected characters, so it tells us what was *produced*, not what
was *decoded*.

Dependencies (torch + torchaudio) are optional. When absent, `align(...)`
returns `None` and the assessor falls back to Whisper-only alignment.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class ForcedWord:
    """One reference word aligned to audio."""

    word: str  # reference token (lowercased, alphanum only)
    ref_index: int  # position in the reference token list
    start: float  # seconds
    end: float  # seconds
    confidence: float  # mean CTC posterior across the span, 0..1
    # Per-character details kept for optional phoneme-span lookup.
    char_spans: list[tuple[str, float, float, float]] = field(default_factory=list)


@dataclass
class ForcedAlignment:
    """Full forced-alignment output for one audio+reference pair."""

    words: list[ForcedWord]
    model_name: str


# ---------------------------------------------------------------------------
# Optional dependency loading
# ---------------------------------------------------------------------------


_BUNDLE = None  # torchaudio.pipelines bundle
_MODEL = None  # wav2vec2 model (quantized)
_LABELS: tuple[str, ...] | None = None
_BLANK_IDX: int | None = None


def is_available() -> bool:
    """Return True if torch+torchaudio are importable."""
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
    except Exception:
        return False
    return True


def _ensure_model() -> bool:
    """Lazy-load wav2vec2. Returns False if unavailable."""
    global _BUNDLE, _MODEL, _LABELS, _BLANK_IDX
    if _MODEL is not None:
        return True
    if not is_available():
        return False
    try:
        import torch
        import torchaudio

        _BUNDLE = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = _BUNDLE.get_model()
        # Dynamic int8 quantization on Linear layers. Runtime memory drops from
        # ~380MB (fp32) to ~95MB on CPU, ~3x faster inference. Accuracy loss is
        # negligible for forced alignment where we care about argmax posteriors.
        try:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception as e:  # pragma: no cover — best-effort
            logger.warning("wav2vec2 quantization failed, using fp32: %s", e)
        model.eval()
        _MODEL = model
        _LABELS = _BUNDLE.get_labels()
        _BLANK_IDX = 0  # wav2vec2 CTC always uses index 0 for blank
        logger.info("wav2vec2 loaded (quantized int8): %d labels", len(_LABELS))
    except Exception as e:
        logger.warning("failed to load wav2vec2: %s", e)
        _MODEL = None
        return False
    return True


# ---------------------------------------------------------------------------
# Text prep — reference -> label-index sequence
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokens_with_spans(reference: str) -> list[tuple[str, int, int]]:
    """Return [(token, char_start, char_end)] for reference word positions."""
    return [(m.group(0).upper(), m.start(), m.end()) for m in _TOKEN_RE.finditer(reference)]


def _prepare_reference(
    reference: str,
    labels: tuple[str, ...],
) -> tuple[list[int], list[tuple[str, int, int]]]:
    """Convert the reference to a sequence of CTC label indices.

    wav2vec2-base-960h labels are uppercase English + `|` (word separator)
    + `-` (blank). We build a transcript with words separated by `|` and
    strip anything outside the label inventory. Returns the index sequence
    plus per-word `(token, seq_start, seq_end)` spans into the index list.
    """
    label_to_idx = {c: i for i, c in enumerate(labels)}
    word_sep = label_to_idx.get("|", 1)

    tokens = _tokens_with_spans(reference)
    seq: list[int] = []
    word_spans: list[tuple[str, int, int]] = []
    for idx, (tok, _s, _e) in enumerate(tokens):
        if idx > 0:
            seq.append(word_sep)
        start_in_seq = len(seq)
        for ch in tok:
            if ch in label_to_idx:
                seq.append(label_to_idx[ch])
            elif ch == "'":
                continue  # drop apostrophes — wav2vec2 labels don't include them
        end_in_seq = len(seq)
        if end_in_seq > start_in_seq:
            word_spans.append((tok, start_in_seq, end_in_seq))
    return seq, word_spans


# ---------------------------------------------------------------------------
# CTC posterior computation
# ---------------------------------------------------------------------------


def _get_posteriors(audio_path: Path):
    """Load audio at 16kHz mono and return CTC log-posteriors [T, V].

    Uses soundfile + librosa for loading to avoid torchaudio 2.11's torchcodec
    dependency. librosa is already a required dep for prosody.
    """
    import librosa
    import numpy as np
    import torch

    audio, sr = librosa.load(str(audio_path), sr=_BUNDLE.sample_rate, mono=True)
    waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    with torch.inference_mode():
        emissions, _ = _MODEL(waveform)
        log_probs = torch.log_softmax(emissions, dim=-1)
    return log_probs[0].numpy(), waveform.shape[1] / _BUNDLE.sample_rate


# ---------------------------------------------------------------------------
# Viterbi trellis alignment (token-level)
# ---------------------------------------------------------------------------


def _viterbi(log_probs, tokens, blank: int):
    """CTC forced-alignment Viterbi.

    Based on the standard two-state-per-token trellis from the torchaudio
    forced-alignment tutorial. Returns per-frame token index (or `blank`).
    """
    import numpy as np

    T = log_probs.shape[0]
    N = len(tokens)
    if N == 0 or T == 0:
        return []

    NEG_INF = -1e30
    # trellis[t, j] = best log-prob to reach token j by frame t.
    trellis = np.full((T, N), NEG_INF, dtype=np.float32)
    backptr = np.zeros((T, N), dtype=np.int8)  # 0=stay on j, 1=move from j-1

    # Initialize: at frame 0, we can emit token 0 OR blank.
    trellis[0, 0] = log_probs[0, tokens[0]]
    for t in range(1, T):
        # Stay on current token j or advance from j-1.
        stay = trellis[t - 1] + log_probs[t, blank]
        advance = np.full(N, NEG_INF, dtype=np.float32)
        # At token j, advance comes from trellis[t-1, j-1] + emit tokens[j]
        advance[1:] = trellis[t - 1, :-1] + log_probs[t, tokens[1:]]
        # Emit token j without consuming from previous (allow doubling blank between identical tokens).
        emit = trellis[t - 1] + log_probs[t, tokens]
        # Transition: max over {stay-with-blank, advance-with-token, emit-token}.
        # For j=0, no advance (can't come from j=-1).
        best = np.maximum(np.maximum(stay, advance), emit)
        trellis[t] = best
        # Only track advance vs stay for traceback simplicity.
        backptr[t] = (advance >= stay).astype(np.int8)

    # Traceback: pick the best ending state (N-1).
    path = [None] * T
    j = N - 1
    for t in range(T - 1, -1, -1):
        # At each frame we could either be emitting tokens[j] or blank —
        # pick whichever has the higher local probability (argmax-style).
        if log_probs[t, tokens[j]] > log_probs[t, blank]:
            path[t] = j
        else:
            path[t] = -1  # blank
        if t > 0 and backptr[t, j] and j > 0:
            j -= 1
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def align(audio_path: Path, reference_text: str) -> ForcedAlignment | None:
    """Forced-align `reference_text` against the audio. Returns None if unavailable.

    On success the result contains one ForcedWord per reference token with
    start/end (seconds) and a confidence score (mean CTC posterior probability
    across the word's frames). Confidence < ~0.4 typically means the user
    skipped or badly mispronounced the word.
    """
    if not _ensure_model():
        return None

    try:
        import numpy as np

        log_probs, duration = _get_posteriors(audio_path)
        seq, word_spans = _prepare_reference(reference_text, _LABELS)
        if not seq or not word_spans:
            return None

        path = _viterbi(log_probs, seq, _BLANK_IDX)
        if not path:
            return None

        T = log_probs.shape[0]
        frame_duration = duration / T

        # For each reference word, find which frames were aligned to one of its
        # token positions in `seq`. The word span in `seq` is [start_in_seq,
        # end_in_seq); find frames whose path is in that range.
        out_words: list[ForcedWord] = []
        for ref_idx, (tok, s_in_seq, e_in_seq) in enumerate(word_spans):
            frames = [
                t
                for t in range(T)
                if path[t] is not None and path[t] != -1 and s_in_seq <= path[t] < e_in_seq
            ]
            if not frames:
                # No frames aligned to this word -> very low confidence, zero span.
                out_words.append(
                    ForcedWord(
                        word=tok.lower(),
                        ref_index=ref_idx,
                        start=0.0,
                        end=0.0,
                        confidence=0.0,
                    )
                )
                continue
            fstart, fend = min(frames), max(frames) + 1
            # Confidence: mean posterior (not log-posterior) across frames on
            # the path tokens within this word.
            posteriors = np.exp([log_probs[t, seq[path[t]]] for t in frames])
            conf = float(np.mean(posteriors))
            out_words.append(
                ForcedWord(
                    word=tok.lower(),
                    ref_index=ref_idx,
                    start=fstart * frame_duration,
                    end=fend * frame_duration,
                    confidence=conf,
                )
            )

        return ForcedAlignment(words=out_words, model_name="wav2vec2_base_960h_int8")
    except Exception as e:
        logger.warning("forced alignment failed: %s", e)
        return None
