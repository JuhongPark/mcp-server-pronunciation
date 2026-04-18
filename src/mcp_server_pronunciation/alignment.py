"""Word-level alignment between reference and hypothesis transcripts.

`difflib.SequenceMatcher` (used previously) cascades a single-word drop into a
chain of phantom substitutions. Needleman-Wunsch with explicit edit operations
produces the minimum-cost edit script and keeps the rest of the sequence
correctly aligned around a single insertion or deletion.

The aligner uses a soft equality metric so minor typos or spelling variants
(e.g. "colour" vs "color") register as a single substitution rather than an
ins/del pair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

Op = Literal["match", "sub", "ins", "del"]


@dataclass
class AlignedWord:
    """One entry in the aligned word list.

    - ref: reference token (None for insertions — extra words the speaker added)
    - hyp: hypothesis token (None for deletions — words the speaker skipped)
    - op:  "match" | "sub" | "ins" | "del"
    - ref_index / hyp_index: original positions in the token lists for callers
      that need to look up timestamps.
    - forced_confidence: mean CTC posterior from wav2vec2 forced alignment when
      available (None otherwise). High values on a `sub`/`del` entry mean the
      user probably said the reference word correctly and Whisper misheard.
    - note: free-form annotation added by the assessor (e.g.
      "Whisper misheard; acoustic evidence matched").
    """

    ref: str | None
    hyp: str | None
    op: Op
    ref_index: int | None = None
    hyp_index: int | None = None
    forced_confidence: float | None = None
    note: str | None = None


# --- tokenization ----------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase + split into word tokens. Strips punctuation and hyphens.

    "front-end" -> ["front", "end"]   (Whisper often splits hyphenated words,
                                       so we do too for consistent alignment)
    "I'm" -> ["i'm"]                  (apostrophes kept inside tokens)
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


# --- soft equality ---------------------------------------------------------


def _similar(a: str, b: str) -> float:
    """Character-ratio similarity in [0, 1]."""
    return SequenceMatcher(None, a, b).ratio()


def _sub_cost(a: str, b: str) -> float:
    """Substitution cost — 0 for exact match, small for near-match, 1 otherwise.

    Near-matches (ratio >= 0.8) are cheaper than 1 so the aligner prefers a
    single substitution over an insert+delete pair for words like
    "colour" -> "color" or "thru" -> "through".
    """
    if a == b:
        return 0.0
    ratio = _similar(a, b)
    if ratio >= 0.85:
        return 0.4
    return 1.0


# --- Needleman-Wunsch ------------------------------------------------------

# Gap (insert or delete) cost. Must be < 1 so that substituting two unrelated
# words is still cheaper than skipping both, otherwise alignment degenerates
# to a staircase of ins/del pairs on long mismatched stretches.
_GAP_COST = 0.9


def align_words(ref: list[str], hyp: list[str]) -> list[AlignedWord]:
    """Align two token lists with Needleman-Wunsch; return edit script."""
    n, m = len(ref), len(hyp)

    if n == 0 and m == 0:
        return []
    if n == 0:
        return [AlignedWord(None, w, "ins", None, j) for j, w in enumerate(hyp)]
    if m == 0:
        return [AlignedWord(w, None, "del", i, None) for i, w in enumerate(ref)]

    # Dynamic programming table. dp[i][j] = min edit cost for ref[:i] vs hyp[:j].
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * _GAP_COST
    for j in range(1, m + 1):
        dp[0][j] = j * _GAP_COST

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = _sub_cost(ref[i - 1], hyp[j - 1])
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,      # match or substitution
                dp[i - 1][j] + _GAP_COST,     # deletion (ref has extra word)
                dp[i][j - 1] + _GAP_COST,     # insertion (hyp has extra word)
            )

    # Traceback. Reconstruct the edit script from the DP table.
    aligned: list[AlignedWord] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = _sub_cost(ref[i - 1], hyp[j - 1])
            diag = dp[i - 1][j - 1] + cost
        else:
            diag = float("inf")
        up = dp[i - 1][j] + _GAP_COST if i > 0 else float("inf")
        left = dp[i][j - 1] + _GAP_COST if j > 0 else float("inf")

        # Tie-break order: diagonal (match/sub) > up (del) > left (ins). This
        # keeps the alignment readable — a substitution is preferred over a
        # pair of gaps when costs are equal.
        best = min(diag, up, left)
        if diag == best:
            op: Op = "match" if ref[i - 1] == hyp[j - 1] else "sub"
            aligned.append(
                AlignedWord(ref[i - 1], hyp[j - 1], op, i - 1, j - 1)
            )
            i -= 1
            j -= 1
        elif up == best:
            aligned.append(AlignedWord(ref[i - 1], None, "del", i - 1, None))
            i -= 1
        else:
            aligned.append(AlignedWord(None, hyp[j - 1], "ins", None, j - 1))
            j -= 1

    aligned.reverse()
    return aligned


# --- convenience -----------------------------------------------------------


def align_texts(reference: str, hypothesis: str) -> list[AlignedWord]:
    """Tokenize both strings and align them."""
    return align_words(tokenize(reference), tokenize(hypothesis))


def summarize(aligned: list[AlignedWord]) -> dict[str, int]:
    """Count each edit op — useful for tests and summary stats."""
    out = {"match": 0, "sub": 0, "ins": 0, "del": 0}
    for a in aligned:
        out[a.op] += 1
    return out
