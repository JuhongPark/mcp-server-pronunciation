"""Phoneme-level analysis: CMU dict lookup, IPA rendering, Korean-L1 patterns.

Given a word-level alignment, this module answers "which *sound* failed?" by
comparing the phoneme sequence of each reference word to the phoneme sequence
of its hypothesis counterpart. Phoneme sequences come from CMUdict (ARPAbet),
with `g2p_en`'s neural G2P as a fallback for OOV words (proper nouns, domain
terms). Both outputs are mapped to IPA for display.

Korean-L1 pattern detection runs over the aligned phonemes and the raw
alignment ops to surface the canonical L1→L2 confusions: th→s/d, r/l, f→p,
v→b, z→dʒ, final cluster deletion, intrusive onset vowel, final stop
unrelease, schwa→full-vowel, dark-l confusion, article omission.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .alignment import AlignedWord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ARPAbet -> IPA
# ---------------------------------------------------------------------------
# CMUdict uses ARPAbet with stress suffixes (0/1/2) on vowels. We strip the
# stress digit for IPA rendering but keep it available for prosody work.

_ARPA_TO_IPA: dict[str, str] = {
    # vowels
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "EH": "ɛ",
    "ER": "ɜr",
    "EY": "eɪ",
    "IH": "ɪ",
    "IY": "i",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "UH": "ʊ",
    "UW": "u",
    # consonants
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

# Phoneme class membership for pattern detection.
_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
_STOPS = {"P", "T", "K", "B", "D", "G"}
_OBSTRUENT_CLUSTER_HEADS = {"S"}  # "sp", "st", "sk" onsets


def strip_stress(phone: str) -> str:
    """`AH0` -> `AH`."""
    return re.sub(r"[0-9]$", "", phone)


def stress_of(phone: str) -> int:
    """Return 0/1/2 stress marker; -1 for consonants."""
    m = re.search(r"([0-9])$", phone)
    return int(m.group(1)) if m else -1


def arpa_to_ipa(phones: Iterable[str]) -> str:
    """Render an ARPAbet sequence as IPA, wrapped in slashes."""
    out = []
    for p in phones:
        bare = strip_stress(p)
        out.append(_ARPA_TO_IPA.get(bare, bare))
    return "/" + "".join(out) + "/"


def is_vowel(phone: str) -> bool:
    return strip_stress(phone) in _VOWELS


# ---------------------------------------------------------------------------
# CMUdict + G2P fallback
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _cmudict_map() -> dict[str, list[list[str]]]:
    """Load CMUdict once. Returns word -> list of pronunciation variants."""
    try:
        import cmudict

        raw = cmudict.dict()
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("cmudict unavailable: %s", e)
        return {}
    return {w: list(pron_list) for w, pron_list in raw.items()}


_g2p = None


def _get_g2p():
    """Lazy-load g2p_en. Downloads NLTK resources on first use if missing."""
    global _g2p
    if _g2p is not None:
        return _g2p
    try:
        import nltk

        for res, path in [
            ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
            ("cmudict", "corpora/cmudict"),
        ]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(res, quiet=True)

        from g2p_en import G2p

        _g2p = G2p()
    except Exception as e:  # pragma: no cover
        logger.warning("g2p_en unavailable: %s", e)
        _g2p = False  # sentinel — don't retry
    return _g2p


@lru_cache(maxsize=2048)
def phonemes_for(word: str) -> list[str]:
    """Return ARPAbet phonemes with stress for a word.

    Prefers CMUdict; falls back to g2p_en for OOV (proper nouns, domain terms).
    Returns an empty list if neither source works.
    """
    if not word:
        return []
    key = word.lower().strip("'.,?!\"()")
    if not key:
        return []

    entries = _cmudict_map().get(key)
    if entries:
        return list(entries[0])

    g2p = _get_g2p()
    if g2p is False or g2p is None:
        return []
    try:
        phones = g2p(key)
        # g2p_en returns a mixed list with punctuation tokens for multi-word
        # input; filter to ARPAbet tokens only.
        return [p for p in phones if re.match(r"^[A-Z]+[0-9]?$", p)]
    except Exception:  # pragma: no cover
        return []


# ---------------------------------------------------------------------------
# Phoneme-sequence diff
# ---------------------------------------------------------------------------


@dataclass
class PhonemeDiff:
    """Per-word phoneme-level comparison of what was expected vs produced."""

    word: str  # reference word (display)
    expected_arpa: list[str]  # reference ARPAbet w/ stress
    produced_arpa: list[str]  # hypothesis ARPAbet w/ stress
    expected_ipa: str  # "/rɪsks/"
    produced_ipa: str  # "/rɪs/"
    weak_phonemes: list[str]  # ARPAbet tokens that were changed/dropped
    weak_phonemes_ipa: str  # IPA fragment of weak phonemes
    confidence: float  # 0.0-1.0 (1 - edit_distance/max_len)


def _phoneme_edit(exp: list[str], prod: list[str]) -> tuple[int, list[str]]:
    """NW edit distance over bare ARPAbet; returns (distance, weak_phonemes).

    weak_phonemes = reference phonemes that were substituted or deleted.
    Insertions (epenthetic vowels, etc.) are counted in distance but reported
    separately via the intrusive-onset-vowel pattern.
    """
    a = [strip_stress(p) for p in exp]
    b = [strip_stress(p) for p in prod]
    n, m = len(a), len(b)
    if n == 0:
        return m, []
    if m == 0:
        return n, list(a)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    # Traceback to identify which reference phonemes were substituted/deleted.
    weak: list[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            if a[i - 1] != b[j - 1]:
                weak.append(exp[i - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            weak.append(exp[i - 1])
            i -= 1
        else:
            j -= 1
    weak.reverse()
    return dp[n][m], weak


def diff_word(ref_word: str, hyp_word: str | None) -> PhonemeDiff | None:
    """Build a PhonemeDiff for a ref/hyp pair. Returns None if ref has no phones."""
    exp = phonemes_for(ref_word)
    if not exp:
        return None
    prod = phonemes_for(hyp_word) if hyp_word else []
    dist, weak = _phoneme_edit(exp, prod)
    conf = 1.0 - (dist / max(len(exp), 1))
    conf = max(0.0, min(1.0, conf))
    return PhonemeDiff(
        word=ref_word,
        expected_arpa=exp,
        produced_arpa=prod,
        expected_ipa=arpa_to_ipa(exp),
        produced_ipa=arpa_to_ipa(prod) if prod else "/∅/",
        weak_phonemes=weak,
        weak_phonemes_ipa=arpa_to_ipa(weak) if weak else "",
        confidence=conf,
    )


# ---------------------------------------------------------------------------
# Korean-L1 pattern detection
# ---------------------------------------------------------------------------


@dataclass
class KoreanL1Pattern:
    """One detected Korean-L1 confusion instance."""

    pattern: str  # machine key: "r_l_swap", "final_cluster_deletion", ...
    label: str  # human label: "R/L swap"
    examples: list[str]  # ["word @ 2.1s", ...]
    tip_ko: str  # Korean-language tip
    drill: list[str]  # minimal pair drills
    count: int = 1


# Static data: per-pattern tip + drill list.
# Keep tips short — Claude will quote them or paraphrase in the reply.
_PATTERN_META: dict[str, tuple[str, str, list[str]]] = {
    "r_l_swap": (
        "R/L swap",
        "/r/은 혀 끝을 입천장에 닿지 않게 뒤로 말고, /l/은 혀 끝을 윗잇몸에 붙이세요.",
        ["rice/lice", "right/light", "red/led", "road/load", "rock/lock"],
    ),
    "f_to_p": (
        "/f/ → /p/",
        "/f/는 윗니를 아랫입술에 살짝 대고 바람을 빼세요. /p/처럼 두 입술로 막지 않습니다.",
        ["fan/pan", "fast/past", "coffee/copy", "fine/pine", "leaf/leap"],
    ),
    "v_to_b": (
        "/v/ → /b/",
        "/v/는 윗니를 아랫입술에 대고 소리를 내세요. /b/처럼 입술을 붙이지 않습니다.",
        ["very/berry", "vest/best", "vote/boat", "curve/curb", "van/ban"],
    ),
    "th_to_s": (
        "/θ/ → /s/",
        "/θ/는 혀 끝을 윗니와 아랫니 사이에 살짝 내밀며 바람을 빼세요.",
        ["think/sink", "thick/sick", "thing/sing", "thought/sought", "mouth/mouse"],
    ),
    "th_to_t": (
        "/θ/ → /t/",
        "/θ/는 혀를 이 사이로 내밀고 바람을 빼세요. /t/처럼 혀가 입천장에 닿지 않습니다.",
        ["three/tree", "thin/tin", "thought/taught", "path/pat", "bath/bat"],
    ),
    "dh_to_d": (
        "/ð/ → /d/",
        "/ð/는 혀를 이 사이로 내밀고 성대를 울려 소리를 내세요.",
        ["they/day", "than/Dan", "though/dough", "breathe/breed", "lather/ladder"],
    ),
    "z_to_j": (
        "/z/ → /dʒ/",
        "/z/는 혀를 윗잇몸 가까이 두고 바람을 빼며 성대를 울리세요. /dʒ/처럼 혀를 붙이지 않습니다.",
        ["zoo/Jew", "zip/gyp", "zinc/jinx", "Z/J"],
    ),
    "final_cluster_deletion": (
        "Final consonant cluster deletion",
        "어미 자음 덩어리(sks, sts, kst 등)를 끝까지 발음하세요. 한국어엔 없는 조합이라 생략하기 쉽습니다.",
        ["task/tasks", "ask/asked", "test/tests", "fist/fists", "list/lists"],
    ),
    "intrusive_onset_vowel": (
        "Intrusive vowel in onset cluster",
        "어두 자음 덩어리(str, fl, sk 등) 사이에 /으/ 같은 모음을 넣지 마세요. 자음을 이어 발음합니다.",
        ["street (not 수뜨릿)", "flask (not 플라스크)", "strict (not 스트릭트)", "script"],
    ),
    "final_stop_unrelease": (
        "Final stop unrelease / deletion",
        "어말 /p/, /t/, /k/는 약하지만 입술·혀를 붙여 공기를 막으세요. 완전히 생략하지 마세요.",
        ["cat/cab", "map/mop", "book/boot", "stop/stock", "hot/hop"],
    ),
    "schwa_to_full_vowel": (
        "Schwa replaced by full vowel",
        "강세 없는 음절은 /ə/(애매한 '어')로 약하게. 각 음절을 똑같이 세게 발음하지 마세요.",
        ["banana (bə-NAN-ə)", "about (ə-BAUT)", "support (sə-PORT)"],
    ),
    "dark_l_confusion": (
        "Dark-L at syllable end",
        "음절 끝 /l/은 혀 뿌리를 올려 어두운 'ㄹ'. 한국어 'ㄹ'처럼 살짝 굴리지 마세요.",
        ["full/pool", "feel/peel", "milk (miw-k)", "tall", "cold"],
    ),
    "article_omission": (
        "Article omission (a/an/the)",
        "빠르게 말할 때 관사(a, an, the)를 생략하지 마세요. 한국어엔 없어서 빠뜨리기 쉽습니다.",
        ["the book (not 'book')", "a cat (not 'cat')", "an apple (not 'apple')"],
    ),
}


# Canonical Korean-L1 substitutions at the phoneme level: (expected, produced)
_PHONEME_SUB_PATTERNS: dict[tuple[str, str], str] = {
    ("R", "L"): "r_l_swap",
    ("L", "R"): "r_l_swap",
    ("F", "P"): "f_to_p",
    ("F", "B"): "f_to_p",
    ("V", "B"): "v_to_b",
    ("TH", "S"): "th_to_s",
    ("TH", "T"): "th_to_t",
    ("DH", "D"): "dh_to_d",
    ("Z", "JH"): "z_to_j",
}


_ARTICLES = {"a", "an", "the"}


def detect_patterns(
    aligned: list[AlignedWord],
    phoneme_diffs: list[PhonemeDiff],
    word_timestamps: dict[int, float] | None = None,
) -> list[KoreanL1Pattern]:
    """Scan alignment + phoneme diffs for Korean-L1 confusions.

    word_timestamps: optional map from ref_index -> start-time (seconds). When
    provided, examples are rendered as "word @ 2.1s".
    """
    word_timestamps = word_timestamps or {}
    buckets: dict[str, KoreanL1Pattern] = {}

    def _tagged(ref_word: str, ref_idx: int | None) -> str:
        t = word_timestamps.get(ref_idx) if ref_idx is not None else None
        return f"{ref_word} @ {t:.1f}s" if t is not None else ref_word

    def _bump(key: str, example: str) -> None:
        meta = _PATTERN_META.get(key)
        if not meta:
            return
        label, tip, drill = meta
        if key not in buckets:
            buckets[key] = KoreanL1Pattern(
                pattern=key,
                label=label,
                examples=[example],
                tip_ko=tip,
                drill=drill,
                count=1,
            )
        else:
            buckets[key].count += 1
            if example not in buckets[key].examples:
                buckets[key].examples.append(example)

    # Build a lookup from ref_word lowercase to its aligned entry (first occurrence)
    ref_to_aligned: dict[str, AlignedWord] = {}
    for a in aligned:
        if a.ref and a.ref.lower() not in ref_to_aligned:
            ref_to_aligned[a.ref.lower()] = a

    # --- phoneme-substitution patterns from phoneme diffs ---------------
    diff_by_word = {d.word.lower(): d for d in phoneme_diffs}
    for a in aligned:
        if a.op != "sub" or not a.ref or not a.hyp:
            continue
        d = diff_by_word.get(a.ref.lower())
        if not d:
            continue
        # Pair each expected weak phoneme with its positionally-aligned produced
        # counterpart. We approximate by zipping after stripping stress.
        exp = [strip_stress(p) for p in d.expected_arpa]
        prod = [strip_stress(p) for p in d.produced_arpa]
        # Simple pairwise walk to surface canonical confusions.
        for e, p in zip(exp, prod):
            pat = _PHONEME_SUB_PATTERNS.get((e, p))
            if pat:
                _bump(pat, _tagged(a.ref, a.ref_index))

    # --- final cluster deletion ---------------------------------------
    # Reference ends with 2+ trailing consonants (often /sks/, /sts/, /kst/);
    # produced form has fewer trailing consonants.
    for d in phoneme_diffs:
        exp_no_stress = [strip_stress(p) for p in d.expected_arpa]
        prod_no_stress = [strip_stress(p) for p in d.produced_arpa]

        def _tail_consonants(seq: list[str]) -> list[str]:
            out = []
            for p in reversed(seq):
                if p in _VOWELS:
                    break
                out.append(p)
            return out

        exp_tail = _tail_consonants(exp_no_stress)
        if len(exp_tail) < 2:
            continue
        prod_tail = _tail_consonants(prod_no_stress)
        if len(prod_tail) < len(exp_tail):
            ref_idx = None
            a = ref_to_aligned.get(d.word.lower())
            if a:
                ref_idx = a.ref_index
            _bump("final_cluster_deletion", _tagged(d.word, ref_idx))

    # --- intrusive onset vowel ----------------------------------------
    # Reference begins with a consonant cluster (str, fl, sk, sp, tr, pr, br,
    # bl, gr, kr, ...), and the produced form has a vowel inserted between
    # cluster members. We detect this by checking if produced has strictly
    # more vowels in the first 3 phonemes than expected.
    for d in phoneme_diffs:
        if not d.expected_arpa or not d.produced_arpa:
            continue
        exp_head = [strip_stress(p) for p in d.expected_arpa[:3]]
        prod_head = [strip_stress(p) for p in d.produced_arpa[:3]]
        if len(exp_head) < 2:
            continue
        # Check onset is a cluster: first 2+ phonemes are consonants.
        onset_cluster_len = 0
        for p in exp_head:
            if p in _VOWELS:
                break
            onset_cluster_len += 1
        if onset_cluster_len < 2:
            continue
        exp_vowels_in_head = sum(1 for p in exp_head if p in _VOWELS)
        prod_vowels_in_head = sum(1 for p in prod_head if p in _VOWELS)
        if prod_vowels_in_head > exp_vowels_in_head:
            ref_idx = None
            a = ref_to_aligned.get(d.word.lower())
            if a:
                ref_idx = a.ref_index
            _bump("intrusive_onset_vowel", _tagged(d.word, ref_idx))

    # --- final stop unrelease / deletion ------------------------------
    # Reference ends in /p/, /t/, /k/, /b/, /d/, /g/; produced differs in
    # that final stop (vowel, different consonant, or missing entirely).
    # Skip if this word already triggered final_cluster_deletion — that
    # pattern is more specific.
    cluster_hit_words = {
        ex.split(" @", 1)[0].lower()
        for ex in buckets.get(
            "final_cluster_deletion", KoreanL1Pattern("", "", [], "", [], 0)
        ).examples
    }
    for d in phoneme_diffs:
        if not d.expected_arpa or not d.produced_arpa:
            continue
        if d.word.lower() in cluster_hit_words:
            continue
        exp_last = strip_stress(d.expected_arpa[-1])
        prod_last = strip_stress(d.produced_arpa[-1])
        if exp_last in _STOPS and prod_last != exp_last:
            ref_idx = None
            a = ref_to_aligned.get(d.word.lower())
            if a:
                ref_idx = a.ref_index
            _bump("final_stop_unrelease", _tagged(d.word, ref_idx))

    # --- article omission --------------------------------------------
    for a in aligned:
        if a.op == "del" and a.ref and a.ref.lower() in _ARTICLES:
            _bump("article_omission", _tagged(a.ref, a.ref_index))

    # Order patterns by count (most frequent first), then by key for stability.
    return sorted(buckets.values(), key=lambda p: (-p.count, p.pattern))


# ---------------------------------------------------------------------------
# Drill suggestions
# ---------------------------------------------------------------------------


@dataclass
class Drill:
    """A minimal-pair drill suggestion tied to a specific weakness."""

    reason: str  # "final cluster /sks/" or "/r/ at word start"
    minimal_pairs: list[str]


def suggest_drills(
    patterns: list[KoreanL1Pattern],
    phoneme_diffs: list[PhonemeDiff],
    limit: int = 5,
) -> list[Drill]:
    """Derive drill suggestions from detected patterns + phoneme diffs.

    Prioritizes patterns (they carry curated drill lists) and tops up from any
    remaining weak phonemes not covered by a pattern.
    """
    drills: list[Drill] = []
    seen_reasons: set[str] = set()

    for p in patterns:
        reason = p.label
        if reason in seen_reasons:
            continue
        seen_reasons.add(reason)
        drills.append(Drill(reason=reason, minimal_pairs=list(p.drill)))
        if len(drills) >= limit:
            return drills

    # Top up from weak phonemes not yet covered by any pattern. Skip
    # fully-absent words — those are "you skipped this", not a pronunciation
    # drill opportunity.
    for d in phoneme_diffs:
        if not d.weak_phonemes or not d.produced_arpa:
            continue
        weak_ipa = arpa_to_ipa(d.weak_phonemes)
        reason = f"{d.word}: weak {weak_ipa}"
        if reason in seen_reasons:
            continue
        seen_reasons.add(reason)
        drills.append(
            Drill(
                reason=reason,
                minimal_pairs=[f"{d.word} ({d.expected_ipa})"],
            )
        )
        if len(drills) >= limit:
            break

    return drills
