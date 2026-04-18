"""Tests for phoneme lookup, IPA rendering, and Korean-L1 pattern detection."""

from mcp_server_pronunciation.alignment import align_texts
from mcp_server_pronunciation.phonemes import (
    arpa_to_ipa,
    detect_patterns,
    diff_word,
    phonemes_for,
    strip_stress,
    suggest_drills,
)


class TestPhonemeLookup:
    def test_common_word(self):
        assert phonemes_for("cat") == ["K", "AE1", "T"]

    def test_cluster_word(self):
        assert phonemes_for("tasks") == ["T", "AE1", "S", "K", "S"]

    def test_uppercase(self):
        assert phonemes_for("CAT") == ["K", "AE1", "T"]

    def test_empty(self):
        assert phonemes_for("") == []

    def test_strip_stress(self):
        assert strip_stress("AE1") == "AE"
        assert strip_stress("K") == "K"


class TestArpaToIpa:
    def test_th(self):
        assert arpa_to_ipa(["TH", "IH1", "NG", "K"]) == "/θɪŋk/"

    def test_cluster(self):
        assert arpa_to_ipa(["R", "IH1", "S", "K", "S"]) == "/rɪsks/"

    def test_empty(self):
        assert arpa_to_ipa([]) == "//"


class TestDiffWord:
    def test_exact_match_no_weak(self):
        d = diff_word("cat", "cat")
        assert d.weak_phonemes == []
        assert d.confidence == 1.0

    def test_th_sub(self):
        d = diff_word("think", "sink")
        assert "TH" in d.weak_phonemes
        assert d.confidence < 1.0

    def test_deleted_word_full_weak(self):
        d = diff_word("be", None)
        # All phonemes weak when the word is absent (stripped of stress marks).
        assert len(d.weak_phonemes) == len(d.expected_arpa)
        assert [strip_stress(p) for p in d.weak_phonemes] == [
            strip_stress(p) for p in d.expected_arpa
        ]
        assert d.produced_arpa == []
        assert d.confidence == 0.0

    def test_oov_fallback(self):
        # Made-up word not in CMUdict triggers g2p_en fallback.
        d = diff_word("flibbertigibbet", "flibbertyget")
        assert d is not None
        assert d.expected_arpa  # non-empty


class TestKoreanL1Patterns:
    def _run(self, ref: str, hyp: str):
        aligned = align_texts(ref, hyp)
        diffs = []
        for a in aligned:
            if a.op in ("sub", "del") and a.ref:
                d = diff_word(a.ref, a.hyp)
                if d:
                    diffs.append(d)
        return detect_patterns(aligned, diffs, None)

    def test_final_cluster_deletion(self):
        patterns = self._run("I have two tasks", "I have two task")
        keys = {p.pattern for p in patterns}
        assert "final_cluster_deletion" in keys

    def test_final_stop_deletion(self):
        patterns = self._run("the cat sat", "the ca sat")
        keys = {p.pattern for p in patterns}
        assert "final_stop_unrelease" in keys

    def test_intrusive_onset_vowel(self):
        patterns = self._run("I saw a street", "I saw a suhtreet")
        keys = {p.pattern for p in patterns}
        assert "intrusive_onset_vowel" in keys

    def test_th_to_s(self):
        patterns = self._run("I think so", "I sink so")
        keys = {p.pattern for p in patterns}
        assert "th_to_s" in keys

    def test_th_to_t(self):
        patterns = self._run("the three men", "the tree men")
        keys = {p.pattern for p in patterns}
        assert "th_to_t" in keys

    def test_r_l_swap(self):
        patterns = self._run("I like rice", "I like lice")
        keys = {p.pattern for p in patterns}
        assert "r_l_swap" in keys

    def test_article_omission(self):
        patterns = self._run("I see a risk", "I see risk")
        keys = {p.pattern for p in patterns}
        assert "article_omission" in keys

    def test_pattern_has_tip_and_drill(self):
        patterns = self._run("I think so", "I sink so")
        assert patterns
        p = patterns[0]
        assert p.tip_ko  # non-empty Korean tip
        assert p.drill   # non-empty drill list

    def test_no_patterns_when_matching(self):
        patterns = self._run("the cat sat", "the cat sat")
        assert patterns == []


class TestSuggestDrills:
    def test_patterns_drive_drills(self):
        aligned = align_texts("I think so", "I sink so")
        diffs = [diff_word(a.ref, a.hyp) for a in aligned
                 if a.op in ("sub", "del") and a.ref]
        diffs = [d for d in diffs if d]
        patterns = detect_patterns(aligned, diffs, None)
        drills = suggest_drills(patterns, diffs)
        assert drills
        assert any("think/sink" in " ".join(d.minimal_pairs) for d in drills)

    def test_fully_absent_word_skipped_from_drills(self):
        """Words the user completely skipped get no drill suggestion —
        they get reported in the alignment table as deletions instead."""
        diffs = [diff_word("be", None)]
        drills = suggest_drills([], diffs)
        # A fully-absent production has produced_arpa == [], so the fallback
        # drill path should skip it.
        assert not any("weak" in d.reason for d in drills)
