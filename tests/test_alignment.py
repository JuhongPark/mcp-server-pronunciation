"""Tests for Needleman-Wunsch word alignment."""

from mcp_server_pronunciation.alignment import (
    align_texts,
    align_words,
    summarize,
    tokenize,
)


class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello world") == ["hello", "world"]

    def test_strips_punct(self):
        assert tokenize("The cat sat on the mat!") == [
            "the",
            "cat",
            "sat",
            "on",
            "the",
            "mat",
        ]

    def test_hyphen_splits(self):
        # Whisper tokenizes hyphenated compounds as separate words; do the same.
        assert tokenize("front-end") == ["front", "end"]

    def test_apostrophes_kept(self):
        assert tokenize("I'm happy") == ["i'm", "happy"]

    def test_empty(self):
        assert tokenize("") == []


class TestAlignWords:
    def test_exact_match(self):
        a = align_words(["the", "cat"], ["the", "cat"])
        assert summarize(a) == {"match": 2, "sub": 0, "ins": 0, "del": 0}

    def test_substitution(self):
        a = align_words(["the", "cat"], ["the", "bat"])
        assert summarize(a) == {"match": 1, "sub": 1, "ins": 0, "del": 0}
        sub_entry = [x for x in a if x.op == "sub"][0]
        assert sub_entry.ref == "cat" and sub_entry.hyp == "bat"

    def test_single_deletion_no_cascade(self):
        """Primary regression: a single dropped word in the middle of a
        sentence must report exactly one `del`, not a chain of phantom
        substitutions (the old SequenceMatcher-based behavior)."""
        ref = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        hyp = ["the", "quick", "brown", "fox", "over", "the", "lazy", "dog"]
        a = align_words(ref, hyp)
        s = summarize(a)
        assert s["del"] == 1
        assert s["sub"] == 0
        assert s["ins"] == 0
        del_entry = [x for x in a if x.op == "del"][0]
        assert del_entry.ref == "jumps"

    def test_single_insertion_no_cascade(self):
        ref = ["i", "think", "about", "it"]
        hyp = ["i", "really", "think", "about", "it"]
        a = align_words(ref, hyp)
        s = summarize(a)
        assert s["ins"] == 1
        assert s["sub"] == 0
        assert s["del"] == 0
        ins_entry = [x for x in a if x.op == "ins"][0]
        assert ins_entry.hyp == "really"

    def test_mixed_edits(self):
        ref = ["the", "three", "brothers"]
        hyp = ["de", "tree", "brothers"]
        a = align_words(ref, hyp)
        s = summarize(a)
        assert s["match"] == 1
        assert s["sub"] == 2

    def test_empty_ref(self):
        a = align_words([], ["hello"])
        assert summarize(a) == {"match": 0, "sub": 0, "ins": 1, "del": 0}

    def test_empty_hyp(self):
        a = align_words(["hello"], [])
        assert summarize(a) == {"match": 0, "sub": 0, "ins": 0, "del": 1}

    def test_both_empty(self):
        assert align_words([], []) == []


class TestAlignTexts:
    def test_single_deletion_preserves_surrounding_matches(self):
        """Full text-to-text: dropping a function word mid-sentence should
        leave every other word correctly matched."""
        ref = "The cat is sitting on the mat."
        hyp = "The cat sitting on the mat."
        a = align_texts(ref, hyp)
        s = summarize(a)
        assert s["del"] == 1
        assert s["sub"] == 0
        del_entry = [x for x in a if x.op == "del"][0]
        assert del_entry.ref == "is"

    def test_ref_indices_preserved(self):
        """ref_index on each aligned entry should map back to the original token list."""
        ref = "I think about it"
        hyp = "I really think about it"
        a = align_texts(ref, hyp)
        # Ref tokens are 0:i, 1:think, 2:about, 3:it
        think_entry = [x for x in a if x.ref == "think"][0]
        assert think_entry.ref_index == 1
