"""Tests for the AssessmentResult data shape and rendering (no Whisper needed)."""

from mcp_server_pronunciation.alignment import align_words
from mcp_server_pronunciation.assessor import (
    AssessmentResult,
    WordResult,
)


def _make_result(
    transcript: str,
    reference: str | None = None,
    words: list[WordResult] | None = None,
) -> AssessmentResult:
    if words is None:
        tokens = transcript.split()
        words = [
            WordResult(word=w, start=i * 0.5, end=i * 0.5 + 0.4, probability=0.95)
            for i, w in enumerate(tokens)
        ]
    # Speech duration = sum of word spans (matches the real assessor).
    speech_dur = sum(max(0.0, w.end - w.start) for w in words)
    return AssessmentResult(
        transcript=transcript,
        reference_text=reference,
        words=words,
        duration_sec=speech_dur + 1.0,
        speech_duration_sec=speech_dur,
    )


# --- scoring -----------------------------------------------------


class TestScoring:
    def test_avg_confidence(self):
        words = [
            WordResult("a", 0, 0.5, 0.9),
            WordResult("b", 0.5, 1.0, 0.7),
        ]
        r = _make_result("a b", words=words)
        assert abs(r.avg_confidence - 0.8) < 0.01

    def test_words_per_minute(self):
        # 3 words, each 0.3s of speech = 0.9s total speech
        words = [
            WordResult("one", 0, 0.3, 0.9),
            WordResult("two", 0.3, 0.6, 0.9),
            WordResult("three", 0.6, 0.9, 0.9),
        ]
        r = _make_result("one two three", words=words)
        # 3 / 0.9 * 60 = 200 WPM
        assert abs(r.words_per_minute - 200) < 1

    def test_wpm_caveat_short_clip(self):
        r = _make_result("hello world")
        assert r.wpm_caveat is not None
        assert "of speech" in r.wpm_caveat

    def test_wpm_caveat_long_clip(self):
        # 30 words of 0.4s each = 12s speech -> no caveat.
        words = [WordResult(f"w{i}", i * 0.4, i * 0.4 + 0.4, 0.9) for i in range(30)]
        r = _make_result(" ".join(f"w{i}" for i in range(30)), words=words)
        assert r.wpm_caveat is None

    def test_clarity_with_reference_penalizes_mismatch(self):
        r = _make_result("cat", reference="bat")
        r.aligned = align_words(["bat"], ["cat"])
        # 0 matches out of 1 ref word; clarity should be lower than whisper-only.
        assert r.clarity_pct < 95


# --- grammar (ported) --------------------------------------------


class TestGrammarNotes:
    def test_detects_buyed(self):
        r = _make_result("I buyed some apples yesterday")
        notes = r.grammar_notes()
        assert len(notes) == 1
        assert notes[0][0] == "buyed"
        assert notes[0][1] == "bought"

    def test_no_errors(self):
        r = _make_result("I bought apples")
        assert r.grammar_notes() == []

    def test_deduplicates(self):
        r = _make_result("I buyed it and then I buyed another")
        assert len(r.grammar_notes()) == 1


# --- pauses -------------------------------------------------------


class TestPauses:
    def test_detects_long_pause(self):
        words = [
            WordResult("hello", 0, 0.5, 0.9),
            WordResult("world", 2.0, 2.5, 0.9),
        ]
        r = _make_result("hello world", words=words)
        pauses = r.get_pauses(0.8)
        assert len(pauses) == 1


# --- to_dict ------------------------------------------------------


class TestToDict:
    def test_basic_shape(self):
        r = _make_result("hello world", reference="hello world")
        r.aligned = align_words(["hello", "world"], ["hello", "world"])
        d = r.to_dict()
        assert "clarity_pct" in d
        assert "speaking_rate_wpm" in d
        assert "alignment" in d
        assert "phoneme_issues" in d
        assert "korean_l1_patterns" in d
        assert "prosody" in d
        assert "drills" in d

    def test_alignment_entries(self):
        r = _make_result("the cat", reference="the bat")
        r.aligned = align_words(["the", "bat"], ["the", "cat"])
        d = r.to_dict()
        ops = {a["op"] for a in d["alignment"]}
        assert "sub" in ops or "match" in ops


# --- format_report ------------------------------------------------


class TestFormatReport:
    def test_clean_when_no_issues(self):
        r = _make_result("hello world", reference="hello world")
        r.aligned = align_words(["hello", "world"], ["hello", "world"])
        report = r.format_report()
        assert "Great job" in report

    def test_alignment_table_on_mismatch(self):
        r = _make_result("the cat", reference="the bat")
        r.aligned = align_words(["the", "bat"], ["the", "cat"])
        report = r.format_report()
        assert "Alignment" in report

    def test_shows_wpm_caveat(self):
        r = _make_result("hi")
        report = r.format_report()
        assert "computed over" in report


# --- converse report ---------------------------------------------


class TestFormatConverseReport:
    def test_has_for_assistant_section(self):
        r = _make_result("hello")
        assert "## For Assistant" in r.format_converse_report()

    def test_surfaces_grammar_error(self):
        r = _make_result("I buyed apples")
        rep = r.format_converse_report()
        assert "bought" in rep

    def test_silent_prompt(self):
        r = _make_result("")
        r.words = []
        rep = r.format_converse_report()
        assert "silent" in rep.lower() or "repeat" in rep.lower()
