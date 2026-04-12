"""Tests for pronunciation assessment logic (no Whisper model needed)."""

from mcp_server_pronunciation.assessor import (
    AssessmentResult,
    WordResult,
    _normalize_words,
)


def _make_result(
    transcript: str,
    reference: str | None = None,
    words: list[WordResult] | None = None,
) -> AssessmentResult:
    """Helper to build an AssessmentResult with sensible defaults."""
    if words is None:
        # Build words from transcript with high confidence
        tokens = transcript.split()
        words = [
            WordResult(word=w, start=i * 0.5, end=i * 0.5 + 0.4, probability=0.95)
            for i, w in enumerate(tokens)
        ]
    speech_dur = words[-1].end - words[0].start if words else 0.0
    return AssessmentResult(
        transcript=transcript,
        reference_text=reference,
        words=words,
        duration_sec=speech_dur + 1.0,
        speech_duration_sec=speech_dur,
    )


# --- _normalize_words ---


class TestNormalizeWords:
    def test_basic(self):
        assert _normalize_words("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert _normalize_words("Hello, world!") == ["hello", "world"]

    def test_empty(self):
        assert _normalize_words("") == []

    def test_mixed_case(self):
        assert _normalize_words("The THREE Brothers") == ["the", "three", "brothers"]


# --- Word alignment (SequenceMatcher) ---


class TestFindMismatches:
    def test_exact_match_no_mismatches(self):
        r = _make_result("the three brothers", "The three brothers.")
        assert r._find_mismatches() == []

    def test_substitution(self):
        r = _make_result("de tree brothers", "The three brothers.")
        mismatches = r._find_mismatches()
        refs = [m[0] for m in mismatches]
        assert "the" in refs
        assert "three" in refs

    def test_substitution_with_hint(self):
        r = _make_result("I sink about it", "I think about it")
        mismatches = r._find_mismatches()
        # "think" -> "sink" should have a hint
        think_mismatch = [m for m in mismatches if m[0] == "think"]
        assert len(think_mismatch) == 1
        assert think_mismatch[0][1] == "sink"
        assert think_mismatch[0][2] != ""  # has a hint

    def test_skipped_word(self):
        r = _make_result("I about it", "I think about it")
        mismatches = r._find_mismatches()
        skipped = [m for m in mismatches if m[1] == "(skipped)"]
        assert len(skipped) >= 1

    def test_extra_word(self):
        r = _make_result("I really think about it", "I think about it")
        mismatches = r._find_mismatches()
        # "really" is extra — should not cause other words to misalign
        non_extra = [m for m in mismatches if m[1] != "(extra)"]
        # The actual words should still match
        assert len(non_extra) == 0 or all(m[1] == "(extra)" for m in mismatches)

    def test_no_reference(self):
        r = _make_result("hello world", None)
        assert r._find_mismatches() == []

    def test_empty_transcript(self):
        r = _make_result("", "hello world")
        r.words = []
        mismatches = r._find_mismatches()
        assert all(m[1] == "(skipped)" for m in mismatches)


# --- Scoring ---


class TestScoring:
    def test_avg_confidence(self):
        words = [
            WordResult("a", 0, 0.5, 0.9),
            WordResult("b", 0.5, 1.0, 0.7),
        ]
        r = _make_result("a b", words=words)
        assert abs(r.avg_confidence - 0.8) < 0.01

    def test_avg_confidence_empty(self):
        r = AssessmentResult(transcript="", reference_text=None, words=[])
        assert r.avg_confidence == 0.0

    def test_low_confidence_words(self):
        words = [
            WordResult("good", 0, 0.5, 0.95),
            WordResult("bad", 0.5, 1.0, 0.3),
            WordResult("ok", 1.0, 1.5, 0.65),
        ]
        r = _make_result("good bad ok", words=words)
        low = r.low_confidence_words
        assert len(low) == 2
        assert low[0].word == "bad"
        assert low[1].word == "ok"

    def test_words_per_minute(self):
        words = [
            WordResult("one", 0, 0.3, 0.9),
            WordResult("two", 0.3, 0.6, 0.9),
            WordResult("three", 0.6, 1.0, 0.9),
        ]
        r = _make_result("one two three", words=words)
        # 3 words in 1 second = 180 WPM
        assert abs(r.words_per_minute - 180) < 1

    def test_words_per_minute_zero_duration(self):
        r = AssessmentResult(transcript="", reference_text=None, words=[])
        assert r.words_per_minute == 0.0


# --- Pauses ---


class TestPauses:
    def test_detects_long_pause(self):
        words = [
            WordResult("hello", 0, 0.5, 0.9),
            WordResult("world", 2.0, 2.5, 0.9),
        ]
        r = _make_result("hello world", words=words)
        pauses = r.get_pauses(0.8)
        assert len(pauses) == 1
        assert abs(pauses[0][2] - 1.5) < 0.01

    def test_no_pause(self):
        words = [
            WordResult("hello", 0, 0.4, 0.9),
            WordResult("world", 0.5, 0.9, 0.9),
        ]
        r = _make_result("hello world", words=words)
        assert r.get_pauses(0.8) == []


# --- Report formatting ---


class TestFormatReport:
    def test_excellent_report(self):
        r = _make_result("hello world", "Hello world.")
        report = r.format_report()
        assert "Great job" in report

    def test_report_with_mismatch(self):
        r = _make_result("de tree brothers", "The three brothers.")
        report = r.format_report()
        assert "What to fix" in report

    def test_report_shows_transcript(self):
        r = _make_result("hello world")
        report = r.format_report()
        assert "hello world" in report

    def test_report_shows_clarity(self):
        r = _make_result("hello world")
        report = r.format_report()
        assert "Clarity" in report


# --- Korean tips ---


class TestKoreanTips:
    def test_th_words_with_low_confidence(self):
        words = [
            WordResult("the", 0, 0.3, 0.4),
            WordResult("cat", 0.3, 0.6, 0.95),
        ]
        r = _make_result("the cat", words=words)
        tips = r._get_korean_tips()
        assert any("/θ/" in t for t in tips)

    def test_no_tips_when_confident(self):
        words = [
            WordResult("the", 0, 0.3, 0.95),
            WordResult("three", 0.3, 0.6, 0.95),
        ]
        r = _make_result("the three", words=words)
        tips = r._get_korean_tips()
        assert len(tips) == 0

    def test_f_words_with_low_confidence(self):
        words = [
            WordResult("five", 0, 0.3, 0.4),
            WordResult("friends", 0.3, 0.6, 0.95),
        ]
        r = _make_result("five friends", words=words)
        tips = r._get_korean_tips()
        assert any("/f/" in t for t in tips)


# --- Grammar notes (irregular past tense) ---


class TestGrammarNotes:
    def test_detects_buyed(self):
        r = _make_result("I buyed some apples yesterday")
        notes = r.grammar_notes()
        assert len(notes) == 1
        wrong, correct, _ = notes[0]
        assert wrong == "buyed"
        assert correct == "bought"

    def test_multiple_errors(self):
        r = _make_result("I goed home and eated dinner")
        notes = r.grammar_notes()
        wrongs = {n[0] for n in notes}
        assert wrongs == {"goed", "eated"}

    def test_case_insensitive(self):
        r = _make_result("Yesterday I BUYED apples")
        notes = r.grammar_notes()
        assert len(notes) == 1
        assert notes[0][0] == "buyed"

    def test_no_errors(self):
        r = _make_result("I bought apples yesterday")
        assert r.grammar_notes() == []

    def test_empty_transcript(self):
        r = _make_result("")
        r.words = []
        assert r.grammar_notes() == []

    def test_deduplicates(self):
        r = _make_result("I buyed the apple and then I buyed another one")
        notes = r.grammar_notes()
        assert len(notes) == 1


# --- Converse report format ---


class TestFormatConverseReport:
    def test_has_user_said_section(self):
        r = _make_result("hello how are you")
        report = r.format_converse_report()
        assert "## User said" in report
        assert "hello how are you" in report

    def test_has_for_claude_section(self):
        r = _make_result("hello")
        report = r.format_converse_report()
        assert "## For Claude" in report

    def test_surfaces_grammar_error(self):
        r = _make_result("I buyed apples")
        report = r.format_converse_report()
        assert "Grammar" in report
        assert "bought" in report

    def test_empty_transcript_tells_claude_to_ask_again(self):
        r = _make_result("")
        r.words = []
        report = r.format_converse_report()
        assert "silent" in report.lower() or "repeat" in report.lower()

    def test_no_errors_clean_report(self):
        r = _make_result("I am having a great day today")
        report = r.format_converse_report()
        assert "Quick feedback" in report

    def test_target_mode_shows_mismatch(self):
        r = _make_result("de tree brothers", reference="The three brothers")
        report = r.format_converse_report(has_target=True)
        assert "Pronunciation" in report
