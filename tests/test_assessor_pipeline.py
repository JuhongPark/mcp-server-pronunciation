"""Pipeline tests for assessor wiring without loading Whisper or audio models."""

from types import SimpleNamespace

import mcp_server_pronunciation.assessor as assessor_module
from mcp_server_pronunciation.assessor import PronunciationAssessor
from mcp_server_pronunciation.forced_align import ForcedAlignment, ForcedWord
from mcp_server_pronunciation.prosody import ProsodyResult


class _FakeModel:
    def __init__(self, text: str):
        self._words = []
        for index, word in enumerate(text.split()):
            self._words.append(
                SimpleNamespace(
                    word=word,
                    start=index * 0.4,
                    end=index * 0.4 + 0.3,
                    probability=0.95,
                )
            )
        self._text = text

    def transcribe(self, *_args, **_kwargs):
        segment = SimpleNamespace(text=self._text, words=self._words)
        info = SimpleNamespace(duration=1.5, language="en", language_probability=0.99)
        return [segment], info


def _assessor_with_fake_model(text: str) -> PronunciationAssessor:
    assessor = PronunciationAssessor(model_size="fake")
    assessor._model = _FakeModel(text)
    return assessor


def test_assess_reference_pipeline_detects_phoneme_pattern(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"not real audio")
    monkeypatch.setattr(assessor_module.forced_align, "align", lambda *_args: None)
    monkeypatch.setattr(assessor_module, "prosody_analyze", lambda *_args: ProsodyResult())

    result = _assessor_with_fake_model("I sink so").assess(audio, "I think so")

    assert result.transcript == "I sink so"
    assert any(item.op == "sub" and item.ref == "think" for item in result.aligned)
    assert any(diff.word == "think" for diff in result.phoneme_diffs)
    assert any(pattern.pattern == "th_to_s" for pattern in result.korean_l1_patterns)
    assert "θ" in result.format_report()


def test_forced_alignment_can_override_whisper_substitution(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"not real audio")
    forced = ForcedAlignment(
        model_name="fake-aligner",
        words=[
            ForcedWord("i", 0, 0.0, 0.2, 0.9),
            ForcedWord("think", 1, 0.2, 0.7, 0.95),
            ForcedWord("so", 2, 0.7, 1.0, 0.9),
        ],
    )
    monkeypatch.setattr(assessor_module.forced_align, "align", lambda *_args: forced)
    monkeypatch.setattr(assessor_module, "prosody_analyze", lambda *_args: ProsodyResult())

    result = _assessor_with_fake_model("I sink so").assess(audio, "I think so")
    think_entry = next(item for item in result.aligned if item.ref == "think")

    assert result.forced_alignment_used is True
    assert think_entry.op == "match"
    assert think_entry.forced_confidence == 0.95
    assert "Whisper misheard" in think_entry.note
    assert result.phoneme_diffs == []
