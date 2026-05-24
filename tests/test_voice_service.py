"""Tests for shared voice session state."""

from pathlib import Path

from mcp_server_pronunciation.assessor import AssessmentResult, WordResult
from mcp_server_pronunciation.service import VoiceSessionService


class _FakeAssessor:
    def assess(self, audio_path: Path, reference_text: str | None = None) -> AssessmentResult:
        return AssessmentResult(
            transcript="testing one two",
            reference_text=reference_text,
            words=[
                WordResult("testing", 0.0, 0.4, 0.9),
                WordResult("one", 0.5, 0.8, 0.8),
                WordResult("two", 0.9, 1.2, 0.8),
            ],
            duration_sec=1.5,
            speech_duration_sec=1.0,
        )


def test_voice_session_runs_recording_and_analysis(tmp_path):
    calls: list[tuple[float, Path]] = []

    def recorder(duration: float, output_path: Path) -> None:
        calls.append((duration, output_path))
        output_path.write_bytes(b"fake wav")

    service = VoiceSessionService(
        assessor_factory=lambda: _FakeAssessor(),
        recorder=recorder,
        retention=lambda: "keep",
    )

    session = service.create_session("conversation", 4.0)
    completed = service.run_session(session.id)

    assert calls == [(4.0, completed.audio_path)]
    assert completed.status == "done"
    assert completed.transcript == "testing one two"
    assert completed.clarity_pct == 83
    assert completed.speaking_rate_wpm == 180
    assert "## User said" in completed.report_markdown
    assert service.latest_session() is completed
    assert completed.to_dict()["audio_path"] == str(completed.audio_path)


def test_voice_session_records_errors(tmp_path):
    def recorder(_duration: float, _output_path: Path) -> None:
        raise RuntimeError("mic unavailable")

    service = VoiceSessionService(
        assessor_factory=lambda: _FakeAssessor(),
        recorder=recorder,
        retention=lambda: "keep",
    )

    session = service.create_session("practice", 0.1, reference_text="testing one two")
    failed = service.run_session(session.id, raise_errors=False)

    assert failed.status == "error"
    assert failed.error == "mic unavailable"
    assert failed.duration == 1.0
