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


def test_voice_session_prunes_old_sessions(tmp_path):
    service = VoiceSessionService(
        assessor_factory=lambda: _FakeAssessor(),
        recorder=lambda _duration, output_path: output_path.write_bytes(b"fake wav"),
        retention=lambda: "keep",
        max_sessions=2,
    )

    first = service.create_session("conversation", 1.0)
    service.run_session(first.id)
    second = service.create_session("conversation", 1.0)
    service.run_session(second.id)
    third = service.create_session("conversation", 1.0)
    service.run_session(third.id)

    assert service.session_count() == 2
    assert service.get_session(first.id) is None
    assert service.get_session(second.id) is not None
    assert service.latest_session() is third


def test_voice_session_pruning_preserves_pending_sessions():
    service = VoiceSessionService(assessor_factory=_FakeAssessor, max_sessions=2)
    sessions = [service.create_session("conversation", 1.0) for _ in range(3)]

    assert service.session_count() == 3
    assert all(service.get_session(session.id) is session for session in sessions)

    service.cancel_session(sessions[0].id)
    service.fail_session(sessions[1].id, "mic unavailable")

    assert service.session_count() == 2
    assert service.get_session(sessions[0].id) is None
    assert service.get_session(sessions[2].id) is sessions[2]
    assert service.latest_session() is sessions[1]


def test_voice_session_pruning_preserves_running_session():
    def recorder(_duration, output_path):
        service.create_session("conversation", 1.0)
        output_path.write_bytes(b"fake wav")

    service = VoiceSessionService(
        assessor_factory=_FakeAssessor,
        recorder=recorder,
        max_sessions=1,
    )
    session = service.create_session("conversation", 1.0)

    completed = service.run_session(session.id)

    assert completed.status == "done"
    assert service.get_session(session.id) is completed
    service.cleanup_recordings()
