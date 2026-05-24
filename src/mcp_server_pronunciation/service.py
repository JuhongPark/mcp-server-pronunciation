"""Shared voice session state for MCP tools and interactive UIs."""

from __future__ import annotations

import atexit
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, TYPE_CHECKING

from .config import audio_retention_value

if TYPE_CHECKING:
    from .assessor import AssessmentResult, PronunciationAssessor

VoiceStatus = Literal["idle", "recording", "analyzing", "done", "error", "cancelled"]
VoiceMode = Literal["conversation", "practice", "assessment", "retry"]

AssessorFactory = Callable[[], "PronunciationAssessor"]
Recorder = Callable[[float, Path], Path | None]


@dataclass
class VoiceSession:
    """One recording and assessment lifecycle."""

    mode: VoiceMode
    duration: float
    reference_text: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: VoiceStatus = "idle"
    status_message: str = "Waiting to start."
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    recording_started_at: float | None = None
    analyzing_started_at: float | None = None
    finished_at: float | None = None
    updated_at: float = field(default_factory=time.time)
    audio_path: Path | None = None
    report_markdown: str = ""
    transcript: str = ""
    clarity_pct: int = 0
    speaking_rate_wpm: int = 0
    assessment: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    result: AssessmentResult | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "status": self.status,
            "status_message": self.status_message,
            "duration": self.duration,
            "reference_text": self.reference_text,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "recording_started_at": self.recording_started_at,
            "analyzing_started_at": self.analyzing_started_at,
            "finished_at": self.finished_at,
            "updated_at": self.updated_at,
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "transcript": self.transcript,
            "clarity_pct": self.clarity_pct,
            "speaking_rate_wpm": self.speaking_rate_wpm,
            "report_markdown": self.report_markdown,
            "assessment": self.assessment,
            "error": self.error,
        }


class VoiceSessionService:
    """Thread-safe session registry and runner."""

    def __init__(
        self,
        assessor_factory: AssessorFactory,
        recorder: Recorder | None = None,
        retention: Callable[[], str] = audio_retention_value,
    ) -> None:
        self._assessor_factory = assessor_factory
        self._recorder = recorder
        self._retention = retention
        self._lock = threading.RLock()
        self._sessions: dict[str, VoiceSession] = {}
        self._latest_id: str | None = None
        self._recordings_to_cleanup: set[Path] = set()
        atexit.register(self.cleanup_recordings)

    def create_session(
        self,
        mode: VoiceMode,
        duration: float,
        reference_text: str | None = None,
    ) -> VoiceSession:
        duration = min(max(duration, 1.0), 120.0)
        session = VoiceSession(mode=mode, duration=duration, reference_text=reference_text)
        with self._lock:
            self._sessions[session.id] = session
            self._latest_id = session.id
        return session

    def get_session(self, session_id: str) -> VoiceSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def latest_session(self) -> VoiceSession | None:
        with self._lock:
            if self._latest_id is None:
                return None
            return self._sessions.get(self._latest_id)

    def run_session(self, session_id: str, *, raise_errors: bool = True) -> VoiceSession:
        session = self.require_session(session_id)
        try:
            self._mark(session, "recording", "Recording microphone audio.")
            output_path = self.new_recording_path()
            self._update(session, audio_path=output_path)
            self._record(session.duration, output_path)

            self._mark(session, "analyzing", "Transcribing and analyzing pronunciation.")
            result = self._assessor_factory().assess(
                output_path, reference_text=session.reference_text
            )
            report = self._format_report(result, session.mode, session.reference_text)
            self.complete_session(session.id, result=result, report_markdown=report)
        except Exception as exc:
            self.fail_session(session.id, str(exc))
            if raise_errors:
                raise
        return self.require_session(session.id)

    def complete_session(
        self,
        session_id: str,
        *,
        result: AssessmentResult,
        report_markdown: str,
    ) -> VoiceSession:
        return self._update(
            self.require_session(session_id),
            status="done",
            status_message="Analysis complete.",
            finished_at=time.time(),
            report_markdown=report_markdown,
            transcript=result.transcript,
            clarity_pct=result.clarity_pct,
            speaking_rate_wpm=int(round(result.words_per_minute)),
            assessment=result.to_dict(),
            result=result,
            error=None,
        )

    def fail_session(self, session_id: str, message: str) -> VoiceSession:
        return self._update(
            self.require_session(session_id),
            status="error",
            status_message="Voice capture failed.",
            finished_at=time.time(),
            error=message,
        )

    def cancel_session(self, session_id: str) -> VoiceSession:
        return self._update(
            self.require_session(session_id),
            status="cancelled",
            status_message="Voice capture cancelled.",
            finished_at=time.time(),
        )

    def require_session(self, session_id: str) -> VoiceSession:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"Unknown voice session: {session_id}")
        return session

    def new_recording_path(self) -> Path:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="pronun_")
        path = Path(tmp.name)
        tmp.close()
        if self._retention() == "session":
            with self._lock:
                self._recordings_to_cleanup.add(path)
        return path

    def cleanup_recordings(self) -> None:
        if self._retention() == "keep":
            return
        with self._lock:
            paths = list(self._recordings_to_cleanup)
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            finally:
                with self._lock:
                    self._recordings_to_cleanup.discard(path)

    def _record(self, duration: float, output_path: Path) -> None:
        if self._recorder is None:
            from .recorder import record_audio

            record_audio(duration, output_path)
            return
        self._recorder(duration, output_path)

    def _mark(self, session: VoiceSession, status: VoiceStatus, message: str) -> VoiceSession:
        now = time.time()
        fields: dict[str, Any] = {
            "status": status,
            "status_message": message,
            "started_at": session.started_at or now,
        }
        if status == "recording":
            fields["recording_started_at"] = now
        if status == "analyzing":
            fields["analyzing_started_at"] = now
        return self._update(session, **fields)

    def _update(self, session: VoiceSession, **fields: Any) -> VoiceSession:
        with self._lock:
            for key, value in fields.items():
                setattr(session, key, value)
            session.updated_at = time.time()
            self._latest_id = session.id
            return session

    @staticmethod
    def _format_report(
        result: AssessmentResult,
        mode: VoiceMode,
        reference_text: str | None,
    ) -> str:
        if mode == "conversation":
            return result.format_converse_report(has_target=reference_text is not None)
        return result.format_report()
