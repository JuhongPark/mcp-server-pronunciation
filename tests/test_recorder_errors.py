"""Tests for actionable recorder errors."""

import pytest

from mcp_server_pronunciation import cli, recorder


def test_sounddevice_recording_failure_mentions_doctor_and_device_hint(tmp_path, monkeypatch):
    class CallbackAbort(Exception):
        pass

    class FakeSoundDevice:
        class InputStream:
            def __init__(self, *args, **kwargs):
                raise OSError("No default input device")

    FakeSoundDevice.CallbackAbort = CallbackAbort
    monkeypatch.setattr(recorder, "_import_sounddevice", lambda: FakeSoundDevice)

    with pytest.raises(RuntimeError) as excinfo:
        recorder._record_sounddevice_vad(0.01, tmp_path / "recording.wav")

    message = str(excinfo.value)
    assert "Recording failed via sounddevice" in message
    assert "mcp-server-pronunciation doctor" in message
    assert "check_mic" in message


def test_empty_sounddevice_capture_has_actionable_message(tmp_path, monkeypatch):
    class CallbackAbort(Exception):
        pass

    class FakeSoundDevice:
        class InputStream:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

    FakeSoundDevice.CallbackAbort = CallbackAbort
    monkeypatch.setattr(recorder, "_import_sounddevice", lambda: FakeSoundDevice)

    with pytest.raises(RuntimeError) as excinfo:
        recorder._record_sounddevice_vad(0.0, tmp_path / "recording.wav")

    message = str(excinfo.value)
    assert "No audio data captured" in message
    assert "Speak after the tool starts recording" in message
    assert "mcp-server-pronunciation doctor" in message


def test_pull_model_rejects_unsupported_model_before_network(capsys):
    assert cli.pull_model("basee.en") == 1

    captured = capsys.readouterr()
    assert "unsupported Whisper model" in captured.err
    assert "base.en" in captured.err
