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


def test_check_audio_devices_reports_configured_vad_and_device(monkeypatch):
    class FakeSoundDevice:
        @staticmethod
        def query_devices(kind=None):
            if kind == "input":
                return {"name": "Built-in Mic"}
            return [
                {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 48000},
                {"name": "USB Mic", "max_input_channels": 1, "default_samplerate": 44100},
            ]

    monkeypatch.setattr(recorder, "_is_wsl", lambda: False)
    monkeypatch.setattr(recorder, "_import_sounddevice", lambda: FakeSoundDevice)
    monkeypatch.setenv("MCP_PRONUNCIATION_INPUT_DEVICE", "1")
    monkeypatch.setenv("MCP_PRONUNCIATION_VAD_SENSITIVITY", "high")
    monkeypatch.setenv("MCP_PRONUNCIATION_SILENCE_DURATION", "2.0")

    output = recorder.check_audio_devices()

    assert "Configured input device: 1" in output
    assert "sensitivity=high" in output
    assert "stop_after=2.0s" in output
    assert "[1] USB Mic" in output
