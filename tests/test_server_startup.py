"""Tests for server startup controls."""

import base64
import importlib
import io
import sys
import wave
from pathlib import Path

import anyio
import pytest

from mcp_server_pronunciation.assessor import AssessmentResult, WordResult
from mcp_server_pronunciation.service import VoiceSessionService


def _load_server_without_preload(monkeypatch):
    monkeypatch.setenv("MCP_PRONUNCIATION_PRELOAD", "0")
    sys.modules.pop("mcp_server_pronunciation.server", None)
    return importlib.import_module("mcp_server_pronunciation.server")


def _wav_bytes(
    *,
    sample_rate: int = 16000,
    sample_width: int = 2,
    channels: int = 1,
    frames: int = 1600,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\0" * frames * sample_width * channels)
    return buffer.getvalue()


def test_model_preload_can_be_disabled(monkeypatch):
    server = _load_server_without_preload(monkeypatch)
    assert server._preload_enabled() is False
    assert server._assessor is None

    monkeypatch.setenv("MCP_PRONUNCIATION_PRELOAD", "off")
    assert server._preload_enabled() is False

    monkeypatch.setenv("MCP_PRONUNCIATION_PRELOAD", "1")
    assert server._preload_enabled() is True


def test_tool_schemas_include_agent_friendly_parameter_metadata(monkeypatch):
    server = _load_server_without_preload(monkeypatch)
    tools = anyio.run(server.mcp.list_tools)
    tools_by_name = {tool.name: tool for tool in tools}

    assert set(tools_by_name) == {
        "converse",
        "practice",
        "retry",
        "quick_practice",
        "open_voice_panel",
        "analyze_uploaded_audio",
        "start_voice_capture",
        "voice_capture_status",
        "wait_for_voice_capture",
        "latest_voice_capture",
        "cancel_voice_capture",
        "suggest_sentence",
        "record",
        "assess",
        "check_mic",
    }

    converse_schema = tools_by_name["converse"].inputSchema["properties"]
    assert "target sentence" in converse_schema["target_hint"]["description"]
    assert "WSL2 PowerShell recording" in converse_schema["duration"]["description"]

    quick_practice_schema = tools_by_name["quick_practice"].inputSchema["properties"]
    focus_schema = quick_practice_schema["focus"]["anyOf"][0]
    difficulty_schema = quick_practice_schema["difficulty"]["anyOf"][0]

    assert focus_schema["enum"] == ["th", "f_v", "r_l", "vowels", "general"]
    assert difficulty_schema["enum"] == ["beginner", "intermediate", "advanced"]
    assert "pronunciation focus filter" in quick_practice_schema["focus"]["description"]

    assess_schema = tools_by_name["assess"].inputSchema["properties"]
    assert "local path to a WAV file" in assess_schema["audio_path"]["description"]

    assert tools_by_name["check_mic"].annotations.readOnlyHint is True
    assert tools_by_name["suggest_sentence"].annotations.readOnlyHint is True
    assert tools_by_name["open_voice_panel"].annotations.readOnlyHint is True
    assert tools_by_name["open_voice_panel"].meta["openai/outputTemplate"].startswith("ui://")
    assert tools_by_name["converse"].annotations.readOnlyHint is False
    assert tools_by_name["practice"].annotations.destructiveHint is False

    practice_output = tools_by_name["practice"].outputSchema["properties"]
    assert "report_markdown" in practice_output
    assert "top_issue" in practice_output
    assert "next_action" in practice_output
    assert "retry_comparison" in practice_output

    upload_output = tools_by_name["analyze_uploaded_audio"].outputSchema["properties"]
    assert "status" in upload_output
    assert "session_id" in upload_output
    assert "report_markdown" in upload_output


def test_prompt_shortcuts_are_discoverable(monkeypatch):
    server = _load_server_without_preload(monkeypatch)
    prompts = anyio.run(server.mcp.list_prompts)
    prompts_by_name = {prompt.name: prompt for prompt in prompts}

    assert set(prompts_by_name) == {
        "start_voice_chat",
        "daily_practice",
        "practice_focus",
        "troubleshoot_mic",
    }
    assert prompts_by_name["start_voice_chat"].title == "Start voice chat"
    assert prompts_by_name["daily_practice"].arguments[0].name == "focus"

    rendered = anyio.run(
        server.mcp.get_prompt,
        "practice_focus",
        {"focus": "th", "difficulty": "beginner"},
    )
    prompt_text = rendered.messages[0].content.text
    assert "quick_practice" in prompt_text
    assert "focus=th" in prompt_text


def test_voice_panel_resource_is_discoverable(monkeypatch):
    server = _load_server_without_preload(monkeypatch)

    resources = anyio.run(server.mcp.list_resources)
    resources_by_name = {resource.name: resource for resource in resources}

    panel = resources_by_name["voice_panel"]
    assert str(panel.uri) == "ui://pronunciation/voice-panel"
    assert panel.mimeType == "text/html"
    assert panel.meta["io.modelcontextprotocol/ui"]["permissions"] == ["microphone"]

    contents = anyio.run(server.mcp.read_resource, "ui://pronunciation/voice-panel")
    assert contents[0].mime_type == "text/html"
    assert "<button" in contents[0].content
    assert "Record" in contents[0].content
    assert "analyze_uploaded_audio" in contents[0].content


def test_mcp_text_only_tools_can_be_called_without_audio_hardware(monkeypatch):
    server = _load_server_without_preload(monkeypatch)

    content, metadata = anyio.run(
        server.mcp.call_tool,
        "suggest_sentence",
        {"focus": "th", "difficulty": "beginner"},
    )
    suggestion = content[0].text

    assert content[0].type == "text"
    assert metadata["result"] == suggestion
    assert "**Practice this:**" in suggestion
    assert "**Focus:** th | **Difficulty:** beginner" in suggestion
    assert "`practice` tool" in suggestion

    result = anyio.run(server.mcp.call_tool, "retry", {"duration": 1.0})
    assert result.isError is True
    assert "No previous practice session" in result.content[0].text
    assert result.structuredContent["mode"] == "retry"
    assert result.structuredContent["next_action"]["tool"] == "practice"

    result = anyio.run(
        server.mcp.call_tool,
        "assess",
        {"reference_text": None, "audio_path": None},
    )
    assert result.isError is True
    assert result.structuredContent["mode"] == "assessment"
    assert "No recording found" in result.content[0].text


def test_background_voice_capture_tools_report_progress(monkeypatch):
    server = _load_server_without_preload(monkeypatch)

    class FakeAssessor:
        def assess(self, audio_path, reference_text=None):
            assert Path(audio_path).exists()
            return AssessmentResult(
                transcript="voice note",
                reference_text=reference_text,
                words=[WordResult("voice", 0.0, 0.4, 0.9)],
                duration_sec=0.5,
                speech_duration_sec=0.4,
            )

    def recorder(_duration, output_path):
        output_path.write_bytes(b"fake wav")

    server._voice_service = VoiceSessionService(
        assessor_factory=lambda: FakeAssessor(),
        recorder=recorder,
        retention=lambda: "keep",
    )

    started = server.start_voice_capture(duration=1.0)
    assert started.session_id
    assert started.status in {"idle", "recording", "analyzing", "done"}

    waited = server.wait_for_voice_capture(started.session_id, timeout=3.0)
    assert waited.status == "done"
    assert waited.transcript == "voice note"
    assert waited.clarity_pct == 90
    with server._voice_threads_lock:
        thread = server._voice_threads.get(started.session_id)
    if thread is not None:
        thread.join(timeout=3.0)
    with server._voice_threads_lock:
        assert started.session_id not in server._voice_threads

    latest = server.latest_voice_capture()
    assert latest.session_id == started.session_id


def test_uploaded_audio_analysis_updates_latest_voice_capture(monkeypatch):
    server = _load_server_without_preload(monkeypatch)

    class FakeAssessor:
        def assess(self, audio_path, reference_text=None):
            assert Path(audio_path).read_bytes().startswith(b"RIFF")
            return AssessmentResult(
                transcript="uploaded voice",
                reference_text=reference_text,
                words=[WordResult("uploaded", 0.0, 0.4, 0.8)],
                duration_sec=0.5,
                speech_duration_sec=0.4,
            )

    server._voice_service = VoiceSessionService(
        assessor_factory=lambda: FakeAssessor(),
        retention=lambda: "keep",
    )

    result = server.analyze_uploaded_audio(
        wav_base64=base64.b64encode(_wav_bytes()).decode("ascii"),
        mode="conversation",
    )

    assert result.isError is False
    assert result.structuredContent["status"] == "done"
    assert result.structuredContent["transcript"] == "uploaded voice"
    assert server.latest_voice_capture().session_id == result.structuredContent["session_id"]


def test_uploaded_audio_rejects_invalid_base64(monkeypatch):
    server = _load_server_without_preload(monkeypatch)

    result = server.analyze_uploaded_audio(wav_base64="not-base64", mode="conversation")

    assert result.isError is True
    assert result.structuredContent["status"] == "error"
    assert result.structuredContent["error"] == "Uploaded audio must be valid base64 WAV data."

    tool_result = anyio.run(
        server.mcp.call_tool,
        "analyze_uploaded_audio",
        {"wav_base64": "not-base64", "mode": "conversation"},
    )
    assert tool_result.isError is True
    assert tool_result.structuredContent["status"] == "error"


def test_uploaded_audio_rejects_non_wav_payload(monkeypatch):
    server = _load_server_without_preload(monkeypatch)

    result = server.analyze_uploaded_audio(
        wav_base64=base64.b64encode(b"plain bytes").decode("ascii"),
        mode="conversation",
    )

    assert result.isError is True
    assert result.structuredContent["status"] == "error"
    assert result.structuredContent["error"] == "Uploaded audio must be a RIFF/WAVE file."


def test_uploaded_audio_rejects_oversized_payload(monkeypatch):
    server = _load_server_without_preload(monkeypatch)
    monkeypatch.setattr(server, "MAX_UPLOADED_WAV_BYTES", 64)

    result = server.analyze_uploaded_audio(
        wav_base64=base64.b64encode(_wav_bytes()).decode("ascii"),
        mode="conversation",
    )

    assert result.isError is True
    assert result.structuredContent["status"] == "error"
    assert "too large" in result.structuredContent["error"]


@pytest.mark.parametrize(
    ("audio_bytes", "message"),
    [
        (_wav_bytes()[:-1], "complete PCM frames"),
        (_wav_bytes()[:20], "RIFF/WAVE file"),
        (_wav_bytes()[:36] + b"JUNK" + _wav_bytes()[40:], "readable WAV file"),
        (_wav_bytes(channels=3), "mono or stereo"),
        (_wav_bytes(sample_width=1), "16-bit PCM"),
        (_wav_bytes(sample_rate=7999), "between 8 kHz and 48 kHz"),
        (_wav_bytes(sample_rate=48001), "between 8 kHz and 48 kHz"),
        (_wav_bytes(frames=0), "no audio frames"),
    ],
    ids=[
        "truncated",
        "short-header",
        "missing-data",
        "channels",
        "bit-depth",
        "low-rate",
        "high-rate",
        "empty",
    ],
)
def test_uploaded_audio_rejects_invalid_wav_data(monkeypatch, audio_bytes, message):
    server = _load_server_without_preload(monkeypatch)

    with pytest.raises(ValueError, match=message):
        server._decode_uploaded_wav(base64.b64encode(audio_bytes).decode("ascii"))


@pytest.mark.parametrize("encoded", ["", "  ", "not-base64", "\N{SNOWMAN}"])
def test_uploaded_audio_rejects_invalid_encoding(monkeypatch, encoded):
    server = _load_server_without_preload(monkeypatch)

    with pytest.raises(ValueError, match="Uploaded audio"):
        server._decode_uploaded_wav(encoded)


@pytest.mark.parametrize(("channels", "sample_rate"), [(1, 8000), (2, 48000)])
def test_uploaded_audio_accepts_supported_wav_formats(monkeypatch, channels, sample_rate):
    server = _load_server_without_preload(monkeypatch)
    audio_bytes = _wav_bytes(channels=channels, sample_rate=sample_rate)

    assert server._decode_uploaded_wav(base64.b64encode(audio_bytes).decode("ascii")) == audio_bytes


def test_uploaded_audio_enforces_duration_limit(monkeypatch):
    server = _load_server_without_preload(monkeypatch)
    monkeypatch.setattr(server, "MAX_UPLOADED_WAV_SECONDS", 1.0)
    at_limit = _wav_bytes(frames=16000)
    too_long = _wav_bytes(frames=16001)

    assert server._decode_uploaded_wav(base64.b64encode(at_limit).decode("ascii")) == at_limit
    with pytest.raises(ValueError, match="too long"):
        server._decode_uploaded_wav(base64.b64encode(too_long).decode("ascii"))


def test_uploaded_audio_enforces_decoded_size_limit(monkeypatch):
    server = _load_server_without_preload(monkeypatch)
    monkeypatch.setattr(server, "MAX_UPLOADED_WAV_BYTES", 64)
    at_limit = _wav_bytes(frames=10)
    too_large = _wav_bytes(frames=11)

    assert server._decode_uploaded_wav(base64.b64encode(at_limit).decode("ascii")) == at_limit
    with pytest.raises(ValueError, match="too large"):
        server._decode_uploaded_wav(base64.b64encode(too_large).decode("ascii"))


def test_retry_comparison_summarizes_clarity_delta(monkeypatch):
    server = _load_server_without_preload(monkeypatch)

    previous = AssessmentResult(
        transcript="i buyed apples",
        reference_text="I bought apples.",
        words=[WordResult("i", 0, 0.2, 0.4), WordResult("buyed", 0.3, 0.7, 0.5)],
        speech_duration_sec=0.6,
    )
    current = AssessmentResult(
        transcript="i bought apples",
        reference_text="I bought apples.",
        words=[WordResult("i", 0, 0.2, 0.8), WordResult("bought", 0.3, 0.7, 0.8)],
        speech_duration_sec=0.6,
    )

    comparison = server._compare_attempts(previous, current)

    assert comparison.previous_clarity_pct == 45
    assert comparison.current_clarity_pct == 80
    assert comparison.clarity_delta == 35
    assert "improved" in comparison.summary
