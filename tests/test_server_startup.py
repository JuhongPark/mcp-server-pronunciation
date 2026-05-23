"""Tests for server startup controls."""

import anyio
import importlib
import sys

from mcp_server_pronunciation.assessor import AssessmentResult, WordResult


def _load_server_without_preload(monkeypatch):
    monkeypatch.setenv("MCP_PRONUNCIATION_PRELOAD", "0")
    sys.modules.pop("mcp_server_pronunciation.server", None)
    return importlib.import_module("mcp_server_pronunciation.server")


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
    assert tools_by_name["converse"].annotations.readOnlyHint is False
    assert tools_by_name["practice"].annotations.destructiveHint is False

    practice_output = tools_by_name["practice"].outputSchema["properties"]
    assert "report_markdown" in practice_output
    assert "top_issue" in practice_output
    assert "next_action" in practice_output
    assert "retry_comparison" in practice_output


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
