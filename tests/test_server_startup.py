"""Tests for server startup controls."""

import anyio
import importlib
import sys


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
    assert "auto-stops earlier on silence" in converse_schema["duration"]["description"]

    quick_practice_schema = tools_by_name["quick_practice"].inputSchema["properties"]
    focus_schema = quick_practice_schema["focus"]["anyOf"][0]
    difficulty_schema = quick_practice_schema["difficulty"]["anyOf"][0]

    assert focus_schema["enum"] == ["th", "f_v", "r_l", "vowels", "general"]
    assert difficulty_schema["enum"] == ["beginner", "intermediate", "advanced"]
    assert "pronunciation focus filter" in quick_practice_schema["focus"]["description"]

    assess_schema = tools_by_name["assess"].inputSchema["properties"]
    assert "local path to a WAV file" in assess_schema["audio_path"]["description"]


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

    content, metadata = anyio.run(server.mcp.call_tool, "retry", {"duration": 1.0})
    assert content[0].text == metadata["result"]
    assert "No previous practice session" in content[0].text

    content, metadata = anyio.run(
        server.mcp.call_tool,
        "assess",
        {"reference_text": None, "audio_path": None},
    )
    assert content[0].text == metadata["result"]
    assert "No recording found" in content[0].text
