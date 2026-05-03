"""Tests for server startup controls."""

import anyio
import importlib
import sys


def test_model_preload_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MCP_PRONUNCIATION_PRELOAD", "0")
    sys.modules.pop("mcp_server_pronunciation.server", None)

    server = importlib.import_module("mcp_server_pronunciation.server")

    assert server._preload_enabled() is False
    assert server._assessor is None

    monkeypatch.setenv("MCP_PRONUNCIATION_PRELOAD", "off")
    assert server._preload_enabled() is False

    monkeypatch.setenv("MCP_PRONUNCIATION_PRELOAD", "1")
    assert server._preload_enabled() is True


def test_tool_schemas_include_agent_friendly_parameter_metadata(monkeypatch):
    monkeypatch.setenv("MCP_PRONUNCIATION_PRELOAD", "0")
    sys.modules.pop("mcp_server_pronunciation.server", None)

    server = importlib.import_module("mcp_server_pronunciation.server")
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
