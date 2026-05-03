"""Tests for server startup controls."""

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
