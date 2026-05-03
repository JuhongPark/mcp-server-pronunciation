"""Tests for shared configuration parsing."""

from mcp_server_pronunciation.config import (
    audio_retention_value,
    is_documented_whisper_model,
    preload_enabled,
    whisper_model_name,
)


def test_whisper_model_name_defaults_and_trims_values():
    assert whisper_model_name({}) == "base.en"
    assert whisper_model_name({"MCP_PRONUNCIATION_MODEL": " small.en "}) == "small.en"
    assert whisper_model_name({"MCP_PRONUNCIATION_MODEL": " "}) == "base.en"


def test_supported_whisper_model_names_are_documented():
    assert is_documented_whisper_model("base.en") is True
    assert is_documented_whisper_model("large-v3-turbo") is True
    assert is_documented_whisper_model("basee.en") is False


def test_audio_retention_uses_safe_default_for_invalid_values():
    assert audio_retention_value({}) == "session"
    assert audio_retention_value({"MCP_PRONUNCIATION_AUDIO_RETENTION": " keep "}) == "keep"
    assert audio_retention_value({"MCP_PRONUNCIATION_AUDIO_RETENTION": "forever"}) == "session"


def test_preload_enabled_accepts_common_false_values():
    assert preload_enabled({}) is True
    assert preload_enabled({"MCP_PRONUNCIATION_PRELOAD": "0"}) is False
    assert preload_enabled({"MCP_PRONUNCIATION_PRELOAD": "OFF"}) is False
    assert preload_enabled({"MCP_PRONUNCIATION_PRELOAD": "yes"}) is True
