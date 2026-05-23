"""Tests for shared configuration parsing."""

from mcp_server_pronunciation.config import (
    audio_retention_value,
    input_device_value,
    is_documented_whisper_model,
    preload_enabled,
    vad_sensitivity_value,
    vad_silence_duration_seconds,
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


def test_input_device_value_trims_blank_values():
    assert input_device_value({}) is None
    assert input_device_value({"MCP_PRONUNCIATION_INPUT_DEVICE": " 2 "}) == "2"
    assert input_device_value({"MCP_PRONUNCIATION_INPUT_DEVICE": " "}) is None


def test_vad_sensitivity_uses_documented_values():
    assert vad_sensitivity_value({}) == "normal"
    assert vad_sensitivity_value({"MCP_PRONUNCIATION_VAD_SENSITIVITY": " HIGH "}) == "high"
    assert vad_sensitivity_value({"MCP_PRONUNCIATION_VAD_SENSITIVITY": "broken"}) == "normal"


def test_vad_silence_duration_clamps_invalid_values():
    assert vad_silence_duration_seconds({}) == 1.5
    assert vad_silence_duration_seconds({"MCP_PRONUNCIATION_SILENCE_DURATION": "2.25"}) == 2.25
    assert vad_silence_duration_seconds({"MCP_PRONUNCIATION_SILENCE_DURATION": "0.1"}) == 0.3
    assert vad_silence_duration_seconds({"MCP_PRONUNCIATION_SILENCE_DURATION": "10"}) == 5.0
    assert vad_silence_duration_seconds({"MCP_PRONUNCIATION_SILENCE_DURATION": "nope"}) == 1.5
