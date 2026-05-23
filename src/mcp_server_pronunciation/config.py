"""Shared configuration helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_WHISPER_MODEL = "base.en"
SUPPORTED_WHISPER_MODELS = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v3",
    "large-v3-turbo",
)

AUDIO_RETENTION_VALUES = ("session", "keep")
PRELOAD_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})
VAD_SENSITIVITY_VALUES = ("low", "normal", "high")

DEFAULT_VAD_SENSITIVITY = "normal"
DEFAULT_VAD_SILENCE_DURATION = 1.5
MIN_VAD_SILENCE_DURATION = 0.3
MAX_VAD_SILENCE_DURATION = 5.0


def _env(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if env is None else env


def whisper_model_name(env: Mapping[str, str] | None = None) -> str:
    value = _env(env).get("MCP_PRONUNCIATION_MODEL", DEFAULT_WHISPER_MODEL).strip()
    return value or DEFAULT_WHISPER_MODEL


def is_documented_whisper_model(value: str) -> bool:
    return value in SUPPORTED_WHISPER_MODELS


def audio_retention_value(env: Mapping[str, str] | None = None) -> str:
    value = _env(env).get("MCP_PRONUNCIATION_AUDIO_RETENTION", "session").strip().lower()
    return value if value in AUDIO_RETENTION_VALUES else "session"


def preload_enabled(env: Mapping[str, str] | None = None) -> bool:
    value = _env(env).get("MCP_PRONUNCIATION_PRELOAD", "1").strip().lower()
    return value not in PRELOAD_DISABLED_VALUES


def input_device_value(env: Mapping[str, str] | None = None) -> str | None:
    value = _env(env).get("MCP_PRONUNCIATION_INPUT_DEVICE", "").strip()
    return value or None


def vad_sensitivity_value(env: Mapping[str, str] | None = None) -> str:
    value = (
        _env(env).get("MCP_PRONUNCIATION_VAD_SENSITIVITY", DEFAULT_VAD_SENSITIVITY).strip().lower()
    )
    return value if value in VAD_SENSITIVITY_VALUES else DEFAULT_VAD_SENSITIVITY


def vad_silence_duration_seconds(env: Mapping[str, str] | None = None) -> float:
    raw = _env(env).get("MCP_PRONUNCIATION_SILENCE_DURATION", "").strip()
    if not raw:
        return DEFAULT_VAD_SILENCE_DURATION
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_VAD_SILENCE_DURATION
    return min(max(value, MIN_VAD_SILENCE_DURATION), MAX_VAD_SILENCE_DURATION)
