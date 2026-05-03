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
