"""Lightweight benchmark manifest adapters and report helpers."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import __version__


SPEECHOCEAN_SCORE_KEYS = (
    "utt_accuracy",
    "utt_completeness",
    "utt_fluency",
    "utt_prosodic",
    "utt_total",
    "words_accuracy",
    "words_stress",
    "words_total",
    "phones_godness",
)


@dataclass
class BenchmarkItem:
    """One utterance entry from a benchmark manifest."""

    dataset: str
    item_id: str
    text: str
    audio_path: str | None = None
    speaker: str | None = None
    scores: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Small JSON report for adapter smoke checks and future scoring runs."""

    dataset: str
    status: str
    generated_at_unix: float
    environment: dict[str, Any]
    summary: dict[str, Any]
    items: list[dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def environment_metadata() -> dict[str, Any]:
    """Return metadata that makes benchmark reports comparable."""
    return {
        "package_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def load_speechocean_jsonl(path: Path, limit: int | None = None) -> list[BenchmarkItem]:
    """Load a JSONL manifest exported from Speechocean762-style metadata.

    Expected fields are intentionally permissive so the adapter can read either
    a hand-authored manifest or a small export from Hugging Face datasets.
    """
    items: list[BenchmarkItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            items.append(_speechocean_item(raw, line_no))
            if limit is not None and len(items) >= limit:
                break
    return items


def make_adapter_report(dataset: str, items: list[BenchmarkItem]) -> BenchmarkReport:
    """Build a report that verifies adapter parsing without running ASR."""
    speaker_count = len({item.speaker for item in items if item.speaker})
    score_counts: dict[str, int] = {}
    for key in SPEECHOCEAN_SCORE_KEYS:
        score_counts[key] = sum(1 for item in items if key in item.scores)

    summary = {
        "sample_count": len(items),
        "items_with_audio_path": sum(1 for item in items if item.audio_path),
        "items_with_sentence_total": sum(1 for item in items if "utt_total" in item.scores),
        "speaker_count": speaker_count,
        "score_counts": score_counts,
    }

    preview = [asdict(item) for item in items[:5]]
    return BenchmarkReport(
        dataset=dataset,
        status="adapter-only",
        generated_at_unix=time.time(),
        environment=environment_metadata(),
        summary=summary,
        items=preview,
    )


def write_report(report: BenchmarkReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.to_json(), encoding="utf-8")


def _speechocean_item(raw: dict[str, Any], line_no: int) -> BenchmarkItem:
    text = raw.get("utt_text") or raw.get("text") or raw.get("sentence")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"manifest line {line_no}: missing `utt_text` or `text`")

    audio_path = raw.get("audio_path") or raw.get("path")
    audio = raw.get("audio")
    if audio_path is None and isinstance(audio, dict):
        audio_path = audio.get("path")
    if audio_path is not None:
        audio_path = str(audio_path)

    item_id = str(raw.get("utt_name") or raw.get("id") or f"line-{line_no}")
    scores = {key: raw[key] for key in SPEECHOCEAN_SCORE_KEYS if key in raw}
    metadata = {
        key: raw[key] for key in ("age", "gender", "split") if key in raw and raw[key] is not None
    }
    return BenchmarkItem(
        dataset="speechocean762",
        item_id=item_id,
        text=text.strip(),
        audio_path=audio_path,
        speaker=str(raw["spk"]) if "spk" in raw else None,
        scores=scores,
        metadata=metadata,
    )
