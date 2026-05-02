"""Lightweight benchmark manifest adapters and report helpers."""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

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


@dataclass
class RegressionMetrics:
    """Score-agreement metrics for human-vs-machine pronunciation scores."""

    count: int
    mae: float | None
    rmse: float | None
    pearson: float | None
    spearman: float | None
    gold_mean: float | None
    predicted_mean: float | None


@dataclass
class DetectionMetrics:
    """Binary mispronunciation-detection metrics.

    The positive class is "mispronounced and flagged".
    """

    count: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float | None
    recall: float | None
    f1: float | None
    accuracy: float | None
    false_alarm_rate: float | None
    miss_rate: float | None


@dataclass
class ScorePrediction:
    """One human-vs-machine score pair loaded from a prediction export."""

    item_id: str
    gold: float | None
    predicted: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreReport:
    """JSON report for score-agreement evaluation."""

    dataset: str
    status: str
    generated_at_unix: float
    environment: dict[str, Any]
    summary: dict[str, Any]
    metrics: dict[str, Any]
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


def regression_metrics(
    gold_values: Iterable[Any],
    predicted_values: Iterable[Any],
) -> RegressionMetrics:
    """Compute score-agreement metrics, skipping non-numeric pairs."""
    gold_list = list(gold_values)
    predicted_list = list(predicted_values)
    if len(gold_list) != len(predicted_list):
        raise ValueError("gold and predicted values must have the same length")

    pairs: list[tuple[float, float]] = []
    for gold, predicted in zip(gold_list, predicted_list):
        gold_float = _to_float(gold)
        predicted_float = _to_float(predicted)
        if gold_float is None or predicted_float is None:
            continue
        pairs.append((gold_float, predicted_float))

    if not pairs:
        return RegressionMetrics(
            count=0,
            mae=None,
            rmse=None,
            pearson=None,
            spearman=None,
            gold_mean=None,
            predicted_mean=None,
        )

    gold = [p[0] for p in pairs]
    predicted = [p[1] for p in pairs]
    errors = [pred - ref for ref, pred in pairs]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    return RegressionMetrics(
        count=len(pairs),
        mae=mae,
        rmse=rmse,
        pearson=_pearson(gold, predicted),
        spearman=_spearman(gold, predicted),
        gold_mean=sum(gold) / len(gold),
        predicted_mean=sum(predicted) / len(predicted),
    )


def detection_metrics(
    gold_mispronounced: Iterable[Any],
    predicted_mispronounced: Iterable[Any],
) -> DetectionMetrics:
    """Compute binary metrics for mispronunciation detection."""
    gold_list = list(gold_mispronounced)
    predicted_list = list(predicted_mispronounced)
    if len(gold_list) != len(predicted_list):
        raise ValueError("gold and predicted labels must have the same length")

    tp = fp = tn = fn = 0
    for gold, predicted in zip(gold_list, predicted_list):
        gold_bool = _to_bool(gold)
        predicted_bool = _to_bool(predicted)
        if gold_bool is None or predicted_bool is None:
            continue
        if gold_bool and predicted_bool:
            tp += 1
        elif not gold_bool and predicted_bool:
            fp += 1
        elif not gold_bool and not predicted_bool:
            tn += 1
        else:
            fn += 1

    count = tp + fp + tn + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None:
        f1 = _safe_div(2 * precision * recall, precision + recall)

    return DetectionMetrics(
        count=count,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=_safe_div(tp + tn, count),
        false_alarm_rate=_safe_div(fp, fp + tn),
        miss_rate=_safe_div(fn, tp + fn),
    )


def load_score_predictions_jsonl(
    path: Path,
    gold_field: str,
    predicted_field: str,
    id_field: str = "id",
    limit: int | None = None,
) -> list[ScorePrediction]:
    """Load human-vs-machine score pairs from a JSONL prediction export.

    Field names may be dotted paths such as `human.utt_total`.
    """
    predictions: list[ScorePrediction] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")

            item_id_raw = _get_dotted(raw, id_field)
            item_id = str(item_id_raw) if item_id_raw is not _MISSING else f"line-{line_no}"
            predictions.append(
                ScorePrediction(
                    item_id=item_id,
                    gold=_to_float(_get_dotted(raw, gold_field)),
                    predicted=_to_float(_get_dotted(raw, predicted_field)),
                    metadata={"line_no": line_no},
                )
            )
            if limit is not None and len(predictions) >= limit:
                break
    return predictions


def make_score_report(
    dataset: str,
    predictions: list[ScorePrediction],
    gold_field: str,
    predicted_field: str,
) -> ScoreReport:
    """Build a score-agreement report from loaded prediction rows."""
    valid = [p for p in predictions if p.gold is not None and p.predicted is not None]
    metrics = regression_metrics(
        [p.gold for p in valid],
        [p.predicted for p in valid],
    )
    summary = {
        "row_count": len(predictions),
        "valid_pair_count": metrics.count,
        "skipped_pair_count": len(predictions) - metrics.count,
        "gold_field": gold_field,
        "predicted_field": predicted_field,
    }
    return ScoreReport(
        dataset=dataset,
        status="score-report",
        generated_at_unix=time.time(),
        environment=environment_metadata(),
        summary=summary,
        metrics=asdict(metrics),
        items=[asdict(item) for item in predictions[:5]],
    )


def write_report(report: BenchmarkReport | ScoreReport, output_path: Path) -> None:
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


_MISSING = object()


def _get_dotted(raw: dict[str, Any], field_name: str) -> Any:
    value: Any = raw
    for part in field_name.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _to_float(value: Any) -> float | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _to_bool(value: Any) -> bool | None:
    if value is None or value is _MISSING:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "mispronounced", "error", "incorrect"}:
            return True
        if normalized in {"0", "false", "no", "n", "correct", "ok", "accepted"}:
            return False
    return None


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_delta = [x - x_mean for x in xs]
    y_delta = [y - y_mean for y in ys]
    numerator = sum(x * y for x, y in zip(x_delta, y_delta))
    x_den = math.sqrt(sum(x * x for x in x_delta))
    y_den = math.sqrt(sum(y * y for y in y_delta))
    if x_den == 0 or y_den == 0:
        return None
    return numerator / (x_den * y_den)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _average_ranks(values: list[float]) -> list[float]:
    sorted_pairs = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_pairs):
        j = i + 1
        while j < len(sorted_pairs) and sorted_pairs[j][0] == sorted_pairs[i][0]:
            j += 1
        average_rank = (i + 1 + j) / 2
        for _value, original_index in sorted_pairs[i:j]:
            ranks[original_index] = average_rank
        i = j
    return ranks
