"""Tests for benchmark score and detection metrics."""

import json
import math

import pytest

from mcp_server_pronunciation.benchmarks import (
    detection_metrics,
    load_score_predictions_jsonl,
    make_score_report,
    regression_metrics,
)
from mcp_server_pronunciation.cli import main


def test_regression_metrics_perfect_agreement():
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])

    assert metrics.count == 3
    assert metrics.mae == 0
    assert metrics.rmse == 0
    assert metrics.pearson == pytest.approx(1.0)
    assert metrics.spearman == pytest.approx(1.0)


def test_regression_metrics_skips_invalid_pairs():
    metrics = regression_metrics([1, "bad", 2, None], [1, 7, 4, 9])

    assert metrics.count == 2
    assert metrics.mae == pytest.approx(1.0)
    assert metrics.rmse == pytest.approx(math.sqrt(2))


def test_regression_metrics_handles_inverse_rank_order():
    metrics = regression_metrics([1, 2, 3], [3, 2, 1])

    assert metrics.pearson == pytest.approx(-1.0)
    assert metrics.spearman == pytest.approx(-1.0)


def test_regression_metrics_returns_null_correlation_for_zero_variance():
    metrics = regression_metrics([5, 5, 5], [1, 2, 3])

    assert metrics.pearson is None
    assert metrics.spearman is None


def test_detection_metrics_counts_binary_outcomes():
    metrics = detection_metrics(
        [True, True, False, False],
        [True, False, True, False],
    )

    assert metrics.count == 4
    assert metrics.true_positive == 1
    assert metrics.false_negative == 1
    assert metrics.false_positive == 1
    assert metrics.true_negative == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.accuracy == pytest.approx(0.5)


def test_detection_metrics_accepts_common_string_labels():
    metrics = detection_metrics(
        ["mispronounced", "correct", "error", "ok"],
        ["true", "false", "false", "false"],
    )

    assert metrics.true_positive == 1
    assert metrics.false_negative == 1
    assert metrics.true_negative == 2


def test_load_score_predictions_jsonl_with_dotted_fields(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [
        {"id": "utt-1", "human": {"utt_total": 8.0}, "predicted": {"utt_total": 7.5}},
        {"id": "utt-2", "human": {"utt_total": 9.0}, "predicted": {"utt_total": "9.5"}},
        {"id": "utt-3", "human": {}, "predicted": {"utt_total": 6.0}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    predictions = load_score_predictions_jsonl(
        path,
        gold_field="human.utt_total",
        predicted_field="predicted.utt_total",
    )
    report = make_score_report(
        "speechocean762",
        predictions,
        gold_field="human.utt_total",
        predicted_field="predicted.utt_total",
    )

    assert len(predictions) == 3
    assert predictions[0].item_id == "utt-1"
    assert report.summary["valid_pair_count"] == 2
    assert report.summary["skipped_pair_count"] == 1
    assert report.metrics["mae"] == pytest.approx(0.5)


def test_cli_bench_score_report_writes_metrics(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "score_report.json"
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"id": "utt-1", "gold": 8, "pred": 7}),
                json.dumps({"id": "utt-2", "gold": 9, "pred": 9}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    code = main(
        [
            "bench",
            "score-report",
            "--predictions",
            str(predictions),
            "--gold-field",
            "gold",
            "--pred-field",
            "pred",
            "--dataset",
            "fixture",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dataset"] == "fixture"
    assert report["summary"]["valid_pair_count"] == 2
    assert report["metrics"]["mae"] == pytest.approx(0.5)
