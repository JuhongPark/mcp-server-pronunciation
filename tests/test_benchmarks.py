"""Tests for benchmark manifest parsing."""

import json

from mcp_server_pronunciation.cli import main
from mcp_server_pronunciation.benchmarks import (
    load_speechocean_jsonl,
    make_adapter_report,
)


def test_load_speechocean_jsonl(tmp_path):
    manifest = tmp_path / "speechocean.jsonl"
    rows = [
        {
            "spk": "S001",
            "utt_name": "utt-1",
            "audio": {"path": "audio/utt-1.wav"},
            "utt_text": "This is a test.",
            "utt_total": 8.5,
            "utt_accuracy": 9,
            "gender": "F",
        },
        {
            "spk": "S002",
            "utt_name": "utt-2",
            "audio_path": "audio/utt-2.wav",
            "text": "Another test.",
            "utt_total": 7.0,
        },
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    items = load_speechocean_jsonl(manifest)

    assert len(items) == 2
    assert items[0].item_id == "utt-1"
    assert items[0].audio_path == "audio/utt-1.wav"
    assert items[0].scores["utt_total"] == 8.5
    assert items[0].metadata["gender"] == "F"
    assert items[1].text == "Another test."


def test_load_speechocean_jsonl_limit(tmp_path):
    manifest = tmp_path / "speechocean.jsonl"
    manifest.write_text(
        "\n".join(json.dumps({"utt_name": f"utt-{i}", "utt_text": "hello"}) for i in range(3))
        + "\n",
        encoding="utf-8",
    )

    items = load_speechocean_jsonl(manifest, limit=2)

    assert [item.item_id for item in items] == ["utt-0", "utt-1"]


def test_adapter_report_summary(tmp_path):
    manifest = tmp_path / "speechocean.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "spk": "S001",
                        "utt_name": "utt-1",
                        "audio_path": "audio/utt-1.wav",
                        "utt_text": "hello",
                        "utt_total": 9,
                    }
                ),
                json.dumps(
                    {
                        "spk": "S001",
                        "utt_name": "utt-2",
                        "utt_text": "world",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    items = load_speechocean_jsonl(manifest)

    report = make_adapter_report("speechocean762", items)

    assert report.status == "adapter-only"
    assert report.summary["sample_count"] == 2
    assert report.summary["items_with_audio_path"] == 1
    assert report.summary["items_with_sentence_total"] == 1
    assert report.summary["speaker_count"] == 1
    assert len(report.items) == 2


def test_cli_bench_speechocean_writes_report(tmp_path):
    manifest = tmp_path / "speechocean.jsonl"
    output = tmp_path / "report.json"
    manifest.write_text(
        json.dumps(
            {
                "spk": "S001",
                "utt_name": "utt-1",
                "utt_text": "hello",
                "utt_total": 9,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    code = main(
        [
            "bench",
            "speechocean",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dataset"] == "speechocean762"
    assert report["summary"]["sample_count"] == 1
