# Benchmarks

Benchmark support is intentionally data-external: this repository contains
adapters and report code, not third-party speech datasets.

## Speechocean762 Adapter

The first benchmark helper parses a small JSONL manifest exported from
Speechocean762 metadata and writes an adapter report with environment metadata.
It does not run ASR or pronunciation scoring yet.

Example:

```bash
mcp-server-pronunciation bench speechocean \
  --manifest /path/to/speechocean_manifest.jsonl \
  --limit 100 \
  --output benchmark/results/speechocean_adapter.json
```

Each JSONL row should include at least `utt_text` or `text`. The adapter also
recognizes common Speechocean-style fields:

- `spk`
- `utt_name`
- `audio.path` or `audio_path`
- `utt_accuracy`
- `utt_completeness`
- `utt_fluency`
- `utt_prosodic`
- `utt_total`
- `words_accuracy`
- `words_stress`
- `words_total`
- `phones_godness`

The report currently records:

- package and Python environment metadata
- parsed sample count
- audio-path coverage
- sentence-score coverage
- speaker count
- score-field coverage
- a small preview of parsed items

## Data Policy

Do not commit benchmark datasets, downloaded audio, model weights, or generated
large reports. Keep local benchmark data under an ignored directory such as
`benchmark/data/` or outside the repository.

