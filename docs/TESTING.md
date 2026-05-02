# Testing And Evaluation Plan

This project needs two kinds of evidence before public users should trust it:
software tests that keep the local MCP server reliable, and evaluation reports
that show how pronunciation feedback agrees with human labels on known speech
datasets. Passing unit tests is necessary, but it is not enough to claim the
coach is accurate.

## Methodology Survey

Automatic pronunciation assessment is normally evaluated as a human-agreement
problem. The classic framing is to collect learner speech, collect expert human
ratings, compute machine scores, and measure how well the machine predicts the
human scores. The core score-level metrics are Pearson correlation, Spearman
rank correlation, mean absolute error, and root mean squared error.

Mispronunciation detection is evaluated as a detection problem. The positive
class is a mispronounced phone or word that the system flags. Reports should
include precision, recall, F1, accuracy, false alarm rate, and miss rate. This
matters because a coach with high recall but many false alarms feels punitive,
while a coach with high precision but low recall may miss the user's main
practice target.

ASR and recording stability are separate from pronunciation scoring. Transcript
quality should use word error rate, empty transcript rate, and runtime per
utterance. Pronunciation feedback can be wrong because the acoustic score is
wrong, because ASR misheard the utterance, or because the feedback rules mapped
the evidence badly. Evaluation reports should keep those layers separate.

References:

- Franco et al., "Automatic scoring of pronunciation quality", Speech
  Communication 2000: https://doi.org/10.1016/S0167-6393(99)00046-1
- Zhang et al., "speechocean762", Interspeech 2021:
  https://www.isca-archive.org/interspeech_2021/zhang21x_interspeech.html
- Zhao et al., "L2-ARCTIC", Interspeech 2018:
  https://www.isca-archive.org/interspeech_2018/zhao18b_interspeech.html
- El Kheir et al., "Automatic Pronunciation Assessment: A Review", Findings of
  EMNLP 2023: https://aclanthology.org/2023.findings-emnlp.557/
- Amrate and Tsai, "Computer-assisted pronunciation training: A systematic
  review", ReCALL 2024:
  https://www.cambridge.org/core/journals/recall/article/computerassisted-pronunciation-training-a-systematic-review/71E786F7DFC99727837909FDED7A2320
- OpenSLR Speechocean762 resource page: https://openslr.org/101/
- L2-ARCTIC corpus documentation:
  https://psi.engr.tamu.edu/l2-arctic-corpus-docs/

## Test Pyramid

### Unit Tests

Goal: keep deterministic behavior stable without microphones, model downloads,
or third-party datasets.

Coverage:

- Text tokenization and Needleman-Wunsch word alignment.
- CMUdict and G2P phoneme lookup behavior.
- Korean-L1 rule detection and drill selection.
- Assessment result scoring and report rendering.
- Benchmark manifest parsing and metric computation.
- CLI command parsing for benchmark reports.

Run:

```bash
uv run pytest -q
```

### Pipeline Tests

Goal: verify the assessor wiring without loading Whisper or reading real audio.

Coverage:

- Fake ASR segments become `AssessmentResult` words and transcripts.
- Reference analysis produces alignment, phoneme diffs, and learner-pattern
  findings.
- Forced-alignment confidence can override Whisper mismatches.
- Prosody can be patched out cleanly when testing non-prosody behavior.

These tests use pytest monkeypatching so they remain offline and fast.

### Local Audio Smoke Tests

Goal: verify the installed package against the user's machine.

Coverage:

- `doctor` checks microphone dependencies, model cache, pronunciation
  resources, optional forced alignment, and disk space.
- `pull-model` pre-warms Whisper weights.
- `record`, `assess`, and `practice` are manually checked with a short target
  sentence.

Run:

```bash
uv run mcp-server-pronunciation doctor
uv run mcp-server-pronunciation pull-model base.en
```

Manual audio tests should record the observed platform, model, command, target
sentence, transcript, and any surprising feedback. Do not commit audio clips
unless they are synthetic or explicitly license-safe.

### Benchmark Reports

Goal: compare releases against human-labeled data without committing datasets.

The benchmark harness should write JSON reports under `benchmark/results/`.
Datasets, audio, downloaded archives, and large generated reports must stay out
of git.

Minimum report metadata:

- Package version.
- Python version and platform.
- Dataset name and local manifest path.
- Sample count and speaker count when available.
- Model configuration.
- Runtime per utterance when scoring runs are implemented.
- Metric names, counts, and null-correlation reasons when applicable.

## Dataset Plan

### Speechocean762

Role: primary public scoring benchmark.

Why: it has a permissive CC BY 4.0 license, 5,000 read English utterances, and
expert labels at sentence, word, and phone levels.

Use:

- Sentence-score Pearson and Spearman correlation against `utt_total`.
- MAE and RMSE against sentence-level total scores.
- Word-score and phone-score coverage once the adapter reaches those levels.
- Regression checks for score calibration and report output.

### L2-ARCTIC

Role: research benchmark for phone-level mispronunciation detection.

Why: it contains non-native English speakers across several L1 groups, including
Korean, with manual phone substitution, deletion, and addition annotations for a
subset.

Use:

- Phone-level precision, recall, F1, false alarm rate, and miss rate.
- Korean speaker subset review for rule-pack behavior.
- Forced-alignment robustness checks.

Constraint: the license is non-commercial. Keep it optional, local-only, and
out of public release accuracy claims.

### Common Voice

Role: ASR and robustness smoke benchmark, not pronunciation scoring.

Use:

- WER against sentence transcripts.
- Empty transcript rate.
- Runtime and crash rate across accent and device diversity.

### Speech Accent Archive

Role: qualitative accent robustness review.

Use:

- Check whether feedback remains conservative across many accent backgrounds.
- Avoid numeric public claims unless the exact data license and split are clear.

## Metrics

Score regression:

- `mae`: average absolute distance between human and machine score.
- `rmse`: square-root mean squared distance.
- `pearson`: linear correlation. Use `null` when either side has zero variance.
- `spearman`: rank correlation. Use average ranks for ties.

Detection:

- `true_positive`: mispronunciation correctly flagged.
- `false_positive`: correct production incorrectly flagged.
- `true_negative`: correct production accepted.
- `false_negative`: mispronunciation missed.
- `precision`: flagged errors that were real errors.
- `recall`: real errors that were flagged.
- `f1`: harmonic mean of precision and recall.
- `false_alarm_rate`: false positives among correct productions.
- `miss_rate`: false negatives among real errors.

ASR:

- Word error rate.
- Empty transcript rate.
- Runtime per utterance.
- Crash or unavailable-audio rate.

## Current Implementation Target

This cycle implements the foundation that can run without external data:

- Public testing and evaluation plan.
- Pure-Python regression and detection metrics.
- CLI score-report generation from JSONL prediction files.
- Deterministic tests for metrics, CLI output, and assessor pipeline wiring.

Next benchmark work should add full scoring runs over a local Speechocean762
manifest and then a research-only L2-ARCTIC adapter.

