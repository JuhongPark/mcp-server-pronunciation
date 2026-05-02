# Roadmap

This project is being developed as a public, local-first MCP server for voice
conversation and English pronunciation coaching. The long-term goal is not only
to provide a useful interactive demo, but to build a measurable pronunciation
assessment tool whose behavior can be explained, benchmarked, and improved over
time.

## North Star

`mcp-server-pronunciation` should let a user speak naturally to an MCP client,
receive a transcript, and get concise English feedback without sending audio to
an external service. In practice mode, it should provide useful sentence-level,
word-level, phoneme-level, and fluency feedback with clear uncertainty when the
evidence is weak.

The public value proposition is:

- Local audio processing by default.
- Clear privacy and model-download behavior.
- Benchmark-backed pronunciation feedback.
- Strong support for Korean L1 English learners, without making the whole
  project Korean-only.
- Simple installation through `uvx`, `uv tool`, or `pip`.

## Development Principles

- Prefer measurable improvements over subjective tuning.
- Keep public claims narrower than the current evidence.
- Do not redistribute benchmark datasets in the repository.
- Keep non-commercial datasets out of release claims and default public
  benchmarks.
- Treat pronunciation scores as coaching signals, not medical, clinical, or
  standardized-test-grade judgments.
- Keep the default install usable without the heavy forced-alignment stack.
- Make optional network downloads explicit in docs and diagnostics.

## Phase 0: Public Alpha Hardening

Goal: make the current package clean, installable, and honest enough for public
users and contributors.

Tasks:

- Fix formatting so `ruff format --check .` passes.
- Align versions across `pyproject.toml`, `server.json`, README, and
  `CHANGELOG.md`.
- Split extras into lighter groups, such as `dev`, `phoneme`, and `bench`, so
  regular development does not always pull `torch` and `torchaudio`.
- Add `SECURITY.md`, `CONTRIBUTING.md`, and issue templates.
- Add README sections for known limitations, privacy, model downloads, and
  benchmark status.
- Expand `doctor` to check optional phoneme dependencies, NLTK/g2p resources,
  cache locations, and likely first-run downloads.
- Document audio retention behavior and add a configurable temp-file retention
  policy.
- Make WSL recording safer by using a collision-resistant Windows temp file.

Success criteria:

- Fresh clone passes tests, lint, and format checks.
- Fresh install instructions work without local repository context.
- Public docs do not overstate scoring accuracy.
- The repository has a clear path for security reports and contributions.

## Phase 1: Benchmark Foundation

Goal: create a reproducible evaluation harness before doing more scoring
calibration.

Tasks:

- Add a `bench` command or script group that runs dataset adapters and writes
  JSON plus Markdown reports.
- Keep datasets outside the repository and cache them locally.
- Support small `--limit` runs for quick smoke checks.
- Record environment metadata in every report: package version, Python version,
  OS, model size, optional extras, dataset version, sample count, and runtime.
- Add benchmark tests for adapter parsing without downloading full datasets.

Primary metrics:

- Transcript word error rate.
- Empty transcript rate.
- Runtime per utterance.
- Sentence-score correlation.
- Word-score correlation.
- Phone-error precision, recall, and F1.
- Forced-alignment match/miss calibration curves.
- Prosody false-positive review rate on clean clips.

Success criteria:

- A developer can run a small benchmark locally with one command.
- Benchmark output is stable enough to compare before and after changes.
- CI can test benchmark code paths without downloading large datasets.

## Phase 2: Scoring Calibration

Goal: make pronunciation and prosody feedback less noisy and more evidence-based.

Tasks:

- Calibrate forced-alignment confidence thresholds on benchmark data.
- Separate alignment confidence, pronunciation confidence, and coaching severity.
- Improve `AssessmentResult.to_dict()` as the stable machine-readable contract.
- Add snapshot tests for representative reports.
- Use forced-alignment timestamps where available for prosody and pause checks.
- Make prosody findings conservative when pitch evidence is weak.
- Add severity levels such as `info`, `practice`, and `important`.
- Add explicit `learner_l1` configuration. Korean-L1 rules should run when
  requested or inferred from a Korean-focused practice mode, not as the only
  possible learner profile.

Success criteria:

- Scoring changes can be compared through benchmark reports.
- Low-confidence findings are clearly labeled.
- Korean-L1 feedback remains useful while the default product works for broader
  English learners.

## Phase 3: Coaching Experience

Goal: turn the analysis pipeline into a better practice loop.

Tasks:

- Add feedback tone modes: `gentle`, `balanced`, and `strict`.
- Improve retry reports by comparing the current attempt to the previous one.
- Expand sentence focus areas: `th`, `r_l`, `f_v`, `vowels`, `clusters`,
  `stress`, `intonation`, and `general`.
- Move the sentence bank out of code and into structured data.
- Add user-facing controls for device selection, VAD sensitivity, silence
  duration, and max recording length.
- Add concise "next drill" suggestions based on the strongest detected issue.
- Keep conversation-mode feedback short enough that the MCP client can respond
  naturally.

Success criteria:

- Practice mode supports a complete loop: target sentence, recording, feedback,
  retry, and improvement summary.
- Converse mode remains a conversation tool, not a report dump.

## Phase 4: Public Beta And Release Operations

Goal: make releases repeatable and low-risk.

Tasks:

- Verify PyPI Trusted Publisher release flow.
- Validate `server.json` against the MCP Registry schema before release.
- Add release checklist documentation.
- Add a manual benchmark workflow for maintainers.
- Publish benchmark summaries without bundling datasets.
- Add a small demo transcript that uses synthetic or license-safe audio.
- Tag releases with concise release notes tied to `CHANGELOG.md`.

Success criteria:

- A release can be built, tested, tagged, and published without manual guessing.
- Public metadata matches the published package version.
- Users can install and run `doctor` before wiring the server into an MCP client.

## Phase 5: Extensibility

Goal: grow from a Korean-L1-focused coach into a broader local pronunciation
assessment toolkit.

Tasks:

- Introduce rule-pack structure for learner profiles such as `ko`, `zh`, `ja`,
  and `es`.
- Keep language-specific tips in data files rather than hard-coding all rules.
- Add APIs for external sentence banks and custom drills.
- Consider a lightweight CLI workflow for users who do not use MCP.
- Document the machine-readable result schema for downstream tools.

Success criteria:

- New learner profiles can be added without rewriting the assessor.
- MCP users and CLI users share the same core assessment pipeline.

## Benchmark Dataset Candidates

### Speechocean762

Primary public benchmark candidate.

- Source: https://openslr.org/101/
- Paper: https://www.isca-archive.org/interspeech_2021/zhang21x_interspeech.html
- License: CC BY 4.0.
- Scope: 5,000 English utterances by 250 Mandarin-L1 non-native speakers.
- Labels: sentence-level, word-level, and phoneme-level expert scores.
- Use: sentence score correlation, word accuracy correlation, word stress
  checks, phone goodness correlation, and end-to-end report regression.

### L2-ARCTIC

Best research benchmark for phone-level error detection and Korean-L1 behavior.

- Source: https://psi.engr.tamu.edu/l2-arctic-corpus/
- Docs: https://psi.engr.tamu.edu/l2-arctic-corpus-docs/
- License: CC BY-NC 4.0.
- Scope: 24 non-native English speakers across Arabic, Chinese, Hindi, Korean,
  Spanish, and Vietnamese L1 groups.
- Labels: forced-aligned word and phone boundaries plus manual phone
  substitution, deletion, and addition annotations for a subset.
- Use: phone error precision/recall/F1, Korean speaker subset checks, alignment
  robustness, and rule-pack evaluation.
- Constraint: non-commercial license. Keep this out of default public release
  claims and treat it as an optional research benchmark.

### Common Voice

Useful for ASR and accent robustness, not direct pronunciation scoring.

- Source: https://commonvoice.mozilla.org/
- Example dataset page:
  https://datacollective.mozillafoundation.org/datasets/cmn1pv5hi00uto1072y1074y7
- License: generally CC0 for released datasets, with dataset-specific terms.
- Use: transcript word error rate, empty transcript rate, accent robustness, and
  runtime smoke tests.
- Constraint: do not attempt speaker identification. Do not re-host data.

### Speech Accent Archive

Useful for broad accent smoke tests and qualitative review.

- Source: https://accent.gmu.edu/
- About: https://accent.gmu.edu/about.php
- Scope: many speakers reading the same English paragraph, with demographic and
  linguistic background metadata.
- Use: accent robustness checks and qualitative analysis.
- Constraint: licensing varies across mirrors and is non-commercial/share-alike
  in common packaged forms. Do not use it as the default public benchmark.

### Speak & Improve Corpus 2025

Potential future spontaneous-speech benchmark.

- Source:
  https://www.repository.cam.ac.uk/items/611e9b8b-c5b2-4cbe-9b7a-036a48334118
- Scope: L2 learner English with holistic scores and language-error annotation.
- Use: future evaluation of open-ended conversation feedback.
- Constraint: non-commercial availability. Treat as a later optional research
  target.

## Near-Term Implementation Order

1. Fix formatting and version metadata.
2. Split dependency extras and lighten default CI.
3. Add public project docs: limitations, security, contributing, and release
   checklist.
4. Add the benchmark harness skeleton with a Speechocean762 adapter stub.
5. Add small benchmark report output and environment metadata.
6. Add L2-ARCTIC adapter support behind an explicit research-only flag.
7. Use benchmark reports to calibrate forced alignment and scoring thresholds.
8. Improve retry and coaching UX once scoring behavior is measurable.

## Non-Goals

- Do not claim clinical, standardized-test, or native-speaker equivalence.
- Do not bundle third-party speech datasets in the repository.
- Do not require GPU or heavy optional dependencies for the default user path.
- Do not make every conversation turn a detailed pronunciation report.
- Do not optimize for benchmark scores at the expense of useful coaching.

