# Contributing

Thanks for considering a contribution. This project is early and the best
changes are focused, measurable, and easy to review.

## Development Setup

```bash
uv sync --extra dev
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

To work on the optional forced-alignment path:

```bash
uv sync --extra dev --extra phoneme
```

## Contribution Priorities

Good first areas:

- Clearer diagnostics in `doctor`.
- Small pronunciation rule fixes with tests.
- Benchmark adapters and report generation.
- Documentation that makes privacy, model downloads, and limitations clearer.
- Platform-specific recording fixes.

Please avoid broad rewrites unless they are tied to a concrete bug, benchmark
result, or design issue.

## Tests And Evidence

For code changes, include at least one of:

- A unit test.
- A benchmark result.
- A manual reproduction note.
- A reason tests are not practical for the change.

For scoring changes, prefer before/after benchmark output over subjective
descriptions.

## Data And Licenses

Do not commit third-party speech datasets, private recordings, generated model
weights, or benchmark caches. Add downloader scripts or adapters instead.

Non-commercial datasets can be useful for research checks, but they should not
be used for default public release claims.

## Pull Request Checklist

- Tests pass.
- `ruff check .` passes.
- `ruff format --check .` passes.
- Public docs do not overstate pronunciation-scoring accuracy.
- New dependencies are justified and optional when possible.
- No private audio, transcripts, credentials, or local config files are included.

