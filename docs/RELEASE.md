# Release Checklist

Use this checklist before publishing a package or registry update.

## Pre-Release

- Confirm the worktree is clean before starting release steps.
- Run tests and lint:

```bash
uv sync --extra dev
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

- Run optional checks when the forced-alignment path changed:

```bash
uv sync --extra dev --extra phoneme
uv run pytest -v
```

- Verify versions match across:
  - `pyproject.toml`
  - `server.json`
  - `CHANGELOG.md`
  - release notes

- Check that `CHANGELOG.md` has a dated entry for the release.
- Confirm `server.json` points at the intended package version.
- Run `mcp-server-pronunciation doctor` in a fresh environment when possible.
- Scan changed files for credentials, private data, and bundled datasets.

## Build

```bash
uv build --no-sources
```

Confirm the wheel contains `record_mic.ps1`.

## Publish

Publishing is handled by the GitHub Actions release workflow through PyPI
Trusted Publishers. Push a version tag only after the pre-release checks pass.

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

## Post-Release

- Confirm the PyPI page shows the new version.
- Confirm `uvx mcp-server-pronunciation doctor` works from a clean cache or a
  documented first-run state.
- If submitting to an MCP registry, validate and submit the matching
  `server.json`.
- Publish benchmark summaries only as reports. Do not upload third-party
  benchmark datasets to this repository.

