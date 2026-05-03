# Release Checklist

Use this checklist before publishing a package or registry update.

## Pre-Release

- Confirm the worktree is clean before starting release steps.
- For public beta releases, confirm `README.md`, `DISCLAIMER.md`, and release
  notes clearly state that the project is an early beta and may contain bugs,
  runtime errors, inaccurate transcripts, inaccurate pronunciation feedback, and
  platform-specific recording issues.
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
- For a beta release, use a PEP 440 beta version such as `0.3.0b2` and a
  matching tag such as `v0.3.0b2`.
- Confirm the PyPI pending Trusted Publisher exists before the first PyPI
  publish. See [PUBLICATION.md](PUBLICATION.md).
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

For a public beta, publish a GitHub pre-release and use the beta release notes:

```bash
git tag v0.3.0b2
git push origin v0.3.0b2
gh release create v0.3.0b2 \
  --title "v0.3.0b2 (public beta)" \
  --notes-file docs/releases/v0.3.0b2.md \
  --prerelease
```

For a stable release:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

## Post-Release

- Confirm the PyPI page shows the new version.
- Confirm the MCP Registry search returns `io.github.JuhongPark/pronunciation`.
- Confirm `uvx mcp-server-pronunciation doctor` works from a clean cache or a
  documented first-run state.
- Publish benchmark summaries only as reports. Do not upload third-party
  benchmark datasets to this repository.
