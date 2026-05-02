# Public Beta Release Plan

This document describes how this project should be prepared and released as a
public beta. The goal is to make the project easy to try while making the risk
profile clear: this is an early local pronunciation coach, not a stable
assessment product.

## Research Summary

The public beta should use standard packaging and release signals:

- Python package versions should follow PEP 440. A beta pre-release should use
  a version such as `0.3.0b1`.
- PyPI Trove classifiers should mark maturity with
  `Development Status :: 4 - Beta`.
- GitHub Releases should be marked as pre-releases when the release is not
  ready for production use.
- MCP Registry metadata should keep the server version and package version
  aligned for local servers.
- Release notes and README installation instructions should state that beta
  releases may contain bugs, runtime errors, inaccurate feedback, and
  platform-specific recording failures.

References:

- Python Packaging User Guide, version specifiers:
  https://packaging.python.org/specifications/version-specifiers/
- PyPI Trove classifiers:
  https://pypi.org/classifiers/
- GitHub Docs, managing releases:
  https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
- MCP Registry versioning:
  https://modelcontextprotocol.io/registry/versioning

## Beta Disclaimer Text

Use this wording wherever the project is presented to new users:

> Public beta notice: this project is an early beta and is still under active
> development. It may contain bugs, runtime errors, inaccurate transcripts,
> inaccurate pronunciation feedback, or platform-specific recording issues. Use
> it for experimentation and language-learning practice only, and review outputs
> carefully before relying on them.

The short form for release notes is:

> Early public beta. Expect bugs, errors, and incomplete behavior. Please use
> with caution and report reproducible issues.

## Release Surface

The beta notice should appear in:

- README near the top, before installation.
- `DISCLAIMER.md` as a dedicated public-facing disclaimer.
- `CHANGELOG.md` for the beta release entry.
- GitHub Release notes for the beta tag.
- Release checklist documentation.

## Versioning

Use `0.3.0b1` for the first public beta because the current unreleased work is
already staged as the `0.3.0` line. Keep these fields aligned:

- `pyproject.toml` project version.
- `server.json` top-level version.
- `server.json` PyPI package version.
- `CHANGELOG.md` release heading.
- Git tag: `v0.3.0b1`.

For a final stable release of the same feature line, use `0.3.0`.

## Release Steps

1. Add and review the public beta disclaimer.
2. Update package metadata to the beta version and beta classifier.
3. Update release docs with beta-specific instructions.
4. Run formatting, lint, and tests.
5. Scan changed files for secrets, private data, bundled datasets, or
   license-incompatible content.
6. Push the preparation commits to `main`.
7. When ready to publish, create a pre-release tag and GitHub Release:

```bash
git tag v0.3.0b1
git push origin v0.3.0b1
gh release create v0.3.0b1 \
  --title "v0.3.0b1 (public beta)" \
  --notes-file docs/releases/v0.3.0b1.md \
  --prerelease
```

The tag push will trigger the PyPI release workflow. Do not push the tag until
the beta disclaimer, metadata, and validation checks are in place.

## Acceptance Criteria

- README clearly labels the project as public beta.
- Dedicated disclaimer exists and is linked from README.
- Package and MCP metadata use the same beta version.
- Release checklist includes beta release steps.
- Release workflow validates tag and metadata before publishing.
- Local checks pass:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest -q
```
