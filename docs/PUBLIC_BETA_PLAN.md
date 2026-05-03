# Public Beta Release Plan (Historical)

This historical document describes how this project was prepared and released
as a public beta before the stable `0.3.0` line. It is retained as release
history. Current release steps live in [RELEASE.md](RELEASE.md) and current
publication status lives in [PUBLICATION.md](PUBLICATION.md).

The beta goal was to make the project easy to try while making the risk profile
clear: this was an early local pronunciation coach, not a stable assessment
product.

## Research Summary

The public beta used standard packaging and release signals:

- Python package versions should follow PEP 440. A beta pre-release can use
  a version such as `0.3.0b3`.
- PyPI Trove classifiers marked maturity with
  `Development Status :: 4 - Beta`.
- GitHub Releases were marked as pre-releases when the release was not
  ready for production use.
- MCP Registry metadata kept the server version and package version
  aligned for local servers.
- Release notes and README installation instructions stated that beta
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

This wording was used when the project was presented to beta users:

> Public beta notice: this project is an early beta and is still under active
> development. It may contain bugs, runtime errors, inaccurate transcripts,
> inaccurate pronunciation feedback, or platform-specific recording issues. Use
> it for experimentation and language-learning practice only, and review outputs
> carefully before relying on them.

The short form for release notes is:

> Early public beta. Expect bugs, errors, and incomplete behavior. Please use
> with caution and report reproducible issues.

## Release Surface

The beta notice appeared in:

- README near the top, before installation.
- `DISCLAIMER.md` as a dedicated public-facing disclaimer.
- `CHANGELOG.md` for the beta release entry.
- GitHub Release notes for the beta tag.
- Release checklist documentation.

## Versioning

The beta line used `0.3.0b3` because the unreleased work was already staged as
the `0.3.0` line. These fields were kept aligned:

- `pyproject.toml` project version.
- `server.json` top-level version.
- `server.json` PyPI package version.
- `CHANGELOG.md` release heading.
- Git tag: `v0.3.0b3`.

The final stable release of the same feature line uses `0.3.0`.

## Release Steps

1. Added and reviewed the public beta disclaimer.
2. Updated package metadata to the beta version and beta classifier.
3. Updated release docs with beta-specific instructions.
4. Reviewed [PUBLICATION.md](PUBLICATION.md) and confirmed registration
   prerequisites.
5. Ran formatting, lint, and tests.
6. Scanned changed files for secrets, private data, bundled datasets, or
   license-incompatible content.
7. Pushed the preparation commits to `main`.
8. Configured the PyPI pending Trusted Publisher from the maintainer's PyPI
   account.
9. Created a pre-release tag and GitHub Release:

```bash
git tag v0.3.0b3
git push origin v0.3.0b3
gh release create v0.3.0b3 \
  --title "v0.3.0b3 (public beta)" \
  --notes-file docs/releases/v0.3.0b3.md \
  --prerelease
```

The tag push triggered the release workflow. That workflow published to PyPI
first and then published `server.json` to the MCP Registry.

## Acceptance Criteria

- README clearly labelled the project as public beta.
- Dedicated disclaimer existed and was linked from README.
- Package and MCP metadata used the same beta version.
- Release checklist included beta release steps.
- Release workflow validated tag and metadata before publishing.
- Local checks passed:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest -q
```
