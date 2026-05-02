"""Tests for public beta metadata consistency."""

import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_package_and_server_versions_match():
    with (ROOT / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    server = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))

    project_version = pyproject["project"]["version"]
    assert server["version"] == project_version
    assert server["packages"][0]["version"] == project_version


def test_beta_version_has_public_beta_disclaimer():
    with (ROOT / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    version = pyproject["project"]["version"]
    if "b" not in version:
        return

    classifiers = pyproject["project"]["classifiers"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    disclaimer = (ROOT / "DISCLAIMER.md").read_text(encoding="utf-8").lower()
    release_notes = (ROOT / "docs" / "releases" / f"v{version}.md").read_text(encoding="utf-8")

    assert "Development Status :: 4 - Beta" in classifiers
    assert "public beta notice" in readme
    assert "early public beta" in disclaimer
    assert "runtime errors" in disclaimer
    assert "public beta" in release_notes.lower()
