"""Tests for public release metadata consistency."""

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


def test_release_version_has_matching_stability_metadata():
    with (ROOT / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    version = pyproject["project"]["version"]
    if "b" not in version:
        return

    classifiers = pyproject["project"]["classifiers"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    disclaimer = (ROOT / "DISCLAIMER.md").read_text(encoding="utf-8").lower()
    release_notes = (ROOT / "docs" / "releases" / f"v{version}.md").read_text(encoding="utf-8")

    if "b" in version:
        assert "Development Status :: 4 - Beta" in classifiers
        assert "public beta notice" in readme
        assert "early public beta" in disclaimer
        assert "runtime errors" in disclaimer
        assert "public beta" in release_notes.lower()
    else:
        assert "Development Status :: 5 - Production/Stable" in classifiers
        assert "accuracy and safety notice" in readme
        assert "public beta notice" not in readme
        assert "--pre" not in readme
        assert "early public beta" not in disclaimer
        assert "runtime errors" in disclaimer
        assert "stable release" in release_notes.lower()


def test_glama_metadata_declares_maintainer():
    metadata = json.loads((ROOT / "glama.json").read_text(encoding="utf-8"))

    assert metadata["$schema"] == "https://glama.ai/mcp/schemas/server.json"
    assert metadata["maintainers"] == ["JuhongPark"]


def test_readme_exposes_glama_badge():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "https://glama.ai/mcp/servers/JuhongPark/mcp-server-pronunciation" in readme
    assert "/badges/score.svg" in readme


def test_server_metadata_documents_registry_inspection_preload_toggle():
    server = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    env_names = {
        env["name"]
        for package in server["packages"]
        for env in package.get("environmentVariables", [])
    }

    assert "MCP_PRONUNCIATION_PRELOAD" in env_names
    assert "MCP_PRONUNCIATION_AUDIO_RETENTION" in env_names
    assert "MCP_PRONUNCIATION_INPUT_DEVICE" in env_names
    assert "MCP_PRONUNCIATION_VAD_SENSITIVITY" in env_names
    assert "MCP_PRONUNCIATION_SILENCE_DURATION" in env_names
    assert "TORCH_HOME" in env_names


def test_dockerfile_disables_model_preload_for_directory_inspection():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "MCP_PRONUNCIATION_PRELOAD=0" in dockerfile
    assert "libportaudio2" in dockerfile
    assert "--no-deps" in dockerfile
    assert '"mcp[cli]>=1.2,<2"' in dockerfile
    assert 'ENTRYPOINT ["mcp-server-pronunciation"]' in dockerfile
    assert 'CMD ["serve"]' in dockerfile
