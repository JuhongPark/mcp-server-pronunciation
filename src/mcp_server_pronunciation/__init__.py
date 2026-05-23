"""MCP server for voice conversation with MCP assistants + English feedback.

Nothing heavy is imported at package init — `cli_main` and `run` are both
lazy so that running `mcp-server-pronunciation doctor` doesn't trigger the
Whisper pre-load thread.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mcp-server-pronunciation")
except PackageNotFoundError:  # pragma: no cover - editable source fallback
    __version__ = "0.0.0+local"

__all__ = ["run", "cli_main", "__version__"]


def cli_main(*args, **kwargs):
    from .cli import main

    return main(*args, **kwargs)


def run(*args, **kwargs):
    from .server import run as _run

    return _run(*args, **kwargs)
