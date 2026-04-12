"""MCP server for voice conversation with Claude + English language feedback.

Nothing heavy is imported at package init — `cli_main` and `run` are both
lazy so that running `mcp-server-pronunciation doctor` doesn't trigger the
Whisper pre-load thread.
"""

__all__ = ["run", "cli_main"]


def cli_main(*args, **kwargs):
    from .cli import main

    return main(*args, **kwargs)


def run(*args, **kwargs):
    from .server import run as _run

    return _run(*args, **kwargs)
