"""Command-line subcommands for diagnostics and setup.

Exposes `doctor` (preflight check) and `pull-model` (pre-download Whisper
weights) alongside the MCP server entry point. These run outside the MCP
stdio loop — they're meant for users to run manually once before wiring the
server into Claude Desktop / Claude Code.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path


def _print_header(title: str) -> None:
    print(f"\n== {title} ==", flush=True)


def _check(label: str, ok: bool, detail: str = "") -> None:
    mark = "OK  " if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f" — {detail}"
    print(line, flush=True)


def _warn(label: str, detail: str = "") -> None:
    line = f"  [WARN] {label}"
    if detail:
        line += f" — {detail}"
    print(line, flush=True)


def doctor() -> int:
    """Run a preflight check. Returns 0 if everything looks good, 1 otherwise."""
    all_ok = True

    _print_header("System")
    print(f"  Python:     {sys.version.split()[0]}", flush=True)
    print(
        f"  Platform:   {platform.system()} {platform.release()} ({platform.machine()})", flush=True
    )
    print(f"  Executable: {sys.executable}", flush=True)

    _print_header("Microphone / PortAudio")
    try:
        from .recorder import _import_sounddevice, _is_wsl

        if _is_wsl():
            _check(
                "Running under WSL2",
                True,
                "recording goes through Windows PowerShell MCI, not PortAudio",
            )
            ps1 = Path(__file__).parent / "record_mic.ps1"
            _check("PowerShell recording script", ps1.exists(), str(ps1))
            if not ps1.exists():
                all_ok = False
        else:
            try:
                sd = _import_sounddevice()
            except RuntimeError as e:
                _check("sounddevice import", False, str(e).splitlines()[0])
                print(f"\n{e}\n", flush=True)
                all_ok = False
            else:
                _check("sounddevice import", True)
                try:
                    devices = sd.query_devices()
                    inputs = [d for d in devices if d["max_input_channels"] > 0]
                    _check(
                        "Input devices",
                        bool(inputs),
                        f"{len(inputs)} found",
                    )
                    if inputs:
                        try:
                            default = sd.query_devices(kind="input")
                            print(f"        default: {default['name']}", flush=True)
                        except Exception as exc:
                            print(f"        could not query default: {exc}", flush=True)
                    else:
                        all_ok = False
                except Exception as e:
                    _check("Query input devices", False, str(e))
                    all_ok = False
    except Exception as e:
        _check("Recorder module import", False, str(e))
        all_ok = False

    _print_header("Whisper model")
    model_size = os.environ.get("MCP_PRONUNCIATION_MODEL", "base.en")
    print(f"  Target model: {model_size}", flush=True)
    hf_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if hf_cache:
        print(f"  HF cache:     {hf_cache}", flush=True)
    else:
        default_cache = Path.home() / ".cache" / "huggingface" / "hub"
        print(f"  HF cache:     {default_cache} (default)", flush=True)

    try:
        try:
            from faster_whisper import download_model  # noqa: F401
        except Exception as e:
            _check("faster-whisper importable", False, str(e))
            all_ok = False
        else:
            _check("faster-whisper importable", True)

        try:
            from huggingface_hub import try_to_load_from_cache

            repo = f"Systran/faster-whisper-{model_size}"
            cached = try_to_load_from_cache(repo, "model.bin")
            if cached is not None and cached is not False:
                _check("Model cached locally", True, f"{cached}")
            else:
                _warn(
                    "Model not yet downloaded",
                    f"first call will download it. Pre-warm with: mcp-server-pronunciation pull-model {model_size}",
                )
        except Exception as e:
            _warn("Cache probe unavailable", str(e))
    except Exception as e:
        _check("Model check", False, str(e))
        all_ok = False

    _print_header("Pronunciation resources")
    try:
        import cmudict

        entries = len(cmudict.dict())
        _check("CMUdict importable", entries > 0, f"{entries:,} entries")
        if entries == 0:
            all_ok = False
    except Exception as e:
        _check("CMUdict importable", False, str(e))
        all_ok = False

    try:
        from g2p_en import G2p  # noqa: F401

        _check("g2p_en importable", True)
    except Exception as e:
        _check("g2p_en importable", False, str(e))
        all_ok = False

    try:
        import nltk

        nltk_data = os.environ.get("NLTK_DATA") or str(Path.home() / "nltk_data")
        print(f"  NLTK data:   {nltk_data}", flush=True)
        for label, path, package in [
            (
                "NLTK averaged perceptron tagger",
                "taggers/averaged_perceptron_tagger_eng",
                "averaged_perceptron_tagger_eng",
            ),
            ("NLTK CMUdict corpus", "corpora/cmudict", "cmudict"),
        ]:
            try:
                found = nltk.data.find(path)
            except LookupError:
                _warn(
                    label,
                    f"not found. OOV pronunciation fallback may download `{package}` on first use",
                )
            else:
                _check(label, True, str(found))
    except Exception as e:
        _warn("NLTK resource probe unavailable", str(e))

    _print_header("Optional forced alignment")
    torch_home = os.environ.get("TORCH_HOME") or str(Path.home() / ".cache" / "torch")
    print(f"  Torch cache: {torch_home}", flush=True)
    try:
        import torch

        _check("torch importable", True, getattr(torch, "__version__", "unknown"))
    except Exception as e:
        _warn("torch not installed", f"{e}. Install with: mcp-server-pronunciation[phoneme]")

    try:
        import torchaudio

        _check("torchaudio importable", True, getattr(torchaudio, "__version__", "unknown"))
    except Exception as e:
        _warn("torchaudio not installed", f"{e}. Install with: mcp-server-pronunciation[phoneme]")

    _print_header("Disk space")
    try:
        usage = shutil.disk_usage(Path.home())
        free_gb = usage.free / (1024**3)
        _check("Free space in $HOME", free_gb >= 2.0, f"{free_gb:.1f} GB free")
        if free_gb < 2.0:
            all_ok = False
    except Exception as e:
        _check("Disk space probe", False, str(e))

    print("", flush=True)
    if all_ok:
        print("All checks passed. Ready to run `mcp-server-pronunciation`.", flush=True)
        return 0
    print("Some checks failed. See messages above.", flush=True)
    return 1


def pull_model(size: str | None = None) -> int:
    """Pre-download a Whisper model so the first MCP call is instant."""
    target = size or os.environ.get("MCP_PRONUNCIATION_MODEL", "base.en")
    print(f"Downloading Whisper model: {target}", flush=True)
    print("(MIT-licensed weights from openai/whisper via Systran/faster-whisper on HF Hub)")
    try:
        from faster_whisper import download_model
    except Exception as e:
        print(f"ERROR: could not import faster-whisper: {e}", file=sys.stderr)
        return 1

    try:
        path = download_model(target)
    except Exception as e:
        print(f"ERROR: download failed: {e}", file=sys.stderr)
        return 1

    print(f"Downloaded to: {path}", flush=True)
    print("Done.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI dispatcher.

    Usage:
      mcp-server-pronunciation                 — run the MCP server (default)
      mcp-server-pronunciation doctor          — preflight check
      mcp-server-pronunciation pull-model [X]  — pre-download Whisper model
    """
    parser = argparse.ArgumentParser(
        prog="mcp-server-pronunciation",
        description="MCP server for voice conversation with Claude + English feedback.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="Run preflight checks (mic, PortAudio, model).")

    pull = sub.add_parser("pull-model", help="Pre-download the Whisper model.")
    pull.add_argument(
        "size",
        nargs="?",
        default=None,
        help="Model size (default: $MCP_PRONUNCIATION_MODEL or base.en)",
    )

    sub.add_parser("serve", help="Run the MCP server (default).")

    args = parser.parse_args(argv)

    if args.command == "doctor":
        return doctor()
    if args.command == "pull-model":
        return pull_model(args.size)

    # Default: run the MCP server (also matches `serve`).
    from .server import run

    run()
    return 0
