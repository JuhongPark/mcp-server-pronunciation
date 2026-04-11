"""Cross-platform audio recording for pronunciation practice.

Supports:
- macOS / native Linux: sounddevice + soundfile
- WSL2: Windows PowerShell MCI (winmm.dll) — because WSLg doesn't
  forward microphone audio through its RDP channel.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

SAMPLE_RATE = 16000  # 16kHz — optimal for Whisper
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit

# Bundled PowerShell script for WSL recording
_PS1_SCRIPT = Path(__file__).parent / "record_mic.ps1"


def _is_wsl() -> bool:
    """Check if running inside WSL."""
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def record_audio(duration: float, output_path: Path | None = None) -> Path:
    """Record audio from the microphone.

    Automatically selects the best recording method for the platform:
    - WSL2: PowerShell MCI via winmm.dll
    - macOS/Linux: sounddevice

    Args:
        duration: Recording duration in seconds.
        output_path: Where to save the WAV file. Uses a temp file if None.

    Returns:
        Path to the saved 16kHz mono WAV file.
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="pronun_")
        output_path = Path(tmp.name)
        tmp.close()

    if _is_wsl():
        _record_wsl(duration, output_path)
    else:
        _record_sounddevice(duration, output_path)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Recording failed — no audio captured. Check microphone access.")

    return output_path


def _record_sounddevice(duration: float, output_path: Path) -> None:
    """Record via sounddevice (macOS, native Linux, native Windows)."""
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        raise RuntimeError(
            "sounddevice is required for recording on this platform. "
            "Install it with: pip install sounddevice numpy"
        )

    frames = int(duration * SAMPLE_RATE)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16")
    sd.wait()

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def _record_wsl(duration: float, output_path: Path) -> None:
    """Record via Windows PowerShell MCI (winmm.dll).

    WSLg's PulseAudio RDPSource doesn't forward actual microphone audio,
    so we record on the Windows side and copy the file back.
    """
    ps1_path = _PS1_SCRIPT
    if not ps1_path.exists():
        raise RuntimeError(f"PowerShell recording script not found: {ps1_path}")

    # Convert .ps1 path to Windows path
    ps1_win = subprocess.run(
        ["wslpath", "-w", str(ps1_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    # Windows temp path for the recording
    win_temp = f"C:\\temp\\pronun_{os.getpid()}.wav"

    # Record via PowerShell
    result = subprocess.run(
        [
            "powershell.exe", "-ExecutionPolicy", "Bypass",
            "-File", ps1_win,
            "-Duration", str(int(duration)),
            "-OutputPath", win_temp,
            "-SampleRate", str(SAMPLE_RATE),
        ],
        capture_output=True, text=True, timeout=duration + 30,
    )

    if result.returncode != 0:
        raise RuntimeError(f"PowerShell recording failed: {result.stderr}")

    # Convert Windows path to WSL and copy
    win_temp_wsl = subprocess.run(
        ["wslpath", win_temp.replace("\\", "/")],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    import shutil
    shutil.copy2(win_temp_wsl, output_path)
    try:
        os.unlink(win_temp_wsl)
    except OSError:
        pass


def check_audio_devices() -> str:
    """Return info about available audio devices."""
    lines = []

    if _is_wsl():
        lines.append("Platform: WSL2 (recording via Windows PowerShell MCI)")
        lines.append(f"Recording script: {_PS1_SCRIPT}")
        lines.append(f"Script found: {_PS1_SCRIPT.exists()}")
    else:
        lines.append("Platform: Native (recording via sounddevice)")
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            default_input = sd.query_devices(kind="input")
            lines.append(f"Default input: {default_input['name']}")
            input_devices = [d for d in devices if d["max_input_channels"] > 0]
            lines.append(f"Input devices ({len(input_devices)}):")
            for d in input_devices:
                lines.append(f"  - {d['name']} ({d['max_input_channels']}ch, {d['default_samplerate']:.0f}Hz)")
        except ImportError:
            lines.append("WARNING: sounddevice not installed. Run: pip install sounddevice")
        except Exception as e:
            lines.append(f"Error querying devices: {e}")

    return "\n".join(lines)
