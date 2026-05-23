"""Cross-platform audio recording for pronunciation practice.

Supports:
- macOS / native Linux: sounddevice with VAD auto-stop
- WSL2: Windows PowerShell MCI (winmm.dll) — because WSLg doesn't
  forward microphone audio through its RDP channel.
"""

from __future__ import annotations

import functools
import logging
import os
import platform
import subprocess
import tempfile
import threading
import uuid
import wave
from pathlib import Path

from .config import input_device_value, vad_sensitivity_value, vad_silence_duration_seconds

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # 16kHz — optimal for Whisper
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit


def _portaudio_install_hint() -> str:
    """Return a platform-specific install hint for missing PortAudio."""
    system = platform.system()
    if system == "Linux":
        return (
            "PortAudio shared library not found.\n"
            "Install it:\n"
            "  Debian/Ubuntu:  sudo apt-get install libportaudio2\n"
            "  Fedora/RHEL:    sudo dnf install portaudio\n"
            "  Arch:           sudo pacman -S portaudio\n"
            "On PipeWire-only systems you may also need: sudo apt-get install pipewire-alsa"
        )
    if system == "Darwin":
        return (
            "PortAudio should ship inside the sounddevice wheel on macOS — "
            "this is unexpected. Try reinstalling sounddevice:\n"
            "  uv pip install --force-reinstall sounddevice"
        )
    if system == "Windows":
        return (
            "PortAudio should ship inside the sounddevice wheel on Windows — "
            "this is unexpected. Try reinstalling sounddevice:\n"
            "  uv pip install --force-reinstall sounddevice"
        )
    return "PortAudio library not found. Install it for your platform."


def _import_sounddevice():
    """Import sounddevice with a helpful error if PortAudio is missing."""
    try:
        import sounddevice as sd
    except OSError as e:
        msg = str(e)
        if "portaudio" in msg.lower() or "PortAudio" in msg:
            raise RuntimeError(f"{_portaudio_install_hint()}\n\nOriginal error: {e}") from e
        raise
    except ImportError as e:
        raise RuntimeError(
            "sounddevice is not installed. Reinstall this package:\n"
            "  uv tool install --reinstall mcp-server-pronunciation"
        ) from e
    return sd


def _format_recording_failure(exc: BaseException) -> str:
    """Return an actionable recording error for end users."""
    message = str(exc).strip() or exc.__class__.__name__
    lower = message.lower()
    hints = ["Run `mcp-server-pronunciation doctor` to check microphone access."]

    if "no default input" in lower or "invalid input device" in lower:
        hints.append("Set a system default microphone, then run the `check_mic` tool.")
    if "permission" in lower or "access" in lower or "unanticipated host error" in lower:
        hints.append(
            "Check OS microphone privacy permissions for the app launching the MCP server."
        )
    if "portaudio" in lower:
        hints.append("Install or repair PortAudio. On Linux, install `libportaudio2`.")

    hint_text = "\n".join(f"- {hint}" for hint in dict.fromkeys(hints))
    return f"Recording failed via sounddevice: {message}\n{hint_text}"


# Bundled PowerShell script for WSL recording
_PS1_SCRIPT = Path(__file__).parent / "record_mic.ps1"

# RMS threshold for voice activity detection
_VAD_SPEECH_THRESHOLD = 500  # Start listening after RMS exceeds this
_VAD_SILENCE_THRESHOLD = 200  # Stop after RMS drops below this
_VAD_SILENCE_DURATION = 1.5  # Seconds of silence before auto-stop
_VAD_MIN_SPEECH = 0.5  # Minimum speech duration before allowing auto-stop
_VAD_LEVELS: dict[str, tuple[int, int]] = {
    "low": (700, 300),
    "normal": (_VAD_SPEECH_THRESHOLD, _VAD_SILENCE_THRESHOLD),
    "high": (300, 120),
}


def _configured_input_device() -> int | str | None:
    value = input_device_value()
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _vad_thresholds() -> tuple[int, int]:
    return _VAD_LEVELS[vad_sensitivity_value()]


@functools.lru_cache(maxsize=1)
def _is_wsl() -> bool:
    """Check if running inside WSL (cached)."""
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


@functools.lru_cache(maxsize=1)
def _ps1_win_path() -> str:
    """Get Windows path for the PS1 script (cached)."""
    return subprocess.run(
        ["wslpath", "-w", str(_PS1_SCRIPT)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@functools.lru_cache(maxsize=1)
def _windows_temp_dir() -> str:
    """Return the Windows temp directory used by PowerShell recording."""
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "[System.IO.Path]::GetTempPath()",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "C:\\Windows\\Temp"
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().rstrip("\\/")
    return "C:\\Windows\\Temp"


def record_audio(duration: float, output_path: Path | None = None) -> Path:
    """Record audio from the microphone.

    On native platforms, uses VAD (voice activity detection) to auto-stop
    recording after the speaker finishes — no need to wait the full duration.

    Args:
        duration: Maximum recording duration in seconds.
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
        _record_sounddevice_vad(duration, output_path)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Recording failed — no audio captured. Check microphone access.")

    return output_path


def _rms(data) -> float:
    """Calculate RMS (root mean square) of int16 audio data."""
    import numpy as np

    samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples**2)))


def _record_sounddevice_vad(duration: float, output_path: Path) -> None:
    """Record with voice activity detection — auto-stops after speech ends."""
    sd = _import_sounddevice()

    chunks: list[bytes] = []
    speech_detected = False
    silence_start: float | None = None
    stop_event = threading.Event()
    speech_threshold, silence_threshold = _vad_thresholds()
    silence_duration = vad_silence_duration_seconds()
    input_device = _configured_input_device()

    import time

    record_start = time.monotonic()

    def callback(indata, frames, time_info, status):
        nonlocal speech_detected, silence_start

        if stop_event.is_set():
            raise sd.CallbackAbort

        raw = indata.tobytes()
        chunks.append(raw)
        level = _rms(raw)

        elapsed = time.monotonic() - record_start

        if not speech_detected and level > speech_threshold:
            speech_detected = True
            silence_start = None
            logger.debug("Speech detected at %.1fs (RMS=%.0f)", elapsed, level)

        if speech_detected:
            if level < silence_threshold:
                if silence_start is None:
                    silence_start = elapsed
                elif elapsed - silence_start > silence_duration and elapsed > _VAD_MIN_SPEECH + 1.0:
                    logger.debug("Auto-stop: silence for %.1fs", elapsed - silence_start)
                    stop_event.set()
                    raise sd.CallbackAbort
            else:
                silence_start = None

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
            callback=callback,
            device=input_device,
        ):
            # Wait for auto-stop or max duration
            stop_event.wait(timeout=duration)
    except sd.CallbackAbort:
        pass
    except Exception as e:
        raise RuntimeError(_format_recording_failure(e)) from e

    if not chunks:
        raise RuntimeError(
            "No audio data captured.\n"
            "- Speak after the tool starts recording.\n"
            "- Run `mcp-server-pronunciation doctor` if the microphone appears unavailable."
        )

    audio_data = b"".join(chunks)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data)


def _record_wsl(duration: float, output_path: Path) -> None:
    """Record via Windows PowerShell MCI (winmm.dll)."""
    if not _PS1_SCRIPT.exists():
        raise RuntimeError(f"PowerShell recording script not found: {_PS1_SCRIPT}")

    ps1_win = _ps1_win_path()
    win_temp = f"{_windows_temp_dir()}\\pronun_{os.getpid()}_{uuid.uuid4().hex}.wav"

    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                ps1_win,
                "-Duration",
                str(int(duration)),
                "-OutputPath",
                win_temp,
                "-SampleRate",
                str(SAMPLE_RATE),
            ],
            capture_output=True,
            text=True,
            timeout=duration + 30,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            "PowerShell recording timed out. Check Windows microphone access, then run "
            "`mcp-server-pronunciation doctor` inside WSL."
        ) from e
    except OSError as e:
        raise RuntimeError(
            "PowerShell recording could not start. Confirm `powershell.exe` is available "
            "from WSL and run `mcp-server-pronunciation doctor`."
        ) from e

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"PowerShell recording failed: {detail}")

    # Convert Windows path to WSL and copy
    win_temp_unix = win_temp.replace("\\", "/")
    win_temp_unix = "/mnt/" + win_temp_unix[0].lower() + win_temp_unix[2:]
    win_temp_wsl = Path(win_temp_unix)

    import shutil

    shutil.copy2(win_temp_wsl, output_path)
    try:
        win_temp_wsl.unlink()
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
        lines.append("Platform: Native (recording via sounddevice with VAD auto-stop)")
        input_device = input_device_value()
        sensitivity = vad_sensitivity_value()
        speech_threshold, silence_threshold = _vad_thresholds()
        silence_duration = vad_silence_duration_seconds()
        lines.append(f"Configured input device: {input_device or '(system default)'}")
        lines.append(
            "VAD: "
            f"sensitivity={sensitivity}, "
            f"speech_rms>{speech_threshold}, "
            f"silence_rms<{silence_threshold}, "
            f"stop_after={silence_duration:.1f}s"
        )
        try:
            sd = _import_sounddevice()
            default_input = sd.query_devices(kind="input")
            lines.append(f"Default input: {default_input['name']}")
            devices = list(sd.query_devices())
            input_devices = [
                (idx, d) for idx, d in enumerate(devices) if d["max_input_channels"] > 0
            ]
            lines.append(f"Input devices ({len(input_devices)}):")
            for idx, d in input_devices:
                lines.append(
                    f"  - [{idx}] {d['name']} "
                    f"({d['max_input_channels']}ch, {d['default_samplerate']:.0f}Hz)"
                )
        except RuntimeError as e:
            lines.append(f"ERROR: {e}")
        except Exception as e:
            lines.append(f"Error querying devices: {e}")

    return "\n".join(lines)
