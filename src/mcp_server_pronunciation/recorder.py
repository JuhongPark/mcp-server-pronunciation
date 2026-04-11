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
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # 16kHz — optimal for Whisper
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit

# Bundled PowerShell script for WSL recording
_PS1_SCRIPT = Path(__file__).parent / "record_mic.ps1"

# RMS threshold for voice activity detection
_VAD_SPEECH_THRESHOLD = 500  # Start listening after RMS exceeds this
_VAD_SILENCE_THRESHOLD = 200  # Stop after RMS drops below this
_VAD_SILENCE_DURATION = 1.5  # Seconds of silence before auto-stop
_VAD_MIN_SPEECH = 0.5  # Minimum speech duration before allowing auto-stop


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
        capture_output=True, text=True, check=True,
    ).stdout.strip()


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
    return float(np.sqrt(np.mean(samples ** 2)))


def _record_sounddevice_vad(duration: float, output_path: Path) -> None:
    """Record with voice activity detection — auto-stops after speech ends."""
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError(
            "sounddevice is required for recording on this platform. "
            "Install it with: pip install sounddevice numpy"
        )

    chunks: list[bytes] = []
    speech_detected = False
    silence_start: float | None = None
    stop_event = threading.Event()

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

        if not speech_detected and level > _VAD_SPEECH_THRESHOLD:
            speech_detected = True
            silence_start = None
            logger.debug("Speech detected at %.1fs (RMS=%.0f)", elapsed, level)

        if speech_detected:
            if level < _VAD_SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = elapsed
                elif (elapsed - silence_start > _VAD_SILENCE_DURATION
                      and elapsed > _VAD_MIN_SPEECH + 1.0):
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
        ):
            # Wait for auto-stop or max duration
            stop_event.wait(timeout=duration)
    except sd.CallbackAbort:
        pass

    if not chunks:
        raise RuntimeError("No audio data captured.")

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
    win_temp = f"C:\\temp\\pronun_{os.getpid()}.wav"

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
        try:
            import sounddevice as sd
            default_input = sd.query_devices(kind="input")
            lines.append(f"Default input: {default_input['name']}")
            devices = sd.query_devices()
            input_devices = [d for d in devices if d["max_input_channels"] > 0]
            lines.append(f"Input devices ({len(input_devices)}):")
            for d in input_devices:
                lines.append(f"  - {d['name']} ({d['max_input_channels']}ch, {d['default_samplerate']:.0f}Hz)")
        except ImportError:
            lines.append("WARNING: sounddevice not installed. Run: pip install sounddevice")
        except Exception as e:
            lines.append(f"Error querying devices: {e}")

    return "\n".join(lines)
