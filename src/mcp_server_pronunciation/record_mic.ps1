<#
.SYNOPSIS
    Records audio from the Windows microphone using MCI (winmm.dll).
    Designed to be called from WSL2 where WSLg mic forwarding doesn't work.

.DESCRIPTION
    Uses the MCI (Media Control Interface) via winmm.dll P/Invoke to record
    from the default Windows microphone. Outputs 16kHz/16bit/mono PCM WAV,
    which is the ideal format for Whisper speech recognition.

.PARAMETER Duration
    Recording duration in seconds. Default: 5

.PARAMETER OutputPath
    Output WAV file path (Windows path). Default: C:\temp\recording.wav

.PARAMETER SampleRate
    Sample rate in Hz. Default: 16000 (optimal for Whisper)

.EXAMPLE
    # From WSL:
    powershell.exe -File /mnt/c/path/to/record_mic.ps1 -Duration 10 -OutputPath "C:\temp\my_recording.wav"

    # From Windows:
    .\record_mic.ps1 -Duration 10
#>

param(
    [int]$Duration = 5,
    [string]$OutputPath = "C:\temp\recording.wav",
    [int]$SampleRate = 16000
)

# --- P/Invoke definitions ---
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public class MciAudio {
    [DllImport("winmm.dll", CharSet = CharSet.Unicode)]
    public static extern int mciSendString(
        string command, StringBuilder returnValue,
        int returnLength, IntPtr callback);

    [DllImport("winmm.dll")]
    public static extern int mciGetErrorString(
        int errorCode, StringBuilder errorText, int errorTextSize);

    public static string Send(string command) {
        var sb = new StringBuilder(256);
        int result = mciSendString(command, sb, 256, IntPtr.Zero);
        if (result != 0) {
            var err = new StringBuilder(256);
            mciGetErrorString(result, err, 256);
            throw new Exception(string.Format(
                "MCI error {0}: {1} (command: {2})", result, err, command));
        }
        return sb.ToString();
    }
}
"@

# --- Ensure output directory exists ---
$outDir = Split-Path $OutputPath -Parent
if ($outDir -and !(Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

# --- Record ---
$alias = "wslmic_" + [System.Diagnostics.Process]::GetCurrentProcess().Id

try {
    [MciAudio]::Send("open new type waveaudio alias $alias")
    [MciAudio]::Send("set $alias bitspersample 16")
    [MciAudio]::Send("set $alias channels 1")
    [MciAudio]::Send("set $alias samplespersec $SampleRate")
    [MciAudio]::Send("set $alias alignment 2")

    Write-Host "Recording for $Duration seconds... (Ctrl+C to stop early)" -ForegroundColor Yellow
    [MciAudio]::Send("record $alias")

    Start-Sleep -Seconds $Duration

    [MciAudio]::Send("stop $alias")
    [MciAudio]::Send("save $alias `"$OutputPath`"")
    [MciAudio]::Send("close $alias")

    $fileInfo = Get-Item $OutputPath
    Write-Host "Saved: $OutputPath ($([math]::Round($fileInfo.Length / 1024, 1)) KB)" -ForegroundColor Green
}
catch {
    Write-Error $_.Exception.Message
    try { [MciAudio]::Send("close $alias") } catch {}
    exit 1
}
