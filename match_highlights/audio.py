"""Audio loading and time-grid helpers."""

from __future__ import annotations

import subprocess

import numpy as np

from . import FRAME_HOP_S


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Decode the audio track of any media file to mono float32 at `sr` Hz.

    Uses ffmpeg directly: it handles every container a match recording comes
    in and is much faster than decoding through moviepy.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", path,
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    if audio.size == 0:
        raise ValueError(f"no audio track decoded from {path}")
    return audio


def frame_times(duration_s: float, hop_s: float = FRAME_HOP_S) -> np.ndarray:
    """Centre times of the shared analysis frames covering `duration_s`."""
    n = int(np.floor(duration_s / hop_s))
    return (np.arange(n) + 0.5) * hop_s


def to_grid(values: np.ndarray, src_times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linearly resample a signal sampled at `src_times` onto `grid`."""
    if len(values) == 0:
        return np.zeros_like(grid)
    return np.interp(grid, src_times, values, left=values[0], right=values[-1])
