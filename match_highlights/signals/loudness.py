"""Loudness signal: the classic 'crowd roar' heuristic, kept as the baseline."""

from __future__ import annotations

import numpy as np
import librosa

from ..audio import to_grid


def robust_z(x: np.ndarray) -> np.ndarray:
    """Z-score using median/MAD so a few huge peaks don't flatten the rest."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-9
    return (x - med) / mad


def compute(audio: np.ndarray, sr: int, grid: np.ndarray) -> dict[str, np.ndarray]:
    hop = 512
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms + 1e-10)
    t = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=hop)
    db = to_grid(db, t, grid)

    loud = robust_z(db)
    # Rise: how much louder this moment is than the previous 10 s. A goal is a
    # sudden jump, while a loud-but-steady stadium is not.
    hop_s = grid[1] - grid[0] if len(grid) > 1 else 0.5
    k = max(1, int(round(10.0 / hop_s)))
    pad = np.concatenate([np.full(k, db[0]), db])
    trailing = np.convolve(pad, np.ones(k) / k, mode="valid")[: len(db)]
    rise = robust_z(db - trailing)
    return {"loud_z": loud, "loud_rise": rise}
