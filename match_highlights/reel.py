"""From an excitement score to highlight clips and a rendered reel."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from . import FRAME_HOP_S


@dataclass
class Clip:
    start: float
    end: float
    peak: float
    score: float

    @property
    def length(self) -> float:
        return self.end - self.start


def pick_peaks(times: np.ndarray, score: np.ndarray, min_gap_s: float = 30.0) -> list[tuple[float, float]]:
    """Local maxima of the score at least `min_gap_s` apart, strongest first."""
    hop = times[1] - times[0] if len(times) > 1 else FRAME_HOP_S
    idx, _ = find_peaks(score, distance=max(1, int(min_gap_s / hop)))
    order = idx[np.argsort(score[idx])[::-1]]
    return [(float(times[i]), float(score[i])) for i in order]


def merge(clips: list[Clip]) -> list[Clip]:
    clips = sorted(clips, key=lambda c: c.start)
    out: list[Clip] = []
    for c in clips:
        if out and c.start <= out[-1].end:
            p = out[-1]
            out[-1] = Clip(p.start, max(p.end, c.end), p.peak, max(p.score, c.score))
        else:
            out.append(c)
    return out


def select(times: np.ndarray, score: np.ndarray, budget_s: float, pre_s: float = 12.0,
           post_s: float = 8.0, min_gap_s: float = 30.0) -> list[Clip]:
    """Greedy: add the strongest peaks as [peak - pre, peak + post] clips until the
    reel reaches `budget_s`. Pre-roll is longer than post-roll because the score
    peaks on the reaction, which comes after the action."""
    duration = float(times[-1])
    chosen: list[Clip] = []
    for t, s in pick_peaks(times, score, min_gap_s):
        cand = Clip(max(0.0, t - pre_s), min(duration, t + post_s), t, s)
        trial = merge(chosen + [cand])
        if sum(c.length for c in trial) > budget_s:
            continue
        chosen = trial
        if budget_s - sum(c.length for c in chosen) < pre_s + post_s:
            break
    return sorted(chosen, key=lambda c: c.start)


def render(media: str, clips: list[Clip], out_path: str) -> None:
    """Cut each clip with ffmpeg (re-encoded for frame-accurate cuts) and concatenate."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, c in enumerate(clips):
            part = Path(tmp) / f"{i:03d}.mp4"
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{c.start:.2f}",
                "-i", media, "-t", f"{c.length:.2f}", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "20", "-c:a", "aac", str(part),
            ], check=True)
            parts.append(part)
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0",
            "-i", str(listing), "-c", "copy", str(out),
        ], check=True)
