"""Excitement curves vs. labelled goals, for the README and for sanity checks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [1, 99.9])
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def excitement_plot(times: np.ndarray, scores: dict[str, np.ndarray], goals: list[float],
                    clips, title: str, out: str) -> None:
    """One row per scorer; red lines = labelled goals, green bands = reel clips."""
    fig, axes = plt.subplots(len(scores), 1, figsize=(12, 1.8 * len(scores)), sharex=True)
    axes = np.atleast_1d(axes)
    minutes = times / 60
    for ax, (name, s) in zip(axes, scores.items()):
        ax.plot(minutes, _norm(s), lw=0.6, color="#1f3b8f")
        for g in goals:
            ax.axvline(g / 60, color="#d62728", lw=1.2, alpha=0.9)
        for c in clips.get(name, []):
            ax.axvspan(c.start / 60, c.end / 60, color="#2ca02c", alpha=0.25, lw=0)
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        ax.set_ylim(0, 1.05)
    axes[-1].set_xlabel("match time (minutes)")
    axes[0].set_title(f"{title}   (red: labelled goals, green: clips chosen for the reel)", fontsize=10)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
