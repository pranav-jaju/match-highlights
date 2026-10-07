"""Metrics for comparing scorers against labelled highlight segments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .reel import Clip, select


def load_segments(path: str) -> list[tuple[float, float]]:
    df = pd.read_csv(path)
    return list(zip(df["start"].astype(float), df["end"].astype(float)))


def frame_labels(times: np.ndarray, segments: list[tuple[float, float]]) -> np.ndarray:
    y = np.zeros(len(times), dtype=int)
    for s, e in segments:
        y[(times >= s) & (times < e)] = 1
    return y


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def reel_metrics(clips: list[Clip], segments: list[tuple[float, float]],
                 min_cover: float = 0.5) -> dict[str, float]:
    """How good is the actual reel?

    recall    = share of true highlights where the reel shows >= `min_cover` of it
    precision = share of reel seconds that fall inside a true highlight
    """
    hit = 0
    for s, e in segments:
        covered = sum(_overlap(c.start, c.end, s, e) for c in clips)
        hit += covered >= min_cover * (e - s)
    reel_len = sum(c.length for c in clips)
    inside = sum(_overlap(c.start, c.end, s, e) for c in clips for s, e in segments)
    recall = hit / len(segments) if segments else 0.0
    precision = inside / reel_len if reel_len else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"reel_recall": recall, "reel_precision": precision, "reel_f1": f1}


def goal_recall(clips: list[Clip], goal_times: list[float]) -> float:
    """Share of goals whose annotated moment is inside the reel."""
    if not goal_times:
        return float("nan")
    return float(np.mean([any(c.start <= g <= c.end for c in clips) for g in goal_times]))


def evaluate(times: np.ndarray, score: np.ndarray, segments: list[tuple[float, float]],
             budget_s: float | None = None, goals: list[float] | None = None) -> dict[str, float]:
    """Frame-level ranking quality plus the quality of a reel of equal length.

    The reel budget defaults to the total length of the true highlights, so every
    scorer is judged on producing a reel as long as the broadcaster's.
    """
    y = frame_labels(times, segments)
    out = {
        "frame_avg_precision": average_precision_score(y, score) if y.any() else float("nan"),
        "frame_roc_auc": roc_auc_score(y, score) if 0 < y.sum() < len(y) else float("nan"),
        "positive_rate": float(y.mean()),
    }
    budget = budget_s if budget_s is not None else sum(e - s for s, e in segments)
    clips = select(times, score, budget)
    out.update(reel_metrics(clips, segments))
    if goals is not None:
        out["goal_recall"] = goal_recall(clips, goals)
        out["n_goals"] = len(goals)
    return out
