"""Build (and cache) the per-frame feature table for one match."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import FRAME_HOP_S
from .audio import frame_times, load_audio
from .signals import loudness

BASE_SIGNALS = ["loud_z", "loud_rise"]


def _window_stat(x: np.ndarray, frames: int, kind: str) -> np.ndarray:
    """Rolling stat over the next (`kind`='ahead_max') or surrounding frames."""
    s = pd.Series(x)
    if kind == "ahead_max":
        # max over [t, t + frames]: the reaction that follows this moment
        return s[::-1].rolling(frames + 1, min_periods=1).max()[::-1].to_numpy()
    if kind == "centre_mean":
        return s.rolling(2 * frames + 1, center=True, min_periods=1).mean().to_numpy()
    raise ValueError(kind)


def add_context(df: pd.DataFrame, hop_s: float = FRAME_HOP_S) -> pd.DataFrame:
    """Add look-ahead and smoothed versions of every raw signal.

    Highlights are cut offline, so the model may look into the future: a goal
    at time t is confirmed by the roar and the commentary over the next ~15 s.
    """
    out = df.copy()
    ahead = int(round(15 / hop_s))
    around = int(round(5 / hop_s))
    for col in [c for c in df.columns if c != "time"]:
        out[f"{col}_ahead15"] = _window_stat(df[col].to_numpy(), ahead, "ahead_max")
        out[f"{col}_avg5"] = _window_stat(df[col].to_numpy(), around, "centre_mean")
    return out


def build(media: str, cache_dir: str = "cache", use_events: bool = True,
          use_commentary: bool = True, whisper_model: str = "base.en",
          verbose: bool = True) -> pd.DataFrame:
    """Compute every signal for `media`, caching the raw table as CSV."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    stem = Path(media).stem
    raw_path = cache / f"{stem}.signals.csv"
    log = print if verbose else (lambda *a, **k: None)

    df = pd.read_csv(raw_path) if raw_path.exists() else None
    if df is None:
        log(f"[{stem}] decoding audio")
        a16 = load_audio(media, 16000)
        grid = frame_times(len(a16) / 16000)
        df = pd.DataFrame({"time": grid})
        log(f"[{stem}] loudness")
        for k, v in loudness.compute(a16, 16000, grid).items():
            df[k] = v
    grid = df["time"].to_numpy()

    emb_path = cache / f"{stem}.emb.npz"
    if use_events and ("ev_cheer" not in df or not emb_path.exists()):
        from .signals import audio_events
        log(f"[{stem}] sound events + embeddings (PANNs CNN14)")
        a32 = load_audio(media, 32000)
        for k, v in audio_events.compute(a32, 32000, grid).items():
            df[k] = v
        centres, emb = audio_events.compute.last
        np.savez_compressed(emb_path, t=centres, emb=emb)

    if use_commentary and "com_keywords" not in df:
        from .signals import commentary
        log(f"[{stem}] commentary (Whisper {whisper_model})")
        a16 = load_audio(media, 16000)
        feats, segments = commentary.compute(a16, 16000, grid, whisper_model)
        for k, v in feats.items():
            df[k] = v
        (cache / f"{stem}.transcript.json").write_text(json.dumps(segments, indent=1))

    df.to_csv(raw_path, index=False)
    return df
