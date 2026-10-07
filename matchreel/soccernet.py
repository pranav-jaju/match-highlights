"""SoccerNet adapter: turn official event annotations into highlight labels and
fetch match audio/video from the SoccerNet Hugging Face organisation.

Labels (SoccerNet/SN-Labels) are public. Videos (SoccerNet/SoccerNet_raw_HQ) are
gated behind the SoccerNet NDA; downloads use the token saved by `hf auth login`.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LABELS_REPO = "SoccerNet/SN-Labels"
VIDEO_REPO = "SoccerNet/SoccerNet_raw_HQ"

# Moments a broadcaster's highlights package normally shows.
HIGHLIGHT_CLASSES = ["Goal", "Penalty", "Shots on target", "Shots off target",
                     "Red card", "Yellow->red card", "Yellow card"]
# Window around each annotated moment that counts as "the highlight":
# the build-up before the timestamp and the reaction/replay start after it.
PRE_S, POST_S = 10.0, 10.0


@dataclass
class Event:
    half: int
    t: float          # seconds into that half's video
    label: str
    visible: bool


def list_games(league: str = "england_epl") -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(LABELS_REPO, repo_type="dataset")
    return sorted({str(Path(f).parent) for f in files
                   if f.endswith("Labels-v2.json") and f.startswith(league + "/")})


def sample_games(n: int, seed: int = 0, league: str = "england_epl") -> list[str]:
    """Random, seeded choice of games, so results aren't cherry-picked."""
    games = list_games(league)
    return sorted(random.Random(seed).sample(games, min(n, len(games))))


def load_events(game: str) -> list[Event]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(LABELS_REPO, f"{game}/Labels-v2.json", repo_type="dataset")
    events = []
    for a in json.loads(Path(path).read_text())["annotations"]:
        half = int(a["gameTime"].split(" - ")[0])
        events.append(Event(half, int(a["position"]) / 1000.0, a["label"],
                            a.get("visibility", "visible") == "visible"))
    return events


def highlight_segments(events: list[Event], half: int,
                       classes: list[str] = HIGHLIGHT_CLASSES) -> list[tuple[float, float]]:
    segs = sorted((max(0.0, e.t - PRE_S), e.t + POST_S)
                  for e in events if e.half == half and e.label in classes)
    merged: list[tuple[float, float]] = []
    for s, e in segs:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def slug(game: str, half: int) -> str:
    """Filesystem-safe unique name for one half, e.g. 'epl_2015-02-21_chelsea_1-1_burnley_h1'."""
    name = Path(game).name.lower()
    name = re.sub(r"^(\d{4}-\d{2}-\d{2}) - \d{2}-\d{2} ", r"\1_", name)
    name = re.sub(r"[^a-z0-9\-]+", "_", name).strip("_")
    return f"{name}_h{half}"


def prepare(games: list[str], data_dir: str = "data/soccernet", labels_dir: str = "labels",
            download_video: bool = True, branch: str = "videos-224p") -> list[str]:
    """Write per-half label CSVs (and fetch videos) for `games`. Returns media paths."""
    from huggingface_hub import hf_hub_download

    data, labels = Path(data_dir), Path(labels_dir)
    data.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    media = []
    for game in games:
        events = load_events(game)
        for half in (1, 2):
            name = slug(game, half)
            segs = highlight_segments(events, half)
            pd.DataFrame(segs, columns=["start", "end"]).to_csv(labels / f"{name}.csv", index=False)
            goals = [e.t for e in events if e.half == half and e.label == "Goal"]
            pd.DataFrame({"time": goals}).to_csv(labels / f"{name}.goals.csv", index=False)
            target = data / f"{name}.mkv"
            if download_video and not target.exists():
                src = hf_hub_download(VIDEO_REPO, f"{game}/{half}_224p.mkv",
                                      repo_type="dataset", revision=branch)
                target.symlink_to(Path(src).resolve())
            if target.exists():
                media.append(str(target))
    return media
