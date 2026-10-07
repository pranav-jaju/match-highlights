"""Tests on synthetic audio where the right answer is known exactly."""

import numpy as np
import pandas as pd
import pytest

from matchreel.align import SR as ALIGN_SR, logmel, match_chunks, to_segments
from matchreel.audio import frame_times
from matchreel.evaluate import evaluate, frame_labels, reel_metrics
from matchreel.reel import Clip, merge, select
from matchreel.scoring import FusionModel, score_loudness
from matchreel.signals import loudness
from matchreel.signals.commentary import phrase_score, segments_to_grid

SR = 16000


def crowd_match(duration_s=600, bursts=(60, 210, 420), seed=0):
    """Quiet murmur with loud 8-second 'roars' starting at each burst time."""
    rng = np.random.default_rng(seed)
    audio = 0.02 * rng.standard_normal(duration_s * SR).astype(np.float32)
    for b in bursts:
        audio[b * SR:(b + 8) * SR] += 0.3 * rng.standard_normal(8 * SR).astype(np.float32)
    return audio


def loudness_frame(audio):
    grid = frame_times(len(audio) / SR)
    df = pd.DataFrame({"time": grid})
    for k, v in loudness.compute(audio, SR, grid).items():
        df[k] = v
    return df


def test_loudness_reel_finds_every_roar():
    bursts = (60, 210, 420)
    df = loudness_frame(crowd_match(bursts=bursts))
    clips = select(df["time"].to_numpy(), score_loudness(df), budget_s=70)
    for b in bursts:
        assert any(c.start <= b + 4 <= c.end for c in clips), f"missed roar at {b}s"


def test_select_respects_budget_and_merges():
    t = np.arange(0, 600, 0.5)
    score = np.zeros_like(t)
    for p in (100, 105, 300):  # two peaks close together should merge
        score[int(p / 0.5)] = 1.0
    clips = select(t, score, budget_s=60, min_gap_s=1)
    assert sum(c.length for c in clips) <= 60
    assert merge(clips) == clips


def test_alignment_recovers_cut_positions():
    rng = np.random.default_rng(1)
    match = (0.1 * rng.standard_normal(300 * ALIGN_SR)).astype(np.float32)
    # Shape the noise so the spectrum changes over time, like real audio.
    env = 0.5 + np.abs(np.sin(np.linspace(0, 40, match.size)))
    match *= env.astype(np.float32)
    truth = [(40.0, 52.0), (130.0, 150.0), (240.0, 256.0)]
    filler = (0.1 * rng.standard_normal(3 * ALIGN_SR)).astype(np.float32)
    reel = np.concatenate([np.concatenate([match[int(s * ALIGN_SR):int(e * ALIGN_SR)], filler])
                           for s, e in truth])
    ms, rs = logmel(match), logmel(reel)
    segs = to_segments(match_chunks(ms, rs), match_spec=ms, reel_spec=rs)
    assert len(segs) == len(truth)
    for (s, e, conf), (ts, te) in zip(segs, truth):
        assert abs(s - ts) < 0.5 and abs(e - te) < 0.5 and conf > 0.8


def test_reel_metrics():
    segs = [(10, 20), (100, 110)]
    m = reel_metrics([Clip(8, 22, 15, 1.0)], segs)
    assert m["reel_recall"] == 0.5
    assert m["reel_precision"] == pytest.approx(10 / 14)


def test_phrase_score_and_grid():
    assert phrase_score("And it's a GOAL! What a goal!") > phrase_score("passed back to the keeper")
    grid = np.arange(0, 10, 0.5) + 0.25
    feats = segments_to_grid([{"start": 2, "end": 4, "text": "goal goal"}], grid)
    assert feats["com_keywords"][(grid > 2) & (grid < 4)].min() > 0
    assert feats["com_keywords"][grid > 5].max() == 0


def test_fusion_learns_on_synthetic_matches():
    frames, labels = [], []
    for seed in range(3):
        bursts = (50 + 30 * seed, 300, 500)
        df = loudness_frame(crowd_match(bursts=bursts, seed=seed))
        frames.append(df)
        labels.append(frame_labels(df["time"].to_numpy(), [(b - 5, b + 10) for b in bursts]))
    model = FusionModel().fit(frames[:2], labels[:2])
    res = evaluate(frames[2]["time"].to_numpy(), model.score(frames[2]),
                   [(b - 5, b + 10) for b in (110, 300, 500)])
    assert res["frame_roc_auc"] > 0.9


def test_best_signal_picks_informative_feature():
    from matchreel.scoring import BestSignal
    frames, labels = [], []
    for seed in range(2):
        bursts = (80, 300, 500)
        df = loudness_frame(crowd_match(bursts=bursts, seed=seed))
        frames.append(df)
        labels.append(frame_labels(df["time"].to_numpy(), [(b - 5, b + 10) for b in bursts]))
    bs = BestSignal().fit(frames, labels)
    assert bs.column.startswith("loud")
    assert bs.train_auc > 0.9
