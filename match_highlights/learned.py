"""Learned highlight scorer on PANNs CNN14 embeddings + hand-crafted signals.

The pretrained tagger's class probabilities ("cheering", "crowd", ...) were
designed for generic YouTube clips and barely beat chance on broadcast audio.
Its 2048-d penultimate embedding, however, encodes much richer acoustic
structure. We project it to a few principal components (fitted on training
matches only), add the loudness and class-probability signals, give every
feature temporal context (what happens in the next 15 s, and around +-5 s),
and train a frame classifier for "within 10 s of a goal".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .audio import to_grid
from .features import add_context

BASE = ["loud_z", "loud_rise", "ev_cheer", "ev_crowd", "ev_shout", "ev_whistle", "ev_music", "ev_speech"]


def load_embeddings(stem: str, grid: np.ndarray, cache_dir: str = "cache") -> np.ndarray:
    d = np.load(Path(cache_dir) / f"{stem}.emb.npz")
    e = d["emb"].astype(np.float32)
    # nearest window for each 0.5 s frame (windows are 1 s apart)
    idx = np.clip(np.searchsorted(d["t"], grid), 0, len(d["t"]) - 1)
    return e[idx]


class EmbeddingScorer:
    def __init__(self, kind: str = "logreg", n_pca: int = 32, seed: int = 0):
        self.kind, self.n_pca, self.seed = kind, n_pca, seed

    def _design(self, df: pd.DataFrame, emb: np.ndarray) -> np.ndarray:
        z = self.pca.transform(np.log1p(np.maximum(emb, 0)))
        frame = pd.DataFrame(z, columns=[f"pc{i}" for i in range(z.shape[1])])
        frame.insert(0, "time", df["time"].to_numpy())
        for c in BASE:
            frame[c] = df[c].to_numpy()
        return add_context(frame).drop(columns=["time"]).to_numpy(np.float32)

    def fit(self, frames: list[pd.DataFrame], embs: list[np.ndarray], labels: list[np.ndarray]):
        rng = np.random.default_rng(self.seed)
        sample = np.concatenate([e[rng.choice(len(e), min(3000, len(e)), replace=False)] for e in embs])
        self.pca = PCA(self.n_pca, random_state=self.seed).fit(np.log1p(np.maximum(sample, 0)))
        X = np.concatenate([self._design(f, e) for f, e in zip(frames, embs)])
        y = np.concatenate(labels)
        if self.kind == "logreg":
            self.clf = make_pipeline(StandardScaler(), LogisticRegression(
                C=0.05, class_weight="balanced", max_iter=3000))
        else:
            self.clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                l2_regularization=1.0, class_weight="balanced", random_state=self.seed)
        self.clf.fit(X, y)
        return self

    def score(self, df: pd.DataFrame, emb: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self._design(df, emb))[:, 1]
