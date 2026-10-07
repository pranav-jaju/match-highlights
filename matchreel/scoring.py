"""Excitement scorers: rule-based baselines and a trained fusion model."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import add_context


def score_loudness(df: pd.DataFrame) -> np.ndarray:
    """Baseline: smoothed loudness, i.e. 'the loudest stretches are the highlights'."""
    ctx = add_context(df[["time", "loud_z"]])
    return ctx["loud_z_avg5"].to_numpy()


def score_events(df: pd.DataFrame) -> np.ndarray:
    """Rule on sound events: crowd reaction in the next 15 s, minus music.

    No training needed, so it's the fair 'pretrained model only' comparison.
    """
    cols = ["time", "ev_cheer", "ev_crowd", "ev_shout", "ev_music"]
    ctx = add_context(df[[c for c in cols if c in df]])
    reaction = ctx[["ev_cheer_ahead15", "ev_shout_ahead15"]].max(axis=1)
    return (reaction + 0.5 * ctx["ev_crowd_avg5"] - ctx["ev_music_avg5"]).to_numpy()


class FusionModel:
    """Logistic regression over every signal plus its look-ahead/smoothed context.

    Deliberately simple: with a handful of labelled matches, a linear model
    generalises better than a deep one and its weights are readable.
    """

    def __init__(self, C: float = 0.5):
        self.pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
        )
        self.columns: list[str] = []

    @staticmethod
    def design(df: pd.DataFrame) -> pd.DataFrame:
        return add_context(df).drop(columns=["time"])

    def fit(self, frames: list[pd.DataFrame], labels: list[np.ndarray]) -> "FusionModel":
        X = pd.concat([self.design(f) for f in frames], ignore_index=True)
        self.columns = list(X.columns)
        self.pipe.fit(X.to_numpy(), np.concatenate(labels))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self.design(df).reindex(columns=self.columns, fill_value=0.0)
        return self.pipe.predict_proba(X.to_numpy())[:, 1]

    def weights(self) -> pd.Series:
        coef = self.pipe[-1].coef_[0]
        return pd.Series(coef, index=self.columns).sort_values(key=np.abs, ascending=False)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipe": self.pipe, "columns": self.columns}, path)

    @classmethod
    def load(cls, path: str) -> "FusionModel":
        d = joblib.load(path)
        m = cls()
        m.pipe, m.columns = d["pipe"], d["columns"]
        return m


class BestSignal:
    """Pick the single context feature with the highest ROC-AUC on the training
    matches, then score with it alone. With very few labelled matches this is far
    less prone to overfitting than a 30-weight model."""

    def fit(self, frames: list[pd.DataFrame], labels: list[np.ndarray]) -> "BestSignal":
        from sklearn.metrics import roc_auc_score

        X = [add_context(f).drop(columns=["time"]) for f in frames]
        auc = {c: np.mean([roc_auc_score(y, x[c]) for x, y in zip(X, labels)]) for c in X[0].columns}
        self.column = max(auc, key=auc.get)
        self.train_auc = auc[self.column]
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return add_context(df)[self.column].to_numpy()


def score_random(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Chance level: smoothed noise. Any real scorer must clearly beat this."""
    noise = np.random.default_rng(seed).standard_normal(len(df))
    return pd.Series(noise).rolling(21, center=True, min_periods=1).mean().to_numpy()


SCORERS = {"random": score_random, "loudness": score_loudness, "events": score_events}
