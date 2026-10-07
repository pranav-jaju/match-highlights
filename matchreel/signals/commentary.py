"""Commentary signal: transcribe with Whisper and score what is being said.

Commentators react to big moments in two measurable ways: they say particular
words ("goal", "what a save", "wicket") and they speak faster. Both are
independent of crowd noise, so they help in empty or quiet stadiums.
"""

from __future__ import annotations

import re

import numpy as np

# Lowercase phrases that mark a big moment; weights reflect how decisive they are.
EXCITEMENT_PHRASES = {
    "goal": 3.0, "scores": 2.5, "what a goal": 3.0, "penalty": 2.0,
    "red card": 2.5, "yellow card": 1.0, "what a save": 2.5, "saved": 1.5,
    "off the post": 2.0, "off the bar": 2.0, "header": 1.0, "free kick": 1.0,
    "wicket": 3.0, "bowled": 2.5, "caught": 2.0, "six": 2.0, "four": 1.0,
    "out": 1.0, "lbw": 2.5, "century": 2.0,
    "unbelievable": 1.5, "incredible": 1.5, "brilliant": 1.0, "wow": 1.0,
    "what a": 1.0, "oh my": 1.5, "magnificent": 1.5, "stunning": 1.5,
}

_model = None


def _get_model(name: str):
    global _model
    if _model is None:
        import whisper  # openai-whisper, lazy because it pulls in torch

        _model = whisper.load_model(name, device="cpu")
    return _model


def phrase_score(text: str) -> float:
    """Weighted count of excitement phrases, plus a bonus for exclamations."""
    t = " " + re.sub(r"[^a-z!' ]", " ", text.lower()) + " "
    score = 0.0
    for phrase, w in EXCITEMENT_PHRASES.items():
        score += w * len(re.findall(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", t))
    score += 0.5 * text.count("!")
    return score


def transcribe(audio_16k: np.ndarray, model_name: str = "base.en") -> list[dict]:
    model = _get_model(model_name)
    result = model.transcribe(audio_16k, fp16=False, language="en",
                              condition_on_previous_text=False, verbose=None)
    return [{"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result["segments"]]


def segments_to_grid(segments: list[dict], grid: np.ndarray) -> dict[str, np.ndarray]:
    """Spread per-segment scores over the frames each segment covers."""
    kw = np.zeros_like(grid)
    wps = np.zeros_like(grid)
    for s in segments:
        dur = max(s["end"] - s["start"], 0.5)
        mask = (grid >= s["start"]) & (grid < s["end"])
        kw[mask] = np.maximum(kw[mask], phrase_score(s["text"]))
        wps[mask] = len(s["text"].split()) / dur
    return {"com_keywords": kw, "com_words_per_s": wps}


def compute(audio: np.ndarray, sr: int, grid: np.ndarray,
            model_name: str = "base.en") -> tuple[dict[str, np.ndarray], list[dict]]:
    if sr != 16000:
        raise ValueError(f"commentary expects 16 kHz audio, got {sr}")
    segments = transcribe(audio, model_name)
    return segments_to_grid(segments, grid), segments
