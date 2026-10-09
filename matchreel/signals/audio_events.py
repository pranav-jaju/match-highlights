"""Sound-event signal from a pretrained AudioSet tagger (PANNs CNN14).

Loudness can't tell a goal from stadium music, an advert or a long whistle.
CNN14 was trained on AudioSet (2M clips, 527 classes), so it gives separate
per-second probabilities for cheering, crowd, applause, music and speech.
"""

from __future__ import annotations

import numpy as np

from ..audio import to_grid

PANNS_SR = 32000

# AudioSet class names -> our feature names. Several classes can feed one feature.
CLASS_GROUPS = {
    "cheer": ["Cheering", "Applause", "Clapping"],
    "crowd": ["Crowd", "Hubbub, speech noise, speech babble", "Chant"],
    "shout": ["Shout", "Yell", "Screaming"],
    "whistle": ["Whistle", "Whistling"],
    "music": ["Music"],
    "speech": ["Speech"],
}

_model = None


def _get_model():
    global _model
    if _model is None:
        from panns_inference import AudioTagging  # heavy import, keep lazy

        _model = AudioTagging(checkpoint_path=None, device="cpu")
    return _model


def _class_indices() -> dict[str, list[int]]:
    from panns_inference import labels

    idx = {name: i for i, name in enumerate(labels)}
    return {
        feat: [idx[n] for n in names if n in idx]
        for feat, names in CLASS_GROUPS.items()
    }


def tag_windows(audio: np.ndarray, win_s: float = 2.0, hop_s: float = 1.0,
                batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Run CNN14 on sliding windows. Returns (window centre times, probs[n, 527])."""
    model = _get_model()
    win, hop = int(win_s * PANNS_SR), int(hop_s * PANNS_SR)
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    starts = np.arange(0, len(audio) - win + 1, hop)
    probs, embs = [], []
    for i in range(0, len(starts), batch):
        chunk = np.stack([audio[s:s + win] for s in starts[i:i + batch]])
        clipwise, emb = model.inference(chunk)
        probs.append(clipwise)
        embs.append(emb.astype(np.float16))
    centres = (starts + win / 2) / PANNS_SR
    tag_windows.last_embeddings = np.concatenate(embs)   # (n_windows, 2048)
    return centres, np.concatenate(probs)


def compute(audio: np.ndarray, sr: int, grid: np.ndarray) -> dict[str, np.ndarray]:
    if sr != PANNS_SR:
        raise ValueError(f"audio_events expects {PANNS_SR} Hz audio, got {sr}")
    centres, probs = tag_windows(audio)
    compute.last = (centres, tag_windows.last_embeddings)
    out = {}
    for feat, cols in _class_indices().items():
        if cols:
            out[f"ev_{feat}"] = to_grid(probs[:, cols].max(axis=1), centres, grid)
    return out
