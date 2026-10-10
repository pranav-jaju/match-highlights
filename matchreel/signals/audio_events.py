"""Sound-event signal from a pretrained AudioSet tagger (PANNs CNN14).

Loudness can't tell a goal from stadium music, an advert or a long whistle.
CNN14 was trained on AudioSet (2M clips, 527 classes), so it gives separate
per-second probabilities for cheering, crowd, applause, music and speech.
"""

from __future__ import annotations

from pathlib import Path

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


PANNS_FILES = {
    "class_labels_indices.csv":
        "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
    "Cnn14_mAP=0.431.pth":
        "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
}


def ensure_panns_files() -> Path:
    """panns_inference downloads with `wget`, which many machines lack; fetch
    the AudioSet label list and CNN14 checkpoint (~330 MB) with urllib instead."""
    import urllib.request

    root = Path.home() / "panns_data"
    root.mkdir(exist_ok=True)
    for name, url in PANNS_FILES.items():
        dest = root / name
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, dest)
    return root


def _get_model():
    global _model
    if _model is None:
        root = ensure_panns_files()
        from panns_inference import AudioTagging  # heavy import, keep lazy

        _model = AudioTagging(checkpoint_path=str(root / "Cnn14_mAP=0.431.pth"), device="cpu")
    return _model


def _class_indices() -> dict[str, list[int]]:
    ensure_panns_files()
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
