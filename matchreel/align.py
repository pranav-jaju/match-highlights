"""Ground truth for free: locate an official highlights reel inside the full match.

Broadcasters cut highlight reels from the same feed, so each highlight clip's
audio appears verbatim somewhere in the full match. We slide short chunks of
the reel's log-mel spectrogram over the match's, using normalised
cross-correlation (computed with FFTs), and keep confident matches. Runs of
chunks that map to consecutive match times become labelled highlight segments.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .audio import load_audio

SR = 16000
HOP = 1600          # 0.1 s per spectrogram frame
N_MELS = 64


def logmel(audio: np.ndarray) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=4096, hop_length=HOP, n_mels=N_MELS)
    # float64: the alignment takes running sums over ~60k frames, and float32
    # loses enough precision there to push the correlation above 1.
    x = librosa.power_to_db(m + 1e-10).astype(np.float64)
    # Standardise each band so overall gain / EQ differences between the
    # broadcast and the re-encoded highlights don't matter.
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)


@dataclass
class ChunkMatch:
    reel_t: float      # chunk start in the highlights reel (s)
    match_t: float     # best matching start in the full match (s)
    ncc: float         # normalised cross-correlation of the best match


class MatchIndex:
    """Precomputed FFT of the match spectrogram for fast repeated correlation."""

    def __init__(self, match_spec: np.ndarray, chunk_frames: int):
        self.spec = match_spec
        self.L = chunk_frames
        self.n = match_spec.shape[1]
        self.nfft = 1 << int(np.ceil(np.log2(self.n + chunk_frames)))
        self.F = np.fft.rfft(match_spec, n=self.nfft, axis=1)
        # Variance of every length-L window of the match around its own per-band
        # mean (summed over bands): the denominator of a true Pearson NCC.
        L = chunk_frames
        z = np.zeros((match_spec.shape[0], 1))
        s1 = np.concatenate([z, np.cumsum(match_spec, axis=1)], axis=1)
        s2 = np.concatenate([z, np.cumsum(match_spec ** 2, axis=1)], axis=1)
        w1 = s1[:, L:] - s1[:, :-L]
        w2 = s2[:, L:] - s2[:, :-L]
        var = (w2 - w1 ** 2 / L).sum(axis=0)
        # Near-silent windows have almost no variance; dividing by it would blow
        # the NCC far past 1. Floor the denominator at a fraction of typical variance.
        self.win_energy = np.maximum(var, 0.05 * np.median(var) + 1e-9)

    def best(self, chunk: np.ndarray) -> tuple[int, float]:
        c = chunk - chunk.mean(axis=1, keepdims=True)
        C = np.fft.rfft(c[:, ::-1], n=self.nfft, axis=1)
        # Sum over bands in the frequency domain, then a single inverse FFT.
        corr = np.fft.irfft((self.F * C).sum(axis=0), n=self.nfft)
        valid = corr[self.L - 1: self.n]                 # one value per window start
        ncc = valid / (np.sqrt(self.win_energy) * np.linalg.norm(c) + 1e-9)
        i = int(np.argmax(ncc))
        return i, float(ncc[i])


def match_chunks(match_spec: np.ndarray, reel_spec: np.ndarray,
                 chunk_s: float = 6.0, hop_s: float = 2.0) -> list[ChunkMatch]:
    L = int(chunk_s * SR / HOP)
    step = int(hop_s * SR / HOP)
    index = MatchIndex(match_spec, L)
    out = []
    for start in range(0, reel_spec.shape[1] - L + 1, step):
        i, score = index.best(reel_spec[:, start:start + L])
        out.append(ChunkMatch(start * HOP / SR, i * HOP / SR, score))
    return out


def refine(match_spec: np.ndarray, reel_spec: np.ndarray, reel_t0: float, reel_t1: float,
           offset_s: float, min_corr: float = 0.8) -> tuple[float, float] | None:
    """Exact segment edges: walk outwards frame by frame from a confident run
    while reel and match frames (at the run's offset) stay highly correlated."""
    fps = SR / HOP
    off = int(round(offset_s * fps))
    lo = max(0, int(reel_t0 * fps) - int(6 * fps), -off)
    hi = min(reel_spec.shape[1], int(reel_t1 * fps) + int(6 * fps), match_spec.shape[1] - off)
    if hi - lo < 3:
        return None
    r = reel_spec[:, lo:hi]
    m = match_spec[:, lo + off:hi + off]
    r = (r - r.mean(0)) / (r.std(0) + 1e-6)
    m = (m - m.mean(0)) / (m.std(0) + 1e-6)
    corr = (r * m).mean(0)
    corr = np.convolve(corr, np.ones(5) / 5, mode="same")   # ignore single-frame dips
    good = corr >= min_corr
    centre = int(((reel_t0 + reel_t1) / 2) * fps) - lo
    if not 0 <= centre < len(good) or not good[centre]:
        return None
    a = centre
    while a > 0 and good[a - 1]:
        a -= 1
    b = centre
    while b < len(good) - 1 and good[b + 1]:
        b += 1
    return (lo + a) / fps + offset_s, (lo + b + 1) / fps + offset_s


def to_segments(matches: list[ChunkMatch], chunk_s: float = 6.0, min_ncc: float = 0.8,
                max_jitter_s: float = 1.0, min_len_s: float = 4.0, match_spec=None,
                reel_spec=None) -> list[tuple[float, float, float]]:
    """Group confident chunks with a consistent reel->match offset into segments,
    then (if spectrograms are given) refine each segment's edges frame by frame."""
    good = [m for m in matches if m.ncc >= min_ncc]
    segs: list[list[ChunkMatch]] = []
    for m in good:
        if segs:
            last = segs[-1][-1]
            same_offset = abs((m.match_t - m.reel_t) - (last.match_t - last.reel_t)) <= max_jitter_s
            if same_offset and m.reel_t - last.reel_t <= chunk_s:
                segs[-1].append(m)
                continue
        segs.append([m])
    out = []
    for run in segs:
        start, end = run[0].match_t, run[-1].match_t + chunk_s
        if match_spec is not None and reel_spec is not None:
            offset = float(np.median([m.match_t - m.reel_t for m in run]))
            edges = refine(match_spec, reel_spec, run[0].reel_t, run[-1].reel_t + chunk_s, offset)
            if edges:
                start, end = edges
        if end - start >= min_len_s:
            out.append((start, end, float(np.mean([m.ncc for m in run]))))
    # Overlapping matches (e.g. a replay shown twice in the reel) become one segment.
    merged: list[tuple[float, float, float]] = []
    for s, e, c in sorted(out):
        if merged and s <= merged[-1][1]:
            ps, pe, pc = merged[-1]
            merged[-1] = (ps, max(pe, e), max(pc, c))
        else:
            merged.append((s, e, c))
    return merged


def align(match_media: str, reel_media: str, **kw) -> tuple[list[tuple[float, float, float]], list[ChunkMatch]]:
    match_spec = logmel(load_audio(match_media, SR))
    reel_spec = logmel(load_audio(reel_media, SR))
    matches = match_chunks(match_spec, reel_spec)
    return to_segments(matches, match_spec=match_spec, reel_spec=reel_spec, **kw), matches
