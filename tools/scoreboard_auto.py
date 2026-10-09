"""Automatic goal labels from the broadcast scoreboard (video only, no audio).

For the LaLiga 2025/26 broadcast graphic, each team's score sits in a fixed
box. We sample one frame per second, keep frames where the scoreboard is on
screen, binarise each score box and track it as a sequence of stable "digit
states". A goal is a switch to a new state that then persists. A match is only
accepted when the number of switches per team equals the final score in the
video title, so a misdetection can't silently enter the labels.

The scoreboard updates a few seconds after the ball crosses the line; the
delay is calibrated against hand-verified goals (see `calibrate`).
"""

from __future__ import annotations

import subprocess

import numpy as np

W, H = 640, 360
# Region holding logo + both score boxes, then sub-boxes inside it.
REGION = (20, 10, 90, 52)          # x, y, w, h on a 640x360 frame
HOME = (60, 8, 24, 20)             # relative to REGION
AWAY = (60, 28, 24, 20)
LOGO = (6, 4, 28, 22)              # red LaLiga block


def region_frames(video: str, fps: float = 1.0):
    x, y, w, h = REGION
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", video, "-vf",
           f"fps={fps},scale={W}:{H},crop={w}:{h}:{x}:{y}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    i = 0
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield i / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        i += 1


def _box(img, b):
    x, y, w, h = b
    return img[y:y + h, x:x + w]


def visible(img) -> bool:
    """Scoreboard on screen: LaLiga red block present and score boxes light."""
    logo = _box(img, LOGO).astype(int)
    red = ((logo[..., 0] > 170) & (logo[..., 1] < 110) & (logo[..., 2] < 110)).mean()
    boxes = np.concatenate([_box(img, HOME), _box(img, AWAY)]).mean()
    return red > 0.35 and boxes > 150


def digit_mask(img, b) -> np.ndarray:
    """Dark digit pixels inside a light box, as a binary vector."""
    g = _box(img, b).mean(axis=2)
    return (g < 0.5 * np.median(g)).astype(np.float32).ravel()


def track(times, masks, min_dist: float = 0.08, confirm: int = 20, window: int = 30):
    """Stable-state change points in a sequence of binary masks.

    A change is declared at the first frame that differs from the current state
    when at least `confirm` of the next `window` frames also differ from it and
    agree with each other. Returns (change times, state templates)."""
    state = masks[:20].mean(axis=0) if len(masks) >= 20 else masks.mean(axis=0)
    changes, states = [], [state]
    i = 0
    n = len(masks)
    while i < n:
        d = np.abs(masks[i] - state).mean()
        if d > min_dist:
            fut = masks[i:i + window]
            far = np.abs(fut - state).mean(axis=1) > min_dist
            if far.sum() >= confirm:
                cand = fut[far].mean(axis=0)
                agree = np.abs(fut[far] - cand).mean(axis=1) < min_dist
                if agree.sum() >= confirm:
                    changes.append(times[i])
                    state = cand
                    states.append(state)
                    i += window
                    continue
        i += 1
    return changes, states


def detect(video: str) -> dict:
    ts, home, away = [], [], []
    for t, img in region_frames(video):
        if visible(img):
            ts.append(t)
            home.append(digit_mask(img, HOME))
            away.append(digit_mask(img, AWAY))
    ts = np.array(ts)
    hc, _ = track(ts, np.array(home))
    ac, _ = track(ts, np.array(away))
    return {"home": hc, "away": ac, "visible_frac": len(ts)}


# ---------------------------------------------------------------------------
# v2: read the digits with templates and require a consistent score sequence.

def crops(video: str, cache: str | None = None):
    """(times, home crops, away crops) for frames with the scoreboard visible."""
    import os
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return d["t"], d["home"], d["away"]
    ts, hs, as_ = [], [], []
    for t, img in region_frames(video):
        if visible(img):
            ts.append(t)
            hs.append(_box(img, HOME).mean(axis=2))
            as_.append(_box(img, AWAY).mean(axis=2))
    out = np.array(ts), np.array(hs, np.float32), np.array(as_, np.float32)
    if cache:
        np.savez_compressed(cache, t=out[0], home=out[1], away=out[2])
    return out


def _norm(x):
    x = x.reshape(len(x), -1) if x.ndim == 3 else x.reshape(1, -1)
    x = x - x.mean(axis=1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


def classify(c: np.ndarray, templates: dict[int, np.ndarray], min_corr=0.8, min_margin=0.05):
    """Digit per crop by normalised correlation; -1 when unsure."""
    keys = sorted(templates)
    T = np.stack([_norm(templates[k])[0] for k in keys])
    s = _norm(c) @ T.T
    order = np.sort(s, axis=1)
    best = np.array(keys)[s.argmax(axis=1)]
    ok = (order[:, -1] >= min_corr) & (order[:, -1] - order[:, -2] >= min_margin)
    return np.where(ok, best, -1)


def goal_times(t, digits, stable: int = 5, window: int = 8):
    """Times the read score first goes up by one and stays there.

    Only increments of exactly one are accepted; anything else (a misread, a
    graphic) is ignored. Returns (times, final score)."""
    cur, out = 0, []
    valid = digits >= 0
    t, d = t[valid], digits[valid]
    i = 0
    while i < len(d):
        if d[i] == cur + 1 and (d[i:i + window] == cur + 1).sum() >= stable:
            out.append(float(t[i]))
            cur += 1
            i += window
            continue
        i += 1
    return out, cur


def glyph(crop: np.ndarray, size=(10, 16)) -> np.ndarray | None:
    """Shift-invariant digit shape: bounding box of the dark strokes, resized."""
    import cv2
    # trim the box border (2 px) so only the digit strokes can be dark
    g = crop[2:-2, 3:-3].astype(np.float32)
    light = np.percentile(g, 75)
    dark = (g < 0.6 * light).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n < 2:
        return None
    # keep components that look like digit strokes (tall enough, not box borders)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_HEIGHT] >= 8
            and stats[i, cv2.CC_STAT_WIDTH] <= 0.8 * crop.shape[1]]
    if not keep:
        return None
    m = np.isin(lab, keep)
    ys, xs = np.nonzero(m)
    box = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    return cv2.resize(box, size, interpolation=cv2.INTER_AREA).ravel()


def glyphs(C: np.ndarray):
    out = np.zeros((len(C), 160), np.float32)
    ok = np.zeros(len(C), bool)
    for i, c in enumerate(C):
        g = glyph(c)
        if g is not None:
            out[i], ok[i] = g, True
    return out, ok


def classify_glyphs(G, ok, templates: dict[int, np.ndarray], max_dist=0.12, min_margin=0.04):
    keys = sorted(templates)
    T = np.stack([templates[k] for k in keys])
    d = np.abs(G[:, None, :] - T[None]).mean(axis=2)
    srt = np.sort(d, axis=1)
    best = np.array(keys)[d.argmin(axis=1)]
    good = ok & (srt[:, 0] <= max_dist) & (srt[:, 1] - srt[:, 0] >= min_margin)
    return np.where(good, best, -1)


# ---------------------------------------------------------------------------
# v3: locate the light score box inside a padded crop before reading the digit
# (the graphic shifts by a few pixels, e.g. when the stoppage-time row appears).

PAD_X, PAD_Y = 6, 3


def _padded(img, b):
    x, y, w, h = b
    return img[max(0, y - PAD_Y):y + h + PAD_Y, max(0, x - PAD_X):x + w + PAD_X].mean(axis=2)


def crops_v3(video: str, cache: str | None = None):
    import os
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return d["t"], d["home"], d["away"]
    ts, hs, as_ = [], [], []
    for t, img in region_frames(video):
        if visible(img):
            ts.append(t)
            hs.append(_padded(img, HOME))
            as_.append(_padded(img, AWAY))
    out = np.array(ts), np.array(hs, np.float32), np.array(as_, np.float32)
    if cache:
        np.savez_compressed(cache, t=out[0], home=out[1], away=out[2])
    return out


def glyph_v3(crop: np.ndarray, size=(10, 16)) -> np.ndarray | None:
    import cv2
    g = crop.astype(np.float32)
    light = (g > 0.8 * np.percentile(g, 95)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(light, connectivity=4)
    if n < 2:
        return None
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    x, y, w, h = st[k, :4]
    if w < 10 or h < 12:
        return None
    inner = g[y + 1:y + h - 1, x + 1:x + w - 1]
    if inner.size == 0:
        return None
    dark = (inner < 0.6 * np.percentile(inner, 90)).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = [i for i in range(1, n2) if st2[i, cv2.CC_STAT_HEIGHT] >= 0.45 * inner.shape[0]]
    if not keep:
        return None
    m = np.isin(lab2, keep)
    ys, xs = np.nonzero(m)
    box = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    return cv2.resize(box, size, interpolation=cv2.INTER_AREA).ravel()


def glyphs_v3(C):
    out = np.zeros((len(C), 160), np.float32)
    ok = np.zeros(len(C), bool)
    for i, c in enumerate(C):
        g = glyph_v3(c)
        if g is not None:
            out[i], ok[i] = g, True
    return out, ok


def classify_shifted(Cpad: np.ndarray, templates: dict[int, np.ndarray],
                     max_dist=0.22, min_margin=0.06):
    """Read a digit from a padded crop by trying every small shift of the
    standard 24x20 window and keeping the best-matching (shift, digit)."""
    keys = sorted(templates)
    T = np.stack([templates[k] for k in keys])
    h, w = HOME[3], HOME[2]
    out = np.full(len(Cpad), -1)
    for i, c in enumerate(Cpad):
        best = np.full(len(keys), np.inf)
        for dy in range(0, c.shape[0] - h + 1):
            for dx in range(0, c.shape[1] - w + 1, 2):
                g = glyph(c[dy:dy + h, dx:dx + w])
                if g is None:
                    continue
                best = np.minimum(best, np.abs(T - g).mean(axis=1))
        srt = np.sort(best)
        if srt[0] <= max_dist and srt[1] - srt[0] >= min_margin:
            out[i] = keys[int(np.argmin(best))]
    return out
