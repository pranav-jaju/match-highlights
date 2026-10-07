"""Ground-truth goal times from the broadcast scoreboard (video only, no audio).

The highlight models only ever see the audio. To evaluate them honestly the
labels must come from somewhere else, so this tool reads the on-screen score
with OCR every few seconds and records when it changes. The exact goal moment
(ball crossing the line) is then confirmed by eye on frame contact sheets, since
broadcasters update the scoreboard some seconds after the goal.

Usage:
  python tools/scoreboard_goals.py data/match2_video.mp4 --layout laliga25 --out results/scoreboard_match2.csv
"""

from __future__ import annotations

import argparse
import re
import subprocess

import numpy as np
import pandas as pd

# Pixel boxes (x, y, w, h) on a 640x360 frame, measured from sample frames.
LAYOUTS = {
    # 2015 broadcast strip: "RMA 0 / 1 FCB 19:39" -> one box holding both scores
    "laliga15": {"both": (50, 18, 110, 24)},
    # 2025 LaLiga box: home score on top, away score below
    "laliga25": {"home": (80, 18, 24, 20), "away": (80, 38, 24, 20)},
}


def frames(path: str, every_s: float, w: int = 640, h: int = 360):
    """Yield (time_s, frame) at one frame every `every_s` seconds."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vf",
           f"fps=1/{every_s},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    i = 0
    while True:
        buf = proc.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        yield i * every_s, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        i += 1


def read_score(reader, frame: np.ndarray, layout: dict) -> tuple[int, int] | None:
    def crop(box, scale=4):
        x, y, w, h = box
        import cv2
        return cv2.resize(frame[y:y + h, x:x + w], None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    if "both" in layout:
        text = " ".join(t for _, t, c in reader.readtext(crop(layout["both"])) if c > 0.3)
        m = re.search(r"(\d)\s*[/|\\Il1]?\s*(\d)", text.replace("O", "0"))
        if not m:
            m = re.search(r"(\d)\D+(\d)", text)
        return (int(m.group(1)), int(m.group(2))) if m else None
    import cv2
    out = []
    for side in ("home", "away"):
        # Single isolated digits slip past EasyOCR's text detector, so skip
        # detection and run the recogniser on the padded crop directly.
        c = cv2.copyMakeBorder(crop(layout[side]), 20, 20, 20, 20,
                               cv2.BORDER_CONSTANT, value=(235, 235, 235))
        _, text, conf = reader.recognize(c, allowlist="0123456789")[0]
        if conf < 0.6 or len(text) != 1:
            return None
        out.append(int(text))
    return tuple(out)


def score_changes(readings: pd.DataFrame, stable: int = 3) -> pd.DataFrame:
    """A new score counts only after it is read `stable` times in a row, which
    filters out OCR misreads and replays/graphics hiding the scoreboard."""
    changes, current, run_score, run_start, run_len = [], None, None, None, 0
    for t, s in readings.dropna().itertuples(index=False):
        if s == run_score:
            run_len += 1
        else:
            run_score, run_start, run_len = s, t, 1
        if run_len == stable and s != current:
            if current is not None:
                changes.append({"first_seen_s": run_start, "from": current, "to": s})
            current = s
    return pd.DataFrame(changes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--layout", choices=LAYOUTS, required=True)
    p.add_argument("--every", type=float, default=2.0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    rows = [{"t": t, "score": read_score(reader, f, LAYOUTS[a.layout])}
            for t, f in frames(a.video, a.every)]
    readings = pd.DataFrame(rows)
    readings.to_csv(a.out.replace(".csv", "_readings.csv"), index=False)
    ch = score_changes(readings)
    ch.to_csv(a.out, index=False)
    ok = readings["score"].notna().mean()
    print(f"{len(readings)} frames, scoreboard read in {ok:.0%}, {len(ch)} score changes:")
    for r in ch.itertuples():
        print(f"  {int(r.first_seen_s // 60):3d}:{r.first_seen_s % 60:04.1f}  {r._2} -> {r.to}")


if __name__ == "__main__":
    main()
