"""Timestamped frame grid around a moment, for confirming goal times by eye.

Usage: python tools/contact_sheet.py data/match2_video.mp4 1012 --before 60 --after 4 --step 2 --out sheet.jpg
"""

import argparse
import subprocess

import cv2
import numpy as np


def grab(video: str, t: float, w: int = 320, h: int = 180) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t:.2f}", "-i", video,
                          "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    img = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy() if raw else np.zeros((h, w, 3), np.uint8)
    label = f"{int(t // 60)}:{t % 60:04.1f}"
    cv2.rectangle(img, (0, h - 22), (110, h), (0, 0, 0), -1)
    cv2.putText(img, label, (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("t", type=float)
    p.add_argument("--before", type=float, default=60)
    p.add_argument("--after", type=float, default=4)
    p.add_argument("--step", type=float, default=2)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    times = np.arange(a.t - a.before, a.t + a.after + 1e-6, a.step)
    tiles = [grab(a.video, max(0.0, t)) for t in times]
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
