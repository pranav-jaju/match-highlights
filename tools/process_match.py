"""Process one manifest match end to end: audio, automatic goal labels, features.

  python tools/process_match.py m06 [m07 ...]

Labels are written only when the scoreboard reading reproduces the final score
in the video title for both teams; otherwise the match is marked rejected.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "tools")
sys.path.insert(0, ".")
import scoreboard_auto as sa  # noqa: E402
from match_highlights import features  # noqa: E402

DELAY_S = 4.5          # median scoreboard delay measured on 14 hand-verified goals
HALF_WINDOW = 10.0     # highlight segment = goal +- 10 s (same as hand labels)
MANIFEST = pd.read_csv("meta/matches.csv").set_index("name")


def fmt(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:04.1f}"


def auto_labels(name: str) -> dict:
    row = MANIFEST.loc[name]
    T = {int(k): v for k, v in np.load("tools/digit_glyphs.npz").items()}
    t, H, A = sa.crops_v3(f"data/{name}_video.mp4", f"cache/{name}.scoreboard3.npz")
    res = {}
    for side, C, want in (("home", H, row.home_goals), ("away", A, row.away_goals)):
        d = sa.classify_shifted(C, T)
        times, final = sa.goal_times(t, d)
        res[side] = (times, final, int(want), float((d >= 0).mean()))
    accepted = all(final == want for _, final, want, _ in res.values())
    status = {"name": name, "accepted": accepted,
              "home_read": f"{res['home'][1]}/{res['home'][2]}",
              "away_read": f"{res['away'][1]}/{res['away'][2]}",
              "digit_read_rate": round(min(r[3] for r in res.values()), 3)}
    if accepted:
        goals = sorted((tt - DELAY_S, side) for side, (times, *_) in res.items() for tt in times)
        team = {"home": row.home, "away": row.away}
        ev = pd.DataFrame({"time": [fmt(g) for g, _ in goals], "type": "goal",
                           "note": [f"{team[s]} (auto: scoreboard)" for _, s in goals]})
        ev.to_csv(f"labels/{name}.events.csv", index=False)
        segs = []
        for g, _ in goals:
            s, e = max(0.0, g - HALF_WINDOW), g + HALF_WINDOW
            if segs and s <= segs[-1][1]:
                segs[-1] = (segs[-1][0], e)
            else:
                segs.append((s, e))
        pd.DataFrame(segs, columns=["start", "end"]).to_csv(f"labels/{name}.csv", index=False)
        pd.DataFrame({"time": [g for g, _ in goals]}).to_csv(f"labels/{name}.goals.csv", index=False)
    return status


def main(names):
    rows = []
    for name in names:
        audio = Path(f"data/{name}.m4a")
        if not audio.exists():
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", f"data/{name}_video.mp4",
                            "-vn", "-c:a", "aac", "-b:a", "96k", str(audio)], check=True)
        status = auto_labels(name) if MANIFEST.loc[name].layout == "laliga25" else {"name": name}
        print(status, flush=True)
        features.build(str(audio), "cache", use_events=True, use_commentary=False)
        rows.append(status)
    log = Path("meta/label_status.csv")
    old = pd.read_csv(log) if log.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old["name"].isin(new["name"])]
    pd.concat([old, new]).sort_values("name").to_csv(log, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
