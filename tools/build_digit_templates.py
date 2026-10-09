"""Build digit glyph templates from the two hand-verified matches and report
how accurately they recover the verified goal times (calibration)."""
import sys
import numpy as np
sys.path.insert(0, "tools")
import scoreboard_auto as sa

def ts(s):
    p = [float(x) for x in s.split(":")]
    return p[0] * 60 + p[1] if len(p) == 2 else p[0] * 3600 + p[1] * 60 + p[2]

V = {"match2": [("11:22", "h"), ("16:27", "a"), ("18:18", "a"), ("36:44", "a"), ("45:08", "a"),
                ("71:17", "a"), ("96:53", "h"), ("102:33", "h")],
     "match3": [("32:07", "h"), ("60:29", "h"), ("62:55", "h"), ("73:27", "h"), ("83:09", "h"),
                ("92:54", "h")]}
samples = {k: [] for k in range(10)}
data = {}
for m, goals in V.items():
    t, H, A = sa.crops(f"data/{m}_video.mp4", f"cache/{m}.scoreboard.npz")
    for side, C in (("h", H), ("a", A)):
        G, ok = sa.glyphs(C)
        data[(m, side)] = (t, G, ok)
        gt = sorted(ts(x) for x, s in goals if s == side)
        for i in np.nonzero(ok)[0]:
            if any(t[i] - 90 < x < t[i] + 10 for x in gt):
                continue
            samples[sum(x <= t[i] - 90 for x in gt)].append(G[i])
T = {k: np.median(np.array(v), axis=0) for k, v in samples.items() if len(v) > 50}
np.savez("tools/digit_glyphs.npz", **{str(k): v for k, v in T.items()})
delays = []
for (m, side), (t, G, ok) in data.items():
    d = sa.classify_glyphs(G, ok, T)
    got, final = sa.goal_times(t, d)
    truth = sorted(ts(x) for x, s in V[m] if s == side)
    dl = [a - b for a, b in zip(got, truth)]
    delays += dl
    print(m, side, "read", round((d >= 0).mean(), 2), "final", final, "/", len(truth), "delays", [round(x) for x in dl])
print("delay median", np.median(delays), "max", max(delays))
