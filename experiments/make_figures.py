"""README figures and the demo reel, all from held-out (leave-one-match-out) scores.

  python experiments/make_figures.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, ".")
from matchreel.evaluate import evaluate  # noqa: E402
from matchreel.plot import excitement_plot  # noqa: E402
from matchreel.reel import render, select  # noqa: E402

OUT = "results/v2"
summary = pd.read_csv(f"{OUT}/summary.csv")
per = pd.read_csv(f"{OUT}/per_match.csv")
scorers = ["random", "loudness", "sound events", "reaction", "embedding logreg", "embedding GBM"]
colors = ["#bbbbbb", "#7f9cc9", "#c9a27f", "#8fbf8f", "#3b6fb6", "#1b3a8f"]

# 1. goal recall vs reel length, for every scorer
fig, ax = plt.subplots(figsize=(7, 4))
budgets = [60, 120, 180, 300, 420, 600]
names = per["match"].unique()
rows = []
for n in names:
    sc = pd.read_csv(f"{OUT}/scores/{n}.csv")
    goals = pd.read_csv(f"labels/{n}.goals.csv")["time"].tolist()
    segs = pd.read_csv(f"labels/{n}.csv")
    segs = list(zip(segs.start, segs.end))
    for s in scorers:
        for b in budgets:
            r = evaluate(sc["time"].to_numpy(), sc[s].to_numpy(), segs, budget_s=b, goals=goals)
            rows.append({"scorer": s, "budget": b, "hit": r["goal_recall"] * len(goals), "n": len(goals)})
curve = pd.DataFrame(rows).groupby(["scorer", "budget"]).sum().reset_index()
curve["recall"] = curve["hit"] / curve["n"]
curve.to_csv(f"{OUT}/recall_vs_budget.csv", index=False)
for s, c in zip(scorers, colors):
    d = curve[curve.scorer == s]
    ax.plot(d.budget / 60, d.recall * 100, marker="o", color=c, lw=2.2 if "GBM" in s else 1.4, label=s)
ax.set_xlabel("reel length (minutes)")
ax.set_ylabel("goals included in reel (%)")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
ax.set_title(f"Held-out goal recall, {len(names)} matches / {int(curve.n.iloc[0])} goals", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/recall_vs_budget.png", dpi=140)

# 2. excitement curves for the demo match
demo = "match2"
sc = pd.read_csv(f"{OUT}/scores/{demo}.csv")
goals = pd.read_csv(f"labels/{demo}.goals.csv")["time"].tolist()
t = sc["time"].to_numpy()
show = {"loudness": sc["loudness"].to_numpy(), "embedding GBM": sc["embedding GBM"].to_numpy()}
clips = {k: select(t, v, 180) for k, v in show.items()}
excitement_plot(t, show, goals, clips, "Betis 3-5 Barcelona (held out)", f"{OUT}/{demo}_excitement.png")

# 3. demo reel (5 min) from the held-out GBM score
reel = select(t, sc["embedding GBM"].to_numpy(), 300)
hit = sum(any(c.start <= g <= c.end for c in reel) for g in goals)
print(f"demo reel: {len(reel)} clips, {hit}/{len(goals)} goals")
render(f"data/{demo}_video.mp4", reel, f"output/highlights_{demo}.mp4")
