"""Leave-one-match-out benchmark over every labelled match, with bootstrap CIs.

  python experiments/benchmark_v2.py

Writes results/v2/{per_match.csv, summary.csv, bootstrap.csv} and keeps every
held-out score curve in results/v2/scores/ so the plots and reels can be redone
without retraining.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from matchreel.evaluate import evaluate, frame_labels, load_segments  # noqa: E402
from matchreel.learned import BASE, EmbeddingScorer, load_embeddings  # noqa: E402
from matchreel.scoring import BestSignal, score_events, score_loudness, score_random  # noqa: E402

OUT = Path("results/v2")
BUDGETS = [180, 300]


def load():
    status = pd.read_csv("meta/label_status.csv") if Path("meta/label_status.csv").exists() else None
    names = []
    for name in pd.read_csv("meta/matches.csv")["name"]:
        if not (Path(f"labels/{name}.csv").exists() and Path(f"cache/{name}.emb.npz").exists()
                and Path(f"cache/{name}.signals.csv").exists()):
            continue
        if status is not None and name in set(status["name"]):
            row = status.set_index("name").loc[name]
            if "accepted" in row and not bool(row["accepted"]):
                continue
        names.append(name)
    data = []
    for n in names:
        # signals every match has (Whisper was only run on the 3-match pilot)
        df = pd.read_csv(f"cache/{n}.signals.csv")[["time", *BASE]]
        t = df["time"].to_numpy()
        segs = load_segments(f"labels/{n}.csv")
        goals = pd.read_csv(f"labels/{n}.goals.csv")["time"].tolist()
        data.append(dict(name=n, df=df, t=t, emb=load_embeddings(n, t), segs=segs, goals=goals,
                         y=frame_labels(t, segs)))
    return data


def main():
    data = load()
    print(f"{len(data)} matches, {sum(len(d['goals']) for d in data)} goals")
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, d in enumerate(data):
        tr = [x for j, x in enumerate(data) if j != i]
        scores = {
            "random": score_random(d["df"]),
            "loudness": score_loudness(d["df"]),
            "sound events": score_events(d["df"]),
            "reaction": BestSignal().fit([x["df"] for x in tr], [x["y"] for x in tr]).score(d["df"]),
        }
        for kind, label in (("logreg", "embedding logreg"), ("gbm", "embedding GBM")):
            m = EmbeddingScorer(kind).fit([x["df"] for x in tr], [x["emb"] for x in tr], [x["y"] for x in tr])
            scores[label] = m.score(d["df"], d["emb"])
        pd.DataFrame({"time": d["t"], **scores}).to_csv(OUT / "scores" / f"{d['name']}.csv", index=False)
        for name, s in scores.items():
            for b in BUDGETS:
                r = evaluate(d["t"], s, d["segs"], budget_s=b, goals=d["goals"])
                rows.append({"match": d["name"], "scorer": name, "budget_s": b, **r,
                             "goals_hit": round(r["goal_recall"] * len(d["goals"]))})
        print(f"  {d['name']} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "per_match.csv", index=False)

    order = ["random", "loudness", "sound events", "reaction", "embedding logreg", "embedding GBM"]
    agg = res.groupby(["scorer", "budget_s"]).agg(
        roc_auc=("frame_roc_auc", "mean"), avg_precision=("frame_avg_precision", "mean"),
        goals_hit=("goals_hit", "sum"), n_goals=("n_goals", "sum"),
        reel_precision=("reel_precision", "mean")).reset_index()
    agg["goal_recall"] = agg["goals_hit"] / agg["n_goals"]
    agg["scorer"] = pd.Categorical(agg["scorer"], order)
    agg = agg.sort_values(["budget_s", "scorer"])
    agg.round(3).to_csv(OUT / "summary.csv", index=False)
    print(agg.round(3).to_string(index=False))

    # Paired bootstrap over matches: difference vs the best non-learned baseline.
    rng = np.random.default_rng(0)
    names = res["match"].unique()
    boot = []
    for metric, budget in (("frame_roc_auc", 180), ("frame_avg_precision", 180),
                           ("goal_recall", 180), ("goal_recall", 300)):
        piv = res[res.budget_s == budget].pivot(index="match", columns="scorer", values=metric)
        w = res[(res.budget_s == budget) & (res.scorer == "reaction")].set_index("match")["n_goals"]
        for model, base in (("embedding logreg", "reaction"), ("embedding GBM", "reaction"),
                            ("embedding logreg", "loudness"), ("embedding GBM", "loudness")):
            diffs = []
            for _ in range(5000):
                s = rng.choice(names, len(names))
                if metric == "goal_recall":   # pooled over goals, like the headline number
                    a = (piv.loc[s, model] * w[s]).sum() / w[s].sum()
                    b = (piv.loc[s, base] * w[s]).sum() / w[s].sum()
                else:
                    a, b = piv.loc[s, model].mean(), piv.loc[s, base].mean()
                diffs.append(a - b)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            boot.append({"metric": metric, "budget_s": budget, "model": model, "vs": base,
                         "mean_diff": np.mean(diffs), "ci_low": lo, "ci_high": hi})
    boot = pd.DataFrame(boot)
    boot.round(3).to_csv(OUT / "bootstrap.csv", index=False)
    print(boot.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
