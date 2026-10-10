"""Command-line interface: python -m match_highlights <command> ..."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import features
from .evaluate import evaluate, frame_labels, load_segments
from .reel import render, select
from .scoring import SCORERS, BestSignal, FusionModel


def cmd_features(a):
    for m in a.media:
        df = features.build(m, a.cache, use_events=not a.no_events,
                            use_commentary=not a.no_commentary, whisper_model=a.whisper)
        print(f"{m}: {len(df)} frames, columns: {', '.join(df.columns[1:])}")


def cmd_align(a):
    from .align import align

    segs, chunks = align(a.match, a.reel, min_ncc=a.min_ncc)
    out = Path(a.out or f"labels/{Path(a.match).stem}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(segs, columns=["start", "end", "confidence"]).to_csv(out, index=False)
    kept = sum(c.ncc >= a.min_ncc for c in chunks)
    print(f"{kept}/{len(chunks)} reel chunks matched confidently -> {len(segs)} segments, "
          f"{sum(e - s for s, e, _ in segs):.0f}s of highlights -> {out}")


def _load_dataset(a):
    data = []
    for m in a.media:
        lab = Path(a.labels) / f"{Path(m).stem}.csv"
        if not lab.exists():
            print(f"skip {m}: no labels at {lab}")
            continue
        df = features.build(m, a.cache, verbose=True)
        goals_file = lab.with_suffix(".goals.csv")
        goals = pd.read_csv(goals_file)["time"].tolist() if goals_file.exists() else None
        data.append((Path(m).stem, df, load_segments(str(lab)), goals))
    return data


def cmd_benchmark(a):
    """Leave-one-match-out: the fusion model never sees the match it is scored on."""
    data = _load_dataset(a)
    if len(data) < 2:
        raise SystemExit("need at least 2 labelled matches for leave-one-out")
    rows = []
    for i, (name, df, segs, goals) in enumerate(data):
        t = df["time"].to_numpy()
        for scorer, fn in SCORERS.items():
            rows.append({"match": name, "scorer": scorer, **evaluate(t, fn(df), segs, budget_s=a.budget, goals=goals)})
        train = [d for j, d in enumerate(data) if j != i]
        model = FusionModel().fit([d[1] for d in train],
                                  [frame_labels(d[1]["time"].to_numpy(), d[2]) for d in train])
        rows.append({"match": name, "scorer": "fusion",
                     **evaluate(t, model.score(df), segs, budget_s=a.budget, goals=goals)})
        best = BestSignal().fit([d[1] for d in train],
                                [frame_labels(d[1]["time"].to_numpy(), d[2]) for d in train])
        rows.append({"match": name, "scorer": "reaction",
                     **evaluate(t, best.score(df), segs, budget_s=a.budget, goals=goals)})
    res = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    res.to_csv("results/benchmark.csv", index=False)
    cols = ["frame_avg_precision", "frame_roc_auc", "reel_recall", "reel_precision", "reel_f1"]
    summary = res.groupby("scorer")[cols].mean().reindex(["random", "loudness", "events", "fusion", "reaction"])
    if "goal_recall" in res:
        # Pool goals across matches (weight by goals, not by match).
        g = res.dropna(subset=["goal_recall"])
        summary["goal_recall"] = (g["goal_recall"] * g["n_goals"]).groupby(g["scorer"]).sum() \
            / g.groupby("scorer")["n_goals"].sum()
        summary["n_goals"] = g.groupby("scorer")["n_goals"].sum()
    print(f"\nLeave-one-match-out over {len(data)} match halves (mean):\n")
    print(summary.round(3).to_string())
    summary.round(4).to_csv("results/benchmark_summary.csv")

    final = FusionModel().fit([d[1] for d in data],
                              [frame_labels(d[1]["time"].to_numpy(), d[2]) for d in data])
    final.save(a.model)
    print(f"\nfusion model trained on all {len(data)} matches -> {a.model}")
    print("top weights:\n" + final.weights().head(8).round(3).to_string())


def cmd_run(a):
    df = features.build(a.media, a.cache, use_events=a.scorer != "loudness",
                        use_commentary=a.scorer == "fusion")
    t = df["time"].to_numpy()
    if a.scorer == "learned":
        import joblib
        from .learned import load_embeddings
        model = joblib.load(a.learned_model)
        score = model.score(df, load_embeddings(Path(a.media).stem, t, a.cache))
    elif a.scorer == "fusion":
        score = FusionModel.load(a.model).score(df)
    else:
        score = SCORERS[a.scorer](df)
    clips = select(t, score, a.budget, pre_s=a.pre, post_s=a.post)
    for c in clips:
        print(f"  {c.start:8.1f}s - {c.end:8.1f}s  (peak {c.peak:.1f}s, score {c.score:.3f})")
    print(f"{len(clips)} clips, {sum(c.length for c in clips):.0f}s total")
    if a.timestamps_only:
        return
    render(a.media, clips, a.out)
    print(f"reel -> {a.out}")


def cmd_plot(a):
    """Excitement curves for one labelled match. Fusion is trained on the *other*
    matches given, so the plot shows the same out-of-sample behaviour as the benchmark."""
    from .plot import excitement_plot

    data = _load_dataset(a)
    for i, (name, df, segs, goals) in enumerate(data):
        t = df["time"].to_numpy()
        scores = {k: fn(df) for k, fn in SCORERS.items() if k != "random"}
        others = [d for j, d in enumerate(data) if j != i]
        if others:
            Y = [frame_labels(d[1]["time"].to_numpy(), d[2]) for d in others]
            scores["fusion"] = FusionModel().fit([d[1] for d in others], Y).score(df)
            scores["reaction"] = BestSignal().fit([d[1] for d in others], Y).score(df)
        budget = a.budget
        clips = {k: select(t, s, budget) for k, s in scores.items()}
        out = f"results/{name}_excitement.png"
        excitement_plot(t, scores, goals or [], clips, name, out)
        print(f"-> {out}")


def _parse_ts(ts: str) -> float:
    """'1:23:45', '83:45' or '5025' -> seconds."""
    parts = [float(p) for p in str(ts).strip().split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def cmd_labels(a):
    """Hand-marked moments (labels/<match>.events.csv: time,type,note) -> label files.

    Every moment becomes a highlight segment from 10 s before to 10 s after it
    (the same window used for SoccerNet); goals are also kept separately for
    the goal-recall metric.
    """
    for path in a.events:
        ev = pd.read_csv(path, dtype=str).dropna(subset=["time"])
        ev["t"] = ev["time"].map(_parse_ts)
        ev["type"] = ev["type"].str.strip().str.lower()
        stem = Path(path).name.replace(".events.csv", "")
        segs = sorted((max(0.0, t - 10), t + 10) for t in ev["t"])
        merged = []
        for s, e in segs:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        out = Path(path).parent
        pd.DataFrame(merged, columns=["start", "end"]).to_csv(out / f"{stem}.csv", index=False)
        goals = ev.loc[ev["type"] == "goal", "t"]
        pd.DataFrame({"time": goals}).to_csv(out / f"{stem}.goals.csv", index=False)
        print(f"{stem}: {len(ev)} moments ({len(goals)} goals) -> {len(merged)} highlight segments")


def cmd_soccernet(a):
    from . import soccernet

    games = soccernet.sample_games(a.n, seed=a.seed)
    print(f"{len(games)} games (seed {a.seed}):")
    for g in games:
        print("  ", g)
    media = soccernet.prepare(games, download_video=not a.labels_only)
    print(f"labels -> labels/, {len(media)} half-match videos -> data/soccernet/")


def main(argv=None):
    p = argparse.ArgumentParser(prog="match-highlights", description=__doc__)
    p.add_argument("--cache", default="cache")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("features", help="compute and cache per-frame signals")
    f.add_argument("media", nargs="+")
    f.add_argument("--no-events", action="store_true")
    f.add_argument("--no-commentary", action="store_true")
    f.add_argument("--whisper", default="base.en")
    f.set_defaults(fn=cmd_features)

    al = sub.add_parser("align", help="label a match from its official highlights reel")
    al.add_argument("match")
    al.add_argument("reel")
    al.add_argument("--out")
    al.add_argument("--min-ncc", type=float, default=0.8)
    al.set_defaults(fn=cmd_align)

    b = sub.add_parser("benchmark", help="leave-one-match-out comparison of scorers")
    b.add_argument("media", nargs="+")
    b.add_argument("--labels", default="labels")
    b.add_argument("--model", default="models/fusion.joblib")
    b.add_argument("--budget", type=float, default=None,
                   help="reel length (s) for reel metrics; default = total labelled length")
    b.set_defaults(fn=cmd_benchmark)

    pl = sub.add_parser("plot", help="excitement curves vs labelled goals")
    pl.add_argument("media", nargs="+")
    pl.add_argument("--labels", default="labels")
    pl.add_argument("--budget", type=float, default=180)
    pl.set_defaults(fn=cmd_plot)

    lb = sub.add_parser("labels", help="convert hand-marked moments to label files")
    lb.add_argument("events", nargs="+", help="labels/<match>.events.csv files")
    lb.set_defaults(fn=cmd_labels)

    sn = sub.add_parser("soccernet", help="fetch SoccerNet EPL games and write highlight labels")
    sn.add_argument("--n", type=int, default=20, help="number of games")
    sn.add_argument("--seed", type=int, default=0)
    sn.add_argument("--labels-only", action="store_true")
    sn.set_defaults(fn=cmd_soccernet)

    r = sub.add_parser("run", help="make a highlight reel")
    r.add_argument("media")
    r.add_argument("--out", default="output/highlights.mp4")
    r.add_argument("--budget", type=float, default=300, help="reel length in seconds")
    r.add_argument("--scorer", choices=["learned", "loudness", "events", "fusion"], default="learned")
    r.add_argument("--learned-model", default="models/embedding_gbm.joblib")
    r.add_argument("--model", default="models/fusion.joblib")
    r.add_argument("--pre", type=float, default=12.0)
    r.add_argument("--post", type=float, default=8.0)
    r.add_argument("--timestamps-only", action="store_true")
    r.set_defaults(fn=cmd_run)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
