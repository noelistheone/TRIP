#!/usr/bin/env python
"""Identify and save baselines where Paired-Probe wins:
   - cos_ours >= cos_baseline - SLACK  (accuracy comparable or better)
   - total_time_ours < total_time_baseline (faster end-to-end)

Output: results/_wins.md (markdown table) and results/_wins.json.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SLACK = 0.05  # accept cos within 5 percentage points of baseline as "comparable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[
        "results/attack_comp_gpu0",
        "results/attack_comp_gpu1",
    ])
    ap.add_argument("--out-md", default="results/_wins.md")
    ap.add_argument("--out-json", default="results/_wins.json")
    ap.add_argument("--slack", type=float, default=SLACK,
                    help="Accept cos within this slack as comparable (default 0.05)")
    args = ap.parse_args()

    bucket = defaultdict(dict)  # (ds, model) -> attack -> summary
    for d in args.runs:
        for p in sorted(Path(d).glob("*_*_*.json")):
            with open(p) as f:
                r = json.load(f)
            bucket[(r["dataset"], r["model"])][r.get("attack", "paired_probe")] = r

    wins = []
    losses = []
    only_speed = []   # cos worse but time better
    only_acc = []     # cos better but time worse

    for (ds, mdl), attacks in sorted(bucket.items()):
        if "paired_probe" not in attacks:
            continue
        ours = attacks["paired_probe"]
        ours_cos = ours["cos"]["cos_mean"]
        ours_time = ours["timing"]["prepare_sec"] + ours["timing"]["solve_sec"]

        for a, s in attacks.items():
            if a == "paired_probe":
                continue
            their_cos = s["cos"]["cos_mean"]
            their_time = s["timing"]["prepare_sec"] + s["timing"]["solve_sec"]
            cos_ok = ours_cos >= their_cos - args.slack
            time_ok = ours_time < their_time
            entry = {
                "dataset": ds, "model": mdl, "baseline": a,
                "ours_cos": round(ours_cos, 4),
                "their_cos": round(their_cos, 4),
                "delta_cos": round(ours_cos - their_cos, 4),
                "ours_total_sec": round(ours_time, 3),
                "their_total_sec": round(their_time, 3),
                "speedup": round(their_time / max(ours_time, 1e-6), 1),
                "ours_solve_sec": round(ours["timing"]["solve_sec"], 4),
                "their_solve_sec": round(s["timing"]["solve_sec"], 4),
                "solve_speedup": round(s["timing"]["solve_sec"] / max(ours["timing"]["solve_sec"], 1e-6), 1),
            }
            if cos_ok and time_ok:
                wins.append(entry)
            elif time_ok:
                only_speed.append(entry)
            elif cos_ok:
                only_acc.append(entry)
            else:
                losses.append(entry)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"slack": args.slack, "wins": wins, "only_speed_win": only_speed,
                   "only_acc_win": only_acc, "losses": losses}, f, indent=2)

    lines = [
        f"# Paired-Probe wins (cos within {args.slack} slack AND faster end-to-end)\n",
        f"Computed across {sum(len(v) for v in bucket.values())} runs over {len(bucket)} "
        f"(dataset, model) pairs. Slack = {args.slack}.\n",
    ]

    def render_table(rows, title):
        lines.append(f"## {title} ({len(rows)} entries)\n")
        if not rows:
            lines.append("(none)\n")
            return
        lines.append("| dataset | model | baseline | ours cos | their cos | Δcos | ours total (s) | their total (s) | speedup | ours solve (s) | their solve (s) | solve speedup |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                f"| {r['dataset']} | {r['model']} | {r['baseline']} | "
                f"{r['ours_cos']:+.4f} | {r['their_cos']:+.4f} | {r['delta_cos']:+.4f} | "
                f"{r['ours_total_sec']:.2f} | {r['their_total_sec']:.2f} | {r['speedup']}× | "
                f"{r['ours_solve_sec']:.4f} | {r['their_solve_sec']:.4f} | {r['solve_speedup']}× |"
            )
        lines.append("")

    render_table(wins, "WINS — comparable accuracy AND faster")
    render_table(only_speed, "Speed-only wins — faster but cos lower than slack")
    render_table(only_acc, "Accuracy-only wins — comparable cos but slower")
    render_table(losses, "Losses — both worse")
    Path(args.out_md).write_text("\n".join(lines))
    print(f"wins={len(wins)} speed-only={len(only_speed)} acc-only={len(only_acc)} losses={len(losses)}")
    print(f"→ {args.out_md}")
    print(f"→ {args.out_json}")


if __name__ == "__main__":
    main()
