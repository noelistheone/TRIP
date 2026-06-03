#!/usr/bin/env python
"""Aggregate per-attack JSON summaries into a single cross-attack comparison table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl.eval import render_attack_comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="One or more directories containing <ds>_<model>_<attack>.json files.")
    ap.add_argument("--out", default="results/_attack_comparison.md")
    args = ap.parse_args()

    runs = []
    for d in args.runs:
        for p in sorted(Path(d).glob("*_*_*.json")):
            with open(p) as f:
                runs.append(json.load(f))

    runs.sort(key=lambda r: (r["dataset"], r["model"], r.get("attack", "paired_probe")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_attack_comparison(runs))
    print(f"Aggregated {len(runs)} per-attack runs → {out}")


if __name__ == "__main__":
    main()
