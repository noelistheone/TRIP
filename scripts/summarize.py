#!/usr/bin/env python
"""Aggregate all per-run JSON summaries into a single markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl.eval import render_attack_comparison


def load_runs(dirs):
    runs = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            if f.name.startswith("_"):
                continue
            try:
                runs.append(json.loads(f.read_text()))
            except Exception as e:
                print(f"[skip] {f}: {e}")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="One or more result dirs.")
    ap.add_argument("--out", default="results/_summary.md")
    args = ap.parse_args()
    runs = load_runs(args.runs)
    md = render_attack_comparison(runs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"Aggregated {len(runs)} runs -> {args.out}")


if __name__ == "__main__":
    main()
