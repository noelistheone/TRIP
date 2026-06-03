#!/usr/bin/env python
"""Aggregate all per-run JSON summaries into a single markdown report."""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl.eval import load_and_render_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="One or more result dirs.")
    ap.add_argument("--out", default="results/_summary.md")
    args = ap.parse_args()
    runs = load_and_render_all([Path(r) for r in args.runs], Path(args.out))
    print(f"Aggregated {len(runs)} runs → {args.out}")


if __name__ == "__main__":
    main()
