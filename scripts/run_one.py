#!/usr/bin/env python
"""Single (dataset, model) run — used for debugging and smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl.eval import render_single, run_experiment
from fl.utils import pick_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=["mf", "lightgcn", "ncf"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--gpu", type=int, default=None,
                    help="GPU index (explicit). If omitted, auto-picks least-used.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU.")
    ap.add_argument("--warmup", type=int, default=None,
                    help="Override warmup rounds (otherwise config per-dataset default).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.warmup is not None:
        cfg.setdefault("warmup_overrides", {})[args.dataset] = args.warmup

    if args.cpu:
        import torch
        device = torch.device("cpu")
    else:
        device = pick_gpu(default=args.gpu)
    print(f"device = {device}")

    out_dir = Path(args.out)
    summary = run_experiment(args.dataset, args.model, cfg, device, out_dir)
    md = render_single(summary)
    md_path = out_dir / f"{args.dataset}_{args.model}.md"
    md_path.write_text(md)
    print(f"\n=== SUMMARY ===")
    print(md)
    print(f"\nSaved: {md_path}")


if __name__ == "__main__":
    main()
