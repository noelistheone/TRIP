#!/usr/bin/env python
"""Run a sweep of (dataset × model × attack) combinations."""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl.eval import render_single, run_experiment_multi_attack
from fl.utils import pick_gpu


DEFAULT_DATASETS = [
    "lastfm", "ml", "delicious", "douban-book",
    "amazon-beauty", "amazon-book", "amazon-kindle",
    "gowalla", "iFashion", "yelp2018",
]
DEFAULT_MODELS = ["mf", "lightgcn", "ncf"]
DEFAULT_ATTACKS = ["paired_probe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS,
                    choices=["paired_probe", "single_probe", "dlg", "idlg",
                             "invert_grad", "lti", "raifle"],
                    help="Attacks to run per (dataset, model). All attacks share the same warmup.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip (dataset, model) pairs whose outputs already exist.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.cpu:
        import torch
        device = torch.device("cpu")
    else:
        device = pick_gpu(default=args.gpu)
    print(f"device = {device}")
    print(f"attacks = {args.attacks}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        for mdl in args.models:
            tag = f"{ds}/{mdl}"
            # skip if all requested attacks already have output
            if args.skip_existing:
                have = {
                    a for a in args.attacks
                    if (out_dir / f"{ds}_{mdl}_{a}.json").exists()
                }
                if have == set(args.attacks):
                    print(f"[SKIP] {tag} (all attacks exist)")
                    continue

            print(f"\n========== {tag} ==========")
            try:
                summaries = run_experiment_multi_attack(
                    ds, mdl, cfg, device, out_dir, attacks=args.attacks,
                )
                for a, s in summaries.items():
                    md = render_single(s)
                    (out_dir / f"{ds}_{mdl}_{a}.md").write_text(md)
                    print(f"[OK] {tag}/{a}  cos_mean = {s['cos']['cos_mean']:.4f}")
            except Exception as e:
                traceback.print_exc()
                (out_dir / f"{ds}_{mdl}.ERROR.txt").write_text(
                    f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
                )
                print(f"[FAIL] {tag}: {e}")


if __name__ == "__main__":
    main()
