#!/usr/bin/env python
"""HE ablation: baseline vs HE-low vs HE-high on small datasets.

Runs (lastfm, ml, delicious) × (mf, lightgcn, ncf) at two HE noise levels
(1e-6 CKKS-realistic, 1e-4 stress). The baseline (noise=0) comes from the main
sweep in results/gpu0.
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl.eval import render_single, run_experiment
from fl.utils import pick_gpu


DATASETS = ["lastfm", "ml", "delicious"]
MODELS = ["mf", "lightgcn", "ncf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out-root", default="results/he_ablation")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--noise-levels", nargs="+", type=float, default=[1e-6, 1e-4],
                    help="HE noise_std values to test.")
    args = ap.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    if args.cpu:
        import torch
        device = torch.device("cpu")
    else:
        device = pick_gpu(default=args.gpu)
    print(f"device = {device}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for noise in args.noise_levels:
        tag = f"he_{noise:.0e}".replace("e-0", "e-").replace("+", "")
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(base_cfg)
        cfg["he"] = {"enabled": True, "noise_std": float(noise)}
        print(f"\n########## HE {tag} (noise_std={noise:.1e}) ##########")
        for ds in DATASETS:
            for mdl in MODELS:
                name = f"{ds}/{mdl}"
                print(f"\n---- {tag} / {name} ----")
                try:
                    summary = run_experiment(ds, mdl, cfg, device, out_dir)
                    md = render_single(summary)
                    (out_dir / f"{ds}_{mdl}.md").write_text(md)
                    print(f"[OK] {tag} {name}  cos_mean = {summary['cos']['cos_mean']:.4f}")
                except Exception as e:
                    traceback.print_exc()
                    (out_dir / f"{ds}_{mdl}.ERROR.txt").write_text(
                        f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
                    )
                    print(f"[FAIL] {tag} {name}: {e}")


if __name__ == "__main__":
    main()
