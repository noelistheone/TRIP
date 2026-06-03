#!/usr/bin/env python
"""Re-run delicious/{mf, ncf}/paired_probe with bumped negative-sampling ratio
to get usable ranking metrics on the 69k-item sparse catalog.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

import yaml
import torch
from fl.eval import run_experiment_multi_attack
from fl.utils import pick_gpu


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    # Bump neg sampling ratio (default 1, set to 5)
    cfg["bpr_neg_ratio"] = 5

    device = pick_gpu(default=0)
    out_dir = Path("results/run4_delicious_fix")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}, out={out_dir}, bpr_neg_ratio=5")

    for model in ["mf", "ncf"]:
        print(f"\n========== delicious/{model} ==========")
        try:
            run_experiment_multi_attack(
                "delicious", model, cfg, device, out_dir,
                attacks=["paired_probe"],
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            (out_dir / f"delicious_{model}.ERROR.txt").write_text(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
            )


if __name__ == "__main__":
    main()
