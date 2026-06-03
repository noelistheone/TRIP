#!/usr/bin/env python
"""Re-run amazon-book/mf with reduced warmup to fix the heavy-user
over-clustering issue (TRIP cos was 0.5316 with warmup=30; expected to
recover when user embeddings have less time to collapse into clusters).
"""
from __future__ import annotations
import os, sys
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

    # Override warmup specifically for amazon-book to avoid heavy-user
    # over-clustering: attacked users average 146 train items, and 30 epochs
    # × 3 local_epochs = 90 epochs of centralized BPR makes them co-converge
    # into a low-rank cluster that pinv(A)·G can't separate.
    cfg.setdefault("warmup_overrides", {})["amazon-book"] = 5

    device = pick_gpu(default=0)
    out_dir = Path("results/run6_amazon_book_fix")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}, out={out_dir}, warmup=10 (reduced from 30)")

    # Only paired_probe needs to be re-run (other attacks unaffected by user-emb clustering)
    run_experiment_multi_attack(
        "amazon-book", "mf", cfg, device, out_dir,
        attacks=["paired_probe"],
    )


if __name__ == "__main__":
    main()
