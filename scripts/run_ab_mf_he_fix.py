#!/usr/bin/env python
"""Re-run amazon-book/mf under all 3 HE schemes with warmup=5 to match the
plaintext baseline (results/run6_amazon_book_fix). The original run7_he sweep
used the default warmup=30 which causes heavy-user clustering on amazon-book
and gives cos=0.53 instead of the 0.82 achievable with warmup=5.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from scripts.run_he_immunity import run_one


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    # Override amazon-book warmup to 5 (matches run6_amazon_book_fix).
    cfg["warmup_overrides"]["amazon-book"] = 5

    from fl.utils import pick_gpu
    device = pick_gpu(default=0)
    out_dir = Path("results/run7_he")

    # Force re-run by deleting existing amazon-book_mf_*.json files.
    for p in out_dir.glob("amazon-book_mf_*.json"):
        print(f"removing {p}")
        p.unlink()
    for p in out_dir.glob("amazon-book_mf_*.npz"):
        p.unlink()

    run_one("amazon-book", "mf", cfg, device, out_dir, skip_existing=False)


if __name__ == "__main__":
    main()
