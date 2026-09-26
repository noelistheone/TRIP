#!/usr/bin/env python
"""How much of each target's 'true' embedding is its random initialization, and does TRIP
recover the learned part?

Every user row starts as u0 = 0.1 * randn(d) from a CPU generator seeded by
(uid + 1) * 9176 + seed (fl/trip/server.py:_ensure_user), so u0 can be rebuilt exactly.
Per cell (mean over the evaluated targets):
  init_cos      cos(u0, u)                    what knowing the init alone gives an attacker
  disp_ratio    median ||u - u0|| / ||u0||    how far training moved the row
  trip_cos      cos(u_hat, u)                 the reported recovery
  learned_cos   cos(P u_hat, P u), P = projection orthogonal to u0
                                              recovery of the part training added
Reads the stored runs under the results root TRIP_RESULTS (default <repo>/results), see SOURCES.
usage: init_share.py   (writes results/v4/analysis/init_share.json)
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
R = Path(os.environ.get("TRIP_RESULTS") or REPO / "results")
DS = ["lastfm", "ml", "delicious", "douban-book", "amazon-beauty", "amazon-book", "amazon-kindle",
      "gowalla", "iFashion", "yelp2018"]
SOURCES = {  # (result dirs, first existing wins; attack): the W=21 TRIP runs behind Table 2,
    # the per-client baselines of Table 2 (DLG re-run in v3), and TRIP under the E2b warm-up
    "paper": ([R / "v3" / "headline_w21", R / "v2" / "headline_w21", R / "headline_w21"], "paired_probe"),
    "paper-invert_grad": ([R / "v3" / "main", R / "main"], "invert_grad"),
    "paper-raifle": ([R / "v3" / "main", R / "main"], "raifle"),
    "paper-dlg": ([R / "v3" / "main", R / "main"], "dlg"),
    "paper-lti": ([R / "v3" / "main", R / "main"], "lti"),
    "e2b": ([R / "v4" / "uniform_l2" / "w21"], "paired_probe"),
}


def u0(uid: int, d: int, seed: int = 42) -> np.ndarray:
    g = torch.Generator(device="cpu").manual_seed((uid + 1) * 9176 + seed)
    return (torch.randn(d, generator=g) * 0.1).numpy().astype(np.float64)


def cosrows(A, B):
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)


def main():
    out = {}
    for tag, (dirs, attack) in SOURCES.items():
        for ds in DS:
            for m in ("mf", "lightgcn", "ncf"):
                f = next((p / f"{ds}_{m}_{attack}.npz" for p in dirs
                          if (p / f"{ds}_{m}_{attack}.npz").exists()), None)
                if f is None:
                    continue
                with np.load(f) as z:
                    U = z["true_U"].astype(np.float64)
                    Uh = z["U_hat"].astype(np.float64)
                    uids = z["attacked_uids"]
                    mask = z["eval_mask"].astype(bool) if z["eval_mask"].size else np.ones(len(U), bool)
                U0 = np.stack([u0(int(u), U.shape[1]) for u in uids])
                n0 = U0 / np.linalg.norm(U0, axis=1, keepdims=True)
                P = lambda X: X - (X * n0).sum(1, keepdims=True) * n0
                rec = {
                    "source": str(f.relative_to(R)),
                    "init_cos": float(cosrows(U0, U)[mask].mean()),
                    "disp_ratio_median": float(np.median((np.linalg.norm(U - U0, axis=1)
                                                          / np.linalg.norm(U0, axis=1))[mask])),
                    "trip_cos": float(cosrows(Uh, U)[mask].mean()),
                    "learned_cos": float(cosrows(P(Uh), P(U))[mask].mean()),
                }
                out[f"{tag}/{ds}/{m}"] = rec
                print(f"{tag:17s} {ds:14s} {m:9s} init_cos {rec['init_cos']:6.3f}  disp {rec['disp_ratio_median']:7.2f}"
                      f"  trip {rec['trip_cos']:6.3f}  learned-part {rec['learned_cos']:6.3f}   ({rec['source']})")
    p = R / "v4" / "analysis" / "init_share.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("->", p)


if __name__ == "__main__":
    main()
