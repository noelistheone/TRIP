#!/usr/bin/env python
"""E3 follow-up: the larger-budget baselines against TRIP (W=21) on the same targets.

The E3 runs attack only the first 100 targets. For each dataset x {LightGCN, NCF} this matches
targets by user id between results/v4/budget/<ds>_<m>_<attack>.npz and the Table 2 TRIP run
(first existing of results/v3/headline_w21, results/v2/headline_w21, results/headline_w21), and
reports, on those targets:
  cos        mean cosine with the true embedding (what Table 2 reports)
  learned    cosine after projecting out each target's random initialization u0
             (see scripts/v4/init_share.py; matters on NCF, whose embeddings barely leave u0)
for TRIP, the 300-step baselines (Table 2 runs, same targets) and the E3 baselines.
Results root: TRIP_RESULTS (default <repo>/results).
usage: budget_vs_trip.py   (writes results/v4/analysis/budget_vs_trip.json)
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
AB = {"lastfm": "LFM", "ml": "ML", "delicious": "Del", "douban-book": "Dou", "amazon-beauty": "Bea",
      "amazon-book": "ABk", "amazon-kindle": "Kin", "gowalla": "Gow", "iFashion": "iFa", "yelp2018": "Yelp"}
ATT = ["invert_grad", "raifle", "dlg"]
TRIP_DIRS = [R / "v3" / "headline_w21", R / "v2" / "headline_w21", R / "headline_w21"]
REF_DIRS = [R / "v3" / "main", R / "main"]


def u0(uid, d, seed=42):
    g = torch.Generator(device="cpu").manual_seed((uid + 1) * 9176 + seed)
    return (torch.randn(d, generator=g) * 0.1).numpy().astype(np.float64)


def first(dirs, name):
    return next((p / name for p in dirs if (p / name).exists()), None)


def load(f):
    with np.load(f) as z:
        m = z["eval_mask"].astype(bool) if z["eval_mask"].size else np.ones(len(z["true_U"]), bool)
        return {int(u): (z["U_hat"][i].astype(np.float64), z["true_U"][i].astype(np.float64))
                for i, u in enumerate(z["attacked_uids"]) if m[i]}


def scores(rows, uids):
    c, l = [], []
    for u in uids:
        h, t = rows[u]
        c.append(h @ t / (np.linalg.norm(h) * np.linalg.norm(t) + 1e-12))
        n0 = u0(u, len(t)); n0 /= np.linalg.norm(n0)
        hp, tp = h - (h @ n0) * n0, t - (t @ n0) * n0
        l.append(hp @ tp / (np.linalg.norm(hp) * np.linalg.norm(tp) + 1e-12))
    return float(np.mean(c)), float(np.mean(l))


def main():
    out = {}
    for m in ("lightgcn", "ncf"):
        print(f"\n== {m}: cosine on the E3 targets (learned part in brackets) ==")
        print(f"{'':6s}{'TRIP W21':>16s}" + "".join(f"{a + ' 300':>16s}{a + ' E3':>16s}" for a in ATT))
        for ds in DS:
            ft = first(TRIP_DIRS, f"{ds}_{m}_paired_probe.npz")
            if ft is None:
                continue
            trip = load(ft)
            big = {a: load(f) for a in ATT if (f := R / "v4" / "budget" / f"{ds}_{m}_{a}.npz").exists()}
            ref = {a: load(f) for a in ATT if (f := first(REF_DIRS, f"{ds}_{m}_{a}.npz")) is not None}
            if not big:
                continue
            uids = sorted(set(trip).intersection(*[set(b) for b in big.values()], *[set(r) for r in ref.values()]))
            rec = {"n": len(uids), "trip": scores(trip, uids)}
            for a in ATT:
                if a in ref:
                    rec[f"{a}_300"] = scores(ref[a], uids)
                if a in big:
                    rec[f"{a}_E3"] = scores(big[a], uids)
            out[f"{ds}/{m}"] = rec
            f = lambda k: (f"{rec[k][0]:.3f}({rec[k][1]:.2f})" if k in rec else "-").rjust(16)
            print(f"{AB[ds]:6s}{f('trip')}" + "".join(f(f"{a}_300") + f(f"{a}_E3") for a in ATT) + f"   n={len(uids)}")
    p = R / "v4" / "analysis" / "budget_vs_trip.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("->", p)


if __name__ == "__main__":
    main()
