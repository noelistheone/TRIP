#!/usr/bin/env python
"""Anonymity ceiling (Theorem 2) for the v4 warm-up variants, against TRIP under TW+BC (t=16).

Same computation as scripts/analysis/anon_ceiling.py (unstratified beacon partition of the N targets,
beacon seed pact-v1), but the true embeddings come from the v4 run itself
(results/v4/<variant>/bc16/<ds>_<model>_paired_probe.npz), so each run is compared with the
ceiling of its own checkpoint. CPU only. Results root: TRIP_RESULTS (default <repo>/results).
usage: v4_ceiling.py [variant ...]      (default: uniform uniform_l2)
Writes results/v4/analysis/<variant>_ceiling.json and prints one line per cell.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("TRIP_RESULTS") or REPO / "results")
sys.path.insert(0, str(REPO))
os.environ.setdefault("FL_DATA_ROOT", str(REPO / "data_local"))
from fl.pact.blocks import beacon_partition

V4 = RESULTS / "v4"
DS = ["lastfm", "ml", "delicious", "amazon-beauty"]
MODELS = ["mf", "lightgcn", "ncf"]
T = 16


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def ceiling(U, t):
    assign = beacon_partition(list(range(len(U))), t, b"pact-v1", None)
    V = unit(U)
    blocks = {}
    for i, b in assign.items():
        blocks.setdefault(b, []).append(i)
    per = np.zeros(len(U))
    for mem in blocks.values():
        per[mem] = np.linalg.norm(V[mem].mean(axis=0))
    return per


def main(variants):
    for var in variants:
        out = {}
        for ds in DS:
            for m in MODELS:
                f = V4 / var / "bc16" / f"{ds}_{m}_paired_probe.npz"
                j = V4 / var / "bc16" / f"{ds}_{m}_paired_probe.json"
                if not f.exists():
                    continue
                with np.load(f) as z:
                    U = z["true_U"].astype(np.float64)
                    mask = z["eval_mask"].astype(bool) if z["eval_mask"].size else np.ones(len(U), bool)
                    measured = float(z["cos"][mask].mean())
                summ = json.loads(j.read_text()) if j.exists() else {}
                aborted = summ.get("tb_aborted_rounds")
                ceil = float(ceiling(U, T)[mask].mean())
                null = float(np.mean((unit(np.repeat(U.mean(0, keepdims=True), len(U), 0)) * unit(U)).sum(1)[mask]))
                best_const = float(np.linalg.norm(unit(U)[mask].mean(axis=0)))
                out[f"{ds}/{m}"] = {"trip_bc16": measured, "ceiling_t16": ceil, "gap": measured - ceil,
                                    "null": null, "best_constant": best_const, "tb_aborted_rounds": aborted}
                print(f"{var:10s} {ds:14s} {m:9s} TRIP {measured:6.3f}  ceiling {ceil:6.3f}  "
                      f"gap {measured - ceil:+.3f}  best-const {best_const:.3f}  TB-aborted {aborted}")
        p = V4 / "analysis" / f"{var}_ceiling.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        print("->", p)


if __name__ == "__main__":
    main(sys.argv[1:] or ["uniform", "uniform_l2"])
