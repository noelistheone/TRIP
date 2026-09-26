#!/usr/bin/env python
"""Anonymity ceiling of Theorem 2 (block invariance), from the true embeddings.

Reads the W=21 TRIP runs (<results>/v3/headline_w21, <results>/v2/headline_w21,
<results>/headline_w21; first existing *_paired_probe.npz per cell) and the datasets under
FL_DATA_ROOT (for the activity strata). Results root: TRIP_RESULTS (default <repo>/results).

For a block Gamma, m_Gamma = mean_{j in Gamma} u_j/||u_j||. If the server's view is
invariant under permutations of the private states inside every block, then for any
estimator and any member i, the expected cosine over a uniformly random assignment of
the block's states to its identities is u_hat^T m_Gamma <= ||m_Gamma||. The bound is
attained by the block oracle u_hat = m_Gamma/||m_Gamma|| (which knows every member's
direction but not who is who). We report mean_i ||m_Gamma(i)|| over the N targets,
with blocks built exactly as the attack runs build them (fl/eval.py: a beacon
partition of the N targets, stratified by activity when the config says so).

Output: <results>/analysis/anon_ceiling.json
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("FL_DATA_ROOT", str(REPO / "data_local"))
from fl.data import load_dataset
from fl.pact.blocks import beacon_partition, activity_bucket

R = Path(os.environ.get("TRIP_RESULTS") or REPO / "results")
OUT = R / "analysis" / "anon_ceiling.json"
DS = ["lastfm", "ml", "delicious", "amazon-beauty"]
MODELS = ["mf", "lightgcn", "ncf"]
TS = [1, 2, 4, 8, 16, 32, 64]


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def truth(ds, m):
    for p in (R / "v3" / "headline_w21", R / "v2" / "headline_w21", R / "headline_w21"):
        f = p / f"{ds}_{m}_paired_probe.npz"
        if f.exists():
            z = np.load(f)
            return z["true_U"].astype(np.float64), z["attacked_uids"], z["eval_mask"]
    return None


def ceiling(U, strata, t):
    N = len(U)
    assign = beacon_partition(list(range(N)), t, b"pact-v1", strata)
    V = unit(U)
    blocks = {}
    for i, b in assign.items():
        blocks.setdefault(b, []).append(i)
    per = np.zeros(N)
    sizes = []
    for b, mem in blocks.items():
        mvec = V[mem].mean(axis=0)
        per[mem] = np.linalg.norm(mvec)
        sizes.append(len(mem))
    return per, (min(sizes), max(sizes), len(blocks))


def main():
    out = {}
    for ds in DS:
        bundle = None
        for m in MODELS:
            tr = truth(ds, m)
            if tr is None:
                continue
            if bundle is None:
                bundle = load_dataset(ds)
            U, uids, mask = tr
            mask = mask.astype(bool) if mask is not None and mask.size else np.ones(len(U), bool)
            strata = {i: activity_bucket(len(bundle.train_pos.get(int(u), [])))
                      for i, u in enumerate(uids)}
            Ubar = U.mean(axis=0, keepdims=True)
            null = float(np.mean((unit(np.repeat(Ubar, len(U), 0)) * unit(U)).sum(1)[mask]))
            best_const = float(np.linalg.norm(unit(U)[mask].mean(axis=0)))
            cell = {"null": null, "best_constant": best_const, "unstratified": {}, "stratified": {}}
            for t in TS:
                for key, st in (("unstratified", None), ("stratified", strata)):
                    per, (smin, smax, nb) = ceiling(U, st, t)
                    cell[key][str(t)] = {"ceiling": float(per[mask].mean()),
                                         "block_size_min": smin, "block_size_max": smax,
                                         "n_blocks": nb}
            out[f"{ds}_{m}"] = cell
            print(ds, m, f"null={null:.3f} best_const={best_const:.3f}",
                  " ".join(f"t{t}:{cell['unstratified'][str(t)]['ceiling']:.3f}" for t in TS))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
