#!/usr/bin/env python
"""Offline re-measurements behind statements in the paper (no training).

Reads, under the results root TRIP_RESULTS (default <repo>/results): results/main (proj, sat),
results/v3/headline_w21 (lgcn), results/biasvar/W*_n1e-06 (noise), results/threatmodel/tm_*_W2[01]
and results/v3/threatmodel/tm3_*_W2[01] (ladder), results/{v3,v2,}/{headline_w21,main} (colspace),
results/v2/rounds/* (rounds), results/{v3/main,main} (sat); plus the configs under <repo>/configs,
the warm-up cache (FL_WARMUP_CACHE, default <repo>/warmup_cache) and the datasets (FL_DATA_ROOT)
for lgcn and sat. Every number is recomputed from the stored systems (G, A, true U) or from the
warm-up checkpoints.

  proj      measured cosine vs. projection ceiling cos(PU,U) on all ten MF datasets (W=20)
  lgcn      LightGCN target mismatch cos(u_tilde, u0) per dataset vs. TRIP W=21
  noise     effective gains K*W*||row_i(A_lam^+)|| and Prop. 2 predictions at sigma=1e-6
  mixed     rank / conditioning of mixed window schedules
  ladder    Adam rungs re-solved with K per-pair offset columns
  colspace  share of G outside col(A) (how structured the residual is), MF vs NCF
  rounds    weighted operator for K*W > N: rank and projection ceiling
  sat       BPR saturation on the default.yaml checkpoints (not run by default; needs a GPU)

usage: offline_checks.py [section ...]   (default: all sections except sat)
Output: <results>/analysis/offline.json  (sections already in the file are kept, selected ones rewritten)
"""
from __future__ import annotations
import glob, json, math, os, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("FL_DATA_ROOT", str(REPO / "data_local"))
R = Path(os.environ.get("TRIP_RESULTS") or REPO / "results")
OUT = R / "analysis" / "offline.json"
CONFIGS = REPO / "configs"
DS10 = ["lastfm", "ml", "delicious", "douban-book", "amazon-beauty", "amazon-kindle",
        "amazon-book", "gowalla", "yelp2018", "iFashion"]


def rc(X, Y):
    return (X * Y).sum(1) / np.maximum(np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1), 1e-300)


def proj_rowspace(A):
    # orthogonal projector onto row(A) via SVD (A is TK x N)
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = int((s > s.max() * 1e-9).sum())
    V = Vt[:r].T
    return V @ V.T, r, s[:r]


def window_A(N, K, W, T):
    ds = N // K
    A = np.zeros((T * K, N))
    for t in range(T):
        for k in range(K):
            for j in range(W):
                A[t * K + k, (t + k * ds + j) % N] = 1.0
    return A


def sec_proj():
    out = {}
    for ds in DS10:
        f = R / "main" / f"{ds}_mf_paired_probe.npz"
        if not f.exists():
            continue
        z = np.load(f)
        A = z["A"].astype(np.float64); U = z["true_U"].astype(np.float64)
        P, r, _ = proj_rowspace(A)
        pc = rc(P @ U, U)
        meas = z["cos"].astype(np.float64)
        n = np.linalg.norm(U, axis=1)
        out[ds] = {"rank": r, "proj_mean": float(pc.mean()), "meas_mean": float(meas.mean()),
                   "abs_diff_mean": float(abs(pc.mean() - meas.mean())),
                   "abs_diff_user_max": float(np.abs(pc - meas).max()),
                   "norm_cv": float(n.std() / n.mean())}
    return out


def sec_lgcn():
    import torch, yaml
    from fl.data import load_dataset
    from fl.eval import _make_warmup_cfg, _resolve_per_dataset, _warmup_cache_path
    cfg = yaml.safe_load(open(CONFIGS / "headline_w21.yaml"))
    out = {}
    for ds in DS10:
        base = R / "v3" / "headline_w21"   # v3: LightGCN re-warmed with star propagation
        jf = base / f"{ds}_lightgcn_paired_probe.json"
        zf = base / f"{ds}_lightgcn_paired_probe.npz"
        if not jf.exists():
            continue
        b = load_dataset(ds)
        warm = _make_warmup_cfg(cfg, ds, "lightgcn")
        n_warm = int(_resolve_per_dataset(cfg, "warmup_overrides", ds, cfg.get("warmup", 200)))
        p = _warmup_cache_path(ds, "lightgcn", cfg, warm, n_warm, b)
        if p is None or not p.exists():
            out[ds] = {"missing_cache": str(p)}; continue
        blob = torch.load(p, map_location="cpu", weights_only=False)
        V = blob["state"]["item_emb.weight"].double().numpy()
        z = np.load(zf)
        uids = z["attacked_uids"]; U0 = z["true_U"].astype(np.float64)
        ut = []
        for i, u in enumerate(uids):
            items = [j for j in b.train_pos.get(int(u), []) if j < V.shape[0]]
            s = V[items].sum(0) / math.sqrt(len(items)) if items else 0.0
            ut.append((2.0 * U0[i] + s) / 3.0)
        mm = rc(np.asarray(ut), U0)
        trip = json.loads(jf.read_text())["cos"]["cos_mean"]
        out[ds] = {"mismatch_mean": float(mm.mean()), "trip_w21": float(trip),
                   "diff": float(trip - mm.mean())}
    return out


def sec_noise():
    N, K, T, lam, eta = 500, 10, 150, 1e-6, 5e-3
    out = {}
    for W in (19, 20, 21, 25):
        A = window_A(N, K, W, T)
        Ad = np.linalg.solve(A.T @ A + lam * np.eye(N), A.T)
        row = np.linalg.norm(Ad, axis=1)
        s = np.linalg.svd(A, compute_uv=False)
        s_nz = s[s > s.max() * 1e-9]
        f = R / "biasvar" / f"W{W}_n1e-06" / "lastfm_mf_paired_probe.npz"
        pred = None
        if f.exists():
            z = np.load(f); U = z["true_U"].astype(np.float64)
            rng = np.random.default_rng(0)
            M = Ad @ A
            preds = []
            for _ in range(200):
                Xi = rng.normal(0.0, math.sqrt(2.0) * K * W * 1e-6, size=(A.shape[0], U.shape[1]))
                Uh = M @ (eta * U) + Ad @ Xi
                preds.append(float(rc(Uh, U).mean()))
            pred = float(np.mean(preds))
        meas = None
        jf = R / "biasvar" / f"W{W}_n1e-06" / "lastfm_mf_paired_probe.json"
        if jf.exists():
            meas = json.loads(jf.read_text())["cos"]["cos_mean"]
        out[str(W)] = {"inv_smin": float(1.0 / s_nz.min()), "row_gain_mean": float(row.mean()),
                       "eff_gain": float(K * W * row.mean()), "pred_1e-6": pred, "meas_1e-6": meas,
                       "rank": int(len(s_nz))}
    return out


def sec_mixed():
    N, K = 500, 10
    def stats(A):
        s = np.linalg.svd(A, compute_uv=False)
        nz = s[s > s.max() * 1e-9]
        return {"rank": int(len(nz)), "inv_smin": float(1 / nz.min())}
    A20 = window_A(N, K, 20, 150); A21 = window_A(N, K, 21, 150)
    Am = np.vstack([window_A(N, K, 20, 100), window_A(N, K, 21, 50)])
    return {"W20": stats(A20), "W21": stats(A21), "mixed_100x20_50x21": stats(Am)}


def sec_ladder():
    out = {}
    files = {}
    for f in sorted(glob.glob(str(R / "threatmodel/tm_*_W2[01]/*_mf_paired_probe.npz"))):
        files[(f.split("/")[-2], f.split("/")[-1])] = f
    for f in sorted(glob.glob(str(R / "v3/threatmodel/tm3_*_W2[01]/*_mf_paired_probe.npz"))):
        files[(f.split("/")[-2].replace("tm3_", "tm_", 1), f.split("/")[-1])] = f   # v3 replaces
    for (rung_w, fname), f in sorted(files.items()):
        ds = fname.split("_mf_")[0]
        z = np.load(f)
        A = z["A"].astype(np.float64); G = z["G"].astype(np.float64); U = z["true_U"].astype(np.float64)
        TK, N = A.shape; K = 10
        E = np.zeros((TK, K)); E[np.arange(TK), np.arange(TK) % K] = 1.0
        X = np.hstack([A, E])
        lam = 1e-6
        S = np.linalg.solve(X.T @ X + lam * np.eye(N + K), X.T @ G)[:N]
        plain = np.linalg.solve(A.T @ A + lam * np.eye(N), A.T @ G)
        null = float(rc(np.repeat(U.mean(0, keepdims=True), N, 0), U).mean())
        out[f"{rung_w}/{ds}"] = {"plain": float(rc(plain, U).mean()),
                                 "offset": float(rc(S, U).mean()),
                                 "offset_vs_sign": float(rc(S, np.sign(U)).mean()),
                                 "plain_vs_sign": float(rc(plain, np.sign(U)).mean()),
                                 "sign_vs_u": float(rc(np.sign(U), U).mean()),
                                 "stored": float(z["cos"].mean()), "null": null,
                                 "resid_plain": float(np.linalg.norm(G - A @ np.linalg.lstsq(A, G, rcond=None)[0]) / np.linalg.norm(G)),
                                 "resid_offset": float(np.linalg.norm(G - X @ np.linalg.lstsq(X, G, rcond=None)[0]) / np.linalg.norm(G))}
    return out


def sec_colspace():
    out = {}
    for sub, tag in (("v3/headline_w21", "W21"), ("v3/main", "W20"), ("v2/headline_w21", "W21"), ("v2/main", "W20"),
                     ("headline_w21", "W21"), ("main", "W20")):
        for f in sorted(glob.glob(str(R / sub / "*_paired_probe.npz"))):
            name = f.split("/")[-1].replace("_paired_probe.npz", "")
            key = f"{tag}/{name}"
            if key in out:
                continue
            z = np.load(f)
            A = z["A"].astype(np.float64); G = z["G"].astype(np.float64)
            coef = np.linalg.lstsq(A, G, rcond=None)[0]
            out[key] = float(np.linalg.norm(G - A @ coef) / np.linalg.norm(G))
    return out


def sec_rounds():
    """K*W > N: the saved npz A (pre-fix) was int8 and lost the 1/m weights, so the
    weighted operator is rebuilt from the allocator exactly as the attack realised it
    (every client's m_u*r <= 256 triples fit one mini-batch, so each weight is 1/m_u(t))."""
    import yaml
    from fl.trip import SlidingWindowAllocator
    out = {}
    for cfgname in ("K25_Tf1", "K50_Tf1"):
        c = yaml.safe_load(open(CONFIGS / f"rounds_{cfgname}.yaml"))
        K, W, Tf = int(c["K"]), int(c["W"]), int(c["T_factor"])
        al = SlidingWindowAllocator(N=500, K=K, W=W, T_factor=Tf)
        A = np.zeros((al.T * K, 500))
        for t in range(al.T):
            for lu, ks in al.assignments_for_round(t).items():
                for k in ks:
                    A[t * K + k, lu] = 1.0 / len(ks)
        P, r, _ = proj_rowspace(A)
        for ds in ("lastfm", "ml"):
            f = R / "v2" / "rounds" / cfgname / f"{ds}_mf_paired_probe.npz"
            if not f.exists():
                continue
            z = np.load(f); U = z["true_U"].astype(np.float64)
            out[f"{cfgname}/{ds}"] = {"K": K, "W": W, "rank": r, "proj": float(rc(P @ U, U).mean()),
                                      "meas": float(z["cos"].mean()), "max_m": int(max(len(v) for t in range(al.T) for v in al.assignments_for_round(t).values()))}
    return out


def sec_sat():
    """Share of users whose honest observed update is ~0 (BPR saturated) on the main
    (default.yaml) checkpoints: the 500 targets and 500 ordinary users (train uids 500-999),
    plus InvGrad's cosine split by whether the target's update was zero (v3 main runs)."""
    import torch, yaml
    from fl.data import load_dataset
    from fl.models import make_model
    from fl.trip.server import TRIPServer
    from fl import eval as E
    from fl.attacks.dlg import _observation_round
    cfg = yaml.safe_load(open(CONFIGS / "default.yaml"))
    dev = torch.device("cuda:0")
    out = {}
    for ds in DS10:
        b = load_dataset(ds)
        for mdl in ("mf", "lightgcn", "ncf"):
            model = make_model(mdl, b.n_users, b.n_items, cfg["d"], cfg).to(dev)
            server = TRIPServer(model, b, cfg, dev)
            nw = int(E._resolve_per_dataset(cfg, "warmup_overrides", ds, 200))
            wc = E._make_warmup_cfg(cfg, ds, mdl); server.cfg = wc
            if E._load_warmup(E._warmup_cache_path(ds, mdl, cfg, wc, nw, b), model, server, dev) is None:
                out[f"{ds}/{mdl}"] = {"missing_ckpt": True}; continue
            model.n_items_original = b.n_items
            acfg = E._make_attack_cfg(cfg, ds, mdl); server.cfg = acfg
            tg = E._select_attacked_uids(b, 500, 42)
            ordin = sorted(b.train_user_ids)[500:1000]
            rec = {}
            for tag, us in (("targets", tg), ("ordinary", ordin)):
                obs = _observation_round(server, b, us, acfg)
                mx = np.array([float(obs[u].abs().max()) for u in us])
                nrm = np.array([float(server.user_states[u].norm()) for u in us])
                rec[tag] = {"zero_share": float((mx < 1e-8).mean()), "median_norm": float(np.median(nrm))}
                if tag == "targets":
                    zero = mx < 1e-8
            for sub in ("v3/main", "main"):
                f = R / sub / f"{ds}_{mdl}_invert_grad.npz"
                if f.exists():
                    z = np.load(f); c = z["cos"]
                    if list(z["attacked_uids"]) == list(tg):
                        rec["ig_cos_zero"] = float(c[zero].mean()) if zero.any() else None
                        rec["ig_cos_nonzero"] = float(c[~zero].mean()) if (~zero).any() else None
                        rec["ig_src"] = sub
                    break
            out[f"{ds}/{mdl}"] = rec
            print(ds, mdl, rec, flush=True)
            del model, server; torch.cuda.empty_cache()
    return out


def main():
    sel = sys.argv[1:] or ["proj", "lgcn", "noise", "mixed", "ladder", "colspace", "rounds"]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for s in sel:
        res[s] = globals()[f"sec_{s}"]()
        print(f"== {s}"); print(json.dumps(res[s], indent=1)[:3000])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
