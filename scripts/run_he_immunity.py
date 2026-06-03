#!/usr/bin/env python
"""HE-Immunity sweep (Exp 1 — Table 3).

For each (dataset, model): run warmup ONCE, snapshot state, then run a list of
(attack, scheme) cells off the same checkpoint. Schemes simulate three real
HE deployments via per-coord noise on the aggregate plus aggregate-only
visibility for baselines (the binding HE constraint):

    paillier  : noise_std = 1e-13   (Paillier 30-bit fixed-point)
    ckks40    : noise_std = 1e-16   (CKKS 40-bit scale, high precision)
    ckks20    : noise_std = 1e-10   (CKKS 20-bit scale, aggressive)

TRIP runs under all three schemes (cheap solves). Baselines run under ckks40
only (canonical realistic scheme); since aggregate-only is the binding HE
defense for them, all three schemes give the same collapse — see paper text.

Output: results/run7_he/<ds>_<mdl>_<attack>_<scheme>.{json,npz}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fl.data import load_dataset
from fl.eval import (
    _build_attack,
    _cos_per_row,
    _evaluate_recovery,
    _make_attack_cfg,
    _make_warmup_cfg,
    _resolve_per_dataset,
    _run_warmup,
    _select_attacked_uids,
)
from fl.models import make_model
from fl.trip.server import TRIPServer
from fl.utils import pick_gpu, set_seed


SCHEMES = {
    # tag       (noise_std, description)
    "paillier": (1.0e-13, "Paillier (30-bit fixed-point)"),
    "ckks40":   (1.0e-16, "CKKS (40-bit scale)"),
    "ckks20":   (1.0e-10, "CKKS (20-bit scale, aggressive)"),
}
DATASETS = [
    "lastfm", "ml", "delicious", "douban-book",
    "amazon-beauty", "amazon-book", "amazon-kindle",
    "gowalla", "iFashion", "yelp2018",
]
MODELS = ["mf", "lightgcn", "ncf"]
# (attack_name, list_of_schemes_to_run)
ATTACK_SCHEMES = [
    ("paired_probe", ["paillier", "ckks40", "ckks20"]),
    ("invert_grad",  ["ckks40"]),
    ("lti",          ["ckks40"]),
    ("raifle",       ["ckks40"]),
]


def run_one(ds: str, mdl: str, base_cfg: dict, device, out_dir: Path,
            skip_existing: bool = True) -> None:
    # Skip warmup entirely if every (attack, scheme) cell already exists.
    if skip_existing:
        all_done = True
        for attack_name, scheme_tags in ATTACK_SCHEMES:
            for scheme in scheme_tags:
                if not (out_dir / f"{ds}_{mdl}_{attack_name}_{scheme}.json").exists():
                    all_done = False
                    break
            if not all_done:
                break
        if all_done:
            print(f"[skip] {ds}/{mdl} (all cells exist)")
            return

    set_seed(int(base_cfg.get("seed", 42)))
    bundle = load_dataset(ds)
    print(f"[{ds}/{mdl}] users={bundle.n_users} items={bundle.n_items} "
          f"train_users={len(bundle.train_user_ids)}")

    d = int(base_cfg["d"])
    model = make_model(mdl, bundle.n_users, bundle.n_items, d, base_cfg).to(device)
    server = TRIPServer(model, bundle, base_cfg, device)

    attacked_uids = _select_attacked_uids(
        bundle, int(base_cfg["N_attack"]), int(base_cfg.get("seed", 42))
    )
    n_warmup = int(_resolve_per_dataset(
        base_cfg, "warmup_overrides", ds, base_cfg.get("warmup", 200)
    ))
    warm_cfg = _make_warmup_cfg(base_cfg, ds, mdl)
    server.cfg = warm_cfg
    print(f"[{ds}/{mdl}] warmup {n_warmup} rounds, opt={warm_cfg['optimizer']}, "
          f"lr={warm_cfg['lr']}")
    t_warm = time.time()
    _run_warmup(server, n_warmup, attacked_uids, int(warm_cfg["clients_per_round"]))
    warm_time = time.time() - t_warm
    print(f"[{ds}/{mdl}] warmup done in {warm_time:.1f}s")

    post_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    post_users = {uid: u.detach().cpu().clone() for uid, u in server.user_states.items()}
    post_round = server.round
    U_true = np.stack([
        post_users[uid].numpy() for uid in attacked_uids
    ]).astype(np.float32)

    attack_cfg_base = _make_attack_cfg(base_cfg, ds, mdl)

    for attack_name, scheme_tags in ATTACK_SCHEMES:
        for scheme in scheme_tags:
            noise_std, scheme_desc = SCHEMES[scheme]
            tag = f"{ds}_{mdl}_{attack_name}_{scheme}"
            json_out = out_dir / f"{tag}.json"
            if skip_existing and json_out.exists():
                print(f"[skip] {tag} (exists)")
                continue
            print(f"\n[{ds}/{mdl}/{attack_name}/{scheme}] starting")
            try:
                a_cfg = dict(attack_cfg_base)
                a_cfg["he"] = {
                    "enabled": True,
                    "aggregate_only": True,
                    "noise_std": float(noise_std),
                }
                a_model = make_model(
                    mdl, bundle.n_users, bundle.n_items, d, base_cfg
                ).to(device)
                a_model.load_state_dict(
                    {k: v.to(device) for k, v in post_state.items()}
                )
                a_model.n_items_original = bundle.n_items
                a_server = TRIPServer(a_model, bundle, a_cfg, device)
                a_server.user_states = {
                    uid: u.to(device).clone() for uid, u in post_users.items()
                }
                a_server.round = post_round

                attack = _build_attack(attack_name, a_cfg)
                attack.prepare(a_server, bundle, attacked_uids)
                U_hat = attack.solve(a_server, bundle, attacked_uids)
                eval_mask = getattr(attack, "eval_mask", None)
                metrics = _evaluate_recovery(
                    a_model, bundle, attacked_uids, U_hat, U_true, eval_mask
                )
                summary = {
                    "dataset": ds,
                    "model": mdl,
                    "attack": attack_name,
                    "scheme": scheme,
                    "noise_std": float(noise_std),
                    "scheme_desc": scheme_desc,
                    "n_attacked": int(len(attacked_uids)),
                    "n_items": int(bundle.n_items),
                    "n_users": int(bundle.n_users),
                    "warmup_rounds": int(n_warmup),
                    "warmup_time_sec": float(warm_time),
                    **metrics,
                    "timing": dict(attack.timing),
                }
                json_out.write_text(json.dumps(summary, indent=2))
                npz_out = out_dir / f"{tag}.npz"
                np.savez_compressed(
                    npz_out,
                    U_hat=U_hat,
                    true_U=U_true,
                    cos=_cos_per_row(U_hat, U_true),
                    attacked_uids=np.asarray(attacked_uids, dtype=np.int64),
                )
                print(f"[OK] {tag} cos_mean={summary['cos']['cos_mean']:.4f}")
            except Exception as e:
                traceback.print_exc()
                (out_dir / f"{tag}.ERROR.txt").write_text(
                    f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
                )
                print(f"[FAIL] {tag}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out-dir", default="results/run7_he")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--no-skip", action="store_true",
                    help="Re-run cells even if output JSON already exists.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = pick_gpu(default=args.gpu)
    print(f"device = {device}")
    print(f"datasets: {args.datasets}")
    print(f"models:   {args.models}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        for mdl in args.models:
            print(f"\n========== {ds}/{mdl} ==========")
            try:
                run_one(ds, mdl, cfg, device, out_dir,
                        skip_existing=not args.no_skip)
            except Exception as e:
                traceback.print_exc()
                (out_dir / f"{ds}_{mdl}.ERROR.txt").write_text(
                    f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
                )


if __name__ == "__main__":
    main()
