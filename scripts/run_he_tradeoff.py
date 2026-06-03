#!/usr/bin/env python
"""Privacy-Utility Tradeoff sweep (Exp 2 — Figure 1).

Hybrid pipeline: centralized BPR warmup (clean, converges fast) followed by
K_OP federated FedAvg rounds with HE noise injected per round. The federated
rounds simulate operational HE deployment: each round publishes a
homomorphically-decrypted aggregate that carries per-coord precision-loss
noise into the global item embeddings. After K_OP rounds the cumulative
noise on item_emb scales as sigma*sqrt(K_OP), so a fixed K_OP lets us sweep
sigma to draw a clean tradeoff curve.

  Step 1.  Centralized BPR warmup    -> converged model (no HE)
  Step 2.  K_OP federated rounds     -> model perturbed by HE noise
  Step 3.  Evaluate ranking metrics  -> recommender quality at this noise
  Step 4.  Run TRIP attack           -> recovery cos at this noise

Pure federated warmup-from-scratch was tried first (run8_tradeoff_centralized
_archive_fed) but converged ~100x worse than centralized BPR on lastfm
(recall@20=0.002 vs 0.18) -- per-round Adam reset wipes the optimizer
state. The hybrid form above gives a strong baseline (clean centralized
warmup) and noise that actually persists into the trained recommender,
which is what the tradeoff figure needs.

Datasets: lastfm, ml, douban-book.
Model:    MF only (one line per panel, cleaner figure).

Output: results/run8_tradeoff/<ds>_<mdl>_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fl.data import load_dataset
from fl.eval import (
    _build_attack,
    _cos_per_row,
    _make_attack_cfg,
    _make_warmup_cfg,
    _ranking_metrics,
    _resolve_per_dataset,
    _run_warmup,
    _score_user_against_items,
    _select_attacked_uids,
)
from fl.models import make_model
from fl.trip.server import TRIPServer
from fl.utils import pick_gpu, set_seed


DATASETS = ["lastfm", "ml", "douban-book"]
MODELS = ["mf"]
# (tag, noise_std) — the lower end (1e-12 .. 1e-6) is the realistic CKKS /
# Paillier deployment range; the upper end (1e-2 / 1) represents broken or
# adversarial HE precision and exists to characterize the noise-robustness
# boundary of TRIP. The recommender breaks well before TRIP at the higher end.
NOISE_LEVELS = [
    ("0",     0.0),
    ("1e-12", 1.0e-12),
    ("1e-8",  1.0e-8),
    ("1e-6",  1.0e-6),
    ("1e-4",  1.0e-4),
    ("1e-2",  1.0e-2),
    ("1",     1.0),
]
# K_OP: number of federated operational rounds run on top of the centralized
# warmup. Each round adds an N(0, sigma^2) i.i.d. perturbation to the
# aggregate; K_OP=50 gives a cumulative item-emb perturbation of
# sigma*sqrt(50) ~ 7*sigma. For sigma=1e-2 that's ~0.07 (significant on
# typical item-emb stds of ~0.1-0.5), consistent with what real FedAvg
# operational deployments would experience after a comparable number of
# encrypted aggregation rounds.
K_OP_DEFAULT = 50


def operational_rounds(server: TRIPServer,
                       n_rounds: int,
                       train_uids: List[int],
                       attacked_uids: List[int],
                       clients_per_round: int,
                       warm_cfg: dict,
                       noise_std: float,
                       seed: int = 42) -> None:
    """Run n_rounds of FedAvg with HE noise applied to the aggregate every
    round. Operational-deployment simulation: each round is a real FedAvg
    update plus per-coord noise. Force-includes attacked uids each round so
    their per-user rows continue to be touched (matches the centralized
    warmup's attacked-user oversampling so the post-noise user_states stay
    representative)."""
    cfg = dict(warm_cfg)
    # Use SGD with a small lr in the operational phase: we are not trying to
    # train more here, only to inject HE noise into the converged state.
    # Adam in this phase would aggressively unlearn the centralized warmup
    # and the no-HE baseline would degrade for reasons unrelated to noise.
    cfg["optimizer"] = "sgd"
    cfg["lr"] = 1.0e-3
    cfg["weight_decay"] = 0.0
    cfg["he"] = {
        "enabled": noise_std > 0,
        "noise_std": float(noise_std),
        "aggregate_only": False,
    }
    server.cfg = cfg
    rng = np.random.default_rng(seed + 9999)
    train_arr = np.asarray(train_uids, dtype=np.int64)
    attacked_set = set(int(u) for u in attacked_uids)

    for round_idx in tqdm(range(n_rounds),
                          desc=f"op rounds (sigma={noise_std:.0e})", ncols=80):
        if clients_per_round >= len(train_arr):
            sampled = [int(u) for u in train_arr]
        else:
            sampled = [int(u) for u in
                       rng.choice(train_arr, size=clients_per_round, replace=False)]
        sampled_set = set(sampled)
        for uid in attacked_set:
            if uid not in sampled_set:
                sampled.append(uid)
                sampled_set.add(uid)
        server.run_round(sampled, probe_assign={}, reps_per_pair=0,
                         apply_he_noise=(noise_std > 0))


def evaluate_ranking(model, bundle, attacked_uids: List[int],
                     post_users: dict) -> dict:
    """recall/ndcg/precision/hr @ {10,20,50} with TRUE user emb queries on
    the trained model."""
    ks = [10, 20, 50]
    n_items_orig = getattr(model, "n_items_original", model.n_items)
    rank_per: dict = {f"{m}@{k}": []
                      for m in ("recall", "ndcg", "precision", "hr") for k in ks}
    for uid in attacked_uids:
        gt = bundle.test_pos.get(uid, [])
        if not gt:
            continue
        u_q = post_users[uid].detach().cpu()
        scores = _score_user_against_items(
            model, u_q, n_items_orig, bundle.train_pos.get(uid, []),
        )
        m = _ranking_metrics(scores, gt, ks)
        for key, v in m.items():
            rank_per[key].append(float(v))
    return {k: (float(np.mean(v)) if v else 0.0) for k, v in rank_per.items()}


def run_dataset_model(ds: str, mdl: str, base_cfg: dict, device,
                      out_dir: Path, k_op: int,
                      skip_existing: bool = True) -> None:
    """Run all noise levels for one (ds, mdl). Centralized warmup is done
    ONCE per (ds, mdl); the operational-rounds phase reuses a snapshot of
    the warmup state per noise level (so the no-HE baseline corresponds to
    the converged centralized model + 50 fed rounds without noise)."""
    # Skip if all cells exist already.
    all_exist = all(
        (out_dir / f"{ds}_{mdl}_{tag}.json").exists()
        for tag, _ in NOISE_LEVELS
    )
    if skip_existing and all_exist:
        print(f"[skip] {ds}/{mdl} (all noise levels exist)")
        return

    set_seed(int(base_cfg.get("seed", 42)))
    bundle = load_dataset(ds)
    print(f"\n========== {ds}/{mdl} ==========")
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
    print(f"[{ds}/{mdl}] centralized warmup {n_warmup} rounds, "
          f"opt={warm_cfg['optimizer']}, lr={warm_cfg['lr']}")
    t_warm = time.time()
    _run_warmup(server, n_warmup, attacked_uids, int(warm_cfg["clients_per_round"]))
    warm_time = time.time() - t_warm
    print(f"[{ds}/{mdl}] centralized warmup done in {warm_time:.1f}s")

    # Snapshot post-warmup state — every noise level reuses this clean checkpoint.
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    base_users = {uid: u.detach().cpu().clone() for uid, u in server.user_states.items()}
    base_round = server.round
    cpr = int(warm_cfg["clients_per_round"])

    for tag, noise in NOISE_LEVELS:
        out_json = out_dir / f"{ds}_{mdl}_{tag}.json"
        if skip_existing and out_json.exists():
            print(f"[skip] {ds}/{mdl}/noise={tag}")
            continue
        print(f"\n---- {ds}/{mdl}/noise={tag} ----")
        try:
            # Restore from centralized-warmup snapshot
            model.load_state_dict({k: v.to(device) for k, v in base_state.items()})
            server.user_states = {uid: u.to(device).clone() for uid, u in base_users.items()}
            server.round = base_round

            # Operational rounds with HE noise
            t_op = time.time()
            operational_rounds(
                server, k_op,
                train_uids=list(bundle.train_user_ids),
                attacked_uids=attacked_uids,
                clients_per_round=cpr,
                warm_cfg=warm_cfg,
                noise_std=float(noise),
                seed=int(base_cfg.get("seed", 42)),
            )
            op_time = time.time() - t_op

            # Snapshot post-operational state (this is what TRIP attacks)
            post_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            post_users = {uid: u.detach().cpu().clone()
                          for uid, u in server.user_states.items()}
            for uid in attacked_uids:
                if uid not in post_users:
                    post_users[uid] = torch.zeros(d)
            post_round = server.round
            U_true = np.stack([
                post_users[uid].numpy() for uid in attacked_uids
            ]).astype(np.float32)

            # Recommender quality on the noisy model
            rank_true = evaluate_ranking(model, bundle, attacked_uids, post_users)

            # TRIP attack on the noisy model. CRITICAL: the threat model
            # assumes HE is on for the WHOLE deployment — the server can't
            # toggle it off during probe-injection rounds. So attack-phase
            # cfg gets the same HE settings as operational rounds; the
            # T=75 attack rounds add per-round Gaussian noise to the
            # aggregated probe-row deltas (snapshot/restore preserves the
            # model state but NOT the noise injected into the captured
            # delta `G`, which TRIP solves over).
            attack_cfg = _make_attack_cfg(base_cfg, ds, mdl)
            attack_cfg["he"] = {
                "enabled": noise > 0,
                "noise_std": float(noise),
                "aggregate_only": False,
            }
            a_model = make_model(mdl, bundle.n_users, bundle.n_items, d, base_cfg).to(device)
            a_model.load_state_dict({k: v.to(device) for k, v in post_state.items()})
            a_model.n_items_original = bundle.n_items
            a_server = TRIPServer(a_model, bundle, attack_cfg, device)
            a_server.user_states = {uid: u.to(device).clone() for uid, u in post_users.items()}
            a_server.round = post_round

            attack = _build_attack("paired_probe", attack_cfg)
            t_att = time.time()
            attack.prepare(a_server, bundle, attacked_uids)
            U_hat = attack.solve(a_server, bundle, attacked_uids)
            attack_time = time.time() - t_att

            cos_per = _cos_per_row(U_hat, U_true)
            cos_summary = {
                "cos_mean":   float(np.mean(cos_per)) if cos_per.size else 0.0,
                "cos_median": float(np.median(cos_per)) if cos_per.size else 0.0,
                "cos_std":    float(np.std(cos_per)) if cos_per.size else 0.0,
            }

            summary = {
                "dataset": ds,
                "model": mdl,
                "he_tag": tag,
                "he_noise_std": float(noise),
                "warmup_rounds": int(n_warmup),
                "k_op_rounds":  int(k_op),
                "n_attacked":   int(len(attacked_uids)),
                "n_items":      int(bundle.n_items),
                "n_users":      int(bundle.n_users),
                "warmup_time_sec": float(warm_time),
                "op_time_sec":     float(op_time),
                "attack_time_sec": float(attack_time),
                "cos": cos_summary,
                "ranking_true_U": rank_true,
                "timing": dict(attack.timing),
            }
            out_json.write_text(json.dumps(summary, indent=2))
            npz_out = out_dir / f"{ds}_{mdl}_{tag}.npz"
            np.savez_compressed(
                npz_out,
                U_hat=U_hat,
                true_U=U_true,
                cos=cos_per,
                attacked_uids=np.asarray(attacked_uids, dtype=np.int64),
            )
            print(f"[OK] {ds}/{mdl}/noise={tag}  cos={cos_summary['cos_mean']:.4f}  "
                  f"recall@20={rank_true['recall@20']:.4f}  "
                  f"ndcg@20={rank_true['ndcg@20']:.4f}")
        except Exception as e:
            traceback.print_exc()
            (out_dir / f"{ds}_{mdl}_{tag}.ERROR.txt").write_text(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
            )
            print(f"[FAIL] {ds}/{mdl}/noise={tag}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out-dir", default="results/run8_tradeoff")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--k-op", type=int, default=K_OP_DEFAULT,
                    help="Operational federated rounds with HE noise per cell")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--no-skip", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    device = pick_gpu(default=args.gpu)
    print(f"device = {device}")
    print(f"datasets: {args.datasets}")
    print(f"models:   {args.models}")
    print(f"noise levels: {[t for t, _ in NOISE_LEVELS]}")
    print(f"k_op: {args.k_op} federated rounds per cell")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        for mdl in args.models:
            try:
                run_dataset_model(ds, mdl, base_cfg, device, out_dir,
                                  k_op=args.k_op,
                                  skip_existing=not args.no_skip)
            except Exception as e:
                traceback.print_exc()
                (out_dir / f"{ds}_{mdl}.ERROR.txt").write_text(
                    f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
                )


if __name__ == "__main__":
    main()
