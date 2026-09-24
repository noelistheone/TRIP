#!/usr/bin/env python
"""Residual passive leakage under full PACT (Theorem 4, measured).

One federated round on MF under the full-PACT client (WS with the pinned
federated recipe, BC cohort of whole blocks). Every client's individual
update is computed with the production `local_train`, so we can count, for
each item row, how many cohort members wrote it. A row with exactly one
writer appears in the exact aggregate as that writer's own update. We report
how many participants own such rows and how well those rows align with the
writer's embedding (|cos| and sign pattern). The server can attribute such a
row to a block (by participation) but not to a member of it.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fl.client import local_train
from fl.data import load_dataset
from fl.eval import _load_warmup, _make_warmup_cfg, _resolve_per_dataset, _warmup_cache_path
from fl.models import make_model
from fl.pact import policy_from_cfg
from fl.pact.blocks import StickyPartition, activity_bucket
from fl.trip.server import TRIPServer
from fl.utils import pick_gpu, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["lastfm", "ml", "delicious", "amazon-beauty"])
    ap.add_argument("--config", default="configs/pact_full.yaml")
    ap.add_argument("--cohort", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", default="/data/lawrence/TRIP/results/v2/residual_leak.json")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    device = pick_gpu(default=a.gpu)
    res = {}
    for ds in a.datasets:
        set_seed(a.seed)
        bundle = load_dataset(ds)
        model = make_model("mf", bundle.n_users, bundle.n_items, int(cfg["d"]), cfg).to(device)
        model.n_items_original = bundle.n_items
        warm_cfg = _make_warmup_cfg(cfg, ds, "mf")
        fed_cfg = dict(warm_cfg); fed_cfg["weight_decay"] = 0.0
        server = TRIPServer(model, bundle, fed_cfg, device)
        n_warm = int(_resolve_per_dataset(cfg, "warmup_overrides", ds, cfg.get("warmup", 200)))
        cache = _warmup_cache_path(ds, "mf", cfg, warm_cfg, n_warm, bundle)
        if _load_warmup(cache, model, server, device) is None:
            print(f"no warm-up cache for {ds}"); continue
        policy = policy_from_cfg(cfg, fed_cfg)
        train_uids = sorted(bundle.train_user_ids)
        strata = {u: activity_bucket(len(bundle.train_pos.get(u, []))) for u in train_uids}
        part = StickyPartition.build(train_uids, policy.t, policy.beacon, strata if policy.stratify else None)
        policy.partition = part
        rng = np.random.default_rng(a.seed + 991)
        bids = list(part.members.keys()); rng.shuffle(bids)
        cohort, block_of = [], {}
        for b in bids:
            if len(cohort) >= a.cohort:
                break
            for u in part.members[b]:
                cohort.append(u); block_of[u] = b
        glob = {k: model.state_dict()[k].detach().clone() for k in model.shared_keys()}
        D, U0 = {}, {}
        for uid in cohort:
            items = bundle.train_pos.get(uid, [])
            if not items:
                continue
            u0 = server._ensure_user(uid).detach().clone()
            delta, _ = local_train(model, glob, u0, uid, items, bundle.n_items, fed_cfg,
                                   round_idx=server.round, policy=policy)
            D[uid] = delta["item_emb.weight"].detach().double().cpu()
            U0[uid] = u0.double().cpu()
        n_items = bundle.n_items
        writers = np.zeros(n_items, dtype=np.int64)
        written = {}
        for uid, d in D.items():
            w = (d.abs().sum(1) > 0).numpy()
            written[uid] = w
            writers += w
        single = writers == 1
        per_client, cos_all, sign_all = [], [], []
        for uid, d in D.items():
            rows = np.where(written[uid] & single)[0]
            u = U0[uid]
            if len(rows):
                R = d[rows]
                c = ((R @ u) / (R.norm(dim=1) * u.norm() + 1e-30)).abs().numpy()
                sg = (torch.sign(R) * torch.sign(u)[None, :]).numpy()
                agree = np.maximum((sg > 0).mean(1), (sg < 0).mean(1))
                cos_all.extend(c.tolist()); sign_all.extend(agree.tolist())
            per_client.append(len(rows))
        per_client = np.asarray(per_client)
        # The best labelled use of a block's sole-writer directions is bounded by the
        # anonymity ceiling of Theorem 2: report it for the cohort's blocks.
        V = {u: (U0[u] / U0[u].norm()).numpy() for u in U0}
        ceil = []
        for b in {block_of[u] for u in D}:
            mem = [u for u in part.members[b] if u in V]
            m = np.mean([V[u] for u in mem], axis=0)
            ceil.extend([float(np.linalg.norm(m))] * len(mem))
        res[ds] = {
            "cohort": len(D), "t": policy.t, "recipe": [policy.optimizer, policy.lr, policy.weight_decay, policy.local_epochs],
            "rows_written": int((writers > 0).sum()), "rows_single_writer": int(single.sum()),
            "frac_clients_ge1_single": float((per_client >= 1).mean()),
            "frac_clients_ge2_single": float((per_client >= 2).mean()),
            "single_rows_per_client_median": float(np.median(per_client)),
            "abs_cos_single_mean": float(np.mean(cos_all)) if cos_all else None,
            "abs_cos_single_median": float(np.median(cos_all)) if cos_all else None,
            "sign_agree_single_mean": float(np.mean(sign_all)) if sign_all else None,
            "anon_ceiling_cohort": float(np.mean(ceil)),
        }
        print(ds, json.dumps(res[ds]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    main()
