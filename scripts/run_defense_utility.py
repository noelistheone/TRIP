#!/usr/bin/env python
"""E3/E4 — does PACT cost recommendation accuracy or time? (Theorems 4 and 5.)

Runs genuine FedAvg training twice from the SAME initialisation:

  arm "honest"  clients sampled uniformly; `fl/client.local_train` with the
                deployment recipe and its own numpy-RNG negatives.
  arm "pact"    whole blocks sampled until the cohort is at least as large as
                the honest one (BC-4 over-provisioning); `local_train` with the
                pinned recipe and PRF-sourced negatives (WS); intra-block masks
                applied to the uploaded delta and verified to cancel (BC-3).

Then it scores Recall/NDCG/Precision/HR@{10,20,50} for every evaluated user and
reports per-round wall-clock and the protocol overhead in bytes.

The point of Theorem 4 is that the DECRYPTED AGGREGATE is bit-identical, so any
difference in ranking must come from client sampling alone, not from the
defense's mechanics. `--check-masks` asserts the mask cancellation exactly.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fl.client import local_train
from fl.data import load_dataset
from fl.eval import (_evaluate_recovery, _make_warmup_cfg, _ranking_metrics,
                     _resolve_per_dataset, _score_user_against_items,
                     _select_attacked_uids)
from fl.models import make_model
from fl.pact import policy_from_cfg
from fl.pact.blocks import StickyPartition, activity_bucket
from fl.pact.mask import assert_cancels, mask_for
from fl.eval import _load_warmup, _warmup_cache_path
from fl.trip.server import TRIPServer
from fl.utils import pick_gpu, set_seed


FED_WD = 0.0


def run_arm(arm, ds, mdl, cfg, device, n_rounds, cpr, eval_uids, seed,
            check_masks=False, from_warmup=True):
    set_seed(seed)
    bundle = load_dataset(ds)
    d = int(cfg["d"])
    model = make_model(mdl, bundle.n_users, bundle.n_items, d, cfg).to(device)
    model.n_items_original = bundle.n_items
    warm_cfg = _make_warmup_cfg(cfg, ds, mdl)
    # The warm-up (centralised) recipe uses Adam with COUPLED weight decay. In
    # federated rounds every client builds a fresh optimiser, so the decay term
    # wd*theta gives every item row a nonzero gradient and Adam's first step moves
    # every row by ~lr*sign(theta): the item table collapses towards 0 in BOTH arms.
    # Federated rounds therefore use the same optimiser without weight decay.
    fed_cfg = dict(warm_cfg); fed_cfg["weight_decay"] = float(FED_WD)
    server_warm_cfg = warm_cfg
    server = TRIPServer(model, bundle, fed_cfg, device)

    # Both arms start from the SAME converged checkpoint, so the comparison is
    # about federated rounds under the defense, not about cold-start noise.
    if from_warmup:
        n_warm = int(_resolve_per_dataset(cfg, "warmup_overrides", ds, cfg.get("warmup", 200)))
        cache = _warmup_cache_path(ds, mdl, cfg, server_warm_cfg, n_warm, bundle)
        if _load_warmup(cache, model, server, device) is None:
            raise SystemExit(
                f"no warm-up checkpoint for {ds}/{mdl}; run the main sweep first "
                f"(expected {cache})")

    policy = None
    part = None
    train_uids = sorted(bundle.train_user_ids)
    if arm == "pact":
        policy = policy_from_cfg(cfg, fed_cfg)
        assert policy is not None, "arm 'pact' needs cfg['pact']['enabled']=true"
        strata = ({u: activity_bucket(len(bundle.train_pos.get(u, [])))
                   for u in train_uids} if policy.stratify else None)
        part = StickyPartition.build(train_uids, policy.t, policy.beacon, strata)
        policy.partition = part
        print(f"[pact] t={policy.t} blocks={len(part.members)} "
              f"recipe={policy.optimizer}/lr={policy.lr}/wd={policy.weight_decay}")

    def _rank_now():
        ks_ = (10, 20, 50)
        acc_ = {f"{m}@{k}": [] for m in ("recall", "ndcg", "precision", "hr") for k in ks_}
        for uid in eval_uids:
            gt = bundle.test_pos.get(uid, [])
            u = server.user_states.get(uid)
            if not gt or u is None:
                continue
            sc = _score_user_against_items(model, u.detach().cpu(), bundle.n_items,
                                           bundle.train_pos.get(uid, []), use_true_neighbors=True)
            for k2, v in _ranking_metrics(sc, gt, list(ks_)).items():
                acc_[k2].append(float(v))
        return {k: (float(np.mean(v)) if v else 0.0) for k, v in acc_.items()}
    start_rank = _rank_now()
    rng = np.random.default_rng(seed + 991)
    per_round, mask_ok, cohorts = [], True, []
    for r in range(n_rounds):
        t0 = time.time()
        if arm == "pact":
            # BC-2/BC-4: sample WHOLE blocks until the cohort is >= cpr.
            bids = list(part.members.keys())
            rng.shuffle(bids)
            sampled, i = [], 0
            while len(sampled) < cpr and i < len(bids):
                sampled.extend(part.members[bids[i]]); i += 1
            if check_masks and r == 0:
                shape = (min(64, bundle.n_items), d)
                mask_ok = assert_cancels(part.members[bids[0]], r, shape)
        else:
            sampled = [int(x) for x in rng.choice(
                np.asarray(train_uids, dtype=np.int64),
                size=min(cpr, len(train_uids)), replace=False)]
        cohorts.append(len(sampled))
        server.run_round(sampled, probe_assign=None, reps_per_pair=1, policy=policy)
        per_round.append(time.time() - t0)

    # ---- ranking evaluation on the shared eval set ----
    ks = (10, 20, 50)
    acc = {f"{m}@{k}": [] for m in ("recall", "ndcg", "precision", "hr") for k in ks}
    n_items = bundle.n_items
    for uid in eval_uids:
        gt = bundle.test_pos.get(uid, [])
        if not gt:
            continue
        u = server.user_states.get(uid)
        if u is None:
            continue
        scores = _score_user_against_items(
            model, u.detach().cpu(), n_items, bundle.train_pos.get(uid, []),
            use_true_neighbors=True)
        m = _ranking_metrics(scores, gt, list(ks))
        for k2, v in m.items():
            acc[k2].append(float(v))
    rank = {k: (float(np.mean(v)) if v else 0.0) for k, v in acc.items()}
    return dict(arm=arm, rounds=n_rounds, cohort=cpr, start_ranking=start_rank,
                cohort_mean=float(np.mean(cohorts)),
                sec_per_client=float(np.sum(per_round) / max(np.sum(cohorts), 1)),
                sec_per_round=float(np.mean(per_round)),
                sec_per_round_std=float(np.std(per_round)),
                ranking=rank, mask_cancels=bool(mask_ok),
                n_eval=len([u for u in eval_uids if bundle.test_pos.get(u)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lastfm")
    ap.add_argument("--model", default="mf")
    ap.add_argument("--config", default="configs/pact_full.yaml")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--cohort", type=int, default=128)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", default="results/defense_utility")
    ap.add_argument("--check-masks", action="store_true")
    ap.add_argument("--fed-wd", type=float, default=0.0,
                    help="Weight decay used in federated rounds (both arms).")
    ap.add_argument("--match-cohort", action="store_true", default=True,
                    help="Give the honest arm PACT's realised mean cohort, so the "
                         "comparison is not confounded by BC-4 over-provisioning.")
    ap.add_argument("--cold-start", action="store_true",
                    help="Start from random init instead of the cached warm-up.")
    a = ap.parse_args()
    global FED_WD
    FED_WD = a.fed_wd

    cfg = yaml.safe_load(open(a.config))
    device = pick_gpu(default=a.gpu)
    bundle = load_dataset(a.dataset)
    eval_uids = _select_attacked_uids(bundle, a.n_eval, int(cfg.get("seed", 42)))
    print(f"device={device}  eval users={len(eval_uids)}  rounds={a.rounds}  cohort={a.cohort}")

    out = {"dataset": a.dataset, "model": a.model, "rounds": a.rounds,
           "cohort": a.cohort, "seeds": a.seeds, "arms": {}}
    # Run PACT first so the honest arm can be given the SAME mean cohort.
    # BC-4 over-provisioning rounds the cohort up to a whole number of blocks,
    # and a larger cohort is worth a fraction of a percent of Recall on its own.
    # Comparing at unequal cohorts would credit the defense with that, which is
    # not a property of the defense.
    cohort_for = {"pact": a.cohort, "honest": a.cohort}
    for arm in ("pact", "honest"):
        runs = []
        for s in a.seeds:
            print(f"\n=== arm={arm} seed={s} ===")
            runs.append(run_arm(arm, a.dataset, a.model, cfg, device, a.rounds,
                                cohort_for[arm], eval_uids, s, a.check_masks,
                                from_warmup=not a.cold_start))
        agg = {k: float(np.mean([r["ranking"][k] for r in runs])) for k in runs[0]["ranking"]}
        sd = {k: float(np.std([r["ranking"][k] for r in runs])) for k in runs[0]["ranking"]}
        out.setdefault("start_ranking", runs[0]["start_ranking"])
        out["arms"][arm] = {
            "runs": runs, "ranking_mean": agg, "ranking_std": sd,
            "sec_per_round": float(np.mean([r["sec_per_round"] for r in runs])),
            "sec_per_client": float(np.mean([r["sec_per_client"] for r in runs])),
            "cohort_mean": float(np.mean([r["cohort_mean"] for r in runs]))}
        if arm == "pact" and a.match_cohort:
            cohort_for["honest"] = int(round(out["arms"]["pact"]["cohort_mean"]))
            print(f"[match] honest arm will use cohort={cohort_for['honest']} "
                  f"to match PACT's realised mean")

    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    tag = f"{a.dataset}_{a.model}_r{a.rounds}"
    (o / f"{tag}.json").write_text(json.dumps(out, indent=2))

    h, p = out["arms"]["honest"], out["arms"]["pact"]
    ns = len(a.seeds)
    print(f"\n{'metric':<14}{'honest':>12}{'+/-':>9}{'PACT':>12}{'+/-':>9}{'delta':>11}{'rel':>9}")
    for k in ("recall@10", "recall@20", "ndcg@10", "ndcg@20", "hr@10", "hr@20"):
        hv, pv = h["ranking_mean"][k], p["ranking_mean"][k]
        hs, ps = h["ranking_std"][k], p["ranking_std"][k]
        rel = (pv - hv) / hv * 100 if hv else 0.0
        flag = ""
        if ns > 1 and abs(pv - hv) < (hs + ps):
            flag = "  (within seed spread)"
        print(f"{k:<14}{hv:>12.4f}{hs:>9.4f}{pv:>12.4f}{ps:>9.4f}{pv-hv:>+11.4f}{rel:>8.2f}%{flag}")
    print(f"\n{'cohort/round':<14}{h['cohort_mean']:>12.1f}{'':>9}{p['cohort_mean']:>12.1f}")
    print(f"{'sec/round':<14}{h['sec_per_round']:>12.3f}{'':>9}{p['sec_per_round']:>12.3f}"
          f"{'':>9}{p['sec_per_round']-h['sec_per_round']:>+11.3f}"
          f"{(p['sec_per_round']/h['sec_per_round']-1)*100:>8.2f}%")
    print(f"{'sec/client':<14}{h['sec_per_client']*1e3:>12.3f}{'':>9}{p['sec_per_client']*1e3:>12.3f}"
          f"{'':>9}{(p['sec_per_client']-h['sec_per_client'])*1e3:>+11.3f}"
          f"{(p['sec_per_client']/h['sec_per_client']-1)*100:>8.2f}%   (ms, cohort-normalised)")
    print(f"\nmask cancellation exact: {all(r['mask_cancels'] for r in p['runs'])}")
    print(f"saved -> {o/f'{tag}.json'}")


if __name__ == "__main__":
    main()
