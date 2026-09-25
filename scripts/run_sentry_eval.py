#!/usr/bin/env python
"""E5 -- does SENTRY actually detect TRIP? Measured on REAL attack rounds.

Benign rounds: honest FedAvg from the cached checkpoint (no injection, no
overrides). Attack rounds: the genuine TRIP attack phase -- real paired probes
appended by `init_paired_probes`, real `damp_ncf_mlp` for NCF, real
snapshot/restore. The detector is fit one-class on benign rounds only and never
sees an attack round during training.

Reports detection rate at the conformal level alpha and the empirical false
alarm rate, plus a per-feature ablation so the result is explained, not just
asserted.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fl.data import load_dataset
from fl.eval import (_load_warmup, _make_warmup_cfg, _resolve_per_dataset,
                     _select_attacked_uids, _warmup_cache_path)
from fl.models import NCFModel, make_model
from fl.pact.sentry import FEATURES, Sentry, extract, synth_attack_rounds
from fl.trip import SlidingWindowAllocator, init_paired_probes, overwrite_ncf_probes_saturating
from fl.trip.probes import damp_ncf_mlp
from fl.trip.server import TRIPServer
from fl.utils import pick_gpu, set_seed


def tnorms(model):
    return {k: float(v.norm()) for k, v in model.state_dict().items()
            if not k.startswith("user_emb.")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lastfm")
    ap.add_argument("--model", default="mf")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--benign-rounds", type=int, default=120)
    ap.add_argument("--fresh-rounds", type=int, default=200,
                    help="Benign rounds AFTER the calibration window, used only to measure false alarms.")
    ap.add_argument("--attack-rounds", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--fed-wd", type=float, default=0.0,
                    help="weight decay of benign/fresh rounds (honest federated recipe: 0)")
    ap.add_argument("--out", default="results/sentry")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    device = pick_gpu(default=a.gpu)
    set_seed(int(cfg.get("seed", 42)))
    bundle = load_dataset(a.dataset)
    d = int(cfg["d"])
    model = make_model(a.model, bundle.n_users, bundle.n_items, d, cfg).to(device)
    model.n_items_original = bundle.n_items
    warm = _make_warmup_cfg(cfg, a.dataset, a.model)
    server = TRIPServer(model, bundle, warm, device)
    n_warm = int(_resolve_per_dataset(cfg, "warmup_overrides", a.dataset, cfg.get("warmup", 200)))
    if _load_warmup(_warmup_cache_path(a.dataset, a.model, cfg, warm, n_warm, bundle),
                    model, server, device) is None:
        raise SystemExit(f"no warm-up checkpoint for {a.dataset}/{a.model}")
    # Benign and fresh rounds follow the honest federated recipe of the utility
    # runs (warm-up optimiser and rate, no weight decay). Set only after the
    # checkpoint is loaded so the warm-up cache key is unchanged.
    server.cfg = dict(warm, weight_decay=float(a.fed_wd))

    uids = sorted(bundle.train_user_ids)
    attacked = _select_attacked_uids(bundle, int(cfg["N_attack"]), 42)
    rng = np.random.default_rng(7)
    d = int(cfg["d"])

    # ---------- benign rounds ----------
    # Honest cold-start items: same count as the attack's 2K probes, drawn
    # independently from the catalog's own distribution. Without this control the
    # detector would only be distinguishing "catalog grew" from "catalog did not".
    # Cold-start rows are drawn at the CURRENT catalogue scale: the catalogue
    # shrinks under weight decay, and a fixed-scale control would drift away
    # from the calibration rounds for reasons unrelated to any attack.
    def benign_append(cat):
        k = int(rng.poisson(2 * int(cfg["K"])))          # variable cold-start volume
        return rng.normal(0.0, float(cat.std()), size=(max(k, 2), d))
    benign, prev_recv, prev_recv_state = [], None, None
    for r in range(a.benign_rounds):
        recv = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                if not k.startswith("user_emb.")}
        ident = 0 if prev_recv_state is None else sum(
            1 for k, v in recv.items()
            if k in prev_recv_state and torch.equal(v, prev_recv_state[k]))
        prev_recv_state = recv
        cur = tnorms(model)
        s = [int(x) for x in rng.choice(np.asarray(uids), size=min(128, len(uids)), replace=False)]
        server.run_round(s, probe_assign=None, reps_per_pair=1)
        cat = model.item_emb.weight.detach().cpu().numpy()
        # A benign round also appends cold-start items; draw a comparable number
        # of genuinely NEW rows so `nn_dup` and `mag_z` are not trivially
        # separable just because benign rounds append nothing.
        benign.append(extract(benign_append(cat), cat, cur, prev_recv, ident, len(cur), 0.0))
        prev_recv = cur
    benign = np.asarray(benign)
    snap_after_benign = server.snapshot()
    prev_fresh = prev_recv
    prev_state_fresh = prev_recv_state

    # ---------- fresh benign rounds, strictly after calibration ----------
    # (the first version measured false alarms on the calibration rounds
    # themselves, which is in-sample and meaningless)
    fresh = []
    # evasion families, in units of the catalogue std: "scaled" = the attack's own
    # construction at probe scale eps (base eps, pair jitter eps/10); "adaptive" =
    # base sqrt(1-s^2), jitter s, so every probe row is marginally N(0, sigma^2)
    EV_FAMILIES = {f"scaled:{e}": (e, 0.1 * e) for e in (1e-3, 1e-1, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0)}
    EV_FAMILIES.update({f"adaptive:{s_}": (float(np.sqrt(1 - s_ * s_)), s_) for s_ in (0.1, 0.3, 0.5, 0.6, 0.707)})
    ev_rng = np.random.default_rng(20241019)
    ev_round = {}
    for r in range(a.fresh_rounds):
        recv = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                if not k.startswith("user_emb.")}
        ident = sum(1 for k, v in recv.items() if k in prev_state_fresh and torch.equal(v, prev_state_fresh[k]))
        prev_state_fresh = recv
        cur = tnorms(model)
        s_ = [int(x) for x in rng.choice(np.asarray(uids), size=min(128, len(uids)), replace=False)]
        server.run_round(s_, probe_assign=None, reps_per_pair=1)
        cat = model.item_emb.weight.detach().cpu().numpy()
        fresh.append(extract(benign_append(cat), cat, cur, prev_fresh, ident, len(cur), 0.0))
        # the same round's catalogue, with probe pairs of each evasion family appended
        # instead of honest cold-start rows (independent RNG so the benign stream is unchanged)
        sv = float(cat.std())
        for key, (b_sd, j_sd) in EV_FAMILIES.items():
            base = ev_rng.normal(0.0, b_sd * sv, size=(int(cfg["K"]), d))
            jit = ev_rng.normal(0.0, j_sd * sv, size=(int(cfg["K"]), d))
            rows = np.concatenate([base + jit, base - jit], axis=0)
            ev_round.setdefault(key, []).append(extract(rows, cat, cur, prev_fresh, ident, len(cur), 1.0))
        prev_last = prev_fresh
        prev_fresh = cur
    fresh = np.asarray(fresh)

    cat_fresh, cur_fresh = cat, cur
    prev_recv, prev_recv_state = prev_fresh, prev_state_fresh   # attack follows the fresh rounds
    import copy as _copy
    model_pre_attack = _copy.deepcopy(model)   # catalogue without probes

    # ---------- real TRIP attack rounds ----------
    K, W, Tf = int(cfg["K"]), int(cfg["W"]), int(cfg["T_factor"])
    n_before = model.item_emb.weight.shape[0]
    pair_ids, _ = init_paired_probes(model, K, cfg["probes"]["eps_rel"],
                                     base_rel=cfg["probes"].get("base_rel"),
                                     jitter_rel=cfg["probes"].get("jitter_rel"))
    if isinstance(model, NCFModel) and bool(cfg.get("ncf", {}).get("saturating_probes", False)):
        overwrite_ncf_probes_saturating(model, pair_ids, M=float(cfg["ncf"]["M"]),
                                        eps_rel=cfg["probes"]["eps_rel"])
    new_rows = model.item_emb.weight.detach().cpu().numpy()[n_before:]
    alloc = SlidingWindowAllocator(N=len(attacked), K=K, W=W, T_factor=Tf)
    acfg = dict(warm); acfg.update(optimizer="sgd", weight_decay=0.0, local_epochs=1,
                                   attack_probes_only=True)
    server.cfg = acfg
    attack = []
    for t in range(a.attack_rounds):
        snap = server.snapshot()
        try:
            if isinstance(model, NCFModel):
                damp_ncf_mlp(model, factor=0.1)
            recv = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    if not k.startswith("user_emb.")}
            ident = 0 if prev_recv_state is None else sum(
                1 for k, v in recv.items()
                if k in prev_recv_state and torch.equal(v, prev_recv_state[k]))
            prev_recv_state = recv
            cur = tnorms(model)              # what the CLIENT receives this round
            la = alloc.assignments_for_round(t)
            pa, forced = {}, []
            for lu, ks in la.items():
                ru = attacked[lu]; pa[ru] = [pair_ids[k] for k in ks]; forced.append(ru)
            server.run_round(list(set(forced)), probe_assign=pa, reps_per_pair=32)
            cat = model.item_emb.weight.detach().cpu().numpy()
            # The probe rows sit in the catalog every round, and the client can
            # see the whole catalog; `alien_pos` is the fraction of instructed
            # positives absent from the client's own history.
            attack.append(extract(new_rows, cat, cur, prev_recv, ident, len(cur), 1.0))
            prev_recv = cur
        finally:
            server.restore(snap)
    attack = np.asarray(attack)

    # Detector features: only the catalogue-based ones, which are stationary
    # under benign training. Norm ratios drift as the model trains, and the
    # staleness indicator has zero benign variance; both are reported, not used.
    USE = [FEATURES.index(f) for f in ("nn_dup", "growth", "mag_z")]
    h = len(benign) // 2          # split conformal: fit on one half, calibrate on the other
    S = Sentry(alpha=a.alpha).fit(benign[:h][:, USE], calib=benign[h:][:, USE])
    det = float(np.mean([S.flag(x[USE])[0] for x in attack]))
    fa = float(np.mean([S.flag(x[USE])[0] for x in fresh]))
    per = {}
    for f in ("nn_dup", "growth", "mag_z"):
        i = FEATURES.index(f)
        Si = Sentry(alpha=a.alpha).fit(benign[:h][:, [i]], calib=benign[h:][:, [i]])
        per[f] = (float(np.mean([Si.flag(x[[i]])[0] for x in attack])),
                  float(np.mean([Si.flag(x[[i]])[0] for x in fresh])))

    # Evasion, measured with the full detector (not extrapolated from one feature).
    # "scaled": the attack's own construction with eps_rel = eps (base and pair
    # separation both grow with eps). "adaptive": base kept at catalogue scale,
    # the pair separation s is widened while the base variance is lowered to
    # 1 - s^2, so every probe row keeps the catalogue's marginal N(0, sigma^2);
    # at s = 1/sqrt(2) the two rows of a pair are independent draws.
    from fl.trip import init_paired_probes as _ipp
    def _probe_feats(**kw):
        mm = _copy.deepcopy(model_pre_attack); n0 = mm.item_emb.weight.shape[0]
        _ipp(mm, int(cfg["K"]), **kw)
        rows = mm.item_emb.weight.detach().cpu().numpy()[n0:]
        x = extract(rows, cat_fresh, cur_fresh, prev_last, 0, len(cur_fresh), 1.0)
        flag, pv = S.flag(x[USE])
        return {"detected": bool(flag), "p": float(pv),
                **{f: float(x[FEATURES.index(f)]) for f in ("nn_dup", "growth", "mag_z")}}
    ev = {str(e): _probe_feats(eps_rel=float(e))
          for e in (1e-3, 1e-1, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0)}
    ev_adapt = {str(s_): _probe_feats(eps_rel=1e-3, base_rel=float(np.sqrt(1 - s_ * s_)),
                                      jitter_rel=float(s_))
                for s_ in (0.1, 0.3, 0.5, 0.6, 0.707)}

    ev_rate = {k: float(np.mean([S.flag(x[USE])[0] for x in v])) for k, v in ev_round.items()}
    i_nn = FEATURES.index("nn_dup")
    S_nn = Sentry(alpha=a.alpha).fit(benign[:h][:, [i_nn]], calib=benign[h:][:, [i_nn]])
    ev_rate_nn = {k: float(np.mean([S_nn.flag(x[[i_nn]])[0] for x in v])) for k, v in ev_round.items()}

    print(f"\n{a.dataset}/{a.model}  calibration={len(benign)}  fresh benign={len(fresh)}  attack={len(attack)}")
    print(f"  detection (real TRIP rounds)      : {det:.3f}")
    print(f"  false alarm (FRESH benign rounds) : {fa:.3f}   (alpha={a.alpha})")
    for f, (d_, fa_) in per.items():
        print(f"    {f:<8} det={d_:.3f} fa={fa_:.3f}")
    print("  scaled  :", {k: v["detected"] for k, v in ev.items()})
    print("  adaptive:", {k: (v["detected"], round(v["nn_dup"], 3), round(v["mag_z"], 3)) for k, v in ev_adapt.items()})
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / f"{a.dataset}_{a.model}.json").write_text(json.dumps(
        {"dataset": a.dataset, "model": a.model, "alpha": a.alpha,
         "features_used": ["nn_dup", "growth", "mag_z"],
         "detection_real": det, "false_alarm_fresh": fa, "per_feature": per,
         "n_fit": h, "n_calibration": len(benign) - h, "n_fresh": len(fresh), "n_attack": len(attack),
         "evasion_scaled": ev, "evasion_adaptive": ev_adapt,
         "evasion_rate_per_round": ev_rate, "evasion_rate_per_round_nn_only": ev_rate_nn,
         "benign_feature_median": {f: float(np.median(benign[:, FEATURES.index(f)]))
                                   for f in ("nn_dup", "growth", "mag_z")},
         "raw": {"features": FEATURES, "benign": benign.tolist(),
                 "fresh": fresh.tolist(), "attack": attack.tolist()}}, indent=2))
    print(f"\nsaved -> {o}/{a.dataset}_{a.model}.json")


if __name__ == "__main__":
    main()
