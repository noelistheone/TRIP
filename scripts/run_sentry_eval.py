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
    ap.add_argument("--attack-rounds", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--gpu", type=int, default=None)
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

    uids = sorted(bundle.train_user_ids)
    attacked = _select_attacked_uids(bundle, int(cfg["N_attack"]), 42)
    rng = np.random.default_rng(7)
    d = int(cfg["d"])

    # ---------- benign rounds ----------
    # Honest cold-start items: same count as the attack's 2K probes, drawn
    # independently from the catalog's own distribution. Without this control the
    # detector would only be distinguishing "catalog grew" from "catalog did not".
    _cat = model.item_emb.weight.detach().cpu().numpy()
    _sv = float(_cat.std())
    def benign_append():
        k = int(rng.poisson(2 * int(cfg["K"])))          # variable cold-start volume
        return rng.normal(0.0, _sv, size=(max(k, 2), d))
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
        benign.append(extract(benign_append(), cat, cur, prev_recv, ident, len(cur), 0.0))
        prev_recv = cur
    benign = np.asarray(benign)

    # ---------- real TRIP attack rounds ----------
    K, W, Tf = int(cfg["K"]), int(cfg["W"]), int(cfg["T_factor"])
    n_before = model.item_emb.weight.shape[0]
    pair_ids, _ = init_paired_probes(model, K, cfg["probes"]["eps_rel"])
    if isinstance(model, NCFModel):
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

    # ---------- fit one-class on benign only ----------
    n = len(benign); cut = n // 2
    S = Sentry(alpha=a.alpha).fit(benign[:cut], benign[cut:])
    det = float(np.mean([S.flag(x)[0] for x in attack]))
    fa = float(np.mean([S.flag(x)[0] for x in benign[cut:]]))
    syn = synth_attack_rounds(benign, 400, rng)
    det_syn = float(np.mean([S.flag(x)[0] for x in syn]))

    # `alien_pos` is a DIRECT observation of the probes-only instruction, so with
    # it detection is trivial and says nothing interesting. The real question is
    # whether the attack is still visible to a client that is told nothing --
    # i.e. from the broadcast model and catalog alone.
    obs = [i for i, f in enumerate(FEATURES) if f != "alien_pos"]
    Sp = Sentry(alpha=a.alpha).fit(benign[:cut][:, obs], benign[cut:][:, obs])
    det_passive = float(np.mean([Sp.flag(x[obs])[0] for x in attack]))
    fa_passive = float(np.mean([Sp.flag(x[obs])[0] for x in benign[cut:]]))

    print(f"\n{a.dataset}/{a.model}  benign={len(benign)}  real attack rounds={len(attack)}")
    print(f"  detection rate (REAL TRIP rounds) : {det:.3f}")
    print(f"  detection rate (synthetic class)  : {det_syn:.3f}")
    print(f"  false-alarm rate (held-out benign): {fa:.3f}   (conformal target alpha={a.alpha})")
    print(f"  -- broadcast-only (no instruction feature) --")
    print(f"  detection rate                    : {det_passive:.3f}")
    print(f"  false-alarm rate                  : {fa_passive:.3f}")
    print(f"\n  {'feature':<12}{'benign med':>12}{'attack med':>12}{'robust z':>11}")
    for i, f in enumerate(FEATURES):
        z = abs((np.median(attack[:, i]) - S.med[i]) / S.mad[i])
        print(f"  {f:<12}{S.med[i]:>12.4g}{np.median(attack[:, i]):>12.4g}{z:>11.2f}")
    # single-feature ablation: which signature is load-bearing?
    print(f"\n  single-feature detection / false-alarm (which signature carries it):")
    for i, f in enumerate(FEATURES):
        Si = Sentry(alpha=a.alpha).fit(benign[:cut][:, [i]], benign[cut:][:, [i]])
        di = float(np.mean([Si.flag(x[[i]])[0] for x in attack]))
        fi = float(np.mean([Si.flag(x[[i]])[0] for x in benign[cut:]]))
        deg = "  <-- degenerate (benign MAD ~ 0)" if S.mad[i] < 1e-6 else ""
        print(f"    {f:<12} det={di:.3f}  fa={fi:.3f}{deg}")

    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / f"{a.dataset}_{a.model}.json").write_text(json.dumps(
        {"dataset": a.dataset, "model": a.model, "alpha": a.alpha,
         "detection_real": det, "detection_synthetic": det_syn, "false_alarm": fa,
         "detection_broadcast_only": det_passive, "false_alarm_broadcast_only": fa_passive,
         "features": FEATURES,
         "benign_median": S.med.tolist(), "attack_median": np.median(attack, 0).tolist()},
        indent=2))
    print(f"\nsaved -> {o}/{a.dataset}_{a.model}.json")


if __name__ == "__main__":
    main()
