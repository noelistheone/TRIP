"""Deep Leakage from Gradients (Zhu, Liu, Han — NeurIPS 2019) adapted to FedRec.

Threat model: per-client item-emb delta is captured in the clear during ONE
"observation round" (no HE). The adversary then runs per-user L-BFGS over
candidate u, minimizing ‖simulate_delta(u) - obs_delta‖² to recover u.

In real FedRec deployments, secure aggregation / HE makes per-client deltas
inaccessible to the server — DLG is included here as a ceiling baseline that
quantifies what is theoretically extractable WITHOUT TRIP's probe trick.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch

from ..client import local_train
from ..models import LightGCNModel, MFModel
from ._simulate import simulate_delta
from .base import AttackBase


def _observation_round(server, bundle, attacked_uids: List[int], cfg: dict) -> Dict[int, torch.Tensor]:
    """Run ONE round of FedAvg with HE temporarily disabled and capture each
    attacked-user's individual item-emb delta in the clear.

    Returns: dict[uid → Tensor of shape (n_items, d)] (item-emb delta only).
    """
    device = next(server.model.parameters()).device
    # Force plain SGD, lr from config (attack-phase math expects Δ = −η·grad).
    cap_cfg = dict(cfg)
    cap_cfg["optimizer"] = "sgd"
    cap_cfg["weight_decay"] = 0.0

    # Snapshot then run a captive round.
    snap = server.snapshot()
    try:
        global_shared = {k: server.model.state_dict()[k].detach().clone()
                         for k in server.model.shared_keys()}
        observations: Dict[int, torch.Tensor] = {}
        round_idx = server.round
        for uid in attacked_uids:
            train_items = bundle.train_pos.get(uid, [])
            if not train_items:
                d = cfg["d"]
                observations[uid] = torch.zeros((server.model.n_items, d))
                continue
            user_init = server.user_states[uid].to(device)
            n_real_items = getattr(server.model, "n_items_original", bundle.n_items)
            delta_dict, _ = local_train(
                server.model, global_shared, user_init, uid, train_items, n_real_items,
                cap_cfg, probe_pair_ids=None, reps_per_pair=0, round_idx=round_idx,
            )
            observations[uid] = delta_dict["item_emb.weight"].detach().cpu()
            # The server.model's state was mutated by local_train; reload global
            # so the next user starts from the same global state.
            with torch.no_grad():
                sd = server.model.state_dict()
                for k, v in global_shared.items():
                    sd[k] = v.to(device).clone()
                server.model.load_state_dict(sd, strict=True)
        # HE-aggregate-only threat model: under homomorphic FedAvg the server
        # only ever sees the decrypted aggregate; per-client deltas remain
        # encrypted forever. DLG / InvGrad / LtI / RAIFLE all consume per-uid
        # observations as the load-bearing input to their per-user solver --
        # structurally, that signal is unavailable under HE. We model this by
        # zeroing each per-uid observation, so the solver runs with no signal
        # and converges to the trivial answer (u_hat -> 0, cos -> 0). This is
        # the same effect as "the attack cannot run", quantified into a number.
        he_cfg = cfg.get("he", {})
        if he_cfg.get("aggregate_only", False):
            for uid in observations:
                observations[uid] = torch.zeros_like(observations[uid])
        return observations
    finally:
        server.restore(snap)


class DLGAttack(AttackBase):
    name = "dlg"

    def prepare(self, server, bundle, attacked_uids: List[int]) -> None:
        with self._time("prepare_sec"):
            self._state["observations"] = _observation_round(
                server, bundle, attacked_uids, self.cfg
            )
            self._state["round_idx"] = server.round

    def solve(self, server, bundle, attacked_uids: List[int]) -> np.ndarray:
        cfg_dlg = self.cfg.get("dlg", {})
        model_key = "mf" if isinstance(server.model, MFModel) else (
            "lightgcn" if isinstance(server.model, LightGCNModel) else "ncf"
        )
        max_iter = int(cfg_dlg.get("max_iter_per_model", {}).get(
            model_key, cfg_dlg.get("max_iter", 500)
        ))
        n_restarts = int(cfg_dlg.get("n_restarts", 3))
        tol = float(cfg_dlg.get("tol", 1.0e-6))

        observations: Dict[int, torch.Tensor] = self._state["observations"]
        round_idx: int = self._state["round_idx"]
        device = next(server.model.parameters()).device
        d = server.cfg["d"]
        N = len(attacked_uids)

        U_hat = np.zeros((N, d), dtype=np.float32)
        per_user_times: List[float] = []

        with self._time("solve_sec"):
            for i, uid in enumerate(attacked_uids):
                obs = observations[uid].to(device)
                train_items = bundle.train_pos.get(uid, [])
                if not train_items:
                    U_hat[i] = np.zeros(d)
                    continue

                t_user = time.time()
                best_u = None
                best_loss = float("inf")
                for r in range(n_restarts):
                    torch.manual_seed((uid + 1) * 11 + r)
                    u = (torch.randn(d, device=device) * 0.1).detach().clone().requires_grad_(True)
                    opt = torch.optim.LBFGS([u], max_iter=max_iter, tolerance_grad=tol,
                                            tolerance_change=tol, history_size=10,
                                            line_search_fn="strong_wolfe")

                    def closure():
                        opt.zero_grad()
                        pred = simulate_delta(u, uid, train_items, server, round_idx)
                        loss = ((pred - obs) ** 2).sum()
                        loss.backward()
                        return loss

                    try:
                        loss_val = opt.step(closure)
                        loss_f = float(loss_val) if loss_val is not None else float("inf")
                    except Exception:
                        loss_f = float("inf")
                    if loss_f < best_loss:
                        best_loss = loss_f
                        best_u = u.detach().cpu().numpy()

                if best_u is None:
                    best_u = np.zeros(d, dtype=np.float32)
                norm = np.linalg.norm(best_u) + 1e-12
                U_hat[i] = best_u / norm
                per_user_times.append(time.time() - t_user)

        self.timing["per_user_solve_mean"] = float(np.mean(per_user_times)) if per_user_times else 0.0
        self.timing["per_user_solve_std"] = float(np.std(per_user_times)) if per_user_times else 0.0
        return U_hat
