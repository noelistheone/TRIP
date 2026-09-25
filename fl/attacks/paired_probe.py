"""Paired-Probe attack adapter.

Wraps the existing fl/trip/* implementation (unchanged) so the AttackBase
contract is satisfied. Behavior is identical to the pre-refactor TRIPServer.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch

from ..models import NCFModel
from ..trip import (
    SlidingWindowAllocator,
    TRIPServer,
    init_paired_probes,
    overwrite_ncf_probes_saturating,
    solve_mf_lgcn,
    solve_ncf,
)
from .base import AttackBase


class PairedProbeAttack(AttackBase):
    name = "paired_probe"

    def prepare(self, server, bundle, attacked_uids: List[int]) -> None:
        with self._time("prepare_sec"):
            model = server.model
            cfg = self.cfg
            K = int(cfg["K"])
            W = int(cfg["W"])
            T_factor = int(cfg["T_factor"])
            N = len(attacked_uids)

            # Extend catalog with 2K paired probes
            model.n_items_original = bundle.n_items
            pair_ids, _ = init_paired_probes(model, K=K, eps_rel=cfg["probes"]["eps_rel"],
                                             base_rel=cfg["probes"].get("base_rel"),
                                             jitter_rel=cfg["probes"].get("jitter_rel"))
            probe_base = None
            if isinstance(model, NCFModel):
                if bool(cfg.get("ncf", {}).get("saturating_probes", False)):
                    # Legacy large-norm probe centres (M * sigma_v). Kept only for
                    # reproducing old results: probe magnitude does not suppress
                    # the MLP residual (ReLU derivatives are scale-invariant).
                    probe_base = overwrite_ncf_probes_saturating(
                        model, pair_ids,
                        M=float(cfg["ncf"]["M"]),
                        eps_rel=cfg["probes"]["eps_rel"],
                    )
                else:
                    d_ = int(cfg["d"])
                    probe_base = model.fusion.weight.detach().clone().squeeze(0)[:d_]

            allocator = SlidingWindowAllocator(N=N, K=K, W=W, T_factor=T_factor)
            trip = TRIPServer(server.model, bundle, cfg, server.device)
            trip.user_states = server.user_states
            trip.round = server.round

            from ..models import MFModel, LightGCNModel
            if isinstance(model, MFModel):
                reps = int(cfg["reps_per_pair"]["mf"])
            elif isinstance(model, LightGCNModel):
                reps = int(cfg["reps_per_pair"]["lightgcn"])
            else:
                reps = int(cfg["reps_per_pair"]["ncf"])

            policy = getattr(self, "policy", None)
            G = trip.attack(attacked_uids, allocator, pair_ids,
                            reps_per_pair=reps, policy=policy)
            # Build the operator that actually generated G. A client training m
            # pairs in one round writes each with weight (reps / size of the
            # mini-batch the pair falls in); probes-only batches are not shuffled.
            # With K*W <= N and no block expansion every m is 1 and this is the
            # 0/1 matrix of Lemma 1. The server chose the assignment, so it knows
            # these weights; fitting the 0/1 matrix instead would handicap it.
            B = int(cfg.get("local_batch", 256))
            probes_only = bool(cfg.get("attack_probes_only", True))
            A = np.zeros((allocator.T * K, len(attacked_uids)), dtype=np.float32)
            for t_idx, assign in enumerate(trip.last_assignments):
                for lu, ks in assign.items():
                    ks_sorted = list(ks)
                    if not probes_only:
                        for k in ks_sorted:      # mixed batches: weights depend on |I_u|
                            A[t_idx * K + k, lu] = 1.0
                        continue
                    total = len(ks_sorted) * reps
                    for idx, k in enumerate(ks_sorted):
                        lo, hi = idx * reps, (idx + 1) * reps
                        wgt = 0.0
                        for b0 in range(0, total, B):
                            b1 = min(b0 + B, total)
                            ov = max(0, min(hi, b1) - max(lo, b0))
                            if ov:
                                wgt += ov / (b1 - b0)
                        A[t_idx * K + k, lu] = wgt

            self._state["G"] = G
            self._state["A"] = A
            self._state["tb_aborted"] = int(getattr(trip, "tb_aborted", 0))
            self._state["tb_rounds"] = len(trip.last_assignments)
            self._state["probe_base"] = probe_base
            self._state["pair_ids"] = pair_ids
            self._state["allocator"] = allocator

    def solve(self, server, bundle, attacked_uids: List[int]) -> np.ndarray:
        with self._time("solve_sec"):
            G = self._state["G"]
            A = self._state["A"]
            probe_base = self._state["probe_base"]
            model = server.model
            if isinstance(model, NCFModel):
                U_hat = solve_ncf(
                    G, A, probe_base, model,
                    lam=float(self.cfg["ncf"]["ridge_lambda"]),
                )
            else:
                U_hat = solve_mf_lgcn(
                    G, A, ridge=float(self.cfg.get('solver_ridge', 1e-6)),
                )
        return U_hat
