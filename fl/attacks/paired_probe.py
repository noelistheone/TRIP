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
            pair_ids, _ = init_paired_probes(model, K=K, eps_rel=cfg["probes"]["eps_rel"])
            probe_base = None
            if isinstance(model, NCFModel):
                probe_base = overwrite_ncf_probes_saturating(
                    model, pair_ids,
                    M=float(cfg["ncf"]["M"]),
                    eps_rel=cfg["probes"]["eps_rel"],
                )

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
            A = allocator.build_A()
            if (policy is not None and getattr(policy, "block_closed", False)
                    and getattr(policy, "partition", None) is not None):
                # Under BC the server can only address whole blocks, so the
                # operator it actually realises is A.Pi.Pi^T (columns constant
                # within a block). Give the ATTACK the correct operator -- the
                # partition is public, so this is the strongest solver the
                # server can mount, not a handicap.
                from ..pact.blocks import pooling_matrix
                import numpy as _np
                Pi = pooling_matrix(range(len(attacked_uids)), policy.partition.assign)
                A = A @ Pi @ Pi.T
                A = _np.minimum(A, 1.0).astype(_np.float32)

            self._state["G"] = G
            self._state["A"] = A
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
