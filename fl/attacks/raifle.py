"""RAIFLE (Pham, Kulkarni, Houmansadr — NDSS 2025) adapted to FedRec.

Threat model: per-client item-emb delta in the clear (no HE), like DLG/IG.
Active component: server perturbs the K most-popular item embeddings to
α-scaled orthonormal directions BEFORE the observation round, sharpening
the gradient signal at known directions. Then per-user Adam reconstruction
with squared-error loss against the observed delta.

After solving, the server restores the clean item-embedding state so
subsequent attacks see undisturbed warmup state.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch

from ..models import LightGCNModel, MFModel
from ._simulate import simulate_delta
from .base import AttackBase
from .dlg import _observation_round


class RaifleAttack(AttackBase):
    name = "raifle"

    def prepare(self, server, bundle, attacked_uids: List[int]) -> None:
        with self._time("prepare_sec"):
            cfg_r = self.cfg.get("raifle", {})
            n_probes = int(cfg_r.get("n_probes", 64))
            alpha = float(cfg_r.get("alpha", 1.0))
            device = next(server.model.parameters()).device

            # Pick the most "popular" items by training-set count over attacked uids.
            counts: Dict[int, int] = {}
            for uid in attacked_uids:
                for it in bundle.train_pos.get(uid, []):
                    if it < server.model.n_items_original:
                        counts[it] = counts.get(it, 0) + 1
            top = sorted(counts.items(), key=lambda x: -x[1])[:n_probes]
            top_ids = [int(i) for (i, _) in top]
            d = server.cfg["d"]

            # Save original embeddings for these rows; overwrite with α-scaled
            # orthonormal directions (random unit vectors approximate this for
            # n_probes ≪ d ranks anyway).
            self._state["restore_ids"] = top_ids
            self._state["restore_vecs"] = (
                server.model.item_emb.weight.data[top_ids].detach().clone()
                if top_ids else None
            )
            if top_ids:
                g = torch.Generator(device=device).manual_seed(2024_2025)
                rand = torch.randn(len(top_ids), d, generator=g, device=device)
                rand = rand / (rand.norm(dim=-1, keepdim=True) + 1e-12) * alpha
                with torch.no_grad():
                    server.model.item_emb.weight.data[top_ids] = rand

            # Now do the standard observation round on perturbed state.
            self._state["observations"] = _observation_round(
                server, bundle, attacked_uids, self.cfg
            )
            self._state["round_idx"] = server.round

    def solve(self, server, bundle, attacked_uids: List[int]) -> np.ndarray:
        cfg_r = self.cfg.get("raifle", {})
        model_key = "mf" if isinstance(server.model, MFModel) else (
            "lightgcn" if isinstance(server.model, LightGCNModel) else "ncf"
        )
        max_iter = int(cfg_r.get("max_iter_per_model", {}).get(
            model_key, cfg_r.get("max_iter", 1000)
        ))
        n_restarts = int(cfg_r.get("n_restarts", 3))
        adam_lr = float(cfg_r.get("adam_lr", 0.05))

        observations: Dict[int, torch.Tensor] = self._state["observations"]
        round_idx: int = self._state["round_idx"]
        device = next(server.model.parameters()).device
        d = server.cfg["d"]
        N = len(attacked_uids)

        U_hat = np.zeros((N, d), dtype=np.float32)
        per_user_times: List[float] = []

        with self._time("solve_sec"):
            try:
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
                        torch.manual_seed((uid + 1) * 13 + r)
                        u = (torch.randn(d, device=device) * 0.05).detach().clone().requires_grad_(True)
                        opt = torch.optim.Adam([u], lr=adam_lr)
                        for _ in range(max_iter):
                            opt.zero_grad()
                            pred = simulate_delta(u, uid, train_items, server, round_idx)
                            loss = ((pred - obs) ** 2).sum()
                            loss.backward()
                            opt.step()
                        # Re-evaluate AFTER the last step: `loss` above is the
                        # objective at the previous iterate, so comparing it
                        # against the post-step `u` selected restarts on a
                        # stale score and handicapped the baseline.
                        # NOTE: no torch.no_grad() here -- simulate_delta for NCF
                        # calls torch.autograd.grad internally, which needs grad
                        # mode enabled. We just discard the graph via detach().
                        loss_val = float(((simulate_delta(
                            u, uid, train_items, server, round_idx) - obs) ** 2
                        ).sum().detach())
                        if loss_val < best_loss:
                            best_loss = loss_val
                            best_u = u.detach().cpu().numpy()
                    if best_u is None:
                        best_u = np.zeros(d, dtype=np.float32)
                    U_hat[i] = best_u / (np.linalg.norm(best_u) + 1e-12)
                    per_user_times.append(time.time() - t_user)
            finally:
                # Restore the perturbed item rows so subsequent attacks are clean.
                ids = self._state.get("restore_ids") or []
                vecs = self._state.get("restore_vecs")
                if ids and vecs is not None:
                    with torch.no_grad():
                        server.model.item_emb.weight.data[ids] = vecs.to(device)

        self.timing["per_user_solve_mean"] = float(np.mean(per_user_times)) if per_user_times else 0.0
        self.timing["per_user_solve_std"] = float(np.std(per_user_times)) if per_user_times else 0.0
        return U_hat
