"""Inverting Gradients (Geiping, Bauermeister, Dröge, Moeller — NeurIPS 2020)
adapted to FedRec.

Same threat model as DLG (per-client item-emb delta captured in the clear,
i.e. no HE). Differences from DLG:
  - Loss: 1 - cos(pred_delta, obs_delta)          (magnitude-invariant)
  - Optimizer: Adam, lr=0.1, 1000 iterations
  - Signed-gradient update step per Geiping §3.2
  - TV prior DROPPED (image-specific; not applicable to vector embeddings).

Per-user iterative optimization: typically slower per iteration than L-BFGS
but with fewer unstable restarts.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch

from ._simulate import simulate_delta
from .base import AttackBase
from .dlg import _observation_round


class InvertGradientsAttack(AttackBase):
    name = "invert_grad"

    def prepare(self, server, bundle, attacked_uids: List[int]) -> None:
        with self._time("prepare_sec"):
            self._state["observations"] = _observation_round(
                server, bundle, attacked_uids, self.cfg
            )
            self._state["round_idx"] = server.round

    def solve(self, server, bundle, attacked_uids: List[int]) -> np.ndarray:
        ig_cfg = self.cfg.get("invert_grad", {})
        from ..models import MFModel, LightGCNModel
        model_key = "mf" if isinstance(server.model, MFModel) else (
            "lightgcn" if isinstance(server.model, LightGCNModel) else "ncf"
        )
        lr = float(ig_cfg.get("lr", 0.1))
        n_iter = int(ig_cfg.get("n_iter_per_model", {}).get(
            model_key, ig_cfg.get("n_iter", 1000)
        ))
        signed_grad = bool(ig_cfg.get("signed_grad", True))
        # Published schedule (inversefed): x0.1 at 3/8, 5/8 and 7/8 of the budget. Off by default.
        lr_decay = bool(ig_cfg.get("lr_decay", False))
        milestones = [int(n_iter * f) for f in (3 / 8, 5 / 8, 7 / 8)]

        observations: Dict[int, torch.Tensor] = self._state["observations"]
        round_idx: int = self._state["round_idx"]
        device = next(server.model.parameters()).device
        d = server.cfg["d"]
        N = len(attacked_uids)

        U_hat = np.zeros((N, d), dtype=np.float32)
        per_user_times: List[float] = []
        best_iters: List[int] = []   # v4 budget diagnostics: step at which the best loss was seen

        with self._time("solve_sec"):
            for i, uid in enumerate(attacked_uids):
                obs = observations[uid].to(device)
                train_items = bundle.train_pos.get(uid, [])
                if not train_items:
                    U_hat[i] = np.zeros(d)
                    continue
                obs_flat = obs.flatten()
                obs_norm = obs_flat.norm().clamp_min(1e-12)

                torch.manual_seed((uid + 1) * 7)
                u = (torch.randn(d, device=device) * 0.01).detach().clone().requires_grad_(True)
                opt = torch.optim.Adam([u], lr=lr)
                sched = (torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)
                         if lr_decay else None)

                t_user = time.time()
                best_loss = float("inf")
                best_u = u.detach().clone()
                best_step = 0
                for step in range(n_iter):
                    opt.zero_grad()
                    pred = simulate_delta(u, uid, train_items, server, round_idx)
                    pred_flat = pred.flatten()
                    pred_norm = pred_flat.norm().clamp_min(1e-12)
                    # 1 − cos: magnitude-invariant loss from Geiping §3.1
                    loss = 1.0 - (pred_flat @ obs_flat) / (pred_norm * obs_norm)
                    # Signed-gradient descent takes fixed-size steps, so the
                    # iterate oscillates around the optimum and the LAST one is
                    # systematically worse than the best seen. Track the best.
                    lv = float(loss.detach())
                    if lv < best_loss:
                        best_loss = lv
                        best_u = u.detach().clone()
                        best_step = step
                    loss.backward()
                    if signed_grad:
                        with torch.no_grad():
                            if u.grad is not None:
                                u.grad = u.grad.sign()
                    opt.step()
                    if sched is not None:
                        sched.step()
                u_np = best_u.cpu().numpy()
                u_np = u_np / (np.linalg.norm(u_np) + 1e-12)
                U_hat[i] = u_np
                per_user_times.append(time.time() - t_user)
                best_iters.append(best_step)

        self.timing["per_user_solve_mean"] = float(np.mean(per_user_times)) if per_user_times else 0.0
        self.timing["per_user_solve_std"] = float(np.std(per_user_times)) if per_user_times else 0.0
        if best_iters:
            bi = np.asarray(best_iters, dtype=float) / max(n_iter - 1, 1)
            self.timing["budget_n_iter"] = int(n_iter)
            self.timing["best_step_frac_mean"] = float(bi.mean())
            self.timing["best_in_last_10pct_frac"] = float((bi >= 0.9).mean())
        return U_hat
