"""Learning to Invert (Wu, Chen, Guo, Weinberger — UAI 2023) adapted to FedRec.

Paper: arxiv.org/abs/2210.10880

The original LtI trains a supervised regressor on auxiliary (gradient → input)
data and uses it to invert gradients at attack time, bypassing per-user
optimization. We adapt to FedRec embedding inversion as follows:

  Threat model (matches Wu et al. §3):
    - Honest-but-curious server captures one round of per-client item-embedding
      updates in the clear (no HE), as in DLG / IG.
    - Server has SHADOW access to the true u of HALF the attacked users to use
      as supervised training labels for the inversion regressor. This is the
      LtI paper's "auxiliary dataset" assumption. Shadow uids are the FIRST
      N/2 attacked_uids; target uids are the REMAINING N/2 — only the target
      half is reported in cos/ranking metrics (eval_mask hides shadow rows).

  Method (faithful to Wu et al. §3):
    1. For each attacked user, capture per-client item-emb delta on one
       observation round (same channel as DLG/IG).
    2. Server-agnostic feature engineering: take the TOP-K rows of |delta|
       by L2 magnitude (sorted descending), stack into (K·d,) feature
       vector. We deliberately do NOT use train_pos knowledge here — that
       is auxiliary information beyond LtI's threat model and would make
       the feature directly proportional to u, collapsing the regression
       to identity. Top-K-magnitude is the model-agnostic feature Wu et al.
       use for image-gradient inversion, transposed to embedding rows.
    3. Train a 3-layer MLP regressor (256→128→d) with Adam + ReLU on the
       shadow half (uids[:N/2], features → u). 200 epochs, early stop on
       in-sample MSE plateau.
    4. Apply regressor to all N users; runner subsets to target half via
       eval_mask, so reported cos reflects genuine out-of-sample
       generalisation.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .base import AttackBase
from .dlg import _observation_round


class LearningToInvertAttack(AttackBase):
    name = "lti"

    def prepare(self, server, bundle, attacked_uids: List[int]) -> None:
        with self._time("prepare_sec"):
            self._state["observations"] = _observation_round(
                server, bundle, attacked_uids, self.cfg
            )
            true_U = torch.stack([
                server.user_states[uid].detach().cpu() for uid in attacked_uids
            ])
            self._state["true_U"] = true_U
            self._state["round_idx"] = server.round
            self._state["d"] = self.cfg["d"]

    def solve(self, server, bundle, attacked_uids: List[int]) -> np.ndarray:
        with self._time("solve_sec"):
            obs: Dict[int, torch.Tensor] = self._state["observations"]
            true_U: torch.Tensor = self._state["true_U"]
            d: int = self._state["d"]
            device = next(server.model.parameters()).device

            N = len(attacked_uids)
            n_shadow = N // 2

            lti_cfg = self.cfg.get("lti", {})
            K = int(lti_cfg.get("topk_rows", 8))
            n_epochs = int(lti_cfg.get("n_epochs", 200))
            lr = float(lti_cfg.get("lr", 1e-3))
            hidden = lti_cfg.get("hidden", [256, 128])

            def fingerprint(delta: torch.Tensor) -> torch.Tensor:
                """Top-K rows of |delta| by L2 magnitude, sorted descending.
                Server-agnostic: no knowledge of which items the user
                actually interacted with. This matches LtI's image-gradient
                feature engineering — the regressor must learn the mapping
                from gradient-pattern shape to input."""
                norms = delta.norm(dim=-1)  # (n_items,)
                k = min(K, norms.shape[0])
                vals, idx = torch.topk(norms, k=k)
                rows = delta[idx]  # (K, d) — already in descending-norm order
                return rows.reshape(-1).to(device)  # (K·d,)

            X = torch.stack([fingerprint(obs[uid]) for uid in attacked_uids])
            Y = true_U.to(device)

            # 3-layer MLP regressor per Wu et al. §3.
            in_dim = X.shape[1]
            layers = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, d))
            mlp = nn.Sequential(*layers).to(device)

            X_shadow = X[:n_shadow]
            Y_shadow = Y[:n_shadow]

            opt = torch.optim.Adam(mlp.parameters(), lr=lr)
            mlp.train()
            for _ in range(n_epochs):
                opt.zero_grad()
                pred = mlp(X_shadow)
                loss = ((pred - Y_shadow) ** 2).mean()
                loss.backward()
                opt.step()

            mlp.eval()
            with torch.no_grad():
                Y_full = mlp(X).cpu().numpy()
            Y_full = Y_full / (np.linalg.norm(Y_full, axis=-1, keepdims=True) + 1e-12)

            # Restrict eval to TARGET half (out-of-sample). Shadow half was
            # used as supervised training labels; including it would inflate
            # cos via in-sample fit.
            mask = np.zeros(N, dtype=bool)
            mask[n_shadow:] = True
            self.eval_mask = mask
        return Y_full
