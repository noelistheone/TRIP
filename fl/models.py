"""FedRec models: MF, LightGCN, NCF.

All three subclass FLModel which exposes a unified contract for the FedAvg
runner / attacks:

    shared_keys()            → state_dict keys that are shared with the server
    set_user_row(t)          → install the client's private user row before
                                local training
    get_user_row()           → read the post-local-training user row out
    forward_pos_neg(pos, neg)→ BPR-friendly score pair (s_u_pos, s_u_neg)
    set_neighbors(items)     → (LightGCN only) install the user's local sub-
                                graph for graph propagation
    extend_items(K)          → grow the item catalog by K rows (paired probes
                                use 2*K) and remember n_items_original

User embedding is kept as a 1-row nn.Embedding so the standard PyTorch optim
loop trains it. Probes are appended to item_emb but are never added to the
LightGCN local graph (they are scoring-only ghosts).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FLModel(nn.Module):
    name: str = "base"

    def __init__(self, n_users: int, n_items: int, d: int):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.d = d
        self.n_items_original: int = n_items
        # Single-row private user embedding. The server hot-swaps the row
        # before each call to local_train via set_user_row.
        self.user_emb = nn.Embedding(1, d)
        nn.init.normal_(self.user_emb.weight, std=0.1)

    # --- user row interface ---------------------------------------------
    def set_user_row(self, u: torch.Tensor) -> None:
        with torch.no_grad():
            self.user_emb.weight.data.copy_(u.detach().to(self.user_emb.weight.device).reshape(1, self.d))

    def get_user_row(self) -> torch.Tensor:
        return self.user_emb.weight.data[0].detach().clone()

    # --- shared params --------------------------------------------------
    def shared_keys(self) -> List[str]:
        return [k for k in self.state_dict().keys() if not k.startswith("user_emb.")]

    # --- catalog growth -------------------------------------------------
    def extend_items(self, K: int) -> None:
        old = self.item_emb.weight.data
        n_old, d = old.shape
        device = old.device
        new = nn.Embedding(n_old + K, d).to(device)
        with torch.no_grad():
            new.weight[:n_old].copy_(old)
            new.weight[n_old:].zero_()
        self.item_emb = new
        self.n_items = n_old + K

    # --- model-specific -------------------------------------------------
    def forward_pos_neg(self, pos: torch.Tensor, neg: torch.Tensor):  # pragma: no cover
        raise NotImplementedError

    def set_neighbors(self, items: List[int]) -> None:
        # default no-op (only LightGCN overrides)
        pass


class MFModel(FLModel):
    name = "mf"

    def __init__(self, n_users: int, n_items: int, d: int):
        super().__init__(n_users, n_items, d)
        self.item_emb = nn.Embedding(n_items, d)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def _user_vec(self) -> torch.Tensor:
        return self.user_emb.weight[0]

    def forward_pos_neg(self, pos: torch.Tensor, neg: torch.Tensor):
        u = self._user_vec()                       # (d,)
        v_pos = self.item_emb(pos)                 # (B, d)
        v_neg = self.item_emb(neg)                 # (B, d)
        pos_s = (v_pos * u.unsqueeze(0)).sum(-1)
        neg_s = (v_neg * u.unsqueeze(0)).sum(-1)
        return pos_s, neg_s

    def score_all(self, item_ids: torch.Tensor | None = None) -> torch.Tensor:
        u = self._user_vec()
        V = self.item_emb.weight if item_ids is None else self.item_emb(item_ids)
        return V @ u


class LightGCNModel(FLModel):
    name = "lightgcn"

    def __init__(self, n_users: int, n_items: int, d: int, hops: int = 2):
        super().__init__(n_users, n_items, d)
        self.hops = hops
        self.item_emb = nn.Embedding(n_items, d)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self._nbr_idx: torch.Tensor | None = None

    def set_neighbors(self, items: List[int]) -> None:
        # Filter out anything past n_items_original (probes never join graph)
        device = self.user_emb.weight.device
        clean = [int(i) for i in items if 0 <= int(i) < self.n_items_original]
        if not clean:
            self._nbr_idx = torch.empty(0, dtype=torch.long, device=device)
        else:
            self._nbr_idx = torch.as_tensor(clean, dtype=torch.long, device=device)

    def _propagate(self) -> torch.Tensor:
        """Return ẽ_u = mean over hops 0..K of e_u^(k) on the local subgraph."""
        u = self.user_emb.weight[0]           # (d,)
        nbr = self._nbr_idx
        if nbr is None or nbr.numel() == 0:
            return u
        v_nbrs = self.item_emb(nbr)           # (M, d)
        M = v_nbrs.shape[0]
        inv = 1.0 / (M ** 0.5)
        layers = [u]
        cur_u = u
        cur_v = v_nbrs
        for _ in range(self.hops):
            new_u = inv * cur_v.sum(dim=0)
            new_v = inv * cur_u.unsqueeze(0).expand(M, -1)
            cur_u, cur_v = new_u, new_v
            layers.append(cur_u)
        return torch.stack(layers, dim=0).mean(dim=0)

    def forward_pos_neg(self, pos: torch.Tensor, neg: torch.Tensor):
        u_tilde = self._propagate()
        v_pos = self.item_emb(pos)
        v_neg = self.item_emb(neg)
        pos_s = (v_pos * u_tilde.unsqueeze(0)).sum(-1)
        neg_s = (v_neg * u_tilde.unsqueeze(0)).sum(-1)
        return pos_s, neg_s

    def score_all(self, item_ids: torch.Tensor | None = None) -> torch.Tensor:
        u_tilde = self._propagate()
        V = self.item_emb.weight if item_ids is None else self.item_emb(item_ids)
        return V @ u_tilde


class NCFModel(FLModel):
    name = "ncf"

    def __init__(self, n_users: int, n_items: int, d: int):
        super().__init__(n_users, n_items, d)
        self.item_emb = nn.Embedding(n_items, d)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        # Architecture per framework spec §6.1: 2d → 128 → 64 → 32
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.h_last = 32
        # Fusion: w = [w_GMF (d); w_MLP (32)]; Linear(d+32, 1) (no bias).
        self.fusion = nn.Linear(d + 32, 1, bias=False)
        nn.init.normal_(self.fusion.weight, std=0.1)

    def _score_batch(self, pos_or_neg: torch.Tensor) -> torch.Tensor:
        u = self.user_emb.weight[0]                       # (d,)
        v = self.item_emb(pos_or_neg)                     # (B, d)
        u_b = u.unsqueeze(0).expand_as(v)
        gmf = u_b * v                                     # (B, d)
        mlp_out = self.mlp(torch.cat([u_b, v], dim=-1))   # (B, 32)
        fused = torch.cat([gmf, mlp_out], dim=-1)         # (B, d+32)
        return self.fusion(fused).squeeze(-1)             # (B,)

    def forward_pos_neg(self, pos: torch.Tensor, neg: torch.Tensor):
        return self._score_batch(pos), self._score_batch(neg)

    def score_all(self, item_ids: torch.Tensor | None = None) -> torch.Tensor:
        ids = torch.arange(self.n_items, device=self.user_emb.weight.device) if item_ids is None else item_ids
        return self._score_batch(ids)


def make_model(name: str, n_users: int, n_items: int, d: int, cfg: dict) -> FLModel:
    if name == "mf":
        return MFModel(n_users, n_items, d)
    if name == "lightgcn":
        return LightGCNModel(n_users, n_items, d, hops=int(cfg.get("lightgcn", {}).get("hops", 2)))
    if name == "ncf":
        return NCFModel(n_users, n_items, d)
    raise ValueError(f"unknown model: {name}")
