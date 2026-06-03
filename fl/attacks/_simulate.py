"""Differentiable simulator of one client's local training round.

Given a hypothesized user embedding `u`, reproduces the exact BPR-SGD update
the real client would have performed, assuming the attacker knows:
  - the global shared state the round began with (item_emb + MLP/fusion weights),
  - the user's train_items,
  - the client's RNG seed (we reuse `(uid, round_idx)` deterministically,
    matching fl/client.py::local_train line 75).

This is faithful to the DLG / Inverting Gradients setting where the adversary
is assumed to know the loss function, model architecture, and hyperparameters.
It simply needs to predict the delta that a candidate `u` would produce and
match it against the observed delta.

For MF: closed-form differentiable expression (no actual SGD loop needed since
the item-embedding gradient is a simple function of κ·u).

For LightGCN / NCF: we run a short differentiable BPR SGD loop through a
deepcopy of the model with the candidate u injected, then read back the shared-
param delta. Slower but correct.
"""
from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from ..data import bpr_sample
from ..models import FLModel, LightGCNModel, MFModel, NCFModel


def _reproduce_triples(uid: int, train_items: List[int], n_real_items: int,
                       local_epochs: int, round_idx: int,
                       probe_pair_ids=None, reps_per_pair: int = 0) -> np.ndarray:
    """Replay the same BPR triples that fl/client.py::local_train sampled.

    Reuses the exact seed formula in client.py:75 so sampled neg items match.
    """
    rng = np.random.default_rng(seed=(uid + 1) * 1_000_003 + round_idx)
    all_triples = []
    for _ in range(local_epochs):
        n_real = max(len(train_items), 1)
        real = bpr_sample(uid, train_items, n_real_items, n_samples=n_real, rng=rng)
        if probe_pair_ids:
            for (pos, neg) in probe_pair_ids:
                for _ in range(reps_per_pair):
                    real = np.vstack([real, np.array([[uid, pos, neg]], dtype=np.int64)])
        rng.shuffle(real, axis=0)
        all_triples.append(real)
    return np.concatenate(all_triples, axis=0) if all_triples else np.zeros((0, 3), dtype=np.int64)


def simulate_local_delta_mf(u: torch.Tensor, uid: int, train_items: List[int],
                            global_item_emb: torch.Tensor, cfg: dict,
                            round_idx: int) -> torch.Tensor:
    """Closed-form differentiable MF item-emb delta prediction.

    For 1 local_epoch, the client walks BPR triples through a plain-SGD loop:
    for each triple (u, pos, neg):
        pos_s = u·v_pos; neg_s = u·v_neg
        κ = σ(neg_s - pos_s)
        v_pos -= lr · (-κ) · u          = + lr·κ·u
        v_neg -= lr · (+κ) · u          = - lr·κ·u
    With Adam the update is non-linear in grad; DLG assumes plain SGD (matching
    our attack-phase server). For warmup+Adam-trained state, the attacker still
    observes ONE round of plain-SGD gradient capture in the 'no HE' observation
    round (we configure it that way in `dlg.prepare`).

    Because BPR triples are sampled per mini-batch and the mini-batch order
    shuffles deltas into v_pos / v_neg rows cumulatively, we process triples
    sequentially in a detach-recompute loop: this is O(n_triples) but each step
    is a d-dim vec update.
    """
    device = u.device
    d = u.shape[0]
    lr = float(cfg["lr"])
    batch = int(cfg["local_batch"])

    # Build triples that the real client used.
    triples = _reproduce_triples(uid, train_items, cfg["_n_real_items"],
                                 local_epochs=1, round_idx=round_idx)
    if triples.shape[0] == 0:
        return torch.zeros_like(global_item_emb)

    # Track the item_emb as a differentiable tensor seeded at the global state.
    # Reproduces fl/client.py::local_train *exactly*: per mini-batch the BPR
    # loss is `.mean()`d (so gradients scale by 1/|batch|), and SGD updates
    # BOTH item_emb AND user_emb every step.
    V = global_item_emb.clone()
    V.requires_grad_(False)
    delta = torch.zeros_like(V)
    u_now = u  # candidate u; will move during the simulated SGD trajectory
    t = torch.from_numpy(triples).to(device)
    for start in range(0, t.shape[0], batch):
        mini = t[start:start + batch]
        B_actual = mini.shape[0]
        pos_ids = mini[:, 1]
        neg_ids = mini[:, 2]
        V_now_pos = V[pos_ids] + delta[pos_ids]
        V_now_neg = V[neg_ids] + delta[neg_ids]
        pos_s = (V_now_pos * u_now.unsqueeze(0)).sum(-1)
        neg_s = (V_now_neg * u_now.unsqueeze(0)).sum(-1)
        kappa = torch.sigmoid(neg_s - pos_s)     # σ(-margin)
        # Real client uses .mean() over batch → scale gradient by 1/B_actual.
        # Per-triple SGD update on V: Δv_pos = +(lr/B)·κ·u, Δv_neg = -(lr/B)·κ·u.
        scale = lr / B_actual
        delta_pos = (scale * kappa).unsqueeze(-1) * u_now.unsqueeze(0)
        delta_neg = -delta_pos
        delta.index_add_(0, pos_ids, delta_pos)
        delta.index_add_(0, neg_ids, delta_neg)
        # u also moves: ∂loss/∂u = -(1/B)·Σ κ_i·(v_pos_i - v_neg_i).
        # SGD: u_new = u - lr·grad = u + (lr/B)·Σ κ·(v_pos - v_neg).
        du = (scale * kappa).unsqueeze(-1) * (V_now_pos - V_now_neg)  # (B, d)
        u_now = u_now + du.sum(0)
    return delta


def simulate_local_delta_lightgcn(u: torch.Tensor, uid: int, train_items: List[int],
                                  global_item_emb: torch.Tensor, nbr_idx: torch.Tensor,
                                  hops: int, cfg: dict, round_idx: int) -> torch.Tensor:
    """LightGCN: user representation passes through graph propagation. We
    compute propagated ũ as a differentiable function of u (and fixed item
    neighbors) and then run the same MF-style BPR loop against ũ.
    """
    if len(train_items) == 0:
        return torch.zeros_like(global_item_emb)
    v_nbrs = global_item_emb[nbr_idx]     # (M, d)
    M = v_nbrs.shape[0]
    inv_sqrt_M = 1.0 / max(M, 1) ** 0.5

    # K_hops-layer propagation, identical to models/lightgcn.py::_propagate.
    u_layers = [u]
    cur_u = u
    cur_v = v_nbrs
    for _ in range(hops):
        new_u = inv_sqrt_M * cur_v.sum(dim=0)
        new_v = inv_sqrt_M * cur_u.unsqueeze(0).expand(M, -1)
        cur_u, cur_v = new_u, new_v
        u_layers.append(cur_u)
    u_tilde = torch.stack(u_layers, dim=0).mean(dim=0)
    return simulate_local_delta_mf(u_tilde, uid, train_items, global_item_emb, cfg, round_idx)


def _ncf_score_functional(u: torch.Tensor, v: torch.Tensor, mlp, fusion_w: torch.Tensor,
                          d: int) -> torch.Tensor:
    """Pure functional NCF forward that treats `u` as a plain tensor (grad-enabled)
    rather than model.user_emb.weight (nn.Parameter).

    Mirrors NCFModel._score_batch but lets grad flow back into `u`.
    """
    # u: [d], v: [B, d] → score: [B]
    u_b = u.unsqueeze(0).expand_as(v)            # [B, d]
    gmf = u_b * v                                # [B, d]
    mlp_out = mlp(torch.cat([u_b, v], dim=-1))   # [B, h_last]
    fused = torch.cat([gmf, mlp_out], dim=-1)    # [B, d + h_last]
    return (fused * fusion_w.unsqueeze(0)).sum(-1)


def simulate_local_delta_ncf(u: torch.Tensor, uid: int, train_items: List[int],
                             ncf_model: NCFModel, cfg: dict, round_idx: int) -> torch.Tensor:
    """NCF per-round item-emb delta simulator.

    To keep `u` in the gradient graph end-to-end, we run the BPR forward as a
    pure function of u (not via ncf.forward_pos_neg which reads from
    ncf.user_emb.weight — an nn.Parameter that breaks gradient flow when we
    .copy_() u into it).

    Plain SGD, 1 local_epoch, same triple order as the real client.
    """
    device = u.device
    n_items = ncf_model.item_emb.weight.shape[0]
    d = ncf_model.d
    # Frozen-weight copies (no grad needed; attacker assumes knowledge of these)
    V_init = ncf_model.item_emb.weight.detach().clone()           # (n_items, d)
    mlp = ncf_model.mlp                                           # shared module, weights frozen during sim
    fusion_w = ncf_model.fusion.weight.detach().clone().squeeze(0)  # (d + h_last,)

    triples = _reproduce_triples(uid, train_items, cfg["_n_real_items"],
                                 local_epochs=1, round_idx=round_idx)
    if triples.shape[0] == 0:
        return torch.zeros_like(V_init)

    t = torch.from_numpy(triples).to(device)
    lr = float(cfg["lr"])
    batch = int(cfg["local_batch"])

    # Track delta = V - V_init. Each mini-batch gradient is computed via
    # second-order autograd (create_graph=True) so the final delta depends
    # on u end-to-end. We use fresh requires_grad leaves for V_now_pos/neg
    # each iteration; the grad of loss w.r.t. those leaves still carries
    # the u-dependency through the MLP + fusion path.
    delta = torch.zeros_like(V_init)
    for start in range(0, t.shape[0], batch):
        mini = t[start:start + batch]
        pos_ids = mini[:, 1]
        neg_ids = mini[:, 2]
        # Value = V_init[ids] + delta[ids]; leaf for autograd.grad.
        V_now_pos = (V_init[pos_ids] + delta[pos_ids]).detach().clone().requires_grad_(True)
        V_now_neg = (V_init[neg_ids] + delta[neg_ids]).detach().clone().requires_grad_(True)
        pos_s = _ncf_score_functional(u, V_now_pos, mlp, fusion_w, d)
        neg_s = _ncf_score_functional(u, V_now_neg, mlp, fusion_w, d)
        loss = -F.logsigmoid(pos_s - neg_s).mean()
        grad_V_pos, grad_V_neg = torch.autograd.grad(
            loss, [V_now_pos, V_now_neg], create_graph=True,
        )
        update = torch.zeros_like(delta)
        update = update.index_add(0, pos_ids, -lr * grad_V_pos)
        update = update.index_add(0, neg_ids, -lr * grad_V_neg)
        delta = delta + update
    return delta


def simulate_delta(u: torch.Tensor, uid: int, train_items: List[int],
                   server, round_idx: int) -> torch.Tensor:
    """Dispatch to the per-model simulator.

    Returns predicted Δitem_emb of shape [n_items, d].
    """
    model = server.model
    cfg = dict(server.cfg)
    cfg["_n_real_items"] = getattr(model, "n_items_original", model.n_items)
    if isinstance(model, MFModel):
        return simulate_local_delta_mf(
            u, uid, train_items, model.item_emb.weight.detach(), cfg, round_idx
        )
    if isinstance(model, LightGCNModel):
        nbr_idx = torch.as_tensor(train_items, dtype=torch.long, device=u.device)
        return simulate_local_delta_lightgcn(
            u, uid, train_items, model.item_emb.weight.detach(),
            nbr_idx, model.hops, cfg, round_idx,
        )
    if isinstance(model, NCFModel):
        return simulate_local_delta_ncf(u, uid, train_items, model, cfg, round_idx)
    raise TypeError(f"unknown model type {type(model)}")
