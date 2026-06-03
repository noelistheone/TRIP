"""Probe-item construction.

init_paired_probes
    Append 2K rows to the item catalog: K (pos, neg) pairs. Each pair is two
    near-identical small-random embeddings (eps_rel = 1e-3 of item-emb std);
    the small magnitude keeps user×probe scores ≈ 0 so the BPR sigmoid
    coefficient κ_u = σ(-(u·p_pos - u·p_neg)) ≈ σ(0) ≈ 0.5 across ALL users.
    Pairing eliminates per-user κ_u variance (framework §3, §4.5).

overwrite_ncf_probes_saturating
    For NCF, the same near-zero probe magnitude is insufficient because the
    MLP branch's gradient remains active. We boost the probe items into the
    "saturating" regime of the MLP's ReLUs: large probe magnitude (M·std)
    drives ReLU activations into either fully-on or fully-off states, so
    the MLP gradient w.r.t. e_i is approximately constant across users —
    leaving the GMF branch as the only user-dependent signal. Returns the
    *baseline GMF response* (scalar per probe, used by solve_ncf to invert
    the diag(w_GMF) scaling).
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from ..models import FLModel, NCFModel


def init_paired_probes(model: FLModel, K: int, eps_rel: float) -> Tuple[List[Tuple[int, int]], None]:
    """Append 2K probe rows; return list of (pos_id, neg_id) pairs."""
    device = model.item_emb.weight.device
    sigma = float(model.item_emb.weight.std().item())
    eps = max(eps_rel * sigma, 1e-6)
    n_old = model.item_emb.weight.shape[0]
    model.extend_items(2 * K)
    pair_ids: List[Tuple[int, int]] = []
    g = torch.Generator(device=device).manual_seed(2024_1019)
    with torch.no_grad():
        for k in range(K):
            base = torch.randn(model.d, generator=g, device=device) * eps
            jitter = torch.randn(model.d, generator=g, device=device) * (eps * 0.1)
            pos_id = n_old + 2 * k
            neg_id = n_old + 2 * k + 1
            model.item_emb.weight.data[pos_id] = base + jitter
            model.item_emb.weight.data[neg_id] = base - jitter
            pair_ids.append((pos_id, neg_id))
    return pair_ids, None


def overwrite_ncf_probes_saturating(model: NCFModel,
                                    pair_ids: List[Tuple[int, int]],
                                    M: float, eps_rel: float) -> torch.Tensor:
    """For NCF: bump probe magnitudes to saturating regime of the MLP first
    layer. Random unit-direction probes ~M*σ_item magnitude; ~half of the
    layer-1 ReLUs go OFF, the rest contribute residual MLP gradient.

    The MLP residual is further suppressed during the attack-only window by
    a 10× damping of the MLP body weights (applied per probe round in
    TRIPServer.attack(), then auto-undone by the snapshot/restore wrapper
    so post-attack model state is bit-identical to post-warmup state — the
    eval phase sees the original NCF scoring function).

    Returns:
        w_gmf: (d,) — the diagonal GMF branch weights, used by solve_ncf
                to invert the per-coord GMF scaling.
    """
    device = model.item_emb.weight.device
    K = len(pair_ids)
    d = model.d
    sigma = float(model.item_emb.weight[: model.n_items_original].std().item())
    big = max(M * sigma, 1e-3)
    eps = max(eps_rel * sigma, 1e-6)

    g = torch.Generator(device=device).manual_seed(2024_1119)
    with torch.no_grad():
        for k in range(K):
            base = torch.randn(d, generator=g, device=device)
            base = base / (base.norm() + 1e-12) * big
            jitter = torch.randn(d, generator=g, device=device) * eps
            pos_id, neg_id = pair_ids[k]
            model.item_emb.weight.data[pos_id] = base + jitter
            model.item_emb.weight.data[neg_id] = base - jitter

    fusion_w = model.fusion.weight.detach().clone().squeeze(0)
    w_gmf = fusion_w[:d].detach().clone()
    return w_gmf


def damp_ncf_mlp(model: NCFModel, factor: float = 0.1) -> None:
    """In-place: damp NCF MLP body weights and the MLP-half of fusion. Used
    inside the attack round (between snapshot and restore) to suppress the
    residual MLP gradient that contaminates GMF recovery."""
    d = model.d
    with torch.no_grad():
        for layer in model.mlp:
            if hasattr(layer, "weight") and layer.weight is not None:
                layer.weight.data.mul_(factor)
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.data.mul_(factor)
        model.fusion.weight.data[:, d:] = model.fusion.weight.data[:, d:] * factor
