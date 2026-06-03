"""Client-side local training for FedAvg.

The server simulator calls `local_train` once per sampled client per round, passing in:
  - the current global shared parameters (dict[str, Tensor])
  - the client's private user embedding (d-dim tensor)
  - the client's train items
  - optionally, a list of probe pair ids (pos_id, neg_id) to inject as extra BPR triples

Returns (delta_shared, new_user_emb).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import bpr_sample
from .models import FLModel, LightGCNModel


def _build_triples(uid: int, train_items: List[int], n_real_items: int,
                   probe_pair_ids: Optional[List[Tuple[int, int]]],
                   reps_per_pair: int, rng: np.random.Generator,
                   probes_only: bool = False) -> np.ndarray:
    """Build BPR triples for one client.

    If probes_only=True AND probe_pair_ids is non-empty, real-item BPR is
    SKIPPED entirely. This is needed for the TRIP attack rounds: heavy users
    (e.g. amazon-book first 500 average 146 items) dilute the probe-row
    gradient through `loss.mean()` to (32·σ/178)·u ≈ 0.1u, four times
    weaker than for sparse users — recovery cos collapses. With probes-only,
    every user contributes the same `-σ·u` per probe step, restoring clean
    signal independent of train-item count. Side effect: user embedding
    barely drifts (paired probes have ≈ identical scores → user grad ≈ 0),
    which is exactly what the linearization in TRIP §4.4 assumes.
    """
    if probes_only and probe_pair_ids:
        probe_rows = []
        for (pos, neg) in probe_pair_ids:
            for _ in range(reps_per_pair):
                probe_rows.append([uid, pos, neg])
        return np.asarray(probe_rows, dtype=np.int64)
    n_real = max(len(train_items), 1)
    real_triples = bpr_sample(uid, train_items, n_real_items, n_samples=n_real, rng=rng)
    if probe_pair_ids:
        probe_rows = []
        for (pos, neg) in probe_pair_ids:
            for _ in range(reps_per_pair):
                probe_rows.append([uid, pos, neg])
        probe_triples = np.asarray(probe_rows, dtype=np.int64)
        all_t = np.concatenate([real_triples, probe_triples], axis=0)
    else:
        all_t = real_triples
    rng.shuffle(all_t, axis=0)
    return all_t


def local_train(model: FLModel,
                global_shared: Dict[str, torch.Tensor],
                user_emb_init: torch.Tensor,
                uid: int,
                train_items: List[int],
                n_real_items: int,
                cfg: dict,
                probe_pair_ids: Optional[List[Tuple[int, int]]] = None,
                reps_per_pair: int = 16,
                round_idx: int = 0,
                ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Run one client's local SGD; return (delta_shared, new_user_emb).

    Invariant: BPR negatives are sampled from [0, n_real_items) so probe rows at indices
    >= n_real_items never leak into the real BPR stream.
    """
    device = next(model.parameters()).device

    # Load global shared params into the model; install user's private row.
    with torch.no_grad():
        sd = model.state_dict()
        for k, v in global_shared.items():
            sd[k] = v.to(device).clone()
        model.load_state_dict(sd, strict=True)
        model.set_user_row(user_emb_init)

    # LightGCN: install user's subgraph (train items). Probes are NOT added.
    if isinstance(model, LightGCNModel):
        model.set_neighbors(train_items)

    # Warmup can use Adam (higher-quality trained embeddings, hence better ranking),
    # but the attack phase MUST use plain SGD because the paired-probe math assumes
    # delta = -η·grad. Route via `optimizer` kwarg (default SGD).
    # `weight_decay` stabilizes MF on sparse large catalogs (otherwise user-emb norms
    # explode). Attack phase sets wd=0 so probe-row deltas stay linear in grad.
    opt_name = str(cfg.get("optimizer", "sgd")).lower()
    wd = float(cfg.get("weight_decay", 0.0))
    if opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=wd)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], weight_decay=wd)
    rng = np.random.default_rng(seed=(uid + 1) * 1_000_003 + round_idx)

    n_samples = 0
    loss_sum = 0.0
    probes_only = bool(cfg.get("attack_probes_only", False))
    for _ in range(cfg["local_epochs"]):
        triples = _build_triples(uid, train_items, n_real_items, probe_pair_ids,
                                 reps_per_pair, rng, probes_only=probes_only)
        if triples.shape[0] == 0:
            continue
        t = torch.from_numpy(triples).to(device)
        B = cfg["local_batch"]
        for start in range(0, t.shape[0], B):
            batch = t[start:start + B]
            pos = batch[:, 1]
            neg = batch[:, 2]
            pos_s, neg_s = model.forward_pos_neg(pos, neg)
            loss = -F.logsigmoid(pos_s - neg_s).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n_samples += batch.shape[0]
            loss_sum += float(loss.item()) * batch.shape[0]

    # Compute delta on shared keys.
    post = model.state_dict()
    delta = {k: post[k].detach().clone() - global_shared[k].to(device)
             for k in model.shared_keys()}
    new_u = model.get_user_row()
    return delta, new_u
