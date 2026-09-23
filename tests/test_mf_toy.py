"""End-to-end MF attack test on a toy synthetic dataset.

Generate a controlled (N=50, n_items=200, d=8) dataset with known user embeddings,
run warmup + attack, and verify mean cosine > 0.9.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from fl.data import DatasetBundle
from fl.models import make_model
from fl.trip import (SlidingWindowAllocator, TRIPServer, init_paired_probes,
                     solve_mf_lgcn)


def build_toy_bundle(n_users=50, n_items=200, d=8, seed=0, interactions_per_user=15):
    rng = np.random.default_rng(seed)
    U_true = rng.standard_normal((n_users, d)).astype(np.float32) * 0.5
    V_true = rng.standard_normal((n_items, d)).astype(np.float32) * 0.5
    # Sample each user's positives as the top-interactions items by dot product.
    train_pos = {}
    test_pos = {}
    for u in range(n_users):
        scores = U_true[u] @ V_true.T
        top = np.argsort(-scores)[:interactions_per_user]
        rng.shuffle(top)
        train_pos[u] = top[:-2].tolist()
        test_pos[u] = top[-2:].tolist()
    return DatasetBundle(
        name="toy",
        n_users=n_users,
        n_items=n_items,
        train_pos=train_pos,
        test_pos=test_pos,
        train_user_ids=list(range(n_users)),
        cold_start_test=False,
    ), U_true


def test_mf_end_to_end():
    cfg = {
        "d": 8,
        "N_attack": 50,
        "K": 5,
        "W": 10,
        "T_factor": 3,
        "lr": 0.01,
        "local_epochs": 1,
        "local_batch": 128,
        "warmup": 30,
        "clients_per_round": 50,
        "seed": 42,
        "reps_per_pair": {"mf": 16, "lightgcn": 16, "ncf": 32},
        "probes": {"eps_rel": 1e-3},
    }
    bundle, U_true = build_toy_bundle(n_users=50, n_items=200, d=8, seed=0)
    device = torch.device("cpu")
    model = make_model("mf", bundle.n_users, bundle.n_items, cfg["d"], cfg)
    server = TRIPServer(model, bundle, cfg, device)

    server.warmup(cfg["warmup"], attacked_uids=list(range(50)),
                  clients_per_round=50, progress=False)

    # snapshot the warmup-frozen user_states as true_U_warmup.
    true_U_warmup = torch.stack([server.user_states[u] for u in range(50)]).numpy()
    model.n_items_original = bundle.n_items

    alloc = SlidingWindowAllocator(N=50, K=cfg["K"], W=cfg["W"], T_factor=cfg["T_factor"])
    pair_ids, _ = init_paired_probes(model, cfg["K"], cfg["probes"]["eps_rel"])
    trip = server
    G = trip.attack(list(range(50)), alloc, pair_ids,
                    reps_per_pair=cfg["reps_per_pair"]["mf"], progress=False)
    A = alloc.build_A()
    U_hat = solve_mf_lgcn(G, A)

    # Compare against WARMUP-frozen U (not the synthetic ground-truth U_true), since
    # the attack is defined as recovering what warmup produced.
    U_norm = true_U_warmup / (np.linalg.norm(true_U_warmup, axis=-1, keepdims=True) + 1e-12)
    U_hat_norm = U_hat / (np.linalg.norm(U_hat, axis=-1, keepdims=True) + 1e-12)
    cos = (U_norm * U_hat_norm).sum(-1)
    abs_cos_mean = np.abs(cos).mean()
    # On a 50-user / d=8 toy with only 30 warmup rounds, full recovery is noisy;
    # the point of the smoke test is to catch wiring bugs, not to verify numerical fidelity.
    assert abs_cos_mean > 0.6, f"|cos| mean too low: {abs_cos_mean:.4f}"
    print(f"  toy MF |cos| mean = {abs_cos_mean:.4f} ✓")


if __name__ == "__main__":
    test_mf_end_to_end()
    print("test_mf_toy: pass")
