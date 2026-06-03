"""Verify TRIPServer snapshot/restore is bit-exact."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from fl.data import load_dataset
from fl.models import build_model
from fl.trip import SlidingWindowAllocator, TRIPServer, init_paired_probes


def test_snapshot_restore_bitexact():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["clients_per_round"] = 16
    cfg["N_attack"] = 50
    bundle = load_dataset("lastfm")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model("mf", bundle.n_users, bundle.n_items, cfg["d"], cfg)
    model.n_items_original = bundle.n_items
    trip = TRIPServer(model, bundle, cfg, device)

    # Warmup a few rounds to get non-trivial state.
    trip.warmup(2, progress=False)

    # Extend catalog with probes BEFORE snapshot — this is the real invocation order.
    attacked = bundle.train_user_ids[:50]
    alloc = SlidingWindowAllocator(N=50, K=5, W=10, T_factor=1)
    pair_ids, _ = init_paired_probes(model, K=5)
    model.n_items_original = bundle.n_items   # reset after extend
    snap = trip.snapshot()

    _ = trip.attack(attacked, alloc, pair_ids, reps_per_pair=4, progress=False)

    # After attack, trip should be restored to the pre-attack state.
    sd = model.state_dict()
    for k in snap[0]:
        assert torch.equal(sd[k], snap[0][k]), f"{k} differs after attack restore"
    for u in snap[1]:
        assert torch.equal(trip.user_states[u], snap[1][u]), f"user {u} differs"


if __name__ == "__main__":
    test_snapshot_restore_bitexact()
    print("freeze/restore bit-exact ✓")
