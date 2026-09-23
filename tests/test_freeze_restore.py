"""Verify TRIPServer snapshot/restore is bit-exact across an attack phase."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from fl.data import load_dataset
from fl.models import make_model
from fl.trip import SlidingWindowAllocator, TRIPServer, init_paired_probes


def test_snapshot_restore_bitexact():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["clients_per_round"] = 16
    cfg["N_attack"] = 50
    cfg["local_epochs"] = 1
    cfg["attack_probes_only"] = True
    bundle = load_dataset("lastfm")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = make_model("mf", bundle.n_users, bundle.n_items, cfg["d"], cfg).to(device)
    model.n_items_original = bundle.n_items
    trip = TRIPServer(model, bundle, cfg, device)

    # A couple of warmup rounds to get non-trivial state.
    attacked = list(bundle.train_user_ids[:50])
    trip.warmup(2, attacked_uids=attacked, clients_per_round=16, progress=False)

    # Extend catalog with probes BEFORE the snapshot — the real invocation order.
    alloc = SlidingWindowAllocator(N=50, K=5, W=10, T_factor=1)
    pair_ids, _ = init_paired_probes(model, 5, cfg["probes"]["eps_rel"])
    model.n_items_original = bundle.n_items   # reset after extend
    snap = trip.snapshot()

    _ = trip.attack(attacked, alloc, pair_ids, reps_per_pair=4, progress=False)

    # After the attack, trip must be restored to the pre-attack state exactly.
    sd = model.state_dict()
    for k, v in snap["shared"].items():
        assert torch.equal(sd[k], v), f"{k} differs after attack restore"
    for u, v in snap["users"].items():
        assert torch.equal(trip.user_states[u], v), f"user {u} differs after restore"
    assert trip.round == snap["round"], "round counter not restored"
    print(f"  snapshot/restore bit-exact over {len(snap['shared'])} tensors "
          f"and {len(snap['users'])} user rows OK")


if __name__ == "__main__":
    test_snapshot_restore_bitexact()
    print("test_freeze_restore: pass")
