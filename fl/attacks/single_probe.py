"""Single-probe ablation of Paired-Probe.

Injects K single probe items (not pairs) into the catalog; each probe is used
as a synthetic NEGATIVE paired with a randomly sampled real positive from the
user's own history. The per-user BPR sigmoid coefficient
    κ_u = σ(−u·v_pos_u)
varies across users because each user's real positive v_pos_u differs, so the
aggregated probe-row delta contains a per-user scalar that pinv cannot
disentangle. Expected: recovered cosine << paired-probe's cosine.

This is our own strawman ablation; no published paper performs this attack.
The point is to empirically demonstrate that the pairing construction is what
makes Paired-Probe work once the recommender passes its initial training phase.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from ..trip import SlidingWindowAllocator, TRIPServer
from ..trip.solver import solve_mf_lgcn
from ..utils import clone_state_dict
from .base import AttackBase


class SingleProbeAttack(AttackBase):
    name = "single_probe"

    def _init_single_probes(self, model, K: int) -> List[Tuple[int, int]]:
        """Extend item catalog by K rows (one per probe) and return a list of
        (real_pos_placeholder, probe_id) — the 'pos' slot will be overwritten
        per-client with the user's actual positive. Here we put -1 as a
        placeholder; client-side injection resolves it to a random real pos.
        """
        device = model.item_emb.weight.device
        sigma = float(model.item_emb.weight.std().item())
        sigma = max(sigma, 1e-3)
        rng = torch.Generator(device=device).manual_seed(1337)
        probes = torch.randn(K, model.d, generator=rng, device=device) * sigma
        model.extend_items(K)
        n_total = model.item_emb.weight.shape[0]
        probe_start = n_total - K
        with torch.no_grad():
            model.item_emb.weight.data[probe_start:probe_start + K] = probes
        return [(-1, probe_start + k) for k in range(K)]  # pos=-1 sentinel

    def prepare(self, server, bundle, attacked_uids: List[int]) -> None:
        with self._time("prepare_sec"):
            model = server.model
            cfg = self.cfg
            K = int(cfg["K"])
            W = int(cfg["W"])
            T_factor = int(cfg["T_factor"])
            N = len(attacked_uids)

            model.n_items_original = bundle.n_items
            probe_pairs = self._init_single_probes(model, K)
            allocator = SlidingWindowAllocator(N=N, K=K, W=W, T_factor=T_factor)

            # Use a dedicated single-probe server that resolves pos=-1 to
            # the sampled user's actual positive per round.
            trip = _SingleProbeServer(server.model, bundle, cfg, server.device)
            trip.user_states = server.user_states
            trip.round = server.round

            from ..models import MFModel, LightGCNModel
            if isinstance(model, MFModel):
                reps = int(cfg["reps_per_pair"]["mf"])
            elif isinstance(model, LightGCNModel):
                reps = int(cfg["reps_per_pair"]["lightgcn"])
            else:
                reps = int(cfg["reps_per_pair"]["ncf"])

            G = trip.attack_single_probe(attacked_uids, allocator, probe_pairs, reps)
            A = allocator.build_A()

            self._state["G"] = G
            self._state["A"] = A
            self._state["probe_pairs"] = probe_pairs

    def solve(self, server, bundle, attacked_uids: List[int]) -> np.ndarray:
        with self._time("solve_sec"):
            G = self._state["G"]
            A = self._state["A"]
            # Same pinv math as paired-probe, but G here comes from a single
            # probe-row delta (no (neg-pos)/2 diff), so per-user κ_u scalars
            # are not cancelled and the solution is contaminated.
            # For NCF the ridge-GMF step is also less well-posed; we use the
            # MF/LGCN-style solver even for NCF (since we don't have paired
            # probes to compute C_MLP) and accept the degraded cos.
            U_hat = solve_mf_lgcn(G, A)
        return U_hat


class _SingleProbeServer(TRIPServer):
    """TRIPServer variant that records only the probe row's delta (not diff)."""

    def attack_single_probe(self, attacked_uids, allocator, probe_pairs, reps_per_pair,
                            progress: bool = True):
        from tqdm import tqdm
        K = allocator.K
        d = self.cfg["d"]
        T = allocator.T
        G = np.zeros((T, K, d), dtype=np.float32)
        rng = np.random.default_rng(self.cfg.get("seed", 42) + 20241)

        iterator = range(T)
        if progress:
            iterator = tqdm(iterator, desc="single-probe attack", ncols=80)

        for t in iterator:
            snap = self.snapshot()
            try:
                # Per-round: resolve pos=-1 to each user's random real positive.
                # This is the essence of single-probe: the probe acts as negative
                # against a DIFFERENT pos per user, giving per-user κ_u variance.
                local_assign = allocator.assignments_for_round(t)
                probe_assign: Dict[int, list] = {}
                forced = set()
                client_rng = np.random.default_rng(
                    self.cfg.get("seed", 42) + 91 * t
                )
                for local_uid, ks in local_assign.items():
                    real_uid = attacked_uids[local_uid]
                    forced.add(real_uid)
                    user_train = self.bundle.train_pos.get(real_uid, [])
                    if not user_train:
                        continue
                    assign_list = []
                    for k in ks:
                        _, probe_id = probe_pairs[k]
                        real_pos = int(user_train[client_rng.integers(0, len(user_train))])
                        assign_list.append((real_pos, probe_id))
                    probe_assign[real_uid] = assign_list
                sampled = self.sample_clients(
                    self.cfg["clients_per_round"], forced=forced, rng=rng,
                )
                assert set(forced).issubset(set(sampled))

                delta = self.run_round(sampled, probe_assign=probe_assign,
                                       reps_per_pair=reps_per_pair)
                item_delta = delta["item_emb.weight"]
                n_sampled = len(sampled)
                for k in range(K):
                    _, probe_id = probe_pairs[k]
                    # Single-probe: record probe row's raw delta scaled by |sampled|
                    # (undo FedAvg averaging → pure sum over window).
                    G[t, k] = (item_delta[probe_id] * n_sampled).detach().cpu().numpy()
            finally:
                self.restore(snap)
        return G
