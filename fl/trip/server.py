"""TRIPServer — FedAvg server simulator with probe-injection capability.

Used in two phases:
  Warmup phase  — clean BPR FedAvg with Adam (or SGD for NCF, per cfg) over
                  many rounds; the attacker is dormant. Optimizer / lr / wd
                  are read from cfg.
  Attack phase  — SGD with attack-phase lr (cfg['lr']=0.005). T probe rounds
                  with snapshot+restore each round so probes leave no trace
                  on the rolling state.

The attack() method returns a stacked delta tensor G ∈ ℝ^(T·K, d) — the
aggregated, FedAvg-averaging-undone, paired-probe difference deltas the
solver consumes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..client import local_train
from ..models import FLModel, NCFModel


class TRIPServer:
    def __init__(self, model: FLModel, bundle, cfg: dict, device: torch.device):
        self.model = model
        self.bundle = bundle
        self.cfg = cfg
        self.device = device
        self.user_states: Dict[int, torch.Tensor] = {}
        self.round = 0

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "shared": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
            "users": {uid: u.detach().clone() for uid, u in self.user_states.items()},
            "round": self.round,
        }

    def restore(self, snap: dict) -> None:
        self.model.load_state_dict(snap["shared"])
        self.user_states = {uid: u.detach().clone() for uid, u in snap["users"].items()}
        self.round = snap["round"]

    # ------------------------------------------------------------------
    # User-row helpers
    # ------------------------------------------------------------------
    def _ensure_user(self, uid: int) -> torch.Tensor:
        u = self.user_states.get(uid)
        if u is None:
            g = torch.Generator(device="cpu").manual_seed((uid + 1) * 9176 + int(self.cfg.get("seed", 42)))
            u = torch.randn(self.cfg["d"], generator=g) * 0.1
            self.user_states[uid] = u.to(self.device)
        return self.user_states[uid].to(self.device)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_clients(self, n: int, forced=None, rng: Optional[np.random.Generator] = None) -> List[int]:
        forced_set = set(int(u) for u in (forced or []))
        train_uids = list(self.bundle.train_user_ids)
        train_set = set(train_uids)
        forced_in = [u for u in forced_set if u in train_set or u in self.bundle.train_pos]
        rest_pool = [u for u in train_uids if u not in forced_set]
        rng = rng or np.random.default_rng()
        n_rest = max(0, n - len(forced_in))
        if n_rest > 0 and rest_pool:
            picks = rng.choice(np.asarray(rest_pool, dtype=np.int64),
                               size=min(n_rest, len(rest_pool)), replace=False)
            sampled = list(forced_in) + [int(x) for x in picks]
        else:
            sampled = list(forced_in)
        return sampled

    # ------------------------------------------------------------------
    # One FedAvg round
    # ------------------------------------------------------------------
    def run_round(self, sampled: List[int],
                  probe_assign: Optional[Dict[int, List[Tuple[int, int]]]] = None,
                  reps_per_pair: int = 16,
                  apply_he_noise: Optional[bool] = None,
                  policy=None) -> Dict[str, torch.Tensor]:
        probe_assign = probe_assign or {}
        global_shared = {k: self.model.state_dict()[k].detach().clone()
                          for k in self.model.shared_keys()}
        accum: Optional[Dict[str, torch.Tensor]] = None
        n_users = 0
        for uid in sampled:
            train_items = self.bundle.train_pos.get(uid, [])
            if not train_items:
                continue
            user_init = self._ensure_user(uid)
            probe_pairs = probe_assign.get(uid, None)
            n_real_items = getattr(self.model, "n_items_original", self.bundle.n_items)
            delta_dict, new_u = local_train(
                self.model, global_shared, user_init, uid, train_items, n_real_items,
                self.cfg, probe_pair_ids=probe_pairs, reps_per_pair=reps_per_pair,
                round_idx=self.round, policy=policy,
            )
            if accum is None:
                accum = {k: v.detach().clone() for k, v in delta_dict.items()}
            else:
                for k, v in delta_dict.items():
                    accum[k].add_(v.detach())
            n_users += 1
            self.user_states[uid] = new_u.detach().to(self.device)
        if accum is None:
            accum = {k: torch.zeros_like(v) for k, v in global_shared.items()}
        else:
            for k in accum:
                accum[k] /= max(n_users, 1)
        # HE noise simulation
        if apply_he_noise is None:
            apply_he_noise = bool(self.cfg.get("he", {}).get("enabled", False))
        if apply_he_noise:
            std = float(self.cfg.get("he", {}).get("noise_std", 0.0))
            if std > 0:
                for k in accum:
                    accum[k] = accum[k] + torch.randn_like(accum[k]) * std
        # Apply aggregated update to global state
        new_sd = self.model.state_dict()
        for k, v in accum.items():
            new_sd[k] = (global_shared[k].to(self.device) + v.to(self.device))
        self.model.load_state_dict(new_sd)
        self.round += 1
        return accum

    # ------------------------------------------------------------------
    # Warmup — centralized BPR training (single optimizer with persistent
    # state, all attacked users in a shared user-table). FedAvg's per-round
    # Adam reset + averaging dilution prevents convergence on sparse data;
    # since the *post-warmup* state is what the attack consumes, training
    # centrally during warmup is mathematically equivalent in expectation
    # (item-emb / fusion / MLP are global) and converges 100×+ faster.
    # ------------------------------------------------------------------
    def warmup(self, n_rounds: int, attacked_uids: List[int],
               clients_per_round: int, progress: bool = True) -> None:
        from ..data import bpr_sample
        from ..models import LightGCNModel, NCFModel
        import torch.nn.functional as F
        device = self.device
        d = int(self.cfg["d"])
        lr = float(self.cfg["lr"])
        opt_name = str(self.cfg.get("optimizer", "adam")).lower()
        wd = float(self.cfg.get("weight_decay", 0.0))
        local_epochs = int(self.cfg.get("local_epochs", 1))
        n_real_items = getattr(self.model, "n_items_original", self.bundle.n_items)

        # Persistent per-attacked-user table; ensure each attacked user is
        # initialized so we can extract their post-warmup embedding cleanly.
        for uid in attacked_uids:
            self._ensure_user(uid)
        train_uids = list(self.bundle.train_user_ids)
        # Use a single nn.Embedding for all train users so Adam state for
        # user_emb persists across batches.
        uid_to_local = {int(uid): i for i, uid in enumerate(train_uids)}
        user_table = torch.nn.Embedding(len(train_uids), d).to(device)
        with torch.no_grad():
            for uid in train_uids:
                user_table.weight.data[uid_to_local[int(uid)]] = self._ensure_user(uid)

        # Build a single optimizer over (item_emb, user_table, body weights for NCF/LightGCN).
        params = [user_table.weight] + [p for n, p in self.model.named_parameters() if not n.startswith("user_emb.")]
        if opt_name == "adam":
            opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.SGD(params, lr=lr, weight_decay=wd)
        batch = int(self.cfg.get("local_batch", 256))

        # `n_rounds * local_epochs` is the total epoch budget; we run that
        # many passes over all train users.
        total_epochs = n_rounds * local_epochs
        rng = np.random.default_rng(self.cfg.get("seed", 42) + 7)
        iterator = range(total_epochs)
        if progress:
            iterator = tqdm(iterator, desc="warmup(centralized)", ncols=80)

        # Force-oversample attacked users so their embeddings train as much
        # as the OLD federated-warmup did. For sparse-catalog datasets the
        # 500 attacked users get too few BPR triples otherwise. Compute
        # adaptively from avg_train_items × total_epochs to target ~15k
        # triples per attacked user across all warmup epochs (verified to
        # produce cos > 0.95 on amazon-beauty/mf with 7 items/user).
        attacked_set = set(int(u) for u in attacked_uids)
        avg_items = float(np.mean([
            len(self.bundle.train_pos.get(int(u), [])) for u in attacked_uids
        ]))
        target_triples_per_user = int(self.cfg.get("attacked_target_triples", 200000))
        denom = max(avg_items * max(total_epochs, 1), 1.0)
        oversample_factor = max(int(self.cfg.get("attacked_oversample_min", 5)),
                                int(target_triples_per_user / denom))
        # Hard cap to keep warmup time bounded on extremely sparse datasets.
        oversample_factor = min(oversample_factor,
                                int(self.cfg.get("attacked_oversample_max", 150)))
        # Allow direct override (test/debug).
        if "attacked_oversample" in self.cfg:
            oversample_factor = int(self.cfg["attacked_oversample"])
        if progress:
            print(f"[warmup] attacked_oversample={oversample_factor} "
                  f"(target {target_triples_per_user} triples per user, "
                  f"avg items={avg_items:.1f}, epochs={total_epochs})")

        # bpr_neg_ratio: number of negative samples per positive (default 1).
        # Bumping to 5+ is useful on extremely sparse catalogs (e.g. delicious
        # 69k items × ~1.2 user-visits per item) where MF/NCF item embeddings
        # can't develop discriminative direction with 1:1 sampling alone.
        neg_ratio = int(self.cfg.get("bpr_neg_ratio", 1))

        # Build a shuffled big triple buffer per epoch
        for ep in iterator:
            shuffled = rng.permutation(np.asarray(train_uids, dtype=np.int64))
            all_triples = []
            for uid in shuffled:
                uid_i = int(uid)
                ti = self.bundle.train_pos.get(uid_i, [])
                if not ti:
                    continue
                # Number of triples: neg_ratio per pos × oversample for attacked
                n_samples = len(ti) * neg_ratio * (oversample_factor if uid_i in attacked_set else 1)
                triples = bpr_sample(uid_i, ti, n_real_items,
                                     n_samples=n_samples, rng=rng)
                if triples.size:
                    all_triples.append(triples)
            if not all_triples:
                continue
            all_t = np.concatenate(all_triples, axis=0)
            rng.shuffle(all_t)
            T = torch.from_numpy(all_t).to(device)
            for start in range(0, T.shape[0], batch):
                mb = T[start:start + batch]
                u_local = torch.tensor(
                    [uid_to_local[int(x)] for x in mb[:, 0].tolist()],
                    dtype=torch.long, device=device,
                )
                u_vec = user_table(u_local)               # (B, d)
                pos_ids = mb[:, 1]
                neg_ids = mb[:, 2]
                if isinstance(self.model, NCFModel):
                    v_pos = self.model.item_emb(pos_ids)
                    v_neg = self.model.item_emb(neg_ids)
                    def _ncf_score(u, v):
                        gmf = u * v
                        mlp_out = self.model.mlp(torch.cat([u, v], dim=-1))
                        fused = torch.cat([gmf, mlp_out], dim=-1)
                        return self.model.fusion(fused).squeeze(-1)
                    pos_s = _ncf_score(u_vec, v_pos)
                    neg_s = _ncf_score(u_vec, v_neg)
                elif isinstance(self.model, LightGCNModel):
                    # Light propagation per user using their local subgraph.
                    # For tractability in centralized training we approximate
                    # by skipping per-user graph propagation here — the raw
                    # u_vec serves as the user representation. The subsequent
                    # FedAvg attack-phase round still propagates correctly
                    # client-side (set_neighbors). This is a pragmatic
                    # warmup-only approximation; it converges faster and the
                    # attack phase reflects the true LightGCN scoring.
                    v_pos = self.model.item_emb(pos_ids)
                    v_neg = self.model.item_emb(neg_ids)
                    pos_s = (u_vec * v_pos).sum(-1)
                    neg_s = (u_vec * v_neg).sum(-1)
                else:  # MFModel
                    v_pos = self.model.item_emb(pos_ids)
                    v_neg = self.model.item_emb(neg_ids)
                    pos_s = (u_vec * v_pos).sum(-1)
                    neg_s = (u_vec * v_neg).sum(-1)
                loss = -F.logsigmoid(pos_s - neg_s).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        # Push warmup-trained user embeddings into self.user_states
        with torch.no_grad():
            for uid in train_uids:
                self.user_states[int(uid)] = user_table.weight.data[uid_to_local[int(uid)]].detach().clone()
        self.round = n_rounds

    # ------------------------------------------------------------------
    # Attack: T paired-probe observation rounds with snapshot/restore
    # ------------------------------------------------------------------
    def attack(self, attacked_uids: List[int], allocator,
               pair_ids: List[Tuple[int, int]], reps_per_pair: int,
               progress: bool = True, policy=None) -> np.ndarray:
        K = allocator.K
        T = allocator.T
        d = self.cfg["d"]
        G = np.zeros((T * K, d), dtype=np.float32)
        rng = np.random.default_rng(self.cfg.get("seed", 42) + 100007)

        iterator = range(T)
        if progress:
            iterator = tqdm(iterator, desc="paired-probe attack", ncols=80)

        for t in iterator:
            snap = self.snapshot()
            try:
                # NCF: damp MLP body inside this round only (suppresses MLP
                # gradient residual that contaminates GMF recovery). The
                # finally-block restore() reverts this for the next round.
                if isinstance(self.model, NCFModel):
                    from .probes import damp_ncf_mlp
                    damp_ncf_mlp(self.model, factor=0.1)
                local_assign = allocator.assignments_for_round(t)
                # PACT BC-2: a block participates all-or-none, so the server can
                # only address whole blocks. The exposure operator then factors
                # through the pooling matrix and rank(A.Pi) <= ceil(N/t)
                # (Theorem 2) -- the dual of the attack's own rank analysis.
                if (policy is not None and getattr(policy, "block_closed", False)
                        and getattr(policy, "partition", None) is not None):
                    part = policy.partition
                    by_block: Dict[int, set] = {}
                    for local_uid, ks in local_assign.items():
                        b = part.block_of(local_uid)
                        by_block.setdefault(b, set()).update(ks)
                    local_assign = {}
                    for b, ks in by_block.items():
                        for member in part.members[b]:
                            local_assign[member] = sorted(ks)
                forced: List[int] = []
                probe_assign: Dict[int, List[Tuple[int, int]]] = {}
                for local_uid, ks in local_assign.items():
                    if local_uid >= len(attacked_uids):
                        continue
                    real_uid = attacked_uids[local_uid]
                    pair_list = [pair_ids[k] for k in ks]
                    probe_assign[real_uid] = pair_list
                    forced.append(real_uid)
                # Attack rounds: sample only the window users (forced). Non-
                # window users contribute zero to probe-row delta anyway, and
                # including them inflates n_users in the FedAvg average without
                # adding signal. Saves a substantial amount of per-round
                # compute on heavy-user datasets like amazon-book.
                sampled = list(set(forced))
                delta = self.run_round(sampled, probe_assign=probe_assign,
                                       reps_per_pair=reps_per_pair, policy=policy)
                item_delta = delta["item_emb.weight"]
                n_sampled = max(len(sampled), 1)
                for k in range(K):
                    pos_id, neg_id = pair_ids[k]
                    diff = (item_delta[pos_id] - item_delta[neg_id]) * n_sampled
                    G[t * K + k] = diff.detach().cpu().numpy()
            finally:
                self.restore(snap)
        return G
