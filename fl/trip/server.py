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


def lgcn_layer_weights(hops: int) -> Tuple[float, float]:
    """On a star graph LightGCN's layers alternate u, S_u, u, ...; return the
    shares of u and of S_u in the mean over hops+1 layers ((2/3, 1/3) for 2 hops)."""
    return (hops // 2 + 1) / (hops + 1), ((hops + 1) // 2) / (hops + 1)


def lgcn_star_sum(lists: List[List[int]], device):
    """CSR of per-user item lists -> (f, deg) with f(u_loc, V) = M_u^{-1/2} sum_{j in I_u} V[j]
    for a batch of local user indices (differentiable in V); deg[u] = M_u."""
    deg_np = np.array([len(l) for l in lists], dtype=np.int64)
    deg = torch.from_numpy(deg_np).to(device)
    indptr = torch.zeros(len(lists) + 1, dtype=torch.long, device=device)
    indptr[1:] = torch.cumsum(deg, 0)
    indices = torch.tensor([i for l in lists for i in l], dtype=torch.long, device=device)
    inv_sqrt = torch.where(deg > 0, deg.clamp_min(1).double().rsqrt(),
                           torch.zeros_like(deg, dtype=torch.double)).float()

    def f(u_loc: torch.Tensor, V: torch.Tensor, total: Optional[int] = None) -> torch.Tensor:
        # `total` = sum of the batch users' degrees, computed on the host by the
        # caller; passing it as output_size keeps the kernels free of GPU syncs.
        lens = deg[u_loc]
        if total is None:
            total = int(lens.sum())
        rows = torch.repeat_interleave(torch.arange(len(u_loc), device=device), lens, output_size=total)
        starts = torch.repeat_interleave(indptr[u_loc], lens, output_size=total)
        offs = torch.arange(total, device=device) - torch.repeat_interleave(
            torch.cumsum(lens, 0) - lens, lens, output_size=total)
        S = torch.zeros(len(u_loc), V.shape[1], device=device, dtype=V.dtype)
        S = S.index_add(0, rows, V[indices[starts + offs]])
        return S * inv_sqrt[u_loc].unsqueeze(-1)
    f.deg_np = deg_np
    return f, deg


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
        uid_lut = np.full(max(int(u) for u in train_uids) + 1, -1, dtype=np.int64)
        uid_lut[np.asarray(train_uids, dtype=np.int64)] = np.arange(len(train_uids), dtype=np.int64)
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

        if isinstance(self.model, LightGCNModel):
            # per-batch neighbour sum S_u over each train user's real items
            lists = [[int(i) for i in self.bundle.train_pos.get(int(uid), []) if 0 <= int(i) < n_real_items]
                     for uid in train_uids]
            lgcn_nbr_sum, lgcn_deg = lgcn_star_sum(lists, device)
            lgcn_a, lgcn_b = lgcn_layer_weights(int(getattr(self.model, "hops", 2)))

        # bpr_neg_ratio: number of negative samples per positive (default 1).
        # Bumping to 5+ is useful on extremely sparse catalogs (e.g. delicious
        # 69k items × ~1.2 user-visits per item) where MF/NCF item embeddings
        # can't develop discriminative direction with 1:1 sampling alone.
        neg_ratio = int(self.cfg.get("bpr_neg_ratio", 1))
        batch_l2 = float(self.cfg.get("warmup_batch_l2", 0.0) or 0.0)   # v4 E2b, off by default

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
            # local user index of every triple, mapped once per epoch on the host
            u_all_np = uid_lut[all_t[:, 0]]
            U_all = torch.from_numpy(u_all_np).to(device)
            for start in range(0, T.shape[0], batch):
                mb = T[start:start + batch]
                u_local = U_all[start:start + batch]
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
                    # Same scoring as a client's LightGCNModel._propagate on its
                    # star graph: layers alternate u, S_u, u, ... with
                    # S_u = M_u^{-1/2} sum_{j in I_u} v_j, averaged over hops+1
                    # layers; a user without items keeps its raw u.
                    S = lgcn_nbr_sum(u_local, self.model.item_emb.weight,
                                     total=int(lgcn_nbr_sum.deg_np[u_all_np[start:start + batch]].sum()))
                    has = (lgcn_deg[u_local] > 0).unsqueeze(-1)
                    u_til = torch.where(has, lgcn_a * u_vec + lgcn_b * S, u_vec)
                    v_pos = self.model.item_emb(pos_ids)
                    v_neg = self.model.item_emb(neg_ids)
                    pos_s = (u_til * v_pos).sum(-1)
                    neg_s = (u_til * v_neg).sum(-1)
                else:  # MFModel
                    v_pos = self.model.item_emb(pos_ids)
                    v_neg = self.model.item_emb(neg_ids)
                    pos_s = (u_vec * v_pos).sum(-1)
                    neg_s = (u_vec * v_neg).sum(-1)
                loss = -F.logsigmoid(pos_s - neg_s).mean()
                if batch_l2 > 0:
                    # LightGCN/NGCF-style: L2 on the batch's own (layer-0) rows only,
                    # decay/2 * (|u|^2 + |v+|^2 + |v-|^2) / B -- untouched rows are not shrunk.
                    loss = loss + batch_l2 * 0.5 * (u_vec.pow(2).sum() + v_pos.pow(2).sum()
                                                    + v_neg.pow(2).sum()) / u_vec.shape[0]
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

        base_round = self.round
        # Realised per-round assignment (after any block expansion): the solver
        # needs it to build the operator with the true 1/m and batch-split weights.
        self.last_assignments: List[Dict[int, List[int]]] = []
        # PACT TB (v4, opt-in via pact.binding.enforce): clients compare each broadcast
        # with the last one they ACCEPTED -- per-tensor norm ratios inside the band and
        # append-only catalogue growth under the cap -- and abort the round otherwise.
        # An aborted round releases nothing: no assignment, zero measurement rows.
        tb = policy is not None and bool(getattr(policy, "tb_enforce", False))
        self.tb_aborted = 0
        if tb:
            from ..pact.catalog import sanity_band

            def _tb_norms() -> Dict[str, float]:
                sd = self.model.state_dict()
                return {k: float(sd[k].float().norm()) for k in self.model.shared_keys()}
            tb_ref_norms = _tb_norms()
            tb_ref_items = int(getattr(self.model, "n_items_original",
                                       self.model.item_emb.weight.shape[0]))
        for t in iterator:
            snap = self.snapshot()
            if policy is not None:
                # A server that wants clients to participate must not replay round
                # indices (clients refuse replays). Give every attack round a fresh one.
                self.round = base_round + t
            try:
                # NCF: damp MLP body inside this round only (suppresses MLP
                # gradient residual that contaminates GMF recovery). The
                # finally-block restore() reverts this for the next round.
                gamma = float((self.cfg.get("ncf") or {}).get("damp_factor", 0.1))
                if isinstance(self.model, NCFModel) and gamma != 1.0:
                    from .probes import damp_ncf_mlp
                    damp_ncf_mlp(self.model, factor=gamma)
                if tb:
                    cur_norms = _tb_norms()
                    cur_items = int(self.model.item_emb.weight.shape[0])
                    grow_ok = (cur_items >= tb_ref_items and
                               cur_items - tb_ref_items <= policy.catalog_growth_max * tb_ref_items)
                    band_ok, why = sanity_band(tb_ref_norms, cur_norms, policy.theta_ratio_band)
                    if not (grow_ok and band_ok):
                        self.tb_aborted += 1
                        policy.aborts.append(f"TB round {t}: " + (why or
                                             f"catalogue {tb_ref_items}->{cur_items} above growth cap"))
                        self.last_assignments.append({})
                        continue                      # finally-block restores the model
                    tb_ref_norms, tb_ref_items = cur_norms, cur_items
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
                self.last_assignments.append({lu: list(ks) for lu, ks in local_assign.items()
                                              if lu < len(attacked_uids)})
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
        if policy is not None:
            self.round = base_round + len(self.last_assignments)
        return G
