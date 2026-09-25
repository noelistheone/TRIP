"""Top-level experiment runner.

Builds the (model, server, bundle) triple, runs FedAvg warmup, then runs one
or more attacks on the same post-warmup state. Reports cosine vs the true
user embedding plus ranking metrics (Recall/NDCG/Precision/HR @ 10/20/50)
computed by using the recovered U_hat as a query into the model's natural
scoring function.

Single-attack: run_experiment(...) → summary dict.
Multi-attack:  run_experiment_multi_attack(...) → dict[attack_name → summary],
               sharing one warmup across all listed attacks (compute ≈ same
               as a single-attack sweep + the per-user optimization overhead
               of DLG/IG/RAIFLE).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .data import DatasetBundle, load_dataset
from .models import FLModel, LightGCNModel, MFModel, NCFModel, make_model
from .trip.server import TRIPServer
from .utils import set_seed


# --------------------------------------------------------------------
# Cosine + ranking metrics
# --------------------------------------------------------------------
def _cos_per_row(U_hat: np.ndarray, U_true: np.ndarray) -> np.ndarray:
    a = U_hat / (np.linalg.norm(U_hat, axis=-1, keepdims=True) + 1e-12)
    b = U_true / (np.linalg.norm(U_true, axis=-1, keepdims=True) + 1e-12)
    return (a * b).sum(-1)


def _ranking_metrics(scores: np.ndarray, gt_items: List[int],
                     ks: List[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n_items = scores.shape[0]
    if not gt_items or n_items == 0:
        for k in ks:
            for m in ("recall", "ndcg", "precision", "hr"):
                out[f"{m}@{k}"] = 0.0
        return out
    gt_set = set(int(x) for x in gt_items)
    order = np.argsort(-scores)
    for k in ks:
        topk = order[:k]
        hits = [int(int(i) in gt_set) for i in topk]
        n_hit = sum(hits)
        out[f"recall@{k}"] = n_hit / max(len(gt_set), 1)
        out[f"precision@{k}"] = n_hit / max(k, 1)
        out[f"hr@{k}"] = 1.0 if n_hit > 0 else 0.0
        # NDCG
        dcg = sum(h / np.log2(rank + 2) for rank, h in enumerate(hits))
        ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(gt_set), k)))
        out[f"ndcg@{k}"] = float(dcg / ideal) if ideal > 0 else 0.0
    return out


def _score_user_against_items(model: FLModel, u_query: torch.Tensor,
                              n_items_original: int,
                              train_pos: List[int],
                              use_true_neighbors: bool = False) -> np.ndarray:
    """Score a candidate user vector against the original (non-probe) item catalog.
    Train items are masked to -inf. Returns ndarray of length n_items_original."""
    device = next(model.parameters()).device
    saved = model.get_user_row()
    try:
        model.set_user_row(u_query.to(device))
        if isinstance(model, LightGCNModel):
            # Sec. III-D defines the LightGCN recovery target as the PROPAGATED
            # representation u_tilde, which is what the probe rows expose. So the
            # recovered vector must be scored directly (V @ u_hat); re-propagating
            # it would both double-smooth it and, far worse, feed the scorer the
            # victim's true interaction list -- the private data the attack is
            # supposed to be inferring.
            #
            # Measured consequence of the old behaviour (LastFM/LightGCN): an
            # attack with cos = -0.0115 scored recall@10 = 0.1257 against a
            # true-u reference of 0.1296, i.e. the ranking metric was almost
            # independent of the recovered embedding. Set
            # `eval_lightgcn_use_true_neighbors: true` to reproduce it as an
            # explicit upper bound.
            if use_true_neighbors:
                model.set_neighbors(train_pos)
            else:
                model.set_neighbors([])
        ids = torch.arange(n_items_original, device=device)
        with torch.no_grad():
            scores = model.score_all(ids).detach().cpu().numpy()
    finally:
        model.set_user_row(saved)
    if train_pos:
        scores = scores.copy()
        for it in train_pos:
            if 0 <= int(it) < n_items_original:
                scores[int(it)] = -np.inf
    return scores


# --------------------------------------------------------------------
# Pipeline helpers
# --------------------------------------------------------------------
def _resolve_per_dataset(cfg: dict, key: str, dataset: str, default):
    if key in cfg and isinstance(cfg[key], dict) and dataset in cfg[key]:
        return cfg[key][dataset]
    return cfg.get(key, default)


def _make_warmup_cfg(cfg: dict, dataset: str, model_name: str) -> dict:
    out = copy.deepcopy(cfg)
    out["optimizer"] = (
        cfg.get("warmup_optimizer_per_model", {}).get(model_name)
        or cfg.get("warmup_optimizer", "adam")
    )
    out["lr"] = float(
        cfg.get("warmup_lr_per_model", {}).get(model_name)
        or cfg.get("warmup_lr", 1.0e-3)
    )
    out["weight_decay"] = float(
        cfg.get("warmup_weight_decay_per_model", {}).get(model_name, 0.0)
    )
    out["local_epochs"] = int(
        _resolve_per_dataset(cfg, "local_epochs_overrides", dataset, cfg.get("local_epochs", 1))
    )
    out["clients_per_round"] = int(
        _resolve_per_dataset(cfg, "warmup_clients_per_round", dataset, cfg.get("clients_per_round", 128))
    )
    return out


def _make_attack_cfg(cfg: dict, dataset: str, model_name: str) -> dict:
    out = copy.deepcopy(cfg)
    # The attack phase normally FORCES plain SGD with wd=0 and probes-only
    # mini-batches. Both are capabilities an honest-but-curious server does not
    # have, so they are overridable to let us measure the attack on a ladder of
    # progressively weaker adversaries (see the threat-model ablation).
    out["optimizer"] = str(cfg.get("attack_optimizer_override", "sgd")).lower()
    out["weight_decay"] = float(cfg.get("attack_weight_decay_override", 0.0))
    out["lr"] = float(cfg.get("attack_lr_override", cfg.get("lr", 0.005)))
    # ATTACK PHASE: local_epochs MUST be small (default 1) so the user
    # embedding stays close to u^(0) during probe injection. With dense
    # datasets like amazon-book (45 items/user), warmup uses local_epochs=10
    # for convergence, but inheriting that into the attack causes 450 SGD
    # steps per round → user drifts ~2× initial norm → probe-row signal
    # decoheres → cos collapses to ~0.3. Cap to 1 universally.
    # "honest" (ladder rungs without C3b): clients keep their own epoch count.
    le = cfg.get("attack_local_epochs", 1)
    out["local_epochs"] = int(
        _resolve_per_dataset(cfg, "local_epochs_overrides", dataset, cfg.get("local_epochs", 1))
        if le == "honest" else le)
    out["attack_probes_only"] = bool(cfg.get("attack_probes_only", True))
    out["clients_per_round"] = int(cfg.get("clients_per_round", 128))
    return out


def _select_attacked_uids(bundle: DatasetBundle, n: int, seed: int) -> List[int]:
    """First N train users (sorted by uid). Warmup still trains ALL users
    (centralized BPR over the full train_user_ids); the attack recovery is
    targeted at the FIRST N for reproducibility / tractable per-user iter.
    """
    train_uids = sorted(bundle.train_user_ids)
    if len(train_uids) <= n:
        return train_uids
    return train_uids[:n]


def _run_warmup(server: TRIPServer, n_rounds: int, attacked_uids: List[int],
                clients_per_round: int) -> None:
    if n_rounds <= 0:
        return
    server.warmup(n_rounds, attacked_uids, clients_per_round, progress=True)


def _build_attack(name: str, cfg: dict):
    if name == "paired_probe":
        from .attacks.paired_probe import PairedProbeAttack
        return PairedProbeAttack(cfg)
    if name == "single_probe":
        from .attacks.single_probe import SingleProbeAttack
        return SingleProbeAttack(cfg)
    if name == "dlg":
        from .attacks.dlg import DLGAttack
        return DLGAttack(cfg)
    if name == "idlg":
        from .attacks.idlg import IDLGAttack
        return IDLGAttack(cfg)
    if name == "invert_grad":
        from .attacks.invert_grad import InvertGradientsAttack
        return InvertGradientsAttack(cfg)
    if name == "lti":
        from .attacks.lti import LearningToInvertAttack
        return LearningToInvertAttack(cfg)
    if name == "raifle":
        from .attacks.raifle import RaifleAttack
        return RaifleAttack(cfg)
    raise ValueError(f"unknown attack: {name}")


# --------------------------------------------------------------------
# Warmup checkpoint cache
# --------------------------------------------------------------------
# Warmup is by far the most expensive phase and is completely independent of
# the attack-phase geometry (W, K, T_factor, ridge, probe construction) and of
# any defense configuration. Caching the post-warmup state keyed on the
# warmup-relevant configuration lets a whole sweep over attack/defense knobs
# reuse one training run. Set FL_WARMUP_CACHE="" to disable.
_WARMUP_CACHE_KEYS = (
    "d", "seed", "N_attack", "local_batch", "bpr_neg_ratio",
    "attacked_target_triples", "attacked_oversample_min",
    "attacked_oversample_max", "attacked_oversample",
)


def _warmup_cache_path(dataset: str, model_name: str, cfg: dict, warm_cfg: dict,
                       n_warmup: int, bundle: DatasetBundle) -> Optional[Path]:
    root = os.environ.get("FL_WARMUP_CACHE", str(Path(__file__).resolve().parents[1] / "warmup_cache"))
    if not root:
        return None
    key = {
        "dataset": dataset, "model": model_name, "n_warmup": n_warmup,
        "n_users": bundle.n_users, "n_items": bundle.n_items,
        "n_inter": sum(len(v) for v in bundle.train_pos.values()),
        "warm": {k: warm_cfg.get(k) for k in
                 ("optimizer", "lr", "weight_decay", "local_epochs", "clients_per_round")},
        "cfg": {k: cfg.get(k) for k in _WARMUP_CACHE_KEYS},
        # attack-phase-only knobs (e.g. ncf.damp_factor) must not change the warm-up key
        "model_cfg": {k: ({kk: vv for kk, vv in (cfg.get(k) or {}).items() if kk != "damp_factor"}
                          if isinstance(cfg.get(k), dict) else cfg.get(k)) for k in ("lightgcn", "ncf")},
    }
    if cfg.get("warmup_batch_l2"):
        # v4 E2b: per-batch L2 regulariser in the warm-up; only enters the key when set,
        # so every existing checkpoint keeps its key.
        key["batch_l2"] = float(cfg["warmup_batch_l2"])
    if model_name == "lightgcn":
        # v3: the centralized warm-up now propagates on each user's star graph
        # (earlier checkpoints scored raw u^T v); keep the two apart.
        key["lgcn_warmup"] = "star-propagate-v3"
    h = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return Path(root) / f"{dataset}_{model_name}_{h}.pt"


def _load_warmup(path: Optional[Path], model: FLModel, server: TRIPServer,
                 device: torch.device) -> Optional[float]:
    """Returns the cached warmup wall-clock on a hit, None on a miss."""
    if path is None or not path.exists():
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[warmup-cache] unreadable ({e}); retraining")
        return None
    model.load_state_dict({k: v.to(device) for k, v in blob["state"].items()})
    server.user_states = {int(u): v.to(device) for u, v in blob["users"].items()}
    server.round = int(blob["round"])
    print(f"[warmup-cache] HIT  {path.name}  (saved {blob.get('warm_time', 0):.0f}s)")
    return float(blob.get("warm_time", 0.0))


def _save_warmup(path: Optional[Path], model: FLModel, server: TRIPServer,
                 warm_time: float) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "users": {int(u): v.detach().cpu().clone() for u, v in server.user_states.items()},
        "round": server.round, "warm_time": warm_time,
    }, tmp)
    tmp.replace(path)
    print(f"[warmup-cache] SAVE {path.name}")


# --------------------------------------------------------------------
# Metric assembly
# --------------------------------------------------------------------
def _evaluate_recovery(model: FLModel, bundle: DatasetBundle,
                       attacked_uids: List[int],
                       U_hat: np.ndarray, U_true: np.ndarray,
                       eval_mask: Optional[np.ndarray],
                       use_true_neighbors: bool = False) -> Dict:
    N = U_hat.shape[0]
    cos = _cos_per_row(U_hat, U_true)
    if eval_mask is not None:
        cos_eval = cos[eval_mask]
    else:
        cos_eval = cos
    # Attack-free floor: what a server gets by predicting the POPULATION MEAN
    # embedding for every user, i.e. with no attack at all. This floor is large
    # and strongly dataset-dependent (0.24 on LastFM, 0.72 on MovieLens), so a
    # raw cosine is not interpretable without it. `gain_norm` rescales the
    # metric onto [0, 1] between "no attack" and "perfect recovery".
    U_bar = U_true.mean(axis=0, keepdims=True)
    cos_triv = _cos_per_row(np.repeat(U_bar, U_true.shape[0], axis=0), U_true)
    cos_triv_eval = cos_triv[eval_mask] if eval_mask is not None else cos_triv
    triv = float(np.mean(cos_triv_eval)) if cos_triv_eval.size else 0.0
    cm = float(np.mean(cos_eval)) if cos_eval.size else 0.0
    cos_summary = {
        "cos_mean": cm,
        "cos_median": float(np.median(cos_eval)) if cos_eval.size else 0.0,
        "cos_std": float(np.std(cos_eval)) if cos_eval.size else 0.0,
        "n_eval": int(cos_eval.size),
        "cos_per_user_mean": float(np.mean(cos)),
        "cos_trivial_floor": triv,
        "gain_norm": float((cm - triv) / (1.0 - triv)) if triv < 1.0 else 0.0,
    }

    ks = [10, 20, 50]
    n_items_orig = getattr(model, "n_items_original", model.n_items)
    rank_per_user: Dict[str, List[float]] = {f"{m}@{k}": [] for m in ("recall", "ndcg", "precision", "hr") for k in ks}

    rank_true_per_user: Dict[str, List[float]] = {f"{m}@{k}": [] for m in ("recall", "ndcg", "precision", "hr") for k in ks}
    eval_idx = np.where(eval_mask)[0] if eval_mask is not None else np.arange(N)
    for i in eval_idx:
        uid = attacked_uids[int(i)]
        gt = bundle.test_pos.get(uid, [])
        if not gt:
            continue
        u_query = torch.tensor(U_hat[int(i)], dtype=torch.float32)
        scores = _score_user_against_items(
            model, u_query, n_items_orig, bundle.train_pos.get(uid, []),
            use_true_neighbors=use_true_neighbors,
        )
        m = _ranking_metrics(scores, gt, ks)
        for key, v in m.items():
            rank_per_user[key].append(float(v))
        # Reference: ranking with true u (upper bound for this warmup quality)
        # Reference = the recommendations the USER would generate for themselves,
        # which legitimately use their own interaction graph.
        u_true_q = torch.tensor(U_true[int(i)], dtype=torch.float32)
        scores_t = _score_user_against_items(
            model, u_true_q, n_items_orig, bundle.train_pos.get(uid, []),
            use_true_neighbors=True,
        )
        m_t = _ranking_metrics(scores_t, gt, ks)
        for key, v in m_t.items():
            rank_true_per_user[key].append(float(v))
    rank_summary = {k: (float(np.mean(v)) if v else 0.0) for k, v in rank_per_user.items()}
    rank_true_summary = {k: (float(np.mean(v)) if v else 0.0) for k, v in rank_true_per_user.items()}
    return {"cos": cos_summary, "ranking": rank_summary, "ranking_true_U": rank_true_summary}


# --------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------
def run_experiment(dataset: str, model_name: str, cfg: dict,
                   device: torch.device, out_dir: Path) -> Dict:
    """Single attack (paired_probe). For multi-attack runs use run_experiment_multi_attack."""
    return run_experiment_multi_attack(
        dataset, model_name, cfg, device, out_dir, attacks=["paired_probe"],
    )["paired_probe"]


def run_experiment_multi_attack(dataset: str, model_name: str, cfg: dict,
                                device: torch.device, out_dir: Path,
                                attacks: List[str]) -> Dict[str, Dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg.get("seed", 42)))
    bundle = load_dataset(dataset)
    print(f"[{dataset}/{model_name}] users={bundle.n_users} items={bundle.n_items} "
          f"train_users={len(bundle.train_user_ids)}")

    d = int(cfg["d"])
    model = make_model(model_name, bundle.n_users, bundle.n_items, d, cfg).to(device)
    server = TRIPServer(model, bundle, cfg, device)

    attacked_uids = _select_attacked_uids(bundle, int(cfg["N_attack"]), int(cfg.get("seed", 42)))
    n_warmup = int(_resolve_per_dataset(cfg, "warmup_overrides", dataset, cfg.get("warmup", 200)))
    warm_cfg = _make_warmup_cfg(cfg, dataset, model_name)
    server.cfg = warm_cfg
    print(f"[{dataset}/{model_name}] warmup {n_warmup} rounds, opt={warm_cfg['optimizer']}, lr={warm_cfg['lr']}")
    cache_path = _warmup_cache_path(dataset, model_name, cfg, warm_cfg, n_warmup, bundle)
    t_warm = time.time()
    _cached = _load_warmup(cache_path, model, server, device)
    if _cached is not None:
        warm_time = _cached
        cached_warmup = True
    else:
        _run_warmup(server, n_warmup, attacked_uids, int(warm_cfg["clients_per_round"]))
        warm_time = time.time() - t_warm
        cached_warmup = False
        _save_warmup(cache_path, model, server, warm_time)
    print(f"[{dataset}/{model_name}] warmup done in {warm_time:.1f}s"
          f"{' (cached)' if cached_warmup else ''}")

    # Snapshot post-warmup model state + user states (so we can rebuild a
    # fresh model per attack — paired_probe extends the catalog, raifle
    # mutates popular rows; we need a clean restore between attacks).
    post_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    post_users = {uid: u.detach().cpu().clone() for uid, u in server.user_states.items()}
    post_round = server.round

    # True U for cosine eval (raw e_u^(0) at start of attack phase).
    U_true = np.stack([
        post_users[uid].numpy() for uid in attacked_uids
    ]).astype(np.float32)

    # ---- PACT ----------------------------------------------------------
    # Built from the ORIGINAL cfg and the HONEST warm-up recipe, and threaded to
    # the client as a separate argument. It deliberately does NOT live in the
    # dict `_make_attack_cfg` deep-copies: otherwise the simulated attacker
    # would rewrite the defense's own optimizer / probes-only settings and every
    # "PACT stops TRIP" number would be vacuous.
    from .pact import policy_from_cfg
    from .pact.blocks import StickyPartition, activity_bucket
    policy = policy_from_cfg(cfg, warm_cfg)
    if policy is not None:
        strata = None
        if policy.stratify:
            strata = {i: activity_bucket(len(bundle.train_pos.get(u, [])))
                      for i, u in enumerate(attacked_uids)}
        policy.partition = StickyPartition.build(
            list(range(len(attacked_uids))), policy.t, policy.beacon, strata)
        print(f"[pact] t={policy.t} blocks={len(policy.partition.members)} "
              f"(ceil(N/t)={-(-len(attacked_uids)//policy.t)}) "
              f"ws={policy.write_set_sovereignty} bc={policy.block_closed} "
              f"recipe={policy.optimizer}/wd={policy.weight_decay}")

    summaries: Dict[str, Dict] = {}
    attack_cfg_base = _make_attack_cfg(cfg, dataset, model_name)
    for attack_name in attacks:
        print(f"\n[{dataset}/{model_name}/{attack_name}] starting")
        # Fresh model + server; attack-phase cfg.
        a_model = make_model(model_name, bundle.n_users, bundle.n_items, d, cfg).to(device)
        a_model.load_state_dict({k: v.to(device) for k, v in post_state.items()})
        a_model.n_items_original = bundle.n_items
        a_server = TRIPServer(a_model, bundle, attack_cfg_base, device)
        a_server.user_states = {uid: u.to(device).clone() for uid, u in post_users.items()}
        a_server.round = post_round

        attack = _build_attack(attack_name, attack_cfg_base)
        attack.policy = policy
        # v4 budget sweeps: per-client baselines may be run on the first n targets only
        # (same warm-up, same ground truth, so they pair with the full runs target by target).
        a_uids, a_Utrue = attacked_uids, U_true
        n_sub = int(cfg.get("baseline_eval_n", 0) or 0)
        if n_sub and attack_name not in ("paired_probe", "single_probe"):
            a_uids, a_Utrue = attacked_uids[:n_sub], U_true[:n_sub]
        try:
            attack.prepare(a_server, bundle, a_uids)
            U_hat = attack.solve(a_server, bundle, a_uids)
        except Exception as e:
            import traceback
            traceback.print_exc()
            (out_dir / f"{dataset}_{model_name}_{attack_name}.ERROR.txt").write_text(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
            )
            continue

        # Metrics
        eval_mask = getattr(attack, "eval_mask", None)
        metrics = _evaluate_recovery(
            a_model, bundle, a_uids, U_hat, a_Utrue, eval_mask,
            use_true_neighbors=bool(cfg.get('eval_lightgcn_use_true_neighbors', False)),
        )
        summary = {
            "dataset": dataset,
            "model": model_name,
            "attack": attack_name,
            "n_attacked": int(len(a_uids)),
            "n_items": int(bundle.n_items),
            "n_users": int(bundle.n_users),
            "warmup_rounds": int(n_warmup),
            "warmup_time_sec": float(warm_time),
            **metrics,
            "timing": dict(attack.timing),
        }
        if policy is not None and getattr(policy, "tb_enforce", False):
            st_ = getattr(attack, "_state", {}) or {}
            summary["tb_enforced"] = True
            summary["tb_aborted_rounds"] = st_.get("tb_aborted")
            summary["tb_rounds"] = st_.get("tb_rounds")
        summaries[attack_name] = summary

        # Persist artifacts
        json_path = out_dir / f"{dataset}_{model_name}_{attack_name}.json"
        json_path.write_text(json.dumps(summary, indent=2))
        npz_path = out_dir / f"{dataset}_{model_name}_{attack_name}.npz"
        # Persist the linear system (G, A) too: the README documents it, and it
        # lets any solver/ridge/rank experiment be replayed offline in seconds
        # instead of re-running warmup + attack on a GPU.
        extra = {}
        _st = getattr(attack, "_state", {})
        if isinstance(_st.get("G"), np.ndarray):
            extra["G"] = _st["G"].astype(np.float32)
        if isinstance(_st.get("A"), np.ndarray):
            extra["A"] = _st["A"].astype(np.float32)   # weights 1/m can be fractional
        np.savez_compressed(
            npz_path,
            U_hat=U_hat, true_U=a_Utrue,
            cos=_cos_per_row(U_hat, a_Utrue),
            attacked_uids=np.asarray(a_uids, dtype=np.int64),
            eval_mask=(eval_mask if eval_mask is not None else np.ones(len(a_uids), dtype=bool)),
            **extra,
        )
        print(f"[{dataset}/{model_name}/{attack_name}] cos_mean={summary['cos']['cos_mean']:.4f} "
              f"recall@10={summary['ranking']['recall@10']:.4f} "
              f"prep={summary['timing'].get('prepare_sec', 0):.1f}s "
              f"solve={summary['timing'].get('solve_sec', 0):.1f}s")

    return summaries


# --------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------
ATTACK_DISPLAY_ORDER = ["paired_probe", "dlg", "idlg", "invert_grad", "lti", "raifle", "single_probe"]
ATTACK_LABEL = {
    "paired_probe": "TRIP (ours)",
    "dlg": "DLG",
    "idlg": "iDLG",
    "invert_grad": "InvGrad",
    "lti": "LtI",
    "raifle": "RAIFLE",
    "single_probe": "single",
}


def render_attack_comparison(runs: List[Dict],
                             exclude_attacks: List[str] | None = None,
                             cos_only: bool = False) -> str:
    """Build the cross-attack aggregate markdown over a flat list of per-attack JSONs.

    exclude_attacks: list of attack names to drop (e.g. ["dlg", "idlg"] when
                     the user has decided those baselines aren't appropriate
                     for the writeup).
    cos_only: if True, only emit the cosine + timing tables (skip ranking).
    """
    exclude_attacks = exclude_attacks or []
    by_pair: Dict[tuple, Dict[str, Dict]] = {}
    for r in runs:
        ds = r.get("dataset", "?")
        mdl = r.get("model", "?")
        atk = r.get("attack", "paired_probe")
        if atk in exclude_attacks:
            continue
        by_pair.setdefault((ds, mdl), {})[atk] = r

    pairs = sorted(by_pair.keys())
    attacks_seen = sorted({a for d in by_pair.values() for a in d.keys()},
                          key=lambda a: ATTACK_DISPLAY_ORDER.index(a) if a in ATTACK_DISPLAY_ORDER else 99)

    lines: List[str] = []
    lines.append("# Attack comparison — recovery quality + downstream ranking + wall-clock")
    lines.append("")
    lines.append("All numbers from the same warmup-frozen checkpoint per (dataset, model). 4 datasets × 3 models. ")
    lines.append("TRIP (paired_probe) is the *only* attack that succeeds under HE-aggregated FedAvg; ")
    lines.append("all other attacks (DLG / iDLG / Inverting Gradients / LtI / RAIFLE) require per-client ")
    lines.append("delta access in the clear (i.e. *without* HE), so the high-cos numbers they report here ")
    lines.append("are the *no-HE upper bound* — under HE they collapse to ≈ random (cos→0).")
    lines.append("")

    # ---------- Table 1: Cosine ----------
    lines.append("## Table 1 — Cosine similarity (recovered Ũ vs true U)")
    lines.append("")
    head = "| dataset | model | " + " | ".join(ATTACK_LABEL.get(a, a) for a in attacks_seen) + " |"
    sep = "|---|---|" + "|".join([":---:"] * len(attacks_seen)) + "|"
    lines.append(head)
    lines.append(sep)
    for (ds, mdl) in pairs:
        cells = []
        for a in attacks_seen:
            r = by_pair[(ds, mdl)].get(a)
            if r is None:
                cells.append("—")
            else:
                cm = r.get("cos", {}).get("cos_mean")
                cells.append(f"{cm:.4f}" if cm is not None else "—")
        lines.append(f"| {ds} | {mdl} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---------- Table 2: Ranking metrics @10/20/50 ----------
    if cos_only:
        # Skip ranking tables; jump to wall-clock.
        pass
    else:
      metrics = ["recall", "ndcg", "precision", "hr"]
      for k in (10, 20, 50):
        for m in metrics:
            key = f"{m}@{k}"
            lines.append(f"## Table — {m.upper()}@{k} (recovered Ũ as query)")
            lines.append("")
            lines.append(head)
            lines.append(sep)
            for (ds, mdl) in pairs:
                cells = []
                for a in attacks_seen:
                    r = by_pair[(ds, mdl)].get(a)
                    if r is None:
                        cells.append("—")
                    else:
                        v = r.get("ranking", {}).get(key)
                        cells.append(f"{v:.4f}" if v is not None else "—")
                lines.append(f"| {ds} | {mdl} | " + " | ".join(cells) + " |")
            lines.append("")
    if cos_only:
        pass

    # ---------- Table 3: Wall-clock ----------
    lines.append("## Table — Wall-clock attack time (seconds)")
    lines.append("")
    lines.append("`prepare` = probe injection / per-client capture. `solve` = pinv or per-user iterative optimization.")
    lines.append("")
    head_t = "| dataset | model |" + "|".join(
        f" {ATTACK_LABEL.get(a, a)} prep | {ATTACK_LABEL.get(a, a)} solve "
        for a in attacks_seen) + "|"
    sep_t = "|---|---|" + "|".join(["---:"] * (2 * len(attacks_seen))) + "|"
    lines.append(head_t)
    lines.append(sep_t)
    for (ds, mdl) in pairs:
        cells = []
        for a in attacks_seen:
            r = by_pair[(ds, mdl)].get(a)
            if r is None:
                cells.extend(["—", "—"])
            else:
                t = r.get("timing", {})
                cells.append(f"{float(t.get('prepare_sec', 0)):.1f}")
                cells.append(f"{float(t.get('solve_sec', 0)):.2f}")
        lines.append(f"| {ds} | {mdl} | " + " | ".join(cells) + " |")
    lines.append("")

    if cos_only:
        return "\n".join(lines) + "\n"

    # ---------- Reference — true_U ranking (warmup quality ceiling) ----------
    lines.append("## Reference — Ranking with true U (warmup quality ceiling)")
    lines.append("")
    lines.append("These numbers show what the trained recommender achieves when given the *true* user embedding (no recovery loss). ")
    lines.append("For TRIP-recovered users to be 'as good as ground truth', ranking with Ũ should match these.")
    lines.append("")
    lines.append("| dataset | model | recall@10 | ndcg@10 | hr@10 | recall@20 | hr@20 | recall@50 | hr@50 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for (ds, mdl) in pairs:
        # Pick any attack's ranking_true_U (they all share warmup; same true_U)
        r0 = next(iter(by_pair[(ds, mdl)].values()), {})
        rk = r0.get("ranking_true_U", {})
        if not rk:
            continue
        cells = [f"{rk.get(k, 0.0):.4f}" for k in
                 ("recall@10", "ndcg@10", "hr@10", "recall@20", "hr@20", "recall@50", "hr@50")]
        lines.append(f"| {ds} | {mdl} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_single(summary: Dict) -> str:
    lines: List[str] = []
    h = f"# {summary['dataset']} / {summary['model']} / {summary.get('attack','paired_probe')}"
    lines.append(h)
    lines.append("")
    lines.append(f"- attacked users: {summary['n_attacked']}")
    lines.append(f"- items: {summary['n_items']}, users: {summary['n_users']}")
    lines.append(f"- warmup: {summary['warmup_rounds']} rounds ({summary['warmup_time_sec']:.1f}s)")
    lines.append("")
    cs = summary["cos"]
    lines.append("## Cosine vs true u")
    lines.append(f"- cos_mean   : {cs['cos_mean']:.4f}")
    lines.append(f"- cos_median : {cs['cos_median']:.4f}")
    lines.append(f"- cos_std    : {cs['cos_std']:.4f}")
    lines.append(f"- n_eval     : {cs['n_eval']}")
    if "cos_trivial_floor" in cs:
        lines.append(f"- trivial floor (mean-embedding, no attack): {cs['cos_trivial_floor']:.4f}")
        lines.append(f"- gain_norm  : {cs['gain_norm']:.4f}   "
                     f"[(cos - floor) / (1 - floor)]")
    lines.append("")
    lines.append("## Ranking metrics (over recovered embedding queries)")
    rk = summary["ranking"]
    for k in (10, 20, 50):
        lines.append(f"- @{k:>3}: "
                     f"recall={rk[f'recall@{k}']:.4f}  "
                     f"ndcg={rk[f'ndcg@{k}']:.4f}  "
                     f"precision={rk[f'precision@{k}']:.4f}  "
                     f"hr={rk[f'hr@{k}']:.4f}")
    lines.append("")
    lines.append("## Timing")
    t = summary["timing"]
    for k, v in t.items():
        lines.append(f"- {k}: {float(v):.3f}")
    return "\n".join(lines) + "\n"
