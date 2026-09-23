"""WS-1/WS-3/WS-4: the client's pinned, non-negotiable training recipe.

CRITICAL SIMULATOR HYGIENE. This object must NOT live inside the cfg dict that
`fl/eval.py::_make_attack_cfg` deep-copies and rewrites, or the simulated
attacker would silently switch the defense off and every "PACT stops TRIP"
number would be vacuous. It is threaded as a separate argument from experiment
setup down to `fl.client.local_train`, and the attack path never receives a
mutable handle to it.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ClientPolicy:
    enabled: bool = True
    # --- WS ---
    write_set_sovereignty: bool = True   # discard server-supplied item lists
    pin_recipe: bool = True              # ignore server optimizer / epoch overrides
    nu: float = 1.0                      # negatives per positive
    optimizer: str = "adam"
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    local_epochs: int = 1
    local_batch: int = 256
    # --- BC ---
    block_closed: bool = True
    t: int = 16
    stratify: bool = True
    beacon: bytes = b"pact-v1"
    mask: bool = True
    # --- TB ---
    binding: bool = True
    theta_ratio_band: Tuple[float, float] = (0.5, 2.0)
    catalog_growth_max: float = 0.02
    # --- SENTRY ---
    sentry: bool = True
    sentry_alpha: float = 0.05
    sentry_mode: str = "report"          # "report" | "abort"
    # --- runtime ---
    secrets: Dict[int, bytes] = field(default_factory=dict, repr=False)
    counters: Dict[int, int] = field(default_factory=dict, repr=False)
    partition: object = None
    aborts: List[str] = field(default_factory=list, repr=False)
    alarms: List[str] = field(default_factory=list, repr=False)

    def secret(self, uid: int) -> bytes:
        s = self.secrets.get(int(uid))
        if s is None:
            # Stand-in for a device-held key. Derived from a process-local salt
            # so it is reproducible within a run but never server-chosen.
            s = hashlib.sha256(b"pact-client-sk" + struct.pack("<q", int(uid))).digest()
            self.secrets[int(uid)] = s
        return s

    def next_counter(self, uid: int, round_idx: int) -> Optional[int]:
        """WS-4: strictly increasing, client-owned. Refuses replay."""
        cur = self.counters.get(int(uid), -1)
        if round_idx <= cur:
            self.aborts.append(f"replay: uid={uid} round={round_idx} <= {cur}")
            return None
        self.counters[int(uid)] = int(round_idx)
        return int(round_idx)

    def recipe(self) -> Dict:
        return {"optimizer": self.optimizer, "lr": self.lr,
                "weight_decay": self.weight_decay,
                "local_epochs": self.local_epochs, "local_batch": self.local_batch,
                "nu": self.nu}


def policy_from_cfg(cfg: dict, warm_cfg: Optional[dict] = None) -> Optional[ClientPolicy]:
    """Build a policy from `cfg['pact']`. Returns None when PACT is disabled.

    The pinned recipe defaults to the HONEST warm-up recipe, which is the point:
    the client keeps training the way it always did, and the attack's
    `optimizer=sgd, weight_decay=0, attack_probes_only=True` overrides are
    discarded rather than negotiated.
    """
    p = (cfg or {}).get("pact") or {}
    if not p.get("enabled", False):
        return None
    ws = p.get("write_set", {}) or {}
    bl = p.get("block", {}) or {}
    bi = p.get("binding", {}) or {}
    se = p.get("sentry", {}) or {}
    w = warm_cfg or {}
    band = bi.get("theta_ratio_band", [0.5, 2.0])
    return ClientPolicy(
        enabled=True,
        write_set_sovereignty=bool(ws.get("sovereignty", True)),
        pin_recipe=bool(ws.get("pin_recipe", True)),
        nu=float(ws.get("nu", 1.0)),
        optimizer=str(w.get("optimizer", cfg.get("warmup_optimizer", "adam"))).lower(),
        lr=float(w.get("lr", cfg.get("warmup_lr", 1e-3))),
        # The pinned recipe is the honest FEDERATED recipe. Its weight decay is
        # set explicitly: coupled decay under a per-round fresh Adam state moves
        # every row and collapses the table, so federated rounds use 0.
        weight_decay=float(ws["weight_decay"]) if "weight_decay" in ws
                     else float(w.get("weight_decay", 0.0)),
        local_epochs=int(w.get("local_epochs", cfg.get("local_epochs", 1))),
        local_batch=int(cfg.get("local_batch", 256)),
        block_closed=bool(bl.get("block_closed", p.get("schedule", {}).get("block_closed", True))),
        t=int(bl.get("t", 16)),
        stratify=bool(bl.get("stratify", True)),
        beacon=str(bl.get("beacon_seed") or "pact-v1").encode(),
        mask=bool((p.get("mask", {}) or {}).get("enabled", True)),
        binding=bool(bi.get("enabled", True)),
        theta_ratio_band=(float(band[0]), float(band[1])),
        catalog_growth_max=float(bi.get("catalog_growth_max", 0.02)),
        sentry=bool(se.get("enabled", True)),
        sentry_alpha=float(se.get("alpha", 0.05)),
        sentry_mode=str(se.get("mode", "report")),
    )
