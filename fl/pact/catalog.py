"""TB: transcript binding — the catalog, the recipe and the participant set are
committed, signed, and verified by every client before it trains.

What this stops, concretely, in this repository:
  * `fl/trip/probes.py::init_paired_probes` appends 2K rows to the catalog. Under
    TB the append is a Merkle extension the client verifies against the root it
    pinned last round, and the growth rate is capped, so probe rows are public,
    counted and rate-limited rather than silent.
  * `fl/trip/probes.py::damp_ncf_mlp(factor=0.1)` rescales the trained MLP body
    before broadcast. The per-tensor sanity band catches a 10x rescale with an
    enormous margin.
  * `fl/eval.py::_make_attack_cfg` ships a modified training recipe. The recipe
    digest is in the signed header and the client compares it to its pinned one.

HMAC-SHA256 stands in for Ed25519 here: the security argument needs an EUF-CMA
signature, and the cost argument needs one verification per round. Swapping in
Ed25519 changes ~50 us per round and no part of the analysis.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


def _h(*parts: bytes) -> bytes:
    d = hashlib.sha256()
    for p in parts:
        d.update(p)
    return d.digest()


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    if not leaves:
        return b"\x00" * 32
    level = [_h(b"\x00", x) for x in leaves]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else a
            nxt.append(_h(b"\x01", a, b))
        level = nxt
    return level[0]


def catalog_leaves(n_items: int) -> List[bytes]:
    return [struct.pack("<q", i) for i in range(int(n_items))]


def catalog_root(n_items: int) -> bytes:
    return merkle_root(catalog_leaves(n_items))


def verify_append(prev_n: int, new_n: int, prev_root: bytes, new_root: bytes,
                  growth_max: float = 0.02) -> Tuple[bool, str]:
    """Append-only check plus the growth cap that bounds injectable rows."""
    if new_n < prev_n:
        return False, "catalog shrank (not append-only)"
    if catalog_root(prev_n) != prev_root:
        return False, "pinned previous root mismatch"
    if catalog_root(new_n) != new_root:
        return False, "announced new root does not match announced size"
    if prev_n > 0 and (new_n - prev_n) > growth_max * prev_n:
        return False, (f"catalog grew by {new_n - prev_n} rows "
                       f"(> {growth_max:.1%} of {prev_n})")
    return True, ""


def recipe_digest(recipe: Dict) -> bytes:
    return _h(json.dumps(recipe, sort_keys=True, default=str).encode())


@dataclass
class RoundHeader:
    round_idx: int
    catalog_n: int
    catalog_root: bytes
    recipe_digest: bytes
    participants: Tuple[int, ...]
    sig: bytes = b""

    def payload(self) -> bytes:
        return _h(
            struct.pack("<qq", int(self.round_idx), int(self.catalog_n)),
            self.catalog_root, self.recipe_digest,
            b"".join(struct.pack("<q", int(u)) for u in self.participants),
        )

    def nbytes(self) -> int:
        return 8 + 8 + 32 + 32 + 8 * len(self.participants) + len(self.sig)


def sign_header(h: RoundHeader, key: bytes) -> RoundHeader:
    h.sig = hmac.new(key, h.payload(), hashlib.sha256).digest()
    return h


def verify_header(h: RoundHeader, key: bytes) -> bool:
    return hmac.compare_digest(
        h.sig, hmac.new(key, h.payload(), hashlib.sha256).digest())


def sanity_band(prev: Dict[str, float], cur: Dict[str, float],
                band: Tuple[float, float] = (0.5, 2.0)) -> Tuple[bool, str]:
    """Per-tensor norm-ratio band. `damp_ncf_mlp(0.1)` gives ratio 0.1."""
    lo, hi = band
    for k, v in cur.items():
        p = prev.get(k)
        if not p:
            continue
        r = v / p
        if r < lo or r > hi:
            return False, f"tensor {k}: norm ratio {r:.4g} outside [{lo}, {hi}]"
    return True, ""
