"""BC-1: sticky, beacon-derived blocks — the anonymity sets.

Clients are partitioned ONCE, by a public randomness beacon the server cannot
influence, into blocks of size `t`. The partition is sticky for the lifetime of
the embedding table (`E_max = 1`): Proposition 7 shows that two independent
partitions let the server identify an individual exactly, by intersecting the
two blocks a given per-row signal is consistent with.

Stratification. Theorem 3's per-user ceiling is `sqrt(alpha + (1-alpha) w_i)`
with `w_i = c_i^2 / sum_u c_u^2` — QUADRATIC in the per-user BPR scalar, so a
single very active member of a block is recovered almost perfectly regardless of
`t`. Grouping by activity bucket `b_u = min(ceil(log2|I_u|), B_max)` bounds the
within-block ratio and hence `w_max = rho_s^2 / (rho_s^2 + t - 1)`.
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


def activity_bucket(n_items: int, b_max: int = 8) -> int:
    if n_items <= 1:
        return 0
    return min(int(math.ceil(math.log2(n_items))), b_max)


def _beacon_order(uids: Sequence[int], beacon: bytes) -> List[int]:
    """Public, server-independent permutation: sort by H(beacon || uid)."""
    keyed = [(hashlib.sha256(beacon + struct.pack("<q", int(u))).digest(), int(u))
             for u in uids]
    keyed.sort()
    return [u for _, u in keyed]


def beacon_partition(uids: Sequence[int], t: int, beacon: bytes = b"pact-v1",
                     strata: Optional[Dict[int, int]] = None) -> Dict[int, int]:
    """Return uid -> block id, with EVERY block holding between t and 2t-1 clients.

    Blocks smaller than t would break the anonymity parameter (a singleton block
    that participates whole isolates one client). So a stratum with fewer than t
    members is carried into the next stratum, and the final short chunk of a
    stratum is merged into that stratum's previous block. With stratification the
    permutation is applied within each (merged) activity bucket, so blocks stay
    activity-homogeneous where the population allows it.
    """
    t = max(1, int(t))
    uids = [int(u) for u in uids]
    if len(uids) < t:
        return {u: 0 for u in uids}           # one block holding everyone
    if strata:
        groups: Dict[int, List[int]] = {}
        for u in uids:
            groups.setdefault(int(strata.get(u, 0)), []).append(u)
        ordered = [groups[k] for k in sorted(groups)]
    else:
        ordered = [list(uids)]
    # carry undersized strata forward so every processed group has >= t members
    merged: List[List[int]] = []
    carry: List[int] = []
    for grp in ordered:
        carry = carry + grp
        if len(carry) >= t:
            merged.append(carry); carry = []
    if carry:
        if merged:
            merged[-1] = merged[-1] + carry
        else:
            merged.append(carry)
    assign: Dict[int, int] = {}
    nxt = 0
    for grp in merged:
        order = _beacon_order(grp, beacon)
        n_full = len(order) // t              # >= 1 because len(order) >= t
        for bi in range(n_full):
            lo = bi * t
            hi = (bi + 1) * t if bi < n_full - 1 else len(order)   # last block absorbs the remainder
            for u in order[lo:hi]:
                assign[u] = nxt
            nxt += 1
    return assign


def pooling_matrix(uids: Sequence[int], assign: Dict[int, int]) -> np.ndarray:
    """Pi in {0,1}^(N x n_blocks) with Pi[i, b] = 1 iff uids[i] is in block b.

    The exposure operator the server can address factors through Pi, so
    rank(A Pi) <= n_blocks = ceil(N / t)  (Theorem 2).
    """
    blocks = sorted({assign[int(u)] for u in uids})
    idx = {b: j for j, b in enumerate(blocks)}
    P = np.zeros((len(uids), len(blocks)), dtype=np.float64)
    for i, u in enumerate(uids):
        P[i, idx[assign[int(u)]]] = 1.0
    return P


@dataclass
class StickyPartition:
    """uid -> block, plus the members of each block. Refresh is refused."""
    t: int
    beacon: bytes = b"pact-v1"
    assign: Dict[int, int] = field(default_factory=dict)
    members: Dict[int, List[int]] = field(default_factory=dict)
    epochs: int = 0
    e_max: int = 1

    @classmethod
    def build(cls, uids: Sequence[int], t: int, beacon: bytes = b"pact-v1",
              strata: Optional[Dict[int, int]] = None) -> "StickyPartition":
        a = beacon_partition(uids, t, beacon, strata)
        mem: Dict[int, List[int]] = {}
        for u, b in a.items():
            mem.setdefault(b, []).append(int(u))
        for b in mem:
            mem[b].sort()
        return cls(t=int(t), beacon=beacon, assign=a, members=mem, epochs=1)

    def refresh(self, *_a, **_k):
        raise RuntimeError(
            "PACT forbids re-partitioning within an embedding's lifetime "
            "(E_max=1). Two partitions let the server intersect the two "
            "candidate blocks of a per-row signal and identify the client "
            "exactly (Proposition 7)."
        )

    def block_of(self, uid: int) -> int:
        return self.assign[int(uid)]

    def close(self, uids: Sequence[int]) -> List[int]:
        """BC-2: expand a set to the union of the whole blocks that touch it."""
        bs = {self.assign[int(u)] for u in uids if int(u) in self.assign}
        out: List[int] = []
        for b in sorted(bs):
            out.extend(self.members[b])
        return out

    def is_block_closed(self, uids: Sequence[int]) -> bool:
        s = {int(u) for u in uids}
        bs = {self.assign[int(u)] for u in s if int(u) in self.assign}
        return all(set(self.members[b]) <= s for b in bs)
