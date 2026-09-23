"""WS-2: client-owned pseudorandomness for negative sampling.

The server chooses the negative-sampling *distribution* (public, signed); the
client chooses the *outcome*, from a PRF keyed by a secret that never leaves the
device. Theorem 1 rests on this: for any item the client has never interacted
with, the server's posterior that the client touched that row in a given round
is within `eps_prf` of the prior it would have without seeing anything.

SHA-256 in counter mode. One PRF draw *replaces* one PRNG draw in the honest
sampler (`fl/data.py::bpr_sample`), so the added cost is a constant factor on a
term that is already dominated by the O(n_items * d) model download.
"""
from __future__ import annotations

import hashlib
import struct
from typing import List, Optional, Sequence

import numpy as np


class PactRng:
    """Deterministic client-side stream keyed by (sk_u, counter, round root)."""

    __slots__ = ("_key", "_ctr", "_buf", "_off")

    def __init__(self, sk: bytes, ctr: int, round_root: bytes = b""):
        self._key = hashlib.sha256(sk + struct.pack("<Q", int(ctr)) + round_root).digest()
        self._ctr = 0
        self._buf = b""
        self._off = 0

    def _refill(self, n: int) -> None:
        out = []
        have = len(self._buf) - self._off
        while have < n:
            out.append(hashlib.sha256(self._key + struct.pack("<Q", self._ctr)).digest())
            self._ctr += 1
            have += 32
        self._buf = self._buf[self._off:] + b"".join(out)
        self._off = 0

    def bytes(self, n: int) -> bytes:
        if len(self._buf) - self._off < n:
            self._refill(n)
        out = self._buf[self._off:self._off + n]
        self._off += n
        return out

    def uint64(self, size: int) -> np.ndarray:
        raw = self.bytes(8 * size)
        return np.frombuffer(raw, dtype="<u8", count=size)

    def uniform(self, size: int) -> np.ndarray:
        return (self.uint64(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        span = int(high) - int(low)
        if span <= 0:
            return np.full(size, int(low), dtype=np.int64)
        return (low + (self.uint64(size) % np.uint64(span))).astype(np.int64)


def sample_negatives(rng: PactRng, pos_set: set, n_items: int, n: int,
                     dist: Optional[np.ndarray] = None,
                     max_rounds: int = 32) -> np.ndarray:
    """Draw `n` negatives from the WHOLE committed catalog under `dist`.

    `dist` is the server's public, signed sampling distribution (None = uniform).
    Rejection is against the client's own positives only; the client never asks
    the server which items exist, and never accepts a server-supplied list.
    """
    out = np.empty(n, dtype=np.int64)
    k = 0
    cdf = None
    if dist is not None:
        cdf = np.cumsum(np.asarray(dist, dtype=np.float64))
        cdf /= cdf[-1]
    for _ in range(max_rounds):
        need = n - k
        if need <= 0:
            break
        draw = max(need * 2, 16)
        if cdf is None:
            cand = rng.integers(0, n_items, draw)
        else:
            cand = np.searchsorted(cdf, rng.uniform(draw)).astype(np.int64)
            np.clip(cand, 0, n_items - 1, out=cand)
        for c in cand:
            ci = int(c)
            if ci in pos_set:
                continue
            out[k] = ci
            k += 1
            if k == n:
                break
    while k < n:                       # degenerate fallback (tiny catalogs)
        out[k] = int(rng.integers(0, n_items, 1)[0])
        k += 1
    return out
