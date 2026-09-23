"""BC-3: pairwise masks that cancel exactly within a block.

Bonawitz-style: client u adds  sum_{v != u} sign(u,v) * PRG(s_uv || round)  to
its uploaded item-table delta. The sum over a FULL block is identically zero, so
the decrypted block sum is bit-identical to the honest one (Theorem 4) — while
any sub-block decryption is uniform garbage over the ring.

There is deliberately NO dropout-recovery threshold below `t`: threshold
recovery reveals the survivor set, and differencing survivor sets across rounds
re-opens exactly the isolation channel PACT closes. Blocks that are not fully
live are skipped and the server over-provisions blocks instead (BC-4).

Fixed point over int64 so cancellation is exact; float would reintroduce the
very rounding the HE-precision analysis is about.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Dict, List, Sequence, Tuple

import numpy as np

RING_BITS = 62
RING = np.int64(1) << np.int64(RING_BITS)


def _pair_seed(a: int, b: int, round_idx: int, bind: bytes) -> bytes:
    lo, hi = (a, b) if a < b else (b, a)
    return hashlib.sha256(
        b"pact-mask" + struct.pack("<qqq", int(lo), int(hi), int(round_idx)) + bind
    ).digest()


def _prg(seed: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    n = int(np.prod(shape))
    need = 8 * n
    buf, ctr = [], 0
    have = 0
    while have < need:
        buf.append(hashlib.sha256(seed + struct.pack("<Q", ctr)).digest())
        ctr += 1
        have += 32
    raw = b"".join(buf)[:need]
    v = np.frombuffer(raw, dtype="<u8", count=n).astype(np.int64)
    return (v % RING).reshape(shape)


def mask_for(uid: int, block: Sequence[int], round_idx: int,
             shape: Tuple[int, ...], bind: bytes = b"") -> np.ndarray:
    """The mask client `uid` adds. Sum over a full block is exactly zero."""
    m = np.zeros(shape, dtype=np.int64)
    for v in block:
        v = int(v)
        if v == int(uid):
            continue
        s = _prg(_pair_seed(int(uid), v, round_idx, bind), shape)
        if int(uid) < v:
            m += s
        else:
            m -= s
    return m


def block_masks(block: Sequence[int], round_idx: int, shape: Tuple[int, ...],
                bind: bytes = b"") -> Dict[int, np.ndarray]:
    return {int(u): mask_for(int(u), block, round_idx, shape, bind) for u in block}


def assert_cancels(block: Sequence[int], round_idx: int, shape: Tuple[int, ...],
                   bind: bytes = b"") -> bool:
    ms = block_masks(block, round_idx, shape, bind)
    tot = np.zeros(shape, dtype=np.int64)
    for m in ms.values():
        tot += m
    return bool(np.all(tot == 0))
