"""Sliding-window probe allocator.

Builds a deterministic schedule that assigns each of K probe slots to a window
of W contiguous attacked-user indices per round. Across T = ceil(N/K)·T_factor
rounds, every user is covered enough times that the resulting allocation
matrix A ∈ {0,1}^(T·K, N) has near-full column rank — the algebraic
precondition for differential isolation (TRIP framework Theorem 2).

Construction
------------
At round t, probe slot k targets users:
    [(t + k * stride + j) % N   for j in range(W)]
where stride = N // K is the spread between K simultaneous probe anchors per
round. Adjacent rounds differ by exactly 1 user per probe, enabling fine-
grained pairwise differential isolation.

Output
------
A : np.ndarray of shape (T·K, N), entries in {0, 1}, every row has sum W.
    Row r = t*K + k corresponds to round t, probe slot k.
T : derived as ceil(N / K) * T_factor.
"""
from __future__ import annotations

from typing import Dict, List

import math
import numpy as np


class SlidingWindowAllocator:
    def __init__(self, N: int, K: int, W: int, T_factor: int = 3):
        self.N = int(N)
        self.K = int(K)
        self.W = int(W)
        self.T_factor = int(T_factor)
        self.T = max(1, math.ceil(self.N / max(self.K, 1)) * self.T_factor)
        self.stride = max(1, self.N // max(self.K, 1))

    def window(self, t: int, k: int) -> List[int]:
        start = (t + k * self.stride) % self.N
        return [(start + j) % self.N for j in range(self.W)]

    def touched(self, t: int) -> List[int]:
        s = set()
        for k in range(self.K):
            s.update(self.window(t, k))
        return sorted(s)

    def assignments_for_round(self, t: int) -> Dict[int, List[int]]:
        """For round t, return dict: local_uid → list of probe slot indices k it was assigned."""
        assign: Dict[int, List[int]] = {}
        for k in range(self.K):
            for uid in self.window(t, k):
                assign.setdefault(uid, []).append(k)
        return assign

    def build_A(self) -> np.ndarray:
        A = np.zeros((self.T * self.K, self.N), dtype=np.float32)
        for t in range(self.T):
            for k in range(self.K):
                row = t * self.K + k
                for uid in self.window(t, k):
                    A[row, uid] = 1.0
        return A
