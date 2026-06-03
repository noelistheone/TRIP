"""Abstract base class for embedding-inversion attacks on FedRec.

Every attack implements:
  - prepare(server, bundle, attacked_uids, cfg): capture whatever state the
    attack needs (probe injection, per-client gradient capture, etc.). Should
    record its own wall-clock time in self.timing['prepare_sec'].
  - solve(server, bundle, attacked_uids, cfg) → np.ndarray [N, d]: produce the
    recovered user-embedding matrix. Records self.timing['solve_sec'].

Runner calls `.prepare()` then `.solve()` and assembles the usual metrics
(cosine vs true_U, downstream ranking) in a model-agnostic way.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import torch


@contextmanager
def _timer(storage: Dict[str, float], key: str):
    t = time.time()
    yield
    storage[key] = time.time() - t


class AttackBase(ABC):
    name: str = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.timing: Dict[str, float] = {"prepare_sec": 0.0, "solve_sec": 0.0}
        # Attacks may store intermediate state (observations, probe pairs, etc.)
        self._state: Dict[str, object] = {}
        # Optional bool mask of length N. When set, the runner restricts
        # cosine + ranking metrics to indices where mask is True. Used by
        # LtI: half the attacked users serve as in-sample shadow training
        # labels and must be excluded from the eval set so we report only
        # out-of-sample (target-half) generalization, matching Wu et al.'s
        # original threat model.
        self.eval_mask: "np.ndarray | None" = None

    @abstractmethod
    def prepare(self, server, bundle, attacked_uids: List[int]) -> None:
        """Inject probes or capture per-client gradients.

        Must set self._state with whatever the solver needs. Must record
        self.timing['prepare_sec'].
        """

    @abstractmethod
    def solve(self, server, bundle, attacked_uids: List[int]) -> np.ndarray:
        """Return recovered U_hat of shape [N, d]. Records self.timing['solve_sec']."""

    # Helper
    def _time(self, key: str):
        return _timer(self.timing, key)
