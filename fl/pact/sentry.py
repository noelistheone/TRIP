"""SENTRY — client-side one-class detector for inverse-probing servers.

Strictly OUTSIDE the security guarantee. WS/BC/TB already bound what the server
can learn; SENTRY's job is *attribution*: telling the client (and its user) that
the server is actively deviating, so the deviation becomes evidence rather than
an invisible protocol detail. It can only escalate to abort-and-report; it can
never weaken the other three rules. A broken detector therefore costs zero
security, and a maximally miscalibrated one costs at most `alpha` aborted rounds.

Every feature is computed from data the client has ALREADY downloaded, so
inference is O(n_items * d) worst case -- the same order as receiving the model.

The features target the attack's structural signatures, not a fingerprint of one
implementation:

  f0  min pairwise distance among newly appended item rows, relative to
      sigma_v * sqrt(2d).  Paired probes are constructed as p +/- delta with
      eps = 1e-3 * sigma_v, so they arrive as NEAR-DUPLICATE PAIRS. Honest
      cold-start items are drawn independently and are not.
  f1  catalog growth this round, as a fraction of the catalog.
  f2  max per-tensor norm ratio vs the previous round (gamma damping: 0.1).
  f3  min per-tensor norm ratio vs the previous round.
  f4  staleness: fraction of shared tensors bit-identical to last round. TRIP's
      snapshot/restore (fl/trip/server.py) reverts the global model every probe
      round, so the model does not move for T rounds while training "proceeds".
  f5  max |z| of appended-row norms against the catalog's norm distribution
      (saturating probes are placed at M * sigma_v, M = 8).
  f6  fraction of server-instructed positives that are NOT in the client's own
      history (probes-only instructions put this at 1.0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FEATURES = ["nn_dup", "growth", "norm_hi", "norm_lo", "staleness", "mag_z", "alien_pos"]


def extract(new_rows: Optional[np.ndarray],
            catalog: np.ndarray,
            tensor_norms: Dict[str, float],
            prev_tensor_norms: Optional[Dict[str, float]],
            n_identical: int = 0, n_tensors: int = 1,
            alien_pos_frac: float = 0.0) -> np.ndarray:
    d = catalog.shape[1] if catalog.ndim == 2 else 1
    sigma_v = float(catalog.std()) if catalog.size else 1.0
    scale = sigma_v * np.sqrt(2.0 * d) + 1e-12

    # f0: nearest-neighbour duplication among the newly appended rows
    nn = 1.0
    if new_rows is not None and len(new_rows) >= 2:
        X = np.asarray(new_rows, dtype=np.float64)
        g = X @ X.T
        sq = np.diag(g)
        D = np.maximum(sq[:, None] + sq[None, :] - 2 * g, 0.0)
        np.fill_diagonal(D, np.inf)
        nn = float(np.sqrt(D.min()) / scale)

    growth = float(len(new_rows) / max(len(catalog), 1)) if new_rows is not None else 0.0

    hi, lo = 1.0, 1.0
    if prev_tensor_norms:
        rs = [tensor_norms[k] / prev_tensor_norms[k]
              for k in tensor_norms if prev_tensor_norms.get(k, 0) > 0]
        if rs:
            hi, lo = float(max(rs)), float(min(rs))

    stale = float(n_identical / max(n_tensors, 1))

    magz = 0.0
    if new_rows is not None and len(new_rows):
        cn = np.linalg.norm(catalog, axis=1)
        mu, sd = float(cn.mean()), float(cn.std()) + 1e-12
        magz = float(np.abs((np.linalg.norm(np.asarray(new_rows), axis=1) - mu) / sd).max())

    return np.array([nn, growth, hi, lo, stale, magz, float(alien_pos_frac)],
                    dtype=np.float64)


@dataclass
class Sentry:
    """Robust one-class model + split-conformal threshold.

    Location/scale are median/MAD so a few poisoned calibration rounds cannot
    drag the model onto the attack. The score is a two-sided robust Mahalanobis
    distance in whitened feature space; the threshold is the empirical
    (1-alpha) quantile of calibration scores, which gives a finite-sample
    false-alarm guarantee of alpha under exchangeability without assuming any
    distribution.
    """
    alpha: float = 0.05
    med: Optional[np.ndarray] = None
    mad: Optional[np.ndarray] = None
    thresh: float = np.inf
    n_fit: int = 0
    calib: List[float] = field(default_factory=list, repr=False)

    def fit(self, benign: np.ndarray, calib: Optional[np.ndarray] = None) -> "Sentry":
        B = np.asarray(benign, dtype=np.float64)
        self.med = np.median(B, axis=0)
        self.mad = np.median(np.abs(B - self.med), axis=0) * 1.4826 + 1e-9
        C = B if calib is None else np.asarray(calib, dtype=np.float64)
        self.calib = sorted(float(s) for s in self._raw(C))
        k = int(np.ceil((len(self.calib) + 1) * (1 - self.alpha))) - 1
        self.thresh = self.calib[min(max(k, 0), len(self.calib) - 1)]
        self.n_fit = len(B)
        return self

    def _raw(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.abs((X - self.med) / self.mad).max(axis=1)

    def score(self, x: np.ndarray) -> float:
        return float(self._raw(x)[0])

    def pvalue(self, x: np.ndarray) -> float:
        s = self.score(x)
        n = len(self.calib)
        return (1 + sum(1 for c in self.calib if c >= s)) / (n + 1)

    def flag(self, x: np.ndarray) -> Tuple[bool, float]:
        p = self.pvalue(x)
        return p < self.alpha, p


def synth_attack_rounds(benign: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generative anomaly class: the structural signatures, not one codebase.

    No labelled attacks exist in the wild, so the detector is trained one-class
    on benign rounds and only EVALUATED against these. Each sample perturbs a
    benign round along one or more signature axes.
    """
    B = np.asarray(benign, dtype=np.float64)
    out = []
    for _ in range(n):
        x = B[rng.integers(0, len(B))].copy()
        which = rng.integers(0, 4)
        if which == 0:      # paired near-duplicate probe rows
            x[0] = rng.uniform(1e-4, 5e-3)
            x[1] = max(x[1], rng.uniform(1e-3, 2e-2))
        elif which == 1:    # gamma-damped broadcast
            x[3] = rng.uniform(0.05, 0.3)
        elif which == 2:    # frozen global model (snapshot/restore)
            x[4] = rng.uniform(0.8, 1.0)
        else:               # saturating probe magnitudes
            x[5] = rng.uniform(6.0, 40.0)
            x[6] = rng.uniform(0.8, 1.0)
        out.append(x)
    return np.asarray(out)
