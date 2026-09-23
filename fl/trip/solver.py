"""Closed-form analytical recovery solvers.

solve_mf_lgcn(G, A) → U_hat
    For MF and LightGCN, framework §4.4 + §5.3 establish that the aggregated
    paired-probe difference G[r=t·K+k] equals A[r] · U up to O(η²) error.
    Solve U_hat = pinv(A) · G in least-squares sense.

solve_ncf(G, A, w_gmf, model, lam) → U_hat
    For NCF, framework §6.3: under the saturating probe construction the MLP
    branch gradient is approximately constant across users, so G[r] ≈ A[r] ·
    diag(w_GMF) · U. Solve U_hat = diag(w_GMF)^-1 · pinv(A) · G with a small
    Tikhonov ridge for numerical stability.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..models import NCFModel


def _pinv_solve(A: np.ndarray, G: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Least-squares solve of A · U = G, i.e. U = (A^T A + λI)^-1 A^T G."""
    AtA = A.T @ A
    n = AtA.shape[0]
    AtA_reg = AtA + ridge * np.eye(n, dtype=A.dtype)
    AtG = A.T @ G
    U = np.linalg.solve(AtA_reg, AtG)
    return U


def solve_mf_lgcn(G: np.ndarray, A: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Returns U_hat ∈ ℝ^(N, d). Each row L2-normalized.

    `ridge` is the Tikhonov parameter of (AᵀA + ridge·I)⁻¹Aᵀ. It is NOT a mere
    numerical guard: once the aggregate carries noise, the optimal ridge grows
    with the noise level, and a window with a large ‖A⁺‖ (i.e. coprime to N)
    needs a larger ridge than a rank-deficient one. Leaving it fixed at 1e-6
    conflates "the full-rank design is worse under noise" with "the full-rank
    design was under-regularised".
    """
    if G.ndim == 3:
        # (T, K, d) → (T·K, d)
        T, K, d = G.shape
        G = G.reshape(T * K, d)
    U = _pinv_solve(A.astype(np.float64), G.astype(np.float64), ridge=ridge).astype(np.float32)
    norms = np.linalg.norm(U, axis=-1, keepdims=True) + 1e-12
    return U / norms


def solve_ncf(G: np.ndarray, A: np.ndarray, probe_base: torch.Tensor,
              model: NCFModel, lam: float = 1.0e-4) -> np.ndarray:
    """U_hat = diag(w_GMF)^-1 · pinv(A) · G. probe_base is w_GMF."""
    if G.ndim == 3:
        T, K, d = G.shape
        G = G.reshape(T * K, d)
    U_raw = _pinv_solve(A.astype(np.float64), G.astype(np.float64),
                        ridge=max(lam, 1e-6)).astype(np.float32)
    w_gmf = probe_base.detach().cpu().numpy().astype(np.float32)  # (d,)
    # Per-coord rescale, with ridge to avoid blow-up where w_gmf is small.
    inv_w = w_gmf / (w_gmf ** 2 + lam)
    U = U_raw * inv_w[None, :]
    norms = np.linalg.norm(U, axis=-1, keepdims=True) + 1e-12
    return U / norms
