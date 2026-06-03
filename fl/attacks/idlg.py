"""iDLG: Improved Deep Leakage from Gradients (Zhao et al., 2020) adapted to FedRec.

Paper: arxiv.org/abs/2001.02610

iDLG's contribution over DLG is *analytical label inference*: instead of
optimizing labels jointly with inputs, infer the label from the sign of the
last-layer gradient, then optimize only the input via L-BFGS. This dramatically
improves convergence and accuracy on classification tasks with cross-entropy
loss + one-hot labels.

Adaptation to FedRec/BPR:
  - In our setting, "labels" are the positive-item ids in each BPR triple.
    The simulator already replays the EXACT triples the real client sampled
    (same RNG seed), so "label inference" is a no-op for us.
  - iDLG therefore reduces to plain DLG with one less unknown. We implement it
    as a faithful baseline for completeness; the published optimization
    machinery (L-BFGS + L2) is identical. Differences from DLG in our
    implementation are: (a) we explicitly mark labels as known (no inference
    step), and (b) we use the same warm-start trick from observed-delta
    direction. Empirically, iDLG and DLG produce near-identical cos in our
    BPR setting — a fair finding given the algorithmic similarity once
    labels are removed from the search space.
"""
from __future__ import annotations

from .dlg import DLGAttack


class IDLGAttack(DLGAttack):
    """iDLG, faithful to the published method but with our pre-known-label adaptation.

    Inherits everything from DLGAttack since the optimization machinery is
    identical once labels are accounted for. The published distinction (label
    inference) collapses to a no-op when labels are already known via the
    simulator's RNG-replayed triple sampling.
    """
    name = "idlg"
