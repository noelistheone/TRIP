#!/usr/bin/env python
"""TRIP noise robustness figure (Figure 1) for the paper.

Loads results/run8_tradeoff/<ds>_<mdl>_<tag>.json for each (dataset, MF, noise)
cell and plots a 1×3 grid (one panel per dataset) of TRIP recovery cosine
vs HE precision-loss noise sigma. A shaded band marks the realistic HE
deployment range (per-coord precision loss <=1e-10 across all standard
schemes); within this band TRIP is bit-exact stable, demonstrating that
realistic HE provides no defense against TRIP. Annotations mark the three
canonical HE schemes (CKKS-40, Paillier, CKKS-20).

Output: draft/fig_he_tradeoff.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["lastfm", "ml", "douban-book"]
DS_LABEL = {"lastfm": "LastFM", "ml": "MovieLens", "douban-book": "Douban-Book"}
MODEL = "mf"
NOISE_LEVELS = [
    ("0",     0.0),
    ("1e-12", 1.0e-12),
    ("1e-8",  1.0e-8),
    ("1e-6",  1.0e-6),
    ("1e-4",  1.0e-4),
    ("1e-2",  1.0e-2),
    ("1",     1.0),
]
# Realistic HE deployments produce per-coord precision-loss noise on the
# aggregate at or below this sigma. CKKS-40 ~1e-16, Paillier ~1e-13, CKKS-20
# ~1e-10. The shaded band ends at 1e-10 (the noisiest realistic scheme).
REALISTIC_HE_MAX = 1.0e-10
# Reference markers for the three canonical HE schemes.
HE_REFS = [
    (1.0e-16, "CKKS-40"),
    (1.0e-13, "Paillier"),
    (1.0e-10, "CKKS-20"),
]


def load_cell(ds: str, tag: str):
    p = Path(f"results/run8_tradeoff/{ds}_{MODEL}_{tag}.json")
    if not p.exists():
        return None
    return json.load(open(p))


def main():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)

    # X-axis: replace 0 with a small value for plotting on log scale.
    # Cap at 1e-18 so the shaded "realistic HE" band has visible left edge.
    x_disp = [n if n > 0 else 1e-18 for _, n in NOISE_LEVELS]

    for ax, ds in zip(axes, DATASETS):
        cos_y = []
        for tag, _ in NOISE_LEVELS:
            r = load_cell(ds, tag)
            cos_y.append(r["cos"]["cos_mean"] if r else np.nan)
        # Shaded "realistic HE" band: from extreme left up to REALISTIC_HE_MAX
        ax.axvspan(1e-19, REALISTIC_HE_MAX, color="#d4f0d4", alpha=0.7,
                   zorder=0, label="Realistic HE range")
        # Vertical reference lines for each canonical HE scheme
        for sig, name in HE_REFS:
            ax.axvline(sig, color="gray", linestyle=":", linewidth=0.8,
                       alpha=0.6, zorder=1)
        # TRIP cos curve
        ax.plot(x_disp, cos_y, "o-", color="black", linewidth=2.2,
                markersize=6, label="TRIP cos", zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(1e-18, 5)
        ax.set_xlabel(r"HE noise $\sigma$")
        ax.set_title(DS_LABEL[ds])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3, zorder=2)
        # Custom x ticks: show "0" label for the leftmost point
        ax.set_xticks(x_disp)
        ax.set_xticklabels(["0", r"$10^{-12}$", r"$10^{-8}$",
                            r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$",
                            r"$10^{0}$"],
                           rotation=30, ha="right", fontsize=9)
        # Annotate canonical schemes only on left panel to avoid clutter
        if ax is axes[0]:
            ax.text(1.0e-16, 0.04, "CKKS-40", rotation=90, fontsize=7.5,
                    color="gray", va="bottom", ha="right")
            ax.text(1.0e-13, 0.04, "Paillier", rotation=90, fontsize=7.5,
                    color="gray", va="bottom", ha="right")
            ax.text(1.0e-10, 0.04, "CKKS-20", rotation=90, fontsize=7.5,
                    color="gray", va="bottom", ha="right")

    axes[0].set_ylabel("TRIP cosine")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    out = Path("draft/fig_he_tradeoff.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")
    out_png = out.with_suffix(".png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
