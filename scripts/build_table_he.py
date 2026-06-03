#!/usr/bin/env python
"""Build the HE-Immunity table* (Table 3) for the paper.

Two stacked blocks per (dataset × model) grid (10 ds × 3 mdl = 30 cols total,
rendered as datasets-as-columns × (model × attack/scheme)-as-rows):

  Block A - TRIP under three HE schemes vs Plaintext
    rows: per-model × {Plaintext, HE-Paillier (1e-13), HE-CKKS-20 (1e-10)}
    (CKKS-40 dropped: bit-exact identical to Paillier within float noise)

  Block B - Baselines collapse under HE-aggregate-only
    rows: per-model × {InvGrad,LtI,RAIFLE} × {Plaintext, HE-aggregated}
    HE column uses CKKS-40 (canonical realistic scheme; all three schemes
    give the same collapse since aggregate-only is the binding HE defense
    for these baselines).

Plaintext numbers come from results/run6/ (with run4_ncf_fix / run4_delicious_fix
/ run6_amazon_*_fix overrides matching build_table_10ds.py priority).
HE numbers come from results/run7_he/.
"""
from __future__ import annotations

import json
from pathlib import Path

DATASETS = [
    "lastfm", "ml", "delicious", "douban-book",
    "amazon-beauty", "amazon-book", "amazon-kindle",
    "gowalla", "iFashion", "yelp2018",
]
DS_SHORT = {
    "lastfm": "LastFM",
    "ml": "MovieLens",
    "delicious": "Delicious",
    "douban-book": "Douban-Book",
    "amazon-beauty": "Amazon-Beauty",
    "amazon-book": "Amazon-Book",
    "amazon-kindle": "Amazon-Kindle",
    "gowalla": "Gowalla",
    "iFashion": "iFashion",
    "yelp2018": "Yelp2018",
}
MODELS = ["mf", "lightgcn", "ncf"]
MODEL_LABEL = {"mf": "MF", "lightgcn": "LightGCN", "ncf": "NCF"}
BASELINES = ["invert_grad", "lti", "raifle"]
BASELINE_LABEL = {
    "invert_grad": r"InvGrad",
    "lti": r"LtI",
    "raifle": r"RAIFLE",
}


def load_plaintext():
    """Load plaintext results from run6 with the same override priority used
    by build_table_10ds.py."""
    runs = {}
    for d in [
        "results/run4", "results/run6", "results/run4_ncf_fix",
        "results/run4_delicious_fix",
        "results/run6_amazon_beauty_timing",
        "results/run6_amazon_book_fix",
    ]:
        p_dir = Path(d)
        if not p_dir.exists():
            continue
        for p in sorted(p_dir.glob("*.json")):
            try:
                r = json.load(open(p))
            except Exception:
                continue
            key = (r["dataset"], r["model"], r.get("attack", "paired_probe"))
            runs[key] = r
    return runs


def load_he():
    """Load HE-aggregate-only results from run7_he. Files named
    <ds>_<mdl>_<attack>_<scheme>.json."""
    runs = {}
    p_dir = Path("results/run7_he")
    if not p_dir.exists():
        return runs
    for p in sorted(p_dir.glob("*.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        key = (r["dataset"], r["model"], r["attack"], r["scheme"])
        runs[key] = r
    return runs


def fmt_cos(v, bold=False):
    if v is None:
        return "--"
    return f"{v:.3f}"


def get_cos(run):
    if run is None:
        return None
    return run.get("cos", {}).get("cos_mean")


def build_table(pt, he):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{HE immunity. \textbf{Top:} TRIP cosine under Paillier "
        r"($\sigma{=}10^{-13}$), CKKS-40 ($\sigma{=}10^{-16}$), and CKKS-20 "
        r"($\sigma{=}10^{-10}$). \textbf{Bottom:} baseline cosine under "
        r"plaintext FedAvg vs.\ HE-CKKS-40 aggregated FedAvg "
        r"(other HE schemes give the same collapse since baselines are blocked "
        r"by aggregation regardless of noise level).}"
    )
    lines.append(r"\label{tab:he_immunity}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\footnotesize")
    col_spec = "l l " + " ".join(["c"] * len(DATASETS))
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    h = r"\textbf{Model} & \textbf{Condition}"
    for ds in DATASETS:
        h += " & " + DS_SHORT[ds]
    h += r" \\"
    lines.append(h)
    n_cols = 2 + len(DATASETS)

    # ---------- Block A ----------
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{" + str(n_cols)
        + r"}{l}{\emph{TRIP cosine across HE schemes} $\uparrow$} \\"
    )
    lines.append(r"\midrule")
    trip_conditions = [
        ("Plaintext",   "paired_probe", None),
        ("HE-Paillier", "paired_probe", "paillier"),
        ("HE-CKKS-40",  "paired_probe", "ckks40"),
        ("HE-CKKS-20",  "paired_probe", "ckks20"),
    ]
    for mdl_idx, mdl in enumerate(MODELS):
        for ci, (cond_label, attack, scheme) in enumerate(trip_conditions):
            cells = []
            if ci == 0:
                cells.append(r"\multirow{4}{*}{" + MODEL_LABEL[mdl] + "}")
            else:
                cells.append("")
            cells.append(cond_label)
            for ds in DATASETS:
                if scheme is None:
                    v = get_cos(pt.get((ds, mdl, attack)))
                else:
                    v = get_cos(he.get((ds, mdl, attack, scheme)))
                cells.append(fmt_cos(v))
            lines.append(" & ".join(cells) + r" \\")
        if mdl_idx < len(MODELS) - 1:
            lines.append(r"\cmidrule(l){2-" + str(n_cols) + "}")

    # ---------- Block B ----------
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{" + str(n_cols)
        + r"}{l}{\emph{Per-client-delta baselines under HE-aggregated FedAvg} $\downarrow$} \\"
    )
    lines.append(r"\midrule")
    for mdl_idx, mdl in enumerate(MODELS):
        rows_for_model = []
        for atk in BASELINES:
            rows_for_model.append((BASELINE_LABEL[atk] + r" / Plain", atk, None))
            rows_for_model.append((BASELINE_LABEL[atk] + r" / HE", atk, "ckks40"))
        for ri, (cond_label, attack, scheme) in enumerate(rows_for_model):
            cells = []
            if ri == 0:
                cells.append(r"\multirow{6}{*}{" + MODEL_LABEL[mdl] + "}")
            else:
                cells.append("")
            cells.append(cond_label)
            for ds in DATASETS:
                if scheme is None:
                    v = get_cos(pt.get((ds, mdl, attack)))
                else:
                    v = get_cos(he.get((ds, mdl, attack, scheme)))
                # Bold HE columns where collapse is most dramatic
                bold = (scheme is not None) and (v is not None) and abs(v) < 0.05
                cells.append(fmt_cos(v, bold=bold))
            lines.append(" & ".join(cells) + r" \\")
        if mdl_idx < len(MODELS) - 1:
            lines.append(r"\cmidrule(l){2-" + str(n_cols) + "}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main():
    pt = load_plaintext()
    he = load_he()
    print(f"Loaded {len(pt)} plaintext runs, {len(he)} HE runs.")

    # Coverage report
    n_pt_cells = 0
    n_he_cells = 0
    for ds in DATASETS:
        for mdl in MODELS:
            for atk in ["paired_probe"] + BASELINES:
                if (ds, mdl, atk) in pt:
                    n_pt_cells += 1
            for atk in ["paired_probe"]:
                for scheme in ("paillier", "ckks20"):
                    if (ds, mdl, atk, scheme) in he:
                        n_he_cells += 1
            for atk in BASELINES:
                if (ds, mdl, atk, "ckks40") in he:
                    n_he_cells += 1
    print(f"Plaintext coverage: {n_pt_cells}/{len(DATASETS) * len(MODELS) * 4} cells")
    print(f"HE coverage:        {n_he_cells}/{len(DATASETS) * len(MODELS) * (2 + 3)} cells "
          "(TRIP×2 schemes + 3 baselines×1 scheme)")

    tex = build_table(pt, he)

    # Inline-update the main .tex file in place between markers.
    main_tex = Path("draft/conference_101719.tex")
    src = main_tex.read_text()
    start_marker = "% AUTO-GENERATED HE TABLE START"
    end_marker = "% AUTO-GENERATED HE TABLE END"
    if start_marker not in src or end_marker not in src:
        raise RuntimeError(
            f"Could not find {start_marker!r} / {end_marker!r} in {main_tex}; "
            "the inlined HE table block must be wrapped between these markers."
        )
    pre, _, rest = src.partition(start_marker)
    _, _, post = rest.partition(end_marker)
    new_src = (
        pre
        + start_marker
        + " (regenerate via scripts/build_table_he.py)\n"
        + tex
        + end_marker
        + post
    )
    main_tex.write_text(new_src)
    print(f"Updated {main_tex} between AUTO-GENERATED HE TABLE markers "
          f"({len(tex)} chars).")


if __name__ == "__main__":
    main()
