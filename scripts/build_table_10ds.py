#!/usr/bin/env python
"""Build the 10-dataset paper-ready LaTeX table*: rows = (model, attack),
columns = 10 datasets. Two stacked blocks: Cosine, then Total time.
"""
from __future__ import annotations
import json
from pathlib import Path

ATTACKS = ["paired_probe", "invert_grad", "lti", "raifle"]
ATTACK_LABEL = {
    "paired_probe": r"\textbf{TRIP (ours)}",
    "invert_grad": r"InvGrad~\cite{invertgrad}",
    "lti": r"LtI~\cite{lti}",
    "raifle": r"RAIFLE~\cite{raifle}",
}
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


DEFAULT_RUN_DIRS = ["results/main"]


def load_runs(dirs=None):
    """Load every per-run JSON from the given result directories.

    Later directories override earlier ones for the same (dataset, model, attack).
    Prefer a SINGLE directory produced by one sweep with one configuration: a
    table stitched together from several partially-reconfigured runs cannot be
    described as a controlled comparison.
    """
    runs = {}
    for d in (dirs or DEFAULT_RUN_DIRS):
        p_dir = Path(d)
        if not p_dir.exists():
            continue
        for p in sorted(p_dir.glob("*.json")):
            r = json.load(open(p))
            key = (r["dataset"], r["model"], r.get("attack", "paired_probe"))
            runs[key] = r
    return runs


def fmt_cos(v, is_best=False):
    if v is None:
        return "--"
    s = f"{v:.3f}"
    return r"\textbf{" + s + "}" if is_best else s


def fmt_time(v):
    if v is None:
        return "--"
    if v < 1.0:
        return f"{v:.1f}"
    if v < 100:
        return f"{int(round(v))}"
    if v < 1000:
        return f"{int(round(v))}"
    return f"{int(round(v/100))*100}"  # round to nearest 100 for big values


def total_time(r):
    if r is None:
        return None
    t = r.get("timing", {})
    return float(t.get("prepare_sec", 0)) + float(t.get("solve_sec", 0))


def build_table(runs):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cosine similarity ($\uparrow$, top) and total "
                 r"recovery wall-clock seconds ($\downarrow$, bottom) of TRIP "
                 r"and three baselines on ten datasets and three backbones; "
                 r"per-column best in bold.}")
    lines.append(r"\label{tab:run10}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\footnotesize")
    # 2 row-headers + 10 datasets = 12 cols
    col_spec = "l l " + " ".join(["c"] * len(DATASETS))
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    h = r"\textbf{Model} & \textbf{Attack}"
    for ds in DATASETS:
        h += " & " + DS_SHORT[ds]
    h += r" \\"
    lines.append(h)
    lines.append(r"\midrule")
    n_cols = 2 + len(DATASETS)

    # ----- Cosine block -----
    lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{\emph{Cosine similarity} $\uparrow$} \\")
    lines.append(r"\midrule")
    for mdl_idx, mdl in enumerate(MODELS):
        for a_idx, a in enumerate(ATTACKS):
            cells = []
            if a_idx == 0:
                cells.append(r"\multirow{4}{*}{" + MODEL_LABEL[mdl] + "}")
            else:
                cells.append("")
            cells.append(ATTACK_LABEL[a])
            # Compute per-dataset cos for this (model, attack), and per-column best
            row_vals = [runs.get((ds, mdl, a)) for ds in DATASETS]
            row_cos = [r["cos"]["cos_mean"] if r else None for r in row_vals]
            # Column best: across attacks for this (dataset, model)
            for d_idx, ds in enumerate(DATASETS):
                col_attack_vals = [
                    runs[(ds, mdl, ax)]["cos"]["cos_mean"]
                    for ax in ATTACKS if (ds, mdl, ax) in runs
                ]
                is_best = False
                if row_cos[d_idx] is not None and col_attack_vals:
                    if row_cos[d_idx] >= max(col_attack_vals) - 1e-9:
                        is_best = True
                cells.append(fmt_cos(row_cos[d_idx], is_best=is_best))
            lines.append(" & ".join(cells) + r" \\")
        if mdl_idx < len(MODELS) - 1:
            lines.append(r"\cmidrule(l){2-" + str(n_cols) + "}")

    # ----- Time block -----
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{\emph{Total recovery time (s)} $\downarrow$} \\")
    lines.append(r"\midrule")
    for mdl_idx, mdl in enumerate(MODELS):
        for a_idx, a in enumerate(ATTACKS):
            cells = []
            if a_idx == 0:
                cells.append(r"\multirow{4}{*}{" + MODEL_LABEL[mdl] + "}")
            else:
                cells.append("")
            cells.append(ATTACK_LABEL[a])
            for ds in DATASETS:
                r = runs.get((ds, mdl, a))
                cells.append(fmt_time(total_time(r)))
            lines.append(" & ".join(cells) + r" \\")
        if mdl_idx < len(MODELS) - 1:
            lines.append(r"\cmidrule(l){2-" + str(n_cols) + "}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUN_DIRS,
                    help="Result directories (later override earlier).")
    ap.add_argument("--tex", default="paper/conference_101719.tex",
                    help="LaTeX file carrying the AUTO-GENERATED RUN10 TABLE markers. "
                         "Use --tex '' to print the table instead of splicing it.")
    args = ap.parse_args()
    runs = load_runs(args.runs)
    cells_total = len(DATASETS) * len(MODELS) * len(ATTACKS)
    cells_have = sum(1 for ds in DATASETS for mdl in MODELS for a in ATTACKS
                     if (ds, mdl, a) in runs)
    print(f"Coverage: {cells_have}/{cells_total} runs available")
    tex = build_table(runs)

    if not args.tex:
        print(tex)
        return
    main_tex = Path(args.tex)
    src = main_tex.read_text()
    start = "% AUTO-GENERATED RUN10 TABLE START"
    end = "% AUTO-GENERATED RUN10 TABLE END"
    if start not in src or end not in src:
        raise RuntimeError(
            f"Could not find {start!r}/{end!r} in {main_tex}; the inlined "
            "Table 2 must be wrapped between these markers."
        )
    pre, _, rest = src.partition(start)
    _, _, post = rest.partition(end)
    new_src = (
        pre + start + " (regenerate via scripts/build_table_10ds.py)\n"
        + tex + end + post
    )
    main_tex.write_text(new_src)
    print(f"Updated {main_tex} between AUTO-GENERATED RUN10 TABLE markers "
          f"({len(tex)} chars).")


if __name__ == "__main__":
    main()
