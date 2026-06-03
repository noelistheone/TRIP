#!/usr/bin/env python
"""Build the paper-ready LaTeX table*: 4-dataset comparison of
TRIP-recovered metrics vs true-u metrics (recall/NDCG/precision/HR @10/20/50).
"""
from __future__ import annotations
import json
from pathlib import Path

DATASETS = ["lastfm", "ml", "delicious"]
DS_LABEL = {
    "lastfm": "LastFM",
    "ml": "MovieLens",
    "delicious": "Delicious",
}
MODELS = ["mf", "lightgcn", "ncf"]
MODEL_LABEL = {"mf": "MF", "lightgcn": "LightGCN", "ncf": "NCF"}
KS = [10, 20, 50]
METRICS = ["recall", "ndcg", "precision", "hr"]


def load_runs():
    """delicious_fix > run4_ncf_fix > run4 (4-dataset table is unaffected by
    the run6_amazon_* fixes since those datasets aren't in this table)."""
    runs = {}
    for p in sorted(Path("results/run4").glob("*paired_probe.json")):
        r = json.load(open(p))
        runs[(r["dataset"], r["model"])] = r
    for p in sorted(Path("results/run4_ncf_fix").glob("*paired_probe.json")):
        r = json.load(open(p))
        runs[(r["dataset"], r["model"])] = r
    fix = Path("results/run4_delicious_fix")
    if fix.exists():
        for p in sorted(fix.glob("*paired_probe.json")):
            r = json.load(open(p))
            runs[(r["dataset"], r["model"])] = r
    return runs


def fmt_num(v):
    if v is None:
        return "--"
    if abs(v) < 0.00005 and v != 0:
        return r"$<$0.0001"
    return f"{v:.4f}"


def build_table(runs):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ranking quality of TRIP-recovered user embeddings "
                 r"vs.\ ground-truth ($\textit{true }u$) on three datasets and "
                 r"three backbones.}")
    lines.append(r"\label{tab:run4}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\footnotesize")
    # 3 row-headers + 4 metrics × 3 ks = 15 cols
    col_spec = "l l l " + " ".join(["c c c"] * 4)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    h1 = (r"& & "
          r"& \multicolumn{3}{c}{\textbf{Recall}} "
          r"& \multicolumn{3}{c}{\textbf{NDCG}} "
          r"& \multicolumn{3}{c}{\textbf{Precision}} "
          r"& \multicolumn{3}{c}{\textbf{HR}} \\")
    lines.append(h1)
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12} \cmidrule(lr){13-15}")
    h2 = (r"\textbf{Dataset} & \textbf{Model} & \textbf{Query} "
          r"& @10 & @20 & @50 & @10 & @20 & @50 & @10 & @20 & @50 & @10 & @20 & @50 \\")
    lines.append(h2)
    lines.append(r"\midrule")

    for ds_idx, ds in enumerate(DATASETS):
        for mdl_idx, mdl in enumerate(MODELS):
            r = runs.get((ds, mdl))
            if r is None:
                continue
            rk = r.get("ranking", {})
            rk_true = r.get("ranking_true_U", {})

            # First row: TRIP recovered
            cells = []
            if mdl_idx == 0:
                cells.append(r"\multirow{6}{*}{" + DS_LABEL[ds] + "}")
            else:
                cells.append("")
            cells.append(r"\multirow{2}{*}{" + MODEL_LABEL[mdl] + "}")
            cells.append(r"\textbf{TRIP}")
            for m in METRICS:
                for k in KS:
                    cells.append(fmt_num(rk.get(f"{m}@{k}")))
            lines.append(" & ".join(cells) + r" \\")

            # Second row: true u
            cells = ["", "", r"\textit{true $u$}"]
            for m in METRICS:
                for k in KS:
                    cells.append(fmt_num(rk_true.get(f"{m}@{k}")))
            lines.append(" & ".join(cells) + r" \\")

            if mdl_idx < len(MODELS) - 1:
                lines.append(r"\cmidrule(l){2-15}")
        if ds_idx < len(DATASETS) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main():
    runs = load_runs()
    have = sum(1 for ds in DATASETS for mdl in MODELS if (ds, mdl) in runs)
    print(f"Coverage: {have}/{len(DATASETS) * len(MODELS)} (dataset, model) cells")
    tex = build_table(runs)

    main_tex = Path("draft/conference_101719.tex")
    src = main_tex.read_text()
    start = "% AUTO-GENERATED RUN4 TABLE START"
    end = "% AUTO-GENERATED RUN4 TABLE END"
    if start not in src or end not in src:
        raise RuntimeError(
            f"Could not find {start!r}/{end!r} in {main_tex}; the inlined "
            "Table 1 must be wrapped between these markers."
        )
    pre, _, rest = src.partition(start)
    _, _, post = rest.partition(end)
    new_src = (
        pre + start + " (regenerate via scripts/build_table_4ds.py)\n"
        + tex + end + post
    )
    main_tex.write_text(new_src)
    print(f"Updated {main_tex} between AUTO-GENERATED RUN4 TABLE markers "
          f"({len(tex)} chars).")


if __name__ == "__main__":
    main()
