#!/usr/bin/env python
"""Summaries of the v4 experiments (results/v4) against the reference runs.

Sections (each skips cleanly when its results are not there yet):
  tb          E1   TB enforced in TW+BC (t=4, 16) and full PACT vs the same runs without TB
  tbnd        E1b  NCF with TB enforced and C3c dropped vs the damped runs without TB
  uniform     E2   uniform warm-up (no target oversampling, no warm-up weight decay) vs the current
                   protocol: TRIP W20/W21, baselines, norms/saturation, utility, TW+BC t=16, full PACT
  uniform_l2  E2b  the same with LightGCN/NGCF-style per-batch L2 in the warm-up (configs/uni2_*.yaml)
  budget      E3   per-client baselines at larger budgets vs the 300-step runs, same first 100 targets
  scope       E5   adaptive evasion on all SENTRY cells; aggregate-only baselines on four datasets
  timing      E6   dedicated timings vs the reference runs in results/main
  half_utility E4  utility (honest vs PACT) from the E2b and the half-length warm-up
Reads everything under the results root TRIP_RESULTS (default <repo>/results): results/v4/... for the
new runs and results/v3, results/v2, results/ for the reference runs (first existing wins).
Writes results/v4/analysis/<section>.json and prints a readable table.
usage: summarize_v4.py [section ...]
"""
from __future__ import annotations
import glob, json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
R = Path(os.environ.get("TRIP_RESULTS") or REPO / "results")
V4 = R / "v4"
OUT = V4 / "analysis"
DS4 = ["lastfm", "ml", "delicious", "amazon-beauty"]
DS10 = ["lastfm", "ml", "delicious", "douban-book", "amazon-beauty", "amazon-book", "amazon-kindle",
        "gowalla", "iFashion", "yelp2018"]
M3 = ["mf", "lightgcn", "ncf"]
AB = {"lastfm": "LFM", "ml": "ML", "delicious": "Del", "douban-book": "Dou", "amazon-beauty": "Bea",
      "amazon-book": "ABk", "amazon-kindle": "Kin", "gowalla": "Gow", "iFashion": "iFa", "yelp2018": "Yelp"}


def jl(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def cos(p):
    j = jl(p)
    return None if j is None else float(j["cos"]["cos_mean"])


def first(*paths):
    for p in paths:
        if Path(p).exists():
            return p
    return paths[-1]


def ref_dir(kind, m):
    """Reference runs as the paper tables read them (v3 > v2 > v1)."""
    if kind == "main":
        return [R / "v3/main", R / "v2/main", R / "main"]      # v3/main holds MF DLG and all LightGCN
    if kind == "w21":
        return [R / "v3/headline_w21", R / "v2/headline_w21", R / "headline_w21"]
    return []


def ref_cos(kind, ds, m, attack="paired_probe"):
    for d in ref_dir(kind, m):
        f = d / f"{ds}_{m}_{attack}.json"
        if f.exists():
            if kind == "main" and m == "ncf" and attack == "paired_probe" and d == R / "main":
                continue            # saturating-probe NCF runs are superseded by v2
            return cos(f)
    return None


def fmt(x, n=3):
    return "--" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


# ------------------------------------------------------------------ E1
def sec_tb():
    out, rows = {}, []
    for conf, old in (("bc4", "defense_gen/bc4"), ("bc16", "defense_gen/bc16"), ("full", "defense_gen/full")):
        for ds in DS4:
            for m in M3:
                new = jl(V4 / f"tb/{conf}/{ds}_{m}_paired_probe.json")
                ref = first(R / f"v3/{old}/{ds}_{m}_paired_probe.json", R / f"v2/{old}/{ds}_{m}_paired_probe.json")
                o = cos(ref)
                if new is None:
                    continue
                rec = {"cos_tb": float(new["cos"]["cos_mean"]), "cos_no_tb": o,
                       "aborted": new.get("tb_aborted_rounds"), "rounds": new.get("tb_rounds")}
                out[f"{conf}/{ds}/{m}"] = rec
                rows.append(f"{conf:5} {AB[ds]:4}/{m:9} TB {fmt(rec['cos_tb'])}  no-TB {fmt(o)}  aborted {rec['aborted']}/{rec['rounds']}")
    return out, rows


# ------------------------------------------------------------------ E1b
def sec_tbnd():
    """NCF with TB enforced and C3c dropped (ncf.damp_factor 1.0), against the damped runs without
    TB, with the null and the anonymity ceiling of the same cell (results/analysis/anon_ceiling.json
    as written by scripts/analysis/anon_ceiling.py, else results/v2/analysis/anon_ceiling.json)."""
    out, rows = {}, []
    ceil = jl(first(R / "analysis/anon_ceiling.json", R / "v2/analysis/anon_ceiling.json")) or {}
    for conf, old, t in (("bc4", "defense_gen/bc4", "4"), ("bc16", "defense_gen/bc16", "16"),
                         ("full", "defense_gen/full", None)):
        for ds in DS4:
            new = jl(V4 / f"tb_nodamp/{conf}/{ds}_ncf_paired_probe.json")
            if new is None:
                continue
            ref = first(R / f"v3/{old}/{ds}_ncf_paired_probe.json", R / f"v2/{old}/{ds}_ncf_paired_probe.json")
            c = ceil.get(f"{ds}_ncf", {})
            rec = {"cos_tb_nodamp": float(new["cos"]["cos_mean"]), "cos_damped_no_tb": cos(ref),
                   "aborted": new.get("tb_aborted_rounds"), "rounds": new.get("tb_rounds"),
                   "null": c.get("null"),
                   "ceiling": (c.get("unstratified", {}).get(t, {}).get("ceiling") if t else None)}
            out[f"{conf}/{ds}/ncf"] = rec
            rows.append(f"{conf:5} {AB[ds]:4}/ncf  TB+no-C3c {fmt(rec['cos_tb_nodamp'])}  damped no-TB "
                        f"{fmt(rec['cos_damped_no_tb'])}  null {fmt(rec['null'])}  ceiling {fmt(rec['ceiling'])}  "
                        f"aborted {rec['aborted']}/{rec['rounds']}")
    return out, rows


# ------------------------------------------------------------------ E6
def sec_timing():
    """Dedicated timings (results/v4/timing: one job per GPU on an otherwise idle machine) with the
    same statistics as the paper's timing table, next to the values computed from the reference runs
    in results/main; DLG (not timed in the paper) is added."""
    out, rows = {"cells": {}, "summary": {}}, []
    atts = ("paired_probe", "invert_grad", "raifle", "dlg")

    def load(root):
        M = {}
        for ds in DS10:
            for m in M3:
                for a in atts:
                    j = jl(root / f"{ds}_{m}_{a}.json")
                    if j and "timing" in j:
                        M[(ds, m, a)] = j["timing"]
        return M

    def stats(M):
        e2e = lambda t: float(t.get("prepare_sec", 0)) + float(t.get("solve_sec", 0))
        ts = [float(M[(d, m, "paired_probe")]["solve_sec"]) for d in DS10 for m in M3 if (d, m, "paired_probe") in M]
        s = {"tsolveMaxAll": max(ts) if ts else None}
        for a, k in (("invert_grad", "igSolveMin"), ("raifle", "raSolveMin"), ("dlg", "dlgSolveMin")):
            v = [float(M[(d, m, a)]["solve_sec"]) for d in DS10 for m in M3 if (d, m, a) in M]
            s[k] = min(v) if v else None
        s["solveRatioMin"] = s["igSolveMin"] / s["tsolveMaxAll"] if s["igSolveMin"] and s["tsolveMaxAll"] else None
        for tag, bs in (("ee", ("invert_grad", "raifle")), ("eeDLG", ("dlg",))):
            ee = [e2e(M[(d, m, b)]) / e2e(M[(d, m, "paired_probe")]) for m in M3 for d in DS10 for b in bs
                  if (d, m, b) in M and (d, m, "paired_probe") in M]
            s[f"{tag}Min"] = min(ee) if ee else None
            s[f"{tag}Max"] = max(ee) if ee else None
            s[f"{tag}BelowOne"] = sum(1 for x in ee if x < 1)
            s[f"{tag}N"] = len(ee)
        return s

    new, old = load(V4 / "timing"), load(R / "main")
    for k, t in new.items():
        out["cells"]["/".join(k)] = {"new": t, "main": old.get(k)}
    out["summary"] = {"v4_timing": stats(new), "main (paper)": stats(old)}
    for tag, s in out["summary"].items():
        rows.append(f"{tag:14} TRIP solve max {fmt(s['tsolveMaxAll'], 2)} s | InvGrad min {fmt(s['igSolveMin'], 0)} s, "
                    f"RAIFLE min {fmt(s['raSolveMin'], 0)} s, DLG min {fmt(s['dlgSolveMin'], 1)} s | "
                    f"solve ratio >= {fmt(s['solveRatioMin'], 0)} | e2e speedup IG/RA {fmt(s['eeMin'], 1)}-"
                    f"{fmt(s['eeMax'], 1)} (<1 in {s['eeBelowOne']}/{s['eeN']}) | DLG {fmt(s['eeDLGMin'], 2)}-"
                    f"{fmt(s['eeDLGMax'], 1)} (<1 in {s['eeDLGBelowOne']}/{s['eeDLGN']})")
    return out, rows


# ------------------------------------------------------------------ E4
def sec_half_utility():
    """Utility (honest vs PACT, 200 rounds, 3 seeds) from the E2b warm-up ("full") and from the
    half-length warm-up ("half"), MF and LightGCN: start/honest/PACT Recall@20, the honest arm's
    relative change, PACT vs honest, 2 s.e. of the difference, and the same for NDCG@20."""
    out, rows = {}, []
    for tag, sub in (("full", "uniform_l2"), ("half", "half")):
        for ds in DS4:
            for m in ("mf", "lightgcn"):
                j = jl(V4 / sub / "utility" / f"{ds}_{m}_r200.json")
                if j is None:
                    continue
                s = j["start_ranking"]; h = j["arms"]["honest"]["ranking_mean"]; p = j["arms"]["pact"]["ranking_mean"]
                hs = j["arms"]["honest"]["ranking_std"]; ps = j["arms"]["pact"]["ranking_std"]
                r = {"start": s["recall@20"], "honest": h["recall@20"], "pact": p["recall@20"],
                     "moved_pct": 100 * (h["recall@20"] - s["recall@20"]) / s["recall@20"],
                     "pact_vs_honest_pct": 100 * (p["recall@20"] - h["recall@20"]) / h["recall@20"],
                     "two_se": 2 * ((hs["recall@20"] ** 2 + ps["recall@20"] ** 2) / 3) ** 0.5,
                     "ndcg_moved_pct": 100 * (h["ndcg@20"] - s["ndcg@20"]) / s["ndcg@20"],
                     "ndcg_pact_vs_honest_pct": 100 * (p["ndcg@20"] - h["ndcg@20"]) / h["ndcg@20"]}
                out[f"{tag}/{ds}/{m}"] = r
                rows.append(f"{tag:5} {AB[ds]:4}/{m:9} R@20 start {r['start']:.4f} honest {r['honest']:.4f} "
                            f"({r['moved_pct']:+.2f}%) PACT {r['pact']:.4f} ({r['pact_vs_honest_pct']:+.2f}% vs honest)")
    return out, rows


# ------------------------------------------------------------------ E2
def sec_uniform(sub="uniform", satfile="uniform_sat.json"):
    U = V4 / sub
    out, rows = {}, []
    for ds in DS4:
        for m in M3:
            rec = {}
            for att in ("paired_probe", "invert_grad", "raifle", "dlg", "lti"):
                rec[f"uni_{att}"] = cos(U / f"main/{ds}_{m}_{att}.json")
                rec[f"ref_{att}"] = ref_cos("main", ds, m, att)
            j = jl(U / f"main/{ds}_{m}_paired_probe.json")
            rec["uni_null"] = None if j is None else float(j["cos"]["cos_trivial_floor"])
            rr = None
            for d in ref_dir("main", m):
                f = d / f"{ds}_{m}_paired_probe.json"
                if f.exists() and not (m == "ncf" and d == R / "main"):
                    rr = jl(f); break
            rec["ref_null"] = None if rr is None else float(rr["cos"]["cos_trivial_floor"])
            rec["uni_w21"] = cos(U / f"w21/{ds}_{m}_paired_probe.json")
            rec["ref_w21"] = ref_cos("w21", ds, m)
            for conf in ("bc16", "full"):
                jj = jl(U / f"{conf}/{ds}_{m}_paired_probe.json")
                rec[f"uni_{conf}"] = None if jj is None else float(jj["cos"]["cos_mean"])
                rec[f"uni_{conf}_aborted"] = None if jj is None else jj.get("tb_aborted_rounds")
            ut = jl(U / f"utility/{ds}_{m}_r200.json")
            if ut:
                st = ut.get("start_ranking", {})
                h = ut["arms"]["honest"].get("ranking_mean", {}); p = ut["arms"]["pact"].get("ranking_mean", {})
                for k in ("recall@20", "ndcg@20"):
                    if k in st and k in h and k in p and st[k]:
                        rec[f"util_{k}_start"] = st[k]; rec[f"util_{k}_honest"] = h[k]; rec[f"util_{k}_pact"] = p[k]
                        rec[f"util_{k}_pact_vs_honest_pct"] = 100 * (p[k] - h[k]) / h[k] if h[k] else None
                        rec[f"util_{k}_honest_moved_pct"] = 100 * (h[k] - st[k]) / st[k]
            out[f"{ds}/{m}"] = rec
            rows.append(f"{AB[ds]:4}/{m:9} TRIP20 {fmt(rec['uni_paired_probe'])} (ref {fmt(rec['ref_paired_probe'])})  "
                        f"TRIP21 {fmt(rec['uni_w21'])} (ref {fmt(rec['ref_w21'])})  "
                        f"IG {fmt(rec['uni_invert_grad'])} ({fmt(rec['ref_invert_grad'])})  RA {fmt(rec['uni_raifle'])} ({fmt(rec['ref_raifle'])})  "
                        f"DLG {fmt(rec['uni_dlg'])} ({fmt(rec['ref_dlg'])})  LtI {fmt(rec['uni_lti'])} ({fmt(rec['ref_lti'])})  null {fmt(rec['uni_null'])} ({fmt(rec['ref_null'])})  "
                        f"bc16 {fmt(rec['uni_bc16'])} full {fmt(rec['uni_full'])}  "
                        f"R@20 moved {fmt(rec.get('util_recall@20_honest_moved_pct'), 2)}% PACT-vs-honest {fmt(rec.get('util_recall@20_pact_vs_honest_pct'), 2)}%")
    sat = jl(OUT / satfile)
    if sat:
        out["_sat"] = sat
    return out, rows


def sec_uniform_l2():
    """E2b: uniform warm-up plus LightGCN/NGCF-style per-batch L2 (configs/uni2_*.yaml)."""
    return sec_uniform("uniform_l2", "uniform_l2_sat.json")


# ------------------------------------------------------------------ E3
def sec_budget():
    out, rows = {}, []
    for m in ("lightgcn", "ncf"):
        for ds in DS10:
            for att in ("invert_grad", "raifle", "dlg"):
                nf = V4 / f"budget/{ds}_{m}_{att}.npz"
                if not nf.exists():
                    continue
                z = np.load(nf); n = len(z["cos"])
                ref = None
                for d in (R / "v3/main", R / "main"):
                    f = d / f"{ds}_{m}_{att}.npz"
                    if f.exists():
                        zr = np.load(f)
                        if list(zr["attacked_uids"][:n]) == list(z["attacked_uids"]):
                            ref = float(zr["cos"][:n].mean())
                        break
                tj = (jl(V4 / f"budget/{ds}_{m}_{att}.json") or {}).get("timing", {})
                rec = {"cos_big": float(z["cos"].mean()), "cos_ref_same_targets": ref, "n": n,
                       **{k: tj[k] for k in ("budget_n_iter", "best_step_frac_mean", "best_in_last_10pct_frac",
                                              "budget_max_iter", "lbfgs_iters_mean", "hit_cap_frac") if k in tj}}
                out[f"{ds}/{m}/{att}"] = rec
                rows.append(f"{AB[ds]:4}/{m:9} {att:12} big {fmt(rec['cos_big'])}  300-step {fmt(ref)}  "
                            f"delta {fmt(None if ref is None else rec['cos_big'] - ref)}  "
                            + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()
                                       if k not in ("cos_big", "cos_ref_same_targets", "n")))
    return out, rows


# ------------------------------------------------------------------ E5
def sec_scope():
    out, rows = {"evasion": {}, "heagg": {}}, []
    for ds in DS4:
        for m in M3:
            s = jl(first(V4 / f"sentry/{ds}_{m}.json", R / f"v3/sentry/{ds}_{m}.json", R / f"v2/sentry/{ds}_{m}.json")) or {}
            rf, rn = s.get("evasion_rate_per_round", {}), s.get("evasion_rate_per_round_nn_only", {})
            for sep in ("0.1", "0.3", "0.5", "0.6", "0.707"):
                f = first(V4 / f"evasion_adaptive/s{sep}/{ds}_{m}_paired_probe.json",
                          R / f"v2/evasion_adaptive/s{sep}/{ds}_{m}_paired_probe.json")
                c = cos(f)
                rec = {"trip_cos": c, "flag_all": rf.get(f"adaptive:{sep}"), "flag_nn": rn.get(f"adaptive:{sep}"),
                       "fa_all": s.get("false_alarm_fresh"),
                       "fa_nn": (s.get("per_feature", {}).get("nn_dup") or [None, None])[1] if isinstance(s.get("per_feature", {}).get("nn_dup"), list) else None}
                out["evasion"][f"{ds}/{m}/s{sep}"] = rec
            r = out["evasion"][f"{ds}/{m}/s0.707"]
            rows.append(f"evasion s=0.707 {AB[ds]:4}/{m:9} TRIP {fmt(r['trip_cos'])}  flag nn {fmt(r['flag_nn'], 2)} all {fmt(r['flag_all'], 2)}  FA all {fmt(r['fa_all'], 3)}")
    for f in sorted(glob.glob(str(V4 / "he_aggregate/*.json")) + glob.glob(str(R / "v3/he_aggregate/*.json")) + glob.glob(str(R / "v2/he_aggregate/*.json"))):
        j = jl(f)
        if not j or "cos" not in j:
            continue
        key = f"{j['dataset']}/{j['model']}/{j.get('attack')}"
        if key in out["heagg"]:
            continue
        out["heagg"][key] = float(j["cos"]["cos_mean"])
    for k, v in sorted(out["heagg"].items()):
        rows.append(f"heagg {k:36} {v:.3f}")
    return out, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    secs = sys.argv[1:] or ["tb", "tbnd", "uniform", "uniform_l2", "budget", "scope", "timing", "half_utility"]
    for s in secs:
        data, rows = globals()[f"sec_{s}"]()
        (OUT / f"{s}.json").write_text(json.dumps(data, indent=1))
        print(f"== {s}: {len(data)} entries -> {OUT / (s + '.json')}")
        for r in rows:
            print("  " + r)


if __name__ == "__main__":
    main()
