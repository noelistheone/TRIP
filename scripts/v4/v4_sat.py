#!/usr/bin/env python
"""Saturation / norm statistics on a given config's warm-up checkpoints (default: the v4 uniform one).

For each dataset x backbone: share of the 500 targets and of 500 ordinary users (train uids 500-999)
whose honest observed update is ~0 (max |delta| < 1e-8), and their median embedding norms.
Same method as scripts/analysis/offline_checks.py:sec_sat, but for an arbitrary config, writing to
results/v4/analysis/ (results root TRIP_RESULTS, default <repo>/results). Reads the warm-up cache
(FL_WARMUP_CACHE, default <repo>/warmup_cache) and the datasets under FL_DATA_ROOT.
usage: v4_sat.py [--config configs/uni_default.yaml] [--out results/v4/analysis/uniform_sat.json]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("TRIP_RESULTS") or REPO / "results")
sys.path.insert(0, str(REPO))
os.environ.setdefault("FL_DATA_ROOT", str(REPO / "data_local"))
from fl.data import load_dataset
from fl.models import make_model
from fl.trip.server import TRIPServer
from fl import eval as E
from fl.attacks.dlg import _observation_round

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(REPO / "configs" / "uni_default.yaml"))
ap.add_argument("--out", default=str(RESULTS / "v4" / "analysis" / "uniform_sat.json"))
ap.add_argument("--datasets", nargs="+", default=["lastfm", "ml", "delicious", "amazon-beauty"])
ap.add_argument("--models", nargs="+", default=["mf", "lightgcn", "ncf"])
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
dev = torch.device(a.device)
out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
for ds in a.datasets:
    b = load_dataset(ds)
    for mdl in a.models:
        model = make_model(mdl, b.n_users, b.n_items, cfg["d"], cfg).to(dev)
        server = TRIPServer(model, b, cfg, dev)
        nw = int(E._resolve_per_dataset(cfg, "warmup_overrides", ds, cfg.get("warmup", 200)))
        wc = E._make_warmup_cfg(cfg, ds, mdl); server.cfg = wc
        p = E._warmup_cache_path(ds, mdl, cfg, wc, nw, b)
        if E._load_warmup(p, model, server, dev) is None:
            out[f"{ds}/{mdl}"] = {"missing_ckpt": p.name}; continue
        model.n_items_original = b.n_items
        acfg = E._make_attack_cfg(cfg, ds, mdl); server.cfg = acfg
        rec = {"ckpt": p.name}
        for tag, us in (("targets", E._select_attacked_uids(b, 500, 42)), ("ordinary", sorted(b.train_user_ids)[500:1000])):
            obs = _observation_round(server, b, us, acfg)
            mx = np.array([float(obs[u].abs().max()) for u in us])
            nrm = np.array([float(server.user_states[u].norm()) for u in us])
            rec[tag] = {"zero_share": float((mx < 1e-8).mean()), "median_norm": float(np.median(nrm))}
        out[f"{ds}/{mdl}"] = rec
        print(ds, mdl, rec, flush=True)
        del model, server
        if dev.type == "cuda": torch.cuda.empty_cache()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(out, indent=1))
print("->", a.out)
