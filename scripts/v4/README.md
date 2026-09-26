# scripts/v4: follow-up experiments E1-E6

Job lists, drivers and analysis tools for the follow-up experiment groups. The jobs call the
ordinary experiment scripts in `scripts/` (`run_one.py`, `run_all.py`, `run_defense_utility.py`,
`run_sentry_eval.py`, `run_he_immunity.py`) with the configs named below, using the active `python`.

## Paths

| what | resolution |
|---|---|
| repository root | parent of `scripts/`, from each script's own location |
| results root | `TRIP_RESULTS` (default `<repo>/results`); the analysis tools read and write here |
| data root | `FL_DATA_ROOT` (default `<repo>/data_local`), exported to every job |
| warm-up cache | `FL_WARMUP_CACHE` (default `<repo>/warmup_cache`), exactly as `fl/eval.py` resolves it |
| logs, claim files, sweep status | `<repo>/logs/sweep/` |

The job lists write to `results/v4/...` relative to the repository root (every job runs after
`cd <repo>`). If `TRIP_RESULTS` points elsewhere, make `<repo>/results` a symlink to it so that
the analysis tools find the runs.

## Running the job lists

`sweep.py` is the runner: `python scripts/v4/sweep.py --jobs scripts/v4/jobs/<list>.json --gpus 0 1`.
A job list is a JSON list of `{"name", "cmd", "done"}`. Each worker owns one GPU and runs one job
at a time; a job is skipped when its `done` file exists, so a list can be restarted after an
interruption. Several `sweep.py` instances (e.g. one per GPU) can share one list through the claim
files in `logs/sweep/claims/`. A worker waits until at least `SWEEP_MIN_FREE_MIB` (default 4000)
MiB are free on its card before starting a job. `--dry-run` prints the composed commands.

The `<group>_g0.json` / `<group>.json` pairs split one group between two GPUs. The drivers run
one job per GPU at a time; the runs were produced as

```bash
python scripts/v4/sweep.py --jobs scripts/v4/jobs/v4_tb.json --gpus 0 1                 # E1
J=scripts/v4/jobs
bash scripts/v4/run_v4_chain.sh 0 0 $J/v4_uniform_g0.json $J/v4_budget_g0.json $J/v4_scope_g0.json $J/v4_half_g0.json & P0=$!
bash scripts/v4/run_v4_chain.sh 1 0 $J/v4_uniform.json    $J/v4_budget.json    $J/v4_scope.json    $J/v4_half.json    & P1=$!
bash scripts/v4/run_v4_timing.sh 0 $P0 $P1 &                                             # E6, after both chains
bash scripts/v4/run_v4_timing.sh 1 $P0 $P1 &
```

`run_v4_chain.sh GPU WAIT_PID list...` runs the given lists one after another on one GPU (after
process `WAIT_PID` exits; `0` = start now). `run_v4_timing.sh GPU PID0 PID1` waits for both chains
and then runs `jobs/v4_timing_g<GPU>.json` alone on its GPU; the GPU-0 instance logs all GPU
processes every 5 min to `logs/sweep/v4_timing_machine_state.log`.

## Experiment groups

The four-dataset set is LastFM, ML, Delicious, Amazon-Beauty; the ten-dataset set adds Douban-Book,
Amazon-Book, Amazon-Kindle, Gowalla, iFashion, Yelp2018. Backbones: MF, LightGCN, NCF. Output
directories below are under `results/v4/`.

**E1: TB enforced** (TW+BC t=4, t=16, full PACT; 4 datasets x 3 backbones, 36 runs). List
`v4_tb.json`. `run_one.py` with `configs/dg_{bc4,bc16,full}_W21_tb.yaml` -> `tb/{bc4,bc16,full}`.
Summary: `summarize_v4.py tb` (vs. the same defenses without TB in `results/v3|v2/defense_gen/*`).

**E1b: TB enforced with C3c dropped on NCF** (4 datasets, 12 runs). Inside `v4_scope.json` and
`v4_scope_g0.json`. `run_one.py` with `configs/dg_{bc4,bc16,full}_W21_tb_nodamp.yaml` ->
`tb_nodamp/{bc4,bc16,full}`. Summary: `summarize_v4.py tbnd` (null and anonymity ceiling from
`results/analysis/anon_ceiling.json`, see `scripts/analysis/`).

**E2: uniform warm-up** (4 datasets x 3 backbones, 60 jobs). Lists `v4_uniform.json` (LastFM,
Amazon-Beauty) + `v4_uniform_g0.json` (ML, Delicious). Per cell:
`run_all.py --attacks paired_probe invert_grad raifle dlg lti --config configs/uni_default.yaml` -> `uniform/main`;
`run_one.py --config configs/uni_w21.yaml` -> `uniform/w21`;
`run_defense_utility.py --config configs/uni_pact_full.yaml --rounds 200 --cohort 128 --seeds 42 43 44 --fed-wd 0.0 --check-masks` -> `uniform/utility`;
`run_one.py` with `configs/uni_dg_bc16_W21_tb.yaml` -> `uniform/bc16` and `configs/uni_dg_full_W21_tb.yaml` -> `uniform/full`.
Summary: `summarize_v4.py uniform`; `v4_sat.py` (saturation -> `analysis/uniform_sat.json`);
`v4_ceiling.py uniform`.

**E2b: uniform warm-up + per-batch L2** (same grid, 40 jobs). First 20 jobs of `v4_budget.json` +
`v4_budget_g0.json`; the E2 commands with `configs/uni2_{default,w21,pact_full,dg_bc16_W21_tb,dg_full_W21_tb}.yaml`
-> `uniform_l2/{main,w21,utility,bc16,full}`. Summary: `summarize_v4.py uniform_l2`;
`v4_sat.py --config configs/uni2_default.yaml --out results/v4/analysis/uniform_l2_sat.json`;
`v4_ceiling.py uniform_l2`; `init_share.py` (source `e2b`).

**E3: baseline budgets** (LightGCN, NCF x 10 datasets; InvGrad, RAIFLE, DLG; 20 jobs). Last 20
jobs of `v4_budget.json` + `v4_budget_g0.json`:
`run_all.py --attacks invert_grad raifle dlg --config configs/budget_v4.yaml` -> `budget`.
Summary: `summarize_v4.py budget` (same first 100 targets as the 300-step runs); `budget_vs_trip.py`
(vs. TRIP W=21 on those targets).

**E4: utility from a half-length warm-up** (MF, LightGCN x 4 datasets, 16 jobs). Lists
`v4_half.json` + `v4_half_g0.json`. `run_all.py --attacks paired_probe --config configs/uni2h_default.yaml`
-> `half/main`; `run_defense_utility.py --config configs/uni2h_pact_full.yaml` (same flags as E2)
-> `half/utility`. Summary: `summarize_v4.py half_utility` (start / honest / PACT Recall@20 and NDCG@20
from the E2b and the half-length warm-up, written to `results/v4/analysis/half_utility.json`).

**E5: scope** (65 jobs; the lists also hold the 12 E1b jobs). Lists `v4_scope.json` + `v4_scope_g0.json`.
Adaptive evasion: `run_one.py --config configs/ev_adapt_s{0.1,0.3,0.5,0.6,0.707}.yaml` on 4 datasets x
3 backbones -> `evasion_adaptive/s<sep>` (LastFM/MF was already in `results/v2/evasion_adaptive`, so
11 new cells per separation; the summary completes the 12-cell grid from there). Aggregate-only
baselines: `run_he_immunity.py --config configs/headline_w21.yaml --baseline-mode aggregate` for
Delicious and Amazon-Beauty x 3 backbones -> `he_aggregate` (LastFM and ML come from
`results/v3|v2/he_aggregate`). SENTRY re-run on NCF, 4 datasets:
`run_sentry_eval.py --config configs/headline_w21.yaml --benign-rounds 120 --fresh-rounds 200 --attack-rounds 40`
-> `sentry`. Summary: `summarize_v4.py scope`.

**E6: dedicated timing** (10 datasets x 3 backbones; TRIP, InvGrad, RAIFLE, DLG; 30 jobs). Lists
`v4_timing_g0.json` + `v4_timing_g1.json`, started by `run_v4_timing.sh` only after all other jobs
have finished: `run_all.py --attacks paired_probe invert_grad raifle dlg --config configs/default.yaml`
-> `timing`. Summary: `summarize_v4.py timing` (vs. the timings stored with `results/main`).

## Analysis tools

All read `TRIP_RESULTS` and write to `results/v4/analysis/`.

- `summarize_v4.py [section ...]`: sections `tb tbnd uniform uniform_l2 budget scope timing` (all by
  default); writes `<section>.json` and prints one table per section. Reference runs are taken from
  `results/v3/...`, then `results/v2/...`, then `results/...`.
- `v4_sat.py [--config C] [--out F]`: share of the 500 targets / 500 ordinary users whose honest
  update is ~0 and their median embedding norms, from the warm-up checkpoints of config C (GPU).
- `v4_ceiling.py [variant ...]`: Theorem-2 anonymity ceiling (t=16, unstratified) from the true
  embeddings stored in `results/v4/<variant>/bc16` (CPU).
- `init_share.py`: share of each target's embedding that is its random initialization, and TRIP's
  recovery of the learned part (W=21 runs, per-client baselines, E2b).
- `budget_vs_trip.py`: E3 baselines vs. TRIP (W=21) on the same targets, with the learned-part cosine.

The offline theory checks (`offline_checks.py`, `anon_ceiling.py`) live in `scripts/analysis/` (own README).
