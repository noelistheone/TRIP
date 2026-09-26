# TRIP: Training-Free User-Embedding Recovery in Federated Recommendation

Reference implementation of **TRIP** (Temporal Reconstruction via Inverse Probing),
a server-side privacy attack that recovers per-user embeddings in federated
recommender systems from **only the aggregated item-embedding update**, the exact
signal that secure aggregation or a homomorphic-encryption deployment releases,
and of **PACT**, the defense that confines what such a server can learn to
within-block anonymity. TRIP appends pairs of near-identical probe items, shifts
which targets train each pair from round to round, and recovers all targets with one
least-squares solve: no auxiliary network, no per-user optimization, no shadow
supervision.

The repository provides:

- A federated RecSys framework (1 user = 1 client, FedAvg + BPR) for three backbones:
  **MF**, **LightGCN**, and **NCF** (`fl/`).
- The **TRIP / paired-probe** attack and a **single-probe** ablation (`fl/trip/`,
  `fl/attacks/paired_probe.py`), with the capability ladder that switches its
  server-side deviations off one at a time (`configs/tm*_*.yaml`).
- Adaptations of four gradient-inversion baselines (**DLG**, **iDLG**,
  **Inverting Gradients**, **Learning-to-Invert (LtI)**) plus a **RAIFLE**
  adaptation, all run in plaintext on individual updates, and an aggregate-only
  mode that gives them the sum instead.
- **PACT** (`fl/pact/`): write-set sovereignty with PRF-sampled negatives
  (`prf.py`, `policy.py`), block-quantized participation with a beacon partition and
  pairwise masks (`blocks.py`, `mask.py`), transcript binding with the norm-band and
  catalogue-growth checks (`catalog.py`, enforced through `pact.binding.enforce`), and
  the client-side **SENTRY** detector (`sentry.py`).
- An HE simulation (per-coordinate precision-loss noise) and aggregate-noise sweeps.
- One YAML config per experimental condition, so every number in the paper maps to a
  config and a script (see "Reproducing the paper").

## Repository layout

```
fl/                     # Core library
  data.py               #   dataset loading + BPR negative sampling
  client.py             #   per-client local BPR training (SGD / Adam, honest recipe)
  models.py             #   MF / LightGCN / NCF definitions
  eval.py               #   experiment runner, warm-up cache, metrics
  utils.py              #   seeding, GPU selection
  attacks/              #   AttackBase + TRIP, baselines, ablation
  trip/                 #   probe construction, sliding-window allocator, solvers, server
  pact/                 #   PACT policy, PRF, blocks, masks, catalogue binding, SENTRY
configs/                # default.yaml (annotated reference) + one file per condition
scripts/                # experiment runners and result aggregation
  v4/                   #   follow-up experiments: job lists, sweep runner, summaries
  analysis/             #   offline checks of the theory (projection limits, ceilings, ...)
tests/                  # unit tests (allocator, freeze/restore, MF toy)
```

## Installation

```bash
conda create -n fl_env python=3.11 -y
conda activate fl_env
# Install PyTorch matching your CUDA version (the reported runs used PyTorch 2.6 / CUDA 12.4):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Data

Datasets are **not** distributed with this repository. Each dataset is a directory
`<name>/{train,test}.txt`, where every line is a user's interaction list:

```
<uid> <item1> <item2> ...
```

(whitespace-separated, 0-indexed). Point the loader at your data root via the
`FL_DATA_ROOT` environment variable (default: `data_local/`):

```bash
export FL_DATA_ROOT=/path/to/datasets
```

The ten datasets used in the experiments are public benchmarks: LastFM, MovieLens-100K,
Delicious, Douban-Book, Amazon-{Beauty,Book,Kindle}, Gowalla, iFashion, and Yelp2018.
`scripts/resplit_ml.py` shows how MovieLens-100K was re-split per user. Warm-up
checkpoints are cached under `warmup_cache/` (override with `FL_WARMUP_CACHE`); every
attack and defense run of a `(dataset, backbone)` cell starts from the same checkpoint.

## Quick start (smoke test)

```bash
python scripts/run_one.py --dataset lastfm --model mf
```

Expected: `cos_mean ≥ 0.97` for LastFM/MF.

## Reproducing the paper

All runs take a config from `configs/` and write to `--out`; `--skip-existing` makes
them restartable. The mapping below lists, per part of the evaluation, the script, the
configs and the output directory the tables were built from.

| Part of the evaluation | Script | Configs | Output |
|---|---|---|---|
| Recovery of all attacks, $W{=}20$ (Table 2) | `run_all.py --attacks paired_probe single_probe dlg invert_grad lti raifle` | `default.yaml` | `results/main` |
| TRIP at $W{=}21$ (Table 2, headline) | `run_all.py --attacks paired_probe single_probe` | `headline_w21.yaml` | `results/headline_w21` |
| Seed variability | `run_all.py` / `run_one.py` | `seed43.yaml`, `seed44.yaml`, `seed43_w21.yaml`, `seed44_w21.yaml` | `results/seeds` |
| Window lengths and rank (theory, no noise) | `run_one.py` | `w19.yaml`, `w21.yaml`, `w25.yaml`, `w50.yaml` | `results/wsweep` |
| Aggregate noise, fixed and oracle $\lambda$ | `run_one.py` | `bv_W{19,20,21,25}_n*.yaml`, `rs_W*_n*.yaml` | `results/biasvar`, `results/ridge_src` |
| Fewer rounds, more pairs | `run_one.py` | `rounds_K*_*.yaml` | `results/rounds` |
| Capability ladder (which deviations TRIP needs) | `run_one.py` | `tm3_hbc_W{20,21}.yaml` (C1+C2 only), `tm3_probesonly_W{20,21}.yaml` (+C3a), `tm_hbc_sgd_W{20,21}.yaml` (+C3b), `tm_full_W{20,21}.yaml`; NCF without C3c: `headline_w21_nodamp.yaml` | `results/threatmodel`, `results/ncf_nodamp` |
| PACT vs. TRIP (Table 3) | `run_one.py` | `dg_{none,bc4,bc16,ws,full}_W21.yaml`; block-size sweep `pd_bc_t{2,...,64}.yaml` | `results/defense_gen`, `results/defense` |
| Transcript binding enforced | `run_one.py` | `dg_{bc4,bc16,full}_W21_tb.yaml`; NCF attacker without C3c: `dg_*_W21_tb_nodamp.yaml` | `results/v4/tb`, `results/v4/tb_nodamp` |
| Ranking accuracy, honest vs. PACT | `run_defense_utility.py --rounds 200 --cohort 128 --seeds 42 43 44 --fed-wd 0.0 --check-masks` | `pact_full.yaml` | `results/defense_utility` |
| SENTRY detector, scaled probes | `run_sentry_eval.py --benign-rounds 120 --fresh-rounds 200 --attack-rounds 40` | `headline_w21.yaml` | `results/sentry` |
| Detector-aware attacker (widened pairs) | `run_one.py` | `ev_adapt_s{0.1,0.3,0.5,0.6,0.707}.yaml`; scaled probes `ev_eps*.yaml` | `results/evasion_adaptive`, `results/evasion` |
| Per-client baselines given only the aggregate | `run_he_immunity.py --baseline-mode aggregate` | `headline_w21.yaml` | `results/he_aggregate` |
| Sole-writer rows under full PACT | `run_residual_leak.py` | `pact_full.yaml` | `results/residual_leak.json` |
| Warm-up variants, baseline budgets, dedicated timing | see `scripts/v4/README.md` | `uni_*.yaml`, `uni2_*.yaml`, `uni2h_*.yaml`, `budget_v4.yaml` | `results/v4/...` |

The offline checks of the theory (projection limits and their isotropic prediction,
the noise decomposition, the anonymity ceiling, saturation statistics) read these
result directories; see `scripts/analysis/`. `scripts/v4/` holds the job lists and the
one-job-per-GPU runner used for the follow-up experiments, plus the scripts that
summarize them.

## HE-immunity experiments

```bash
# TRIP under three HE schemes (Paillier-30 / CKKS-40 / CKKS-20) vs. baselines:
python scripts/run_he_immunity.py --datasets lastfm ml delicious --models mf lightgcn ncf

# HE precision-loss noise sweep on TRIP:
python scripts/run_he_ablation.py --noise-levels 1e-10 1e-8 1e-6 1e-4 --out-root results/he_ablation

# Privacy-utility trade-off (operational HE rounds):
python scripts/run_he_tradeoff.py --datasets lastfm ml douban-book --models mf
```

## Aggregation and figures

```bash
python scripts/summarize_attacks.py --runs results/main --out results/_attack_comparison.md
python scripts/summarize.py    --runs results/run0 results/run1 --out results/_summary.md
python scripts/summarize_he.py --baseline results/run0 --he-root results/he_ablation --out results/_he_summary.md
```

The `scripts/build_table_*.py` and `scripts/build_figure_tradeoff.py` helpers
regenerate LaTeX tables / figures from the `results/` artifacts.

## Output convention

Each `(dataset, model[, attack])` run writes to its `--out` directory:

- `<dataset>_<model>[_<attack>].md`: human-readable report
- `<dataset>_<model>[_<attack>].json`: machine-readable metrics (cosine statistics,
  timing, and for PACT runs the number of rounds rejected by transcript binding)
- `<dataset>_<model>[_<attack>].npz`: `U_hat`, `true_U`, `cos`, `attacked_uids`,
  `eval_mask`, and for TRIP the measurements `G` and the realised operator `A`

## Tests

```bash
export FL_DATA_ROOT=/path/to/datasets   # one test loads LastFM
bash tests/run_all.sh            # all tests (env must be active)
python tests/test_allocator.py   # single test
```

## Key configuration knobs

`configs/default.yaml` is the annotated reference (comments explain each per-dataset
and per-model override). Core attack-geometry knobs:

| knob | meaning | default |
|------|---------|---------|
| `d` | embedding dimension | 64 |
| `N_attack` | number of attacked users | 500 |
| `K` | number of probe pairs (catalog grows by `2K`) | 10 |
| `W` | sliding-window length per probe per round | 20 |
| `lr` | attack-phase SGD learning rate | 0.005 |
| `attack_optimizer_override`, `attack_weight_decay_override`, `attack_local_epochs` | what the server prescribes in attack rounds; `attack_local_epochs: honest` keeps each client's own epoch count | sgd, 0, 1 |
| `attack_probes_only` | clients train on probe triples only (C3a) | true |
| `ncf.damp_factor` | MLP scaling broadcast on NCF (C3c); 1.0 disables it | 0.1 |
| `pact.*` | write-set sovereignty, block size `t`, masks, `binding.enforce` (clients check norm band and catalogue growth every round), SENTRY | see `pact_full.yaml` |

The recovery math assumes the attack-phase update of a probe row is a multiple of the
user's embedding, which holds for one step of plain SGD (`Δθ = −η·∇L`). The ladder
configs drop that prescription to measure what an honest Adam recipe still leaks.
`configs/he.yaml` is the HE-enabled variant (`he.enabled: true`).
