# TRIP: Training-Free User-Embedding Recovery in Federated Recommendation

Reference implementation of **TRIP** (Temporal Reconstruction via Inverse Probing),
a server-side privacy attack that recovers per-user embeddings in federated
recommender systems from **only the FedAvg-aggregated item-embedding update** — the
exact signal a homomorphic-encryption (HE) deployment is designed to release. The
recovery is a single closed-form matrix pseudo-inverse: no auxiliary network, no
per-user optimization, no shadow supervision.

The repository provides:

- A federated RecSys framework (1 user = 1 client, FedAvg + BPR) for three backbones:
  **MF**, **LightGCN**, and **NCF**.
- The **TRIP / Paired-Probe** attack and a **Single-Probe** ablation.
- Faithful adaptations of four gradient-inversion baselines — **DLG**, **iDLG**,
  **Inverting Gradients**, and **Learning-to-Invert (LtI)** — plus a **RAIFLE** adaptation.
- An HE simulation (per-coordinate precision-loss noise) for the encryption-immunity
  experiments.

## Repository layout

```
fl/                     # Core library
  data.py               #   dataset loading + BPR negative sampling
  client.py             #   per-client local BPR-SGD training
  models.py             #   MF / LightGCN / NCF definitions
  eval.py               #   experiment runner + metric/table rendering
  utils.py              #   seeding, GPU selection
  attacks/              #   AttackBase + TRIP, baselines, ablation
  trip/                 #   probe construction, sliding-window allocator, solvers
configs/                # default.yaml (annotated reference) + he.yaml
scripts/                # experiment runners, sweeps, and result aggregation
tests/                  # unit tests (allocator, freeze/restore, MF toy)
```

## Installation

```bash
conda create -n fl_env python=3.11 -y
conda activate fl_env
# Install PyTorch matching your CUDA version (example: CUDA 12.1):
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
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

The ten datasets used in our experiments are public benchmarks: LastFM, MovieLens,
Delicious, Douban-Book, Amazon-{Beauty,Book,Kindle}, Gowalla, iFashion, and Yelp2018.
`scripts/resplit_ml.py` shows how MovieLens-100k was re-split per user.

## Quick start (smoke test)

```bash
python scripts/run_one.py --dataset lastfm --model mf
```

Expected: `cos_mean ≥ 0.97` for LastFM/MF.

## Attack comparison

Run several attacks per `(dataset, model)`; all attacks share one FedAvg warmup:

```bash
python scripts/run_all.py \
    --datasets lastfm ml delicious \
    --models mf lightgcn ncf \
    --attacks paired_probe single_probe dlg invert_grad lti raifle \
    --out results/attack_comp \
    [--skip-existing]

python scripts/summarize_attacks.py --runs results/attack_comp --out results/_attack_comparison.md
```

For a detached full 2-GPU sweep, see `scripts/launch_all.sh` and
`scripts/launch_attack_comparison.sh` (set `FL_CONDA_ACTIVATE` to your env-activation
command if the environment is not already active).

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
python scripts/summarize.py    --runs results/run0 results/run1 --out results/_summary.md
python scripts/summarize_he.py --baseline results/run0 --he-root results/he_ablation --out results/_he_summary.md
python scripts/save_wins.py
```

The `scripts/build_table_*.py` and `scripts/build_figure_tradeoff.py` helpers
regenerate the LaTeX tables / figure reported in the paper from the `results/` artifacts.

## Output convention

Each `(dataset, model[, attack])` run writes to its `--out` directory:

- `<dataset>_<model>[_<attack>].md` — human-readable report
- `<dataset>_<model>[_<attack>].json` — machine-readable metrics
- `<dataset>_<model>.npz` — `U_hat`, `true_U`, `G`, `A`, `cos`

## Tests

```bash
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

The attack-phase optimizer must be plain SGD (`weight_decay=0`): the recovery math
assumes `Δθ = −η·∇L`. Warmup may use Adam for faster ranking convergence.
`configs/he.yaml` is the HE-enabled variant (`he.enabled: true`).
