#!/bin/bash
# Launch the full re-run: 30 main-sweep runs across 2 GPUs + 36 HE ablation runs.
# All processes are fully detached (setsid + nohup) so they survive SSH disconnect.
#
# Pipeline:
#   GPU 0: main sweep group A (5 datasets × 3 models = 15 runs), then HE ablation
#   GPU 1: main sweep group B (5 datasets × 3 models = 15 runs)
#
# PIDs are written to results/pids/*.pid for later inspection.
# Logs land in results/gpu{0,1}.log and results/he_ablation.log.
set -euo pipefail

# Repo root = parent of this script's directory.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p results/pids
rm -rf results/gpu0 results/gpu1 results/he_ablation results/smoke 2>/dev/null || true
mkdir -p results/gpu0 results/gpu1 results/he_ablation

# Optional: command to activate your Python environment inside the detached
# shells (e.g. "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fl_env").
# Leave empty if the environment is already active in your shell.
CONDA_ACTIVATE="${FL_CONDA_ACTIVATE:-}"
# Data root holding <dataset>/{train,test}.txt (default: <repo>/data_local).
export FL_DATA_ROOT="${FL_DATA_ROOT:-$ROOT/data_local}"

# ---- GPU 0: main group A → then HE ablation ----
setsid bash -c "
  $CONDA_ACTIVATE
  cd $ROOT
  CUDA_VISIBLE_DEVICES=0 python -u scripts/run_all.py \
    --datasets lastfm ml delicious douban-book amazon-beauty \
    --out results/gpu0 \
    > results/gpu0.log 2>&1
  echo '[gpu0 main done, starting HE ablation]' >> results/gpu0.log
  CUDA_VISIBLE_DEVICES=0 python -u scripts/run_he_ablation.py \
    --noise-levels 1e-10 1e-8 1e-6 1e-4 \
    --out-root results/he_ablation \
    > results/he_ablation.log 2>&1
  echo '[gpu0 all done]' >> results/gpu0.log
" < /dev/null > /dev/null 2>&1 &
GPU0_PID=$!
echo "$GPU0_PID" > results/pids/gpu0.pid
disown
echo "GPU 0 launched: PID $GPU0_PID (main group A → HE ablation)"

# ---- GPU 1: main group B ----
setsid bash -c "
  $CONDA_ACTIVATE
  cd $ROOT
  CUDA_VISIBLE_DEVICES=1 python -u scripts/run_all.py \
    --datasets amazon-book amazon-kindle gowalla iFashion yelp2018 \
    --out results/gpu1 \
    > results/gpu1.log 2>&1
  echo '[gpu1 all done]' >> results/gpu1.log
" < /dev/null > /dev/null 2>&1 &
GPU1_PID=$!
echo "$GPU1_PID" > results/pids/gpu1.pid
disown
echo "GPU 1 launched: PID $GPU1_PID (main group B)"

echo
echo "Both runners detached (setsid + disown). Safe to close SSH."
echo "Monitor with:"
echo "  tail -f results/gpu0.log results/gpu1.log results/he_ablation.log"
echo "Check completion with:"
echo "  grep -E '\\[OK\\]|\\[FAIL\\]|\\[.*all done\\]' results/*.log"
echo "After ALL jobs finish:"
echo "  python scripts/summarize.py --runs results/gpu0 results/gpu1 --out results/_summary.md"
echo "  python scripts/summarize_he.py --baseline results/gpu0 --he-root results/he_ablation --out results/_he_summary.md"
