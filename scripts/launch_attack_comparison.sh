#!/bin/bash
# Attack-comparison sweep: 10 datasets × 3 models × 4 attacks = 120 runs.
# Shares one warmup per (dataset, model) across all 4 attacks → compute ≈ main
# sweep + DLG/IG per-user optimization overhead.
#
# All processes are fully detached (setsid + disown) so they survive SSH
# disconnect. PIDs in results/pids/attack_*.pid; logs in results/attack_gpu{0,1}.log.
set -euo pipefail

# Repo root = parent of this script's directory.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p results/pids
rm -rf results/attack_comp_gpu0 results/attack_comp_gpu1 2>/dev/null || true
mkdir -p results/attack_comp_gpu0 results/attack_comp_gpu1

ATTACKS="paired_probe single_probe dlg invert_grad"
# Optional environment-activation command for the detached shells; leave empty
# if your Python environment is already active. Data root defaults to <repo>/data_local.
CONDA_ACTIVATE="${FL_CONDA_ACTIVATE:-}"
export FL_DATA_ROOT="${FL_DATA_ROOT:-$ROOT/data_local}"

# ---- GPU 0: datasets 1–5 ----
setsid bash -c "
  $CONDA_ACTIVATE
  cd $ROOT
  CUDA_VISIBLE_DEVICES=0 python -u scripts/run_all.py \
    --datasets lastfm ml delicious douban-book amazon-beauty \
    --attacks $ATTACKS \
    --out results/attack_comp_gpu0 \
    > results/attack_gpu0.log 2>&1
  echo '[attack_gpu0 all done]' >> results/attack_gpu0.log
" < /dev/null > /dev/null 2>&1 &
GPU0_PID=$!
echo "$GPU0_PID" > results/pids/attack_gpu0.pid
disown
echo "GPU 0 launched: PID $GPU0_PID (5 datasets × 3 models × 4 attacks)"

# ---- GPU 1: datasets 6–10 ----
setsid bash -c "
  $CONDA_ACTIVATE
  cd $ROOT
  CUDA_VISIBLE_DEVICES=1 python -u scripts/run_all.py \
    --datasets amazon-book amazon-kindle gowalla iFashion yelp2018 \
    --attacks $ATTACKS \
    --out results/attack_comp_gpu1 \
    > results/attack_gpu1.log 2>&1
  echo '[attack_gpu1 all done]' >> results/attack_gpu1.log
" < /dev/null > /dev/null 2>&1 &
GPU1_PID=$!
echo "$GPU1_PID" > results/pids/attack_gpu1.pid
disown
echo "GPU 1 launched: PID $GPU1_PID (5 datasets × 3 models × 4 attacks)"

echo
echo "Both runners detached. Safe to close SSH."
echo "Monitor:   tail -f results/attack_gpu{0,1}.log"
echo "Progress:  grep -cE '^\\[OK\\]' results/attack_gpu*.log"
echo "Aggregate: python scripts/summarize_attacks.py --runs results/attack_comp_gpu0 results/attack_comp_gpu1 --out results/_attack_comparison.md"
