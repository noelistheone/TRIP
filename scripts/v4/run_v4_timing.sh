#!/bin/bash
# E6 dedicated timing: start only when BOTH v4 chains have finished, so that no other job of this
# experiment set is running, then run the timing jobs one at a time on this GPU. The GPU-0 instance
# also records every GPU process on the machine every 5 min (logs/sweep/v4_timing_machine_state.log),
# to document that the card was otherwise idle while the timings were taken.
# usage: run_v4_timing.sh GPU CHAIN_PID0 CHAIN_PID1
GPU=$1; P0=$2; P1=$3
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOG=$ROOT/logs/sweep
mkdir -p "$LOG"
while kill -0 "$P0" 2>/dev/null || kill -0 "$P1" 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] timing gpu$GPU start" >> "$LOG/v4_timing_chain.out"
if [ "$GPU" = "0" ]; then
  ( while true; do { date +%H:%M:%S; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader; uptime; } >> "$LOG/v4_timing_machine_state.log"; sleep 300; done ) &
  MON=$!
fi
python "$ROOT/scripts/v4/sweep.py" --jobs "$ROOT/scripts/v4/jobs/v4_timing_g$GPU.json" --gpus $GPU >> "$LOG/v4_timing_g${GPU}_gpu$GPU.out" 2>&1
echo "[$(date +%H:%M:%S)] timing gpu$GPU done" >> "$LOG/v4_timing_chain.out"
if [ "$GPU" = "0" ]; then
  while pgrep -f "jobs/v4_timing_g1.json" > /dev/null; do sleep 60; done
  kill $MON 2>/dev/null
fi
