#!/bin/bash
# v4 experiment chain: run job lists one after another on ONE GPU, one job at a time.
# usage: run_v4_chain.sh GPU WAIT_PID jobfile1 [jobfile2 ...]   (WAIT_PID=0 to start at once;
#        otherwise the chain starts when process WAIT_PID has exited)
# Job-file paths are taken as given (relative to the current directory); the jobs themselves run
# from the repository root (see sweep.py). Logs go to <repo>/logs/sweep/.
GPU=$1; WAIT=$2; shift 2
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=python
LOG=$ROOT/logs/sweep
mkdir -p "$LOG"
if [ "$WAIT" != "0" ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 30; done; fi
for jf in "$@"; do
  name=$(basename "$jf" .json)
  echo "[$(date +%H:%M:%S)] chain gpu$GPU start $name" >> "$LOG/v4_chain_gpu$GPU.out"
  $PY "$ROOT/scripts/v4/sweep.py" --jobs "$jf" --gpus $GPU >> "$LOG/${name}_gpu$GPU.out" 2>&1
  echo "[$(date +%H:%M:%S)] chain gpu$GPU done  $name" >> "$LOG/v4_chain_gpu$GPU.out"
done
echo "[$(date +%H:%M:%S)] chain gpu$GPU ALL DONE" >> "$LOG/v4_chain_gpu$GPU.out"
