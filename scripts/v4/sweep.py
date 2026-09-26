#!/usr/bin/env python
"""Two-GPU job-queue runner for the TRIP experiment matrix.

Jobs are shell commands tagged with a name; each worker owns one GPU and pulls
the next unfinished job from a shared queue. A job is 'done' when its sentinel
file exists, so the sweep is restartable and `--skip-existing` is implicit.

Usage (from the repository root):
    python scripts/v4/sweep.py --jobs scripts/v4/jobs/v4_tb.json [--gpus 0 1] [--dry-run]

A job list is a JSON list of {"name":..., "cmd":..., "done":<sentinel path>}.
`cmd` runs with the repository root as working directory, using the `python` of
the environment that started sweep.py, with CUDA_VISIBLE_DEVICES set to the
worker's GPU and FL_DATA_ROOT exported. A relative `done` path is taken relative
to the repository root (the job's working directory).

Paths: repository root = parent of scripts/; data root FL_DATA_ROOT (default
<repo>/data_local); logs, cross-process claim files and the final status file
under <repo>/logs/sweep/. SWEEP_MIN_FREE_MIB (default 4000) is the free-memory
gate described in worker().
"""
from __future__ import annotations
import argparse, json, os, queue, shlex, subprocess, threading, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("FL_DATA_ROOT") or REPO / "data_local")
LOGS = REPO / "logs" / "sweep"
ENV_PRE = (f"cd {shlex.quote(str(REPO))} && export FL_DATA_ROOT={shlex.quote(str(DATA_ROOT))} && "
           "export PYTHONUNBUFFERED=1 && ")

MIN_FREE_MIB = int(os.environ.get("SWEEP_MIN_FREE_MIB", "4000"))


def _free_mib(gpu: int):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                              "-i", str(gpu)], capture_output=True, text=True, timeout=30).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def _done_path(done):
    """Sentinel path of a job; relative paths are relative to the repository root."""
    if not done:
        return None
    p = Path(done)
    return p if p.is_absolute() else REPO / p


lock = threading.Lock()
state = {}


def log(msg):
    with lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def worker(gpu: int, q: "queue.Queue", dry: bool):
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        name, cmd, done = job["name"], job["cmd"], _done_path(job.get("done"))
        if done and done.exists():
            log(f"gpu{gpu} SKIP {name}")
            state[name] = "skip"
            q.task_done(); continue
        # Cross-process claim so several sweep.py instances can share one job
        # file (e.g. one per GPU, started at different times).
        claim = LOGS / "claims" / f"{name}.claim"
        claim.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"gpu{gpu} pid{os.getpid()}\n".encode()); os.close(fd)
        except FileExistsError:
            log(f"gpu{gpu} CLAIMED-BY-OTHER {name}")
            state[name] = "claimed-elsewhere"
            q.task_done(); continue
        logf = LOGS / f"{name}.log"
        logf.parent.mkdir(parents=True, exist_ok=True)
        full = ENV_PRE + f"CUDA_VISIBLE_DEVICES={gpu} " + cmd
        # Memory gate: another process may hold most of the card's memory, so wait until at
        # least MIN_FREE_MIB are free before starting (a running job cannot yield mid-run).
        while True:
            free = _free_mib(gpu)
            if free is None or free >= MIN_FREE_MIB:
                break
            log(f"gpu{gpu} WAIT {name}: {free} MiB free < {MIN_FREE_MIB}")
            time.sleep(60)
        log(f"gpu{gpu} START {name}")
        if dry:
            log(f"    {full}")
            state[name] = "dry"; q.task_done(); continue
        t0 = time.time()
        with open(logf, "w") as f:
            f.write(f"# {full}\n\n"); f.flush()
            rc = subprocess.call(["bash", "-c", full], stdout=f, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        ok = (rc == 0) and (not done or done.exists())
        if not ok:
            try: claim.unlink()
            except OSError: pass
        state[name] = "ok" if ok else f"FAIL(rc={rc})"
        log(f"gpu{gpu} {'OK  ' if ok else 'FAIL'} {name}  {dt/60:.1f} min  -> {logf}")
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--per-gpu", type=int, default=1, help="concurrent jobs per GPU")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="walk the job list back to front")
    a = ap.parse_args()

    jobs = json.loads(Path(a.jobs).read_text())
    if a.reverse:
        jobs = jobs[::-1]
    q: "queue.Queue" = queue.Queue()
    for j in jobs:
        q.put(j)
    log(f"{len(jobs)} jobs, gpus={a.gpus} x{a.per_gpu}")

    threads = []
    for g in a.gpus:
        for _ in range(a.per_gpu):
            t = threading.Thread(target=worker, args=(g, q, a.dry_run), daemon=True)
            t.start(); threads.append(t)
    for t in threads:
        t.join()

    LOGS.mkdir(parents=True, exist_ok=True)
    status = LOGS / f"{Path(a.jobs).stem}.status.gpu{'-'.join(map(str, a.gpus))}.json"
    status.write_text(json.dumps(state, indent=2))
    bad = {k: v for k, v in state.items() if v not in ("ok", "skip", "dry")}
    log(f"DONE  ok={sum(1 for v in state.values() if v=='ok')} "
        f"skip={sum(1 for v in state.values() if v=='skip')} fail={len(bad)}  status -> {status}")
    if bad:
        log("FAILED: " + ", ".join(bad))


if __name__ == "__main__":
    main()
