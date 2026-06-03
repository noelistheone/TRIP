from __future__ import annotations

import os
import random
import subprocess

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_gpu(default: int | None = None) -> torch.device:
    """Pick the least-utilized GPU. Falls back to CPU if no CUDA."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if default is not None:
        return torch.device(f"cuda:{default}")
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env != "":
        # Already masked — just use cuda:0 of whatever is visible
        return torch.device("cuda:0")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip().splitlines()
        mems = [int(x) for x in out]
        idx = int(np.argmin(mems))
        return torch.device(f"cuda:{idx}")
    except Exception:
        return torch.device("cuda:0")


def clone_state_dict(sd):
    return {k: v.detach().clone() for k, v in sd.items()}
