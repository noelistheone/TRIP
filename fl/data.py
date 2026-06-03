"""Dataset loading for Adap_tau_Denoise-format implicit-feedback files.

Each line in train.txt / test.txt: `uid item1 item2 ...` (whitespace-separated, 0-indexed).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Default data root (relative to the repo). Override with the FL_DATA_ROOT
# environment variable or by passing an explicit `root` to load_dataset().
# Datasets are expected at <root>/<name>/{train,test}.txt in Adap_tau_Denoise
# format (each line: `uid item1 item2 ...`).
DATA_ROOT_DEFAULT = "data_local"


@dataclass
class DatasetBundle:
    name: str
    n_users: int
    n_items: int
    train_pos: Dict[int, List[int]]
    test_pos: Dict[int, List[int]]
    train_user_ids: List[int]
    cold_start_test: bool = False
    _train_set: Dict[int, set] = field(default_factory=dict, repr=False)

    def train_items_of(self, uid: int) -> List[int]:
        return self.train_pos.get(uid, [])

    def in_train(self, uid: int, iid: int) -> bool:
        if uid not in self._train_set:
            self._train_set[uid] = set(self.train_pos.get(uid, []))
        return iid in self._train_set[uid]


def _parse(path: str) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            uid = int(parts[0])
            items = [int(x) for x in parts[1:]]
            if items:
                out[uid] = items
    return out


def load_dataset(name: str, root: str | None = None) -> DatasetBundle:
    root = root or os.environ.get("FL_DATA_ROOT", DATA_ROOT_DEFAULT)
    base = Path(root) / name
    train = _parse(str(base / "train.txt"))
    test = _parse(str(base / "test.txt"))

    max_uid = max([-1] + list(train.keys()) + list(test.keys()))
    max_iid = -1
    for d in (train, test):
        for items in d.values():
            if items:
                mi = max(items)
                if mi > max_iid:
                    max_iid = mi
    n_users = max_uid + 1
    n_items = max_iid + 1

    train_uids = sorted(train.keys())
    cold = len(set(train.keys()) & set(test.keys())) == 0 and len(test) > 0

    return DatasetBundle(
        name=name,
        n_users=n_users,
        n_items=n_items,
        train_pos=train,
        test_pos=test,
        train_user_ids=train_uids,
        cold_start_test=cold,
    )


def bpr_sample(uid: int, pos_items: List[int], n_real_items: int,
               n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Return triples array of shape [n_samples, 3] = [uid, pos, neg].

    Negatives are drawn uniformly from [0, n_real_items), rejecting user's own positives.
    n_real_items constrains the sampler to the true catalog so probe rows (appended afterward)
    never leak into BPR negatives.
    """
    if not pos_items:
        return np.zeros((0, 3), dtype=np.int64)
    pos_set = set(pos_items)
    pos_arr = np.asarray(pos_items, dtype=np.int64)
    pos_draw = pos_arr[rng.integers(0, len(pos_arr), size=n_samples)]
    negs = np.empty(n_samples, dtype=np.int64)
    k = 0
    # Vectorized batch rejection sampling
    attempts = 0
    while k < n_samples and attempts < 20:
        need = n_samples - k
        cand = rng.integers(0, n_real_items, size=need * 2)
        for c in cand:
            if c in pos_set:
                continue
            negs[k] = c
            k += 1
            if k == n_samples:
                break
        attempts += 1
    # Fallback: if still short, fill with any valid non-pos
    while k < n_samples:
        c = int(rng.integers(0, n_real_items))
        if c in pos_set:
            continue
        negs[k] = c
        k += 1
    triples = np.stack([np.full(n_samples, uid, dtype=np.int64), pos_draw, negs], axis=1)
    return triples
