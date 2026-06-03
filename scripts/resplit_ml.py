#!/usr/bin/env python
"""Re-split MovieLens-100k into per-user train/test (same users in both files).

Reads the raw u.data file and:
- Re-indexes user/item IDs contiguously 0..N-1.
- For each user, randomly puts 20% of their interactions in test, 80% in train.
- Ensures every user appears in BOTH train.txt and test.txt (unless they have <2 items).

Writes to <FL_DATA_ROOT>/ml/{train,test}.txt. Other datasets live under
FL_DATA_ROOT via the loader's env var fallback.

Paths are configurable via environment variables:
  ML_RAW_PATH   path to the raw MovieLens-100k u.data file (required)
  FL_DATA_ROOT  output data root (default: data_local)
"""
from __future__ import annotations

import os
import random
from pathlib import Path


SRC = Path(os.environ.get("ML_RAW_PATH", "data_local/ml/u.data"))
DST_DIR = Path(os.environ.get("FL_DATA_ROOT", "data_local")) / "ml"
SEED = 42
TEST_FRAC = 0.2


def main():
    random.seed(SEED)
    DST_DIR.mkdir(parents=True, exist_ok=True)

    user_map: dict[str, int] = {}
    item_map: dict[str, int] = {}
    per_user: dict[int, list[int]] = {}

    with open(SRC) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            u_raw, i_raw = parts[0], parts[1]
            if u_raw not in user_map:
                user_map[u_raw] = len(user_map)
            if i_raw not in item_map:
                item_map[i_raw] = len(item_map)
            u, i = user_map[u_raw], item_map[i_raw]
            per_user.setdefault(u, []).append(i)

    n_users = len(user_map)
    n_items = len(item_map)
    print(f"ml: {n_users} users, {n_items} items, "
          f"{sum(len(v) for v in per_user.values())} interactions")

    train_lines, test_lines = [], []
    n_train_users = n_test_users = 0
    skipped = 0
    for u in sorted(per_user.keys()):
        items = per_user[u]
        # Dedup preserving order (raw file has potential duplicates)
        seen, uniq = set(), []
        for it in items:
            if it not in seen:
                seen.add(it)
                uniq.append(it)
        items = uniq
        if len(items) < 2:
            # Keep in train only, no test split possible.
            train_lines.append(f"{u} " + " ".join(str(x) for x in items))
            n_train_users += 1
            skipped += 1
            continue
        random.shuffle(items)
        n_test = max(1, int(len(items) * TEST_FRAC))
        test_items = items[:n_test]
        train_items = items[n_test:]
        if not train_items:
            train_items = test_items
            test_items = []
        if train_items:
            train_lines.append(f"{u} " + " ".join(str(x) for x in train_items))
            n_train_users += 1
        if test_items:
            test_lines.append(f"{u} " + " ".join(str(x) for x in test_items))
            n_test_users += 1

    (DST_DIR / "train.txt").write_text("\n".join(train_lines) + "\n")
    (DST_DIR / "test.txt").write_text("\n".join(test_lines) + "\n")

    print(f"  wrote {len(train_lines)} train lines, {len(test_lines)} test lines")
    print(f"  train users: {n_train_users}, test users: {n_test_users}, "
          f"skipped-single-item: {skipped}")
    print(f"  overlap = {len(set(range(n_train_users)) & set(range(n_test_users)))}")
    print(f"  → {DST_DIR}/")


if __name__ == "__main__":
    main()
