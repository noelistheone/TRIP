"""Unit tests for SlidingWindowAllocator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fl.trip.allocator import SlidingWindowAllocator


def test_A_shape_and_row_sum():
    alloc = SlidingWindowAllocator(N=500, K=10, W=20, T_factor=3)
    assert alloc.T == 150
    A = alloc.build_A()
    assert A.shape == (150 * 10, 500)
    assert int(A.sum(axis=1).max()) == 20
    assert int(A.sum(axis=1).min()) == 20  # all rows hit exactly W users
    # Rank should be near N so pinv is well-conditioned.
    rank = np.linalg.matrix_rank(A)
    assert rank >= 450, f"rank={rank} too low"


def test_window_wraparound():
    alloc = SlidingWindowAllocator(N=100, K=5, W=10, T_factor=2)
    # first round, first probe: window starts at offset_0 = 0
    w = alloc.window(0, 0)
    assert w == list(range(0, 10))
    # Advancing rounds shifts by 1
    assert alloc.window(1, 0) == list(range(1, 11))


def test_touched_is_union():
    alloc = SlidingWindowAllocator(N=500, K=10, W=20, T_factor=3)
    t = 7
    u = set()
    for k in range(10):
        u.update(alloc.window(t, k))
    assert set(alloc.touched(t)) == u


if __name__ == "__main__":
    test_A_shape_and_row_sum()
    test_window_wraparound()
    test_touched_is_union()
    print("All allocator tests pass.")
