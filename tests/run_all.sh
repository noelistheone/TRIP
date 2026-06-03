#!/bin/bash
# Run all unit tests sequentially.
# Assumes the Python environment (see README) is already active.
set -e
cd "$(dirname "$0")/.."

echo "=== test_allocator ==="
python tests/test_allocator.py

echo "=== test_freeze_restore ==="
CUDA_VISIBLE_DEVICES= python tests/test_freeze_restore.py

echo "=== test_mf_toy ==="
python tests/test_mf_toy.py

echo
echo "All tests passed ✓"
