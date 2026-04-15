#!/usr/bin/env bash
# replay=0.5 (PLR), seeds 16–30; other flags use Args defaults in policy/ppo.py.
# From repo root:  bash Work/to_run/sh/p0p5_seeds_16_to_30.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORK_DIR" || exit 1

for seed in $(seq 16 30); do
  echo "========== replay=0.5 seed=$seed =========="
  python policy/ppo.py --seed "$seed" --replay-resample-prob 0.5 || true
done

echo "========== p0p5_seeds_16_to_30 finished =========="
