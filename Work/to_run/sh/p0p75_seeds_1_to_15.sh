#!/usr/bin/env bash
# replay=0.75 (PLR), seeds 1–15; other flags use Args defaults in policy/ppo.py.
# From repo root:  bash Work/to_run/sh/p0p75_seeds_1_to_15.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORK_DIR" || exit 1

for seed in $(seq 1 15); do
  echo "========== replay=0.75 seed=$seed =========="
  python policy/ppo.py --seed "$seed" --replay-resample-prob 0.75 || true
done

echo "========== p0p75_seeds_1_to_15 finished =========="
