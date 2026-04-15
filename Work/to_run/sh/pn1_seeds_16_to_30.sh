#!/usr/bin/env bash
# replay=-1 (no PLR), seeds 16–30. From repo root:  bash Work/to_run/sh/pn1_seeds_16_to_30.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORK_DIR" || exit 1

for seed in $(seq 16 30); do
  echo "========== replay=-1 seed=$seed =========="
  python policy/ppo.py --seed "$seed" --replay-resample-prob -1 || true
done

echo "========== pn1_seeds_16_to_30 finished =========="
