#!/usr/bin/env bash
# Buffer-style name: replay 0.5 -> p0p5, stale 0.01 -> st0p01, sampler random.
REPLAY_P=0.5
PLR_STALE_COEF=0.01
# From repo:  bash Work/to_run/sh/p0p5_srandom_st0p01_seeds_1_to_5.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORK_DIR" || exit 1

for seed in 1 2 3 4 5; do
  echo "========== PLR seed=$seed replay_p=$REPLAY_P stale=$PLR_STALE_COEF =========="
  python policy/ppo.py \
    --seed "$seed" \
    --replay-resample-prob "$REPLAY_P" \
    --plr-stale-coef "$PLR_STALE_COEF" \
    --sampler-type random \
    || true
done

echo "========== p0p5_srandom_st0p01_seeds_1_to_5 finished =========="
