#!/usr/bin/env bash
# Buffer-style name: p0p5_srandom_st0p01; seeds 6–10.
REPLAY_P=0.5
PLR_STALE_COEF=0.01
# From repo:  bash Work/to_run/sh/p0p5_srandom_st0p01_seeds_6_to_10.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORK_DIR" || exit 1

for seed in 6 7 8 9 10; do
  echo "========== PLR seed=$seed replay_p=$REPLAY_P stale=$PLR_STALE_COEF =========="
  python policy/ppo.py \
    --seed "$seed" \
    --replay-resample-prob "$REPLAY_P" \
    --plr-stale-coef "$PLR_STALE_COEF" \
    --sampler-type random \
    || true
done

echo "========== p0p5_srandom_st0p01_seeds_6_to_10 finished =========="
