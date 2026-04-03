#!/usr/bin/env bash
# p<replay>_s<sampler>_st<stale>_seeds_* — replay=-1 (pn1), seeds 6–10.
# From repo:  bash Work/to_run/sh/pn1_srandom_st0_seeds_6_to_10.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORK_DIR" || exit 1

for seed in 6 7 8 9 10; do
  echo "========== no PLR seed=$seed =========="
  python policy/ppo.py --seed "$seed" --replay-resample-prob -1 --plr-stale-coef 0 --sampler-type random || true
done

echo "========== pn1_srandom_st0_seeds_6_to_10 finished =========="
