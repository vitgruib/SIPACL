#!/usr/bin/env bash
# Names match buffer dir prefix: p<replay>_s<sampler>_st<stale>_… (see policy/ppo.py _float_slug).
# replay=-1 -> pn1; PLR off; seeds 1–5; sampler random; stale coef 0 (default when unused).
# From repo:  bash Work/to_run/sh/pn1_srandom_st0_seeds_1_to_5.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORK_DIR" || exit 1

for seed in 1 2 3 4 5; do
  echo "========== no PLR seed=$seed =========="
  python policy/ppo.py --seed "$seed" --replay-resample-prob -1 --plr-stale-coef 0 --sampler-type random || true
done

echo "========== pn1_srandom_st0_seeds_1_to_5 finished =========="
