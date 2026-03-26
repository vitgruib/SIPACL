#!/usr/bin/env bash
# Part 2 of 4 — sequential runs (HPC-friendly: no parallel library).
# random × replay_resample_prob=0.25 × all plr_stale_coef (3 runs).
# Full sweep = part1 + part2 + part3 + part4 (12 runs; stale ∈ {0, 0.01, 0.05}).
#
# From Work/:  bash run_resample_sweep_part2.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

for stale in 0 0.01 0.05; do
  echo "========== part2/4 random p=0.25 plr_stale_coef=$stale =========="
  python policy/ppo.py --replay-resample-prob 0.25 --sampler-type random --plr-stale-coef "$stale" || true
done

echo "========== part2 finished =========="
