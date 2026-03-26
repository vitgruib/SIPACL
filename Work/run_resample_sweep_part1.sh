#!/usr/bin/env bash
# Part 1 of 4: random sampler, replay_resample_prob=-1, three plr_stale_coef values.

set -u
# Error on unset variables (catches typos).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Absolute path to the directory containing this script (Work/).

cd "$SCRIPT_DIR" || exit 1
# Run from Work/ so policy/ppo.py resolves; fail if cd fails.

for stale in 0 0.01 0.05; do
  # Loop over plr_stale_coef values for this sweep part (3 runs).
  echo "========== part1/4 random p=-1 plr_stale_coef=$stale =========="
  python policy/ppo.py --replay-resample-prob -1 --sampler-type random --plr-stale-coef "$stale" || true
  # Train PPO; || true continues if python exits with an error.
done

echo "========== part1 finished =========="
