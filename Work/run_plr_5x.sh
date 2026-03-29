#!/usr/bin/env bash
# Five training runs with prioritized level replay ON.
# Hardcoded hyperparameters (edit here if needed):
REPLAY_P=0.5          # buffer resample probability (0–1)
PLR_STALE_COEF=0.01   # staleness weight; effective score = LP * (1 + coef * episode_staleness)
#
# From Work/:  bash run_plr_5x.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

for seed in 1 2 3 4 5; do
  echo "========== PLR seed=$seed replay_p=$REPLAY_P stale=$PLR_STALE_COEF =========="
  python policy/ppo.py \
    --seed "$seed" \
    --replay-resample-prob "$REPLAY_P" \
    --plr-stale-coef "$PLR_STALE_COEF" \
    --sampler-type random \
    || true
done

echo "========== run_plr_5x finished =========="
