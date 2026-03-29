#!/usr/bin/env bash
# Five training runs with prioritized level replay OFF (always new scenes).
# replay_resample_prob=-1 disables PLR in MetaDriveEnv (see policy/custom_gym).
#
# From Work/:  bash run_no_plr_5x.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

for seed in 1 2 3 4 5; do
  echo "========== no PLR seed=$seed =========="
  python policy/ppo.py --seed "$seed" --replay-resample-prob -1 --sampler-type random || true
done

echo "========== run_no_plr_5x finished =========="
