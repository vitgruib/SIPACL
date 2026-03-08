#!/usr/bin/env bash
# Run policy/ppo.py for replay_resample_prob = -1, 0.25, 0.5, 0.75
# Only --replay-resample-prob is overridden; all other ppo.py defaults are used.
# (No set -e: we keep running the remaining probs even if one run fails.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for p in -1 0.25 0.5 0.75; do
  echo "========== replay_resample_prob=$p =========="
  python policy/ppo.py --replay-resample-prob "$p" || true
done

echo "========== sweep finished =========="
