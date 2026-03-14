#!/usr/bin/env bash
# Run policy/ppo.py for each (replay_resample_prob, sampler_type) combination.
# replay_resample_prob: -1, 0.25, 0.5, 0.75
# sampler_type: halton, random
# (No set -e: we keep running the remaining runs even if one fails.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for sampler in halton random; do
  for p in -1 0.25 0.5 0.75; do
    echo "========== replay_resample_prob=$p sampler_type=$sampler =========="
    python policy/ppo.py --replay-resample-prob "$p" --sampler-type "$sampler" || true
  done
done

echo "========== sweep finished =========="
