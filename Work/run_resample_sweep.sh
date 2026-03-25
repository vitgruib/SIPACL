#!/usr/bin/env bash
# Grid over (replay_resample_prob, sampler_type, plr_stale_coef) using GNU parallel.
# From Work/:  J=4 bash run_resample_sweep.sh
# Requires: GNU parallel (e.g. apt install parallel, pacman -S parallel).
# Unique buffer dirs: ppo.py defaults (Work/buffer_runs/...).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
J="${J:-4}"

if ! command -v parallel >/dev/null 2>&1; then
  echo "GNU parallel is required. Install the 'parallel' package, or on Windows use: pwsh ./run_resample_sweep.ps1 (PowerShell 7+)." >&2
  exit 1
fi

export SCRIPT_DIR
# {1}=sampler (slowest), {2}=p, {3}=stale
parallel -j "$J" 'echo "========== replay_resample_prob={2} sampler_type={1} plr_stale_coef={3} =========="; cd "$SCRIPT_DIR" && python policy/ppo.py --replay-resample-prob {2} --sampler-type {1} --plr-stale-coef {3} || true' \
  ::: halton random ::: -1 0.25 0.5 0.75 ::: 0 0.001 0.01 0.05

echo "========== sweep finished =========="
