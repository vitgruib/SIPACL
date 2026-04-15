# replay=0.5 (PLR), seeds 1-15; other flags use Args defaults in policy/ppo.py.
# From repo root:  .\Work\to_run\ps1\p0p5_seeds_1_to_15.ps1

$WorkDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $WorkDir

foreach ($seed in 1..15) {
    Write-Host "========== replay=0.5 seed=$seed =========="
    python policy/ppo.py --seed $seed --replay-resample-prob 0.5
}

Write-Host "========== p0p5_seeds_1_to_15 finished =========="
