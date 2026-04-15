# replay=0.5 (PLR), seeds 16-30; other flags use Args defaults in policy/ppo.py.
# From repo root:  .\Work\to_run\ps1\p0p5_seeds_16_to_30.ps1

$WorkDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $WorkDir

foreach ($seed in 16..30) {
    Write-Host "========== replay=0.5 seed=$seed =========="
    python policy/ppo.py --seed $seed --replay-resample-prob 0.5
}

Write-Host "========== p0p5_seeds_16_to_30 finished =========="
