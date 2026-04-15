# replay=-1 (no PLR), seeds 1-15. From repo root:  .\Work\to_run\ps1\pn1_seeds_1_to_15.ps1

$WorkDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $WorkDir

foreach ($seed in 1..15) {
    Write-Host "========== replay=-1 seed=$seed =========="
    python policy/ppo.py --seed $seed --replay-resample-prob -1
}

Write-Host "========== pn1_seeds_1_to_15 finished =========="
