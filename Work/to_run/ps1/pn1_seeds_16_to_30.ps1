# replay=-1 (no PLR), seeds 16-30. From repo root:  .\Work\to_run\ps1\pn1_seeds_16_to_30.ps1

$WorkDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $WorkDir

foreach ($seed in 16..30) {
    Write-Host "========== replay=-1 seed=$seed =========="
    python policy/ppo.py --seed $seed --replay-resample-prob -1
}

Write-Host "========== pn1_seeds_16_to_30 finished =========="
