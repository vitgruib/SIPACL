# pn1 (replay off); seeds 6-10.
# From repo root:  .\Work\to_run\ps1\pn1_srandom_st0_seeds_6_to_10.ps1

$WorkDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $WorkDir

foreach ($seed in 6..10) {
    Write-Host "========== no PLR seed=$seed =========="
    python policy/ppo.py --seed $seed --replay-resample-prob -1 --plr-stale-coef 0 --sampler-type random
}

Write-Host "========== pn1_srandom_st0_seeds_6_to_10 finished =========="
