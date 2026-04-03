# p0p5_srandom_st0p01; seeds 6-10.
# From repo root:  .\Work\to_run\ps1\p0p5_srandom_st0p01_seeds_6_to_10.ps1
$ReplayP = 0.5
$PlrStaleCoef = 0.01

$WorkDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $WorkDir

foreach ($seed in 6..10) {
    Write-Host "========== PLR seed=$seed replay_p=$ReplayP stale=$PlrStaleCoef =========="
    python policy/ppo.py `
        --seed $seed `
        --replay-resample-prob $ReplayP `
        --plr-stale-coef $PlrStaleCoef `
        --sampler-type random
}

Write-Host "========== p0p5_srandom_st0p01_seeds_6_to_10 finished =========="
