# p0p5_srandom_st0p01 — matches default buffer dir token pattern (policy/ppo.py).
# From repo root:  .\Work\to_run\ps1\p0p5_srandom_st0p01_seeds_1_to_5.ps1
$ReplayP = 0.5
$PlrStaleCoef = 0.01

$WorkDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $WorkDir

foreach ($seed in 1..5) {
    Write-Host "========== PLR seed=$seed replay_p=$ReplayP stale=$PlrStaleCoef =========="
    python policy/ppo.py `
        --seed $seed `
        --replay-resample-prob $ReplayP `
        --plr-stale-coef $PlrStaleCoef `
        --sampler-type random
}

Write-Host "========== p0p5_srandom_st0p01_seeds_1_to_5 finished =========="
