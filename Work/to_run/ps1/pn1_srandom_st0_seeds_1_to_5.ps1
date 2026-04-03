# p<replay>_s<sampler>_st<stale>_seeds_* — pn1 = replay -1; seeds 1-5.
# From repo:  .\Work\to_run\ps1\pn1_srandom_st0_seeds_1_to_5.ps1  (cwd can be anywhere if you use full path)

$WorkDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $WorkDir

foreach ($seed in 1..5) {
    Write-Host "========== no PLR seed=$seed =========="
    python policy/ppo.py --seed $seed --replay-resample-prob -1 --plr-stale-coef 0 --sampler-type random
}

Write-Host "========== pn1_srandom_st0_seeds_1_to_5 finished =========="
