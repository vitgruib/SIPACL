# Five training runs with prioritized level replay ON.
# Hardcoded hyperparameters (edit here if needed):
$ReplayP = 0.5          # buffer resample probability (0-1)
$PlrStaleCoef = 0.01   # staleness weight; effective score = LP * (1 + coef * episode_staleness)
#
# From Work/:  .\run_plr_5x.ps1

Set-Location $PSScriptRoot

foreach ($seed in 1..5) {
    Write-Host "========== PLR seed=$seed replay_p=$ReplayP stale=$PlrStaleCoef =========="
    python policy/ppo.py `
        --seed $seed `
        --replay-resample-prob $ReplayP `
        --plr-stale-coef $PlrStaleCoef `
        --sampler-type random
}

Write-Host "========== run_plr_5x finished =========="
