# Five training runs with prioritized level replay OFF (always new scenes).
# replay_resample_prob=-1 disables PLR in MetaDriveEnv (see policy/custom_gym).
#
# From Work/:  .\run_no_plr_5x.ps1

Set-Location $PSScriptRoot

foreach ($seed in 1..5) {
    Write-Host "========== no PLR seed=$seed =========="
    python policy/ppo.py --seed $seed --replay-resample-prob -1 --sampler-type random
}

Write-Host "========== run_no_plr_5x finished =========="
