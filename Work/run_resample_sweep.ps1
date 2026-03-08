# Run policy/ppo.py for replay_resample_prob = -1, 0.25, 0.5, 0.75
# Only --replay-resample-prob is overridden; all other ppo.py defaults are used.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

foreach ($p in @(-1, 0.25, 0.5, 0.75)) {
    Write-Host "========== replay_resample_prob=$p =========="
    python policy/ppo.py --replay-resample-prob $p
    # Continue with next value even if this run failed
}

Write-Host "========== sweep finished =========="
