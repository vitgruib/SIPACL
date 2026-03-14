# Run policy/ppo.py for each (replay_resample_prob, sampler_type) combination.
# replay_resample_prob: -1, 0.25, 0.5, 0.75
# sampler_type: halton, random
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

foreach ($sampler in @("halton", "random")) {
    foreach ($p in @(-1, 0.25, 0.5, 0.75)) {
        Write-Host "========== replay_resample_prob=$p sampler_type=$sampler =========="
        python policy/ppo.py --replay-resample-prob $p --sampler-type $sampler
        # Continue with next combination even if this run failed
    }
}

Write-Host "========== sweep finished =========="
