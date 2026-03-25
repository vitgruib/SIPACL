#Requires -Version 7.0
<#
  Grid over (replay_resample_prob, sampler_type, plr_stale_coef); runs multiple policy/ppo.py jobs in parallel.
  Requires PowerShell 7+ (ForEach-Object -Parallel). Install: winget install Microsoft.PowerShell

  From Work/:  pwsh ./run_resample_sweep.ps1
               pwsh ./run_resample_sweep.ps1 -MaxParallel 8

  Parameters:
    -MaxParallel  How many ppo.py processes at once (default 4). Tune to CPU/GPU memory.

  Unique buffer dirs: ppo.py defaults (Work/buffer_runs/p<...>_s<...>_st<...>).
#>
param(
    [int]$MaxParallel = 4
)

$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }

$combos = foreach ($sampler in @("halton", "random")) {
    foreach ($p in @(-1, 0.25, 0.5, 0.75)) {
        foreach ($stale in @(0, 0.001, 0.01, 0.05)) {
            [pscustomobject]@{
                P       = $p
                Sampler = $sampler
                Stale   = $stale
            }
        }
    }
}

$combos | ForEach-Object -Parallel {
    $c = $_
    Set-Location $using:ScriptDir
    Write-Host "========== replay_resample_prob=$($c.P) sampler_type=$($c.Sampler) plr_stale_coef=$($c.Stale) =========="
    & python policy/ppo.py --replay-resample-prob $c.P --sampler-type $c.Sampler --plr-stale-coef $c.Stale
    if (-not $?) {
        Write-Host "Warning: run failed (continuing) p=$($c.P) sampler=$($c.Sampler) stale=$($c.Stale)"
    }
} -ThrottleLimit $MaxParallel

Write-Host "========== sweep finished =========="
