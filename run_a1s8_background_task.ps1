$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root

$python = 'C:\Users\12238\.conda\envs\pytorch_cpu\python.exe'
$runName = 'run67-m2w-120-500-background-a1s8-m00-preserve'
$runDir = Join-Path $root ('checkpoints_fsia\' + $runName)
$taskLogDir = Join-Path $root 'a1s8_task_logs'
$statusPath = Join-Path $taskLogDir 'a1s8_task_status.json'
$stdoutPath = Join-Path $taskLogDir 'a1s8_task_stdout.log'
$stderrPath = Join-Path $taskLogDir 'a1s8_task_stderr.log'
New-Item -ItemType Directory -Force -Path $taskLogDir | Out-Null

function Write-Status([hashtable] $State) {
    $State.utc = [DateTime]::UtcNow.ToString('o')
    $State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

if (Test-Path -LiteralPath (Join-Path $runDir 'training_summary.json')) {
    Write-Status @{ status = 'already_complete'; run_name = $runName; exit_code = 0 }
    exit 0
}

$statePath = Join-Path $runDir 'last_training_state.pth'
$arguments = @(
    'main_fsia.py',
    '--run-name', $runName,
    '--model-domain', '120-500-obs-200-500',
    '--date-blocked-split',
    '--date-split-manifest', 'isr_validation_outputs\run66-modelonly-m0\date_split_manifest.json',
    '--background-only',
    '--background-condition-adapter', 'maglat2_smooth_v1',
    '--background-training-scope', 'condition_adapter_only_dual_maglat_v1',
    '--background-adapter-seed-checkpoint', 'checkpoints_fsia\run67-m2w-120-500-background-stratum-bias-a1s\best_background_model.pth',
    '--background-adapter-rng-seed', '42',
    '--background-adapter-gate-semantics', 'abs_sinI_025_050_smoothstep_v1',
    '--background-sampling-semantics', 'a1s8_mixed_gate_aware_v1',
    '--background-fixed-updates-per-epoch', '960',
    '--background-anchor-semantics', 'a1s_output_trust_region_v1',
    '--background-gate-balance-semantics', 'effective_gate_mass_clipped_v1',
    '--background-source-mix-schedule', 'fy3_cosmic1_v1',
    '--background-mix-legacy-fraction', '0.75',
    '--background-mix-balanced-fraction', '0.25',
    '--background-anchor-gradient-target', '0.10',
    '--background-anchor-calibration-batches', '20',
    '--background-anchor-warmup-epochs', '1',
    '--background-exposure-weight-cap', '3.0',
    '--background-gradient-ratio-min', '0.25',
    '--background-gradient-ratio-max', '4.0',
    '--background-low-effective-mass-min', '0.25',
    '--background-low-effective-mass-max', '0.35',
    '--background-high-effective-mass-min', '0.65',
    '--background-high-effective-mass-max', '0.75',
    '--background-adapter-residual-cap', '0.1',
    '--background-checkpoint-selection', 'bias-constrained-v1',
    '--background-selection-baseline', 'checkpoints_fsia\run66-m2w-120-500-obs-200-500-gate-off\background_development_replay.json',
    '--background-stratum-bias-gradient-target', '0.015',
    '--background-epochs', '5',
    '--analysis-epochs', '10',
    '--skip-hash-verification'
)
if (Test-Path -LiteralPath $statePath) {
    $arguments += @('--resume', $statePath)
}

Write-Status @{ status = 'running'; run_name = $runName; pid = $PID; resume = (Test-Path -LiteralPath $statePath) }
try {
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    Write-Status @{ status = 'running'; run_name = $runName; pid = $process.Id; resume = (Test-Path -LiteralPath $statePath) }
    $process.WaitForExit()
    $exitCode = $process.ExitCode
    if ($exitCode -eq 0) {
        Write-Status @{ status = 'completed'; run_name = $runName; exit_code = 0 }
    } else {
        Write-Status @{ status = 'failed'; run_name = $runName; exit_code = $exitCode }
        exit $exitCode
    }
} catch {
    Write-Status @{ status = 'failed'; run_name = $runName; error = $_.Exception.ToString(); exit_code = 1 }
    throw
}
