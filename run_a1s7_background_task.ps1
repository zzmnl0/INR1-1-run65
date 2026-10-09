$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root

$python = 'C:\Users\12238\.conda\envs\pytorch_cpu\python.exe'
$runName = 'run67-m2w-120-500-background-a1s7-m00-b'
$runDir = Join-Path $root ('checkpoints_fsia\' + $runName)
$taskLogDir = Join-Path $root 'a1s7_task_logs'
$statusPath = Join-Path $taskLogDir 'a1s7_task_status.json'
$stdoutPath = Join-Path $taskLogDir 'a1s7_task_stdout.log'
$stderrPath = Join-Path $taskLogDir 'a1s7_task_stderr.log'

New-Item -ItemType Directory -Force -Path $taskLogDir | Out-Null

function Write-Status {
    param([hashtable]$State)
    $State.utc = [DateTime]::UtcNow.ToString('o')
    $State | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

try {
    if (Test-Path -LiteralPath (Join-Path $runDir 'training_summary.json')) {
        Write-Status @{ status = 'already_complete'; run_name = $runName }
        exit 0
    }

    $statePath = Join-Path $runDir 'last_training_state.pth'
    $resume = Test-Path -LiteralPath $statePath
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
        '--background-adapter-residual-cap', '0.1',
        '--background-checkpoint-selection', 'bias-constrained-v1',
        '--background-selection-baseline', 'checkpoints_fsia\run66-m2w-120-500-obs-200-500-gate-off\background_development_replay.json',
        '--background-stratum-bias-gradient-target', '0.05',
        '--background-epochs', '5',
        '--skip-hash-verification'
    )
    if ($resume) {
        $arguments += @('--resume', 'auto')
    }

    Write-Status @{ status = 'running'; run_name = $runName; pid = $PID; resumed = $resume }
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    $process.WaitForExit()
    $exitCode = $process.ExitCode
    if ($exitCode -ne 0) {
        throw "Training process exited with code $exitCode"
    }
    Write-Status @{ status = 'completed'; run_name = $runName; exit_code = $exitCode; resumed = $resume }
    exit 0
} catch {
    $message = $_.Exception.ToString()
    $message | Out-File -LiteralPath $stderrPath -Encoding UTF8 -Append
    try {
        Write-Status @{ status = 'failed'; run_name = $runName; error = $message }
    } catch { }
    exit 1
}
