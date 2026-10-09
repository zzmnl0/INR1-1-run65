$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root

$python = 'C:\Users\12238\.conda\envs\pytorch_cpu\python.exe'
$runName = 'run67-m2w-120-500-background-m00-observation-error'
$runDir = Join-Path $root ('checkpoints_fsia\' + $runName)
$taskLogDir = Join-Path $root 'm00_observation_task_logs'
$statusPath = Join-Path $taskLogDir 'm00_observation_task_status.json'
$calibrationPath = Join-Path $root 'checkpoints_fsia\m00_observation_calibration_train_only.json'
$calibrationStdout = Join-Path $taskLogDir 'calibration_stdout.log'
$calibrationStderr = Join-Path $taskLogDir 'calibration_stderr.log'
$trainingStdout = Join-Path $taskLogDir 'training_stdout.log'
$trainingStderr = Join-Path $taskLogDir 'training_stderr.log'

New-Item -ItemType Directory -Force -Path $taskLogDir | Out-Null

function Write-Status {
    param([hashtable]$State)
    $State.utc = [DateTime]::UtcNow.ToString('o')
    $State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

try {
    if (Test-Path -LiteralPath (Join-Path $runDir 'training_summary.json')) {
        Write-Status @{ status = 'already_complete'; run_name = $runName; exit_code = 0 }
        exit 0
    }

    if (-not (Test-Path -LiteralPath $calibrationPath)) {
        Write-Status @{ status = 'calibrating'; run_name = $runName }
        $calibrationArgs = @(
            'prepare_m00_observation_calibration.py',
            '--output', $calibrationPath,
            '--date-split-manifest', 'isr_validation_outputs\run66-modelonly-m0\date_split_manifest.json'
        )
        $calibration = Start-Process -FilePath $python -ArgumentList $calibrationArgs `
            -WorkingDirectory $root -WindowStyle Hidden `
            -RedirectStandardOutput $calibrationStdout `
            -RedirectStandardError $calibrationStderr -PassThru
        Write-Status @{ status = 'calibrating'; run_name = $runName; pid = $calibration.Id }
        $calibration.WaitForExit()
        $calibration.Refresh()
        if ($calibration.ExitCode -ne 0 -and
            -not (Test-Path -LiteralPath $calibrationPath)) {
            throw "Calibration process exited with code $($calibration.ExitCode)"
        }
    }

    if (-not (Test-Path -LiteralPath $calibrationPath)) {
        throw 'Calibration process completed without its output JSON'
    }

    # Fail closed if a stale table or a different date contract is present.
    $split = Get-Content -Raw -Encoding UTF8 `
        (Join-Path $root 'isr_validation_outputs\run66-modelonly-m0\date_split_manifest.json') |
        ConvertFrom-Json
    $calibrationPayload = Get-Content -Raw -Encoding UTF8 $calibrationPath | ConvertFrom-Json
    $expectedTrain = @($split.partitions.train | ForEach-Object { [int]$_ })
    $actualTrain = @($calibrationPayload.train_dates | ForEach-Object { [int]$_ })
    $expectedDevelopment = @($split.partitions.development | ForEach-Object { [int]$_ })
    $expectedLocked = @($split.partitions.locked_test | ForEach-Object { [int]$_ })
    if ($expectedTrain.Count -ne 20 -or $expectedDevelopment.Count -ne 5 -or
        $expectedLocked.Count -ne 5 -or
        (@($expectedTrain + $expectedDevelopment + $expectedLocked) |
            Sort-Object -Unique).Count -ne 30) {
        throw 'A1-S date contract must be exactly 20 Train / 5 Development / 5 Locked-test over 30 days'
    }
    if (($expectedTrain -join ',') -ne ($actualTrain -join ',')) {
        throw 'Calibration Train dates do not match the active A1-S manifest'
    }
    $calibrationAlt = @($calibrationPayload.observation_alt_range_km |
        ForEach-Object { [double]$_ })
    if ($calibrationAlt.Count -ne 2 -or
        [math]::Abs($calibrationAlt[0] - 200.0) -gt 1e-6 -or
        [math]::Abs($calibrationAlt[1] - 500.0) -gt 1e-6) {
        throw 'Calibration altitude domain is not the strict 200-500 km target domain'
    }
    if ($calibrationPayload.locked_test -ne 'excluded' -or
        $calibrationPayload.train_only -ne $true) {
        throw 'Calibration is not marked Train-only with Locked-test excluded'
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
        '--background-epochs', '5',
        '--analysis-epochs', '10',
        '--background-checkpoint-selection', 'bias-constrained-v1',
        '--background-selection-baseline', 'checkpoints_fsia\run66-m2w-120-500-obs-200-500-gate-off\background_development_replay.json',
        '--m00-observation-calibration-semantics', 'm00_train_only_bias_r_v1',
        '--m00-observation-bias-path', $calibrationPath,
        '--m00-observation-stage', 'full',
        '--m00-observation-weight-cap', '4.0',
        '--m00-observation-seed-checkpoint', 'checkpoints_fsia\run67-m2w-120-500-background-stratum-bias-a1s\best_background_model.pth',
        '--skip-hash-verification'
    )
    if ($resume) {
        $arguments += @('--resume', 'auto')
    }

    Write-Status @{ status = 'training'; run_name = $runName; resumed = $resume }
    $training = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput $trainingStdout `
        -RedirectStandardError $trainingStderr -PassThru
    Write-Status @{ status = 'training'; run_name = $runName; pid = $training.Id; resumed = $resume }
    $training.WaitForExit()
    $training.Refresh()
    $summaryPath = Join-Path $runDir 'training_summary.json'
    if ($training.ExitCode -ne 0 -and -not (Test-Path -LiteralPath $summaryPath)) {
        throw "Training process exited with code $($training.ExitCode)"
    }
    Write-Status @{ status = 'completed'; run_name = $runName; exit_code = 0; resumed = $resume }
    exit 0
} catch {
    $message = $_.Exception.ToString()
    $message | Out-File -LiteralPath $trainingStderr -Encoding UTF8 -Append
    try { Write-Status @{ status = 'failed'; run_name = $runName; error = $message } } catch { }
    exit 1
}
