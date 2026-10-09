$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root
$python = 'C:\Users\12238\.conda\envs\pytorch_cpu\python.exe'
$logDir = Join-Path $root 'm00_residual_correction_task_logs'
$outputRoot = Join-Path $root 'checkpoints_fsia\run67-m2w-120-500-background-m00-residual-correction'
$statePath = Join-Path $outputRoot 'experiment_state.json'
New-Item -ItemType Directory -Force -Path $logDir, $outputRoot | Out-Null
$trainingSummary = Join-Path $root 'checkpoints_fsia\run67-m2w-120-500-background-stratum-bias-a1s\training_summary.json'
$a1sSummary = Get-Content -LiteralPath $trainingSummary -Raw -Encoding UTF8 | ConvertFrom-Json
if ($a1sSummary.best_epochs.background -ne 3 -or $a1sSummary.checkpoint_stage -ne 'background') {
    throw 'A1-S training summary does not identify the epoch-3 Background checkpoint'
}
$seedCheckpoint = [System.IO.Path]::GetFullPath([string]$a1sSummary.checkpoint)
if (-not (Test-Path -LiteralPath $seedCheckpoint)) { throw 'A1-S epoch-3 checkpoint is missing' }

function Write-ExperimentState([string]$Stage, [string]$Status, [hashtable]$Extra = @{}) {
    $value = @{}
    if (Test-Path -LiteralPath $statePath) {
        $old = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($item in $old.PSObject.Properties) { $value[$item.Name] = $item.Value }
    }
    $value['semantics'] = 'a1s_m00_residual_correction_state_v1'
    $value['stage'] = $Stage
    $value['status'] = $Status
    $value['updated_utc'] = [DateTime]::UtcNow.ToString('o')
    foreach ($key in $Extra.Keys) { $value[$key] = $Extra[$key] }
    $tmp = $statePath + '.tmp'
    $value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $tmp -Encoding UTF8
    Move-Item -LiteralPath $tmp -Destination $statePath -Force
}

function Invoke-PythonLogged([string[]]$Arguments, [string]$LogPath) {
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python @Arguments *> $LogPath
    $code = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    return $code
}

trap {
    Write-ExperimentState 'stopped' 'runtime_failed' @{ error = $_.Exception.Message }
    exit 1
}

$qdSpecs = @(
    @('FY', 'D:\FYsatellite\EDP_data\fy_202409_qc_v2.npy', 'D:\FYsatellite\EDP_data\fy_202409_qc_v2_index.npz', (Join-Path $root 'm00_residual_outputs\fy_qd.npz')),
    @('COSMIC', 'D:\cosmic2\cosmic245-274-September\cosmic_september_2024_qc.npy', 'D:\cosmic2\cosmic245-274-September\cosmic_september_2024_qc_index.npz', (Join-Path $root 'm00_residual_outputs\cosmic_qd.npz'))
)
foreach ($spec in $qdSpecs) {
    $stage = 'qd_' + $spec[0].ToLowerInvariant()
    if (-not (Test-Path -LiteralPath $spec[3])) {
        Write-ExperimentState $stage 'running'
        $log = Join-Path $logDir ($stage + '.log')
        $args = @('generate_qd_sidecar.py', '--source', $spec[0], '--data', $spec[1],
                  '--index', $spec[2], '--output', $spec[3],
                  '--date-split-manifest', (Join-Path $root 'isr_validation_outputs\run66-modelonly-m0\date_split_manifest.json'),
                  '--authorized-partitions', 'train,development')
        $exitCode = Invoke-PythonLogged $args $log
        if ($exitCode -ne 0) { throw "QD $($spec[0]) failed with exit code $exitCode" }
    }
    Write-ExperimentState $stage 'passed' @{ artifact = $spec[3] }
}

$diagnostic = Join-Path $root 'm00_residual_outputs\m00_qd_residual_reliability.json'
if (-not (Test-Path -LiteralPath $diagnostic)) {
    Write-ExperimentState 'reliability' 'running'
    $args = @('prepare_m00_residual_reliability.py', '--output', $diagnostic,
              '--checkpoint', $seedCheckpoint,
              '--date-split-manifest', (Join-Path $root 'isr_validation_outputs\run66-modelonly-m0\date_split_manifest.json'),
              '--sidecar-fy', (Join-Path $root 'm00_residual_outputs\fy_qd.npz'),
              '--sidecar-cosmic', (Join-Path $root 'm00_residual_outputs\cosmic_qd.npz'))
    $log = Join-Path $logDir 'reliability.log'
    $exitCode = Invoke-PythonLogged $args $log
    if ($exitCode -ne 0) { throw "reliability diagnostic failed with exit code $exitCode" }
}
Write-ExperimentState 'reliability' 'passed' @{ artifact = $diagnostic }

$args = @('run_a1s_m00_residual_correction.py', '--training-summary', $trainingSummary,
          '--manifest', (Join-Path $root 'isr_validation_outputs\run66-modelonly-m0\date_split_manifest.json'),
          '--sidecar-fy', (Join-Path $root 'm00_residual_outputs\fy_qd.npz'),
          '--sidecar-cosmic', (Join-Path $root 'm00_residual_outputs\cosmic_qd.npz'),
          '--diagnostic-json', $diagnostic, '--output-root', $outputRoot,
          '--state-path', $statePath)
$exitCode = Invoke-PythonLogged $args (Join-Path $logDir 'stdout.log')
if ($exitCode -ne 0) {
    if (Test-Path -LiteralPath $statePath) {
        $childState = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($childState.status -in @('contract_failed', 'quality_failed', 'awaiting_external_isr')) {
            exit $exitCode
        }
    }
    throw "M00 experiment failed with exit code $exitCode"
}

$final = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($final.stage -eq 'complete' -and $final.status -eq 'passed') {
    Write-ExperimentState 'complete' 'passed' @{ task_exit_code = 0 }
} elseif ($final.status -ne 'awaiting_external_isr') {
    throw "M00 experiment returned without terminal or awaiting-ISR state"
}
