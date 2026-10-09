$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = 'C:\Users\12238\.conda\envs\pytorch_cpu\python.exe'
$statusPath = Join-Path $root 'a1s5_postprocess_status.txt'
$baseline = Join-Path $root 'checkpoints_fsia\run67-m2w-120-500-analysis-a1s-10epoch\best_fsia_model.pth'
$etkf = Join-Path $root 'isr_validation_outputs\traditional-static-etkf-raw-iri-full30-isr\isr_etkf_cache.npz'
$split = Join-Path $root 'checkpoints_fsia\run67-m2w-120-500-background-a1s2\profile_split.npz'
$isrScript = Join-Path $root 'isr_evaluation\main_isr_eval.py'
$compareScript = Join-Path $root 'isr_evaluation\compare_a1s2_a1s_caches.py'

foreach ($path in @($baseline, $etkf, $split, $isrScript, $compareScript)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required artifact missing: $path"
    }
}

function Write-Status([string]$message) {
    "$(Get-Date -Format o) $message" | Add-Content -LiteralPath $statusPath -Encoding UTF8
}

function Select-Checkpoint([string]$runDir) {
    $best = Join-Path $runDir 'best_fsia_model.pth'
    if (Test-Path -LiteralPath $best -PathType Leaf) { return $best }
    $epochs = @(Get-ChildItem -LiteralPath $runDir -Filter 'epoch_*_model.pth' -File |
        Sort-Object LastWriteTime)
    if ($epochs.Count -eq 0) { throw "No Analysis checkpoint in $runDir" }
    return $epochs[-1].FullName
}

$runs = @(
    [pscustomobject]@{
        Name = 'a1s5'
        Dir = Join-Path $root 'checkpoints_fsia\run67-m2w-120-500-analysis-a1s5-retry-a1s-causal-past-feature-letkf'
        IsrDir = Join-Path $root 'isr_validation_outputs\m2w-a1s5-retry-a1s2split-causal-full30-isr'
        CompareDir = Join-Path $root 'isr_validation_outputs\m2w-a1s5-retry-vs-a1s-full30-comparison-isr-fixed3'
        Prefix = 'A1S5_RETRY_FULL30'
    }
    [pscustomobject]@{
        Name = 'a1s5-1'
        Dir = Join-Path $root 'checkpoints_fsia\run67-m2w-120-500-analysis-a1s5-1-a1s-native-causal-past-feature-letkf'
        IsrDir = Join-Path $root 'isr_validation_outputs\m2w-a1s5-1-native-causal-full30-isr'
        CompareDir = Join-Path $root 'isr_validation_outputs\m2w-a1s5-1-vs-a1s-full30-comparison-isr'
        Prefix = 'A1S5_1_NATIVE_FULL30'
    }
)

Set-Content -LiteralPath $statusPath -Value "START $(Get-Date -Format o)" -Encoding UTF8
while ($true) {
    $ready = $true
    foreach ($run in $runs) {
        if (-not (Test-Path -LiteralPath (Join-Path $run.Dir 'training_summary.json') -PathType Leaf)) {
            $ready = $false
        }
    }
    if ($ready) { break }
    Start-Sleep -Seconds 60
}
Write-Status 'Both training summaries detected.'

foreach ($run in $runs) {
    $checkpoint = Select-Checkpoint $run.Dir
    Write-Status "$($run.Name) checkpoint=$checkpoint"

    $isrCache = Join-Path $run.IsrDir 'isr_evaluation_cache.npz'
    if (-not (Test-Path -LiteralPath $isrCache -PathType Leaf)) {
        if (Test-Path -LiteralPath $run.IsrDir) {
            $items = @(Get-ChildItem -LiteralPath $run.IsrDir -Force)
            if ($items.Count -gt 0) { throw "ISR output directory is not empty: $($run.IsrDir)" }
        }
        & $python $isrScript `
            --checkpoint $checkpoint `
            --baseline-checkpoint $baseline `
            --baseline-label A1S_FULL30 `
            --save-dir $run.IsrDir `
            --observation-profile-pool shared-full30 `
            --shared-full30-profile-split $split `
            --skip-hash-verification
        if ($LASTEXITCODE -ne 0) { throw "ISR failed for $($run.Name), exit=$LASTEXITCODE" }
    }
    Write-Status "$($run.Name) ISR cache ready."

    $report = Join-Path $run.CompareDir 'isr_a1s2_a1s_full30_report.json'
    if (-not (Test-Path -LiteralPath $report -PathType Leaf)) {
        if (Test-Path -LiteralPath $run.CompareDir) {
            $items = @(Get-ChildItem -LiteralPath $run.CompareDir -Force)
            if ($items.Count -gt 0) { throw "Comparison output directory is not empty: $($run.CompareDir)" }
        }
        & $python $compareScript `
            --a1s2-cache $isrCache `
            --a1s-cache (Join-Path $root 'isr_validation_outputs\m2w-a1s-full30-analysis-isr\isr_evaluation_cache.npz') `
            --etkf-cache $etkf `
            --candidate-prefix $run.Prefix `
            --baseline-prefix A1S_FULL30 `
            --full30-output-dir $run.CompareDir
        if ($LASTEXITCODE -ne 0) { throw "Comparison failed for $($run.Name), exit=$LASTEXITCODE" }
    }
    Write-Status "$($run.Name) comparison ready."
}
Write-Status 'DONE'
