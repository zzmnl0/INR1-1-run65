$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = 'C:\Users\12238\.conda\envs\pytorch_cpu\python.exe'
$runName = 'run67-m2w-120-500-background-a1s2-m00-conditional-shape'
$runDir = Join-Path $root (Join-Path 'checkpoints_fsia' $runName)
$isrDir = Join-Path $root (Join-Path 'isr_validation_outputs' 'm2w-background-a1s2-m00-conditional-shape-isr')
$splitPath = Join-Path $root 'checkpoints_fsia\run67-m2w-120-500-background-a1s2\profile_split.npz'
$baseline = Join-Path $root 'checkpoints_fsia\run67-m2w-120-500-background-a1s2\best_background_model.pth'
$comparison = Join-Path $root 'isr_validation_outputs\m2w-background-a1s2-m00-condadapter-joint-fresh-isr\isr_m00_cache.npz'
$canonical = Join-Path $root 'isr_validation_outputs\traditional-static-etkf-raw-iri-full30-isr\isr_etkf_cache.npz'

foreach ($path in @($splitPath, $baseline, $comparison, $canonical)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required artifact missing: $path"
    }
}
if (Test-Path -LiteralPath $runDir) {
    $items = @(Get-ChildItem -LiteralPath $runDir -Force)
    if ($items.Count -gt 0) { throw "Training output directory is not empty: $runDir" }
}
if (Test-Path -LiteralPath $isrDir) {
    $items = @(Get-ChildItem -LiteralPath $isrDir -Force)
    if ($items.Count -gt 0) { throw "ISR output directory is not empty: $isrDir" }
}

Push-Location $root
try {
    & $python main_fsia.py `
        --run-name $runName `
        --model-domain '120-500-obs-200-500' `
        --qc-data `
        --profile-split-semantics 'source_time_local_block80_dev20_v1' `
        --profile-split-path $splitPath `
        --background-only `
        --background-epochs 5 `
        --analysis-epochs 10 `
        --background-stratum-bias-gradient-target 0 `
        --background-checkpoint-selection 'conditional-shape-constrained-v1' `
        --background-selection-baseline-checkpoint $baseline `
        --background-shape-loss 'conditional-profile-centered-scale-v1' `
        --background-shape-gradient-target 0.02 `
        --background-shape-calibration-batches 20 `
        --background-shape-min-profiles-per-cell 8 `
        --skip-hash-verification
    if ($LASTEXITCODE -ne 0) { throw "Background training failed with exit code $LASTEXITCODE" }

    $checkpoint = Join-Path $runDir 'best_background_model.pth'
    $diagnostic = $false
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        $checkpoint = Join-Path $runDir 'diagnostic_background_model.pth'
        $diagnostic = $true
    }
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        throw "No finite Background checkpoint produced in $runDir"
    }

    $args = @(
        'isr_evaluation/background_m00_eval.py',
        '--checkpoint', $checkpoint,
        '--cache', $canonical,
        '--comparison-cache', $comparison,
        '--output-dir', $isrDir,
        '--candidate-name', 'M00_A1S2_CONDITIONAL_SHAPE',
        '--reference-name', 'M00_A1S2',
        '--static-etkf-name', 'Static_ETKF_FULL30'
    )
    if ($diagnostic) { $args += '--diagnostic-only' }
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "M00 ISR failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
