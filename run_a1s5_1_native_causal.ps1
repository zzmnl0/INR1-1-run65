$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = 'C:\Users\12238\.conda\envs\pytorch_cpu\python.exe'
$runName = 'run67-m2w-120-500-analysis-a1s5-1-a1s-native-causal-past-feature-letkf'
$runDir = Join-Path $root (Join-Path 'checkpoints_fsia' $runName)
$manifest = Join-Path $root 'isr_validation_outputs\run66-modelonly-m0\date_split_manifest.json'
$background = Join-Path $root 'checkpoints_fsia\run67-m2w-120-500-analysis-a1s-10epoch\best_background_model.pth'
foreach ($path in @($manifest, $background)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required artifact missing: $path" }
}
if (Test-Path -LiteralPath $runDir) {
    $items = @(Get-ChildItem -LiteralPath $runDir -Force)
    if ($items.Count -gt 0) { throw "Output directory is not empty: $runDir" }
}
Push-Location $root
try {
    & $python 'main_fsia.py' `
      --run-name $runName `
      --model-domain '120-500-obs-200-500' `
      --date-blocked-split `
      --date-split-semantics 'a1s_native_date_split_v1' `
      --date-split-manifest $manifest `
      --analysis-seed-background $background `
      --analysis-training-scope 'causal_history_only_v1' `
      --analysis-output-semantics 'causal_context_background_v1' `
      --analysis-observation-time-support 'causal_past_lag_0p5_1p5_v1' `
      --causal-background-anchor 'a1s_m00_v1' `
      --r-source 'a1s_frozen_global_v1' `
      --r-fy-fixed 0.03446704712846726 `
      --r-cosmic-fixed 0.02256251445677149 `
      --background-epochs 5 `
 --analysis-epochs 10 `
 --train-only `
 --skip-hash-verification
    if ($LASTEXITCODE -ne 0) { throw "A1-S5-1 training failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
