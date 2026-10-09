$ErrorActionPreference = 'Stop'
$workdir = 'D:\code11\IRI01\IRI03\INR1-1-run65'
$python = 'C:\Users\12238\.conda\envs\pytorch_cpu\python.exe'
$run = 'checkpoints_fsia\run67-m2w-120-500-analysis-a1s6-train-plus-locked'
$stdout = Join-Path $workdir 'a1s6_train_stdout.log'
$stderr = Join-Path $workdir 'a1s6_train_stderr.log'
$status = Join-Path $workdir 'a1s6_train_status.txt'
Set-Location -LiteralPath $workdir
if (Test-Path -LiteralPath $run) {
    $items = @(Get-ChildItem -LiteralPath $run -Force -File -ErrorAction SilentlyContinue)
    if ($items.Count -gt 0) { "STOP: output not empty ($($items.Count) files)" | Set-Content $status; exit 2 }
}
"START $(Get-Date -Format o)" | Set-Content -LiteralPath $status -Encoding UTF8
& $python 'main_fsia.py' `
  --run-name 'run67-m2w-120-500-analysis-a1s6-train-plus-locked' `
  --model-domain '120-500-obs-200-500' `
  --date-blocked-split `
  --date-split-semantics 'a1s_train_plus_locked_dev5_v1' `
  --date-split-manifest 'isr_validation_outputs\run66-modelonly-m0\date_split_manifest.json' `
  --background-epochs 5 `
  --analysis-epochs 10 `
  --background-checkpoint-selection 'bias-constrained-v1' `
  --background-selection-baseline 'checkpoints_fsia\run66-m2w-120-500-obs-200-500-gate-off\background_development_replay.json' `
  1> $stdout 2> $stderr
$exitCode = $LASTEXITCODE
"END $(Get-Date -Format o) exit=$exitCode" | Add-Content -LiteralPath $status -Encoding UTF8
exit $exitCode
