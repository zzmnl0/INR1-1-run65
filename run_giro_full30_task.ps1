$ErrorActionPreference = 'Stop'

$workdir = 'D:\code11\IRI01\IRI03\INR1-1-run65'
$python = 'C:\Users\12238\.conda\envs\pytorch_cpu\python.exe'
$script = Join-Path $workdir 'evaluate_giro_peak.py'
$outdir = Join-Path $workdir 'isr_validation_outputs\m2w-a1s2-vs-a1s-giro-full30'
$stdout = Join-Path $workdir 'giro_full30_task_stdout.log'
$stderr = Join-Path $workdir 'giro_full30_task_stderr.log'
$status = Join-Path $workdir 'giro_full30_task_status.txt'

Set-Location -LiteralPath $workdir
if (-not (Test-Path -LiteralPath $outdir)) {
    New-Item -ItemType Directory -Path $outdir | Out-Null
}
$existing = @(Get-ChildItem -LiteralPath $outdir -Force -File -ErrorAction SilentlyContinue)
if ($existing.Count -gt 0) {
    "STOP: output directory is not empty ($($existing.Count) files)" | Set-Content -LiteralPath $status -Encoding UTF8
    exit 2
}

"START $(Get-Date -Format o)" | Set-Content -LiteralPath $status -Encoding UTF8
& $python $script `
    --checkpoint 'checkpoints_fsia/run67-m2w-120-500-background-a1s2/best_fsia_model.pth' `
    --baseline-checkpoint 'checkpoints_fsia/run67-m2w-120-500-analysis-a1s-10epoch/best_fsia_model.pth' `
    --baseline-label A1S `
    --observation-pool shared-full30 `
    --save-dir 'isr_validation_outputs/m2w-a1s2-vs-a1s-giro-full30' `
    --skip-sha256 1> $stdout 2> $stderr
$exitCode = $LASTEXITCODE
"END $(Get-Date -Format o) exit=$exitCode" | Add-Content -LiteralPath $status -Encoding UTF8
exit $exitCode
