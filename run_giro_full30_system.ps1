$ErrorActionPreference = 'Stop'

$workdir = 'D:\code11\IRI01\IRI03\INR1-1-run65'
$python = 'C:\Users\12238\.conda\envs\pytorch_cpu\python.exe'
$script = Join-Path $workdir 'evaluate_giro_peak.py'
$outdir = Join-Path $workdir 'isr_validation_outputs\m2w-a1s2-vs-a1s-giro-full30'
$resumeDir = Join-Path $workdir 'isr_validation_outputs\m2w-a1s2-vs-a1s-giro-full30-resume'
$stdout = Join-Path $workdir 'giro_full30_system_stdout.log'
$stderr = Join-Path $workdir 'giro_full30_system_stderr.log'
$status = Join-Path $workdir 'giro_full30_system_status.txt'

Set-Location -LiteralPath $workdir
foreach ($required in @($python, $script)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        "STOP: missing $required" | Set-Content -LiteralPath $status -Encoding UTF8
        exit 2
    }
}
if (-not (Test-Path -LiteralPath $outdir)) {
    New-Item -ItemType Directory -Path $outdir | Out-Null
}
$existing = @(Get-ChildItem -LiteralPath $outdir -Force -File -ErrorAction SilentlyContinue)
if ($existing.Count -gt 0) {
    "STOP: output directory is not empty ($($existing.Count) files)" | Set-Content -LiteralPath $status -Encoding UTF8
    exit 2
}

"START $(Get-Date -Format o) account=TASK_SCHEDULER" | Set-Content -LiteralPath $status -Encoding UTF8
& $python -u $script `
    --checkpoint 'checkpoints_fsia/run67-m2w-120-500-background-a1s2/best_fsia_model.pth' `
    --baseline-checkpoint 'checkpoints_fsia/run67-m2w-120-500-analysis-a1s-10epoch/best_fsia_model.pth' `
    --baseline-label A1S `
    --observation-pool shared-full30 `
    --save-dir 'isr_validation_outputs/m2w-a1s2-vs-a1s-giro-full30' `
    --resume-dir $resumeDir `
    --skip-sha256 1> $stdout 2> $stderr
$exitCode = $LASTEXITCODE
"END $(Get-Date -Format o) exit=$exitCode" | Add-Content -LiteralPath $status -Encoding UTF8
exit $exitCode
