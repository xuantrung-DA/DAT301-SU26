param(
    [string]$ProjectRoot = 'D:\LOW_LIGHT\DAT301-SU26',
    [int]$EvaluationPid,
    [string]$RunName,
    [string]$OutputDirectory,
    [int]$IntervalSeconds = 30,
    [string]$ExpectedMethods = 'none,gamma,clahe,m0,m1,m2'
)
$statusPath = Join-Path $ProjectRoot 'runs\LIVE_TRAINING_STATUS.json'
$logPath = Join-Path $ProjectRoot 'runs\LIVE_TRAINING_STATUS.log'
$outputPath = Join-Path $ProjectRoot $OutputDirectory
$expected = @($ExpectedMethods -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
while (Get-Process -Id $EvaluationPid -ErrorAction SilentlyContinue) {
    $completed = @($expected | Where-Object { Test-Path (Join-Path $outputPath "${_}_seed_3407_summary.json") })
    $gpuRaw = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null)
    $gpu = $gpuRaw -split ',\s*'
    $status = [ordered]@{
        updated_at = (Get-Date).ToString('o'); running = $true; process_id = $EvaluationPid
        job = 'evaluation'; run = $RunName; completed_methods = $completed
        methods_completed = $completed.Count; methods_total = $expected.Count
        progress_percent = [math]::Round(100.0 * $completed.Count / $expected.Count, 2)
        gpu = [ordered]@{ utilization_percent=[double]$gpu[0]; memory_used_mib=[double]$gpu[1]; memory_total_mib=[double]$gpu[2]; temperature_c=[double]$gpu[3] }
    }
    $status | ConvertTo-Json -Depth 6 | Set-Content $statusPath -Encoding utf8
    "$(Get-Date -Format o) job=evaluation run=$RunName methods=$($completed.Count)/$($expected.Count) gpu=$($gpu[0])% vram=$($gpu[1])MiB" | Add-Content $logPath -Encoding utf8
    Start-Sleep -Seconds $IntervalSeconds
}
@{updated_at=(Get-Date).ToString('o');running=$false;job='evaluation';run=$RunName;message='Evaluation process ended; inspect output summaries.'} | ConvertTo-Json | Set-Content $statusPath -Encoding utf8
"$(Get-Date -Format o) job=evaluation run=$RunName process-ended" | Add-Content $logPath -Encoding utf8
