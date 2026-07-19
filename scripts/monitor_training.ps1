param(
    [string]$ProjectRoot = 'D:\LOW_LIGHT\DAT301-SU26',
    [int]$TrainingPid = 36372,
    [int]$IntervalSeconds = 30,
    [string]$Stage = 'C',
    [string]$CheckpointRun = 'seed_3407',
    [int]$TargetEpochs = 30
)

$statusPath = Join-Path $ProjectRoot 'runs\LIVE_TRAINING_STATUS.json'
$logPath = Join-Path $ProjectRoot 'runs\LIVE_TRAINING_STATUS.log'
$stageLower = $Stage.ToLowerInvariant()
$historyPath = Join-Path $ProjectRoot "runs\ladd_uav\checkpoints\$CheckpointRun\stage_${stageLower}_history.json"

while (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) {
    $history = @()
    if (Test-Path $historyPath) {
        try { $history = Get-Content $historyPath -Raw | ConvertFrom-Json } catch { $history = @() }
    }
    $gpuRaw = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null)
    $gpu = $gpuRaw -split ',\s*'
    $historyCount = if ($null -eq $history) { 0 } else { $history.Count }
    $latest = if ($historyCount -gt 0) { $history[$historyCount - 1] } else { $null }
    $status = [ordered]@{
        updated_at = (Get-Date).ToString('o')
        running = $true
        process_id = $TrainingPid
        stage = $Stage
        epochs_completed = $historyCount
        checkpoint_run = $CheckpointRun
        target_epochs = $TargetEpochs
        progress_percent = [math]::Round(100.0 * $historyCount / $TargetEpochs, 2)
        latest_metrics = $latest
        gpu = [ordered]@{
            utilization_percent = if ($gpu.Count -ge 1) { [double]$gpu[0] } else { $null }
            memory_used_mib = if ($gpu.Count -ge 2) { [double]$gpu[1] } else { $null }
            memory_total_mib = if ($gpu.Count -ge 3) { [double]$gpu[2] } else { $null }
            temperature_c = if ($gpu.Count -ge 4) { [double]$gpu[3] } else { $null }
            power_w = if ($gpu.Count -ge 5) { [double]$gpu[4] } else { $null }
        }
    }
    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding utf8
    "$(Get-Date -Format o) run=$CheckpointRun stage=$Stage epoch=$historyCount/$TargetEpochs gpu=$($status.gpu.utilization_percent)% vram=$($status.gpu.memory_used_mib)MiB temp=$($status.gpu.temperature_c)C" |
        Add-Content -LiteralPath $logPath -Encoding utf8
    Start-Sleep -Seconds $IntervalSeconds
}

$final = [ordered]@{
    updated_at = (Get-Date).ToString('o')
    running = $false
    process_id = $TrainingPid
    stage = $Stage
    message = "Training process ended; inspect stage_${stageLower}_history.json and checkpoint metadata."
}
$final | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statusPath -Encoding utf8
"$(Get-Date -Format o) stage=$Stage process-ended" | Add-Content -LiteralPath $logPath -Encoding utf8
